# Copyright (c) 2024  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import paddle
import paddle.nn as nn
from paddle.nn import functional as F
import math
from paddlenlp.utils.log import logger
from paddle.distributed.fleet.base import topology as tp
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from ..advanced.trellis_utils import Pad, cal_sign, decode_twos_complement
# from wintx import weight_only_linear_int2_symm, weight_only_linear_int4_symm

class TrellisQuantLinear(nn.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        lut_bits,
        states,
        state_bits,
        group_size=64,
        super_group_size=256,
        group_scale_bits=4,
        group_zp_bits=4,
        quant_scale=False,
        symmetric=True,
        pack=False,
        isolate_outliers=False,
        bias=None,
        hadamard=False,
        extract_sign=False,
        weight_sign=None,
        enable_norm=False,
        enable_perm=False,
        enable_cluster=False,
        redundant_bits=0,
        parity_sign=False,
        enable_razor=False,
        enable_completion=False,
        enable_compensation=False,
        enable_circle=False,
        **kwargs,
    ):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self.lut_bits = lut_bits
        self.states = states
        self.state_bits = state_bits
        self.group_size = group_size
        self.super_group_size = super_group_size
        self.group_scale_bits = group_scale_bits
        self.group_zp_bits = group_zp_bits
        self.quant_scale = quant_scale
        self.extract_sign = extract_sign
        self._symmetric = symmetric
        self.enable_circle = enable_circle
        if self._symmetric and not self.extract_sign:
            self._qmin = -(2 ** (self.lut_bits - 1))
            self._qmax = 2 ** (self.lut_bits - 1) - 1
        else:
            self._qmin = 0
            self._qmax = 2 ** self.lut_bits - 1

        self._thread_num = os.cpu_count()
        
        if extract_sign:
            assert pack is False, "extract sign requires non packed"
            assert weight_sign is not None, "extract sign requires weight_sign"
        self.weight_sign = weight_sign

        self.packed = pack # only pack weight now
        
        self.enable_norm = enable_norm
        if self.group_size == -1:
            self.group_size = self._in_features
        self._group_nums = self._in_features // self.group_size
        if isinstance(self.states, (list, tuple)):
            self.states_length = sum(self.states)
        else:
            self.states_length = self.states
  
        if self.group_size % self.states_length == 0:
            self._pad_cols = 0 
        else:
            self._pad_cols = self.states_length - self.group_size % self.states_length
        self._state_nums = (self.group_size + self._pad_cols) // self.states_length
        
        if isinstance(self.states, (list, tuple)):
            code_length = 0
            for i in range(len(self.states)):
                code_length += self.lut_bits + (self.states[i] - 1)*self.state_bits[i]
            data_bits = code_length
        else:
            data_bits = self.lut_bits + (self.states_length - 1)*self.state_bits
        if data_bits > 16:
            dtype = 'int32'
        elif data_bits > 8:
            dtype = 'int16'
        else:
            dtype = 'int8'
        if enable_cluster:
            dtype = 'uint8'
        self.data_bits = data_bits
        if enable_circle:
            self.data_bits = self.lut_bits
            dtype = 'uint8'
        if self.data_bits > 8:
            assert self.packed is False, f"only data_bits<8, it can pack weight, now data_bits:{data_bits}"
        if self.packed:
            # only consider pack to int8 format now
            pack_num = 8 // data_bits
            self.register_buffer(
                'weight',
                paddle.zeros([self._group_nums*self._state_nums//pack_num, self._out_features],
                            dtype=dtype),
            )
        else:
            self.register_buffer(
                'weight',
                paddle.zeros([self._group_nums*self._state_nums, self._out_features],
                            dtype=dtype),
            )
        
        if self.quant_scale:
            scale_dtype = 'uint8'
            scale_shape = [self._group_nums//(8//self.group_scale_bits), self._out_features]
        else:
            scale_dtype = 'bfloat16'
            scale_shape = [self._group_nums, self._out_features]
        self.register_buffer(
            'scales',
            paddle.zeros(scale_shape, dtype=scale_dtype),
        )
        if self.quant_scale:
            self.register_buffer(
                'super_scales',
                paddle.zeros([self._out_features], dtype='bfloat16'),
            )
        if not self._symmetric:
            self.register_buffer(
                'zero_points',
                paddle.zeros([self._group_nums, self._out_features], dtype=scale_dtype),
            )
    
        self._isolate = isolate_outliers
        if self._isolate:
            self.register_buffer(
                'outliers_weight',
                paddle.zeros([group_nums, self.group_size], dtype='bfloat16'),
            )
        if self.enable_norm:
            self.register_buffer(
                'norm_scale',
                paddle.zeros([self._in_features], dtype='bfloat16'),
            )
            self.register_buffer(
                'norm_bias',
                paddle.zeros([self._in_features], dtype='bfloat16'),
            )
        self.enable_cluster = enable_cluster
        if self.enable_cluster:
            self.register_buffer(
                'code_scale',
                paddle.zeros([self._out_features], dtype='float32'),
            )
            self.register_buffer(
                'code_zp',
                paddle.zeros([self._out_features], dtype='float32'),
            )
        self.redundant_bits = redundant_bits

        self.enable_perm = enable_perm
        if self.enable_perm:
            self.perm = None
        self.hadamard = hadamard
        self.dq_weight = None
        self.bias = bias
        self.parity_sign = parity_sign
        self.enable_razor = enable_razor
        self.enable_completion = enable_completion
        self.enable_compensation = enable_compensation

        if enable_razor:
            quant_bits = 8
            self._qmin = -(2 ** (quant_bits - 1))
            self._qmax = 2 ** (quant_bits - 1) - 1
        self.tq_config = {
            'lut_bits': self.lut_bits,
            'states': self.states,
            'state_bits': self.state_bits,
            'group_size': self.group_size,
            'quant_scale': self.quant_scale,
            'group_scale_bits': self.group_scale_bits,
            'redundant_bits': self.redundant_bits,
            'parity_sign': self.parity_sign,
            'pack': self.packed,
            'enable_completion': self.enable_completion,
            'enable_compensation': self.enable_compensation,
            'enable_cluster': self.enable_cluster,
            'enable_circle': self.enable_circle,
        }

    def init_parameters(self, 
        quant_weight, 
        scales, 
        super_scales=None, 
        zero_points=None, 
        super_zero_points=None, 
        is_pack=False, 
        outliers_weight=None,
        norm_scale=None,
        norm_bias=None,
        perm=None,
        indices_scale=None,
        indices_zp=None,
        svd_bias=None,
        for_infer=True,
    ):
        if self.packed and not is_pack:
            # [total group nums, state nums] ->  [total group nums, state nums//pack nums]
            quant_weight = self.pack(quant_weight, self.data_bits)
        if not for_infer:
            # reshape is not need for load quant ckpt.
            # [total group nums, state nums] -> [group_nums * state_nums, out_features]
            quant_weight = quant_weight.reshape([self._out_features, -1]).t()
            if scales is not None:
                scales = scales.reshape([self._out_features, -1])
                if self.quant_scale and self.group_scale_bits < 8:
                    scales = self.pack(scales, self.group_scale_bits)
                scales = scales.t()
            if not self._symmetric:
                zero_points = zero_points.reshape([self._out_features, -1]).t()

        faster_set_value(quant_weight, self.weight)
        if scales is not None:
            faster_set_value(scales, self.scales)
        if not self._symmetric:
            faster_set_value(zero_points, self.zero_points)
        if self.quant_scale:
            faster_set_value(super_scales, self.super_scales)
        if self.enable_norm:
            faster_set_value(norm_scale, self.norm_scale)
            if norm_bias is not None:
                faster_set_value(norm_bias, self.norm_bias)
        if self.enable_perm:
            assert perm is not None
            self.perm = perm
            self.inv_perm = None
        if self.enable_cluster:
            faster_set_value(indices_scale, self.code_scale)
            faster_set_value(indices_zp, self.code_zp)
        if self.enable_compensation:
            self.svd_bias = svd_bias

    def pack(self, src, bits):
        # pack parameters: use int8 to save multiple weights and scales
        pack_num = 8 // bits
        shift_bits = (paddle.arange(0, pack_num) * bits).cast('uint8')
        
        src = src.cast('uint8')
        if len(src.shape) == 2:
            row, col = src.shape
            src = src.reshape((row, col//pack_num, pack_num))
        else:
            src = src.reshape((src.shape[0]//pack_num, pack_num))
        src = src << shift_bits 
        return src.sum(axis=-1)
    
    def unpack(self, src, bits):
        pack_num = 8 // bits 
        bnt = paddle.to_tensor(2**bits - 1, dtype='uint8')
        shift_bits = (paddle.arange(0, pack_num) * bits).cast('uint8')

        if len(src.shape) == 2:
            row, col = src.shape
            src = src.reshape((row, col, 1)).expand((row, col, pack_num))
            src = src >> shift_bits & bnt
            src = src.reshape((row, col*pack_num))
        else:
            src = src.reshape((src.shape[0], 1)).expand((src.shape[0], pack_num))
            src = src >> shift_bits & bnt
            src = src.reshape([src.shape[0]*pack_num])
        return src

    def decode(self, weight):
        """
        weight: [group nums, state nums]
        """
        if self.enable_cluster:
            group_nums, state_nums = weight.shape
            weight = weight.reshape([self._out_features, -1])
            weight = (weight.cast('float32') * self.code_scale.unsqueeze([-1]) + self.code_zp.unsqueeze([-1])).round().cast('int32') # [out_features, -1]
            weight = weight.reshape([group_nums, state_nums])

        if self.enable_circle:
            mask = paddle.to_tensor(2**self.lut_bits - 1, dtype='int32')
            shifts = (paddle.arange(0, self.states, 1) * self.state_bits).cast('int32') % self.lut_bits
            weight = weight.unsqueeze(-1).cast('int32')
            weight = (weight >> shifts) | ((weight << (self.lut_bits - shifts)) & mask)
        elif isinstance(self.states, (list, tuple)):
            # for multi code format
            code_length = []
            for i in range(len(self.states)):
                code_length.append(self.lut_bits + self.state_bits[i] * (self.states[i] - 1))
            shifts = []
            for i in range(len(self.states)): # [3, 4] bf16 -> 13bits
                if i == len(self.states) - 1:
                    cur = paddle.arange(self.states[i]-1, -1, -1) * self.state_bits[i]
                else:
                    cur = paddle.arange(self.states[i]-1, -1, -1) * self.state_bits[i] + code_length[i+1]
                shifts.append(cur)
                # [13, 11, 9, 6, 4, 2, 0]
            shifts = paddle.concat(shifts, axis=0).cast('int32')
            mask = paddle.to_tensor(2**self.lut_bits - 1, dtype='int32')
            weight = (weight.unsqueeze(-1).cast('int32') >> shifts) & mask
        else:
            shifts = (paddle.arange(self.states-1, -1, -1) * self.state_bits).cast('int32')
            mask = paddle.to_tensor((2**self.lut_bits) - 1, dtype='int32') << shifts
            weight = (weight.unsqueeze(-1).cast('int32') & mask) >> shifts
        
        if self.enable_razor:
            weight = weight << paddle.to_tensor(4, dtype='int32')

        if self.parity_sign:
            weight = cal_sign(weight).cast('bfloat16')
        elif self.enable_completion:
            weight = decode_twos_complement(weight, self.lut_bits).cast('bfloat16')
        else:
            weight = weight.cast('bfloat16') + self._qmin
        
        # weight += self._qmin # This will result in numerical overflow
        weight = weight.reshape([weight.shape[0], -1]) # [group_nums, group_size]
        assert weight.shape[1] == self.group_size + self._pad_cols, f"{weight.shape[1]} != {self.group_size} + {self._pad_cols}"

        if weight.shape[1] > self.group_size:
            weight = weight[:, :self.group_size]
    
        return weight

    def dequant(self, use_triton=False):
        weight = self.weight.t()
        if self.packed:
            weight = self.unpack(weight, self.data_bits) # [out_features, -1]

        weight = weight.reshape([self._out_features, self._group_nums, self._state_nums])
        weight = weight.reshape([-1, self._state_nums])
        scales = self.scales.t().reshape([-1])
        if self.quant_scale:
            if self.redundant_bits > 0 and not self.enable_cluster:
                mask = 2**self.redundant_bits - 1
                scales = weight[:, -1] & paddle.to_tensor(mask, dtype=weight.dtype)
            super_scales = self.super_scales

        weight = self.decode(weight)
        
        if not self._symmetric:
            zero_points = self.zero_points.t().reshape([-1]) 
            # super_zero_points = self.super_zero_points

        if self.quant_scale:
            # dequant scales
            scales = scales.reshape([self._out_features, -1])
            if self.redundant_bits == 0 or self.enable_cluster:
                scales = self.unpack(scales, self.group_scale_bits)
            scales = (scales.cast('bfloat16') * self.super_scales.unsqueeze(-1)).reshape([-1]).cast('bfloat16')

        if self._symmetric:
            # dequant weight
            dequant_weight = weight.cast("bfloat16") * scales.unsqueeze(-1)
        else:
            dequant_weight = weight.cast('bfloat16') * scales.unsqueeze(-1) + zero_points.unsqueeze(-1)

        if self._isolate:
            dequant_weight = dequant_weight + self.outliers_weight.cast('bfloat16')
            # clear params
            self.outliers_weight.value().get_tensor()._clear()
        
        if self.hadamard:
            h = paddle.load('hadamard_matrix_64.pdtensors')
            if dequant_weight.shape[1] == self._in_features:
                dequant_weight = dequant_weight.reshape([-1, 64]).cast('float32') @ h.t() 
                dequant_weight = dequant_weight.reshape([self._out_features, -1])
            else:
                dequant_weight = dequant_weight.cast("float32") @ h.t() 
            dequant_weight = dequant_weight.cast('bfloat16')
            del h

        if self.enable_perm:
            if self.inv_perm is None:
                self.inv_perm = paddle.argsort(self.perm)
            dequant_weight = dequant_weight[:, self.inv_perm]

        dequant_weight = dequant_weight.reshape([self._out_features, self._in_features])
        if self.enable_norm:
            dequant_weight = dequant_weight / self.norm_scale.unsqueeze(0) + self.norm_bias.unsqueeze(0)
        if self.enable_compensation:
            dequant_weight = dequant_weight + self.svd_bias.cast('bfloat16')
            # logger.info(f"svd_bias: {self.svd_bias}")
        dequant_weight = dequant_weight.t()
        if self.extract_sign:
            dequant_weight *= self.weight_sign
        return dequant_weight

    def forward(self, x):
        use_triton = False
        if use_triton and self.packed:
            if self.quant_bits == 2:
                return weight_only_linear_int2_symm(x, self.weight, self.bias, self.scales)
            elif self.quant_bits == 4:
                return weight_only_linear_int4_symm(x, self.weight, self.bias, self.scales)

        if self.dq_weight is None:
            self.dq_weight = self.dequant()
        dq_weight = self.dq_weight
        # dq_weight = self.dequant()
        # if self.enable_norm:
        #     x = x * self.norm_scale
        res = F.linear(x, dq_weight, self.bias)
        return res

class TrellisQuantRowParallelLinear(nn.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        lut_bits,
        states,
        state_bits,
        group_size=64,
        super_group_size=256,
        group_scale_bits=4,
        group_zp_bits=4,
        quant_scale=False,
        symmetric=True,
        pack=False,
        isolate_outliers=False,
        bias=None,
        hadamard=False,
        extract_sign=False,
        weight_sign=None,
        enable_cluster=False,
        redundant_bits=0,
        input_is_parallel=False,
        mp_group=None,
        parity_sign=False,
        enable_norm=False,
        enable_perm=False,
        enable_circle=False,
        **kwargs,
    ):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self.lut_bits = lut_bits
        self.states = states
        self.state_bits = state_bits
        self.group_size = group_size
        self.super_group_size = super_group_size
        self.group_scale_bits = group_scale_bits
        self.group_zp_bits = group_zp_bits
        self.quant_scale = quant_scale
        self._symmetric = symmetric
        if self._symmetric:
            self._qmin = -(2 ** (self.lut_bits - 1))
            self._qmax = 2 ** (self.lut_bits - 1) - 1
        else:
            self._qmin = 0
            self._qmax = 2 ** self.lut_bits - 1
        self._thread_num = os.cpu_count()
        self.extract_sign = extract_sign
        if extract_sign:
            assert pack is False, "extract sign requires non packed"
            assert weight_sign is not None, "extract sign requires weight_sign"
        self.weight_sign = weight_sign

        self.packed = pack # only pack weight now

        if self.group_size == -1:
            self.group_size = self._in_features
        self._group_nums = self._in_features // self.group_size
        if isinstance(self.states, (list, tuple)):
            self.states_length = sum(self.states)
        else:
            self.states_length = self.states

        if self.group_size % self.states_length == 0:
            self._pad_cols = 0 
        else:
            self._pad_cols = self.states_length - self.group_size % self.states_length
        self._state_nums = (self.group_size + self._pad_cols) // self.states_length
        if isinstance(self.states, (list, tuple)):
            code_length = 0
            for i in range(len(self.states)):
                code_length += self.lut_bits + (self.states[i] - 1)*self.state_bits[i]
            data_bits = code_length
        else:
            data_bits = self.lut_bits + (self.states_length - 1)*self.state_bits
        if data_bits > 8:
            dtype = 'int16'
        else:
            dtype = 'int8'
        if enable_cluster:
            dtype = 'uint8'
        self.data_bits = data_bits

        if self.packed:
            # only consider pack to int8 format now
            pack_num = 8 // data_bits
            self.register_buffer(
                'weight',
                paddle.zeros([self._group_nums*self._state_nums//pack_num, self._out_features],
                            dtype=dtype),
            )
        else:
            self.register_buffer(
                'weight',
                paddle.zeros([self._group_nums*self._state_nums, self._out_features],
                            dtype=dtype),
            )
        if self.quant_scale:
            scale_dtype = 'uint8'
            scale_shape = [self._group_nums//(8//self.group_scale_bits), self._out_features]
        else:
            scale_dtype = 'bfloat16'
            scale_shape = [self._group_nums, self._out_features]
        self.register_buffer(
            'scales',
            paddle.zeros(scale_shape, dtype=scale_dtype),
        )
        
        if self.quant_scale:
            self.register_buffer(
                'super_scales',
                paddle.zeros([self._out_features],
                            dtype='bfloat16'),
            )
        if not self._symmetric:
            self.register_buffer(
                'zero_points',
                paddle.zeros([group_nums], dtype=scale_dtype),
            )
            if self.quant_scale:
                self.register_buffer(
                    'super_zero_points',
                    paddle.zeros([group_nums//(self.super_group_size//self.group_size)],
                                dtype='bfloat16'),
                )

        self._isolate = isolate_outliers
        if self._isolate:
            self.register_buffer(
                'outliers_weight',
                paddle.zeros([group_nums, self.group_size], dtype='bfloat16'),
            )
        self.enable_cluster = enable_cluster
        if self.enable_cluster:
            self.register_buffer(
                'code_scale',
                paddle.zeros([self._out_features], dtype='float32'),
            )
            self.register_buffer(
                'code_zp',
                paddle.zeros([self._out_features], dtype='float32'),
            )
        self.redundant_bits = redundant_bits

        self.hadamard = hadamard
        self.enable_perm = enable_perm
        self.enable_norm = enable_norm
        self.enable_circle = enable_circle
        self.dq_weight = None
        self.bias = bias
        self.parity_sign = parity_sign
        self.tq_config = {
            'lut_bits': self.lut_bits,
            'states': self.states,
            'state_bits': self.state_bits,
            'group_size': self.group_size,
            'quant_scale': self.quant_scale,
            'redundant_bits': self.redundant_bits,
            'parity_sign': self.parity_sign,
            'pack': self.packed,
        }

        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group() if mp_group is None else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size() if mp_group is None else mp_group.nranks
        )
        self.is_mp = self.world_size > 1
        self.input_is_parallel = input_is_parallel
        # self.weight.is_distributed = True if self.is_mp else False
        # self.scales.is_distributed = True if self.is_mp else False

    def init_parameters(self, 
        quant_weight, 
        scales, 
        super_scales=None, 
        zero_points=None, 
        super_zero_points=None, 
        is_pack=False, 
        outliers_weight=None,
        norm_scale=None,
        norm_bias=None,
        perm=None,
        indices_scale=None,
        indices_zp=None,
        svd_bias=None,
        for_infer=True,
    ):
        if self.packed and not is_pack:
            # [total group nums, state nums] ->  [total group nums, state nums//pack nums]
            quant_weight = self.pack(quant_weight, self.data_bits)
        if not for_infer:
            # reshape is not need for load quant ckpt.
            # [total group nums, state nums] -> [group_nums * state_nums, out_features]
            quant_weight = quant_weight.reshape([self._out_features, -1]).t()
            if scales is not None:
                scales = scales.reshape([self._out_features, -1])
                if self.quant_scale and self.group_scale_bits < 8:
                    scales = self.pack(scales, self.group_scale_bits)
                scales = scales.t()
            if not self._symmetric:
                zero_points = zero_points.reshape([self._out_features, -1]).t()

        faster_set_value(quant_weight, self.weight)
        if scales is not None:
            faster_set_value(scales, self.scales)
        if not self._symmetric:
            faster_set_value(zero_points, self.zero_points)
        if self.quant_scale:
            faster_set_value(super_scales, self.super_scales)
        if self.enable_norm:
            faster_set_value(norm_scale, self.norm_scale)
            if norm_bias is not None:
                faster_set_value(norm_bias, self.norm_bias)
        if self.enable_perm:
            assert perm is not None
            self.perm = perm
            self.inv_perm = None
        if self.enable_cluster:
            faster_set_value(indices_scale, self.code_scale)
            faster_set_value(indices_zp, self.code_zp)
   
    def pack(self, src, bits):
        # pack parameters: use int32 to save multiple weights and scales
        pack_num = 8 // bits
        shift_bits = (paddle.arange(0, pack_num) * bits).cast('uint8')
        
        src = src.cast('uint8')
        if len(src.shape) == 2:
            row, col = src.shape
            src = src.reshape((row, col//pack_num, pack_num))
        else:
            src = src.reshape((src.shape[0]//pack_num, pack_num))
        src = src << shift_bits 
        return src.sum(axis=-1)
    
    def unpack(self, src, bits):
        pack_num = 8 // bits 
        bnt = paddle.to_tensor(2**bits - 1, dtype='uint8')
        shift_bits = (paddle.arange(0, pack_num) * bits).cast('uint8')

        if len(src.shape) == 2:
            row, col = src.shape
            src = src.reshape((row, col, 1)).expand((row, col, pack_num))
            src = src >> shift_bits & bnt
            src = src.reshape((row, col*pack_num))
        else:
            src = src.reshape((src.shape[0], 1)).expand((src.shape[0], pack_num))
            src = src >> shift_bits & bnt
            src = src.reshape([src.shape[0]*pack_num])
        return src

    def decode(self, weight):
        """
        weight: [group nums, state nums]
        """
        if self.enable_cluster:
            group_nums, state_nums = weight.shape
            weight = weight.reshape([self._out_features, -1])
            weight = (weight.cast('float32') * self.code_scale.unsqueeze([-1]) + self.code_zp.unsqueeze([-1])).round().cast('int32') # [out_features, -1]
            weight = weight.reshape([group_nums, state_nums])

        if self.enable_circle:
            mask = paddle.to_tensor(2**self.lut_bits - 1, dtype='int32')
            shifts = (paddle.arange(0, self.states, 1) * self.state_bits).cast('int32') % self.lut_bits
            weight = weight.unsqueeze(-1).cast('int32')
            weight = (weight >> shifts) | ((weight << (self.lut_bits - shifts)) & mask)
        elif isinstance(self.states, (list, tuple)):
            # for multi code format
            code_length = []
            for i in range(len(self.states)):
                code_length.append(self.lut_bits + self.state_bits[i] * (self.states[i] - 1))
            shifts = []
            for i in range(len(self.states)): # [3, 4] bf16 -> 13bits
                if i == len(self.states) - 1:
                    cur = paddle.arange(self.states[i]-1, -1, -1) * self.state_bits[i]
                else:
                    cur = paddle.arange(self.states[i]-1, -1, -1) * self.state_bits[i] + code_length[i+1]
                shifts.append(cur)
                # [13, 11, 9, 6, 4, 2, 0]
            shifts = paddle.concat(shifts, axis=0).cast('int32')
            mask = paddle.to_tensor(2**self.lut_bits - 1, dtype='int32')
            weight = (weight.unsqueeze(-1).cast('int32') >> shifts) & mask
        else:
            shifts = (paddle.arange(self.states-1, -1, -1) * self.state_bits).cast('int32')
            mask = paddle.to_tensor((2**self.lut_bits) - 1, dtype='int32') << shifts
            weight = (weight.unsqueeze(-1).cast('int32') & mask) >> shifts

        if self.parity_sign:
            weight = cal_sign(weight).cast('bfloat16')
        else:
            weight = weight.cast('bfloat16') + self._qmin
        
        # weight += self._qmin # This will result in numerical overflow
        weight = weight.reshape([weight.shape[0], -1]) # [group_nums, group_size]
        assert weight.shape[1] == self.group_size + self._pad_cols, f"{weight.shape[1]} != {self.group_size} + {self._pad_cols}"

        if weight.shape[1] > self.group_size:
            weight = weight[:, :self.group_size]
    
        return weight

    def dequant(self, use_triton=False):
        weight = self.weight.t()
        if self.packed:
            weight = self.unpack(weight, self.data_bits) # [out_features, -1]

        weight = weight.reshape([self._out_features, self._group_nums, self._state_nums])
        weight = weight.reshape([-1, self._state_nums])
        scales = self.scales.t().reshape([-1])
        if self.quant_scale:
            if self.redundant_bits > 0 and not self.enable_cluster:
                mask = 2**self.redundant_bits - 1
                scales = weight[:, -1] & paddle.to_tensor(mask, dtype=weight.dtype)
            super_scales = self.super_scales

        weight = self.decode(weight)
        
        if not self._symmetric:
            zero_points = self.zero_points.t().reshape([-1]) 
            # super_zero_points = self.super_zero_points

        if self.quant_scale:
            # dequant scales
            scales = scales.reshape([self._out_features, -1])
            if self.redundant_bits == 0 or self.enable_cluster:
                scales = self.unpack(scales, self.group_scale_bits)
            scales = (scales.cast('bfloat16') * self.super_scales.unsqueeze(-1)).reshape([-1]).cast('bfloat16')

        if self._symmetric:
            # dequant weight
            dequant_weight = weight.cast("bfloat16") * scales.unsqueeze(-1)
        else:
            dequant_weight = weight.cast('bfloat16') * scales.unsqueeze(-1) + zero_points.unsqueeze(-1)

        if self._isolate:
            dequant_weight = dequant_weight + self.outliers_weight.cast('bfloat16')
            # clear params
            self.outliers_weight.value().get_tensor()._clear()
        
        if self.hadamard:
            h = paddle.load('hadamard_matrix_64.pdtensors')
            if dequant_weight.shape[1] == self._in_features:
                dequant_weight = dequant_weight.reshape([-1, 64]).cast('float32') @ h.t() 
                dequant_weight = dequant_weight.reshape([self._out_features, -1])
            else:
                dequant_weight = dequant_weight.cast("float32") @ h.t() 
            dequant_weight = dequant_weight.cast('bfloat16')
            del h

        dequant_weight = dequant_weight.reshape([self._out_features, self._in_features])
        dequant_weight = dequant_weight.t()
        if self.extract_sign:
            dequant_weight *= self.weight_sign

        return dequant_weight

    def forward(self, x):
        if self.input_is_parallel or (not self.is_mp):
            input_parallel = x
        else:
            # split last dim
            input_parallel = mp_ops._c_split(x, group=self.model_parallel_group)

        if self.is_mp:
            with paddle.amp.auto_cast(enable=False):
                use_triton = False
                if use_triton and self.packed:
                    if self.quant_bits == 2:
                        output_parallel = weight_only_linear_int2_symm(input_parallel, self.weight, self.bias, self.scales)
                    elif self.quant_bits == 4:
                        output_parallel = weight_only_linear_int4_symm(input_parallel, self.weight, self.bias, self.scales)
                else:
                    # if self.dq_weight is None:
                        # self.dq_weight = self.dequant()
                    dq_weight = self.dequant()
                    output_parallel = F.linear(input_parallel, dq_weight)

                output_ = mp_ops._mp_allreduce(
                    output_parallel,
                    group=self.model_parallel_group,
                    use_calc_stream=True,
                    use_model_parallel=True,
                )
                output = output_ + self.bias if self.bias is not None else output_
                return output
        else:
            with paddle.amp.auto_cast(enable=False):
                dq_weight = self.dequant()
                output = F.linear(input_parallel, dq_weight, self.bias)
                return output
    

class TrellisQuantColumnParallelLinear(nn.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        lut_bits,
        states,
        state_bits,
        group_size=64,
        super_group_size=256,
        group_scale_bits=4,
        group_zp_bits=4,
        quant_scale=False,
        symmetric=True,
        pack=False,
        isolate_outliers=False,
        bias=None,
        hadamard=False,
        extract_sign=False,
        weight_sign=None,
        enable_cluster=False,
        redundant_bits=0,
        gather_output=True,
        mp_group=None,
        parity_sign=False,
        enable_norm=False,
        enable_perm=False,
        enable_circle=False,
        **kwargs,
    ):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self.lut_bits = lut_bits
        self.states = states
        self.state_bits = state_bits
        self.group_size = group_size
        self.super_group_size = super_group_size
        self.group_scale_bits = group_scale_bits
        self.group_zp_bits = group_zp_bits
        self.quant_scale = quant_scale
        self._symmetric = symmetric
        if self._symmetric:
            self._qmin = -(2 ** (self.lut_bits - 1))
            self._qmax = 2 ** (self.lut_bits - 1) - 1
        else:
            self._qmin = 0
            self._qmax = 2 ** self.lut_bits - 1
        self._thread_num = os.cpu_count()
        self.extract_sign = extract_sign
        if extract_sign:
            assert pack is False, "extract sign requires non packed"
            assert weight_sign is not None, "extract sign requires weight_sign"
        self.weight_sign = weight_sign

        self.packed = pack # only pack weight now

        if self.group_size == -1:
            self.group_size = self._in_features
        self._group_nums = self._in_features // self.group_size
        if isinstance(self.states, (list, tuple)):
            self.states_length = sum(self.states)
        else:
            self.states_length = self.states

        if self.group_size % self.states_length == 0:
            self._pad_cols = 0 
        else:
            self._pad_cols = self.states_length - self.group_size % self.states_length
        self._state_nums = (self.group_size + self._pad_cols) // self.states_length
        if isinstance(self.states, (list, tuple)):
            code_length = 0
            for i in range(len(self.states)):
                code_length += self.lut_bits + (self.states[i] - 1)*self.state_bits[i]
            data_bits = code_length
        else:
            data_bits = self.lut_bits + (self.states_length - 1)*self.state_bits
        if data_bits > 8:
            dtype = 'int16'
        else:
            dtype = 'int8'
        if enable_cluster:
            dtype = 'uint8'
        self.data_bits = data_bits
        if self.packed:
            # only consider pack to int8 format now
            pack_num = 8 // data_bits
            self.register_buffer(
                'weight',
                paddle.zeros([self._group_nums*self._state_nums//pack_num, self._out_features],
                            dtype=dtype),
            )
        else:
            self.register_buffer(
                'weight',
                paddle.zeros([self._group_nums*self._state_nums, self._out_features],
                            dtype=dtype),
            )
        if self.quant_scale:
            scale_dtype = 'uint8'
            scale_shape = [self._group_nums//(8//self.group_scale_bits), self._out_features]
        else:
            scale_dtype = 'bfloat16'
            scale_shape = [self._group_nums, self._out_features]
        self.register_buffer(
            'scales',
            paddle.zeros(scale_shape, dtype=scale_dtype),
        )

        if self.quant_scale:
            self.register_buffer(
                'super_scales',
                paddle.zeros([self._out_features], dtype='bfloat16'),
            )
        if not self._symmetric:
            self.register_buffer(
                'zero_points',
                paddle.zeros([self._group_nums], dtype=scale_dtype),
            )
            if self.quant_scale:
                self.register_buffer(
                    'super_zero_points',
                    paddle.zeros([self._group_nums//(self.super_group_size//self.group_size)],
                                dtype='bfloat16'),
                )

        self._isolate = isolate_outliers
        if self._isolate:
            self.register_buffer(
                'outliers_weight',
                paddle.zeros([group_nums, self.group_size], dtype='bfloat16'),
            )
        self.enable_cluster = enable_cluster
        if self.enable_cluster:
            self.register_buffer(
                'code_scale',
                paddle.zeros([self._out_features], dtype='float32'),
            )
            self.register_buffer(
                'code_zp',
                paddle.zeros([self._out_features], dtype='float32'),
            )
        self.redundant_bits = redundant_bits

        self.hadamard = hadamard
        self.enable_perm = enable_perm
        self.enable_norm = enable_norm
        self.enable_circle = enable_circle
        self.dq_weight = None
        self.bias = bias
        self.parity_sign = parity_sign
        self.tq_config = {
            'lut_bits': self.lut_bits,
            'states': self.states,
            'state_bits': self.state_bits,
            'group_size': self.group_size,
            'quant_scale': self.quant_scale,
            'redundant_bits': self.redundant_bits,
            'parity_sign': self.parity_sign,
            'pack': self.packed,
        }

        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group() if mp_group is None else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size() if mp_group is None else mp_group.nranks
        )
        self.is_mp = self.world_size > 1
        self.gather_output = gather_output
        # self.weight.is_distributed = True if self.is_mp else False
        # self.scales.is_distributed = True if self.is_mp else False

    def init_parameters(self, 
        quant_weight, 
        scales, 
        super_scales=None, 
        zero_points=None, 
        super_zero_points=None, 
        is_pack=False, 
        outliers_weight=None,
        norm_scale=None,
        norm_bias=None,
        perm=None,
        indices_scale=None,
        indices_zp=None,
        svd_bias=None,
        for_infer=True,
    ):
        if self.packed and not is_pack:
            # [total group nums, state nums] ->  [total group nums, state nums//pack nums]
            quant_weight = self.pack(quant_weight, self.data_bits)
        if not for_infer:
            # reshape is not need for load quant ckpt.
            # [total group nums, state nums] -> [group_nums * state_nums, out_features]
            quant_weight = quant_weight.reshape([self._out_features, -1]).t()
            if scales is not None:
                scales = scales.reshape([self._out_features, -1])
                if self.quant_scale and self.group_scale_bits < 8:
                    scales = self.pack(scales, self.group_scale_bits)
                scales = scales.t()
            if not self._symmetric:
                zero_points = zero_points.reshape([self._out_features, -1]).t()

        faster_set_value(quant_weight, self.weight)
        if scales is not None:
            faster_set_value(scales, self.scales)
        if not self._symmetric:
            faster_set_value(zero_points, self.zero_points)
        if self.quant_scale:
            faster_set_value(super_scales, self.super_scales)
        if self.enable_norm:
            faster_set_value(norm_scale, self.norm_scale)
            if norm_bias is not None:
                faster_set_value(norm_bias, self.norm_bias)
        if self.enable_perm:
            assert perm is not None
            self.perm = perm
            self.inv_perm = None
        if self.enable_cluster:
            faster_set_value(indices_scale, self.code_scale)
            faster_set_value(indices_zp, self.code_zp)

    def pack(self, src, bits):
        # pack parameters: use int32 to save multiple weights and scales
        pack_num = 8 // bits
        shift_bits = (paddle.arange(0, pack_num) * bits).cast('uint8')
        
        src = src.cast('uint8')
        if len(src.shape) == 2:
            row, col = src.shape
            src = src.reshape((row, col//pack_num, pack_num))
        else:
            src = src.reshape((src.shape[0]//pack_num, pack_num))
        src = src << shift_bits 
        return src.sum(axis=-1)
    
    def unpack(self, src, bits):
        pack_num = 8 // bits 
        bnt = paddle.to_tensor(2**bits - 1, dtype='uint8')
        shift_bits = (paddle.arange(0, pack_num) * bits).cast('uint8')

        if len(src.shape) == 2:
            row, col = src.shape
            src = src.reshape((row, col, 1)).expand((row, col, pack_num))
            src = src >> shift_bits & bnt
            src = src.reshape((row, col*pack_num))
        else:
            src = src.reshape((src.shape[0], 1)).expand((src.shape[0], pack_num))
            src = src >> shift_bits & bnt
            src = src.reshape([src.shape[0]*pack_num])
        return src

    def decode(self, weight):
        """
        weight: [group nums, state nums]
        """
        if self.enable_cluster:
            group_nums, state_nums = weight.shape
            weight = weight.reshape([self._out_features, -1])
            weight = (weight.cast('float32') * self.code_scale.unsqueeze([-1]) + self.code_zp.unsqueeze([-1])).round().cast('int32') # [out_features, -1]
            weight = weight.reshape([group_nums, state_nums])

        if self.enable_circle:
            mask = paddle.to_tensor(2**self.lut_bits - 1, dtype='int32')
            shifts = (paddle.arange(0, self.states, 1) * self.state_bits).cast('int32') % self.lut_bits
            weight = weight.unsqueeze(-1).cast('int32')
            weight = (weight >> shifts) | ((weight << (self.lut_bits - shifts)) & mask)
        elif isinstance(self.states, (list, tuple)):
            # for multi code format
            code_length = []
            for i in range(len(self.states)):
                code_length.append(self.lut_bits + self.state_bits[i] * (self.states[i] - 1))
            shifts = []
            for i in range(len(self.states)): # [3, 4] bf16 -> 13bits
                if i == len(self.states) - 1:
                    cur = paddle.arange(self.states[i]-1, -1, -1) * self.state_bits[i]
                else:
                    cur = paddle.arange(self.states[i]-1, -1, -1) * self.state_bits[i] + code_length[i+1]
                shifts.append(cur)
                # [13, 11, 9, 6, 4, 2, 0]
            shifts = paddle.concat(shifts, axis=0).cast('int32')
            mask = paddle.to_tensor(2**self.lut_bits - 1, dtype='int32')
            weight = (weight.unsqueeze(-1).cast('int32') >> shifts) & mask
        else:
            shifts = (paddle.arange(self.states-1, -1, -1) * self.state_bits).cast('int32')
            mask = paddle.to_tensor((2**self.lut_bits) - 1, dtype='int32') << shifts
            weight = (weight.unsqueeze(-1).cast('int32') & mask) >> shifts
        

        if self.parity_sign:
            weight = cal_sign(weight).cast('bfloat16')
        else:
            weight = weight.cast('bfloat16') + self._qmin
        
        # weight += self._qmin # This will result in numerical overflow
        weight = weight.reshape([weight.shape[0], -1]) # [group_nums, group_size]
        assert weight.shape[1] == self.group_size + self._pad_cols, f"{weight.shape[1]} != {self.group_size} + {self._pad_cols}"

        if weight.shape[1] > self.group_size:
            weight = weight[:, :self.group_size]
    
        return weight

    def dequant(self, use_triton=False):
        weight = self.weight.t()
        if self.packed:
            weight = self.unpack(weight, self.data_bits) # [out_features, -1]

        weight = weight.reshape([self._out_features, self._group_nums, self._state_nums])
        weight = weight.reshape([-1, self._state_nums])
        scales = self.scales.t().reshape([-1])
        if self.quant_scale:
            if self.redundant_bits > 0 and not self.enable_cluster:
                mask = 2**self.redundant_bits - 1
                scales = weight[:, -1] & paddle.to_tensor(mask, dtype=weight.dtype)
            super_scales = self.super_scales

        weight = self.decode(weight)
        
        if not self._symmetric:
            zero_points = self.zero_points.t().reshape([-1]) 
            # super_zero_points = self.super_zero_points

        if self.quant_scale:
            # dequant scales
            scales = scales.reshape([self._out_features, -1])
            if self.redundant_bits == 0 or self.enable_cluster:
                scales = self.unpack(scales, self.group_scale_bits)
            scales = (scales.cast('bfloat16') * self.super_scales.unsqueeze(-1)).reshape([-1]).cast('bfloat16')

        if self._symmetric:
            # dequant weight
            dequant_weight = weight.cast("bfloat16") * scales.unsqueeze(-1)
        else:
            dequant_weight = weight.cast('bfloat16') * scales.unsqueeze(-1) + zero_points.unsqueeze(-1)

        if self._isolate:
            dequant_weight = dequant_weight + self.outliers_weight.cast('bfloat16')
            # clear params
            self.outliers_weight.value().get_tensor()._clear()
        
        if self.hadamard:
            h = paddle.load('hadamard_matrix_64.pdtensors')
            if dequant_weight.shape[1] == self._in_features:
                dequant_weight = dequant_weight.reshape([-1, 64]).cast('float32') @ h.t() 
                dequant_weight = dequant_weight.reshape([self._out_features, -1])
            else:
                dequant_weight = dequant_weight.cast("float32") @ h.t() 
            dequant_weight = dequant_weight.cast('bfloat16')
            del h

        dequant_weight = dequant_weight.reshape([self._out_features, self._in_features])
        dequant_weight = dequant_weight.t()
        if self.extract_sign:
            dequant_weight *= self.weight_sign

        return dequant_weight

    def forward(self, x):
        if self.is_mp:
            input_parallel = mp_ops._c_identity(x, group=self.model_parallel_group)
        else:
            input_parallel = x

        with paddle.amp.auto_cast(enable=False):
            use_triton = False
            if use_triton and self.packed:
                if self.quant_bits == 2:
                    output = weight_only_linear_int2_symm(input_parallel, self.weight, self.bias, self.scales)
                elif self.quant_bits == 4:
                    output = weight_only_linear_int4_symm(input_parallel, self.weight, self.bias, self.scales)
            else:
                # if self.dq_weight is None:
                    # self.dq_weight = self.dequant()
                dq_weight = self.dequant()
                output = F.linear(input_parallel, dq_weight, self.bias)

            if self.gather_output and self.is_mp:
                output = mp_ops._c_concat(output, group=self.model_parallel_group)
            
            return output

def faster_set_value(src_tensor, param):
    assert src_tensor.shape == param.shape, f"{src_tensor.shape} != {param.shape}"
    # set src_tensor to param
    if src_tensor.dtype != param.dtype:
        src_tensor = src_tensor.astype(param.dtype)
    
    dst_tensor = param.value().get_tensor()
    place = param.place

    if not src_tensor.place._equals(place):
        # clear dst_tensor for save memory
        dst_tensor._clear()
        # v_new = v_new._copy_to(paddle.CUDAPinnedPlace(), False)
        new_t = src_tensor._copy_to(place, False)
    else:
        new_t = src_tensor
    
    dst_tensor._share_data_with(new_t.value().get_tensor())