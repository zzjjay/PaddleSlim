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
# from .wintx import weight_only_linear_int2_symm, weight_only_linear_int4_symm
class IQuantLinear(nn.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        quant_bits=2,
        group_size=32,
        super_group_size=256,
        group_scale_bits=4,
        group_zp_bits=4,
        quant_scale=True,
        symmetric=True,
        pack=False,
        isolate_outliers=False,
        bias=None,
        hadamard=False,
        extract_sign=False,
        weight_sign=None,
        **kwargs,
    ):
        super().__init__()
        self.quant_bits = quant_bits
        self.group_size = group_size
        self.super_group_size = super_group_size
        self.group_scale_bits = group_scale_bits
        self.group_zp_bits = group_zp_bits
        self.quant_scale = quant_scale
        self._symmetric = symmetric
        self._thread_num = os.cpu_count()
        self.extract_sign = extract_sign
        if extract_sign:
            assert pack is False, "extract sign requires non packed"
            assert weight_sign is not None, "extract sign requires weight_sign"
        self.weight_sign = weight_sign
        self._in_features = in_features
        self._out_features = out_features

        self.packed = pack
        if pack:
            assert (quant_bits == 2 or quant_bits == 4), 'only pack in 2 bits or 4 bits.'
        group_nums = self._in_features // self.group_size * self._out_features 
        if not self.packed:
            self.register_buffer(
                'weight',
                paddle.zeros([group_nums, self.group_size],
                            dtype='int8'),
            )
            scale_dtype = 'int8' if self.quant_scale else 'bfloat16'
            self.register_buffer(
                'scales',
                paddle.zeros([group_nums], dtype=scale_dtype),
            )
            if self.quant_scale:
                self.register_buffer(
                    'super_scales',
                    paddle.zeros([group_nums//(self.super_group_size//self.group_size)],
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
        else:
            # packing parameters: use int32 to save multiple weights and scales
            # only consider pack 2bits weight for now
            pack_num = 32 // quant_bits # for weight
            self.register_buffer(
                'weight',
                paddle.zeros([self._in_features//pack_num, self._out_features],
                            dtype='int32'),
            )
            if self.quant_scale:
                pack_num = 32 // self.group_scale_bits # for scale
                self.register_buffer(
                    'scales',
                    paddle.zeros([group_nums//pack_num], dtype='int32'),
                )
                self.register_buffer(
                    'super_scales',
                    paddle.zeros([group_nums//(self.super_group_size//self.group_size)],
                                dtype='bfloat16'),
                )
            else:
                self.register_buffer(
                    'scales',
                    paddle.zeros([self._in_features//self.group_size, self._out_features], dtype='bfloat16'),
                )
            if not self._symmetric:
                self.register_buffer(
                    'zero_points',
                    paddle.zeros([group_nums//pack_num], dtype='int32'),
                )
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

        self.hadamard = hadamard
        self.dq_weight = None
        self.bias = bias
        # logger.info(f"quant_scale: {self.quant_scale}, pack:{self.packed}")
        self.iq_config = {
            'quant_bits': self.quant_bits,
            'group_size': self.group_size,
            'pack': self.packed,
            'quant_scale': self.quant_scale,
        }

    def init_parameters(self, quant_weight, scales, super_scales=None, zero_points=None, super_zero_points=None, is_pack=False, outliers_weight=None):
        if self.packed and not is_pack:
            # pack parameters
            # [group nums, group size] -> [out_features, in_features]
            quant_weight = quant_weight.reshape([self._out_features, self._in_features])
            # transfer int2 to uint2
            quant_weight += 2**(self.quant_bits - 1) # 2
            quant_weight = self.pack(quant_weight, self.quant_bits) # [out_features, in_features//pack_num]
            quant_weight = quant_weight.t() # [in_features//pack_num, out_features]
            if self.quant_scale:
                # scales: [group nums] 
                scales = self.pack(scales, self.group_scale_bits)
                if not self._symmetric:
                    zero_points = self.pack(zero_points, self.group_zp_bits)

            # [group nums] -> [in_features//group_size, out_features]
            # scales = scales.reshape([-1, self._out_features])

            scales = scales.reshape([self._out_features, -1]).t()

        faster_set_value(quant_weight, self.weight)
        faster_set_value(scales, self.scales)
        if self.quant_scale:
            faster_set_value(super_scales, self.super_scales)
        '''
        self.weight.set_value(quant_weight.cast(self.weight.dtype))
        self.scales.set_value(scales.cast(self.scales.dtype))
        if self.quant_scale:
            self.super_scales.set_value(super_scales.cast(self.super_scales.dtype))
        if not self._symmetric:
            self.zero_points.set_value(zero_points.cast(self.zero_points.dtype))
            if self.quant_scale:
                self.super_zero_points.set_value(super_zero_points.cast(self.super_zero_points.dtype))
        if outliers_weight is not None:
            self.outliers_weight.set_value(outliers_weight.cast(self.outliers_weight.dtype))
        '''

    def pack(self, src, bits):
        # pack parameters: use int32 to save multiple weights and scales
        pack_num = 32 // bits
        shift_bits = (paddle.arange(0, pack_num) * bits).cast('int32')
        
        src = src.cast('int32')
        if len(src.shape) == 2:
            row, col = src.shape
            src = src.reshape((row, col//pack_num, pack_num))
        else:
            src = src.reshape((src.shape[0]//pack_num, pack_num))
        src = src << shift_bits 
        return src.sum(axis=-1)
    
    def unpack(self, src, bits):
        pack_num = 32 // bits 
        bnt = paddle.to_tensor(2**bits - 1, dtype='int32')
        shift_bits = (paddle.arange(0, pack_num) * bits).cast('int32')

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

    def dequant(self, use_triton=False):
        if self.packed:
            weight = self.unpack(self.weight.t(), self.quant_bits)

            # uint2 -> int2
            weight -= 2**(self.quant_bits - 1) 

            # [out_features, in_features//pack_num, pack_num] -> [group_nums, group_size] [* , 32]
            weight = weight.reshape([self._out_features, self._in_features]).reshape([-1, self.group_size])

            if self.quant_scale:
                # scales: [group nums] -> [group_nums//pack_num, pack_num]
                scales = self.unpack(self.scales, self.group_scale_bits)
                # [group_nums//pack_num, pack_num] -> [group_nums]
                scales = scales.reshape([-1])
            else:
                scales = self.scales.t().reshape([-1]) # [group_nums]

        else:
            weight = self.weight
            scales = self.scales
            if self.quant_scale:
                super_scales = self.super_scales
            if not self._symmetric:
                zero_points = self.zero_points
                super_zero_points = self.super_zero_points

        step = self.super_group_size // self.group_size

        if self.quant_scale:
            # dequant group scales
            # [group nums, group size] -> [super group nums, 8, 32]
            scales = scales.reshape([-1, step]).t().cast("bfloat16") 
            assert scales.shape[-1] == self.super_scales.shape[0]
            scales = (scales * self.super_scales).t().reshape([-1]) # [group_nums]

        if self._symmetric:
            # dequant weight
            dequant_weight = weight.cast("bfloat16") * scales.unsqueeze(-1)
        else:
            # dequant group zero points
            zero_points = self.zero_points.reshape([-1, step]).t().cast("bfloat16") 
            assert zero_points.shape[-1] == self.super_zero_points.shape[0]
            dequant_zero_points = (zero_points * self.super_zero_points).t().reshape([-1])

            # dequant weight
            dequant_weight = weight.cast('bfloat16') * scales.unsqueeze(-1) + dequant_zero_points.unsqueeze(-1)

        if self._isolate:
            dequant_weight = dequant_weight + self.outliers_weight.cast('bfloat16')
            # clear params
            self.outliers_weight.value().get_tensor()._clear()
        
        if self.hadamard:
            h = paddle.load('hadamard_matrix_32.pdtensors')
            dequant_weight = dequant_weight.cast("float32") @ h.t() 
            dequant_weight = dequant_weight.cast('bfloat16')
            del h

        dequant_weight = dequant_weight.reshape([self._out_features, self._in_features]).t()
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
        res = F.linear(x, dq_weight, self.bias)
        return res
    

class IQuantRowParallelLinear(nn.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        quant_bits=2,
        group_size=32,
        super_group_size=256,
        group_scale_bits=4,
        group_zp_bits=4,
        quant_scale=True,
        symmetric=True,
        pack=False,
        isolate_outliers=False,
        bias=None,
        hadamard=False,
        extract_sign=False,
        weight_sign=None,
        input_is_parallel=False,
        mp_group=None,
        **kwargs,
    ):
        super().__init__()
        self.quant_bits = quant_bits
        self.group_size = group_size
        self.super_group_size = super_group_size
        self.group_scale_bits = group_scale_bits
        self.group_zp_bits = group_zp_bits
        self.quant_scale = quant_scale
        self._symmetric = symmetric
        self._thread_num = os.cpu_count()
        self.extract_sign = extract_sign
        if extract_sign:
            assert pack is False, "extract sign requires non packed"
            assert weight_sign is not None, "extract sign requires weight_sign"
        self.weight_sign = weight_sign
        self._in_features = in_features
        self._out_features = out_features

        self.packed = pack
        if pack:
            assert (quant_bits == 2 or quant_bits == 4), 'only pack in 2 bits or 4 bits.'
        group_nums = self._in_features // self.group_size * self._out_features 
        if not self.packed:
            self.register_buffer(
                'weight',
                paddle.zeros([group_nums, self.group_size],
                            dtype='int8'),
            )
            scale_dtype = 'int8' if self.quant_scale else 'bfloat16'
            self.register_buffer(
                'scales',
                paddle.zeros([group_nums], dtype=scale_dtype),
            )
            if self.quant_scale:
                self.register_buffer(
                    'super_scales',
                    paddle.zeros([group_nums//(self.super_group_size//self.group_size)],
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
        else:
            # packing parameters: use int32 to save multiple weights and scales
            # only consider pack 2bits weight for now
            pack_num = 32 // quant_bits # for weight
            self.register_buffer(
                'weight',
                paddle.zeros([self._in_features//pack_num, self._out_features],
                            dtype='int32'),
            )
            if self.quant_scale:
                pack_num = 32 // self.group_scale_bits # for scale
                self.register_buffer(
                    'scales',
                    paddle.zeros([group_nums//pack_num], dtype='int32'),
                )
                self.register_buffer(
                    'super_scales',
                    paddle.zeros([group_nums//(self.super_group_size//self.group_size)],
                                dtype='bfloat16'),
                )
            else:
                self.register_buffer(
                    'scales',
                    paddle.zeros([self._in_features//self.group_size, self._out_features], dtype='bfloat16'),
                )
            if not self._symmetric:
                self.register_buffer(
                    'zero_points',
                    paddle.zeros([group_nums//pack_num], dtype='int32'),
                )
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

        self.hadamard = hadamard
        self.dq_weight = None
        self.bias = bias
        # logger.info(f"quant_scale: {self.quant_scale}, pack:{self.packed}")

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

    def init_parameters(self, quant_weight, scales, super_scales=None, zero_points=None, super_zero_points=None, is_pack=False, outliers_weight=None):
        if self.packed and not is_pack:
            # pack parameters
            # [group nums, group size] -> [out_features, in_features]
            quant_weight = quant_weight.reshape([self._out_features, self._in_features])
            # transfer int2 to uint2
            quant_weight += 2**(self.quant_bits - 1) # 2
            quant_weight = self.pack(quant_weight, self.quant_bits) # [out_features, in_features//pack_num]
            quant_weight = quant_weight.t() # [in_features//pack_num, out_features]
            if self.quant_scale:
                # scales: [group nums] 
                scales = self.pack(scales, self.group_scale_bits)
                if not self._symmetric:
                    zero_points = self.pack(zero_points, self.group_zp_bits)

            # [group nums] -> [out_features, in_features//group_size]
            scales = scales.reshape([self._out_features, -1]).t()
        
        faster_set_value(quant_weight, self.weight)
        faster_set_value(scales, self.scales)
        if self.quant_scale:
            faster_set_value(super_scales, self.super_scales)

        '''
        self.weight.set_value(quant_weight.cast(self.weight.dtype))
        self.scales.set_value(scales.cast(self.scales.dtype))
        if self.quant_scale:
            self.super_scales.set_value(super_scales.cast(self.super_scales.dtype))
        if not self._symmetric:
            self.zero_points.set_value(zero_points.cast(self.zero_points.dtype))
            if self.quant_scale:
                self.super_zero_points.set_value(super_zero_points.cast(self.super_zero_points.dtype))
        if outliers_weight is not None:
            self.outliers_weight.set_value(outliers_weight.cast(self.outliers_weight.dtype))
        '''

    def pack(self, src, bits):
        # pack parameters: use int32 to save multiple weights and scales
        pack_num = 32 // bits
        shift_bits = (paddle.arange(0, pack_num) * bits).cast('int32')
        
        src = src.cast('int32')
        if len(src.shape) == 2:
            row, col = src.shape
            src = src.reshape((row, col//pack_num, pack_num))
        else:
            src = src.reshape((src.shape[0]//pack_num, pack_num))
        src = src << shift_bits 
        return src.sum(axis=-1)
    
    def unpack(self, src, bits):
        pack_num = 32 // bits 
        bnt = paddle.to_tensor(2**bits - 1, dtype='int32')
        shift_bits = (paddle.arange(0, pack_num) * bits).cast('int32')

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

    def dequant(self):
        if self.packed:
            weight = self.unpack(self.weight.t(), self.quant_bits)

            # uint2 -> int2
            weight -= 2**(self.quant_bits - 1) 

            # [out_features, in_features//pack_num, pack_num] -> [group_nums, group_size] [* , 32]
            weight = weight.reshape([self._out_features, self._in_features]).reshape([-1, self.group_size])

            if self.quant_scale:
                # scales: [group nums] -> [group_nums//pack_num, pack_num]
                scales = self.unpack(self.scales, self.group_scale_bits)
                # [group_nums//pack_num, pack_num] -> [group_nums]
                scales = scales.reshape([-1])
            else:
                scales = self.scales.t().reshape([-1]) # [group_nums]
        else:
            weight = self.weight
            scales = self.scales
            if self.quant_scale:
                super_scales = self.super_scales
            if not self._symmetric:
                zero_points = self.zero_points
                super_zero_points = self.super_zero_points

        step = self.super_group_size // self.group_size

        if self.quant_scale:
            # dequant group scales
            # [group nums, group size] -> [super group nums, 8, 32]
            scales = scales.reshape([-1, step]).t().cast("bfloat16") 
            assert scales.shape[-1] == self.super_scales.shape[0]
            scales = (scales * self.super_scales).t().reshape([-1]) # [group_nums]

        if self._symmetric:
            # dequant weight
            dequant_weight = weight.cast("bfloat16") * scales.unsqueeze(-1)
        else:
            # dequant group zero points
            zero_points = self.zero_points.reshape([-1, step]).t().cast("bfloat16") 
            assert zero_points.shape[-1] == self.super_zero_points.shape[0]
            dequant_zero_points = (zero_points * self.super_zero_points).t().reshape([-1])

            # dequant weight
            dequant_weight = weight.cast('bfloat16') * scales.unsqueeze(-1) + dequant_zero_points.unsqueeze(-1)

        if self._isolate:
            dequant_weight = dequant_weight + self.outliers_weight.cast('bfloat16')
            # clear params
            self.outliers_weight.value().get_tensor()._clear()
        
        if self.hadamard:
            h = paddle.load('hadamard_matrix_32.pdtensors')
            dequant_weight = dequant_weight.cast("float32") @ h.t() 
            dequant_weight = dequant_weight.cast('bfloat16')
            del h

        dequant_weight = dequant_weight.reshape([self._out_features, self._in_features]).t()
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
                    if self.dq_weight is None:
                        self.dq_weight = self.dequant()
                    dq_weight = self.dq_weight
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
                # if self.dq_weight is None:
                #     self.dq_weight = self.dequant()
                dq_weight = self.dequant()
                output = F.linear(input_parallel, dq_weight, self.bias)
                return output
    

class IQuantColumnParallelLinear(nn.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        quant_bits=2,
        group_size=32,
        super_group_size=256,
        group_scale_bits=4,
        group_zp_bits=4,
        quant_scale=True,
        symmetric=True,
        pack=False,
        isolate_outliers=False,
        bias=None,
        hadamard=False,
        extract_sign=False,
        weight_sign=None,
        gather_output=True,
        mp_group=None,
        **kwargs,
    ):
        super().__init__()
        self.quant_bits = quant_bits
        self.group_size = group_size
        self.super_group_size = super_group_size
        self.group_scale_bits = group_scale_bits
        self.group_zp_bits = group_zp_bits
        self.quant_scale = quant_scale
        self._symmetric = symmetric
        self._thread_num = os.cpu_count()
        self.extract_sign = extract_sign
        if extract_sign:
            assert pack is False, "extract sign requires non packed"
            assert weight_sign is not None, "extract sign requires weight_sign"
        self.weight_sign = weight_sign
        self._in_features = in_features
        self._out_features = out_features

        self.packed = pack
        if pack:
            assert (quant_bits == 2 or quant_bits == 4), 'only pack in 2 bits or 4 bits.'
        group_nums = self._in_features // self.group_size * self._out_features 
        if not self.packed:
            self.register_buffer(
                'weight',
                paddle.zeros([group_nums, self.group_size],
                            dtype='int8'),
            )
            scale_dtype = 'int8' if self.quant_scale else 'bfloat16'
            self.register_buffer(
                'scales',
                paddle.zeros([group_nums], dtype=scale_dtype),
            )
            if self.quant_scale:
                self.register_buffer(
                    'super_scales',
                    paddle.zeros([group_nums//(self.super_group_size//self.group_size)],
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
        else:
            # packing parameters: use int32 to save multiple weights and scales
            # only consider pack 2bits weight for now
            pack_num = 32 // quant_bits # for weight
            self.register_buffer(
                'weight',
                paddle.zeros([self._in_features//pack_num, self._out_features],
                            dtype='int32'),
            )
            if self.quant_scale:
                pack_num = 32 // self.group_scale_bits # for scale
                self.register_buffer(
                    'scales',
                    paddle.zeros([group_nums//pack_num], dtype='int32'),
                )
                self.register_buffer(
                    'super_scales',
                    paddle.zeros([group_nums//(self.super_group_size//self.group_size)],
                                dtype='bfloat16'),
                )
            else:
                self.register_buffer(
                    'scales',
                    paddle.zeros([self._in_features//self.group_size, self._out_features], dtype='bfloat16'),
                )
            if not self._symmetric:
                self.register_buffer(
                    'zero_points',
                    paddle.zeros([group_nums//pack_num], dtype='int32'),
                )
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

        self.hadamard = hadamard
        self.dq_weight = None
        self.bias = bias
        # logger.info(f"quant_scale: {self.quant_scale}, pack:{self.packed}")

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

    def init_parameters(self, quant_weight, scales, super_scales=None, zero_points=None, super_zero_points=None, is_pack=False, outliers_weight=None):
        if self.packed and not is_pack:
            # pack parameters
            # [group nums, group size] -> [out_features, in_features]
            quant_weight = quant_weight.reshape([self._out_features, self._in_features])
            # transfer int2 to uint2
            quant_weight += 2**(self.quant_bits - 1) # 2
            quant_weight = self.pack(quant_weight, self.quant_bits) # [out_features, in_features//pack_num]
            quant_weight = quant_weight.t() # [in_features//pack_num, out_features]
            if self.quant_scale:
                # scales: [group nums] 
                scales = self.pack(scales, self.group_scale_bits)
                if not self._symmetric:
                    zero_points = self.pack(zero_points, self.group_zp_bits)

            # [group nums] -> [out_features, in_features//group_size]
            scales = scales.reshape([self._out_features, -1]).t()

        faster_set_value(quant_weight, self.weight)
        faster_set_value(scales, self.scales)
        if self.quant_scale:
            faster_set_value(super_scales, self.super_scales)
        '''
        self.weight.set_value(quant_weight.cast(self.weight.dtype))
        self.scales.set_value(scales.cast(self.scales.dtype))
        if self.quant_scale:
            self.super_scales.set_value(super_scales.cast(self.super_scales.dtype))
        if not self._symmetric:
            self.zero_points.set_value(zero_points.cast(self.zero_points.dtype))
            if self.quant_scale:
                self.super_zero_points.set_value(super_zero_points.cast(self.super_zero_points.dtype))
        if outliers_weight is not None:
            self.outliers_weight.set_value(outliers_weight.cast(self.outliers_weight.dtype))
        '''

    def pack(self, src, bits):
        # pack parameters: use int32 to save multiple weights and scales
        pack_num = 32 // bits
        shift_bits = (paddle.arange(0, pack_num) * bits).cast('int32')
        
        src = src.cast('int32')
        if len(src.shape) == 2:
            row, col = src.shape
            src = src.reshape((row, col//pack_num, pack_num))
        else:
            src = src.reshape((src.shape[0]//pack_num, pack_num))
        src = src << shift_bits 
        return src.sum(axis=-1)
    
    def unpack(self, src, bits):
        pack_num = 32 // bits 
        bnt = paddle.to_tensor(2**bits - 1, dtype='int32')
        shift_bits = (paddle.arange(0, pack_num) * bits).cast('int32')

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

    def dequant(self):
        if self.packed:
            weight = self.unpack(self.weight.t(), self.quant_bits)

            # uint2 -> int2
            weight -= 2**(self.quant_bits - 1) 

            # [out_features, in_features//pack_num, pack_num] -> [group_nums, group_size] [* , 32]
            weight = weight.reshape([self._out_features, self._in_features]).reshape([-1, self.group_size])

            if self.quant_scale:
                # scales: [group nums] -> [group_nums//pack_num, pack_num]
                scales = self.unpack(self.scales, self.group_scale_bits)
                # [group_nums//pack_num, pack_num] -> [group_nums]
                scales = scales.reshape([-1])
            else:
                scales = self.scales.t().reshape([-1]) # [group_nums]
        else:
            weight = self.weight
            scales = self.scales
            if self.quant_scale:
                super_scales = self.super_scales
            if not self._symmetric:
                zero_points = self.zero_points
                super_zero_points = self.super_zero_points

        step = self.super_group_size // self.group_size

        if self.quant_scale:
            # dequant group scales
            # [group nums, group size] -> [super group nums, 8, 32]
            scales = scales.reshape([-1, step]).t().cast("bfloat16") 
            assert scales.shape[-1] == self.super_scales.shape[0]
            scales = (scales * self.super_scales).t().reshape([-1]) # [group_nums]

        if self._symmetric:
            # dequant weight
            dequant_weight = weight.cast("bfloat16") * scales.unsqueeze(-1)
        else:
            # dequant group zero points
            zero_points = self.zero_points.reshape([-1, step]).t().cast("bfloat16") 
            assert zero_points.shape[-1] == self.super_zero_points.shape[0]
            dequant_zero_points = (zero_points * self.super_zero_points).t().reshape([-1])

            # dequant weight
            dequant_weight = weight.cast('bfloat16') * scales.unsqueeze(-1) + dequant_zero_points.unsqueeze(-1)

        if self._isolate:
            dequant_weight = dequant_weight + self.outliers_weight.cast('bfloat16')
            # clear params
            self.outliers_weight.value().get_tensor()._clear()
        
        if self.hadamard:
            h = paddle.load('hadamard_matrix_32.pdtensors')
            dequant_weight = dequant_weight.cast("float32") @ h.t() 
            dequant_weight = dequant_weight.cast('bfloat16')
            del h

        dequant_weight = dequant_weight.reshape([self._out_features, self._in_features]).t()
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
                #     self.dq_weight = self.dequant()
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