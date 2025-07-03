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
"""
Trellis Quant.
"""
import os
import gc
import math
import traceback
import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import ColumnParallelLinear, RowParallelLinear
from .utils import find_parent_layer_and_sub_name
from paddlenlp.utils.log import logger
from .iquant_utils import IQuantizer, read_file, save_file
from ..layers.trellis_quant_layers import TrellisQuantLinear, TrellisQuantRowParallelLinear, TrellisQuantColumnParallelLinear
from .trellis_utils import *
import concurrent.futures
from paddle.distributed.fleet.layers.mpu import mp_ops
from multiprocessing import Pool
__all__ = ['TrellisQuant']
    

class TrellisQuant:
    """
    TrellisQuant.
    """
    def __init__(self,
                 model,
                 lut_bits=4,
                 states=4,
                 state_bits=2,
                 group_size=64,
                 super_group_size=512,
                 group_scale_bits=4,
                 quant_scale=False,
                 target_layers=None,
                 collect_hessian=False,
                 is_deepseekv3=False,
                 hook_targets=None,
                 enable_cluster=False,
                 magR=False,
                 IQ=None,
        ):
        super(TrellisQuant, self).__init__()
        self.model = model
        self.lut_bits = lut_bits
        self.states = states
        self.state_bits = state_bits
        
        self.group_size = group_size
        self.super_group_size = super_group_size
        self.group_scale_bits = group_scale_bits
        self.quant_scale = quant_scale
        self._target_layers = target_layers
        self.save_state_dict = {}
        self.collect_hessian = collect_hessian
        self.is_deepseekv3 = is_deepseekv3
        self._hook_targets = hook_targets
        self._forward_hook_list = [] 
        self.sampled_hessian = {}
        self.sampled_means = {}
        self.sampled_num = {}
        if collect_hessian:
            self._apply_hook()
        logger.info(f"[TQ] group_size: {group_size}, \
                lut_bits: {lut_bits}, states: {states}, state_bits: {state_bits}, \
                super_group_size: {super_group_size}, \
                group_scale_bits: {group_scale_bits}, \
                quant_scale: {quant_scale}, \
                collect_hessian: {collect_hessian}, \
                enable_cluster: {enable_cluster}")
        self.enable_cluster = enable_cluster
        self.IQ = IQ
        self.magR = magR

    def _apply_hook(self):
        self._forward_hook_list = []
        for cur_name, sub_layer in self.model.named_sublayers():
            if isinstance(sub_layer, (ColumnParallelLinear, RowParallelLinear, paddle.nn.Linear)):
                skip = False
                if self._hook_targets is not None:
                    for key in self._hook_targets:
                        if key in cur_name:
                            skip = False
                            break
                        else:
                            skip = True
                if skip:
                    continue
                forward_pre_hook_handle = sub_layer.register_forward_pre_hook(
                    self._forward_pre_hook)
                self._forward_hook_list.append(forward_pre_hook_handle)

    def _forward_pre_hook(self, layer, input):
        weight = layer.weight.t() # [out_features, in_features]
        self._collect_hessian(input, layer.full_name(), weight.shape[1], layer)
        return input
    
    def _collect_hessian(self, input, layer_name, columns, layer=None):
        inp = input[0] if type(input) == tuple else input
        if self.is_deepseekv3:
            if isinstance(layer, RowParallelLinear) and layer.input_is_parallel==False:
                inp = mp_ops._c_split(inp, group=layer.model_parallel_group)
        inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.cast_('float32')
        if self.magR:
            logger.info(f"[MagR] {layer.full_name()}")
            self.magR_weight(layer.weight.cast('float32'), inp, layer)

        if layer_name not in self.sampled_hessian:
            self.sampled_hessian[layer_name] = paddle.zeros(
                    (columns, columns), dtype='float32')
            self.sampled_num[layer_name] = 0
            # self.sampled_means[layer_name] = paddle.zeros(
            #         (columns), dtype='float32')

        hessian = self.sampled_hessian[layer_name]
        self.sampled_num[layer_name] += inp.shape[0]
        hessian += paddle.matmul(inp.t(), inp)
        self.sampled_hessian[layer_name] = hessian.cpu()
        # self.sampled_means[layer_name] += inp.sum(axis=0) #.cpu()
        del inp, hessian
    
    def _remove_hook(self):
        for hook in self._forward_hook_list:
            hook.remove()
        self._forward_hook_list = [] 

    def magR_weight(self, weight, inputs, layer):
        def test(weight):
            w_max = weight.reshape([-1, self.group_size]).max(axis=0)
            logger.info(f"max: {w_max.max().item()}, min: {w_max.min().item()}, mean: {w_max.mean().item()}")
        logger.info("Before MagR...")
        test(weight)
        weight = W_proximal_preprocess_groupwise(weight, inputs, n_iters=200, group_size=self.group_size)
        logger.info("After MagR...")
        test(weight)
        layer.weight.set_value(weight.cast(layer.weight.dtype))

    @paddle.no_grad()
    def trellis_quantize(self, hessian_file):
        logger.info(f"Loading Hessian Matrix from {hessian_file}")
        paddle.set_device('cpu')
        if os.path.exists(hessian_file):
            self.sampled_hessian = read_file(hessian_file)
        else:
            self.sampled_hessian = {}
        paddle.set_device(f'gpu')
        lookup_table_dict = {}
        def _quantize_layer(model_state, layer_name_dict, sampled_hessian, target_layers):
        
            save_state = {}
            for cur_name in layer_name_dict:
                layer_name = layer_name_dict[cur_name]
                if layer_name in target_layers:
                    # if not ('mlp' in cur_name or 'mlp.w2' in cur_name):
                    #     continue
                    layer_idx = int(cur_name.split(".")[2])
    
                    logger.info(f"[TQ] quantize: {cur_name} --- {layer_name}")

                    layer_idx = int(cur_name.split(".")[2])
                    lut_bits, states, state_bits = self.lut_bits, self.states, self.state_bits
                    symmetric = True
                    isolate = False
                    hadamard = False
                    enable_norm = False
                    enable_perm = False
                    group_size = self.group_size
                    enable_ldlq = False  # GPTQ
                    enable_cluster = self.enable_cluster
                    extract_sign = False
                    quant_scale = self.quant_scale
                    group_scale_bits = 4 # self.group_scale_bits
                    parity_sign = False
                    pack = False
                    enable_razor = False
                    enable_completion = False
                    enable_compensation = False
                    enable_circle = False
                    # lut_bits = 8
                    # states = 4
                    # state_bits = 6
                    
                    # lut_bits = 3
                    # states = [3, 4]
                    # state_bits = [2, 2]

                    '''
                    if 'k_proj' in cur_name or 'v_proj' in cur_name or 'shared_expert' in cur_name:
                        lut_bits = 6
                        states = 2
                        state_bits = 4
                    elif 'q_proj' in cur_name or 'o_proj' in cur_name:
                        lut_bits = 5
                        states = 4
                        state_bits = 3
                    else:
                        # for experts
                        if layer_idx < 6:
                            lut_bits = 5
                            states = 4
                            state_bits = 3
                        else:
                            lut_bits = 4
                            states = 4
                            state_bits = 2
                    '''

                    if isinstance(states, list):
                        table_key = (lut_bits, tuple(states), tuple(state_bits))
                    else:
                        table_key = (lut_bits, states, state_bits)

                    if table_key in lookup_table_dict:
                        lookup_table = lookup_table_dict[table_key]
                    elif enable_circle:
                        lookup_table = construct_lookup_table_circle(lut_bits, states, state_bits)
                    elif isinstance(table_key[1], tuple):
                        lookup_table = construct_lookup_table_multi2(lut_bits, states, state_bits)
                        lookup_table_dict[table_key] = lookup_table
                    else:
                        lookup_table = construct_lookup_table(lut_bits, states, state_bits)
                        lookup_table_dict[table_key] = lookup_table

                    weight_quanter = TrellisQuantizer(
                        lut_bits=lut_bits,
                        states=states,
                        state_bits=state_bits,
                        group_size=group_size,
                        lookup_table=lookup_table,
                        super_group_size=self.super_group_size,
                        group_scale_bits=group_scale_bits,
                        quant_scale=quant_scale,
                        symmetric=symmetric,
                        isolate_outliers=isolate,
                        enable_norm=enable_norm,
                        enable_perm=enable_perm,
                        extract_sign=extract_sign,
                        parity_sign=parity_sign,
                        enable_completion=enable_completion,
                        enable_compensation=enable_compensation,
                        enable_circle=enable_circle,
                    )
                    weight = model_state[cur_name+'.weight'].t().cuda() # [out_features, in_features]
                    
                    if extract_sign:
                        logger.info(f"extract weight's sign...")
                        weight = weight.abs()
                
                    if enable_perm:
                        # logger.info(f"Permute weight...")
                        # # weight = weight_quanter.permute(weight)
                        # weight_quanter.perm = paddle.argsort(paddle.diag(hessian_matrix), descending=True)
                        # weight = weight[:, weight_quanter.perm]
                        # hessian_matrix = hessian_matrix[weight_quanter.perm][:, weight_quanter.perm]

                        logger.info(f"Permute group weight...")
                        old_shape = weight.shape
                        weight = weight.reshape([-1, self.group_size])
                        tmp_weight = weight.abs().mean(axis=0)
                        weight_quanter.perm = paddle.argsort(tmp_weight, descending=True)
                        weight = weight[:, weight_quanter.perm].reshape(old_shape)

                    if enable_norm:
                        logger.info(f"Normalizing weight...")
                        # weight_quanter.norm_scale = paddle.ones(weight.shape[1], dtype='bfloat16')
                        means = weight.abs().max(axis=0).mean()
                        denominator = weight.abs().max(axis=0)
                        scale_out_channel = paddle.pow(paddle.diag(hessian_matrix), 0.5) / paddle.pow(denominator, 0.5)
                        # scale_out_channel = paddle.diag(hessian_matrix) * denominator
                        # scale_out_channel = paddle.pow(weight.abs().max(axis=0), 0.5) / paddle.pow(denominator, 0.5)
                        # scale_out_channel = 1 / scale_out_channel
                        logger.info(f"denominator: {denominator.min().item()} ~ {denominator.max().item()}")
                        logger.info(f"norm scale: {scale_out_channel.min().item()} ~ {scale_out_channel.max().item()}")
                        weight_quanter.norm_scale = scale_out_channel
                        weight_quanter.norm_bias = paddle.zeros(weight.shape[1], dtype='bfloat16')
                        weight = (weight - weight_quanter.norm_bias) * weight_quanter.norm_scale
                        
                    if hadamard:
                        H, W  = weight.shape
                        weight_square = weight**2
                        hadamard_matrix = paddle.load("hadamard_matrix_64.pdtensors").cast('float32')
                        logger.debug(f"[debug] hadamard_matrix: {hadamard_matrix.shape}")
                        weight = weight.reshape([-1, self.group_size]).cast("float32") @ hadamard_matrix
                        weight = weight.reshape([H, W]).cast("bfloat16")
                        # hessian_matrix = hessian_matrix.reshape([-1, self.group_size])
                        # hessian_matrix = hessian_matrix.cast("float32") @ hadamard_matrix
                        # hessian_matrix = hessian_matrix.reshape([W, W])

                    if layer_name in sampled_hessian:
                        logger.info(f"Using hessian...")
                        hessian_matrix = sampled_hessian.pop(layer_name).cuda() # [in_fea, in_fea]
                        info_matrix = paddle.diag(hessian_matrix).unsqueeze(0).tile((weight.shape[0], 1))#.cast('bfloat16') # [out_features, in_features]
                    else:
                        info_matrix = None

                    def isolate_outliers(weight, info_matrix, p1=0.005, p2=0.0):
                        percentile1 = paddle.quantile(info_matrix.abs(), q=1-p1)
                        sensitive_mask = paddle.where(info_matrix.abs() >= percentile1, 1., 0.).cast("bfloat16") 
                        remain_weight = weight *(1-sensitive_mask)
                        sensitive_weight = weight * sensitive_mask
                        if p2 > 0:
                            percentile2 = paddle.quantile(remain_weight.abs(), q=1-p2)
                            outliers_mask = paddle.where(remain_weight.abs() >= percentile2, 1., 0.).cast("bfloat16") 
                            outliers_weight = weight * outliers_mask
                            remain_weight = remain_weight * (1. - outliers_mask)
                            outliers_weight += sensitive_weight
                        else:
                            outliers_weight = sensitive_weight
                        # info_matrix = info_matrix * (1-sensitive_mask) * (1-outliers_mask)
                        return remain_weight, info_matrix, outliers_weight
                    
                    if isolate:
                        weight, info_matrix, outliers_weight = isolate_outliers(weight, info_matrix)
                        outliers_weight = outliers_weight.reshape([-1, self.group_size])
                        logger.debug(f"[debug] outliers_weight: {outliers_weight.shape}")
                    else:
                        outliers_weight = None

                    # group-wise quantization 
                    logger.info(f"[TQ] Before quant, weight shape: {weight.shape}")
                    weight_amp = weight.sum()
                    weight_channel_amp = weight.sum(axis=-1).mean()
                    logger.info(f"weight amplitude: {weight_amp.item()}, channel_amp: {weight_channel_amp.item()}")
                    if enable_ldlq:
                        # hessian_matrix = self._regularize_H(hessian_matrix.cast('float32'), 10.)
                        weight_quanter.group_quantize_with_LDLQ(weight, hessian_matrix.cast('float32'), info_matrix)
                    elif enable_cluster:
                        weight_quanter.group_quantize_cluster(weight, info_matrix)
                    elif enable_razor:
                        weight_quanter.qrazor_quantize(weight, info_matrix)
                    else:
                        weight_quanter.group_quantize(weight, info_matrix) # [out_features, in_features]
                    
                    logger.info(f"[TQ] Set {cur_name} to be TrellisQuantLinear")
                    # use QuantLinear to save quant weight and scale
                    qlayer = TrellisQuantLinear(
                        in_features=weight.shape[1],
                        out_features=weight.shape[0],
                        lut_bits=lut_bits,
                        states=states,
                        state_bits=state_bits,
                        group_size=group_size,
                        super_group_size=self.super_group_size,
                        group_scale_bits=group_scale_bits,
                        quant_scale=quant_scale,
                        symmetric=symmetric,
                        pack=pack,
                        isolate_outliers=isolate,
                        bias=None,
                        enable_norm=enable_norm,
                        enable_perm=enable_perm,
                        enable_cluster=enable_cluster,
                        redundant_bits=weight_quanter.redundant_bits,
                        parity_sign=parity_sign,
                        enable_razor=enable_razor,
                        enable_completion=enable_completion,
                        enable_compensation=enable_compensation,
                        enable_circle=enable_circle,
                    )
                    qlayer.init_parameters( 
                        weight_quanter.quant_weight, 
                        weight_quanter.scales, 
                        weight_quanter.super_scales,
                        weight_quanter.zero_points,
                        weight_quanter.super_zero_points,
                        outliers_weight=outliers_weight,
                        norm_scale=weight_quanter.norm_scale,
                        norm_bias=weight_quanter.norm_bias,
                        perm=weight_quanter.perm,
                        indices_scale=weight_quanter.indices_scale,
                        indices_zp=weight_quanter.indices_zp,
                        for_infer=False,
                    )

                    # parent_layer, sub_name = find_parent_layer_and_sub_name(self.model, cur_name)
                    # setattr(parent_layer, sub_name, qlayer)

                    # only save quant weight and scale
                    save_state.update({cur_name + '.weight': qlayer.weight,
                                        cur_name + '.scales': qlayer.scales,
                                        cur_name + '.tq_config': qlayer.tq_config
                    })
                    if getattr(qlayer, "super_scales", None) is not None:
                        save_state.update({cur_name + '.super_scales': qlayer.super_scales})
                    if getattr(qlayer, "zero_points", None) is not None:
                        save_state.update({cur_name + '.zero_points': qlayer.zero_points})
                    if getattr(qlayer, "super_zero_points", None) is not None:
                        save_state.update({cur_name + '.super_zero_points': qlayer.super_zero_points})
                    if getattr(qlayer, "outliers_weight", None) is not None:
                        save_state.update({cur_name + '.outliers_weight': qlayer.outliers_weight})
                    if enable_norm:
                        save_state.update({cur_name + '.norm_scale': qlayer.norm_scale})
                        save_state.update({cur_name + '.norm_bias': qlayer.norm_bias})
                    if enable_perm:
                        save_state.update({cur_name + '.perm': qlayer.perm})
                    if extract_sign:
                        save_state.update({cur_name + '.extract_sign': True})
                    if enable_cluster:
                        save_state.update({cur_name + '.code_scale': qlayer.code_scale})
                        save_state.update({cur_name + '.code_zp': qlayer.code_zp})
                    if enable_compensation:
                        save_state.update({cur_name + '.svd_bias': weight_quanter.svd_bias})
                    # clear params
                    model_state[cur_name+'.weight'].value().get_tensor()._clear()
                    del weight_quanter, info_matrix, qlayer
                    paddle.device.cuda.empty_cache()
                    # gc.collect()
            return save_state
        
        layer_name_dict = {}
        for cur_name, sub_layer in self.model.named_sublayers():
            if type(sub_layer) in [
                ColumnParallelLinear,
                RowParallelLinear,
                paddle.nn.Linear,
            ]:
                layer_name_dict[cur_name] = sub_layer.full_name()
        if True:
            logger.info(f"IQ start...")
            self.save_state_dict = self.IQ._quantize_layer(self.model.state_dict(), layer_name_dict, self.sampled_hessian, self.IQ._target_layers)
            logger.info(f"[IQ] state_dict: {len(self.save_state_dict)}")
            logger.info(f"IQ end...")
            gc.collect()

        _state_dict = _quantize_layer(self.model.state_dict(), layer_name_dict, self.sampled_hessian, self._target_layers)
        self.save_state_dict.update(_state_dict)
        for key, value in self.model.state_dict().items():
            if key not in self.save_state_dict:
                self.save_state_dict[key] = value

    def trellis_quantize_for_GBQ(self, target_layers, layer_idx):
        self._remove_hook()
        logger.info(f"[TQ] Prepare Hessian Matrix")
        self._cal_hessian()
        logger.info(f"Hessian: [{len(list(self.sampled_hessian.keys()))}] - {list(self.sampled_hessian.keys())}")
        self._target_layers = []
        
        logger.info(f"[TQ] target_layers: [{len(target_layers)}] - {target_layers}")
        
        lookup_table_dict = {}
        def _quantize_layer(target_layers):
            save_state = {}
            for cur_name, sub_layer in self.model.named_sublayers():
                layer_name = sub_layer.full_name()
                if not isinstance(sub_layer, (ColumnParallelLinear, RowParallelLinear, paddle.nn.Linear)):
                    continue
                if layer_name in target_layers:
                    lut_bits, states, state_bits = self.lut_bits, self.states, self.state_bits
                    symmetric = True
                    isolate = False
                    hadamard = False
                    enable_norm = False
                    enable_ldlq = False
                    quant_scale = self.quant_scale
                    enable_cluster = self.enable_cluster
                    group_size = self.group_size
                    pack = False
                    parity_sign = False

                    logger.info(f"[TQ] [layer {layer_idx}] quantize: {cur_name} --- {layer_name}")
                    if isinstance(states, list):
                        table_key = (lut_bits, tuple(states), tuple(state_bits))
                    else:
                        table_key = (lut_bits, states, state_bits)
                    if table_key in lookup_table_dict:
                        lookup_table = lookup_table_dict[table_key]
                    elif isinstance(table_key[1], tuple):
                        lookup_table = construct_lookup_table_multi2(lut_bits, states, state_bits)
                        lookup_table_dict[table_key] = lookup_table
                    else:
                        lookup_table = construct_lookup_table(lut_bits, states, state_bits)
                        lookup_table_dict[table_key] = lookup_table

                    weight_quanter = TrellisQuantizer(
                        lut_bits=lut_bits,
                        states=states,
                        state_bits=state_bits,
                        group_size=group_size,
                        lookup_table=lookup_table,
                        super_group_size=self.super_group_size,
                        group_scale_bits=self.group_scale_bits,
                        quant_scale=quant_scale,
                        symmetric=symmetric,
                        isolate_outliers=isolate,
                        enable_norm=enable_norm,
                        parity_sign=parity_sign,
                    )
                    weight = sub_layer.weight.t().cuda() # [out_features, in_features]
                    if layer_name in self.sampled_hessian:
                        logger.info(f"hessian nums: {self.sampled_num[layer_name]}")
                        hessian_matrix = self.sampled_hessian.pop(layer_name).cuda()
                        info_matrix = paddle.diag(hessian_matrix).unsqueeze(0).tile((weight.shape[0], 1)).cast('bfloat16') # [out_features, in_features]
                        if paddle.isinf(info_matrix).any():
                            logger.warning(f"[warning] inf hessian matrix, set to none")
                            info_matrix = None
                        if paddle.isnan(info_matrix).any():
                            logger.warning(f"[warning] nan hessian matrix, set to none")
                            info_matrix = None
                    else:
                        logger.info(f"[TQ] No Hessian matrix for {cur_name}")
                        hessian_matrix = None
                        info_matrix = None

                    def isolate_outliers(weight, info_matrix, p1=0.005, p2=0.0):
                        percentile1 = paddle.quantile(info_matrix.abs(), q=1-p1)
                        sensitive_mask = paddle.where(info_matrix.abs() >= percentile1, 1., 0.).cast("bfloat16") 
                        remain_weight = weight *(1-sensitive_mask)
                        sensitive_weight = weight * sensitive_mask
                        if p2 > 0:
                            percentile2 = paddle.quantile(remain_weight.abs(), q=1-p2)
                            outliers_mask = paddle.where(remain_weight.abs() >= percentile2, 1., 0.).cast("bfloat16") 
                            outliers_weight = weight * outliers_mask
                            remain_weight = remain_weight * (1. - outliers_mask)
                            outliers_weight += sensitive_weight
                        else:
                            outliers_weight = sensitive_weight
                        # info_matrix = info_matrix * (1-sensitive_mask) * (1-outliers_mask)
                        return remain_weight, info_matrix, outliers_weight
                    
                    if isolate:
                        weight, info_matrix, outliers_weight = isolate_outliers(weight, info_matrix)
                        outliers_weight = outliers_weight.reshape([-1, self.group_size])
                        logger.debug(f"[debug] outliers_weight: {outliers_weight.shape}")
                    else:
                        outliers_weight = None

                    if hadamard:
                        H, W  = weight.shape
                        hadamard_matrix = paddle.load("hadamard_matrix_32.pdtensors")
                        logger.debug(f"[debug] hadamard_matrix: {hadamard_matrix.shape}")
                        weight = weight.reshape([-1, self.group_size]).cast("float32") @ hadamard_matrix
                        weight = weight.reshape([H, W]).cast("bfloat16")

                    # group-wise quantization 
                    logger.info(f"[TQ] Before quant, weight shape: {weight.shape}")
                    if enable_ldlq and hessian_matrix is not None:
                        weight_quanter.group_quantize_with_LDLQ(weight, hessian_matrix, info_matrix)
                    elif enable_cluster:
                        weight_quanter.group_quantize_cluster(weight, info_matrix)
                    else:
                        qdq_weight = weight_quanter.group_quantize(weight, info_matrix) #[out_features, in_features]

                        ### for GPTAQ
                        # sub_layer.weight.set_value(qdq_weight.cast('bfloat16').t())

                    logger.info(f"[TQ] Set {cur_name} to be TrellisQuantLinear")
                    # use QuantLinear to save quant weight and scale
                    qlayer = TrellisQuantLinear(
                        in_features=weight.shape[1],
                        out_features=weight.shape[0],
                        lut_bits=lut_bits,
                        states=states,
                        state_bits=state_bits,
                        group_size=group_size,
                        super_group_size=self.super_group_size,
                        group_scale_bits=self.group_scale_bits,
                        quant_scale=quant_scale,
                        symmetric=symmetric,
                        bias=None,
                        enable_cluster=enable_cluster,
                        redundant_bits=weight_quanter.redundant_bits,
                        parity_sign=parity_sign,
                        pack=pack,
                    )
                    qlayer.init_parameters( 
                        weight_quanter.quant_weight, 
                        weight_quanter.scales, 
                        weight_quanter.super_scales,
                        weight_quanter.zero_points,
                        weight_quanter.super_zero_points,
                        outliers_weight=outliers_weight,
                        norm_scale=weight_quanter.norm_scale,
                        norm_bias=weight_quanter.norm_bias,
                        indices_scale=weight_quanter.indices_scale,
                        indices_zp=weight_quanter.indices_zp,
                        for_infer=False,
                    )

                    # only save quant weight and scale
                    save_state.update({cur_name + '.weight': qlayer.weight,
                                        cur_name + '.scales': qlayer.scales,
                                        cur_name + '.tq_config': qlayer.tq_config
                    })
                    if getattr(qlayer, "super_scales", None) is not None:
                        save_state.update({cur_name + '.super_scales': qlayer.super_scales})
                    if getattr(qlayer, "zero_points", None) is not None:
                        save_state.update({cur_name + '.zero_points': qlayer.zero_points})
                    if getattr(qlayer, "super_zero_points", None) is not None:
                        save_state.update({cur_name + '.super_zero_points': qlayer.super_zero_points})
                    if getattr(qlayer, "outliers_weight", None) is not None:
                        save_state.update({cur_name + '.outliers_weight': qlayer.outliers_weight})
                    if enable_norm:
                        save_state.update({cur_name + '.norm_scale': qlayer.norm_scale})
                        save_state.update({cur_name + '.norm_bias': qlayer.norm_bias})
                    if enable_cluster:
                        save_state.update({cur_name + '.code_scale': qlayer.code_scale})
                        save_state.update({cur_name + '.code_zp': qlayer.code_zp})
                    # clear params
                    # sub_layer.weight.value().get_tensor()._clear()
                    del weight_quanter, info_matrix, qlayer
                    # paddle.device.cuda.empty_cache()
                    # gc.collect()
            return save_state
        
        _state_dict = _quantize_layer(target_layers)
        self.save_state_dict.update(_state_dict)

        logger.info(f"[TQ] Final quant_layers: [{len(target_layers)}] - {target_layers}")

    def replace_with_quant_layer(self, load_path=None, model_state=None):
        logger.info("[TQ] Begin replace with quant linear...")
        if model_state is None:
            logger.info(f"Load quant model from {load_path}")
            paddle.set_device('cpu')
            qmodel_state = paddle.load(load_path)
            paddle.set_device('gpu')
        else:
            qmodel_state = model_state
        all_target_layers = []
        for key in qmodel_state.keys():
            if '.tq_config' in key:
                all_target_layers.append(key)
                # self.model.state_dict()[key.replace('tq_config', 'weight')].value().get_tensor()._clear()
        logger.info(f"All target layers: {all_target_layers}")
        logger.info(f"All target layers: {len(all_target_layers)}")
        total_sparse_ratio = []

        def _replace_layer(target_layers):
            sparse_ratios = []
            for cur_name, sub_layer in self.model.named_sublayers():
                layer_name = sub_layer.full_name()
                if isinstance(sub_layer, (ColumnParallelLinear, RowParallelLinear, paddle.nn.Linear)) \
                 and (cur_name + '.tq_config' in target_layers):
                    if cur_name + '.tq_config' not in qmodel_state:
                        continue
                    gather_output, input_is_parallel = None, None
                    if isinstance(sub_layer, ColumnParallelLinear):
                        QuantLayer = TrellisQuantColumnParallelLinear
                        gather_output = sub_layer.gather_output
                    elif isinstance(sub_layer, RowParallelLinear):
                        QuantLayer = TrellisQuantRowParallelLinear
                        input_is_parallel = sub_layer.input_is_parallel
                    else:
                        QuantLayer = TrellisQuantLinear
                    logger.info(f"Replace: {cur_name} --- {layer_name} -> {QuantLayer}")
                    layer_idx = int(cur_name.split(".")[2])
                    tq_config = qmodel_state[cur_name + '.tq_config']
                    lut_bits, states, state_bits = tq_config['lut_bits'], tq_config['states'], tq_config['state_bits']
                    group_size = tq_config['group_size']

                    hadamard = False
                    symmetric = True
                    enable_cluster = False
                    if cur_name + '.code_scale' in qmodel_state:
                        enable_cluster = True
                    isolate_outliers = False
                    if cur_name + '.outliers_weight' in qmodel_state:
                        isolate_outliers = True

                    extract_sign = qmodel_state.get(cur_name + '.extract_sign', False)
                    if extract_sign:
                        weight_sign = paddle.sign(sub_layer.weight)
                    else:
                        weight_sign = None

                    logger.info(f"[TQ] {cur_name} - quant_config: {tq_config}")
                    with paddle.LazyGuard():
                        qlayer = QuantLayer(
                            in_features=sub_layer.weight.shape[0],
                            out_features=sub_layer.weight.shape[1],
                            lut_bits=lut_bits,
                            states=states,
                            state_bits=state_bits,
                            group_size=group_size,
                            symmetric=symmetric,
                            quant_scale=tq_config.get('quant_scale', False),
                            group_scale_bits=tq_config.get('group_scale_bits', 4),
                            redundant_bits=tq_config['redundant_bits'],
                            pack=tq_config.get('pack', False),
                            bias=getattr(sub_layer, 'bias', None),
                            hadamard=False,
                            enable_norm=False,
                            gather_output=gather_output,
                            input_is_parallel=input_is_parallel,
                            enable_cluster=enable_cluster,
                            parity_sign=tq_config.get('parity_sign', False),
                            enable_circle=tq_config.get('enable_circle', False),
                        )
                    qlayer.init_parameters(
                        quant_weight=qmodel_state[cur_name + '.weight'],
                        scales=qmodel_state[cur_name + '.scales'],
                        super_scales=qmodel_state.get(cur_name + '.super_scales', None),
                        zero_points=qmodel_state.get(cur_name + '.zero_points', None),
                        super_zero_points=qmodel_state.get(cur_name + '.super_zero_points', None),
                        is_pack=tq_config.get('pack', False),
                        outliers_weight=qmodel_state.get(cur_name + '.outliers_weight', None),
                        indices_scale=qmodel_state.get(cur_name + '.code_scale', None),
                        indices_zp=qmodel_state.get(cur_name + '.code_zp', None),
                    )
                    parent_layer, sub_name = find_parent_layer_and_sub_name(self.model, cur_name)
                    setattr(parent_layer, sub_name, qlayer)
                    # clear ori weight
                    sub_layer.weight.value().get_tensor()._clear()

                    quant_weight = qmodel_state[cur_name + '.weight']
                    sparse_ratio = paddle.sum(quant_weight == 0).item() / (quant_weight.shape[0] * quant_weight.shape[1])
                    sparse_ratios.append(sparse_ratio)
                    qmodel_state.pop(cur_name + '.weight')
                    del quant_weight, sub_layer
                    paddle.device.cuda.empty_cache()
                    
            return sparse_ratios
        if len(all_target_layers) == 0:
            logger.info(f"[TQ] No need to init quant linear")
            return qmodel_state
        self._thread_num = os.cpu_count()
        if len(all_target_layers) < self._thread_num:
            self._thread_num = 2
        if self._thread_num == 1:
            total_sparse_ratio = _replace_layer(all_target_layers)
        else:
            step = math.ceil(len(all_target_layers)/self._thread_num)
            seg_target_layers = []
            for i in range(0, len(all_target_layers), step):
                seg_target_layers.append(all_target_layers[i: i + step])
            logger.info(f"Seg parts: {len(seg_target_layers)} Step: {step}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._thread_num) as executor:
                futures = [
                    executor.submit(_replace_layer, part) for part in seg_target_layers
                ]

            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                total_sparse_ratio.extend(future.result())
        
        logger.info(f"Average Sparse Ratio: {sum(total_sparse_ratio) / len(total_sparse_ratio)}")
        logger.info(f"Initialize Quant Linear Layers Done!")
        return qmodel_state

    def replace_with_quant_layer_for_DS(self, load_path=None, model_state=None):
        logger.info("[TQ] Begin replace with quant linear...")
        if model_state is None:
            logger.info(f"Load quant model from {load_path}")
            paddle.set_device('cpu')
            qmodel_state = paddle.load(load_path)
            paddle.set_device('gpu')
        else:
            qmodel_state = model_state
        all_target_layers = []
        for key in qmodel_state.keys():
            if '.tq_config' in key:
                all_target_layers.append(key)
        logger.info(f"All target layers: {all_target_layers}")
        logger.info(f"All target layers: {len(all_target_layers)}")
        total_sparse_ratio = []

        def _replace_layer(target_layers):
            sparse_ratios = []
            for cur_name, sub_layer in self.model.named_sublayers():
                layer_name = sub_layer.full_name()
                if isinstance(sub_layer, (ColumnParallelLinear, RowParallelLinear, paddle.nn.Linear)) \
                 and (cur_name + '.tq_config' in target_layers):
                    if cur_name + '.tq_config' not in qmodel_state:
                        continue
                    gather_output, input_is_parallel = None, None
                    if isinstance(sub_layer, ColumnParallelLinear):
                        QuantLayer = TrellisQuantColumnParallelLinear
                        gather_output = sub_layer.gather_output
                    elif isinstance(sub_layer, RowParallelLinear):
                        QuantLayer = TrellisQuantRowParallelLinear
                        input_is_parallel = sub_layer.input_is_parallel
                    else:
                        QuantLayer = TrellisQuantLinear
                    logger.info(f"Replace: {cur_name} --- {layer_name} -> {QuantLayer}")
                    layer_idx = int(cur_name.split(".")[2])
                    tq_config = qmodel_state[cur_name + '.tq_config']
                    lut_bits, states, state_bits = tq_config['lut_bits'], tq_config['states'], tq_config['state_bits']
                    group_size = tq_config['group_size']

                    hadamard = False
                    symmetric = True
                    enable_cluster = False
                    
                    if cur_name + '.code_scale' in qmodel_state:
                        enable_cluster = True
                    isolate_outliers = False
                    if cur_name + '.outliers_weight' in qmodel_state:
                        isolate_outliers = True

                    extract_sign = qmodel_state.get(cur_name + '.extract_sign', False)
                    if extract_sign:
                        weight_sign = paddle.sign(sub_layer.weight)
                    else:
                        weight_sign = None

                    logger.info(f"[TQ] {cur_name} - quant_config: {tq_config} - enable_cluster: {enable_cluster}")
                    with paddle.LazyGuard():
                        qlayer = QuantLayer(
                            in_features=sub_layer.weight.shape[0],
                            out_features=sub_layer.weight.shape[1],
                            lut_bits=lut_bits,
                            states=states,
                            state_bits=state_bits,
                            group_size=group_size,
                            symmetric=symmetric,
                            pack=tq_config.get('pack', False),
                            bias=getattr(sub_layer, 'bias', None),
                            hadamard=False,
                            enable_norm=False,
                            gather_output=gather_output,
                            input_is_parallel=input_is_parallel,
                            enable_cluster=enable_cluster,
                            parity_sign=tq_config.get('parity_sign', False),
                        )
                    qlayer.init_parameters(
                        quant_weight=qmodel_state[cur_name + '.quant_weight'],
                        scales=qmodel_state[cur_name + '.quant_scale'],
                        super_scales=qmodel_state.get(cur_name + '.super_scales', None),
                        zero_points=qmodel_state.get(cur_name + '.zero_points', None),
                        super_zero_points=qmodel_state.get(cur_name + '.super_zero_points', None),
                        is_pack=tq_config.get('pack', False),
                        outliers_weight=qmodel_state.get(cur_name + '.outliers_weight', None),
                        indices_scale=qmodel_state.get(cur_name + '.code_scale', None),
                        indices_zp=qmodel_state.get(cur_name + '.code_zp', None),
                    )
                    parent_layer, sub_name = find_parent_layer_and_sub_name(self.model, cur_name)
                    setattr(parent_layer, sub_name, qlayer)
                    # clear ori weight
                    sub_layer.weight.value().get_tensor()._clear()

                    quant_weight = qmodel_state[cur_name + '.quant_weight']
                    sparse_ratio = paddle.sum(quant_weight == 0).item() / (quant_weight.shape[0] * quant_weight.shape[1])
                    sparse_ratios.append(sparse_ratio)
                    qmodel_state.pop(cur_name + '.quant_weight')
                    del quant_weight
                    paddle.device.cuda.empty_cache()
                    
            return sparse_ratios
        if len(all_target_layers) == 0:
            logger.info(f"[TQ] No need to init quant linear")
            return
        self._thread_num = os.cpu_count()
        if len(all_target_layers) < self._thread_num:
            self._thread_num = 2
        if self._thread_num == 1:
            total_sparse_ratio = _replace_layer(all_target_layers)
        else:
            step = math.ceil(len(all_target_layers)/self._thread_num)
            seg_target_layers = []
            for i in range(0, len(all_target_layers), step):
                seg_target_layers.append(all_target_layers[i: i + step])
            logger.info(f"Seg parts: {len(seg_target_layers)} Step: {step}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._thread_num) as executor:
                futures = [
                    executor.submit(_replace_layer, part) for part in seg_target_layers
                ]

            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                total_sparse_ratio.extend(future.result())
        
        logger.info(f"Average Sparse Ratio: {sum(total_sparse_ratio) / len(total_sparse_ratio)}")

        logger.info(f"Initialize Quant Linear Layers Done!")
        
        paddle.device.cuda.empty_cache()
        gc.collect()
        return qmodel_state

    def save_quant_model(self, save_path, dp_degree=1):
        assert self.save_state_dict != {}, "save_state_dict should not be empty!"
        try:
            hcg = fleet.get_hybrid_communicate_group()
            rank = hcg.get_model_parallel_rank()
            nranks = hcg.get_model_parallel_world_size()
            dp_id = hcg.get_data_parallel_rank()
        except:
            rank = paddle.distributed.get_rank()
            nranks = paddle.distributed.get_world_size()
            dp_id = 0

        if nranks == 1:
            model_path = os.path.join(save_path, "model_state.pdparams")
        else:
            model_path = os.path.join(save_path, f"model_state.tp0{rank}.pdparams")
           
        logger.info(f"Save quant model to {model_path}")
        paddle.save(self.save_state_dict, model_path)
        logger.info(f"Save quant model done.")
    
    def _cal_hessian(self):
        for key in self.sampled_hessian:
            H = self.sampled_hessian[key] / self.sampled_num[key]
            self.sampled_hessian[key] = self._regularize_H(H).cpu()

        paddle.device.cuda.empty_cache()
        gc.collect()

    def save_hessian(self, save_path, hessian_nums=128):
        self._remove_hook()
        gc.collect()
        paddle.device.cuda.empty_cache()
        self._cal_hessian()

        try:
            hcg = fleet.get_hybrid_communicate_group()
            rank = hcg.get_model_parallel_rank()
            nranks = hcg.get_model_parallel_world_size()
            dp_id = hcg.get_data_parallel_rank()
        except:
            rank = paddle.distributed.get_rank()
            nranks = paddle.distributed.get_world_size()
            dp_id = 0
        if nranks == 1:
            path = os.path.join(save_path, f"sampled_hessian_{hessian_nums}.pdtensors")
        else:
            path = os.path.join(save_path, f"sampled_hessian_{hessian_nums}.tp0{rank}.pdtensors")
        logger.info(f"Save hessian file to {path}")
        # paddle.save(self.sampled_hessian, path)
        save_file(save_path, f"sampled_hessian_{hessian_nums}.tp0{rank}.safetensors", self.sampled_hessian)
        logger.info(f"Save hessian done.")
        if nranks != 1:
            paddle.distributed.barrier()
    
    def _regularize_H(self, H, sigma=1e-2):
        # zero_idx = paddle.where(paddle.diag(H) == 0)
        # if not paddle.is_empty(zero_idx):
        #     H[zero_idx, zero_idx] = 1
        # damp = sigma * paddle.mean(paddle.diag(H))
        # diag = paddle.arange(H.shape[0])
        # H[diag, diag] += damp

        H = H / paddle.diag(H).mean()
        idx = paddle.arange(H.shape[0])
        H[idx, idx] += sigma
        return H
    
def quantize_layer_parallel(model_state, layer_name_dict, sampled_hessian, target_layers):
    save_state = {}
    for cur_name in layer_name_dict:
        layer_name = layer_name_dict[cur_name]
        if layer_name in sampled_hessian and layer_name in target_layers:
            pass