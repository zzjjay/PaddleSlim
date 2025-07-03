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
IQ.
"""
import os
import gc
import math
import traceback
import numpy as np
import paddle
from paddle.distributed import fleet
# from paddle.distributed.fleet.meta_parallel import ColumnParallelLinear, RowParallelLinear
from .utils import find_parent_layer_and_sub_name
from paddlenlp.utils.log import logger
from .iquant_utils import IQuantizer, read_file, save_file
from ..layers.iquant_layers import IQuantLinear, IQuantColumnParallelLinear, IQuantRowParallelLinear
import concurrent.futures
from paddle.distributed.fleet.layers.mpu import mp_ops
from multiprocessing import Pool
from paddle.distributed.fleet.meta_parallel import ColumnParallelLinear, RowParallelLinear
from paddle.nn import Linear
from paddle.nn.quant import weight_quantize
# import paddlenlp.transformers.deepseek_v2.fp8_linear as linear_utils
# ColumnParallelLinear = linear_utils.ColumnParallelLinear
# RowParallelLinear = linear_utils.RowParallelLinear
# Linear = linear_utils.Linear
__all__ = ['IQuant']
    

class IQuant:
    """
    IQuant.
    """
    def __init__(self,
                 model,
                 group_size=32,
                 quant_bits=2,
                 super_group_size=256,
                 group_scale_bits=4,
                 quant_scale=True,
                 target_layers=None,
                 collect_hessian=False,
                 is_deepseekv3=False,
                 hook_targets=None,
                 weight_only_int4=False,
        ):
        super(IQuant, self).__init__()
        self.model = model
        self.group_size = group_size
        self.quant_bits = quant_bits
        self.super_group_size = super_group_size
        self.group_scale_bits = group_scale_bits
        self.quant_scale = quant_scale
        self._target_layers = target_layers
        self.save_state_dict = {}
        self.collect_hessian = collect_hessian
        self.is_deepseekv3 = is_deepseekv3
        self._hook_targets = hook_targets
        self.sampled_hessian = {}
        self.sampled_means = {}
        self.sampled_num = {}
        self._forward_hook_list = []
        self.weight_only_int4 = weight_only_int4
        if collect_hessian:
            self._apply_hook()
            
        logger.info(f"[IQ] group_size: {group_size}, \
                quant_bits: {quant_bits}, \
                super_group_size: {super_group_size}, \
                group_scale_bits: {group_scale_bits}, \
                quant_scale: {quant_scale}, \
                collect_hessian: {collect_hessian}")
        # logger.info(f"[IQ] target_layers: {target_layers}")
        logger.info(f"[IQ] hook_targets: {hook_targets}")

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
        inp = inp.cast("float32")
        tmp = inp.shape[0]
        inp = inp.reshape((-1, inp.shape[-1]))
        if layer_name not in self.sampled_hessian:
            self.sampled_hessian[layer_name] = paddle.zeros(
                    (columns, columns), dtype='float32')
            self.sampled_num[layer_name] = 0
            # self.sampled_means[layer_name] = paddle.zeros(
            #         (columns), dtype='float32')

        hessian = self.sampled_hessian[layer_name]

        # self.sampled_num[layer_name] += inp.shape[0]
        # hessian = paddle.matmul(inp.t(), inp) + hessian #.cuda()
        
        hessian *= self.sampled_num[layer_name] / (self.sampled_num[layer_name] + tmp)
        self.sampled_num[layer_name] += tmp
        inp = math.sqrt(2 / self.sampled_num[layer_name]) * inp
        hessian += paddle.matmul(inp.t(), inp)

        self.sampled_hessian[layer_name] = hessian #.pin_memory()
        # means = self.sampled_means[layer_name] + inp.sum(axis=0)
        # self.sampled_means[layer_name] = means #.pin_memory()
        del inp, hessian
    
    def _remove_hook(self):
        for hook in self._forward_hook_list:
            hook.remove()
        self._forward_hook_list = [] 

    def _quantize_layer(self, model_state, layer_name_dict, sampled_hessian, target_layers):
        save_state = {}
        for cur_name in layer_name_dict:
            layer_name = layer_name_dict[cur_name]
            if layer_name in target_layers:
                # if not ('mlp.w1' in cur_name or 'mlp.w2' in cur_name):
                #     continue
                logger.info(f"[IQ] quantize: {cur_name} --- {layer_name}")
                layer_idx = int(cur_name.split(".")[2])
                quant_bits = self.quant_bits
                symmetric = True
                isolate = False
                hadamard = False
                extract_sign = False
                logger.info(f"[IQ] quant_bits: {quant_bits}")
                if self.weight_only_int4:
                    # logger.info(f"Use bf16 weight ...")
                    # logger.info(f"ori weight: {sub_layer.weight.shape}")
                    # save_state.update({cur_name + '.weight': sub_layer.weight,
                    # })
                    # continue
                    logger.info(f"Use weight_only_int4 ...")
                    group_size = self.group_size
                    logger.info(f"ori weight: {sub_layer.weight.shape}, group_size: {group_size}")
                    quant_weight, quant_scale = weight_quantize(
                        x=sub_layer.weight.cpu(),
                        algo='weight_only_int4',
                        group_size=group_size,
                    )
                    quant_config = {
                        'quant_bits': quant_bits,
                        'group_size': group_size,
                    }
                    logger.info(f"quant weight: {quant_weight.shape}")
                    save_state.update({cur_name + '.weight': quant_weight,
                                    cur_name + '.scales': quant_scale,
                                    cur_name + '.quant_config': quant_config,
                    })
                    continue

                weight_quanter = IQuantizer(
                    quant_bits=quant_bits,
                    group_size=self.group_size,
                    super_group_size=self.super_group_size,
                    group_scale_bits=self.group_scale_bits,
                    quant_scale=self.quant_scale,
                    symmetric=symmetric,
                    isolate_outliers=isolate,
                    extract_sign=extract_sign,
                )
                weight = model_state[cur_name+'.weight'].t() # [out_features, in_features]
                if extract_sign:
                    weight = weight.abs()
                if layer_name in sampled_hessian:
                    logger.info(f"Using hessian...")
                    hessian_matrix = sampled_hessian.pop(layer_name).cuda()
                    info_matrix = paddle.diag(hessian_matrix).unsqueeze(0).tile((weight.shape[0], 1)) # [out_features, in_features]
                else:
                    info_matrix = None
                def isolate_outliers(weight, info_matrix, p1=0.005, p2=0.0):
                    percentile1 = paddle.quantile(info_matrix.abs(), q=1-p1)
                    sensitive_mask = paddle.where(info_matrix.abs() >= percentile1, 1., 0.).cast_("bfloat16") 
                    remain_weight = weight *(1-sensitive_mask)
                    sensitive_weight = weight * sensitive_mask
                    if p2 > 0:
                        percentile2 = paddle.quantile(remain_weight.abs(), q=1-p2)
                        outliers_mask = paddle.where(remain_weight.abs() >= percentile2, 1., 0.).cast_("bfloat16") 
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
                    weight = weight.reshape([-1, self.group_size]).cast_("float32") @ hadamard_matrix
                    weight = weight.reshape([H, W]).cast_("bfloat16")

                # group-wise quantization 
                logger.info(f"[IQ] Before quant, weight shape: {weight.shape}")
                weight_amp = weight.sum()
                weight_channel_amp = weight.sum(axis=-1).mean()
                logger.info(f"weight amplitude: {weight_amp.item()}, channel_amp: {weight_channel_amp.item()}")
                weight_quanter.group_quantize(weight, info_matrix)

                logger.info(f"[IQ] Set {cur_name} to be IQuantLinear")
                # use QuantLinear to save quant weight and scale
                if quant_bits == 2 or quant_bits == 4:
                    pack = True
                else:
                    pack = False
                qlayer = IQuantLinear(
                    model_state[cur_name+'.weight'].shape[0],
                    model_state[cur_name+'.weight'].shape[1],
                    quant_bits,
                    self.group_size,
                    self.super_group_size,
                    self.group_scale_bits,
                    quant_scale=self.quant_scale,
                    symmetric=symmetric,
                    pack=pack,
                    isolate_outliers=isolate,
                    bias=None,
                )
                qlayer.init_parameters( 
                    weight_quanter.quant_weight, 
                    weight_quanter.scales, 
                    weight_quanter.super_scales,
                    weight_quanter.zero_points,
                    weight_quanter.super_zero_points,
                    outliers_weight=outliers_weight,
                )

                # parent_layer, sub_name = find_parent_layer_and_sub_name(self.model, cur_name)
                # setattr(parent_layer, sub_name, qlayer)

                # only save quant weight and scale
                save_state.update({cur_name + '.weight': qlayer.weight,
                                    cur_name + '.scales': qlayer.scales,
                                    cur_name + '.iq_config': qlayer.iq_config
                })
                if getattr(qlayer, "super_scales", None) is not None:
                    save_state.update({cur_name + '.super_scales': qlayer.super_scales})
                if getattr(qlayer, "zero_points", None) is not None:
                    save_state.update({cur_name + '.zero_points': qlayer.zero_points})
                if getattr(qlayer, "super_zero_points", None) is not None:
                    save_state.update({cur_name + '.super_zero_points': qlayer.super_zero_points})
                if getattr(qlayer, "outliers_weight", None) is not None:
                    save_state.update({cur_name + '.outliers_weight': qlayer.outliers_weight})
                if extract_sign:
                    save_state.update({cur_name + '.extract_sign': True})

                # clear params
                model_state[cur_name+'.weight'].value().get_tensor()._clear()
                del weight_quanter, info_matrix, qlayer
                paddle.device.cuda.empty_cache()
                # gc.collect()
        return save_state
        
    @paddle.no_grad()
    def importance_quantize(self, hessian_file):
        logger.info(f"Loading Hessian Matrix from {hessian_file}")
        paddle.set_device('cpu')
        # self.sampled_hessian = paddle.load(hessian_file)
        self.sampled_hessian = read_file(hessian_file)
        # model_path = os.path.join(model_path, 'model_state.pdparams')
        # logger.info(f"Loading model from {model_path}")
        # model_state_dict = paddle.load(model_path)
        paddle.set_device(f'gpu')

        layer_name_dict = {}
        for cur_name, sub_layer in self.model.named_sublayers():
            if type(sub_layer) in [
                ColumnParallelLinear,
                RowParallelLinear,
                paddle.nn.Linear,
            ]:
                layer_name_dict[cur_name] = sub_layer.full_name()

        self._thread_num = 1
        if len(self._target_layers) < self._thread_num:
            self._thread_num = 1
        if self._thread_num == 1:
            self.save_state_dict = self._quantize_layer(self.model.state_dict(), layer_name_dict, self.sampled_hessian, self._target_layers)
        else:
            step = math.ceil(len(self._target_layers)/self._thread_num)
            seg_target_layers = []
            for i in range(0, len(self._target_layers), step):
                seg_target_layers.append(self._target_layers[i: i + step])
            logger.info(f"Seg parts: {len(seg_target_layers)} Step: {step}")

            # '''
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._thread_num) as executor:
                futures = [
                    executor.submit(
                        _quantize_layer, 
                        self.model.state_dict(),
                        layer_name_dict,
                        self.sampled_hessian,
                        part,
                    ) for part in seg_target_layers
                ]

            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                state_dict = future.result()
                self.save_state_dict.update(state_dict)
            # '''

    def importance_quantize_for_GBQ(self, target_layers, layer_idx):
        self._remove_hook()
        logger.info(f"[IQ] Prepare Hessian Matrix")
        self._cal_hessian()
        logger.info(f"Hessian: [{len(list(self.sampled_hessian.keys()))}] - {list(self.sampled_hessian.keys())}")
        self._target_layers = []
        # if len(self.sampled_hessian) > 0:
        #     for cur_name, sub_layer in self.model.named_sublayers():
        #         layer_name = sub_layer.full_name()
        #         if layer_name in self.sampled_hessian:
        #             self._target_layers.append(layer_name)
        logger.info(f"[IQ] target_layers: [{len(target_layers)}] - {target_layers}")
        def _quantize_layer(target_layers):
            save_state = {}
            for cur_name, sub_layer in self.model.named_sublayers():
                layer_name = sub_layer.full_name()
                if not isinstance(sub_layer, (ColumnParallelLinear, RowParallelLinear, Linear)):
                    continue
                if layer_name in target_layers:
                    quant_bits = self.quant_bits
                    symmetric = True
                    use_hessian = False
                    '''
                    if 'self_attn.kv' in cur_name:
                        quant_bits = 4
                    elif 'self_attn.q' in cur_name:
                        quant_bits = 4
                    elif 'self_attn.o' in cur_name:
                        quant_bits = 4
                    elif 'mlp.up_proj' in cur_name or 'mlp.gate_proj' in cur_name:
                        quant_bits = 4
                    elif 'mlp.down_proj' in cur_name:
                        quant_bits = 4
                    elif 'shared_expert' in cur_name:    
                        quant_bits = 4
                    else:
                        # for experts
                        if layer_idx < 6:
                            quant_bits = 4
                        else:
                            quant_bits = 2
                    if quant_bits == 2:
                        continue
                    '''
                    logger.info(f"[IQ] [layer {layer_idx}] quantize: {cur_name} --- {layer_name}")
                    logger.info(f"quant_bits: {quant_bits}")
                    if self.weight_only_int4:
                        # logger.info(f"Use bf16 weight ...")
                        # logger.info(f"ori weight: {sub_layer.weight.shape}")
                        # save_state.update({cur_name + '.weight': sub_layer.weight,
                        # })
                        # continue

                        logger.info(f"Use weight_only_int4 ...")
                        group_size = self.group_size
                        logger.info(f"ori weight: {sub_layer.weight.shape}, group_size: {group_size}")
                        quant_weight, quant_scale = weight_quantize(
                            x=sub_layer.weight.cpu(),
                            algo='weight_only_int4',
                            group_size=group_size,
                        )
                        quant_config = {
                            'quant_bits': quant_bits,
                            'group_size': group_size,
                        }
                        logger.info(f"quant weight: {quant_weight.shape}")
                        save_state.update({cur_name + '.weight': quant_weight,
                                        cur_name + '.scales': quant_scale,
                                        cur_name + '.quant_config': quant_config,
                        })
                        continue

                    # '''
                    weight_quanter = IQuantizer(
                        quant_bits=quant_bits,
                        group_size=self.group_size,
                        super_group_size=self.super_group_size,
                        group_scale_bits=self.group_scale_bits,
                        quant_scale=self.quant_scale,
                        symmetric=symmetric,
                    )
                    # sub_layer.weight_quanter = weight_quanter
                    weight_name = cur_name + '.weight'
                    weight = sub_layer.weight.t().cuda() # [out_features, in_features]
                    if layer_name in self.sampled_hessian:
                        logger.info(f"hessian nums: {self.sampled_num[layer_name]}")

                        # if self.sampled_num[layer_name] < 1000:
                        #     logger.info(f"set hessian to none")
                        #     info_matrix = None

                        if not use_hessian:
                            logger.info(f"Don't use hessian, so set hessian to none")
                            info_matrix = None
                        else:
                            hessian_matrix = self.sampled_hessian.pop(layer_name).cuda().cast_("bfloat16")
                            info_matrix = paddle.diag(hessian_matrix).unsqueeze(0).tile((weight.shape[0], 1)) # [out_features, in_features]
                            if paddle.isinf(info_matrix).any():
                                logger.warning(f"[warning] inf hessian matrix, set to none")
                                info_matrix = None
                            if paddle.isnan(info_matrix).any():
                                logger.warning(f"[warning] nan hessian matrix, set to none")
                                info_matrix = None
                    else:
                        logger.info(f"cur layer has no hessian matrix.")
                        info_matrix = None
                    # group-wise quantization 
                    logger.info(f"[IQ] Before quant, weight shape: {weight.shape}")
                    weight_quanter.group_quantize(weight, info_matrix)

                    logger.info(f"[IQ] Set {cur_name} to be IQuantLinear")
                    # use QuantLinear to save quant weight and scale
                    if quant_bits == 2 or quant_bits == 4:
                        pack = True
                    else:
                        pack = False
                    qlayer = IQuantLinear(
                        sub_layer.weight.shape,
                        quant_bits=quant_bits,
                        group_size=self.group_size,
                        super_group_size=self.super_group_size,
                        group_scale_bits=self.group_scale_bits,
                        quant_scale=self.quant_scale,
                        symmetric=symmetric,
                        pack=pack,
                    )
                    qlayer.init_parameters( 
                        weight_quanter.quant_weight, 
                        weight_quanter.scales, 
                        weight_quanter.super_scales,
                        weight_quanter.zero_points,
                        weight_quanter.super_zero_points,
                    )

                    logger.info(f"[IQ] {cur_name} Quantized!")
                    # only save quant weight and scale
                    save_state.update({cur_name + '.weight': qlayer.weight,
                                        cur_name + '.scales': qlayer.scales,
                                        cur_name + '.iq_config': qlayer.iq_config,
                    })
                    if getattr(qlayer, "super_scales", None) is not None:
                        save_state.update({cur_name + '.super_scales': qlayer.zero_points})
                    if getattr(qlayer, "zero_points", None) is not None:
                        save_state.update({cur_name + '.zero_points': qlayer.zero_points})
                    if getattr(qlayer, "super_zero_points", None) is not None:
                        save_state.update({cur_name + '.super_zero_points': qlayer.super_zero_points})
                    if getattr(qlayer, "outliers_weight", None) is not None:
                        save_state.update({cur_name + '.outliers_weight': qlayer.outliers_weight})

                    # del weight_quanter, weight, info_matrix, hessian_matrix
                    del qlayer
                    # paddle.device.cuda.empty_cache()
                    # gc.collect()
            return save_state
        
        logger.info(f"[IQ] Final quant layers: [{len(self._target_layers)}] - {self._target_layers}")
                
        self.save_state_dict = _quantize_layer(target_layers)

    def replace_with_quant_layer(self, load_path=None):
        logger.info("[IQ] Begin replace with quant linear...")
        logger.info(f"Load quant model from {load_path}")
        paddle.set_device('cpu')
        qmodel_state = paddle.load(load_path)
        paddle.set_device('gpu')
        all_target_layers = []
        for key in qmodel_state.keys():
            if '.iq_config' in key:
                all_target_layers.append(key)
                self.model.state_dict()[key.replace('iq_config', 'weight')].value().get_tensor()._clear()
        logger.info(f"All target layers: {all_target_layers}")
        logger.info(f"All target layers: {len(all_target_layers)}")
        total_sparse_ratio = []

        def _replace_layer(target_layers):
            sparse_ratios = []
            for cur_name, sub_layer in self.model.named_sublayers():
                layer_name = sub_layer.full_name()
                if isinstance(sub_layer, (ColumnParallelLinear, RowParallelLinear, paddle.nn.Linear)) \
                 and (cur_name + '.iq_config' in target_layers):
                    if cur_name + '.iq_config' not in qmodel_state:
                        continue
                    gather_output, input_is_parallel = None, None
                    if isinstance(sub_layer, ColumnParallelLinear):
                        QuantLayer = IQuantColumnParallelLinear
                        gather_output = sub_layer.gather_output
                    elif isinstance(sub_layer, RowParallelLinear):
                        QuantLayer = IQuantRowParallelLinear
                        input_is_parallel = sub_layer.input_is_parallel
                    else:
                        QuantLayer = IQuantLinear
                    logger.info(f"Replace: {cur_name} --- {layer_name} -> {QuantLayer}")
                    layer_idx = int(cur_name.split(".")[2])
                    quant_config = qmodel_state[cur_name + '.iq_config']
                    quant_bits = quant_config['quant_bits']
                    group_size = quant_config['group_size']
                    quant_scale = quant_config['quant_scale']
                    is_pack = quant_config['pack']

                    hadamard = False
                    symmetric = True

                    isolate_outliers = False
                    if cur_name + '.outliers_weight' in qmodel_state:
                        isolate_outliers = True

                    extract_sign = qmodel_state.get(cur_name + '.extract_sign', False)
                    if extract_sign:
                        weight_sign = paddle.sign(sub_layer.weight)
                    else:
                        weight_sign = None

                    logger.info(f"[IQ] {cur_name} - quant_config: {quant_config}")
                    with paddle.LazyGuard():
                        qlayer = QuantLayer(
                            in_features=sub_layer.weight.shape[0],
                            out_features=sub_layer.weight.shape[1],
                            quant_bits=quant_bits,
                            group_size=group_size,
                            super_group_size=self.super_group_size,
                            group_scale_bits=self.group_scale_bits,
                            quant_scale=quant_scale,
                            symmetric=symmetric,
                            pack=is_pack,
                            isolate_outliers=isolate_outliers,
                            bias=getattr(sub_layer, 'bias', None),
                            hadamard=hadamard,
                            extract_sign=extract_sign,
                            weight_sign=weight_sign,
                            gather_output=gather_output,
                            input_is_parallel=input_is_parallel,
                        )
                    qlayer.init_parameters(
                        quant_weight=qmodel_state[cur_name + '.weight'],
                        scales=qmodel_state[cur_name + '.scales'],
                        super_scales=qmodel_state.get(cur_name + '.super_scales', None),
                        zero_points=qmodel_state.get(cur_name + '.zero_points', None),
                        super_zero_points=qmodel_state.get(cur_name + '.super_zero_points', None),
                        is_pack=is_pack,
                        outliers_weight=qmodel_state.get(cur_name + '.outliers_weight', None),
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
            logger.info(f"[IQ] No need to init quant linear")
            return qmodel_state
        self._thread_num = os.cpu_count()
        if len(all_target_layers) < self._thread_num:
            self._thread_num = 2
        # self._thread_num = 1
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

    def replace_with_quant_layer_for_DS(self, load_path=None):
        logger.info("[IQ] Begin replace with quant linear...")
        logger.info(f"Load quant model from {load_path}")
        paddle.set_device('cpu')
        qmodel_state = paddle.load(load_path)
        paddle.set_device('gpu')
        all_target_layers = []
        for key in qmodel_state.keys():
            if '.iq_config' in key:
                all_target_layers.append(key)
        logger.info(f"All target layers: {all_target_layers}")
        logger.info(f"All target layers: {len(all_target_layers)}")
        total_sparse_ratio = []

        def _replace_layer(target_layers):
            sparse_ratios = []
            for cur_name, sub_layer in self.model.named_sublayers():
                layer_name = sub_layer.full_name()
                if isinstance(sub_layer, (ColumnParallelLinear, RowParallelLinear, paddle.nn.Linear)) \
                 and (cur_name + '.iq_config' in target_layers):
                    if cur_name + '.iq_config' not in qmodel_state:
                        continue
                    gather_output, input_is_parallel = None, None
                    if isinstance(sub_layer, ColumnParallelLinear):
                        QuantLayer = IQuantColumnParallelLinear
                        gather_output = sub_layer.gather_output
                    elif isinstance(sub_layer, RowParallelLinear):
                        QuantLayer = IQuantRowParallelLinear
                        input_is_parallel = sub_layer.input_is_parallel
                    else:
                        QuantLayer = IQuantLinear
                    logger.info(f"Replace: {cur_name} --- {layer_name} -> {QuantLayer}")
                    layer_idx = int(cur_name.split(".")[2])
                    quant_config = qmodel_state[cur_name + '.iq_config']
                    quant_bits = quant_config['quant_bits']
                    group_size = quant_config['group_size']
                    quant_scale = quant_config['quant_scale']
                    is_pack = quant_config['pack']

                    hadamard = False
                    symmetric = True

                    isolate_outliers = False
                    if cur_name + '.outliers_weight' in qmodel_state:
                        isolate_outliers = True

                    extract_sign = qmodel_state.get(cur_name + '.extract_sign', False)
                    if extract_sign:
                        weight_sign = paddle.sign(sub_layer.weight)
                    else:
                        weight_sign = None

                    logger.info(f"[IQ] {cur_name} - quant_config: {quant_config}")
                    with paddle.LazyGuard():
                        qlayer = QuantLayer(
                            in_features=sub_layer.weight.shape[0],
                            out_features=sub_layer.weight.shape[1],
                            quant_bits=quant_bits,
                            group_size=group_size,
                            super_group_size=self.super_group_size,
                            group_scale_bits=self.group_scale_bits,
                            quant_scale=quant_scale,
                            symmetric=symmetric,
                            pack=is_pack,
                            isolate_outliers=isolate_outliers,
                            bias=getattr(sub_layer, 'bias', None),
                            hadamard=hadamard,
                            extract_sign=extract_sign,
                            weight_sign=weight_sign,
                            gather_output=gather_output,
                            input_is_parallel=input_is_parallel,
                        )
                    qlayer.init_parameters(
                        quant_weight=qmodel_state[cur_name + '.quant_weight'],
                        scales=qmodel_state[cur_name + '.quant_scale'],
                        super_scales=qmodel_state.get(cur_name + '.super_scales', None),
                        zero_points=qmodel_state.get(cur_name + '.zero_points', None),
                        super_zero_points=qmodel_state.get(cur_name + '.super_zero_points', None),
                        is_pack=is_pack,
                        outliers_weight=qmodel_state.get(cur_name + '.outliers_weight', None),
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
            logger.info(f"[IQ] No need to init quant linear")
            return
        self._thread_num = os.cpu_count()
        if len(all_target_layers) < self._thread_num:
            self._thread_num = 2
        # self._thread_num = 1
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
        if dp_degree == 1:
            model_path = os.path.join(save_path, "model_state.pdparams")
        else:
            hcg = fleet.get_hybrid_communicate_group()
            dp_id = hcg.get_data_parallel_rank()
            model_path = os.path.join(save_path, f"model_state.seg{dp_id}.pdparams")
           
        logger.info(f"Save quant model to {model_path}")
        paddle.save(self.save_state_dict, model_path)
        logger.info(f"Save quant model done.")
    
    def _cal_hessian(self):
        for key in self.sampled_hessian:
            # H = self.sampled_hessian[key] / self.sampled_num[key]
            # means = self.sampled_means[key] / self.sampled_num[key]
            # m = means.unsqueeze(-1) @ means.unsqueeze(0)
            ### use covariance matrix to estimate the hessian matrix:  E[x^T*x] - E[x^T]E[x],
            # H = H - m
            H = self.sampled_hessian[key]
            self.sampled_hessian[key] = self._regularize_H(H)

        paddle.device.cuda.empty_cache()
        gc.collect()

    def _regularize_H(self, H, sigma=1e-2):
        zero_idx = paddle.where(paddle.diag(H) == 0)
        if not paddle.is_empty(zero_idx):
            H[zero_idx, zero_idx] = 1
        damp = sigma * paddle.mean(paddle.diag(H))
        diag = paddle.arange(H.shape[0])
        H[diag, diag] += damp

        # H = H / paddle.diag(H).mean()
        # idx = paddle.arange(H.shape[0])
        # H[idx, idx] += sigma
        return H

    def _preprocess_H(self, H, percdamp=1e-2):
        zero_idx = paddle.where(paddle.diag(H) == 0)
        if not paddle.is_empty(zero_idx):
            H[zero_idx, zero_idx] = 1.
        columns = H.shape[1]
        damp = percdamp * paddle.mean(paddle.diag(H))
        diag = paddle.arange(columns)
        H[diag, diag] += damp
        return H

    def save_hessian(self, save_path, hessian_nums=128):
        self._remove_hook()
        gc.collect()
        paddle.device.cuda.empty_cache()
        self._cal_hessian()

        logger.info(f"Save hessian to {save_path}")
        paddle.save(self.sampled_hessian, save_path)
        # save_file(save_path, f"sampled_hessian_{hessian_nums}.safetensors", self.sampled_hessian)
        logger.info(f"Save hessian done.")
    