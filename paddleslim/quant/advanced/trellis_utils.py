# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import gc
import paddle
import math
import numpy as np
from paddlenlp.utils.log import logger
from paddle.distributed import fleet
import concurrent.futures
# import torch
from sklearn.cluster import KMeans
# from cuml.cluster import KMeans

class TrellisQuantizer:
    def __init__(
        self,
        lut_bits=4,
        states=4,
        state_bits=2,
        group_size=64,
        lookup_table=None,
        super_group_size=512,
        group_scale_bits=4,
        group_zp_bits=4,
        quant_scale=False,
        symmetric=True,
        isolate_outliers=True,
        hadamard=False,
        enable_norm=False,
        enable_perm=False,
        enable_ldlq=False,
        extract_sign=False,
        parity_sign=False,
        enable_completion=False,
        enable_compensation=False,
        enable_circle=False,
    ):
        self.lut_bits = lut_bits
        self.states = states
        self.state_bits = state_bits
        self.group_size = group_size
        self.super_group_size = super_group_size
        self.extract_sign = extract_sign
        self._symmetric = symmetric
        self.enable_completion = enable_completion
        self.enable_circle = enable_circle
        if self._symmetric and not self.extract_sign and not parity_sign:
            self._qmin = -(2 ** (self.lut_bits - 1))
            self._qmax = 2 ** (self.lut_bits - 1) - 1
        else:
            self._qmin = 0
            self._qmax = (2 ** self.lut_bits) - 1
        # if parity_sign:
        #     self._qmin = -2 ** (self.lut_bits) + 1
        #     self._qmax = 2 ** (self.lut_bits) - 2

        self.quant_scale = quant_scale
        self.group_scale_bits = group_scale_bits
        self.enable_norm = enable_norm
        self.norm_scale, self.norm_bias = None, None
        self.enable_perm = enable_perm
        self.perm = None
        self.enable_ldlq = enable_ldlq

        # All scales have the same symbol
        self._s_qmin = 0
        self._s_qmax = 2 ** self.group_scale_bits - 1
        self.group_zp_bits = group_zp_bits
        self._zp_qmin = -(2 ** (self.group_zp_bits - 1))
        self._zp_qmax = 2 ** (self.group_zp_bits - 1) - 1

        self.super_scales, self.zero_points, self.super_zero_points = None, None, None

        self._isolate = isolate_outliers
        self._thread_num = os.cpu_count() 

        if lookup_table is None:
            self.lookup_table = construct_lookup_table(self.lut_bits, self.states, self.state_bits)
        else:
            self.lookup_table = lookup_table
        if isinstance(self.lookup_table, list):
            logger.info(f"[TQ] Lookup table: {self.lookup_table[0].shape}-{self.lookup_table[1].shape}, LUT bits: {self.lut_bits}, States: {self.states}, State bits: {self.state_bits}" )
        else:
            logger.info(f"[TQ] Lookup table: {self.lookup_table.shape}, LUT bits: {self.lut_bits}, States: {self.states}, State bits: {self.state_bits}, group size:{self.group_size}" )
        if isinstance(self.states, (list, tuple)):
            self.states_length = sum(self.states)
            code_length = 0
            for i in range(len(self.states)):
                code_length += self.lut_bits + (self.states[i] - 1)*self.state_bits[i]

            residual = group_size % self.states_length
            residual_bits = self.lut_bits + (residual - 1) * self.state_bits[0]
            redundant_bits = code_length - residual_bits
        else:
            self.states_length = self.states
            code_length = self.lut_bits + (self.states_length - 1)*self.state_bits

            residual = group_size % self.states_length
            residual_bits = self.lut_bits + (residual - 1) * self.state_bits
            redundant_bits = code_length - residual_bits
        if residual > 0:
            # self.redundant_mask = (2**residual_bits - 1)<< redundant_bits
            self.redundant_mask = ((2**residual_bits - 1) * (2**redundant_bits))
        else:
            redundant_bits = 0
            self.redundant_mask = 0
        self.redundant_bits = redundant_bits
        
        if self.redundant_bits > 0:
            self.group_scale_bits = self.redundant_bits
            self._s_qmax = 2 ** self.group_scale_bits - 1
        self.indices_cluster = None
        self.indices_scale = None
        self.indices_zp = None
        self.parity_sign = parity_sign
        self.enable_compensation = enable_compensation
        logger.info(f"[TQ] redundant_bits: {redundant_bits}, redundant_mask: {self.redundant_mask}, parity_sign: {self.parity_sign}")
        
    def group_quantize(self, weight, info_matrix=None):
        """
        weight: [out_features, in_features]
        info_matrix: [out_features, in_features]
        """
        if self.enable_perm:
            # logger.info(f"Permute weight...")
            # weight = self.permute(weight)
            info_matrix = info_matrix.reshape([-1, self.group_size])[:, self.perm]
        # if self.enable_norm:
        #     logger.info(f"Normalizing weights...")
        #     weight = self.normalize(weight)

        self.out_features, self.in_features = weight.shape
        if self.group_size > -1:
            weight = weight.reshape([weight.shape[0], -1, self.group_size])
            weight = weight.reshape([-1, self.group_size]) # [group numbers, group size]
        else:
            # when group_size=-1, it equals channel-wise quant
            self.group_size = weight.shape[1]

        # add pad
        self.pad = Pad(self.states_length, self.group_size)
        weight = self.pad.add_padding(weight)
        weight_square = weight**2 
        logger.info(f"group-wise weight: {weight.shape} - pad_cols: {self.pad.pad_cols}")

        # thres = paddle.quantile(weight.abs(), q=0.005)
        # self.eliminate_mask = (weight.abs() > thres).cast('bfloat16')

        # cal info matrix for each super group weight
        if info_matrix is not None:
            step = self.super_group_size // self.group_size
            # if self.enable_perm:
            #     info_matrix = info_matrix[:, self.perm]
            info_matrix = info_matrix.reshape([-1, self.group_size])
            info_matrix = self.pad.add_padding(info_matrix)
            assert info_matrix.shape == weight.shape, f"{info_matrix.shape} != {weight.shape}"
            sigma2 = paddle.sum(weight_square, axis=-1, keepdim=True) / self.group_size
            info_matrix *= paddle.sqrt(sigma2 + weight_square)
        else:
            info_matrix = weight_square
            # logger.info(f"info matrix set to be ones...")
            # info_matrix = paddle.ones_like(weight)
        
        min_s, max_s = 0.8, 2.
        step = (max_s - min_s) / 20.
        ranges = paddle.arange(min_s, max_s, step).tolist()
        best_loss = None
        best_scales = None
        ranges.append(1.0)
        ranges = [1.0]
        # search best group scale, but its not good now.
        for idx, s in enumerate(ranges):
            # Firstly, calculate the scale of each group 
            quant_w, scales, zero_points = self.quant(weight.t(), s=s)

            if not self._symmetric:
                self.zero_points = zero_points

            # Secondly, search the best code for each states relying on MSE
            if False:
                logger.info(f"Searching code with MSE...")
                info_m = None
            else:
                # search with weighted MSE has problem...
                logger.info(f"Searching code with Weighted MSE...")
                info_m = info_matrix
             
            if isinstance(self.states, (list, tuple)):
                indices, loss = self.search_code_with_mse_multi(weight, scales, zero_points, info_m) # [group numbers, state_nums]
            else:
                indices, loss = self.search_code_with_mse(weight, scales, zero_points, info_m) # [group numbers, state_nums]
             
            logger.debug(f"loss: {loss.shape}, scales: {scales.shape}")
            if best_loss is None:
                best_loss = loss
                best_scales = scales
            else:
                mask = loss < best_loss
                best_loss[mask] = loss[mask]
                best_scales[mask] = scales[mask]
        scales = best_scales
         
        if len(ranges) > 1:
            if isinstance(self.states, (list, tuple)):
                indices, best_loss = self.search_code_with_mse_multi(weight, scales, zero_points, info_m) # [group numbers, state_nums]
            else:
                indices, best_loss = self.search_code_with_mse(weight, scales, zero_points, info_m) # [group numbers, state_nums]
        
        logger.info(f"indices: {indices.shape}, scales: {scales.shape}") 
        enable_compensation = False
        if enable_compensation:
            logger.info(f"Computing SVD compensations ...")
            qdq_weight = self.cal_dequant_weight(indices, scales, zero_points)
            weight_diff = weight - qdq_weight
            weight_diff = weight_diff.reshape([self.out_features, -1]).cast('float32')
            logger.info(f"weight_diff: {weight_diff.shape}")
            U, S, V = paddle.linalg.svd(weight_diff)
            # best_loss = None
            step = 512
            for k in range(0, S.shape[0], step):
                if k > 0:
                    Uk = U[:, :k]
                    Sk = S[:k]
                    Vk = V[:k, :]
                    svd_weight = (Uk @ paddle.diag(Sk)) @ Vk
                    weight_compensation = qdq_weight.reshape([self.out_features, -1]) + svd_weight
                    weight_compensation = weight_compensation.reshape(qdq_weight.shape).cast('bfloat16')
                else:
                    weight_compensation = weight_diff.reshape(qdq_weight.shape).cast('bfloat16') + qdq_weight
                
                if isinstance(self.states, (list, tuple)):
                    cur_indices, _ = self.search_code_with_mse_multi(weight_compensation, scales, zero_points, info_m) # [group numbers, state_nums]
                else:
                    cur_indices, _ = self.search_code_with_mse(weight_compensation, scales, zero_points, info_m) # [group numbers, state_nums]
                
                tmp_qdq_weight = self.cal_dequant_weight(cur_indices, scales, zero_points)
                cur_loss = ((weight - tmp_qdq_weight)**2).sum(axis=-1)
                
                mask = cur_loss < best_loss
                best_loss[mask] = cur_loss[mask]
                indices[mask] = cur_indices[mask]
        
        # q = 0.05
        # logger.debug(f"Eliminate the small values... q:{q}")
        # thres = paddle.quantile(info_matrix, q=q)
        # info_matrix = (info_matrix > thres).cast('bfloat16') * info_matrix

        ### Lastly, optimize the scale of each group
        scales, zero_points = self.optimize_scale(weight, indices, info_matrix) # [group numbers]

        if self.enable_compensation:
            logger.info(f"[bias] Computing SVD compensations ...")
            qdq_weight = self.cal_dequant_weight(indices, scales, zero_points)
            weight_diff = weight[:, :self.group_size] - qdq_weight[:, :self.group_size]
            weight_diff = weight_diff.reshape([self.out_features, -1]).cast('float32')
            logger.info(f"weight_diff: {weight_diff.shape}")
            U, S, V = paddle.linalg.svd(weight_diff)
            k = 2
            Uk = U[:, :k]
            Sk = S[:k]
            Vk = V[:k, :]
            svd_bias = (Uk @ paddle.diag(Sk)) @ Vk
            logger.info(f"SVD bias shape: {svd_bias.shape}")
            # US = Uk @ paddle.diag(Sk)
            # logger.info(f"US: {US.shape}, Vk: {Vk.shape}")
            self.svd_bias = svd_bias

        self.scales = scales
        self.quant_weight = indices
        if not self._symmetric:
            self.zero_points = zero_points

        if self.quant_scale:
            logger.info(f"Next, quant and optimize super scale...")
            self.super_scales, scales = self.optimize_super_scale(weight, indices, scales, info_matrix)
            logger.info(f"super scales:{self.super_scales.shape}, scales:{scales.shape}")
            self.scales = scales

            '''
            # use dequant scale to search code again
            logger.info(f"Last, search code with quanted scale")
            dequant_scales = scales.reshape([self.out_features, -1]).cast('bfloat16') * self.super_scales.unsqueeze(-1)
            dequant_scales = dequant_scales.reshape([-1]).cast('bfloat16')
            if isinstance(self.states, (list, tuple)):
                indices, _ = self.search_code_with_mse_multi(weight, dequant_scales, zero_points, info_m) # [group numbers, state_nums]
            else:
                indices, _ = self.search_code_with_mse(weight, dequant_scales, zero_points, info_m) # [group numbers, state_nums]
            dequant_scales = dequant_scales.reshape([self.out_features, -1])
            self.super_scales, _ = self.optimize_super_scale(weight, indices, None, info_matrix, dequant_scales, self.super_scales)
            '''
            if self.redundant_mask > 0:
                # Place the scale in the redundant bits of the weight
                indices[:, -1] = indices[:, -1] & paddle.to_tensor(self.redundant_mask, dtype='int32')
                indices[:, -1] = indices[:, -1] + self.scales
                self.scales = None
                # self.scales = scales
                self.quant_weight = indices
        # '''
        # for debug
        if paddle.isinf(scales).any():
            logger.debug(f"scales has inf!")
            inf_indices = paddle.where(paddle.isinf(scales))[0]
            logger.debug(f"inf_indices: {inf_indices.tolist()}")
            max_w = paddle.max(weight[inf_indices], axis=-1).cast("float32")
            logger.debug(f"max_w: {max_w.tolist()}")
            max_info = paddle.max(info_matrix[inf_indices], axis=-1).cast("float32")
            logger.debug(f"max_info: {max_info.tolist()}")
            w_inf = paddle.isinf(weight[inf_indices]).any()
            info_inf = paddle.isinf(info_matrix[inf_indices]).any()
            logger.debug(f"w_inf: {w_inf.item()}, info_inf: {info_inf.item()}")
            scales = paddle.where(paddle.isinf(scales), paddle.zeros_like(scales), scales)
        if paddle.isnan(scales).any():
            logger.debug(f"scales has nan!")
            nan_indices = paddle.where(paddle.isnan(scales))[0]
            logger.debug(f"nan_indices: {nan_indices.tolist()}")
            max_w = paddle.max(weight[nan_indices], axis=0)
            logger.debug(f"max_w: {max_w.tolist()}")
            max_info = paddle.max(info_matrix[nan_indices], axis=-1).cast("float32")
            logger.debug(f"max_info: {max_info.tolist()}")
            w_nan = paddle.isnan(weight[nan_indices]).any()
            info_nan = paddle.isnan(info_matrix[nan_indices]).any()
            logger.debug(f"w_nan: {w_nan.item()}, info_nan: {info_nan.item()}")
            scales = paddle.where(paddle.isnan(scales), paddle.zeros_like(scales), scales)
        # '''
        # for debug
        qdq_weight = self.cal_dequant_weight(indices, scales, zero_points)
        weight = weight[:, :self.group_size]
        qdq_weight = qdq_weight[:, :self.group_size]
        info_matrix = info_matrix[:, :self.group_size]
        # logger.debug(f"test3")
        mse_loss = paddle.mean(((weight - qdq_weight)**2).sum(axis=-1))
        info_mse_loss = paddle.mean((info_matrix*(weight - qdq_weight)**2).sum(axis=-1))
        logger.info(f"[TQ] MSE loss: {mse_loss.item()}, Info MSE Loss: {info_mse_loss.item()}")
        # '''

        # for GPTAQ
        return qdq_weight.reshape([self.out_features, -1])

    def cal_dequant_weight(self, indices, scales, zero_points=None):
        """
        indices: [group numbers, state_nums]
        scales: [group numbers]
        """
        if isinstance(self.states, (list, tuple)):
            quant_weight = self.decode_multi(indices)
        else:
            if self.parity_sign:
                quant_weight = cal_sign(self.lookup_table[indices]).cast('bfloat16')
            elif self.enable_completion:
                quant_weight = decode_twos_complement(self.lookup_table[indices], self.lut_bits)
            else:
                quant_weight = self.lookup_table[indices].cast('bfloat16') + self._qmin

        quant_weight = quant_weight.reshape([quant_weight.shape[0], -1]).cast("bfloat16")
        if self.quant_scale:
            scales = scales.reshape([self.out_features, -1]).cast('float32')
            scales = scales * self.super_scales.unsqueeze(-1)
            scales = scales.reshape([-1])
        if self._symmetric:
            qdq_weight = quant_weight * scales.unsqueeze([-1]).cast("bfloat16")
        else:
            qdq_weight = quant_weight * scales.unsqueeze([-1]).cast("bfloat16") + zero_points.unsqueeze([-1]).cast("bfloat16")
        return qdq_weight 

    def qrazor_quantize(self, weight, info_matrix=None):
        """
        weight: [out_features, in_features]
        info_matrix: [out_features, in_features]
        """
        
        self.out_features, self.in_features = weight.shape
        if self.group_size > -1:
            weight = weight.reshape([weight.shape[0], -1, self.group_size])
            weight = weight.reshape([-1, self.group_size]) # [group numbers, group size]
        else:
            # when group_size=-1, it equals channel-wise quant
            self.group_size = weight.shape[1]

        # add pad
        self.pad = Pad(self.states_length, self.group_size)
        weight = self.pad.add_padding(weight)
        weight_square = weight**2 
        logger.info(f"group-wise weight: {weight.shape} - pad_cols: {self.pad.pad_cols}")

        # cal info matrix for each super group weight
        if info_matrix is not None:
            step = self.super_group_size // self.group_size
            # if self.enable_perm:
            #     info_matrix = info_matrix[:, self.perm]
            info_matrix = info_matrix.reshape([-1, self.group_size])
            info_matrix = self.pad.add_padding(info_matrix)
            assert info_matrix.shape == weight.shape, f"{info_matrix.shape} != {weight.shape}"
            sigma2 = paddle.sum(weight_square, axis=-1, keepdim=True) / self.group_size
            info_matrix *= paddle.sqrt(sigma2 + weight_square)
        else:
            info_matrix = weight_square
        
        gs = 8
        keep_bits = 4
        logger.info(f"razor gs: {gs}, keep_bits: {keep_bits}")
        quant_bits = 8
        self._qmin = -(2 ** (quant_bits - 1))
        self._qmax = 2 ** (quant_bits - 1) - 1

        # Firstly, calculate the scale of each group 
        quant_w, scales, zero_points = self.quant(weight.t())
        quant_w -= self._qmin
        q_w_gs_max = quant_w.t().reshape([-1, gs]).max(axis=-1)
        pos_bits = MSB_pos(q_w_gs_max)
        logger.info(f"pos_bits: {pos_bits.shape}")
        logger.info(f"pos: {pos_bits.min().item()} ~ {pos_bits.max().item()}")
        shift_bits = pos_bits - keep_bits

        best_loss = None
        best_shift = None
        # search best shift bits
        ranges = [0,1,2,3]
        ranges = [0]
        for idx, s in enumerate(ranges):
            tmp_shift_bits = shift_bits - s

            # Secondly, search the best code for each states relying on MSE
            if False:
                logger.info(f"Searching code with MSE...")
                info_m = None
            else:
                # search with weighted MSE has problem...
                logger.info(f"Searching code with Weighted MSE...")
                info_m = info_matrix
       
            indices, loss = self.search_code_with_mse_qrazor(weight, scales, zero_points, info_m, tmp_shift_bits, gs) # [group numbers, state_nums]
            loss = loss.reshape([-1])
            logger.debug(f"loss: {loss.shape}, scales: {scales.shape}")
            if best_loss is None:
                best_loss = loss
                best_shift = tmp_shift_bits
            else:
                mask = loss < best_loss
                best_loss[mask] = loss[mask]
                best_shift[mask] = tmp_shift_bits[mask]
        shift_bits = best_shift
        if len(ranges) > 1:
            logger.info(f"final shift_bits: {shift_bits.shape}, {shift_bits.min().item()}~{shift_bits.max().item()}")
            indices, loss = self.search_code_with_mse_qrazor(weight, scales, zero_points, info_m, shift_bits, gs) # [group numbers, state_nums]
            
        logger.info(f"indices: {indices.shape}") 

        ## Lastly, optimize the scale of each group
        # scales, zero_points = self.optimize_scale(weight, indices, info_matrix) # [group numbers]

        self.scales = scales
        self.quant_weight = indices
        if not self._symmetric:
            self.zero_points = zero_points

        # '''
        # for debug
        if paddle.isinf(scales).any():
            logger.debug(f"scales has inf!")
            inf_indices = paddle.where(paddle.isinf(scales))[0]
            logger.debug(f"inf_indices: {inf_indices.tolist()}")
            max_w = paddle.max(weight[inf_indices], axis=-1).cast("float32")
            logger.debug(f"max_w: {max_w.tolist()}")
            max_info = paddle.max(info_matrix[inf_indices], axis=-1).cast("float32")
            logger.debug(f"max_info: {max_info.tolist()}")
            w_inf = paddle.isinf(weight[inf_indices]).any()
            info_inf = paddle.isinf(info_matrix[inf_indices]).any()
            logger.debug(f"w_inf: {w_inf.item()}, info_inf: {info_inf.item()}")
            scales = paddle.where(paddle.isinf(scales), paddle.zeros_like(scales), scales)
        if paddle.isnan(scales).any():
            logger.debug(f"scales has nan!")
            nan_indices = paddle.where(paddle.isnan(scales))[0]
            logger.debug(f"nan_indices: {nan_indices.tolist()}")
            max_w = paddle.max(weight[nan_indices], axis=0)
            logger.debug(f"max_w: {max_w.tolist()}")
            max_info = paddle.max(info_matrix[nan_indices], axis=-1).cast("float32")
            logger.debug(f"max_info: {max_info.tolist()}")
            w_nan = paddle.isnan(weight[nan_indices]).any()
            info_nan = paddle.isnan(info_matrix[nan_indices]).any()
            logger.debug(f"w_nan: {w_nan.item()}, info_nan: {info_nan.item()}")
            scales = paddle.where(paddle.isnan(scales), paddle.zeros_like(scales), scales)
        
        # for debug
        qdq_weight = self.cal_dequant_weight_qrazor(indices, scales, zero_points, shift_bits, gs)
        weight = weight[:, :self.group_size].reshape([-1, 64])
        qdq_weight = qdq_weight[:, :self.group_size].reshape([-1, 64])
        info_matrix = info_matrix[:, :self.group_size].reshape([-1, 64])
        # logger.debug(f"test3")
        mse_loss = paddle.mean(((weight - qdq_weight)**2).sum(axis=-1))
        info_mse_loss = paddle.mean((info_matrix*(weight - qdq_weight)**2).sum(axis=-1))
        logger.info(f"[TQ] MSE loss: {mse_loss.item()}, Info MSE Loss: {info_mse_loss.item()}")
        # '''

    def cal_dequant_weight_qrazor(self, indices, scales, zero_points, shift_bits, gs):
        """
        indices: [out_features, -1]
        scales: [out_features]
        shift_bits: [group_nums]
        """
        quant_weight = self.lookup_table[indices].reshape([self.out_features, -1])
        quant_weight = quant_weight.reshape([-1, gs]) << shift_bits.unsqueeze(-1)
        quant_weight = quant_weight.cast('bfloat16') + self._qmin
        quant_weight = quant_weight.reshape([self.out_features, -1])

        if self.quant_scale:
            scales = scales.reshape([self.out_features, -1]).cast('float32')
            scales = scales * self.super_scales.unsqueeze(-1)
            scales = scales.reshape([-1])
        if self._symmetric:
            qdq_weight = quant_weight * scales.unsqueeze([-1]).cast("bfloat16")
        else:
            qdq_weight = quant_weight * scales.unsqueeze([-1]).cast("bfloat16") + zero_points.unsqueeze([-1]).cast("bfloat16")
        return qdq_weight 

    def group_quantize_cluster(self, weight, info_matrix=None):
        """
        weight: [out_features, in_features]
        info_matrix: [out_features, in_features]
        """
        self.out_features, self.in_features = weight.shape
        if self.group_size > -1:
            weight = weight.reshape([weight.shape[0], -1, self.group_size])
            weight = weight.reshape([-1, self.group_size]) # [group numbers, group size]
        else:
            # when group_size=-1, it equals channel-wise quant
            self.group_size = weight.shape[1]

        # add pad
        self.pad = Pad(self.states_length, self.group_size)
        weight = self.pad.add_padding(weight)
        weight_square = weight**2 # It can be calculated in advance and transmitted from outside
        logger.info(f"group-wise weight: {weight.shape} - pad_cols: {self.pad.pad_cols}")

        # '''
        # cal info matrix for each super group weight
        if info_matrix is not None:
            step = self.super_group_size // self.group_size
            # if self.enable_perm:
            #     info_matrix = info_matrix[:, self.perm]
            info_matrix = info_matrix.reshape([-1, self.group_size])
            info_matrix = self.pad.add_padding(info_matrix)
            assert info_matrix.shape == weight.shape, f"{info_matrix.shape} != {weight.shape}"
            sigma2 = paddle.sum(weight_square, axis=-1, keepdim=True) / self.group_size
            info_matrix *= paddle.sqrt(sigma2 + weight_square)
        else:
            info_matrix = weight_square
        # '''

        # Firstly, calculate the scale of each group 
        _, scales, zero_points = self.quant(weight.t())
        if not self._symmetric:
            self.zero_points = zero_points
        
        # Secondly, search the best code for each states relying on MSE
        if self.lut_bits > 4 or True:
            logger.info(f"Searching code with MSE...")
            info_m = None
        else:
            # search with weighted MSE has problem...
            logger.info(f"Searching code with Weighted MSE...")
            info_m = info_matrix

        if isinstance(self.states, (list, tuple)):
            indices, _ = self.search_code_with_mse_multi(weight, scales, zero_points, info_m) # [group numbers, state_nums]
        else:
            indices, _ = self.search_code_with_mse(weight, scales, zero_points, info_m) # [group numbers, state_nums]
    
        logger.info(f"indices: {indices.shape}, scales: {scales.shape}") 
        
        if isinstance(self.states, (list, tuple)):
            # The method is bad
            indices, real_indices = self.cluster_search_multi_double(weight, scales, indices, info_matrix)
        else:
            indices, real_indices = self.cluster_search(weight, scales, indices, info_matrix)

        ### Lastly, optimize the scale of each group
        weight = weight[:, :self.group_size]
        info_matrix = info_matrix[:, :self.group_size]
        # info_matrix = paddle.ones_like(weight)
        scales, zero_points = self.optimize_scale(weight, real_indices, info_matrix) # [group numbers]
         
        if self.quant_scale:
            logger.info(f"Next, quant and optimize super scale...")
            self.super_scales, scales = self.optimize_super_scale(weight, real_indices, scales, info_matrix)
            logger.info(f"super scales:{self.super_scales.shape}, scales:{scales.shape}")

        # for debug
        qdq_weight = self.cal_dequant_weight(real_indices, scales, zero_points)
        if weight.shape[1] == self.group_size:
            qdq_weight = qdq_weight[:, :self.group_size]
        mse_loss = paddle.mean(((weight - qdq_weight)**2).sum(axis=-1))
        info_mse_loss = paddle.mean((info_matrix*(weight - qdq_weight)**2).sum(axis=-1))
        logger.info(f"[TQ] MSE loss: {mse_loss.item()}, Info MSE Loss: {info_mse_loss.item()}")
 
        self.scales = scales
        self.quant_weight = indices
        if not self._symmetric:
            self.zero_points = zero_points

    def cluster(self, weight, scales, indices, info_matrix):
        use_kmeans = False
        if self.indices_cluster is None or True:
            logger.info(f"Begin cluster...")

            # indices = indices.flatten()#.unsqueeze(-1)
            # indice_counts = paddle.bincount(indices)
            # _, indices_cluster = paddle.topk(indice_counts, k=256)
            # logger.debug(f"indice_counts: {indice_counts.shape}, indices:{indices_cluster.min().item()}~{indices_cluster.max().item()}")
            if use_kmeans:
                kmeans = KMeans(n_clusters=256, tol=1e-5, init='random', max_iter=50, random_state=0, n_init=1)
                kmeans.fit(indices.flatten().unsqueeze(-1).cast('float32'))
                indices_cluster = kmeans.cluster_centers_
                indices_cluster = paddle.to_tensor(indices_cluster).cast('int32').squeeze(1)
                logger.debug(f"cluster centers: {indices_cluster.shape}, min: {indices_cluster.min().item()}, max: {indices_cluster.max().item()}")

                self.indices_scale = indices_cluster.max() / 255
                self.indices_zp = indices_cluster.min().cast('float32')
                # qdq indices
                indices_cluster = paddle.clip(((indices_cluster-self.indices_zp)/self.indices_scale).round(), min=0, max=255).cast('int32')
                indices_cluster = (indices_cluster * self.indices_scale + self.indices_zp).round().cast('int32')
                indices_cluster = paddle.unique(indices_cluster)
            else:
                # Use uniform quant to map to the range of 0-255
                indices = indices.reshape([self.out_features, -1, indices.shape[-1]]).reshape([self.out_features, -1])
                cluster_nums = 2**(2*self.states)
                logger.info(f"cluster nums: {cluster_nums}")
                self.indices_scale = (indices.max(axis=-1) - indices.min(axis=-1))/ (cluster_nums - 1)
                self.indices_zp = indices.min(axis=-1).cast('float32')
                logger.debug(f"indices_scale: {self.indices_scale.shape}, indices_zp: {self.indices_zp.shape}")
                ### use dq indices to search final indices
                indices_cluster = paddle.arange(cluster_nums, dtype='float32')
                indices_cluster = (indices_cluster * self.indices_scale.unsqueeze(1) + self.indices_zp.unsqueeze(1)).round().cast('int32')
         
            logger.debug(f"indices_cluster:{indices_cluster.min().item()}~{indices_cluster.max().item()}")
            # import pdb; pdb.set_trace()
        else:
            logger.info(f"Use clustered indices")
            indices_cluster = indices
        
        cluster_lookup_table = self.lookup_table[indices_cluster]
        logger.info(f"cluster_lookup_table: {cluster_lookup_table.shape}")

        # search code on cluster_lookup_table
        group_nums, group_size = weight.shape
        table_length, states = cluster_lookup_table.shape[-2:]
        weight = weight.reshape([group_nums, -1, self.states]) # [group numbers, state_nums, states]
        if use_kmeans:
            code_book = cluster_lookup_table.unsqueeze(0).expand([group_nums, table_length, states]).cast('bfloat16') + self._qmin
        else:
            code_book = cluster_lookup_table.unsqueeze(1).expand([self.out_features, group_nums//self.out_features, table_length, states]).cast('bfloat16') + self._qmin
            code_book = code_book.reshape([-1, table_length, states])
        code_book = code_book * scales.unsqueeze([-1, -1])
        cur_loss = paddle.cdist(weight, code_book, p=2.0)
        index = paddle.argmin(cur_loss, axis=-1).cast('float32') # [group_nums, state_nums]
        if use_kmeans:
            real_index =  ((index * self.indices_scale) + self.indices_zp).round().cast('int32')
        else:
            index = index.reshape([self.out_features, -1, index.shape[1]])
            real_index = (index * self.indices_scale.unsqueeze([-1,-2]) + self.indices_zp.unsqueeze([-1,-2])).round().cast('int32')
            real_index = real_index.reshape([-1, real_index.shape[-1]])
        
        logger.info(f"final indices: {real_index.shape}")
        return real_index
    
    def cluster_search(self, weight, scales, indices, info_matrix):
        group_nums, group_size = weight.shape
        state_nums = group_size // self.states_length
        # weight = weight.reshape([group_nums, -1, self.states]) # [group numbers, state_nums, states]
        # info_matrix = info_matrix.reshape([group_nums, -1, self.states_length])
        weight = weight.reshape([self.out_features, -1, state_nums, self.states_length])
        info_matrix = info_matrix.reshape([self.out_features, -1, state_nums, self.states_length])

        if self.extract_sign:
            cluster_nums = 2**(2*self.states - self.states)
        else:
            cluster_nums = 2**(2*self.states)
        cluster_nums = 256 # hard code for int8 dtype
        indices_cluster = paddle.arange(cluster_nums, dtype='float32')

        def cluster_lut(weight, scales, indices_scale, indices_zp, info_matrix, indices_group):
            ### use dq indices to search final indices
            # indices_cluster: [cluster nums]

            indices_quant = (indices_cluster * indices_scale.unsqueeze(-1) + indices_zp.unsqueeze(-1)).round().cast('int32')
            # indices_quant = paddle.clip(indices_quant, 0, self.lookup_table.shape[0]-1)
            logger.debug(f"indices_quant: {indices_quant.shape}, {indices_quant.min().item()}~{indices_quant.max().item()}")

            cluster_lookup_table = self.lookup_table[indices_quant]
            logger.info(f"cluster_lookup_table: {cluster_lookup_table.shape}") # [out_features, cluster_nums, states]

            # search code on cluster_lookup_table
            table_length, states = cluster_lookup_table.shape[-2:]

            code_book = cluster_lookup_table.unsqueeze(1)
            if indices_group == 1:
                # code_book = code_book.expand([self.out_features, group_nums//self.out_features, table_length, states])
                code_book = code_book.expand([weight.shape[0], weight.shape[1], table_length, states])

            if self.parity_sign:
                code_book = cal_sign(code_book).cast("bfloat16")
            elif self.enable_completion:
                code_book = decode_twos_complement(code_book, self.lut_bits).cast('bfloat16')
            else:
                code_book = code_book.cast("bfloat16") + self._qmin
            code_book = code_book.reshape([-1, table_length, states])  # [group numbers, table_length, states]
            s = scales.reshape([-1])
            code_book = code_book * s.unsqueeze([-1, -1])

            weight = weight.reshape([weight.shape[0]*weight.shape[1], state_nums, self.states_length])
            
            assert weight.shape[0] == code_book.shape[0], f"{weight.shape[0]} != {code_book.shape[0]}"
            if True:
                dist = paddle.cdist(weight, code_book, p=2.0) # [group numbers, state_nums, table_length]
                # dist = self.weighted_mse(weight, code_book, info_matrix=None)
            else:
                logger.debug(f"use weighted mse...")
                info_matrix = info_matrix.reshape([info_matrix.shape[0]*info_matrix.shape[1], state_nums, self.states_length])
                assert weight.shape == info_matrix.shape, f"{weight.shape} != {info_matrix.shape}"
                dist = self.weighted_mse(weight, code_book, info_matrix) 
            return dist, indices_quant

        logger.info(f"Begin cluster searching...")
        # Use uniform quant to map to the range of 0-255
        indices = indices.reshape([self.out_features, -1, indices.shape[-1]]).reshape([self.out_features, -1]) # [out_features, all states]
        logger.info(f"cluster nums: {cluster_nums}")
        
        best_loss = None
        indices_group = 1 # indices.shape[1]//state_nums
        logger.debug(f"indices: {indices.shape}, group:{indices_group}")
        assert indices.shape[-1] % indices_group == 0
        if indices_group == 1:
            indices = indices.reshape([self.out_features, -1])
        else:
            indices = indices.reshape([self.out_features, indices_group, -1])
        scales = scales.reshape([self.out_features, -1])
        indices = paddle.sort(indices, axis=-1)
        max_q = 0.1
        iters = 10
        for i in range(0, iters+1):
            if i == 0:
                indices_scale = (indices.max(axis=-1) - indices.min(axis=-1))/ (cluster_nums - 1)
                indices_zp = indices.min(axis=-1).cast('float32')
            else:
                q = i * max_q / iters
                indices_zp = paddle.quantile(indices.cast('float32'), q=q, axis=-1)
                max_value = indices.max(axis=-1).cast('float32')
                indices_scale = (max_value - indices_zp) / (cluster_nums - 1)
            # indices_scale = indices_scale.cast('bfloat16')
            # indices_zp = indices_zp.cast('bfloat16')
            logger.debug(f"indices_scale: {indices_scale.shape}, indices_zp: {indices_zp.shape}")

            if cluster_nums >=512:
                bs_nums = 8
            elif cluster_nums >= 256:
                bs_nums = 4
            else:
                bs_nums = 1
            step = self.out_features // bs_nums
            if step < 1:
                step = self.out_features
            dist = []
            for idx in range(0, self.out_features, step):
                end_idx = min(idx+step, self.out_features)
                logger.info(f"{idx}-{end_idx}")
                cur_dist, _ = cluster_lut(weight[idx:end_idx], scales[idx:end_idx], indices_scale[idx:end_idx], indices_zp[idx:end_idx], info_matrix[idx:end_idx], indices_group)
                dist.append(cur_dist.min(axis=-1))
            dist = paddle.concat(dist, axis=0)
            # Prevent accessing out of bounds when the matrix is too large
            if indices_group == 1:
                judge_loss = dist.reshape([self.out_features, -1, state_nums]).sum(axis=[-1,-2])
            else:
                judge_loss = dist.reshape([self.out_features, indices_group, -1]).sum(axis=-1)
            if best_loss is None:
                best_loss = judge_loss
                self.indices_scale = indices_scale
                self.indices_zp = indices_zp
            else:
                mask = best_loss > judge_loss
                best_loss[mask] = judge_loss[mask]
                self.indices_scale[mask] = indices_scale[mask]
                self.indices_zp[mask] = indices_zp[mask]
            paddle.device.cuda.empty_cache()

        # dist, indice_map = cluster_lut(weight, scales, self.indices_scale, self.indices_zp, info_matrix, indices_group)
        if cluster_nums >=512:
            bs_nums = 8
        elif cluster_nums >= 256:
            bs_nums = 4
        else:
            bs_nums = 1
        step = self.out_features // bs_nums
        if step < 1:
            step = self.out_features
        index = []
        for idx in range(0, self.out_features, step):
            end_idx = min(idx+step, self.out_features)
            logger.info(f"{idx}-{end_idx}")
            cur_dist, _ = cluster_lut(weight[idx:end_idx], scales[idx:end_idx], self.indices_scale[idx:end_idx], self.indices_zp[idx:end_idx], info_matrix[idx:end_idx], indices_group)
            index.append(cur_dist.argmin(axis=-1))
        index = paddle.concat(index, axis=0).cast('int32')  # [group_nums, state_nums]
        
        logger.info(f"index: {index.shape}, {index.min().item()}~{index.max().item()}")
        if indices_group == 1:
            real_index = index.reshape([self.out_features, -1, index.shape[1]]).reshape([self.out_features, -1])
            # # optimize index scale
            # logger.debug(f"new index: {index.shape}")
            # ori_index = paddle.take_along_axis(indice_map, index, axis=-1)
            # logger.debug(f"ori_index: {ori_index.shape}, {ori_index.min().item()}~{ori_index.max().item()}")
            # assert index.shape == ori_index.shape, f"{index.shape} != {ori_index.shape}"
            # sum_wq = paddle.sum(ori_index * index, axis=[-1]).cast('float32')
            # sum_q2 = paddle.sum(index**2, axis=[-1]).cast('float32')
            # q_sum = paddle.sum(index, axis=[-1]).cast('float32')
            # new_scales = (sum_wq - self.indices_zp*q_sum) / sum_q2
            # logger.debug(f"new_scales: {new_scales.shape}")
            # self.indices_scale = new_scales
            real_index = (real_index.cast(self.indices_scale.dtype) * self.indices_scale.unsqueeze([-1]) + self.indices_zp.unsqueeze([-1])).round().cast('int32') # [out_features, -1]
            real_index = real_index.reshape([-1, state_nums])
        else:
            real_index = index.reshape([self.out_features, -1, index.shape[1]]).reshape([self.out_features, indices_group, -1])
            real_index = (real_index.cast(self.indices_scale.dtype) * self.indices_scale.unsqueeze([-1]) + self.indices_zp.unsqueeze([-1])).round().cast('int32') # [out_features, indices_group, -1]
            real_index = real_index.reshape([-1, state_nums])
        logger.info(f"final indices: {real_index.shape}, {real_index.min().item()}~{real_index.max().item()}")
        return index, real_index

    def cluster_search_channel(self, weight, scales, indices, info_matrix):
        group_nums, group_size = weight.shape
        assert group_size == self.in_features + self.pad.pad_cols, f"{group_size} != {self.in_features}+{self.pad.pad_cols}"
        state_nums = group_size // self.states_length
        weight = weight.reshape([group_nums, -1, self.states_length]) # [group numbers, state_nums, states]
        info_matrix = info_matrix.reshape([group_nums, -1, self.states_length])
        def cluster_lut(weight, indices_cluster, indices_scale, indices_zp, info_matrix, indices_group, group_len):
            ### use dq indices to search final indices
            # if indices_group != 1:
            #     new_indices_scales = []
            #     new_indices_zp = []
            #     # expand indices_scale and zp
            #     for i in range(indices_group):
            #         tmp_scales = indices_scale[:, i].unsqueeze(-1).expand([self.out_features, group_len])
            #         tmp_zp = indices_zp[:, i].unsqueeze(-1).expand([self.out_features, group_len])
            #         new_indices_scales.append(tmp_scales)
            #         new_indices_zp.append(tmp_zp)
            #     indices_scale = paddle.concat(new_indices_scales, axis=-1)
            #     indices_zp = paddle.concat(new_indices_zp, axis=-1)

            indices_quant = (indices_cluster * indices_scale.unsqueeze(-1) + indices_zp.unsqueeze(-1)).round().cast('int32')
            logger.debug(f"indices_quant: {indices_quant.shape}, {indices_quant.min().item()}~{indices_quant.max().item()}")

            cluster_lookup_table = self.lookup_table[indices_quant]
            logger.info(f"cluster_lookup_table: {cluster_lookup_table.shape}") # [out_features, indices_group, cluster_nums, states]
            code_book = cluster_lookup_table.cast("bfloat16") + self._qmin
            code_book = code_book * scales.unsqueeze([-1, -1, -1])

            # weight: [out_features, state_nums, states] -> [out_features, indices_group, state_nums, states]
            weight = weight.reshape([self.out_features, indices_group, -1, self.states_length])            
            # logger.info(f"weight: {weight.shape}")
            
            if True:
                dist = paddle.cdist(weight, code_book, p=2.0) # [out_features, indices_group, state_nums, cluster_nums]
            else:
                logger.debug(f"use weighted mse...")
                dist = self.weighted_mse(weight, code_book, info_matrix) 
            return dist

        logger.info(f"Begin cluster searching...")
        # Use uniform quant to map to the range of 0-255
        indices = indices.reshape([self.out_features, -1, indices.shape[-1]]).reshape([self.out_features, -1]) # [out_features, all states]
        
        if self.extract_sign:
            cluster_nums = 2**(2*self.states - self.states + 1)
        else:
            cluster_nums = 2**(2*self.states)
        logger.info(f"cluster nums: {cluster_nums}")
        indices_cluster = paddle.arange(cluster_nums, dtype='float32')
        best_loss = None
        group_len = group_size // 64 // 2
        indices_group = indices.shape[-1] // group_len
        logger.debug(f"indices: {indices.shape}, group:{indices_group}")
        assert indices.shape[-1] % indices_group == 0
        if indices_group == 1:
            indices = indices.reshape([self.out_features, -1])
        else:
            indices = indices.reshape([self.out_features, indices_group, -1])
        for i in range(0, 50):
            if i == 0:
                indices_scale = (indices.max(axis=-1) - indices.min(axis=-1))/ (cluster_nums - 1)
                indices_zp = indices.min(axis=-1).cast('float32')
            else:
                q = i * 0.002
                # indices_zp = paddle.quantile(indices.cast('float32'), q=q, axis=-1)
                q = int(i * 0.002 * indices.shape[-1]) - 1
                # indices_zp = paddle.quantile(indices.cast('float32'), q=q, axis=-1)
                indices_zp = indices[:, :, q].cast('float32')
                max_value = indices.max(axis=-1).cast('float32')
                indices_scale = (max_value - indices_zp) / (cluster_nums - 1)
            logger.debug(f"indices_scale: {indices_scale.shape}, indices_zp: {indices_zp.shape}")

            dist = cluster_lut(weight, indices_cluster, indices_scale, indices_zp, info_matrix, indices_group, group_len)
            # judge_loss = dist.min(axis=-1).reshape([self.out_features, -1, state_nums]).sum(axis=[1,2]) # [out features]
            judge_loss = dist.min(axis=-1).sum(axis=-1) # [out_features, indices_group]
            if best_loss is None:
                best_loss = judge_loss
                self.indices_scale = indices_scale
                self.indices_zp = indices_zp
            else:
                mask = best_loss > judge_loss
                best_loss[mask] = judge_loss[mask]
                self.indices_scale[mask] = indices_scale[mask]
                self.indices_zp[mask] = indices_zp[mask]
        
        dist = cluster_lut(weight, indices_cluster, self.indices_scale, self.indices_zp, info_matrix, indices_group, group_len)
        index = paddle.argmin(dist, axis=-1).cast('float32')  # [out_features, indices_group, state_nums]
        # logger.info(f"index: {index.shape}")
        real_index = (index * self.indices_scale.unsqueeze([-1]) + self.indices_zp.unsqueeze([-1])).round().cast('int32') # [out_features, indices_group, -1]
        real_index = real_index.reshape([self.out_features, -1])

        logger.info(f"final indices: {real_index.shape}")
        # import pdb; pdb.set_trace()
        return real_index

    def cluster_search_old(self, weight, scales, indices, info_matrix):
        group_nums, group_size = weight.shape
        state_nums = group_size // self.states_length
        weight = weight.reshape([group_nums, -1, self.states]) # [group numbers, state_nums, states]
        info_matrix = info_matrix.reshape([group_nums, -1, self.states_length])
        def cluster_lut(weight, indices_cluster, indices_scale, indices_zp, info_matrix):
            ### use dq indices to search final indices
            indices_quant = (indices_cluster * indices_scale.unsqueeze(-1) + indices_zp.unsqueeze(-1)).round().cast('int32')
            logger.debug(f"indices_quant:{indices_quant.min().item()}~{indices_quant.max().item()}")
            cluster_lookup_table = self.lookup_table[indices_quant].cast("bfloat16")
            logger.info(f"cluster_lookup_table: {cluster_lookup_table.shape}") # [out_features, 256, states]

            # search code on cluster_lookup_table
            table_length, states = cluster_lookup_table.shape[-2:]

            code_book = cluster_lookup_table.unsqueeze(1).expand([self.out_features, group_nums//self.out_features, table_length, states])
            code_book = code_book.cast("bfloat16") + self._qmin
            code_book = code_book.reshape([-1, table_length, states])  # [group numbers, table_length, states]
            code_book = code_book * scales.unsqueeze([-1, -1])
            if True:
                dist = paddle.cdist(weight, code_book, p=2.0) # [group numbers, state_nums, table_length]
            else:
                logger.debug(f"use weighted mse...")
                dist = self.weighted_mse(weight, code_book, info_matrix) 
            return dist

        logger.info(f"Begin cluster searching...")
        # Use uniform quant to map to the range of 0-255
        indices = indices.reshape([self.out_features, -1, indices.shape[-1]]).reshape([self.out_features, -1]) # [out_features, all states]
        
        if self.extract_sign:
            cluster_nums = 2**(2*self.states - self.states + 1)
        else:
            cluster_nums = 2**(2*self.states)
        logger.info(f"cluster nums: {cluster_nums}")
        indices_cluster = paddle.arange(cluster_nums, dtype='float32')
        best_loss = None
        for i in range(0, 1):
            if i == 0:
                indices_scale = (indices.max(axis=-1) - indices.min(axis=-1))/ (cluster_nums - 1)
                indices_zp = indices.min(axis=-1).cast('float32')
            else:
                q = i * 0.002
                indices_zp = paddle.quantile(indices.cast('float32'), q=q, axis=-1)
                # max_value = paddle.quantile(indices.cast('float32'), q=1-q, axis=-1)
                max_value = indices.max(axis=-1).cast('float32')
                indices_scale = (max_value - indices_zp) / (cluster_nums - 1)
            logger.debug(f"indices_scale: {indices_scale.shape}, indices_zp: {indices_zp.shape}")

            dist = cluster_lut(weight, indices_cluster, indices_scale, indices_zp, info_matrix)
            judge_loss = dist.min(axis=-1).reshape([self.out_features, -1, state_nums]).sum(axis=[1,2]) # [out features]
            if best_loss is None:
                best_loss = judge_loss
                self.indices_scale = indices_scale
                self.indices_zp = indices_zp
            else:
                mask = best_loss > judge_loss
                best_loss[mask] = judge_loss[mask]
                self.indices_scale[mask] = indices_scale[mask]
                self.indices_zp[mask] = indices_zp[mask]
        
        dist = cluster_lut(weight, indices_cluster, self.indices_scale, self.indices_zp, info_matrix)
        index = paddle.argmin(dist, axis=-1).cast('float32') # [group_nums, state_nums]
        
        index = index.reshape([self.out_features, -1, index.shape[1]]).reshape([self.out_features, -1])
        real_index = (index * self.indices_scale.unsqueeze([-1]) + self.indices_zp.unsqueeze([-1])).round().cast('int32') # [out_features, -1]
        real_index = real_index.reshape([-1, state_nums])

        logger.info(f"final indices: {real_index.shape}")
        return real_index

    def cluster_search_scale(self, weight, scales, indices, info_matrix):
        group_nums, group_size = weight.shape
        state_nums = group_size // self.states_length
        weight = weight.reshape([group_nums, -1, self.states]) # [group numbers, state_nums, states]
        def cluster_lut(weight, indices_cluster, indices_scale, indices_zp):
            ### use dq indices to search final indices
            indices_quant = (indices_cluster * indices_scale.unsqueeze(1) + indices_zp.unsqueeze(1)).round().cast('int32')
            logger.debug(f"indices_quant:{indices_quant.min().item()}~{indices_quant.max().item()}")
            cluster_lookup_table = self.lookup_table[indices_quant]
            logger.info(f"cluster_lookup_table: {cluster_lookup_table.shape}")

            # search code on cluster_lookup_table
            table_length, states = cluster_lookup_table.shape[-2:]

            code_book = cluster_lookup_table.unsqueeze(1).expand([self.out_features, group_nums//self.out_features, table_length, states]).cast('bfloat16') + self._qmin
            code_book = code_book.reshape([-1, table_length, states])
            real_code_book = code_book * scales.unsqueeze([-1, -1]) # [group numbers, table_length, states]
            dist = paddle.cdist(weight, real_code_book, p=2.0) # [group numbers, state_nums, table_length]
            return dist, code_book

        logger.info(f"Begin cluster searching2...")
        # Use uniform quant to map to the range of 0-255
        indices = indices.reshape([self.out_features, -1, indices.shape[-1]]).reshape([self.out_features, -1])
        cluster_nums = 2**(2*self.states)
        logger.info(f"cluster nums: {cluster_nums}")
        indices_cluster = paddle.arange(cluster_nums, dtype='float32')

        self.indices_scale = (indices.max(axis=-1) - indices.min(axis=-1))/ (cluster_nums - 1)
        self.indices_zp = indices.min(axis=-1).cast('float32')
        logger.debug(f"indices_scale: {self.indices_scale.shape}, indices_zp: {self.indices_zp.shape}")

        dist, code_book = cluster_lut(weight, indices_cluster, self.indices_scale, self.indices_zp)
        best_loss = dist.min(axis=-1).sum(axis=-1) # [group_nums]
        min_s = 0.5
        max_s = 1.5
        step = (max_s - min_s) / 100.
        ranges = paddle.arange(min_s, max_s, step)
        for i in ranges:
            new_scales = scales * i
            real_code_book = code_book * new_scales.unsqueeze([-1, -1])
            dist = paddle.cdist(weight, real_code_book, p=2.0) # [group numbers, state_nums, table_length]
            cur_loss = dist.min(axis=-1).sum(axis=-1) # [group_nums]
            mask = cur_loss < best_loss
            best_loss[mask] = cur_loss[mask]
            scales[mask] = new_scales[mask]

        real_code_book = code_book * scales.unsqueeze([-1, -1])
        dist = paddle.cdist(weight, real_code_book, p=2.0) # [group numbers, state_nums, table_length]
        index = paddle.argmin(dist, axis=-1).cast('float32') # [group_nums, state_nums]
        
        index = index.reshape([self.out_features, -1, index.shape[1]])
        real_index = (index * self.indices_scale.unsqueeze([-1,-2]) + self.indices_zp.unsqueeze([-1,-2])).round().cast('int32')
        real_index = real_index.reshape([-1, real_index.shape[-1]])
        logger.info(f"final indices: {real_index.shape}")
        return real_index

    def search_code_with_mse(self, weight, scales, zero_points, info_matrix):
        """
        weight: [group numbers, group size]
        scales: [group numbers]
        """
        group_nums, group_size = weight.shape
        table_length, states = self.lookup_table.shape
        weight = weight.reshape([group_nums, -1, self.states_length]) # [group numbers, state_nums, states]
        if info_matrix is not None:
            info_matrix = info_matrix.reshape([group_nums, -1, self.states_length])

        if self.lookup_table.shape[0] >= 2**20:
            bs_nums = 2000
        elif self.lookup_table.shape[0] >= 2**16:
            bs_nums = 400
        elif self.lookup_table.shape[0] >= 2**13:
            bs_nums = 100 
        elif self.lookup_table.shape[0] >= 2**10:
            bs_nums = 50
        else:
            bs_nums = 100 # must be greater than 4, otherwise, the min() will cause cuda error 700.
        if self.group_size <  64:
            bs_nums *= 4
        step = group_nums // bs_nums
        if step < 1:
            step = group_nums
        indices = []
        loss = []
        for idx in range(0, group_nums, step):
            end_idx = min(idx+step, group_nums)
            # dequant the code
            # [length, states] -> [group nums, length, states]
            code_book = self.lookup_table.unsqueeze(0).expand([end_idx-idx, table_length, states])
            
            if self.parity_sign:
                code_book = cal_sign(code_book).cast('bfloat16')
            elif self.enable_completion:
                code_book = decode_twos_complement(code_book, self.lut_bits).cast('bfloat16')
            else:
                code_book = code_book.cast('bfloat16')+ self._qmin # !!!

            # logger.debug(f"code book: {code_book.shape} {idx}:{end_idx}")
            if self._symmetric:
                code_book = code_book * scales[idx:end_idx].unsqueeze([-1, -1])
            else:
                code_book = code_book * scales[idx:end_idx].unsqueeze([-1, -1]) + zero_points[idx:end_idx].unsqueeze([-1, -1])
            if info_matrix is None:
                cur_loss = paddle.cdist(weight[idx:end_idx], code_book, p=2.0)
            else:
                cur_loss = self.weighted_mse_old(weight[idx:end_idx], code_book, info_matrix[idx:end_idx])
            cur_index = paddle.argmin(cur_loss, axis=-1).cast('int32') #[group numbers, state_nums]
            # logger.info(f"cur loss: {cur_loss.shape}")
            loss.append(cur_loss.min(axis=-1).sum(axis=-1))
            indices.append(cur_index)
        indices = paddle.concat(indices, axis=0)
        loss = paddle.concat(loss, axis=0)
        return indices, loss

    def search_code_with_mse_qrazor(self, weight, scales, zero_points, info_matrix, shift_bits, gs):
        """
        weight: [out_features, in_features]
        scales: [out_features]
        pos_bits: [group nums]
        """
        weight = weight.reshape([self.out_features, -1, gs])
        scales = scales.unsqueeze(-1).expand([self.out_features, weight.shape[1]])

        _, group_nums = weight.shape[0:2]
        table_length, states = self.lookup_table.shape
        weight = weight.reshape(weight.shape[0:2] + [-1, self.states_length]) 
        
        shift_bits = shift_bits.reshape([self.out_features, -1]).unsqueeze([-1,-1])
        logger.info(f"scales: {scales.shape}, shift_bits: {shift_bits.shape}")
        logger.info(f"shifts: {shift_bits.min().item()} ~ {shift_bits.max().item()}")

        if self.lookup_table.shape[0] >= 2**20:
            bs_nums = 2000
        elif self.lookup_table.shape[0] >= 2**16:
            bs_nums = 400
        elif self.lookup_table.shape[0] >= 2**13:
            bs_nums = 100 
        elif self.lookup_table.shape[0] >= 2**10:
            bs_nums = 50
        else:
            bs_nums = 4
        if gs <  64:
            bs_nums *= 4
        step = self.out_features // bs_nums
        if step < 1:
            step = self.out_features
        indices = []
        loss = []
        for idx in range(0, self.out_features, step):
            end_idx = min(idx+step, self.out_features)
            # dequant the code
            # [length, states] -> [group nums, length, states]
            code_book = self.lookup_table.unsqueeze([0,1]).expand([end_idx-idx, group_nums, table_length, states])
            
            code_book = code_book << shift_bits[end_idx-idx] # for razor shifts
            
            if self.parity_sign:
                code_book = cal_sign(code_book).cast('bfloat16')
            elif enable_completion:
                code_book = decode_twos_complement(code_book, self.lut_bits).cast('bfloat16')
            else:
                code_book = code_book.cast('bfloat16')+ self._qmin # !!!

            # logger.debug(f"code book: {code_book.shape} {idx}:{end_idx}")
            if self._symmetric:
                code_book = code_book * scales[idx:end_idx].unsqueeze([-1, -1])
            else:
                code_book = code_book * scales[idx:end_idx].unsqueeze([-1, -1]) + zero_points[idx:end_idx].unsqueeze([-1, -1])
            if info_matrix is None:
                cur_loss = paddle.cdist(weight[idx:end_idx], code_book, p=2.0)
            else:
                cur_loss = self.weighted_mse(weight[idx:end_idx], code_book, info_matrix[idx:end_idx])
            cur_index = paddle.argmin(cur_loss, axis=-1).cast('int32') #[group numbers, state_nums]
            loss.append(cur_loss.min(axis=-1).sum(axis=-1))
            indices.append(cur_index)
        indices = paddle.concat(indices, axis=0)
        loss = paddle.concat(loss, axis=0)
        return indices, loss

    def weighted_mse(self, weight, code_book, info_matrix=None):
        """
        weight: [group numbers, state_nums, states]
        code_book: [group numbers, table_length, states]
        info_matrix: [group numbers, state_nums, states]
        d^2 = A^2 + B^2 - 2 * A@B.T # A^2 could be omitted
        """
        thres = paddle.quantile(info_matrix, q=0.05)
        # info_matrix = (info_matrix > thres).cast('bfloat16')
        info_matrix = paddle.ones_like(info_matrix)
        if info_matrix is not None:
            assert weight.shape == info_matrix.shape, f"{weight.shape}!={info_matrix.shape}"
            # A_sq = (info_matrix *(weight**2)).sum(axis=-1, keepdim=True) # [group numbers, state_nums, 1]
            code2 = (code_book**2).unsqueeze(1) # [group numbers, 1, table_length, states]
            B_sq = (info_matrix.unsqueeze(2) * code2).sum(axis=-1) # [group numbers, state_nums, table_length]
            # dist = A_sq + B_sq - 2 * ((info_matrix * weight) @ code_book.transpose([0, 2, 1])) # [group numbers, state_nums, table_length]
            dist = B_sq - 2 * ((info_matrix * weight) @ code_book.transpose([0, 2, 1]))
        else:
            # A_sq = (weight**2).sum(axis=-1, keepdim=True) # [group numbers, state_nums, 1]
            B_sq = (code_book**2).sum(axis=-1).unsqueeze(1) # [group numbers, 1, table_length]
            dist = B_sq - 2 * (weight @ code_book.transpose([0, 2, 1]))
        return dist
    
    def weighted_mse_old(self, weight, code_book, info_matrix):
        """
        weight: [group numbers, state_nums, states]
        code_book: [group numbers, table_length, states]
        info_matrix: [group numbers, state_nums, states]
        """
        assert weight.shape == info_matrix.shape
        weight = weight.unsqueeze(-2)
        code_book = code_book.unsqueeze(1)
        info_matrix = info_matrix.unsqueeze(-2)
        
        # diff = paddle.cdist(weight, code_book, p=2)
        diff = weight - code_book
        # logger.info(f"weight: {weight.shape}, code_book: {code_book.shape}, info_matrix: {info_matrix.shape}, diff: {diff.shape}")
        info_diff = info_matrix * (diff**2) # [group numbers, state_nums, table_length, states]
        mse_loss = info_diff.sum(axis=-1)
        # logger.info(f"weighted mse_loss: {mse_loss.shape}")
        return mse_loss

    def search_code_with_mse_multi(self, weight, scales, zero_points, info_matrix):
        """
        for multi code formats
        weight: [group numbers, group size]
        scales: [group numbers]
        """
        group_nums, group_size = weight.shape
        weight = weight.reshape([group_nums, -1, self.states_length]) # [group numbers, state_nums, states]
        weight = paddle.split(weight, self.states, axis=-1)
        if info_matrix is not None:
            info_matrix = info_matrix.reshape([group_nums, -1, self.states_length])
            info_matrix = paddle.split(info_matrix, self.states, axis=-1)
        total_indices = []
        total_loss = []
        for i in range(len(self.states)):
            cur_weight = weight[i]
            if info_matrix is not None:
                cur_info = info_matrix[i]
            logger.info(f"search for state: {self.states[i]}, cur_weight:{cur_weight.shape}")
            lookup_table = self.lookup_table[i]
            table_length, states = lookup_table.shape
            assert states == self.states[i]
            if lookup_table.shape[0] >= 2**16:
                bs_nums = 2000
            if lookup_table.shape[0] > 2**15:
                bs_nums = 150
            elif lookup_table.shape[0] > 2**10:
                bs_nums = 50
            else:
                bs_nums = 10
            step = group_nums // bs_nums
            if step < 1:
                step = group_nums
            indices = []
            loss = []
            for idx in range(0, group_nums, step):
                end_idx = min(idx+step, group_nums)
                # dequant the code
                # [length, states] -> [group nums, length, states]
                code_book = lookup_table.unsqueeze(0).expand([end_idx-idx, table_length, states])
                if self.parity_sign:
                    code_book = cal_sign(code_book).cast('bfloat16') 
                elif self.enable_completion:
                    quant_weight = decode_twos_complement(code_book, self.lut_bits).cast('bfloat16') 
                else:
                    code_book = code_book.cast('bfloat16') + self._qmin
                # logger.debug(f"code book: {code_book.shape} {idx}:{end_idx}")
                if self._symmetric:
                    code_book = code_book.cast("bfloat16") * scales[idx:end_idx].unsqueeze([-1, -1])
                else:
                    code_book = code_book.cast("bfloat16") * scales[idx:end_idx].unsqueeze([-1, -1]) + zero_points[idx:end_idx].unsqueeze([-1, -1])
                if info_matrix is None:
                    cur_loss = paddle.cdist(cur_weight[idx:end_idx], code_book, p=2.0)
                else:
                    cur_loss = self.weighted_mse(cur_weight[idx:end_idx], code_book, cur_info[idx:end_idx])
                cur_index = paddle.argmin(cur_loss, axis=-1).cast('int32') #[group numbers, state_nums]
                indices.append(cur_index)
                loss.append(cur_loss.min(axis=-1).sum(axis=-1))
            indices = paddle.concat(indices, axis=0)
            loss = paddle.concat(loss, axis=0)
            total_indices.append(indices)
            total_loss.append(loss)
        total_loss = paddle.stack(total_loss, axis=-1).sum(axis=-1)
        final_indices = None
        code_length = 0
        for i in reversed(range(len(total_indices))):
            if i == len(total_indices) - 1:
                final_indices = total_indices[i]
            else:
                code_length += int(math.log2(self.lookup_table[i+1].shape[0]))
                # final_indices += (total_indices[i] << code_length)
                final_indices += (total_indices[i] * (2**code_length))
        # import pdb; pdb.set_trace()
        return final_indices, total_loss

    def group_quantize_with_LDLQ(self, weight, hessian_matrix, info_matrix=None):
        """
        GPTQ is a special case of LDLQ
        weight: [out_features, in_features]
        hessian_matrix: [out_features, in_features]
        """
        self.out_features, self.in_features = weight.shape
        logger.info(f"[TQ] Using LDLQ to update weight...")

        qdq_weight, indices, scales = self.GPTQ(weight, hessian_matrix, info_matrix)
        # qdq_weight, indices, scales = self.coordinate_descent(weight, hessian_matrix)
        logger.info(f"qdq_weight: {qdq_weight.shape}, indices: {indices.shape}, scales: {scales.shape}")
        self.scales = scales
        self.quant_weight = indices

        if self.quant_scale:
            logger.info(f"Next, quant and optimize super scale...")
            weight = weight.reshape([-1, self.group_size])
            indices = indices.reshape(scales.shape + [-1])
            info_matrix = info_matrix.reshape([-1, self.group_size])
            self.super_scales, scales = self.optimize_super_scale(weight, indices, scales, info_matrix)
            logger.info(f"super scales:{self.super_scales.shape}, scales:{scales.shape}")
            
            if self.redundant_mask > 0:
                # Place the scale in the redundant bits of the weight
                indices = indices.reshape([-1, indices.shape[-1]])
                indices[:, -1] = indices[:, -1] & paddle.to_tensor(self.redundant_mask, dtype='int32')
                indices[:, -1] = indices[:, -1] + scales
                self.scales = None
                # self.scales = scales
                self.quant_weight = indices
        
            qdq_weight = self.cal_dequant_weight(indices, scales)
            qdq_weight = qdq_weight[:, :self.group_size]

        weight = weight.reshape([-1, self.group_size])
        qdq_weight = qdq_weight.reshape([-1, self.group_size])
        mse_loss = paddle.mean(((weight - qdq_weight)**2).sum(axis=-1))
        logger.info(f"[TQ] MSE loss: {mse_loss.item()}")


    def LDLQ_old(self, weight, hessian_matrix):
        """
        weight: [out_features, in_features]
        hessian_matrix: [in_features, in_features]
        """
        m, n = weight.shape
        state_nums = math.ceil(self.group_size / self.states)
        group_nums = n // self.group_size

        indices = []
        L = self.block_LDL(hessian_matrix, self.group_size) # [in_features, in_features]
        prod_cache = paddle.zeros_like(weight)
        weight_hat = weight.clone() # fake quant weight
        scales = paddle.zeros([m, group_nums])
        for idx in range(n, 0, -self.group_size):
            idx1 = max(idx - self.group_size, 0)
            count = idx - idx1
            block_W = weight[:, idx1:idx]
            block_W_hat = weight_hat[:, idx1:idx]
            block_L = L[idx1:idx]
            block_prod = prod_cache[:, idx1:idx]
            offset = idx1
            block_index = [] 

            # cal scale for cur group
            cur_group_idx = idx1 // self.group_size
            # scale = block_W.square().mean(axis=-1).sqrt() # [out_features]
            scale = block_W.max(axis=-1) - block_W.min(axis=-1)
            scale = scale / (self._qmax - self._qmin)
            # block_W /= scale.unsqueeze([-1])
            # weight[:, idx1:idx] = block_W
            # logger.debug(f"scale1: {scale[0].item()}, scale2: {scale2[0].item()}")
            scales[:, cur_group_idx] = scale
            # logger.info(f"block_W: {block_W.shape}, cur_group_idx: {cur_group_idx}")
            # logger.info(f"{idx1} : {idx}")
            for i in reversed(range(state_nums)):
                if i == state_nums - 1:
                    # logger.info(f"{offset + self.states*i}")
                    # consider the last idx
                    q_w, index = self.encode(block_W[:, self.states*i:], scale)
                    block_W_hat[:, self.states*i:] = q_w
                    # weight_hat[:, offset+self.states*i:] = q_w
                    block_index = [index] + block_index
                else:
                    # Prevent crossing over to the previous group
                    # max_idx = min(idx, offset + self.states*(i+1))
                    # logger.info(f"{offset + self.states*i}:{offset + self.states*(i+1)}")
                    eta = (block_W[:, self.states*(i+1):] - block_W_hat[:, self.states*(i+1):]) @ \
                            block_L[self.states*(i+1):, offset + self.states*i:offset + self.states*(i+1)] + \
                                block_prod[:, self.states*i:self.states*(i+1)]
                    
                    # eta = (weight[:, max_idx:] - weight_hat[:, max_idx:]) @ \
                    #         L[max_idx:, offset + self.states*i:max_idx]

                    w = block_W[:, self.states*i:self.states*(i+1)] + eta
                    q_w, index = self.encode(w, scale)
                    block_W_hat[:, self.states*i:self.states*(i+1)] = q_w  # fake quant
                    block_index = [index] + block_index

            # prod_cache += (block_W - block_W_hat) @ block_L
            weight_hat[:, idx1:idx] = block_W_hat #* scale.unsqueeze([-1])
            block_index = paddle.stack(block_index, axis=1)
            # logger.info(f"block_index: {block_index.shape}")
            indices = [block_index] + indices
        indices = paddle.concat(indices, axis=1)

        # tmp_scale = scales.unsqueeze(-1).expand([m, group_nums, self.group_size]).reshape([m, -1])
        # weight_hat = weight_hat * tmp_scale

        '''
        # last tune
        tune_iters = 0
        for ie in range(tune_iters):
            tmp_scale = scales.unsqueeze(-1).expand([m, group_nums, self.group_size]).reshape([m, -1])
            final_indices = []
            # delta = (weight - weight_hat)/tmp_scale  # quant error
            delta = weight - weight_hat
            for idx in range(n, 0, -self.group_size):
                idx1 = max(idx - self.group_size, 0)
                count = idx - idx1
                # block_W = weight[:, idx1:idx]
                block_W_hat = weight_hat[:, idx1:idx]
                block_delta = delta[:, idx1:idx]
                block_H = hessian_matrix[:, idx1:idx]
                offset = idx1
                block_index = []
                cur_group_idx = idx1 // self.group_size
                scale = scales[:, cur_group_idx]
                # block_W_hat /= scale.unsqueeze([-1])
                # logger.info(f"offset: {offset}, block_H: {block_H.shape}")
                for i in reversed(range(state_nums)):
                    # Prevent crossing over to the previous group
                    max_idx = min(idx, offset + self.states*(i+1))
                    # logger.info(f"{offset + self.states*i}:{max_idx}")
                    h = block_H[offset + self.states*i:max_idx, self.states*i:self.states*(i+1)]
                    inv_H = paddle.linalg.inv(h)
                    w = block_W_hat[:, self.states*i:self.states*(i+1)] + \
                        delta @ block_H[:, self.states*i:self.states*(i+1)] @ inv_H
                    
                    q_w, index = self.encode(w, scale)
                    block_delta[:, self.states*i:self.states*(i+1)] += block_W_hat[:, self.states*i:self.states*(i+1)]
                    block_delta[:, self.states*i:self.states*(i+1)] -= q_w
                    block_W_hat[:, self.states*i:self.states*(i+1)] = q_w
                    block_index = [index] + block_index
                delta[:, idx1:idx] = block_delta
                weight_hat[:, idx1:idx] = block_W_hat #* scale.unsqueeze([-1])
                block_index = paddle.stack(block_index, axis=1)
                final_indices = [block_index] + final_indices
      
            indices = paddle.concat(final_indices, axis=1)
        '''
        return weight_hat, indices, scales

    def LDLQ(self, weight, hessian_matrix):
        """
        weight: [out_features, in_features]
        hessian_matrix: [in_features, in_features]
        """
        m, n = weight.shape
        state_nums = math.ceil(self.group_size / self.states)
        group_nums = n // self.group_size

        indices = []
        # L = self.block_LDL(hessian_matrix, self.group_size) # [in_features, in_features]

        try:
            L = torch.linalg.cholesky(hessian_matrix)
        except:
            logger.info(f"The hessian is not PD, and fix it.")
            H = fix_hessian(hessian_matrix.cast('float32'))
            L = paddle.linalg.cholesky(H.cast('float32'))
        # L = paddle.flip(L,[0,1])
        L = L @ paddle.diag(1/paddle.diag(L))
        L = L - paddle.eye(H.shape[0])

        prod_cache = paddle.zeros_like(weight)
        weight_hat = weight.clone() # fake quant weight
        scales = paddle.zeros([m, group_nums])
        for idx in range(n, 0, -self.group_size):
            idx1 = max(idx - self.group_size, 0)
            count = idx - idx1
            block_W = weight[:, idx1:idx]
            block_W_hat = weight_hat[:, idx1:idx]
            block_L = L[:, idx1:idx]
            W2Hdiff = weight[:, idx:] - weight_hat[:, idx:]
            offset = idx1
            block_index = [] 

            # cal scale for cur group
            cur_group_idx = idx1 // self.group_size
            # scale = block_W.square().mean(axis=-1).sqrt() # [out_features]
            scale = block_W.max(axis=-1) - block_W.min(axis=-1)
            scale = scale / (self._qmax - self._qmin)
            # block_W /= scale.unsqueeze([-1])
            # weight[:, idx1:idx] = block_W
            # logger.debug(f"scale1: {scale[0].item()}, scale2: {scale2[0].item()}")
            scales[:, cur_group_idx] = scale
            # logger.info(f"{idx1}:{idx}, block_W: {block_W.shape}, cur_group_idx: {cur_group_idx}")
            for i in reversed(range(state_nums)):
                if i == state_nums - 1 and idx == n:
                    # logger.info(f"{offset + self.states*i}")
                    # consider the last idx
                    q_w, index = self.encode(block_W[:, self.states*i:], scale)
                    block_W_hat[:, self.states*i:] = q_w
                    block_index = [index] + block_index
                else:
                    # Prevent crossing over to the previous group
                    max_idx = min(idx, offset + self.states*(i+1))
                    # logger.info(f"{offset + self.states*i}:{max_idx}")
                    # logger.info(f"{offset + self.states*i}:{offset + self.states*(i+1)}")
                    # eta = (block_W[:, self.states*(i+1):] - block_W_hat[:, self.states*(i+1):]) @ \
                    #         block_L[self.states*(i+1):, offset + self.states*i:offset + self.states*(i+1)] + \
                    #             block_prod[:, self.states*i:self.states*(i+1)]
                    
                    # eta = (weight[:, max_idx:] - weight_hat[:, max_idx:]) @ \
                    #         L[max_idx:, offset + self.states*i:max_idx]

                    if idx == n or True:
                        eta = (block_W - block_W_hat) @ block_L[idx1:idx, self.states*i:self.states*(i+1)]
                    else:
                        eta = (block_W - block_W_hat) @ block_L[idx1:idx, self.states*i:self.states*(i+1)] + \
                                W2Hdiff @ block_L[idx:, self.states*i:self.states*(i+1)]

                    w = block_W[:, self.states*i:self.states*(i+1)]
                    w += eta
                    q_w, index = self.encode(w, scale)
                    block_W_hat[:, self.states*i:self.states*(i+1)] = q_w  # fake quant
                    block_index = [index] + block_index
            
            # prod_cache += (block_W - block_W_hat) @ block_L
            weight_hat[:, idx1:idx] = block_W_hat #* scale.unsqueeze([-1])
            block_index = paddle.stack(block_index, axis=1)
            # logger.info(f"block_index: {block_index.shape}")
            indices = [block_index] + indices
        indices = paddle.concat(indices, axis=1)
        # tmp_scale = scales.unsqueeze(-1).expand([m, group_nums, self.group_size]).reshape([m, -1])
        # weight_hat = weight_hat * tmp_scale

        # last tune
        tune_iters = 0
        for ie in range(tune_iters):
            tmp_scale = scales.unsqueeze(-1).expand([m, group_nums, self.group_size]).reshape([m, -1])
            final_indices = []
            delta = (weight - weight_hat)/tmp_scale  # quant error
            for idx in range(n, 0, -self.group_size):
                idx1 = max(idx - self.group_size, 0)
                count = idx - idx1
                # block_W = weight[:, idx1:idx]
                block_W_hat = weight_hat[:, idx1:idx]
                block_delta = delta[:, idx1:idx]
                block_H = hessian_matrix[:, idx1:idx]
                offset = idx1
                block_index = []
                cur_group_idx = idx1 // self.group_size
                scale = scales[:, cur_group_idx]
                block_W_hat /= scale.unsqueeze([-1])
                # logger.info(f"offset: {offset}, block_H: {block_H.shape}")
                for i in reversed(range(state_nums)):
                    # Prevent crossing over to the previous group
                    max_idx = min(idx, offset + self.states*(i+1))
                    # logger.info(f"{offset + self.states*i}:{max_idx}")
                    h = block_H[offset + self.states*i:max_idx, self.states*i:self.states*(i+1)]
                    inv_H = paddle.linalg.inv(h)
                    w = block_W_hat[:, self.states*i:self.states*(i+1)] + \
                        delta @ block_H[:, self.states*i:self.states*(i+1)] @ inv_H
                    
                    q_w, index = self.encode(w, scale)
                    block_delta[:, self.states*i:self.states*(i+1)] += block_W_hat[:, self.states*i:self.states*(i+1)]
                    block_delta[:, self.states*i:self.states*(i+1)] -= q_w
                    block_W_hat[:, self.states*i:self.states*(i+1)] = q_w
                    block_index = [index] + block_index
                delta[:, idx1:idx] = block_delta
                weight_hat[:, idx1:idx] = block_W_hat * scale.unsqueeze([-1])
                block_index = paddle.stack(block_index, axis=1)
                final_indices = [block_index] + final_indices
      
            indices = paddle.concat(final_indices, axis=1)

        return weight_hat, indices, scales

    def encode(self, w, scale, optimal=False, info_matrix=None):
        """
        encode a group for LDLQ or GPTQ
        w: [out_features, states] or [out_features, group_size]
        scale: [out_features]
        """
        pad_nums = 0
        if w.shape[1] % self.states_length!= 0:
            pad_nums = self.states_length - w.shape[1] % self.states_length
            padded_tensor = paddle.zeros((w.shape[0], pad_nums), dtype=w.dtype)
            w = paddle.concat([w, padded_tensor], axis=1)
        assert w.shape[-1] % self.states_length == 0, f"{w.shape}"
        if w.shape[1] == self.states_length:
            w = w.unsqueeze(1)
        else:
            w = w.reshape([w.shape[0], -1, self.states_length])

        if isinstance(self.states, (list, tuple)):
            # for multi
            split_w = paddle.split(w, self.states, axis=-1)
            total_indices = []
            total_loss = []
            q_w = []
            for i in range(len(self.states)):
                cur_w = split_w[i]
                lookup_table = self.lookup_table[i]
                table_length, states = lookup_table.shape
                assert states == self.states[i]
                code_book = lookup_table.unsqueeze(0).expand([cur_w.shape[0], table_length, states]).cast('bfloat16') + self._qmin
                code_book = code_book.cast("bfloat16") * scale.unsqueeze([-1, -1])
                cur_loss = paddle.cdist(cur_w, code_book, p=2.0)
                cur_index = paddle.argmin(cur_loss, axis=-1).cast('int32')
                # logger.debug(f"cur_index: {cur_index.shape}")

                cur_qw = lookup_table[cur_index].cast('bfloat16') + self._qmin
                q_w.append(cur_qw)
                total_indices.append(cur_index)
                total_loss.append(cur_loss)
            q_w = paddle.concat(q_w, axis=-1)
            # logger.info(f"q_w:{q_w.shape}")
            index = None
            code_length = 0
            for i in reversed(range(len(total_indices))):
                if i == len(total_indices) - 1:
                    index = total_indices[i]
                else:
                    code_length += int(math.log2(self.lookup_table[i+1].shape[0]))
                    index += (total_indices[i] << code_length)
            # logger.debug(f"index: {index.shape}")

            # # for debug
            # new_q_w = self.decode_multi(index)
            # error = new_q_w.cast('bfloat16') - q_w.cast('bfloat16')
            # logger.debug(f"error: {error.sum().item()}")

        else:
            # search code with MSE
            table_length, states = self.lookup_table.shape
            code_book = self.lookup_table.unsqueeze(0).expand([w.shape[0], table_length, states]).cast('bfloat16') + self._qmin
            code_book = code_book.cast("bfloat16") * scale.unsqueeze([-1, -1])
            loss = paddle.cdist(w.cast('bfloat16'), code_book.cast('bfloat16'), p=2.0) # [out_features, 1, table_length]
            index = paddle.argmin(loss, axis=-1).cast('int32').squeeze(-1)
            q_w = self.lookup_table[index].cast('bfloat16') + self._qmin

        if optimal:
            w = w.reshape([w.shape[0], -1])
            assert info_matrix is not None
            # info_matrix = info_matrix * (w[:, :self.group_size]**2)

            w_square = w[:, :self.group_size]**2
            # sigma2 = paddle.sum(w_square, axis=-1, keepdim=True) / self.group_size
            # info_matrix *= paddle.sqrt(sigma2 + w_square)
            info_matrix = w_square

            scale, _ = self.optimize_scale(w[:, :self.group_size], index, info_matrix)

        q_w = q_w.reshape([w.shape[0], -1])
        qdq_w = q_w * scale.unsqueeze([-1]).cast('bfloat16')

        # loss = paddle.mean(((w.squeeze() - qdq_w)**2).sum(axis=-1))
        # logger.info(f"group loss: {loss.item()}")

        if pad_nums > 0:
            qdq_w = qdq_w[:, :-pad_nums]
        return qdq_w, index, scale

    def block_LDL(self, hessian_matrix, block_size):
        """
        hessian_matrix: [in_features, in_features]
        block_size: 
        """
        in_features = hessian_matrix.shape[0]
        assert (in_features % block_size == 0)
        nums = in_features // block_size
        try:
            L = paddle.linalg.cholesky(hessian_matrix.cast('float32'))
        except:
            logger.info(f"The hessian is not PD, and fix it.")
            H = fix_hessian(hessian_matrix.cast('float32'))
            L = paddle.linalg.cholesky(H.cast('float32'))

        DL = paddle.diagonal(L.reshape([nums, block_size, nums, block_size]), axis1=0, axis2=2).transpose([2,0,1])
        DL = paddle.linalg.inv(DL)
        L = L.reshape([in_features, nums, block_size])
        for i in range(nums):
            L[:, i, :] = L[:, i, :] @ DL[i, :, :]
        if L.isnan().any():
            logger.debug("L contains NaN.")

        L = L.reshape([in_features, in_features]).cast('bfloat16')
        return L

    def GPTQ(self, weight, hessian_matrix, info_matrix):
        """
        weight: [out_features, in_features]
        hessian_matrix: [in_features, in_features]
        info_matrix: [out_features, in_features]
        """
        m, n = weight.shape
        state_nums = math.ceil(self.group_size / self.states_length)
        group_nums = n // self.group_size

        # preproces
        Losses = paddle.zeros_like(weight)
        Q = paddle.zeros_like(weight)
        '''
        try:
            H = paddle.linalg.cholesky(hessian_matrix)
        except:
            logger.info(f"The hessian is not PD, and fix it.")
            H = fix_hessian(hessian_matrix)
            H = paddle.linalg.cholesky(H)
        H = paddle.linalg.cholesky_inverse(H)
        H = paddle.linalg.cholesky(H, upper=True)
        '''
        # Torch version is faster than Paddle version
        logger.info("convert to torch version...")
        hessian_matrix = fix_hessian(hessian_matrix)
        hessian = torch.tensor(hessian_matrix.cast('float32').numpy(), dtype=torch.float32).cuda()
        hessian = torch.linalg.cholesky(hessian)
        hessian = torch.cholesky_inverse(hessian)
        hessian = torch.linalg.cholesky(hessian, upper=True)
        # hessian = np.array(hessian, dtype=np.float32)
        # hessian = paddle.to_tensor(hessian, dtype='float32')
        # hessian = paddle.linalg.cholesky(hessian, upper=True)

        H = paddle.to_tensor(hessian.cpu().numpy())

        W = weight
        Hinv = H.cast('bfloat16')
        scales = paddle.zeros([m, group_nums])
        indices = []
        logger.info(f"Start GPTQ...")
        for i1 in range(0, n, self.group_size):
            i2 = min(i1 + self.group_size, n)
            count = i2 - i1
            W1 = W[:, i1:i2]
            Q1 = paddle.zeros_like(W1)
            Err1 = paddle.zeros_like(W1)
            Losses1 = paddle.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            scale = W1.max(axis=-1) - W1.min(axis=-1)
            # scale = W1.abs().max(axis=-1)
            scale = scale / (self._qmax - self._qmin)
            cur_group_idx = i1 // self.group_size
            scales[:, cur_group_idx] = scale
            block_index = [] 
            # logger.info(f"{i1}:{i2}, {W1.shape}")

            # '''
            for i in range(state_nums):
                w = W1[:, self.states_length*i:self.states_length*(i+1)]
                d = Hinv1[self.states_length*i:self.states_length*(i+1), self.states_length*i:self.states_length*(i+1)]
                d_inv = paddle.linalg.inv(d.cast('float32')).cast('bfloat16')
                if i == state_nums - 1:
                    q, index, _ = self.encode(w, scale)
                    Q1[:, self.states_length*i:] = q
                    err1 = (w - q).matmul(d_inv)
                    Err1[:, self.states_length*i:self.states_length*(i+1)] = err1
                else:
                    q, index, _ = self.encode(w, scale)
                    Q1[:, self.states_length*i:self.states_length*(i+1)] = q
                    # Losses1[:, self.states_length*i:self.states_length*(i+1)] = (w - q)**2 / d**2
                    err1 = (w - q).matmul(d_inv)
                    W1[:, self.states_length*(i+1):] -= err1.matmul(Hinv1[self.states_length*i:self.states_length*(i+1), self.states_length*(i+1):])
                    Err1[:, self.states_length*i:self.states_length*(i+1)] = err1
                block_index.append(index)
            # '''
            ### for block GPTQ + opimize scale
            # info_m = info_matrix[:, i1:i2]
            # q, index, scale = self.encode(W1, scale, optimal=False, info_matrix=info_m)
            # scales[:, cur_group_idx] = scale
            # block_index = [index]
            # Q1 = q

            d_inv = paddle.linalg.inv(Hinv1.cast('float32')).cast('bfloat16')
            Err1 = (W1 - Q1).matmul(d_inv)

            Q[:, i1:i2] = Q1
            W[:, i1:i2] = W1
            # Losses[:, i1:i2] = Losses1 / 2
            if Hinv[i1:i2, i2:].shape[1] > 0:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            block_index = paddle.stack(block_index, axis=1)   
            indices.append(block_index)

        indices = paddle.concat(indices, axis=1)
        return Q, indices, scales

    def optimize_scale(self, weight, indices, info_matrix):
        """ 
        weight: [group numbers, group size]
        indices: [group numbers, group size]
        info_matrix: [group numbers, group size]
        """
        if info_matrix is not None:
            info_matrix = info_matrix.cast("float32")
        weight = weight.cast("float32")

        # lookup table is a list when states is a list
        if isinstance(self.states, (list, tuple)):
            quant_weight = self.decode_multi(indices)
        else:
            if self.parity_sign:
                quant_weight = cal_sign(self.lookup_table[indices]).cast("float32")
            elif self.enable_completion:
                quant_weight = decode_twos_complement(self.lookup_table[indices], self.lut_bits).cast("float32")
            else:
                quant_weight = self.lookup_table[indices].cast("float32") + self._qmin
        logger.info(f"quant_weight: {quant_weight.shape}")
        quant_weight = quant_weight.reshape([quant_weight.shape[0], -1]) # [group numbers, group size]
        if weight.shape[1] == self.group_size:
            quant_weight = quant_weight[:, :self.group_size]
        assert quant_weight.shape == weight.shape, f"quant_weight:{quant_weight.shape}, weight: {weight.shape}"
        if info_matrix is None:
            info_matrix = 1.0
        else:
            assert weight.shape == info_matrix.shape
        if self._symmetric:
            sum_wq = paddle.sum(info_matrix * weight * quant_weight, axis=-1)
            sum_q2 = paddle.sum(info_matrix * quant_weight**2, axis=-1)
            new_scales = sum_wq / sum_q2
            return new_scales, None
        else:
            sum_wq = paddle.sum(info_matrix * weight * quant_weight, axis=-1) 
            info_m_qw = info_matrix * quant_weight
            sum_q2 = paddle.sum(info_m_qw * quant_weight, axis=-1)
            q_sum = paddle.sum(info_m_qw, axis=-1)
            N = info_matrix.sum(axis=-1)
            w_sum = paddle.sum(info_matrix * weight, axis=-1) # [group_nums]
            # D = N * sum_q2 - q_sum**2
            # new_scales = (N * sum_wq - w_sum * q_sum) / D
            # new_zp = (w_sum * sum_q2 - sum_wq * q_sum) / D
            # return new_scales, new_zp
            
            new_scales = (sum_wq - self.zero_points*q_sum)/sum_q2
            new_zp = (w_sum - q_sum * new_scales) / N
            return new_scales, new_zp

    def optimize_super_scale(self, weight, indices, scales, info_matrix, dequant_scales=None, super_scales=None):
        """
        weight: [all group nums, group size]
        indices: [all group nums, state nums]
        scales: [all group nums]
        quant scales per-channel.
        """
        weight = weight.reshape([self.out_features, -1, weight.shape[-1]]).cast('float32')
        info_matrix = info_matrix.reshape([self.out_features, -1, info_matrix.shape[-1]]).cast('float32')
        assert weight.shape == info_matrix.shape, f"{weight.shape} != {info_matrix.shape}"
        indices = indices.reshape([self.out_features, -1, indices.shape[-1]])
        logger.info(f"quant scales to {self.group_scale_bits} bits...")
        if dequant_scales is None:
            # quant each chanel's scale
            scales = scales.reshape([self.out_features, -1]).cast('float32')
            # To prevent a small number of negative scales from causing quantization to be 0, abs() is not used
            super_scales = scales.max(axis=-1, keepdim=True) / self._s_qmax # [out_features, 1]
            quant_scales = paddle.clip(paddle.round(scales / super_scales), self._s_qmin, self._s_qmax)
            dequant_scales = quant_scales * super_scales # [out_features, group nums]
            super_scales = super_scales.squeeze(axis=-1)
        else:
            assert super_scales is not None
        if paddle.isnan(super_scales).any():
            logger.debug("First get super sclaes has nan")
        if dequant_scales.isnan().any():
            logger.debug("dequant scales has nan")

        # dequant weight
        if isinstance(self.states, (list, tuple)):
            quant_weight = self.decode_multi(indices)
        else:
            quant_weight = self.lookup_table[indices].cast('float32') + self._qmin
        quant_weight = quant_weight.reshape(quant_weight.shape[0:2] + [-1])
        logger.debug(f"quant_weight: {quant_weight.shape}")
        dequant_weight = quant_weight * dequant_scales.unsqueeze([-1]).cast('float32')
        if weight.shape[-1] <= self.group_size:
            dequant_weight = dequant_weight[:, :, :self.group_size]
        assert weight.shape == dequant_weight.shape, f"{weight.shape} != {dequant_weight.shape}"
        if dequant_weight.isnan().any():
            logger.debug("dequant weight has nan")

        # optimize super scales
        sum_wq = paddle.sum(info_matrix * weight * dequant_weight, axis=[1,2])
        sum_q2 = paddle.sum(info_matrix * dequant_weight**2, axis=[1,2])
        new_super_scales = super_scales * (sum_wq / sum_q2) # [out_features]
        if new_super_scales.isnan().any():
            logger.debug("Final super scales has nan...")
            nan_indices = paddle.where(paddle.isnan(new_super_scales))[0]
            logger.debug(f"nan_indices: {nan_indices.tolist()}")
            max_w = paddle.max(sum_q2[nan_indices], axis=0)
            logger.debug(f"max sum_q2: {max_w.tolist()}")
            new_super_scales[nan_indices] = super_scales[nan_indices]
            # hcg = fleet.get_hybrid_communicate_group()
            # rank = hcg.get_model_parallel_rank()
            # paddle.save(scales, f'scalse_test_tp{rank}.pd')
            # paddle.save(quant_scales, f"quant_scales_test_tp{rank}.pd")
            # paddle.save(dequant_scales, f"dequant_scales_test_tp{rank}.pd")
            # paddle.save(super_scales, f"super_scales_test_tp{rank}.pd")
            logger.debug(f"fix it...")
        super_scales = new_super_scales
        if scales is None:
            return super_scales, None
        else:
            quant_scales = quant_scales.reshape([-1]).cast('int32')
            return super_scales, quant_scales
        
    def decode_multi(self, indices):
        # if self.enable_circle:
        #     shifts = []
        #     for i in range(len(self.states)):
        #         cur_shift = paddle.arange(0, self.states[i], dtype='int32') * self.state_bits[i]
        #         cur_shift = cur_shift % self.lut_bits
        #         if i == 0:
        #             cur_shift += self.lut_bits
        #         shifts.append(cur_shift)
        #     shifts = paddle.concat(shifts)
        #     mask = paddle.to_tensor(2**self.lut_bits - 1, dtype='int32')
        #     indices = indices.unsqueeze(-1).cast('int32')
        #     quant_weight = (indices >> shifts)

        code_length = []
        for i in range(len(self.states)):
            code_length.append(self.lut_bits + self.state_bits[i] * (self.states[i] - 1))
        
        shifts = []
        for i in range(len(self.states)):
            if i == len(self.states) - 1:
                cur = paddle.arange(self.states[i]-1, -1, -1) * self.state_bits[i]
            else:
                cur = paddle.arange(self.states[i]-1, -1, -1) * self.state_bits[i] + code_length[i+1]
            shifts.append(cur)
        shifts = paddle.concat(shifts, axis=0).cast('int32')
        mask = paddle.to_tensor(2**self.lut_bits - 1, dtype='int32')
        quant_weight = (indices.unsqueeze(-1).cast('int32') >> shifts) & mask
        # quant_weight = (indices.unsqueeze(-1).cast('int32') // (2**shifts)) & mask
        if self.parity_sign:
            quant_weight = cal_sign(quant_weight).cast("bfloat16")
        elif self.enable_completion:
            quant_weight = decode_twos_complement(quant_weight, self.lut_bits).cast("bfloat16")
        else:
            quant_weight = quant_weight.cast("bfloat16") + self._qmin
        return quant_weight

    def normalize(self, weight, axis=0):
        """
        weight: [out_features, in_features]
        """
        weight_min = weight.min(axis=axis)  
        weight_max = weight.max(axis=axis)  
        self.norm_scale = weight_max - weight_min
        self.norm_bias = weight_min
        weight = (weight - self.norm_bias.unsqueeze(axis)) / self.norm_scale.unsqueeze(axis)
        return weight

    def permute(self, weight):
        """
        weight: [out_features, in_features]
        """
        value = weight.max(axis=0) - weight.min(axis=0)
        # value = weight.abs().max(axis=0)
        self.perm = paddle.argsort(value, descending=True)
        weight = weight[:, self.perm]
        return weight

    def quant(self, x, offset=0., s=1.0):
        """
        x: [group size, group nums]
        """
        if self._symmetric:
            if self.parity_sign:
                scale = (x.max(axis=0) - x.min(axis=0)) / (2*self._qmax)
            else:
                scale = (x.max(axis=0) - x.min(axis=0)) / (self._qmax - self._qmin) 
            # scale = x.abs().max(axis=0) / (self._qmin + offset)
            scale = paddle.where(scale == paddle.to_tensor( 0, dtype=x.dtype), \
                paddle.to_tensor(1e-8, dtype=x.dtype), scale)
            scale *= s
            quant_x = paddle.clip(paddle.round(x / scale), self._qmin, self._qmax)
            zero_point = 0
        else:
            scale = (x.max(axis=0) - x.min(axis=0)) / (self._qmax - self._qmin)
            zero_point = x.min(axis=0)
            quant_x = paddle.clip(paddle.round((x - zero_point) / scale),
                                  self._qmin, self._qmax)
        return quant_x, scale, zero_point
    
    def dequant(self, x, scale, zero_point=None):
        if self._symmetric:
            dequant_x = x * scale
        else:
            assert zero_point is not None
            dequant_x = x * scale + zero_point
        return dequant_x

    def coordinate_descent(self, weight, hessian_matrix, iters=10):
        """
        weight: [out_features, in_features]
        hessian_matrix: [in_features, in_fetures]
        """
        logger.info(f"Begin coordinate descent...")
        columns = weight.shape[1]
        state_nums = math.ceil(self.group_size / self.states_length)
        group_nums = columns // self.group_size
        scales = paddle.zeros([weight.shape[0], group_nums])
        Q = weight.clone()

        indices = []
        # XtX
        diag_sigma = paddle.diag(hessian_matrix)
        diag_sigma += 0.75*paddle.diag(hessian_matrix).mean()
        H = hessian_matrix / (diag_sigma+0.1) # [in_features, in_features]
        P = weight @ H # [out_features, in_features]
        H.fill_diagonal_(0)
        H = H.t()
        for _ in range(iters):
            delta_Q = Q.clone().t() # [in_features, out_features]
            P_hat = (H @ delta_Q).t() # [out_features, in_features]
            indices = []
            for i in range(0, columns, self.group_size):
                block_index = []
                block_w = Q[:, i:i+self.group_size]
                
                scale = block_w.max(axis=-1) - block_w.min(axis=-1)
                scale = scale / (self._qmax - self._qmin)
                cur_group_idx = i // self.group_size
                scales[:, cur_group_idx] = scale
                for j in range(state_nums):
                    idx1 = i + j * self.states_length
                    idx2 = idx1 + self.states_length
                    if j == state_nums - 1:
                        idx2 = min(idx2, i + self.group_size)

                    u = P[:, idx1:idx2] - P_hat[:, idx1:idx2]
                    u += (H[idx1:idx2, :idx1] @ delta_Q[:idx1, :]).t()

                    q_w, index, _ = self.encode(u, scale)
                    Q[:, idx1:idx2] = q_w
                    delta_Q[idx1:idx2, :] -= u.t()
                    block_index.append(index)
                block_index = paddle.stack(block_index, axis=1)
                indices.append(block_index)
            indices = paddle.concat(indices, axis=-1)
        return Q, indices, scales

    def coordinate_descent2(self, weight, hessian_matrix, iters=10):
        """
        weight: [out_features, in_features]
        hessian_matrix: [in_features, in_fetures]
        """
        logger.info(f"Begin coordinate descent...")
        columns = weight.shape[1]
        state_nums = math.ceil(self.group_size / self.states_length)
        group_nums = columns // self.group_size
        scales = paddle.zeros([weight.shape[0], group_nums])
        Q = weight.clone()

        indices = []
        # XtX
        diag_sigma = paddle.diag(hessian_matrix)
        diag_sigma += 0.75*paddle.diag(hessian_matrix).mean()
        H = hessian_matrix / (diag_sigma+0.1) # [in_features, in_features]
        P = weight @ H # [out_features, in_features]
        H.fill_diagonal_(0)
        H = H.t()
        for _ in range(iters):
            delta_Q = Q.clone().t() # [in_features, out_features]
            P_hat = (H @ delta_Q).t() # [out_features, in_features]
            indices = []
            for i in range(0, columns, self.group_size):
                block_index = []
                block_w = Q[:, i:i+self.group_size]
                
                scale = block_w.max(axis=-1) - block_w.min(axis=-1)
                scale = scale / (self._qmax - self._qmin)
                cur_group_idx = i // self.group_size
                scales[:, cur_group_idx] = scale
                for j in range(state_nums):
                    idx1 = i + j * self.states_length
                    idx2 = idx1 + self.states_length
                    if j == state_nums - 1:
                        idx2 = min(idx2, i + self.group_size)

                    u = P[:, idx1:idx2] - P_hat[:, idx1:idx2]
                    u += (H[idx1:idx2, :idx1] @ delta_Q[:idx1, :]).t()

                    q_w, index, _ = self.encode(u, scale)
                    Q[:, idx1:idx2] = q_w
                    delta_Q[idx1:idx2, :] -= u.t()
                    block_index.append(index)
                block_index = paddle.stack(block_index, axis=1)
                indices.append(block_index)
            indices = paddle.concat(indices, axis=-1)
        return Q, indices, scales


def decode_twos_complement(x, n_bits):
    sign_bit = paddle.to_tensor(1 << (n_bits - 1), dtype='int32')
    # mask = paddle.to_tensor((1 << n_bits) - 1, dtype='int32')
    # 用 bit 运算恢复负数（如果符号位是 1，就减去 2^n）
    # return (x & mask) - ((x & sign_bit) << 1)
    return x - ((x & sign_bit) << 1)

def MSB_pos(x):
    # zero_mask = x == 0
    # x[zero_mask] = 1
    bitlen = paddle.floor(paddle.log2(x.cast('float32'))).cast('int32') + 1
    return bitlen

def cal_sign(x):
    """
    Calculate signs based on parity, 
    where even numbers are positive and odd numbers are negative.
    """
    mask = paddle.to_tensor(1, dtype='int32')
    sign = 1 - 2 * (x & mask)
    return x * sign

def construct_lookup_table(lut_bits, states, state_bits):
    code_length = lut_bits + state_bits * (states - 1)
    total_length = 2**code_length
    shifts = (paddle.arange(states-1, -1, -1) * state_bits).cast('int32')
    # mask = paddle.to_tensor(2**lut_bits - 1, dtype='int32') << shifts
    mask = paddle.to_tensor(2**lut_bits - 1, dtype='int32') * (2**shifts)

    lookup_table = paddle.arange(total_length, dtype="int32").reshape([total_length, 1])
    lookup_table = (lookup_table & mask) >> shifts
    # lookup_table = (lookup_table & mask) // (2**shifts)
    return lookup_table

def construct_lookup_table_multi2(lut_bits=3, states=[3,4], state_bits=[2, 2]):
    lookup_tables = []
    for i in range(len(states)):
        table = construct_lookup_table(lut_bits, states[i], state_bits[i])
        lookup_tables.append(table)
    return lookup_tables

def construct_lookup_table_circle(lut_bits, states, state_bits):
    total_length = 2**lut_bits
    lookup_table = paddle.arange(total_length, dtype='int32').reshape([total_length, 1])
    
    mask = paddle.to_tensor(2**lut_bits - 1, dtype='int32')
    shifts = (paddle.arange(0, states, 1) * state_bits).cast('int32') % lut_bits

    lookup_table = (lookup_table >> shifts) | ((lookup_table << (lut_bits - shifts)) & mask)

    return lookup_table

def fix_hessian(H, eps=1e-2):
    eig, _ = paddle.linalg.eigh(H)
    min_eig = eig.min().abs()
    delta = min_eig + eps # ignore the zero eigenvalue
    # logger.debug(f"delta: {delta.item()}")
    idx = paddle.arange(H.shape[0])
    H[idx, idx] += delta # Increase on the diagonal
    return H

class Pad:
    def __init__(self, states, group_size):
        self.states = states
        self.group_size = group_size
        self.is_padded = False
        self.pad_cols = 0

    def add_padding(self, data):
        '''
        Check if data need padding columns
        data: [group_nums, group_size]
        Returns padded data
        '''
        remainder = data.shape[1] % self.states
        if remainder != 0:
            padded_tensor = paddle.zeros((data.shape[0], self.states - remainder),
                                        dtype=data.dtype)
            self.is_padded = True
            self.pad_cols = self.states - remainder
            return paddle.concat((data, padded_tensor), axis=1)
        return data

    def remove_padding(self, data):
        '''
        input data: [group_nums, group_size]
        Remove padding
        '''
        if self.is_padded:
            data = data[:, :-self.pad_cols]
        return data

def project_onto_l1_ball_groupwise(x, eps=1.0):
    """
    Compute Euclidean projection onto the L1 ball for a batch.

    Parameters:
    x: (batch_size, num_groups, group_size) torch array
      batch of grouped tensors to project, possibly on GPU

    eps: float
      radius of the L-1 ball to project onto

    Returns:
    u: (batch_size, num_groups, group_size) 
      batch of projected tensors, reshaped to match the original
    """
    # Flattening within each group but keeping batch and group separations
    batch_size, num_groups, group_size = x.shape
    x = x.reshape([batch_size * num_groups, group_size])
    
    mask = (paddle.linalg.norm(x, p=1, axis=1) < eps).cast('float32').unsqueeze(1)
    mu = paddle.sort(paddle.abs(x), axis=1, descending=True)
    cumsum = paddle.cumsum(mu, axis=1)
    arange = paddle.arange(1, group_size + 1, dtype='float32')
    rho = paddle.max((mu * arange > (cumsum - eps)).cast('float32') * arange, axis=1).cast('int32')
    theta = (cumsum[paddle.arange(batch_size * num_groups, dtype='int32'), rho - 1] - eps) / rho.cast('float32')
    proj = (paddle.abs(x) - theta.unsqueeze(1)).clip(min=0)
    x = mask * x + (1 - mask) * proj * paddle.sign(x)
    
    # Reshape back to the original grouped shape
    return x.reshape([batch_size, num_groups, group_size])

def linfty_proximal_groupwise(x, scale, group_size=128):
    """
    x: [out_Features, in_features]
    """
    assert scale != 0

    # Reshape x to have groups of `group_size`
    num_features = x.shape[1]
    
    if num_features % group_size != 0:
        raise ValueError("The number of features must be divisible by the group size.")
    
    num_groups = num_features // group_size
    
    x = x.reshape([-1, num_groups, group_size])

    # Apply the projection for each group
    proximal_result = x - scale * project_onto_l1_ball_groupwise(x / scale)
    
    # Reshape back to the original shape
    return proximal_result.reshape([-1, num_features])

def W_proximal_preprocess_groupwise(W, X, alpha=0.0001, n_iters=200, group_size=64):
    W_hat = W.clone()
    m, n = X.shape

    U, s, Vt = paddle.linalg.svd(X, full_matrices=False)
    del X
    s /= paddle.max(s)
    S = paddle.diag(s)

    X = U @ S @ Vt
    XtX = paddle.matmul(X.t(), X) # [in_features, in_features]

    for idx in range(n_iters):
        if idx % 10 == 0 or idx == n_iters - 1:
            logger.info(f"[MagR] n_iters: {idx}")
        W_hat = linfty_proximal_groupwise(
            (W_hat - paddle.matmul(XtX, W_hat - W)).t(), scale=alpha, group_size=group_size).t()

    del XtX
    return W_hat