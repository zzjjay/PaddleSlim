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
import concurrent.futures

class IQuantizer:
    def __init__(
        self,
        quant_bits=2,
        group_size=32,
        super_group_size=256,
        group_scale_bits=4,
        group_zp_bits=4,
        quant_scale=True,
        symmetric=True,
        isolate_outliers=True,
        hadamard=False,
        extract_sign=False,
    ):
        self.quant_bits = quant_bits
        self.group_size = group_size
        self.super_group_size = super_group_size
        # assert symmetric, "only support symmetric quantization now"
        self.extract_sign = extract_sign
        self._symmetric = symmetric
        if self._symmetric and not self.extract_sign:
            self._qmin = -(2 ** (self.quant_bits - 1))
            self._qmax = 2 ** (self.quant_bits - 1) - 1
        else:
            self._qmin = 0
            self._qmax = 2 ** self.quant_bits - 1
        self.scales = None
        self.zero_points = None

        self.quant_scale = quant_scale
        self.group_scale_bits = group_scale_bits

        # All scales have the same symbol
        self._s_qmin = 0
        self._s_qmax = 2 ** self.group_scale_bits - 1
        self.group_zp_bits = group_zp_bits
        self._zp_qmin = -(2 ** (self.group_zp_bits - 1))
        self._zp_qmax = 2 ** (self.group_zp_bits - 1) - 1

        self._isolate = isolate_outliers
        self._thread_num = os.cpu_count() 

        # if hadamard:
        #     self._hadamard_matrix = paddle.load('hadamard_matrix_32.pdtensors')
        # else:
        #     self._hadamard_matrix = None

    def group_quantize(self, weight, info_matrix=None):
        """
        weight: [out_features, in_features]
        info_matrix: [out_features, in_features]
        """
        out_features, in_features = weight.shape
        weight = weight.reshape([out_features, -1, self.group_size]) # [out_features, group numbers, group size]
        weight = weight.reshape([-1, self.group_size]) # [group numbers, group size]
        weight_square = weight**2 # It can be calculated in advance and transmitted from outside
        logger.info(f"group-wise weight: {weight.shape}")

        step = self.super_group_size // self.group_size
        # cal info matrix for each super group weight
        if info_matrix is not None:
            # '''
            weight = weight.reshape([-1, step, self.group_size]) # [super group numbers, step, group size]
            info_matrix = info_matrix.reshape([-1, step, self.group_size])
            weight_square = weight_square.reshape([-1, step, self.group_size])
            assert info_matrix.shape == weight.shape
            # cal each super group's sigma2
            sigma2 = paddle.sum(weight_square, axis=[1,2], keepdim=True) / self.super_group_size # [super group numbers, 1, 1]
            # '''
            # sigma2 = weight_square.sum(axis=-1, keepdim=True) / (weight.shape[0] * weight.shape[1])
            info_matrix *= paddle.sqrt(sigma2 + weight_square)

            weight = weight.reshape([-1, self.group_size])
            info_matrix = info_matrix.reshape([-1, self.group_size])
        else:
            info_matrix = weight_square
            logger.info(f"info matrix set to be ones...")
            info_matrix = paddle.ones_like(weight)

        self.scales = paddle.zeros(weight.shape[0])
        self.quant_weight = paddle.zeros(weight.shape)

        # Firstly, optimize the scale of each super group in parallel
        # For efficiency, we optimize the scales in each super group simultaneously
        self.quant_weight, self.scales, self.zero_points = self.search_scale_for_block(weight, info_matrix)
        logger.info(f"Group scale optimization finished.")
        self.super_scales = None
        self.super_zero_points = None
        if self.quant_scale:
            # Secondly, quant group scale and optimize each super group scale
            logger.info(f" Next, optimizing super group scale")
            self.super_scales, self.scales, self.super_zero_points, self.zero_points = self.optimize_super_scale_parallel(weight, self.quant_weight, self.scales, self.zero_points, info_matrix)
            assert self.super_scales.shape[0] == weight.shape[0] // step
            if not self._symmetric:
                assert self.super_scales.shape[0] == self.super_zero_points.shape[0]
                assert self.scales.shape[0] == self.zero_points.shape[0]

            logger.info(f'super scales: {self.super_scales.shape}')
        
        # for debug
        if paddle.isinf(self.scales).any():
            logger.debug(f"scales has inf!")
            inf_indices = paddle.where(paddle.isinf(self.scales))[0]
            logger.debug(f"inf_indices: {inf_indices.tolist()}")
            max_w = paddle.max(weight[:, inf_indices], axis=0)
            logger.debug(f"max_w: {max_w.tolist()}")
            w_inf = paddle.isinf(weight[:, inf_indices]).any()
            info_inf = paddle.isinf(info_matrix[:, inf_indices]).any()
            logger.debug(f"w_inf: {w_inf.item()}, info_inf: {info_inf.item()}")
            self.scales = paddle.where(paddle.isinf(self.scales), paddle.zeros_like(self.scales), self.scales)
        if paddle.isnan(self.scales).any():
            logger.debug(f"scales has nan!")
            nan_indices = paddle.where(paddle.isnan(self.scales))[0]
            logger.debug(f"nan_indices: {nan_indices.tolist()}")
            max_w = paddle.max(weight[:, nan_indices], axis=0)
            logger.debug(f"max_w: {max_w.tolist()}")
            w_nan = paddle.isnan(weight[:, nan_indices]).any()
            info_nan = paddle.isnan(info_matrix[:, nan_indices]).any()
            logger.debug(f"w_nan: {w_nan.item()}, info_nan: {info_nan.item()}")
            self.scales = paddle.where(paddle.isnan(self.scales), paddle.zeros_like(self.scales), self.scales)
        
        # cal quant loss
        qdq_weight = self.dequant(self.quant_weight, self.scales.unsqueeze(-1)) # [group size, group nums]
        logger.info(f"qdq_weight: {qdq_weight.shape}")
        mse_loss = paddle.mean(((weight - qdq_weight)**2).sum(axis=-1))
        info_mse_loss = paddle.mean((info_matrix*(weight - qdq_weight)**2).sum(axis=-1))
        logger.info(f"[IQ] MSE loss: {mse_loss.item()}, Info MSE Loss: {info_mse_loss.item()}")


    def search_scale_for_block(self, weight, info_matrix):
        """ 
        weight: [group numbers, group size]
        info_matrix: [group numbers, group size]
        return: 
            quant_weight [group_nums, group_size]
        """
        # [group size, group nums]
        weight = weight.t() 
        info_matrix = info_matrix.t() 

        # for sum_wq
        new_info_m = info_matrix * weight

        best = paddle.zeros(weight.shape[1]) # [group nums]
        scales = paddle.zeros(weight.shape[1]) # [group nums]
        if self._symmetric:
            zero_points = None
        else:
            zero_points = paddle.zeros(weight.shape[1]) # [group nums]
        quant_weight = paddle.zeros(weight.shape) # [group size, group nums]

        min_s = 0.5
        max_s = 1.5
        step = (max_s - min_s) / 100.
        ranges = paddle.arange(min_s, max_s, step, dtype='bfloat16')
        ranges = [1.0]
        # Quantile filtering for values with low importance
        # percentile = paddle.quantile(info_matrix.abs(), q=0.01, axis=0) # [group nums]
        # w_mask = paddle.where(info_matrix.abs() > percentile, 1., 0.).cast("bfloat16") # [group size, group nums]
        # info_matrix *= w_mask
        # new_info_m *= w_mask

        if self._symmetric:
            # only search scale
            for idx, off in enumerate(ranges): 
            # for off in range(-9, 10):
                # logger.info(f"idx: {idx} --- {paddle.device.cuda.memory_allocated()}")
                # q_w, _, _ = self.quant(weight, offset=off*0.1) # [group size, group nums]
                q_w, _, _ = self.quant(weight, 0, s=off)
                sum_wq = paddle.sum(new_info_m * q_w, axis=0) # [group nums] 
                sum_q2 = paddle.sum(info_matrix * q_w**2, axis=0) # [group nums]

                new_scales = sum_wq / sum_q2
                mask = (sum_wq**2) > (best * sum_q2)
                new_thres = new_scales * sum_wq 

                ## Boolean index
                scales[mask] = new_scales[mask]
                best[mask] = new_thres[mask]
                quant_weight[:, mask] = q_w[:, mask]

                # scales = paddle.where((sum_wq**2) > (best * sum_q2), sum_wq / sum_q2, scales)
                # quant_weight = paddle.where((sum_wq**2) > (best * sum_q2), q_w, quant_weight)
                # best = paddle.where((sum_wq**2) > (best * sum_q2), sum_wq**2 / sum_q2, best)

        else:
            best = paddle.full(best.shape, float('inf'), dtype='bfloat16')
            # search scale and update zero point
            N = info_matrix.sum(axis=0, keepdim=False) # [1, group nums]
            w_sum = new_info_m.sum(axis=0, keepdim=False) # [1, group nums]
            for off in ranges:
                q_w, _, _ = self.quant(weight, 0, s=off)
                sum_wq = paddle.sum(new_info_m * q_w, axis=0, keepdim=False) # [1, group nums] 
                info_m_qw = info_matrix * q_w # [group size, group nums]
                q_sum = paddle.sum(info_m_qw, axis=0, keepdim=False) # [1, group nums]
                sum_q2 = paddle.sum(info_m_qw*q_w, axis=0, keepdim=False) # [1, group nums]
                D = N * sum_q2 - q_sum**2
                new_scale = (N * sum_wq - w_sum * q_sum) / D     # [1, group nums]
                new_zp = (w_sum * sum_q2 - sum_wq * q_sum) / D   # [1, group nums]

                new_thres = paddle.sum(info_matrix * (self.dequant(q_w, new_scale, new_zp)-weight)**2, axis=0)
                mask = new_thres < best
                # Boolean index
                scales[mask] = new_scale[mask]
                zero_points[mask] = new_zp[mask]
                best[mask] = new_thres[mask]
                quant_weight[:, mask] = q_w[:, mask]

        return quant_weight.t(), scales, zero_points

    def optimize_super_scale(self, weight, quant_weight, scales, info_matrix):
        """
        weight: original super group weight (float) [group nums, group size]
        quant_weight: quantized super group weight [group nums, group size]
        scales: each group scale in the super group [group nums]
        info_matrix: importance matrix of each group in the super group [group nums, group size]
        return:
            super_scale: the optimal super group scale 
            quant_scales: the quant group scales [group nums]
        """
        
        if info_matrix is not None:
            assert info_matrix.shape == weight.shape
            sigma2 = paddle.sum(weight**2)/weight.shape[0]
            info_matrix *= paddle.sqrt(sigma2 + weight**2)
        else:
            info_matrix = weight**2
        super_scale = scales.abs().max() / self._s_qmin
        quant_scales = paddle.clip(paddle.round(scales / super_scale), self._s_qmin, self._s_qmax)
        dequant_scales = quant_scales * super_scale # [group nums]
        # group_nums wise dequant
        qdq_weight = self.dequant(quant_weight.t(), dequant_scales).t() # [group nums, group size]
        assert weight.shape == qdq_weight.shape

        group_nums = self.super_group_size // self.group_size
        sum_wq = paddle.sum(info_matrix * weight * qdq_weight)
        sum_q2 = paddle.sum(info_matrix * qdq_weight**2)
        super_scale *= (sum_wq / sum_q2)

        return super_scale, quant_scales
    
    def optimize_super_scale_parallel(self, weight, quant_weight, scales, zero_points, info_matrix=None):
        # [group numbers, group size] -> [super group nums, step, group size]
        assert weight.shape == info_matrix.shape
        assert weight.shape == quant_weight.shape
        step = self.super_group_size // self.group_size
        weight = weight.reshape([-1, step, self.group_size])
        quant_weight = quant_weight.reshape([-1, step, self.group_size])
        info_matrix = info_matrix.reshape([-1, step, self.group_size])

        # quant each group scales
        # [group numbers] -> [super group nums, step]
        scales = scales.reshape([-1, step])
        super_scales = scales.abs().max(axis=-1, keepdim=True) / self._s_qmax # [super group nums, 1]
        quant_scales = paddle.clip(paddle.round(scales / super_scales), self._s_qmin, self._s_qmax) # [super group nums, step]
        dequant_scales = quant_scales * super_scales # [super group nums, step]

        if self._symmetric:
            # dequant each group weight
            dequant_weight = self.dequant(quant_weight.reshape([-1, self.group_size]).t(), dequant_scales.reshape([-1])) 
            dequant_weight = dequant_weight.t().reshape([-1, step, self.group_size])  # [super group nums, step, group size]
            assert weight.shape == dequant_weight.shape

            # optimize super scales
            sum_wq = paddle.sum(info_matrix * weight * dequant_weight, axis=[1,2]) # [super group nums]
            sum_q2 = paddle.sum(info_matrix * dequant_weight**2, axis=[1,2]) # [super group nums]
            super_scales = super_scales.squeeze(axis=-1) * (sum_wq / sum_q2) # [super group nums]

            return super_scales, quant_scales.reshape([-1]), None, None
        else:
            # quant each group zero points
            # [group numbers] -> [super group nums, step]
            zero_points = zero_points.reshape([-1, step])
            super_zero_points = zero_points.abs().max(axis=-1, keepdim=True) / self._zp_qmin # [super group nums, 1]
            quant_zero_points = paddle.clip(paddle.round(zero_points / super_zero_points), self._zp_qmin, self._zp_qmax)  # [super group nums, step]
            dequant_zero_points = quant_zero_points * super_zero_points # [group numbers, step]
            # return super_scales.reshape([-1]), quant_scales.reshape([-1]), super_zero_points.reshape([-1]), quant_zero_points.reshape([-1])

            # dequant each group weight
            dequant_weight = self.dequant(quant_weight.reshape([-1, self.group_size]).t(), dequant_scales.reshape([-1]), dequant_zero_points.reshape([-1])) 
            dequant_weight = dequant_weight.t().reshape([-1, step, self.group_size])  # [super group nums, step, group size]
            assert weight.shape == dequant_weight.shape

            # optimize super scales
            sum_sqw_sqz = paddle.sum(info_matrix*dequant_scales.unsqueeze(-1)*quant_weight*(weight - dequant_zero_points.unsqueeze(-1)), axis=[1,2]) # [super group nums]
            sum_s2q2 = paddle.sum(info_matrix*dequant_scales.unsqueeze(-1)**2*quant_weight**2, axis=[1,2]) # [super group nums]
            super_scales = super_scales.squeeze(axis=-1) * (sum_sqw_sqz / sum_s2q2) # [super group nums]

            # return super_scales.reshape([-1]), quant_scales.reshape([-1]), super_zero_points.reshape([-1]), quant_zero_points.reshape([-1])

            dequant_scales = super_scales.unsqueeze(-1) * quant_scales
            # optimize super zero points
            sum_zw_sqz = paddle.sum(info_matrix * dequant_zero_points.unsqueeze(-1)*(weight - dequant_scales.unsqueeze(-1)*quant_weight))
            sum_z2 = paddle.sum(info_matrix * dequant_zero_points.unsqueeze(-1)**2)
            super_zero_points = super_zero_points.squeeze(axis=-1) * (sum_zw_sqz / sum_z2) # [super group nums]

            return super_scales, quant_scales.reshape([-1]), super_zero_points, quant_zero_points.reshape([-1])

    def isolate_outliers(self, weight, info_matrix, percent=0.005):
        """
        weight: [group nums, group size]
        info_matrix: [group nums, group size]
        """
        percentile = paddle.quantile(info_matrix.abs(), q=1-percent)
        outliers_mask = paddle.where(info_matrix.abs() >= percentile, 1., 0.).cast("bfloat16") 
        remain_mask = (1 - outliers_mask).cast("bfloat16") 
        outliers_weight = weight * outliers_mask
        # info_matrix *= remain_mask
        weight *= remain_mask
        return weight, info_matrix, outliers_weight

    def quant(self, x, offset=0., s=1.0):
        if self._symmetric:
            # scale = x.abs().max(axis=0) / (self._qmin + offset) 
            scale = (x.max(axis=0) - x.min(axis=0)) / (self._qmax - self._qmin)
            scale = paddle.where(scale == paddle.to_tensor( 0, dtype=x.dtype), \
                paddle.to_tensor(1e-5, dtype=x.dtype), scale)
            scale *= s
            quant_x = paddle.clip(paddle.round(x / scale), self._qmin, self._qmax)
            zero_point = 0
        else:
            scale = (x.max(axis=0) - x.min(axis=0)) / (self._qmax - self._qmin)
            scale *= s
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


def read_file(file_name):
    if file_name.endswith("safetensors"):
        try:
            from paddlenlp.utils.safetensors import fast_load_file as load_file
        except:
            from safetensors.numpy import load_file

        read_tensors = load_file(file_name)
        for key in list(read_tensors.keys()):
            if isinstance(read_tensors[key], np.ndarray):
                read_tensors[key] = paddle.Tensor(read_tensors.pop(key), zero_copy=True)
    else:
        read_tensors = paddle.load(file_name)
    return read_tensors


def save_file(output_path, file_name, tensors, safe_serialization=True):
    # tensors: dict {key: tensor}
    if safe_serialization:
        from safetensors.numpy import save_file as _save_file
        
        if file_name == "model_state.pdparams":
            file_name = "model.safetensors"

        for key in list(tensors.keys()):
            if isinstance(tensors[key], paddle.Tensor):
                tensors[key] = tensors.pop(key).cpu().numpy()
        _save_file(tensors, os.path.join(output_path, file_name), metadata={"format": "np"})
    else:
        paddle.save(tensors, os.path.join(output_path, file_name))
