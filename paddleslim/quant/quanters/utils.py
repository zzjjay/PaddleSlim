# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import random
import math
import numpy as np
from typing import Dict

__all__ = [
    "sigma",
    "min_max",
    "histogram",
    "CHANNEL_AXIS",
    "init_lsq",
    "percent",
    "avg_min_max",
]

CHANNEL_AXIS: Dict[type, int] = {
    paddle.nn.Conv2D: 0,
    paddle.nn.Linear: 1,
    paddle.distributed.fleet.meta_parallel.ColumnParallelLinear: 1,
    paddle.distributed.fleet.meta_parallel.RowParallelLinear: 1,
}


def sigma(src, axis=None):
    # src: Paddle.Tensor
    mean = src.mean(axis=axis)
    std = src.std(axis=axis)
    min_value = mean - 3. * std
    max_value = mean + 3. * std
    return min_value, max_value


def min_max(src, axis=None):
    # src: Paddle.Tensor
    min_value = src.min(axis=axis)
    max_value = src.max(axis=axis)
    return min_value, max_value


def avg_min_max(src):
    # src: B, N, H, W
    B, N, H, W = src.shape
    new_src = src.transpose([1, 0, 2, 3]).reshape([N, -1])
    min_value = new_src.min(axis=1).mean()
    max_value = new_src.max(axis=1).mean()
    return min_value, max_value


def histogram(src):
    # scr: Paddle.Tensor
    hist_array, min_value, max_value, mult_factor, offset = _tensor_histogram(
        src)
    if hist_array is None:
        return min_value, max_value

    new_mn_scaled, new_mx_scaled = _extrema_hist_search(hist_array)
    new_mn = (new_mn_scaled / mult_factor) + offset
    new_mx = (new_mx_scaled / mult_factor) + offset

    new_mn = max(min_value, new_mn)
    new_mx = min(max_value, new_mx)
    return new_mn, new_mx


def _tensor_histogram(src, fast_mode=True):
    # downsample for fast_mode
    fast_stride = 2
    fast_stride2 = fast_stride * 2
    if fast_mode and len(src.shape) == 4 and (src.shape[2] > fast_stride2) and (
            src.shape[3] > fast_stride2):
        r_start = random.randint(0, fast_stride - 1)
        c_start = random.randint(0, fast_stride - 1)
        src = src[..., r_start::fast_stride, c_start::fast_stride]

    mn = src.min()
    mx = src.max()
    if mn == 0 and mx == 0:
        return None, mn, mx, 1.0, 0.0

    num_bins = 255.0
    cum_freq = float(100.0)
    offset = mn
    range_val = paddle.abs(mx - mn)
    mult_factor = (num_bins / range_val)
    tensor_int = (src.flatten() - offset) * mult_factor
    tensor_int = paddle.round(tensor_int)

    hist = np.bincount(tensor_int.numpy().astype(np.int32))
    hist_sum = np.sum(hist)
    hist_array = hist.astype(np.float32) * cum_freq / float(hist_sum)
    return hist_array, mn, mx, mult_factor, offset


def _extrema_hist_search(hist_array, range_shrink_percentile=0.01):
    # hist_array: numpy array
    new_mn_scaled = 0
    new_mx_scaled = len(hist_array) - 1
    hist_sum_left = 0.0
    hist_sum_right = 0.0
    for h_idx in range(len(hist_array)):
        r_idx = len(hist_array) - 1 - h_idx
        hist_sum_left += hist_array[h_idx]
        hist_sum_right += hist_array[r_idx]
        if hist_sum_left < range_shrink_percentile:
            new_mn_scaled = h_idx
        if hist_sum_right < range_shrink_percentile:
            new_mx_scaled = r_idx
    return paddle.to_tensor(new_mn_scaled), paddle.to_tensor(new_mx_scaled)


def init_lsq(src, qmax, axis=None):
    # Initialization method from lsq
    # src: Paddle.Tensor
    # qmax: float
    value = src.abs().mean(axis=axis) * 2 * math.sqrt(qmax)
    return -value, value


def percent(src, ratio=0.9):
    sorted = src.abs().flatten().sort()
    value = sorted[int(len(sorted) * ratio)]
    return -value, value


def floor_ceil(x):
    # x: Paddle.Tensor
    return paddle.where((x < 0) & (x - x.floor() == 0.5), x.ceil(), x.round())
