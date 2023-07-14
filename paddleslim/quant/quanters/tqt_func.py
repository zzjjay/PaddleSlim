# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import math
import paddle
from paddle.autograd import PyLayer


class TQTFunc(PyLayer):
    @staticmethod
    def forward(ctx, x, log2_t, qmin, qmax):
        s = 2.**paddle.ceil(log2_t) / qmax
        quant_x = paddle.clip(paddle.round(x / s), qmin, qmax)
        dequant_x = quant_x * s
        ctx.save_for_backward(x / s, s)
        ctx.other = qmin, qmax
        return dequant_x

    @staticmethod
    def backward(ctx, grad_output):
        x_div_s, s = ctx.saved_tensor()
        qmin, qmax = ctx.other
        rounded = paddle.round(x_div_s)
        lower = paddle.cast(rounded < qmin, rounded.dtype)
        upper = paddle.cast(rounded > qmax, rounded.dtype)
        middle = paddle.cast((qmin <= rounded) & (rounded <= qmax),
                             rounded.dtype)
        grad_s = (rounded - x_div_s) * middle + qmin * lower + qmax * upper
        grad_log_2_t = math.log(2) * s * grad_s
        grad_x = middle
        return grad_output * grad_x, grad_output * grad_log_2_t, None, None
