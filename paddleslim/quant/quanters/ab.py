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
from paddle.nn.initializer import Constant
from paddle.utils import unique_name
from paddle.framework import ParamAttr
from paddle.quantization.factory import QuanterFactory
from .base_fake_quanter import BaseFakeQuanterLayer
from .utils import *


class ABQuanter(QuanterFactory):
    def __init__(self,
                 bit_length=8,
                 channel_wise=False,
                 sign=True,
                 dtype='float32',
                 name=None,
                 windows=[0, 0],
                 quanter=None):
        super().__init__(
            bit_length=bit_length,
            channel_wise=channel_wise,
            sign=sign,
            dtype=dtype,
            name=name,
            windows=windows,
            quanter=quanter)

    def _get_class(self):
        return ABQuanterLayer


class ABQuanterLayer(BaseFakeQuanterLayer):
    def __init__(self,
                 layer,
                 bit_length=8,
                 channel_wise=False,
                 sign=True,
                 dtype='float32',
                 name=None,
                 windows=[0, 0],
                 quanter=None):
        super().__init__()
        self._bit_length = bit_length
        self._sign = sign
        self._qmin, self._qmax = self.qmin_qmax
        self._current_iters = -1
        self._windows = windows
        if quanter:
            self._quanter = quanter._instance(layer)
            self._scale = self._quanter.scales()
            self._quant_axis = self._quanter.quant_axis()
        else:
            self._init_scale(layer, channel_wise, name, dtype)
            self._quanter = None

    def _init_scale(self, layer, channel_wise, name, dtype):
        if channel_wise:
            for key in CHANNEL_AXIS.keys():
                if issubclass(type(layer), key):
                    self._quant_axis = CHANNEL_AXIS[key]
                    break
            self._channel_num = layer.weight.shape[self._quant_axis]
        else:
            self._quant_axis = -1
            self._channel_num = 1

        scale_prefix = f"{name}.scale" if name else 'quant_dequant.scale'
        self._scale_name = unique_name.generate(scale_prefix)
        scale_attr = ParamAttr(
            name=self._scale_name,
            initializer=Constant(0.001),
            trainable=False, )
        self._scale = self.create_parameter(
            shape=[self._channel_num], attr=scale_attr, dtype=dtype)
        self._scale.stop_gradient = True

    def forward(self, inputs):
        if self.training:
            alpha = self._update_params(inputs.detach())
            if self._quanter is None:
                if self._current_iters <= self._windows[1]:
                    with paddle.no_grad():
                        qdq_inputs = self._quant_dequant(inputs)
                else:
                    qdq_inputs = self._quant_dequant(inputs, update=True)
                return inputs * (1 - alpha) + alpha * qdq_inputs
            else:
                if self._current_iters <= self._windows[1]:
                    with paddle.no_grad():
                        qdq_inputs = self._quanter(inputs)
                else:
                    qdq_inputs = self._quanter(inputs)
                return inputs * (1 - alpha) + alpha * qdq_inputs
        else:
            if self._quanter is None:
                return self._quant_dequant(inputs)
            else:
                return self._quanter(inputs)

    def _update_params(self, inputs):
        if self._quanter is None:
            if self._channel_num > 1:
                reduce_axis = tuple([
                    i for i in range(len(inputs.shape)) if i != self._quant_axis
                ])
                abs_max_values = paddle.max(
                    paddle.abs(inputs), axis=reduce_axis)
                self._scale.set_value(abs_max_values)
            else:
                min_value, max_value = avg_min_max(inputs)
                cur_scale = (max_value - min_value) / (
                    self._qmax - self._qmin) * self._qmax
                if self._current_iters < 0:
                    self._scale.set_value(cur_scale.unsqueeze(0))
                else:
                    self._scale.set_value(
                        cur_scale.unsqueeze(0) * 0.1 + 0.9 * self._scale)

        t0, t1 = self._windows
        self._current_iters += 1
        if t0 == t1 or self._current_iters > t1:
            return 1.
        if self._current_iters <= t0:
            return 0.
        if self._current_iters <= t1:
            return 1 - math.pow((t1 - self._current_iters) / (t1 - t0), 3)

    def _quant_dequant(self, x, update=False):
        if self._scale.shape[0] == 1:
            s = self._scale / self._qmax
        else:
            weight_shape = x.shape
            scale = self._scale.reshape([self._scale.shape[0], 1])
            if len(weight_shape) == 2:
                scale = scale.repeat_interleave(weight_shape[0], axis=1).t()
            else:
                scale = scale.repeat_interleave(
                    weight_shape[1] * weight_shape[2] * weight_shape[3], axis=1)
                scale = scale.reshape(weight_shape)
            s = scale / self._qmax
        quant_x = paddle.clip(paddle.round(x / s), self._qmin, self._qmax)
        dequant_x = s * quant_x
        if update:
            return x + (dequant_x - x).detach()
        return dequant_x

    def bit_length(self):
        return self._bit_length

    def quant_axis(self):
        return self._quant_axis

    def scales(self):
        return self._scale

    def zero_points(self):
        return None
