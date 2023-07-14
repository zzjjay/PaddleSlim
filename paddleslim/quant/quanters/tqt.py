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

import paddle
from paddle.nn.initializer import Constant
from paddle.utils import unique_name
from paddle.framework import ParamAttr
from paddle.quantization.factory import QuanterFactory
from .base_fake_quanter import BaseFakeQuanterLayer
from .utils import *
from .tqt_func import TQTFunc


class TQTQuanter(QuanterFactory):
    def __init__(self,
                 bit_length=8,
                 channel_wise=False,
                 sign=True,
                 dtype='float32',
                 name=None):
        super().__init__(
            bit_length=bit_length,
            channel_wise=channel_wise,
            sign=sign,
            dtype=dtype,
            name=name)

    def _get_class(self):
        return TQTQuanterLayer


class TQTQuanterLayer(BaseFakeQuanterLayer):
    def __init__(self,
                 layer,
                 bit_length=8,
                 channel_wise=False,
                 sign=True,
                 dtype='float32',
                 name=None):
        super().__init__()
        self._bit_length = bit_length
        self._sign = sign
        self._qmin, self._qmax = self.qmin_qmax
        self._channel_wise = channel_wise

        if self._channel_wise:
            for key in CHANNEL_AXIS.keys():
                if issubclass(type(layer), key):
                    self._quant_axis = CHANNEL_AXIS[key]
                    break
            self._channel_num = layer.weight.shape[self._quant_axis]
        else:
            self._quant_axis = -1
            self._channel_num = 1

        log2_t_prefix = f"{name}.log2_t" if name else 'quant_dequant.log2_t'
        log2_t_attr = ParamAttr(
            name=unique_name.generate(log2_t_prefix),
            initializer=Constant(1.),
            trainable=False, )
        self._log2_t = self.create_parameter(
            shape=[self._channel_num], attr=log2_t_attr, dtype=dtype)

        # scale = 2 ** log2_t
        scale_prefix = f"{name}.scale" if name else 'quant_dequant.scale'
        scale_attr = ParamAttr(
            name=unique_name.generate(scale_prefix),
            initializer=Constant(1.),
            trainable=False, )
        self._scale = self.create_parameter(
            shape=[self._channel_num], attr=scale_attr, dtype=dtype)
        self._init_state = 0

    def forward(self, inputs):
        if self.training:
            if self._init_state == 0:
                with paddle.no_grad():
                    self._init_params(inputs)
            outputs = TQTFunc.apply(inputs, self._log2_t, self._qmin,
                                    self._qmax)
            return outputs
        else:
            return self._quant_dequant(inputs)

    def _init_params(self, inputs):
        _, thershold = percent(inputs)
        log2_t = paddle.log2(thershold)
        if len(log2_t.shape) == 0:
            log2_t = log2_t.unsqueeze(0)
        self._log2_t.set_value(log2_t)
        self._init_state += 1

    def _quant_dequant(self, x):
        scale = 2.**paddle.ceil(self._log2_t)
        s = scale / self._qmax
        quant_x = paddle.clip(paddle.round(x / s), self._qmin, self._qmax)
        dequant_x = quant_x * s
        return dequant_x

    def bit_length(self):
        return self._bit_length

    def quant_axis(self):
        return self._quant_axis

    def scales(self):
        scale = 2.**paddle.ceil(self._log2_t)
        self._scale.set_value(scale)
        return self._scale

    def zero_points(self):
        return None
