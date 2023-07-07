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

from typing import Dict

import paddle
from paddle.nn.initializer import Constant
from paddle.utils import unique_name
from paddle.framework import ParamAttr
from paddle.quantization.factory import QuanterFactory
from .base_fake_quanter import BaseFakeQuanterLayer

CHANNEL_AXIS: Dict[type, int] = {
    paddle.nn.Conv2D: 0,
    paddle.nn.Linear: 1,
    paddle.distributed.fleet.meta_parallel.ColumnParallelLinear: 1,
    paddle.distributed.fleet.meta_parallel.RowParallelLinear: 1,
}


class FreezeQuanter(QuanterFactory):
    def __init__(self,
                 bit_length=8,
                 channel_wise=False,
                 sign=True,
                 dtype='float32',
                 name=None,
                 quanter=None):
        super().__init__(
            bit_length=bit_length,
            channel_wise=channel_wise,
            sign=sign,
            dtype=dtype,
            name=name,
            quanter=quanter)

    def _get_class(self):
        return FreezeQuanterLayer


class FreezeQuanterLayer(BaseFakeQuanterLayer):
    def __init__(self,
                 layer,
                 bit_length=8,
                 channel_wise=False,
                 sign=True,
                 dtype='float32',
                 name=None,
                 quanter=None):
        super().__init__()
        self._bit_length = bit_length
        self._sign = sign
        if quanter is None:
            self._init_scale(layer, channel_wise, name, dtype)
            self._quanter = None
            self._qmin, self._qmax = self.qmin_qmax
        else:
            self._quanter = quanter._instance(layer)
            self._scale = self._quanter.scales()
            self._quant_axis = self._quanter.quant_axis()
            self.bit_length = self._quanter.bit_length()

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
        if self._quanter is None:
            return self._quant_dequant(inputs)
        else:
            self._quanter.eval()
            return self._quanter(inputs)

    def _quant_dequant(self, x, weight_shape):
        if self._scale.shape[0] == 1:
            s = self._scale / self._qmax
        else:
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
        return x + (dequant_x - x).detach()

    def bit_length(self):
        return self._bit_length

    def quant_axis(self):
        return self._quant_axis

    def scales(self):
        return self._scale

    def zero_points(self):
        return None
