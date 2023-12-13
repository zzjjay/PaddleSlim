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
import numpy as np
import math
from paddle.framework import ParamAttr
from paddle.nn import Layer
from paddle.nn.initializer import Constant
from paddle.utils import unique_name
from paddle.quantization.factory import QuanterFactory
from .base_fake_quanter import BaseFakeQuanterLayer
from .lsq_func import LsqFunc, LsqPlusActFunc, Round
from .channel_wise_abs_max import CHANNEL_AXIS


class ActLSQplusQuanter(QuanterFactory):
    r"""
    Activation quantizer. More details can be found in 
    https://arxiv.org/pdf/1902.08153.pdf and https://arxiv.org/pdf/2004.09576.pdf.
    Args:
        per_channel(bool): whether layer-wise or channel-wise quantization, where True for layer-wise quantization and False for channel-wise quantization.
        batch_init(int): number of batches that collect Gaussian approximation for the weight distribution in each layer.
        dtype(str): data type.
        name(str): the name of the layer.
    Examples:
       .. code-block:: python
            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import ActLSQplusQuanter, WeightLSQplusQuanter
            weight_quanter = WeightLSQplusQuanter()
            act_quanter = ActLSQplusQuanter()
            q_config = QuantConfig(activation=act_quanter, weight=weight_quanter)
    """

    def __init__(self,
                 quant_bits=8,
                 sign=True,
                 symmetric=True,
                 per_channel=False,
                 batch_init=20,
                 dtype='float32',
                 name=None):
        super(ActLSQplusQuanter, self).__init__(
            quant_bits=quant_bits,
            sign=sign,
            symmetric=symmetric,
            per_channel=per_channel,
            batch_init=batch_init,
            dtype=dtype,
            name=name)

    def _get_class(self):
        return ActLSQplusQuanterLayer


class ActLSQplusQuanterLayer(BaseFakeQuanterLayer):
    def __init__(self,
                 layer,
                 quant_bits=8,
                 sign=True,
                 symmetric=True,
                 per_channel=False,
                 batch_init=20,
                 dtype='float32',
                 name=None):
        super(ActLSQplusQuanterLayer, self).__init__()
        self._symmetric = symmetric
        self._per_channel = per_channel
        self._batch_init = batch_init
        self._name = name
        if per_channel:
            for key in CHANNEL_AXIS.keys():
                if issubclass(type(layer), key):
                    self._quant_axis = CHANNEL_AXIS[key]
                    break
        self.qmin, self.qmax = self.qmin_qmax

        self._init_state = 0

        scale_prefix = ("{}.scale".format(name)
                        if name else 'quant_dequant.scale')
        self._scale_name = unique_name.generate(scale_prefix)

        s_attr = ParamAttr(
            name=self._scale_name, initializer=Constant(1.0), trainable=True)
        self._scale = self.create_parameter(shape=[1], attr=s_attr, dtype=dtype)
        self._scale.stop_gradient = False

        if not self._symmetric:
            beta_prefix = ("{}.beta".format(name)
                           if name else 'quant_dequant.beta')
            self._beta_name = unique_name.generate(beta_prefix)

            beta_attr = ParamAttr(
                name=self._beta_name, initializer=Constant(0.0), trainable=True)
            self._beta = self.create_parameter(
                shape=[1], attr=beta_attr, dtype='float32')
            self._beta.stop_gradient = False

    def _init_params(self, activation):
        self.g = paddle.to_tensor(
            1.0 / math.sqrt(activation.numel() * self.qmax))
        min_a = paddle.min(activation.detach())
        max_a = paddle.max(activation.detach())
        scale = (max_a - min_a) / (self.qmax - self.qmin)
        if len(scale.shape) == 0:
            scale = scale.unsqueeze(0)
        self._scale.set_value(scale)
        if not self._symmetric:
            self._beta.set_value(min_a - self._scale * self.qmin)
        self._init_state += 1

    def _collect_gaussian(self, activation):
        min_a = paddle.min(activation.detach())
        max_a = paddle.max(activation.detach())
        if len(scale.shape) == 0:
            scale = scale.unsqueeze(0)
        self._scale.set_value(self._scale * 0.9 + 0.1 * scale)
        if not self._symmetric:
            self._beta.set_value(self._scale * 0.9 + 0.1 *
                                 (min_a - self._scale * self.qmin))
        self._init_state += 1

    def forward(self, activation):
        if self._init_state == 0:
            self._init_params(activation)
        elif self._init_state < self._batch_init:
            self._collect_gaussian(activation)

        activation.stop_gradient = False
        if not self._symmetric:
            q_a = LsqPlusActFunc.apply(activation, self._scale, self._beta,
                                       self.g, self.qmin, self.qmax)
        else:
            q_a = LsqFunc.apply(
                activation,
                self._scale,
                self.g,
                self.qmin,
                self.qmax,
                per_channel=False)
        return q_a

    def bit_length(self):
        """ Return the bit length of quantized data.
        """
        return self._quant_bits

    def quant_axis(self):
        """ Return quantization axis.
        """
        return self._quant_axis

    def scales(self):
        """ Return output scales.
        For LSQ, scale = scale / qmax, so the output scales should multiply qmax
        """
        return self._scale * self.qmax

    def zero_points(self):
        """ Return output zero points.
        """
        if self._zero_point is None:
            if self._symmetric:
                if self._sign:
                    self._zero_point = 0
                else:
                    self._zero_point = (self.qmax + self.qmin) / 2
            else:
                self._zero_point = self.qmin - Round(self.qmin / self._scale)
                self._zero_point = paddle.clip(self._zero_point, self.qmin,
                                               self.qmax)
        return self._zero_point
