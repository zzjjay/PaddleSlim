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

import logging
import numpy as np
import paddle
from paddle.utils import unique_name
from paddle.nn.initializer import Assign
from paddle.quantization.factory import ObserverFactory
from .uniform import UniformObserver
from ...common import get_logger

_logger = get_logger(
    __name__,
    logging.INFO,
    fmt='%(asctime)s-%(levelname)s: %(message)s', )

GAMMA = -0.1
ZETA = 1.1


class AdaroundObserver(ObserverFactory):
    r"""
    It collects maximum absolute values of target tensor.
    Args:
        bit_length(int, optional): Number of bits to represent an quantized integer in binary.
        dtype(str, optional): The data type of input tensor.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.
    Examples:
       .. code-block:: python
            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
            quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.99)
            q_config = QuantConfig(activation=quanter, weight=quanter)
    """

    def __init__(self, ptq_observer, warmup=10, quant_bits=8):
        super(AdaroundObserver, self).__init__(
            ptq_observer=ptq_observer, warmup=warmup, quant_bits=quant_bits)

    def _get_class(self):
        return AdaroundObserverLayer


class AdaroundObserverLayer(UniformObserver):
    def __init__(self, layer, ptq_observer, warmup=10, quant_bits=8):
        super(AdaroundObserverLayer, self).__init__(quant_bits=quant_bits)
        self._quant_bits = quant_bits
        self.qmin, self.qmax = self.qmin_qmax
        self._layer = layer
        self._ptq_observer = ptq_observer
        self._warmup = warmup
        self._currnt_iters = 0
        self.alpha = None

        # scale_prefix = ("{}.scale".format(layer.full_name()))
        # self._scale_name = unique_name.generate(scale_prefix)
        # scale_attr = paddle.ParamAttr(
        #     name=self._scale_name, initializer=Assign(value=scale),trainable=False)
        # self._scale = self.create_parameter(
        #     shape=scale.shape, attr=scale_attr, dtype=dtype)
        # self._scale.stop_gradient = True

    def _init_alpha(self):
        """ Initialize alpha
        """
        weight = self._layer.weight
        scale = self._ptq_observer.scales()

        quantized_weight = np.clip(
            self._quant(weight.numpy(), scale), self.qmin, self.qmax)
        floor_weight = np.floor(quantized_weight)
        mantissa = quantized_weight - floor_weight
        init_alpha = -np.log((ZETA - GAMMA) / (mantissa - GAMMA) - 1)

        alpha_prefix = ("{}.adaround".format(self._layer.full_name()))
        self._alpha_name = unique_name.generate(alpha_prefix)
        alpha_attr = paddle.ParamAttr(
            name=self._alpha_name,
            initializer=Assign(value=init_alpha),
            trainable=True)
        self.alpha = self.create_parameter(
            shape=weight.shape, attr=alpha_attr, dtype=weight.dtype)

    def forward(self, weights):
        """ Calculate forward pass.
        """
        self._current_iters += 1
        if self._current_batch_id <= self._warmup:
            return self._observer(weights)

        if self.alpha is None:
            self._init_alpha()

        scale = self._ptq_observer.scales()
        h_v = paddle.clip(
            paddle.nn.functional.sigmoid(self.alpha) * (ZETA - GAMMA) + GAMMA,
            0,
            1, )

        quantized_weight = self._quant(weights, scale)
        floor_weight = (paddle.floor(quantized_weight) -
                        quantized_weight).detach() + quantized_weight
        clip_weight = paddle.clip(floor_weight + h_v, self.qmin, self.qmax)
        dequant_weight = self._dequant(clip_weight, scale)
        return dequant_weight

    def cal_thresholds(self):
        """ Compute thresholds for adaround function.
        """
        self._min, self._max = self._ptq_observer._min, self._ptq_observer._max
        self._scale, self._zero_point = self.cal_scales_zero_points()

    def _quant(self, x, scale):
        s = scale / self.qmax
        quant_x = x / s
        return quant_x

    def _dequant(self, x, scale):
        s = scale / self.qmax
        dequant_x = s * x
        return dequant_x

    def min_value(self) -> float:
        """ The minimum value of floating-point numbers."""
        return self._min

    def max_value(self) -> float:
        """ The maximum value of floating-point numbers."""
        return self._max

    def bit_length(self):
        """ Return the bit length of quantized data.
        """
        return self._quant_bits

    def quant_axis(self):
        """ Return quantization axis.
        """
        return -1

    def scales(self):
        """ Return output scales.
        """
        return self._ptq_observer.scales()

    def zero_points(self):
        """ Return output zero points.
        """
        return self._zero_point
