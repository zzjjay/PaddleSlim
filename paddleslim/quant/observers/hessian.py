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

import numpy as np
import paddle
from .uniform import UniformObserver
from paddle.quantization.factory import ObserverFactory

from paddle.nn import Layer
from paddle.nn import functional as F
from paddle.nn.quant.format import ConvertibleQuantedLayer


class HessianObserver(ObserverFactory):
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

    def __init__(self, quant_bits=8):
        super(HessianObserver, self).__init__(quant_bits=quant_bits)

    def _get_class(self):
        return HessianObserverLayer


class HessianObserverLayer(UniformObserver):
    def __init__(self,
                 layer,
                 quant_bits=8,
                 alpha=0.0,
                 beta=1.2,
                 n=100,
                 parallel_n=1):
        super(HessianObserverLayer, self).__init__(quant_bits=quant_bits)
        self._quant_bits = quant_bits
        self._qmin, self._qmax = self.qmin_qmax
        self._alpha = alpha
        self._beta = beta
        self._n = n
        self._interval = paddle.to_tensor([
            self._alpha + i * (self._beta - self._alpha) / self._n
            for i in range(self._n + 1)
        ])
        self._parallel_n = parallel_n
        self.mode = 'normal'

    def forward(self, inputs):
        """ Calculate forward pass.
        """
        if self.mode == 'quant_prepare':
            self.candidates = self._generate_candidates(inputs)
        if self.mode == 'quant_search':
            self._scale = self.candidates.pop()
            inputs = self._quant_dequant(inputs, self._scale)
        if self.mode == 'quant_forward':
            inputs = self._quant_dequant(inputs, self._scale)
        return inputs

    def _generate_candidates(self, inputs):
        """
        Generate candidate scales.
        """
        max_value = inputs.abs().max()
        return max_value * self._interval

    def _quant_dequant(self, x, scale):
        s = scale / self._qmax
        quant_x = paddle.clip(paddle.round(x / s), self._qmin, self._qmax)
        dequant_x = quant_x * s
        return dequant_x

    def min_value(self) -> float:
        return self._min

    def max_value(self) -> float:
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
        return self._scale

    def zero_points(self):
        """ Return output zero points.
        """
        return self._zero_point


class HessianConv2D(ConvertibleQuantedLayer):
    """
    The computational logic of QuantizedConv2D is the same as Conv2D.
    The only difference is that its inputs are all fake quantized.
    """

    def __init__(self, layer: Layer, q_config):
        super().__init__()

        # For Conv2D
        self._groups = layer._groups
        self._stride = layer._stride
        self._padding = layer._padding
        self._padding_mode = layer._padding_mode
        if self._padding_mode != 'zeros':
            self._reversed_padding_repeated_twice = (
                layer._reversed_padding_repeated_twice)
        self._dilation = layer._dilation
        self._data_format = layer._data_format
        self.weight = layer.weight
        self.bias = layer.bias

        self.weight_quanter = None
        self.activation_quanter = None
        if q_config.weight is not None:
            self.weight_quanter = q_config.weight._instance(layer)
        if q_config.activation is not None:
            self.activation_quanter = q_config.activation._instance(layer)
        self.output_hook = None
        self.current_iters = -1
        self._layer_name = layer.full_name()

    def forward(self, input):
        quant_input = input
        quant_weight = self.weight

        if not self.training:
            self.activation_quanter.mode = 'quant_forward'

        if self.weight_quanter is not None:
            quant_weight = self.weight_quanter(self.weight)
        if self.activation_quanter is not None:
            if self.current_iters == 0:
                self.activation_quanter.mode = 'quant_prepare'
            if self.current_iters == 1:
                print("********Current Layer:", self._layer_name)
                self._search_hessian(input, quant_weight)
            quant_input = self.activation_quanter(input)

        output = self._conv_forward(quant_input, quant_weight)

        if self.output_hook is None and self.current_iters == -1:
            self.output_hook = output.register_hook(self._grad_hook)
        if self.current_iters == 0:
            self.original_output = output

        self.current_iters += 1

    def _search_hessian(self, input, weight):
        self.activation_quanter.mode = 'quant_search'
        best_sim = paddle.to_tensor(float('inf'))
        print("======Begin searching hessian======")
        for i in range(len(self.activation_quanter._interval)):
            if i == 0:
                print("abs max:", input.abs().max())
            if i % 10 == 0:
                print("Current Iter:", i)
            quant_input = self.activation_quanter(input)
            with paddle.no_grad():
                qdq_output = self._conv_forward(quant_input, weight)
            cur_sim = self._cal_similarity(self.original_output, qdq_output)
            if cur_sim < best_sim:
                print(
                    f"scales change [{best_scales}] to [{self.activation_quanter.scales()}]"
                )
                best_sim = cur_sim
                best_scales = self.activation_quanter.scales()

        print("====== Best_scales:", best_scales)
        self.activation_quanter._scale = best_scales
        self.activation_quanter.mode = 'normal'
        self.output_hook.remove()
        del self.original_output, self.output_grad

    def _cal_similarity(self, ori_output, qdq_output):
        grad = self.output_grad.reshape(ori_output.shape)
        similarity = -(grad * (ori_output - qdq_output))**2
        return similarity.mean()

    def _grad_hook(self, grad):
        self.output_grad = grad

    def _conv_forward(self, inputs, weights):
        if self._padding_mode != 'zeros':
            inputs = F.pad(
                inputs,
                self._reversed_padding_repeated_twice,
                mode=self._padding_mode,
                data_format=self._data_format, )
            self._padding = 0

        return F.conv2d(
            inputs,
            weights,
            bias=self.bias,
            padding=self._padding,
            stride=self._stride,
            dilation=self._dilation,
            groups=self._groups,
            data_format=self._data_format, )

    def weights_to_quanters(self):
        return [('weight', 'weight_quanter')]

    def activation_quanters(self):
        return ['activation_quanter']
