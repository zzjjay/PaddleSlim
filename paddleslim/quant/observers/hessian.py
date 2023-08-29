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
import numpy as np
import paddle
from .uniform import UniformObserver
from paddle.quantization.factory import ObserverFactory

from paddle.nn import Layer
from paddle.nn import functional as F
from paddle.nn.quant.format import ConvertibleQuantedLayer
from .channel_wise import CHANNEL_AXIS


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

    def __init__(self,
                 quant_bits=8,
                 alpha=0.1,
                 beta=1.2,
                 n=100,
                 parallel_n=1,
                 channel_wise=False):
        super(HessianObserver, self).__init__(
            quant_bits=quant_bits,
            alpha=alpha,
            beta=beta,
            n=n,
            parallel_n=parallel_n,
            channel_wise=channel_wise)

    def _get_class(self):
        return HessianObserverLayer


class HessianObserverLayer(UniformObserver):
    def __init__(self,
                 layer,
                 quant_bits=8,
                 alpha=0.1,
                 beta=1.2,
                 n=100,
                 parallel_n=1,
                 channel_wise=False):
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
        if channel_wise:
            for key in CHANNEL_AXIS.keys():
                if issubclass(type(layer), key):
                    self._quant_axis = CHANNEL_AXIS[key]
                    break
        else:
            self._quant_axis = -1

    def forward(self, inputs, idx=0):
        """ Calculate forward pass.
        """
        if self.mode == 'quant_prepare':
            with paddle.no_grad():
                self._generate_candidates(inputs)
        if self.mode == 'quant_search':
            self._scale = self._interval[idx] * self._tmp_scale
            return self._quant_dequant(inputs, self._scale)
        if self.mode == 'quant_forward':
            return self._quant_dequant(inputs, self._scale)
        return inputs

    def _generate_candidates(self, inputs):
        """
        Generate candidate scales.
        """
        if self._quant_axis != -1:
            reduce_axis = tuple(
                [i for i in range(len(inputs.shape)) if i != self._quant_axis])
            self._tmp_scale = inputs.abs().max(axis=reduce_axis)
        else:
            self._tmp_scale = inputs.abs().max()
        if self._scale is None:
            self._scale = self._tmp_scale

    def _quant_dequant(self, x, scale):
        if scale.size == 1:
            s = scale / self._qmax
        else:
            weight_shape = x.shape
            scale = scale.reshape([scale.shape[0], 1])
            if len(weight_shape) == 2:
                scale = scale.repeat_interleave(weight_shape[0], axis=1).t()
            else:
                scale = scale.repeat_interleave(
                    weight_shape[1] * weight_shape[2] * weight_shape[3], axis=1)
                scale = scale.reshape(weight_shape)

            s = scale / self._qmax
        quant_x = paddle.clip(paddle.round(x / s), self._qmin, self._qmax)
        dequant_x = quant_x * s
        return dequant_x

    def cal_min_max(self, inputs):
        abs_avg_value = paddle.abs(inputs.reshape((inputs.shape[0], -1)))
        abs_avg_value = float(paddle.mean(paddle.max(abs_avg_value, axis=(1))))
        return 0, abs_avg_value

    def cal_thresholds(self):
        """ Compute thresholds for MAX function.
        """
        self._min, self._max = self._avg_min, paddle.mean(
            paddle.to_tensor(self._avg_list))
        self._scale, self._zero_point = self.cal_scales_zero_points()

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
        return self._quant_axis

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
        self.output_grad = None

    def forward(self, input):
        self.current_iters += 1
        quant_input = input
        quant_weight = self.weight

        if self.current_iters == 0:
            self.activation_quanter.mode = 'quant_prepare'
            self.weight_quanter.mode = 'quant_prepare'

        if self.current_iters == 1:
            print("********Current Layer:", self._layer_name)
            if self.output_grad is not None:
                self._search_hessian(input, self.weight)

        quant_input = self.activation_quanter(input)
        quant_weight = self.weight_quanter(self.weight)

        output = self._conv_forward(quant_input, quant_weight)

        if output.stop_gradient is False:
            self.output_hook = output.register_hook(self._grad_hook)

        return output

    def _search_hessian(self, input, weight):
        self.original_output = self._conv_forward(input, weight)
        print("======Begin searching hessian======")
        print('input:', input)
        rounds = 3
        for r in range(rounds):
            print("Round:", r)
            self.activation_quanter.mode = 'quant_forward'
            qdq_input = self.activation_quanter(input)
            self._search_w(qdq_input, weight)

            self.weight_quanter.mode = 'quant_forward'
            qdq_weight = self.weight_quanter(weight)
            self._search_a(input, qdq_weight)

        self.output_hook.remove()
        del self.output_grad, self.original_output
        self.output_grad = None
        #return original_output

    def _search_a(self, input, weight):
        print("******Begin search act")
        best_sim = paddle.to_tensor(float('-inf'))
        best_scales = 0
        self.activation_quanter.mode = 'quant_search'
        for i in range(len(self.activation_quanter._interval)):
            if i == 0:
                print("abs max:", float(input.abs().max()))
            if i % 10 == 0:
                print("Current Iter:", i)
            quant_input = self.activation_quanter(input, i)
            with paddle.no_grad():
                qdq_output = self._conv_forward(quant_input, weight)
            cur_sim = self._cal_similarity(self.original_output, qdq_output)
            if cur_sim > best_sim:
                print(
                    f"Iters:{i}, scales change [{float(best_scales)}] to [{float(self.activation_quanter.scales())}]"
                )
                best_sim = cur_sim
                best_scales = self.activation_quanter.scales()

        print("======Act Best_scales:", float(best_scales))
        self.activation_quanter._scale = best_scales
        self.activation_quanter.mode = 'quant_forward'

    def _search_w(self, input, weight):
        print("******Begin search weight")
        print("act scale:", self.activation_quanter._scale)
        best_sim = paddle.to_tensor(float('-inf'))
        best_scales = 0
        self.weight_quanter.mode = 'quant_search'
        for i in range(35, len(self.weight_quanter._interval)):
            if i == 0:
                print("weight max:", float(weight.abs().max()))
            if i % 10 == 0:
                print("Current Iter:", i)
            quant_weight = self.weight_quanter(weight, i)
            with paddle.no_grad():
                qdq_output = self._conv_forward(input, quant_weight)
            cur_sim = self._cal_similarity(self.original_output, qdq_output)
            if cur_sim > best_sim:
                print(f"Iters:{i}, scales changed")

                best_sim = cur_sim
                best_scales = self.weight_quanter.scales()

        print("======Weight Best_scales:", best_scales.shape)
        self.weight_quanter._scale = best_scales
        self.weight_quanter.mode = 'quant_forward'

    def _cal_similarity(self, ori_output, qdq_output):
        grad = self.output_grad.reshape(ori_output.shape)

        similarity = -(grad * (ori_output - qdq_output))**2
        #similarity = -((ori_output - qdq_output))**2  # mse
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


class HessianLinear(ConvertibleQuantedLayer):
    """
    The computational logic of QuantizedLinear is the same as Linear.
    The only difference is that its inputs are all fake quantized.
    """

    def __init__(self, layer: Layer, q_config):
        super().__init__()
        # For Linear
        self.weight = layer.weight
        self.bias = layer.bias
        self.name = layer.name
        # For FakeQuant

        self.weight_quanter = None
        self.activation_quanter = None
        if q_config.weight is not None:
            self.weight_quanter = q_config.weight._instance(layer)
        if q_config.activation is not None:
            self.activation_quanter = q_config.activation._instance(layer)

        self.output_hook = None
        self.current_iters = -1
        self._layer_name = layer.full_name()
        self.output_grad = None

    def forward(self, input):
        self.current_iters += 1
        quant_input = input
        quant_weight = self.weight

        if self.current_iters == 0:
            self.activation_quanter.mode = 'quant_prepare'
            self.weight_quanter.mode = 'quant_prepare'

        if self.current_iters == 1:
            print("********Current Layer:", self._layer_name)
            if self.output_grad is not None:
                self._search_hessian(input, self.weight)

        quant_input = self.activation_quanter(input)
        quant_weight = self.weight_quanter(self.weight)

        output = self._linear_forward(quant_input, quant_weight)

        if output.stop_gradient is False:
            self.output_hook = output.register_hook(self._grad_hook)

        return output

    def _search_hessian(self, input, weight):
        self.original_output = self._linear_forward(input, weight)
        print("======Begin searching hessian======")
        rounds = 3
        print('input:', input.max(), input.min())
        if input.sum() == 0.0:
            return
        for r in range(rounds):
            print("Round:", r)
            self.activation_quanter.mode = 'quant_forward'
            qdq_input = self.activation_quanter(input)
            self._search_w(qdq_input, weight)

            self.weight_quanter.mode = 'quant_forward'
            qdq_weight = self.weight_quanter(weight)
            self._search_a(input, qdq_weight)

        self.output_hook.remove()
        del self.output_grad, self.original_output
        self.output_grad = None
        #return original_output

    def _search_a(self, input, weight):
        print("******Begin search act")
        best_sim = paddle.to_tensor(float('-inf'))
        best_scales = 0
        self.activation_quanter.mode = 'quant_search'
        for i in range(len(self.activation_quanter._interval)):
            if i == 0:
                print("abs max:", float(input.abs().max()))
            if i % 10 == 0:
                print("Current Iter:", i)
            quant_input = self.activation_quanter(input, i)
            with paddle.no_grad():
                qdq_output = self._linear_forward(quant_input, weight)
            cur_sim = self._cal_similarity(self.original_output, qdq_output)
            if cur_sim > best_sim:
                print(
                    f"Iters:{i}, scales change [{float(best_scales)}] to [{float(self.activation_quanter.scales())}]"
                )
                best_sim = cur_sim
                best_scales = self.activation_quanter.scales()

        print("======Act Best_scales:", float(best_scales))
        self.activation_quanter._scale = best_scales
        self.activation_quanter.mode = 'quant_forward'

    def _search_w(self, input, weight):
        print("******Begin search weight")
        print("act scale:", self.activation_quanter._scale)
        best_sim = paddle.to_tensor(float('-inf'))
        best_scales = 0
        self.weight_quanter.mode = 'quant_search'
        for i in range(35, len(self.weight_quanter._interval)):
            if i == 0:
                print("weight max:", float(weight.abs().max()))
            if i % 10 == 0:
                print("Current Iter:", i)
            quant_weight = self.weight_quanter(weight, i)
            with paddle.no_grad():
                qdq_output = self._linear_forward(input, quant_weight)
            cur_sim = self._cal_similarity(self.original_output, qdq_output)
            if cur_sim > best_sim:
                print(f"Iters:{i}, scales changed")

                best_sim = cur_sim
                best_scales = self.weight_quanter.scales()

        print("======Weight Best_scales:", best_scales.shape)
        self.weight_quanter._scale = best_scales
        self.weight_quanter.mode = 'quant_forward'

    def _cal_similarity(self, ori_output, qdq_output):
        grad = self.output_grad.reshape(ori_output.shape)

        similarity = -(grad * (ori_output - qdq_output))**2
        #similarity = -((ori_output - qdq_output))**2  # mse
        return similarity.mean()

    def _grad_hook(self, grad):
        self.output_grad = grad
        #print('grad:', grad.max(), grad.min())

    def _linear_forward(self, input, weight):
        out = F.linear(x=input, weight=weight, bias=self.bias, name=self.name)
        return out

    def weights_to_quanters(self):
        return [('weight', 'weight_quanter')]

    def activation_quanters(self):
        return ['activation_quanter']
