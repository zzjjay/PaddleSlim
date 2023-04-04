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
import time
import logging
import numpy as np
import tqdm
import paddle
from paddle.quantization import PTQ
from paddle.quantization import QuantConfig
from paddle.quantization.config import DEFAULT_QAT_LAYER_MAPPINGS
from .adaround_act import AdaroundActObserver
from .adaround_weight import AdaroundWeightObserver, AdaroundWeightObserverLayer
from ...common import get_logger

_logger = get_logger(
    __name__,
    logging.INFO,
    fmt='%(asctime)s-%(levelname)s: %(message)s', )

GAMMA = -0.1
ZETA = 1.1


class Adaround:
    def __init__(self,
                 model,
                 quant_config,
                 data_loader,
                 epochs=10,
                 batch_nums=10,
                 lr=0.1):
        self.origin_model = model
        self.quant_config = quant_config
        self._data_loader = data_loader
        self._batch_nums = batch_nums
        self._epochs = epochs
        self._lr = lr

        self.all_quant_layer_outputs = {}
        self.all_origin_layer_outputs = {}

    def init_ptq(self, inplace=False):
        """ Initialize PTQ.
        """
        self.ptq = PTQ(self.quant_config)
        self.quant_model = self.ptq.quantize(self.origin_model, inplace=inplace)

        # apply hook
        for layer in self.quant_model.sublayers():
            if isinstance(layer, tuple(DEFAULT_QAT_LAYER_MAPPINGS.items())):
                layer.register_forward_hook(self._quant_forward_post_hook)
        for layer in self.origin_model.sublayers():
            if isinstance(layer, tuple(DEFAULT_QAT_LAYER_MAPPINGS.keys())):
                layer.register_forward_hook(self._origin_forward_post_hook)

    def _quant_forward_post_hook(self, layer, inputs, outputs):
        weight_name = layer.weight.name.split("_deepcopy")[0]
        self.all_quant_layer_outputs[weight_name](outputs)
        return outputs

    def _origin_forward_post_hook(self, layer, inputs, outputs):
        weight_name = layer.weight.name
        self.all_origin_layer_outputs[weight_name] = outputs
        return outputs

    def run(self):
        self.model.eval()
        self.quant_model.eval()
        with tqdm(
                total=self._batch_nums,
                bar_format=
                'Sampling stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
                ncols=80, ) as t:
            for batch_id, data in enumerate(self._data_loader()):
                # data (dict)
                self.quant_model(**data)
                t.update()
                if batch_id + 1 == self._batch_nums:
                    break

        _logger.info("Begin updating quant model")
        for layer in self.quant_model.sublayers():
            if isinstance(layer, AdaroundWeightObserverLayer):
                weight_name = layer.alpha.split('.adaround_alpha')
                _logger.info(
                    f'Current layer: {layer.full_name()}, weight: {weight_name}'
                )

                opt = paddle.optimizer.Adam(
                    learning_rate=self._lr, parameters=[layer.alpha])
                for epoch in range(self._epochs):
                    for batch_id, data in enumerate(self._data_loader()):
                        # data (dict)
                        start_time = time.time()
                        self.origin_model(**data)
                        self.quant_model(**data)
                        h_v = layer.compute_soft_rounding()
                        round_loss = self._round_loss(h_v)
                        recon_loss = self._recon_loss(weight_name)
                        total_loss = round_loss + recon_loss
                        total_loss.backward()
                        opt.step()
                        opt.clear_grad()
                        cur_time = time.time()

                        _logger.info(
                            "Epoch {:d}, Iter {:d}, lr {}, total_loss {:.5f}, recon_loss {:.5f}, round_loss {:.5f}, time {:.5f}s"
                            .format(epoch, batch_id, self._lr,
                                    total_loss.numpy(),
                                    recon_loss.numpy(),
                                    round_loss.numpy(),
                                    cur_time - start_time), )

                        if batch_id + 1 == self._batch_nums:
                            break

    def _round_loss(self, h_v):
        return paddle.sum(-paddle.pow(paddle.abs(2 * h_v - 1), 3) + 1)

    def _recon_loss(self, weight_name):
        return paddle.nn.functional.mse_loss(
            self.all_quant_layer_outputs[weight_name],
            self.all_origin_layer_outputs[weight_name], )
