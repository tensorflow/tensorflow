# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""A model that places a hard decision tree embedding before a neural net."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.tensor_forest.hybrid.python import hybrid_model
from tensorflow.contrib.tensor_forest.hybrid.python.layers import decisions_to_data
from tensorflow.contrib.tensor_forest.hybrid.python.layers import fully_connected
from tensorflow.python.ops import nn_ops
from tensorflow.python.training import adagrad


class HardDecisionsToDataThenNN(hybrid_model.HybridModel):
  """A model that treats tree inference as hard at test."""

  def __init__(self,
               params,
               device_assigner=None,
               optimizer_class=adagrad.AdagradOptimizer,
               **kwargs):

    super(HardDecisionsToDataThenNN, self).__init__(
        params,
        device_assigner=device_assigner,
        optimizer_class=optimizer_class,
        **kwargs)

    self.layers = [decisions_to_data.HardDecisionsToDataLayer(
        params, 0, device_assigner),
                   fully_connected.FullyConnectedLayer(
                       params, 1, device_assigner=device_assigner)]

  def _base_inference(self, data, data_spec=None, soft=False):
    if soft:
      inference_result = self.layers[0].soft_inference_graph(data)
    else:
      inference_result = self._do_layer_inference(self.layers[0], data)

    for layer in self.layers[1:]:
      inference_result = self._do_layer_inference(layer, inference_result)

    output_size = 1 if self.is_regression else self.params.num_classes
    output = layers.fully_connected(
        inference_result, output_size, activation_fn=nn_ops.softmax)
    return output

  def inference_graph(self, data, data_spec=None):
    """Returns the op that performs inference on a batch of data."""

    return nn_ops.softmax(
        self._base_inference(
            data, data_spec=data_spec, soft=True))

  # pylint: disable=unused-argument
  def training_inference_graph(self, data, data_spec=None):
    return self._base_inference(data, data_spec=data_spec, soft=False)
