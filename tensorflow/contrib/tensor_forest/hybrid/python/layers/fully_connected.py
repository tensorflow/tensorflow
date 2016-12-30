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
"""Neural network components for hybrid models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.tensor_forest.hybrid.python import hybrid_layer

from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops


class FullyConnectedLayer(hybrid_layer.HybridLayer):
  """A stacked, fully-connected feed-forward neural network layer."""

  def _define_vars(self, params):
    pass

  def inference_graph(self, data):
    with ops.device(self.device_assigner.get_device(self.layer_num)):
      # Compute activations for the neural network.
      nn_activations = layers.fully_connected(data, self.params.layer_size)

      for _ in range(1, self.params.num_layers):
        # pylint: disable=W0106
        nn_activations = layers.fully_connected(nn_activations,
                                                self.params.layer_size)
      return nn_activations


class ManyToOneLayer(hybrid_layer.HybridLayer):

  def _define_vars(self, params):
    pass

  def inference_graph(self, data):
    with ops.device(self.device_assigner.get_device(self.layer_num)):
      # Compute activations for the neural network.
      nn_activations = layers.fully_connected(data, 1)

      # There is always one activation per instance by definition, so squeeze
      # away the extra dimension.
      return array_ops.squeeze(nn_activations, squeeze_dims=[1])


class FlattenedFullyConnectedLayer(hybrid_layer.HybridLayer):
  """A stacked, fully-connected flattened feed-forward neural network layer."""

  def _define_vars(self, params):
    pass

  def inference_graph(self, data):
    with ops.device(self.device_assigner.get_device(self.layer_num)):
      # Compute activations for the neural network.
      nn_activations = [layers.fully_connected(data, self.params.layer_size)]

      for _ in range(1, self.params.num_layers):
        # pylint: disable=W0106
        nn_activations.append(
            layers.fully_connected(
                nn_activations[-1],
                self.params.layer_size))

      nn_activations_tensor = array_ops.concat_v2(
          nn_activations, 1, name="flattened_nn_activations")

      return nn_activations_tensor
