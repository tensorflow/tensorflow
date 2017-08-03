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
"""Defines the model abstraction for hybrid models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import variables as framework_variables

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables

from tensorflow.python.training import adagrad


class HybridModel(object):
  """Defines a hybrid model.

  Models chain together the results of inference layers and provide training
  capabilities.
  """

  # pylint: disable=unused-argument
  def __init__(self,
               params,
               device_assigner=None,
               optimizer_class=adagrad.AdagradOptimizer,
               **kwargs):

    self.device_assigner = (
        device_assigner or framework_variables.VariableDeviceChooser())

    self.params = params

    self.optimizer = optimizer_class(self.params.learning_rate)

    self.is_regression = params.regression

    self.regularizer = None
    if params.regularization == "l1":
      self.regularizer = layers.l1_regularizer(
          self.params.regularization_strength)
    elif params.regularization == "l2":
      self.regularizer = layers.l2_regularizer(
          self.params.regularization_strength)

  def _do_layer_inference(self, layer, data):

    # If this is a collection of layers, return the mean of their inference
    # results.
    if isinstance(layer, collections.Iterable):
      return math_ops.reduce_mean(
          array_ops.stack([l.inference_graph(data) for l in layer]), 0)
    # If this is a single layer, return its inference result.
    else:
      return layer.inference_graph(data)

  def _base_inference(self, data, data_spec=None):
    """Returns an op that performs inference without a softmax."""
    inference_result = self._do_layer_inference(self.layers[0], data)

    for layer in self.layers[1:]:
      inference_result = self._do_layer_inference(layer, inference_result)

    output_size = 1 if self.is_regression else self.params.num_classes
    output = layers.fully_connected(
        inference_result, output_size, activation_fn=array_ops.identity)

    return output

  def inference_graph(self, data, data_spec=None):
    """Returns the op that performs inference on a batch of data."""

    return nn_ops.softmax(self._base_inference(data, data_spec=data_spec))

  def training_inference_graph(self, data, data_spec=None):
    """Returns an inference-without-softmax op for training purposes."""

    return self._base_inference(data, data_spec=data_spec)

  def predict_proba(self, data, data_spec=None):
    inference_result = self.inference_graph(data, data_spec=data_spec)

    probabilities = nn_ops.softmax(inference_result, name="probabilities")

    return probabilities

  def training_graph(self, data, labels, data_spec=None, epoch=None):
    """Returns the op that trains the hybrid model."""
    return self.optimizer.minimize(self.training_loss(data, labels))

  def loss(self, data, labels):
    """The loss to minimize while training."""

    if self.is_regression:
      diff = self.training_inference_graph(data) - math_ops.to_float(labels)
      mean_squared_error = math_ops.reduce_mean(diff * diff)
      root_mean_squared_error = math_ops.sqrt(mean_squared_error, name="loss")
      loss = root_mean_squared_error
    else:
      loss = math_ops.reduce_mean(
          nn_ops.sparse_softmax_cross_entropy_with_logits(
              labels=array_ops.squeeze(math_ops.to_int32(labels)),
              logits=self.training_inference_graph(data)),
          name="loss")
    if self.regularizer:
      loss += layers.apply_regularization(self.regularizer,
                                          variables.trainable_variables())
    return loss

  def training_loss(self, data, labels):
    return self.loss(data, labels)

  def validation_loss(self, data, labels):
    return self.loss(data, labels)
