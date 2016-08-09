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

"""TensorFlow Ops for loss computation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.losses.python.losses import loss_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops as array_ops_
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


def mean_squared_error_regressor(tensor_in, labels, weights, biases, name=None):
  """Returns prediction and loss for mean squared error regression."""
  with ops.name_scope(name, "mean_squared_error_regressor",
                      [tensor_in, labels]):
    predictions = nn.xw_plus_b(tensor_in, weights, biases)
    if len(labels.get_shape()) == 1 and len(predictions.get_shape()) == 2:
      predictions = array_ops_.squeeze(predictions, squeeze_dims=[1])
    return predictions, loss_ops.sum_of_squares(predictions, labels)


def softmax_classifier(tensor_in,
                       labels,
                       weights,
                       biases,
                       class_weight=None,
                       name=None):
  """Returns prediction and loss for softmax classifier.

  Args:
    tensor_in: Input tensor, [batch_size, feature_size], features.
    labels: Tensor, [batch_size, n_classes], labels of the output classes.
    weights: Tensor, [batch_size, feature_size], linear transformation
      matrix.
    biases: Tensor, [batch_size], biases.
    class_weight: Tensor, optional, [n_classes], weight for each class.
      If not given, all classes are supposed to have weight one.
    name: Operation name.

  Returns:
    Prediction and loss tensors.
  """
  with ops.name_scope(name, "softmax_classifier", [tensor_in, labels]):
    logits = nn.xw_plus_b(tensor_in, weights, biases)
    if class_weight is not None:
      logits = math_ops.mul(logits, class_weight)
    return nn.softmax(logits), loss_ops.softmax_cross_entropy(logits, labels)
