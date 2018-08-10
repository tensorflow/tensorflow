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

"""TensorFlow Ops for loss computation (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework import deprecated
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops as array_ops_
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses


@deprecated('2016-12-01', 'Use `tf.losses.mean_squared_error` '
            'and explicit logits computation.')
def mean_squared_error_regressor(tensor_in, labels, weights, biases, name=None):
  """Returns prediction and loss for mean squared error regression."""
  with ops.name_scope(name, 'mean_squared_error_regressor',
                      [tensor_in, labels]):
    predictions = nn.xw_plus_b(tensor_in, weights, biases)
    if len(labels.get_shape()) == 1 and len(predictions.get_shape()) == 2:
      predictions = array_ops_.squeeze(predictions, axis=[1])
    return predictions, losses.mean_squared_error(labels, predictions)


@deprecated('2016-12-01', 'Use `tf.losses.softmax_cross_entropy` '
            'and explicit logits computation.')
def softmax_classifier(tensor_in,
                       labels,
                       weights,
                       biases,
                       class_weight=None,
                       name=None):
  """Returns prediction and loss for softmax classifier.

  This function returns "probabilities" and a cross entropy loss. To obtain
  predictions, use `tf.argmax` on the returned probabilities.

  This function requires labels to be passed in one-hot encoding.

  Args:
    tensor_in: Input tensor, [batch_size, feature_size], features.
    labels: Tensor, [batch_size, n_classes], one-hot labels of the output
      classes.
    weights: Tensor, [batch_size, feature_size], linear transformation
      matrix.
    biases: Tensor, [batch_size], biases.
    class_weight: Tensor, optional, [n_classes], weight for each class.
      If not given, all classes are supposed to have weight one.
    name: Operation name.

  Returns:
    `tuple` of softmax predictions and loss `Tensor`s.
  """
  with ops.name_scope(name, 'softmax_classifier', [tensor_in, labels]):
    logits = nn.xw_plus_b(tensor_in, weights, biases)
    if class_weight is not None:
      logits = math_ops.multiply(logits, class_weight)
    return nn.softmax(logits), losses.softmax_cross_entropy(labels, logits)
