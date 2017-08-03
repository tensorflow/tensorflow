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
"""Classification metrics library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

# TODO(nsilberman): move into metrics/python/ops/


def accuracy(predictions, labels, weights=None, name=None):
  """Computes the percentage of times that predictions matches labels.

  Args:
    predictions: the predicted values, a `Tensor` whose dtype and shape
                 matches 'labels'.
    labels: the ground truth values, a `Tensor` of any shape and
            bool, integer, or string dtype.
    weights: None or `Tensor` of float values to reweight the accuracy.
    name: A name for the operation (optional).

  Returns:
    Accuracy `Tensor`.

  Raises:
    ValueError: if dtypes don't match or
                if dtype is not bool, integer, or string.
  """
  if not (labels.dtype.is_integer or
          labels.dtype in (dtypes.bool, dtypes.string)):
    raise ValueError(
        'Labels should have bool, integer, or string dtype, not %r' %
        labels.dtype)
  if not labels.dtype.is_compatible_with(predictions.dtype):
    raise ValueError('Dtypes of predictions and labels should match. '
                     'Given: predictions (%r) and labels (%r)' %
                     (predictions.dtype, labels.dtype))
  with ops.name_scope(name, 'accuracy', values=[predictions, labels]):
    is_correct = math_ops.cast(
        math_ops.equal(predictions, labels), dtypes.float32)
    if weights is not None:
      is_correct = math_ops.multiply(is_correct, weights)
      num_values = math_ops.multiply(weights, array_ops.ones_like(is_correct))
      return math_ops.div(math_ops.reduce_sum(is_correct),
                          math_ops.reduce_sum(num_values))
    return math_ops.reduce_mean(is_correct)
