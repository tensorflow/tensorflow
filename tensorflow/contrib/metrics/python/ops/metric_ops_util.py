# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions for metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


def remove_squeezable_dimensions(predictions, labels):
  """Squeeze last dim if ranks of `predictions` and `labels` differ by 1.

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    predictions: Predicted values, a `Tensor` of arbitrary dimensions.
    labels: Label values, a `Tensor` whose dimensions match `predictions`.

  Returns:
    Tuple of `predictions` and `labels`, possibly with last dim squeezed.
  """
  predictions = ops.convert_to_tensor(predictions)
  labels = ops.convert_to_tensor(labels)
  predictions_shape = predictions.get_shape()
  predictions_rank = predictions_shape.ndims
  labels_shape = labels.get_shape()
  labels_rank = labels_shape.ndims
  if (labels_rank is not None) and (predictions_rank is not None):
    # Use static rank.
    rank_diff = predictions_rank - labels_rank
    if rank_diff == -1:
      labels = array_ops.squeeze(labels, [-1])
    elif rank_diff == 1:
      predictions = array_ops.squeeze(predictions, [-1])
    return predictions, labels

  # Use dynamic rank.
  rank_diff = array_ops.rank(predictions) - array_ops.rank(labels)
  if (predictions_rank is None) or (
      predictions_shape.dims[-1].is_compatible_with(1)):
    predictions = control_flow_ops.cond(
        math_ops.equal(1, rank_diff),
        lambda: array_ops.squeeze(predictions, [-1]),
        lambda: predictions)
  if (labels_rank is None) or (
      labels_shape.dims[-1].is_compatible_with(1)):
    labels = control_flow_ops.cond(
        math_ops.equal(-1, rank_diff),
        lambda: array_ops.squeeze(labels, [-1]),
        lambda: labels)
  return predictions, labels
