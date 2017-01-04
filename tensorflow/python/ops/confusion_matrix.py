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
"""Confusion matrix related utilities.


@@remove_squeezable_dimensions
@@confusion_matrix
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops


def remove_squeezable_dimensions(labels, predictions, name=None):
  """Squeeze last dim if ranks of `predictions` and `labels` differ by 1.

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    labels: Label values, a `Tensor` whose dimensions match `predictions`.
    predictions: Predicted values, a `Tensor` of arbitrary dimensions.
    name: Name of the op.

  Returns:
    Tuple of `labels` and `predictions`, possibly with last dim squeezed.
  """
  with ops.name_scope(name, 'remove_squeezable_dimensions',
                      [labels, predictions]):
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
      return labels, predictions

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
    return labels, predictions


def confusion_matrix(labels, predictions, num_classes=None, dtype=dtypes.int32,
                     name=None, weights=None):
  """Computes the confusion matrix from predictions and labels.

  Calculate the Confusion Matrix for a pair of prediction and
  label 1-D int arrays.

  The matrix rows represent the prediction labels and the columns
  represents the real labels. The confusion matrix is always a 2-D array
  of shape `[n, n]`, where `n` is the number of valid labels for a given
  classification task. Both prediction and labels must be 1-D arrays of
  the same shape in order for this function to work.

  If `num_classes` is None, then `num_classes` will be set to the one plus
  the maximum value in either predictions or labels.
  Class labels are expected to start at 0. E.g., if `num_classes` was
  three, then the possible labels would be `[0, 1, 2]`.

  If `weights` is not `None`, then each prediction contributes its
  corresponding weight to the total value of the confusion matrix cell.

  For example:

  ```python
    tf.contrib.metrics.confusion_matrix([1, 2, 4], [2, 2, 4]) ==>
        [[0 0 0 0 0]
         [0 0 1 0 0]
         [0 0 1 0 0]
         [0 0 0 0 0]
         [0 0 0 0 1]]
  ```

  Note that the possible labels are assumed to be `[0, 1, 2, 3, 4]`,
  resulting in a 5x5 confusion matrix.

  Args:
    labels: A 1-D representing the real labels for the classification task.
    predictions: A 1-D array representing the predictions for a given
                 classification.
    num_classes: The possible number of labels the classification task can
                 have. If this value is not provided, it will be calculated
                 using both predictions and labels array.
    dtype: Data type of the confusion matrix.
    name: Scope name.
    weights: An optional `Tensor` whose shape matches `predictions`.

  Returns:
    A k X k matrix representing the confusion matrix, where k is the number of
    possible labels in the classification task.

  Raises:
    ValueError: If both predictions and labels are not 1-D vectors and have
      mismatched shapes, or if `weights` is not `None` and its shape doesn't
      match `predictions`.
  """
  with ops.name_scope(name, 'confusion_matrix',
                      [predictions, labels, num_classes]) as name:
    labels, predictions = remove_squeezable_dimensions(
        ops.convert_to_tensor(labels, name='labels'),
        ops.convert_to_tensor(
            predictions, name='predictions'))
    predictions = math_ops.cast(predictions, dtypes.int64)
    labels = math_ops.cast(labels, dtypes.int64)

    if num_classes is None:
      num_classes = math_ops.maximum(math_ops.reduce_max(predictions),
                                     math_ops.reduce_max(labels)) + 1

    if weights is not None:
      predictions.get_shape().assert_is_compatible_with(weights.get_shape())
      weights = math_ops.cast(weights, dtype)

    shape = array_ops.stack([num_classes, num_classes])
    indices = array_ops.transpose(array_ops.stack([predictions, labels]))
    values = (array_ops.ones_like(predictions, dtype)
              if weights is None else weights)
    cm_sparse = sparse_tensor.SparseTensor(
        indices=indices, values=values, dense_shape=math_ops.to_int64(shape))
    zero_matrix = array_ops.zeros(math_ops.to_int32(shape), dtype)

    return sparse_ops.sparse_add(zero_matrix, cm_sparse)
