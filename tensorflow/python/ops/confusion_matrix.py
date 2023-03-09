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
"""Confusion matrix related utilities."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


def remove_squeezable_dimensions(
    labels, predictions, expected_rank_diff=0, name=None):
  """Squeeze last dim if ranks differ from expected by exactly 1.

  In the common case where we expect shapes to match, `expected_rank_diff`
  defaults to 0, and we squeeze the last dimension of the larger rank if they
  differ by 1.

  But, for example, if `labels` contains class IDs and `predictions` contains 1
  probability per class, we expect `predictions` to have 1 more dimension than
  `labels`, so `expected_rank_diff` would be 1. In this case, we'd squeeze
  `labels` if `rank(predictions) - rank(labels) == 0`, and
  `predictions` if `rank(predictions) - rank(labels) == 2`.

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    labels: Label values, a `Tensor` whose dimensions match `predictions`.
    predictions: Predicted values, a `Tensor` of arbitrary dimensions.
    expected_rank_diff: Expected result of `rank(predictions) - rank(labels)`.
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
      if (rank_diff == expected_rank_diff + 1 and
          predictions_shape.dims[-1].is_compatible_with(1)):
        predictions = array_ops.squeeze(predictions, [-1])
      elif (rank_diff == expected_rank_diff - 1 and
            labels_shape.dims[-1].is_compatible_with(1)):
        labels = array_ops.squeeze(labels, [-1])
      return labels, predictions

    # Use dynamic rank.
    rank_diff = array_ops.rank(predictions) - array_ops.rank(labels)
    if (predictions_rank is None) or (
        predictions_shape.dims[-1].is_compatible_with(1)):
      predictions = control_flow_ops.cond(
          math_ops.equal(expected_rank_diff + 1, rank_diff),
          lambda: array_ops.squeeze(predictions, [-1]),
          lambda: predictions)
    if (labels_rank is None) or (
        labels_shape.dims[-1].is_compatible_with(1)):
      labels = control_flow_ops.cond(
          math_ops.equal(expected_rank_diff - 1, rank_diff),
          lambda: array_ops.squeeze(labels, [-1]),
          lambda: labels)
    return labels, predictions


@tf_export('math.confusion_matrix', v1=[])
@dispatch.add_dispatch_support
def confusion_matrix(labels,
                     predictions,
                     num_classes=None,
                     weights=None,
                     dtype=dtypes.int32,
                     name=None):
  """Computes the confusion matrix from predictions and labels.

  The matrix columns represent the prediction labels and the rows represent the
  real labels. The confusion matrix is always a 2-D array of shape `[n, n]`,
  where `n` is the number of valid labels for a given classification task. Both
  prediction and labels must be 1-D arrays of the same shape in order for this
  function to work.

  If `num_classes` is `None`, then `num_classes` will be set to one plus the
  maximum value in either predictions or labels. Class labels are expected to
  start at 0. For example, if `num_classes` is 3, then the possible labels
  would be `[0, 1, 2]`.

  If `weights` is not `None`, then each prediction contributes its
  corresponding weight to the total value of the confusion matrix cell.

  For example:

  ```python
    tf.math.confusion_matrix([1, 2, 4], [2, 2, 4]) ==>
        [[0 0 0 0 0]
         [0 0 1 0 0]
         [0 0 1 0 0]
         [0 0 0 0 0]
         [0 0 0 0 1]]
  ```

  Note that the possible labels are assumed to be `[0, 1, 2, 3, 4]`,
  resulting in a 5x5 confusion matrix.

  Args:
    labels: 1-D `Tensor` of real labels for the classification task.
    predictions: 1-D `Tensor` of predictions for a given classification.
    num_classes: The possible number of labels the classification task can
                 have. If this value is not provided, it will be calculated
                 using both predictions and labels array.
    weights: An optional `Tensor` whose shape matches `predictions`.
    dtype: Data type of the confusion matrix.
    name: Scope name.

  Returns:
    A `Tensor` of type `dtype` with shape `[n, n]` representing the confusion
    matrix, where `n` is the number of possible labels in the classification
    task.

  Raises:
    ValueError: If both predictions and labels are not 1-D vectors and have
      mismatched shapes, or if `weights` is not `None` and its shape doesn't
      match `predictions`.
  """
  with ops.name_scope(name, 'confusion_matrix',
                      (predictions, labels, num_classes, weights)) as name:
    labels, predictions = remove_squeezable_dimensions(
        ops.convert_to_tensor(labels, name='labels'),
        ops.convert_to_tensor(
            predictions, name='predictions'))
    predictions = math_ops.cast(predictions, dtypes.int64)
    labels = math_ops.cast(labels, dtypes.int64)

    # Sanity checks - underflow or overflow can cause memory corruption.
    labels = control_flow_ops.with_dependencies(
        [check_ops.assert_non_negative(
            labels, message='`labels` contains negative values')],
        labels)
    predictions = control_flow_ops.with_dependencies(
        [check_ops.assert_non_negative(
            predictions, message='`predictions` contains negative values')],
        predictions)

    if num_classes is None:
      num_classes = math_ops.maximum(math_ops.reduce_max(predictions),
                                     math_ops.reduce_max(labels)) + 1
    else:
      num_classes_int64 = math_ops.cast(num_classes, dtypes.int64)
      labels = control_flow_ops.with_dependencies(
          [check_ops.assert_less(
              labels, num_classes_int64, message='`labels` out of bound')],
          labels)
      predictions = control_flow_ops.with_dependencies(
          [check_ops.assert_less(
              predictions, num_classes_int64,
              message='`predictions` out of bound')],
          predictions)

    if weights is not None:
      weights = ops.convert_to_tensor(weights, name='weights')
      predictions.get_shape().assert_is_compatible_with(weights.get_shape())
      weights = math_ops.cast(weights, dtype)

    shape = array_ops_stack.stack([num_classes, num_classes])
    indices = array_ops_stack.stack([labels, predictions], axis=1)
    values = (array_ops.ones_like(predictions, dtype)
              if weights is None else weights)
    return array_ops.scatter_nd(
        indices=indices,
        updates=values,
        shape=math_ops.cast(shape, dtypes.int64))


@tf_export(v1=['math.confusion_matrix', 'confusion_matrix'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('confusion_matrix', 'train.confusion_matrix')
def confusion_matrix_v1(labels,
                        predictions,
                        num_classes=None,
                        dtype=dtypes.int32,
                        name=None,
                        weights=None):
  """Computes the confusion matrix from predictions and labels.

  The matrix columns represent the prediction labels and the rows represent the
  real labels. The confusion matrix is always a 2-D array of shape `[n, n]`,
  where `n` is the number of valid labels for a given classification task. Both
  prediction and labels must be 1-D arrays of the same shape in order for this
  function to work.

  If `num_classes` is `None`, then `num_classes` will be set to one plus the
  maximum value in either predictions or labels. Class labels are expected to
  start at 0. For example, if `num_classes` is 3, then the possible labels
  would be `[0, 1, 2]`.

  If `weights` is not `None`, then each prediction contributes its
  corresponding weight to the total value of the confusion matrix cell.

  For example:

  ```python
    tf.math.confusion_matrix([1, 2, 4], [2, 2, 4]) ==>
        [[0 0 0 0 0]
         [0 0 1 0 0]
         [0 0 1 0 0]
         [0 0 0 0 0]
         [0 0 0 0 1]]
  ```

  Note that the possible labels are assumed to be `[0, 1, 2, 3, 4]`,
  resulting in a 5x5 confusion matrix.

  Args:
    labels: 1-D `Tensor` of real labels for the classification task.
    predictions: 1-D `Tensor` of predictions for a given classification.
    num_classes: The possible number of labels the classification task can have.
      If this value is not provided, it will be calculated using both
      predictions and labels array.
    dtype: Data type of the confusion matrix.
    name: Scope name.
    weights: An optional `Tensor` whose shape matches `predictions`.

  Returns:
    A `Tensor` of type `dtype` with shape `[n, n]` representing the confusion
    matrix, where `n` is the number of possible labels in the classification
    task.

  Raises:
    ValueError: If both predictions and labels are not 1-D vectors and have
      mismatched shapes, or if `weights` is not `None` and its shape doesn't
      match `predictions`.
  """
  return confusion_matrix(labels, predictions, num_classes, weights, dtype,
                          name)
