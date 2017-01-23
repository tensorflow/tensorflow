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
"""Contains metric-computing operations on streamed tensors.

Module documentation, including "@@" callouts, should be put in
third_party/tensorflow/contrib/metrics/__init__.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework import tensor_util
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.metrics.python.ops import set_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables


def _safe_div(numerator, denominator, name):
  """Divides two values, returning 0 if the denominator is <= 0.

  Args:
    numerator: A real `Tensor`.
    denominator: A real `Tensor`, with dtype matching `numerator`.
    name: Name for the returned op.

  Returns:
    0 if `denominator` <= 0, else `numerator` / `denominator`
  """
  return array_ops.where(
      math_ops.greater(denominator, 0),
      math_ops.truediv(numerator, denominator),
      0,
      name=name)


def _safe_scalar_div(numerator, denominator, name):
  """Divides two values, returning 0 if the denominator is 0.

  Args:
    numerator: A scalar `float64` `Tensor`.
    denominator: A scalar `float64` `Tensor`.
    name: Name for the returned op.

  Returns:
    0 if `denominator` == 0, else `numerator` / `denominator`
  """
  numerator.get_shape().with_rank_at_most(1)
  denominator.get_shape().with_rank_at_most(1)
  return control_flow_ops.cond(
      math_ops.equal(
          array_ops.constant(0.0, dtype=dtypes.float64), denominator),
      lambda: array_ops.constant(0.0, dtype=dtypes.float64),
      lambda: math_ops.div(numerator, denominator),
      name=name)


def _create_local(name, shape, collections=None, validate_shape=True,
                  dtype=dtypes.float32):
  """Creates a new local variable.

  Args:
    name: The name of the new or existing variable.
    shape: Shape of the new or existing variable.
    collections: A list of collection names to which the Variable will be added.
    validate_shape: Whether to validate the shape of the variable.
    dtype: Data type of the variables.

  Returns:
    The created variable.
  """
  # Make sure local variables are added to tf.GraphKeys.LOCAL_VARIABLES
  collections = list(collections or [])
  collections += [ops.GraphKeys.LOCAL_VARIABLES]
  return variables.Variable(
      initial_value=array_ops.zeros(shape, dtype=dtype),
      name=name,
      trainable=False,
      collections=collections,
      validate_shape=validate_shape)


# TODO(ptucker): Move this somewhere common, to share with ops/losses/losses.py.
def _assert_weights_rank(weights, values):
  """`weights` rank must be either `0`, or the same as 'values'."""
  return check_ops.assert_rank_in(weights, (0, array_ops.rank(values)))


def _count_condition(values, weights=None, metrics_collections=None,
                     updates_collections=None):
  """Sums the weights of cases where the given values are True.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    values: A `bool` `Tensor` of arbitrary size.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `values`, and must be broadcastable to `values` (i.e., all dimensions
      must be either `1`, or the same as the corresponding `values`
      dimension).
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.

  Returns:
    value_tensor: A `Tensor` representing the current value of the metric.
    update_op: An operation that accumulates the error from a batch of data.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`,
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
  """
  check_ops.assert_type(values, dtypes.bool)
  count = _create_local('count', shape=[])

  values = math_ops.to_float(values)
  if weights is not None:
    weights = math_ops.to_float(weights)
    with ops.control_dependencies((_assert_weights_rank(weights, values),)):
      values = math_ops.multiply(values, weights)

  value_tensor = array_ops.identity(count)
  update_op = state_ops.assign_add(count, math_ops.reduce_sum(values))

  if metrics_collections:
    ops.add_to_collections(metrics_collections, value_tensor)

  if updates_collections:
    ops.add_to_collections(updates_collections, update_op)

  return value_tensor, update_op


def streaming_true_positives(predictions, labels, weights=None,
                             metrics_collections=None,
                             updates_collections=None,
                             name=None):
  """Sum the weights of true_positives.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will
      be cast to `bool`.
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions
      must be either `1`, or the same as the corresponding `labels`
      dimension).
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    value_tensor: A `Tensor` representing the current value of the metric.
    update_op: An operation that accumulates the error from a batch of data.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  return metrics.true_positives(
      predictions=predictions, labels=labels, weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def streaming_true_negatives(predictions, labels, weights=None,
                             metrics_collections=None,
                             updates_collections=None,
                             name=None):
  """Sum the weights of true_negatives.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will
      be cast to `bool`.
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions
      must be either `1`, or the same as the corresponding `labels`
      dimension).
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    value_tensor: A `Tensor` representing the current value of the metric.
    update_op: An operation that accumulates the error from a batch of data.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  with variable_scope.variable_scope(
      name, 'true_negatives', (predictions, labels, weights)):

    predictions = math_ops.cast(predictions, dtype=dtypes.bool)
    labels = math_ops.cast(labels, dtype=dtypes.bool)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    is_true_negative = math_ops.logical_and(math_ops.equal(labels, False),
                                            math_ops.equal(predictions, False))
    return _count_condition(is_true_negative, weights, metrics_collections,
                            updates_collections)


def streaming_false_positives(predictions, labels, weights=None,
                              metrics_collections=None,
                              updates_collections=None,
                              name=None):
  """Sum the weights of false positives.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will
      be cast to `bool`.
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions
      must be either `1`, or the same as the corresponding `labels`
      dimension).
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    value_tensor: A `Tensor` representing the current value of the metric.
    update_op: An operation that accumulates the error from a batch of data.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  return metrics.false_positives(
      predictions=predictions, labels=labels, weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def streaming_false_negatives(predictions, labels, weights=None,
                              metrics_collections=None,
                              updates_collections=None,
                              name=None):
  """Computes the total number of false positives.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will
      be cast to `bool`.
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions
      must be either `1`, or the same as the corresponding `labels`
      dimension).
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    value_tensor: A `Tensor` representing the current value of the metric.
    update_op: An operation that accumulates the error from a batch of data.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`,
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
  """
  return metrics.false_negatives(
      predictions=predictions, labels=labels, weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


# TODO(ptucker): Move this somewhere common, to share with ops/losses/losses.py.
def _broadcast_weights(weights, values):
  """Broadcast `weights` to the same shape as `values`.

  This returns a version of `weights` following the same broadcast rules as
  `mul(weights, values)`. When computing a weighted average, use this function
  to broadcast `weights` before summing them; e.g.,
  `reduce_sum(w * v) / reduce_sum(_broadcast_weights(w, v))`.

  Args:
    weights: `Tensor` whose rank is either 0, or the same rank as `values`, and
      must be broadcastable to `values` (i.e., all dimensions must be either
      `1`, or the same as the corresponding `values` dimension).
    values: `Tensor` of any shape.

  Returns:
    `weights` broadcast to `values` shape.
  """
  with ops.name_scope(None, 'broadcast_weights', (values, weights)) as scope:
    weights_shape = weights.get_shape()
    values_shape = values.get_shape()
    if (weights_shape.is_fully_defined() and
        values_shape.is_fully_defined() and
        weights_shape.is_compatible_with(values_shape)):
      return weights
    with ops.control_dependencies((_assert_weights_rank(weights, values),)):
      return math_ops.multiply(
          weights, array_ops.ones_like(values), name=scope)


def streaming_mean(values, weights=None, metrics_collections=None,
                   updates_collections=None, name=None):
  """Computes the (weighted) mean of the given values.

  The `streaming_mean` function creates two local variables, `total` and `count`
  that are used to compute the average of `values`. This average is ultimately
  returned as `mean` which is an idempotent operation that simply divides
  `total` by `count`.

  For estimation of the metric  over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `mean`.
  `update_op` increments `total` with the reduced sum of the product of `values`
  and `weights`, and it increments `count` with the reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    values: A `Tensor` of arbitrary dimensions.
    weights: `Tensor` whose rank is either 0, or the same rank as `values`, and
      must be broadcastable to `values` (i.e., all dimensions must be either
      `1`, or the same as the corresponding `values` dimension).
    metrics_collections: An optional list of collections that `mean`
      should be added to.
    updates_collections: An optional list of collections that `update_op`
      should be added to.
    name: An optional variable_scope name.

  Returns:
    mean: A `Tensor` representing the current mean, the value of `total` divided
      by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_value`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`,
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
  """
  return metrics.mean(
      values=values, weights=weights, metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def streaming_mean_tensor(values, weights=None, metrics_collections=None,
                          updates_collections=None, name=None):
  """Computes the element-wise (weighted) mean of the given tensors.

  In contrast to the `streaming_mean` function which returns a scalar with the
  mean,  this function returns an average tensor with the same shape as the
  input tensors.

  The `streaming_mean_tensor` function creates two local variables,
  `total_tensor` and `count_tensor` that are used to compute the average of
  `values`. This average is ultimately returned as `mean` which is an idempotent
  operation that simply divides `total` by `count`.

  For estimation of the metric  over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `mean`.
  `update_op` increments `total` with the reduced sum of the product of `values`
  and `weights`, and it increments `count` with the reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    values: A `Tensor` of arbitrary dimensions.
    weights: `Tensor` whose rank is either 0, or the same rank as `values`, and
      must be broadcastable to `values` (i.e., all dimensions must be either
      `1`, or the same as the corresponding `values` dimension).
    metrics_collections: An optional list of collections that `mean`
      should be added to.
    updates_collections: An optional list of collections that `update_op`
      should be added to.
    name: An optional variable_scope name.

  Returns:
    mean: A float `Tensor` representing the current mean, the value of `total`
      divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_value`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`,
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
  """
  return metrics.mean_tensor(
      values=values, weights=weights, metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def streaming_accuracy(predictions, labels, weights=None,
                       metrics_collections=None, updates_collections=None,
                       name=None):
  """Calculates how often `predictions` matches `labels`.

  The `streaming_accuracy` function creates two local variables, `total` and
  `count` that are used to compute the frequency with which `predictions`
  matches `labels`. This frequency is ultimately returned as `accuracy`: an
  idempotent operation that simply divides `total` by `count`.

  For estimation of the metric  over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `accuracy`.
  Internally, an `is_correct` operation computes a `Tensor` with elements 1.0
  where the corresponding elements of `predictions` and `labels` match and 0.0
  otherwise. Then `update_op` increments `total` with the reduced sum of the
  product of `weights` and `is_correct`, and it increments `count` with the
  reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: The predicted values, a `Tensor` of any shape.
    labels: The ground truth values, a `Tensor` whose shape matches
      `predictions`.
    weights: `Tensor` whose rank is either 0, or the same rank as `labels`, and
      must be broadcastable to `labels` (i.e., all dimensions must be either
      `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `accuracy` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    accuracy: A `Tensor` representing the accuracy, the value of `total` divided
      by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `accuracy`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  return metrics.accuracy(
      predictions=predictions, labels=labels, weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def streaming_precision(predictions, labels, weights=None,
                        metrics_collections=None, updates_collections=None,
                        name=None):
  """Computes the precision of the predictions with respect to the labels.

  The `streaming_precision` function creates two local variables,
  `true_positives` and `false_positives`, that are used to compute the
  precision. This value is ultimately returned as `precision`, an idempotent
  operation that simply divides `true_positives` by the sum of `true_positives`
  and `false_positives`.

  For estimation of the metric  over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `precision`. `update_op` weights each prediction by the corresponding value in
  `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: The predicted values, a `bool` `Tensor` of arbitrary shape.
    labels: The ground truth values, a `bool` `Tensor` whose dimensions must
      match `predictions`.
    weights: `Tensor` whose rank is either 0, or the same rank as `labels`, and
      must be broadcastable to `labels` (i.e., all dimensions must be either
      `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `precision` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    precision: Scalar float `Tensor` with the value of `true_positives`
      divided by the sum of `true_positives` and `false_positives`.
    update_op: `Operation` that increments `true_positives` and
      `false_positives` variables appropriately and whose value matches
      `precision`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  return metrics.precision(
      predictions=predictions, labels=labels, weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def streaming_recall(predictions, labels, weights=None,
                     metrics_collections=None, updates_collections=None,
                     name=None):
  """Computes the recall of the predictions with respect to the labels.

  The `streaming_recall` function creates two local variables, `true_positives`
  and `false_negatives`, that are used to compute the recall. This value is
  ultimately returned as `recall`, an idempotent operation that simply divides
  `true_positives` by the sum of `true_positives`  and `false_negatives`.

  For estimation of the metric  over a stream of data, the function creates an
  `update_op` that updates these variables and returns the `recall`. `update_op`
  weights each prediction by the corresponding value in `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: The predicted values, a `bool` `Tensor` of arbitrary shape.
    labels: The ground truth values, a `bool` `Tensor` whose dimensions must
      match `predictions`.
    weights: `Tensor` whose rank is either 0, or the same rank as `labels`, and
      must be broadcastable to `labels` (i.e., all dimensions must be either
      `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `recall` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    recall: Scalar float `Tensor` with the value of `true_positives` divided
      by the sum of `true_positives` and `false_negatives`.
    update_op: `Operation` that increments `true_positives` and
      `false_negatives` variables appropriately and whose value matches
      `recall`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  return metrics.recall(
      predictions=predictions, labels=labels, weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def _streaming_confusion_matrix_at_thresholds(
    predictions, labels, thresholds, weights=None, includes=None):
  """Computes true_positives, false_negatives, true_negatives, false_positives.

  This function creates up to four local variables, `true_positives`,
  `true_negatives`, `false_positives` and `false_negatives`.
  `true_positive[i]` is defined as the total weight of values in `predictions`
  above `thresholds[i]` whose corresponding entry in `labels` is `True`.
  `false_negatives[i]` is defined as the total weight of values in `predictions`
  at most `thresholds[i]` whose corresponding entry in `labels` is `True`.
  `true_negatives[i]` is defined as the total weight of values in `predictions`
  at most `thresholds[i]` whose corresponding entry in `labels` is `False`.
  `false_positives[i]` is defined as the total weight of values in `predictions`
  above `thresholds[i]` whose corresponding entry in `labels` is `False`.

  For estimation of these metrics over a stream of data, for each metric the
  function respectively creates an `update_op` operation that updates the
  variable and returns its value.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    labels: A `Tensor` whose shape matches `predictions`. `labels` will be cast
      to `bool`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions
      must be either `1`, or the same as the corresponding `labels`
      dimension).
    includes: Tuple of keys to return, from 'tp', 'fn', 'tn', fp'. If `None`,
      default to all four.

  Returns:
    values: Dict of variables of shape `[len(thresholds)]`. Keys are from
        `includes`.
    update_ops: Dict of operations that increments the `values`. Keys are from
        `includes`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      `includes` contains invalid keys.
  """
  all_includes = ('tp', 'fn', 'tn', 'fp')
  if includes is None:
    includes = all_includes
  else:
    for include in includes:
      if include not in all_includes:
        raise ValueError('Invaild key: %s.' % include)

  predictions, labels, weights = _remove_squeezable_dimensions(
      predictions, labels, weights)
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())

  num_thresholds = len(thresholds)

  # Reshape predictions and labels.
  predictions_2d = array_ops.reshape(predictions, [-1, 1])
  labels_2d = array_ops.reshape(
      math_ops.cast(labels, dtype=dtypes.bool), [1, -1])

  # Use static shape if known.
  num_predictions = predictions_2d.get_shape().as_list()[0]

  # Otherwise use dynamic shape.
  if num_predictions is None:
    num_predictions = array_ops.shape(predictions_2d)[0]
  thresh_tiled = array_ops.tile(
      array_ops.expand_dims(array_ops.constant(thresholds), [1]),
      array_ops.stack([1, num_predictions]))

  # Tile the predictions after thresholding them across different thresholds.
  pred_is_pos = math_ops.greater(
      array_ops.tile(array_ops.transpose(predictions_2d), [num_thresholds, 1]),
      thresh_tiled)
  if ('fn' in includes) or ('tn' in includes):
    pred_is_neg = math_ops.logical_not(pred_is_pos)

  # Tile labels by number of thresholds
  label_is_pos = array_ops.tile(labels_2d, [num_thresholds, 1])
  if ('fp' in includes) or ('tn' in includes):
    label_is_neg = math_ops.logical_not(label_is_pos)

  if weights is not None:
    broadcast_weights = _broadcast_weights(
        math_ops.to_float(weights), predictions)
    weights_tiled = array_ops.tile(array_ops.reshape(
        broadcast_weights, [1, -1]), [num_thresholds, 1])
    thresh_tiled.get_shape().assert_is_compatible_with(
        weights_tiled.get_shape())
  else:
    weights_tiled = None

  values = {}
  update_ops = {}

  if 'tp' in includes:
    true_positives = _create_local('true_positives', shape=[num_thresholds])
    is_true_positive = math_ops.to_float(
        math_ops.logical_and(label_is_pos, pred_is_pos))
    if weights_tiled is not None:
      is_true_positive *= weights_tiled
    update_ops['tp'] = state_ops.assign_add(
        true_positives, math_ops.reduce_sum(is_true_positive, 1))
    values['tp'] = true_positives

  if 'fn' in includes:
    false_negatives = _create_local('false_negatives', shape=[num_thresholds])
    is_false_negative = math_ops.to_float(
        math_ops.logical_and(label_is_pos, pred_is_neg))
    if weights_tiled is not None:
      is_false_negative *= weights_tiled
    update_ops['fn'] = state_ops.assign_add(
        false_negatives, math_ops.reduce_sum(is_false_negative, 1))
    values['fn'] = false_negatives

  if 'tn' in includes:
    true_negatives = _create_local('true_negatives', shape=[num_thresholds])
    is_true_negative = math_ops.to_float(
        math_ops.logical_and(label_is_neg, pred_is_neg))
    if weights_tiled is not None:
      is_true_negative *= weights_tiled
    update_ops['tn'] = state_ops.assign_add(
        true_negatives, math_ops.reduce_sum(is_true_negative, 1))
    values['tn'] = true_negatives

  if 'fp' in includes:
    false_positives = _create_local('false_positives', shape=[num_thresholds])
    is_false_positive = math_ops.to_float(
        math_ops.logical_and(label_is_neg, pred_is_pos))
    if weights_tiled is not None:
      is_false_positive *= weights_tiled
    update_ops['fp'] = state_ops.assign_add(
        false_positives, math_ops.reduce_sum(is_false_positive, 1))
    values['fp'] = false_positives

  return values, update_ops


def streaming_true_positives_at_thresholds(
    predictions, labels, thresholds, weights=None):
  values, update_ops = _streaming_confusion_matrix_at_thresholds(
      predictions, labels, thresholds, weights=weights, includes=('tp',))
  return values['tp'], update_ops['tp']


def streaming_false_negatives_at_thresholds(
    predictions, labels, thresholds, weights=None):
  values, update_ops = _streaming_confusion_matrix_at_thresholds(
      predictions, labels, thresholds, weights=weights, includes=('fn',))
  return values['fn'], update_ops['fn']


def streaming_false_positives_at_thresholds(
    predictions, labels, thresholds, weights=None):
  values, update_ops = _streaming_confusion_matrix_at_thresholds(
      predictions, labels, thresholds, weights=weights, includes=('fp',))
  return values['fp'], update_ops['fp']


def streaming_true_negatives_at_thresholds(
    predictions, labels, thresholds, weights=None):
  values, update_ops = _streaming_confusion_matrix_at_thresholds(
      predictions, labels, thresholds, weights=weights, includes=('tn',))
  return values['tn'], update_ops['tn']


def streaming_auc(predictions, labels, weights=None, num_thresholds=200,
                  metrics_collections=None, updates_collections=None,
                  curve='ROC', name=None):
  """Computes the approximate AUC via a Riemann sum.

  The `streaming_auc` function creates four local variables, `true_positives`,
  `true_negatives`, `false_positives` and `false_negatives` that are used to
  compute the AUC. To discretize the AUC curve, a linearly spaced set of
  thresholds is used to compute pairs of recall and precision values. The area
  under the ROC-curve is therefore computed using the height of the recall
  values by the false positive rate, while the area under the PR-curve is the
  computed using the height of the precision values by the recall.

  This value is ultimately returned as `auc`, an idempotent operation that
  computes the area under a discretized curve of precision versus recall values
  (computed using the aforementioned variables). The `num_thresholds` variable
  controls the degree of discretization with larger numbers of thresholds more
  closely approximating the true AUC. The quality of the approximation may vary
  dramatically depending on `num_thresholds`.

  For best results, `predictions` should be distributed approximately uniformly
  in the range [0, 1] and not peaked around 0 or 1. The quality of the AUC
  approximation may be poor if this is not the case.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `auc`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    labels: A `bool` `Tensor` whose shape matches `predictions`.
    weights: `Tensor` whose rank is either 0, or the same rank as `labels`, and
      must be broadcastable to `labels` (i.e., all dimensions must be either
      `1`, or the same as the corresponding `labels` dimension).
    num_thresholds: The number of thresholds to use when discretizing the roc
      curve.
    metrics_collections: An optional list of collections that `auc` should be
      added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    curve: Specifies the name of the curve to be computed, 'ROC' [default] or
    'PR' for the Precision-Recall-curve.
    name: An optional variable_scope name.

  Returns:
    auc: A scalar `Tensor` representing the current area-under-curve.
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables
      appropriately and whose value matches `auc`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  return metrics.auc(
      predictions=predictions, labels=labels, weights=weights,
      metrics_collections=metrics_collections, num_thresholds=num_thresholds,
      curve=curve, updates_collections=updates_collections, name=name)


def streaming_specificity_at_sensitivity(
    predictions, labels, sensitivity, weights=None, num_thresholds=200,
    metrics_collections=None, updates_collections=None, name=None):
  """Computes the specificity at a given sensitivity.

  The `streaming_specificity_at_sensitivity` function creates four local
  variables, `true_positives`, `true_negatives`, `false_positives` and
  `false_negatives` that are used to compute the specificity at the given
  sensitivity value. The threshold for the given sensitivity value is computed
  and used to evaluate the corresponding specificity.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `specificity`. `update_op` increments the `true_positives`, `true_negatives`,
  `false_positives` and `false_negatives` counts with the weight of each case
  found in the `predictions` and `labels`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  For additional information about specificity and sensitivity, see the
  following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity

  Args:
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    labels: A `bool` `Tensor` whose shape matches `predictions`.
    sensitivity: A scalar value in range `[0, 1]`.
    weights: `Tensor` whose rank is either 0, or the same rank as `labels`, and
      must be broadcastable to `labels` (i.e., all dimensions must be either
      `1`, or the same as the corresponding `labels` dimension).
    num_thresholds: The number of thresholds to use for matching the given
      sensitivity.
    metrics_collections: An optional list of collections that `specificity`
      should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    specificity: A scalar `Tensor` representing the specificity at the given
      `specificity` value.
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables
      appropriately and whose value matches `specificity`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      `sensitivity` is not between 0 and 1, or if either `metrics_collections`
      or `updates_collections` are not a list or tuple.
  """
  return metrics.specificity_at_sensitivity(
      sensitivity=sensitivity, num_thresholds=num_thresholds,
      predictions=predictions, labels=labels, weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def streaming_sensitivity_at_specificity(
    predictions, labels, specificity, weights=None, num_thresholds=200,
    metrics_collections=None, updates_collections=None, name=None):
  """Computes the specificity at a given sensitivity.

  The `streaming_sensitivity_at_specificity` function creates four local
  variables, `true_positives`, `true_negatives`, `false_positives` and
  `false_negatives` that are used to compute the sensitivity at the given
  specificity value. The threshold for the given specificity value is computed
  and used to evaluate the corresponding sensitivity.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `sensitivity`. `update_op` increments the `true_positives`, `true_negatives`,
  `false_positives` and `false_negatives` counts with the weight of each case
  found in the `predictions` and `labels`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  For additional information about specificity and sensitivity, see the
  following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity

  Args:
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    labels: A `bool` `Tensor` whose shape matches `predictions`.
    specificity: A scalar value in range `[0, 1]`.
    weights: `Tensor` whose rank is either 0, or the same rank as `labels`, and
      must be broadcastable to `labels` (i.e., all dimensions must be either
      `1`, or the same as the corresponding `labels` dimension).
    num_thresholds: The number of thresholds to use for matching the given
      specificity.
    metrics_collections: An optional list of collections that `sensitivity`
      should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    sensitivity: A scalar `Tensor` representing the sensitivity at the given
      `specificity` value.
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables
      appropriately and whose value matches `sensitivity`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      `specificity` is not between 0 and 1, or if either `metrics_collections`
      or `updates_collections` are not a list or tuple.
  """
  return metrics.sensitivity_at_specificity(
      specificity=specificity, num_thresholds=num_thresholds,
      predictions=predictions, labels=labels, weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def streaming_precision_at_thresholds(predictions, labels, thresholds,
                                      weights=None,
                                      metrics_collections=None,
                                      updates_collections=None, name=None):
  """Computes precision values for different `thresholds` on `predictions`.

  The `streaming_precision_at_thresholds` function creates four local variables,
  `true_positives`, `true_negatives`, `false_positives` and `false_negatives`
  for various values of thresholds. `precision[i]` is defined as the total
  weight of values in `predictions` above `thresholds[i]` whose corresponding
  entry in `labels` is `True`, divided by the total weight of values in
  `predictions` above `thresholds[i]` (`true_positives[i] / (true_positives[i] +
  false_positives[i])`).

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `precision`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    labels: A `bool` `Tensor` whose shape matches `predictions`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    weights: `Tensor` whose rank is either 0, or the same rank as `labels`, and
      must be broadcastable to `labels` (i.e., all dimensions must be either
      `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `auc` should be
      added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    precision: A float `Tensor` of shape `[len(thresholds)]`.
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables that
      are used in the computation of `precision`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  return metrics.precision_at_thresholds(
      thresholds=thresholds,
      predictions=predictions, labels=labels, weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def streaming_recall_at_thresholds(predictions, labels, thresholds,
                                   weights=None, metrics_collections=None,
                                   updates_collections=None, name=None):
  """Computes various recall values for different `thresholds` on `predictions`.

  The `streaming_recall_at_thresholds` function creates four local variables,
  `true_positives`, `true_negatives`, `false_positives` and `false_negatives`
  for various values of thresholds. `recall[i]` is defined as the total weight
  of values in `predictions` above `thresholds[i]` whose corresponding entry in
  `labels` is `True`, divided by the total weight of `True` values in `labels`
  (`true_positives[i] / (true_positives[i] + false_negatives[i])`).

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `recall`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    labels: A `bool` `Tensor` whose shape matches `predictions`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    weights: `Tensor` whose rank is either 0, or the same rank as `labels`, and
      must be broadcastable to `labels` (i.e., all dimensions must be either
      `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `recall` should be
      added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    recall: A float `Tensor` of shape `[len(thresholds)]`.
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables that
      are used in the computation of `recall`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  return metrics.recall_at_thresholds(
      thresholds=thresholds,
      predictions=predictions, labels=labels, weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def _at_k_name(name, k=None, class_id=None):
  if k is not None:
    name = '%s_at_%d' % (name, k)
  else:
    name = '%s_at_k' % (name)
  if class_id is not None:
    name = '%s_class%d' % (name, class_id)
  return name


@deprecated('2016-11-08', 'Please use `streaming_sparse_recall_at_k`, '
            'and reshape labels from [batch_size] to [batch_size, 1].')
def streaming_recall_at_k(predictions, labels, k, weights=None,
                          metrics_collections=None, updates_collections=None,
                          name=None):
  """Computes the recall@k of the predictions with respect to dense labels.

  The `streaming_recall_at_k` function creates two local variables, `total` and
  `count`, that are used to compute the recall@k frequency. This frequency is
  ultimately returned as `recall_at_<k>`: an idempotent operation that simply
  divides `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `recall_at_<k>`. Internally, an `in_top_k` operation computes a `Tensor` with
  shape [batch_size] whose elements indicate whether or not the corresponding
  label is in the top `k` `predictions`. Then `update_op` increments `total`
  with the reduced sum of `weights` where `in_top_k` is `True`, and it
  increments `count` with the reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: A float `Tensor` of dimension [batch_size, num_classes].
    labels: A `Tensor` of dimension [batch_size] whose type is in `int32`,
      `int64`.
    k: The number of top elements to look at for computing recall.
    weights: `Tensor` whose rank is either 0, or the same rank as `labels`, and
      must be broadcastable to `labels` (i.e., all dimensions must be either
      `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `recall_at_k`
      should be added to.
    updates_collections: An optional list of collections `update_op` should be
      added to.
    name: An optional variable_scope name.

  Returns:
    recall_at_k: A `Tensor` representing the recall@k, the fraction of labels
      which fall into the top `k` predictions.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `recall_at_k`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  in_top_k = math_ops.to_float(nn.in_top_k(predictions, labels, k))
  return streaming_mean(in_top_k,
                        weights,
                        metrics_collections,
                        updates_collections,
                        name or _at_k_name('recall', k))


# TODO(ptucker): Validate range of values in labels?
def streaming_sparse_recall_at_k(predictions,
                                 labels,
                                 k,
                                 class_id=None,
                                 weights=None,
                                 metrics_collections=None,
                                 updates_collections=None,
                                 name=None):
  """Computes recall@k of the predictions with respect to sparse labels.

  If `class_id` is not specified, we'll calculate recall as the ratio of true
      positives (i.e., correct predictions, items in the top `k` highest
      `predictions` that are found in the corresponding row in `labels`) to
      actual positives (the full `labels` row).
  If `class_id` is specified, we calculate recall by considering only the rows
      in the batch for which `class_id` is in `labels`, and computing the
      fraction of them for which `class_id` is in the corresponding row in
      `labels`.

  `streaming_sparse_recall_at_k` creates two local variables,
  `true_positive_at_<k>` and `false_negative_at_<k>`, that are used to compute
  the recall_at_k frequency. This frequency is ultimately returned as
  `recall_at_<k>`: an idempotent operation that simply divides
  `true_positive_at_<k>` by total (`true_positive_at_<k>` +
  `false_negative_at_<k>`).

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `recall_at_<k>`. Internally, a `top_k` operation computes a `Tensor`
  indicating the top `k` `predictions`. Set operations applied to `top_k` and
  `labels` calculate the true positives and false negatives weighted by
  `weights`. Then `update_op` increments `true_positive_at_<k>` and
  `false_negative_at_<k>` using these values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: Float `Tensor` with shape [D1, ... DN, num_classes] where
      N >= 1. Commonly, N=1 and predictions has shape [batch size, num_classes].
      The final dimension contains the logit values for each class. [D1, ... DN]
      must match `labels`.
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match `predictions`.
      Values should be in range [0, num_classes), where num_classes is the last
      dimension of `predictions`. Values outside this range always count
      towards `false_negative_at_<k>`.
    k: Integer, k for @k metric.
    class_id: Integer class ID for which we want binary metrics. This should be
      in range [0, num_classes), where num_classes is the last dimension of
      `predictions`. If class_id is outside this range, the method returns NAN.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of new update operation, and namespace for other dependent ops.

  Returns:
    recall: Scalar `float64` `Tensor` with the value of `true_positives` divided
      by the sum of `true_positives` and `false_negatives`.
    update_op: `Operation` that increments `true_positives` and
      `false_negatives` variables appropriately, and whose value matches
      `recall`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match
    `predictions`, or if either `metrics_collections` or `updates_collections`
    are not a list or tuple.
  """
  return metrics.recall_at_k(
      k=k, class_id=class_id,
      predictions=predictions, labels=labels, weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def _streaming_sparse_precision_at_k(top_k_idx,
                                     labels,
                                     k=None,
                                     class_id=None,
                                     weights=None,
                                     metrics_collections=None,
                                     updates_collections=None,
                                     name=None):
  """Computes precision@k of the top-k indices with respect to sparse labels.

  This method contains the code shared by streaming_sparse_precision_at_k and
  streaming_sparse_precision_at_top_k. Refer to those methods for more details.

  Args:
    top_k_idx: Integer `Tensor` with shape [D1, ... DN, k] where
      N >= 1. Commonly, N=1 and top_k_idx has shape [batch size, k].
      The final dimension contains the indices of top-k labels. [D1, ... DN]
      must match `labels`.
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`. Values should be in range [0, num_classes), where
      num_classes is the last dimension of `predictions`. Values outside this
      range are ignored.
    k: Integer, k for @k metric or `None`. Only used for default op name.
    class_id: Integer class ID for which we want binary metrics. This should be
      in range [0, num_classes), where num_classes is the last dimension of
      `predictions`. If `class_id` is outside this range, the method returns
      NAN.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of the metric and of the enclosing scope.

  Returns:
    precision: Scalar `float64` `Tensor` with the value of `true_positives`
      divided by the sum of `true_positives` and `false_positives`.
    update_op: `Operation` that increments `true_positives` and
      `false_positives` variables appropriately, and whose value matches
      `precision`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match
      `predictions`, or if either `metrics_collections` or `updates_collections`
      are not a list or tuple.
  """
  top_k_idx = math_ops.to_int64(top_k_idx)
  tp, tp_update = _streaming_sparse_true_positive_at_k(
      predictions_idx=top_k_idx, labels=labels, k=k, class_id=class_id,
      weights=weights)
  fp, fp_update = _streaming_sparse_false_positive_at_k(
      predictions_idx=top_k_idx, labels=labels, k=k, class_id=class_id,
      weights=weights)

  metric = math_ops.div(tp, math_ops.add(tp, fp), name=name)
  update = math_ops.div(
      tp_update, math_ops.add(tp_update, fp_update), name='update')
  if metrics_collections:
    ops.add_to_collections(metrics_collections, metric)
  if updates_collections:
    ops.add_to_collections(updates_collections, update)
  return metric, update


# TODO(ptucker): Validate range of values in labels?
def streaming_sparse_precision_at_k(predictions,
                                    labels,
                                    k,
                                    class_id=None,
                                    weights=None,
                                    metrics_collections=None,
                                    updates_collections=None,
                                    name=None):
  """Computes precision@k of the predictions with respect to sparse labels.

  If `class_id` is not specified, we calculate precision as the ratio of true
      positives (i.e., correct predictions, items in the top `k` highest
      `predictions` that are found in the corresponding row in `labels`) to
      positives (all top `k` `predictions`).
  If `class_id` is specified, we calculate precision by considering only the
      rows in the batch for which `class_id` is in the top `k` highest
      `predictions`, and computing the fraction of them for which `class_id` is
      in the corresponding row in `labels`.

  We expect precision to decrease as `k` increases.

  `streaming_sparse_precision_at_k` creates two local variables,
  `true_positive_at_<k>` and `false_positive_at_<k>`, that are used to compute
  the precision@k frequency. This frequency is ultimately returned as
  `precision_at_<k>`: an idempotent operation that simply divides
  `true_positive_at_<k>` by total (`true_positive_at_<k>` +
  `false_positive_at_<k>`).

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `precision_at_<k>`. Internally, a `top_k` operation computes a `Tensor`
  indicating the top `k` `predictions`. Set operations applied to `top_k` and
  `labels` calculate the true positives and false positives weighted by
  `weights`. Then `update_op` increments `true_positive_at_<k>` and
  `false_positive_at_<k>` using these values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: Float `Tensor` with shape [D1, ... DN, num_classes] where
      N >= 1. Commonly, N=1 and predictions has shape [batch size, num_classes].
      The final dimension contains the logit values for each class. [D1, ... DN]
      must match `labels`.
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions`. Values should be in range [0, num_classes), where
      num_classes is the last dimension of `predictions`. Values outside this
      range are ignored.
    k: Integer, k for @k metric.
    class_id: Integer class ID for which we want binary metrics. This should be
      in range [0, num_classes], where num_classes is the last dimension of
      `predictions`. If `class_id` is outside this range, the method returns
      NAN.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of new update operation, and namespace for other dependent ops.

  Returns:
    precision: Scalar `float64` `Tensor` with the value of `true_positives`
      divided by the sum of `true_positives` and `false_positives`.
    update_op: `Operation` that increments `true_positives` and
      `false_positives` variables appropriately, and whose value matches
      `precision`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match
      `predictions`, or if either `metrics_collections` or `updates_collections`
      are not a list or tuple.
  """
  return metrics.sparse_precision_at_k(
      k=k, class_id=class_id,
      predictions=predictions, labels=labels, weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


# TODO(ptucker): Validate range of values in labels?
def streaming_sparse_precision_at_top_k(top_k_predictions,
                                        labels,
                                        class_id=None,
                                        weights=None,
                                        metrics_collections=None,
                                        updates_collections=None,
                                        name=None):
  """Computes precision@k of top-k predictions with respect to sparse labels.

  If `class_id` is not specified, we calculate precision as the ratio of
      true positives (i.e., correct predictions, items in `top_k_predictions`
      that are found in the corresponding row in `labels`) to positives (all
      `top_k_predictions`).
  If `class_id` is specified, we calculate precision by considering only the
      rows in the batch for which `class_id` is in the top `k` highest
      `predictions`, and computing the fraction of them for which `class_id` is
      in the corresponding row in `labels`.

  We expect precision to decrease as `k` increases.

  `streaming_sparse_precision_at_top_k` creates two local variables,
  `true_positive_at_k` and `false_positive_at_k`, that are used to compute
  the precision@k frequency. This frequency is ultimately returned as
  `precision_at_k`: an idempotent operation that simply divides
  `true_positive_at_k` by total (`true_positive_at_k` + `false_positive_at_k`).

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `precision_at_k`. Internally, set operations applied to `top_k_predictions`
  and `labels` calculate the true positives and false positives weighted by
  `weights`. Then `update_op` increments `true_positive_at_k` and
  `false_positive_at_k` using these values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    top_k_predictions: Integer `Tensor` with shape [D1, ... DN, k] where
      N >= 1. Commonly, N=1 and top_k_predictions has shape [batch size, k].
      The final dimension contains the indices of top-k labels. [D1, ... DN]
      must match `labels`.
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `top_k_predictions`. Values should be in range [0, num_classes), where
      num_classes is the last dimension of `predictions`. Values outside this
      range are ignored.
    class_id: Integer class ID for which we want binary metrics. This should be
      in range [0, num_classes), where num_classes is the last dimension of
      `predictions`. If `class_id` is outside this range, the method returns
      NAN.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of new update operation, and namespace for other dependent ops.

  Returns:
    precision: Scalar `float64` `Tensor` with the value of `true_positives`
      divided by the sum of `true_positives` and `false_positives`.
    update_op: `Operation` that increments `true_positives` and
      `false_positives` variables appropriately, and whose value matches
      `precision`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match
      `predictions`, or if either `metrics_collections` or `updates_collections`
      are not a list or tuple.
    ValueError: If `top_k_predictions` has rank < 2.
  """
  default_name = _at_k_name('precision', class_id=class_id)
  with ops.name_scope(
      name, default_name,
      (top_k_predictions, labels, weights)) as name_scope:
    return _streaming_sparse_precision_at_k(
        top_k_idx=top_k_predictions,
        labels=labels,
        class_id=class_id,
        weights=weights,
        metrics_collections=metrics_collections,
        updates_collections=updates_collections,
        name=name_scope)


def num_relevant(labels, k):
  """Computes number of relevant values for each row in labels.

  For labels with shape [D1, ... DN, num_labels], this is the minimum of
  `num_labels` and `k`.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels].
    k: Integer, k for @k metric.

  Returns:
    Integer `Tensor` of shape [D1, ... DN], where each value is the number of
    relevant values for that row.

  Raises:
    ValueError: if inputs have invalid dtypes or values.
  """
  if k < 1:
    raise ValueError('Invalid k=%s.' % k)
  with ops.name_scope(None, 'num_relevant', (labels,)) as scope:
    # For SparseTensor, calculate separate count for each row.
    if isinstance(
        labels, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
      labels_sizes = set_ops.set_size(labels)
      return math_ops.minimum(labels_sizes, k, name=scope)

    # For dense Tensor, calculate scalar count based on last dimension, and
    # tile across labels shape.
    labels_shape = array_ops.shape(labels)
    labels_size = labels_shape[-1]
    num_relevant_scalar = math_ops.minimum(labels_size, k)
    return array_ops.fill(labels_shape[0:-1], num_relevant_scalar, name=scope)


def expand_and_tile(tensor, multiple, dim=0, name=None):
  """Slice `tensor` shape in 2, then tile along the sliced dimension.

  A new dimension is inserted in shape of `tensor` before `dim`, then values are
  tiled `multiple` times along the new dimension.

  Args:
    tensor: Input `Tensor` or `SparseTensor`.
    multiple: Integer, number of times to tile.
    dim: Integer, dimension along which to tile.
    name: Name of operation.

  Returns:
    `Tensor` result of expanding and tiling `tensor`.

  Raises:
    ValueError: if `multiple` is less than 1, or `dim` is not in
    `[-rank(tensor), rank(tensor)]`.
  """
  if multiple < 1:
    raise ValueError('Invalid multiple %s, must be > 0.' % multiple)
  with ops.name_scope(
      name, 'expand_and_tile', (tensor, multiple, dim)) as scope:
    # Sparse.
    if isinstance(tensor, sparse_tensor.SparseTensorValue):
      tensor = sparse_tensor.SparseTensor.from_value(tensor)
    if isinstance(tensor, sparse_tensor.SparseTensor):
      if dim < 0:
        expand_dims = array_ops.reshape(
            array_ops.size(tensor.dense_shape) + dim, [1])
      else:
        expand_dims = [dim]
      expanded_shape = array_ops.concat(
          (array_ops.strided_slice(tensor.dense_shape, [0], expand_dims), [1],
           array_ops.strided_slice(
               tensor.dense_shape, expand_dims, [-1], end_mask=1 << 0)),
          0,
          name='expanded_shape')
      expanded = sparse_ops.sparse_reshape(
          tensor, shape=expanded_shape, name='expand')
      if multiple == 1:
        return expanded
      return sparse_ops.sparse_concat(
          dim - 1 if dim < 0 else dim, [expanded] * multiple, name=scope)

    # Dense.
    expanded = array_ops.expand_dims(
        tensor, dim if (dim >= 0) else (dim - 1), name='expand')
    if multiple == 1:
      return expanded
    ones = array_ops.ones_like(array_ops.shape(tensor))
    tile_multiples = array_ops.concat(
        (ones[:dim], (multiple,), ones[dim:]), 0, name='multiples')
    return array_ops.tile(expanded, tile_multiples, name=scope)


def sparse_average_precision_at_k(predictions, labels, k):
  """Computes average precision@k of predictions with respect to sparse labels.

  From en.wikipedia.org/wiki/Information_retrieval#Average_precision, formula
  for each row is:

    AveP = sum_{i=1...k} P_{i} * rel_{i} / num_relevant_items

  A "row" is the elements in dimension [D1, ... DN] of `predictions`, `labels`,
  and the result `Tensors`. In the common case, this is [batch_size]. Each row
  of the results contains the average precision for that row.

  Internally, a `top_k` operation computes a `Tensor` indicating the top `k`
  `predictions`. Set operations applied to `top_k` and `labels` calculate the
  true positives, which are used to calculate the precision ("P_{i}" term,
  above).

  Args:
    predictions: Float `Tensor` with shape [D1, ... DN, num_classes] where
      N >= 1. Commonly, N=1 and `predictions` has shape
      [batch size, num_classes]. The final dimension contains the logit values
      for each class. [D1, ... DN] must match `labels`.
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions`. Values should be in range [0, num_classes), where
      num_classes is the last dimension of `predictions`. Values outside this
      range are ignored.
    k: Integer, k for @k metric. This will calculate an average precision for
      range `[1,k]`, as documented above.

  Returns:
    `float64` `Tensor` of shape [D1, ... DN], where each value is the average
    precision for that row.

  Raises:
    ValueError: if k is invalid.
  """
  if k < 1:
    raise ValueError('Invalid k=%s.' % k)
  with ops.name_scope(
      None, 'average_precision', (predictions, labels, k)) as scope:
    # Calculate top k indices to produce [D1, ... DN, k] tensor.
    _, predictions_idx = nn.top_k(predictions, k)
    predictions_idx = math_ops.to_int64(predictions_idx, name='predictions_idx')

    # Expand dims to produce [D1, ... DN, k, 1] tensor. This gives us a separate
    # prediction for each k, so we can calculate separate true positive values
    # for each k.
    predictions_idx_per_k = array_ops.expand_dims(
        predictions_idx, -1, name='predictions_idx_per_k')

    # Replicate labels k times to produce [D1, ... DN, k, num_labels] tensor.
    labels_per_k = expand_and_tile(
        labels, multiple=k, dim=-1, name='labels_per_k')

    # The following tensors are all of shape [D1, ... DN, k], containing values
    # per row, per k value.
    # `relevant_per_k` (int32) - Relevance indicator, 1 if the prediction at
    #     that k value is correct, 0 otherwise. This is the "rel_{i}" term from
    #     the formula above.
    # `tp_per_k` (int32) - True positive counts.
    # `retrieved_per_k` (int32) - Number of predicted values at each k. This is
    #     the precision denominator.
    # `precision_per_k` (float64) - Precision at each k. This is the "P_{i}"
    #     term from the formula above.
    # `relevant_precision_per_k` (float64) - Relevant precisions; i.e.,
    #     precisions at all k for which relevance indicator is true.
    relevant_per_k = _sparse_true_positive_at_k(
        predictions_idx_per_k, labels_per_k, name='relevant_per_k')
    tp_per_k = math_ops.cumsum(relevant_per_k, axis=-1, name='tp_per_k')
    retrieved_per_k = math_ops.cumsum(
        array_ops.ones_like(relevant_per_k), axis=-1, name='retrieved_per_k')
    precision_per_k = math_ops.div(
        math_ops.to_double(tp_per_k), math_ops.to_double(retrieved_per_k),
        name='precision_per_k')
    relevant_precision_per_k = math_ops.multiply(
        precision_per_k, math_ops.to_double(relevant_per_k),
        name='relevant_precision_per_k')

    # Reduce along k dimension to get the sum, yielding a [D1, ... DN] tensor.
    precision_sum = math_ops.reduce_sum(
        relevant_precision_per_k, reduction_indices=(-1,), name='precision_sum')

    # Divide by number of relevant items to get average precision. These are
    # the "num_relevant_items" and "AveP" terms from the formula above.
    num_relevant_items = math_ops.to_double(num_relevant(labels, k))
    return math_ops.div(precision_sum, num_relevant_items, name=scope)


def streaming_sparse_average_precision_at_k(predictions,
                                            labels,
                                            k,
                                            weights=None,
                                            metrics_collections=None,
                                            updates_collections=None,
                                            name=None):
  """Computes average precision@k of predictions with respect to sparse labels.

  See `sparse_average_precision_at_k` for details on formula. `weights` are
  applied to the result of `sparse_average_precision_at_k`

  `streaming_sparse_average_precision_at_k` creates two local variables,
  `average_precision_at_<k>/total` and `average_precision_at_<k>/max`, that
  are used to compute the frequency. This frequency is ultimately returned as
  `average_precision_at_<k>`: an idempotent operation that simply divides
  `average_precision_at_<k>/total` by `average_precision_at_<k>/max`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `precision_at_<k>`. Internally, a `top_k` operation computes a `Tensor`
  indicating the top `k` `predictions`. Set operations applied to `top_k` and
  `labels` calculate the true positives and false positives weighted by
  `weights`. Then `update_op` increments `true_positive_at_<k>` and
  `false_positive_at_<k>` using these values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: Float `Tensor` with shape [D1, ... DN, num_classes] where
      N >= 1. Commonly, N=1 and `predictions` has shape
      [batch size, num_classes]. The final dimension contains the logit values
      for each class. [D1, ... DN] must match `labels`.
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_`. Values should be in range [0, num_classes), where
      num_classes is the last dimension of `predictions`. Values outside this
      range are ignored.
    k: Integer, k for @k metric. This will calculate an average precision for
      range `[1,k]`, as documented above.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of new update operation, and namespace for other dependent ops.

  Returns:
    mean_average_precision: Scalar `float64` `Tensor` with the mean average
      precision values.
    update: `Operation` that increments  variables appropriately, and whose
      value matches `metric`.
  """
  return metrics.sparse_average_precision_at_k(
      k=k, predictions=predictions, labels=labels, weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def _select_class_id(ids, selected_id):
  """Filter all but `selected_id` out of `ids`.

  Args:
    ids: `int64` `Tensor` or `SparseTensor` of IDs.
    selected_id: Int id to select.

  Returns:
    `SparseTensor` of same dimensions as `ids`. This contains only the entries
    equal to `selected_id`.
  """
  if isinstance(
      ids, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
    return sparse_ops.sparse_retain(
        ids, math_ops.equal(ids.values, selected_id))

  # TODO(ptucker): Make this more efficient, maybe add a sparse version of
  # tf.equal and tf.reduce_any?

  # Shape of filled IDs is the same as `ids` with the last dim collapsed to 1.
  ids_shape = array_ops.shape(ids, out_type=dtypes.int64)
  ids_last_dim = array_ops.size(ids_shape) - 1
  filled_selected_id_shape = math_ops.reduced_shape(
      ids_shape, array_ops.reshape(ids_last_dim, [1]))

  # Intersect `ids` with the selected ID.
  filled_selected_id = array_ops.fill(
      filled_selected_id_shape, math_ops.to_int64(selected_id))
  result = set_ops.set_intersection(filled_selected_id, ids)
  return sparse_tensor.SparseTensor(
      indices=result.indices, values=result.values, dense_shape=ids_shape)


def _maybe_select_class_id(labels, predictions_idx, selected_id=None):
  """If class ID is specified, filter all other classes.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`.
    predictions_idx: `int64` `Tensor` of class IDs, with shape [D1, ... DN, k]
      where N >= 1. Commonly, N=1 and `predictions_idx` has shape
      [batch size, k].
    selected_id: Int id to select.

  Returns:
    Tuple of `labels` and `predictions_idx`, possibly with classes removed.
  """
  if selected_id is None:
    return labels, predictions_idx
  return (_select_class_id(labels, selected_id),
          _select_class_id(predictions_idx, selected_id))


def _sparse_true_positive_at_k(predictions_idx,
                               labels,
                               class_id=None,
                               weights=None,
                               name=None):
  """Calculates true positives for recall@k and precision@k.

  If `class_id` is specified, calculate binary true positives for `class_id`
      only.
  If `class_id` is not specified, calculate metrics for `k` predicted vs
      `n` label classes, where `n` is the 2nd dimension of `labels_sparse`.

  Args:
    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,
      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must
      match `labels`.
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`.
    class_id: Class for which we want binary metrics.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    name: Name of operation.

  Returns:
    A [D1, ... DN] `Tensor` of true positive counts.
  """
  with ops.name_scope(
      name, 'true_positives', (predictions_idx, labels, weights)):
    labels, predictions_idx = _maybe_select_class_id(
        labels, predictions_idx, class_id)
    tp = set_ops.set_size(set_ops.set_intersection(predictions_idx, labels))
    tp = math_ops.to_double(tp)
    if weights is not None:
      weights = math_ops.to_double(weights)
      with ops.control_dependencies((_assert_weights_rank(weights, tp),)):
        tp = math_ops.multiply(tp, weights)
    return tp


def _streaming_sparse_true_positive_at_k(predictions_idx,
                                         labels,
                                         k=None,
                                         class_id=None,
                                         weights=None,
                                         name=None):
  """Calculates weighted per step true positives for recall@k and precision@k.

  If `class_id` is specified, calculate binary true positives for `class_id`
      only.
  If `class_id` is not specified, calculate metrics for `k` predicted vs
      `n` label classes, where `n` is the 2nd dimension of `labels`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,
      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must
      match `labels`.
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`.
    k: Integer, k for @k metric. This is only used for default op name.
    class_id: Class for which we want binary metrics.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    name: Name of new variable, and namespace for other dependent ops.

  Returns:
    A tuple of `Variable` and update `Operation`.

  Raises:
    ValueError: If `weights` is not `None` and has an incomptable shape.
  """
  default_name = _at_k_name('true_positive', k, class_id=class_id)
  with ops.name_scope(
      name, default_name, (predictions_idx, labels, weights)) as scope:
    tp = _sparse_true_positive_at_k(
        predictions_idx=predictions_idx, labels=labels, class_id=class_id,
        weights=weights)
    batch_total_tp = math_ops.to_double(math_ops.reduce_sum(tp))

    var = contrib_variables.local_variable(
        array_ops.zeros([], dtype=dtypes.float64), name=scope)
    return var, state_ops.assign_add(var, batch_total_tp, name='update')


def _sparse_false_positive_at_k(predictions_idx,
                                labels,
                                class_id=None,
                                weights=None):
  """Calculates false positives for precision@k.

  If `class_id` is specified, calculate binary true positives for `class_id`
      only.
  If `class_id` is not specified, calculate metrics for `k` predicted vs
      `n` label classes, where `n` is the 2nd dimension of `labels_sparse`.

  Args:
    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,
      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must
      match `labels`.
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`.
    class_id: Class for which we want binary metrics.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).

  Returns:
    A [D1, ... DN] `Tensor` of false positive counts.
  """
  with ops.name_scope(
      None, 'false_positives', (predictions_idx, labels, weights)):
    labels, predictions_idx = _maybe_select_class_id(labels,
                                                     predictions_idx,
                                                     class_id)
    fp = set_ops.set_size(set_ops.set_difference(
        predictions_idx, labels, aminusb=True))
    fp = math_ops.to_double(fp)
    if weights is not None:
      weights = math_ops.to_double(weights)
      with ops.control_dependencies((_assert_weights_rank(weights, fp),)):
        fp = math_ops.multiply(fp, weights)
    return fp


def _streaming_sparse_false_positive_at_k(predictions_idx,
                                          labels,
                                          k=None,
                                          class_id=None,
                                          weights=None,
                                          name=None):
  """Calculates weighted per step false positives for precision@k.

  If `class_id` is specified, calculate binary true positives for `class_id`
      only.
  If `class_id` is not specified, calculate metrics for `k` predicted vs
      `n` label classes, where `n` is the 2nd dimension of `labels`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,
      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must
      match `labels`.
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`.
    k: Integer, k for @k metric. This is only used for default op name.
    class_id: Class for which we want binary metrics.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    name: Name of new variable, and namespace for other dependent ops.

  Returns:
    A tuple of `Variable` and update `Operation`.

  Raises:
    ValueError: If `weights` is not `None` and has an incomptable shape.
  """
  with ops.name_scope(
      name, _at_k_name('false_positive', k, class_id=class_id),
      (predictions_idx, labels, weights)) as scope:
    fp = _sparse_false_positive_at_k(
        predictions_idx=predictions_idx, labels=labels, class_id=class_id,
        weights=weights)
    batch_total_fp = math_ops.to_double(math_ops.reduce_sum(fp))

    var = contrib_variables.local_variable(
        array_ops.zeros([], dtype=dtypes.float64), name=scope)
    return var, state_ops.assign_add(var, batch_total_fp, name='update')


def _sparse_false_negative_at_k(predictions_idx,
                                labels,
                                class_id=None,
                                weights=None):
  """Calculates false negatives for recall@k.

  If `class_id` is specified, calculate binary true positives for `class_id`
      only.
  If `class_id` is not specified, calculate metrics for `k` predicted vs
      `n` label classes, where `n` is the 2nd dimension of `labels_sparse`.

  Args:
    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,
      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must
      match `labels`.
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`.
    class_id: Class for which we want binary metrics.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).

  Returns:
    A [D1, ... DN] `Tensor` of false negative counts.
  """
  with ops.name_scope(
      None, 'false_negatives', (predictions_idx, labels, weights)):
    labels, predictions_idx = _maybe_select_class_id(labels,
                                                     predictions_idx,
                                                     class_id)
    fn = set_ops.set_size(set_ops.set_difference(predictions_idx,
                                                 labels,
                                                 aminusb=False))
    fn = math_ops.to_double(fn)
    if weights is not None:
      weights = math_ops.to_double(weights)
      with ops.control_dependencies((_assert_weights_rank(weights, fn),)):
        fn = math_ops.multiply(fn, weights)
    return fn


def _streaming_sparse_false_negative_at_k(predictions_idx,
                                          labels,
                                          k,
                                          class_id=None,
                                          weights=None,
                                          name=None):
  """Calculates weighted per step false negatives for recall@k.

  If `class_id` is specified, calculate binary true positives for `class_id`
      only.
  If `class_id` is not specified, calculate metrics for `k` predicted vs
      `n` label classes, where `n` is the 2nd dimension of `labels`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions_idx: 1-D or higher `int64` `Tensor` with last dimension `k`,
      top `k` predicted classes. For rank `n`, the first `n-1` dimensions must
      match `labels`.
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`.
    k: Integer, k for @k metric. This is only used for default op name.
    class_id: Class for which we want binary metrics.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    name: Name of new variable, and namespace for other dependent ops.

  Returns:
    A tuple of `Variable` and update `Operation`.

  Raises:
    ValueError: If `weights` is not `None` and has an incomptable shape.
  """
  with ops.name_scope(
      name, _at_k_name('false_negative', k, class_id=class_id),
      (predictions_idx, labels, weights)) as scope:
    fn = _sparse_false_negative_at_k(
        predictions_idx=predictions_idx, labels=labels, class_id=class_id,
        weights=weights)
    batch_total_fn = math_ops.to_double(math_ops.reduce_sum(fn))

    var = contrib_variables.local_variable(
        array_ops.zeros([], dtype=dtypes.float64), name=scope)
    return var, state_ops.assign_add(var, batch_total_fn, name='update')


def streaming_mean_absolute_error(predictions, labels, weights=None,
                                  metrics_collections=None,
                                  updates_collections=None,
                                  name=None):
  """Computes the mean absolute error between the labels and predictions.

  The `streaming_mean_absolute_error` function creates two local variables,
  `total` and `count` that are used to compute the mean absolute error. This
  average is weighted by `weights`, and it is ultimately returned as
  `mean_absolute_error`: an idempotent operation that simply divides `total` by
  `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `mean_absolute_error`. Internally, an `absolute_errors` operation computes the
  absolute value of the differences between `predictions` and `labels`. Then
  `update_op` increments `total` with the reduced sum of the product of
  `weights` and `absolute_errors`, and it increments `count` with the reduced
  sum of `weights`

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: A `Tensor` of arbitrary shape.
    labels: A `Tensor` of the same shape as `predictions`.
    weights: Optional `Tensor` indicating the frequency with which an example is
      sampled. Rank must be 0, or the same rank as `labels`, and must be
      broadcastable to `labels` (i.e., all dimensions must be either `1`, or
      the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that
      `mean_absolute_error` should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    mean_absolute_error: A `Tensor` representing the current mean, the value of
      `total` divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_absolute_error`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  return metrics.mean_absolute_error(
      predictions=predictions, labels=labels, weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def streaming_mean_relative_error(predictions, labels, normalizer, weights=None,
                                  metrics_collections=None,
                                  updates_collections=None,
                                  name=None):
  """Computes the mean relative error by normalizing with the given values.

  The `streaming_mean_relative_error` function creates two local variables,
  `total` and `count` that are used to compute the mean relative absolute error.
  This average is weighted by `weights`, and it is ultimately returned as
  `mean_relative_error`: an idempotent operation that simply divides `total` by
  `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `mean_reative_error`. Internally, a `relative_errors` operation divides the
  absolute value of the differences between `predictions` and `labels` by the
  `normalizer`. Then `update_op` increments `total` with the reduced sum of the
  product of `weights` and `relative_errors`, and it increments `count` with the
  reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: A `Tensor` of arbitrary shape.
    labels: A `Tensor` of the same shape as `predictions`.
    normalizer: A `Tensor` of the same shape as `predictions`.
    weights: Optional `Tensor` indicating the frequency with which an example is
      sampled. Rank must be 0, or the same rank as `labels`, and must be
      broadcastable to `labels` (i.e., all dimensions must be either `1`, or
      the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that
      `mean_relative_error` should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    mean_relative_error: A `Tensor` representing the current mean, the value of
      `total` divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_relative_error`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  return metrics.mean_relative_error(
      normalizer=normalizer, predictions=predictions, labels=labels,
      weights=weights, metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def streaming_mean_squared_error(predictions, labels, weights=None,
                                 metrics_collections=None,
                                 updates_collections=None,
                                 name=None):
  """Computes the mean squared error between the labels and predictions.

  The `streaming_mean_squared_error` function creates two local variables,
  `total` and `count` that are used to compute the mean squared error.
  This average is weighted by `weights`, and it is ultimately returned as
  `mean_squared_error`: an idempotent operation that simply divides `total` by
  `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `mean_squared_error`. Internally, a `squared_error` operation computes the
  element-wise square of the difference between `predictions` and `labels`. Then
  `update_op` increments `total` with the reduced sum of the product of
  `weights` and `squared_error`, and it increments `count` with the reduced sum
  of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: A `Tensor` of arbitrary shape.
    labels: A `Tensor` of the same shape as `predictions`.
    weights: Optional `Tensor` indicating the frequency with which an example is
      sampled. Rank must be 0, or the same rank as `labels`, and must be
      broadcastable to `labels` (i.e., all dimensions must be either `1`, or
      the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that
      `mean_squared_error` should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    mean_squared_error: A `Tensor` representing the current mean, the value of
      `total` divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_squared_error`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  return metrics.mean_squared_error(
      predictions=predictions, labels=labels, weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def streaming_root_mean_squared_error(predictions, labels, weights=None,
                                      metrics_collections=None,
                                      updates_collections=None,
                                      name=None):
  """Computes the root mean squared error between the labels and predictions.

  The `streaming_root_mean_squared_error` function creates two local variables,
  `total` and `count` that are used to compute the root mean squared error.
  This average is weighted by `weights`, and it is ultimately returned as
  `root_mean_squared_error`: an idempotent operation that takes the square root
  of the division of `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `root_mean_squared_error`. Internally, a `squared_error` operation computes
  the element-wise square of the difference between `predictions` and `labels`.
  Then `update_op` increments `total` with the reduced sum of the product of
  `weights` and `squared_error`, and it increments `count` with the reduced sum
  of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: A `Tensor` of arbitrary shape.
    labels: A `Tensor` of the same shape as `predictions`.
    weights: Optional `Tensor` indicating the frequency with which an example is
      sampled. Rank must be 0, or the same rank as `labels`, and must be
      broadcastable to `labels` (i.e., all dimensions must be either `1`, or
      the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that
      `root_mean_squared_error` should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    root_mean_squared_error: A `Tensor` representing the current mean, the value
      of `total` divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `root_mean_squared_error`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  return metrics.root_mean_squared_error(
      predictions=predictions, labels=labels, weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def streaming_covariance(predictions,
                         labels,
                         weights=None,
                         metrics_collections=None,
                         updates_collections=None,
                         name=None):
  """Computes the unbiased sample covariance between `predictions` and `labels`.

  The `streaming_covariance` function creates four local variables,
  `comoment`, `mean_prediction`, `mean_label`, and `count`, which are used to
  compute the sample covariance between predictions and labels across multiple
  batches of data. The covariance is ultimately returned as an idempotent
  operation that simply divides `comoment` by `count` - 1. We use `count` - 1
  in order to get an unbiased estimate.

  The algorithm used for this online computation is described in
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.
  Specifically, the formula used to combine two sample comoments is
  `C_AB = C_A + C_B + (E[x_A] - E[x_B]) * (E[y_A] - E[y_B]) * n_A * n_B / n_AB`
  The comoment for a single batch of data is simply
  `sum((x - E[x]) * (y - E[y]))`, optionally weighted.

  If `weights` is not None, then it is used to compute weighted comoments,
  means, and count. NOTE: these weights are treated as "frequency weights", as
  opposed to "reliability weights". See discussion of the difference on
  https://wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance

  To facilitate the computation of covariance across multiple batches of data,
  the function creates an `update_op` operation, which updates underlying
  variables and returns the updated covariance.

  Args:
    predictions: A `Tensor` of arbitrary size.
    labels: A `Tensor` of the same size as `predictions`.
    weights: Optional `Tensor` indicating the frequency with which an example is
      sampled. Rank must be 0, or the same rank as `labels`, and must be
      broadcastable to `labels` (i.e., all dimensions must be either `1`, or
      the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    covariance: A `Tensor` representing the current unbiased sample covariance,
      `comoment` / (`count` - 1).
    update_op: An operation that updates the local variables appropriately.

  Raises:
    ValueError: If labels and predictions are of different sizes or if either
      `metrics_collections` or `updates_collections` are not a list or tuple.
  """
  with variable_scope.variable_scope(
      name, 'covariance', (predictions, labels, weights)):
    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions, labels, weights)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    count = _create_local('count', [])
    mean_prediction = _create_local('mean_prediction', [])
    mean_label = _create_local('mean_label', [])
    comoment = _create_local('comoment', [])  # C_A in update equation

    if weights is None:
      batch_count = math_ops.to_float(array_ops.size(labels))  # n_B in eqn
      weighted_predictions = predictions
      weighted_labels = labels
    else:
      weights = _broadcast_weights(weights, labels)
      batch_count = math_ops.reduce_sum(weights)  # n_B in eqn
      weighted_predictions = math_ops.multiply(predictions, weights)
      weighted_labels = math_ops.multiply(labels, weights)

    update_count = state_ops.assign_add(count, batch_count)  # n_AB in eqn
    prev_count = update_count - batch_count  # n_A in update equation

    # We update the means by Delta=Error*BatchCount/(BatchCount+PrevCount)
    # batch_mean_prediction is E[x_B] in the update equation
    batch_mean_prediction = _safe_div(
        math_ops.reduce_sum(weighted_predictions), batch_count,
        'batch_mean_prediction')
    delta_mean_prediction = _safe_div(
        (batch_mean_prediction - mean_prediction) * batch_count, update_count,
        'delta_mean_prediction')
    update_mean_prediction = state_ops.assign_add(mean_prediction,
                                                  delta_mean_prediction)
    # prev_mean_prediction is E[x_A] in the update equation
    prev_mean_prediction = update_mean_prediction - delta_mean_prediction

    # batch_mean_label is E[y_B] in the update equation
    batch_mean_label = _safe_div(
        math_ops.reduce_sum(weighted_labels), batch_count, 'batch_mean_label')
    delta_mean_label = _safe_div((batch_mean_label - mean_label) * batch_count,
                                 update_count, 'delta_mean_label')
    update_mean_label = state_ops.assign_add(mean_label, delta_mean_label)
    # prev_mean_label is E[y_A] in the update equation
    prev_mean_label = update_mean_label - delta_mean_label

    unweighted_batch_coresiduals = (
        (predictions - batch_mean_prediction) * (labels - batch_mean_label))
    # batch_comoment is C_B in the update equation
    if weights is None:
      batch_comoment = math_ops.reduce_sum(unweighted_batch_coresiduals)
    else:
      batch_comoment = math_ops.reduce_sum(unweighted_batch_coresiduals *
                                           weights)

    # View delta_comoment as = C_AB - C_A in the update equation above.
    # Since C_A is stored in a var, by how much do we need to increment that var
    # to make the var = C_AB?
    delta_comoment = (batch_comoment +
                      (prev_mean_prediction - batch_mean_prediction) *
                      (prev_mean_label - batch_mean_label) *
                      (prev_count * batch_count / update_count))
    update_comoment = state_ops.assign_add(comoment, delta_comoment)

    covariance = _safe_div(comoment, count - 1, 'covariance')
    with ops.control_dependencies([update_comoment]):
      update_op = _safe_div(comoment, count - 1, 'update_op')

  if metrics_collections:
    ops.add_to_collections(metrics_collections, covariance)

  if updates_collections:
    ops.add_to_collections(updates_collections, update_op)

  return covariance, update_op


def streaming_pearson_correlation(predictions,
                                  labels,
                                  weights=None,
                                  metrics_collections=None,
                                  updates_collections=None,
                                  name=None):
  """Computes Pearson correlation coefficient between `predictions`, `labels`.

  The `streaming_pearson_correlation` function delegates to
  `streaming_covariance` the tracking of three [co]variances:

  - `streaming_covariance(predictions, labels)`, i.e. covariance
  - `streaming_covariance(predictions, predictions)`, i.e. variance
  - `streaming_covariance(labels, labels)`, i.e. variance

  The product-moment correlation ultimately returned is an idempotent operation
  `cov(predictions, labels) / sqrt(var(predictions) * var(labels))`. To
  facilitate correlation computation across multiple batches, the function
  groups the `update_op`s of the underlying streaming_covariance and returns an
  `update_op`.

  If `weights` is not None, then it is used to compute a weighted correlation.
  NOTE: these weights are treated as "frequency weights", as opposed to
  "reliability weights". See discussion of the difference on
  https://wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance

  Args:
    predictions: A `Tensor` of arbitrary size.
    labels: A `Tensor` of the same size as predictions.
    weights: Optional `Tensor` indicating the frequency with which an example is
      sampled. Rank must be 0, or the same rank as `labels`, and must be
      broadcastable to `labels` (i.e., all dimensions must be either `1`, or
      the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    pearson_r: A `Tensor` representing the current Pearson product-moment
      correlation coefficient, the value of
      `cov(predictions, labels) / sqrt(var(predictions) * var(labels))`.
    update_op: An operation that updates the underlying variables appropriately.

  Raises:
    ValueError: If `labels` and `predictions` are of different sizes, or if
      `weights` is the wrong size, or if either `metrics_collections` or
      `updates_collections` are not a `list` or `tuple`.
  """
  with variable_scope.variable_scope(
      name, 'pearson_r', (predictions, labels, weights)):
    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions, labels, weights)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    # Broadcast weights here to avoid duplicate broadcasting in each call to
    # `streaming_covariance`.
    if weights is not None:
      weights = _broadcast_weights(weights, labels)
    cov, update_cov = streaming_covariance(
        predictions, labels, weights=weights, name='covariance')
    var_predictions, update_var_predictions = streaming_covariance(
        predictions, predictions, weights=weights, name='variance_predictions')
    var_labels, update_var_labels = streaming_covariance(
        labels, labels, weights=weights, name='variance_labels')

    pearson_r = _safe_div(
        cov,
        math_ops.multiply(math_ops.sqrt(var_predictions),
                          math_ops.sqrt(var_labels)),
        'pearson_r')
    with ops.control_dependencies(
        [update_cov, update_var_predictions, update_var_labels]):
      update_op = _safe_div(update_cov, math_ops.multiply(
          math_ops.sqrt(update_var_predictions),
          math_ops.sqrt(update_var_labels)), 'update_op')

  if metrics_collections:
    ops.add_to_collections(metrics_collections, pearson_r)

  if updates_collections:
    ops.add_to_collections(updates_collections, update_op)

  return pearson_r, update_op


# TODO(nsilberman): add a 'normalized' flag so that the user can request
# normalization if the inputs are not normalized.
def streaming_mean_cosine_distance(predictions, labels, dim, weights=None,
                                   metrics_collections=None,
                                   updates_collections=None,
                                   name=None):
  """Computes the cosine distance between the labels and predictions.

  The `streaming_mean_cosine_distance` function creates two local variables,
  `total` and `count` that are used to compute the average cosine distance
  between `predictions` and `labels`. This average is weighted by `weights`,
  and it is ultimately returned as `mean_distance`, which is an idempotent
  operation that simply divides `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `mean_distance`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: A `Tensor` of the same shape as `labels`.
    labels: A `Tensor` of arbitrary shape.
    dim: The dimension along which the cosine distance is computed.
    weights: An optional `Tensor` whose shape is broadcastable to `predictions`,
      and whose dimension `dim` is 1.
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    mean_distance: A `Tensor` representing the current mean, the value of
      `total` divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  predictions, labels, weights = _remove_squeezable_dimensions(
      predictions, labels, weights)
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())
  radial_diffs = math_ops.multiply(predictions, labels)
  radial_diffs = math_ops.reduce_sum(radial_diffs,
                                     reduction_indices=[dim,],
                                     keep_dims=True)
  mean_distance, update_op = streaming_mean(radial_diffs, weights,
                                            None,
                                            None,
                                            name or 'mean_cosine_distance')
  mean_distance = math_ops.subtract(1.0, mean_distance)
  update_op = math_ops.subtract(1.0, update_op)

  if metrics_collections:
    ops.add_to_collections(metrics_collections, mean_distance)

  if updates_collections:
    ops.add_to_collections(updates_collections, update_op)

  return mean_distance, update_op


def streaming_percentage_less(values, threshold, weights=None,
                              metrics_collections=None,
                              updates_collections=None,
                              name=None):
  """Computes the percentage of values less than the given threshold.

  The `streaming_percentage_less` function creates two local variables,
  `total` and `count` that are used to compute the percentage of `values` that
  fall below `threshold`. This rate is weighted by `weights`, and it is
  ultimately returned as `percentage` which is an idempotent operation that
  simply divides `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `percentage`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    values: A numeric `Tensor` of arbitrary size.
    threshold: A scalar threshold.
    weights: An optional `Tensor` whose shape is broadcastable to `values`.
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    percentage: A `Tensor` representing the current mean, the value of `total`
      divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`,
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
  """
  return metrics.percentage_below(
      values=values, threshold=threshold, weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def streaming_mean_iou(predictions,
                       labels,
                       num_classes,
                       weights=None,
                       metrics_collections=None,
                       updates_collections=None,
                       name=None):
  """Calculate per-step mean Intersection-Over-Union (mIOU).

  Mean Intersection-Over-Union is a common evaluation metric for
  semantic image segmentation, which first computes the IOU for each
  semantic class and then computes the average over classes.
  IOU is defined as follows:
    IOU = true_positive / (true_positive + false_positive + false_negative).
  The predictions are accumulated in a confusion matrix, weighted by `weights`,
  and mIOU is then calculated from it.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `mean_iou`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: A `Tensor` of prediction results for semantic labels, whose
      shape is [batch size] and type `int32` or `int64`. The tensor will be
      flattened, if its rank > 1.
    labels: A `Tensor` of ground truth labels with shape [batch size] and of
      type `int32` or `int64`. The tensor will be flattened, if its rank > 1.
    num_classes: The possible number of labels the prediction task can
      have. This value must be provided, since a confusion matrix of
      dimension = [num_classes, num_classes] will be allocated.
    weights: An optional `Tensor` whose shape is broadcastable to `predictions`.
    metrics_collections: An optional list of collections that `mean_iou`
      should be added to.
    updates_collections: An optional list of collections `update_op` should be
      added to.
    name: An optional variable_scope name.

  Returns:
    mean_iou: A `Tensor` representing the mean intersection-over-union.
    update_op: An operation that increments the confusion matrix.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  return metrics.mean_iou(
      num_classes=num_classes, predictions=predictions, labels=labels,
      weights=weights, metrics_collections=metrics_collections,
      updates_collections=updates_collections, name=name)


def _next_array_size(required_size, growth_factor=1.5):
  """Calculate the next size for reallocating a dynamic array.

  Args:
    required_size: number or tf.Tensor specifying required array capacity.
    growth_factor: optional number or tf.Tensor specifying the growth factor
      between subsequent allocations.

  Returns:
    tf.Tensor with dtype=int32 giving the next array size.
  """
  exponent = math_ops.ceil(
      math_ops.log(math_ops.cast(required_size, dtypes.float32))
      / math_ops.log(math_ops.cast(growth_factor, dtypes.float32)))
  return math_ops.cast(math_ops.ceil(growth_factor ** exponent), dtypes.int32)


def streaming_concat(values,
                     axis=0,
                     max_size=None,
                     metrics_collections=None,
                     updates_collections=None,
                     name=None):
  """Concatenate values along an axis across batches.

  The function `streaming_concat` creates two local variables, `array` and
  `size`, that are used to store concatenated values. Internally, `array` is
  used as storage for a dynamic array (if `maxsize` is `None`), which ensures
  that updates can be run in amortized constant time.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that appends the values of a tensor and returns the
  `value` of the concatenated tensors.

  This op allows for evaluating metrics that cannot be updated incrementally
  using the same framework as other streaming metrics.

  Args:
    values: `Tensor` to concatenate. Rank and the shape along all axes other
      than the axis to concatenate along must be statically known.
    axis: optional integer axis to concatenate along.
    max_size: optional integer maximum size of `value` along the given axis.
      Once the maximum size is reached, further updates are no-ops. By default,
      there is no maximum size: the array is resized as necessary.
    metrics_collections: An optional list of collections that `value`
      should be added to.
    updates_collections: An optional list of collections `update_op` should be
      added to.
    name: An optional variable_scope name.

  Returns:
    value: A `Tensor` representing the concatenated values.
    update_op: An operation that concatenates the next values.

  Raises:
    ValueError: if `values` does not have a statically known rank, `axis` is
      not in the valid range or the size of `values` is not statically known
      along any axis other than `axis`.
  """
  with variable_scope.variable_scope(name, 'streaming_concat', (values,)):
    # pylint: disable=invalid-slice-index
    values_shape = values.get_shape()
    if values_shape.dims is None:
      raise ValueError('`values` must have known statically known rank')

    ndim = len(values_shape)
    if axis < 0:
      axis += ndim
    if not 0 <= axis < ndim:
      raise ValueError('axis = %r not in [0, %r)' % (axis, ndim))

    fixed_shape = [dim.value for n, dim in enumerate(values_shape)
                   if n != axis]
    if any(value is None for value in fixed_shape):
      raise ValueError('all dimensions of `values` other than the dimension to '
                       'concatenate along must have statically known size')

    # We move `axis` to the front of the internal array so assign ops can be
    # applied to contiguous slices
    init_size = 0 if max_size is None else max_size
    init_shape = [init_size] + fixed_shape
    array = _create_local(
        'array', shape=init_shape, validate_shape=False, dtype=values.dtype)
    size = _create_local('size', shape=[], dtype=dtypes.int32)

    perm = [0 if n == axis else n + 1 if n < axis else n for n in range(ndim)]
    valid_array = array[:size]
    valid_array.set_shape([None] + fixed_shape)
    value = array_ops.transpose(valid_array, perm, name='concat')

    values_size = array_ops.shape(values)[axis]
    if max_size is None:
      batch_size = values_size
    else:
      batch_size = math_ops.minimum(values_size, max_size - size)

    perm = [axis] + [n for n in range(ndim) if n != axis]
    batch_values = array_ops.transpose(values, perm)[:batch_size]

    def reallocate():
      next_size = _next_array_size(new_size)
      next_shape = array_ops.stack([next_size] + fixed_shape)
      new_value = array_ops.zeros(next_shape, dtype=values.dtype)
      old_value = array.value()
      assign_op = state_ops.assign(array, new_value, validate_shape=False)
      with ops.control_dependencies([assign_op]):
        copy_op = array[:size].assign(old_value[:size])
      # return value needs to be the same dtype as no_op() for cond
      with ops.control_dependencies([copy_op]):
        return control_flow_ops.no_op()

    new_size = size + batch_size
    array_size = array_ops.shape_internal(array, optimize=False)[0]
    maybe_reallocate_op = control_flow_ops.cond(
        new_size > array_size, reallocate, control_flow_ops.no_op)
    with ops.control_dependencies([maybe_reallocate_op]):
      append_values_op = array[size:new_size].assign(batch_values)
    with ops.control_dependencies([append_values_op]):
      update_op = size.assign(new_size)

    if metrics_collections:
      ops.add_to_collections(metrics_collections, value)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return value, update_op
    # pylint: enable=invalid-slice-index


def aggregate_metrics(*value_update_tuples):
  """Aggregates the metric value tensors and update ops into two lists.

  Args:
    *value_update_tuples: a variable number of tuples, each of which contain the
      pair of (value_tensor, update_op) from a streaming metric.

  Returns:
    A list of value `Tensor` objects and a list of update ops.

  Raises:
    ValueError: if `value_update_tuples` is empty.
  """
  if not value_update_tuples:
    raise ValueError('Expected at least one value_tensor/update_op pair')
  value_ops, update_ops = zip(*value_update_tuples)
  return list(value_ops), list(update_ops)


def aggregate_metric_map(names_to_tuples):
  """Aggregates the metric names to tuple dictionary.

  This function is useful for pairing metric names with their associated value
  and update ops when the list of metrics is long. For example:

  ```python
    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
        'Mean Absolute Error': new_slim.metrics.streaming_mean_absolute_error(
            predictions, labels, weights),
        'Mean Relative Error': new_slim.metrics.streaming_mean_relative_error(
            predictions, labels, labels, weights),
        'RMSE Linear': new_slim.metrics.streaming_root_mean_squared_error(
            predictions, labels, weights),
        'RMSE Log': new_slim.metrics.streaming_root_mean_squared_error(
            predictions, labels, weights),
    })
  ```

  Args:
    names_to_tuples: a map of metric names to tuples, each of which contain the
      pair of (value_tensor, update_op) from a streaming metric.

  Returns:
    A dictionary from metric names to value ops and a dictionary from metric
    names to update ops.
  """
  metric_names = names_to_tuples.keys()
  value_ops, update_ops = zip(*names_to_tuples.values())
  return dict(zip(metric_names, value_ops)), dict(zip(metric_names, update_ops))


def _remove_squeezable_dimensions(predictions, labels, weights):
  """Squeeze last dim if needed.

  Squeezes `predictions` and `labels` if their rank differs by 1.
  Squeezes `weights` if its rank is 1 more than the new rank of `predictions`

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    predictions: Predicted values, a `Tensor` of arbitrary dimensions.
    labels: Label values, a `Tensor` whose dimensions match `predictions`.
    weights: Optional weight `Tensor`. It will be squeezed if its rank is 1
      more than the new rank of `predictions`

  Returns:
    Tuple of `predictions`, `labels` and `weights`, possibly with the last
    dimension squeezed.
  """
  predictions, labels = tensor_util.remove_squeezable_dimensions(
      predictions, labels)
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())

  if weights is not None:
    weights = ops.convert_to_tensor(weights)
    predictions_shape = predictions.get_shape()
    predictions_rank = predictions_shape.ndims
    weights_shape = weights.get_shape()
    weights_rank = weights_shape.ndims

    if (predictions_rank is not None) and (weights_rank is not None):
      # Use static rank.
      if weights_rank - predictions_rank == 1:
        weights = array_ops.squeeze(weights, [-1])
    elif (weights_rank is None) or (
        weights_shape.dims[-1].is_compatible_with(1)):
      # Use dynamic rank
      weights = control_flow_ops.cond(
          math_ops.equal(array_ops.rank(weights),
                         math_ops.add(array_ops.rank(predictions), 1)),
          lambda: array_ops.squeeze(weights, [-1]),
          lambda: weights)
  return predictions, labels, weights


__all__ = [
    'aggregate_metric_map',
    'aggregate_metrics',
    'streaming_accuracy',
    'streaming_auc',
    'streaming_false_negatives',
    'streaming_false_negatives_at_thresholds',
    'streaming_false_positives',
    'streaming_false_positives_at_thresholds',
    'streaming_mean',
    'streaming_mean_absolute_error',
    'streaming_mean_cosine_distance',
    'streaming_mean_iou',
    'streaming_mean_relative_error',
    'streaming_mean_squared_error',
    'streaming_mean_tensor',
    'streaming_percentage_less',
    'streaming_precision',
    'streaming_precision_at_thresholds',
    'streaming_recall',
    'streaming_recall_at_k',
    'streaming_recall_at_thresholds',
    'streaming_root_mean_squared_error',
    'streaming_sensitivity_at_specificity',
    'streaming_sparse_average_precision_at_k',
    'streaming_sparse_precision_at_k',
    'streaming_sparse_recall_at_k',
    'streaming_specificity_at_sensitivity',
    'streaming_true_negatives',
    'streaming_true_negatives_at_thresholds',
    'streaming_true_positives',
    'streaming_true_positives_at_thresholds',
]
