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

import collections as collections_lib

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.util.deprecation import deprecated

# Epsilon constant used to represent extremely small quantity.
_EPSILON = 1e-7


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


def streaming_true_positives(predictions,
                             labels,
                             weights=None,
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
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


def streaming_true_negatives(predictions,
                             labels,
                             weights=None,
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
  return metrics.true_negatives(
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


def streaming_false_positives(predictions,
                              labels,
                              weights=None,
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
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


def streaming_false_negatives(predictions,
                              labels,
                              weights=None,
                              metrics_collections=None,
                              updates_collections=None,
                              name=None):
  """Computes the total number of false negatives.

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
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


def streaming_mean(values,
                   weights=None,
                   metrics_collections=None,
                   updates_collections=None,
                   name=None):
  """Computes the (weighted) mean of the given values.

  The `streaming_mean` function creates two local variables, `total` and `count`
  that are used to compute the average of `values`. This average is ultimately
  returned as `mean` which is an idempotent operation that simply divides
  `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
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
      appropriately and whose value matches `mean`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`,
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
  """
  return metrics.mean(
      values=values,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


def streaming_mean_tensor(values,
                          weights=None,
                          metrics_collections=None,
                          updates_collections=None,
                          name=None):
  """Computes the element-wise (weighted) mean of the given tensors.

  In contrast to the `streaming_mean` function which returns a scalar with the
  mean,  this function returns an average tensor with the same shape as the
  input tensors.

  The `streaming_mean_tensor` function creates two local variables,
  `total_tensor` and `count_tensor` that are used to compute the average of
  `values`. This average is ultimately returned as `mean` which is an idempotent
  operation that simply divides `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
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
      appropriately and whose value matches `mean`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`,
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
  """
  return metrics.mean_tensor(
      values=values,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


@deprecated(
    None, 'Please switch to tf.metrics.accuracy. Note that the order of the '
    'labels and predictions arguments has been switched.')
def streaming_accuracy(predictions,
                       labels,
                       weights=None,
                       metrics_collections=None,
                       updates_collections=None,
                       name=None):
  """Calculates how often `predictions` matches `labels`.

  The `streaming_accuracy` function creates two local variables, `total` and
  `count` that are used to compute the frequency with which `predictions`
  matches `labels`. This frequency is ultimately returned as `accuracy`: an
  idempotent operation that simply divides `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
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
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


def streaming_precision(predictions,
                        labels,
                        weights=None,
                        metrics_collections=None,
                        updates_collections=None,
                        name=None):
  """Computes the precision of the predictions with respect to the labels.

  The `streaming_precision` function creates two local variables,
  `true_positives` and `false_positives`, that are used to compute the
  precision. This value is ultimately returned as `precision`, an idempotent
  operation that simply divides `true_positives` by the sum of `true_positives`
  and `false_positives`.

  For estimation of the metric over a stream of data, the function creates an
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
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


def streaming_recall(predictions,
                     labels,
                     weights=None,
                     metrics_collections=None,
                     updates_collections=None,
                     name=None):
  """Computes the recall of the predictions with respect to the labels.

  The `streaming_recall` function creates two local variables, `true_positives`
  and `false_negatives`, that are used to compute the recall. This value is
  ultimately returned as `recall`, an idempotent operation that simply divides
  `true_positives` by the sum of `true_positives`  and `false_negatives`.

  For estimation of the metric over a stream of data, the function creates an
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
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


def streaming_false_positive_rate(predictions,
                                  labels,
                                  weights=None,
                                  metrics_collections=None,
                                  updates_collections=None,
                                  name=None):
  """Computes the false positive rate of predictions with respect to labels.

  The `false_positive_rate` function creates two local variables,
  `false_positives` and `true_negatives`, that are used to compute the
  false positive rate. This value is ultimately returned as
  `false_positive_rate`, an idempotent operation that simply divides
  `false_positives` by the sum of `false_positives` and `true_negatives`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `false_positive_rate`. `update_op` weights each prediction by the
  corresponding value in `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will
      be cast to `bool`.
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that
     `false_positive_rate` should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    false_positive_rate: Scalar float `Tensor` with the value of
      `false_positives` divided by the sum of `false_positives` and
      `true_negatives`.
    update_op: `Operation` that increments `false_positives` and
      `true_negatives` variables appropriately and whose value matches
      `false_positive_rate`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  with variable_scope.variable_scope(name, 'false_positive_rate',
                                     (predictions, labels, weights)):
    predictions, labels, weights = metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
        predictions=math_ops.cast(predictions, dtype=dtypes.bool),
        labels=math_ops.cast(labels, dtype=dtypes.bool),
        weights=weights)

    false_p, false_positives_update_op = metrics.false_positives(
        labels=labels,
        predictions=predictions,
        weights=weights,
        metrics_collections=None,
        updates_collections=None,
        name=None)
    true_n, true_negatives_update_op = metrics.true_negatives(
        labels=labels,
        predictions=predictions,
        weights=weights,
        metrics_collections=None,
        updates_collections=None,
        name=None)

    def compute_fpr(fp, tn, name):
      return array_ops.where(
          math_ops.greater(fp + tn, 0), math_ops.div(fp, fp + tn), 0, name)

    fpr = compute_fpr(false_p, true_n, 'value')
    update_op = compute_fpr(false_positives_update_op, true_negatives_update_op,
                            'update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, fpr)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return fpr, update_op


def streaming_false_negative_rate(predictions,
                                  labels,
                                  weights=None,
                                  metrics_collections=None,
                                  updates_collections=None,
                                  name=None):
  """Computes the false negative rate of predictions with respect to labels.

  The `false_negative_rate` function creates two local variables,
  `false_negatives` and `true_positives`, that are used to compute the
  false positive rate. This value is ultimately returned as
  `false_negative_rate`, an idempotent operation that simply divides
  `false_negatives` by the sum of `false_negatives` and `true_positives`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `false_negative_rate`. `update_op` weights each prediction by the
  corresponding value in `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will
      be cast to `bool`.
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that
      `false_negative_rate` should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    false_negative_rate: Scalar float `Tensor` with the value of
      `false_negatives` divided by the sum of `false_negatives` and
      `true_positives`.
    update_op: `Operation` that increments `false_negatives` and
      `true_positives` variables appropriately and whose value matches
      `false_negative_rate`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  with variable_scope.variable_scope(name, 'false_negative_rate',
                                     (predictions, labels, weights)):
    predictions, labels, weights = metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
        predictions=math_ops.cast(predictions, dtype=dtypes.bool),
        labels=math_ops.cast(labels, dtype=dtypes.bool),
        weights=weights)

    false_n, false_negatives_update_op = metrics.false_negatives(
        labels,
        predictions,
        weights,
        metrics_collections=None,
        updates_collections=None,
        name=None)
    true_p, true_positives_update_op = metrics.true_positives(
        labels,
        predictions,
        weights,
        metrics_collections=None,
        updates_collections=None,
        name=None)

    def compute_fnr(fn, tp, name):
      return array_ops.where(
          math_ops.greater(fn + tp, 0), math_ops.div(fn, fn + tp), 0, name)

    fnr = compute_fnr(false_n, true_p, 'value')
    update_op = compute_fnr(false_negatives_update_op, true_positives_update_op,
                            'update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, fnr)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return fnr, update_op


def _streaming_confusion_matrix_at_thresholds(predictions,
                                              labels,
                                              thresholds,
                                              weights=None,
                                              includes=None):
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

  predictions, labels, weights = metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
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
    broadcast_weights = weights_broadcast_ops.broadcast_weights(
        math_ops.to_float(weights), predictions)
    weights_tiled = array_ops.tile(
        array_ops.reshape(broadcast_weights, [1, -1]), [num_thresholds, 1])
    thresh_tiled.get_shape().assert_is_compatible_with(
        weights_tiled.get_shape())
  else:
    weights_tiled = None

  values = {}
  update_ops = {}

  if 'tp' in includes:
    true_positives = metrics_impl.metric_variable(
        [num_thresholds], dtypes.float32, name='true_positives')
    is_true_positive = math_ops.to_float(
        math_ops.logical_and(label_is_pos, pred_is_pos))
    if weights_tiled is not None:
      is_true_positive *= weights_tiled
    update_ops['tp'] = state_ops.assign_add(true_positives,
                                            math_ops.reduce_sum(
                                                is_true_positive, 1))
    values['tp'] = true_positives

  if 'fn' in includes:
    false_negatives = metrics_impl.metric_variable(
        [num_thresholds], dtypes.float32, name='false_negatives')
    is_false_negative = math_ops.to_float(
        math_ops.logical_and(label_is_pos, pred_is_neg))
    if weights_tiled is not None:
      is_false_negative *= weights_tiled
    update_ops['fn'] = state_ops.assign_add(false_negatives,
                                            math_ops.reduce_sum(
                                                is_false_negative, 1))
    values['fn'] = false_negatives

  if 'tn' in includes:
    true_negatives = metrics_impl.metric_variable(
        [num_thresholds], dtypes.float32, name='true_negatives')
    is_true_negative = math_ops.to_float(
        math_ops.logical_and(label_is_neg, pred_is_neg))
    if weights_tiled is not None:
      is_true_negative *= weights_tiled
    update_ops['tn'] = state_ops.assign_add(true_negatives,
                                            math_ops.reduce_sum(
                                                is_true_negative, 1))
    values['tn'] = true_negatives

  if 'fp' in includes:
    false_positives = metrics_impl.metric_variable(
        [num_thresholds], dtypes.float32, name='false_positives')
    is_false_positive = math_ops.to_float(
        math_ops.logical_and(label_is_neg, pred_is_pos))
    if weights_tiled is not None:
      is_false_positive *= weights_tiled
    update_ops['fp'] = state_ops.assign_add(false_positives,
                                            math_ops.reduce_sum(
                                                is_false_positive, 1))
    values['fp'] = false_positives

  return values, update_ops


def streaming_true_positives_at_thresholds(predictions,
                                           labels,
                                           thresholds,
                                           weights=None):
  values, update_ops = _streaming_confusion_matrix_at_thresholds(
      predictions, labels, thresholds, weights=weights, includes=('tp',))
  return values['tp'], update_ops['tp']


def streaming_false_negatives_at_thresholds(predictions,
                                            labels,
                                            thresholds,
                                            weights=None):
  values, update_ops = _streaming_confusion_matrix_at_thresholds(
      predictions, labels, thresholds, weights=weights, includes=('fn',))
  return values['fn'], update_ops['fn']


def streaming_false_positives_at_thresholds(predictions,
                                            labels,
                                            thresholds,
                                            weights=None):
  values, update_ops = _streaming_confusion_matrix_at_thresholds(
      predictions, labels, thresholds, weights=weights, includes=('fp',))
  return values['fp'], update_ops['fp']


def streaming_true_negatives_at_thresholds(predictions,
                                           labels,
                                           thresholds,
                                           weights=None):
  values, update_ops = _streaming_confusion_matrix_at_thresholds(
      predictions, labels, thresholds, weights=weights, includes=('tn',))
  return values['tn'], update_ops['tn']


def streaming_curve_points(labels=None,
                           predictions=None,
                           weights=None,
                           num_thresholds=200,
                           metrics_collections=None,
                           updates_collections=None,
                           curve='ROC',
                           name=None):
  """Computes curve (ROC or PR) values for a prespecified number of points.

  The `streaming_curve_points` function creates four local variables,
  `true_positives`, `true_negatives`, `false_positives` and `false_negatives`
  that are used to compute the curve values. To discretize the curve, a linearly
  spaced set of thresholds is used to compute pairs of recall and precision
  values.

  For best results, `predictions` should be distributed approximately uniformly
  in the range [0, 1] and not peaked around 0 or 1.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
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
    points: A `Tensor` with shape [num_thresholds, 2] that contains points of
      the curve.
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.

  TODO(chizeng): Consider rewriting this method to make use of logic within the
  streaming_precision_recall_at_equal_thresholds method (to improve run time).
  """
  with variable_scope.variable_scope(name, 'curve_points',
                                     (labels, predictions, weights)):
    if curve != 'ROC' and curve != 'PR':
      raise ValueError('curve must be either ROC or PR, %s unknown' % (curve))
    kepsilon = _EPSILON  # to account for floating point imprecisions
    thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                  for i in range(num_thresholds - 2)]
    thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

    values, update_ops = _streaming_confusion_matrix_at_thresholds(
        labels=labels,
        predictions=predictions,
        thresholds=thresholds,
        weights=weights)

    # Add epsilons to avoid dividing by 0.
    epsilon = 1.0e-6

    def compute_points(tp, fn, tn, fp):
      """Computes the roc-auc or pr-auc based on confusion counts."""
      rec = math_ops.div(tp + epsilon, tp + fn + epsilon)
      if curve == 'ROC':
        fp_rate = math_ops.div(fp, fp + tn + epsilon)
        return fp_rate, rec
      else:  # curve == 'PR'.
        prec = math_ops.div(tp + epsilon, tp + fp + epsilon)
        return rec, prec

    xs, ys = compute_points(values['tp'], values['fn'], values['tn'],
                            values['fp'])
    points = array_ops.stack([xs, ys], axis=1)
    update_op = control_flow_ops.group(*update_ops.values())

    if metrics_collections:
      ops.add_to_collections(metrics_collections, points)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return points, update_op


@deprecated(
    None, 'Please switch to tf.metrics.auc. Note that the order of the '
    'labels and predictions arguments has been switched.')
def streaming_auc(predictions,
                  labels,
                  weights=None,
                  num_thresholds=200,
                  metrics_collections=None,
                  updates_collections=None,
                  curve='ROC',
                  name=None):
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
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      num_thresholds=num_thresholds,
      curve=curve,
      updates_collections=updates_collections,
      name=name)


def _compute_dynamic_auc(labels, predictions, curve='ROC'):
  """Computes the apporixmate AUC by a Riemann sum with data-derived thresholds.

  Computes the area under the ROC or PR curve using each prediction as a
  threshold. This could be slow for large batches, but has the advantage of not
  having its results degrade depending on the distribution of predictions.

  Args:
    labels: A `Tensor` of ground truth labels with the same shape as
      `predictions` with values of 0 or 1 and type `int64`.
    predictions: A 1-D `Tensor` of predictions whose values are `float64`.
    curve: The name of the curve to be computed, 'ROC' for the Receiving
      Operating Characteristic or 'PR' for the Precision-Recall curve.

  Returns:
    A scalar `Tensor` containing the area-under-curve value for the input.
  """
  # Count the total number of positive and negative labels in the input.
  size = array_ops.size(predictions)
  total_positive = math_ops.cast(math_ops.reduce_sum(labels), dtypes.int32)

  def continue_computing_dynamic_auc():
    """Continues dynamic auc computation, entered if labels are not all equal.

    Returns:
      A scalar `Tensor` containing the area-under-curve value.
    """
    # Sort the predictions descending, and the corresponding labels as well.
    ordered_predictions, indices = nn.top_k(predictions, k=size)
    ordered_labels = array_ops.gather(labels, indices)

    # Get the counts of the unique ordered predictions.
    _, _, counts = array_ops.unique_with_counts(ordered_predictions)

    # Compute the indices of the split points between different predictions.
    splits = math_ops.cast(
        array_ops.pad(math_ops.cumsum(counts), paddings=[[1, 0]]), dtypes.int32)

    # Count the positives to the left of the split indices.
    positives = math_ops.cast(
        array_ops.pad(math_ops.cumsum(ordered_labels), paddings=[[1, 0]]),
        dtypes.int32)
    true_positives = array_ops.gather(positives, splits)
    if curve == 'ROC':
      # Count the negatives to the left of every split point and the total
      # number of negatives for computing the FPR.
      false_positives = math_ops.subtract(splits, true_positives)
      total_negative = size - total_positive
      x_axis_values = math_ops.truediv(false_positives, total_negative)
      y_axis_values = math_ops.truediv(true_positives, total_positive)
    elif curve == 'PR':
      x_axis_values = math_ops.truediv(true_positives, total_positive)
      # For conformance, set precision to 1 when the number of positive
      # classifications is 0.
      y_axis_values = array_ops.where(
          math_ops.greater(splits, 0),
          math_ops.truediv(true_positives, splits),
          array_ops.ones_like(true_positives, dtype=dtypes.float64))

    # Calculate trapezoid areas.
    heights = math_ops.add(y_axis_values[1:], y_axis_values[:-1]) / 2.0
    widths = math_ops.abs(
        math_ops.subtract(x_axis_values[1:], x_axis_values[:-1]))
    return math_ops.reduce_sum(math_ops.multiply(heights, widths))

  # If all the labels are the same, AUC isn't well-defined (but raising an
  # exception seems excessive) so we return 0, otherwise we finish computing.
  return control_flow_ops.cond(
      math_ops.logical_or(
          math_ops.equal(total_positive, 0),
          math_ops.equal(total_positive, size)
      ),
      true_fn=lambda: array_ops.constant(0, dtypes.float64),
      false_fn=continue_computing_dynamic_auc)


def streaming_dynamic_auc(labels,
                          predictions,
                          curve='ROC',
                          metrics_collections=(),
                          updates_collections=(),
                          name=None):
  """Computes the apporixmate AUC by a Riemann sum with data-derived thresholds.

  USAGE NOTE: this approach requires storing all of the predictions and labels
  for a single evaluation in memory, so it may not be usable when the evaluation
  batch size and/or the number of evaluation steps is very large.

  Computes the area under the ROC or PR curve using each prediction as a
  threshold. This has the advantage of being resilient to the distribution of
  predictions by aggregating across batches, accumulating labels and predictions
  and performing the final calculation using all of the concatenated values.

  Args:
    labels: A `Tensor` of ground truth labels with the same shape as `labels`
      and with values of 0 or 1 whose values are castable to `int64`.
    predictions: A `Tensor` of predictions whose values are castable to
      `float64`. Will be flattened into a 1-D `Tensor`.
    curve: The name of the curve for which to compute AUC, 'ROC' for the
      Receiving Operating Characteristic or 'PR' for the Precision-Recall curve.
    metrics_collections: An optional iterable of collections that `auc` should
      be added to.
    updates_collections: An optional iterable of collections that `update_op`
      should be added to.
    name: An optional name for the variable_scope that contains the metric
      variables.

  Returns:
    auc: A scalar `Tensor` containing the current area-under-curve value.
    update_op: An operation that concatenates the input labels and predictions
      to the accumulated values.

  Raises:
    ValueError: If `labels` and `predictions` have mismatched shapes or if
      `curve` isn't a recognized curve type.
  """

  if curve not in ['PR', 'ROC']:
    raise ValueError('curve must be either ROC or PR, %s unknown' % curve)

  with variable_scope.variable_scope(name, default_name='dynamic_auc'):
    labels.get_shape().assert_is_compatible_with(predictions.get_shape())
    predictions = array_ops.reshape(
        math_ops.cast(predictions, dtypes.float64), [-1])
    labels = array_ops.reshape(math_ops.cast(labels, dtypes.int64), [-1])
    with ops.control_dependencies([
        check_ops.assert_greater_equal(
            labels,
            array_ops.zeros_like(labels, dtypes.int64),
            message='labels must be 0 or 1, at least one is <0'),
        check_ops.assert_less_equal(
            labels,
            array_ops.ones_like(labels, dtypes.int64),
            message='labels must be 0 or 1, at least one is >1')
    ]):
      preds_accum, update_preds = streaming_concat(predictions,
                                                   name='concat_preds')
      labels_accum, update_labels = streaming_concat(labels,
                                                     name='concat_labels')
      update_op = control_flow_ops.group(update_labels, update_preds)
      auc = _compute_dynamic_auc(labels_accum, preds_accum, curve=curve)
      if updates_collections:
        ops.add_to_collections(updates_collections, update_op)
      if metrics_collections:
        ops.add_to_collections(metrics_collections, auc)
      return auc, update_op


def streaming_precision_recall_at_equal_thresholds(predictions,
                                                   labels,
                                                   num_thresholds=None,
                                                   weights=None,
                                                   name=None,
                                                   use_locking=None):
  """A helper method for creating metrics related to precision-recall curves.

  These values are true positives, false negatives, true negatives, false
  positives, precision, and recall. This function returns a data structure that
  contains ops within it.

  Unlike _streaming_confusion_matrix_at_thresholds (which exhibits O(T * N)
  space and run time), this op exhibits O(T + N) space and run time, where T is
  the number of thresholds and N is the size of the predictions tensor. Hence,
  it may be advantageous to use this function when `predictions` is big.

  For instance, prefer this method for per-pixel classification tasks, for which
  the predictions tensor may be very large.

  Each number in `predictions`, a float in `[0, 1]`, is compared with its
  corresponding label in `labels`, and counts as a single tp/fp/tn/fn value at
  each threshold. This is then multiplied with `weights` which can be used to
  reweight certain values, or more commonly used for masking values.

  Args:
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    labels: A bool `Tensor` whose shape matches `predictions`.
    num_thresholds: Optional; Number of thresholds, evenly distributed in
      `[0, 1]`. Should be `>= 2`. Defaults to 201. Note that the number of bins
      is 1 less than `num_thresholds`. Using an even `num_thresholds` value
      instead of an odd one may yield unfriendly edges for bins.
    weights: Optional; If provided, a `Tensor` that has the same dtype as,
      and broadcastable to, `predictions`. This tensor is multplied by counts.
    name: Optional; variable_scope name. If not provided, the string
      'precision_recall_at_equal_threshold' is used.
    use_locking: Optional; If True, the op will be protected by a lock.
      Otherwise, the behavior is undefined, but may exhibit less contention.
      Defaults to True.

  Returns:
    result: A named tuple (See PrecisionRecallData within the implementation of
      this function) with properties that are variables of shape
      `[num_thresholds]`. The names of the properties are tp, fp, tn, fn,
      precision, recall, thresholds.
    update_op: An op that accumulates values.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      `includes` contains invalid keys.
  """
  # Disable the invalid-name checker so that we can capitalize the name.
  # pylint: disable=invalid-name
  PrecisionRecallData = collections_lib.namedtuple(
      'PrecisionRecallData',
      ['tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'thresholds'])
  # pylint: enable=invalid-name

  if num_thresholds is None:
    num_thresholds = 201

  if weights is None:
    weights = 1.0

  if use_locking is None:
    use_locking = True

  check_ops.assert_type(labels, dtypes.bool)

  dtype = predictions.dtype
  with variable_scope.variable_scope(name,
                                     'precision_recall_at_equal_thresholds',
                                     (labels, predictions, weights)):
    # Make sure that predictions are within [0.0, 1.0].
    with ops.control_dependencies([
        check_ops.assert_greater_equal(
            predictions,
            math_ops.cast(0.0, dtype=predictions.dtype),
            message='predictions must be in [0, 1]'),
        check_ops.assert_less_equal(
            predictions,
            math_ops.cast(1.0, dtype=predictions.dtype),
            message='predictions must be in [0, 1]')
    ]):
      predictions, labels, weights = metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
          predictions=predictions,
          labels=labels,
          weights=weights)

    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    # We cast to float to ensure we have 0.0 or 1.0.
    f_labels = math_ops.cast(labels, dtype)

    # Get weighted true/false labels.
    true_labels = f_labels * weights
    false_labels = (1.0 - f_labels) * weights

    # Flatten predictions and labels.
    predictions = array_ops.reshape(predictions, [-1])
    true_labels = array_ops.reshape(true_labels, [-1])
    false_labels = array_ops.reshape(false_labels, [-1])

    # To compute TP/FP/TN/FN, we are measuring a binary classifier
    #   C(t) = (predictions >= t)
    # at each threshold 't'. So we have
    #   TP(t) = sum( C(t) * true_labels )
    #   FP(t) = sum( C(t) * false_labels )
    #
    # But, computing C(t) requires computation for each t. To make it fast,
    # observe that C(t) is a cumulative integral, and so if we have
    #   thresholds = [t_0, ..., t_{n-1}];  t_0 < ... < t_{n-1}
    # where n = num_thresholds, and if we can compute the bucket function
    #   B(i) = Sum( (predictions == t), t_i <= t < t{i+1} )
    # then we get
    #   C(t_i) = sum( B(j), j >= i )
    # which is the reversed cumulative sum in tf.cumsum().
    #
    # We can compute B(i) efficiently by taking advantage of the fact that
    # our thresholds are evenly distributed, in that
    #   width = 1.0 / (num_thresholds - 1)
    #   thresholds = [0.0, 1*width, 2*width, 3*width, ..., 1.0]
    # Given a prediction value p, we can map it to its bucket by
    #   bucket_index(p) = floor( p * (num_thresholds - 1) )
    # so we can use tf.scatter_add() to update the buckets in one pass.
    #
    # This implementation exhibits a run time and space complexity of O(T + N),
    # where T is the number of thresholds and N is the size of predictions.
    # Metrics that rely on _streaming_confusion_matrix_at_thresholds instead
    # exhibit a complexity of O(T * N).

    # Compute the bucket indices for each prediction value.
    bucket_indices = math_ops.cast(
        math_ops.floor(predictions * (num_thresholds - 1)), dtypes.int32)

    with ops.name_scope('variables'):
      tp_buckets_v = metrics_impl.metric_variable(
          [num_thresholds], dtype, name='tp_buckets')
      fp_buckets_v = metrics_impl.metric_variable(
          [num_thresholds], dtype, name='fp_buckets')

    with ops.name_scope('update_op'):
      update_tp = state_ops.scatter_add(
          tp_buckets_v, bucket_indices, true_labels, use_locking=use_locking)
      update_fp = state_ops.scatter_add(
          fp_buckets_v, bucket_indices, false_labels, use_locking=use_locking)

    # Set up the cumulative sums to compute the actual metrics.
    tp = math_ops.cumsum(tp_buckets_v, reverse=True, name='tp')
    fp = math_ops.cumsum(fp_buckets_v, reverse=True, name='fp')
    # fn = sum(true_labels) - tp
    #    = sum(tp_buckets) - tp
    #    = tp[0] - tp
    # Similarly,
    # tn = fp[0] - fp
    tn = fp[0] - fp
    fn = tp[0] - tp

    # We use a minimum to prevent division by 0.
    epsilon = 1e-7
    precision = tp / math_ops.maximum(epsilon, tp + fp)
    recall = tp / math_ops.maximum(epsilon, tp + fn)

    result = PrecisionRecallData(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        thresholds=math_ops.lin_space(0.0, 1.0, num_thresholds))
    update_op = control_flow_ops.group(update_tp, update_fp)
    return result, update_op


def streaming_specificity_at_sensitivity(predictions,
                                         labels,
                                         sensitivity,
                                         weights=None,
                                         num_thresholds=200,
                                         metrics_collections=None,
                                         updates_collections=None,
                                         name=None):
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
      sensitivity=sensitivity,
      num_thresholds=num_thresholds,
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


def streaming_sensitivity_at_specificity(predictions,
                                         labels,
                                         specificity,
                                         weights=None,
                                         num_thresholds=200,
                                         metrics_collections=None,
                                         updates_collections=None,
                                         name=None):
  """Computes the sensitivity at a given specificity.

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
      specificity=specificity,
      num_thresholds=num_thresholds,
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


@deprecated(
    None, 'Please switch to tf.metrics.precision_at_thresholds. Note that the '
    'order of the labels and predictions arguments has been switched.')
def streaming_precision_at_thresholds(predictions,
                                      labels,
                                      thresholds,
                                      weights=None,
                                      metrics_collections=None,
                                      updates_collections=None,
                                      name=None):
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
    metrics_collections: An optional list of collections that `precision` should
      be added to.
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
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


@deprecated(
    None, 'Please switch to tf.metrics.recall_at_thresholds. Note that the '
    'order of the labels and predictions arguments has been switched.')
def streaming_recall_at_thresholds(predictions,
                                   labels,
                                   thresholds,
                                   weights=None,
                                   metrics_collections=None,
                                   updates_collections=None,
                                   name=None):
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
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


def streaming_false_positive_rate_at_thresholds(predictions,
                                                labels,
                                                thresholds,
                                                weights=None,
                                                metrics_collections=None,
                                                updates_collections=None,
                                                name=None):
  """Computes various fpr values for different `thresholds` on `predictions`.

  The `streaming_false_positive_rate_at_thresholds` function creates two
  local variables, `false_positives`, `true_negatives`, for various values of
  thresholds. `false_positive_rate[i]` is defined as the total weight
  of values in `predictions` above `thresholds[i]` whose corresponding entry in
  `labels` is `False`, divided by the total weight of `False` values in `labels`
  (`false_positives[i] / (false_positives[i] + true_negatives[i])`).

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `false_positive_rate`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    labels: A `bool` `Tensor` whose shape matches `predictions`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    weights: `Tensor` whose rank is either 0, or the same rank as `labels`, and
      must be broadcastable to `labels` (i.e., all dimensions must be either
      `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that
      `false_positive_rate` should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    false_positive_rate: A float `Tensor` of shape `[len(thresholds)]`.
    update_op: An operation that increments the `false_positives` and
      `true_negatives` variables that are used in the computation of
      `false_positive_rate`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  with variable_scope.variable_scope(name, 'false_positive_rate_at_thresholds',
                                     (predictions, labels, weights)):
    values, update_ops = _streaming_confusion_matrix_at_thresholds(
        predictions, labels, thresholds, weights, includes=('fp', 'tn'))

    # Avoid division by zero.
    epsilon = _EPSILON

    def compute_fpr(fp, tn, name):
      return math_ops.div(fp, epsilon + fp + tn, name='fpr_' + name)

    fpr = compute_fpr(values['fp'], values['tn'], 'value')
    update_op = compute_fpr(update_ops['fp'], update_ops['tn'], 'update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, fpr)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return fpr, update_op


def streaming_false_negative_rate_at_thresholds(predictions,
                                                labels,
                                                thresholds,
                                                weights=None,
                                                metrics_collections=None,
                                                updates_collections=None,
                                                name=None):
  """Computes various fnr values for different `thresholds` on `predictions`.

  The `streaming_false_negative_rate_at_thresholds` function creates two
  local variables, `false_negatives`, `true_positives`, for various values of
  thresholds. `false_negative_rate[i]` is defined as the total weight
  of values in `predictions` above `thresholds[i]` whose corresponding entry in
  `labels` is `False`, divided by the total weight of `True` values in `labels`
  (`false_negatives[i] / (false_negatives[i] + true_positives[i])`).

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `false_positive_rate`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    labels: A `bool` `Tensor` whose shape matches `predictions`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    weights: `Tensor` whose rank is either 0, or the same rank as `labels`, and
      must be broadcastable to `labels` (i.e., all dimensions must be either
      `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that
      `false_negative_rate` should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    false_negative_rate: A float `Tensor` of shape `[len(thresholds)]`.
    update_op: An operation that increments the `false_negatives` and
      `true_positives` variables that are used in the computation of
      `false_negative_rate`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  with variable_scope.variable_scope(name, 'false_negative_rate_at_thresholds',
                                     (predictions, labels, weights)):
    values, update_ops = _streaming_confusion_matrix_at_thresholds(
        predictions, labels, thresholds, weights, includes=('fn', 'tp'))

    # Avoid division by zero.
    epsilon = _EPSILON

    def compute_fnr(fn, tp, name):
      return math_ops.div(fn, epsilon + fn + tp, name='fnr_' + name)

    fnr = compute_fnr(values['fn'], values['tp'], 'value')
    update_op = compute_fnr(update_ops['fn'], update_ops['tp'], 'update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, fnr)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return fnr, update_op


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
def streaming_recall_at_k(predictions,
                          labels,
                          k,
                          weights=None,
                          metrics_collections=None,
                          updates_collections=None,
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
  return streaming_mean(in_top_k, weights, metrics_collections,
                        updates_collections, name or _at_k_name('recall', k))


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
      k=k,
      class_id=class_id,
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


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
  return metrics.precision_at_k(
      k=k,
      class_id=class_id,
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


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
  with ops.name_scope(name, default_name,
                      (top_k_predictions, labels, weights)) as name_scope:
    return metrics_impl.precision_at_top_k(
        labels=labels,
        predictions_idx=top_k_predictions,
        class_id=class_id,
        weights=weights,
        metrics_collections=metrics_collections,
        updates_collections=updates_collections,
        name=name_scope)


def sparse_recall_at_top_k(labels,
                           top_k_predictions,
                           class_id=None,
                           weights=None,
                           metrics_collections=None,
                           updates_collections=None,
                           name=None):
  """Computes recall@k of top-k predictions with respect to sparse labels.

  If `class_id` is specified, we calculate recall by considering only the
      entries in the batch for which `class_id` is in the label, and computing
      the fraction of them for which `class_id` is in the top-k `predictions`.
  If `class_id` is not specified, we'll calculate recall as how often on
      average a class among the labels of a batch entry is in the top-k
      `predictions`.

  `sparse_recall_at_top_k` creates two local variables, `true_positive_at_<k>`
  and `false_negative_at_<k>`, that are used to compute the recall_at_k
  frequency. This frequency is ultimately returned as `recall_at_<k>`: an
  idempotent operation that simply divides `true_positive_at_<k>` by total
  (`true_positive_at_<k>` + `false_negative_at_<k>`).

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `recall_at_<k>`. Set operations applied to `top_k` and `labels` calculate the
  true positives and false negatives weighted by `weights`. Then `update_op`
  increments `true_positive_at_<k>` and `false_negative_at_<k>` using these
  values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `top_k_predictions`. Values should be in range [0, num_classes), where
      num_classes is the last dimension of `predictions`. Values outside this
      range always count towards `false_negative_at_<k>`.
    top_k_predictions: Integer `Tensor` with shape [D1, ... DN, k] where
      N >= 1. Commonly, N=1 and top_k_predictions has shape [batch size, k].
      The final dimension contains the indices of top-k labels. [D1, ... DN]
      must match `labels`.
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
  default_name = _at_k_name('recall', class_id=class_id)
  with ops.name_scope(name, default_name,
                      (top_k_predictions, labels, weights)) as name_scope:
    return metrics_impl.recall_at_top_k(
        labels=labels,
        predictions_idx=top_k_predictions,
        class_id=class_id,
        weights=weights,
        metrics_collections=metrics_collections,
        updates_collections=updates_collections,
        name=name_scope)


def _compute_recall_at_precision(tp, fp, fn, precision, name):
  """Helper function to compute recall at a given `precision`.

  Args:
    tp: The number of true positives.
    fp: The number of false positives.
    fn: The number of false negatives.
    precision: The precision for which the recall will be calculated.
    name: An optional variable_scope name.

  Returns:
    The recall at a the given `precision`.
  """
  precisions = math_ops.div(tp, tp + fp + _EPSILON)
  tf_index = math_ops.argmin(
      math_ops.abs(precisions - precision), 0, output_type=dtypes.int32)

  # Now, we have the implicit threshold, so compute the recall:
  return math_ops.div(tp[tf_index], tp[tf_index] + fn[tf_index] + _EPSILON,
                      name)


def recall_at_precision(labels,
                        predictions,
                        precision,
                        weights=None,
                        num_thresholds=200,
                        metrics_collections=None,
                        updates_collections=None,
                        name=None):
  """Computes `recall` at `precision`.

  The `recall_at_precision` function creates four local variables,
  `tp` (true positives), `fp` (false positives) and `fn` (false negatives)
  that are used to compute the `recall` at the given `precision` value. The
  threshold for the given `precision` value is computed and used to evaluate the
  corresponding `recall`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `recall`. `update_op` increments the `tp`, `fp` and `fn` counts with the
  weight of each case found in the `predictions` and `labels`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    precision: A scalar value in range `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    num_thresholds: The number of thresholds to use for matching the given
      `precision`.
    metrics_collections: An optional list of collections that `recall`
      should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    recall: A scalar `Tensor` representing the recall at the given
      `precision` value.
    update_op: An operation that increments the `tp`, `fp` and `fn`
      variables appropriately and whose value matches `recall`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      `precision` is not between 0 and 1, or if either `metrics_collections`
      or `updates_collections` are not a list or tuple.

  """
  if not 0 <= precision <= 1:
    raise ValueError('`precision` must be in the range [0, 1].')

  with variable_scope.variable_scope(name, 'recall_at_precision',
                                     (predictions, labels, weights)):
    thresholds = [
        i * 1.0 / (num_thresholds - 1) for i in range(1, num_thresholds - 1)
    ]
    thresholds = [0.0 - _EPSILON] + thresholds + [1.0 + _EPSILON]

    values, update_ops = _streaming_confusion_matrix_at_thresholds(
        labels, predictions, thresholds, weights)

    recall = _compute_recall_at_precision(values['tp'], values['fp'],
                                          values['fn'], precision, 'value')
    update_op = _compute_recall_at_precision(update_ops['tp'], update_ops['fp'],
                                             update_ops['fn'], precision,
                                             'update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, recall)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return recall, update_op


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
    update: `Operation` that increments variables appropriately, and whose
      value matches `metric`.
  """
  return metrics.average_precision_at_k(
      k=k,
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


def streaming_sparse_average_precision_at_top_k(top_k_predictions,
                                                labels,
                                                weights=None,
                                                metrics_collections=None,
                                                updates_collections=None,
                                                name=None):
  """Computes average precision@k of predictions with respect to sparse labels.

  `streaming_sparse_average_precision_at_top_k` creates two local variables,
  `average_precision_at_<k>/total` and `average_precision_at_<k>/max`, that
  are used to compute the frequency. This frequency is ultimately returned as
  `average_precision_at_<k>`: an idempotent operation that simply divides
  `average_precision_at_<k>/total` by `average_precision_at_<k>/max`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `precision_at_<k>`. Set operations applied to `top_k` and `labels` calculate
  the true positives and false positives weighted by `weights`. Then `update_op`
  increments `true_positive_at_<k>` and `false_positive_at_<k>` using these
  values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    top_k_predictions: Integer `Tensor` with shape [D1, ... DN, k] where N >= 1.
      Commonly, N=1 and `predictions_idx` has shape [batch size, k]. The final
      dimension must be set and contains the top `k` predicted class indices.
      [D1, ... DN] must match `labels`. Values should be in range
      [0, num_classes).
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies
      num_labels=1. N >= 1 and num_labels is the number of target classes for
      the associated prediction. Commonly, N=1 and `labels` has shape
      [batch_size, num_labels]. [D1, ... DN] must match `top_k_predictions`.
      Values should be in range [0, num_classes).
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
    update: `Operation` that increments variables appropriately, and whose
      value matches `metric`.

  Raises:
    ValueError: if the last dimension of top_k_predictions is not set.
  """
  return metrics_impl._streaming_sparse_average_precision_at_top_k(  # pylint: disable=protected-access
      predictions_idx=top_k_predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


@deprecated(None, 'Please switch to tf.metrics.mean.')
def streaming_mean_absolute_error(predictions,
                                  labels,
                                  weights=None,
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
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


def streaming_mean_relative_error(predictions,
                                  labels,
                                  normalizer,
                                  weights=None,
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
      normalizer=normalizer,
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


def streaming_mean_squared_error(predictions,
                                 labels,
                                 weights=None,
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
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


def streaming_root_mean_squared_error(predictions,
                                      labels,
                                      weights=None,
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
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


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
  with variable_scope.variable_scope(name, 'covariance',
                                     (predictions, labels, weights)):
    predictions, labels, weights = metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
        predictions, labels, weights)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    count_ = metrics_impl.metric_variable([], dtypes.float32, name='count')
    mean_prediction = metrics_impl.metric_variable(
        [], dtypes.float32, name='mean_prediction')
    mean_label = metrics_impl.metric_variable(
        [], dtypes.float32, name='mean_label')
    comoment = metrics_impl.metric_variable(  # C_A in update equation
        [], dtypes.float32, name='comoment')

    if weights is None:
      batch_count = math_ops.to_float(array_ops.size(labels))  # n_B in eqn
      weighted_predictions = predictions
      weighted_labels = labels
    else:
      weights = weights_broadcast_ops.broadcast_weights(weights, labels)
      batch_count = math_ops.reduce_sum(weights)  # n_B in eqn
      weighted_predictions = math_ops.multiply(predictions, weights)
      weighted_labels = math_ops.multiply(labels, weights)

    update_count = state_ops.assign_add(count_, batch_count)  # n_AB in eqn
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

    unweighted_batch_coresiduals = ((predictions - batch_mean_prediction) *
                                    (labels - batch_mean_label))
    # batch_comoment is C_B in the update equation
    if weights is None:
      batch_comoment = math_ops.reduce_sum(unweighted_batch_coresiduals)
    else:
      batch_comoment = math_ops.reduce_sum(
          unweighted_batch_coresiduals * weights)

    # View delta_comoment as = C_AB - C_A in the update equation above.
    # Since C_A is stored in a var, by how much do we need to increment that var
    # to make the var = C_AB?
    delta_comoment = (
        batch_comoment + (prev_mean_prediction - batch_mean_prediction) *
        (prev_mean_label - batch_mean_label) *
        (prev_count * batch_count / update_count))
    update_comoment = state_ops.assign_add(comoment, delta_comoment)

    covariance = array_ops.where(
        math_ops.less_equal(count_, 1.),
        float('nan'),
        math_ops.truediv(comoment, count_ - 1),
        name='covariance')
    with ops.control_dependencies([update_comoment]):
      update_op = array_ops.where(
          math_ops.less_equal(count_, 1.),
          float('nan'),
          math_ops.truediv(comoment, count_ - 1),
          name='update_op')

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
  with variable_scope.variable_scope(name, 'pearson_r',
                                     (predictions, labels, weights)):
    predictions, labels, weights = metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
        predictions, labels, weights)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    # Broadcast weights here to avoid duplicate broadcasting in each call to
    # `streaming_covariance`.
    if weights is not None:
      weights = weights_broadcast_ops.broadcast_weights(weights, labels)
    cov, update_cov = streaming_covariance(
        predictions, labels, weights=weights, name='covariance')
    var_predictions, update_var_predictions = streaming_covariance(
        predictions, predictions, weights=weights, name='variance_predictions')
    var_labels, update_var_labels = streaming_covariance(
        labels, labels, weights=weights, name='variance_labels')

    pearson_r = math_ops.truediv(
        cov,
        math_ops.multiply(
            math_ops.sqrt(var_predictions), math_ops.sqrt(var_labels)),
        name='pearson_r')
    update_op = math_ops.truediv(
        update_cov,
        math_ops.multiply(
            math_ops.sqrt(update_var_predictions),
            math_ops.sqrt(update_var_labels)),
        name='update_op')

  if metrics_collections:
    ops.add_to_collections(metrics_collections, pearson_r)

  if updates_collections:
    ops.add_to_collections(updates_collections, update_op)

  return pearson_r, update_op


# TODO(nsilberman): add a 'normalized' flag so that the user can request
# normalization if the inputs are not normalized.
def streaming_mean_cosine_distance(predictions,
                                   labels,
                                   dim,
                                   weights=None,
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
  predictions, labels, weights = metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
      predictions, labels, weights)
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())
  radial_diffs = math_ops.multiply(predictions, labels)
  radial_diffs = math_ops.reduce_sum(
      radial_diffs, reduction_indices=[
          dim,
      ], keep_dims=True)
  mean_distance, update_op = streaming_mean(radial_diffs, weights, None, None,
                                            name or 'mean_cosine_distance')
  mean_distance = math_ops.subtract(1.0, mean_distance)
  update_op = math_ops.subtract(1.0, update_op)

  if metrics_collections:
    ops.add_to_collections(metrics_collections, mean_distance)

  if updates_collections:
    ops.add_to_collections(updates_collections, update_op)

  return mean_distance, update_op


def streaming_percentage_less(values,
                              threshold,
                              weights=None,
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
      values=values,
      threshold=threshold,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


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
      num_classes=num_classes,
      predictions=predictions,
      labels=labels,
      weights=weights,
      metrics_collections=metrics_collections,
      updates_collections=updates_collections,
      name=name)


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
      math_ops.log(math_ops.cast(required_size, dtypes.float32)) / math_ops.log(
          math_ops.cast(growth_factor, dtypes.float32)))
  return math_ops.cast(math_ops.ceil(growth_factor**exponent), dtypes.int32)


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
  length of the concatenated axis.

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

    fixed_shape = [dim.value for n, dim in enumerate(values_shape) if n != axis]
    if any(value is None for value in fixed_shape):
      raise ValueError('all dimensions of `values` other than the dimension to '
                       'concatenate along must have statically known size')

    # We move `axis` to the front of the internal array so assign ops can be
    # applied to contiguous slices
    init_size = 0 if max_size is None else max_size
    init_shape = [init_size] + fixed_shape
    array = metrics_impl.metric_variable(
        init_shape, values.dtype, validate_shape=False, name='array')
    size = metrics_impl.metric_variable([], dtypes.int32, name='size')

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


def count(values,
          weights=None,
          metrics_collections=None,
          updates_collections=None,
          name=None):
  """Computes the number of examples, or sum of `weights`.

  When evaluating some metric (e.g. mean) on one or more subsets of the data,
  this auxiliary metric is useful for keeping track of how many examples there
  are in each subset.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    values: A `Tensor` of arbitrary dimensions. Only it's shape is used.
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
    count: A `Tensor` representing the current value of the metric.
    update_op: An operation that accumulates the metric from a batch of data.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`,
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
  """

  with variable_scope.variable_scope(name, 'count', (values, weights)):
    count_ = metrics_impl.metric_variable([], dtypes.float32, name='count')

    if weights is None:
      num_values = math_ops.to_float(array_ops.size(values))
    else:
      _, _, weights = metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
          predictions=values,
          labels=None,
          weights=weights)
      weights = weights_broadcast_ops.broadcast_weights(
          math_ops.to_float(weights), values)
      num_values = math_ops.reduce_sum(weights)

    with ops.control_dependencies([values]):
      update_op = state_ops.assign_add(count_, num_values)

    if metrics_collections:
      ops.add_to_collections(metrics_collections, count_)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return count_, update_op


__all__ = [
    'aggregate_metric_map',
    'aggregate_metrics',
    'count',
    'recall_at_precision',
    'sparse_recall_at_top_k',
    'streaming_accuracy',
    'streaming_auc',
    'streaming_curve_points',
    'streaming_dynamic_auc',
    'streaming_false_negative_rate',
    'streaming_false_negative_rate_at_thresholds',
    'streaming_false_negatives',
    'streaming_false_negatives_at_thresholds',
    'streaming_false_positive_rate',
    'streaming_false_positive_rate_at_thresholds',
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
    'streaming_sparse_average_precision_at_top_k',
    'streaming_sparse_precision_at_k',
    'streaming_sparse_precision_at_top_k',
    'streaming_sparse_recall_at_k',
    'streaming_specificity_at_sensitivity',
    'streaming_true_negatives',
    'streaming_true_negatives_at_thresholds',
    'streaming_true_positives',
    'streaming_true_positives_at_thresholds',
]
