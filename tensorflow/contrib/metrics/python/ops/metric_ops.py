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

from tensorflow.contrib import framework
from tensorflow.contrib.metrics.python.ops import set_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.util.all_util import make_all


def _mask_to_weights(mask=None):
  """Converts a binary mask to a set of weights.

  Args:
    mask: A binary `Tensor`.

  Returns:
    The corresponding set of weights if `mask` is not `None`, otherwise `None`.
  """
  if mask is not None:
    check_ops.assert_type(mask, dtypes.bool)
    weights = math_ops.logical_not(mask)
  else:
    weights = None
  return weights


def _create_local(name, shape=None, collections=None):
  """Creates a new local variable.

  Args:
    name: The name of the new or existing variable.
    shape: Shape of the new or existing variable.
    collections: A list of collection names to which the Variable will be added.

  Returns:
    The created variable.
  """
  # Make sure local variables are added to tf.GraphKeys.LOCAL_VARIABLES
  collections = list(collections or [])
  collections += [ops.GraphKeys.LOCAL_VARIABLES]
  return variables.Variable(
      initial_value=array_ops.zeros(shape),
      name=name,
      trainable=False,
      collections=collections)


def _remove_squeezable_dimensions(predictions, labels):
  predictions_rank = predictions.get_shape().ndims
  labels_rank = labels.get_shape().ndims
  if not (labels_rank is None or predictions_rank is None):
    if labels_rank == (predictions_rank + 1):
      labels = array_ops.squeeze(labels, [-1])
    elif predictions_rank == (labels_rank + 1):
      predictions = array_ops.squeeze(predictions, [-1])
  return predictions, labels


def _count_condition(values, ignore_mask=None, metrics_collections=None,
                     updates_collections=None):
  """Computes the total number of cases where the given values are True.

  Args:
    values: A binary `Tensor` of arbitrary size.
    ignore_mask: An optional, binary tensor whose size matches 'values'.
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.

  Returns:
    value_tensor: A tensor representing the current value of the metric.
    update_op: An operation that accumulates the error from a batch of data.

  Raises:
    ValueError: If either `metrics_collections` or `updates_collections` are not
      a list or tuple.
  """
  check_ops.assert_type(values, dtypes.bool)
  count = _create_local('count', shape=[])

  if ignore_mask is not None:
    values.get_shape().assert_is_compatible_with(ignore_mask.get_shape())
    check_ops.assert_type(ignore_mask, dtypes.bool)
    values = math_ops.select(
        ignore_mask,
        array_ops.zeros_like(values),
        values)
  values = math_ops.to_float(values)

  value_tensor = array_ops.identity(count)
  update_op = state_ops.assign_add(count, math_ops.reduce_sum(values))

  if metrics_collections:
    ops.add_to_collections(metrics_collections, value_tensor)

  if updates_collections:
    ops.add_to_collections(updates_collections, update_op)

  return value_tensor, update_op


def _streaming_true_positives(predictions, labels, ignore_mask=None,
                              metrics_collections=None,
                              updates_collections=None,
                              name=None):
  """Computes the total number of true_positives.

  Args:
    predictions: The predicted values, a binary `Tensor` of arbitrary
      dimensions.
    labels: The ground truth values, a binary `Tensor` whose dimensions must
      match `predictions`.
    ignore_mask: An optional, binary tensor whose size matches 'predictions'.
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_op_scope name.

  Returns:
    value_tensor: A tensor representing the current value of the metric.
    update_op: An operation that accumulates the error from a batch of data.

  Raises:
    ValueError: If either `metrics_collections` or `updates_collections` are not
      a list or tuple.
  """
  with variable_scope.variable_op_scope(
      [predictions, labels], name, 'true_positives'):

    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    is_true_positive = math_ops.logical_and(math_ops.equal(labels, 1),
                                            math_ops.equal(predictions, 1))
    return _count_condition(is_true_positive, ignore_mask, metrics_collections,
                            updates_collections)


def _streaming_false_positives(predictions, labels, ignore_mask=None,
                               metrics_collections=None,
                               updates_collections=None,
                               name=None):
  """Computes the total number of false positives.

  Args:
    predictions: The predicted values, a binary `Tensor` of arbitrary
      dimensions.
    labels: The ground truth values, a binary `Tensor` whose dimensions must
      match `predictions`.
    ignore_mask: An optional, binary tensor whose size matches 'predictions'.
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_op_scope name.

  Returns:
    value_tensor: A tensor representing the current value of the metric.
    update_op: An operation that accumulates the error from a batch of data.

  Raises:
    ValueError: If either `metrics_collections` or `updates_collections` are not
      a list or tuple.
  """
  with variable_scope.variable_op_scope(
      [predictions, labels], name, 'false_positives'):

    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    is_false_positive = math_ops.logical_and(math_ops.equal(labels, 0),
                                             math_ops.equal(predictions, 1))
    return _count_condition(is_false_positive, ignore_mask,
                            metrics_collections, updates_collections)


def _streaming_false_negatives(predictions, labels, ignore_mask=None,
                               metrics_collections=None,
                               updates_collections=None,
                               name=None):
  """Computes the total number of false positives.

  Args:
    predictions: The predicted values, a binary `Tensor` of arbitrary
      dimensions.
    labels: The ground truth values, a binary `Tensor` whose dimensions must
      match `predictions`.
    ignore_mask: An optional, binary tensor whose size matches 'predictions'.
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_op_scope name.

  Returns:
    value_tensor: A tensor representing the current value of the metric.
    update_op: An operation that accumulates the error from a batch of data.

  Raises:
    ValueError: If either `metrics_collections` or `updates_collections` are not
      a list or tuple.
  """
  with variable_scope.variable_op_scope(
      [predictions, labels], name, 'false_negatives'):

    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    is_false_negative = math_ops.logical_and(math_ops.equal(labels, 1),
                                             math_ops.equal(predictions, 0))
    return _count_condition(is_false_negative, ignore_mask,
                            metrics_collections, updates_collections)


def streaming_mean(values, weights=None, metrics_collections=None,
                   updates_collections=None, name=None):
  """Computes the (weighted) mean of the given values.

  The `streaming_mean` function creates two local variables, `total` and `count`
  that are used to compute the average of `values`. This average is ultimately
  returned as `mean` which is an idempotent operation that simply divides
  `total` by `count`. To facilitate the estimation of a mean over a stream
  of data, the function creates an `update_op` operation whose behavior is
  dependent on the value of `weights`. If `weights` is None, then `update_op`
  increments `total` with the reduced sum of `values` and increments `count`
  with the number of elements in `values`. If `weights` is not `None`, then
  `update_op` increments `total` with the reduced sum of the product of `values`
  and `weights` and increments `count` with the reduced sum of weights.
  In addition to performing the updates, `update_op` also returns the
  `mean`.

  Args:
    values: A `Tensor` of arbitrary dimensions.
    weights: An optional set of weights of the same shape as `values`. If
      `weights` is not None, the function computes a weighted mean.
    metrics_collections: An optional list of collections that `mean`
      should be added to.
    updates_collections: An optional list of collections that `update_op`
      should be added to.
    name: An optional variable_op_scope name.

  Returns:
    mean: A tensor representing the current mean, the value of `total` divided
      by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_value`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
  """
  with variable_scope.variable_op_scope([values, weights], name, 'mean'):
    values = math_ops.to_float(values)

    total = _create_local('total', shape=[])
    count = _create_local('count', shape=[])

    if weights is not None:
      values.get_shape().assert_is_compatible_with(weights.get_shape())
      weights = math_ops.to_float(weights)
      values = math_ops.mul(values, weights)
      num_values = math_ops.reduce_sum(weights)
    else:
      num_values = math_ops.to_float(array_ops.size(values))

    total_compute_op = state_ops.assign_add(total, math_ops.reduce_sum(values))
    count_compute_op = state_ops.assign_add(count, num_values)

    def compute_mean(total, count, name):
      return math_ops.select(math_ops.greater(count, 0),
                             math_ops.div(total, count),
                             0, name)

    mean = compute_mean(total, count, 'value')
    with ops.control_dependencies([total_compute_op, count_compute_op]):
      update_op = compute_mean(total, count, 'update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, mean)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return mean, update_op


def streaming_accuracy(predictions, labels, weights=None,
                       metrics_collections=None, updates_collections=None,
                       name=None):
  """Calculates how often `predictions` matches `labels`.

  The `streaming_accuracy` function creates two local variables, `total` and
  `count` that are used to compute the frequency with which `predictions`
  matches `labels`. This frequency is ultimately returned as `accuracy`: an
  idempotent operation that simply divides `total` by `count`.
  To facilitate the estimation of the accuracy over a stream of data, the
  function utilizes two operations. First, an `is_correct` operation that
  computes a tensor whose shape matches `predictions` and whose elements are
  set to 1.0 when the corresponding values of `predictions` and `labels match
  and 0.0 otherwise. Second, an `update_op` operation whose behavior is
  dependent on the value of `weights`. If `weights` is None, then `update_op`
  increments `total` with the number of elements of `predictions` that match
  `labels` and increments `count` with the number of elements in `values`. If
  `weights` is not `None`, then `update_op` increments `total` with the reduced
  sum of the product of `weights` and `is_correct` and increments `count` with
  the reduced sum of `weights`. In addition to performing the updates,
  `update_op` also returns the `accuracy` value.

  Args:
    predictions: The predicted values, a `Tensor` of any shape.
    labels: The ground truth values, a `Tensor` whose shape matches
      `predictions`.
    weights: An optional set of weights whose shape matches `predictions`
      which, when not `None`, produces a weighted mean accuracy.
    metrics_collections: An optional list of collections that `accuracy` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_op_scope name.

  Returns:
    accuracy: A tensor representing the accuracy, the value of `total` divided
      by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `accuracy`.

  Raises:
    ValueError: If the dimensions of `predictions` and `labels` don't match or
      if `weight` is not `None` and its shape doesn't match `predictions` or
      if either `metrics_collections` or `updates_collections` are not
      a list or tuple.
  """
  predictions, labels = _remove_squeezable_dimensions(predictions, labels)
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())
  is_correct = math_ops.to_float(math_ops.equal(predictions, labels))
  return streaming_mean(is_correct, weights, metrics_collections,
                        updates_collections, name or 'accuracy')


def streaming_precision(predictions, labels, ignore_mask=None,
                        metrics_collections=None, updates_collections=None,
                        name=None):
  """Computes the precision of the predictions with respect to the labels.

  The `streaming_precision` function creates two local variables,
  `true_positives` and `false_positives`, that are used to compute the
  precision. This value is ultimately returned as `precision`, an idempotent
  operation that simply divides `true_positives` by the sum of `true_positives`
  and `false_positives`. To facilitate the calculation of the precision over a
  stream of data, the function creates an `update_op` operation whose behavior
  is dependent on the value of `ignore_mask`. If `ignore_mask` is None, then
  `update_op` increments `true_positives` with the number of elements of
  `predictions` and `labels` that are both `True` and increments
  `false_positives` with the number of elements of `predictions` that are `True`
  whose corresponding `labels` element is `False`. If `ignore_mask` is not
  `None`, then the increments for `true_positives` and `false_positives` are
  only computed using elements of `predictions` and `labels` whose corresponding
  values in `ignore_mask` are `False`. In addition to performing the updates,
  `update_op` also returns the value of `precision`.

  Args:
    predictions: The predicted values, a binary `Tensor` of arbitrary shape.
    labels: The ground truth values, a binary `Tensor` whose dimensions must
      match `predictions`.
    ignore_mask: An optional, binary tensor whose size matches `predictions`.
    metrics_collections: An optional list of collections that `precision` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_op_scope name.

  Returns:
    precision: Scalar float `Tensor` with the value of `true_positives`
      divided by the sum of `true_positives` and `false_positives`.
    update_op: `Operation` that increments `true_positives` and
      `false_positives` variables appropriately and whose value matches
      `precision`.

  Raises:
    ValueError: If the dimensions of `predictions` and `labels` don't match or
      if `ignore_mask` is not `None` and its shape doesn't match `predictions`
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
  """
  with variable_scope.variable_op_scope(
      [predictions, labels], name, 'precision'):

    predictions, labels = _remove_squeezable_dimensions(predictions, labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    true_positives, true_positives_update_op = _streaming_true_positives(
        predictions, labels, ignore_mask, metrics_collections=None,
        updates_collections=None, name=None)
    false_positives, false_positives_update_op = _streaming_false_positives(
        predictions, labels, ignore_mask, metrics_collections=None,
        updates_collections=None, name=None)

    def compute_precision(name):
      return math_ops.select(
          math_ops.greater(true_positives + false_positives, 0),
          math_ops.div(true_positives, true_positives + false_positives),
          0,
          name)

    precision = compute_precision('value')
    with ops.control_dependencies([true_positives_update_op,
                                   false_positives_update_op]):
      update_op = compute_precision('update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, precision)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return precision, update_op


def streaming_recall(predictions, labels, ignore_mask=None,
                     metrics_collections=None, updates_collections=None,
                     name=None):
  """Computes the recall of the predictions with respect to the labels.

  The `streaming_recall` function creates two local variables,
  `true_positives` and `false_negatives`, that are used to compute the
  recall. This value is ultimately returned as `recall`, an idempotent
  operation that simply divides `true_positives` by the sum of `true_positives`
  and `false_negatives`. To facilitate the calculation of the recall over a
  stream of data, the function creates an `update_op` operation whose behavior
  is dependent on the value of `ignore_mask`. If `ignore_mask` is None, then
  `update_op` increments `true_positives` with the number of elements of
  `predictions` and `labels` that are both `True` and increments
  `false_negatives` with the number of elements of `predictions` that are
  `False` whose corresponding `labels` element is `False`. If `ignore_mask` is
  not `None`, then the increments for `true_positives` and `false_negatives` are
  only computed using elements of `predictions` and `labels` whose corresponding
  values in `ignore_mask` are `False`. In addition to performing the updates,
  `update_op` also returns the value of `recall`.

  Args:
    predictions: The predicted values, a binary `Tensor` of arbitrary shape.
    labels: The ground truth values, a binary `Tensor` whose dimensions must
      match `predictions`.
    ignore_mask: An optional, binary tensor whose size matches `predictions`.
    metrics_collections: An optional list of collections that `precision` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_op_scope name.

  Returns:
    recall: Scalar float `Tensor` with the value of `true_positives` divided
      by the sum of `true_positives` and `false_negatives`.
    update_op: `Operation` that increments `true_positives` and
      `false_negatives` variables appropriately and whose value matches
      `recall`.

  Raises:
    ValueError: If the dimensions of `predictions` and `labels` don't match or
      if `ignore_mask` is not `None` and its shape doesn't match `predictions`
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
  """
  with variable_scope.variable_op_scope([predictions, labels], name, 'recall'):
    predictions, labels = _remove_squeezable_dimensions(predictions, labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    true_positives, true_positives_update_op = _streaming_true_positives(
        predictions, labels, ignore_mask, metrics_collections=None,
        updates_collections=None, name=None)
    false_negatives, false_negatives_update_op = _streaming_false_negatives(
        predictions, labels, ignore_mask, metrics_collections=None,
        updates_collections=None, name=None)

    def compute_recall(true_positives, false_negatives, name):
      return math_ops.select(
          math_ops.greater(true_positives + false_negatives, 0),
          math_ops.div(true_positives, true_positives + false_negatives),
          0,
          name)

    recall = compute_recall(true_positives, false_negatives, 'value')
    with ops.control_dependencies([true_positives_update_op,
                                   false_negatives_update_op]):
      update_op = compute_recall(true_positives, false_negatives, 'update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, recall)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return recall, update_op


def _tp_fn_tn_fp(predictions, labels, thresholds, ignore_mask=None):
  """Computes true_positives, false_negatives, true_negatives, false_positives.

  The `_tp_fn_tn_fp` function creates four local variables, `true_positives`,
  `true_negatives`, `false_positives` and `false_negatives`.
  `true_positive[i]` is defined` as the number of values in `predictions` above
  `thresholds[i]` whose corresponding entry in `labels` is `True`.
  `false_negatives[i]` is defined` as the number of values in `predictions` at
  most `thresholds[i]` whose corresponding entry in `labels` is `True`.
  `true_negatives[i]` is defined` as the number of values in `predictions` at
  most `thresholds[i]` whose corresponding entry in `labels` is `False`.
  `false_positives[i]` is defined` as the number of values in `predictions`
  above `thresholds[i]` whose corresponding entry in `labels` is `False`.

  These four variables are updated through the `update_op`.
  The streaming behavior is that the values of the variables after a few
  `update_op`s is the same as if the inputs had been concatenated and a single
  `update_op` had been performed.

  If `ignore_mask` is not `None`, then the increment of the variables is
  performed using only the elements of `predictions` and `labels` whose
  corresponding value in `ignore_mask` is `False`.

  Args:
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    labels: A binary `Tensor` whose shape matches `predictions`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    ignore_mask: An optional, binary tensor whose size matches `predictions`.

  Returns:
    true_positive: A variable of shape [len(thresholds)].
    false_negative: A variable of shape [len(thresholds)].
    true_negatives: A variable of shape [len(thresholds)].
    false_positives: A variable of shape [len(thresholds)].
    true_positives_update_op: An operation that increments the `true_positives`.
    false_negative_update_op: An operation that increments the `false_negative`.
    true_negatives_update_op: An operation that increments the `true_negatives`.
    false_positives_update_op: An operation that increments the
      `false_positives`.

  Raises:
    ValueError: If the shape of `predictions` and `labels` do not match or if
      `ignore_mask` is not `None` and its shape doesn't match `predictions`
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
  """
  predictions, labels = _remove_squeezable_dimensions(predictions, labels)
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())

  num_thresholds = len(thresholds)

  # Reshape predictions and labels to be column vectors
  predictions = array_ops.reshape(predictions, [-1, 1])
  labels = array_ops.reshape(labels, [-1, 1])

  # Use static shape if known.
  num_predictions = predictions.get_shape().as_list()[0]

  # Otherwise use dynamic shape.
  if num_predictions is None:
    num_predictions = array_ops.shape(predictions)[0]
  thresh_tiled = array_ops.tile(
      array_ops.expand_dims(array_ops.constant(thresholds), [1]),
      array_ops.pack([1, num_predictions]))

  # Tile the predictions after thresholding them across different thresholds.
  pred_tiled = math_ops.cast(
      math_ops.greater(
          array_ops.tile(
              array_ops.transpose(predictions), [num_thresholds, 1]),
          thresh_tiled),
      dtype=dtypes.int32)
  # Tile labels by number of thresholds
  labels_tiled = array_ops.tile(array_ops.transpose(labels),
                                [num_thresholds, 1])

  true_positives = _create_local('true_positives', shape=[num_thresholds])
  false_negatives = _create_local('false_negatives', shape=[num_thresholds])
  true_negatives = _create_local('true_negatives', shape=[num_thresholds])
  false_positives = _create_local('false_positives', shape=[num_thresholds])

  is_true_positive = math_ops.to_float(
      math_ops.logical_and(
          math_ops.equal(labels_tiled, 1), math_ops.equal(pred_tiled, 1)))
  is_false_negative = math_ops.to_float(
      math_ops.logical_and(
          math_ops.equal(labels_tiled, 1), math_ops.equal(pred_tiled, 0)))
  is_false_positive = math_ops.to_float(
      math_ops.logical_and(
          math_ops.equal(labels_tiled, 0), math_ops.equal(pred_tiled, 1)))
  is_true_negative = math_ops.to_float(
      math_ops.logical_and(
          math_ops.equal(labels_tiled, 0), math_ops.equal(pred_tiled, 0)))

  if ignore_mask is not None:
    ignore_mask = array_ops.reshape(ignore_mask, [-1, 1])
    mask_tiled = array_ops.tile(array_ops.transpose(ignore_mask),
                                [num_thresholds, 1])

    labels_tiled.get_shape().assert_is_compatible_with(mask_tiled.get_shape())
    check_ops.assert_type(mask_tiled, dtypes.bool)
    is_true_positive = math_ops.select(
        mask_tiled,
        array_ops.zeros_like(labels_tiled, dtype=dtypes.float32),
        is_true_positive)
    is_false_negative = math_ops.select(
        mask_tiled,
        array_ops.zeros_like(labels_tiled, dtype=dtypes.float32),
        is_false_negative)
    is_false_positive = math_ops.select(
        mask_tiled,
        array_ops.zeros_like(labels_tiled, dtype=dtypes.float32),
        is_false_positive)
    is_true_negative = math_ops.select(
        mask_tiled,
        array_ops.zeros_like(labels_tiled, dtype=dtypes.float32),
        is_true_negative)

  true_positives_update_op = state_ops.assign_add(
      true_positives, math_ops.reduce_sum(is_true_positive, 1))
  false_negatives_update_op = state_ops.assign_add(
      false_negatives, math_ops.reduce_sum(is_false_negative, 1))
  true_negatives_update_op = state_ops.assign_add(
      true_negatives, math_ops.reduce_sum(is_true_negative, 1))
  false_positives_update_op = state_ops.assign_add(
      false_positives, math_ops.reduce_sum(is_false_positive, 1))

  return (true_positives, false_negatives, true_negatives, false_positives,
          true_positives_update_op, false_negatives_update_op,
          true_negatives_update_op, false_positives_update_op)


def streaming_auc(predictions, labels, ignore_mask=None, num_thresholds=200,
                  metrics_collections=None, updates_collections=None,
                  name=None):
  """Computes the approximate AUC via a Riemann sum.

  The `streaming_auc` function creates four local variables, `true_positives`,
  `true_negatives`, `false_positives` and `false_negatives` that are used to
  compute the AUC. To discretize the AUC curve, a linearly spaced set of
  thresholds is used to compute pairs of recall and precision values. The area
  under the curve is therefore computed using the height of the recall values
  by the false positive rate.

  This value is ultimately returned as `auc`, an idempotent
  operation the computes the area under a discretized curve of precision versus
  recall values (computed using the afformentioned variables). The
  `num_thresholds` variable controls the degree of discretization with larger
  numbers of thresholds more closely approximating the true AUC.

  To faciliate the estimation of the AUC over a stream of data, the function
  creates an `update_op` operation whose behavior is dependent on the value of
  `ignore_mask`. If `ignore_mask` is None, then `update_op` increments the
  `true_positives`, `true_negatives`, `false_positives` and `false_negatives`
  counts with the number of each found in the current `predictions` and `labels`
  `Tensors`. If `ignore_mask` is not `None`, then the increment is performed
  using only the elements of `predictions` and `labels` whose corresponding
  value in `ignore_mask` is `False`. In addition to performing the updates,
  `update_op` also returns the `auc`.

  Args:
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    labels: A binary `Tensor` whose shape matches `predictions`.
    ignore_mask: An optional, binary tensor whose size matches `predictions`.
    num_thresholds: The number of thresholds to use when discretizing the roc
      curve.
    metrics_collections: An optional list of collections that `auc` should be
      added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_op_scope name.

  Returns:
    auc: A scalar tensor representing the current area-under-curve.
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables
      appropriately and whose value matches `auc`.

  Raises:
    ValueError: If the shape of `predictions` and `labels` do not match or if
      `ignore_mask` is not `None` and its shape doesn't match `predictions` or
      if either `metrics_collections` or `updates_collections` are not a list or
      tuple.
  """
  with variable_scope.variable_op_scope([predictions, labels], name, 'auc'):
    kepsilon = 1e-7  # to account for floating point imprecisions
    thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                  for i in range(num_thresholds-2)]
    thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

    (true_positives, false_negatives, true_negatives, false_positives,
     true_positives_compute_op, false_negatives_compute_op,
     true_negatives_compute_op, false_positives_compute_op) = _tp_fn_tn_fp(
         predictions, labels, thresholds, ignore_mask)

    epsilon = 1.0e-6
    assert array_ops.squeeze(
        false_positives).get_shape().as_list()[0] == num_thresholds
    # Add epsilons to avoid dividing by 0.
    false_positive_rate = math_ops.div(
        false_positives,
        false_positives + true_negatives + epsilon)
    recall = math_ops.div(true_positives + epsilon,
                          true_positives + false_negatives + epsilon)

    def compute_auc(name):
      return math_ops.reduce_sum(math_ops.mul(
          false_positive_rate[:num_thresholds - 1] - false_positive_rate[1:],
          (recall[:num_thresholds - 1] + recall[1:]) / 2.), name=name)

    # sum up the areas of all the trapeziums
    auc = compute_auc('value')
    with ops.control_dependencies([true_positives_compute_op,
                                   false_negatives_compute_op,
                                   true_negatives_compute_op,
                                   false_positives_compute_op]):
      update_op = compute_auc('update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, auc)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return auc, update_op


def streaming_precision_at_thresholds(predictions, labels, thresholds,
                                      ignore_mask=None,
                                      metrics_collections=None,
                                      updates_collections=None, name=None):
  """Computes precision values for different `thresholds` on `predictions`.

  The `streaming_precision_at_thresholds` function creates four local variables,
  `true_positives`, `true_negatives`, `false_positives` and `false_negatives`
  for various values of thresholds.
  `precision[i]` is defined as the number of values in `predictions` above
  `thresholds[i]` whose corresponding entry in `labels` is `True`
  (`true_positives[i]`) divided by the number of values in `predictions`
  above `thresholds[i]` (`true_positives[i] + false_positives[i]`).

  If `ignore_mask` is not None then only values whose corresponding value in
  `ignore_mask` is `False` are considered.

  `precision` is returned along with an `update_op` whose value equals that of
  `precision`.

  Args:
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    labels: A binary `Tensor` whose shape matches `predictions`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    ignore_mask: An optional, binary tensor whose size matches `predictions`.
    metrics_collections: An optional list of collections that `auc` should be
      added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_op_scope name.

  Returns:
    precision: A float tensor of shape [len(thresholds)].
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables that
      are used in the computation of `precision`.

  Raises:
    ValueError: If the shape of `predictions` and `labels` do not match or if
      `ignore_mask` is not `None` and its shape doesn't match `predictions`
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
  """
  with variable_scope.variable_op_scope([predictions, labels], name,
                                        'precision_at_thresholds'):
    (true_positives, _, _, false_positives, true_positives_compute_op, _, _,
     false_positives_compute_op,) = _tp_fn_tn_fp(
         predictions, labels, thresholds, ignore_mask)

    # avoid division by zero
    epsilon = 1e-7
    def compute_precision(name):
      precision = math_ops.div(true_positives,
                               epsilon + true_positives + false_positives,
                               name='precision_' + name)
      return precision

    precision = compute_precision('value')
    with ops.control_dependencies([true_positives_compute_op,
                                   false_positives_compute_op]):
      update_op = compute_precision('update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, precision)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return precision, update_op


def streaming_recall_at_thresholds(predictions, labels, thresholds,
                                   ignore_mask=None, metrics_collections=None,
                                   updates_collections=None, name=None):
  """Computes various recall values for different `thresholds` on `predictions`.

  The `streaming_recall_at_thresholds` function creates four local variables,
  `true_positives`, `true_negatives`, `false_positives` and `false_negatives`
  for various values of thresholds.
  `recall[i]` is defined as the number of values in `predictions` above
  `thresholds[i]` whose corresponding entry in `labels` is `True`
  (`true_positives[i]`) divided by the number of True values in `labels`
  (`true_positives[i] + false_negatives[i]`).

  If `ignore_mask` is not None then only values whose corresponding value in
  `ignore_mask` is `False` are considered.

  `recall` are returned along with an `update_op` whose value equals that of
  `recall`.

  Args:
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    labels: A binary `Tensor` whose shape matches `predictions`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    ignore_mask: An optional, binary tensor whose size matches `predictions`.
    metrics_collections: An optional list of collections that `auc` should be
      added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_op_scope name.

  Returns:
    recall: A float tensor of shape [len(thresholds)].
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables that
      are used in the computation of `recall`.

  Raises:
    ValueError: If the shape of `predictions` and `labels` do not match or if
      `ignore_mask` is not `None` and its shape doesn't match `predictions`
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
  """
  with variable_scope.variable_op_scope([predictions, labels], name,
                                        'recall_at_thresholds'):
    (true_positives, false_negatives, _, _, true_positives_compute_op,
     false_negatives_compute_op, _, _,) = _tp_fn_tn_fp(
         predictions, labels, thresholds, ignore_mask)

    # avoid division by zero
    epsilon = 1e-7
    def compute_recall(name):
      recall = math_ops.div(true_positives,
                            epsilon + true_positives + false_negatives,
                            name='recall_' + name)
      return recall

    recall = compute_recall('value')
    with ops.control_dependencies([true_positives_compute_op,
                                   false_negatives_compute_op]):
      update_op = compute_recall('update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, recall)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return recall, update_op


def streaming_recall_at_k(predictions, labels, k, ignore_mask=None,
                          metrics_collections=None, updates_collections=None,
                          name=None):
  """Computes the recall@k of the predictions with respect to dense labels.

  The `streaming_recall_at_k` function creates two local variables, `total` and
  `count`, that are used to compute the recall@k frequency. This frequency is
  ultimately returned as `recall_at_<k>`: an idempotent operation that simply
  divides `total` by `count`. To facilitate the estimation of recall@k over a
  stream of data, the function utilizes two operations. First, an `in_top_k`
  operation computes a tensor with shape [batch_size] whose elements indicate
  whether or not the corresponding label is in the top `k` predictions of the
  `predictions` `Tensor`. Second, an `update_op` operation whose behavior is
  dependent on the value of `ignore_mask`. If `ignore_mask` is None, then
  `update_op` increments `total` with the number of elements of `in_top_k` that
  are set to `True` and increments `count` with the batch size. If `ignore_mask`
  is not `None`, then `update_op` increments `total` with the number of elements
  in `in_top_k` that are `True` whose corresponding element in `ignore_mask` is
  `False`. In addition to performing the updates, `update_op` also returns the
  recall value.

  Args:
    predictions: A floating point tensor of dimension [batch_size, num_classes]
    labels: A tensor of dimension [batch_size] whose type is in `int32`,
      `int64`.
    k: The number of top elements to look at for computing precision.
    ignore_mask: An optional, binary tensor whose size matches `labels`. If an
      element of `ignore_mask` is True, the corresponding prediction and label
      pair is used to compute the metrics. Otherwise, the pair is ignored.
    metrics_collections: An optional list of collections that `recall_at_k`
      should be added to.
    updates_collections: An optional list of collections `update_op` should be
      added to.
    name: An optional variable_op_scope name.

  Returns:
    recall_at_k: A tensor representing the recall@k, the fraction of labels
      which fall into the top `k` predictions.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `recall_at_k`.

  Raises:
    ValueError: If the dimensions of `predictions` and `labels` don't match or
      if `ignore_mask` is not `None` and its shape doesn't match `predictions`
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
  """
  in_top_k = math_ops.to_float(nn.in_top_k(predictions, labels, k))
  return streaming_mean(in_top_k, _mask_to_weights(ignore_mask),
                        metrics_collections,
                        updates_collections,
                        name or ('recall_at_%d' % k))


# TODO(ptucker): Validate range of values in labels?
def streaming_sparse_recall_at_k(predictions,
                                 labels,
                                 k,
                                 class_id=None,
                                 ignore_mask=None,
                                 metrics_collections=None,
                                 updates_collections=None,
                                 name=None):
  """Computes recall@k of the predictions with respect to sparse labels.

  If `class_id` is specified, we calculate recall by considering only the
      entries in the batch for which `class_id` is in the label, and computing
      the fraction of them for which `class_id` is in the top-k `predictions`.
  If `class_id` is not specified, we'll calculate recall as how often on
      average a class among the labels of a batch entry is in the top-k
      `predictions`.

  `streaming_sparse_recall_at_k` creates two local variables,
  `true_positive_at_<k>` and `false_negative_at_<k>`, that are used to compute
  the recall_at_k frequency. This frequency is ultimately returned as
  `recall_at_<k>`: an idempotent operation that simply divides
  `true_positive_at_<k>` by total (`true_positive_at_<k>` + `recall_at_<k>`). To
  facilitate the estimation of recall@k over a stream of data, the function
  utilizes three steps.
  * A `top_k` operation computes a tensor whose elements indicate the top `k`
    predictions of the `predictions` `Tensor`.
  * Set operations are applied to `top_k` and `labels` to calculate true
    positives and false negatives.
  * An `update_op` operation increments `true_positive_at_<k>` and
    `false_negative_at_<k>`. It also returns the recall value.

  Args:
    predictions: Float `Tensor` with shape [D1, ... DN, num_classes] where
      N >= 1. Commonly, N=1 and predictions has shape [batch size, num_classes].
      The final dimension contains the logit values for each class. [D1, ... DN]
      must match `labels`.
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match `labels`.
      Values should be in range [0, num_classes], where num_classes is the last
      dimension of `predictions`.
    k: Integer, k for @k metric.
    class_id: Integer class ID for which we want binary metrics. This should be
      in range [0, num_classes], where num_classes is the last dimension of
      `predictions`.
    ignore_mask: An optional, binary tensor whose shape is broadcastable to the
      the first [D1, ... DN] dimensions of `predictions_idx` and `labels`.
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of new update operation, and namespace for other dependant ops.

  Returns:
    recall: Scalar `float64` `Tensor` with the value of `true_positives` divided
      by the sum of `true_positives` and `false_negatives`.
    update_op: `Operation` that increments `true_positives` and
      `false_negatives` variables appropriately, and whose value matches
      `recall`.
  """
  default_name = 'recall_at_%d' % k
  if class_id is not None:
    default_name = '%s_class%d' % (default_name, class_id)

  with ops.op_scope([predictions, labels], name, default_name) as scope:
    _, top_k_idx = nn.top_k(predictions, k)
    top_k_idx = math_ops.to_int64(top_k_idx)
    tp, tp_update = _streaming_sparse_true_positive_at_k(
        predictions_idx=top_k_idx, labels=labels, k=k, class_id=class_id,
        ignore_mask=ignore_mask)
    fn, fn_update = _streaming_sparse_false_negative_at_k(
        predictions_idx=top_k_idx, labels=labels, k=k, class_id=class_id,
        ignore_mask=ignore_mask)

    metric = math_ops.div(tp, math_ops.add(tp, fn), name=scope)
    update = math_ops.div(
        tp_update, math_ops.add(tp_update, fn_update), name='update')
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
                                    ignore_mask=None,
                                    metrics_collections=None,
                                    updates_collections=None,
                                    name=None):
  """Computes precision@k of the predictions with respect to sparse labels.

  If `class_id` is specified, we calculate precision by considering only the
      entries in the batch for which `class_id` is in the top-k highest
      `predictions`, and computing the fraction of them for which `class_id` is
      indeed a correct label.
  If `class_id` is not specified, we'll calculate precision as how often on
      average a class among the top-k classes with the highest predicted values
      of a batch entry is correct and can be found in the label for that entry.

  `streaming_sparse_precision_at_k` creates two local variables,
  `true_positive_at_<k>` and `false_positive_at_<k>`, that are used to compute
  the precision@k frequency. This frequency is ultimately returned as
  `recall_at_<k>`: an idempotent operation that simply divides
  `true_positive_at_<k>` by total (`true_positive_at_<k>` + `recall_at_<k>`). To
  facilitate the estimation of precision@k over a stream of data, the function
  utilizes three steps.
  * A `top_k` operation computes a tensor whose elements indicate the top `k`
    predictions of the `predictions` `Tensor`.
  * Set operations are applied to `top_k` and `labels` to calculate true
    positives and false positives.
  * An `update_op` operation increments `true_positive_at_<k>` and
    `false_positive_at_<k>`. It also returns the recall value.

  Args:
    predictions: Float `Tensor` with shape [D1, ... DN, num_classes] where
      N >= 1. Commonly, N=1 and predictions has shape [batch size, num_classes].
      The final dimension contains the logit values for each class. [D1, ... DN]
      must match `labels`.
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`. Values should be in range [0, num_classes], where
      num_classes is the last dimension of `predictions`.
    k: Integer, k for @k metric.
    class_id: Integer class ID for which we want binary metrics. This should be
      in range [0, num_classes], where num_classes is the last dimension of
      `predictions`.
    ignore_mask: An optional, binary tensor whose shape is broadcastable to the
      the first [D1, ... DN] dimensions of `predictions_idx` and `labels`.
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of new update operation, and namespace for other dependant ops.

  Returns:
    precision: Scalar `float64` `Tensor` with the value of `true_positives`
      divided by the sum of `true_positives` and `false_positives`.
    update_op: `Operation` that increments `true_positives` and
      `false_positives` variables appropriately, and whose value matches
      `precision`.
  """
  default_name = 'precision_at_%d' % k
  if class_id is not None:
    default_name = '%s_class%d' % (default_name, class_id)
  with ops.op_scope([predictions, labels], name, default_name) as scope:
    _, top_k_idx = nn.top_k(predictions, k)
    top_k_idx = math_ops.to_int64(top_k_idx)
    tp, tp_update = _streaming_sparse_true_positive_at_k(
        predictions_idx=top_k_idx, labels=labels, k=k, class_id=class_id,
        ignore_mask=ignore_mask)
    fp, fp_update = _streaming_sparse_false_positive_at_k(
        predictions_idx=top_k_idx, labels=labels, k=k, class_id=class_id,
        ignore_mask=ignore_mask)

    metric = math_ops.div(tp, math_ops.add(tp, fp), name=scope)
    update = math_ops.div(
        tp_update, math_ops.add(tp_update, fp_update), name='update')
    if metrics_collections:
      ops.add_to_collections(metrics_collections, metric)
    if updates_collections:
      ops.add_to_collections(updates_collections, update)
    return metric, update


def _select_class_id(ids, selected_id):
  """Filter all but `selected_id` out of `ids`.

  Args:
    ids: `int64` `Tensor` or `SparseTensor` of IDs.
    selected_id: Int id to select.

  Returns:
    `SparseTensor` of same dimensions as `ids`, except for the last dimension,
    which might be smaller. This contains only the entries equal to
    `selected_id`.
  """
  if isinstance(ids, ops.SparseTensor):
    return sparse_ops.sparse_retain(
        ids, math_ops.equal(ids.values, selected_id))

  # TODO(ptucker): Make this more efficient, maybe add a sparse version of
  # tf.equal and tf.reduce_any?

  # Shape of filled IDs is the same as `ids` with the last dim collapsed to 1.
  ids_shape = array_ops.shape(ids)
  ids_last_dim = array_ops.size(ids_shape) - 1
  filled_selected_id_shape = math_ops.reduced_shape(
      ids_shape, array_ops.reshape(ids_last_dim, [1]))

  # Intersect `ids` with the selected ID.
  filled_selected_id = array_ops.fill(
      filled_selected_id_shape, math_ops.to_int64(selected_id))
  return set_ops.set_intersection(filled_selected_id, ids)


def _maybe_select_class_id(labels, predictions_idx, selected_id=None):
  """If class ID is specified, filter all other classes.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`.
    predictions_idx: `int64` `Tensor` of class IDs, with shape [D1, ... DN, k]
      where N >= 1. Commonly, N=1 and predictions has shape [batch size, k].
    selected_id: Int id to select.

  Returns:
    Tuple of `labels` and `predictions_idx`, possibly with classes removed.
  """
  if selected_id is None:
    return labels, predictions_idx
  return (_select_class_id(labels, selected_id),
          _select_class_id(predictions_idx, selected_id))


def _streaming_sparse_true_positive_at_k(predictions_idx,
                                         labels,
                                         k,
                                         class_id=None,
                                         ignore_mask=None,
                                         name=None):
  """Calculates per step true positives for recall@k and precision@k.

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
    k: Integer, k for @k metric. This is only used for default op name.
    class_id: Class for which we want binary metrics.
    ignore_mask: An optional, binary tensor whose shape is broadcastable to the
      the first [D1, ... DN] dimensions of `predictions_idx` and `labels`.
    name: Name of new variable, and namespace for other dependant ops.

  Returns:
    A tuple of `Variable` and update `Operation`.
  """
  default_name = 'true_positive_at_%d' % k
  if class_id is not None:
    default_name = '%s_class%d' % (default_name, class_id)
  with ops.op_scope([predictions_idx, labels], name, default_name) as scope:
    labels, predictions_idx = _maybe_select_class_id(labels,
                                                     predictions_idx,
                                                     class_id)
    tp = set_ops.set_size(set_ops.set_intersection(predictions_idx, labels))
    if ignore_mask is not None:
      tp = math_ops.select(ignore_mask, array_ops.zeros_like(tp), tp)
    batch_total_tp = math_ops.cast(
        math_ops.reduce_sum(tp), dtype=dtypes.float64)

    var = framework.local_variable(
        array_ops.zeros([], dtype=dtypes.float64), name=scope)
    return var, state_ops.assign_add(var, batch_total_tp, name='update')


def _streaming_sparse_false_positive_at_k(predictions_idx,
                                          labels,
                                          k,
                                          class_id=None,
                                          ignore_mask=None,
                                          name=None):
  """Calculates per step false positives for precision@k.

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
    k: Integer, k for @k metric. This is only used for default op name.
    class_id: Class for which we want binary metrics.
    ignore_mask: An optional, binary tensor whose shape is broadcastable to the
      the first [D1, ... DN] dimensions of `predictions_idx` and `labels`.
    name: Name of new variable, and namespace for other dependant ops.

  Returns:
    A tuple of `Variable` and update `Operation`.
  """
  default_name = 'false_positive_at_%d' % k
  if class_id is not None:
    default_name = '%s_class%d' % (default_name, class_id)
  with ops.op_scope([predictions_idx, labels], name, default_name) as scope:
    labels, predictions_idx = _maybe_select_class_id(labels,
                                                     predictions_idx,
                                                     class_id)
    fp = set_ops.set_size(set_ops.set_difference(predictions_idx,
                                                 labels,
                                                 aminusb=True))
    if ignore_mask is not None:
      fp = math_ops.select(ignore_mask, array_ops.zeros_like(fp), fp)
    batch_total_fp = math_ops.cast(
        math_ops.reduce_sum(fp), dtype=dtypes.float64)

    var = framework.local_variable(
        array_ops.zeros([], dtype=dtypes.float64), name=scope)
    return var, state_ops.assign_add(var, batch_total_fp, name='update')


def _streaming_sparse_false_negative_at_k(predictions_idx,
                                          labels,
                                          k,
                                          class_id=None,
                                          ignore_mask=None,
                                          name=None):
  """Calculates per step false negatives for recall@k.

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
    k: Integer, k for @k metric. This is only used for default op name.
    class_id: Class for which we want binary metrics.
    ignore_mask: An optional, binary tensor whose shape is broadcastable to the
      the first [D1, ... DN] dimensions of `predictions_idx` and `labels`.
    name: Name of new variable, and namespace for other dependant ops.

  Returns:
    A tuple of `Variable` and update `Operation`.
  """
  default_name = 'false_negative_at_%d' % k
  if class_id is not None:
    default_name = '%s_class%d' % (default_name, class_id)
  with ops.op_scope([predictions_idx, labels], name, default_name) as scope:
    labels, predictions_idx = _maybe_select_class_id(labels,
                                                     predictions_idx,
                                                     class_id)
    fn = set_ops.set_size(set_ops.set_difference(predictions_idx,
                                                 labels,
                                                 aminusb=False))
    if ignore_mask is not None:
      fn = math_ops.select(ignore_mask, array_ops.zeros_like(fn), fn)
    batch_total_fn = math_ops.cast(
        math_ops.reduce_sum(fn), dtype=dtypes.float64)

    var = framework.local_variable(
        array_ops.zeros([], dtype=dtypes.float64), name=scope)
    return var, state_ops.assign_add(var, batch_total_fn, name='update')


def streaming_mean_absolute_error(predictions, labels, weights=None,
                                  metrics_collections=None,
                                  updates_collections=None,
                                  name=None):
  """Computes the mean absolute error between the labels and predictions.

  The `streaming_mean_absolute_error` function creates two local variables,
  `total` and `count` that are used to compute the mean absolute error. This
  average is ultimately returned as `mean_absolute_error`: an idempotent
  operation that simply divides `total` by `count`. To facilitate the estimation
  of the mean absolute error over a stream of data, the function utilizes two
  operations. First, an `absolute_errors` operation computes the absolute value
  of the differences between `predictions` and `labels`. Second, an `update_op`
  operation whose behavior is dependent on the value of `weights`. If `weights`
  is None, then `update_op` increments `total` with the reduced sum of
  `absolute_errors` and increments `count` with the number of elements in
  `absolute_errors`. If `weights` is not `None`, then `update_op` increments
  `total` with the reduced sum of the product of `weights` and `absolute_errors`
  and increments `count` with the reduced sum of `weights`. In addition to
  performing the updates, `update_op` also returns the `mean_absolute_error`
  value.

  Args:
    predictions: A `Tensor` of arbitrary shape.
    labels: A `Tensor` of the same shape as `predictions`.
    weights: An optional set of weights of the same shape as `predictions`. If
      `weights` is not None, the function computes a weighted mean.
    metrics_collections: An optional list of collections that
      `mean_absolute_error` should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_op_scope name.

  Returns:
    mean_absolute_error: A tensor representing the current mean, the value of
      `total` divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_absolute_error`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match
      `predictions` or if either `metrics_collections` or `updates_collections`
      are not a list or tuple.
  """
  predictions, labels = _remove_squeezable_dimensions(predictions, labels)
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())
  absolute_errors = math_ops.abs(predictions - labels)
  return streaming_mean(absolute_errors, weights, metrics_collections,
                        updates_collections, name or 'mean_absolute_error')


def streaming_mean_relative_error(predictions, labels, normalizer, weights=None,
                                  metrics_collections=None,
                                  updates_collections=None,
                                  name=None):
  """Computes the mean relative error by normalizing with the given values.

  The `streaming_mean_relative_error` function creates two local variables,
  `total` and `count` that are used to compute the mean relative absolute error.
  This average is ultimately returned as `mean_relative_error`: an idempotent
  operation that simply divides `total` by `count`. To facilitate the estimation
  of the mean relative error over a stream of data, the function utilizes two
  operations. First, a `relative_errors` operation divides the absolute value
  of the differences between `predictions` and `labels` by the `normalizer`.
  Second, an `update_op` operation whose behavior is dependent on the value of
  `weights`. If `weights` is None, then `update_op` increments `total` with the
  reduced sum of `relative_errors` and increments `count` with the number of
  elements in `relative_errors`. If `weights` is not `None`, then `update_op`
  increments `total` with the reduced sum of the product of `weights` and
  `relative_errors` and increments `count` with the reduced sum of `weights`. In
  addition to performing the updates, `update_op` also returns the
  `mean_relative_error` value.

  Args:
    predictions: A `Tensor` of arbitrary shape.
    labels: A `Tensor` of the same shape as `predictions`.
    normalizer: A `Tensor` of the same shape as `predictions`.
    weights: An optional set of weights of the same shape as `predictions`. If
      `weights` is not None, the function computes a weighted mean.
    metrics_collections: An optional list of collections that
      `mean_relative_error` should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_op_scope name.

  Returns:
    mean_relative_error: A tensor representing the current mean, the value of
      `total` divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_relative_error`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match
      `predictions` or if either `metrics_collections` or `updates_collections`
      are not a list or tuple.
  """
  predictions, labels = _remove_squeezable_dimensions(predictions, labels)
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())

  predictions, normalizer = _remove_squeezable_dimensions(
      predictions, normalizer)
  predictions.get_shape().assert_is_compatible_with(normalizer.get_shape())
  relative_errors = math_ops.select(
      math_ops.equal(normalizer, 0.0),
      array_ops.zeros_like(labels),
      math_ops.div(math_ops.abs(labels - predictions), normalizer))
  return streaming_mean(relative_errors, weights, metrics_collections,
                        updates_collections, name or 'mean_relative_error')


def streaming_mean_squared_error(predictions, labels, weights=None,
                                 metrics_collections=None,
                                 updates_collections=None,
                                 name=None):
  """Computes the mean squared error between the labels and predictions.

  The `streaming_mean_squared_error` function creates two local variables,
  `total` and `count` that are used to compute the mean squared error.
  This average is ultimately returned as `mean_squared_error`: an idempotent
  operation that simply divides `total` by `count`. To facilitate the estimation
  of the mean squared error over a stream of data, the function utilizes two
  operations. First, a `squared_error` operation computes the element-wise
  square of the difference between `predictions` and `labels`. Second, an
  `update_op` operation whose behavior is dependent on the value of `weights`.
  If `weights` is None, then `update_op` increments `total` with the
  reduced sum of `squared_error` and increments `count` with the number of
  elements in `squared_error`. If `weights` is not `None`, then `update_op`
  increments `total` with the reduced sum of the product of `weights` and
  `squared_error` and increments `count` with the reduced sum of `weights`. In
  addition to performing the updates, `update_op` also returns the
  `mean_squared_error` value.

  Args:
    predictions: A `Tensor` of arbitrary shape.
    labels: A `Tensor` of the same shape as `predictions`.
    weights: An optional set of weights of the same shape as `predictions`. If
      `weights` is not None, the function computes a weighted mean.
    metrics_collections: An optional list of collections that
      `mean_squared_error` should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_op_scope name.

  Returns:
    mean_squared_error: A tensor representing the current mean, the value of
      `total` divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_squared_error`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match
      `predictions` or if either `metrics_collections` or `updates_collections`
      are not a list or tuple.
  """
  predictions, labels = _remove_squeezable_dimensions(predictions, labels)
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())
  squared_error = math_ops.square(labels - predictions)
  return streaming_mean(squared_error, weights, metrics_collections,
                        updates_collections, name or 'mean_squared_error')


def streaming_root_mean_squared_error(predictions, labels, weights=None,
                                      metrics_collections=None,
                                      updates_collections=None,
                                      name=None):
  """Computes the root mean squared error between the labels and predictions.

  The `streaming_root_mean_squared_error` function creates two local variables,
  `total` and `count` that are used to compute the root mean squared error.
  This average is ultimately returned as `root_mean_squared_error`: an
  idempotent operation that takes the square root of the division of `total`
  by `count`. To facilitate the estimation of the root mean squared error over a
  stream of data, the function utilizes two operations. First, a `squared_error`
  operation computes the element-wise square of the difference between
  `predictions` and `labels`. Second, an `update_op` operation whose behavior is
  dependent on the value of `weights`. If `weights` is None, then `update_op`
  increments `total` with the reduced sum of `squared_error` and increments
  `count` with the number of elements in `squared_error`. If `weights` is not
  `None`, then `update_op` increments `total` with the reduced sum of the
  product of `weights` and `squared_error` and increments `count` with the
  reduced sum of `weights`. In addition to performing the updates, `update_op`
  also returns the `root_mean_squared_error` value.

  Args:
    predictions: A `Tensor` of arbitrary shape.
    labels: A `Tensor` of the same shape as `predictions`.
    weights: An optional set of weights of the same shape as `predictions`. If
      `weights` is not None, the function computes a weighted mean.
    metrics_collections: An optional list of collections that
      `root_mean_squared_error` should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_op_scope name.

  Returns:
    root_mean_squared_error: A tensor representing the current mean, the value
      of `total` divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `root_mean_squared_error`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match
      `predictions` or if either `metrics_collections` or `updates_collections`
      are not a list or tuple.
  """
  predictions, labels = _remove_squeezable_dimensions(predictions, labels)
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())
  value_tensor, update_op = streaming_mean_squared_error(
      predictions, labels, weights, None, None,
      name or 'root_mean_squared_error')

  root_mean_squared_error = math_ops.sqrt(value_tensor)
  with ops.control_dependencies([update_op]):
    update_op = math_ops.sqrt(update_op)

  if metrics_collections:
    ops.add_to_collections(metrics_collections, root_mean_squared_error)

  if updates_collections:
    ops.add_to_collections(updates_collections, update_op)

  return root_mean_squared_error, update_op


# TODO(nsilberman): add a 'normalized' flag so that the user can request
# normalization if the inputs are not normalized.
def streaming_mean_cosine_distance(predictions, labels, dim, weights=None,
                                   metrics_collections=None,
                                   updates_collections=None,
                                   name=None):
  """Computes the cosine distance between the labels and predictions.

  The `streaming_mean_cosine_distance` function creates two local variables,
  `total` and `count` that are used to compute the average cosine distance
  between `predictions` and `labels`. This average is ultimately returned as
  `mean_distance` which is an idempotent operation that simply divides `total`
  by `count. To facilitate the estimation of a mean over multiple batches
  of data, the function creates an `update_op` operation whose behavior is
  dependent on the value of `weights`. If `weights` is None, then `update_op`
  increments `total` with the reduced sum of `values and increments `count` with
  the number of elements in `values`. If `weights` is not `None`, then
  `update_op` increments `total` with the reduced sum of the product of `values`
  and `weights` and increments `count` with the reduced sum of weights.

  Args:
    predictions: A tensor of the same size as labels.
    labels: A tensor of arbitrary size.
    dim: The dimension along which the cosine distance is computed.
    weights: An optional set of weights which indicates which predictions to
      ignore during metric computation. Its size matches that of labels except
      for the value of 'dim' which should be 1. For example if labels has
      dimensions [32, 100, 200, 3], then `weights` should have dimensions
      [32, 100, 200, 1].
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_op_scope name.

  Returns:
    mean_distance: A tensor representing the current mean, the value of `total`
      divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately.

  Raises:
    ValueError: If labels and predictions are of different sizes or if the
      ignore_mask is of the wrong size or if either `metrics_collections` or
      `updates_collections` are not a list or tuple.
  """
  predictions, labels = _remove_squeezable_dimensions(predictions, labels)
  predictions.get_shape().assert_is_compatible_with(labels.get_shape())
  radial_diffs = math_ops.mul(predictions, labels)
  radial_diffs = math_ops.reduce_sum(radial_diffs,
                                     reduction_indices=[dim,],
                                     keep_dims=True)
  mean_distance, update_op = streaming_mean(radial_diffs, weights,
                                            None,
                                            None,
                                            name or 'mean_cosine_distance')
  mean_distance = math_ops.sub(1.0, mean_distance)
  update_op = math_ops.sub(1.0, update_op)

  if metrics_collections:
    ops.add_to_collections(metrics_collections, mean_distance)

  if updates_collections:
    ops.add_to_collections(updates_collections, update_op)

  return mean_distance, update_op


def streaming_percentage_less(values, threshold, ignore_mask=None,
                              metrics_collections=None,
                              updates_collections=None,
                              name=None):
  """Computes the percentage of values less than the given threshold.

  The `streaming_percentage_less` function creates two local variables,
  `total` and `count` that are used to compute the percentage of `values` that
  fall below `threshold`. This rate is ultimately returned as `percentage`
  which is an idempotent operation that simply divides `total` by `count.
  To facilitate the estimation of the percentage of values that fall under
  `threshold` over multiple batches of data, the function creates an
  `update_op` operation whose behavior is dependent on the value of
  `ignore_mask`. If `ignore_mask` is None, then `update_op`
  increments `total` with the number of elements of `values` that are less
  than `threshold` and `count` with the number of elements in `values`. If
  `ignore_mask` is not `None`, then `update_op` increments `total` with the
  number of elements of `values` that are less than `threshold` and whose
  corresponding entries in `ignore_mask` are False, and `count` is incremented
  with the number of elements of `ignore_mask` that are False.

  Args:
    values: A numeric `Tensor` of arbitrary size.
    threshold: A scalar threshold.
    ignore_mask: An optional mask of the same shape as 'values' which indicates
      which elements to ignore during metric computation.
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_op_scope name.

  Returns:
    percentage: A tensor representing the current mean, the value of `total`
      divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately.

  Raises:
    ValueError: If `ignore_mask` is not None and its shape doesn't match `values
      or if either `metrics_collections` or `updates_collections` are supplied
      but are not a list or tuple.
  """
  is_below_threshold = math_ops.to_float(math_ops.less(values, threshold))
  return streaming_mean(is_below_threshold, _mask_to_weights(ignore_mask),
                        metrics_collections, updates_collections,
                        name or 'percentage_below_threshold')


__all__ = make_all(__name__)
