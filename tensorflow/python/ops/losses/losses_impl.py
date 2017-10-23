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
"""Implementation of Loss operations for use in neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util
from tensorflow.python.util.deprecation import deprecated_args


class Reduction(object):
  """Types of loss reduction."""

  # Un-reduced weighted losses with the same shape as input.
  NONE = "none"

  # Scalar sum of `NONE`.
  SUM = "weighted_sum"

  # Scalar `SUM` divided by sum of weights.
  MEAN = "weighted_mean"

  # Scalar `SUM` divided by number of non-zero weights.
  SUM_BY_NONZERO_WEIGHTS = "weighted_sum_by_nonzero_weights"

  @classmethod
  def all(cls):
    return (
        cls.NONE,
        cls.SUM,
        cls.MEAN,
        cls.SUM_BY_NONZERO_WEIGHTS)

  @classmethod
  def validate(cls, key):
    if key not in cls.all():
      raise ValueError("Invalid ReductionKey %s." % key)


def _safe_div(numerator, denominator, name="value"):
  """Computes a safe divide which returns 0 if the denominator is zero.

  Note that the function contains an additional conditional check that is
  necessary for avoiding situations where the loss is zero causing NaNs to
  creep into the gradient computation.

  Args:
    numerator: An arbitrary `Tensor`.
    denominator: `Tensor` whose shape matches `numerator` and whose values are
      assumed to be non-negative.
    name: An optional name for the returned op.

  Returns:
    The element-wise value of the numerator divided by the denominator.
  """
  return array_ops.where(
      math_ops.greater(denominator, 0),
      math_ops.div(numerator, array_ops.where(
          math_ops.equal(denominator, 0),
          array_ops.ones_like(denominator), denominator)),
      array_ops.zeros_like(numerator),
      name=name)


def _safe_mean(losses, num_present):
  """Computes a safe mean of the losses.

  Args:
    losses: `Tensor` whose elements contain individual loss measurements.
    num_present: The number of measurable elements in `losses`.

  Returns:
    A scalar representing the mean of `losses`. If `num_present` is zero,
      then zero is returned.
  """
  total_loss = math_ops.reduce_sum(losses)
  return _safe_div(total_loss, num_present)


def _num_present(losses, weights, per_batch=False):
  """Computes the number of elements in the loss function induced by `weights`.

  A given weights tensor induces different numbers of usable elements in the
  `losses` tensor. The `weights` tensor is broadcast across `losses` for all
  possible dimensions. For example, if `losses` is a tensor of dimension
  `[4, 5, 6, 3]` and `weights` is a tensor of shape `[4, 5]`, then `weights` is,
  in effect, tiled to match the shape of `losses`. Following this effective
  tile, the total number of present elements is the number of non-zero weights.

  Args:
    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
    weights: `Tensor` of shape `[]`, `[batch_size]` or
      `[batch_size, d1, ... dK]`, where K < N.
    per_batch: Whether to return the number of elements per batch or as a sum
      total.

  Returns:
    The number of present (non-zero) elements in the losses tensor. If
      `per_batch` is `True`, the value is returned as a tensor of size
      `[batch_size]`. Otherwise, a single scalar tensor is returned.
  """
  with ops.name_scope(None, "num_present", (losses, weights)) as scope:
    weights = math_ops.to_float(weights)
    present = array_ops.where(
        math_ops.equal(weights, 0.0),
        array_ops.zeros_like(weights),
        array_ops.ones_like(weights))
    present = weights_broadcast_ops.broadcast_weights(present, losses)
    if per_batch:
      return math_ops.reduce_sum(
          present, axis=math_ops.range(1, array_ops.rank(present)),
          keep_dims=True, name=scope)
    return math_ops.reduce_sum(present, name=scope)


def compute_weighted_loss(
    losses, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Computes the weighted loss.

  Args:
    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `losses`, and must be broadcastable to `losses` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: the scope for the operations performed in computing the loss.
    loss_collection: the loss will be added to these collections.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss `Tensor` of the same type as `losses`. If `reduction` is
    `NONE`, this has the same shape as `losses`; otherwise, it is scalar.

  Raises:
    ValueError: If `weights` is `None` or the shape is not compatible with
      `losses`, or if the number of dimensions (rank) of either `losses` or
      `weights` is missing.
  """
  Reduction.validate(reduction)
  with ops.name_scope(scope, "weighted_loss", (losses, weights)):
    with ops.control_dependencies((
        weights_broadcast_ops.assert_broadcastable(weights, losses),)):
      losses = ops.convert_to_tensor(losses)
      input_dtype = losses.dtype
      losses = math_ops.to_float(losses)
      weights = math_ops.to_float(weights)
      weighted_losses = math_ops.multiply(losses, weights)
      if reduction == Reduction.NONE:
        loss = weighted_losses
      else:
        loss = math_ops.reduce_sum(weighted_losses)
        if reduction == Reduction.MEAN:
          loss = _safe_mean(
              loss,
              math_ops.reduce_sum(array_ops.ones_like(losses) * weights))
        elif reduction == Reduction.SUM_BY_NONZERO_WEIGHTS:
          loss = _safe_mean(loss, _num_present(losses, weights))

      # Convert the result back to the input type.
      loss = math_ops.cast(loss, input_dtype)
      util.add_loss(loss, loss_collection)
      return loss


def absolute_difference(
    labels, predictions, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Adds an Absolute Difference loss to the training procedure.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a `Tensor` of
  shape `[batch_size]`, then the total loss for each sample of the batch is
  rescaled by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of
      `labels` or if the shape of `weights` is invalid or if `labels`
      or `predictions` is None.
  """
  if labels is None:
    raise ValueError("labels must not be None.")
  if predictions is None:
    raise ValueError("predictions must not be None.")
  with ops.name_scope(scope, "absolute_difference",
                      (predictions, labels, weights)) as scope:
    predictions = math_ops.to_float(predictions)
    labels = math_ops.to_float(labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    losses = math_ops.abs(math_ops.subtract(predictions, labels))
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


@deprecated_args(None, "dim is deprecated, use axis instead", "dim")
def cosine_distance(
    labels, predictions, axis=None, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS,
    dim=None):
  """Adds a cosine-distance loss to the training procedure.

  Note that the function assumes that `predictions` and `labels` are already
  unit-normalized.

  Args:
    labels: `Tensor` whose shape matches 'predictions'
    predictions: An arbitrary matrix.
    axis: The dimension along which the cosine distance is computed.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: Type of reduction to apply to loss.
    dim: The old (deprecated) name for `axis`.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If `predictions` shape doesn't match `labels` shape, or
      `axis`, `labels`, `predictions` or `weights` is `None`.
  """
  if dim is not None:
    if axis is not None:
      raise ValueError("Cannot specify both 'axis' and 'dim'")
    axis = dim
  if axis is None and dim is None:
    raise ValueError("You must specify 'axis'.")
  if labels is None:
    raise ValueError("labels must not be None.")
  if predictions is None:
    raise ValueError("predictions must not be None.")
  with ops.name_scope(scope, "cosine_distance_loss",
                      (predictions, labels, weights)) as scope:
    predictions = math_ops.to_float(predictions)
    labels = math_ops.to_float(labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    radial_diffs = math_ops.multiply(predictions, labels)
    losses = 1 - math_ops.reduce_sum(radial_diffs, axis=(axis,), keep_dims=True)
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


def hinge_loss(labels, logits, weights=1.0, scope=None,
               loss_collection=ops.GraphKeys.LOSSES,
               reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Adds a hinge loss to the training procedure.

  Args:
    labels: The ground truth output tensor. Its shape should match the shape of
      logits. The values of the tensor are expected to be 0.0 or 1.0.
    logits: The logits, a float tensor.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shapes of `logits` and `labels` don't match or
      if `labels` or `logits` is None.
  """
  if labels is None:
    raise ValueError("labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with ops.name_scope(scope, "hinge_loss", (logits, labels, weights)) as scope:
    logits = math_ops.to_float(logits)
    labels = math_ops.to_float(labels)
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    # We first need to convert binary labels to -1/1 labels (as floats).
    all_ones = array_ops.ones_like(labels)
    labels = math_ops.subtract(2 * labels, all_ones)
    losses = nn_ops.relu(
        math_ops.subtract(all_ones, math_ops.multiply(labels, logits)))
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


def huber_loss(labels, predictions, weights=1.0, delta=1.0, scope=None,
               loss_collection=ops.GraphKeys.LOSSES,
               reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Adds a Huber Loss term to the training procedure.

  For each value x in `error=labels-predictions`, the following is calculated:

  ```
    0.5 * x^2                  if |x| <= d
    0.5 * d^2 + d * (|x| - d)  if |x| > d
  ```

  where d is `delta`.

  See: https://en.wikipedia.org/wiki/Huber_loss

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    delta: `float`, the point where the huber loss function
      changes from a quadratic to linear.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.  Also if `labels` or
     `predictions` is None.
  """
  if labels is None:
    raise ValueError("labels must not be None.")
  if predictions is None:
    raise ValueError("predictions must not be None.")
  with ops.name_scope(scope, "huber_loss",
                      (predictions, labels, weights)) as scope:
    predictions = math_ops.to_float(predictions)
    labels = math_ops.to_float(labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    error = math_ops.subtract(predictions, labels)
    abs_error = math_ops.abs(error)
    quadratic = math_ops.minimum(abs_error, delta)
    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


def log_loss(labels, predictions, weights=1.0, epsilon=1e-7, scope=None,
             loss_collection=ops.GraphKeys.LOSSES,
             reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Adds a Log Loss term to the training procedure.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    epsilon: A small increment to add to avoid taking a log of zero.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.  Also if `labels` or `predictions`
      is None.
  """
  if labels is None:
    raise ValueError("labels must not be None.")
  if predictions is None:
    raise ValueError("predictions must not be None.")
  with ops.name_scope(scope, "log_loss",
                      (predictions, labels, weights)) as scope:
    predictions = math_ops.to_float(predictions)
    labels = math_ops.to_float(labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    losses = -math_ops.multiply(
        labels,
        math_ops.log(predictions + epsilon)) - math_ops.multiply(
            (1 - labels), math_ops.log(1 - predictions + epsilon))
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


# TODO(b/37208492): Add reduction arg.
def mean_pairwise_squared_error(
    labels, predictions, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES):
  """Adds a pairwise-errors-squared loss to the training procedure.

  Unlike `mean_squared_error`, which is a measure of the differences between
  corresponding elements of `predictions` and `labels`,
  `mean_pairwise_squared_error` is a measure of the differences between pairs of
  corresponding elements of `predictions` and `labels`.

  For example, if `labels`=[a, b, c] and `predictions`=[x, y, z], there are
  three pairs of differences are summed to compute the loss:
    loss = [ ((a-b) - (x-y)).^2 + ((a-c) - (x-z)).^2 + ((b-c) - (y-z)).^2 ] / 3

  Note that since the inputs are of shape `[batch_size, d0, ... dN]`, the
  corresponding pairs are computed within each batch sample but not across
  samples within a batch. For example, if `predictions` represents a batch of
  16 grayscale images of dimension [batch_size, 100, 200], then the set of pairs
  is drawn from each image, but not across images.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector.

  Args:
    labels: The ground truth output tensor, whose shape must match the shape of
      `predictions`.
    predictions: The predicted outputs, a tensor of size
      `[batch_size, d0, .. dN]` where N+1 is the total number of dimensions in
      `predictions`.
    weights: Coefficients for the loss a scalar, a tensor of shape
      `[batch_size]` or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.

  Returns:
    A scalar `Tensor` that returns the weighted loss.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.  Also if `labels` or `predictions
      is None.
  """
  if labels is None:
    raise ValueError("labels must not be None.")
  if predictions is None:
    raise ValueError("predictions must not be None.")
  with ops.name_scope(scope, "mean_pairwise_squared_error",
                      (predictions, labels, weights)) as scope:
    weights = math_ops.to_float(weights)
    labels = math_ops.to_float(labels)
    with ops.control_dependencies((
        weights_broadcast_ops.assert_broadcastable(weights, labels),)):
      predictions = math_ops.to_float(predictions)
      predictions.get_shape().assert_is_compatible_with(labels.get_shape())

      diffs = math_ops.subtract(predictions, labels)

      reduction_indices = math_ops.range(1, array_ops.rank(diffs))

      sum_squares_diff_per_batch = math_ops.reduce_sum(
          math_ops.square(diffs), reduction_indices=reduction_indices,
          keep_dims=True)
      num_present_per_batch = _num_present(diffs, weights, per_batch=True)

      term1 = 2.0 * _safe_div(sum_squares_diff_per_batch,
                              num_present_per_batch)

      sum_diff = math_ops.reduce_sum(
          diffs, reduction_indices=reduction_indices, keep_dims=True)
      term2 = 2.0 * _safe_div(math_ops.square(sum_diff),
                              math_ops.square(num_present_per_batch))

      weighted_losses = math_ops.multiply(term1 - term2, weights)
      loss = math_ops.reduce_sum(weighted_losses)

      mean_loss = array_ops.where(
          math_ops.reduce_sum(num_present_per_batch) > 0,
          loss,
          array_ops.zeros_like(loss),
          name="value")
      util.add_loss(mean_loss, loss_collection)
      return mean_loss


def mean_squared_error(
    labels, predictions, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Adds a Sum-of-Squares loss to the training procedure.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.  Also if `labels` or `predictions`
      is None.
  """
  if labels is None:
    raise ValueError("labels must not be None.")
  if predictions is None:
    raise ValueError("predictions must not be None.")
  with ops.name_scope(scope, "mean_squared_error",
                      (predictions, labels, weights)) as scope:
    predictions = math_ops.to_float(predictions)
    labels = math_ops.to_float(labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    losses = math_ops.squared_difference(predictions, labels)
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


def sigmoid_cross_entropy(
    multi_class_labels, logits, weights=1.0, label_smoothing=0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Creates a cross-entropy loss using tf.nn.sigmoid_cross_entropy_with_logits.

  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of shape `[batch_size]`, then the loss weights apply to each
  corresponding sample.

  If `label_smoothing` is nonzero, smooth the labels towards 1/2:

      new_multiclass_labels = multiclass_labels * (1 - label_smoothing)
                              + 0.5 * label_smoothing

  Args:
    multi_class_labels: `[batch_size, num_classes]` target integer labels in
      `(0, 1)`.
    logits: Float `[batch_size, num_classes]` logits outputs of the network.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    label_smoothing: If greater than `0` then smooth the labels.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss `Tensor` of the same type as `logits`. If `reduction` is
    `NONE`, this has the same shape as `logits`; otherwise, it is scalar.

  Raises:
    ValueError: If the shape of `logits` doesn't match that of
      `multi_class_labels` or if the shape of `weights` is invalid, or if
      `weights` is None.  Also if `multi_class_labels` or `logits` is None.
  """
  if multi_class_labels is None:
    raise ValueError("multi_class_labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with ops.name_scope(scope, "sigmoid_cross_entropy_loss",
                      (logits, multi_class_labels, weights)) as scope:
    logits = ops.convert_to_tensor(logits)
    multi_class_labels = math_ops.cast(multi_class_labels, logits.dtype)
    logits.get_shape().assert_is_compatible_with(multi_class_labels.get_shape())

    if label_smoothing > 0:
      multi_class_labels = (multi_class_labels * (1 - label_smoothing) +
                            0.5 * label_smoothing)

    losses = nn.sigmoid_cross_entropy_with_logits(labels=multi_class_labels,
                                                  logits=logits,
                                                  name="xentropy")
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


def softmax_cross_entropy(
    onehot_labels, logits, weights=1.0, label_smoothing=0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Creates a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits.

  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of shape `[batch_size]`, then the loss weights apply to each
  corresponding sample.

  If `label_smoothing` is nonzero, smooth the labels towards 1/num_classes:
      new_onehot_labels = onehot_labels * (1 - label_smoothing)
                          + label_smoothing / num_classes

  Args:
    onehot_labels: `[batch_size, num_classes]` target one-hot-encoded labels.
    logits: [batch_size, num_classes] logits outputs of the network .
    weights: Optional `Tensor` whose rank is either 0, or rank 1 and is
      broadcastable to the loss which is a `Tensor` of shape `[batch_size]`.
    label_smoothing: If greater than 0 then smooth the labels.
    scope: the scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss `Tensor` of the same type as `logits`. If `reduction` is
    `NONE`, this has shape `[batch_size]`; otherwise, it is scalar.

  Raises:
    ValueError: If the shape of `logits` doesn't match that of `onehot_labels`
      or if the shape of `weights` is invalid or if `weights` is None.  Also if
      `onehot_labels` or `logits` is None.
  """
  if onehot_labels is None:
    raise ValueError("onehot_labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with ops.name_scope(scope, "softmax_cross_entropy_loss",
                      (logits, onehot_labels, weights)) as scope:
    logits = ops.convert_to_tensor(logits)
    onehot_labels = math_ops.cast(onehot_labels, logits.dtype)
    logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())

    if label_smoothing > 0:
      num_classes = math_ops.cast(
          array_ops.shape(onehot_labels)[1], logits.dtype)
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      onehot_labels = onehot_labels * smooth_positives + smooth_negatives

    losses = nn.softmax_cross_entropy_with_logits(labels=onehot_labels,
                                                  logits=logits,
                                                  name="xentropy")
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


# TODO(ptucker): Merge this with similar method in metrics_impl.
def _remove_squeezable_dimensions(
    labels, predictions, weights=None, expected_rank_diff=0):
  """Internal version of _remove_squeezable_dimensions which handles weights.

  Squeezes `predictions` and `labels` if their ranks differ from expected by
  exactly 1.
  Squeezes `weights` if its rank is 1 more than the new rank of `predictions`

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    labels: Label values, a `Tensor` whose dimensions match `predictions`.
    predictions: Predicted values, a `Tensor` of arbitrary dimensions.
    weights: Optional weight `Tensor`. It will be squeezed if it's not scalar,
      and its rank is 1 more than the new rank of `labels`.
    expected_rank_diff: Expected result of `rank(predictions) - rank(labels)`.

  Returns:
    Tuple of `predictions`, `labels` and `weights`, possibly with the last
    dimension squeezed.
  """
  labels, predictions = confusion_matrix.remove_squeezable_dimensions(
      labels, predictions, expected_rank_diff=expected_rank_diff)

  if weights is not None:
    weights = ops.convert_to_tensor(weights)
    labels_rank = labels.get_shape().ndims
    weights_shape = weights.get_shape()
    weights_rank = weights_shape.ndims

    if (labels_rank is not None) and (weights_rank is not None):
      # Use static rank.
      rank_diff = weights_rank - labels_rank
      if rank_diff == 1:
        weights = array_ops.squeeze(weights, [-1])
      return labels, predictions, weights

    # Use dynamic rank.
    rank_diff = array_ops.rank(weights) - array_ops.rank(labels)
    if (weights_rank is None) or (
        weights_rank > 0 and weights_shape.dims[-1].is_compatible_with(1)):
      weights = control_flow_ops.cond(
          math_ops.equal(1, rank_diff),
          lambda: array_ops.squeeze(weights, [-1]),
          lambda: weights)

  return labels, predictions, weights


def sparse_softmax_cross_entropy(
    labels, logits, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Cross-entropy loss using `tf.nn.sparse_softmax_cross_entropy_with_logits`.

  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of shape [`batch_size`], then the loss weights apply to each
  corresponding sample.

  Args:
    labels: `Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` (where `r` is rank of
      `labels` and result) and dtype `int32` or `int64`. Each entry in `labels`
      must be an index in `[0, num_classes)`. Other values will raise an
      exception when this op is run on CPU, and return `NaN` for corresponding
      loss and gradient rows on GPU.
    logits: Unscaled log probabilities of shape
      `[d_0, d_1, ..., d_{r-1}, num_classes]` and dtype `float32` or `float64`.
    weights: Coefficients for the loss. This must be scalar or broadcastable to
      `labels` (i.e. same rank and each dimension is either 1 or the same).
    scope: the scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss `Tensor` of the same type as `logits`. If `reduction` is
    `NONE`, this has the same shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shapes of `logits`, `labels`, and `weights` are
      incompatible, or if any of them are None.
  """
  if labels is None:
    raise ValueError("labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with ops.name_scope(scope, "sparse_softmax_cross_entropy_loss",
                      (logits, labels, weights)) as scope:
    # As documented above in Args, labels contain class IDs and logits contains
    # 1 probability per class ID, so we expect rank(logits) - rank(labels) == 1;
    # therefore, expected_rank_diff=1.
    labels, logits, weights = _remove_squeezable_dimensions(
        labels, logits, weights, expected_rank_diff=1)
    losses = nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                         logits=logits,
                                                         name="xentropy")
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)
