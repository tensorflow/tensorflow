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
"""Loss operations for use in neural networks.

Note: All the losses are added to the `GraphKeys.LOSSES` collection.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup

__all__ = [
    "absolute_difference", "add_loss", "cosine_distance",
    "compute_weighted_loss", "get_losses", "get_regularization_losses",
    "get_total_loss", "hinge_loss", "log_loss", "mean_pairwise_squared_error",
    "mean_squared_error", "sigmoid_cross_entropy", "softmax_cross_entropy",
    "sparse_softmax_cross_entropy"
]


def _scale_losses(losses, weights):
  """Computes the scaled loss.

  Args:
    losses: A `Tensor` of size [batch_size, d1, ... dN].
    weights: A `Tensor` of size [1], [batch_size] or [batch_size, d1, ... dN].
      The `losses` are reduced (tf.reduce_sum) until its dimension matches
      that of `weights` at which point the reduced `losses` are element-wise
      multiplied by `weights` and a final reduce_sum is computed on the result.
      Conceptually, this operation is equivalent to broadcasting (tiling)
      `weights` to be the same size as `losses`, performing an element-wise
      multiplication, and summing the result.

  Returns:
    A scalar tf.float32 `Tensor` whose value represents the sum of the scaled
      `losses`.
  """
  # First, compute the sum of the losses over all elements:
  start_index = max(0, weights.get_shape().ndims)
  reduction_indices = list(range(start_index, losses.get_shape().ndims))
  reduced_losses = math_ops.reduce_sum(
      losses, reduction_indices=reduction_indices)
  reduced_losses = math_ops.multiply(reduced_losses, weights)
  return math_ops.reduce_sum(reduced_losses)


def _safe_div(numerator, denominator, name="value"):
  """Computes a safe divide which returns 0 if the denominator is zero.

  Note that the function contains an additional conditional check that is
  necessary for avoiding situations where the loss is zero causing NaNs to
  creep into the gradient computation.

  Args:
    numerator: An arbitrary `Tensor`.
    denominator: A `Tensor` whose shape matches `numerator` and whose values are
      assumed to be non-negative.
    name: An optional name for the returned op.

  Returns:
    The element-wise value of the numerator divided by the denominator.
  """
  return array_ops.where(
      math_ops.greater(denominator, 0),
      math_ops.div(numerator,
                   array_ops.where(
                       math_ops.equal(denominator, 0),
                       array_ops.ones_like(denominator), denominator)),
      array_ops.zeros_like(numerator),
      name=name)


def _safe_mean(losses, num_present):
  """Computes a safe mean of the losses.

  Args:
    losses: A tensor whose elements contain individual loss measurements.
    num_present: The number of measurable losses in the tensor.

  Returns:
    A scalar representing the mean of the losses. If `num_present` is zero,
      then zero is returned.
  """
  total_loss = math_ops.reduce_sum(losses)
  return _safe_div(total_loss, num_present)


@deprecated("2016-12-30", "Use tf.losses.compute_weighted_loss instead.")
def compute_weighted_loss(losses, weights=1.0, scope=None):
  """Computes the weighted loss.

  Args:
    losses: A tensor of size [batch_size, d1, ... dN].
    weights: A tensor of size [1] or [batch_size, d1, ... dK] where K < N.
    scope: the scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` that returns the weighted loss.

  Raises:
    ValueError: If `weights` is `None` or the shape is not compatible with
      `losses`, or if the number of dimensions (rank) of either `losses` or
      `weights` is missing.
  """
  with ops.name_scope(scope, "weighted_loss", [losses, weights]):
    losses = ops.convert_to_tensor(losses)
    input_dtype = losses.dtype
    losses = math_ops.to_float(losses)
    weights = math_ops.to_float(ops.convert_to_tensor(weights))

    if losses.get_shape().ndims is None:
      raise ValueError("losses.get_shape().ndims cannot be None")
    weights_shape = weights.get_shape()
    if weights_shape.ndims is None:
      raise ValueError("weights.get_shape().ndims cannot be None")

    if weights_shape.ndims > 1 and weights_shape.dims[-1].is_compatible_with(1):
      weights = array_ops.squeeze(weights, [-1])

    total_loss = _scale_losses(losses, weights)
    num_present = _num_present(losses, weights)
    mean_loss = _safe_mean(total_loss, num_present)
    # convert the result back to the input type
    mean_loss = math_ops.cast(mean_loss, input_dtype)
    add_loss(mean_loss)
    return mean_loss


def _num_present(losses, weights, per_batch=False):
  """Computes the number of elements in the loss function induced by `weights`.

  A given weights tensor induces different numbers of usable elements in the
  `losses` tensor. The `weights` tensor is broadcast across `losses` for all
  possible dimensions. For example, if `losses` is a tensor of dimension
  [4, 5, 6, 3] and `weights` is a tensor of size [4, 5], then `weights` is, in
  effect, tiled to match the size of `losses`. Following this effective tile,
  the total number of present elements is the number of non-zero weights.

  Args:
    losses: A tensor of size [batch_size, d1, ... dN].
    weights: A tensor of size [1] or [batch_size, d1, ... dK] where K < N.
    per_batch: Whether to return the number of elements per batch or as a sum
      total.

  Returns:
    The number of present (non-zero) elements in the losses tensor. If
      `per_batch` is True, the value is returned as a tensor of size
      [batch_size]. Otherwise, a single scalar tensor is returned.
  """
  # If weights is a scalar, its easy to compute:
  if weights.get_shape().ndims == 0:
    batch_size = array_ops.reshape(
        array_ops.slice(array_ops.shape(losses), [0], [1]), [])
    num_per_batch = math_ops.div(
        math_ops.to_float(array_ops.size(losses)),
        math_ops.to_float(batch_size))
    num_per_batch = array_ops.where(
        math_ops.equal(weights, 0), 0.0, num_per_batch)
    num_per_batch = math_ops.multiply(
        array_ops.ones(array_ops.reshape(batch_size, [1])), num_per_batch)
    return num_per_batch if per_batch else math_ops.reduce_sum(num_per_batch)

  # First, count the number of nonzero weights:
  if weights.get_shape().ndims >= 1:
    reduction_indices = list(range(1, weights.get_shape().ndims))
    num_nonzero_per_batch = math_ops.reduce_sum(
        math_ops.to_float(math_ops.not_equal(weights, 0)),
        reduction_indices=reduction_indices)

  # Next, determine the number of elements that weights would broadcast to:
  broadcast_dims = array_ops.slice(
      array_ops.shape(losses), [weights.get_shape().ndims], [-1])
  num_to_broadcast = math_ops.to_float(math_ops.reduce_prod(broadcast_dims))

  num_per_batch = math_ops.multiply(num_nonzero_per_batch, num_to_broadcast)
  return num_per_batch if per_batch else math_ops.reduce_sum(num_per_batch)


@deprecated("2016-12-30", "Use tf.losses.add_loss instead.")
@add_arg_scope
def add_loss(loss, loss_collection=ops.GraphKeys.LOSSES):
  """Adds a externally defined loss to the collection of losses.

  Args:
    loss: A loss `Tensor`.
    loss_collection: Optional collection to add the loss to.
  """
  if loss_collection:
    ops.add_to_collection(loss_collection, loss)


@deprecated("2016-12-30", "Use tf.losses.get_losses instead.")
def get_losses(scope=None, loss_collection=ops.GraphKeys.LOSSES):
  """Gets the list of losses from the loss_collection.

  Args:
    scope: an optional scope for filtering the losses to return.
    loss_collection: Optional losses collection.

  Returns:
    a list of loss tensors.
  """
  return ops.get_collection(loss_collection, scope)


@deprecated("2016-12-30", "Use tf.losses.get_regularization_losses instead.")
def get_regularization_losses(scope=None):
  """Gets the regularization losses.

  Args:
    scope: an optional scope for filtering the losses to return.

  Returns:
    A list of regularization losses as Tensors.
  """
  return ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES, scope)


@deprecated("2016-12-30", "Use tf.losses.get_total_loss instead.")
def get_total_loss(add_regularization_losses=True, name="total_loss"):
  """Returns a tensor whose value represents the total loss.

  Notice that the function adds the given losses to the regularization losses.

  Args:
    add_regularization_losses: A boolean indicating whether or not to use the
      regularization losses in the sum.
    name: The name of the returned tensor.

  Returns:
    A `Tensor` whose value represents the total loss.

  Raises:
    ValueError: if `losses` is not iterable.
  """
  losses = get_losses()
  if add_regularization_losses:
    losses += get_regularization_losses()
  return math_ops.add_n(losses, name=name)


@deprecated("2016-12-30", "Use tf.losses.absolute_difference instead.")
def absolute_difference(predictions, labels=None, weights=1.0, scope=None):
  """Adds an Absolute Difference loss to the training procedure.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  Args:
    predictions: The predicted outputs.
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    weights: Coefficients for the loss a scalar, a tensor of shape
      [batch_size] or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.
  """
  with ops.name_scope(scope, "absolute_difference",
                      [predictions, labels, weights]) as scope:
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    predictions = math_ops.to_float(predictions)
    labels = math_ops.to_float(labels)
    losses = math_ops.abs(math_ops.subtract(predictions, labels))
    return compute_weighted_loss(losses, weights, scope=scope)


@deprecated("2016-12-30",
            "Use tf.losses.sigmoid_cross_entropy instead. Note that the order "
            "of the predictions and labels arguments has been changed.")
def sigmoid_cross_entropy(logits,
                          multi_class_labels,
                          weights=1.0,
                          label_smoothing=0,
                          scope=None):
  """Creates a cross-entropy loss using tf.nn.sigmoid_cross_entropy_with_logits.

  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of size [`batch_size`], then the loss weights apply to each
  corresponding sample.

  If `label_smoothing` is nonzero, smooth the labels towards 1/2:

      new_multiclass_labels = multiclass_labels * (1 - label_smoothing)
                              + 0.5 * label_smoothing

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    multi_class_labels: [batch_size, num_classes] labels in (0, 1).
    weights: Coefficients for the loss. The tensor must be a scalar, a tensor of
      shape [batch_size] or shape [batch_size, num_classes].
    label_smoothing: If greater than 0 then smooth the labels.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `logits` doesn't match that of
      `multi_class_labels` or if the shape of `weights` is invalid, or if
      `weights` is None.
  """
  with ops.name_scope(scope, "sigmoid_cross_entropy_loss",
                      [logits, multi_class_labels, weights]) as scope:
    logits.get_shape().assert_is_compatible_with(multi_class_labels.get_shape())

    multi_class_labels = math_ops.cast(multi_class_labels, logits.dtype)

    if label_smoothing > 0:
      multi_class_labels = (
          multi_class_labels * (1 - label_smoothing) + 0.5 * label_smoothing)

    losses = nn.sigmoid_cross_entropy_with_logits(
        labels=multi_class_labels, logits=logits, name="xentropy")
    return compute_weighted_loss(losses, weights, scope=scope)


@deprecated("2016-12-30",
            "Use tf.losses.softmax_cross_entropy instead. Note that the order "
            "of the logits and labels arguments has been changed.")
def softmax_cross_entropy(logits,
                          onehot_labels,
                          weights=1.0,
                          label_smoothing=0,
                          scope=None):
  """Creates a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits.

  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of size [`batch_size`], then the loss weights apply to each
  corresponding sample.

  If `label_smoothing` is nonzero, smooth the labels towards 1/num_classes:
      new_onehot_labels = onehot_labels * (1 - label_smoothing)
                          + label_smoothing / num_classes

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    onehot_labels: [batch_size, num_classes] one-hot-encoded labels.
    weights: Coefficients for the loss. The tensor must be a scalar or a tensor
      of shape [batch_size].
    label_smoothing: If greater than 0 then smooth the labels.
    scope: the scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the mean loss value.

  Raises:
    ValueError: If the shape of `logits` doesn't match that of `onehot_labels`
      or if the shape of `weights` is invalid or if `weights` is None.
  """
  with ops.name_scope(scope, "softmax_cross_entropy_loss",
                      [logits, onehot_labels, weights]) as scope:
    logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())

    onehot_labels = math_ops.cast(onehot_labels, logits.dtype)

    if label_smoothing > 0:
      num_classes = math_ops.cast(
          array_ops.shape(onehot_labels)[1], logits.dtype)
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      onehot_labels = onehot_labels * smooth_positives + smooth_negatives

    losses = nn.softmax_cross_entropy_with_logits(
        labels=onehot_labels, logits=logits, name="xentropy")
    return compute_weighted_loss(losses, weights, scope=scope)


@deprecated("2016-12-30",
            "Use tf.losses.sparse_softmax_cross_entropy instead. Note that "
            "the order of the logits and labels arguments has been changed.")
def sparse_softmax_cross_entropy(logits, labels, weights=1.0, scope=None):
  """Cross-entropy loss using `tf.nn.sparse_softmax_cross_entropy_with_logits`.

  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of size [`batch_size`], then the loss weights apply to each
  corresponding sample.

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    labels: [batch_size, 1] or [batch_size] labels of dtype `int32` or `int64`
      in the range `[0, num_classes)`.
    weights: Coefficients for the loss. The tensor must be a scalar or a tensor
      of shape [batch_size] or [batch_size, 1].
    scope: the scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the mean loss value.

  Raises:
    ValueError: If the shapes of `logits`, `labels`, and `weights` are
      incompatible, or if `weights` is None.
  """
  with ops.name_scope(scope, "sparse_softmax_cross_entropy_loss",
                      [logits, labels, weights]) as scope:
    labels = array_ops.reshape(labels, shape=[array_ops.shape(labels)[0]])

    losses = nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name="xentropy")
    return compute_weighted_loss(losses, weights, scope=scope)


@deprecated("2016-12-30",
            "Use tf.losses.log_loss instead. Note that the order of the "
            "predictions and labels arguments has been changed.")
def log_loss(predictions, labels=None, weights=1.0, epsilon=1e-7, scope=None):
  """Adds a Log Loss term to the training procedure.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  Args:
    predictions: The predicted outputs.
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    weights: Coefficients for the loss a scalar, a tensor of shape
      [batch_size] or a tensor whose shape matches `predictions`.
    epsilon: A small increment to add to avoid taking a log of zero.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.
  """
  with ops.name_scope(scope, "log_loss",
                      [predictions, labels, weights]) as scope:
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    predictions = math_ops.to_float(predictions)
    labels = math_ops.to_float(labels)
    losses = -math_ops.multiply(
        labels, math_ops.log(predictions + epsilon)) - math_ops.multiply(
            (1 - labels), math_ops.log(1 - predictions + epsilon))
    return compute_weighted_loss(losses, weights, scope=scope)


@deprecated("2016-12-30",
            "Use tf.losses.hinge_loss instead. Note that the order of the "
            "logits and labels arguments has been changed, and to stay "
            "unweighted, reduction=Reduction.NONE")
def hinge_loss(logits, labels=None, scope=None):
  """Method that returns the loss tensor for hinge loss.

  Args:
    logits: The logits, a float tensor. Note that logits are assumed to be
      unbounded and 0-centered. A value > 0 (resp. < 0) is considered a positive
      (resp. negative) binary prediction.
    labels: The ground truth output tensor. Its shape should match the shape of
      logits. The values of the tensor are expected to be 0.0 or 1.0. Internally
      the {0,1} labels are converted to {-1,1} when calculating the hinge loss.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    An unweighted `Tensor` of same shape as `logits` and `labels` representing
    the
      loss values across the batch.

  Raises:
    ValueError: If the shapes of `logits` and `labels` don't match.
  """
  with ops.name_scope(scope, "hinge_loss", [logits, labels]) as scope:
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    # We first need to convert binary labels to -1/1 labels (as floats).
    labels = math_ops.to_float(labels)
    all_ones = array_ops.ones_like(labels)
    labels = math_ops.subtract(2 * labels, all_ones)
    return nn_ops.relu(
        math_ops.subtract(all_ones, math_ops.multiply(labels, logits)))


@deprecated("2016-12-30", "Use tf.losses.mean_squared_error instead.")
def mean_squared_error(predictions, labels=None, weights=1.0, scope=None):
  """Adds a Sum-of-Squares loss to the training procedure.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  Args:
    predictions: The predicted outputs.
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    weights: Coefficients for the loss a scalar, a tensor of shape
      [batch_size] or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.
  """
  with ops.name_scope(scope, "mean_squared_error",
                      [predictions, labels, weights]) as scope:
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    predictions = math_ops.to_float(predictions)
    labels = math_ops.to_float(labels)
    losses = math_ops.square(math_ops.subtract(predictions, labels))
    return compute_weighted_loss(losses, weights, scope=scope)


@deprecated("2016-12-30",
            "Use tf.losses.mean_pairwise_squared_error instead. Note that the "
            "order of the predictions and labels arguments has been changed.")
def mean_pairwise_squared_error(predictions,
                                labels=None,
                                weights=1.0,
                                scope=None):
  """Adds a pairwise-errors-squared loss to the training procedure.

  Unlike `mean_squared_error`, which is a measure of the differences between
  corresponding elements of `predictions` and `labels`,
  `mean_pairwise_squared_error` is a measure of the differences between pairs of
  corresponding elements of `predictions` and `labels`.

  For example, if `labels`=[a, b, c] and `predictions`=[x, y, z], there are
  three pairs of differences are summed to compute the loss:
    loss = [ ((a-b) - (x-y)).^2 + ((a-c) - (x-z)).^2 + ((b-c) - (y-z)).^2 ] / 3

  Note that since the inputs are of size [batch_size, d0, ... dN], the
  corresponding pairs are computed within each batch sample but not across
  samples within a batch. For example, if `predictions` represents a batch of
  16 grayscale images of dimension [batch_size, 100, 200], then the set of pairs
  is drawn from each image, but not across images.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector.

  Args:
    predictions: The predicted outputs, a tensor of size [batch_size, d0, .. dN]
      where N+1 is the total number of dimensions in `predictions`.
    labels: The ground truth output tensor, whose shape must match the shape of
      the `predictions` tensor.
    weights: Coefficients for the loss a scalar, a tensor of shape [batch_size]
      or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.
  """
  with ops.name_scope(scope, "mean_pairwise_squared_error",
                      [predictions, labels, weights]) as scope:
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    predictions = math_ops.to_float(predictions)
    labels = math_ops.to_float(labels)
    weights = math_ops.to_float(ops.convert_to_tensor(weights))

    diffs = math_ops.subtract(predictions, labels)

    # Need to verify here since the function doesn't use compute_weighted_loss
    if diffs.get_shape().ndims is None:
      raise ValueError("diffs.get_shape().ndims cannot be None")
    if weights.get_shape().ndims is None:
      raise ValueError("weights.get_shape().ndims cannot be None")

    reduction_indices = list(range(1, diffs.get_shape().ndims))

    sum_squares_diff_per_batch = math_ops.reduce_sum(
        math_ops.square(diffs), reduction_indices=reduction_indices)
    num_present_per_batch = _num_present(diffs, weights, per_batch=True)

    term1 = 2.0 * _safe_div(sum_squares_diff_per_batch, num_present_per_batch)

    sum_diff = math_ops.reduce_sum(diffs, reduction_indices=reduction_indices)
    term2 = 2.0 * _safe_div(
        math_ops.square(sum_diff), math_ops.square(num_present_per_batch))

    loss = _scale_losses(term1 - term2, weights)

    mean_loss = array_ops.where(
        math_ops.reduce_sum(num_present_per_batch) > 0,
        loss,
        array_ops.zeros_like(loss),
        name="value")
    add_loss(mean_loss)
    return mean_loss


@deprecated("2016-12-30", "Use tf.losses.cosine_distance instead.")
@deprecated_args(None, "dim is deprecated, use axis instead", "dim")
def cosine_distance(predictions,
                    labels=None,
                    axis=None,
                    weights=1.0,
                    scope=None,
                    dim=None):
  """Adds a cosine-distance loss to the training procedure.

  Note that the function assumes that `predictions` and `labels` are already
  unit-normalized.

  Args:
    predictions: An arbitrary matrix.
    labels: A `Tensor` whose shape matches 'predictions'
    axis: The dimension along which the cosine distance is computed.
    weights: Coefficients for the loss a scalar, a tensor of shape
      [batch_size] or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.
    dim: The old (deprecated) name for `axis`.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If `predictions` shape doesn't match `labels` shape, or
      `weights` is `None`.
  """
  axis = deprecated_argument_lookup(
      "axis", axis, "dim", dim)
  if axis is None:
    raise ValueError("You must specify 'axis'.")
  with ops.name_scope(scope, "cosine_distance_loss",
                      [predictions, labels, weights]) as scope:
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    predictions = math_ops.to_float(predictions)
    labels = math_ops.to_float(labels)

    radial_diffs = math_ops.multiply(predictions, labels)
    losses = 1 - math_ops.reduce_sum(
        radial_diffs, reduction_indices=[
            axis,
        ])
    return compute_weighted_loss(losses, weights, scope=scope)
