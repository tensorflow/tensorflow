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

from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops


__all__ = ["absolute_difference",
           "add_loss",
           "cosine_distance",
           "compute_weighted_loss",
           "get_losses",
           "get_regularization_losses",
           "get_total_loss",
           "hinge_loss",
           "log_loss",
           "mean_pairwise_squared_error",
           "mean_squared_error",
           "sigmoid_cross_entropy",
           "softmax_cross_entropy",
           "sparse_softmax_cross_entropy",
           "sum_of_pairwise_squares",
           "sum_of_squares"]


def _scale_losses(losses, weight):
  """Computes the scaled loss.

  Args:
    losses: A `Tensor` of size [batch_size, d1, ... dN].
    weight: A `Tensor` of size [1], [batch_size] or [batch_size, d1, ... dN].
      The `losses` are reduced (tf.reduce_sum) until its dimension matches
      that of `weight` at which point the reduced `losses` are element-wise
      multiplied by `weight` and a final reduce_sum is computed on the result.
      Conceptually, this operation is equivalent to broadcasting (tiling)
      `weight` to be the same size as `losses`, performing an element-wise
      multiplication, and summing the result.

  Returns:
    A scalar tf.float32 `Tensor` whose value represents the sum of the scaled
      `losses`.
  """
  # First, compute the sum of the losses over all elements:
  start_index = max(0, weight.get_shape().ndims)
  reduction_indices = list(range(start_index, losses.get_shape().ndims))
  reduced_losses = math_ops.reduce_sum(losses,
                                       reduction_indices=reduction_indices)
  reduced_losses = math_ops.mul(reduced_losses, weight)
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
  return math_ops.select(
      math_ops.greater(denominator, 0),
      math_ops.div(numerator, math_ops.select(
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


def compute_weighted_loss(losses, weight=1.0):
  """Computes the weighted loss.

  Args:
    losses: A tensor of size [batch_size, d1, ... dN].
    weight: A tensor of size [1] or [batch_size, d1, ... dK] where K < N.

  Returns:
    A scalar `Tensor` that returns the weighted loss.

  Raises:
    ValueError: If the weight is None or the shape is not compatible with the
      losses shape or if the number of dimensions (rank) of either losses or
      weight is missing.
  """
  if weight is None:
    raise ValueError("`weight` cannot be None")
  input_dtype = losses.dtype
  losses = math_ops.to_float(losses)
  weight = math_ops.to_float(ops.convert_to_tensor(weight))

  if losses.get_shape().ndims is None:
    raise ValueError("losses.get_shape().ndims cannot be None")
  if weight.get_shape().ndims is None:
    raise ValueError("weight.get_shape().ndims cannot be None")

  total_loss = _scale_losses(losses, weight)
  num_present = _num_present(losses, weight)
  mean_loss = _safe_mean(total_loss, num_present)
  # convert the result back to the input type
  mean_loss = math_ops.cast(mean_loss, input_dtype)
  add_loss(mean_loss)
  return mean_loss


def _num_present(losses, weight, per_batch=False):
  """Computes the number of elements in the loss function induced by `weight`.

  A given weight tensor induces different numbers of usable elements in the
  `losses` tensor. The `weight` tensor is broadcast across `losses` for all
  possible dimensions. For example, if `losses` is a tensor of dimension
  [4, 5, 6, 3] and weight is a tensor of size [4, 5], then weight is, in effect,
  tiled to match the size of `losses`. Following this effective tile, the total
  number of present elements is the number of non-zero weights.

  Args:
    losses: A tensor of size [batch_size, d1, ... dN].
    weight: A tensor of size [1] or [batch_size, d1, ... dK] where K < N.
    per_batch: Whether to return the number of elements per batch or as a sum
      total.

  Returns:
    The number of present (non-zero) elements in the losses tensor. If
      `per_batch` is True, the value is returned as a tensor of size
      [batch_size]. Otherwise, a single scalar tensor is returned.
  """
  # To ensure that dims of [2, 1] gets mapped to [2,]
  weight = array_ops.squeeze(weight)

  # If the weight is a scalar, its easy to compute:
  if weight.get_shape().ndims == 0:
    batch_size = array_ops.reshape(array_ops.slice(array_ops.shape(losses),
                                                   [0], [1]), [])
    num_per_batch = math_ops.div(math_ops.to_float(array_ops.size(losses)),
                                 math_ops.to_float(batch_size))
    num_per_batch = math_ops.select(math_ops.equal(weight, 0),
                                    0.0, num_per_batch)
    num_per_batch = math_ops.mul(array_ops.ones(
        array_ops.reshape(batch_size, [1])), num_per_batch)
    return num_per_batch if per_batch else math_ops.reduce_sum(num_per_batch)

  # First, count the number of nonzero weights:
  if weight.get_shape().ndims >= 1:
    reduction_indices = list(range(1, weight.get_shape().ndims))
    num_nonzero_per_batch = math_ops.reduce_sum(
        math_ops.to_float(math_ops.not_equal(weight, 0)),
        reduction_indices=reduction_indices)

  # Next, determine the number of elements that weight would broadcast to:
  broadcast_dims = array_ops.slice(array_ops.shape(losses),
                                   [weight.get_shape().ndims], [-1])
  num_to_broadcast = math_ops.to_float(math_ops.reduce_prod(broadcast_dims))

  num_per_batch = math_ops.mul(num_nonzero_per_batch, num_to_broadcast)
  return num_per_batch if per_batch else math_ops.reduce_sum(num_per_batch)


@add_arg_scope
def add_loss(loss, loss_collection=ops.GraphKeys.LOSSES):
  """Adds a externally defined loss to the collection of losses.

  Args:
    loss: A loss `Tensor`.
    loss_collection: Optional collection to add the loss to.
  """
  if loss_collection:
    ops.add_to_collection(loss_collection, loss)


def get_losses(scope=None, loss_collection=ops.GraphKeys.LOSSES):
  """Gets the list of losses from the loss_collection.

  Args:
    scope: an optional scope for filtering the losses to return.
    loss_collection: Optional losses collection.

  Returns:
    a list of loss tensors.
  """
  return ops.get_collection(loss_collection, scope)


def get_regularization_losses(scope=None):
  """Gets the regularization losses.

  Args:
    scope: an optional scope for filtering the losses to return.

  Returns:
    A list of loss variables.
  """
  return ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES, scope)


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


def absolute_difference(predictions, targets, weight=1.0, scope=None):
  """Adds an Absolute Difference loss to the training procedure.

  `weight` acts as a coefficient for the loss. If a scalar is provided, then the
  loss is simply scaled by the given value. If `weight` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weight` vector. If the shape of
  `weight` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weight`.

  Args:
    predictions: The predicted outputs.
    targets: The ground truth output tensor, same dimensions as 'predictions'.
    weight: Coefficients for the loss a scalar, a tensor of shape
      [batch_size] or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `targets` or
      if the shape of `weight` is invalid.
  """
  with ops.name_scope(scope, "absolute_difference",
                      [predictions, targets]) as scope:
    predictions.get_shape().assert_is_compatible_with(targets.get_shape())
    if weight is None:
      raise ValueError("`weight` cannot be None")
    predictions = math_ops.to_float(predictions)
    targets = math_ops.to_float(targets)
    losses = math_ops.abs(math_ops.sub(predictions, targets))
    return compute_weighted_loss(losses, weight)


def sigmoid_cross_entropy(logits, multi_class_labels, weight=1.0,
                          label_smoothing=0, scope=None):
  """Creates a cross-entropy loss using tf.nn.sigmoid_cross_entropy_with_logits.

  `weight` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weight` is a
  tensor of size [`batch_size`], then the loss weights apply to each
  corresponding sample.

  If `label_smoothing` is nonzero, smooth the labels towards 1/2:
      new_multiclass_labels = multiclass_labels * (1 - label_smoothing)
                              + 0.5 * label_smoothing

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    multi_class_labels: [batch_size, num_classes] target labels in (0, 1).
    weight: Coefficients for the loss. The tensor must be a scalar, a tensor of
      shape [batch_size] or shape [batch_size, num_classes].
    label_smoothing: If greater than 0 then smooth the labels.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `targets` or
      if the shape of `weight` is invalid or if `weight` is None.
  """
  with ops.name_scope(scope, "sigmoid_cross_entropy_loss",
                      [logits, multi_class_labels]):
    logits.get_shape().assert_is_compatible_with(multi_class_labels.get_shape())

    multi_class_labels = math_ops.cast(multi_class_labels, logits.dtype)

    if label_smoothing > 0:
      multi_class_labels = (multi_class_labels * (1 - label_smoothing) +
                            0.5 * label_smoothing)

    losses = nn.sigmoid_cross_entropy_with_logits(logits, multi_class_labels,
                                                  name="xentropy")
    return compute_weighted_loss(losses, weight)


def softmax_cross_entropy(logits, onehot_labels, weight=1.0,
                          label_smoothing=0, scope=None):
  """Creates a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits.

  `weight` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weight` is a
  tensor of size [`batch_size`], then the loss weights apply to each
  corresponding sample.

  If `label_smoothing` is nonzero, smooth the labels towards 1/num_classes:
      new_onehot_labels = onehot_labels * (1 - label_smoothing)
                          + label_smoothing / num_classes

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    onehot_labels: [batch_size, num_classes] target one_hot_encoded labels.
    weight: Coefficients for the loss. The tensor must be a scalar or a tensor
      of shape [batch_size].
    label_smoothing: If greater than 0 then smooth the labels.
    scope: the scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `logits` doesn't match that of `onehot_labels`
      or if the shape of `weight` is invalid or if `weight` is None.
  """
  with ops.name_scope(scope, "softmax_cross_entropy_loss",
                      [logits, onehot_labels]):
    logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())

    onehot_labels = math_ops.cast(onehot_labels, logits.dtype)

    if label_smoothing > 0:
      num_classes = math_ops.cast(
          array_ops.shape(onehot_labels)[1], logits.dtype)
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      onehot_labels = onehot_labels * smooth_positives + smooth_negatives

    losses = nn.softmax_cross_entropy_with_logits(logits, onehot_labels,
                                                  name="xentropy")
    return compute_weighted_loss(losses, weight)


def sparse_softmax_cross_entropy(logits, labels, weight=1.0, scope=None):
  """Cross-entropy loss using tf.nn.sparse_softmax_cross_entropy_with_logits.

  `weight` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weight` is a
  tensor of size [`batch_size`], then the loss weights apply to each
  corresponding sample.

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    labels: [batch_size, 1] or [batch_size] target labels of dtype `int32` or
      `int64` in the range `[0, num_classes)`.
    weight: Coefficients for the loss. The tensor must be a scalar or a tensor
      of shape [batch_size] or [batch_size, 1].
    scope: the scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shapes of logits, labels, and weight are incompatible, or
      if `weight` is None.
  """
  with ops.name_scope(scope, "sparse_softmax_cross_entropy_loss",
                      [logits, labels]):
    labels = array_ops.reshape(labels, shape=[array_ops.shape(labels)[0]])
    weight = array_ops.squeeze(weight)

    losses = nn.sparse_softmax_cross_entropy_with_logits(logits, labels,
                                                         name="xentropy")
    return compute_weighted_loss(losses, weight)


def log_loss(predictions, targets, weight=1.0, epsilon=1e-7, scope=None):
  """Adds a Log Loss term to the training procedure.

  `weight` acts as a coefficient for the loss. If a scalar is provided, then the
  loss is simply scaled by the given value. If `weight` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weight` vector. If the shape of
  `weight` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weight`.

  Args:
    predictions: The predicted outputs.
    targets: The ground truth output tensor, same dimensions as 'predictions'.
    weight: Coefficients for the loss a scalar, a tensor of shape
      [batch_size] or a tensor whose shape matches `predictions`.
    epsilon: A small increment to add to avoid taking a log of zero.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `targets` or
      if the shape of `weight` is invalid.
  """
  with ops.name_scope(scope, "log_loss",
                      [predictions, targets]) as scope:
    predictions.get_shape().assert_is_compatible_with(targets.get_shape())
    if weight is None:
      raise ValueError("`weight` cannot be None")
    predictions = math_ops.to_float(predictions)
    targets = math_ops.to_float(targets)
    losses = -math_ops.mul(
        targets,
        math_ops.log(predictions + epsilon)) - math_ops.mul(
            (1 - targets), math_ops.log(1 - predictions + epsilon))
    return compute_weighted_loss(losses, weight)


def hinge_loss(logits, target, scope=None):
  """Method that returns the loss tensor for hinge loss.

  Args:
    logits: The logits, a float tensor.
    target: The ground truth output tensor. Its shape should match the shape of
      logits. The values of the tensor are expected to be 0.0 or 1.0.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A `Tensor` of same shape as logits and target representing the loss values
      across the batch.

  Raises:
    ValueError: If the shapes of `logits` and `target` don't match.
  """
  with ops.name_scope(scope, "hinge_loss", [logits, target]) as scope:
    logits.get_shape().assert_is_compatible_with(target.get_shape())
    # We first need to convert binary labels to -1/1 labels (as floats).
    target = math_ops.to_float(target)
    all_ones = array_ops.ones_like(target)
    labels = math_ops.sub(2 * target, all_ones)
    losses = nn_ops.relu(math_ops.sub(all_ones, math_ops.mul(labels, logits)))
    return losses


@deprecated("2016-10-01", "Use mean_squared_error.")
def sum_of_squares(predictions, targets, weight=1.0, scope=None):
  """Adds a Sum-of-Squares loss to the training procedure.

  `weight` acts as a coefficient for the loss. If a scalar is provided, then the
  loss is simply scaled by the given value. If `weight` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weight` vector. If the shape of
  `weight` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weight`.

  Args:
    predictions: The predicted outputs.
    targets: The ground truth output tensor, same dimensions as 'predictions'.
    weight: Coefficients for the loss a scalar, a tensor of shape
      [batch_size] or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `targets` or
      if the shape of `weight` is invalid.
  """
  with ops.name_scope(scope, "sum_of_squares_loss",
                      [predictions, targets]) as scope:
    predictions.get_shape().assert_is_compatible_with(targets.get_shape())
    if weight is None:
      raise ValueError("`weight` cannot be None")
    predictions = math_ops.to_float(predictions)
    targets = math_ops.to_float(targets)
    losses = math_ops.square(math_ops.sub(predictions, targets))
    return compute_weighted_loss(losses, weight)


mean_squared_error = sum_of_squares


@deprecated("2016-10-01", "Use mean_pairwise_squared_error.")
def sum_of_pairwise_squares(predictions, targets, weight=1.0, scope=None):
  """Adds a pairwise-errors-squared loss to the training procedure.

  Unlike the sum_of_squares loss, which is a measure of the differences between
  corresponding elements of `predictions` and `targets`, sum_of_pairwise_squares
  is a measure of the differences between pairs of corresponding elements of
  `predictions` and `targets`.

  For example, if `targets`=[a, b, c] and `predictions`=[x, y, z], there are
  three pairs of differences are summed to compute the loss:
    loss = [ ((a-b) - (x-y)).^2 + ((a-c) - (x-z)).^2 + ((b-c) - (y-z)).^2 ] / 3

  Note that since the inputs are of size [batch_size, d0, ... dN], the
  corresponding pairs are computed within each batch sample but not across
  samples within a batch. For example, if `predictions` represents a batch of
  16 grayscale images of dimenion [batch_size, 100, 200], then the set of pairs
  is drawn from each image, but not across images.

  `weight` acts as a coefficient for the loss. If a scalar is provided, then the
  loss is simply scaled by the given value. If `weight` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weight` vector.

  Args:
    predictions: The predicted outputs, a tensor of size [batch_size, d0, .. dN]
      where N+1 is the total number of dimensions in `predictions`.
    targets: The ground truth output tensor, whose shape must match the shape of
      the `predictions` tensor.
    weight: Coefficients for the loss a scalar, a tensor of shape [batch_size]
      or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `targets` or
      if the shape of `weight` is invalid.
  """
  with ops.name_scope(scope, "sum_of_pairwise_squares_loss",
                      [predictions, targets]) as scope:
    predictions.get_shape().assert_is_compatible_with(targets.get_shape())
    if weight is None:
      raise ValueError("`weight` cannot be None")
    predictions = math_ops.to_float(predictions)
    targets = math_ops.to_float(targets)
    weight = math_ops.to_float(ops.convert_to_tensor(weight))

    diffs = math_ops.sub(predictions, targets)

    # Need to verify here since the function doesn't use compute_weighted_loss
    if diffs.get_shape().ndims is None:
      raise ValueError("diffs.get_shape().ndims cannot be None")
    if weight.get_shape().ndims is None:
      raise ValueError("weight.get_shape().ndims cannot be None")

    reduction_indices = list(range(1, diffs.get_shape().ndims))

    sum_squares_diff_per_batch = math_ops.reduce_sum(
        math_ops.square(diffs),
        reduction_indices=reduction_indices)
    num_present_per_batch = _num_present(diffs, weight, per_batch=True)

    term1 = 2.0 * _safe_div(sum_squares_diff_per_batch,
                            num_present_per_batch)

    sum_diff = math_ops.reduce_sum(diffs, reduction_indices=reduction_indices)
    term2 = 2.0 * _safe_div(math_ops.square(sum_diff),
                            math_ops.square(num_present_per_batch))

    loss = _scale_losses(term1 - term2, weight)

    mean_loss = math_ops.select(math_ops.reduce_sum(num_present_per_batch) > 0,
                                loss,
                                array_ops.zeros_like(loss),
                                name="value")
    add_loss(mean_loss)
    return mean_loss


mean_pairwise_squared_error = sum_of_pairwise_squares


def cosine_distance(predictions, targets, dim, weight=1.0, scope=None):
  """Adds a cosine-distance loss to the training procedure.

  Note that the function assumes that the predictions and targets are already
  unit-normalized.

  Args:
    predictions: An arbitrary matrix.
    targets: A `Tensor` whose shape matches 'predictions'
    dim: The dimension along which the cosine distance is computed.
    weight: Coefficients for the loss a scalar, a tensor of shape
      [batch_size] or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If predictions.shape doesn't match targets.shape, if the ignore
                mask is provided and its shape doesn't match targets.shape or if
                the ignore mask is not boolean valued.
  """
  with ops.name_scope(scope, "cosine_distance_loss",
                      [predictions, targets]) as scope:
    predictions.get_shape().assert_is_compatible_with(targets.get_shape())
    if weight is None:
      raise ValueError("`weight` cannot be None")

    predictions = math_ops.to_float(predictions)
    targets = math_ops.to_float(targets)

    radial_diffs = math_ops.mul(predictions, targets)
    losses = 1 - math_ops.reduce_sum(radial_diffs, reduction_indices=[dim,])
    return compute_weighted_loss(losses, weight)
