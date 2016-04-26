# Copyright 2016 Google Inc. All Rights Reserved.
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
"""## Loss operations for use in neural networks.

All of the loss functions take a pair of predictions and ground truth labels,
from which the loss is computed. It is assumed that the shape of both these
tensors is of the form [batch_size, d1, ... dN] where `batch_size` is the number
of samples in the batch and `d1` ... `dN` are the remaining dimensions.

It is common, when training with multiple loss functions, to adjust the relative
strengths of individual losses. This is performed by rescaling the losses via
a `weight` parameter passed to the loss functions. For example, if we were
training with both log_loss and sum_of_squares_loss, and we wished that the
log_loss penalty be twice as severe as the sum_of_squares_loss, we would
implement this as:

  # Explicitely set the weight.
  tf.contrib.losses.log(predictions, targets, weight=2.0)

  # Uses default weight of 1.0
  tf.contrib.losses.sum_of_squares(predictions, targets)

While specifying a scalar loss rescales the loss over the entire batch,
we sometimes want to rescale the loss per batch sample. For example, if we have
certain examples that matter more to us to get correctly, we might want to have
a higher loss that other samples whose mistakes matter less. In this case, we
can provide a weight vector of length `batch_size` which results in the loss
for each sample in the batch being scaled by the corresponding weight element.
For example, consider the case of a classification problem where we want to
maximize our accuracy but we especially interested in obtaining high accuracy
for a specific class:

  inputs, labels = LoadData(batch_size=3)
  logits = MyModelPredictions(inputs)

  # Ensures that the loss for examples whose ground truth class is `3` is 5x
  # higher than the loss for all other examples.
  weight = tf.mul(4, tf.cast(tf.equal(labels, 3), tf.float32)) + 1

  onehot_labels = tf.one_hot(labels, num_classes=5)
  tf.contrib.losses.softmax_cross_entropy(logits, onehot_labels, weight=weight)

Finally, in certain cases, we may want to specify a different loss for every
single measurable value. For example, if we are performing per-pixel depth
prediction, or per-pixel denoising, a single batch sample has P values where P
is the number of pixels in the image. For many losses, the number of measurable
values matches the number of elements in the predictions and targets tensors.
For others, such as softmax_cross_entropy and cosine_distance, the
loss functions reduces the dimensions of the inputs to produces a tensor of
losses for each measurable value. For example, softmax_cross_entropy takes as
input predictions and labels of dimension [batch_size, num_classes] but the
number of measurable values is [batch_size]. Consequently, when passing a weight
tensor to specify a different loss for every measurable value, the dimension of
the tensor will depend on the loss being used.

For a concrete example, consider the case of per-pixel depth prediction where
certain ground truth depth values are missing (due to sensor noise in the
capture process). In this case, we want to assign zero weight to losses for
these predictions.

  # 'depths' that are missing have a value of 0:
  images, depths = LoadData(...)
  predictions = MyModelPredictions(images)

  weight = tf.cast(tf.greater(depths, 0), tf.float32)
  tf.contrib.losses.sum_of_squares(predictions, depths, weight)

Note that when using weights for the losses, the final average is computed
by rescaling the losses by the weights and then dividing by the total number of
non-zero samples. For an arbitrary set of weights, this may not necessarily
produce a weighted average. Instead, it simply and transparently rescales the
per-element losses before averaging over the number of observations. For example
if the losses computed by the loss function is an array [4, 1, 2, 3] and the
weights are an array [1, 0.5, 3, 9], then the average loss is:

  (4*1 + 1*0.5 + 2*3 + 3*9) / 4

However, with a single loss function and an arbitrary set of weights, one can
still easily create a loss function such that the resulting loss is a
weighted average over the individual prediction errors:

  images, labels = LoadData(...)
  predictions = MyModelPredictions(images)

  weight = MyComplicatedWeightingFunction(labels)
  weight = tf.div(weight, tf.size(weight))
  tf.contrib.losses.sum_of_squares(predictions, depths, weight)


@@absolute_difference
@@cosine_distance
@@log
@@sigmoid_cross_entropy
@@softmax_cross_entropy
@@sum_of_pairwise_squares
@@sum_of_squares
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

__all__ = [
    "absolute_difference",
    "cosine_distance",
    "log",
    "sigmoid_cross_entropy",
    "softmax_cross_entropy",
    "sum_of_pairwise_squares",
    "sum_of_squares",
]


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
  return math_ops.select(
      math_ops.greater(num_present, 0),
      math_ops.div(total_loss, math_ops.select(
          math_ops.equal(num_present, 0), 1.0, num_present)),
      array_ops.zeros_like(total_loss),
      name="value")


def _compute_weighted_loss(losses, weight):
  """Computes the weighted loss.

  Args:
    losses: A tensor of size [batch_size, d1, ... dN].
    weight: A tensor of size [1] or [batch_size, d1, ... dK] where K < N.

  Returns:
    A scalar `Tensor` that returns the weighted loss.

  Raises:
    ValueError: If the weight shape is not compatible with the losses shape or
      if the number of dimensions (rank) of either losses or weight is missing.
  """
  losses = math_ops.to_float(losses)
  weight = math_ops.to_float(ops.convert_to_tensor(weight))

  if losses.get_shape().ndims is None:
    raise ValueError("losses.get_shape().ndims cannot be None")
  if weight.get_shape().ndims is None:
    raise ValueError("weight.get_shape().ndims cannot be None")

  total_loss = _scale_losses(losses, weight)
  num_present = _num_present(losses, weight)
  return _safe_mean(total_loss, num_present)


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
  with ops.op_scope([predictions, targets],
                    scope, "sum_of_squares_loss") as scope:
    predictions.get_shape().assert_is_compatible_with(targets.get_shape())
    if weight is None:
      raise ValueError("`weight` cannot be None")
    predictions = math_ops.to_float(predictions)
    targets = math_ops.to_float(targets)
    losses = math_ops.abs(math_ops.sub(predictions, targets))
    return _compute_weighted_loss(losses, weight)


def sigmoid_cross_entropy(logits, multi_class_labels, weight=1.0,
                          label_smoothing=0, scope=None):
  """Creates a cross-entropy loss using tf.nn.sigmoid_cross_entropy_with_logits.

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    multi_class_labels: [batch_size, num_classes] target labels in (0, 1).
    weight: Coefficients for the loss. The tensor must be a scalar, a tensor of
      shape [batch_size] or shape [batch_size, num_classes].
    label_smoothing: If greater than 0 then smooth the labels.
    scope: The scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.
  """
  with ops.op_scope([logits, multi_class_labels],
                    scope, "sigmoid_cross_entropy_loss"):
    return _cross_entropy(logits, multi_class_labels, weight,
                          label_smoothing,
                          activation_fn=nn.sigmoid_cross_entropy_with_logits)


def softmax_cross_entropy(logits, onehot_labels, weight=1.0,
                          label_smoothing=0, scope=None):
  """Creates a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits.

  It can scale the loss by weight factor, and smooth the labels.

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    onehot_labels: [batch_size, num_classes] target one_hot_encoded labels.
    weight: Coefficients for the loss. The tensor must be a scalar or a tensor
      of shape [batch_size].
    label_smoothing: If greater than 0 then smooth the labels.
    scope: the scope for the operations performed in computing the loss.

  Returns:
    A scalar `Tensor` representing the loss value.
  """
  with ops.op_scope([logits, onehot_labels],
                    scope, "softmax_cross_entropy_loss"):
    return _cross_entropy(logits, onehot_labels, weight,
                          label_smoothing,
                          activation_fn=nn.softmax_cross_entropy_with_logits)


def _cross_entropy(logits, onehot_labels, weight, label_smoothing,
                   activation_fn):
  """Adds a CrossEntropyLoss to the losses collection.

  `weight` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weight` is a
  tensor of size [`batch_size`], then the loss weights apply to each
  corresponding sample.

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    onehot_labels: [batch_size, num_classes] target one_hot_encoded labels.
    weight: Coefficients for the loss. If the activation is SIGMOID, then the
      weight shape must be one of [1], [batch_size] or logits.shape().
      Otherwise, the weight shape must be either [1] or [batch_size].
    label_smoothing: If greater than 0 then smooth the labels.
    activation_fn: The activation function to use. The method must take three
      arguments, the logits, the labels, and an operation name.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `targets` or
      if the shape of `weight` is invalid or if `weight` is None.
  """
  logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())
  if weight is None:
    raise ValueError("`weight` cannot be None")

  onehot_labels = math_ops.cast(onehot_labels, logits.dtype)

  if label_smoothing > 0:
    num_classes = onehot_labels.get_shape()[1].value
    smooth_positives = 1.0 - label_smoothing
    smooth_negatives = label_smoothing / num_classes
    onehot_labels = onehot_labels * smooth_positives + smooth_negatives

  losses = activation_fn(logits, onehot_labels, name="xentropy")
  return _compute_weighted_loss(losses, weight)


def log(predictions, targets, weight=1.0, epsilon=1e-7, scope=None):
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
  with ops.op_scope([predictions, targets],
                    scope, "log_loss") as scope:
    predictions.get_shape().assert_is_compatible_with(targets.get_shape())
    if weight is None:
      raise ValueError("`weight` cannot be None")
    predictions = math_ops.to_float(predictions)
    targets = math_ops.to_float(targets)
    losses = -math_ops.mul(
        targets,
        math_ops.log(predictions + epsilon)) - math_ops.mul(
            (1 - targets), math_ops.log(1 - predictions + epsilon))
    return _compute_weighted_loss(losses, weight)


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
  with ops.op_scope([predictions, targets],
                    scope, "sum_of_squares_loss") as scope:
    predictions.get_shape().assert_is_compatible_with(targets.get_shape())
    if weight is None:
      raise ValueError("`weight` cannot be None")
    predictions = math_ops.to_float(predictions)
    targets = math_ops.to_float(targets)
    losses = math_ops.square(math_ops.sub(predictions, targets))
    return _compute_weighted_loss(losses, weight)


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
  with ops.op_scope([predictions, targets],
                    scope, "sum_of_pairwise_squares_loss") as scope:
    predictions.get_shape().assert_is_compatible_with(targets.get_shape())
    if weight is None:
      raise ValueError("`weight` cannot be None")
    predictions = math_ops.to_float(predictions)
    targets = math_ops.to_float(targets)
    weight = math_ops.to_float(ops.convert_to_tensor(weight))

    diffs = math_ops.sub(predictions, targets)

    # Need to verify here since the function doesn't use _compute_weighted_loss
    if diffs.get_shape().ndims is None:
      raise ValueError("diffs.get_shape().ndims cannot be None")
    if weight.get_shape().ndims is None:
      raise ValueError("weight.get_shape().ndims cannot be None")

    reduction_indices = list(range(1, diffs.get_shape().ndims))

    sum_squares_diff_per_batch = math_ops.reduce_sum(
        math_ops.square(diffs),
        reduction_indices=reduction_indices)
    num_present_per_batch = _num_present(diffs, weight, per_batch=True)

    term1 = 2.0 * math_ops.div(sum_squares_diff_per_batch,
                               num_present_per_batch)

    sum_diff = math_ops.reduce_sum(diffs, reduction_indices=reduction_indices)
    term2 = 2.0 * math_ops.div(math_ops.square(sum_diff),
                               math_ops.square(num_present_per_batch))

    loss = _scale_losses(term1 - term2, weight)

    return math_ops.select(math_ops.reduce_sum(num_present_per_batch) > 0,
                           loss,
                           array_ops.zeros_like(loss),
                           name="value")


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
  with ops.op_scope([predictions, targets],
                    scope, "cosine_distance_loss") as scope:
    predictions.get_shape().assert_is_compatible_with(targets.get_shape())
    if weight is None:
      raise ValueError("`weight` cannot be None")

    predictions = math_ops.to_float(predictions)
    targets = math_ops.to_float(targets)

    radial_diffs = math_ops.mul(predictions, targets)
    losses = 1 - math_ops.reduce_sum(radial_diffs, reduction_indices=[dim,])
    return _compute_weighted_loss(losses, weight)
