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

Note: All the losses are added to the `GraphKeys.LOSSES` collection by default.

@@absolute_difference
@@compute_weighted_loss
@@cosine_distance
@@hinge_loss
@@log_loss
@@mean_pairwise_squared_error
@@mean_squared_error
@@sigmoid_cross_entropy
@@softmax_cross_entropy
@@sparse_softmax_cross_entropy

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.losses import util
from tensorflow.python.platform import tf_logging as logging


def _scale_losses(losses, weights):
  """Computes the scaled loss.

  Args:
    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
    weights: `Tensor` of shape `[]`, `[batch_size]` or
      `[batch_size, d1, ... dN]`. The `losses` are reduced (`tf.reduce_sum`)
      until its dimension matches that of `weights` at which point the reduced
      `losses` are element-wise multiplied by `weights` and a final `reduce_sum`
      is computed on the result. Conceptually, this operation is similar to
      broadcasting (tiling) `weights` to be the same shape as `losses`,
      performing an element-wise multiplication, and summing the result. Note,
      however, that the dimension matching is right-to-left, not left-to-right;
      i.e., the opposite of standard NumPy/Tensorflow broadcasting.

  Returns:
    A scalar tf.float32 `Tensor` whose value represents the sum of the scaled
      `losses`.
  """
  # First, compute the sum of the losses over all elements:
  start_index = max(0, weights.get_shape().ndims)
  reduction_indices = list(range(start_index, losses.get_shape().ndims))
  reduced_losses = math_ops.reduce_sum(losses,
                                       reduction_indices=reduction_indices)
  reduced_losses = math_ops.mul(reduced_losses, weights)
  return math_ops.reduce_sum(reduced_losses)


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
  # If weights is a scalar, its easy to compute:
  if weights.get_shape().ndims == 0:
    batch_size = array_ops.reshape(array_ops.slice(array_ops.shape(losses),
                                                   [0], [1]), [])
    num_per_batch = math_ops.div(math_ops.to_float(array_ops.size(losses)),
                                 math_ops.to_float(batch_size))
    num_per_batch = array_ops.where(math_ops.equal(weights, 0),
                                    0.0, num_per_batch)
    num_per_batch = math_ops.mul(array_ops.ones(
        array_ops.reshape(batch_size, [1])), num_per_batch)
    return num_per_batch if per_batch else math_ops.reduce_sum(num_per_batch)

  # First, count the number of nonzero weights.
  if weights.get_shape().ndims >= 1:
    reduction_indices = list(range(1, weights.get_shape().ndims))
    num_nonzero_per_batch = math_ops.reduce_sum(
        math_ops.to_float(math_ops.not_equal(weights, 0)),
        reduction_indices=reduction_indices)

  # Next, determine the number of elements that weight would broadcast to:
  broadcast_dims = array_ops.slice(array_ops.shape(losses),
                                   [weights.get_shape().ndims], [-1])
  num_to_broadcast = math_ops.to_float(math_ops.reduce_prod(broadcast_dims))

  num_per_batch = math_ops.mul(num_nonzero_per_batch, num_to_broadcast)
  return num_per_batch if per_batch else math_ops.reduce_sum(num_per_batch)


def compute_weighted_loss(
    losses, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES):
  """Computes the weighted loss.

  WARNING: `weights` also supports dimensions of 1, but the broadcasting does
  not work as advertised, you'll wind up with weighted sum instead of weighted
  mean for any but the last dimension. This will be cleaned up soon, so please
  do not rely on the current behavior for anything but the shapes documented for
  `weights` below.

  Args:
    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
    weights: `Tensor` of shape `[]`, `[batch_size]` or
      `[batch_size, d1, ... dK]`, where K < N.
    scope: the scope for the operations performed in computing the loss.
    loss_collection: the loss will be added to these collections.

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

    losses_shape = losses.get_shape()
    if losses_shape.ndims is None:
      raise ValueError("losses.get_shape().ndims cannot be None")
    weights_shape = weights.get_shape()
    if weights_shape.ndims is None:
      raise ValueError("weight.get_shape().ndims cannot be None")

    # TODO(b/33556118): Remove `ndims > 1` check so shapes [] and [1] behave the
    # same.
    if weights_shape.ndims > 1 and weights_shape.dims[-1].is_compatible_with(1):
      weights = array_ops.squeeze(weights, [-1])

    # TODO(b/33556118): Remove this when we require weights shape be either
    # scalar or the same as losses.
    weights_dims = weights_shape.as_list()
    losses_dims = losses_shape.as_list()
    if len(weights_dims) > len(losses_dims):
      raise ValueError(
          "Invalid weights shape %s can not be broadcast to losses %s." % (
              weights_shape, losses_shape))
    for i in range(len(weights_dims)):
      if ((losses_dims[i] is not None) and (losses_dims[i] != 1) and
          (weights_dims[i] is not None) and (weights_dims[i] == 1)):
        logging.warn(
            "WARNING: Weights %s with dimension 1 will result in a sum"
            ", not average, across dimension %d.", weights_shape, i)

    total_loss = _scale_losses(losses, weights)
    num_present = _num_present(losses, weights)
    mean_loss = _safe_mean(total_loss, num_present)
    # Convert the result back to the input type.
    mean_loss = math_ops.cast(mean_loss, input_dtype)
    util.add_loss(mean_loss, loss_collection)
    return mean_loss


def absolute_difference(
    labels, predictions, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES):
  """Adds an Absolute Difference loss to the training procedure.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a `Tensor` of
  shape `[batch_size]`, then the total loss for each sample of the batch is
  rescaled by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  WARNING: `weights` also supports dimensions of 1, but the broadcasting does
  not work as advertised, you'll wind up with weighted sum instead of weighted
  mean for any but the last dimension. This will be cleaned up soon, so please
  do not rely on the current behavior for anything but the shapes documented for
  `weights` below.

  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    weights: Coefficients for the loss a scalar, a tensor of shape
      `[batch_size]` or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.

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
    losses = math_ops.abs(math_ops.sub(predictions, labels))
    return compute_weighted_loss(losses, weights, scope, loss_collection)


def cosine_distance(
    labels, predictions, dim=None, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES):
  """Adds a cosine-distance loss to the training procedure.

  Note that the function assumes that `predictions` and `labels` are already
  unit-normalized.

  WARNING: `weights` also supports dimensions of 1, but the broadcasting does
  not work as advertised, you'll wind up with weighted sum instead of weighted
  mean for any but the last dimension. This will be cleaned up soon, so please
  do not rely on the current behavior for anything but the shapes documented for
  `weights` below.

  Args:
    labels: `Tensor` whose shape matches 'predictions'
    predictions: An arbitrary matrix.
    dim: The dimension along which the cosine distance is computed.
    weights: Coefficients for the loss a scalar, a tensor of shape
      `[batch_size]` or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If `predictions` shape doesn't match `labels` shape, or
      `weights` is `None`.
  """
  if dim is None:
    raise ValueError("`dim` cannot be None.")
  with ops.name_scope(scope, "cosine_distance_loss",
                      [predictions, labels, weights]) as scope:
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    predictions = math_ops.to_float(predictions)
    labels = math_ops.to_float(labels)

    radial_diffs = math_ops.mul(predictions, labels)
    losses = 1 - math_ops.reduce_sum(radial_diffs, reduction_indices=[dim,])
    return compute_weighted_loss(losses, weights, scope, loss_collection)


def hinge_loss(labels, logits, weights=1.0, scope=None,
               loss_collection=ops.GraphKeys.LOSSES):
  """Adds a hinge loss to the training procedure.

  WARNING: `weights` also supports dimensions of 1, but the broadcasting does
  not work as advertised, you'll wind up with weighted sum instead of weighted
  mean for any but the last dimension. This will be cleaned up soon, so please
  do not rely on the current behavior for anything but the shapes documented for
  `weights` below.

  Args:
    labels: The ground truth output tensor. Its shape should match the shape of
      logits. The values of the tensor are expected to be 0.0 or 1.0.
    logits: The logits, a float tensor.
    weights: Coefficients for the loss a scalar, a tensor of shape
      `[batch_size]` or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.

  Returns:
    A scalar `Tensor` of the loss value.

  Raises:
    ValueError: If the shapes of `logits` and `labels` don't match.
  """
  with ops.name_scope(scope, "hinge_loss", [logits, labels]) as scope:
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    # We first need to convert binary labels to -1/1 labels (as floats).
    labels = math_ops.to_float(labels)
    all_ones = array_ops.ones_like(labels)
    labels = math_ops.sub(2 * labels, all_ones)
    losses = nn_ops.relu(math_ops.sub(all_ones, math_ops.mul(labels, logits)))
    return compute_weighted_loss(losses, weights, scope, loss_collection)


def log_loss(labels, predictions, weights=1.0, epsilon=1e-7, scope=None,
             loss_collection=ops.GraphKeys.LOSSES):
  """Adds a Log Loss term to the training procedure.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  WARNING: `weights` also supports dimensions of 1, but the broadcasting does
  not work as advertised, you'll wind up with weighted sum instead of weighted
  mean for any but the last dimension. This will be cleaned up soon, so please
  do not rely on the current behavior for anything but the shapes documented for
  `weights` below.

  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    weights: Coefficients for the loss a scalar, a tensor of shape
      `[batch_size]` or a tensor whose shape matches `predictions`.
    epsilon: A small increment to add to avoid taking a log of zero.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.

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
    losses = -math_ops.mul(
        labels,
        math_ops.log(predictions + epsilon)) - math_ops.mul(
            (1 - labels), math_ops.log(1 - predictions + epsilon))
    return compute_weighted_loss(losses, weights, scope, loss_collection)


def mean_pairwise_squared_error(labels, predictions, weights=1.0, scope=None,
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

    diffs = math_ops.sub(predictions, labels)

    # Need to verify here since the function doesn't use compute_weighted_loss
    if diffs.get_shape().ndims is None:
      raise ValueError("diffs.get_shape().ndims cannot be None")
    if weights.get_shape().ndims is None:
      raise ValueError("weights.get_shape().ndims cannot be None")

    reduction_indices = list(range(1, diffs.get_shape().ndims))

    sum_squares_diff_per_batch = math_ops.reduce_sum(
        math_ops.square(diffs),
        reduction_indices=reduction_indices)
    num_present_per_batch = _num_present(diffs, weights, per_batch=True)

    term1 = 2.0 * _safe_div(sum_squares_diff_per_batch,
                            num_present_per_batch)

    sum_diff = math_ops.reduce_sum(diffs, reduction_indices=reduction_indices)
    term2 = 2.0 * _safe_div(math_ops.square(sum_diff),
                            math_ops.square(num_present_per_batch))

    loss = _scale_losses(term1 - term2, weights)

    mean_loss = array_ops.where(math_ops.reduce_sum(num_present_per_batch) > 0,
                                loss,
                                array_ops.zeros_like(loss),
                                name="value")
    util.add_loss(mean_loss, loss_collection)
    return mean_loss


def mean_squared_error(labels, predictions, weights=1.0, scope=None,
                       loss_collection=ops.GraphKeys.LOSSES):
  """Adds a Sum-of-Squares loss to the training procedure.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  WARNING: `weights` also supports dimensions of 1, but the broadcasting does
  not work as advertised, you'll wind up with weighted sum instead of weighted
  mean for any but the last dimension. This will be cleaned up soon, so please
  do not rely on the current behavior for anything but the shapes documented for
  `weights` below.

  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    weights: Coefficients for the loss a scalar, a tensor of shape
      `[batch_size]` or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.

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
    losses = math_ops.square(math_ops.sub(predictions, labels))
    return compute_weighted_loss(losses, weights, scope, loss_collection)


def sigmoid_cross_entropy(
    multi_class_labels, logits, weights=1.0, label_smoothing=0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES):
  """Creates a cross-entropy loss using tf.nn.sigmoid_cross_entropy_with_logits.

  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of shape `[batch_size]`, then the loss weights apply to each
  corresponding sample.

  WARNING: `weights` also supports dimensions of 1, but the broadcasting does
  not work as advertised, you'll wind up with weighted sum instead of weighted
  mean for any but the last dimension. This will be cleaned up soon, so please
  do not rely on the current behavior for anything but the shapes documented for
  `weights` below.

  If `label_smoothing` is nonzero, smooth the labels towards 1/2:

      new_multiclass_labels = multiclass_labels * (1 - label_smoothing)
                              + 0.5 * label_smoothing

  Args:
    multi_class_labels: `[batch_size, num_classes]` target integer labels in
      `(0, 1)`.
    logits: `[batch_size, num_classes]` logits outputs of the network.
    weights: Coefficients for the loss. This must be of shape `[]`,
      `[batch_size]` or `[batch_size, num_classes]`.
    label_smoothing: If greater than `0` then smooth the labels.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.

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
      multi_class_labels = (multi_class_labels * (1 - label_smoothing) +
                            0.5 * label_smoothing)

    losses = nn.sigmoid_cross_entropy_with_logits(logits, multi_class_labels,
                                                  name="xentropy")
    return compute_weighted_loss(losses, weights, scope, loss_collection)


def softmax_cross_entropy(
    onehot_labels, logits, weights=1.0, label_smoothing=0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES):
  """Creates a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits.

  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of shape `[batch_size]`, then the loss weights apply to each
  corresponding sample.

  WARNING: `weights` also supports dimensions of 1, but the broadcasting does
  not work as advertised, you'll wind up with weighted sum instead of weighted
  mean for any but the last dimension. This will be cleaned up soon, so please
  do not rely on the current behavior for anything but the shapes documented for
  `weights` below.

  If `label_smoothing` is nonzero, smooth the labels towards 1/num_classes:
      new_onehot_labels = onehot_labels * (1 - label_smoothing)
                          + label_smoothing / num_classes

  Args:
    onehot_labels: `[batch_size, num_classes]` target one-hot-encoded labels.
    logits: [batch_size, num_classes] logits outputs of the network .
    weights: Coefficients for the loss. This must be of shape `[]`,
      `[batch_size]` or `[batch_size, num_classes]`.
    label_smoothing: If greater than 0 then smooth the labels.
    scope: the scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.

  Returns:
    A scalar `Tensor` representing the loss value.

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

    losses = nn.softmax_cross_entropy_with_logits(logits, onehot_labels,
                                                  name="xentropy")
    return compute_weighted_loss(losses, weights, scope, loss_collection)


def sparse_softmax_cross_entropy(labels, logits, weights=1.0, scope=None,
                                 loss_collection=ops.GraphKeys.LOSSES):
  """Cross-entropy loss using `tf.nn.sparse_softmax_cross_entropy_with_logits`.

  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of shape [`batch_size`], then the loss weights apply to each
  corresponding sample.

  WARNING: `weights` also supports dimensions of 1, but the broadcasting does
  not work as advertised, you'll wind up with weighted sum instead of weighted
  mean for any but the last dimension. This will be cleaned up soon, so please
  do not rely on the current behavior for anything but the shapes documented for
  `weights` below.

  Args:
    labels: [batch_size, 1] or [batch_size] target labels of dtype `int32` or
      `int64` in the range `[0, num_classes)`.
    logits: [batch_size, num_classes] logits outputs of the network .
    weights: Coefficients for the loss. This must be of shape `[batch_size]` or
      `[batch_size, 1]`.
    scope: the scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.

  Returns:
    A scalar `Tensor` representing the loss value.

  Raises:
    ValueError: If the shapes of logits, labels, and weight are incompatible, or
      if `weights` is None.
  """
  with ops.name_scope(scope, "sparse_softmax_cross_entropy_loss",
                      [logits, labels, weights]) as scope:
    labels = array_ops.reshape(labels, shape=[array_ops.shape(labels)[0]])
    weights = array_ops.squeeze(weights)

    losses = nn.sparse_softmax_cross_entropy_with_logits(logits, labels,
                                                         name="xentropy")
    return compute_weighted_loss(losses, weights, scope, loss_collection)
