# Copyright 2015 Google Inc. All Rights Reserved.
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

The loss ops measure error for use in neural networks. These losses
can be used for measuring accuracy of a network in a regression task
or for regularization purposes (e.g., weight decay).

These loss ops are, by design, minimal, enabling flexibility in how
their output can be used.

@@absolute
@@squared
@@logistic
@@softmax
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

__all__ = ["absolute", "squared", "logistic", "softmax"]


def _reduce_batch(x, reduce_fn, name=None):
  """Given a tensor `x`, calls reduce_fn to reduce it across dimensions.

  Given a tensor with number of dimensions > 1, _reduce_batch will reduce the
  tensor across all dimensions except for dimension 0. As an example, given a
  tensor of shape [batch_size, d1, d2], this function will reduce across
  dimensions d1 and d2, returning a tensor of shape [batch_size].

  Tensors of dimension 1 are returned as-is, while tensors of dimension 0
  raise a ValueError.

  Args:
    x: A `Tensor` with dimension > 0.
    reduce_fn: A math_ops reduce function that takes arguments of
      `x`, `reduction_indices`, and `name`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with values reduced by reduce_fn across all dimensions > 0.

  Raises:
    ValueError: If `x` has dimension 0.
  """
  x = ops.convert_to_tensor(x, name="x")
  with ops.op_scope([x], name, "reduce_batch"):
    ndims = x.get_shape().ndims
    if ndims == 0:
      raise ValueError("Cannot reduce a scalar into batches.")
    elif ndims == 1:
      return x  # Don't include a useless reduction.
    elif ndims:
      reduction_indices = math_ops.range(1, ndims)
      shape = [x.get_shape().dims[0]]
    else:
      reduction_indices = math_ops.range(1, array_ops.size(array_ops.shape(x)))
      shape = [None]  # We don't know much about the shape, but it is rank 1.
    result = reduce_fn(x, reduction_indices=reduction_indices)

    # Give a shape hint in case we have extra information.
    result.set_shape(shape)
    return result


def _reduce_batch_sum(x, name=None):
  """Given a tensor `x`, sums across all dimensions except dimension 0.

  Given a tensor with the number of dimensions > 1, this will sum across all
  dimensions except for dimension 0. This function is useful for summing the
  loss (error) across all examples in a batch when training. As an example,
  given a tensor of shape [batch_size, d1, d2], this function will sum across
  dimensions d1 and d2, returning a tensor of shape [batch_size].

  Tensors of dimension 1 are returned as-is, while tensors of dimension 0
  raise a ValueError.

  Args:
    x: A `Tensor` with dimension > 0.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with values summed across all dimensions > 0.

  Raises:
    ValueError: If `x` has dimension 0.

  """
  return _reduce_batch(x, math_ops.reduce_sum, name)


def _reduce_to_scalar(x, name=None):
  """Reduces losses to a scalar.

  Given a tensor `x`, sums across all dimensions except dimension 0, then
  average across dimension 0.

  Args:
    x: A `Tensor` with dimension > 0.
    name: A name for the operation (optional).

  Returns:
    Caculate sum of losses per example, then average across batch.
  """
  with ops.op_scope([x], name, "scalar") as scope:
    return math_ops.reduce_mean(_reduce_batch_sum(x), name=scope)


def _validate_predicted_and_target(predicted, target):
  # TODO(ptucker): Optionally add assert op for shape check, for cases when
  # shape is not fully defined at graph construction time?
  predicted.get_shape().assert_is_compatible_with(target.get_shape())
  tensor_util.assert_same_float_dtype([predicted, target])


def _raw_absolute(predicted, target, name=None):
  """Computes and returns the per-example absolute loss.

  Computes the per-example absolute value of the difference between
  the target and predicted tensors. The tensors must have the same
  shape.

  Args:
    predicted: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]`
      of predicted values.
    target: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]` of
      target values. The shape of the target tensor should match the
      `predicted` tensor.
    name: A name for the operation (optional).

  Returns:
    A `[batch_size, dim_1, ..., dim_n]` tensor of per-example absolute losses.

  Raises:
    ValueError: If `predicted` and `target` shapes do not match.

  """
  with ops.op_scope([predicted, target], name, "absolute_loss") as scope:
    predicted = ops.convert_to_tensor(predicted, name="predicted")
    target = ops.convert_to_tensor(target, name="target")
    _validate_predicted_and_target(predicted, target)
    return math_ops.abs(target - predicted, name=scope)


def _raw_squared(predicted, target, name=None):
  """Computes and returns the per-example squared loss, divided by 2.

  Computes the per-example squared difference between the target and
  predicted tensors. The tensors must have the same shape.

  Args:
    predicted: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]`
      of predicted values.
    target: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]` of
      target values. The shape of the target tensor should match the
      `predicted` tensor.
    name: A name for the operation (optional).

  Returns:
    A `[batch_size, dim_1, ..., dim_n]` tensor of per-example squared losses.

  Raises:
    ValueError: If `predicted` and `target` shapes do not match.

  """
  with ops.op_scope([predicted, target], name, "squared_loss") as scope:
    predicted = ops.convert_to_tensor(predicted, name="predicted")
    target = ops.convert_to_tensor(target, name="target")
    _validate_predicted_and_target(predicted, target)
    return math_ops.div(math_ops.square(target - predicted), 2.0, name=scope)


def absolute(predicted, target, name=None):
  """Reduces absolute losses to a scalar.

  Args:
    predicted: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]`
      of predicted values.
    target: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]` of
      target values. The shape of the target tensor should match the
      `predicted` tensor.
    name: A name for the operation (optional).

  Returns:
    Caculate sum of absolute losses per example, then average across batch.
  """
  with ops.op_scope([predicted, target], name, "absolute_loss") as scope:
    return _reduce_to_scalar(_raw_absolute(predicted, target), name=scope)


def squared(predicted, target, name=None):
  """Reduces squared losses to a scalar.

  Args:
    predicted: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]`
      of predicted values.
    target: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]` of
      target values. The shape of the target tensor should match the
      `predicted` tensor.
    name: A name for the operation (optional).

  Returns:
    Caculate sum of squared losses per example, then average across batch.
  """
  with ops.op_scope([predicted, target], name, "squared_loss") as scope:
    return _reduce_to_scalar(_raw_squared(predicted, target), name=scope)


def logistic(logit, target, name=None):
  """Calculates the logistic cross-entropy loss, averaged across batches.

  **WARNING:** `logit` must be unscaled.
  See `tf.nn.sigmoid_cross_entropy_with_logits` for more details.

  Args:
    logit: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]`
      of predicted logit values.
    target: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]` of
      target values. The shape of the target tensor should match the
      `logit` tensor.
    name: A name for the operation (optional).

  Returns:
    A scalar `tensor` of the logistic cross-entropy loss, averaged across
    batches.

  Raises:
    ValueError: If `logit` and `target` shapes do not match.
  """
  with ops.op_scope([logit, target], name, "logistic_loss") as scope:
    return _reduce_to_scalar(
        nn.sigmoid_cross_entropy_with_logits(logit, target), name=scope)


def softmax(logit, target, name=None):
  """Calculates the softmax cross-entropy loss, averaged across batches.

  **WARNING:** `logit` must be unscaled, while the `target` should be a
  normalized probability prediction.
  See `tf.nn.softmax_cross_entropy_with_logits` for more details.

  Args:
    logit: Tensor of actual values. Shape must have rank 2, generally
        (batch, num_classes). num_classes must be > 1. For single-class
        regression, use `logistic`. Type must be `tf.float32` or `tf.float64`.
    target: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]` of
      target values. The shape of the target tensor should match the
      `logit` tensor.
    name: A name for the operation (optional).

  Returns:
    A scalar `tensor` of the softmax cross-entropy loss, averaged across
    batches.

  Raises:
    ValueError: If `logit` and `target` shapes do not match.
  """
  with ops.op_scope([logit, target], name, "softmax_loss") as scope:
    shape = logit.get_shape().with_rank(2)
    if shape.dims[1] and shape.dims[1] < 2:
      raise ValueError(
          "Invalid shape %s; use logistic() instead for only 1 class." %
          shape)
    return _reduce_to_scalar(
        nn.softmax_cross_entropy_with_logits(logit, target), name=scope)
