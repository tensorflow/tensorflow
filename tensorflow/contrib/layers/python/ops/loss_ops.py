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

@@reduce_batch_sum
@@reduce_batch_mean

@@absolute_loss
@@squared_loss

@@sum_squared_loss
@@mean_absolute_loss
@@mean_squared_loss
@@root_mean_squared_loss
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


__all__ = ["reduce_batch_sum", "reduce_batch_mean", "absolute_loss",
           "squared_loss", "sum_squared_loss", "mean_absolute_loss",
           "mean_squared_loss", "root_mean_squared_loss"]


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
      reduction_indices = list(range(1, ndims))
      shape = [x.get_shape().dims[0]]
    else:
      reduction_indices = math_ops.range(1, array_ops.size(array_ops.shape(x)))
      shape = [None]  # We don't know much about the shape, but it is rank 1.
    result = reduce_fn(x, reduction_indices=reduction_indices)

    # Give a shape hint in case we have extra information.
    result.set_shape(shape)
    return result


def reduce_batch_sum(x, name=None):
  """Given a tensor `x`, sums across all dimensions except dimension 0.

  Given a tensor with the number of dimensions > 1, reduce_batch_sum
  will sum across all dimensions except for dimension 0. This function
  is useful for summing the loss (error) across all examples in a
  batch when training. As an example, given a tensor of shape
  [batch_size, d1, d2], this function will sum across dimensions d1
  and d2, returning a tensor of shape [batch_size].

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


def reduce_batch_mean(x, name=None):
  """Given a tensor `x`, returns the mean across all dimensions except dim 0.

  Given a tensor with the number of dimensions > 1, reduce_batch_mean
  will calculate the mean across all dimensions except for dimension
  0. This function is useful for calculating the mean loss (error)
  across all examples in a batch when training. As an example, given a
  tensor of shape [batch_size, d1, d2], this function will calculate
  the mean across dimensions d1 and d2, returning a tensor of shape
  [batch_size].

  Tensors of dimension 1 are returned as-is.

  Args:
    x: A `Tensor` with dimension > 0.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with values averaged across all dimensions > 0.

  Raises:
    ValueError: If `x` has dimension 0.

  """
  return _reduce_batch(x, math_ops.reduce_mean, name)


def absolute_loss(predicted, target, name=None):
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
    predicted.get_shape().assert_is_compatible_with(target.get_shape())
    return math_ops.abs(target - predicted, name=scope)


def squared_loss(predicted, target, name=None):
  """Computes and returns the per-example squared loss.

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
    predicted.get_shape().assert_is_compatible_with(target.get_shape())
    return math_ops.square(target - predicted, name=scope)


def sum_squared_loss(predicted, target, name=None):
  # pylint: disable=line-too-long
  """Calculates 1/2 the sum of the squared loss across batches.

  Computes the squared difference between the target and predicted
  tensors, sums across all dimensions except dimension 0, and divides
  by 2:

      losses = reduce_batch_sum(squared_loss(predicted, target)) / 2.0

  where `losses` is a tensor with dimensions [batch_size].

  The tensors must have the same shape.

  This function is equivalent to typical formulations of L2 loss, and similar
  to TensorFlow's l2_loss function. It differs from the l2_loss function
  by allowing the caller to specify both the predicted and target tensors.

  Args:
    predicted: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]`
      of predicted values.
    target: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]` of
      target values. The shape of the target tensor should match the
      `predicted` tensor.
    name: A name for the operation (optional).

  Returns:
    A `[batch_size]` tensor of squared losses summed across all dimensions
    except dimension 0, divided by 2.

  Raises:
    ValueError: If `predicted` and `target` shapes do not match.

  """
  # pylint: enable=line-too-long
  with ops.op_scope(
      [predicted, target],
      name,
      "sum_squared_loss") as scope:
    return math_ops.div(reduce_batch_sum(squared_loss(predicted, target)),
                        2.0,
                        name=scope)


def mean_absolute_loss(predicted, target, name=None):
  """Calculates the mean absolute loss across batches.

  Computes the absolute difference between the target and predicted
  tensors, averaged across all dimensions except dimension 0:

        losses = reduce_batch_mean(absolute_loss(predicted, target))

  where `losses` is a tensor with dimensions [batch_size].

  The tensors must have the same shape.

  This loss function is a form of L1 loss.

  Args:
    predicted: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]`
      of predicted values.
    target: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]` of
      target values. The shape of the target tensor should match the
      `predicted` tensor.
    name: A name for the operation (optional).

  Returns:
    A `[batch_size]` tensor of absolute differences, averaged across all
    dimensions except dimension 0.

  Raises:
    ValueError: If `predicted` and `target` shapes do not match.

  """
  with ops.op_scope([predicted, target], name, "mean_absolute_loss") as scope:
    return reduce_batch_mean(absolute_loss(predicted, target), name=scope)


def mean_squared_loss(predicted, target, name=None):
  """Calculates the mean squared loss across batches.

  Computes the squared difference between the target and predicted
  tensors, and averages across all dimensions except dimension 0:

        losses = reduce_batch_mean(squared_loss(predicted, target))

  where `losses` is a tensor with dimensions [batch_size].

  The tensors must have the same shape.

  Args:
    predicted: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]`
      of predicted values.
    target: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]` of
      target values. The shape of the target tensor should match the
      `predicted` tensor.
    name: A name for the operation (optional).

  Returns:
    A `[batch_size]` tensor of squared differences, averaged across
    all dimensions except dimension 0.

  Raises:
    ValueError: If `predicted` and `target` shapes do not match.

  """
  with ops.op_scope([predicted, target], name, "mean_squared_loss") as scope:
    return reduce_batch_mean(squared_loss(predicted, target), name=scope)


def root_mean_squared_loss(predicted, target, name=None):
  """Calculates the root mean squared loss across batches.

  Computes the root mean squared loss between the target and predicted
  tensors, which is the square root of the mean squared differences
  between the predicted and target tensors:

        losses = sqrt(mean_squared_loss(predicted, target))

  where `losses` is a tensor with dimensions [batch_size].

  The tensors must have the same shape.

  Args:
    predicted: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]`
      of predicted values.
    target: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]` of
      target values. The shape of the target tensor should match the
      `predicted` tensor.
    name: A name for the operation (optional).

  Returns:
    A `[batch_size]` tensor of the root mean squared differences.

  Raises:
    ValueError: If `predicted` and `target` shapes do not match.

  """
  with ops.op_scope([predicted, target],
                    name,
                    "root_mean_squared_loss") as scope:
    return math_ops.sqrt(mean_squared_loss(predicted, target),
                         name=scope)
