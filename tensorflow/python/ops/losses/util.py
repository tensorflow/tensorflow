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
"""Utilities for manipulating the loss collections."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export


def squeeze_or_expand_dimensions(y_pred, y_true=None, sample_weight=None):
  """Squeeze or expand last dimension if needed.

  1. Squeezes last dim of `y_pred` or `y_true` if their rank differs by 1
  (using `confusion_matrix.remove_squeezable_dimensions`).
  2. Squeezes or expands last dim of `sample_weight` if its rank differs by 1
  from the new rank of `y_pred`.
  If `sample_weight` is scalar, it is kept scalar.

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    y_pred: Predicted values, a `Tensor` of arbitrary dimensions.
    y_true: Optional label `Tensor` whose dimensions match `y_pred`.
    sample_weight: Optional weight scalar or `Tensor` whose dimensions match
      `y_pred`.

  Returns:
    Tuple of `y_pred`, `y_true` and `sample_weight`. Each of them possibly has
    the last dimension squeezed,
    `sample_weight` could be extended by one dimension.
    If `sample_weight` is None, (y_pred, y_true) is returned.
  """
  y_pred_shape = y_pred.shape
  y_pred_rank = y_pred_shape.ndims
  if y_true is not None:

    # If sparse matrix is provided as `y_true`, the last dimension in `y_pred`
    # may be > 1. Eg: y_true = [0, 1, 2] (shape=(3,)),
    # y_pred = [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]] (shape=(3, 3))
    # In this case, we should not try to remove squeezable dimension.
    y_true_shape = y_true.shape
    y_true_rank = y_true_shape.ndims
    if (y_true_rank is not None) and (y_pred_rank is not None):
      # Use static rank for `y_true` and `y_pred`.
      if (y_pred_rank - y_true_rank != 1) or y_pred_shape[-1] == 1:
        y_true, y_pred = confusion_matrix.remove_squeezable_dimensions(
            y_true, y_pred)
    else:
      # Use dynamic rank.
      rank_diff = array_ops.rank(y_pred) - array_ops.rank(y_true)
      squeeze_dims = lambda: confusion_matrix.remove_squeezable_dimensions(  # pylint: disable=g-long-lambda
          y_true, y_pred)
      is_last_dim_1 = math_ops.equal(1, array_ops.shape(y_pred)[-1])
      maybe_squeeze_dims = lambda: control_flow_ops.cond(  # pylint: disable=g-long-lambda
          is_last_dim_1, squeeze_dims, lambda: (y_true, y_pred))
      y_true, y_pred = control_flow_ops.cond(
          math_ops.equal(1, rank_diff), maybe_squeeze_dims, squeeze_dims)

  if sample_weight is None:
    return y_pred, y_true

  weights_shape = sample_weight.shape
  weights_rank = weights_shape.ndims
  if weights_rank == 0:  # If weights is scalar, do nothing.
    return y_pred, y_true, sample_weight

  if (y_pred_rank is not None) and (weights_rank is not None):
    # Use static rank.
    if weights_rank - y_pred_rank == 1:
      sample_weight = array_ops.squeeze(sample_weight, [-1])
    elif y_pred_rank - weights_rank == 1:
      sample_weight = array_ops.expand_dims(sample_weight, [-1])
    return y_pred, y_true, sample_weight

  # Use dynamic rank.
  weights_rank_tensor = array_ops.rank(sample_weight)
  rank_diff = weights_rank_tensor - array_ops.rank(y_pred)
  maybe_squeeze_weights = lambda: array_ops.squeeze(sample_weight, [-1])

  def _maybe_expand_weights():
    expand_weights = lambda: array_ops.expand_dims(sample_weight, [-1])
    return control_flow_ops.cond(
        math_ops.equal(rank_diff, -1), expand_weights, lambda: sample_weight)

  def _maybe_adjust_weights():
    return control_flow_ops.cond(
        math_ops.equal(rank_diff, 1), maybe_squeeze_weights,
        _maybe_expand_weights)

  # squeeze or expand last dim of `sample_weight` if its rank differs by 1
  # from the new rank of `y_pred`.
  sample_weight = control_flow_ops.cond(
      math_ops.equal(weights_rank_tensor, 0), lambda: sample_weight,
      _maybe_adjust_weights)
  return y_pred, y_true, sample_weight


def scale_losses_by_sample_weight(losses, sample_weight):
  """Scales loss values by the given sample weights.

  `sample_weight` dimensions are updated to match with the dimension of `losses`
  if possible by using squeeze/expand/broadcast.

  Args:
    losses: Loss tensor.
    sample_weight: Sample weights tensor.

  Returns:
    `losses` scaled by `sample_weight` with dtype float32.
  """
  # TODO(psv): Handle the casting here in a better way, eg. if losses is float64
  # we do not want to lose precision.
  losses = math_ops.cast(losses, dtypes.float32)
  sample_weight = math_ops.cast(sample_weight, dtypes.float32)

  # Update dimensions of `sample_weight` to match with `losses` if possible.
  losses, _, sample_weight = squeeze_or_expand_dimensions(
      losses, None, sample_weight)
  return math_ops.multiply(losses, sample_weight)


@tf_contextlib.contextmanager
def check_per_example_loss_rank(per_example_loss):
  """Context manager that checks that the rank of per_example_loss is at least 1.

  Args:
    per_example_loss: Per example loss tensor.

  Yields:
    A context manager.
  """
  loss_rank = per_example_loss.shape.rank
  if loss_rank is not None:
    # Handle static rank.
    if loss_rank == 0:
      raise ValueError(
          "Invalid value passed for `per_example_loss`. Expected a tensor with "
          f"at least rank 1. Received per_example_loss={per_example_loss} with "
          f"rank {loss_rank}")
    yield
  else:
    # Handle dynamic rank.
    with ops.control_dependencies([
        check_ops.assert_greater_equal(
            array_ops.rank(per_example_loss),
            math_ops.cast(1, dtype=dtypes.int32),
            message="Invalid value passed for `per_example_loss`. Expected a "
            "tensor with at least rank 1.")
    ]):
      yield


@tf_export(v1=["losses.add_loss"])
def add_loss(loss, loss_collection=ops.GraphKeys.LOSSES):
  """Adds a externally defined loss to the collection of losses.

  Args:
    loss: A loss `Tensor`.
    loss_collection: Optional collection to add the loss to.
  """
  # Since we have no way of figuring out when a training iteration starts or
  # ends, holding on to a loss when executing eagerly is indistinguishable from
  # leaking memory. We instead leave the collection empty.
  if loss_collection and not context.executing_eagerly():
    ops.add_to_collection(loss_collection, loss)


@tf_export(v1=["losses.get_losses"])
def get_losses(scope=None, loss_collection=ops.GraphKeys.LOSSES):
  """Gets the list of losses from the loss_collection.

  Args:
    scope: An optional scope name for filtering the losses to return.
    loss_collection: Optional losses collection.

  Returns:
    a list of loss tensors.
  """
  return ops.get_collection(loss_collection, scope)


@tf_export(v1=["losses.get_regularization_losses"])
def get_regularization_losses(scope=None):
  """Gets the list of regularization losses.

  Args:
    scope: An optional scope name for filtering the losses to return.

  Returns:
    A list of regularization losses as Tensors.
  """
  return ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES, scope)


@tf_export(v1=["losses.get_regularization_loss"])
def get_regularization_loss(scope=None, name="total_regularization_loss"):
  """Gets the total regularization loss.

  Args:
    scope: An optional scope name for filtering the losses to return.
    name: The name of the returned tensor.

  Returns:
    A scalar regularization loss.
  """
  losses = get_regularization_losses(scope)
  if losses:
    return math_ops.add_n(losses, name=name)
  else:
    return constant_op.constant(0.0)


@tf_export(v1=["losses.get_total_loss"])
def get_total_loss(add_regularization_losses=True,
                   name="total_loss",
                   scope=None):
  """Returns a tensor whose value represents the total loss.

  In particular, this adds any losses you have added with `tf.add_loss()` to
  any regularization losses that have been added by regularization parameters
  on layers constructors e.g. `tf.layers`. Be very sure to use this if you
  are constructing a loss_op manually. Otherwise regularization arguments
  on `tf.layers` methods will not function.

  Args:
    add_regularization_losses: A boolean indicating whether or not to use the
      regularization losses in the sum.
    name: The name of the returned tensor.
    scope: An optional scope name for filtering the losses to return. Note that
      this filters the losses added with `tf.add_loss()` as well as the
      regularization losses to that scope.

  Returns:
    A `Tensor` whose value represents the total loss.

  Raises:
    ValueError: if `losses` is not iterable.
  """
  losses = get_losses(scope=scope)
  if add_regularization_losses:
    losses += get_regularization_losses(scope=scope)
  return math_ops.add_n(losses, name=name)
