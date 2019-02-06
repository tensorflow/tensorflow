# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=protected-access
"""Utilities related to loss functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import losses_impl


def squeeze_or_expand_dimensions(y_pred, y_true, sample_weight):
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
  """
  y_pred_shape = y_pred.get_shape()
  y_pred_rank = y_pred_shape.ndims
  if y_true is not None:

    # If sparse matrix is provided as `y_true`, the last dimension in `y_pred`
    # may be > 1. Eg: y_true = [0, 1, 2] (shape=(3,)),
    # y_pred = [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]] (shape=(3, 3))
    # In this case, we should not try to remove squeezable dimension.
    y_true_shape = y_true.get_shape()
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
    return y_pred, y_true, None

  sample_weight = ops.convert_to_tensor(sample_weight)
  weights_shape = sample_weight.get_shape()
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
    return control_flow_ops.cond(
        math_ops.equal(rank_diff,
                       -1), lambda: array_ops.expand_dims(sample_weight, [-1]),
        lambda: sample_weight)

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
  return math_ops.div_no_nan(total_loss, num_present, name='value')


def _num_elements(losses):
  """Computes the number of elements in `losses` tensor."""
  with ops.name_scope(None, 'num_elements', values=[losses]) as scope:
    return math_ops.cast(array_ops.size(losses, name=scope), dtype=losses.dtype)


def _reduce_weighted_loss(
    weighted_losses, reduction=losses_impl.ReductionV2.SUM_OVER_BATCH_SIZE):
  """Reduces the individual weighted loss measurements."""
  if reduction == losses_impl.ReductionV2.NONE:
    loss = weighted_losses
  else:
    loss = math_ops.reduce_sum(weighted_losses)
    if reduction == losses_impl.ReductionV2.SUM_OVER_BATCH_SIZE:
      num_replicas = (  # Used to convert from local to global batch size.
          distribution_strategy_context.get_strategy().num_replicas_in_sync)
      loss = _safe_mean(loss, num_replicas * _num_elements(weighted_losses))
  return loss


def compute_weighted_loss(losses,
                          sample_weight=None,
                          reduction=losses_impl.ReductionV2.SUM_OVER_BATCH_SIZE,
                          name=None):
  """Computes the weighted loss.

  Args:
    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
    sample_weight: Optional `Tensor` whose rank is either 0, or the same rank as
      `losses`, or be broadcastable to `losses`.
    reduction: Type of `tf.losses.Reduction` to apply to loss. Default value is
      `SUM_OVER_BATCH_SIZE`.
    name: Optional name for the op.

  Raises:
    ValueError: If the shape of `sample_weight` is not compatible with `losses`.

  Returns:
    Weighted loss `Tensor` of the same type as `losses`. If `reduction` is
    `NONE`, this has the same shape as `losses`; otherwise, it is scalar.
  """
  losses_impl.ReductionV2.validate(reduction)
  if sample_weight is None:
    sample_weight = 1.0
  with ops.name_scope(name, 'weighted_loss', (losses, sample_weight)):
    # Update dimensions of `sample_weight` to match with `losses` if possible.
    losses, _, sample_weight = squeeze_or_expand_dimensions(
        losses, None, sample_weight)
    losses = ops.convert_to_tensor(losses)
    input_dtype = losses.dtype
    losses = math_ops.cast(losses, dtypes.float32)
    sample_weight = math_ops.cast(sample_weight, dtypes.float32)

    try:
      # Broadcast weights if possible.
      sample_weight = weights_broadcast_ops.broadcast_weights(
          sample_weight, losses)
    except ValueError:
      # Reduce values to same ndim as weight array.
      ndim = K.ndim(losses)
      weight_ndim = K.ndim(sample_weight)
      losses = K.mean(losses, axis=list(range(weight_ndim, ndim)))

    sample_weight.get_shape().assert_is_compatible_with(losses.get_shape())
    weighted_losses = math_ops.multiply(losses, sample_weight)
    # Apply reduction function to the individual weighted losses.
    loss = _reduce_weighted_loss(weighted_losses, reduction)
    # Convert the result back to the input type.
    loss = math_ops.cast(loss, input_dtype)
    return loss
