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
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import loss_reduction
from tensorflow.python.ops.losses import util as tf_losses_utils
from tensorflow.python.util.tf_export import keras_export


# TODO(joshl/psv): Update references to ReductionV2 to point to its
# new location.
ReductionV2 = keras_export(  # pylint: disable=invalid-name
    'keras.losses.Reduction', v1=[])(loss_reduction.ReductionV2)


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
  with K.name_scope('num_elements') as scope:
    return math_ops.cast(array_ops.size(losses, name=scope), dtype=losses.dtype)


def reduce_weighted_loss(weighted_losses,
                         reduction=ReductionV2.SUM_OVER_BATCH_SIZE):
  """Reduces the individual weighted loss measurements."""
  if reduction == ReductionV2.NONE:
    loss = weighted_losses
  else:
    loss = math_ops.reduce_sum(weighted_losses)
    if reduction == ReductionV2.SUM_OVER_BATCH_SIZE:
      loss = _safe_mean(loss, _num_elements(weighted_losses))
  return loss


def compute_weighted_loss(losses,
                          sample_weight=None,
                          reduction=ReductionV2.SUM_OVER_BATCH_SIZE,
                          name=None):
  """Computes the weighted loss.

  Args:
    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
    sample_weight: Optional `Tensor` whose rank is either 0, or the same rank as
      `losses`, or be broadcastable to `losses`.
    reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `SUM_OVER_BATCH_SIZE`.
    name: Optional name for the op.

  Raises:
    ValueError: If the shape of `sample_weight` is not compatible with `losses`.

  Returns:
    Weighted loss `Tensor` of the same type as `losses`. If `reduction` is
    `NONE`, this has the same shape as `losses`; otherwise, it is scalar.
  """
  ReductionV2.validate(reduction)

  # If this function is called directly, then we just default 'AUTO' to
  # 'SUM_OVER_BATCH_SIZE'. Eg. Canned estimator use cases.
  if reduction == ReductionV2.AUTO:
    reduction = ReductionV2.SUM_OVER_BATCH_SIZE
  if sample_weight is None:
    sample_weight = 1.0
  with K.name_scope(name or 'weighted_loss'):
    # Save the `reduction` argument for loss normalization when distributing
    # to multiple replicas. Used only for estimator + v1 optimizer flow.
    ops.get_default_graph()._last_loss_reduction = reduction  # pylint: disable=protected-access

    losses = ops.convert_to_tensor_v2(losses)
    input_dtype = losses.dtype
    weighted_losses = tf_losses_utils.scale_losses_by_sample_weight(
        losses, sample_weight)
    # Apply reduction function to the individual weighted losses.
    loss = reduce_weighted_loss(weighted_losses, reduction)
    # Convert the result back to the input type.
    loss = math_ops.cast(loss, input_dtype)
    return loss


def scale_loss_for_distribution(loss_value):
  """Scales and returns the given loss value by the number of replicas."""
  num_replicas = (
      distribution_strategy_context.get_strategy().num_replicas_in_sync)
  if num_replicas > 1:
    loss_value *= (1. / num_replicas)
  return loss_value


def cast_losses_to_common_dtype(losses):
  """Cast a list of losses to a common dtype.

  If any loss is floating-point, they will all be casted to the most-precise
  floating-point loss. Otherwise the losses are not casted. We also skip casting
  losses if there are any complex losses.

  Args:
    losses: A list of losses.

  Returns:
    `losses`, but they have been casted to a common dtype.
  """
  highest_float = None
  for loss in losses:
    if loss.dtype.is_floating:
      if highest_float is None or loss.dtype.size > highest_float.size:
        highest_float = loss.dtype
      elif {loss.dtype, highest_float} == {'bfloat16', 'float16'}:
        highest_float = 'float32'
    if loss.dtype.is_complex:
      return losses  # If we find any complex losses, do not cast any losses
  if highest_float:
    losses = [math_ops.cast(loss, highest_float) for loss in losses]
  return losses
