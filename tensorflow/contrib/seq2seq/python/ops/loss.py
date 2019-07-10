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
"""Seq2seq loss operations for use in sequence models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras.losses import Loss
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

__all__ = ["sequence_loss", "SequenceLoss"]


def sequence_loss(logits,
                  targets,
                  weights,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  sum_over_timesteps=False,
                  sum_over_batch=False,
                  softmax_loss_function=None,
                  name=None):
  """Weighted cross-entropy loss for a sequence of logits.

  Depending on the values of `average_across_timesteps` / `sum_over_timesteps`
  and `average_across_batch` / `sum_over_batch`, the return Tensor will have
  rank 0, 1, or 2 as these arguments reduce the cross-entropy at each target,
  which has shape `[batch_size, sequence_length]`, over their respective
  dimensions. For example, if `average_across_timesteps` is `True` and
  `average_across_batch` is `False`, then the return Tensor will have shape
  `[batch_size]`.

  Note that `average_across_timesteps` and `sum_over_timesteps` cannot be True
  at same time. Same for `average_across_batch` and `sum_over_batch`.

  The recommended loss reduction in tf 2.0 has been changed to sum_over, instead
  of weighted average. User are recommend to use `sum_over_timesteps` and
  `sum_over_batch` for reduction.

  Args:
    logits: A Tensor of shape
      `[batch_size, sequence_length, num_decoder_symbols]` and dtype float.
      The logits correspond to the prediction across all classes at each
      timestep.
    targets: A Tensor of shape `[batch_size, sequence_length]` and dtype
      int. The target represents the true class at each timestep.
    weights: A Tensor of shape `[batch_size, sequence_length]` and dtype
      float. `weights` constitutes the weighting of each prediction in the
      sequence. When using `weights` as masking, set all valid timesteps to 1
      and all padded timesteps to 0, e.g. a mask returned by `tf.sequence_mask`.
    average_across_timesteps: If set, sum the cost across the sequence
      dimension and divide the cost by the total label weight across timesteps.
    average_across_batch: If set, sum the cost across the batch dimension and
      divide the returned cost by the batch size.
    sum_over_timesteps: If set, sum the cost across the sequence dimension and
      divide the size of the sequence. Note that any element with 0 weights will
      be excluded from size calculation.
    sum_over_batch: if set, sum the cost across the batch dimension and divide
      the total cost by the batch size. Not that any element with 0 weights will
      be excluded from size calculation.
    softmax_loss_function: Function (labels, logits) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
      **Note that to avoid confusion, it is required for the function to accept
      named arguments.**
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A float Tensor of rank 0, 1, or 2 depending on the
    `average_across_timesteps` and `average_across_batch` arguments. By default,
    it has rank 0 (scalar) and is the weighted average cross-entropy
    (log-perplexity) per symbol.

  Raises:
    ValueError: logits does not have 3 dimensions or targets does not have 2
                dimensions or weights does not have 2 dimensions.
  """
  if len(logits.get_shape()) != 3:
    raise ValueError("Logits must be a "
                     "[batch_size x sequence_length x logits] tensor")
  if len(targets.get_shape()) != 2:
    raise ValueError("Targets must be a [batch_size x sequence_length] tensor")
  if len(weights.get_shape()) != 2:
    raise ValueError("Weights must be a [batch_size x sequence_length] tensor")
  if average_across_timesteps and sum_over_timesteps:
    raise ValueError("average_across_timesteps and sum_over_timesteps cannot "
                     "be set to True at same time.")
  if average_across_batch and sum_over_batch:
    raise ValueError("average_across_batch and sum_over_batch cannot be set "
                     "to True at same time.")
  with ops.name_scope(name, "sequence_loss", [logits, targets, weights]):
    num_classes = array_ops.shape(logits)[2]
    logits_flat = array_ops.reshape(logits, [-1, num_classes])
    targets = array_ops.reshape(targets, [-1])
    if softmax_loss_function is None:
      crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
          labels=targets, logits=logits_flat)
    else:
      crossent = softmax_loss_function(labels=targets, logits=logits_flat)
    crossent *= array_ops.reshape(weights, [-1])
    if average_across_timesteps and average_across_batch:
      crossent = math_ops.reduce_sum(crossent)
      total_size = math_ops.reduce_sum(weights)
      crossent = math_ops.div_no_nan(crossent, total_size)
    elif sum_over_timesteps and sum_over_batch:
      crossent = math_ops.reduce_sum(crossent)
      total_count = math_ops.cast(math_ops.count_nonzero(weights),
                                  crossent.dtype)
      crossent = math_ops.div_no_nan(crossent, total_count)
    else:
      crossent = array_ops.reshape(crossent, array_ops.shape(logits)[0:2])
      if average_across_timesteps or average_across_batch:
        reduce_axis = [0] if average_across_batch else [1]
        crossent = math_ops.reduce_sum(crossent, axis=reduce_axis)
        total_size = math_ops.reduce_sum(weights, axis=reduce_axis)
        crossent = math_ops.div_no_nan(crossent, total_size)
      elif sum_over_timesteps or sum_over_batch:
        reduce_axis = [0] if sum_over_batch else [1]
        crossent = math_ops.reduce_sum(crossent, axis=reduce_axis)
        total_count = math_ops.cast(
            math_ops.count_nonzero(weights, axis=reduce_axis),
            dtype=crossent.dtype)
        crossent = math_ops.div_no_nan(crossent, total_count)
    return crossent


class SequenceLoss(Loss):
  """Weighted cross-entropy loss for a sequence of logits."""

  def __init__(self,
               average_across_timesteps=False,
               average_across_batch=False,
               sum_over_timesteps=True,
               sum_over_batch=True,
               softmax_loss_function=None,
               name=None):
    super(SequenceLoss, self).__init__(name=name)
    self.average_across_timesteps = average_across_timesteps
    self.average_across_batch = average_across_batch
    self.sum_over_timesteps = sum_over_timesteps
    self.sum_over_batch = sum_over_batch
    self.softmax_loss_function = softmax_loss_function

  def __call__(self, y_true, y_pred, sample_weight=None):
    """Override the parent __call__ to have a customized reduce behavior."""
    return sequence_loss(y_pred, y_true, sample_weight,
                         average_across_timesteps=self.average_across_timesteps,
                         average_across_batch=self.average_across_batch,
                         sum_over_timesteps=self.sum_over_timesteps,
                         sum_over_batch=self.sum_over_batch,
                         softmax_loss_function=self.softmax_loss_function,
                         name=self.name)

  def call(self, y_true, y_pred):
    # Skip this method since the __call__ contains real implementation.
    pass
