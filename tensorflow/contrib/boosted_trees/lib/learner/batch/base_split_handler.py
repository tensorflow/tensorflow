# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Base class for creating split nodes using one or more features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from tensorflow.contrib.boosted_trees.python.ops import batch_ops_utils
from tensorflow.python.ops import control_flow_ops


class BaseSplitHandler(object):
  """Abstract Base class defining split handlers interface."""

  __metaclass__ = abc.ABCMeta

  def __init__(self,
               l1_regularization,
               l2_regularization,
               tree_complexity_regularization,
               min_node_weight,
               feature_column_group_id,
               gradient_shape,
               hessian_shape,
               multiclass_strategy,
               loss_uses_sum_reduction=False,
               name=None):
    """Constructor for BaseSplitHandler.

    Args:
      l1_regularization: L1 regularization applied for this split handler.
      l2_regularization: L2 regularization applied for this split handler.
      tree_complexity_regularization: Tree complexity regularization applied
          for this split handler.
      min_node_weight: Minimum sum of weights of examples in each partition to
          be considered for splitting.
      feature_column_group_id: Feature column group index.
      gradient_shape: A TensorShape, containing shape of gradients.
      hessian_shape: A TensorShape, containing shape of hessians.
      multiclass_strategy: Strategy describing how to treat multiclass problems.
      loss_uses_sum_reduction: A scalar boolean tensor that specifies whether
          SUM or MEAN reduction was used for the loss.
      name: An optional handler name.
    """
    self._l1_regularization = l1_regularization
    self._l2_regularization = l2_regularization
    self._tree_complexity_regularization = tree_complexity_regularization
    self._min_node_weight = min_node_weight
    self._feature_column_group_id = feature_column_group_id
    self._name = name or ""
    self._multiclass_strategy = multiclass_strategy
    self._hessian_shape = hessian_shape
    self._gradient_shape = gradient_shape
    self._loss_uses_sum_reduction = loss_uses_sum_reduction

  def scheduled_reads(self):
    """Returns the list of `ScheduledOp`s required for update_stats."""
    return []

  @abc.abstractmethod
  def update_stats(self, stamp_token, example_partition_ids, gradients,
                   hessians, empty_gradients, empty_hessians, weights,
                   is_active, scheduled_reads):
    """Updates the state for this split handler.

    Args:
      stamp_token: An int32 scalar tensor containing the current stamp token.
      example_partition_ids: A dense tensor, containing an int32 for each
        example which is the partition id that the example ends up in.
      gradients: A dense tensor of gradients.
      hessians: A dense tensor of hessians.
      empty_gradients: A dense empty tensor of the same shape (for dimensions >
        0) as gradients.
      empty_hessians: A dense empty tensor of the same shape (for dimensions >
        0) as hessians.
      weights: A dense float32 tensor with a weight for each example.
      is_active: A boolean tensor that says if this handler is active or not.
          One value for the current layer and one value for the next layer.
      scheduled_reads: List of results from the scheduled reads.

    Returns:
      A tuple of the op that updates the stats for this handler and a list of
      `ScheduledOp`s.
    """

  def update_stats_sync(self, stamp_token, example_partition_ids, gradients,
                        hessians, empty_gradients, empty_hessians, weights,
                        is_active):
    """Updates the state for this split handler running the scheduled I/O.

    Args:
      stamp_token: An int32 scalar tensor containing the current stamp token.
      example_partition_ids: A dense tensor, containing an int32 for each
        example which is the partition id that the example ends up in.
      gradients: A dense tensor of gradients.
      hessians: A dense tensor of hessians.
      empty_gradients: A dense empty tensor of the same shape (for dimensions >
        0) as gradients.
      empty_hessians: A dense empty tensor of the same shape (for dimensions >
        0) as hessians.
      weights: A dense float32 tensor with a weight for each example.
      is_active: A boolean tensor that says if this handler is active or not.
          One value for the current layer and one value for the next layer.

    Returns:
      Op that updates the stats for this handler.
    """
    handler_reads = {self: self.scheduled_reads()}
    handler_results = batch_ops_utils.run_handler_scheduled_ops(
        handler_reads, stamp_token, None)
    update_1, scheduled_updates = self.update_stats(
        stamp_token, example_partition_ids, gradients, hessians,
        empty_gradients, empty_hessians, weights, is_active,
        handler_results[self])
    update_2 = batch_ops_utils.run_handler_scheduled_ops({
        self: scheduled_updates
    }, stamp_token, None)
    return control_flow_ops.group(update_1, *update_2[self])

  @abc.abstractmethod
  def make_splits(self, stamp_token, next_stamp_token, class_id):
    """Create the best split using the accumulated stats and flush the state.

    This should only be called by the master.

    Args:
      stamp_token: An int32 scalar tensor containing the current stamp token.
      next_stamp_token: An int32 scalar tensor containing the stamp token for
        the next iteration.
      class_id: what class id the handler gathers stats for (for tree per class
        strategy). When class_id=-1, the strategy is not tree per class.
    Returns:
      A tuple (are_splits_ready, partition_id, gain, split_info) where
      are_splits_ready is a scalar boolean tensor, partition_id is a rank 1,
      int32 tensor, gain is a rank 1 float32 tensor and split_info is a rank 1
      string tensor containing serialized SplitInfo protos.
    """
