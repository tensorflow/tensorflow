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
"""Implementation of handler for split nodes for categorical columns."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.boosted_trees.lib.learner.batch import base_split_handler
from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.boosted_trees.python.ops import split_handler_ops
from tensorflow.contrib.boosted_trees.python.ops import stats_accumulator_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

_BIAS_FEATURE_ID = int(dtypes.int64.min)


class EqualitySplitHandler(base_split_handler.BaseSplitHandler):
  """Creates equality split type for categorical features."""

  def __init__(self,
               sparse_int_column,
               l1_regularization,
               l2_regularization,
               tree_complexity_regularization,
               min_node_weight,
               feature_column_group_id,
               gradient_shape,
               hessian_shape,
               multiclass_strategy,
               init_stamp_token=0,
               loss_uses_sum_reduction=False,
               weak_learner_type=learner_pb2.LearnerConfig.NORMAL_DECISION_TREE,
               name=None):
    """Initialize the internal state for this split handler.

    Args:
      sparse_int_column: A `SparseTensor` column with int64 values associated
        with this handler.
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
      init_stamp_token: A tensor containing an scalar for initial stamp of the
         stamped objects.
      loss_uses_sum_reduction: A scalar boolean tensor that specifies whether
          SUM or MEAN reduction was used for the loss.
      weak_learner_type: Specifies the type of weak learner to use.
      name: An optional handler name.
    """
    super(EqualitySplitHandler, self).__init__(
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization,
        tree_complexity_regularization=tree_complexity_regularization,
        min_node_weight=min_node_weight,
        feature_column_group_id=feature_column_group_id,
        gradient_shape=gradient_shape,
        hessian_shape=hessian_shape,
        multiclass_strategy=multiclass_strategy,
        loss_uses_sum_reduction=loss_uses_sum_reduction,
        name=name)
    self._stats_accumulator = stats_accumulator_ops.StatsAccumulator(
        init_stamp_token,
        gradient_shape,
        hessian_shape,
        name="StatsAccumulator/{}".format(self._name))
    self._sparse_int_column = sparse_int_column
    self._weak_learner_type = weak_learner_type

  def update_stats(self, stamp_token, example_partition_ids, gradients,
                   hessians, empty_gradients, empty_hessians, weights,
                   is_active, scheduled_reads):
    """Updates the state for equality split handler.

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
      The op that updates the stats for this handler.
    Raises:
      ValueError: If example_columns is not a single sparse column.

    """
    del scheduled_reads  # Unused by the categorical split handler.

    def not_active_inputs():
      return (constant_op.constant([], dtype=dtypes.int32),
              constant_op.constant([], dtype=dtypes.int64, shape=[1, 2]),
              empty_gradients, empty_hessians)

    def active_inputs():
      """The normal flow when the handler is active."""
      # Remove the second column of example indices matrix since it is not
      # useful.
      example_indices, _ = array_ops.split(
          self._sparse_int_column.indices, num_or_size_splits=2, axis=1)
      example_indices = array_ops.squeeze(example_indices, [1])

      filtered_gradients = array_ops.gather(gradients, example_indices)
      filtered_hessians = array_ops.gather(hessians, example_indices)
      filtered_partition_ids = array_ops.gather(example_partition_ids,
                                                example_indices)
      unique_partitions, mapped_partitions = array_ops.unique(
          example_partition_ids)

      # Compute aggregate stats for each partition.
      # The bias is computed on gradients and hessians (and not
      # filtered_gradients) which have exactly one value per example, so we
      # don't double count a gradient in multivalent columns.
      # Since unsorted_segment_sum can be numerically unstable, use 64bit
      # operation.
      gradients64 = math_ops.cast(gradients, dtypes.float64)
      hessians64 = math_ops.cast(hessians, dtypes.float64)
      per_partition_gradients = math_ops.unsorted_segment_sum(
          gradients64, mapped_partitions, array_ops.size(unique_partitions))
      per_partition_hessians = math_ops.unsorted_segment_sum(
          hessians64, mapped_partitions, array_ops.size(unique_partitions))
      per_partition_gradients = math_ops.cast(per_partition_gradients,
                                              dtypes.float32)
      per_partition_hessians = math_ops.cast(per_partition_hessians,
                                             dtypes.float32)
      # Prepend a bias feature per partition that accumulates the stats for all
      # examples in that partition.
      # Bias is added to the stats even if there are no examples with values in
      # the current sparse column. The reason is that the other example batches
      # might have values in these partitions so we have to keep the bias
      # updated.
      bias_feature_ids = array_ops.fill(
          array_ops.shape(unique_partitions), _BIAS_FEATURE_ID)
      bias_feature_ids = math_ops.cast(bias_feature_ids, dtypes.int64)
      partition_ids = array_ops.concat(
          [unique_partitions, filtered_partition_ids], 0)
      filtered_gradients = array_ops.concat(
          [per_partition_gradients, filtered_gradients], 0)
      filtered_hessians = array_ops.concat(
          [per_partition_hessians, filtered_hessians], 0)
      feature_ids = array_ops.concat(
          [bias_feature_ids, self._sparse_int_column.values], 0)
      # Dimension is always zero for sparse int features.
      dimension_ids = array_ops.zeros_like(feature_ids, dtype=dtypes.int64)
      feature_ids_and_dimensions = array_ops.stack(
          [feature_ids, dimension_ids], axis=1)
      return (partition_ids, feature_ids_and_dimensions, filtered_gradients,
              filtered_hessians)

    partition_ids, feature_ids, gradients_out, hessians_out = (
        control_flow_ops.cond(is_active[0], active_inputs, not_active_inputs))
    result = self._stats_accumulator.schedule_add(partition_ids, feature_ids,
                                                  gradients_out, hessians_out)
    return (control_flow_ops.no_op(), [result])

  def make_splits(self, stamp_token, next_stamp_token, class_id):
    """Create the best split using the accumulated stats and flush the state."""
    # Get the aggregated gradients and hessians per <partition_id, feature_id>
    # pair.
    num_minibatches, partition_ids, feature_ids, gradients, hessians = (
        self._stats_accumulator.flush(stamp_token, next_stamp_token))
    # For sum_reduction, we don't need to divide by number of minibatches.

    num_minibatches = control_flow_ops.cond(
        ops.convert_to_tensor(self._loss_uses_sum_reduction),
        lambda: math_ops.to_int64(1), lambda: num_minibatches)
    partition_ids, gains, split_infos = (
        split_handler_ops.build_categorical_equality_splits(
            num_minibatches=num_minibatches,
            partition_ids=partition_ids,
            feature_ids=feature_ids,
            gradients=gradients,
            hessians=hessians,
            class_id=class_id,
            feature_column_group_id=self._feature_column_group_id,
            l1_regularization=self._l1_regularization,
            l2_regularization=self._l2_regularization,
            tree_complexity_regularization=self._tree_complexity_regularization,
            min_node_weight=self._min_node_weight,
            bias_feature_id=_BIAS_FEATURE_ID,
            multiclass_strategy=self._multiclass_strategy,
            weak_learner_type=self._weak_learner_type))
    # There are no warm-up rounds needed in the equality column handler. So we
    # always return ready.
    are_splits_ready = constant_op.constant(True)
    return (are_splits_ready, partition_ids, gains, split_infos)

  def reset(self, stamp_token, next_stamp_token):
    reset = self._stats_accumulator.flush(stamp_token, next_stamp_token)
    return reset
