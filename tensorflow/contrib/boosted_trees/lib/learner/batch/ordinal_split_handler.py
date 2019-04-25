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
"""Implementation of handler for split nodes for float columns.

The general idea in batch split finding is that each handler will accumulate its
own statistics on multiple workers. After some steps, the master runs
make_splits() sub-graph of each handler and each handler returns its best split
per partition.

The way we ensure consistency of statistics is by using stamp_tokens for read
and write operations. During each update of the model, a new stamp token is
created. This stamp token makes sure that updates from the previous iterations
are not included in the statistics for this iteration.

Inequality splits for float features are created similar to the method described
in Approximate Algorithm described in https://arxiv.org/pdf/1603.02754v3.pdf.
Weighted quantiles of the feature columns are computed in a distributed fashion
using quantile_ops.quantile_accumulator.
After certain number of steps of parallel accumulation of quantile statistics,
we decide on bucket boundaries. These bucket boundaries are then used for the
next N steps to accumulate gradients and hessians per bucket.

In this implementation, we gather quantile statistics and gradient statistics
concurrently. That means that we don't wait until we have enough quantile
statistics for bucketization before we start gathering gradient stats. Instead
during each step we create quantile stats for the next iteration and use the
previous quantile buckets for gradient stats accumulation.
In make_splits, we do these steps:
1) Get the buckets that were used creating for the gradient stats.
2) Create bucket boundaries for the next N iterations and clear the accumulated
   quantile stats.
n3) Get the accumulated gradient stats and clear the accumulator. This step can
   run in parallel to step 2.
4) For each leaf node in the current tree (partition):
   4.1) Get the overall gain computed with gradients and hessians of all
        examples that end up in this partition.
   4.2) Compute tensors of left and right cumulative sum of gradients, hessians
        and gain. The first dimension of these tensors are the bucket
        boundaries.
   4.3) Find the gains for all bucket boundaries:
        split_gains = left_gain + right_gain - overall_gain.
   4.4) Find the bucket boundary that has the best gain (argmax(split_gains))
   4.5) For Sparse handler, we also consider the gain for when the examples go
        the left child and when the examples go to the right child and pick the
        default direction that yields the most gain.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.contrib.boosted_trees.lib.learner.batch import base_split_handler
from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.boosted_trees.python.ops import gen_quantile_ops
from tensorflow.contrib.boosted_trees.python.ops import gen_stats_accumulator_ops
from tensorflow.contrib.boosted_trees.python.ops import quantile_ops
from tensorflow.contrib.boosted_trees.python.ops import split_handler_ops
from tensorflow.contrib.boosted_trees.python.ops import stats_accumulator_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


_BIAS_FEATURE_ID = -1
# Pattern to remove all non alpha numeric from a string.
_PATTERN = re.compile(r"[\W_]+")


class InequalitySplitHandler(base_split_handler.BaseSplitHandler):
  """Base class for handlers of inequality splits."""

  def __init__(self,
               l1_regularization,
               l2_regularization,
               tree_complexity_regularization,
               min_node_weight,
               feature_column_group_id,
               epsilon,
               num_quantiles,
               gradient_shape,
               hessian_shape,
               multiclass_strategy,
               init_stamp_token=0,
               loss_uses_sum_reduction=False,
               name=None):
    """Initialize the internal state for this split handler.

    Args:
      l1_regularization: L1 regularization applied for this split handler.
      l2_regularization: L2 regularization applied for this split handler.
      tree_complexity_regularization: Tree complexity regularization applied
          for this split handler.
      min_node_weight: Minimum sum of weights of examples in each partition to
          be considered for splitting.
      feature_column_group_id: Feature column group index.
      epsilon: A float, the error bound for quantile computation.
      num_quantiles: An int, the number of buckets to create from the histogram.
      gradient_shape: A TensorShape, containing shape of gradients.
      hessian_shape: A TensorShape, containing shape of hessians.
      multiclass_strategy: Strategy describing how to treat multiclass problems.
      init_stamp_token: A tensor containing an scalar for initial stamp of the
         stamped objects.
      loss_uses_sum_reduction: A scalar boolean tensor that specifies whether
          SUM or MEAN reduction was used for the loss.
      name: An optional handler name.
    """
    super(InequalitySplitHandler, self).__init__(
        name=name,
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization,
        tree_complexity_regularization=tree_complexity_regularization,
        min_node_weight=min_node_weight,
        feature_column_group_id=feature_column_group_id,
        gradient_shape=gradient_shape,
        hessian_shape=hessian_shape,
        multiclass_strategy=multiclass_strategy,
        loss_uses_sum_reduction=loss_uses_sum_reduction)
    self._stats_accumulator = stats_accumulator_ops.StatsAccumulator(
        init_stamp_token,
        gradient_shape,
        hessian_shape,
        name="StatsAccumulator/{}".format(self._name))
    # Allocate both stats accumulator and quantile accumulator on the same
    # device so that we can build splits with fewer RPCs.
    with ops.colocate_with(self._stats_accumulator.resource_handle):
      self._quantile_accumulator = quantile_ops.QuantileAccumulator(
          init_stamp_token,
          epsilon=epsilon,
          num_quantiles=num_quantiles,
          name="QuantileAccumulator/{}".format(self._name))

  def reset(self, stamp_token, next_stamp_token):
    reset_1 = self._stats_accumulator.flush(stamp_token, next_stamp_token)
    reset_2 = self._quantile_accumulator.flush(stamp_token, next_stamp_token)
    return control_flow_ops.group([reset_1, reset_2])


class DenseSplitHandler(InequalitySplitHandler):
  """Computes stats and finds the best inequality splits on dense columns."""

  def __init__(self,
               dense_float_column,
               l1_regularization,
               l2_regularization,
               tree_complexity_regularization,
               min_node_weight,
               feature_column_group_id,
               epsilon,
               num_quantiles,
               gradient_shape,
               hessian_shape,
               multiclass_strategy,
               init_stamp_token=0,
               loss_uses_sum_reduction=False,
               weak_learner_type=learner_pb2.LearnerConfig.NORMAL_DECISION_TREE,
               name=None):
    """Initialize the internal state for this split handler.

    Args:
      dense_float_column: A `Tensor` column associated with this handler.
      l1_regularization: L1 regularization applied for this split handler.
      l2_regularization: L2 regularization applied for this split handler.
      tree_complexity_regularization: Tree complexity regularization applied
          for this split handler.
      min_node_weight: Minimum sum of weights of examples in each partition to
          be considered for splitting.
      feature_column_group_id: Feature column group index.
      epsilon: A float, the error bound for quantile computation.
      num_quantiles: An int, the number of buckets to create from the histogram.
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
    super(DenseSplitHandler, self).__init__(
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization,
        tree_complexity_regularization=tree_complexity_regularization,
        min_node_weight=min_node_weight,
        feature_column_group_id=feature_column_group_id,
        epsilon=epsilon,
        num_quantiles=num_quantiles,
        init_stamp_token=init_stamp_token,
        name=name,
        gradient_shape=gradient_shape,
        hessian_shape=hessian_shape,
        multiclass_strategy=multiclass_strategy,
        loss_uses_sum_reduction=loss_uses_sum_reduction)
    self._dense_float_column = dense_float_column
    self._weak_learner_type = weak_learner_type
    # Register dense_make_stats_update function as an Op to the graph.
    g = ops.get_default_graph()
    dense_make_stats_update.add_to_graph(g)

  def scheduled_reads(self):
    return [self._quantile_accumulator.schedule_get_buckets()]

  def update_stats(self, stamp_token, example_partition_ids, gradients,
                   hessians, empty_gradients, empty_hessians, weights,
                   is_active, scheduled_reads):
    """Updates the state for dense split handler.

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
      scheduled_reads: List of scheduled reads for this handler.

    Returns:
      The op that updates the stats for this handler.
    """
    name = _PATTERN.sub("", self._name)
    with ops.name_scope(name, "DenseSplitHandler"):
      are_buckets_ready, buckets = scheduled_reads[0]
      (quantile_values, quantile_weights, example_partition_ids,
       feature_ids, gradients, hessians) = dense_make_stats_update(
           is_active, are_buckets_ready, self._dense_float_column, buckets,
           example_partition_ids, gradients, hessians, weights, empty_gradients,
           empty_hessians)
      update_quantiles = self._quantile_accumulator.schedule_add_summary(
          stamp_token=stamp_token,
          column=quantile_values,
          example_weights=quantile_weights)
      update_stats = self._stats_accumulator.schedule_add(
          example_partition_ids, feature_ids, gradients, hessians)
      return control_flow_ops.no_op(), [update_quantiles, update_stats]

  def make_splits(self, stamp_token, next_stamp_token, class_id):
    """Create the best split using the accumulated stats and flush the state."""
    if (self._gradient_shape == tensor_shape.scalar() and
        self._hessian_shape == tensor_shape.scalar()):
      handler = make_dense_split_scalar
    else:
      handler = make_dense_split_tensor

    are_splits_ready, partition_ids, gains, split_infos = (
        handler(self._quantile_accumulator.resource_handle,
                self._stats_accumulator.resource_handle, stamp_token,
                next_stamp_token, self._multiclass_strategy, class_id,
                self._feature_column_group_id, self._l1_regularization,
                self._l2_regularization, self._tree_complexity_regularization,
                self._min_node_weight, self._loss_uses_sum_reduction,
                self._weak_learner_type))
    return are_splits_ready, partition_ids, gains, split_infos


def _make_dense_split(quantile_accumulator_handle, stats_accumulator_handle,
                      stamp_token, next_stamp_token, multiclass_strategy,
                      class_id, feature_column_id, l1_regularization,
                      l2_regularization, tree_complexity_regularization,
                      min_node_weight, is_multi_dimentional,
                      loss_uses_sum_reduction, weak_learner_type):
  """Function that builds splits for a dense feature column."""
  # Get the bucket boundaries
  are_splits_ready, buckets = (
      gen_quantile_ops.quantile_accumulator_get_buckets(
          quantile_accumulator_handles=[quantile_accumulator_handle],
          stamp_token=stamp_token))
  # quantile_accumulator_get_buckets returns a list of results per handle that
  # we pass to it. In this case we're getting results just for one resource.
  are_splits_ready = are_splits_ready[0]
  buckets = buckets[0]

  # After we receive the boundaries from previous iteration we can flush
  # the quantile accumulator.
  with ops.control_dependencies([buckets]):
    flush_quantiles = gen_quantile_ops.quantile_accumulator_flush(
        quantile_accumulator_handle=quantile_accumulator_handle,
        stamp_token=stamp_token,
        next_stamp_token=next_stamp_token)

  if is_multi_dimentional:
    num_minibatches, partition_ids, bucket_ids, gradients, hessians = (
        gen_stats_accumulator_ops.stats_accumulator_tensor_flush(
            stats_accumulator_handle, stamp_token, next_stamp_token))
  else:
    num_minibatches, partition_ids, bucket_ids, gradients, hessians = (
        gen_stats_accumulator_ops.stats_accumulator_scalar_flush(
            stats_accumulator_handle, stamp_token, next_stamp_token))
  # For sum_reduction, we don't need to divide by number of minibatches.
  num_minibatches = control_flow_ops.cond(
      loss_uses_sum_reduction,
      lambda: math_ops.cast(1, dtypes.int64),
      lambda: num_minibatches)
  # Put quantile and stats accumulator flushing in the dependency path.
  with ops.control_dependencies([flush_quantiles, partition_ids]):
    are_splits_ready = array_ops.identity(are_splits_ready)
  partition_ids, gains, split_infos = (
      split_handler_ops.build_dense_inequality_splits(
          num_minibatches=num_minibatches,
          bucket_boundaries=buckets,
          partition_ids=partition_ids,
          bucket_ids=bucket_ids,
          gradients=gradients,
          hessians=hessians,
          class_id=class_id,
          feature_column_group_id=feature_column_id,
          l1_regularization=l1_regularization,
          l2_regularization=l2_regularization,
          tree_complexity_regularization=tree_complexity_regularization,
          min_node_weight=min_node_weight,
          multiclass_strategy=multiclass_strategy,
          weak_learner_type=weak_learner_type))
  return are_splits_ready, partition_ids, gains, split_infos


class SparseSplitHandler(InequalitySplitHandler):
  """Computes stats and finds the best inequality splits on sparse columns."""

  def __init__(self,
               sparse_float_column,
               l1_regularization,
               l2_regularization,
               tree_complexity_regularization,
               min_node_weight,
               feature_column_group_id,
               epsilon,
               num_quantiles,
               gradient_shape,
               hessian_shape,
               multiclass_strategy,
               init_stamp_token=0,
               loss_uses_sum_reduction=False,
               name=None):
    """Initialize the internal state for this split handler.

    Args:
      sparse_float_column: A `SparseTensor` column associated with this handler.
      l1_regularization: L1 regularization applied for this split handler.
      l2_regularization: L2 regularization applied for this split handler.
      tree_complexity_regularization: Tree complexity regularization applied
          for this split handler.
      min_node_weight: Minimum sum of weights of examples in each partition to
          be considered for splitting.
      feature_column_group_id: Feature column group index.
      epsilon: A float, the error bound for quantile computation.
      num_quantiles: An int, the number of buckets to create from the histogram.
      gradient_shape: A TensorShape, containing shape of gradients.
      hessian_shape: A TensorShape, containing shape of hessians.
      multiclass_strategy: Strategy describing how to treat multiclass problems.
      init_stamp_token: A tensor containing an scalar for initial stamp of the
         stamped objects.
      loss_uses_sum_reduction: A scalar boolean tensor that specifies whether
          SUM or MEAN reduction was used for the loss.
      name: An optional handler name.
    """
    super(SparseSplitHandler, self).__init__(
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization,
        tree_complexity_regularization=tree_complexity_regularization,
        min_node_weight=min_node_weight,
        feature_column_group_id=feature_column_group_id,
        epsilon=epsilon,
        num_quantiles=num_quantiles,
        gradient_shape=gradient_shape,
        hessian_shape=hessian_shape,
        multiclass_strategy=multiclass_strategy,
        init_stamp_token=init_stamp_token,
        loss_uses_sum_reduction=loss_uses_sum_reduction,
        name=name)
    self._sparse_float_column = sparse_float_column

  def scheduled_reads(self):
    return [self._quantile_accumulator.schedule_get_buckets()]

  def update_stats(self, stamp_token, example_partition_ids, gradients,
                   hessians, empty_gradients, empty_hessians, weights,
                   is_active, scheduled_reads):
    """Updates the state for dense split handler.

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
    """
    are_buckets_ready, buckets = scheduled_reads[0]
    with ops.name_scope(self._name, "SparseSplitHandler"):
      (quantile_indices, quantile_values, quantile_shapes, quantile_weights,
       example_partition_ids, feature_ids, gradients,
       hessians) = sparse_make_stats_update(
           is_active, are_buckets_ready, self._sparse_float_column.indices,
           self._sparse_float_column.values,
           self._sparse_float_column.dense_shape, buckets,
           example_partition_ids, gradients, hessians, weights, empty_gradients,
           empty_hessians)
      update_quantiles = self._quantile_accumulator.schedule_add_summary(
          stamp_token=stamp_token,
          column=sparse_tensor.SparseTensor(quantile_indices, quantile_values,
                                            quantile_shapes),
          example_weights=quantile_weights)
      update_stats = self._stats_accumulator.schedule_add(
          example_partition_ids, feature_ids, gradients, hessians)
      return (control_flow_ops.no_op(), [update_quantiles, update_stats])

  def make_splits(self, stamp_token, next_stamp_token, class_id):
    """Create the best split using the accumulated stats and flush the state."""
    if (self._gradient_shape == tensor_shape.scalar() and
        self._hessian_shape == tensor_shape.scalar()):
      handler = make_sparse_split_scalar
    else:
      handler = make_sparse_split_tensor

    are_splits_ready, partition_ids, gains, split_infos = (
        handler(self._quantile_accumulator.resource_handle,
                self._stats_accumulator.resource_handle, stamp_token,
                next_stamp_token, self._multiclass_strategy, class_id,
                self._feature_column_group_id, self._l1_regularization,
                self._l2_regularization, self._tree_complexity_regularization,
                self._min_node_weight, self._loss_uses_sum_reduction))
    return are_splits_ready, partition_ids, gains, split_infos


def _make_sparse_split(
    quantile_accumulator_handle, stats_accumulator_handle, stamp_token,
    next_stamp_token, multiclass_strategy, class_id, feature_column_id,
    l1_regularization, l2_regularization, tree_complexity_regularization,
    min_node_weight, is_multi_dimentional, loss_uses_sum_reduction):
  """Function that builds splits for a sparse feature column."""
  # Get the bucket boundaries
  are_splits_ready, buckets = (
      gen_quantile_ops.quantile_accumulator_get_buckets(
          quantile_accumulator_handles=[quantile_accumulator_handle],
          stamp_token=stamp_token))
  # quantile_accumulator_get_buckets returns a list of results per handle that
  # we pass to it. In this case we're getting results just for one resource.
  are_splits_ready = are_splits_ready[0]
  buckets = buckets[0]

  # After we receive the boundaries from previous iteration we can flush
  # the quantile accumulator.
  with ops.control_dependencies([buckets]):
    flush_quantiles = gen_quantile_ops.quantile_accumulator_flush(
        quantile_accumulator_handle=quantile_accumulator_handle,
        stamp_token=stamp_token,
        next_stamp_token=next_stamp_token)

  if is_multi_dimentional:
    num_minibatches, partition_ids, bucket_ids, gradients, hessians = (
        gen_stats_accumulator_ops.stats_accumulator_tensor_flush(
            stats_accumulator_handle, stamp_token, next_stamp_token))
  else:
    num_minibatches, partition_ids, bucket_ids, gradients, hessians = (
        gen_stats_accumulator_ops.stats_accumulator_scalar_flush(
            stats_accumulator_handle, stamp_token, next_stamp_token))
  num_minibatches = control_flow_ops.cond(
      loss_uses_sum_reduction,
      lambda: math_ops.cast(1, dtypes.int64),
      lambda: num_minibatches)
  # Put quantile and stats accumulator flushing in the dependency path.
  with ops.control_dependencies([flush_quantiles, partition_ids]):
    are_splits_ready = array_ops.identity(are_splits_ready)
  partition_ids, gains, split_infos = (
      split_handler_ops.build_sparse_inequality_splits(
          num_minibatches=num_minibatches,
          bucket_boundaries=buckets,
          partition_ids=partition_ids,
          bucket_ids=bucket_ids,
          gradients=gradients,
          hessians=hessians,
          class_id=class_id,
          feature_column_group_id=feature_column_id,
          l1_regularization=l1_regularization,
          l2_regularization=l2_regularization,
          tree_complexity_regularization=tree_complexity_regularization,
          min_node_weight=min_node_weight,
          bias_feature_id=_BIAS_FEATURE_ID,
          multiclass_strategy=multiclass_strategy))
  return are_splits_ready, partition_ids, gains, split_infos


def _specialize_make_split_dense(func, is_multi_dimentional):
  """Builds a specialized version of the function."""

  @function.Defun(
      dtypes.resource,
      dtypes.resource,
      dtypes.int64,
      dtypes.int64,
      dtypes.int32,
      dtypes.int32,
      dtypes.int32,
      dtypes.float32,
      dtypes.float32,
      dtypes.float32,
      dtypes.float32,
      dtypes.bool,
      dtypes.int32,
      noinline=True)
  def f(quantile_accumulator_handle, stats_accumulator_handle, stamp_token,
        next_stamp_token, multiclass_strategy, class_id, feature_column_id,
        l1_regularization, l2_regularization, tree_complexity_regularization,
        min_node_weight, loss_uses_sum_reduction, weak_learner_type):
    """Function that builds splits for a sparse feature column."""
    return func(quantile_accumulator_handle, stats_accumulator_handle,
                stamp_token, next_stamp_token, multiclass_strategy, class_id,
                feature_column_id, l1_regularization, l2_regularization,
                tree_complexity_regularization, min_node_weight,
                is_multi_dimentional, loss_uses_sum_reduction,
                weak_learner_type)

  return f


def _specialize_make_split_sparse(func, is_multi_dimentional):
  """Builds a specialized version of the function."""

  @function.Defun(
      dtypes.resource,
      dtypes.resource,
      dtypes.int64,
      dtypes.int64,
      dtypes.int32,
      dtypes.int32,
      dtypes.int32,
      dtypes.float32,
      dtypes.float32,
      dtypes.float32,
      dtypes.float32,
      dtypes.bool,
      noinline=True)
  def f(quantile_accumulator_handle, stats_accumulator_handle, stamp_token,
        next_stamp_token, multiclass_strategy, class_id, feature_column_id,
        l1_regularization, l2_regularization, tree_complexity_regularization,
        min_node_weight, loss_uses_sum_reduction):
    """Function that builds splits for a sparse feature column."""
    return func(quantile_accumulator_handle, stats_accumulator_handle,
                stamp_token, next_stamp_token, multiclass_strategy, class_id,
                feature_column_id, l1_regularization, l2_regularization,
                tree_complexity_regularization, min_node_weight,
                is_multi_dimentional, loss_uses_sum_reduction)

  return f


make_dense_split_scalar = _specialize_make_split_dense(
    _make_dense_split, is_multi_dimentional=False)

make_dense_split_tensor = _specialize_make_split_dense(
    _make_dense_split, is_multi_dimentional=True)

make_sparse_split_scalar = _specialize_make_split_sparse(
    _make_sparse_split, is_multi_dimentional=False)
make_sparse_split_tensor = _specialize_make_split_sparse(
    _make_sparse_split, is_multi_dimentional=True)


@function.Defun(
    dtypes.bool,
    dtypes.bool,
    dtypes.float32,
    dtypes.float32,
    dtypes.int32,
    dtypes.float32,
    dtypes.float32,
    dtypes.float32,
    dtypes.float32,
    dtypes.float32,
    noinline=True)
def dense_make_stats_update(is_active, are_buckets_ready, float_column,
                            quantile_buckets, example_partition_ids, gradients,
                            hessians, weights, empty_gradients, empty_hessians):
  """Updates the state for dense split handler."""
  empty_float = constant_op.constant_v1([], dtype=dtypes.float32)

  quantile_values, quantile_weights = control_flow_ops.cond(
      is_active[1],  # For the next layer, this handler is inactive.
      lambda: (float_column, weights),
      lambda: (empty_float, empty_float))

  def ready_inputs_fn():
    """Branch to execute when quantiles are ready."""
    quantized_feature = quantile_ops.quantiles([float_column], [],
                                               [quantile_buckets], [], [])
    quantized_feature = math_ops.cast(quantized_feature[0], dtypes.int64)
    quantized_feature = array_ops.squeeze(quantized_feature, axis=0)
    return (example_partition_ids, quantized_feature, gradients, hessians)

  def not_ready_inputs_fn():
    return (constant_op.constant_v1([], dtype=dtypes.int32),
            constant_op.constant_v1([[]], dtype=dtypes.int64, shape=[1, 2]),
            empty_gradients, empty_hessians)

  example_partition_ids, feature_ids, gradients, hessians = (
      control_flow_ops.cond(
          math_ops.logical_and(
              math_ops.logical_and(are_buckets_ready,
                                   array_ops.size(quantile_buckets) > 0),
              is_active[0]), ready_inputs_fn, not_ready_inputs_fn))
  return (quantile_values, quantile_weights, example_partition_ids, feature_ids,
          gradients, hessians)


@function.Defun(
    dtypes.bool,
    dtypes.bool,
    dtypes.int64,
    dtypes.float32,
    dtypes.int64,
    dtypes.float32,
    dtypes.int32,
    dtypes.float32,
    dtypes.float32,
    dtypes.float32,
    dtypes.float32,
    dtypes.float32,
    noinline=True)
def sparse_make_stats_update(
    is_active, are_buckets_ready, sparse_column_indices, sparse_column_values,
    sparse_column_shape, quantile_buckets, example_partition_ids, gradients,
    hessians, weights, empty_gradients, empty_hessians):
  """Updates the state for this split handler."""

  def quantiles_ready():
    """The subgraph for when the quantiles are ready."""
    quantized_feature = quantile_ops.quantiles([], [sparse_column_values], [],
                                               [quantile_buckets],
                                               [sparse_column_indices])

    quantized_feature = math_ops.cast(quantized_feature[1], dtypes.int64)
    quantized_feature = array_ops.squeeze(quantized_feature, axis=0)

    example_indices, _ = array_ops.split(
        sparse_column_indices, num_or_size_splits=2, axis=1)
    example_indices = array_ops.squeeze(example_indices, [1])
    filtered_gradients = array_ops.gather(gradients, example_indices)
    filtered_hessians = array_ops.gather(hessians, example_indices)
    filtered_partition_ids = array_ops.gather(example_partition_ids,
                                              example_indices)
    unique_partitions, mapped_partitions = array_ops.unique(
        example_partition_ids)

    # Compute aggregate stats for each partition.
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
    bias_feature_ids = array_ops.fill(
        array_ops.shape(unique_partitions), _BIAS_FEATURE_ID)
    bias_feature_ids = math_ops.cast(bias_feature_ids, dtypes.int64)
    zeros = array_ops.zeros_like(bias_feature_ids)
    bias_feature_ids = array_ops.stack([bias_feature_ids, zeros], axis=1)

    partition_ids = array_ops.concat(
        [unique_partitions, filtered_partition_ids], 0)
    filtered_gradients = array_ops.concat(
        [per_partition_gradients, filtered_gradients], 0)
    filtered_hessians = array_ops.concat(
        [per_partition_hessians, filtered_hessians], 0)

    bucket_ids = array_ops.concat([bias_feature_ids, quantized_feature], 0)

    return partition_ids, bucket_ids, filtered_gradients, filtered_hessians

  def quantiles_not_ready():
    """The subgraph for when the quantiles are not ready."""
    return (constant_op.constant_v1([], dtype=dtypes.int32),
            constant_op.constant_v1([], dtype=dtypes.int64, shape=[1, 2]),
            empty_gradients, empty_hessians)

  empty_float = constant_op.constant_v1([], dtype=dtypes.float32)
  handler_not_active = (constant_op.constant(
      [], dtype=dtypes.int64, shape=[0, 2]), empty_float,
                        constant_op.constant([0, 1], dtype=dtypes.int64),
                        empty_float)
  handler_active = (sparse_column_indices, sparse_column_values,
                    sparse_column_shape, weights)
  quantile_indices, quantile_values, quantile_shape, quantile_weights = (
      control_flow_ops.cond(is_active[1], lambda: handler_active,
                            lambda: handler_not_active))

  example_partition_ids, feature_ids, gradients, hessians = (
      control_flow_ops.cond(
          math_ops.logical_and(are_buckets_ready,
                               array_ops.size(quantile_buckets) > 0),
          quantiles_ready, quantiles_not_ready))

  return (quantile_indices, quantile_values, quantile_shape, quantile_weights,
          example_partition_ids, feature_ids, gradients, hessians)
