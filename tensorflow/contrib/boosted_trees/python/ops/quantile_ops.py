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
"""Quantile ops python wrappers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re

from tensorflow.contrib.boosted_trees.python.ops import batch_ops_utils
# pylint: disable=unused-import
from tensorflow.contrib.boosted_trees.python.ops import boosted_trees_ops_loader
# pylint: enable=unused-import
from tensorflow.contrib.boosted_trees.python.ops import gen_quantile_ops

# go/tf-wildcard-import
# pylint: disable=wildcard-import,undefined-variable
from tensorflow.contrib.boosted_trees.python.ops.gen_quantile_ops import *
# pylint: enable=wildcard-import,undefined-variable

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import resources
from tensorflow.python.training import saver

# Pattern to remove all non alpha numeric from a string.
_PATTERN = re.compile(r"[\W_]+")


class QuantileAccumulator(saver.BaseSaverBuilder.SaveableObject):
  """A resource that allows distributed quantile computation."""

  def __init__(self,
               init_stamp_token,
               epsilon,
               num_quantiles,
               max_elements=None,
               name=None,
               container=None,
               generate_quantiles=False):
    """Creates a QuantileAccumulator object.

    Args:
      init_stamp_token: The initial value for the stamp token.
      epsilon: Error bound on the quantile computation.
      num_quantiles: Number of quantiles to produce from the final summary.
      max_elements: Maximum number of elements added to the accumulator.
      name: the name to save the accumulator under.
      container: An optional `string`. Defaults to `""`
      generate_quantiles: Generate quantiles instead of approximate boundaries.
        If true, exactly `num_quantiles` will be produced in the final summary.
    """
    self._epsilon = epsilon
    self._generate_quantiles = generate_quantiles

    name = _PATTERN.sub("", name)
    with ops.name_scope(name, "QuantileAccumulator") as name:
      self._quantile_accumulator_handle = (
          gen_quantile_ops.quantile_stream_resource_handle_op(
              container=container, shared_name=name, name=name))
      self._create_op = gen_quantile_ops.create_quantile_accumulator(
          self._quantile_accumulator_handle,
          init_stamp_token,
          epsilon=epsilon,
          max_elements=max_elements,
          num_quantiles=num_quantiles,
          generate_quantiles=generate_quantiles)
      is_initialized_op = gen_quantile_ops.quantile_accumulator_is_initialized(
          self._quantile_accumulator_handle)
    resources.register_resource(self._quantile_accumulator_handle,
                                self._create_op, is_initialized_op)
    self._make_savable(name)

  def _make_savable(self, name):
    stamp_token, state, are_buckets_ready, buckets = (
        gen_quantile_ops.quantile_accumulator_serialize(
            self._quantile_accumulator_handle))
    # slice_spec is useful for saving a slice from a variable.
    # It's not meaningful in quantile accumulator.
    slice_spec = ""
    def make_save_spec(tensor, suffix):
      return saver.BaseSaverBuilder.SaveSpec(tensor, slice_spec, name + suffix)

    specs = [make_save_spec(stamp_token, "_stamp")]
    specs += [make_save_spec(state, "_state")]
    specs += [make_save_spec(are_buckets_ready, "_are_buckets_ready")]
    specs += [make_save_spec(buckets, "buckets")]
    super(QuantileAccumulator,
          self).__init__(self._quantile_accumulator_handle, specs, name)
    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, self)

  def restore(self, restored_tensors, unused_restored_shapes):
    """Restores the associated quantile accumulator from 'restored_tensors'.

    Args:
      restored_tensors: the tensors that were loaded from a checkpoint.
      unused_restored_shapes: the shapes this object should conform to after
        restore.

    Returns:
      The operation that restores the state of the quantile accumulator.
    """
    # Read the restored tensors with the same order that were added to saving
    # spec.
    stamp_token = restored_tensors[:1]
    state = restored_tensors[1:2]
    are_buckets_ready = restored_tensors[2:3]
    buckets = restored_tensors[3]
    with ops.control_dependencies([self._create_op]):
      return gen_quantile_ops.quantile_accumulator_deserialize(
          self._quantile_accumulator_handle,
          stamp_token=stamp_token,
          stream_state=state,
          are_buckets_ready=are_buckets_ready,
          buckets=buckets)

  def get_buckets(self, stamp_token):
    """Returns quantile buckets created during previous flush."""
    are_buckets_ready, buckets = (
        gen_quantile_ops.quantile_accumulator_get_buckets(
            quantile_accumulator_handles=[self._quantile_accumulator_handle],
            stamp_token=stamp_token))
    return are_buckets_ready[0], buckets[0]

  def schedule_get_buckets(self):
    """Returns a scheduled read of buckets created during previous flush."""
    return batch_ops_utils.ScheduledStampedResourceOp(
        resource_handle=self._quantile_accumulator_handle,
        op=gen_quantile_ops.quantile_accumulator_get_buckets)

  def _make_summary(self, column, example_weights):
    if isinstance(column, sparse_tensor.SparseTensor):
      return gen_quantile_ops.make_quantile_summaries(
          dense_float_features=[],
          sparse_float_feature_indices=[column.indices],
          sparse_float_feature_values=[column.values],
          sparse_float_feature_shapes=[column.dense_shape],
          example_weights=example_weights,
          epsilon=self._epsilon / 2).sparse_summaries[0]
    else:
      return gen_quantile_ops.make_quantile_summaries(
          dense_float_features=[column],
          sparse_float_feature_indices=[],
          sparse_float_feature_values=[],
          sparse_float_feature_shapes=[],
          example_weights=example_weights,
          epsilon=self._epsilon / 2).dense_summaries[0]

  def add_summary(self, stamp_token, column, example_weights):
    """Adds quantile summary to its stream in resource."""
    summary = self._make_summary(column, example_weights)
    return gen_quantile_ops.quantile_accumulator_add_summaries(
        quantile_accumulator_handles=[self._quantile_accumulator_handle],
        stamp_token=stamp_token,
        summaries=[summary])

  def add_prebuilt_summary(self, stamp_token, summary):
    """Adds quantile summary to its stream in resource."""
    return gen_quantile_ops.quantile_accumulator_add_summaries(
        quantile_accumulator_handles=[self._quantile_accumulator_handle],
        stamp_token=stamp_token,
        summaries=[summary])

  def schedule_add_summary(self, stamp_token, column, example_weights):
    """Schedules to add a quantile summary to its stream in resource."""
    summary = self._make_summary(column, example_weights)
    return batch_ops_utils.ScheduledStampedResourceOp(
        op=gen_quantile_ops.quantile_accumulator_add_summaries,
        resource_handle=self._quantile_accumulator_handle,
        summaries=summary)

  def flush(self, stamp_token, next_stamp_token):
    """Finalizes quantile summary stream and resets it for next iteration.

    Args:
      stamp_token: Exepcted current token.
      next_stamp_token: Next value for the token.
    Returns:
      A list of quantiles or approximate boundaries.
    """
    return gen_quantile_ops.quantile_accumulator_flush(
        quantile_accumulator_handle=self._quantile_accumulator_handle,
        stamp_token=stamp_token,
        next_stamp_token=next_stamp_token)

  def flush_summary(self, stamp_token, next_stamp_token):
    """Finalizes quantile summary stream and resets it for next iteration."""
    result = gen_quantile_ops.quantile_accumulator_flush_summary(
        quantile_accumulator_handle=self._quantile_accumulator_handle,
        stamp_token=stamp_token,
        next_stamp_token=next_stamp_token)
    return result
