# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Checkpoint policies that determine how tensors are split into shards."""

import math
from typing import Sequence

from absl import logging

from tensorflow.python.checkpoint.sharding import sharding_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.util import tf_export


@tf_export.tf_export("train.experimental.ShardByTaskPolicy")
class ShardByTaskPolicy(sharding_util.ShardingCallback):
  """Policy that splits tensors into shards based on their device spec task."""

  @property
  def description(self) -> str:
    return "Split tensors into shards based on their device spec task."

  def __call__(
      self,
      shardable_tensors: Sequence[sharding_util.ShardableTensor]
  ) -> Sequence[sharding_util.Shard]:
    """Callback to split tensors into shards based on their device spec task.

    Args:
      shardable_tensors: A list of ShardableTensors.

    Returns:
      List of shard dicts containing tensors.
          [ {checkpoint key: {slice_spec: tensor} } ]
    """
    tensors_by_task = {}

    for shardable_tensor in shardable_tensors:
      tensor = shardable_tensor.tensor
      checkpoint_key = shardable_tensor.checkpoint_key
      slice_spec = shardable_tensor.slice_spec

      (tensors_by_task
       .setdefault(checkpoint_key, {})[slice_spec]) = tensor

    return [tensors_by_task]


_OffsetAndShape = tuple[Sequence[int], Sequence[int]]


@tf_export.tf_export("train.experimental.MaxShardSizePolicy")
class MaxShardSizePolicy(sharding_util.ShardingCallback):
  """Policy that splits tensors into shards with a max shard size.

  Shards may exceed the max shard size if they contain 1. a single scalar/string
  tensor that could not be sliced and exceeds the max shard size or 2. the
  checkpoint object graph, whose size cannot be calculated when saving.
  """

  class MaxShardSizePartitioner():
    """Partition tensors into shards with a max shard size."""

    def _get_next_partition(self) -> tuple[int, float]:
      """Gets tensor partition with size closest to shard_size_remaining.

      Returns:
        A tuple containing the axis and size of the next partition.
      """
      rank = self._working_tensor_shape.rank
      if rank is None or rank == 0:
        return 0, math.inf

      num_elems = self._working_tensor_shape.num_elements()

      def num_partitions(axis: int) -> float:
        axis_len = self._working_tensor_shape.dims[axis].value
        slice_elems = num_elems // axis_len
        bytes_per_slice = slice_elems * self._dtype_size
        slices_per_shard = self._shard_size_remaining // bytes_per_slice
        if slices_per_shard == 0:
          return math.inf
        return math.ceil(axis_len / slices_per_shard)

      # Find axis with minimum partitions. (axis with maximum partition size)
      # (max partition size is as close as possible to the shard_size_remaining)
      min_parts = num_partitions(0)
      min_axis = 0
      for axis in range(1, rank):
        parts_along_axis = num_partitions(axis)
        part_size = num_elems * self._dtype_size / parts_along_axis
        if (parts_along_axis < min_parts and
            part_size <= self._shard_size_remaining):
          min_axis, min_parts = axis, int(parts_along_axis)
      return (min_axis,
              math.ceil(int(self._working_tensor_shape[min_axis]) / min_parts))

    def _add_partition(
        self, part_axis: int, part_size: float
    ) -> tuple[tensor_lib.Tensor, _OffsetAndShape]:
      """Adds the tensor partition to the shard, if possible.

      Args:
        part_axis: The axis of the partition.
        part_size: The size of the partition.

      Returns:
        A tuple containing the size of the slice that was added to the shard and
            the offset & shape of the remaining portion of the tensor.
      """
      if self._root_shape.rank is None or self._root_shape.rank == 0:
        return None, (None, None)

      # Add what we can to the current shard.
      slice_offset = self._working_tensor_offset
      slice_shape = [self._root_shape[i] - slice_offset[i]
                     for i in range(self._root_shape.rank)]
      slice_shape[part_axis] = part_size
      slice_size_in_bytes = int(math.prod(slice_shape)) * self._dtype_size
      with ops.device(self._device):
        tensor_slice = array_ops.slice(
            self._root_tensor, begin=slice_offset, size=slice_shape)
      slice_spec = variables.Variable.SaveSliceInfo(
          full_name=self._checkpoint_key,
          full_shape=self._root_shape,
          var_offset=slice_offset,
          var_shape=slice_shape).spec.strip()
      remaining_size = self._shard_size_remaining
      if slice_size_in_bytes > self.max_shard_size:
        logging.warning("Slice %s of tensor %s is a scalar of size %s bytes "
                        "and cannot be partitioned into a shard of max shard "
                        "size %s bytes. It will be added as an individual "
                        "shard that exceeds the max shard size.", slice_spec,
                        self._checkpoint_key, slice_size_in_bytes,
                        self.max_shard_size)
        self._large_scalars.append(
            {self._checkpoint_key: {slice_spec: tensor_slice}})
      elif slice_size_in_bytes > self._shard_size_remaining:
        # Smallest partition can't fit in the remaining shard space. Start fresh
        # with a new shard.
        return None, (None, None)
      else:
        if not self._tensors_by_shard or self._shard_size_remaining < 1:
          self._tensors_by_shard.append({})
          remaining_size = self.max_shard_size
        (self._tensors_by_shard[-1]
         .setdefault(self._checkpoint_key, {})[slice_spec]) = tensor_slice
        remaining_size -= slice_size_in_bytes

      # Get remaining portion of tensor to add to the next shard(s).
      slice_offset[part_axis] += part_size
      slice_shape = [self._root_shape[i] - slice_offset[i]
                     for i in range(self._root_shape.rank)]

      return (remaining_size, (slice_offset, slice_shape))

    def get_shards(
        self,
        max_shard_size: int,
        shardable_tensors: Sequence[sharding_util.ShardableTensor]
    ) -> Sequence[sharding_util.Shard]:
      """Callback to split tensors into shards with a max shard size.

      Args:
        max_shard_size: The maximum size of a shard file in bytes.
        shardable_tensors: A list of ShardableTensors.

      Returns:
        List of shard dicts containing tensors.
            [ {checkpoint key: {slice_spec: tensor} } ]
      """
      self.max_shard_size = max_shard_size
      self._tensors_by_shard = []
      self._large_scalars = []

      self._shard_size_remaining = self.max_shard_size
      for shardable_tensor in shardable_tensors:
        self._root_tensor = shardable_tensor.tensor
        self._root_shape = shardable_tensor.shape
        self._dtype = shardable_tensor.dtype
        self._device = shardable_tensor.device
        self._checkpoint_key = shardable_tensor.checkpoint_key
        self._slice_spec = shardable_tensor.slice_spec

        self._dtype_size = dtypes.as_dtype(self._dtype).size
        total_size = self._root_shape.num_elements() * self._dtype_size  # bytes

        # Calculate string tensor sizes.
        if self._checkpoint_key == base.OBJECT_GRAPH_PROTO_KEY:
          # In graph mode, the object graph is populated using feed_additions
          # when the session is run. So, we can't calculate the size here.
          # Fortunately, the serialized object graph string will never be that
          # big, so we just place it in the current shard without worrying about
          # its size.
          total_size = self._dtype_size = 0
        elif self._dtype == dtypes.string:
          if not context.executing_eagerly():
            with ops.device(self._device):
              self._root_tensor = ops.get_default_session().run(
                  self._root_tensor)

          if self._root_shape.rank is None or self._root_shape.rank == 0:
            sizes = [string_ops.string_length(self._root_tensor, unit="BYTE")]
          else:
            sizes = [string_ops.string_length(elem, unit="BYTE")
                     for elem in self._root_tensor]

          if context.executing_eagerly():
            sizes = [size.numpy() for size in sizes]
          else:
            with ops.device(self._device):
              sizes = ops.get_default_session().run(sizes)

          total_size = sum(sizes)
          self._dtype_size = max(sizes)

        if (total_size > self.max_shard_size and
            (self._root_shape.rank is None or self._root_shape.rank == 0)):
          logging.warning("Tensor %s is a scalar of size %s bytes and cannot "
                          "be partitioned into a shard of max shard size %s "
                          "bytes. It will be added as an individual shard that "
                          "exceeds the max shard size.",
                          self._checkpoint_key, total_size, self.max_shard_size)
          self._large_scalars.append(
              {self._checkpoint_key: {self._slice_spec: self._root_tensor}})
          continue

        # Partition tensor and add partitions to shards.
        self._working_tensor_offset = [0] * self._root_shape.rank
        self._working_tensor_shape = self._root_shape
        working_tensor_size = total_size
        while working_tensor_size > self._shard_size_remaining:
          (part_axis, part_size) = self._get_next_partition()

          if part_size == 0:
            # Tensor partition couldn't fit in remaining shard space. Try again
            # with the next full shard.
            self._tensors_by_shard.append({})
            self._shard_size_remaining = self.max_shard_size
            continue

          (remaining_size,
           (remaining_offset, remaining_shape)) = self._add_partition(
               part_axis=part_axis, part_size=part_size)

          if remaining_size is None:
            # Tensor partition couldn't fit in remaining shard space. Try again
            # with the next full shard.
            self._tensors_by_shard.append({})
            self._shard_size_remaining = self.max_shard_size
          else:
            self._working_tensor_offset = remaining_offset
            self._working_tensor_shape = tensor_shape.TensorShape(
                remaining_shape)
            working_tensor_size = (
                int(math.prod(remaining_shape)) * self._dtype_size)
            self._shard_size_remaining = remaining_size

        if self._working_tensor_shape.num_elements() > 0:
          if self._working_tensor_offset and self._working_tensor_shape:
            with ops.device(self._device):
              working_tensor = array_ops.slice(
                  self._root_tensor,
                  begin=self._working_tensor_offset,
                  size=self._working_tensor_shape.as_list())
          else:
            working_tensor = self._root_tensor
          remaining_tensor_slice_spec = variables.Variable.SaveSliceInfo(
              full_name=self._checkpoint_key,
              full_shape=self._root_shape,
              var_offset=self._working_tensor_offset,
              var_shape=self._working_tensor_shape).spec.strip()
          if not self._tensors_by_shard:
            self._tensors_by_shard.append({})
          (self._tensors_by_shard[-1]
           .setdefault(self._checkpoint_key, {})
           [remaining_tensor_slice_spec]) = working_tensor
        self._shard_size_remaining -= working_tensor_size

      return self._tensors_by_shard + self._large_scalars

  def __init__(self, max_shard_size: int):
    self.max_shard_size = max_shard_size

  @property
  def description(self) -> str:
    return "Split tensors into shards with a max shard size."

  def __call__(
      self, shardable_tensors: Sequence[sharding_util.ShardableTensor]
  ) -> Sequence[sharding_util.Shard]:
    return self.MaxShardSizePartitioner().get_shards(
        self.max_shard_size, shardable_tensors)
