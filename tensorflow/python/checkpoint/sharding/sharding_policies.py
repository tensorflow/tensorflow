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
from typing import MutableSequence, Sequence

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
  ) -> Sequence[sharding_util.TensorSliceDict]:
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


_PartitionAxisAndSize = tuple[int, int]
_OffsetAndShape = tuple[Sequence[int], Sequence[int]]


@tf_export.tf_export("train.experimental.MaxShardSizePolicy")
class MaxShardSizePolicy(sharding_util.ShardingCallback):
  """Policy that splits tensors into shards with a max shard size.

  Shards may exceed the max shard size if they contain 1. a single scalar/string
  tensor that could not be sliced and exceeds the max shard size or 2. the
  checkpoint object graph, whose size cannot be calculated when saving.
  """

  def __init__(self, max_shard_size: int):
    self.max_shard_size = max_shard_size

  @property
  def description(self) -> str:
    return "Split tensors into shards with a max shard size."

  def _get_next_partition(
      self,
      shard_size_remaining: int,
      shape: tensor_shape.TensorShape,
      dtype_size: int,
      num_elems: int
  ) -> _PartitionAxisAndSize:
    """Gets tensor partition with size closest to shard_size_remaining.

    Args:
      shard_size_remaining: Size in bytes of the space remaining in the shard.
      shape: Shape of the working tensor to partition in the remaining
          shard space.
      dtype_size: Size in bytes of the dtype of the working tensor.
      num_elems: Number of elements in the working tensor.

    Returns:
      A tuple containing the axis of the next partition and that partition size.
    """
    if shape.rank is None or shape.rank == 0:
      return 0, math.inf

    # Find axis with minimum partitions. (aka axis with maximum partition size)
    # (max partition size is as close as possible to the shard_size_remaining)
    bytes_per_slice = num_elems // shape.dims[0].value * dtype_size
    slices_per_shard = max(
        1, math.floor(shard_size_remaining / bytes_per_slice))
    min_parts = math.ceil(shape.dims[0].value / slices_per_shard)
    min_axis = 0
    for axis in range(1, shape.rank):
      bytes_per_slice = num_elems // shape.dims[axis].value * dtype_size
      slices_per_shard = max(
          1, math.floor(shard_size_remaining / bytes_per_slice))
      axis_parts = math.ceil(shape.dims[axis].value / slices_per_shard)
      partition_size = num_elems * dtype_size / axis_parts
      if (axis_parts < min_parts and
          partition_size < shard_size_remaining):
        min_axis, min_parts = axis, int(axis_parts)
    return min_axis, math.ceil(int(shape[min_axis]) / min_parts)

  def _add_partition(
      self,
      root_shardable_tensor: sharding_util.ShardableTensor,
      dtype_size: int,
      working_tensor_offset: Sequence[int],
      part_axis_and_size: _PartitionAxisAndSize,
      shard_size_remaining: int,
      max_shard_size: int,
      tensors_by_shard: MutableSequence[sharding_util.TensorSliceDict],
      large_scalars: MutableSequence[sharding_util.TensorSliceDict],
  ) -> tuple[tensor_lib.Tensor, _OffsetAndShape]:
    """Adds the tensor partition to the shard, if possible.

    Args:
      root_shardable_tensor: The full tensor being partitioned.
      dtype_size: Size in bytes of the dtype of the working tensor.
      working_tensor_offset: The offset of the working tensor in the full
          tensor.
      part_axis_and_size: A tuple containing the axis of the partition and that
          partition size.
      shard_size_remaining: Size in bytes of the space remaining in the shard.
      max_shard_size: Max size in bytes allowed for a checkpoint shard.
      tensors_by_shard: List of shard dicts containing tensors.
          [ {checkpoint key: {slice_spec: tensor} } ]
      large_scalars: List of shard dicts containing scalars too large to fit in
          the max_shard_size. [ {checkpoint key: {slice_spec: tensor} } ]

    Returns:
      A tuple containing the size of the slice that was added to the shard and
          the offset & shape of the remaining portion of the tensor.
    """
    root_tensor = root_shardable_tensor.tensor
    root_tensor_shape = root_shardable_tensor.shape
    checkpoint_key = root_shardable_tensor.checkpoint_key

    if root_tensor_shape.rank is None or root_tensor_shape.rank == 0:
      return None, (None, None)

    min_axis, part_size = part_axis_and_size

    # Add what we can to the current shard.
    slice_offset = working_tensor_offset
    slice_shape = [root_tensor_shape[i] - slice_offset[i]
                   for i in range(root_tensor_shape.rank)]
    slice_shape[min_axis] = part_size
    slice_size_in_bytes = int(math.prod(slice_shape)) * dtype_size
    with ops.device(root_shardable_tensor.device):
      tensor_slice = array_ops.slice(
          root_tensor, begin=slice_offset, size=slice_shape)
    slice_spec = variables.Variable.SaveSliceInfo(
        full_name=checkpoint_key,
        full_shape=root_tensor_shape,
        var_offset=slice_offset,
        var_shape=slice_shape).spec.strip()
    remaining_size = shard_size_remaining
    if slice_size_in_bytes > max_shard_size:
      logging.warning("Slice %s of tensor %s is a scalar of size %s bytes and "
                      "cannot be partitioned into a shard of max shard size %s "
                      "bytes. It will be added as an individual shard that "
                      "exceeds the max shard size.", slice_spec, checkpoint_key,
                      slice_size_in_bytes, max_shard_size)
      large_scalars.append({checkpoint_key: {slice_spec: tensor_slice}})
    elif slice_size_in_bytes > shard_size_remaining:
      # Smallest partition can't fit in the remaining shard space. Start fresh
      # with a new shard.
      return None, (None, None)
    else:
      if not tensors_by_shard or shard_size_remaining < 1:
        tensors_by_shard.append({})
        remaining_size = max_shard_size
      (tensors_by_shard[-1]
       .setdefault(checkpoint_key, {})[slice_spec]) = tensor_slice
      remaining_size -= slice_size_in_bytes

    # Get remaining portion of tensor to add to the next shard(s).
    slice_offset[min_axis] += part_size
    slice_shape = [root_tensor_shape[i] - slice_offset[i]
                   for i in range(root_tensor_shape.rank)]

    return (remaining_size, (slice_offset, slice_shape))

  def __call__(
      self, shardable_tensors: Sequence[sharding_util.ShardableTensor]
  ) -> Sequence[sharding_util.TensorSliceDict]:
    """Callback to split tensors into shards with a max shard size.

    Args:
      shardable_tensors: A list of ShardableTensors.

    Returns:
      List of shard dicts containing tensors.
          [ {checkpoint key: {slice_spec: tensor} } ]
    """
    tensors_by_shard = []
    large_scalars = []

    shard_size_remaining = self.max_shard_size
    for shardable_tensor in shardable_tensors:
      root_tensor = shardable_tensor.tensor
      root_shape = shardable_tensor.shape
      dtype = shardable_tensor.dtype
      checkpoint_key = shardable_tensor.checkpoint_key

      dtype_size = dtypes.as_dtype(dtype).size
      total_size = root_shape.num_elements() * dtype_size  # in bytes

      # Calculate string tensor sizes.
      if checkpoint_key == base.OBJECT_GRAPH_PROTO_KEY:
        # In graph mode, the object graph is populated using feed_additions when
        # the session is run. So, we can't calculate the size here. Fortunately,
        # the serialized object graph string will never be that big, so we just
        # place it in the current shard without worrying about its size.
        total_size = dtype_size = 0
      elif dtype == dtypes.string:
        if not context.executing_eagerly():
          with ops.device(shardable_tensor.device):
            root_tensor = ops.get_default_session().run(root_tensor)

        if root_shape.rank is None or root_shape.rank == 0:
          sizes = [string_ops.string_length(root_tensor, unit="BYTE")]
        else:
          sizes = [string_ops.string_length(elem, unit="BYTE")
                   for elem in root_tensor]

        if context.executing_eagerly():
          sizes = [size.numpy() for size in sizes]
        else:
          with ops.device(shardable_tensor.device):
            sizes = ops.get_default_session().run(sizes)

        total_size = sum(sizes)
        dtype_size = max(sizes)

      if (total_size > self.max_shard_size and
          (root_shape.rank is None or root_shape.rank == 0)):
        logging.warning("Tensor %s is a scalar of size %s bytes and cannot be "
                        "partitioned into a shard of max shard size %s bytes. "
                        "It will be added as an individual shard that exceeds "
                        "the max shard size.",
                        checkpoint_key, total_size, self.max_shard_size)
        large_scalars.append(
            {checkpoint_key: {shardable_tensor.slice_spec: root_tensor}})
        continue

      # Partition tensor and add partitions to shards.
      working_tensor = root_tensor
      working_tensor_var_offset = [0] * root_shape.rank
      working_tensor_shape = root_shape
      working_tensor_size = total_size
      while working_tensor_size > shard_size_remaining:
        part_axis_and_size = self._get_next_partition(
            shard_size_remaining=shard_size_remaining,
            shape=working_tensor_shape,
            dtype_size=dtype_size,
            num_elems=working_tensor_shape.num_elements())

        (remaining_size,
         (remaining_offset, remaining_shape)) = self._add_partition(
             root_shardable_tensor=shardable_tensor,
             dtype_size=dtype_size,
             working_tensor_offset=working_tensor_var_offset,
             part_axis_and_size=part_axis_and_size,
             shard_size_remaining=shard_size_remaining,
             max_shard_size=self.max_shard_size,
             tensors_by_shard=tensors_by_shard,
             large_scalars=large_scalars)

        if remaining_size is None:
          # Tensor partition couldn't fit in remaining shard space. Try again
          # with the next full shard.
          tensors_by_shard.append({})
          shard_size_remaining = self.max_shard_size
        else:
          working_tensor = array_ops.slice(
              root_tensor, begin=remaining_offset, size=remaining_shape)
          working_tensor_var_offset = remaining_offset
          working_tensor_shape = working_tensor.shape
          working_tensor_size = int(math.prod(remaining_shape)) * dtype_size
          shard_size_remaining = remaining_size

      if working_tensor_shape.num_elements() > 0:
        remaining_tensor_slice_spec = variables.Variable.SaveSliceInfo(
            full_name=checkpoint_key,
            full_shape=root_shape,
            var_offset=working_tensor_var_offset,
            var_shape=working_tensor_shape).spec.strip()
        if not tensors_by_shard:
          tensors_by_shard.append({})
        (tensors_by_shard[-1]
         .setdefault(checkpoint_key, {})
         [remaining_tensor_slice_spec]) = working_tensor
      shard_size_remaining -= working_tensor_size

    return tensors_by_shard + large_scalars
