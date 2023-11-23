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
"""Utils for Sparsecore Checkpoints."""

import functools
from typing import Any, Dict
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.framework.constant_op import constant as tf_constant
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.trackable import base as trackable_base

SPARSECORE_LAYOUTS_CHECKPOINT_KEY = "_sparse_core_table_layouts"


def unshuffle_from_sc_to_cpu(
    t: tensor.Tensor,
    num_sparse_cores: int,
    offset_in_shard: int,
    size_in_shard: int,
    shard_rotation: int = 0,
) -> tensor.Tensor:
  """Unshuffles the sparse core sharded embedding tables to unsharded.

  This converts an input tensor respresenting stacked and sharded embedding
  table into a specific embedding table variable by using the provided
  metadata about the said table within the stacked, sharded embedding table.
  Args:
    t: The input stacked and sharded embedding table from sparsecore.
    num_sparse_cores: The number of sparsecores, this determines the number of
      shards that are present in the input t.
    offset_in_shard: Offset within a shard where the queried table starts.
    size_in_shard: size (number of rows) of this queried table within each shard
      of the input t.
    shard_rotation: The rotation of this table's shards.

  Returns:
    An embedding table which is part of the stacked embedding table t.
  """
  old_shape = t.shape
  # The width of the table must be a multiple of number of SC devices. The
  # tpu strategy does this round off at training time so we expect the
  # checkpoints value to meet this requirement.
  if t.shape[0] % num_sparse_cores != 0:
    raise ValueError(
        "The dim of table ({}) should be multiple of number of sparse cores"
        " ({})".format(t.shape[1], num_sparse_cores)
    )
  # get shards in the input t
  shards_t = array_ops.reshape(
      t,
      (
          num_sparse_cores,
          t.shape[0] // num_sparse_cores,
          t.shape[1],
      ),
  )
  # From each shard in t, get the part for just the queried table.
  shards = shards_t[:, offset_in_shard : offset_in_shard + size_in_shard, :]
  # This table's shards were rotated by `shard_rotation`, so we need to rotate
  # the same amount in opposite direction
  if shard_rotation:
    shards = manip_ops.roll(shards, -shard_rotation, axis=0)
  # Re-arrange (transpose and reshape) the shards to get the queried embedding
  # table.
  intermediate_tensor = array_ops.transpose(shards, (1, 0, 2))
  new_shape = size_in_shard * num_sparse_cores, old_shape[1]
  return array_ops.reshape(intermediate_tensor, new_shape)


def remove_padding_from_sc(
    value_in_checkpoint: tensor.Tensor, variable_shape: tuple[int, int]
) -> tensor.Tensor:
  """Removes padding, if any, from sparsecore checkpoint.

  Args:
    value_in_checkpoint: input tensor value, usually from checkpoint.
    variable_shape: Expected shape of tensor after removing padding.

  Returns:
    A slice of the input tensor to match the variable_shape if the
    variable shape is a valid slice if the input tensor.
  """
  checkpoint_value_shape = value_in_checkpoint.shape.as_list()
  # If the checkpoint shape is at least the size of the variable, we conclude
  # that the extra rows and cols must be padding.
  is_init_value_padded = all(
      [i >= j for i, j in zip(checkpoint_value_shape, variable_shape)]
  )
  if not is_init_value_padded:
    return value_in_checkpoint
  # checkpoint has padding so we can remove it.
  begin = [0] * len(checkpoint_value_shape)
  return array_ops.slice(value_in_checkpoint, begin=begin, size=variable_shape)


def map_indices_in_shard(
    num_sparse_cores: int,
    offset_in_shard: int,
    shard_rotation: int,
    row_indices: tensor.Tensor,
) -> tuple[tensor.Tensor, tensor.Tensor]:
  """Maps a row of a given table to its sparse core shard and position.

  Maps a given a row index of a logical table and its layout in sparse core,
  returns the index of the shard where the row is placed and its relative
  position within
  that sparse core shard.
  Args:
    num_sparse_cores: The number of sparsecores, this determines the number of
      shards present.
    offset_in_shard: Offset within a shard where the queried table starts.
    shard_rotation: The rotation of this table's shards.
    row_indices: row indices of the embedding table being looked up.

  Returns:
    A Tuple representing shard_index and position of the row in that shard.
  """
  shard_index = (
      (row_indices % num_sparse_cores) + shard_rotation
  ) % num_sparse_cores
  position_in_shard = offset_in_shard + row_indices // num_sparse_cores
  return (shard_index, position_in_shard)


class SparseCoreLayoutsTrackable(trackable_base.Trackable):
  """Trackable for sparsecore layouts used in training."""

  def __init__(self, proto_str_tensor: tensor.Tensor):
    self.value = proto_str_tensor

  def _serialize_to_tensors(self) -> Dict[str, tensor.Tensor]:
    return {trackable_base.VARIABLE_VALUE_KEY: self.value}

  def _restore_from_tensors(
      self, restored_tensors: Dict[str, tensor.Tensor]
  ) -> None:
    self.value = restored_tensors[trackable_base.VARIABLE_VALUE_KEY]


class SparseCoreStackedTableTrackable(trackable_base.Trackable):
  """Trackable for stacked tables generated from sparse core."""

  def __init__(self, stacked_layouts, table_to_config):
    self.vars = {}
    self._stacked_layouts = stacked_layouts
    for table_layout in stacked_layouts:
      variable_shape = tuple(table_layout.unsharded_shape)
      self.vars[table_layout.table_name] = tf_variables.Variable(
          name=table_layout.table_name,
          initial_value=functools.partial(
              table_to_config[table_layout.table_name].initializer,
              variable_shape,
              dtype=dtypes.float32,
          ),
          shape=variable_shape,
          dtype=dtypes.float32,
      )

  def _serialize_to_tensors(self) -> Any:
    return {
        # We need to export some variable here for restore to pick
        # the checkpoint key the actual value is not important so 0 works
        trackable_base.VARIABLE_VALUE_KEY: tf_constant(
            0.0, dtype=dtypes.float32
        ),
    }

  def _restore_from_tensors(self, restored_tensors: Dict[str, tensor.Tensor]):
    def fn(restored_tensors):
      value_from_checkpoint = restored_tensors[
          trackable_base.VARIABLE_VALUE_KEY
      ]
      # Do unsharding to get the individual tables from the stacked table in
      # checkpoint
      for layout in self._stacked_layouts:
        variable_shape = (
            layout.unsharded_shape[0],
            layout.unsharded_shape[1],
        )
        t_part = unshuffle_from_sc_to_cpu(
            t=value_from_checkpoint,
            num_sparse_cores=layout.num_sparse_cores,
            offset_in_shard=layout.sparse_core_shard_row_offset,
            size_in_shard=(
                layout.unsharded_padded_shape[0] // layout.num_sparse_cores
            ),
            shard_rotation=layout.sparse_core_shard_rotation,
        )
        t_part = remove_padding_from_sc(t_part, variable_shape)
        self.vars[layout.table_name].assign(t_part)

    return fn(restored_tensors)

  def get_var(self, name: str) -> tf_variables.Variable:
    return self.vars[name]

  def get_vars(self) -> Dict[str, tf_variables.Variable]:
    return self.vars

  def __repr__(self):
    return "SparseCoreStackedTableTrackable({})".format(self.vars.keys())
