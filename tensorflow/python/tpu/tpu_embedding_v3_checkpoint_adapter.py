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
"""Checkpoint adapter for TPUEmbedding."""

import collections
import time
from typing import Mapping, Optional, Sequence

from absl import logging

from tensorflow.core.tpu.kernels import sparse_core_layout_pb2
from tensorflow.python.checkpoint import checkpoint_adapter
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.tpu import tpu_embedding_v3_utils
from tensorflow.python.trackable import base as trackable_base
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.util.protobuf import compare


def _parse_shard_info_str(
    spec: str,
) -> tuple[list[int], trackable_base.ShardInfo]:
  """Parses shape and shard_info string."""

  shape = [int(x) for x in spec.split()[:-1]]
  slices = spec.split()[-1].split(":")
  offset = [int(x.split(",")[0]) for x in slices]
  shard_shape = [int(x.split(",")[1]) for x in slices]
  return shape, trackable_base.ShardInfo(offset=offset, shape=shard_shape)


def _shard_info_str(shape, shard_info) -> str:
  """Created shape and shard_info string."""

  full_shape_str = " ".join("%d" % d for d in shape) + " "
  slice_spec = ":".join(
      "%d,%d" % (o, s) for o, s in zip(shard_info.offset, shard_info.shape)
  )
  return full_shape_str + slice_spec


def _shard_from_cpu_to_sc(
    feature_values: tensor.Tensor,
    shape_and_slice: str,
    to_shard_layout: Sequence[sparse_core_layout_pb2.SparseCoreTableLayout],
) -> tensor.Tensor:
  """Shards the feature tables from CPU to SparseCore."""

  def pad_value(value, variable_shape, table_shape):
    return array_ops.pad(
        value,
        [
            [0, variable_shape[0] - table_shape[0]],
            [0, variable_shape[1] - table_shape[1]],
        ],
        "CONSTANT",
    )

  var_full_shape, shard_info = _parse_shard_info_str(shape_and_slice)
  if shard_info.offset > var_full_shape:
    raise ValueError(
        "Invalid shard offset: {}. Offset should be less than the full shape"
        " of the variable: {}".format(
            shard_info.offset,
            var_full_shape,
        )
    )
  num_sc_per_partition = (
      to_shard_layout[0].num_sparse_cores // to_shard_layout[0].num_partitions
  )

  total_rows_per_sc = to_shard_layout[0].total_rows_per_sparse_core_shard
  total_rows_per_partition = total_rows_per_sc * num_sc_per_partition
  full_values = {}
  if (shard_info.shape[0] % total_rows_per_partition) != 0:
    raise ValueError(
        "Invalid shard shape: {}. Number of rows in input shard slice should"
        " be multiple of number of rows in a partition({})".format(
            shard_info.shape,
            total_rows_per_partition,
        )
    )
  # From the shard info, get the row offsets corresponding to the slice
  # being looked up.
  required_shard_offsets = range(
      shard_info.offset[0],
      shard_info.offset[0] + shard_info.shape[0],
      total_rows_per_partition,
  )
  output_shards = []
  for required_shard_offset in required_shard_offsets:
    sharded_tensors = []
    for i in range(num_sc_per_partition):
      shard_idx = (required_shard_offset // total_rows_per_sc) + i
      for table_idx, layout in enumerate(to_shard_layout):
        if table_idx not in full_values:
          full_values[table_idx] = pad_value(
              feature_values[table_idx],
              layout.unsharded_padded_shape,
              layout.unsharded_shape,
          )

        table_value = full_values[table_idx]
        # Apply rotation to get this table's shard index
        table_shard_offset = (
            shard_idx
            + (layout.num_sparse_cores - layout.sparse_core_shard_rotation)
        ) % layout.num_sparse_cores
        sharded_tensors.append(
            table_value[
                table_shard_offset :: layout.num_sparse_cores,
                :,
            ]
        )
    output_shards.append(array_ops.concat(sharded_tensors, axis=0))
    logging.vlog(
        1,
        "_shard_from_cpu_to_sc: last output_shards.shape: %s",
        output_shards[-1].shape,
    )
  return array_ops.concat(output_shards, axis=0)


def _unshard_from_sc_to_cpu(
    stacked_table: tensor.Tensor,
    from_shard_layouts: Sequence[sparse_core_layout_pb2.SparseCoreTableLayout],
) -> Sequence[tensor.Tensor]:
  """Undo the shard the feature tables into SparseCore stacked table.

  Args:
    stacked_table: The value of a SparseCore stacked and sharded table.
    from_shard_layouts: The target layouts for the target hardware.

  Returns:
    The unsharded feature tables.
  """
  logging.vlog(
      1,
      "To unshuffle_from_sc_to_cpu on stacked_table.shape: %s",
      stacked_table[0].shape,
  )
  ret_tensors = []

  for layout in from_shard_layouts:
    padded_table = tpu_embedding_v3_utils.unshuffle_from_sc_to_cpu(
        stacked_table[0],
        num_sparse_cores=layout.num_sparse_cores,
        offset_in_shard=layout.sparse_core_shard_row_offset,
        size_in_shard=layout.unsharded_padded_shape[0]
        // layout.num_sparse_cores,
        shard_rotation=layout.sparse_core_shard_rotation,
    )

    orig_table = tpu_embedding_v3_utils.remove_padding_from_sc(
        padded_table, layout.unsharded_shape
    )

    logging.vlog(
        1, "orig_tensors.shape[%s]: %s", layout.table_name, orig_table.shape
    )
    ret_tensors.append(orig_table)

  return ret_tensors


class EmbeddingUnshardToShardCallback(checkpoint_adapter.ReshardCallback):
  """Reshard callback for embeddings."""

  def __init__(
      self,
      object_local_name: str,
      checkpoint_local_names: Sequence[str],
      to_shard_layout: Optional[
          Sequence[sparse_core_layout_pb2.SparseCoreTableLayout]
      ] = None,
      to_unshard_layout: Optional[
          Sequence[sparse_core_layout_pb2.SparseCoreTableLayout]
      ] = None,
  ):
    """Initializes  Reshard callback.

    Args:
      object_local_name:  The local name of the object being restored.
      checkpoint_local_names: The local names of the checkpoint positions that
        need to be read.
      to_shard_layout: (Optional) Target layouts as specified in the embedding
        being restored.
      to_unshard_layout: (Optional) Layouts as stored in checkpoint being
        restored from.
    """
    self._object_local_name = object_local_name
    self._checkpoint_local_names = checkpoint_local_names
    self._to_shard_layout = to_shard_layout
    self._to_unshard_layout = to_unshard_layout
    self._main_checkpoint_name = checkpoint_local_names[0]

  def object_name(self) -> str:
    return self._object_local_name

  def update_restore_inputs(
      self, checkpoint_key: str, shape_and_slice_spec: str
  ) -> tuple[Sequence[str], Sequence[str]]:
    """Updates checkpoint key and slice spec acorrding to the resharding plan.

    Args:
      checkpoint_key: The input checkpoint key to be read.
      shape_and_slice_spec: The shape and slice spec of the checkpoint key to be
        read.

    Returns:
      A tuple of (keys, slices) that should be passed to restore_v2 inorder to
      reshard according to the resharding plan. The restored tensors from
      restore_v2 op will usually be passed to reshard method of this class to
      get the final resharded value.
    """
    keys = []
    slices = []
    # TODO(b/398016624): Make this a vlog this log after bug is fixed.
    logging.info(
        "Updating restore v2 inputs for %s: %s",
        checkpoint_key,
        shape_and_slice_spec,
    )
    for i, layout in enumerate(self._to_shard_layout):
      sub_checkpoint_key = checkpoint_key.replace(
          self._main_checkpoint_name, self._checkpoint_local_names[i]
      )
      # For resharding later, we need to read the full value here.
      # TODO(b/398016624): Make this a vlog this log after bug is fixed.
      logging.info(
          "Will read sub key %s: %s",
          sub_checkpoint_key,
          layout.unsharded_shape,
      )
      keys.append(sub_checkpoint_key)
      slices.append(
          _shard_info_str(
              layout.unsharded_shape,
              trackable_base.ShardInfo(
                  offset=[0, 0], shape=layout.unsharded_shape
              ),
          )
      )
    return (keys, slices)

  def reshard(
      self, checkpoint_values: tensor.Tensor, shape_and_slice: str
  ) -> tensor.Tensor:
    """Reshards the checkpoint values according to the resharding plan.

    Args:
      checkpoint_values: The checkpoint values to be resharded.
      shape_and_slice: The shape and slice spec to be returned after resharding.

    Returns:
      The resharded tensor slice.
    """
    return _shard_from_cpu_to_sc(
        checkpoint_values, shape_and_slice, self._to_shard_layout
    )


class EmbeddingReshardCallback(checkpoint_adapter.ReshardCallback):
  """Reshard callback for embeddings."""

  def __init__(
      self,
      object_local_name: str,
      from_shard_layouts: Sequence[
          sparse_core_layout_pb2.SparseCoreTableLayout
      ],  # table name to layout
      to_shard_layouts: Sequence[
          sparse_core_layout_pb2.SparseCoreTableLayout
      ],  # table name to layout
  ):
    """Initializes  Reshard callback.

    Args:
      object_local_name:  The local name of the object being restored.
      from_shard_layouts: layouts as in checkpoint being restored from.
      to_shard_layouts: target layouts as specified in the embedding being
        restored.
    """
    logging.info("Creating EmbeddingReshardCallback for %s", object_local_name)
    self._object_local_name = object_local_name
    self._from_shard_layouts = from_shard_layouts
    self._to_shard_layouts = to_shard_layouts

  def object_name(self) -> str:
    return self._object_local_name

  def update_restore_inputs(
      self, checkpoint_key: str, shape_and_slice_spec: str
  ) -> tuple[Sequence[str], Sequence[str]]:
    """Return the full shape of the stacked that is passed into restore_v2.

    This shape information is required by the restore_v2 process to ensure it
    loads the complete tensor from the checkpoint. The full tensor is required
    to perform resharding operations.

    Args:
      checkpoint_key: The input checkpoint key to be read.
      shape_and_slice_spec: The shape and slice spec of the checkpoint key to be
        read.

    Returns:
      A tuple of (keys, slices) that should be passed to restore_v2 in order to
      reshard according to the resharding plan. The restored tensors from
      restore_v2 op will usually be passed to reshard method of this class to
      get the final resharded value.
    """
    logging.vlog(
        1,
        "Updating restore v2 inputs for %s[%s]: %s",
        checkpoint_key,
        self._object_local_name,
        shape_and_slice_spec,
    )

    slices = []

    # use the first layout get the full shape of the stacked table
    first_layout = self._from_shard_layouts[0]
    full_vocab_size = (
        first_layout.total_rows_per_sparse_core_shard
        * first_layout.num_sparse_cores
    )
    stack_dim = first_layout.unsharded_padded_shape[1]
    full_shape = [full_vocab_size, stack_dim]
    logging.vlog(
        1,
        "Read checkpoint_key %s: %s",
        checkpoint_key,
        full_shape,
    )

    slices.append(
        _shard_info_str(
            full_shape,
            trackable_base.ShardInfo(offset=[0, 0], shape=full_shape),
        )
    )
    return ([checkpoint_key], slices)

  def reshard(
      self, checkpoint_values: tensor.Tensor, shape_and_slice: str
  ) -> tensor.Tensor:
    # unshard
    stime = time.time()
    logging.vlog(
        1,
        "EmbeddingReshardCallback: starting to reshard [%s]",
        self._object_local_name,
    )
    unsharded_tensors = _unshard_from_sc_to_cpu(
        checkpoint_values, self._from_shard_layouts
    )

    ret = _shard_from_cpu_to_sc(
        unsharded_tensors, shape_and_slice, self._to_shard_layouts
    )

    etime = time.time()
    logging.info(
        "EmbeddingReshardCallback: reshard [%s] took %s",
        self._object_local_name,
        etime - stime,
    )
    return ret


def _reorg_layouts(
    layouts: Sequence[sparse_core_layout_pb2.SparseCoreTableLayout],
) -> Mapping[str, Sequence[sparse_core_layout_pb2.SparseCoreTableLayout]]:
  """Reorg the layouts to be in the order of the logical table."""
  stacked_name_to_table_names = collections.defaultdict(list)
  for layout in layouts:
    stacked_name_to_table_names[layout.stacked_table_name].append(layout)
  for stacked_name in stacked_name_to_table_names.keys():
    sorted_layouts = sorted(
        stacked_name_to_table_names[stacked_name],
        key=lambda layout: layout.sparse_core_shard_row_offset,
    )
    stacked_name_to_table_names[stacked_name] = sorted_layouts

  return stacked_name_to_table_names


class TpuEmbeddingV3CheckpointAdapter(
    checkpoint_adapter.AbstractCheckpointAdapter
):
  """Adapter for TPU Embedding V3 to handle checkpoint resharding."""

  def __init__(
      self,
      layouts: Optional[sparse_core_layout_pb2.SparseCoreTableLayouts] = None,
  ):
    """An adapter for TPUEmbeddingV3 checkpoints.

    Constructs an adapter for TPUEmbeddingV3 to handle layout changes. between
    checkpoint values and embedding object being restored.

    Args:
     layouts: The target layouts required.
    """
    self._checkpoint_layouts = {}
    self._checkpoint_to_reshard_callback = {}
    if layouts:
      for layout in layouts.tables:
        self._checkpoint_layouts[layout.table_name] = layout

  @classmethod
  def create_from_checkpoint(cls, save_path: str):
    reader = py_checkpoint_reader.NewCheckpointReader(save_path)
    sparsecore_layouts_str = None
    for name in reader.get_variable_to_dtype_map():
      if tpu_embedding_v3_utils.SPARSECORE_LAYOUTS_CHECKPOINT_KEY in name:
        sparsecore_layouts_str = reader.get_tensor(name)
        break
    if sparsecore_layouts_str is None:
      return cls(None)
    layouts = sparse_core_layout_pb2.SparseCoreTableLayouts()
    layouts.ParseFromString(sparsecore_layouts_str)
    logging.info("Loaded layouts from checkpoint: %s", layouts)
    return cls(layouts)

  def initialize_reshard_callbacks(
      self,
      embedding_layouts: Optional[
          Mapping[str, sparse_core_layout_pb2.SparseCoreTableLayout]
      ] = None,
  ):
    if not self._checkpoint_layouts and embedding_layouts:
      # From Unsharded to Sharded
      stacked_name_to_table_names = collections.defaultdict(list)
      for layout in embedding_layouts.values():
        stacked_name_to_table_names[layout.stacked_table_name].append(layout)
      for stacked_name, layouts in stacked_name_to_table_names.items():
        # Make the first table name as the key for checkpoint position
        # The sorting here is by the position of the logical table in the shard
        sorted_layouts = sorted(
            layouts, key=lambda layout: layout.sparse_core_shard_row_offset
        )
        logging.info("Creating resharding plan for %s", stacked_name)
        self._checkpoint_to_reshard_callback[sorted_layouts[0].table_name] = (
            EmbeddingUnshardToShardCallback(
                stacked_name,
                [l.table_name for l in sorted_layouts],
                sorted_layouts,
                None,
            )
        )
      return
    if not embedding_layouts:
      # TODO(b/326644306): From sharded to unsharded
      raise NotImplementedError("Sharded to Unsharded is not implemented yet.")
    # Reshard to different SC Layout
    from_layouts = _reorg_layouts(list(self._checkpoint_layouts.values()))
    to_layouts = _reorg_layouts(list(embedding_layouts.values()))
    for stacked_name in from_layouts.keys():
      logging.info("Creating resharding plan for %s", stacked_name)
      self._checkpoint_to_reshard_callback[stacked_name] = (
          EmbeddingReshardCallback(
              object_local_name=stacked_name,
              from_shard_layouts=from_layouts[stacked_name],
              to_shard_layouts=to_layouts[stacked_name],
          )
      )

  def is_layouts_same(self, embedding_layouts) -> bool:
    """Returns True if the all the embedding and checkpoint layouts are the same.

    Args:
      embedding_layouts: dict of layouts for embedding tables.

    Raises: ValueError if the embedding layouts and checkpoint layouts do not
      have the same keys.
    Returns: Bool representing if the embedding layouts match the layouts in
      checkpoint.
    """
    if self._checkpoint_layouts.keys() != embedding_layouts.keys():
      raise ValueError(
          "Layouts in checkpoint and embedding must have the same keys. found"
          " {} and {}".format(
              self._checkpoint_layouts.keys(), embedding_layouts.keys()
          )
      )

    for key, layout in self._checkpoint_layouts.items():
      if not compare.ProtoEq(layout, embedding_layouts[key]):
        logging.info(
            "Layouts do not match for %s this will require resharding; %s"
            " vs %s",
            key,
            layout,
            embedding_layouts[key],
        )
        return False
    return True

  def is_applicable(self, trackable: trackable_base.Trackable) -> bool:
    # issubclass(trackable, TPUEmbeddingBase) adds circular deps, hence using
    #  a workaround to select the applicable embedding implementations.
    allowed_class_names = [".TPUEmbeddingV2Plus", ".TPUEmbeddingV2"]
    if not any(x in str(type(trackable)) for x in allowed_class_names):
      return False
    embedding_layouts = None
    if hasattr(trackable, "embedding_layouts"):
      embedding_layouts = trackable.embedding_layouts
    # Neither checkpoint not target embedding has layout, no resharding needed.
    if not self._checkpoint_layouts and not embedding_layouts:
      logging.info("No resharding needed, no layouts")
      return False
    # Only if both checkpoint and embedding have layouts and they match,
    # no resharding needed.
    if (
        self._checkpoint_layouts
        and embedding_layouts
        and self.is_layouts_same(embedding_layouts)
    ):
      logging.info("No resharding needed; layouts match")
      return False
    # Else we need to reshard.
    self.initialize_reshard_callbacks(embedding_layouts)
    return True

  def get_reshard_callback(
      self, name: str
  ) -> Optional[checkpoint_adapter.ReshardCallback]:
    if name in self._checkpoint_to_reshard_callback:
      return self._checkpoint_to_reshard_callback[name]
    # Check if this is slot variable
    var_name = name.split("/")[0]
    if var_name in self._checkpoint_to_reshard_callback:
      return self._checkpoint_to_reshard_callback[var_name]
    return None
