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

from typing import Mapping, Sequence, Optional

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


class EmbeddingReshardCallback(checkpoint_adapter.ReshardCallback):
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
    def pad_value(value, variable_shape, table_shape):
      return array_ops.pad(
          value,
          [
              [0, variable_shape[0] - table_shape[0]],
              [0, variable_shape[1] - table_shape[1]],
          ],
          "CONSTANT",
      )

    _, shard_info = _parse_shard_info_str(shape_and_slice)
    num_sc_per_partition = (
        self._to_shard_layout[0].num_sparse_cores
        // self._to_shard_layout[0].num_partitions
    )

    total_rows = self._to_shard_layout[0].total_rows_per_sparse_core_shard
    sharded_tensors = []
    full_values = {}
    required_shard_offset = shard_info.offset[0]
    for i in range(num_sc_per_partition):
      shard_idx = (required_shard_offset // total_rows) + i
      for table_idx, layout in enumerate(self._to_shard_layout):
        if table_idx not in full_values:
          full_values[table_idx] = pad_value(
              checkpoint_values[table_idx],
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
    return array_ops.concat(sharded_tensors, axis=0)


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
            EmbeddingReshardCallback(
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
    # TODO(b/326644391): First unshard then shard.
    raise NotImplementedError("Changing topology is not implemented yet.")

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
