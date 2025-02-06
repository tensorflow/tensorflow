# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tpu_embedding_v3_checkpoint_adapter."""
from tensorflow.core.tpu.kernels import sparse_core_layout_pb2
from tensorflow.python.framework.constant_op import constant as tf_constant
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v3_checkpoint_adapter


def create_layout(
    tables_name: str,
    stacked_table_name: str,
    num_sparse_cores: int,
    num_partitions: int,
    unsharded_shape: tuple[int, int],
    unsharded_padded_shape: tuple[int, int],
    row_offset: int,
    shard_rotation: int,
    total_rows_per_sparse_core_shard=None,
) -> sparse_core_layout_pb2.SparseCoreTableLayout():
  layout = sparse_core_layout_pb2.SparseCoreTableLayout()
  layout.table_name = tables_name
  layout.stacked_table_name = stacked_table_name
  layout.num_sparse_cores = num_sparse_cores
  layout.num_partitions = num_partitions
  layout.total_rows_per_sparse_core_shard = (
      (unsharded_padded_shape[0] // num_sparse_cores)
      if total_rows_per_sparse_core_shard is None
      else total_rows_per_sparse_core_shard
  )
  layout.unsharded_shape.extend(unsharded_shape)
  layout.unsharded_padded_shape.extend(unsharded_padded_shape)
  layout.sparse_core_shard_row_offset = row_offset
  layout.sparse_core_shard_rotation = shard_rotation
  return layout


class TpuEmbeddingV3CheckpointAdapterTest(test.TestCase):

  def test_adapt_unsharded_to_sharded_simple(self):
    adapter = (
        tpu_embedding_v3_checkpoint_adapter.TpuEmbeddingV3CheckpointAdapter(
            None
        )
    )
    layout = create_layout(
        tables_name="some_feature",
        stacked_table_name="some_feature",
        num_sparse_cores=8,
        num_partitions=2,
        unsharded_shape=(20, 4),
        unsharded_padded_shape=(24, 8),
        row_offset=0,
        shard_rotation=8,
    )
    t = math_ops.range(start=0.0, limit=20.0, delta=1)[
        :, None
    ] * array_ops.ones((20, 4))
    adapter.initialize_reshard_callbacks({"some_feature": layout})
    callback = adapter.get_reshard_callback("some_feature")
    # Check partition index 1 (second parition)
    self.assertAllEqual(
        callback.reshard([t], "128 8 8,16:0,8"),
        tf_constant([
            [2, 2, 2, 2, 0, 0, 0, 0],
            [10, 10, 10, 10, 0, 0, 0, 0],
            [18, 18, 18, 18, 0, 0, 0, 0],
            [3, 3, 3, 3, 0, 0, 0, 0],
            [11, 11, 11, 11, 0, 0, 0, 0],
            [19, 19, 19, 19, 0, 0, 0, 0],
            [4, 4, 4, 4, 0, 0, 0, 0],
            [12, 12, 12, 12, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [5, 5, 5, 5, 0, 0, 0, 0],
            [13, 13, 13, 13, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]),
    )

  def test_adapt_unsharded_to_sharded_stacked(self):
    adapter = (
        tpu_embedding_v3_checkpoint_adapter.TpuEmbeddingV3CheckpointAdapter(
            None
        )
    )
    layouts = {
        "one": create_layout(
            tables_name="one",
            stacked_table_name="one_two",
            num_sparse_cores=8,
            num_partitions=4,
            unsharded_shape=(20, 4),
            unsharded_padded_shape=(24, 8),
            row_offset=0,
            shard_rotation=0,
            total_rows_per_sparse_core_shard=7,
        ),
        "two": create_layout(
            tables_name="two",
            stacked_table_name="one_two",
            num_sparse_cores=8,
            num_partitions=4,
            unsharded_shape=(32, 4),
            unsharded_padded_shape=(32, 8),
            row_offset=3,
            shard_rotation=1,
            total_rows_per_sparse_core_shard=7,
        ),
    }
    one_t = math_ops.range(start=0.0, limit=20.0, delta=1)[
        :, None
    ] * array_ops.ones((20, 4))
    two_t = math_ops.range(start=50.0, limit=82.0, delta=1)[
        :, None
    ] * array_ops.ones((32, 4))
    adapter.initialize_reshard_callbacks(layouts)
    callback = adapter.get_reshard_callback("one")
    self.assertAllEqual(
        callback.reshard([one_t, two_t], "56 8 14,28:0,8"),
        tf_constant([
            # table one shard 2
            [2, 2, 2, 2, 0, 0, 0, 0],
            [10, 10, 10, 10, 0, 0, 0, 0],
            [18, 18, 18, 18, 0, 0, 0, 0],
            # table two shard 2
            [53, 53, 53, 53, 0, 0, 0, 0],
            [61, 61, 61, 61, 0, 0, 0, 0],
            [69, 69, 69, 69, 0, 0, 0, 0],
            [77, 77, 77, 77, 0, 0, 0, 0],
            # table one shard 3
            [3, 3, 3, 3, 0, 0, 0, 0],
            [11, 11, 11, 11, 0, 0, 0, 0],
            [19, 19, 19, 19, 0, 0, 0, 0],
            # table two shard 3
            [54, 54, 54, 54, 0, 0, 0, 0],
            [62, 62, 62, 62, 0, 0, 0, 0],
            [70, 70, 70, 70, 0, 0, 0, 0],
            [78, 78, 78, 78, 0, 0, 0, 0],
        ]),
    )

  def test_adapt_sharded_to_unsharded_simple(self):
    pass

  def test_adapt_sharded_to_unsharded_stacked(self):
    pass

  def test_is_layouts_same_works(self):
    layout = create_layout(
        tables_name="some_feature",
        stacked_table_name="some_feature",
        num_sparse_cores=8,
        num_partitions=8,
        unsharded_shape=(100, 4),
        unsharded_padded_shape=(128, 8),
        row_offset=0,
        shard_rotation=0,
    )
    layouts = sparse_core_layout_pb2.SparseCoreTableLayouts()
    layouts.tables.append(layout)
    adapter = (
        tpu_embedding_v3_checkpoint_adapter.TpuEmbeddingV3CheckpointAdapter(
            layouts
        )
    )
    self.assertTrue(adapter.is_layouts_same({layout.table_name: layout}))
    layout.num_sparse_cores = 3
    self.assertFalse(adapter.is_layouts_same({layout.table_name: layout}))


if __name__ == "__main__":
  test.main()
