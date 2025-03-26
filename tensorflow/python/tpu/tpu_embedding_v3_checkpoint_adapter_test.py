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
from tensorflow.python.compat import v2_compat
from tensorflow.python.framework import dtypes
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
        callback.reshard([t], "128 8 8,12:0,8"),
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
    }
    one_t = math_ops.range(start=0.0, limit=20.0, delta=1)[
        :, None
    ] * array_ops.ones((20, 4))
    two_t = math_ops.range(start=50.0, limit=82.0, delta=1)[
        :, None
    ] * array_ops.ones((32, 4))
    adapter.initialize_reshard_callbacks(layouts)
    callback = adapter.get_reshard_callback("one")
    self.assertEqual(callback.object_name(), "one_two")
    updated_keys, updated_slices = callback.update_restore_inputs(
        "path/to/embedding/one/in/checkpoint", "56 8 14,28:0,8"
    )
    self.assertAllEqual(
        updated_keys,
        [
            "path/to/embedding/one/in/checkpoint",
            "path/to/embedding/two/in/checkpoint",
        ],
    )
    self.assertAllEqual(
        updated_slices,
        ["20 4 0,20:0,4", "32 4 0,32:0,4"],
    )
    actual = callback.reshard([one_t, two_t], "56 8 14,14:0,8")
    self.assertAllEqual(
        actual,
        tf_constant([
            # table one shard 2
            [2, 2, 2, 2, 0, 0, 0, 0],
            [10, 10, 10, 10, 0, 0, 0, 0],
            [18, 18, 18, 18, 0, 0, 0, 0],
            # table two shard 2
            [51, 51, 51, 51, 0, 0, 0, 0],
            [59, 59, 59, 59, 0, 0, 0, 0],
            [67, 67, 67, 67, 0, 0, 0, 0],
            [75, 75, 75, 75, 0, 0, 0, 0],
            # table one shard 3
            [3, 3, 3, 3, 0, 0, 0, 0],
            [11, 11, 11, 11, 0, 0, 0, 0],
            [19, 19, 19, 19, 0, 0, 0, 0],
            # table two shard 3
            [52, 52, 52, 52, 0, 0, 0, 0],
            [60, 60, 60, 60, 0, 0, 0, 0],
            [68, 68, 68, 68, 0, 0, 0, 0],
            [76, 76, 76, 76, 0, 0, 0, 0],
        ]),
    )
    # Check that full resharding works.
    actual_full = callback.reshard([one_t, two_t], "56 8 0,56:0,8")
    self.assertAllEqual(
        actual_full,
        tf_constant(
            [
                # table one shard 0
                [0, 0, 0, 0, 0, 0, 0, 0],
                [8, 8, 8, 8, 0, 0, 0, 0],
                [16, 16, 16, 16, 0, 0, 0, 0],
                # table two shard 0
                [57, 57, 57, 57, 0, 0, 0, 0],
                [65, 65, 65, 65, 0, 0, 0, 0],
                [73, 73, 73, 73, 0, 0, 0, 0],
                [81, 81, 81, 81, 0, 0, 0, 0],
                # table one shard 1
                [1, 1, 1, 1, 0, 0, 0, 0],
                [9, 9, 9, 9, 0, 0, 0, 0],
                [17, 17, 17, 17, 0, 0, 0, 0],
                # table two shard 1
                [50, 50, 50, 50, 0, 0, 0, 0],
                [58, 58, 58, 58, 0, 0, 0, 0],
                [66, 66, 66, 66, 0, 0, 0, 0],
                [74, 74, 74, 74, 0, 0, 0, 0],
                # table one shard 2
                [2, 2, 2, 2, 0, 0, 0, 0],
                [10, 10, 10, 10, 0, 0, 0, 0],
                [18, 18, 18, 18, 0, 0, 0, 0],
                # table two shard 2
                [51, 51, 51, 51, 0, 0, 0, 0],
                [59, 59, 59, 59, 0, 0, 0, 0],
                [67, 67, 67, 67, 0, 0, 0, 0],
                [75, 75, 75, 75, 0, 0, 0, 0],
                # table one shard 3
                [3, 3, 3, 3, 0, 0, 0, 0],
                [11, 11, 11, 11, 0, 0, 0, 0],
                [19, 19, 19, 19, 0, 0, 0, 0],
                # table two shard 3
                [52, 52, 52, 52, 0, 0, 0, 0],
                [60, 60, 60, 60, 0, 0, 0, 0],
                [68, 68, 68, 68, 0, 0, 0, 0],
                [76, 76, 76, 76, 0, 0, 0, 0],
                # table one shard 4
                [4, 4, 4, 4, 0, 0, 0, 0],
                [12, 12, 12, 12, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                # table two shard 4
                [53, 53, 53, 53, 0, 0, 0, 0],
                [61, 61, 61, 61, 0, 0, 0, 0],
                [69, 69, 69, 69, 0, 0, 0, 0],
                [77, 77, 77, 77, 0, 0, 0, 0],
                # table one shard 5
                [5, 5, 5, 5, 0, 0, 0, 0],
                [13, 13, 13, 13, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                # table two shard 5
                [54, 54, 54, 54, 0, 0, 0, 0],
                [62, 62, 62, 62, 0, 0, 0, 0],
                [70, 70, 70, 70, 0, 0, 0, 0],
                [78, 78, 78, 78, 0, 0, 0, 0],
                # table one shard 6
                [6, 6, 6, 6, 0, 0, 0, 0],
                [14, 14, 14, 14, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                # table two shard 6
                [55, 55, 55, 55, 0, 0, 0, 0],
                [63, 63, 63, 63, 0, 0, 0, 0],
                [71, 71, 71, 71, 0, 0, 0, 0],
                [79, 79, 79, 79, 0, 0, 0, 0],
                # table one shard 7
                [7, 7, 7, 7, 0, 0, 0, 0],
                [15, 15, 15, 15, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                # table two shard 7
                [56, 56, 56, 56, 0, 0, 0, 0],
                [64, 64, 64, 64, 0, 0, 0, 0],
                [72, 72, 72, 72, 0, 0, 0, 0],
                [80, 80, 80, 80, 0, 0, 0, 0],
            ],
            dtype=dtypes.float32,
        ),
    )
    self.assertAllEqual(callback._checkpoint_local_names, ["one", "two"])
    self.assertAllEqual(
        [l.table_name for l in callback._to_shard_layout],
        ["one", "two"],
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
  v2_compat.enable_v2_behavior()
  test.main()
