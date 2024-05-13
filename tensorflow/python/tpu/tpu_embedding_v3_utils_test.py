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
"""Test for tpu_embedding_v3_utils."""

import collections

from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.framework.constant_op import constant as tf_constant
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v3_utils as v3_utils

TestTable = collections.namedtuple("Table", ["vocab", "dim", "shift"])


def create_test_table_shards(
    table: TestTable, num_sc_shards: int, table_data_start=0
):
  t = array_ops.reshape(
      math_ops.range(
          start=table_data_start,
          delta=1,
          limit=table_data_start + table.vocab * table.dim,
      ),
      (table.vocab, table.dim),
  )
  shards = [t[i::num_sc_shards, :] for i in range(num_sc_shards)]
  if table.shift:
    shards = collections.deque(shards)
    shards.rotate(table.shift)
    return (t, list(shards))
  else:
    return (t, shards)


class TpuEmbeddingV3UtilsTest(test.TestCase, parameterized.TestCase):

  def test_unpadding(self):
    self.assertAllEqual(
        v3_utils.remove_padding_from_sc(
            array_ops.ones((4, 5)), variable_shape=(3, 2)
        ),
        array_ops.ones((3, 2)),
    )
    x = array_ops.reshape(math_ops.range(12), (3, 4))
    self.assertAllEqual(
        v3_utils.remove_padding_from_sc(x, variable_shape=(2, 2)),
        tf_constant([[0, 1], [4, 5]]),
    )
    self.assertAllEqual(
        v3_utils.remove_padding_from_sc(x, variable_shape=(3, 5)),
        x,
    )

  @parameterized.named_parameters(
      ("one", 8, 4, 4), ("two", 27, 6, 3), ("three", 128, 8, 4)
  )
  def test_unshuffle_one_table_basic(self, vocab, dim, num_sc):
    # input vocab should be multiple of num_sc
    self.assertEqual(vocab % num_sc, 0)
    x, shards = create_test_table_shards(
        TestTable(vocab=vocab, dim=dim, shift=0), num_sc
    )
    x_sharded = array_ops.concat(shards, axis=0)
    self.assertAllEqual(
        v3_utils.unshuffle_from_sc_to_cpu(
            t=x_sharded,
            num_sparse_cores=num_sc,
            offset_in_shard=0,
            size_in_shard=vocab // num_sc,
            shard_rotation=0,
        ),
        x,
    )

  def test_unshuffle_stacking_basic(self):
    num_sc = 4
    ta = TestTable(vocab=12, dim=4, shift=0)
    tb = TestTable(vocab=32, dim=4, shift=1)
    x, x_shards = create_test_table_shards(ta, num_sc)
    y, y_shards = create_test_table_shards(tb, num_sc)
    stacked_shards = [
        array_ops.concat([i, j], axis=0) for i, j in zip(x_shards, y_shards)
    ]
    stacked = array_ops.concat(stacked_shards, axis=0)
    self.assertAllEqual(
        v3_utils.unshuffle_from_sc_to_cpu(
            t=stacked,
            num_sparse_cores=num_sc,
            offset_in_shard=0,
            size_in_shard=ta.vocab // num_sc,
            shard_rotation=ta.shift,
        ),
        x,
    )
    self.assertAllEqual(
        v3_utils.unshuffle_from_sc_to_cpu(
            t=stacked,
            num_sparse_cores=num_sc,
            offset_in_shard=ta.vocab // num_sc,
            size_in_shard=tb.vocab // num_sc,
            shard_rotation=tb.shift,
        ),
        y,
    )

  def test_unshuffle_stacking_many_tables(self):
    num_sc = 4
    tables = [
        TestTable(vocab=12, dim=4, shift=0),
        TestTable(vocab=32, dim=4, shift=1),
        TestTable(vocab=32, dim=4, shift=2),
        TestTable(vocab=32, dim=4, shift=3),
        TestTable(vocab=32, dim=4, shift=4),
        TestTable(vocab=32, dim=4, shift=5),
    ]
    u, u_shards = create_test_table_shards(tables[0], num_sc)
    v, v_shards = create_test_table_shards(tables[1], num_sc)
    w, w_shards = create_test_table_shards(tables[2], num_sc)
    x, x_shards = create_test_table_shards(tables[3], num_sc)
    y, y_shards = create_test_table_shards(tables[4], num_sc)
    z, z_shards = create_test_table_shards(tables[5], num_sc)
    stacked_shards = [
        array_ops.concat([i, j, k, l, m, n], axis=0)
        for i, j, k, l, m, n in zip(
            u_shards, v_shards, w_shards, x_shards, y_shards, z_shards
        )
    ]
    stacked = array_ops.concat(stacked_shards, axis=0)
    self.assertAllEqual(
        v3_utils.unshuffle_from_sc_to_cpu(
            t=stacked,
            num_sparse_cores=num_sc,
            offset_in_shard=0,
            size_in_shard=tables[0].vocab // num_sc,
            shard_rotation=tables[0].shift,
        ),
        u,
    )
    self.assertAllEqual(
        v3_utils.unshuffle_from_sc_to_cpu(
            t=stacked,
            num_sparse_cores=num_sc,
            offset_in_shard=tables[0].vocab // num_sc,
            size_in_shard=tables[1].vocab // num_sc,
            shard_rotation=tables[1].shift,
        ),
        v,
    )
    self.assertAllEqual(
        v3_utils.unshuffle_from_sc_to_cpu(
            t=stacked,
            num_sparse_cores=num_sc,
            offset_in_shard=(tables[0].vocab + tables[1].vocab) // num_sc,
            size_in_shard=tables[2].vocab // num_sc,
            shard_rotation=tables[2].shift,
        ),
        w,
    )
    self.assertAllEqual(
        v3_utils.unshuffle_from_sc_to_cpu(
            t=stacked,
            num_sparse_cores=num_sc,
            offset_in_shard=(
                tables[0].vocab + tables[1].vocab + tables[2].vocab
            )
            // num_sc,
            size_in_shard=tables[3].vocab // num_sc,
            shard_rotation=tables[3].shift,
        ),
        x,
    )
    self.assertAllEqual(
        v3_utils.unshuffle_from_sc_to_cpu(
            t=stacked,
            num_sparse_cores=num_sc,
            offset_in_shard=(
                tables[0].vocab
                + tables[1].vocab
                + tables[2].vocab
                + tables[3].vocab
            )
            // num_sc,
            size_in_shard=tables[4].vocab // num_sc,
            shard_rotation=tables[4].shift,
        ),
        y,
    )
    self.assertAllEqual(
        v3_utils.unshuffle_from_sc_to_cpu(
            t=stacked,
            num_sparse_cores=num_sc,
            offset_in_shard=(
                tables[0].vocab
                + tables[1].vocab
                + tables[2].vocab
                + tables[3].vocab
                + tables[4].vocab
            )
            // num_sc,
            size_in_shard=tables[5].vocab // num_sc,
            shard_rotation=tables[5].shift,
        ),
        z,
    )

  def test_index_mapping_one_table(self):
    num_sc = 4
    x, shards = create_test_table_shards(
        TestTable(vocab=12, dim=4, shift=0), num_sc
    )
    indices = tf_constant([1, 2, 5, 7, 9])
    shard_idx, position_in_shard = v3_utils.map_indices_in_shard(
        num_sparse_cores=num_sc,
        offset_in_shard=0,
        shard_rotation=0,
        row_indices=indices,
    )
    self.assertAllEqual(
        shard_idx,
        indices % num_sc,
    )
    self.assertAllEqual(
        [x[i] for i in indices],
        [shards[j][k] for j, k in zip(shard_idx, position_in_shard)],
    )

  def test_index_mapping_stacked_tables(self):
    num_sc = 4
    ta = TestTable(vocab=12, dim=4, shift=0)
    tb = TestTable(vocab=32, dim=4, shift=1)
    x, x_shards = create_test_table_shards(ta, num_sc)
    y, y_shards = create_test_table_shards(tb, num_sc, table_data_start=100)
    stacked_shards = [
        array_ops.concat([i, j], axis=0) for i, j in zip(x_shards, y_shards)
    ]
    indices_ta = tf_constant([1, 2, 7, 9, 11])
    shard_idx, position_in_shard = v3_utils.map_indices_in_shard(
        num_sparse_cores=num_sc,
        offset_in_shard=0,
        shard_rotation=ta.shift,
        row_indices=indices_ta,
    )
    self.assertAllEqual(
        [x[i] for i in indices_ta],
        [stacked_shards[j][k] for j, k in zip(shard_idx, position_in_shard)],
    )
    indices_tb = tf_constant([1, 2, 7, 9, 15, 27])
    shard_idx, position_in_shard = v3_utils.map_indices_in_shard(
        num_sparse_cores=num_sc,
        offset_in_shard=ta.vocab // num_sc,
        shard_rotation=tb.shift,
        row_indices=indices_tb,
    )
    self.assertAllEqual(
        [y[i] for i in indices_tb],
        [stacked_shards[j][k] for j, k in zip(shard_idx, position_in_shard)],
    )


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
