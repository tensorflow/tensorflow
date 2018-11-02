# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for sharded_mutable_dense_hashtable.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.linear_optimizer.python.ops.sharded_mutable_dense_hashtable import ShardedMutableDenseHashTable
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.platform import googletest


class ShardedMutableDenseHashTableTest(TensorFlowTestCase):
  """Tests for the ShardedMutableHashTable class."""

  def testShardedMutableHashTable(self):
    for num_shards in [1, 3, 10]:
      with self.cached_session():
        default_val = -1
        empty_key = 0
        deleted_key = -1
        keys = constant_op.constant([11, 12, 13], dtypes.int64)
        values = constant_op.constant([0, 1, 2], dtypes.int64)
        table = ShardedMutableDenseHashTable(
            dtypes.int64,
            dtypes.int64,
            default_val,
            empty_key,
            deleted_key,
            num_shards=num_shards)
        self.assertAllEqual(0, table.size().eval())

        table.insert(keys, values).run()
        self.assertAllEqual(3, table.size().eval())

        input_string = constant_op.constant([11, 12, 14], dtypes.int64)
        output = table.lookup(input_string)
        self.assertAllEqual([3], output.get_shape())
        self.assertAllEqual([0, 1, -1], output.eval())

  def testShardedMutableHashTableVectors(self):
    for num_shards in [1, 3, 10]:
      with self.cached_session():
        default_val = [-0.1, 0.2]
        empty_key = [0, 1]
        deleted_key = [1, 0]
        keys = constant_op.constant([[11, 12], [13, 14], [15, 16]],
                                    dtypes.int64)
        values = constant_op.constant([[0.5, 0.6], [1.5, 1.6], [2.5, 2.6]],
                                      dtypes.float32)
        table = ShardedMutableDenseHashTable(
            dtypes.int64,
            dtypes.float32,
            default_val,
            empty_key,
            deleted_key,
            num_shards=num_shards)
        self.assertAllEqual(0, table.size().eval())

        table.insert(keys, values).run()
        self.assertAllEqual(3, table.size().eval())

        input_string = constant_op.constant([[11, 12], [13, 14], [11, 14]],
                                            dtypes.int64)
        output = table.lookup(input_string)
        self.assertAllEqual([3, 2], output.get_shape())
        self.assertAllClose([[0.5, 0.6], [1.5, 1.6], [-0.1, 0.2]],
                            output.eval())

  def testExportSharded(self):
    with self.cached_session():
      empty_key = -2
      deleted_key = -3
      default_val = -1
      num_shards = 2
      keys = constant_op.constant([10, 11, 12], dtypes.int64)
      values = constant_op.constant([2, 3, 4], dtypes.int64)
      table = ShardedMutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_val,
          empty_key,
          deleted_key,
          num_shards=num_shards)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      keys_list, values_list = table.export_sharded()
      self.assertAllEqual(num_shards, len(keys_list))
      self.assertAllEqual(num_shards, len(values_list))

      # Exported keys include empty key buckets set to the empty_key
      self.assertAllEqual(set([-2, 10, 12]), set(keys_list[0].eval().flatten()))
      self.assertAllEqual(set([-2, 11]), set(keys_list[1].eval().flatten()))
      # Exported values include empty value buckets set to 0
      self.assertAllEqual(set([0, 2, 4]), set(values_list[0].eval().flatten()))
      self.assertAllEqual(set([0, 3]), set(values_list[1].eval().flatten()))


if __name__ == '__main__':
  googletest.main()
