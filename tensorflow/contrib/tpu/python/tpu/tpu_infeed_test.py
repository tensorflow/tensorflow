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
# =============================================================================

"""Tests for TPU InfeedQueue methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tpu.python.tpu import tpu_feed

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


class InfeedTest(test.TestCase):

  def testConstructor(self):
    """Tests that the constructor can be called with different arguments."""
    i = tpu_feed.InfeedQueue(number_of_tuple_elements=2)
    self.assertEqual(i.number_of_tuple_elements, 2)
    self.assertEqual(i.tuple_types, None)
    self.assertEqual(i.tuple_shapes, None)
    self.assertEqual(i.number_of_shards, None)
    i = tpu_feed.InfeedQueue(
        tuple_types=[dtypes.float32, dtypes.int32, dtypes.int32])
    self.assertEqual(i.number_of_tuple_elements, 3)
    self.assertEqual(i.tuple_types,
                     [dtypes.float32, dtypes.int32, dtypes.int32])
    self.assertEqual(i.tuple_shapes, None)
    self.assertEqual(i.number_of_shards, None)
    i = tpu_feed.InfeedQueue(tuple_shapes=[[1], [2, 3]])
    self.assertEqual(i.number_of_tuple_elements, 2)
    self.assertEqual(i.tuple_types, None)
    self.assertEqual(i.tuple_shapes, [[1], [2, 3]])
    self.assertEqual(i.number_of_shards, None)
    i = tpu_feed.InfeedQueue(shard_dimensions=[1, 0, 7])
    self.assertEqual(i.number_of_tuple_elements, 3)
    self.assertEqual(i.tuple_types, None)
    self.assertEqual(i.tuple_shapes, None)
    self.assertEqual([p.shard_dimension
                      for p in i.sharding_policies], [1, 0, 7])
    with self.assertRaises(ValueError):
      i = tpu_feed.InfeedQueue()
    with self.assertRaises(ValueError):
      i = tpu_feed.InfeedQueue(
          number_of_tuple_elements=2, tuple_types=[dtypes.float32])
    with self.assertRaises(ValueError):
      i = tpu_feed.InfeedQueue(number_of_tuple_elements=2, tuple_shapes=[[1]])
    with self.assertRaises(ValueError):
      i = tpu_feed.InfeedQueue(number_of_tuple_elements=2, shard_dimensions=[1])
    with self.assertRaises(ValueError):
      i = tpu_feed.InfeedQueue(tuple_shapes=[[1], [2, 3]], shard_dimensions=[1])

  def testModification(self):
    """Tests modification of the queue post-construction."""
    i = tpu_feed.InfeedQueue(number_of_tuple_elements=2)
    i.set_tuple_types([dtypes.float32, dtypes.int32])
    self.assertEqual(i.tuple_types, [dtypes.float32, dtypes.int32])
    i.set_tuple_types([dtypes.float32, dtypes.float32])
    self.assertEqual(i.tuple_types, [dtypes.float32, dtypes.float32])
    with self.assertRaises(ValueError):
      i.set_tuple_types([dtypes.float32])
    i.set_tuple_shapes([[1], [2, 3]])
    self.assertEqual(i.tuple_shapes, [[1], [2, 3]])
    i.set_tuple_shapes([[1, 2], [3, 4]])
    self.assertEqual(i.tuple_shapes, [[1, 2], [3, 4]])
    with self.assertRaises(ValueError):
      i.set_tuple_shapes([[1, 2]])
    i.set_number_of_shards(2)
    self.assertEqual(i.number_of_shards, 2)
    i.set_number_of_shards(3)
    self.assertEqual(i.number_of_shards, 3)
    t1 = constant_op.constant(1, dtypes.int32, shape=[6])
    t2 = constant_op.constant(2.0, dtypes.float32, shape=[3, 18])
    i.set_configuration_from_input_tensors([t1, t2])
    self.assertEqual(i.tuple_shapes, [[6], [3, 18]])
    self.assertEqual(i.tuple_types, [dtypes.int32, dtypes.float32])
    i.set_configuration_from_sharded_input_tensors([[t2, t1], [t2, t1]])
    self.assertEqual(i.number_of_shards, 2)
    self.assertEqual(i.tuple_shapes, [[6, 18], [12]])
    self.assertEqual(i.tuple_types, [dtypes.float32, dtypes.int32])
    i.set_shard_dimensions([1, 0])
    i.set_number_of_shards(3)
    with self.assertRaises(ValueError):
      i.set_number_of_shards(4)

  def testFreezing(self):
    """Tests freezing the queue."""
    i = tpu_feed.InfeedQueue(number_of_tuple_elements=2)
    t1 = constant_op.constant(1, dtypes.int32, shape=[2])
    t2 = constant_op.constant(2.0, dtypes.float32, shape=[2, 4])
    i.set_configuration_from_sharded_input_tensors([[t2, t1], [t2, t1]])
    self.assertEqual(i.number_of_shards, 2)
    self.assertEqual(i.tuple_shapes, [[4, 4], [4]])
    self.assertEqual(i.tuple_types, [dtypes.float32, dtypes.int32])
    self.assertEqual(i.shard_dimensions, [0, 0])
    i.freeze()
    i.set_number_of_shards(2)
    i.set_tuple_shapes([[4, 4], [4]])
    i.set_tuple_types([dtypes.float32, dtypes.int32])
    i.set_shard_dimensions([0, 0])
    with self.assertRaises(ValueError):
      i.set_number_of_shards(1)
    with self.assertRaises(ValueError):
      i.set_tuple_shapes([[8, 8], [8]])
    with self.assertRaises(ValueError):
      i.set_tuple_types([dtypes.int32, dtypes.float32])
    with self.assertRaises(ValueError):
      i.set_shard_dimensions([1, 0])
    self.assertEqual(i.number_of_shards, 2)
    self.assertEqual(i.tuple_shapes, [[4, 4], [4]])
    self.assertEqual(i.tuple_types, [dtypes.float32, dtypes.int32])
    self.assertEqual(i.shard_dimensions, [0, 0])

if __name__ == '__main__':
  test.main()
