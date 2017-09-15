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
# ==============================================================================
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data.python.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class ShardDatasetOpTest(test.TestCase):

  def testSimpleCase(self):
    dataset = dataset_ops.Dataset.range(10).shard(5, 2)
    iterator = dataset.make_one_shot_iterator()

    with self.test_session() as sess:
      self.assertEqual(2, sess.run(iterator.get_next()))
      self.assertEqual(7, sess.run(iterator.get_next()))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(iterator.get_next())

  def testNestedData(self):
    dataset_a = dataset_ops.Dataset.range(10)
    dataset_b = dataset_ops.Dataset.range(10, 0, -1)
    dataset = dataset_ops.Dataset.zip((dataset_a, dataset_b)).shard(5, 2)
    iterator = dataset.make_one_shot_iterator()

    with self.test_session() as sess:
      self.assertEqual((2, 8), sess.run(iterator.get_next()))
      self.assertEqual((7, 3), sess.run(iterator.get_next()))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(iterator.get_next())

  def testOffsetZero(self):
    dataset = dataset_ops.Dataset.range(10).shard(5, 0)
    iterator = dataset.make_one_shot_iterator()

    with self.test_session() as sess:
      self.assertEqual(0, sess.run(iterator.get_next()))
      self.assertEqual(5, sess.run(iterator.get_next()))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(iterator.get_next())

  def testOffsetGreaterNumShards(self):
    with self.assertRaises(ValueError):
      dataset_ops.Dataset.range(10).shard(5, 7)

  def testNegativeOffset(self):
    with self.assertRaises(ValueError):
      dataset_ops.Dataset.range(10).shard(5, -3)

  def testNegativeNumShards(self):
    with self.assertRaises(ValueError):
      dataset_ops.Dataset.range(10).shard(-3, 1)

  def testZeroNumShards(self):
    with self.assertRaises(ValueError):
      dataset_ops.Dataset.range(10).shard(0, 1)

  def testIteratorEndsBeforeFirstElem(self):
    dataset = dataset_ops.Dataset.range(1).shard(5, 2)
    iterator = dataset.make_one_shot_iterator()

    with self.test_session() as sess:
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(iterator.get_next())

  def testLargerWorkerPool(self):
    dataset = dataset_ops.Dataset.range(10).shard(7, 5)
    iterator = dataset.make_one_shot_iterator()
    with self.test_session() as sess:
      self.assertEqual(5, sess.run(iterator.get_next()))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(iterator.get_next())

  def testIndexEqualsNumShards(self):
    dataset = dataset_ops.Dataset.range(10).shard(5, 4)
    iterator = dataset.make_one_shot_iterator()
    with self.test_session() as sess:
      self.assertEqual(4, sess.run(iterator.get_next()))
      self.assertEqual(9, sess.run(iterator.get_next()))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(iterator.get_next())

  def testIndexEqualsNumShards2(self):
    dataset = dataset_ops.Dataset.range(10).shard(4, 3)
    iterator = dataset.make_one_shot_iterator()
    with self.test_session() as sess:
      self.assertEqual(3, sess.run(iterator.get_next()))
      self.assertEqual(7, sess.run(iterator.get_next()))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(iterator.get_next())


if __name__ == "__main__":
  test.main()
