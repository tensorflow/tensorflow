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
"""Tests for `tf.data.Dataset.shard()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_v1_only("deprecated API, no eager or V2 test coverage")
class ShardTest(test_base.DatasetTestBase):

  def testSimpleCase(self):
    dataset = dataset_ops.Dataset.range(10).shard(5, 2)
    self.assertDatasetProduces(dataset, expected_output=[2, 7])

  def testNestedData(self):
    dataset_a = dataset_ops.Dataset.range(10)
    dataset_b = dataset_ops.Dataset.range(10, 0, -1)
    dataset = dataset_ops.Dataset.zip((dataset_a, dataset_b)).shard(5, 2)
    self.assertDatasetProduces(dataset, expected_output=[(2, 8), (7, 3)])

  def testOffsetZero(self):
    dataset = dataset_ops.Dataset.range(10).shard(5, 0)
    self.assertDatasetProduces(dataset, expected_output=[0, 5])

  def testOffsetGreaterNumShards(self):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10).shard(5, 7)
      self.evaluate(self.getNext(dataset)())

  def testNegativeOffset(self):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10).shard(5, -3)
      self.evaluate(self.getNext(dataset)())

  def testNegativeNumShards(self):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10).shard(-3, 1)
      self.evaluate(self.getNext(dataset)())

  def testZeroNumShards(self):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10).shard(0, 1)
      self.evaluate(self.getNext(dataset)())

  def testIteratorEndsBeforeFirstElem(self):
    dataset = dataset_ops.Dataset.range(1).shard(5, 2)
    self.assertDatasetProduces(dataset, expected_output=[])

  def testLargerWorkerPool(self):
    dataset = dataset_ops.Dataset.range(10).shard(7, 5)
    self.assertDatasetProduces(dataset, expected_output=[5])

  def testIndexEqualsNumShards(self):
    dataset = dataset_ops.Dataset.range(10).shard(5, 4)
    self.assertDatasetProduces(dataset, expected_output=[4, 9])

  def testIndexEqualsNumShards2(self):
    dataset = dataset_ops.Dataset.range(10).shard(4, 3)
    self.assertDatasetProduces(dataset, expected_output=[3, 7])

  def testNumShardsLargerThanDataset(self):
    dataset = dataset_ops.Dataset.range(10).shard(20, 5)
    self.assertDatasetProduces(dataset, expected_output=[5])


if __name__ == "__main__":
  test.main()
