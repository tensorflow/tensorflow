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

from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class ShardTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testSimpleCase(self):
    dataset = dataset_ops.Dataset.range(10).shard(5, 2)
    self.assertDatasetProduces(dataset, expected_output=[2, 7])

  @combinations.generate(test_base.default_test_combinations())
  def testNestedData(self):
    dataset_a = dataset_ops.Dataset.range(10)
    dataset_b = dataset_ops.Dataset.range(10, 0, -1)
    dataset = dataset_ops.Dataset.zip((dataset_a, dataset_b)).shard(5, 2)
    self.assertDatasetProduces(dataset, expected_output=[(2, 8), (7, 3)])

  @combinations.generate(test_base.default_test_combinations())
  def testOffsetZero(self):
    dataset = dataset_ops.Dataset.range(10).shard(5, 0)
    self.assertDatasetProduces(dataset, expected_output=[0, 5])

  @combinations.generate(test_base.default_test_combinations())
  def testOffsetGreaterNumShards(self):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10).shard(5, 7)
      self.evaluate(self.getNext(dataset)())

  @combinations.generate(test_base.default_test_combinations())
  def testNegativeOffset(self):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10).shard(5, -3)
      self.evaluate(self.getNext(dataset)())

  @combinations.generate(test_base.default_test_combinations())
  def testNegativeNumShards(self):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10).shard(-3, 1)
      self.evaluate(self.getNext(dataset)())

  @combinations.generate(test_base.default_test_combinations())
  def testZeroNumShards(self):
    with self.assertRaises(errors.InvalidArgumentError):
      dataset = dataset_ops.Dataset.range(10).shard(0, 1)
      self.evaluate(self.getNext(dataset)())

  @combinations.generate(test_base.default_test_combinations())
  def testIteratorEndsBeforeFirstElem(self):
    dataset = dataset_ops.Dataset.range(1).shard(5, 2)
    self.assertDatasetProduces(dataset, expected_output=[])

  @combinations.generate(test_base.default_test_combinations())
  def testLargerWorkerPool(self):
    dataset = dataset_ops.Dataset.range(10).shard(7, 5)
    self.assertDatasetProduces(dataset, expected_output=[5])

  @combinations.generate(test_base.default_test_combinations())
  def testIndexEqualsNumShards(self):
    dataset = dataset_ops.Dataset.range(10).shard(5, 4)
    self.assertDatasetProduces(dataset, expected_output=[4, 9])

  @combinations.generate(test_base.default_test_combinations())
  def testIndexEqualsNumShards2(self):
    dataset = dataset_ops.Dataset.range(10).shard(4, 3)
    self.assertDatasetProduces(dataset, expected_output=[3, 7])

  @combinations.generate(test_base.default_test_combinations())
  def testNumShardsLargerThanDataset(self):
    dataset = dataset_ops.Dataset.range(10).shard(20, 5)
    self.assertDatasetProduces(dataset, expected_output=[5])


class ShardCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                          parameterized.TestCase):

  def _build_dataset(self, num_elements, num_shards, index):
    return dataset_ops.Dataset.range(num_elements).shard(num_shards, index)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              elems=[10, 100], num_shards=[2, 5], index=[0, 1])))
  def testCore(self, elems, num_shards, index):
    self.run_core_tests(lambda: self._build_dataset(elems, num_shards, index),
                        elems // num_shards)


if __name__ == "__main__":
  test.main()
