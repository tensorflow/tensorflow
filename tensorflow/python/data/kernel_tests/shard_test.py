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
from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
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

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    dataset = dataset_ops.Dataset.from_tensors(42).shard(1, 0, name="shard")
    self.assertDatasetProduces(dataset, [42])


class ShardCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                          parameterized.TestCase):

  def _build_dataset(self, num_elements, num_shards, index, options=None):
    dataset = dataset_ops.Dataset.range(num_elements).shard(num_shards, index)
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True]),
          combinations.combine(
              elems=[10, 100], num_shards=[2, 5], index=[0, 1])))
  def test(self, verify_fn, symbolic_checkpoint, elems, num_shards, index):
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(
        self,
        lambda: self._build_dataset(elems, num_shards, index, options),
        num_outputs=elems // num_shards)


class ShardRandomAccessTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 2, 3, 4])))
  def testInvalidIndex(self, index):
    dataset = dataset_ops.Dataset.range(4).shard(num_shards=2, index=0)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyDataset(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([]).shard(
        num_shards=2, index=1)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=0))

  @combinations.generate(test_base.default_test_combinations())
  def testNumShardsAndIndexLessThanNumElements(self):
    dataset = dataset_ops.Dataset.range(10).shard(5, 0)
    self.assertEqual(0, self.evaluate(random_access.at(dataset, 0)))
    self.assertEqual(5, self.evaluate(random_access.at(dataset, 1)))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 2))

  @combinations.generate(test_base.default_test_combinations())
  def testNumShardsGreaterThanNumElementsIndexLess(self):
    dataset = dataset_ops.Dataset.range(7).shard(8, 3)
    self.assertEqual(3, self.evaluate(random_access.at(dataset, 0)))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 1))

  @combinations.generate(test_base.default_test_combinations())
  def testNumShardsAndIndexGreaterThanNumElements(self):
    dataset = dataset_ops.Dataset.range(13).shard(23, 21)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 0))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              elements=[0, 10, 50],
              num_shards=[5, 7, 10],
              index=[0, 1, 2, 3, 4],
          )))
  def testMultipleCombinations(self, elements, num_shards, index):
    components = range(elements)
    dataset = dataset_ops.Dataset.range(elements).shard(
        num_shards=num_shards, index=index)
    len_dataset = self.evaluate(dataset.cardinality())
    for i in range(self.evaluate(dataset.cardinality())):
      self.assertAllEqual(components[index + (num_shards * i)],
                          self.evaluate(random_access.at(dataset, i)))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=len_dataset))


if __name__ == "__main__":
  test.main()
