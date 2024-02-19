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
"""Tests for global shuffling of tf.data datasets."""

from typing import Optional

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class GlobalShuffleTest(test_base.DatasetTestBase, parameterized.TestCase):
  """Tests for global shuffling of tf.data datasets."""

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(seed=[None, 42])))
  def testRange(self, seed: Optional[int]):
    dataset_range = 100
    dataset = dataset_ops.Dataset.range(dataset_range)
    dataset = global_shuffle_op._global_shuffle(dataset, seed=seed)
    dataset = dataset.repeat(3)
    output = self.getDatasetOutput(dataset, requires_initialization=True)
    self.assertCountEqual(output, list(range(dataset_range)) * 3)

    output_per_iteration = [
        output[i : i + dataset_range]
        for i in range(0, len(output), dataset_range)]
    self.assertCountEqual(output_per_iteration[0], list(range(dataset_range)))
    self.assertCountEqual(output_per_iteration[1], list(range(dataset_range)))
    self.assertCountEqual(output_per_iteration[2], list(range(dataset_range)))
    self.assertNotEqual(output_per_iteration[0], output_per_iteration[1])
    self.assertNotEqual(output_per_iteration[0], output_per_iteration[2])
    self.assertNotEqual(output_per_iteration[1], output_per_iteration[2])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(seed=[None, 42])))
  def testNegativeRange(self, seed: Optional[int]):
    dataset_range = 10
    dataset = dataset_ops.Dataset.range(dataset_range, -dataset_range, -1)
    dataset = global_shuffle_op._global_shuffle(dataset)
    dataset = dataset.repeat(3)
    output = self.getDatasetOutput(dataset, requires_initialization=True)
    self.assertCountEqual(
        output, list(range(dataset_range, -dataset_range, -1)) * 3)

    output_per_iteration = [
        output[i : i + dataset_range * 2]
        for i in range(0, len(output), dataset_range * 2)]
    self.assertCountEqual(output_per_iteration[0],
                          list(range(dataset_range, -dataset_range, -1)))
    self.assertCountEqual(output_per_iteration[1],
                          list(range(dataset_range, -dataset_range, -1)))
    self.assertCountEqual(output_per_iteration[2],
                          list(range(dataset_range, -dataset_range, -1)))
    self.assertNotEqual(output_per_iteration[0], output_per_iteration[1])
    self.assertNotEqual(output_per_iteration[0], output_per_iteration[2])
    self.assertNotEqual(output_per_iteration[1], output_per_iteration[2])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(seed=[None, 42])))
  def testNoReshuffleEachIteration(self, seed: Optional[int]):
    dataset_range = 100
    dataset = dataset_ops.Dataset.range(dataset_range)
    dataset = global_shuffle_op._global_shuffle(
        dataset, seed=seed, reshuffle_each_iteration=False)
    dataset = dataset.repeat(3)
    output = self.getDatasetOutput(dataset, requires_initialization=True)
    self.assertCountEqual(output, list(range(dataset_range)) * 3)

    output_per_iteration = [
        output[i : i + dataset_range]
        for i in range(0, len(output), dataset_range)]
    self.assertCountEqual(output_per_iteration[0], list(range(dataset_range)))
    self.assertCountEqual(output_per_iteration[1], list(range(dataset_range)))
    self.assertCountEqual(output_per_iteration[2], list(range(dataset_range)))
    self.assertEqual(output_per_iteration[0], output_per_iteration[1])
    self.assertEqual(output_per_iteration[0], output_per_iteration[2])
    self.assertEqual(output_per_iteration[1], output_per_iteration[2])

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyDataset(self):
    dataset = dataset_ops.Dataset.range(0)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "`global_shuffle` requires the input dataset to have a non-empty "
        "finite cardinality."):
      dataset = global_shuffle_op._global_shuffle(dataset)
      self.getDatasetOutput(dataset, requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testUnsupportedDataset(self):
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset.shuffle(buffer_size=1)
    with self.assertRaisesRegex(
        errors.FailedPreconditionError,
        "`global_shuffle` requires all upstream transformations be compatible "
        "with random access."):
      dataset = global_shuffle_op._global_shuffle(dataset)
      self.getDatasetOutput(dataset, requires_initialization=True)


if __name__ == "__main__":
  test.main()
