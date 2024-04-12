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
"""Tests for `tf.data.Dataset.weighted_flat_map()`."""

from typing import Callable

from absl.testing import parameterized

import numpy as np

from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.experimental.ops import weighted_flat_map_op
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class WeightedFlatMapTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testWeightedFlatMap(self):
    dataset1 = dataset_ops.Dataset.range(10)
    dataset2 = dataset_ops.Dataset.range(10, 20)
    dataset3 = dataset_ops.Dataset.range(20, 30)
    dataset = weighted_flat_map_op._weighted_flat_map(
        [dataset1, dataset2, dataset3], np.asarray([1, 1, 2]))
    self.assertDatasetProduces(
        dataset, expected_output=list(range(5)) + list(range(10, 15)) +
        list(range(20, 30)))

  @combinations.generate(test_base.default_test_combinations())
  def testInvalidCardinality(self):
    dataset1 = dataset_ops.Dataset.range(100)
    dataset2 = dataset_ops.Dataset.range(100, 200)
    dataset3 = dataset_ops.Dataset.range(200, 210)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, "Input.*needs to have at least."):
      dataset = weighted_flat_map_op._weighted_flat_map(
          [dataset1, dataset2, dataset3], np.asarray([1, 1, 100]))
      self.getDatasetOutput(dataset, requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testInfiniteCardinality(self):
    dataset1 = dataset_ops.Dataset.range(10).repeat()
    dataset2 = dataset_ops.Dataset.range(10, 20)
    dataset3 = dataset_ops.Dataset.range(20, 30)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Cardinalities of the inputs must be known."):
      dataset = weighted_flat_map_op._weighted_flat_map(
          [dataset1, dataset2, dataset3])
      self.getDatasetOutput(dataset, requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyInputDatasets(self):
    dataset1 = dataset_ops.Dataset.from_tensor_slices([])
    dataset2 = dataset_ops.Dataset.range(10, 20)
    dataset3 = dataset_ops.Dataset.range(20, 30)
    with self.assertRaisesRegex(
        TypeError,
        "Incompatible dataset elements"):
      dataset = weighted_flat_map_op._weighted_flat_map(
          [dataset1, dataset2, dataset3])
      self.getDatasetOutput(dataset, requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testZeroWeight(self):
    dataset1 = dataset_ops.Dataset.range(10)
    dataset2 = dataset_ops.Dataset.range(10, 20)
    dataset3 = dataset_ops.Dataset.range(20, 30)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "`weights` must be greater than 0.0"):
      dataset = weighted_flat_map_op._weighted_flat_map(
          [dataset1, dataset2, dataset3], [0, 1.0, 1.0])
      self.getDatasetOutput(dataset, requires_initialization=True)


class GlobalShuffleTest(test_base.DatasetTestBase, parameterized.TestCase):
  """Tests for global shuffling of tf.data datasets."""

  @combinations.generate(test_base.default_test_combinations())
  def testShuffledOutput(self):
    dataset1 = dataset_ops.Dataset.range(10)
    dataset2 = dataset_ops.Dataset.range(10, 20)
    dataset3 = dataset_ops.Dataset.range(20, 30)
    dataset = weighted_flat_map_op._weighted_flat_map(
        [dataset1, dataset2, dataset3], np.asarray([0.25, 0.25, 0.5]))
    dataset = global_shuffle_op._global_shuffle(dataset)
    output = self.getDatasetOutput(dataset, requires_initialization=True)
    self.assertCountEqual(
        output, list(range(5)) + list(range(10, 15)) + list(range(20, 30)))

  @combinations.generate(test_base.default_test_combinations())
  def testShuffledInputs(self):
    dataset1 = dataset_ops.Dataset.range(10)
    dataset2 = dataset_ops.Dataset.range(10, 20)
    dataset3 = dataset_ops.Dataset.range(20, 30)
    dataset1 = global_shuffle_op._global_shuffle(dataset1, seed=42)
    dataset2 = global_shuffle_op._global_shuffle(dataset2, seed=42)
    dataset3 = global_shuffle_op._global_shuffle(dataset3, seed=42)
    dataset = weighted_flat_map_op._weighted_flat_map(
        [dataset1, dataset2, dataset3], np.asarray([0.25, 0.25, 0.5]))
    output = self.getDatasetOutput(dataset, requires_initialization=True)
    # Verifies that the first 5 elements are from `dataset1` in a random order.
    self.assertFalse(set(output[:5]).issubset(set(range(5))))
    self.assertTrue(set(output[:5]).issubset(set(range(10))))
    # Verifies that the second 5 elements are from `dataset2` in a random order.
    self.assertFalse(set(output[5:10]).issubset(set(range(10, 15))))
    self.assertTrue(set(output[5:10]).issubset(set(range(10, 20))))
    # Verifies that the last 10 elements are from `dataset3` in a random order.
    self.assertCountEqual(output[10:], range(20, 30))
    self.assertNotEqual(output[10:], range(20, 30))

  @combinations.generate(test_base.default_test_combinations())
  def testShuffledInputsAndOutput(self):
    dataset1 = dataset_ops.Dataset.range(10)
    dataset2 = dataset_ops.Dataset.range(10, 20)
    dataset3 = dataset_ops.Dataset.range(20, 30)
    dataset1 = global_shuffle_op._global_shuffle(dataset1, seed=42)
    dataset2 = global_shuffle_op._global_shuffle(dataset2, seed=42)
    dataset3 = global_shuffle_op._global_shuffle(dataset3, seed=42)
    dataset = weighted_flat_map_op._weighted_flat_map(
        [dataset1, dataset2, dataset3], np.asarray([0.25, 0.25, 0.5]))
    dataset = global_shuffle_op._global_shuffle(dataset, seed=42)
    output = self.getDatasetOutput(dataset, requires_initialization=True)
    # Verifies that not all first 5 elements are from `dataset1`.
    self.assertFalse(set(output[:5]).issubset(set(range(10))))
    # Verifies that not all second 5 elements are from `dataset2`.
    self.assertFalse(set(output[5:10]).issubset(set(range(10, 20))))
    # Verifies that not all last 10 elements are from `dataset3`.
    self.assertFalse(set(output[10:]).issubset(set(range(20, 30))))

    sorted_output = sorted(output)
    # Verifies that there are 5 elements from dataset1
    self.assertTrue(set(sorted_output[:5]).issubset(set(range(10))))
    # Verifies that there are 5 elements from dataset2
    self.assertTrue(set(sorted_output[5:10]).issubset(set(range(10, 20))))
    # Verifies that there are 10 elements from dataset3
    self.assertTrue(set(sorted_output[10:]).issubset(set(range(20, 30))))


class WeightedFlatMapGlobalShuffleCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase
):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False])))
  def testWeightedFlatMap(
      self,
      verify_fn: Callable[..., None],
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool):

    def _build_dataset() -> dataset_ops.Dataset:
      dataset1 = dataset_ops.Dataset.range(10)
      dataset2 = dataset_ops.Dataset.range(10, 20)
      dataset3 = dataset_ops.Dataset.range(20, 30)
      dataset = weighted_flat_map_op._weighted_flat_map(
          [dataset1, dataset2, dataset3], np.asarray([0.25, 0.25, 0.5])
      )
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration
      )
      options = options_lib.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=20,
        assert_items_equal=reshuffle_each_iteration)


if __name__ == "__main__":
  test.main()
