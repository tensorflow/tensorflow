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

from typing import Callable, Optional

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import test


class GlobalShuffleTest(test_base.DatasetTestBase, parameterized.TestCase):
  """Tests for global shuffling of tf.data datasets."""

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[1, 100],
              seed=[None, 42],
              use_tensor_seed=[True, False],
              prefetch=[True, False])))
  def testRange(
      self,
      dataset_range: int,
      seed: Optional[int],
      use_tensor_seed: bool,
      prefetch: bool):
    dataset = dataset_ops.Dataset.range(dataset_range)
    if prefetch:
      dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    seed = (constant_op.constant(seed, dtype=dtypes.int64)
            if seed and use_tensor_seed else seed)
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
    if dataset_range > 1:
      self.assertNotEqual(output_per_iteration[0], output_per_iteration[1])
      self.assertNotEqual(output_per_iteration[0], output_per_iteration[2])
      self.assertNotEqual(output_per_iteration[1], output_per_iteration[2])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(dataset_range=[1, 100], seed=[None, 42])))
  def testNegativeRange(self, dataset_range: int, seed: Optional[int]):
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
    if dataset_range > 1:
      self.assertNotEqual(output_per_iteration[0], output_per_iteration[1])
      self.assertNotEqual(output_per_iteration[0], output_per_iteration[2])
      self.assertNotEqual(output_per_iteration[1], output_per_iteration[2])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(reshuffle=[True, False], seed=[None, 42])))
  def testReshuffleRepeatEpochs(self, reshuffle: bool, seed: Optional[int]):
    dataset_range = 100
    dataset = dataset_ops.Dataset.range(dataset_range)
    dataset = global_shuffle_op._global_shuffle(
        dataset, seed=seed, reshuffle_each_iteration=reshuffle)
    dataset = dataset.repeat(2)

    output = self.getDatasetOutput(dataset, requires_initialization=True)
    self.assertCountEqual(output, list(range(dataset_range)) * 2)
    output_per_iteration = [
        output[i : i + dataset_range]
        for i in range(0, len(output), dataset_range)]
    if reshuffle:
      self.assertNotEqual(output_per_iteration[0], output_per_iteration[1])
    else:
      self.assertEqual(output_per_iteration[0], output_per_iteration[1])

  # Creating multiple iterators with the same seed is only supported in v2 API.
  @combinations.generate(
      combinations.times(
          combinations.combine(tf_api_version=2, mode="eager"),
          combinations.combine(reshuffle=[True, False], seed=[None, 42])))
  def testReshuffleIterationEpochs(self, reshuffle: bool, seed: Optional[int]):
    # TensorFlow unit tests set the global graph seed. We unset it here so that
    # we can control determinism via the `seed` parameter.
    random_seed.set_random_seed(None)
    dataset_range = 100
    dataset = dataset_ops.Dataset.range(dataset_range)
    dataset = global_shuffle_op._global_shuffle(
        dataset, seed=seed, reshuffle_each_iteration=reshuffle)

    first_epoch = self.getDatasetOutput(dataset)
    second_epoch = self.getDatasetOutput(dataset)
    if reshuffle:
      self.assertNotEqual(first_epoch, second_epoch)
    else:
      self.assertEqual(first_epoch, second_epoch)

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


class GlobalShuffleCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                  parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[1, 10],
              reshuffle_each_iteration=[True, False],
              prefetch=[True, False],
              symbolic_checkpoint=[True, False])))
  def testRange(
      self,
      verify_fn: Callable[..., None],
      dataset_range: int,
      reshuffle_each_iteration: bool,
      prefetch: bool,
      symbolic_checkpoint: bool):

    def _build_dataset() -> dataset_ops.Dataset:
      dataset = dataset_ops.Dataset.range(dataset_range)
      if prefetch:
        dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration)
      if symbolic_checkpoint:
        options = options_lib.Options()
        options.experimental_symbolic_checkpoint = symbolic_checkpoint
        dataset = dataset.with_options(options)
      return dataset

    verify_fn(
        self,
        _build_dataset,
        num_outputs=dataset_range,
        assert_items_equal=reshuffle_each_iteration)


if __name__ == "__main__":
  test.main()
