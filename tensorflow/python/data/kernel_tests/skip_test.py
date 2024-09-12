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
"""Tests for `tf.data.Dataset.skip()`."""

from typing import Callable, Optional

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class SkipTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(count=[-1, 0, 4, 10, 25])))
  def testBasic(self, count):
    components = (np.arange(10),)
    dataset = dataset_ops.Dataset.from_tensor_slices(components).skip(count)
    self.assertEqual(
        [c.shape[1:] for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    start_range = min(count, 10) if count != -1 else 10
    self.assertDatasetProduces(
        dataset,
        [tuple(components[0][i:i + 1]) for i in range(start_range, 10)])

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    dataset = dataset_ops.Dataset.from_tensors(42).skip(0, name="skip")
    self.assertDatasetProduces(dataset, [42])


class SkipDatasetCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                parameterized.TestCase):

  def _build_skip_dataset(self, count, options=None):
    dataset = dataset_ops.Dataset.range(100).skip(count)
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True]),
          combinations.combine(count=[50], num_outputs=[50]) +
          combinations.combine(count=[200, 100, -1], num_outputs=[0]) +
          combinations.combine(count=[0], num_outputs=[100])))
  def test(self, verify_fn, count, num_outputs, symbolic_checkpoint):
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(self, lambda: self._build_skip_dataset(count, options),
              num_outputs)


class SkipRandomAccessTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 2, 3])))
  def testInvalidIndex(self, index):
    dataset = dataset_ops.Dataset.range(10).skip(8)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 0])))
  def testEmptyDataset(self, index):
    dataset = dataset_ops.Dataset.from_tensor_slices([]).skip(8)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testBasic(self):
    dataset = dataset_ops.Dataset.range(11).skip(3)
    for i in range(8):
      self.assertEqual(self.evaluate(random_access.at(dataset, index=i)), i + 3)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(skip=[-2, -1])))
  def testNegativeSkip(self, skip):
    dataset = dataset_ops.Dataset.range(11).skip(skip)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=0))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(skip=[5, 8])))
  def testSkipGreaterThanNumElements(self, skip):
    dataset = dataset_ops.Dataset.range(4).skip(skip)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=0))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(elements=[0, 5, 10], skip=[-1, 0, 5, 15])))
  def testMultipleCombinations(self, elements, skip):
    dataset = dataset_ops.Dataset.range(elements).skip(skip)
    for i in range(self.evaluate(dataset.cardinality())):
      self.assertEqual(
          self.evaluate(random_access.at(dataset, index=i)), i + skip)


class SkipGlobalShuffleTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[10],
              count=[0, 2],
              repetitions=[1, 2],
              seed=[None, 42],
              reshuffle_each_iteration=[True, False])))
  def testSkip(
      self,
      dataset_range: int,
      count: int,
      repetitions: int,
      seed: Optional[int],
      reshuffle_each_iteration: bool):
    dataset = dataset_ops.Dataset.range(dataset_range)
    dataset = dataset.skip(count)
    dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
    if repetitions > 1:
      dataset = dataset.repeat(repetitions)
    dataset = global_shuffle_op._global_shuffle(
        dataset, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)

    expected = list(range(count, dataset_range)) * repetitions
    dataset_output = self.getDatasetOutput(
        dataset, requires_initialization=True)
    self.assertCountEqual(dataset_output, expected)
    self.assertNotEqual(dataset_output, expected)
    self.assertLen(dataset_output, self.evaluate(dataset.cardinality()))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(skip=[-2, -1])))
  def testNegativeSkip(self, skip: int):
    dataset = dataset_ops.Dataset.range(10).skip(skip)
    with self.assertRaises(errors.FailedPreconditionError):
      dataset = global_shuffle_op._global_shuffle(dataset)
      self.getDatasetOutput(dataset, requires_initialization=True)


class SkipGlobalShuffleCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[10],
              count=[0, 2],
              repetitions=[1, 2],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False])))
  def testSkip(
      self,
      verify_fn: Callable[..., None],
      dataset_range: int,
      count: int,
      repetitions: int,
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool):
    def _build_dataset() -> dataset_ops.Dataset:
      dataset = dataset_ops.Dataset.range(dataset_range)
      dataset = dataset.skip(count)
      dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
      if repetitions > 1:
        dataset = dataset.repeat(repetitions)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration)
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=(dataset_range - count) * repetitions,
        assert_items_equal=reshuffle_each_iteration,
    )


if __name__ == "__main__":
  test.main()
