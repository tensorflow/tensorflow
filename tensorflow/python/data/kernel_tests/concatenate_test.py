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
"""Tests for `tf.data.Dataset.concatenate()."""
from typing import Callable, Tuple
from absl.testing import parameterized
import numpy as np
from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test


class ConcatenateTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testBasic(self):
    input_components = (
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 15),
        np.array([37.0, 38.0, 39.0, 40.0]))
    to_concatenate_components = (
        np.tile(np.array([[1], [2], [3], [4], [5]]), 20),
        np.tile(np.array([[12], [13], [14], [15], [16]]), 15),
        np.array([37.0, 38.0, 39.0, 40.0, 41.0]))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)
    concatenated = input_dataset.concatenate(dataset_to_concatenate)
    self.assertEqual(
        dataset_ops.get_legacy_output_shapes(concatenated),
        (tensor_shape.TensorShape([20]), tensor_shape.TensorShape([15]),
         tensor_shape.TensorShape([])))

    get_next = self.getNext(concatenated)

    for i in range(9):
      result = self.evaluate(get_next())
      if i < 4:
        for component, result_component in zip(input_components, result):
          self.assertAllEqual(component[i], result_component)
      else:
        for component, result_component in zip(to_concatenate_components,
                                               result):
          self.assertAllEqual(component[i - 4], result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testDifferentShape(self):
    input_components = (
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 4))
    to_concatenate_components = (
        np.tile(np.array([[1], [2], [3], [4], [5]]), 20),
        np.tile(np.array([[12], [13], [14], [15], [16]]), 15))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)
    concatenated = input_dataset.concatenate(dataset_to_concatenate)
    self.assertEqual(
        [ts.as_list()
         for ts in nest.flatten(
             dataset_ops.get_legacy_output_shapes(concatenated))],
        [[20], [None]])
    get_next = self.getNext(concatenated)
    for i in range(9):
      result = self.evaluate(get_next())
      if i < 4:
        for component, result_component in zip(input_components, result):
          self.assertAllEqual(component[i], result_component)
      else:
        for component, result_component in zip(to_concatenate_components,
                                               result):
          self.assertAllEqual(component[i - 4], result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testDifferentStructure(self):
    input_components = (
        np.tile(np.array([[1], [2], [3], [4]]), 5),
        np.tile(np.array([[12], [13], [14], [15]]), 4))
    to_concatenate_components = (
        np.tile(np.array([[1], [2], [3], [4], [5]]), 20),
        np.tile(np.array([[12], [13], [14], [15], [16]]), 15),
        np.array([37.0, 38.0, 39.0, 40.0, 41.0]))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)

    with self.assertRaisesRegex(TypeError, "Incompatible dataset elements"):
      input_dataset.concatenate(dataset_to_concatenate)

  @combinations.generate(test_base.default_test_combinations())
  def testDifferentKeys(self):
    input_components = {
        "foo": np.array([[1], [2], [3], [4]]),
        "bar": np.array([[12], [13], [14], [15]])
    }
    to_concatenate_components = {
        "foo": np.array([[1], [2], [3], [4]]),
        "baz": np.array([[5], [6], [7], [8]])
    }

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)

    with self.assertRaisesRegex(TypeError, "Incompatible dataset elements"):
      input_dataset.concatenate(dataset_to_concatenate)

  @combinations.generate(test_base.default_test_combinations())
  def testDifferentType(self):
    input_components = (
        np.tile(np.array([[1], [2], [3], [4]]), 5),
        np.tile(np.array([[12], [13], [14], [15]]), 4))
    to_concatenate_components = (
        np.tile(np.array([[1.0], [2.0], [3.0], [4.0]]), 5),
        np.tile(np.array([[12], [13], [14], [15]]), 15))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)

    with self.assertRaisesRegex(TypeError, "Incompatible dataset elements"):
      input_dataset.concatenate(dataset_to_concatenate)

  @combinations.generate(test_base.default_test_combinations())
  def testWindows(self):
    a = dataset_ops.Dataset.range(5).window(1)
    b = dataset_ops.Dataset.range(5, 10).window(1)
    c = a.concatenate(b).flat_map(lambda x: x)
    self.assertDatasetProduces(c, list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    a = dataset_ops.Dataset.range(5)
    b = dataset_ops.Dataset.range(5, 10)
    c = a.concatenate(b, name="concatenate")
    self.assertDatasetProduces(c, list(range(10)))


class ConcatenateCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                parameterized.TestCase):

  def _build_concatenate_dataset(self, var_array, options=None):
    input_components = (np.tile(np.array([[1], [2], [3], [4]]), 20),
                        np.tile(np.array([[12], [13], [14], [15]]), 4))
    to_concatenate_components = (np.tile(
        np.array([[5], [6], [7], [8], [9]]), 20), var_array)

    dataset = dataset_ops.Dataset.from_tensor_slices(
        input_components).concatenate(
            dataset_ops.Dataset.from_tensor_slices(to_concatenate_components))
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def test(self, verify_fn, symbolic_checkpoint):
    num_outputs = 9
    array = np.tile(np.array([[16], [17], [18], [19], [20]]), 15)
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(self, lambda: self._build_concatenate_dataset(array, options),
              num_outputs)


class ConcatenateRandomAccessTest(test_base.DatasetTestBase,
                                  parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 3, 4])))
  def testInvalidIndex(self, index):
    input_dataset = dataset_ops.Dataset.from_tensor_slices([-1])
    concatenate_dataset = dataset_ops.Dataset.from_tensor_slices([1, 2])
    concatenated = input_dataset.concatenate(concatenate_dataset)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(concatenated, index=index))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testConcatenateTwoEmptyDatasets(self):
    input_dataset = dataset_ops.Dataset.from_tensor_slices([])
    concatenate_dataset = dataset_ops.Dataset.from_tensor_slices([])
    concatenated = input_dataset.concatenate(concatenate_dataset)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(concatenated, index=0))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testConcatenateAnEmptyDataset(self):
    input_dataset = dataset_ops.Dataset.from_tensor_slices([1.0])
    concatenate_dataset = dataset_ops.Dataset.from_tensor_slices([])
    concatenated = input_dataset.concatenate(concatenate_dataset)
    self.assertAllEqual(
        self.evaluate(random_access.at(concatenated, index=0)), 1.0)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(concatenated, index=1))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testConcatenateOntoEmptyDataset(self):
    input_dataset = dataset_ops.Dataset.from_tensor_slices([])
    concatenate_dataset = dataset_ops.Dataset.from_tensor_slices([2.0, 3.0])
    concatenated = input_dataset.concatenate(concatenate_dataset)
    self.assertAllEqual(
        self.evaluate(random_access.at(concatenated, index=0)), 2.0)
    self.assertAllEqual(
        self.evaluate(random_access.at(concatenated, index=1)), 3.0)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(concatenated, index=2))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testConcatenateTwoNonEmptyDatasets(self):
    input_dataset = dataset_ops.Dataset.from_tensor_slices([0, 1, 2])
    concatenate_dataset = dataset_ops.Dataset.from_tensor_slices([3, 4])
    concatenated = input_dataset.concatenate(concatenate_dataset)
    for i in range(5):
      self.assertAllEqual(random_access.at(concatenated, index=i), i)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(concatenated, index=5))


class GlobalShuffleTest(test_base.DatasetTestBase, parameterized.TestCase):
  """Tests for global shuffling of tf.data datasets."""

  @combinations.generate(test_base.default_test_combinations())
  def testShuffledOutput(self):
    dataset1 = dataset_ops.Dataset.range(0, 5)
    dataset2 = dataset_ops.Dataset.range(5, 17)

    dataset = dataset1.concatenate(dataset2)

    dataset = global_shuffle_op._global_shuffle(dataset)

    output = self.getDatasetOutput(dataset, requires_initialization=True)
    self.assertCountEqual(output, range(0, 17))

  @combinations.generate(test_base.default_test_combinations())
  def testShuffledWithBatchOutput(self):
    """Testing with `.batch()` ensures the global shuffle map is stateless."""
    dataset1 = dataset_ops.Dataset.range(0, 4)
    dataset2 = dataset_ops.Dataset.range(4, 10)

    dataset = dataset1.concatenate(dataset2)
    dataset = dataset.batch(3, drop_remainder=True)

    dataset = global_shuffle_op._global_shuffle(dataset)

    got = self.getDatasetOutput(dataset, requires_initialization=True)
    expected = [
        np.array([0, 1, 2], dtype=np.int32),
        np.array([3, 4, 5], dtype=np.int32),
        np.array([6, 7, 8], dtype=np.int32),
    ]

    self.assertIsInstance(got, list)
    # Converts to tuples for lexicographically sort
    got.sort(key=tuple)

    self.assertLen(got, len(expected))

    for element_got, element_expected in zip(got, expected):
      self.assertAllEqual(element_got, element_expected)

  @combinations.generate(test_base.default_test_combinations())
  def testNestedConcatenateShuffledOutput(self):
    dataset1 = dataset_ops.Dataset.range(0, 3)
    dataset2 = dataset_ops.Dataset.range(3, 6)
    dataset3 = dataset_ops.Dataset.range(6, 9)

    dataset = dataset1.concatenate(dataset2)
    dataset = dataset.concatenate(dataset3)

    dataset = global_shuffle_op._global_shuffle(dataset)

    output = self.getDatasetOutput(dataset, requires_initialization=True)
    self.assertCountEqual(output, range(0, 9))


class ConcatenateGlobalShuffleCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase
):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_ranges=[(10, 8), (9, 5), (4, 7), (5, 8)],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False],
          ),
      )
  )
  def testConcatenate(
      self,
      verify_fn: Callable[..., None],
      dataset_ranges: Tuple[int, int],
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool,
  ):

    def _build_dataset():
      first_dataset = dataset_ops.Dataset.range(dataset_ranges[0])
      second_dataset = dataset_ops.Dataset.range(
          dataset_ranges[0], dataset_ranges[0] + dataset_ranges[1]
      )
      dataset = first_dataset.concatenate(second_dataset)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=10, reshuffle_each_iteration=reshuffle_each_iteration
      )

      options = options_lib.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=sum(dataset_ranges),
        assert_items_equal=reshuffle_each_iteration,
    )

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_ranges=[(10, 8, 11), (9, 5, 3)],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False],
          ),
      )
  )
  def testNestedConcatenate(
      self,
      verify_fn: Callable[..., None],
      dataset_ranges: Tuple[int, int],
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool,
  ):

    def _build_dataset():
      first_dataset = dataset_ops.Dataset.range(dataset_ranges[0])
      second_dataset = dataset_ops.Dataset.range(
          dataset_ranges[0], dataset_ranges[0] + dataset_ranges[1]
      )
      third_dataset = dataset_ops.Dataset.range(
          sum(dataset_ranges[:2]), sum(dataset_ranges[:3])
      )

      dataset = first_dataset.concatenate(second_dataset)
      dataset = dataset.concatenate(third_dataset)

      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=10, reshuffle_each_iteration=reshuffle_each_iteration
      )

      options = options_lib.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=sum(dataset_ranges),
        assert_items_equal=reshuffle_each_iteration,
    )

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_ranges=[(3, 4, 6, 5)],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False],
          ),
      )
  )
  def testFourNestedConcatenate(
      self,
      verify_fn: Callable[..., None],
      dataset_ranges: Tuple[int, int],
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool,
  ):
    def _build_dataset():
      first_dataset = dataset_ops.Dataset.range(dataset_ranges[0])
      second_dataset = dataset_ops.Dataset.range(
          dataset_ranges[0], sum(dataset_ranges[:2])
      )
      third_dataset = dataset_ops.Dataset.range(
          sum(dataset_ranges[:2]), sum(dataset_ranges[:3])
      )
      fourth_dataset = dataset_ops.Dataset.range(
          sum(dataset_ranges[:3]), sum(dataset_ranges)
      )

      left = first_dataset.concatenate(second_dataset)
      right = third_dataset.concatenate(fourth_dataset)

      dataset = left.concatenate(right)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=10, reshuffle_each_iteration=reshuffle_each_iteration
      )

      options = options_lib.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=sum(dataset_ranges),
        assert_items_equal=reshuffle_each_iteration,
    )

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_ranges=[(1, 2, 3, 4, 5, 6)],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False],
          ),
      )
  )
  def testDeepConcatenate(
      self,
      verify_fn: Callable[..., None],
      dataset_ranges: Tuple[int, ...],
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool,
  ):
    def _build_dataset():
      prefix_sums = [0] * (len(dataset_ranges) + 1)
      for i, value in enumerate(dataset_ranges):
        prefix_sums[i + 1] = prefix_sums[i] + value

      dataset = dataset_ops.Dataset.range(prefix_sums[0], prefix_sums[1])
      for i in range(1, len(dataset_ranges)):
        to_concat = dataset_ops.Dataset.range(
            prefix_sums[i], prefix_sums[i + 1]
        )
        dataset = dataset.concatenate(to_concat)

      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=10, reshuffle_each_iteration=reshuffle_each_iteration
      )

      options = options_lib.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=sum(dataset_ranges),
        assert_items_equal=reshuffle_each_iteration,
    )

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_ranges=[(1, 2, 3, 4, 5, 6)],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False],
          ),
      )
  )
  def testDeepConcatenateWithBatchAndPrefetch(
      self,
      verify_fn: Callable[..., None],
      dataset_ranges: Tuple[int, ...],
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool,
  ):
    def _build_dataset():
      prefix_sums = [0] * (len(dataset_ranges) + 1)
      for i, value in enumerate(dataset_ranges):
        prefix_sums[i + 1] = prefix_sums[i] + value

      dataset = dataset_ops.Dataset.range(prefix_sums[0], prefix_sums[1])
      for i in range(1, len(dataset_ranges)):
        to_concat = dataset_ops.Dataset.range(
            prefix_sums[i], prefix_sums[i + 1]
        )
        dataset = dataset.concatenate(to_concat)

      dataset = dataset.batch(2, drop_remainder=True)
      dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)

      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=10, reshuffle_each_iteration=reshuffle_each_iteration
      )
      dataset = dataset.unbatch()

      options = options_lib.Options()
      options.experimental_optimization.apply_default_optimizations = False
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      return dataset.with_options(options)

    verify_fn(
        self,
        _build_dataset,
        num_outputs=(sum(dataset_ranges) // 2) * 2,
        assert_items_equal=reshuffle_each_iteration,
    )


if __name__ == "__main__":
  test.main()
