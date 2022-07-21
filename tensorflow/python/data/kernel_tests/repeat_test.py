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
"""Tests for `tf.data.Dataset.repeat()`."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class RepeatTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(count=[0, 3, 7])))
  def testFiniteRepeat(self, count):
    """Test a dataset that repeats its input multiple times."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
    dataset = dataset_ops.Dataset.from_tensors(components).repeat(count)
    self.assertEqual(
        [c.shape for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    self.assertDatasetProduces(dataset, [components] * count)

  @combinations.generate(test_base.default_test_combinations())
  def testInfiniteRepeat(self):
    # NOTE(mrry): There's not a good way to test that the sequence is infinite.
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
    dataset = dataset_ops.Dataset.from_tensors(components).repeat(-1)
    self.assertEqual(
        [c.shape for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    get_next = self.getNext(dataset)
    for _ in range(17):
      results = self.evaluate(get_next())
      for component, result_component in zip(components, results):
        self.assertAllEqual(component, result_component)

  @combinations.generate(test_base.default_test_combinations())
  def testRepeatRepeat(self):
    """Test the composition of repeat datasets."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
    inner_count, outer_count = 7, 14

    dataset = dataset_ops.Dataset.from_tensors(components).repeat(
        inner_count).repeat(outer_count)
    self.assertEqual(
        [c.shape for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    self.assertDatasetProduces(dataset,
                               [components] * (inner_count * outer_count))

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    dataset = dataset_ops.Dataset.from_tensors(42).repeat(1, name="repeat")
    self.assertDatasetProduces(dataset, [42])


class RepeatDatasetCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                  parameterized.TestCase):

  def _build_repeat_dataset(self, count, take_count=3):
    components = (np.arange(10),)
    return dataset_ops.Dataset.from_tensor_slices(components).take(
        take_count).repeat(count)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testFiniteRepeat(self, verify_fn):
    count = 10
    verify_fn(
        self,
        lambda: self._build_repeat_dataset(count),
        num_outputs=(3 * count))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testEmptyRepeat(self, verify_fn):
    verify_fn(self, lambda: self._build_repeat_dataset(0), num_outputs=0)

  @combinations.generate(test_base.default_test_combinations())
  def testInfiniteRepeat(self):
    self.verify_unused_iterator(
        lambda: self._build_repeat_dataset(-1), 10, verify_exhausted=False)
    self.verify_multiple_breaks(
        lambda: self._build_repeat_dataset(-1), 20, verify_exhausted=False)
    self.verify_reset_restored_iterator(
        lambda: self._build_repeat_dataset(-1), 20, verify_exhausted=False)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testInfiniteEmptyRepeat(self, verify_fn):
    verify_fn(self, lambda: self._build_repeat_dataset(-1, 0), num_outputs=0)


class RepeatRandomAccessTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 6, 7])))
  def testInvalidIndex(self, index):
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3]).repeat(2)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(index=[-1, 0])))
  def testEmptyDataset(self, index):
    dataset = dataset_ops.Dataset.from_tensor_slices([]).repeat(2)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(elements=[0, 5, 10],
                                              count=[0, 3, 8])))
  def testFiniteRepeat(self, elements, count):
    dataset = dataset_ops.Dataset.range(elements).repeat(count)
    expected_dataset = np.tile(
        np.arange(
            start=0, stop=elements, step=1, dtype=dtypes.int64.as_numpy_dtype),
        count)
    for i in range(elements * count):
      self.assertEqual(
          self.evaluate(random_access.at(dataset, index=i)),
          expected_dataset[i])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              elements=[0, 3, 5], count_1=[0, 1, 2], count_2=[3, 4, 5])))
  def testRepeatRepeat(self, elements, count_1, count_2):
    dataset = dataset_ops.Dataset.range(elements).repeat(count_1).repeat(
        count_2)
    expected_dataset = np.tile(
        np.arange(
            start=0, stop=elements, step=1, dtype=dtypes.int64.as_numpy_dtype),
        count_1 * count_2)
    for i in range(elements * count_1 * count_2):
      self.assertEqual(
          self.evaluate(random_access.at(dataset, index=i)),
          expected_dataset[i])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(elements=[3, 5], count=[None, -1, -2])))
  def testInfiniteRepeat(self, elements, count):
    dataset = dataset_ops.Dataset.range(elements).repeat(count=count)

    # Datasets with infinite cardinality do not support random access.
    with self.assertRaises(errors.FailedPreconditionError):
      self.evaluate(random_access.at(dataset, index=0))


if __name__ == "__main__":
  test.main()
