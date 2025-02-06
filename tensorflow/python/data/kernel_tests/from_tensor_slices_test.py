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
"""Tests for `tf.data.Dataset.from_tensor_slices()."""
import collections
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
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


class FromTensorSlicesTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorSlicesEmptyComponent(self):
    components = ()

    with self.assertRaises(ValueError):
      dataset_ops.Dataset.from_tensor_slices(components)

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorSlices(self):
    """Test a dataset that represents the slices from a tuple of tensors."""
    components = (
        np.tile(np.array([[1], [2], [3], [4]]), 20), np.tile(
            np.array([[12], [13], [14], [15]]), 22),
        np.array([37.0, 38.0, 39.0, 40.0])
    )

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    get_next = self.getNext(dataset)

    self.assertEqual(
        [c.shape[1:] for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])

    for i in range(4):
      results = self.evaluate(get_next())
      for component, result_component in zip(components, results):
        self.assertAllEqual(component[i], result_component)
    with self.assertRaises(errors.OutOfRangeError):
      results = self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorSlicesDataset(self):
    dss = [dataset_ops.Dataset.range(10) for _ in range(10)]
    ds = dataset_ops.Dataset.from_tensor_slices(dss)
    ds = ds.flat_map(lambda x: x)
    self.assertDatasetProduces(ds, expected_output=list(range(10)) * 10)

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorSlicesDatasetOfOrderedDict(self):
    dss = [dataset_ops.Dataset.range(10).map(
        lambda x: collections.OrderedDict([("x", x)])) for _ in range(10)]
    ds = dataset_ops.Dataset.from_tensor_slices(dss)
    ds = ds.flat_map(lambda x: x)
    self.assertDatasetProduces(
        ds,
        expected_output=[collections.OrderedDict([("x", x)])
                         for x in list(range(10)) * 10])

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorSlicesDatasetInFunction(self):
    dss = [dataset_ops.Dataset.range(10) for _ in range(10)]
    ds = dataset_ops.Dataset.from_tensors(dss)
    ds = ds.flat_map(dataset_ops.Dataset.from_tensor_slices)
    ds = ds.flat_map(lambda x: x)
    self.assertDatasetProduces(ds, expected_output=list(range(10)) * 10)

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorSlicesSparse(self):
    """Test a dataset that represents the slices from a tuple of tensors."""
    components = (sparse_tensor.SparseTensorValue(
        indices=np.array([[0, 0], [1, 0], [2, 0]]),
        values=np.array([0, 0, 0]),
        dense_shape=np.array([3, 1])),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 1], [2, 2]]),
                      values=np.array([1, 2, 3]),
                      dense_shape=np.array([3, 3])))

    dataset = dataset_ops.Dataset.from_tensor_slices(components)

    self.assertEqual(
        [tensor_shape.TensorShape(c.dense_shape[1:]) for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])

    expected = [
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[0]]),
             values=np.array([1]),
             dense_shape=np.array([3]))),
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[1]]),
             values=np.array([2]),
             dense_shape=np.array([3]))),
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[2]]),
             values=np.array([3]),
             dense_shape=np.array([3]))),
    ]
    self.assertDatasetProduces(dataset, expected_output=expected)

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorSlicesMixed(self):
    """Test a dataset that represents the slices from a tuple of tensors."""
    components = (np.tile(np.array([[1], [2], [3]]), 20),
                  np.tile(np.array([[12], [13], [14]]), 22),
                  np.array([37.0, 38.0, 39.0]),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 0], [2, 0]]),
                      values=np.array([0, 0, 0]),
                      dense_shape=np.array([3, 1])),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 1], [2, 2]]),
                      values=np.array([1, 2, 3]),
                      dense_shape=np.array([3, 3])))

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    get_next = self.getNext(dataset)
    self.assertEqual([
        tensor_shape.TensorShape(c.dense_shape[1:])
        if sparse_tensor.is_sparse(c) else c.shape[1:] for c in components
    ], [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])

    expected = [
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[0]]),
             values=np.array([1]),
             dense_shape=np.array([3]))),
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[1]]),
             values=np.array([2]),
             dense_shape=np.array([3]))),
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[2]]),
             values=np.array([3]),
             dense_shape=np.array([3]))),
    ]
    for i in range(3):
      results = self.evaluate(get_next())
      for component, result_component in zip(
          (list(zip(*components[:3]))[i] + expected[i]), results):
        self.assertValuesEqual(component, result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorSlicesWithDict(self):
    components = {"foo": [1, 2, 3], "bar": [[4.0], [5.0], [6.0]]}
    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    get_next = self.getNext(dataset)

    self.assertEqual(dtypes.int32,
                     dataset_ops.get_legacy_output_types(dataset)["foo"])
    self.assertEqual(dtypes.float32,
                     dataset_ops.get_legacy_output_types(dataset)["bar"])
    self.assertEqual((), dataset_ops.get_legacy_output_shapes(dataset)["foo"])
    self.assertEqual((1,), dataset_ops.get_legacy_output_shapes(dataset)["bar"])

    for i in range(3):
      results = self.evaluate(get_next())
      self.assertEqual(components["foo"][i], results["foo"])
      self.assertEqual(components["bar"][i], results["bar"])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorSlicesRagged(self):
    components = (
        ragged_factory_ops.constant_value([[[0]], [[1]], [[2]]]),
        ragged_factory_ops.constant_value([[[3]], [[4]], [[5]]]),
    )
    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    expected = [(ragged_factory_ops.constant_value([[0]]),
                 ragged_factory_ops.constant_value([[3]])),
                (ragged_factory_ops.constant_value([[1]]),
                 ragged_factory_ops.constant_value([[4]])),
                (ragged_factory_ops.constant_value([[2]]),
                 ragged_factory_ops.constant_value([[5]]))]
    self.assertDatasetProduces(dataset, expected_output=expected)

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorSlicesMixedRagged(self):
    components = (np.tile(np.array([[1], [2], [3]]),
                          20), np.tile(np.array([[12], [13], [14]]),
                                       22), np.array([37.0, 38.0, 39.0]),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 0], [2, 0]]),
                      values=np.array([0, 0, 0]),
                      dense_shape=np.array([3, 1])),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 1], [2, 2]]),
                      values=np.array([1, 2, 3]),
                      dense_shape=np.array([3, 3])),
                  ragged_factory_ops.constant_value([[[0]], [[1]], [[2]]]))

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    get_next = self.getNext(dataset)

    expected = [
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[0]]),
             values=np.array([1]),
             dense_shape=np.array([3])), ragged_factory_ops.constant_value([[0]
                                                                           ])),
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[1]]),
             values=np.array([2]),
             dense_shape=np.array([3])), ragged_factory_ops.constant_value([[1]
                                                                           ])),
        (sparse_tensor.SparseTensorValue(
            indices=np.array([[0]]),
            values=np.array([0]),
            dense_shape=np.array([1])),
         sparse_tensor.SparseTensorValue(
             indices=np.array([[2]]),
             values=np.array([3]),
             dense_shape=np.array([3])), ragged_factory_ops.constant_value([[2]
                                                                           ])),
    ]
    for i in range(3):
      results = self.evaluate(get_next())
      for component, result_component in zip(
          (list(zip(*components[:3]))[i] + expected[i]), results):
        self.assertValuesEqual(component, result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorSlicesWithUintDtypes(self):
    components = (
        np.tile(np.array([[0], [1]], dtype=np.uint8), 2),
        np.tile(np.array([[2], [256]], dtype=np.uint16), 2),
        np.tile(np.array([[4], [65536]], dtype=np.uint32), 2),
        np.tile(np.array([[8], [4294967296]], dtype=np.uint64), 2),
    )
    expected_types = (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
    expected_output = [tuple([c[i] for c in components]) for i in range(2)]

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    self.assertEqual(expected_types,
                     dataset_ops.get_legacy_output_types(dataset))
    self.assertDatasetProduces(dataset, expected_output)


class FromTensorSlicesRandomAccessTest(test_base.DatasetTestBase,
                                       parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testInvalidIndex(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, -1))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 3))

  @combinations.generate(test_base.default_test_combinations())
  def testOneDimensionalArray(self):
    tensor = [1, 2, 3]
    dataset = dataset_ops.Dataset.from_tensor_slices(tensor)
    for i in range(len(tensor)):
      results = self.evaluate(random_access.at(dataset, i))
      self.assertAllEqual(tensor[i], results)

  @combinations.generate(test_base.default_test_combinations())
  def testTwoDimensionalArray(self):
    tensor = [[1, 2], [3, 4]]
    dataset = dataset_ops.Dataset.from_tensor_slices(tensor)
    for i in range(2):
      results = self.evaluate(random_access.at(dataset, i))
      self.assertAllEqual(tensor[i], results)

  @combinations.generate(test_base.default_test_combinations())
  def testMultipleComponents(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(([1, 2], [3, 4], [5, 6]))
    self.assertEqual((1, 3, 5), self.evaluate(random_access.at(dataset, 0)))
    self.assertEqual((2, 4, 6), self.evaluate(random_access.at(dataset, 1)))

  @combinations.generate(test_base.default_test_combinations())
  def testDictionary(self):
    dataset = dataset_ops.Dataset.from_tensor_slices({"a": [1, 2], "b": [3, 4]})
    self.assertEqual({
        "a": 1,
        "b": 3
    }, self.evaluate(random_access.at(dataset, 0)))
    self.assertEqual({
        "a": 2,
        "b": 4
    }, self.evaluate(random_access.at(dataset, 1)))

  @combinations.generate(test_base.default_test_combinations())
  def testNumpy(self):
    components = (
        np.tile(np.array([[0], [1]], dtype=np.uint8), 2),
        np.tile(np.array([[2], [256]], dtype=np.uint16), 2),
        np.tile(np.array([[4], [65536]], dtype=np.uint32), 2),
        np.tile(np.array([[8], [4294967296]], dtype=np.uint64), 2),
    )
    expected_output = [tuple([c[i] for c in components]) for i in range(2)]

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    for i in range(2):
      result = self.evaluate(random_access.at(dataset, i))
      self.assertAllEqual(expected_output[i], result)

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([42],
                                                     name="from_tensor_slices")
    self.assertDatasetProduces(dataset, [42])


class FromTensorSlicesCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                     parameterized.TestCase):

  def _build_tensor_slices_dataset(self, components, options=None):
    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def test(self, verify_fn, symbolic_checkpoint):
    # Equal length components
    components = (np.tile(np.array([[1], [2], [3], [4]]),
                          20), np.tile(np.array([[12], [13], [14], [15]]),
                                       22), np.array([37.0, 38.0, 39.0, 40.0]))
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(
        self,
        lambda: self._build_tensor_slices_dataset(components, options),
        num_outputs=4)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testDict(self, verify_fn):
    dict_components = {"foo": [1, 2, 3], "bar": [[4.0], [5.0], [6.0]]}

    verify_fn(
        self,
        lambda: self._build_tensor_slices_dataset(dict_components),
        num_outputs=3)


class FromTensorSlicesGlobalShuffleTest(
    test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[10, 100],
              repetitions=[1, 3],
              seed=[None, 19],
              reshuffle_each_iteration=[True, False])))
  def testGlobalShuffleTensorSlicesDataset(
      self,
      dataset_range: int,
      repetitions: int,
      seed: Optional[int],
      reshuffle_each_iteration: bool):
    dataset = dataset_ops.Dataset.from_tensor_slices(list(range(dataset_range)))
    if repetitions > 1:
      dataset = dataset.repeat(repetitions)
    dataset = global_shuffle_op._global_shuffle(
        dataset, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)
    dataset_output = self.getDatasetOutput(
        dataset, requires_initialization=True)

    expected = list(range(dataset_range)) * repetitions
    self.assertCountEqual(dataset_output, expected)
    self.assertNotEqual(dataset_output, expected)
    self.assertLen(expected, self.evaluate(dataset.cardinality()))


class FromTensorSlicesGlobalShuffleCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[10],
              repetitions=[1, 3],
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False])))
  def testGlobalShuffleTensorSlicesDataset(
      self,
      verify_fn: Callable[..., None],
      dataset_range: int,
      repetitions: int,
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool):

    def _build_dataset() -> dataset_ops.Dataset:
      dataset = dataset_ops.Dataset.from_tensor_slices(
          list(range(dataset_range)))
      if repetitions > 1:
        dataset = dataset.repeat(repetitions)
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
        num_outputs=dataset_range * repetitions,
        assert_items_equal=reshuffle_each_iteration)


if __name__ == "__main__":
  test.main()
