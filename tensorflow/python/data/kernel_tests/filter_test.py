# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.Dataset.filter()`."""

from typing import Callable

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import global_shuffle_op
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def _test_combinations():

  def filter_fn(dataset, predicate):
    return dataset.filter(predicate)

  def legacy_filter_fn(dataset, predicate):
    return dataset.filter_with_legacy_function(predicate)

  filter_combinations = combinations.combine(
      tf_api_version=[1, 2],
      mode=["eager", "graph"],
      apply_filter=combinations.NamedObject("filter_fn", filter_fn))

  legacy_filter_combinations = combinations.combine(
      tf_api_version=1,
      mode=["eager", "graph"],
      apply_filter=combinations.NamedObject("legacy_filter_fn",
                                            legacy_filter_fn))

  return filter_combinations + legacy_filter_combinations


class FilterTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(_test_combinations())
  def testFilterDataset(self, apply_filter):
    components = (np.arange(7, dtype=np.int64),
                  np.array([[1, 2, 3]], dtype=np.int64) *
                  np.arange(7, dtype=np.int64)[:, np.newaxis],
                  np.array(37.0, dtype=np.float64) * np.arange(7))

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    def do_test(count, modulus):  # pylint: disable=missing-docstring
      dataset = dataset_ops.Dataset.from_tensor_slices(components).map(
          _map_fn).repeat(count)
      # pylint: disable=g-long-lambda
      dataset = apply_filter(
          dataset,
          lambda x, _y, _z: math_ops.equal(math_ops.mod(x, modulus), 0))
      # pylint: enable=g-long-lambda
      self.assertEqual(
          [c.shape[1:] for c in components],
          [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
      get_next = self.getNext(dataset)
      for _ in range(count):
        for i in [x for x in range(7) if x**2 % modulus == 0]:
          result = self.evaluate(get_next())
          for component, result_component in zip(components, result):
            self.assertAllEqual(component[i]**2, result_component)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())

    do_test(14, 2)
    do_test(4, 18)

    # Test an empty dataset.
    do_test(0, 1)

  @combinations.generate(_test_combinations())
  def testFilterRange(self, apply_filter):
    dataset = dataset_ops.Dataset.range(4)
    dataset = apply_filter(dataset,
                           lambda x: math_ops.not_equal(math_ops.mod(x, 3), 2))
    self.assertDatasetProduces(dataset, expected_output=[0, 1, 3])

  @combinations.generate(_test_combinations())
  def testFilterDict(self, apply_filter):
    dataset = dataset_ops.Dataset.range(10).map(
        lambda x: {"foo": x * 2, "bar": x**2})
    dataset = apply_filter(dataset, lambda d: math_ops.equal(d["bar"] % 2, 0))
    dataset = dataset.map(lambda d: d["foo"] + d["bar"])
    self.assertDatasetProduces(
        dataset,
        expected_output=[(i * 2 + i**2) for i in range(10) if not (i**2) % 2])

  @combinations.generate(_test_combinations())
  def testUseStepContainerInFilter(self, apply_filter):
    input_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

    # Define a predicate that returns true for the first element of
    # the sequence and not the second, and uses `tf.map_fn()`.
    def _predicate(xs):
      squared_xs = map_fn.map_fn(lambda x: x * x, xs)
      summed = math_ops.reduce_sum(squared_xs)
      return math_ops.equal(summed, 1 + 4 + 9)

    dataset = dataset_ops.Dataset.from_tensor_slices([[1, 2, 3], [4, 5, 6]])
    dataset = apply_filter(dataset, _predicate)
    self.assertDatasetProduces(dataset, expected_output=[input_data[0]])

  @combinations.generate(_test_combinations())
  def testSparse(self, apply_filter):

    def _map_fn(i):
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1])), i

    def _filter_fn(_, i):
      return math_ops.equal(i % 2, 0)

    dataset = dataset_ops.Dataset.range(10).map(_map_fn)
    dataset = apply_filter(dataset, _filter_fn)
    dataset = dataset.map(lambda x, i: x)
    self.assertDatasetProduces(
        dataset, expected_output=[_map_fn(i * 2)[0] for i in range(5)])

  @combinations.generate(_test_combinations())
  def testShortCircuit(self, apply_filter):
    dataset = dataset_ops.Dataset.zip(
        (dataset_ops.Dataset.range(10),
         dataset_ops.Dataset.from_tensors(True).repeat(None)))
    dataset = apply_filter(dataset, lambda x, y: y)
    self.assertDatasetProduces(
        dataset, expected_output=[(i, True) for i in range(10)])

  @combinations.generate(_test_combinations())
  def testParallelFilters(self, apply_filter):
    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_filter(dataset, lambda x: math_ops.equal(x % 2, 0))
    next_elements = [self.getNext(dataset) for _ in range(10)]
    self.assertEqual([0 for _ in range(10)],
                     self.evaluate(
                         [next_element() for next_element in next_elements]))

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    dataset = dataset_ops.Dataset.from_tensors(42).filter(
        lambda x: True, name="filter")
    self.assertDatasetProduces(dataset, [42])

  @combinations.generate(test_base.default_test_combinations())
  def testPredicateFailWithErrorContext(self):
    dataset = dataset_ops.Dataset.from_tensors(42).filter(
        lambda x: (x // 0) > 0, name="filter")
    get_next = self.getNext(dataset)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r".*Error in user-defined function passed to .* transformation with "
        r"iterator: Iterator::Root::.*"):
      self.evaluate(get_next())


class FilterCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                           parameterized.TestCase):

  def _build_filter_range_dataset(self, div, options=None):
    dataset = dataset_ops.Dataset.range(100).filter(
        lambda x: math_ops.not_equal(math_ops.mod(x, div), 2))
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def test(self, verify_fn, symbolic_checkpoint):
    div = 3
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    num_outputs = sum(x % 3 != 2 for x in range(100))
    verify_fn(self, lambda: self._build_filter_range_dataset(div, options),
              num_outputs)

  def _build_filter_dict_dataset(self):
    return dataset_ops.Dataset.range(10).map(lambda x: {
        "foo": x * 2,
        "bar": x**2
    }).filter(lambda d: math_ops.equal(d["bar"] % 2, 0)).map(
        lambda d: d["foo"] + d["bar"])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testDict(self, verify_fn):
    num_outputs = sum((x**2) % 2 == 0 for x in range(10))
    verify_fn(self, self._build_filter_dict_dataset, num_outputs)

  def _build_sparse_filter_dataset(self):

    def _map_fn(i):
      return sparse_tensor.SparseTensor(
          indices=[[0, 0]], values=(i * [1]), dense_shape=[1, 1]), i

    def _filter_fn(_, i):
      return math_ops.equal(i % 2, 0)

    return dataset_ops.Dataset.range(10).map(_map_fn).filter(_filter_fn).map(
        lambda x, i: x)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testSparse(self, verify_fn):
    verify_fn(self, self._build_sparse_filter_dataset, num_outputs=5)


class FilterGlobalShuffleTest(
    test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testShuffleFilter(self):
    dataset = dataset_ops.Dataset.range(100)
    dataset = global_shuffle_op._global_shuffle(dataset)
    dataset = dataset.filter(lambda x: math_ops.equal(x % 2, 0))
    self.assertDatasetProduces(
        dataset,
        list(range(0, 100, 2)),
        requires_initialization=True,
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testFilterShuffle(self):
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset.filter(lambda x: math_ops.equal(x % 2, 0))
    with self.assertRaisesRegex(
        errors.FailedPreconditionError,
        "`global_shuffle` requires all upstream transformations be compatible "
        "with random access."):
      dataset = global_shuffle_op._global_shuffle(dataset)
      self.getDatasetOutput(dataset, requires_initialization=True)


class FilterGlobalShuffleCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              reshuffle_each_iteration=[True, False],
              symbolic_checkpoint=[True, False])))
  def testShuffleFilter(
      self,
      verify_fn: Callable[..., None],
      reshuffle_each_iteration: bool,
      symbolic_checkpoint: bool):

    def _build_dataset() -> dataset_ops.Dataset:
      dataset = dataset_ops.Dataset.range(10)
      dataset = global_shuffle_op._global_shuffle(
          dataset, seed=42, reshuffle_each_iteration=reshuffle_each_iteration)
      dataset = dataset.filter(lambda x: math_ops.equal(x % 2, 0))
      if symbolic_checkpoint:
        options = options_lib.Options()
        options.experimental_symbolic_checkpoint = symbolic_checkpoint
        dataset = dataset.with_options(options)
      return dataset

    verify_fn(
        self,
        _build_dataset,
        num_outputs=5,
        assert_items_equal=reshuffle_each_iteration)


if __name__ == "__main__":
  test.main()
