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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
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


if __name__ == "__main__":
  test.main()
