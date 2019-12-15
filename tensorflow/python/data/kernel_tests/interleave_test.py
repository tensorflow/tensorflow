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
"""Tests for `tf.data.Dataset.interleave()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


def _interleave(lists, cycle_length, block_length):
  """Reference implementation of interleave used for testing.

  Args:
    lists: a list of lists to interleave
    cycle_length: the length of the interleave cycle
    block_length: the length of the interleave block

  Yields:
    Elements of `lists` interleaved in the order determined by `cycle_length`
    and `block_length`.
  """
  num_open = 0

  # `all_iterators` acts as a queue of iterators over each element of `lists`.
  all_iterators = [iter(l) for l in lists]

  # `open_iterators` are the iterators whose elements are currently being
  # interleaved.
  open_iterators = []
  if cycle_length == dataset_ops.AUTOTUNE:
    cycle_length = multiprocessing.cpu_count()
  for i in range(cycle_length):
    if all_iterators:
      open_iterators.append(all_iterators.pop(0))
      num_open += 1
    else:
      open_iterators.append(None)

  while num_open or all_iterators:
    for i in range(cycle_length):
      if open_iterators[i] is None:
        if all_iterators:
          open_iterators[i] = all_iterators.pop(0)
          num_open += 1
        else:
          continue
      for _ in range(block_length):
        try:
          yield next(open_iterators[i])
        except StopIteration:
          open_iterators[i] = None
          num_open -= 1
          break


def _repeat(values, count):
  """Produces a list of lists suitable for testing interleave.

  Args:
    values: for each element `x` the result contains `[x] * x`
    count: determines how many times to repeat `[x] * x` in the result

  Returns:
    A list of lists of values suitable for testing interleave.
  """
  return [[value] * value for value in np.tile(values, count)]


class InterleaveTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              input_values=[[4, 5, 6]],
              cycle_length=1,
              block_length=1,
              expected_elements=[[
                  4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 5, 5,
                  5, 5, 5, 6, 6, 6, 6, 6, 6
              ]]) + combinations.combine(
                  input_values=[[4, 5, 6]],
                  cycle_length=2,
                  block_length=1,
                  expected_elements=[[
                      4, 5, 4, 5, 4, 5, 4, 5, 5, 6, 6, 4, 6, 4, 6, 4, 6, 4, 6,
                      5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 6
                  ]]) + combinations.combine(
                      input_values=[[4, 5, 6]],
                      cycle_length=2,
                      block_length=3,
                      expected_elements=[[
                          4, 4, 4, 5, 5, 5, 4, 5, 5, 6, 6, 6, 4, 4, 4, 6, 6, 6,
                          4, 5, 5, 5, 6, 6, 6, 5, 5, 6, 6, 6
                      ]]) + combinations.combine(
                          input_values=[[4, 5, 6]],
                          cycle_length=7,
                          block_length=2,
                          expected_elements=[[
                              4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6,
                              6, 4, 4, 5, 5, 6, 6, 5, 6, 6, 5, 6, 6
                          ]]) +
          combinations.combine(
              input_values=[[4, 0, 6]],
              cycle_length=2,
              block_length=1,
              expected_elements=[[
                  4, 4, 6, 4, 6, 4, 6, 6, 4, 6, 4, 6, 4, 4, 6, 6, 6, 6, 6, 6
              ]])))
  def testPythonImplementation(self, input_values, cycle_length, block_length,
                               expected_elements):
    input_lists = _repeat(input_values, 2)

    for expected, produced in zip(
        expected_elements, _interleave(input_lists, cycle_length,
                                       block_length)):
      self.assertEqual(expected, produced)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              input_values=[np.int64([4, 5, 6])],
              cycle_length=1,
              block_length=3,
              num_parallel_calls=[None, 1]) + combinations.combine(
                  input_values=[np.int64([4, 5, 6])],
                  cycle_length=2,
                  block_length=[1, 3],
                  num_parallel_calls=[None, 1, 2]) + combinations.combine(
                      input_values=[np.int64([4, 5, 6])],
                      cycle_length=7,
                      block_length=2,
                      num_parallel_calls=[None, 1, 3, 5, 7]) +
          combinations.combine(
              input_values=[np.int64([4, 5, 6, 7])],
              cycle_length=dataset_ops.AUTOTUNE,
              block_length=3,
              num_parallel_calls=[None, 1]) + combinations.combine(
                  input_values=[np.int64([]), np.int64([0, 0, 0])],
                  cycle_length=2,
                  block_length=3,
                  num_parallel_calls=[None]) + combinations.combine(
                      input_values=[np.int64([4, 0, 6])],
                      cycle_length=2,
                      block_length=3,
                      num_parallel_calls=[None, 1, 2])))
  def testInterleaveDataset(self, input_values, cycle_length, block_length,
                            num_parallel_calls):
    count = 2
    dataset = dataset_ops.Dataset.from_tensor_slices(input_values).repeat(
        count).interleave(
            lambda x: dataset_ops.Dataset.from_tensors(x).repeat(x),
            cycle_length, block_length, num_parallel_calls)
    expected_output = [
        element for element in _interleave(
            _repeat(input_values, count), cycle_length, block_length)
    ]
    self.assertDatasetProduces(dataset, expected_output)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              input_values=[np.float32([1., np.nan, 2., np.nan, 3.])],
              cycle_length=1,
              block_length=3,
              num_parallel_calls=[None, 1]) + combinations.combine(
                  input_values=[np.float32([1., np.nan, 2., np.nan, 3.])],
                  cycle_length=2,
                  block_length=[1, 3],
                  num_parallel_calls=[None, 1, 2]) + combinations.combine(
                      input_values=[np.float32([1., np.nan, 2., np.nan, 3.])],
                      cycle_length=7,
                      block_length=2,
                      num_parallel_calls=[None, 1, 3, 5, 7])))
  def testInterleaveDatasetError(self, input_values, cycle_length, block_length,
                                 num_parallel_calls):
    dataset = dataset_ops.Dataset.from_tensor_slices(input_values).map(
        lambda x: array_ops.check_numerics(x, "message")).interleave(
            dataset_ops.Dataset.from_tensors, cycle_length, block_length,
            num_parallel_calls)
    get_next = self.getNext(dataset)

    for value in input_values:
      if np.isnan(value):
        with self.assertRaises(errors.InvalidArgumentError):
          self.evaluate(get_next())
      else:
        self.assertEqual(value, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testInterleaveSparse(self):

    def _map_fn(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0], [1, 1]], values=(i * [1, -1]), dense_shape=[2, 2])

    def _interleave_fn(x):
      return dataset_ops.Dataset.from_tensor_slices(
          sparse_ops.sparse_to_dense(x.indices, x.dense_shape, x.values))

    dataset = dataset_ops.Dataset.range(10).map(_map_fn).interleave(
        _interleave_fn, cycle_length=1)
    get_next = self.getNext(dataset)
    for i in range(10):
      for j in range(2):
        expected = [i, 0] if j % 2 == 0 else [0, -i]
        self.assertAllEqual(expected, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              input_values=[np.int64([4, 5, 6])],
              cycle_length=1,
              block_length=3,
              num_parallel_calls=1) + combinations.combine(
                  input_values=[np.int64([4, 5, 6])],
                  cycle_length=2,
                  block_length=[1, 3],
                  num_parallel_calls=[1, 2]) + combinations.combine(
                      input_values=[np.int64([4, 5, 6])],
                      cycle_length=7,
                      block_length=2,
                      num_parallel_calls=[1, 3, 5, 7]) + combinations.combine(
                          input_values=[np.int64([4, 5, 6, 7])],
                          cycle_length=dataset_ops.AUTOTUNE,
                          block_length=3,
                          num_parallel_calls=1) + combinations.combine(
                              input_values=[np.int64([4, 0, 6])],
                              cycle_length=2,
                              block_length=3,
                              num_parallel_calls=[1, 2])))
  def testSloppyInterleaveDataset(self, input_values, cycle_length,
                                  block_length, num_parallel_calls):
    count = 2
    dataset = dataset_ops.Dataset.from_tensor_slices(input_values).repeat(
        count).interleave(
            lambda x: dataset_ops.Dataset.from_tensors(x).repeat(x),
            cycle_length, block_length, num_parallel_calls)
    options = dataset_ops.Options()
    options.experimental_deterministic = False
    dataset = dataset.with_options(options)
    expected_output = [
        element for element in _interleave(
            _repeat(input_values, count), cycle_length, block_length)
    ]
    get_next = self.getNext(dataset)
    actual_output = []
    for _ in range(len(expected_output)):
      actual_output.append(self.evaluate(get_next()))
    self.assertAllEqual(expected_output.sort(), actual_output.sort())

  @combinations.generate(test_base.default_test_combinations())
  def testInterleaveMap(self):
    dataset = dataset_ops.Dataset.range(100)

    def interleave_fn(x):
      dataset = dataset_ops.Dataset.from_tensors(x)
      return dataset.map(lambda x: x + x)

    dataset = dataset.interleave(interleave_fn, cycle_length=5)
    dataset = dataset.interleave(interleave_fn, cycle_length=5)

    self.assertDatasetProduces(dataset, [4 * x for x in range(100)])

  @combinations.generate(test_base.default_test_combinations())
  def testParallelInterleaveCached(self):
    dataset = dataset_ops.Dataset.range(5)
    dataset = dataset.cache(os.path.join(self.get_temp_dir(), "cache_dir"))

    def interleave_fn(x):
      return dataset_ops.Dataset.from_tensors(x)

    dataset = dataset.interleave(
        interleave_fn, cycle_length=2, num_parallel_calls=2)
    self.assertDatasetProduces(dataset, list(range(5)))


if __name__ == "__main__":
  test.main()
