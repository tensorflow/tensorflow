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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


class InterleaveDatasetTest(test.TestCase):

  def _interleave(self, lists, cycle_length, block_length):
    num_open = 0

    # `all_iterators` acts as a queue of iterators over each element of `lists`.
    all_iterators = [iter(l) for l in lists]

    # `open_iterators` are the iterators whose elements are currently being
    # interleaved.
    open_iterators = []
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

  def testPythonImplementation(self):
    input_lists = [[4, 4, 4, 4], [5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6],
                   [4, 4, 4, 4], [5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]]

    # Cycle length 1 acts like `Dataset.flat_map()`.
    expected_elements = itertools.chain(*input_lists)
    for expected, produced in zip(
        expected_elements, self._interleave(input_lists, 1, 1)):
      self.assertEqual(expected, produced)

    # Cycle length > 1.
    expected_elements = [4, 5, 4, 5, 4, 5, 4,
                         5, 5, 6, 6,  # NOTE(mrry): When we cycle back
                                      # to a list and are already at
                                      # the end of that list, we move
                                      # on to the next element.
                         4, 6, 4, 6, 4, 6, 4, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5]
    for expected, produced in zip(
        expected_elements, self._interleave(input_lists, 2, 1)):
      self.assertEqual(expected, produced)

    # Cycle length > 1 and block length > 1.
    expected_elements = [4, 4, 4, 5, 5, 5, 4, 5, 5, 6, 6, 6, 4, 4, 4, 6, 6, 6,
                         4, 5, 5, 5, 6, 6, 6, 5, 5, 6, 6, 6]
    for expected, produced in zip(
        expected_elements, self._interleave(input_lists, 2, 3)):
      self.assertEqual(expected, produced)

    # Cycle length > len(input_values).
    expected_elements = [4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6,
                         4, 4, 5, 5, 6, 6, 5, 6, 6, 5, 6, 6]
    for expected, produced in zip(
        expected_elements, self._interleave(input_lists, 7, 2)):
      self.assertEqual(expected, produced)

  def testInterleaveDataset(self):
    input_values = array_ops.placeholder(dtypes.int64, shape=[None])
    cycle_length = array_ops.placeholder(dtypes.int64, shape=[])
    block_length = array_ops.placeholder(dtypes.int64, shape=[])

    repeat_count = 2

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(input_values)
        .repeat(repeat_count)
        .interleave(lambda x: dataset_ops.Dataset.from_tensors(x).repeat(x),
                    cycle_length, block_length))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    next_element = iterator.get_next()

    with self.test_session() as sess:
      # Cycle length 1 acts like `Dataset.flat_map()`.
      sess.run(init_op, feed_dict={input_values: [4, 5, 6],
                                   cycle_length: 1, block_length: 3})

      for expected_element in self._interleave(
          [[4] * 4, [5] * 5, [6] * 6] * repeat_count, 1, 3):
        self.assertEqual(expected_element, sess.run(next_element))

      # Cycle length > 1.
      # expected: [4, 5, 4, 5, 4, 5, 4, 5, 5, 6, 6, 4, 6, 4, 6, 4, 6, 4, 6, 5,
      #            6, 5, 6, 5, 6, 5, 6, 5]
      sess.run(init_op, feed_dict={input_values: [4, 5, 6],
                                   cycle_length: 2, block_length: 1})
      for expected_element in self._interleave(
          [[4] * 4, [5] * 5, [6] * 6] * repeat_count, 2, 1):
        self.assertEqual(expected_element, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

      # Cycle length > 1 and block length > 1.
      # expected: [4, 4, 4, 5, 5, 5, 4, 5, 5, 6, 6, 6, 4, 4, 4, 6, 6, 6, 4, 5,
      #            5, 5, 6, 6, 6, 5, 5, 6, 6, 6]
      sess.run(init_op, feed_dict={input_values: [4, 5, 6],
                                   cycle_length: 2, block_length: 3})
      for expected_element in self._interleave(
          [[4] * 4, [5] * 5, [6] * 6] * repeat_count, 2, 3):
        self.assertEqual(expected_element, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

      # Cycle length > len(input_values) * repeat_count.
      # expected: [4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6, 4, 4,
      #            5, 5, 6, 6, 5, 6, 6, 5, 6, 6]
      sess.run(init_op, feed_dict={input_values: [4, 5, 6],
                                   cycle_length: 7, block_length: 2})
      for expected_element in self._interleave(
          [[4] * 4, [5] * 5, [6] * 6] * repeat_count, 7, 2):
        self.assertEqual(expected_element, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

      # Empty input.
      sess.run(init_op, feed_dict={input_values: [],
                                   cycle_length: 2, block_length: 3})
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

      # Non-empty input leading to empty output.
      sess.run(init_op, feed_dict={input_values: [0, 0, 0],
                                   cycle_length: 2, block_length: 3})
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

      # Mixture of non-empty and empty interleaved datasets.
      sess.run(init_op, feed_dict={input_values: [4, 0, 6],
                                   cycle_length: 2, block_length: 3})
      for expected_element in self._interleave(
          [[4] * 4, [], [6] * 6] * repeat_count, 2, 3):
        self.assertEqual(expected_element, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testSparse(self):

    def _map_fn(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0], [1, 1]], values=(i * [1, -1]), dense_shape=[2, 2])

    def _interleave_fn(x):
      return dataset_ops.Dataset.from_tensor_slices(
          sparse_ops.sparse_to_dense(x.indices, x.dense_shape, x.values))

    iterator = (
        dataset_ops.Dataset.range(10).map(_map_fn).interleave(
            _interleave_fn, cycle_length=1).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for i in range(10):
        for j in range(2):
          expected = [i, 0] if j % 2 == 0 else [0, -i]
          self.assertAllEqual(expected, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testEmptyInput(self):
    iterator = (
        dataset_ops.Dataset.from_tensor_slices([])
        .repeat(None)
        .interleave(dataset_ops.Dataset.from_tensors, cycle_length=2)
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
