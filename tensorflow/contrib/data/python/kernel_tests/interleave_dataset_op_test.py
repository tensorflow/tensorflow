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
import math
import threading
import time

import numpy as np
from six.moves import zip_longest

from tensorflow.contrib.data.python.kernel_tests import dataset_serialization_test_base
from tensorflow.contrib.data.python.ops import dataset_ops
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
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


class InterleaveDatasetSeriazationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_iterator_graph(self, input_values, cycle_length, block_length):
    repeat_count = 2
    return dataset_ops.Dataset.from_tensor_slices(input_values).repeat(
        repeat_count).interleave(
            lambda x: dataset_ops.Dataset.from_tensors(x).repeat(x),
            cycle_length, block_length)

  def testSerializationCore(self):
    input_values = np.array([4, 5, 6], dtype=np.int64)
    num_outputs = np.sum(input_values) * 2
    # cycle_length > 1, block_length > 1
    cycle_length = 2
    block_length = 3
    # pylint: disable=g-long-lambda
    self.run_core_tests(
        lambda: self._build_iterator_graph(
            input_values, cycle_length, block_length),
        lambda: self._build_iterator_graph(
            input_values, cycle_length * 2, block_length * 1),
        num_outputs)
    # cycle_length = 1
    cycle_length = 1
    block_length = 3
    self.run_core_tests(
        lambda: self._build_iterator_graph(
            input_values, cycle_length, block_length),
        None, num_outputs)
    # block_length = 1
    cycle_length = 2
    block_length = 1
    self.run_core_tests(
        lambda: self._build_iterator_graph(
            input_values, cycle_length, block_length),
        None, num_outputs)
    # pylint: enable=g-long-lambda


class ParallelInterleaveDatasetTest(test.TestCase):

  def setUp(self):
    self.input_values = array_ops.placeholder(dtypes.int64, shape=[None])
    self.cycle_length = array_ops.placeholder(dtypes.int64, shape=[])
    self.block_length = array_ops.placeholder(dtypes.int64, shape=[])
    self.sloppy = array_ops.placeholder(dtypes.bool, shape=[])

    self.repeat_count = 2

    # Set up threading events used to sequence when items are produced that
    # are subsequently interleaved. These events allow us to deterministically
    # simulate slowdowns and force sloppiness.
    self.read_coordination_events = {}
    self.write_coordination_events = {}
    # input values [4, 5, 6] are the common case for the tests; set defaults
    for i in range(4, 7):
      self.read_coordination_events[i] = threading.Semaphore(0)
      self.write_coordination_events[i] = threading.Event()

    def map_py_fn(x):
      self.write_coordination_events[x].wait()
      self.write_coordination_events[x].clear()
      self.read_coordination_events[x].release()
      return x * x

    def map_fn(x):
      return script_ops.py_func(map_py_fn, [x], x.dtype)

    def interleave_fn(x):
      dataset = dataset_ops.Dataset.from_tensors(x)
      dataset = dataset.repeat(x)
      return dataset.map(map_fn)

    self.dataset = (dataset_ops.Dataset.from_tensor_slices(self.input_values)
                    .repeat(self.repeat_count).apply(
                        interleave_ops.parallel_interleave(
                            interleave_fn, self.cycle_length,
                            self.block_length, self.sloppy)))
    self.iterator = self.dataset.make_initializable_iterator()
    self.init_op = self.iterator.initializer
    self.next_element = self.iterator.get_next()

  def _interleave(self, lists, cycle_length, block_length):
    """Python implementation of interleave used for testing."""
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
    for expected, produced in zip(expected_elements,
                                  self._interleave(input_lists, 1, 1)):
      self.assertEqual(expected, produced)

    # Cycle length > 1.
    expected_elements = [
        4, 5, 4, 5, 4, 5, 4, 5, 5, 6, 6, 4, 6, 4, 6, 4, 6, 4, 6, 5, 6, 5, 6, 5,
        6, 5, 6, 5, 6, 6
    ]
    for index, (expected, produced) in enumerate(
        zip_longest(expected_elements, self._interleave(input_lists, 2, 1))):
      self.assertEqual(expected, produced, "Values differ at %s. %s != %s" %
                       (index, expected, produced))

  def testPythonImplementationBlockLength(self):
    input_lists = [[4] * 4, [5] * 5, [6] * 6] * 2
    expected_elements = [
        4, 4, 5, 5, 4, 4, 5, 5, 5, 6, 6, 4, 4, 6, 6, 4, 4, 6, 6, 5, 5, 6, 6, 5,
        5, 6, 6, 5, 6, 6
    ]
    for index, (expected, produced) in enumerate(
        zip_longest(expected_elements, self._interleave(input_lists, 2, 2))):
      self.assertEqual(expected, produced, "Values differ at %s. %s != %s" %
                       (index, expected, produced))

  def testPythonImplementationEmptyLists(self):
    input_lists = [[4, 4, 4, 4], [], [6, 6, 6, 6, 6, 6], [4, 4, 4, 4], [],
                   [6, 6, 6, 6, 6, 6]]

    expected_elements = [
        4, 4, 6, 4, 6, 4, 6, 6, 4, 6, 4, 6, 4, 4, 6, 6, 6, 6, 6, 6
    ]
    for index, (expected, produced) in enumerate(
        zip_longest(expected_elements, self._interleave(input_lists, 2, 1))):
      self.assertEqual(expected, produced, "Values differ at %s. %s != %s" %
                       (index, expected, produced))

  def _clear_coordination_events(self):
    for i in range(4, 7):
      self.read_coordination_events[i] = threading.Semaphore(0)
      self.write_coordination_events[i].clear()

  def _allow_all_map_threads(self):
    for i in range(4, 7):
      self.write_coordination_events[i].set()

  def _testSingleThreaded(self, sloppy=False):
    # cycle_length=1,block_length=1 acts like `Dataset.interleave()` and
    # `Dataset.flat_map()` and is single-threaded. No synchronization required.
    with self.test_session() as sess:
      self._clear_coordination_events()
      sess.run(
          self.init_op,
          feed_dict={
              self.input_values: [4, 5, 6],
              self.cycle_length: 1,
              self.block_length: 1,
              self.sloppy: sloppy
          })

      for expected_element in self._interleave(
          [[4] * 4, [5] * 5, [6] * 6] * self.repeat_count, 1, 1):
        self.write_coordination_events[expected_element].set()
        self.assertEqual(expected_element * expected_element,
                         sess.run(self.next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(self.next_element)

  def testSingleThreaded(self):
    self._testSingleThreaded()

  def testSingleThreadedSloppy(self):
    self._testSingleThreaded(sloppy=True)

  def _testTwoThreadsNoContention(self, sloppy=False):
    # num_threads > 1.
    # Explicit coordination should result in `Dataset.interleave()` behavior
    with self.test_session() as sess:
      self._clear_coordination_events()
      done_first_event = False
      sess.run(
          self.init_op,
          feed_dict={
              self.input_values: [4, 5, 6],
              self.cycle_length: 2,
              self.block_length: 1,
              self.sloppy: sloppy
          })
      for i, expected_element in enumerate(
          self._interleave([[4] * 4, [5] * 5, [6] * 6] * self.repeat_count, 2,
                           1)):
        self.write_coordination_events[expected_element].set()
        if done_first_event:  # First event starts the worker threads.
          self.read_coordination_events[expected_element].acquire()
        actual_element = sess.run(self.next_element)
        if not done_first_event:
          self.read_coordination_events[expected_element].acquire()
          done_first_event = True
        self.assertEqual(expected_element * expected_element, actual_element,
                         "At index %s: %s expected, got: %s" %
                         (i, expected_element, actual_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(self.next_element)

  def testTwoThreadsNoContention(self):
    self._testTwoThreadsNoContention()

  def testTwoThreadsNoContentionSloppy(self):
    self._testTwoThreadsNoContention(sloppy=True)

  def _testTwoThreadsNoContentionWithRaces(self, sloppy=False):
    """Tests where all the workers race in producing elements.

    Note: this is in contrast with the prevous test which carefully sequences
    the execution of the map functions.

    Args:
      sloppy: Whether to be sloppy or not.
    """
    with self.test_session() as sess:
      self._clear_coordination_events()
      done_first_event = False
      sess.run(
          self.init_op,
          feed_dict={
              self.input_values: [4, 5, 6],
              self.cycle_length: 2,
              self.block_length: 1,
              self.sloppy: sloppy,
          })
      for i, expected_element in enumerate(
          self._interleave([[4] * 4, [5] * 5, [6] * 6] * self.repeat_count, 2,
                           1)):
        if done_first_event:  # First event starts the worker threads.
          self._allow_all_map_threads()
          self.read_coordination_events[expected_element].acquire()
        else:
          self.write_coordination_events[expected_element].set()
        time.sleep(0.1)  # Sleep to consistently "avoid" the race condition.
        actual_element = sess.run(self.next_element)
        if not done_first_event:
          done_first_event = True
          self.assertTrue(
              self.read_coordination_events[expected_element].acquire(False))
        self.assertEqual(expected_element * expected_element, actual_element,
                         "At index %s: %s expected, got: %s" %
                         (i, expected_element, actual_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(self.next_element)

  def testTwoThreadsNoContentionWithRaces(self):
    self._testTwoThreadsNoContentionWithRaces()

  def testTwoThreadsNoContentionWithRacesSloppy(self):
    self._testTwoThreadsNoContentionWithRaces(sloppy=True)

  def _testTwoThreadsNoContentionBlockLength(self, sloppy=False):
    # num_threads > 1.
    # Explicit coordination should result in `Dataset.interleave()` behavior
    with self.test_session() as sess:
      self._clear_coordination_events()
      done_first_event = False
      sess.run(
          self.init_op,
          feed_dict={
              self.input_values: [4, 5, 6],
              self.cycle_length: 2,
              self.block_length: 2,
              self.sloppy: sloppy
          })
      for i, expected_element in enumerate(
          self._interleave([[4] * 4, [5] * 5, [6] * 6] * self.repeat_count, 2,
                           2)):
        self.write_coordination_events[expected_element].set()
        if done_first_event:  # First event starts the worker threads.
          self.read_coordination_events[expected_element].acquire()
        actual_element = sess.run(self.next_element)
        if not done_first_event:
          done_first_event = True
          self.read_coordination_events[expected_element].acquire()
        self.assertEqual(expected_element * expected_element, actual_element,
                         "At index %s: %s expected, got: %s" %
                         (i, expected_element, actual_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(self.next_element)

  def testTwoThreadsNoContentionBlockLength(self):
    self._testTwoThreadsNoContentionBlockLength()

  def testTwoThreadsNoContentionBlockLengthSloppy(self):
    self._testTwoThreadsNoContentionBlockLength(sloppy=True)

  def _testTwoThreadsNoContentionWithRacesAndBlocking(self, sloppy=False):
    """Tests where all the workers race in producing elements.

    Note: this is in contrast with the prevous test which carefully sequences
    the execution of the map functions.


    Args:
      sloppy: Whether to be sloppy or not.
    """
    with self.test_session() as sess:
      self._clear_coordination_events()
      done_first_event = False
      sess.run(
          self.init_op,
          feed_dict={
              self.input_values: [4, 5, 6],
              self.cycle_length: 2,
              self.block_length: 2,
              self.sloppy: sloppy
          })
      for i, expected_element in enumerate(
          self._interleave([[4] * 4, [5] * 5, [6] * 6] * self.repeat_count, 2,
                           2)):
        if done_first_event:  # First event starts the worker threads.
          self._allow_all_map_threads()
          self.read_coordination_events[expected_element].acquire()
        else:
          self.write_coordination_events[expected_element].set()
        time.sleep(0.1)  # Sleep to consistently "avoid" the race condition.
        actual_element = sess.run(self.next_element)
        if not done_first_event:
          done_first_event = True
          self.assertTrue(
              self.read_coordination_events[expected_element].acquire(False))
        self.assertEqual(expected_element * expected_element, actual_element,
                         "At index %s: %s expected, got: %s" %
                         (i, expected_element, actual_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(self.next_element)

  def testTwoThreadsNoContentionWithRacesAndBlocking(self):
    self._testTwoThreadsNoContentionWithRacesAndBlocking()

  def testTwoThreadsNoContentionWithRacesAndBlockingSloppy(self):
    self._testTwoThreadsNoContentionWithRacesAndBlocking(sloppy=True)

  def _testEmptyInput(self, sloppy=False):
    with self.test_session() as sess:
      # Empty input.
      self._clear_coordination_events()
      sess.run(
          self.init_op,
          feed_dict={
              self.input_values: [],
              self.cycle_length: 2,
              self.block_length: 3,
              self.sloppy: sloppy
          })
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(self.next_element)

  def testEmptyInput(self):
    self._testEmptyInput()

  def testEmptyInputSloppy(self):
    self._testEmptyInput(sloppy=True)

  def _testNonEmptyInputIntoEmptyOutputs(self, sloppy=False):
    # Non-empty input leading to empty output.
    with self.test_session() as sess:
      self._clear_coordination_events()
      sess.run(
          self.init_op,
          feed_dict={
              self.input_values: [0, 0, 0],
              self.cycle_length: 2,
              self.block_length: 3,
              self.sloppy: sloppy
          })
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(self.next_element)

  def testNonEmptyInputIntoEmptyOutputs(self):
    self._testNonEmptyInputIntoEmptyOutputs()

  def testNonEmptyInputIntoEmptyOutputsSloppy(self):
    self._testNonEmptyInputIntoEmptyOutputs(sloppy=True)

  def _testPartiallyEmptyOutputs(self, sloppy=False):
    # Mixture of non-empty and empty interleaved datasets.
    with self.test_session() as sess:
      self._clear_coordination_events()
      done_first_event = False
      sess.run(
          self.init_op,
          feed_dict={
              self.input_values: [4, 0, 6],
              self.cycle_length: 2,
              self.block_length: 1,
              self.sloppy: sloppy,
          })
      for i, expected_element in enumerate(
          self._interleave([[4] * 4, [], [6] * 6] * self.repeat_count, 2, 1)):
        self.write_coordination_events[expected_element].set()
        if done_first_event:  # First event starts the worker threads
          self.read_coordination_events[expected_element].acquire()
        actual_element = sess.run(self.next_element)
        if not done_first_event:
          done_first_event = True
          self.read_coordination_events[expected_element].acquire()
        self.assertEqual(expected_element * expected_element, actual_element,
                         "At index %s: %s expected, got: %s" %
                         (i, expected_element, actual_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(self.next_element)

  def testPartiallyEmptyOutputs(self):
    self._testPartiallyEmptyOutputs()

  def testPartiallyEmptyOutputsSloppy(self):
    self._testPartiallyEmptyOutputs(sloppy=True)

  def testDelayedOutputSloppy(self):
    # Explicitly control the sequence of events to ensure we correctly avoid
    # head-of-line blocking.
    with self.test_session() as sess:
      self._clear_coordination_events()
      sess.run(
          self.init_op,
          feed_dict={
              self.input_values: [4, 5, 6],
              self.cycle_length: 2,
              self.block_length: 1,
              self.sloppy: True,
          })

      mis_ordering = [
          4, 4, 5, 4, 5, 5, 4, 5, 6, 6, 6, 5, 4, 4, 6, 6, 4, 4, 6, 5, 6, 6, 6,
          6, 5, 5, 5, 5, 6, 6
      ]
      for element in mis_ordering:
        self.write_coordination_events[element].set()
        self.assertEqual(element * element, sess.run(self.next_element))
        self.assertTrue(self.read_coordination_events[element].acquire(False))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(self.next_element)

  def testBlockLengthWithContentionSloppy(self):
    with self.test_session() as sess:
      self._clear_coordination_events()
      done_first_event = False
      sess.run(
          self.init_op,
          feed_dict={
              self.input_values: [4, 5, 6],
              self.cycle_length: 2,
              self.block_length: 3,
              self.sloppy: True
          })
      # Test against a generating sequence that differs from the uncontended
      # case, in order to prove sloppy correctness.
      for i, expected_element in enumerate(
          self._interleave(
              [[4] * 4, [5] * 5, [6] * 6] * self.repeat_count,
              cycle_length=2,
              block_length=2)):
        self.write_coordination_events[expected_element].set()
        if done_first_event:  # First event starts the worker threads.
          self.read_coordination_events[expected_element].acquire()
        actual_element = sess.run(self.next_element)
        if not done_first_event:
          self.read_coordination_events[expected_element].acquire()
          done_first_event = True
        self.assertEqual(expected_element * expected_element, actual_element,
                         "At index %s: %s expected, got: %s" %
                         (i, expected_element, actual_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(self.next_element)

  def _testEarlyExit(self, sloppy=False):
    # Exiting without consuming all input should not block
    with self.test_session() as sess:
      self._clear_coordination_events()
      sess.run(
          self.init_op,
          feed_dict={
              self.input_values: [4, 5, 6],
              self.cycle_length: 3,
              self.block_length: 2,
              self.sloppy: sloppy
          })
      for i in range(4, 7):
        self.write_coordination_events[i].set()
      elem = sess.run(self.next_element)  # Start all workers
      # Allow the one successful worker to progress beyond the py_func again.
      elem = int(math.sqrt(elem))
      self.write_coordination_events[elem].set()
      self.read_coordination_events[elem].acquire()
      # Allow the prefetch to succeed
      for i in range(4, 7):
        self.read_coordination_events[i].acquire()
        self.write_coordination_events[i].set()

  def testEarlyExit(self):
    self._testEarlyExit()

  def testEarlyExitSloppy(self):
    self._testEarlyExit(sloppy=True)

  def _testTooManyReaders(self, sloppy=False):

    def interleave_fn(x):
      dataset = dataset_ops.Dataset.from_tensors(x)
      dataset = dataset.repeat(math_ops.cast(x, dtype=dtypes.int64))
      return dataset

    dataset = dataset_ops.Dataset.from_tensor_slices([4, 5, 6])
    dataset = dataset.repeat(self.repeat_count)
    dataset = dataset.apply(
        interleave_ops.parallel_interleave(
            interleave_fn, cycle_length=16, block_length=2, sloppy=sloppy))
    iterator = dataset.make_one_shot_iterator()

    with self.test_session() as sess:
      output_values = []
      for _ in range(30):
        output_values.append(sess.run(iterator.get_next()))

    expected_values = self._interleave(
        [[4] * 4, [5] * 5, [6] * 6] * self.repeat_count, 1, 2)
    self.assertItemsEqual(output_values, expected_values)

  def testTooManyReaders(self):
    self._testTooManyReaders()

  def testTooManyReadersSloppy(self):
    self._testTooManyReaders(sloppy=True)

  def testSparse(self):
    def _map_fn(i):
      return sparse_tensor.SparseTensor(
          indices=[[0, 0], [1, 1]], values=(i * [1, -1]), dense_shape=[2, 2])

    def _interleave_fn(x):
      return dataset_ops.Dataset.from_tensor_slices(
          sparse_ops.sparse_to_dense(x.indices, x.dense_shape, x.values))

    dataset = dataset_ops.Dataset.range(10).map(_map_fn)
    iterator = dataset.apply(
        interleave_ops.parallel_interleave(
            _interleave_fn, cycle_length=1)).make_initializable_iterator()
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


if __name__ == "__main__":
  test.main()
