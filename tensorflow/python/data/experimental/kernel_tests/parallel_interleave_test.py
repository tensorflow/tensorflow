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
"""Tests for `tf.data.experimental.parallel_interleave()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math
import threading
import time

from absl.testing import parameterized
import numpy as np
from six.moves import zip_longest

from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


#TODO(feihugis): refactor this test to be parameterized.
class ParallelInterleaveTest(test_base.DatasetTestBase, parameterized.TestCase):

  def setUp(self):

    self.error = None
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

  def dataset_fn(self, input_values, cycle_length, block_length, sloppy,
                 buffer_output_elements, prefetch_input_elements):

    def map_py_fn(x):
      self.write_coordination_events[x].wait()
      self.write_coordination_events[x].clear()
      self.read_coordination_events[x].release()
      if self.error:
        err = self.error
        self.error = None
        raise err  # pylint: disable=raising-bad-type
      return x * x

    def map_fn(x):
      return script_ops.py_func(map_py_fn, [x], x.dtype)

    def interleave_fn(x):
      dataset = dataset_ops.Dataset.from_tensors(x)
      dataset = dataset.repeat(x)
      return dataset.map(map_fn)

    return dataset_ops.Dataset.from_tensor_slices(input_values).repeat(
        self.repeat_count).apply(
            interleave_ops.parallel_interleave(
                interleave_fn, cycle_length, block_length, sloppy,
                buffer_output_elements, prefetch_input_elements))

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

  @combinations.generate(test_base.default_test_combinations())
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

  @combinations.generate(test_base.default_test_combinations())
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

  @combinations.generate(test_base.default_test_combinations())
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

  def _testSingleThreaded(self, sloppy=False, prefetch_input_elements=0):
    # cycle_length=1,block_length=1 acts like `Dataset.interleave()` and
    # `Dataset.flat_map()` and is single-threaded. No synchronization required.
    self.skipTest("b/131722904")
    self._clear_coordination_events()
    next_element = self.getNext(
        self.dataset_fn(
            input_values=np.int64([4, 5, 6]),
            cycle_length=1,
            block_length=1,
            sloppy=sloppy,
            buffer_output_elements=1,
            prefetch_input_elements=prefetch_input_elements))
    for expected_element in self._interleave(
        [[4] * 4, [5] * 5, [6] * 6] * self.repeat_count, 1, 1):
      self.write_coordination_events[expected_element].set()
      self.assertEqual(expected_element * expected_element,
                       self.evaluate(next_element()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testSingleThreaded(self):
    self._testSingleThreaded()

  @combinations.generate(test_base.default_test_combinations())
  def testSingleThreadedSloppy(self):
    self._testSingleThreaded(sloppy=True)

  @combinations.generate(test_base.default_test_combinations())
  def testSingleThreadedPrefetch1Itr(self):
    self._testSingleThreaded(prefetch_input_elements=1)

  @combinations.generate(test_base.default_test_combinations())
  def testSingleThreadedPrefetch1ItrSloppy(self):
    self._testSingleThreaded(prefetch_input_elements=1, sloppy=True)

  @combinations.generate(test_base.default_test_combinations())
  def testSingleThreadedRagged(self):
    # Tests a sequence with wildly different elements per iterator.
    self.skipTest("b/131722904")
    self._clear_coordination_events()
    next_element = self.getNext(
        self.dataset_fn(
            input_values=np.int64([3, 7, 4]),
            cycle_length=2,
            block_length=1,
            sloppy=False,
            buffer_output_elements=1,
            prefetch_input_elements=1))

    # Add coordination values for 3 and 7
    self.read_coordination_events[3] = threading.Semaphore(0)
    self.write_coordination_events[3] = threading.Event()
    self.read_coordination_events[7] = threading.Semaphore(0)
    self.write_coordination_events[7] = threading.Event()

    for expected_element in self._interleave(
        [[3] * 3, [7] * 7, [4] * 4] * self.repeat_count, 2, 1):
      self.write_coordination_events[expected_element].set()
      output = self.evaluate(next_element())
      self.assertEqual(expected_element * expected_element, output)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  def _testTwoThreadsNoContention(self, sloppy=False):
    # num_threads > 1.
    # Explicit coordination should result in `Dataset.interleave()` behavior
    self.skipTest("b/131722904")
    self._clear_coordination_events()
    done_first_event = False
    next_element = self.getNext(
        self.dataset_fn(
            input_values=np.int64([4, 5, 6]),
            cycle_length=2,
            block_length=1,
            sloppy=sloppy,
            buffer_output_elements=1,
            prefetch_input_elements=1))
    for i, expected_element in enumerate(
        self._interleave([[4] * 4, [5] * 5, [6] * 6] * self.repeat_count, 2,
                         1)):
      self.write_coordination_events[expected_element].set()
      if done_first_event:  # First event starts the worker threads.
        self.read_coordination_events[expected_element].acquire()
      actual_element = self.evaluate(next_element())
      if not done_first_event:
        self.read_coordination_events[expected_element].acquire()
        done_first_event = True
      self.assertEqual(
          expected_element * expected_element, actual_element,
          "At index %s: %s expected, got: %s" % (i, expected_element,
                                                 actual_element))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testTwoThreadsNoContention(self):
    self._testTwoThreadsNoContention()

  @combinations.generate(test_base.default_test_combinations())
  def testTwoThreadsNoContentionSloppy(self):
    self._testTwoThreadsNoContention(sloppy=True)

  def _testTwoThreadsNoContentionWithRaces(self, sloppy=False):
    """Tests where all the workers race in producing elements.

    Note: this is in contrast with the previous test which carefully sequences
    the execution of the map functions.

    Args:
      sloppy: Whether to be sloppy or not.
    """
    self.skipTest("b/131722904")
    self._clear_coordination_events()
    done_first_event = False
    next_element = self.getNext(
        self.dataset_fn(
            input_values=np.int64([4, 5, 6]),
            cycle_length=2,
            block_length=1,
            sloppy=sloppy,
            buffer_output_elements=1,
            prefetch_input_elements=1))
    for i, expected_element in enumerate(
        self._interleave([[4] * 4, [5] * 5, [6] * 6] * self.repeat_count, 2,
                         1)):
      if done_first_event:  # First event starts the worker threads.
        self._allow_all_map_threads()
        self.read_coordination_events[expected_element].acquire()
      else:
        self.write_coordination_events[expected_element].set()
      time.sleep(0.5)  # Sleep to consistently "avoid" the race condition.
      actual_element = self.evaluate(next_element())
      if not done_first_event:
        done_first_event = True
        self.assertTrue(
            self.read_coordination_events[expected_element].acquire(False))
      self.assertEqual(
          expected_element * expected_element, actual_element,
          "At index %s: %s expected, got: %s" % (i, expected_element,
                                                 actual_element))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testTwoThreadsNoContentionWithRaces(self):
    self._testTwoThreadsNoContentionWithRaces()

  @combinations.generate(test_base.default_test_combinations())
  def testTwoThreadsNoContentionWithRacesSloppy(self):
    self._testTwoThreadsNoContentionWithRaces(sloppy=True)

  def _testTwoThreadsNoContentionBlockLength(self, sloppy=False):
    # num_threads > 1.
    # Explicit coordination should result in `Dataset.interleave()` behavior
    self.skipTest("b/131722904")
    self._clear_coordination_events()
    done_first_event = False
    next_element = self.getNext(
        self.dataset_fn(
            input_values=np.int64([4, 5, 6]),
            cycle_length=2,
            block_length=2,
            sloppy=sloppy,
            buffer_output_elements=1,
            prefetch_input_elements=1))
    for i, expected_element in enumerate(
        self._interleave([[4] * 4, [5] * 5, [6] * 6] * self.repeat_count, 2,
                         2)):
      self.write_coordination_events[expected_element].set()
      if done_first_event:  # First event starts the worker threads.
        self.read_coordination_events[expected_element].acquire()
      actual_element = self.evaluate(next_element())
      if not done_first_event:
        done_first_event = True
        self.read_coordination_events[expected_element].acquire()
      self.assertEqual(
          expected_element * expected_element, actual_element,
          "At index %s: %s expected, got: %s" % (i, expected_element,
                                                 actual_element))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testTwoThreadsNoContentionBlockLength(self):
    self._testTwoThreadsNoContentionBlockLength()

  @combinations.generate(test_base.default_test_combinations())
  def testTwoThreadsNoContentionBlockLengthSloppy(self):
    self._testTwoThreadsNoContentionBlockLength(sloppy=True)

  def _testTwoThreadsNoContentionWithRacesAndBlocking(self, sloppy=False):
    """Tests where all the workers race in producing elements.

    Note: this is in contrast with the previous test which carefully sequences
    the execution of the map functions.


    Args:
      sloppy: Whether to be sloppy or not.
    """
    self.skipTest("b/131722904")
    self._clear_coordination_events()
    done_first_event = False
    next_element = self.getNext(
        self.dataset_fn(
            input_values=np.int64([4, 5, 6]),
            cycle_length=2,
            block_length=2,
            sloppy=sloppy,
            buffer_output_elements=1,
            prefetch_input_elements=1))
    for i, expected_element in enumerate(
        self._interleave([[4] * 4, [5] * 5, [6] * 6] * self.repeat_count, 2,
                         2)):
      if done_first_event:  # First event starts the worker threads.
        self._allow_all_map_threads()
        self.read_coordination_events[expected_element].acquire()
      else:
        self.write_coordination_events[expected_element].set()
      time.sleep(0.5)  # Sleep to consistently "avoid" the race condition.
      actual_element = self.evaluate(next_element())
      if not done_first_event:
        done_first_event = True
        self.assertTrue(
            self.read_coordination_events[expected_element].acquire(False))
      self.assertEqual(
          expected_element * expected_element, actual_element,
          "At index %s: %s expected, got: %s" % (i, expected_element,
                                                 actual_element))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testTwoThreadsNoContentionWithRacesAndBlocking(self):
    self._testTwoThreadsNoContentionWithRacesAndBlocking()

  @combinations.generate(test_base.default_test_combinations())
  def testTwoThreadsNoContentionWithRacesAndBlockingSloppy(self):
    self._testTwoThreadsNoContentionWithRacesAndBlocking(sloppy=True)

  def _testEmptyInput(self, sloppy=False):
    # Empty input.
    self._clear_coordination_events()
    next_element = self.getNext(
        self.dataset_fn(
            input_values=np.int64([]),
            cycle_length=2,
            block_length=3,
            sloppy=sloppy,
            buffer_output_elements=1,
            prefetch_input_elements=0))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyInput(self):
    self._testEmptyInput()

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyInputSloppy(self):
    self._testEmptyInput(sloppy=True)

  def _testNonEmptyInputIntoEmptyOutputs(self, sloppy=False):
    # Non-empty input leading to empty output.
    self._clear_coordination_events()
    next_element = self.getNext(
        self.dataset_fn(
            input_values=np.int64([0, 0, 0]),
            cycle_length=2,
            block_length=3,
            sloppy=sloppy,
            buffer_output_elements=1,
            prefetch_input_elements=0))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testNonEmptyInputIntoEmptyOutputs(self):
    self._testNonEmptyInputIntoEmptyOutputs()

  @combinations.generate(test_base.default_test_combinations())
  def testNonEmptyInputIntoEmptyOutputsSloppy(self):
    self._testNonEmptyInputIntoEmptyOutputs(sloppy=True)

  def _testPartiallyEmptyOutputs(self, sloppy=False, prefetch_input_elements=1):
    race_indices = {2, 8, 14}  # Sequence points when sloppy mode has race conds
    # Mixture of non-empty and empty interleaved datasets.
    self.skipTest("b/131722904")
    self._clear_coordination_events()
    done_first_event = False
    next_element = self.getNext(
        self.dataset_fn(
            input_values=np.int64([4, 0, 6]),
            cycle_length=2,
            block_length=1,
            sloppy=sloppy,
            buffer_output_elements=1,
            prefetch_input_elements=prefetch_input_elements))
    for i, expected_element in enumerate(
        self._interleave([[4] * 4, [], [6] * 6] * self.repeat_count, 2, 1)):
      self.write_coordination_events[expected_element].set()
      # First event starts the worker threads. Additionally, when running the
      # sloppy case with prefetch_input_elements=0, we get stuck if we wait
      # for the read coordination event for certain event orderings in the
      # presence of finishing iterators.
      if done_first_event and not (sloppy and (i in race_indices)):
        self.read_coordination_events[expected_element].acquire()
      actual_element = self.evaluate(next_element())
      if not done_first_event or (sloppy and (i in race_indices)):
        done_first_event = True
        self.read_coordination_events[expected_element].acquire()
      self.assertEqual(
          expected_element * expected_element, actual_element,
          "At index %s: %s expected, got: %s" % (i, expected_element,
                                                 actual_element))

  @combinations.generate(test_base.default_test_combinations())
  def testPartiallyEmptyOutputs(self):
    self._testPartiallyEmptyOutputs()

  @combinations.generate(test_base.default_test_combinations())
  def testPartiallyEmptyOutputsSloppy(self):
    self._testPartiallyEmptyOutputs(sloppy=True, prefetch_input_elements=0)

  @combinations.generate(test_base.default_test_combinations())
  def testDelayedOutputSloppy(self):
    # Explicitly control the sequence of events to ensure we correctly avoid
    # head-of-line blocking.
    self.skipTest("b/131722904")
    self._clear_coordination_events()
    next_element = self.getNext(
        self.dataset_fn(
            input_values=np.int64([4, 5, 6]),
            cycle_length=2,
            block_length=1,
            sloppy=True,
            buffer_output_elements=1,
            prefetch_input_elements=0))

    mis_ordering = [
        4, 4, 5, 4, 5, 5, 4, 5, 6, 6, 6, 5, 4, 4, 6, 6, 4, 4, 6, 5, 6, 6, 6, 6,
        5, 5, 5, 5, 6, 6
    ]
    for element in mis_ordering:
      self.write_coordination_events[element].set()
      self.assertEqual(element * element, self.evaluate(next_element()))
      self.assertTrue(self.read_coordination_events[element].acquire(False))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testBlockLengthWithContentionSloppy(self):
    self.skipTest("b/131722904")
    self._clear_coordination_events()
    done_first_event = False
    next_element = self.getNext(
        self.dataset_fn(
            input_values=np.int64([4, 5, 6]),
            cycle_length=2,
            block_length=1,
            sloppy=True,
            buffer_output_elements=1,
            prefetch_input_elements=1))
    # Test against a generating sequence that differs from the uncontended
    # case, in order to prove sloppy correctness.
    for i, expected_element in enumerate(
        self._interleave(
            [[4] * 4, [5] * 5, [6] * 6] * self.repeat_count,
            cycle_length=2,
            block_length=3)):
      self.write_coordination_events[expected_element].set()
      if done_first_event:  # First event starts the worker threads.
        self.read_coordination_events[expected_element].acquire()
      actual_element = self.evaluate(next_element())
      if not done_first_event:
        self.read_coordination_events[expected_element].acquire()
        done_first_event = True
      self.assertEqual(
          expected_element * expected_element, actual_element,
          "At index %s: %s expected, got: %s" % (i, expected_element,
                                                 actual_element))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  def _testEarlyExit(self, sloppy=False):
    # Exiting without consuming all input should not block
    self.skipTest("b/131722904")
    self._clear_coordination_events()
    next_element = self.getNext(
        self.dataset_fn(
            input_values=np.int64([4, 5, 6]),
            cycle_length=3,
            block_length=2,
            sloppy=sloppy,
            buffer_output_elements=1,
            prefetch_input_elements=0))
    for i in range(4, 7):
      self.write_coordination_events[i].set()
    elem = self.evaluate(next_element())  # Start all workers
    # Allow the one successful worker to progress beyond the py_func again.
    elem = int(math.sqrt(elem))
    self.write_coordination_events[elem].set()
    self.read_coordination_events[elem].acquire()
    # Allow the prefetch to succeed
    for i in range(4, 7):
      self.read_coordination_events[i].acquire()
      self.write_coordination_events[i].set()

  @combinations.generate(test_base.default_test_combinations())
  def testEarlyExit(self):
    self._testEarlyExit()

  @combinations.generate(test_base.default_test_combinations())
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
    get_next = self.getNext(dataset)
    output_values = []
    for _ in range(30):
      output_values.append(self.evaluate(get_next()))

    expected_values = self._interleave(
        [[4] * 4, [5] * 5, [6] * 6] * self.repeat_count, 1, 2)
    self.assertItemsEqual(output_values, expected_values)

  @combinations.generate(test_base.default_test_combinations())
  def testTooManyReaders(self):
    self._testTooManyReaders()

  @combinations.generate(test_base.default_test_combinations())
  def testTooManyReadersSloppy(self):
    self._testTooManyReaders(sloppy=True)

  @combinations.generate(test_base.default_test_combinations())
  def testSparse(self):
    def _map_fn(i):
      return sparse_tensor.SparseTensor(
          indices=[[0, 0], [1, 1]], values=(i * [1, -1]), dense_shape=[2, 2])

    def _interleave_fn(x):
      return dataset_ops.Dataset.from_tensor_slices(
          sparse_ops.sparse_to_dense(x.indices, x.dense_shape, x.values))

    dataset = dataset_ops.Dataset.range(10).map(_map_fn).apply(
        interleave_ops.parallel_interleave(_interleave_fn, cycle_length=1))
    get_next = self.getNext(dataset)

    for i in range(10):
      for j in range(2):
        expected = [i, 0] if j % 2 == 0 else [0, -i]
        self.assertAllEqual(expected, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testErrorsInOutputFn(self):
    self.skipTest("b/131722904")
    self._clear_coordination_events()
    next_element = self.getNext(
        self.dataset_fn(
            input_values=np.int64([4, 5, 6]),
            cycle_length=2,
            block_length=1,
            sloppy=False,
            buffer_output_elements=1,
            prefetch_input_elements=0))

    except_on_element_indices = set([3])

    for i, expected_element in enumerate(
        self._interleave([[4] * 4, [5] * 5, [6] * 6] * self.repeat_count, 2,
                         1)):
      if i in except_on_element_indices:
        self.error = ValueError()
        self.write_coordination_events[expected_element].set()
        with self.assertRaises(errors.InvalidArgumentError):
          self.evaluate(next_element())
      else:
        self.write_coordination_events[expected_element].set()
        actual_element = self.evaluate(next_element())
        self.assertEqual(
            expected_element * expected_element, actual_element,
            "At index %s: %s expected, got: %s" % (i, expected_element,
                                                   actual_element))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testErrorsInInputFn(self):

    def map_py_fn(x):
      if x == 5:
        raise ValueError()
      return x

    def map_fn(x):
      return script_ops.py_func(map_py_fn, [x], x.dtype)

    def interleave_fn(x):
      dataset = dataset_ops.Dataset.from_tensors(x)
      dataset = dataset.repeat(x)
      return dataset

    def dataset_fn(input_values, cycle_length, block_length, sloppy,
                   buffer_output_elements, prefetch_input_elements):
      return dataset_ops.Dataset.from_tensor_slices(input_values).map(
          map_fn).repeat(self.repeat_count).apply(
              interleave_ops.parallel_interleave(
                  interleave_fn, cycle_length, block_length, sloppy,
                  buffer_output_elements, prefetch_input_elements))

    next_element = self.getNext(
        dataset_fn(
            input_values=np.int64([4, 5, 6]),
            cycle_length=2,
            block_length=1,
            sloppy=False,
            buffer_output_elements=1,
            prefetch_input_elements=0))
    for i, expected_element in enumerate(
        self._interleave([[4] * 4, [5], [6] * 6] * self.repeat_count, 2, 1)):
      if expected_element == 5:
        with self.assertRaises(errors.InvalidArgumentError):
          self.evaluate(next_element())
      else:
        actual_element = self.evaluate(next_element())
        self.assertEqual(
            expected_element, actual_element,
            "At index %s: %s expected, got: %s" % (i, expected_element,
                                                   actual_element))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testErrorsInInterleaveFn(self):

    def map_py_fn(x):
      if x == 5:
        raise ValueError()
      return x

    def interleave_fn(x):
      dataset = dataset_ops.Dataset.from_tensors(x)
      y = script_ops.py_func(map_py_fn, [x], x.dtype)
      dataset = dataset.repeat(y)
      return dataset

    def dataset_fn(input_values, cycle_length, block_length, sloppy,
                   buffer_output_elements, prefetch_input_elements):
      return dataset_ops.Dataset.from_tensor_slices(input_values).repeat(
          self.repeat_count).apply(
              interleave_ops.parallel_interleave(
                  interleave_fn, cycle_length, block_length, sloppy,
                  buffer_output_elements, prefetch_input_elements))

    next_element = self.getNext(
        dataset_fn(
            input_values=np.int64([4, 5, 6]),
            cycle_length=2,
            block_length=1,
            sloppy=False,
            buffer_output_elements=1,
            prefetch_input_elements=0))
    for i, expected_element in enumerate(
        self._interleave([[4] * 4, [5], [6] * 6] * self.repeat_count, 2, 1)):
      if expected_element == 5:
        with self.assertRaises(errors.InvalidArgumentError):
          self.evaluate(next_element())
      else:
        actual_element = self.evaluate(next_element())
        self.assertEqual(
            expected_element, actual_element,
            "At index %s: %s expected, got: %s" % (i, expected_element,
                                                   actual_element))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testShutdownRace(self):
    dataset = dataset_ops.Dataset.range(20)
    map_fn = lambda x: dataset_ops.Dataset.range(20 * x, 20 * (x + 1))
    dataset = dataset.apply(
        interleave_ops.parallel_interleave(
            map_fn,
            cycle_length=3,
            sloppy=False,
            buffer_output_elements=1,
            prefetch_input_elements=0))
    dataset = dataset.batch(32)

    results = []
    for _ in range(2):
      elements = []
      next_element = self.getNext(dataset)
      try:
        while True:
          elements.extend(self.evaluate(next_element()))
      except errors.OutOfRangeError:
        pass
      results.append(elements)
    self.assertAllEqual(results[0], results[1])


if __name__ == "__main__":
  test.main()
