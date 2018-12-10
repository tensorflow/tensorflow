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

import threading

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import script_ops
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


def _make_coordinated_sloppy_dataset(input_values, cycle_length, block_length,
                                     num_parallel_calls):
  """Produces a dataset iterator and events to control the order of elements.

  Args:
    input_values: the values to generate lists to interleave from
    cycle_length: the length of the interleave cycle
    block_length: the length of the interleave block
    num_parallel_calls: the degree of interleave parallelism

  Returns:
    A dataset iterator (represented as `get_next` op) and events that can be
    used to control the order of output elements.
  """

  # Set up threading events used to sequence when items are produced that
  # are subsequently interleaved. These events allow us to deterministically
  # simulate slowdowns and force sloppiness.
  coordination_events = {i: threading.Event() for i in input_values}

  def map_py_fn(x):
    coordination_events[x].wait()
    coordination_events[x].clear()
    return x * x

  def map_fn(x):
    return script_ops.py_func(map_py_fn, [x], x.dtype)

  def interleave_fn(x):
    dataset = dataset_ops.Dataset.from_tensors(x)
    dataset = dataset.repeat(x)
    return dataset.map(map_fn)

  options = dataset_ops.Options()
  options.experimental_deterministic = False
  dataset = dataset_ops.Dataset.from_tensor_slices(input_values).repeat(
      2).interleave(interleave_fn, cycle_length, block_length,
                    num_parallel_calls).with_options(options)
  iterator = dataset_ops.make_one_shot_iterator(dataset)
  get_next = iterator.get_next()
  return get_next, coordination_events


def _repeat(values, count):
  """Produces a list of lists suitable for testing interleave.

  Args:
    values: for each element `x` the result contains `[x] * x`
    count: determines how many times to repeat `[x] * x` in the result

  Returns:
    A list of lists of values suitable for testing interleave.
  """
  return [[value] * value for value in np.tile(values, count)]


@test_util.run_all_in_graph_and_eager_modes
class InterleaveTest(test_base.DatasetTestBase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("1", [4, 5, 6], 1, 1, [
          4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 5, 5, 5, 5,
          5, 6, 6, 6, 6, 6, 6
      ]),
      ("2", [4, 5, 6], 2, 1, [
          4, 5, 4, 5, 4, 5, 4, 5, 5, 6, 6, 4, 6, 4, 6, 4, 6, 4, 6, 5, 6, 5, 6,
          5, 6, 5, 6, 5, 6, 6
      ]),
      ("3", [4, 5, 6], 2, 3, [
          4, 4, 4, 5, 5, 5, 4, 5, 5, 6, 6, 6, 4, 4, 4, 6, 6, 6, 4, 5, 5, 5, 6,
          6, 6, 5, 5, 6, 6, 6
      ]),
      ("4", [4, 5, 6], 7, 2, [
          4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6,
          6, 5, 6, 6, 5, 6, 6
      ]),
      ("5", [4, 0, 6], 2, 1,
       [4, 4, 6, 4, 6, 4, 6, 6, 4, 6, 4, 6, 4, 4, 6, 6, 6, 6, 6, 6]),
  )
  def testPythonImplementation(self, input_values, cycle_length, block_length,
                               expected_elements):
    input_lists = _repeat(input_values, 2)

    for expected, produced in zip(
        expected_elements, _interleave(input_lists, cycle_length,
                                       block_length)):
      self.assertEqual(expected, produced)

  @parameterized.named_parameters(
      ("1", np.int64([4, 5, 6]), 1, 3, None),
      ("2", np.int64([4, 5, 6]), 1, 3, 1),
      ("3", np.int64([4, 5, 6]), 2, 1, None),
      ("4", np.int64([4, 5, 6]), 2, 1, 1),
      ("5", np.int64([4, 5, 6]), 2, 1, 2),
      ("6", np.int64([4, 5, 6]), 2, 3, None),
      ("7", np.int64([4, 5, 6]), 2, 3, 1),
      ("8", np.int64([4, 5, 6]), 2, 3, 2),
      ("9", np.int64([4, 5, 6]), 7, 2, None),
      ("10", np.int64([4, 5, 6]), 7, 2, 1),
      ("11", np.int64([4, 5, 6]), 7, 2, 3),
      ("12", np.int64([4, 5, 6]), 7, 2, 5),
      ("13", np.int64([4, 5, 6]), 7, 2, 7),
      ("14", np.int64([]), 2, 3, None),
      ("15", np.int64([0, 0, 0]), 2, 3, None),
      ("16", np.int64([4, 0, 6]), 2, 3, None),
      ("17", np.int64([4, 0, 6]), 2, 3, 1),
      ("18", np.int64([4, 0, 6]), 2, 3, 2),
  )
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

  @parameterized.named_parameters(
      ("1", np.float32([1., np.nan, 2., np.nan, 3.]), 1, 3, None),
      ("2", np.float32([1., np.nan, 2., np.nan, 3.]), 1, 3, 1),
      ("3", np.float32([1., np.nan, 2., np.nan, 3.]), 2, 1, None),
      ("4", np.float32([1., np.nan, 2., np.nan, 3.]), 2, 1, 1),
      ("5", np.float32([1., np.nan, 2., np.nan, 3.]), 2, 1, 2),
      ("6", np.float32([1., np.nan, 2., np.nan, 3.]), 2, 3, None),
      ("7", np.float32([1., np.nan, 2., np.nan, 3.]), 2, 3, 1),
      ("8", np.float32([1., np.nan, 2., np.nan, 3.]), 2, 3, 2),
      ("9", np.float32([1., np.nan, 2., np.nan, 3.]), 7, 2, None),
      ("10", np.float32([1., np.nan, 2., np.nan, 3.]), 7, 2, 1),
      ("11", np.float32([1., np.nan, 2., np.nan, 3.]), 7, 2, 3),
      ("12", np.float32([1., np.nan, 2., np.nan, 3.]), 7, 2, 5),
      ("13", np.float32([1., np.nan, 2., np.nan, 3.]), 7, 2, 7),
  )
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

  @parameterized.named_parameters(
      ("1", np.int64([4, 5, 6]), 2, 1, 1),
      ("2", np.int64([4, 5, 6]), 2, 1, 2),
      ("3", np.int64([4, 5, 6]), 2, 3, 1),
      ("4", np.int64([4, 5, 6]), 2, 3, 2),
      ("5", np.int64([4, 5, 6]), 3, 2, 1),
      ("6", np.int64([4, 5, 6]), 3, 2, 2),
      ("7", np.int64([4, 5, 6]), 3, 2, 3),
      ("8", np.int64([4, 0, 6]), 2, 3, 1),
      ("9", np.int64([4, 0, 6]), 2, 3, 2),
  )
  @test_util.run_v1_only("b/120545219")
  def testSkipEagerSloppyInterleaveInOrder(self, input_values, cycle_length,
                                           block_length, num_parallel_calls):
    get_next, coordination_events = _make_coordinated_sloppy_dataset(
        input_values, cycle_length, block_length, num_parallel_calls)
    config = config_pb2.ConfigProto(
        inter_op_parallelism_threads=num_parallel_calls + 1,
        use_per_session_threads=True)
    with self.cached_session(config=config) as sess:
      for expected_element in _interleave(
          _repeat(input_values, 2), cycle_length, block_length):
        coordination_events[expected_element].set()
        self.assertEqual(expected_element * expected_element,
                         self.evaluate(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  @parameterized.named_parameters(
      ("1", np.int64([4, 5, 6]), 2, 1, 2),
      ("2", np.int64([4, 5, 6]), 2, 3, 2),
      ("3", np.int64([4, 5, 6]), 3, 2, 3),
      ("4", np.int64([4, 0, 6]), 2, 3, 2),
  )
  @test_util.run_v1_only("b/120545219")
  def testSkipEagerSloppyInterleaveOutOfOrder(self, input_values, cycle_length,
                                              block_length, num_parallel_calls):
    get_next, coordination_events = _make_coordinated_sloppy_dataset(
        input_values, cycle_length, block_length, num_parallel_calls)
    config = config_pb2.ConfigProto(
        inter_op_parallelism_threads=num_parallel_calls + 1,
        use_per_session_threads=True)
    with self.cached_session(config=config) as sess:
      elements = [
          x for x in _interleave(
              _repeat(input_values, 2), cycle_length, block_length)
      ]
      for i in [1, 4, 7]:
        elements[i], elements[i + 1] = elements[i + 1], elements[i]

      for element in elements:
        coordination_events[element].set()
        self.assertEqual(element * element, self.evaluate(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()
