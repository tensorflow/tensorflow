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
"""Tests for `tf.data.experimental.scan()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from tensorflow.python.data.experimental.ops import scan_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class ScanTest(test_base.DatasetTestBase):

  def _counting_dataset(self, start, scan_fn):
    return dataset_ops.Dataset.from_tensors(0).repeat().apply(
        scan_ops.scan(start, scan_fn))

  def testCount(self):
    def make_scan_fn(step):
      return lambda state, _: (state + step, state)

    def dataset_fn(start, step, take):
      return self._counting_dataset(start, make_scan_fn(step)).take(take)

    for start_val, step_val, take_val in [(0, 1, 10), (0, 1, 0), (10, 1, 10),
                                          (10, 2, 10), (10, -1, 10), (10, -2,
                                                                      10)]:
      next_element = self.getNext(dataset_fn(start_val, step_val, take_val))
      for expected, _ in zip(
          itertools.count(start_val, step_val), range(take_val)):
        self.assertEqual(expected, self.evaluate(next_element()))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element())

  def testFibonacci(self):
    data = dataset_ops.Dataset.from_tensors(1).repeat(None).apply(
        scan_ops.scan([0, 1], lambda a, _: ([a[1], a[0] + a[1]], a[1])))
    next_element = self.getNext(data)

    self.assertEqual(1, self.evaluate(next_element()))
    self.assertEqual(1, self.evaluate(next_element()))
    self.assertEqual(2, self.evaluate(next_element()))
    self.assertEqual(3, self.evaluate(next_element()))
    self.assertEqual(5, self.evaluate(next_element()))
    self.assertEqual(8, self.evaluate(next_element()))

  def testSparseCount(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1]))

    def make_scan_fn(step):
      return lambda state, _: (_sparse(state.values[0] + step), state)

    def dataset_fn(start, step, take):
      return self._counting_dataset(_sparse(start),
                                    make_scan_fn(step)).take(take)

    for start_val, step_val, take_val in [(0, 1, 10), (0, 1, 0), (10, 1, 10),
                                          (10, 2, 10), (10, -1, 10), (10, -2,
                                                                      10)]:
      next_element = self.getNext(dataset_fn(start_val, step_val, take_val))
      for expected, _ in zip(
          itertools.count(start_val, step_val), range(take_val)):
        self.assertEqual(expected, self.evaluate(next_element()).values[0])
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element())

  def testChangingStateShape(self):
    # Test the fixed-point shape invariant calculations: start with
    # initial values with known shapes, and use a scan function that
    # changes the size of the state on each element.
    def _scan_fn(state, input_value):
      # Statically known rank, but dynamic length.
      ret_longer_vector = array_ops.concat([state[0], state[0]], 0)
      # Statically unknown rank.
      ret_larger_rank = array_ops.expand_dims(state[1], 0)
      return (ret_longer_vector, ret_larger_rank), (state, input_value)

    dataset = dataset_ops.Dataset.from_tensors(0).repeat(5).apply(
        scan_ops.scan(([0], 1), _scan_fn))
    self.assertEqual([None], dataset.output_shapes[0][0].as_list())
    self.assertIs(None, dataset.output_shapes[0][1].ndims)
    self.assertEqual([], dataset.output_shapes[1].as_list())

    next_element = self.getNext(dataset)

    for i in range(5):
      (longer_vector_val, larger_rank_val), _ = self.evaluate(next_element())
      self.assertAllEqual([0] * (2**i), longer_vector_val)
      self.assertAllEqual(np.array(1, ndmin=i), larger_rank_val)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  def testIncorrectStateType(self):

    def _scan_fn(state, _):
      return constant_op.constant(1, dtype=dtypes.int64), state

    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegexp(
        TypeError,
        "The element types for the new state must match the initial state."):
      dataset.apply(
          scan_ops.scan(constant_op.constant(1, dtype=dtypes.int32), _scan_fn))

  def testIncorrectReturnType(self):

    def _scan_fn(unused_state, unused_input_value):
      return constant_op.constant(1, dtype=dtypes.int64)

    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegexp(
        TypeError,
        "The scan function must return a pair comprising the new state and the "
        "output value."):
      dataset.apply(
          scan_ops.scan(constant_op.constant(1, dtype=dtypes.int32), _scan_fn))

  def testPreserveCardinality(self):

    def scan_fn(state, val):

      def py_fn(_):
        raise StopIteration()

      return state, script_ops.py_func(py_fn, [val], dtypes.int64)

    dataset = dataset_ops.Dataset.from_tensors(0).apply(
        scan_ops.scan(constant_op.constant(1), scan_fn))
    get_next = self.getNext(dataset)
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())


if __name__ == "__main__":
  test.main()
