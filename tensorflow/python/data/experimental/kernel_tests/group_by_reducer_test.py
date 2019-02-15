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
"""Tests for `tf.data.experimental.group_by_reducer()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class GroupByReducerTest(test_base.DatasetTestBase):

  def testSum(self):
    reducer = grouping.Reducer(
        init_func=lambda _: np.int64(0),
        reduce_func=lambda x, y: x + y,
        finalize_func=lambda x: x)
    for i in range(1, 11):
      dataset = dataset_ops.Dataset.range(2 * i).apply(
          grouping.group_by_reducer(lambda x: x % 2, reducer))
      self.assertDatasetProduces(
          dataset,
          expected_shapes=tensor_shape.scalar(),
          expected_output=[(i - 1) * i, i * i])

  def testAverage(self):

    def reduce_fn(x, y):
      return (x[0] * x[1] + math_ops.cast(y, dtypes.float32)) / (
          x[1] + 1), x[1] + 1

    reducer = grouping.Reducer(
        init_func=lambda _: (0.0, 0.0),
        reduce_func=reduce_fn,
        finalize_func=lambda x, _: x)
    for i in range(1, 11):
      dataset = dataset_ops.Dataset.range(2 * i).apply(
          grouping.group_by_reducer(
              lambda x: math_ops.cast(x, dtypes.int64) % 2, reducer))
      self.assertDatasetProduces(
          dataset,
          expected_shapes=tensor_shape.scalar(),
          expected_output=[i - 1, i])

  def testConcat(self):
    components = np.array(list("abcdefghijklmnopqrst")).view(np.chararray)
    reducer = grouping.Reducer(
        init_func=lambda x: "",
        reduce_func=lambda x, y: x + y[0],
        finalize_func=lambda x: x)
    for i in range(1, 11):
      dataset = dataset_ops.Dataset.zip(
          (dataset_ops.Dataset.from_tensor_slices(components),
           dataset_ops.Dataset.range(2 * i))).apply(
               grouping.group_by_reducer(lambda x, y: y % 2, reducer))
      self.assertDatasetProduces(
          dataset,
          expected_shapes=tensor_shape.scalar(),
          expected_output=[b"acegikmoqs" [:i], b"bdfhjlnprt" [:i]])

  def testSparseSum(self):
    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1], dtype=np.int64)),
          dense_shape=np.array([1, 1]))

    reducer = grouping.Reducer(
        init_func=lambda _: _sparse(np.int64(0)),
        reduce_func=lambda x, y: _sparse(x.values[0] + y.values[0]),
        finalize_func=lambda x: x.values[0])
    for i in range(1, 11):
      dataset = dataset_ops.Dataset.range(2 * i).map(_sparse).apply(
          grouping.group_by_reducer(lambda x: x.values[0] % 2, reducer))
      self.assertDatasetProduces(
          dataset,
          expected_shapes=tensor_shape.scalar(),
          expected_output=[(i - 1) * i, i * i])

  def testChangingStateShape(self):

    def reduce_fn(x, _):
      # Statically known rank, but dynamic length.
      larger_dim = array_ops.concat([x[0], x[0]], 0)
      # Statically unknown rank.
      larger_rank = array_ops.expand_dims(x[1], 0)
      return larger_dim, larger_rank

    reducer = grouping.Reducer(
        init_func=lambda x: ([0], 1),
        reduce_func=reduce_fn,
        finalize_func=lambda x, y: (x, y))

    for i in range(1, 11):
      dataset = dataset_ops.Dataset.from_tensors(np.int64(0)).repeat(i).apply(
          grouping.group_by_reducer(lambda x: x, reducer))
      self.assertEqual([None], dataset.output_shapes[0].as_list())
      self.assertIs(None, dataset.output_shapes[1].ndims)
      get_next = self.getNext(dataset)
      x, y = self.evaluate(get_next())
      self.assertAllEqual([0] * (2**i), x)
      self.assertAllEqual(np.array(1, ndmin=i), y)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())

  def testTypeMismatch(self):
    reducer = grouping.Reducer(
        init_func=lambda x: constant_op.constant(1, dtype=dtypes.int32),
        reduce_func=lambda x, y: constant_op.constant(1, dtype=dtypes.int64),
        finalize_func=lambda x: x)

    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegexp(
        TypeError,
        "The element types for the new state must match the initial state."):
      dataset.apply(
          grouping.group_by_reducer(lambda _: np.int64(0), reducer))

  # TODO(b/78665031): Remove once non-scalar keys are supported.
  def testInvalidKeyShape(self):
    reducer = grouping.Reducer(
        init_func=lambda x: np.int64(0),
        reduce_func=lambda x, y: x + y,
        finalize_func=lambda x: x)

    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegexp(
        ValueError, "`key_func` must return a single tf.int64 tensor."):
      dataset.apply(
          grouping.group_by_reducer(lambda _: np.int64((0, 0)), reducer))

  # TODO(b/78665031): Remove once non-int64 keys are supported.
  def testInvalidKeyType(self):
    reducer = grouping.Reducer(
        init_func=lambda x: np.int64(0),
        reduce_func=lambda x, y: x + y,
        finalize_func=lambda x: x)

    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegexp(
        ValueError, "`key_func` must return a single tf.int64 tensor."):
      dataset.apply(
          grouping.group_by_reducer(lambda _: "wrong", reducer))

  def testTuple(self):
    def init_fn(_):
      return np.array([], dtype=np.int64), np.int64(0)

    def reduce_fn(state, value):
      s1, s2 = state
      v1, v2 = value
      return array_ops.concat([s1, [v1]], 0), s2 + v2

    def finalize_fn(s1, s2):
      return s1, s2

    reducer = grouping.Reducer(init_fn, reduce_fn, finalize_fn)
    dataset = dataset_ops.Dataset.zip(
        (dataset_ops.Dataset.range(10), dataset_ops.Dataset.range(10))).apply(
            grouping.group_by_reducer(lambda x, y: np.int64(0), reducer))
    get_next = self.getNext(dataset)
    x, y = self.evaluate(get_next())
    self.assertAllEqual(x, np.asarray([x for x in range(10)]))
    self.assertEqual(y, 45)


if __name__ == "__main__":
  test.main()
