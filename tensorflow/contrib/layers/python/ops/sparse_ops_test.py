# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.contrib.layers.python.ops.sparse_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.layers.python.ops import sparse_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


def _assert_sparse_tensor_value(test_case, expected, actual):
  test_case.assertEqual(np.int64, np.array(actual.indices).dtype)
  test_case.assertAllEqual(expected.indices, actual.indices)

  test_case.assertEqual(
      np.array(expected.values).dtype, np.array(actual.values).dtype)
  test_case.assertAllEqual(expected.values, actual.values)

  test_case.assertEqual(np.int64, np.array(actual.dense_shape).dtype)
  test_case.assertAllEqual(expected.dense_shape, actual.dense_shape)


class DenseToSparseTensorTest(test.TestCase):

  def test_dense_to_sparse_tensor_1d(self):
    with self.cached_session() as sess:
      st = sparse_ops.dense_to_sparse_tensor([1, 0, 2, 0])
      result = sess.run(st)
    self.assertEqual(result.indices.dtype, np.int64)
    self.assertEqual(result.values.dtype, np.int32)
    self.assertEqual(result.dense_shape.dtype, np.int64)
    self.assertAllEqual([[0], [2]], result.indices)
    self.assertAllEqual([1, 2], result.values)
    self.assertAllEqual([4], result.dense_shape)

  def test_dense_to_sparse_tensor_1d_float(self):
    with self.cached_session() as sess:
      st = sparse_ops.dense_to_sparse_tensor([1.5, 0.0, 2.3, 0.0])
      result = sess.run(st)
    self.assertEqual(result.indices.dtype, np.int64)
    self.assertEqual(result.values.dtype, np.float32)
    self.assertEqual(result.dense_shape.dtype, np.int64)
    self.assertAllEqual([[0], [2]], result.indices)
    self.assertAllClose([1.5, 2.3], result.values)
    self.assertAllEqual([4], result.dense_shape)

  def test_dense_to_sparse_tensor_1d_bool(self):
    with self.cached_session() as sess:
      st = sparse_ops.dense_to_sparse_tensor([True, False, True, False])
      result = sess.run(st)
    self.assertEqual(result.indices.dtype, np.int64)
    self.assertEqual(result.values.dtype, np.bool)
    self.assertEqual(result.dense_shape.dtype, np.int64)
    self.assertAllEqual([[0], [2]], result.indices)
    self.assertAllEqual([True, True], result.values)
    self.assertAllEqual([4], result.dense_shape)

  def test_dense_to_sparse_tensor_1d_str(self):
    with self.cached_session() as sess:
      st = sparse_ops.dense_to_sparse_tensor([b'qwe', b'', b'ewq', b''])
      result = sess.run(st)
    self.assertEqual(result.indices.dtype, np.int64)
    self.assertEqual(result.values.dtype, np.object)
    self.assertEqual(result.dense_shape.dtype, np.int64)
    self.assertAllEqual([[0], [2]], result.indices)
    self.assertAllEqual([b'qwe', b'ewq'], result.values)
    self.assertAllEqual([4], result.dense_shape)

  def test_dense_to_sparse_tensor_1d_str_special_ignore(self):
    with self.cached_session() as sess:
      st = sparse_ops.dense_to_sparse_tensor(
          [b'qwe', b'', b'ewq', b''], ignore_value=b'qwe')
      result = sess.run(st)
    self.assertEqual(result.indices.dtype, np.int64)
    self.assertEqual(result.values.dtype, np.object)
    self.assertEqual(result.dense_shape.dtype, np.int64)
    self.assertAllEqual([[1], [2], [3]], result.indices)
    self.assertAllEqual([b'', b'ewq', b''], result.values)
    self.assertAllEqual([4], result.dense_shape)

  def test_dense_to_sparse_tensor_2d(self):
    with self.cached_session() as sess:
      st = sparse_ops.dense_to_sparse_tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
      result = sess.run(st)
    self.assertAllEqual([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]],
                        result.indices)
    self.assertAllEqual([1, 2, 3, 4, 5], result.values)
    self.assertAllEqual([2, 4], result.dense_shape)

  def test_dense_to_sparse_tensor_3d(self):
    with self.cached_session() as sess:
      st = sparse_ops.dense_to_sparse_tensor([[[1, 2, 0, 0], [3, 4, 5, 0]],
                                              [[7, 8, 0, 0], [9, 0, 0, 0]]])
      result = sess.run(st)
    self.assertAllEqual([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 2],
                         [1, 0, 0], [1, 0, 1], [1, 1, 0]], result.indices)
    self.assertAllEqual([1, 2, 3, 4, 5, 7, 8, 9], result.values)
    self.assertAllEqual([2, 2, 4], result.dense_shape)

  def test_dense_to_sparse_tensor_unknown_1d_shape(self):
    with self.cached_session() as sess:
      tensor = array_ops.placeholder(shape=[None], dtype=dtypes.int32)
      st = sparse_ops.dense_to_sparse_tensor(tensor)
      result = sess.run(st, feed_dict={tensor: [0, 100, 0, 3]})
    self.assertAllEqual([[1], [3]], result.indices)
    self.assertAllEqual([100, 3], result.values)
    self.assertAllEqual([4], result.dense_shape)

  def test_dense_to_sparse_tensor_unknown_3d_shape(self):
    with self.cached_session() as sess:
      tensor = array_ops.placeholder(
          shape=[None, None, None], dtype=dtypes.int32)
      st = sparse_ops.dense_to_sparse_tensor(tensor)
      result = sess.run(st,
                        feed_dict={
                            tensor: [[[1, 2, 0, 0], [3, 4, 5, 0]],
                                     [[7, 8, 0, 0], [9, 0, 0, 0]]]
                        })
    self.assertAllEqual([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 2],
                         [1, 0, 0], [1, 0, 1], [1, 1, 0]], result.indices)
    self.assertAllEqual([1, 2, 3, 4, 5, 7, 8, 9], result.values)
    self.assertAllEqual([2, 2, 4], result.dense_shape)

  def test_dense_to_sparse_unknown_rank(self):
    ph = array_ops.placeholder(dtype=dtypes.int32)
    with self.cached_session() as sess:
      st = sparse_ops.dense_to_sparse_tensor(ph)
      result = sess.run(st, feed_dict={ph: [[1, 2, 0, 0], [3, 4, 5, 0]]})
    self.assertAllEqual([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]],
                        result.indices)
    self.assertAllEqual([1, 2, 3, 4, 5], result.values)
    self.assertAllEqual([2, 4], result.dense_shape)


class SparseRowEnvelopeTest(test.TestCase):

  def test_sparse_row_envelope(self):
    expected_sparse_row_envelope = [1, 0, 3]
    with self.cached_session() as sess:
      sparse_input = sparse_tensor.SparseTensor(
          indices=[[0, 0], [2, 0], [2, 1], [2, 2]],
          values=[0, 1, 2, 3],
          dense_shape=[3, 3])
      sparse_row_envelope = sess.run(
          sparse_ops.sparse_row_envelope(sparse_input))
      self.assertAllEqual(expected_sparse_row_envelope,
                          sparse_row_envelope)

  def test_sparse_row_envelope_unsorted_indices(self):
    expected_sparse_row_envelope = [1, 0, 3]
    with self.cached_session() as sess:
      sparse_input = sparse_tensor.SparseTensor(
          indices=[[2, 0], [2, 2], [2, 1], [0, 0]],
          values=[0, 1, 2, 3],
          dense_shape=[3, 3])
      sparse_row_envelope = sess.run(
          sparse_ops.sparse_row_envelope(sparse_input))
      self.assertAllEqual(expected_sparse_row_envelope,
                          sparse_row_envelope)

  def test_sparse_row_envelope_empty_in_the_end(self):
    expected_sparse_row_envelope = [1, 0, 3, 0, 0]
    with self.cached_session() as sess:
      sparse_input = sparse_tensor.SparseTensor(
          indices=[[0, 0], [2, 0], [2, 1], [2, 2]],
          values=[0, 1, 2, 3],
          dense_shape=[5, 3])
      sparse_row_envelope = sess.run(
          sparse_ops.sparse_row_envelope(sparse_input))
      self.assertAllEqual(expected_sparse_row_envelope,
                          sparse_row_envelope)

  def test_sparse_row_envelope_empty_3d(self):
    expected_sparse_row_envelope = [1, 0, 3, 0, 0]
    with self.cached_session() as sess:
      sparse_input = sparse_tensor.SparseTensor(
          indices=[[0, 0, 0], [0, 2, 0], [0, 2, 1], [0, 2, 2]],
          values=[0, 1, 2, 3],
          dense_shape=[1, 5, 3])
      sparse_row_envelope = sess.run(
          sparse_ops.sparse_row_envelope(sparse_input, 1, 2))
      self.assertAllEqual(expected_sparse_row_envelope,
                          sparse_row_envelope)


class IndicatorToSparseIdsTest(test.TestCase):

  def test_indicators_to_sparse_ids_1d(self):
    indicators = (0, 0, 1, 0)
    sparse_ids = sparse_ops.indicators_to_sparse_ids(indicators)
    with self.cached_session():
      _assert_sparse_tensor_value(self, sparse_tensor.SparseTensorValue(
          indices=((0,),),
          values=(2,),
          dense_shape=(1,),
      ), sparse_ids.eval())

  def test_indicators_to_sparse_ids_2d(self):
    indicators = (
        (0, 0, 1, 0),
        (1, 0, 0, 1),
    )
    sparse_ids = sparse_ops.indicators_to_sparse_ids(indicators)
    with self.cached_session():
      _assert_sparse_tensor_value(self, sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=(2, 0, 3),
          dense_shape=(2, 2),
      ), sparse_ids.eval())

  def test_indicators_to_sparse_ids_3d(self):
    indicators = (
        ((0, 0, 1, 0, 0), (0, 0, 0, 0, 0)),
        ((1, 0, 0, 1, 0), (0, 0, 1, 0, 0)),
        ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        ((1, 0, 0, 1, 1), (0, 0, 1, 0, 0)),
    )
    sparse_ids = sparse_ops.indicators_to_sparse_ids(indicators)
    with self.cached_session():
      _assert_sparse_tensor_value(self, sparse_tensor.SparseTensorValue(
          indices=(
              (0, 0, 0),
              (1, 0, 0), (1, 0, 1), (1, 1, 0),
              (3, 0, 0), (3, 0, 1), (3, 0, 2), (3, 1, 0)
          ), values=(
              2,
              0, 3, 2,
              0, 3, 4, 2
          ), dense_shape=(4, 2, 3),
      ), sparse_ids.eval())

  def test_int16_to_sparse_ids_2d(self):
    indicators = (
        (0, 0, 1, 0),
        (1, 0, 0, 1),
    )
    sparse_ids = sparse_ops.indicators_to_sparse_ids(
        indicators, dtype=dtypes.int16)
    with self.cached_session():
      _assert_sparse_tensor_value(self, sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=np.array((2, 0, 3), dtype=np.int16),
          dense_shape=(2, 2),
      ), sparse_ids.eval())

  def test_indicators_to_sparse_ids_ignore_value(self):
    indicators = (
        ((-1, -1, 10, -1), (-1, -1, -1, -1)),
        ((11, -1, -1, 12), (-1, -1, 13, -1)),
    )
    sparse_ids = sparse_ops.indicators_to_sparse_ids(
        indicators, ignore_value=-1)
    with self.cached_session():
      _assert_sparse_tensor_value(self, sparse_tensor.SparseTensorValue(
          indices=((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)),
          values=(2, 0, 3, 2),
          dense_shape=(2, 2, 2),
      ), sparse_ids.eval())

  def test_string_indicators_to_sparse_ids(self):
    indicators = (
        (('', '', 'A', ''), ('', '', '', '')),
        (('B', '', '', 'C'), ('', '', 'D', '')),
    )
    sparse_ids = sparse_ops.indicators_to_sparse_ids(indicators)
    with self.cached_session():
      _assert_sparse_tensor_value(self, sparse_tensor.SparseTensorValue(
          indices=((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)),
          values=(2, 0, 3, 2),
          dense_shape=(2, 2, 2),
      ), sparse_ids.eval())

  def test_string_indicators_to_sparse_ids_ignore_value(self):
    indicators = (
        (('x', 'x', 'A', 'x'), ('x', 'x', 'x', 'x')),
        (('B', 'x', 'x', 'C'), ('x', 'x', 'D', 'x')),
    )
    sparse_ids = sparse_ops.indicators_to_sparse_ids(
        indicators, ignore_value='x')
    with self.cached_session():
      _assert_sparse_tensor_value(self, sparse_tensor.SparseTensorValue(
          indices=((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)),
          values=(2, 0, 3, 2),
          dense_shape=(2, 2, 2),
      ), sparse_ids.eval())

  def test_indicators_to_sparse_ids_unknown_3d_shape(self):
    indicators_values = (
        ((0, 0, 1, 0), (0, 0, 0, 0)),
        ((1, 0, 0, 1), (0, 0, 1, 0)),
    )
    indicators = array_ops.placeholder(
        dtype=dtypes.int32, shape=(None, None, None))
    sparse_ids = sparse_ops.indicators_to_sparse_ids(indicators)
    with self.cached_session():
      _assert_sparse_tensor_value(self, sparse_tensor.SparseTensorValue(
          indices=((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)),
          values=(2, 0, 3, 2),
          dense_shape=(2, 2, 2),
      ), sparse_ids.eval(feed_dict={indicators: indicators_values}))

  def test_indicators_to_sparse_ids_unknown_rank(self):
    indicators_values = (
        ((0, 0, 1, 0), (0, 0, 0, 0)),
        ((1, 0, 0, 1), (0, 0, 1, 0)),
    )
    indicators = array_ops.placeholder(dtype=dtypes.int32)
    sparse_ids = sparse_ops.indicators_to_sparse_ids(indicators)
    with self.cached_session():
      _assert_sparse_tensor_value(self, sparse_tensor.SparseTensorValue(
          indices=((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)),
          values=(2, 0, 3, 2),
          dense_shape=(2, 2, 2),
      ), sparse_ids.eval(feed_dict={indicators: indicators_values}))


if __name__ == '__main__':
  test.main()
