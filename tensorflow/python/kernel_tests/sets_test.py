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
"""Tests for set_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


_DTYPES = set([
    tf.int8, tf.int16, tf.int32, tf.int64, tf.uint8, tf.uint16, tf.string])


def _values(values, dtype):
  return np.array(
      values,
      dtype=(np.unicode if (dtype == tf.string) else dtype.as_numpy_dtype))


def _constant(values, dtype):
  return tf.constant(_values(values, dtype), dtype=dtype)


def _dense_to_sparse(dense, dtype):
  indices = []
  values = []
  max_row_len = 0
  for row in dense:
    max_row_len = max(max_row_len, len(row))
  shape = [len(dense), max_row_len]
  row_ix = 0
  for row in dense:
    col_ix = 0
    for cell in row:
      indices.append([row_ix, col_ix])
      values.append(str(cell) if dtype == tf.string else cell)
      col_ix += 1
    row_ix += 1
  return tf.SparseTensor(
      tf.constant(indices, tf.int64),
      tf.constant(values, dtype),
      tf.constant(shape, tf.int64))


class SetOpsTest(test_util.TensorFlowTestCase):

  def test_set_size_2d(self):
    for dtype in _DTYPES:
      self._test_set_size_2d(dtype)

  def _test_set_size_2d(self, dtype):
    self.assertAllEqual(
        [1], self._set_size(_dense_to_sparse([[1]], dtype)))
    self.assertAllEqual(
        [2, 1], self._set_size(_dense_to_sparse([[1, 9], [1]], dtype)))
    self.assertAllEqual(
        [3, 0], self._set_size(_dense_to_sparse([[1, 9, 2], []], dtype)))
    self.assertAllEqual(
        [0, 3], self._set_size(_dense_to_sparse([[], [1, 9, 2]], dtype)))

  def test_set_size_duplicates_2d(self):
    for dtype in _DTYPES:
      self._test_set_size_duplicates_2d(dtype)

  def _test_set_size_duplicates_2d(self, dtype):
    self.assertAllEqual(
        [1], self._set_size(_dense_to_sparse([[1, 1, 1, 1, 1, 1]], dtype)))
    self.assertAllEqual(
        [2, 7, 3, 0, 1],
        self._set_size(_dense_to_sparse([
            [1, 9],
            [6, 7, 8, 8, 6, 7, 5, 3, 3, 0, 6, 6, 9, 0, 0, 0],
            [999, 1, -1000],
            [],
            [-1]
        ], dtype)))

  def test_set_size_3d(self):
    for dtype in _DTYPES:
      self._test_set_size_3d(dtype)

  def test_set_size_3d_invalid_indices(self):
    for dtype in _DTYPES:
      self._test_set_size_3d(dtype, invalid_indices=True)

  def _test_set_size_3d(self, dtype, invalid_indices=False):
    if invalid_indices:
      indices = tf.constant([
          [0, 1, 0], [0, 1, 1],             # 0,1
          [1, 0, 0],                        # 1,0
          [1, 1, 0], [1, 1, 1], [1, 1, 2],  # 1,1
          [0, 0, 0], [0, 0, 2],             # 0,0
                                            # 2,0
          [2, 1, 1]                         # 2,1
      ], tf.int64)
    else:
      indices = tf.constant([
          [0, 0, 0], [0, 0, 2],             # 0,0
          [0, 1, 0], [0, 1, 1],             # 0,1
          [1, 0, 0],                        # 1,0
          [1, 1, 0], [1, 1, 1], [1, 1, 2],  # 1,1
                                            # 2,0
          [2, 1, 1]                         # 2,1
      ], tf.int64)

    sp = tf.SparseTensor(
        indices,
        _constant([
            1, 9,     # 0,0
            3, 3,     # 0,1
            1,        # 1,0
            9, 7, 8,  # 1,1
                      # 2,0
            5         # 2,1
        ], dtype),
        tf.constant([3, 2, 3], tf.int64))

    if invalid_indices:
      with self.assertRaisesRegexp(tf.OpError, "out of order"):
        self._set_size(sp)
    else:
      self.assertAllEqual([
          [2,   # 0,0
           1],  # 0,1
          [1,   # 1,0
           3],  # 1,1
          [0,   # 2,0
           1]   # 2,1
      ], self._set_size(sp))

  def _set_size(self, sparse_data):
    # Validate that we get the same results with or without `validate_indices`.
    ops = [
        tf.contrib.metrics.set_size(sparse_data, validate_indices=True),
        tf.contrib.metrics.set_size(sparse_data, validate_indices=False)
    ]
    for op in ops:
      self.assertEqual(None, op.get_shape().dims)
      self.assertEqual(tf.int32, op.dtype)
    with self.test_session() as sess:
      results = sess.run(ops)
    self.assertAllEqual(results[0], results[1])
    return results[0]

  def test_set_intersection_multirow_2d(self):
    for dtype in _DTYPES:
      self._test_set_intersection_multirow_2d(dtype)

  def _test_set_intersection_multirow_2d(self, dtype):
    a_values = [[9, 1, 5], [2, 4, 3]]
    b_values = [[1, 9], [1]]
    expected_indices = [[0, 0], [0, 1]]
    expected_values = _values([1, 9], dtype)
    expected_shape = [2, 2]
    expected_counts = [2, 0]

    # Dense to sparse.
    a = _constant(a_values, dtype=dtype)
    sp_b = _dense_to_sparse(b_values, dtype=dtype)
    intersection = self._set_intersection(a, sp_b)
    self._assert_set_operation(expected_indices, expected_values,
                               expected_shape, intersection, dtype=dtype)
    self.assertAllEqual(expected_counts, self._set_intersection_count(a, sp_b))

    # Sparse to sparse.
    sp_a = _dense_to_sparse(a_values, dtype=dtype)
    intersection = self._set_intersection(sp_a, sp_b)
    self._assert_set_operation(expected_indices, expected_values,
                               expected_shape, intersection, dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_intersection_count(sp_a, sp_b))

  def test_dense_set_intersection_multirow_2d(self):
    for dtype in _DTYPES:
      self._test_dense_set_intersection_multirow_2d(dtype)

  def _test_dense_set_intersection_multirow_2d(self, dtype):
    a_values = [[9, 1, 5], [2, 4, 3]]
    b_values = [[1, 9], [1, 5]]
    expected_indices = [[0, 0], [0, 1]]
    expected_values = _values([1, 9], dtype)
    expected_shape = [2, 2]
    expected_counts = [2, 0]

    # Dense to dense.
    a = _constant(a_values, dtype)
    b = _constant(b_values, dtype)
    intersection = self._set_intersection(a, b)
    self._assert_set_operation(expected_indices, expected_values,
                               expected_shape, intersection, dtype=dtype)
    self.assertAllEqual(expected_counts, self._set_intersection_count(a, b))

  def test_set_intersection_duplicates_2d(self):
    for dtype in _DTYPES:
      self._test_set_intersection_duplicates_2d(dtype)

  def _test_set_intersection_duplicates_2d(self, dtype):
    a_values = [[1, 1, 3]]
    b_values = [[1]]
    expected_indices = [[0, 0]]
    expected_values = _values([1], dtype)
    expected_shape = [1, 1]
    expected_counts = [1]

    # Dense to dense.
    a = _constant(a_values, dtype=dtype)
    b = _constant(b_values, dtype=dtype)
    intersection = self._set_intersection(a, b)
    self._assert_set_operation(expected_indices, expected_values,
                               expected_shape, intersection, dtype=dtype)
    self.assertAllEqual(expected_counts, self._set_intersection_count(a, b))

    # Dense to sparse.
    sp_b = _dense_to_sparse(b_values, dtype=dtype)
    intersection = self._set_intersection(a, sp_b)
    self._assert_set_operation(expected_indices, expected_values,
                               expected_shape, intersection, dtype=dtype)
    self.assertAllEqual(expected_counts, self._set_intersection_count(a, sp_b))

    # Sparse to sparse.
    sp_a = _dense_to_sparse(a_values, dtype=dtype)
    intersection = self._set_intersection(sp_a, sp_b)
    self._assert_set_operation(expected_indices, expected_values,
                               expected_shape, intersection, dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_intersection_count(sp_a, sp_b))

  def test_set_intersection_3d(self):
    for dtype in _DTYPES:
      self._test_set_intersection_3d(dtype=dtype)

  def test_set_intersection_3d_invalid_indices(self):
    for dtype in _DTYPES:
      self._test_set_intersection_3d(dtype=dtype, invalid_indices=True)

  def _test_set_intersection_3d(self, dtype, invalid_indices=False):
    if invalid_indices:
      indices = tf.constant([
          [0, 1, 0], [0, 1, 1],             # 0,1
          [1, 0, 0],                        # 1,0
          [1, 1, 0], [1, 1, 1], [1, 1, 2],  # 1,1
          [0, 0, 0], [0, 0, 2],             # 0,0
                                            # 2,0
          [2, 1, 1]                         # 2,1
                                            # 3,*
      ], tf.int64)
    else:
      indices = tf.constant([
          [0, 0, 0], [0, 0, 2],             # 0,0
          [0, 1, 0], [0, 1, 1],             # 0,1
          [1, 0, 0],                        # 1,0
          [1, 1, 0], [1, 1, 1], [1, 1, 2],  # 1,1
                                            # 2,0
          [2, 1, 1]                         # 2,1
                                            # 3,*
      ], tf.int64)
    sp_a = tf.SparseTensor(
        indices,
        _constant([
            1, 9,     # 0,0
            3, 3,     # 0,1
            1,        # 1,0
            9, 7, 8,  # 1,1
                      # 2,0
            5         # 2,1
                      # 3,*
        ], dtype),
        tf.constant([4, 2, 3], tf.int64))
    sp_b = tf.SparseTensor(
        tf.constant([
            [0, 0, 0], [0, 0, 3],  # 0,0
                                   # 0,1
            [1, 0, 0],             # 1,0
            [1, 1, 0], [1, 1, 1],  # 1,1
            [2, 0, 1],             # 2,0
            [2, 1, 1],             # 2,1
            [3, 0, 0],             # 3,0
            [3, 1, 0]              # 3,1
        ], tf.int64),
        _constant([
            1, 3,  # 0,0
                   # 0,1
            3,     # 1,0
            7, 8,  # 1,1
            2,     # 2,0
            5,     # 2,1
            4,     # 3,0
            4      # 3,1
        ], dtype),
        tf.constant([4, 2, 4], tf.int64))

    if invalid_indices:
      with self.assertRaisesRegexp(tf.OpError, "out of order"):
        self._set_intersection(sp_a, sp_b)
    else:
      expected_indices = [
          [0, 0, 0],             # 0,0
                                 # 0,1
                                 # 1,0
          [1, 1, 0], [1, 1, 1],  # 1,1
                                 # 2,0
          [2, 1, 0],             # 2,1
                                 # 3,*
      ]
      expected_values = _values([
          1,     # 0,0
                 # 0,1
                 # 1,0
          7, 8,  # 1,1
                 # 2,0
          5,     # 2,1
                 # 3,*
      ], dtype)
      expected_shape = [4, 2, 2]
      expected_counts = [[
          1,  # 0,0
          0   # 0,1
      ], [
          0,  # 1,0
          2   # 1,1
      ], [
          0,  # 2,0
          1   # 2,1
      ], [
          0,  # 3,0
          0   # 3,1
      ]]

      # Sparse to sparse.
      intersection = self._set_intersection(sp_a, sp_b)
      self._assert_set_operation(expected_indices, expected_values,
                                 expected_shape, intersection, dtype=dtype)
      self.assertAllEqual(expected_counts,
                          self._set_intersection_count(sp_a, sp_b))

      # NOTE: sparse_to_dense doesn't support uint8 and uint16.
      if dtype not in [tf.uint8, tf.uint16]:
        # Dense to sparse.
        a = tf.cast(
            tf.sparse_to_dense(
                sp_a.indices,
                sp_a.dense_shape,
                sp_a.values,
                default_value="-1" if dtype == tf.string else -1),
            dtype=dtype)
        intersection = self._set_intersection(a, sp_b)
        self._assert_set_operation(expected_indices, expected_values,
                                   expected_shape, intersection, dtype=dtype)
        self.assertAllEqual(
            expected_counts, self._set_intersection_count(a, sp_b))

        # Dense to dense.
        b = tf.cast(
            tf.sparse_to_dense(
                sp_b.indices,
                sp_b.dense_shape,
                sp_b.values,
                default_value="-2" if dtype == tf.string else -2),
            dtype=dtype)
        intersection = self._set_intersection(a, b)
        self._assert_set_operation(expected_indices, expected_values,
                                   expected_shape, intersection, dtype=dtype)
        self.assertAllEqual(expected_counts, self._set_intersection_count(a, b))

  def _assert_shapes(self, input_tensor, result_sparse_tensor):
    expected_rows = (None if isinstance(input_tensor, tf.SparseTensor) else
                     input_tensor.get_shape().as_list()[0])
    expected_rank = (None if isinstance(input_tensor, tf.SparseTensor) else
                     input_tensor.get_shape().ndims)
    self.assertAllEqual((expected_rows, expected_rank),
                        result_sparse_tensor.indices.get_shape().as_list())
    self.assertAllEqual((expected_rows,),
                        result_sparse_tensor.values.get_shape().as_list())
    self.assertAllEqual((expected_rank,),
                        result_sparse_tensor.dense_shape.get_shape().as_list())

  def _set_intersection(self, a, b):
    # Validate that we get the same results with or without `validate_indices`,
    # and with a & b swapped.
    ops = (
        tf.contrib.metrics.set_intersection(a, b, validate_indices=True),
        tf.contrib.metrics.set_intersection(a, b, validate_indices=False),
        tf.contrib.metrics.set_intersection(b, a, validate_indices=True),
        tf.contrib.metrics.set_intersection(b, a, validate_indices=False),
    )
    for op in ops:
      self._assert_shapes(a, op)
    with self.test_session() as sess:
      results = sess.run(ops)
    for i in range(1, 4):
      self.assertAllEqual(results[0].indices, results[i].indices)
      self.assertAllEqual(results[0].values, results[i].values)
      self.assertAllEqual(results[0].dense_shape, results[i].dense_shape)
    return results[0]

  def _set_intersection_count(self, a, b):
    op = tf.contrib.metrics.set_size(tf.contrib.metrics.set_intersection(a, b))
    with self.test_session() as sess:
      return sess.run(op)

  def test_set_difference_multirow_2d(self):
    for dtype in _DTYPES:
      self._test_set_difference_multirow_2d(dtype)

  def _test_set_difference_multirow_2d(self, dtype):
    a_values = [[1, 1, 1], [1, 5, 9], [4, 5, 3], [5, 5, 1]]
    b_values = [[], [1, 2], [1, 2, 2], []]

    # a - b.
    expected_indices = [
        [0, 0], [1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]
    ]
    expected_values = _values([1, 5, 9, 3, 4, 5, 1, 5], dtype)
    expected_shape = [4, 3]
    expected_counts = [1, 2, 3, 2]

    # Dense to sparse.
    a = _constant(a_values, dtype=dtype)
    sp_b = _dense_to_sparse(b_values, dtype=dtype)
    difference = self._set_difference(a, sp_b, True)
    self._assert_set_operation(expected_indices, expected_values,
                               expected_shape, difference, dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(a, sp_b, True))

    # Sparse to sparse.
    sp_a = _dense_to_sparse(a_values, dtype=dtype)
    difference = self._set_difference(sp_a, sp_b, True)
    self._assert_set_operation(expected_indices, expected_values,
                               expected_shape, difference, dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(sp_a, sp_b, True))

    # b - a.
    expected_indices = [[1, 0], [2, 0], [2, 1]]
    expected_values = _values([2, 1, 2], dtype)
    expected_shape = [4, 2]
    expected_counts = [0, 1, 2, 0]

    # Dense to sparse.
    difference = self._set_difference(a, sp_b, False)
    self._assert_set_operation(expected_indices, expected_values,
                               expected_shape, difference, dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(a, sp_b, False))

    # Sparse to sparse.
    difference = self._set_difference(sp_a, sp_b, False)
    self._assert_set_operation(expected_indices, expected_values,
                               expected_shape, difference, dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(sp_a, sp_b, False))

  def test_dense_set_difference_multirow_2d(self):
    for dtype in _DTYPES:
      self._test_dense_set_difference_multirow_2d(dtype)

  def _test_dense_set_difference_multirow_2d(self, dtype):
    a_values = [[1, 5, 9], [4, 5, 3]]
    b_values = [[1, 2, 6], [1, 2, 2]]

    # a - b.
    expected_indices = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]]
    expected_values = _values([5, 9, 3, 4, 5], dtype)
    expected_shape = [2, 3]
    expected_counts = [2, 3]

    # Dense to dense.
    a = _constant(a_values, dtype=dtype)
    b = _constant(b_values, dtype=dtype)
    difference = self._set_difference(a, b, True)
    self._assert_set_operation(expected_indices, expected_values,
                               expected_shape, difference, dtype=dtype)
    self.assertAllEqual(expected_counts, self._set_difference_count(a, b, True))

    # b - a.
    expected_indices = [[0, 0], [0, 1], [1, 0], [1, 1]]
    expected_values = _values([2, 6, 1, 2], dtype)
    expected_shape = [2, 2]
    expected_counts = [2, 2]

    # Dense to dense.
    difference = self._set_difference(a, b, False)
    self._assert_set_operation(expected_indices, expected_values,
                               expected_shape, difference, dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(a, b, False))

  def test_sparse_set_difference_multirow_2d(self):
    for dtype in _DTYPES:
      self._test_sparse_set_difference_multirow_2d(dtype)

  def _test_sparse_set_difference_multirow_2d(self, dtype):
    sp_a = _dense_to_sparse(
        [[], [1, 5, 9], [4, 5, 3, 3, 4, 5], [5, 1]], dtype=dtype)
    sp_b = _dense_to_sparse([[], [1, 2], [1, 2, 2], []], dtype=dtype)

    # a - b.
    expected_indices = [[1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1]]
    expected_values = _values([5, 9, 3, 4, 5, 1, 5], dtype)
    expected_shape = [4, 3]
    expected_counts = [0, 2, 3, 2]

    difference = self._set_difference(sp_a, sp_b, True)
    self._assert_set_operation(expected_indices, expected_values,
                               expected_shape, difference, dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(sp_a, sp_b, True))

    # b - a.
    expected_indices = [[1, 0], [2, 0], [2, 1]]
    expected_values = _values([2, 1, 2], dtype)
    expected_shape = [4, 2]
    expected_counts = [0, 1, 2, 0]

    difference = self._set_difference(sp_a, sp_b, False)
    self._assert_set_operation(expected_indices, expected_values,
                               expected_shape, difference, dtype=dtype)
    self.assertAllEqual(expected_counts,
                        self._set_difference_count(sp_a, sp_b, False))

  def test_set_difference_duplicates_2d(self):
    for dtype in _DTYPES:
      self._test_set_difference_duplicates_2d(dtype)

  def _test_set_difference_duplicates_2d(self, dtype):
    a_values = [[1, 1, 3]]
    b_values = [[1, 2, 2]]

    # a - b.
    expected_indices = [[0, 0]]
    expected_values = _values([3], dtype)
    expected_shape = [1, 1]
    expected_counts = [1]

    # Dense to sparse.
    a = _constant(a_values, dtype=dtype)
    sp_b = _dense_to_sparse(b_values, dtype=dtype)
    difference = self._set_difference(a, sp_b, True)
    self._assert_set_operation(
        expected_indices, expected_values, expected_shape, difference,
        dtype=dtype)
    self.assertAllEqual(
        expected_counts, self._set_difference_count(a, sp_b, True))

    # Sparse to sparse.
    sp_a = _dense_to_sparse(a_values, dtype=dtype)
    difference = self._set_difference(sp_a, sp_b, True)
    self._assert_set_operation(
        expected_indices, expected_values, expected_shape, difference,
        dtype=dtype)
    self.assertAllEqual(
        expected_counts, self._set_difference_count(a, sp_b, True))

    # b - a.
    expected_indices = [[0, 0]]
    expected_values = _values([2], dtype)
    expected_shape = [1, 1]
    expected_counts = [1]

    # Dense to sparse.
    difference = self._set_difference(a, sp_b, False)
    self._assert_set_operation(
        expected_indices, expected_values, expected_shape, difference,
        dtype=dtype)
    self.assertAllEqual(
        expected_counts, self._set_difference_count(a, sp_b, False))

    # Sparse to sparse.
    difference = self._set_difference(sp_a, sp_b, False)
    self._assert_set_operation(
        expected_indices, expected_values, expected_shape, difference,
        dtype=dtype)
    self.assertAllEqual(
        expected_counts, self._set_difference_count(a, sp_b, False))

  def test_sparse_set_difference_3d(self):
    for dtype in _DTYPES:
      self._test_sparse_set_difference_3d(dtype)

  def test_sparse_set_difference_3d_invalid_indices(self):
    for dtype in _DTYPES:
      self._test_sparse_set_difference_3d(dtype, invalid_indices=True)

  def _test_sparse_set_difference_3d(self, dtype, invalid_indices=False):
    if invalid_indices:
      indices = tf.constant([
          [0, 1, 0], [0, 1, 1],             # 0,1
          [1, 0, 0],                        # 1,0
          [1, 1, 0], [1, 1, 1], [1, 1, 2],  # 1,1
          [0, 0, 0], [0, 0, 2],             # 0,0
                                            # 2,0
          [2, 1, 1]                         # 2,1
                                            # 3,*
      ], tf.int64)
    else:
      indices = tf.constant([
          [0, 0, 0], [0, 0, 2],             # 0,0
          [0, 1, 0], [0, 1, 1],             # 0,1
          [1, 0, 0],                        # 1,0
          [1, 1, 0], [1, 1, 1], [1, 1, 2],  # 1,1
                                            # 2,0
          [2, 1, 1]                         # 2,1
                                            # 3,*
      ], tf.int64)
    sp_a = tf.SparseTensor(
        indices,
        _constant([
            1, 9,     # 0,0
            3, 3,     # 0,1
            1,        # 1,0
            9, 7, 8,  # 1,1
                      # 2,0
            5         # 2,1
                      # 3,*
        ], dtype),
        tf.constant([4, 2, 3], tf.int64))
    sp_b = tf.SparseTensor(
        tf.constant([
            [0, 0, 0], [0, 0, 3],  # 0,0
                                   # 0,1
            [1, 0, 0],             # 1,0
            [1, 1, 0], [1, 1, 1],  # 1,1
            [2, 0, 1],             # 2,0
            [2, 1, 1],             # 2,1
            [3, 0, 0],             # 3,0
            [3, 1, 0]              # 3,1
        ], tf.int64),
        _constant([
            1, 3,  # 0,0
                   # 0,1
            3,     # 1,0
            7, 8,  # 1,1
            2,     # 2,0
            5,     # 2,1
            4,     # 3,0
            4      # 3,1
        ], dtype),
        tf.constant([4, 2, 4], tf.int64))

    if invalid_indices:
      with self.assertRaisesRegexp(tf.OpError, "out of order"):
        self._set_difference(sp_a, sp_b, False)
      with self.assertRaisesRegexp(tf.OpError, "out of order"):
        self._set_difference(sp_a, sp_b, True)
    else:
      # a-b
      expected_indices = [
          [0, 0, 0],  # 0,0
          [0, 1, 0],  # 0,1
          [1, 0, 0],  # 1,0
          [1, 1, 0],  # 1,1
                      # 2,*
                      # 3,*
      ]
      expected_values = _values([
          9,  # 0,0
          3,  # 0,1
          1,  # 1,0
          9,  # 1,1
              # 2,*
              # 3,*
      ], dtype)
      expected_shape = [4, 2, 1]
      expected_counts = [[
          1,  # 0,0
          1   # 0,1
      ], [
          1,  # 1,0
          1   # 1,1
      ], [
          0,  # 2,0
          0   # 2,1
      ], [
          0,  # 3,0
          0   # 3,1
      ]]

      difference = self._set_difference(sp_a, sp_b, True)
      self._assert_set_operation(expected_indices, expected_values,
                                 expected_shape, difference, dtype=dtype)
      self.assertAllEqual(
          expected_counts, self._set_difference_count(sp_a, sp_b))

      # b-a
      expected_indices = [
          [0, 0, 0],  # 0,0
                      # 0,1
          [1, 0, 0],  # 1,0
                      # 1,1
          [2, 0, 0],  # 2,0
                      # 2,1
          [3, 0, 0],  # 3,0
          [3, 1, 0]   # 3,1
      ]
      expected_values = _values([
          3,  # 0,0
              # 0,1
          3,  # 1,0
              # 1,1
          2,  # 2,0
              # 2,1
          4,  # 3,0
          4,  # 3,1
      ], dtype)
      expected_shape = [4, 2, 1]
      expected_counts = [[
          1,  # 0,0
          0   # 0,1
      ], [
          1,  # 1,0
          0   # 1,1
      ], [
          1,  # 2,0
          0   # 2,1
      ], [
          1,  # 3,0
          1   # 3,1
      ]]

      difference = self._set_difference(sp_a, sp_b, False)
      self._assert_set_operation(expected_indices, expected_values,
                                 expected_shape, difference, dtype=dtype)
      self.assertAllEqual(expected_counts,
                          self._set_difference_count(sp_a, sp_b, False))

  def _set_difference(self, a, b, aminusb=True):
    # Validate that we get the same results with or without `validate_indices`,
    # and with a & b swapped.
    ops = (
        tf.contrib.metrics.set_difference(
            a, b, aminusb=aminusb, validate_indices=True),
        tf.contrib.metrics.set_difference(
            a, b, aminusb=aminusb, validate_indices=False),
        tf.contrib.metrics.set_difference(
            b, a, aminusb=not aminusb, validate_indices=True),
        tf.contrib.metrics.set_difference(
            b, a, aminusb=not aminusb, validate_indices=False),
    )
    for op in ops:
      self._assert_shapes(a, op)
    with self.test_session() as sess:
      results = sess.run(ops)
    for i in range(1, 4):
      self.assertAllEqual(results[0].indices, results[i].indices)
      self.assertAllEqual(results[0].values, results[i].values)
      self.assertAllEqual(results[0].dense_shape, results[i].dense_shape)
    return results[0]

  def _set_difference_count(self, a, b, aminusb=True):
    op = tf.contrib.metrics.set_size(
        tf.contrib.metrics.set_difference(a, b, aminusb))
    with self.test_session() as sess:
      return sess.run(op)

  def test_set_union_multirow_2d(self):
    for dtype in _DTYPES:
      self._test_set_union_multirow_2d(dtype)

  def _test_set_union_multirow_2d(self, dtype):
    a_values = [[9, 1, 5], [2, 4, 3]]
    b_values = [[1, 9], [1]]
    expected_indices = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [1, 3]]
    expected_values = _values([1, 5, 9, 1, 2, 3, 4], dtype)
    expected_shape = [2, 4]
    expected_counts = [3, 4]

    # Dense to sparse.
    a = _constant(a_values, dtype=dtype)
    sp_b = _dense_to_sparse(b_values, dtype=dtype)
    union = self._set_union(a, sp_b)
    self._assert_set_operation(expected_indices, expected_values,
                               expected_shape, union, dtype=dtype)
    self.assertAllEqual(expected_counts, self._set_union_count(a, sp_b))

    # Sparse to sparse.
    sp_a = _dense_to_sparse(a_values, dtype=dtype)
    union = self._set_union(sp_a, sp_b)
    self._assert_set_operation(expected_indices, expected_values,
                               expected_shape, union, dtype=dtype)
    self.assertAllEqual(expected_counts, self._set_union_count(sp_a, sp_b))

  def test_dense_set_union_multirow_2d(self):
    for dtype in _DTYPES:
      self._test_dense_set_union_multirow_2d(dtype)

  def _test_dense_set_union_multirow_2d(self, dtype):
    a_values = [[9, 1, 5], [2, 4, 3]]
    b_values = [[1, 9], [1, 2]]
    expected_indices = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [1, 3]]
    expected_values = _values([1, 5, 9, 1, 2, 3, 4], dtype)
    expected_shape = [2, 4]
    expected_counts = [3, 4]

    # Dense to dense.
    a = _constant(a_values, dtype=dtype)
    b = _constant(b_values, dtype=dtype)
    union = self._set_union(a, b)
    self._assert_set_operation(expected_indices, expected_values,
                               expected_shape, union, dtype=dtype)
    self.assertAllEqual(expected_counts, self._set_union_count(a, b))

  def test_set_union_duplicates_2d(self):
    for dtype in _DTYPES:
      self._test_set_union_duplicates_2d(dtype)

  def _test_set_union_duplicates_2d(self, dtype):
    a_values = [[1, 1, 3]]
    b_values = [[1]]
    expected_indices = [[0, 0], [0, 1]]
    expected_values = _values([1, 3], dtype)
    expected_shape = [1, 2]

    # Dense to sparse.
    a = _constant(a_values, dtype=dtype)
    sp_b = _dense_to_sparse(b_values, dtype=dtype)
    union = self._set_union(a, sp_b)
    self._assert_set_operation(
        expected_indices, expected_values, expected_shape, union, dtype=dtype)
    self.assertAllEqual([2], self._set_union_count(a, sp_b))

    # Sparse to sparse.
    sp_a = _dense_to_sparse(a_values, dtype=dtype)
    union = self._set_union(sp_a, sp_b)
    self._assert_set_operation(
        expected_indices, expected_values, expected_shape, union, dtype=dtype)
    self.assertAllEqual([2], self._set_union_count(sp_a, sp_b))

  def test_sparse_set_union_3d(self):
    for dtype in _DTYPES:
      self._test_sparse_set_union_3d(dtype)

  def test_sparse_set_union_3d_invalid_indices(self):
    for dtype in _DTYPES:
      self._test_sparse_set_union_3d(dtype, invalid_indices=True)

  def _test_sparse_set_union_3d(self, dtype, invalid_indices=False):
    if invalid_indices:
      indices = tf.constant([
          [0, 1, 0], [0, 1, 1],             # 0,1
          [1, 0, 0],                        # 1,0
          [0, 0, 0], [0, 0, 2],             # 0,0
          [1, 1, 0], [1, 1, 1], [1, 1, 2],  # 1,1
                                            # 2,0
          [2, 1, 1]                         # 2,1
                                            # 3,*
      ], tf.int64)
    else:
      indices = tf.constant([
          [0, 0, 0], [0, 0, 2],             # 0,0
          [0, 1, 0], [0, 1, 1],             # 0,1
          [1, 0, 0],                        # 1,0
          [1, 1, 0], [1, 1, 1], [1, 1, 2],  # 1,1
                                            # 2,0
          [2, 1, 1]                         # 2,1
                                            # 3,*
      ], tf.int64)
    sp_a = tf.SparseTensor(
        indices,
        _constant([
            1, 9,     # 0,0
            3, 3,     # 0,1
            1,        # 1,0
            9, 7, 8,  # 1,1
                      # 2,0
            5         # 2,1
                      # 3,*
        ], dtype),
        tf.constant([4, 2, 3], tf.int64))
    sp_b = tf.SparseTensor(
        tf.constant([
            [0, 0, 0], [0, 0, 3],  # 0,0
                                   # 0,1
            [1, 0, 0],             # 1,0
            [1, 1, 0], [1, 1, 1],  # 1,1
            [2, 0, 1],             # 2,0
            [2, 1, 1],             # 2,1
            [3, 0, 0],             # 3,0
            [3, 1, 0]              # 3,1
        ], tf.int64),
        _constant([
            1, 3,  # 0,0
                   # 0,1
            3,     # 1,0
            7, 8,  # 1,1
            2,     # 2,0
            5,     # 2,1
            4,     # 3,0
            4      # 3,1
        ], dtype),
        tf.constant([4, 2, 4], tf.int64))

    if invalid_indices:
      with self.assertRaisesRegexp(tf.OpError, "out of order"):
        self._set_union(sp_a, sp_b)
    else:
      expected_indices = [
          [0, 0, 0], [0, 0, 1], [0, 0, 2],  # 0,0
          [0, 1, 0],                        # 0,1
          [1, 0, 0], [1, 0, 1],             # 1,0
          [1, 1, 0], [1, 1, 1], [1, 1, 2],  # 1,1
          [2, 0, 0],                        # 2,0
          [2, 1, 0],                        # 2,1
          [3, 0, 0],                        # 3,0
          [3, 1, 0],                        # 3,1
      ]
      expected_values = _values([
          1, 3, 9,  # 0,0
          3,        # 0,1
          1, 3,     # 1,0
          7, 8, 9,  # 1,1
          2,        # 2,0
          5,        # 2,1
          4,        # 3,0
          4,        # 3,1
      ], dtype)
      expected_shape = [4, 2, 3]
      expected_counts = [[
          3,  # 0,0
          1   # 0,1
      ], [
          2,  # 1,0
          3   # 1,1
      ], [
          1,  # 2,0
          1   # 2,1
      ], [
          1,  # 3,0
          1   # 3,1
      ]]

      intersection = self._set_union(sp_a, sp_b)
      self._assert_set_operation(expected_indices, expected_values,
                                 expected_shape, intersection, dtype=dtype)
      self.assertAllEqual(expected_counts, self._set_union_count(sp_a, sp_b))

  def _set_union(self, a, b):
    # Validate that we get the same results with or without `validate_indices`,
    # and with a & b swapped.
    ops = (
        tf.contrib.metrics.set_union(a, b, validate_indices=True),
        tf.contrib.metrics.set_union(a, b, validate_indices=False),
        tf.contrib.metrics.set_union(b, a, validate_indices=True),
        tf.contrib.metrics.set_union(b, a, validate_indices=False),
    )
    for op in ops:
      self._assert_shapes(a, op)
    with self.test_session() as sess:
      results = sess.run(ops)
    for i in range(1, 4):
      self.assertAllEqual(results[0].indices, results[i].indices)
      self.assertAllEqual(results[0].values, results[i].values)
      self.assertAllEqual(results[0].dense_shape, results[i].dense_shape)
    return results[0]

  def _set_union_count(self, a, b):
    op = tf.contrib.metrics.set_size(tf.contrib.metrics.set_union(a, b))
    with self.test_session() as sess:
      return sess.run(op)

  def _assert_set_operation(self, expected_indices, expected_values,
                            expected_shape, sparse_tensor, dtype):
    self.assertAllEqual(expected_indices, sparse_tensor.indices)
    self.assertAllEqual(len(expected_indices), len(expected_values))
    self.assertAllEqual(len(expected_values), len(sparse_tensor.values))
    expected_set = set()
    actual_set = set()
    last_indices = None
    for indices, expected_value, actual_value in zip(
        expected_indices, expected_values, sparse_tensor.values):
      if dtype == tf.string:
        actual_value = actual_value.decode("utf-8")
      if last_indices and (last_indices[:-1] != indices[:-1]):
        self.assertEqual(
            expected_set, actual_set, "Expected %s, got %s, at %s." % (
                expected_set, actual_set, indices))
        expected_set.clear()
        actual_set.clear()
      expected_set.add(expected_value)
      actual_set.add(actual_value)
      last_indices = indices
    self.assertEqual(
        expected_set, actual_set, "Expected %s, got %s, at %s." % (
            expected_set, actual_set, last_indices))
    self.assertAllEqual(expected_shape, sparse_tensor.dense_shape)


if __name__ == "__main__":
  googletest.main()
