# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for Python ops defined in sparse_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import googletest


class SparseToIndicatorTest(test_util.TensorFlowTestCase):

  def _SparseTensor_5x6(self, dtype):
    ind = np.array([
        [0, 0],
        [1, 0], [1, 3], [1, 4],
        [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    return ops.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtype),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_2x3x4(self, dtype):
    ind = np.array([
        [0, 0, 1],
        [0, 1, 0], [0, 1, 2],
        [1, 0, 3],
        [1, 1, 1], [1, 1, 3],
        [1, 2, 2]])
    val = np.array([1, 10, 12, 103, 111, 113, 122])
    shape = np.array([2, 3, 4])
    return ops.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtype),
        constant_op.constant(shape, dtypes.int64))

  def testInt32(self):
    with self.test_session(use_gpu=False):
      sp_input = self._SparseTensor_5x6(dtypes.int32)
      output = sparse_ops.sparse_to_indicator(sp_input, 50).eval()

      expected_output = np.zeros((5, 50), dtype=np.bool)
      expected_trues = ((0, 0), (1, 10), (1, 13), (1, 14), (3, 32), (3, 33))
      for expected_true in expected_trues:
        expected_output[expected_true] = True

      self.assertAllEqual(output, expected_output)

  def testInt64(self):
    with self.test_session(use_gpu=False):
      sp_input = self._SparseTensor_5x6(dtypes.int64)
      output = sparse_ops.sparse_to_indicator(sp_input, 50).eval()

      expected_output = np.zeros((5, 50), dtype=np.bool)
      expected_trues = [(0, 0), (1, 10), (1, 13), (1, 14), (3, 32), (3, 33)]
      for expected_true in expected_trues:
        expected_output[expected_true] = True

      self.assertAllEqual(output, expected_output)

  def testHigherRank(self):
    with self.test_session(use_gpu=False):
      sp_input = self._SparseTensor_2x3x4(dtypes.int64)
      output = sparse_ops.sparse_to_indicator(sp_input, 200).eval()

      expected_output = np.zeros((2, 3, 200), dtype=np.bool)
      expected_trues = [(0, 0, 1), (0, 1, 10), (0, 1, 12),
                        (1, 0, 103), (1, 1, 111), (1, 1, 113), (1, 2, 122)]
      for expected_true in expected_trues:
        expected_output[expected_true] = True

      self.assertAllEqual(output, expected_output)


class SparseRetainTest(test_util.TensorFlowTestCase):

  def _SparseTensor_5x6(self):
    ind = np.array([
        [0, 0],
        [1, 0], [1, 3], [1, 4],
        [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    return ops.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.int32),
        constant_op.constant(shape, dtypes.int64))

  def testBasic(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensor_5x6()
      to_retain = np.array([1, 0, 0, 1, 1, 0], dtype=np.bool)
      sp_output = sparse_ops.sparse_retain(sp_input, to_retain)

      output = sess.run(sp_output)

      self.assertAllEqual(output.indices, [[0, 0], [1, 4], [3, 2]])
      self.assertAllEqual(output.values, [0, 14, 32])
      self.assertAllEqual(output.shape, [5, 6])

  def testRetainNone(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensor_5x6()
      to_retain = np.zeros((6,), dtype=np.bool)
      sp_output = sparse_ops.sparse_retain(sp_input, to_retain)

      output = sess.run(sp_output)

      self.assertAllEqual(output.indices, np.array([]).reshape((0, 2)))
      self.assertAllEqual(output.values, [])
      self.assertAllEqual(output.shape, [5, 6])

  def testMismatchedRetainShape(self):
    with self.test_session(use_gpu=False):
      sp_input = self._SparseTensor_5x6()
      to_retain = np.array([1, 0, 0, 1, 0], dtype=np.bool)
      with self.assertRaises(ValueError):
        sparse_ops.sparse_retain(sp_input, to_retain)


class SparseFillEmptyRowsTest(test_util.TensorFlowTestCase):

  def _SparseTensor_5x6(self):
    ind = np.array([
        [0, 0],
        [1, 0], [1, 3], [1, 4],
        [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    return ops.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.int32),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_String5x6(self):
    ind = np.array([
        [0, 0],
        [1, 0], [1, 3], [1, 4],
        [3, 2], [3, 3]])
    val = np.array(["a", "b", "c", "d", "e", "f"])
    shape = np.array([5, 6])
    return ops.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.string),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_2x6(self):
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4]])
    val = np.array([0, 10, 13, 14])
    shape = np.array([2, 6])
    return ops.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.int32),
        constant_op.constant(shape, dtypes.int64))

  def testFillNumber(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensor_5x6()
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, -1))

      output, empty_row_indicator_out = sess.run(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(
          output.indices,
          [[0, 0], [1, 0], [1, 3], [1, 4], [2, 0], [3, 2], [3, 3], [4, 0]])
      self.assertAllEqual(output.values, [0, 10, 13, 14, -1, 32, 33, -1])
      self.assertAllEqual(output.shape, [5, 6])
      self.assertAllEqual(empty_row_indicator_out,
                          np.array([0, 0, 1, 0, 1]).astype(np.bool))

  def testFillString(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensor_String5x6()
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, ""))

      output, empty_row_indicator_out = sess.run(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(
          output.indices,
          [[0, 0], [1, 0], [1, 3], [1, 4], [2, 0], [3, 2], [3, 3], [4, 0]])
      self.assertAllEqual(output.values,
                          [b"a", b"b", b"c", b"d", b"", b"e", b"f", b""])
      self.assertAllEqual(output.shape, [5, 6])
      self.assertAllEqual(empty_row_indicator_out,
                          np.array([0, 0, 1, 0, 1]).astype(np.bool))

  def testNoEmptyRows(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensor_2x6()
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, -1))

      output, empty_row_indicator_out = sess.run(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(output.indices, [[0, 0], [1, 0], [1, 3], [1, 4]])
      self.assertAllEqual(output.values, [0, 10, 13, 14])
      self.assertAllEqual(output.shape, [2, 6])
      self.assertAllEqual(empty_row_indicator_out, np.zeros(2).astype(np.bool))


if __name__ == "__main__":
  googletest.main()
