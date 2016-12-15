# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
import tensorflow.python.ops.sparse_grad  # pylint: disable=unused-import
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


# TODO(zongheng): it'd be great to factor out this function and various random
# SparseTensor gen funcs.
def _sparsify(x, thresh=0.5, index_dtype=np.int64):
  x[x < thresh] = 0

  non_zero = np.where(x)
  x_indices = np.vstack(non_zero).astype(index_dtype).T
  x_values = x[non_zero]
  x_shape = x.shape

  return sparse_tensor.SparseTensor(
      indices=x_indices, values=x_values, dense_shape=x_shape), len(x_values)


class SparseToIndicatorTest(test_util.TensorFlowTestCase):

  def _SparseTensor_5x6(self, dtype):
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtype),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_2x3x4(self, dtype):
    # Includes two entries with the form [1, 1, x] : 150.
    ind = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 2], [1, 0, 3], [1, 1, 0],
                    [1, 1, 1], [1, 1, 2], [1, 2, 2]])
    val = np.array([1, 10, 12, 103, 150, 149, 150, 122])
    shape = np.array([2, 3, 4])
    return sparse_tensor.SparseTensor(
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
      expected_trues = [(0, 0, 1), (0, 1, 10), (0, 1, 12), (1, 0, 103),
                        (1, 1, 149), (1, 1, 150), (1, 2, 122)]
      for expected_true in expected_trues:
        expected_output[expected_true] = True

      self.assertAllEqual(output, expected_output)


class SparseMergeTest(test_util.TensorFlowTestCase):

  def _SparseTensorValue_3x50(self, indices_dtype, values_dtype):
    # NOTE: This input is intentionally not sorted to validate the
    # already_sorted flag below.
    ind = np.array([[0, 0], [1, 0], [1, 2], [2, 0], [2, 1], [1, 1]])
    # NB: these are not sorted
    indices = np.array([0, 13, 10, 33, 32, 14])
    values = np.array([-3, 4, 1, 9, 5, 1])
    shape = np.array([3, 3])
    indices = sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(indices, indices_dtype), np.array(shape, np.int64))
    values = sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(values, values_dtype), np.array(shape, np.int64))
    return indices, values

  def _SparseTensor_3x50(self, indices_dtype, values_dtype):
    indices, values = self._SparseTensorValue_3x50(indices_dtype, values_dtype)
    return (sparse_tensor.SparseTensor.from_value(indices),
            sparse_tensor.SparseTensor.from_value(values))

  def _AssertResultsSorted(self, output, vocab_size):
    self.assertAllEqual(output.indices,
                        [[0, 0], [1, 10], [1, 13], [1, 14], [2, 32], [2, 33]])
    self.assertAllEqual(output.values, [-3, 1, 4, 1, 5, 9])
    self.assertAllEqual(output.dense_shape, [3, vocab_size])

  def _AssertResultsNotSorted(self, output, vocab_size):
    self.assertAllEqual(output.indices,
                        [[0, 0], [1, 13], [1, 10], [2, 33], [2, 32], [1, 14]])
    self.assertAllEqual(output.values, [-3, 4, 1, 9, 5, 1])
    self.assertAllEqual(output.dense_shape, [3, vocab_size])

  def testInt32AndFloat32(self):
    vocab_size = 50
    indices_v, values_v = self._SparseTensorValue_3x50(np.int32, np.float32)
    with self.test_session(use_gpu=False) as sess:
      for indices in (indices_v,
                      sparse_tensor.SparseTensor.from_value(indices_v)):
        for values in (values_v,
                       sparse_tensor.SparseTensor.from_value(values_v)):
          sp_output = sparse_ops.sparse_merge(indices, values, vocab_size)

          output = sess.run(sp_output)
          self._AssertResultsSorted(output, vocab_size)

  def testInt64AndFloat32(self):
    vocab_size = 50
    with self.test_session(use_gpu=False) as sess:
      indices, values = self._SparseTensor_3x50(np.int64, np.float32)
      sp_output = sparse_ops.sparse_merge(indices, values, vocab_size)

      output = sess.run(sp_output)
      self._AssertResultsSorted(output, vocab_size)

  def testInt64AndFloat64(self):
    vocab_size = 50
    with self.test_session(use_gpu=False) as sess:
      indices, values = self._SparseTensor_3x50(np.int64, np.float64)
      sp_output = sparse_ops.sparse_merge(indices, values, vocab_size)

      output = sess.run(sp_output)
      self._AssertResultsSorted(output, vocab_size)

  def testInt32AndFloat32NonCanonicalOrder(self):
    vocab_size = 50
    with self.test_session(use_gpu=False) as sess:
      indices, values = self._SparseTensor_3x50(np.int32, np.float32)
      sp_output = sparse_ops.sparse_merge(
          indices, values, vocab_size, already_sorted=True)

      output = sess.run(sp_output)
      self._AssertResultsNotSorted(output, vocab_size)

  def testInt64AndFloat32NonCanonicalOrder(self):
    vocab_size = 50
    with self.test_session(use_gpu=False) as sess:
      indices, values = self._SparseTensor_3x50(np.int64, np.float32)
      sp_output = sparse_ops.sparse_merge(
          indices, values, vocab_size, already_sorted=True)

      output = sess.run(sp_output)
      self._AssertResultsNotSorted(output, vocab_size)

  def testInt64AndFloat64NonCanonicalOrder(self):
    vocab_size = 50
    with self.test_session(use_gpu=False) as sess:
      indices, values = self._SparseTensor_3x50(np.int64, np.float64)
      sp_output = sparse_ops.sparse_merge(
          indices, values, vocab_size, already_sorted=True)

      output = sess.run(sp_output)
      self._AssertResultsNotSorted(output, vocab_size)


class SparseRetainTest(test_util.TensorFlowTestCase):

  def _SparseTensorValue_5x6(self):
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    return sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(val, np.int32), np.array(shape, np.int64))

  def _SparseTensor_5x6(self):
    return sparse_tensor.SparseTensor.from_value(self._SparseTensorValue_5x6())

  def testBasic(self):
    with self.test_session(use_gpu=False) as sess:
      for sp_input in (self._SparseTensorValue_5x6(), self._SparseTensor_5x6()):
        to_retain = np.array([1, 0, 0, 1, 1, 0], dtype=np.bool)
        sp_output = sparse_ops.sparse_retain(sp_input, to_retain)

        output = sess.run(sp_output)

        self.assertAllEqual(output.indices, [[0, 0], [1, 4], [3, 2]])
        self.assertAllEqual(output.values, [0, 14, 32])
        self.assertAllEqual(output.dense_shape, [5, 6])

  def testRetainNone(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensor_5x6()
      to_retain = np.zeros((6,), dtype=np.bool)
      sp_output = sparse_ops.sparse_retain(sp_input, to_retain)

      output = sess.run(sp_output)

      self.assertAllEqual(output.indices, np.array([]).reshape((0, 2)))
      self.assertAllEqual(output.values, [])
      self.assertAllEqual(output.dense_shape, [5, 6])

  def testMismatchedRetainShape(self):
    with self.test_session(use_gpu=False):
      sp_input = self._SparseTensor_5x6()
      to_retain = np.array([1, 0, 0, 1, 0], dtype=np.bool)
      with self.assertRaises(ValueError):
        sparse_ops.sparse_retain(sp_input, to_retain)


class SparseResetShapeTest(test_util.TensorFlowTestCase):

  _IND_2_5_6 = np.array(
      [[0, 0, 0], [0, 1, 0], [0, 1, 3], [1, 1, 4], [1, 3, 2], [1, 3, 3]],
      dtype=np.int64)
  _VAL_2_5_6 = np.array([0, 10, 13, 14, 32, 33], dtype=np.int32)
  _SHP_2_5_6 = np.array([2, 5, 6], dtype=np.int64)

  def _SparseTensor_2x5x6(self):
    return sparse_tensor.SparseTensor(
        constant_op.constant(self._IND_2_5_6, dtypes.int64),
        constant_op.constant(self._VAL_2_5_6, dtypes.int32),
        constant_op.constant(self._SHP_2_5_6, dtypes.int64))

  def _SparseTensorValue_2x5x6(self):
    return sparse_tensor.SparseTensorValue(self._IND_2_5_6, self._VAL_2_5_6,
                                           self._SHP_2_5_6)

  def testBasic(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensor_2x5x6()
      new_shape = np.array([3, 6, 7], dtype=np.int64)
      sp_output = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      output = sess.run(sp_output)

      self.assertAllEqual(output.indices, [[0, 0, 0], [0, 1, 0], [0, 1, 3],
                                           [1, 1, 4], [1, 3, 2], [1, 3, 3]])
      self.assertAllEqual(output.values, [0, 10, 13, 14, 32, 33])
      self.assertAllEqual(output.dense_shape, [3, 6, 7])

  def testInputUnavailableInGraphConstructionOk(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorValue_2x5x6()
      new_shape = np.array([3, 6, 7], dtype=np.int64)
      sp_output = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      output = sess.run(sp_output)

      self.assertAllEqual(output.indices, [[0, 0, 0], [0, 1, 0], [0, 1, 3],
                                           [1, 1, 4], [1, 3, 2], [1, 3, 3]])
      self.assertAllEqual(output.values, [0, 10, 13, 14, 32, 33])
      self.assertAllEqual(output.dense_shape, [3, 6, 7])

  def testFeedInputUnavailableInGraphConstructionOk(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = array_ops.sparse_placeholder(dtype=dtypes.int32)
      new_shape = np.array([3, 6, 7], dtype=np.int64)
      sp_output = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      output = sess.run(sp_output,
                        feed_dict={sp_input: self._SparseTensorValue_2x5x6()})

      self.assertAllEqual(output.indices, [[0, 0, 0], [0, 1, 0], [0, 1, 3],
                                           [1, 1, 4], [1, 3, 2], [1, 3, 3]])
      self.assertAllEqual(output.values, [0, 10, 13, 14, 32, 33])
      self.assertAllEqual(output.dense_shape, [3, 6, 7])

  def testTightBoundingBox(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensor_2x5x6()
      sp_output = sparse_ops.sparse_reset_shape(sp_input)

      output = sess.run(sp_output)

      self.assertAllEqual(output.indices, [[0, 0, 0], [0, 1, 0], [0, 1, 3],
                                           [1, 1, 4], [1, 3, 2], [1, 3, 3]])
      self.assertAllEqual(output.values, [0, 10, 13, 14, 32, 33])
      self.assertAllEqual(output.dense_shape, [2, 4, 5])

  def testInvalidRank(self):
    with self.test_session(use_gpu=False):
      sp_input = self._SparseTensor_2x5x6()
      new_shape = np.array([3, 7], dtype=np.int64)

      with self.assertRaises(ValueError):
        sparse_ops.sparse_reset_shape(sp_input, new_shape)

  def testInvalidRankNewShapeUnavailableInGraphConstruction(self):
    with self.test_session(use_gpu=False) as sess:
      new_shape = array_ops.placeholder(dtype=dtypes.int64)
      sp_input = self._SparseTensor_2x5x6()
      out = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      with self.assertRaisesOpError("x == y did not hold element-wise"):
        sess.run(out, feed_dict={new_shape: np.array([3, 7], dtype=np.int64)})

  def testInvalidDimensionSize(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensor_2x5x6()
      new_shape = np.array([3, 7, 5], dtype=np.int64)
      out = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      with self.assertRaisesOpError("x <= y did not hold element-wise"):
        sess.run(out)

  def testInvalidDimensionSizeInputUnavailableInGraphConstruction(self):
    sp_input = array_ops.sparse_placeholder(dtype=dtypes.int32)
    with self.test_session(use_gpu=False) as sess:
      new_shape = np.array([3, 7, 5], dtype=np.int64)
      out = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      with self.assertRaisesOpError("x <= y did not hold element-wise"):
        sess.run(out, feed_dict={sp_input: self._SparseTensorValue_2x5x6()})


class SparseFillEmptyRowsTest(test_util.TensorFlowTestCase):

  def _SparseTensorValue_5x6(self):
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    return sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(val, np.int32), np.array(shape, np.int64))

  def _SparseTensor_5x6(self):
    return sparse_tensor.SparseTensor.from_value(self._SparseTensorValue_5x6())

  def _SparseTensor_String5x6(self):
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
    val = np.array(["a", "b", "c", "d", "e", "f"])
    shape = np.array([5, 6])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.string),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_2x6(self):
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4]])
    val = np.array([0, 10, 13, 14])
    shape = np.array([2, 6])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.int32),
        constant_op.constant(shape, dtypes.int64))

  def testFillNumber(self):
    with self.test_session(use_gpu=False) as sess:
      for sp_input in (self._SparseTensorValue_5x6(), self._SparseTensor_5x6()):
        sp_output, empty_row_indicator = (
            sparse_ops.sparse_fill_empty_rows(sp_input, -1))

        output, empty_row_indicator_out = sess.run(
            [sp_output, empty_row_indicator])

        self.assertAllEqual(
            output.indices,
            [[0, 0], [1, 0], [1, 3], [1, 4], [2, 0], [3, 2], [3, 3], [4, 0]])
        self.assertAllEqual(output.values, [0, 10, 13, 14, -1, 32, 33, -1])
        self.assertAllEqual(output.dense_shape, [5, 6])
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
      self.assertAllEqual(output.dense_shape, [5, 6])
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
      self.assertAllEqual(output.dense_shape, [2, 6])
      self.assertAllEqual(empty_row_indicator_out, np.zeros(2).astype(np.bool))


class SparseReduceSumTest(test_util.TensorFlowTestCase):

  # [[1, ?, 1]
  #  [?, 1, ?]]
  # where ? is implictly-zero.
  ind = np.array([[0, 0], [0, 2], [1, 1]]).astype(np.int64)
  vals = np.array([1, 1, 1]).astype(np.int32)
  dense_shape = np.array([2, 3]).astype(np.int64)

  def _compare(self, sp_t, reduction_axes, ndims, keep_dims):
    densified = sparse_ops.sparse_tensor_to_dense(sp_t).eval()

    np_ans = densified
    if reduction_axes is None:
      np_ans = np.sum(np_ans, keepdims=keep_dims)
    else:
      if not isinstance(reduction_axes, list):  # Single scalar.
        reduction_axes = [reduction_axes]
      reduction_axes = np.array(reduction_axes).astype(np.int32)
      # Handles negative axes.
      reduction_axes = (reduction_axes + ndims) % ndims
      # Loop below depends on sorted.
      reduction_axes.sort()
      for ra in reduction_axes.ravel()[::-1]:
        np_ans = np.sum(np_ans, axis=ra, keepdims=keep_dims)

    with self.test_session():
      tf_dense_ans = sparse_ops.sparse_reduce_sum(sp_t, reduction_axes,
                                                  keep_dims)
      out_dense = tf_dense_ans.eval()

      tf_sparse_ans = sparse_ops.sparse_reduce_sum_sparse(sp_t, reduction_axes,
                                                          keep_dims)
      # Convert to dense for comparison purposes.
      out_sparse = sparse_ops.sparse_tensor_to_dense(tf_sparse_ans).eval()

    self.assertAllClose(np_ans, out_dense)
    self.assertAllClose(np_ans, out_sparse)

  def _compare_all(self, sp_t, reduction_axes, ndims):
    self._compare(sp_t, reduction_axes, ndims, False)
    self._compare(sp_t, reduction_axes, ndims, True)

  def testSimpleAndRandomInputs(self):
    sp_t = sparse_tensor.SparseTensor(self.ind, self.vals, self.dense_shape)

    with self.test_session(use_gpu=False):
      self._compare_all(sp_t, None, ndims=2)
      self._compare_all(sp_t, 0, ndims=2)
      self._compare_all(sp_t, [1], ndims=2)
      self._compare_all(sp_t, [0, 1], ndims=2)
      self._compare_all(sp_t, [1, 0], ndims=2)
      self._compare_all(sp_t, [-1], ndims=2)
      self._compare_all(sp_t, [1, -2], ndims=2)

    np.random.seed(1618)
    test_dims = [(1618, 1, 11, 7, 1), (1,), (1, 1, 1)]
    with self.test_session(use_gpu=False):
      for dims in test_dims:
        sp_t, unused_nnz = _sparsify(np.random.randn(*dims))
        # reduce all using None
        self._compare_all(sp_t, None, ndims=len(dims))
        # reduce random axes from 1D to N-D
        for d in range(1, len(dims) + 1):
          axes = np.random.choice(len(dims), size=d, replace=False).tolist()
          self._compare_all(sp_t, axes, ndims=len(dims))

  def testInvalidAxes(self):
    sp_t = sparse_tensor.SparseTensor(self.ind, self.vals, self.dense_shape)
    with self.test_session(use_gpu=False):
      with self.assertRaisesOpError("Invalid reduction dimension -3"):
        sparse_ops.sparse_reduce_sum(sp_t, -3).eval()
      with self.assertRaisesOpError("Invalid reduction dimension 2"):
        sparse_ops.sparse_reduce_sum(sp_t, 2).eval()

  def testGradient(self):
    np.random.seed(8161)
    test_dims = [(11, 1, 5, 7, 1), (2, 2)]
    with self.test_session(use_gpu=False):
      for dims in test_dims:
        sp_t, nnz = _sparsify(np.random.randn(*dims))
        # reduce random axes from 1D to N-D
        for d in range(1, len(dims) + 1):
          axes = np.random.choice(len(dims), size=d, replace=False).tolist()
          reduced = sparse_ops.sparse_reduce_sum(sp_t, axes)

          err = gradient_checker.compute_gradient_error(sp_t.values, (nnz,),
                                                        reduced,
                                                        reduced.eval().shape)
          self.assertLess(err, 1e-3)

        # Tests for negative axes.
        reduced = sparse_ops.sparse_reduce_sum(sp_t, -1)
        err = gradient_checker.compute_gradient_error(sp_t.values, (nnz,),
                                                      reduced,
                                                      reduced.eval().shape)
        self.assertLess(err, 1e-3)


class SparseMathOpsTest(test_util.TensorFlowTestCase):

  def _check(self, result_tensor, result_np, input_sp_t):
    self.assertTrue(isinstance(result_tensor, sparse_tensor.SparseTensor))
    self.assertTrue(isinstance(input_sp_t, sparse_tensor.SparseTensor))
    self.assertAllEqual(input_sp_t.indices.eval(), result_tensor.indices.eval())
    self.assertAllEqual(input_sp_t.dense_shape.eval(),
                        result_tensor.dense_shape.eval())

    res_densified = sparse_ops.sparse_to_dense(result_tensor.indices,
                                               result_tensor.dense_shape,
                                               result_tensor.values).eval()
    self.assertAllEqual(result_np, res_densified)

  def testCwiseDivAndMul(self):
    np.random.seed(1618)
    sp_shapes = [(10, 10, 10), (5, 5), (1618,), (3, 3, 7)]
    dense_shapes = [(10, 10, 1), (5, 5), (1,), (1, 7)]

    with self.test_session(use_gpu=False):
      for dtype in [np.float32, np.float64, np.int32, np.int64]:
        for sp_shape, dense_shape in zip(sp_shapes, dense_shapes):
          sp_vals_np = np.random.rand(*sp_shape).astype(dtype) + 1
          dense_vals_np = np.random.rand(*dense_shape).astype(dtype) + 1
          sp_t, unused_nnz = _sparsify(sp_vals_np, thresh=1.5)
          sp_t_densified = sparse_ops.sparse_tensor_to_dense(sp_t).eval()
          dense_t = constant_op.constant(dense_vals_np)

          self._check(sp_t / dense_t, sp_t_densified / dense_vals_np, sp_t)
          # Check commutative.
          self._check(sp_t * dense_t, sp_t_densified * dense_vals_np, sp_t)
          self._check(dense_t * sp_t, sp_t_densified * dense_vals_np, sp_t)

          if dtype in [np.int32, np.int64]:
            res = sp_t / dense_t  # should invoke "__truediv__"
            self.assertEqual(res.values.eval().dtype, np.float64)

  def testCwiseAdd(self):
    with self.test_session(use_gpu=False):
      # Identity(2) + AllOnes(2,2).  Should be equal to 2 * Identity(2).
      indices = [[0, 0], [1, 1]]
      vals = [1, 1]
      shape = (2, 2)

      sp_t = sparse_tensor.SparseTensor(indices, vals, shape)
      dense_t = array_ops.ones(shape, dtype=dtypes.int32)
      self._check(
          sparse_ops.sparse_dense_cwise_add(sp_t, dense_t),
          np.identity(2) * 2, sp_t)

      # Variant of above, but broadcasts the dense side.
      dense_t = array_ops.ones([1], dtype=dtypes.int32)
      self._check(
          sparse_ops.sparse_dense_cwise_add(sp_t, dense_t),
          np.identity(2) * 2, sp_t)

  def testGradients(self):
    np.random.seed(1618)
    sp_shapes = [(10, 10, 10), (5, 5), (1618,), (3, 3, 7)]
    dense_shapes = [(10, 10, 1), (5, 5), (1,), (1, 7)]

    with self.test_session(use_gpu=False):
      for dtype in [np.float32, np.float64]:
        for sp_shape, dense_shape in zip(sp_shapes, dense_shapes):
          sp_vals_np = np.random.rand(*sp_shape).astype(dtype) + 1
          dense_vals_np = np.random.rand(*dense_shape).astype(dtype) + 1
          sp_t, nnz = _sparsify(sp_vals_np, thresh=1.5)
          dense_t = constant_op.constant(dense_vals_np)

          cmul = sp_t * dense_t
          err = gradient_checker.compute_gradient_error([sp_t.values, dense_t],
                                                        [(nnz,), dense_shape],
                                                        cmul.values, (nnz,))
          self.assertLess(err, 1e-4)

          cdiv = sp_t / dense_t
          err = gradient_checker.compute_gradient_error(sp_t.values, (nnz,),
                                                        cdiv.values, (nnz,))
          self.assertLess(err, 1e-4)
          err = gradient_checker.compute_gradient_error(
              dense_t,
              dense_shape,
              cdiv.values, (nnz,),
              x_init_value=dense_vals_np)
          self.assertLess(err, 2e-4)


class SparseSoftmaxTest(test_util.TensorFlowTestCase):

  def testEquivalentToDensified(self):
    np.random.seed(1618)
    n, m = np.random.choice(20, size=2)

    for dtype in [np.float32, np.float64]:
      sp_vals_np = np.random.rand(n, m).astype(dtype)

      batched_sp_t, unused_nnz1 = _sparsify(
          sp_vals_np.reshape((1, n, m)), thresh=0.)  # No masking.

      with self.test_session(use_gpu=False):
        densified = constant_op.constant(sp_vals_np)

        sp_result = sparse_ops.sparse_softmax(batched_sp_t).eval(
        ).values.reshape((n, m))
        dense_result = nn_ops.softmax(densified)

        self.assertAllClose(dense_result.eval(), sp_result)

  def testHigherRanks(self):
    # For the first shape:
    # First batch:
    # [?   e.]
    # [1.  ? ]
    # Second batch:
    # [e   ? ]
    # [e   e ]
    #
    # The softmax results should be:
    # [?   1.]     [1    ?]
    # [1.  ? ] and [.5  .5]
    # where ? means implicitly zero.
    #
    # The second shape: same input data, but with a higher-rank shape.
    shapes = [[2, 2, 2], [2, 1, 2, 2]]
    for shape in shapes:
      values = np.asarray(
          [0., np.e, 1., 0., np.e, 0., np.e, np.e]).reshape(shape)
      sp_t, unused_nnz = _sparsify(values, thresh=1e-2)
      expected_values = [1., 1., 1., .5, .5]

      with self.test_session(use_gpu=False):
        result = sparse_ops.sparse_softmax(sp_t).eval()

        self.assertAllEqual(expected_values, result.values)
        self.assertAllEqual(sp_t.indices.eval(), result.indices)
        self.assertAllEqual(shape, result.dense_shape)

  def testGradient(self):
    x_shape = [2, 5, 10]
    with self.test_session(use_gpu=False):
      for dtype in [np.float32, np.float64]:
        x_np = np.random.randn(*x_shape).astype(dtype)
        x_tf, nnz = _sparsify(x_np)
        y_tf = sparse_ops.sparse_softmax(x_tf)
        err = gradient_checker.compute_gradient_error(x_tf.values, (nnz,),
                                                      y_tf.values, (nnz,))
        self.assertLess(err, 1e-4)


class SparseMinimumMaximumTest(test_util.TensorFlowTestCase):

  def _assertSparseTensorValueEqual(self, a, b):
    self.assertAllEqual(a.indices, b.indices)
    self.assertAllEqual(a.values, b.values)
    self.assertAllEqual(a.dense_shape, b.dense_shape)

  def testBasic(self):
    with self.test_session(use_gpu=False):
      # 1-D, values at index 0.
      sp_zero = sparse_tensor.SparseTensor([[0]], [0], [7])
      sp_one = sparse_tensor.SparseTensor([[0]], [1], [7])
      max_tf = sparse_ops.sparse_maximum(sp_zero, sp_one).eval()
      min_tf = sparse_ops.sparse_minimum(sp_zero, sp_one).eval()
      self._assertSparseTensorValueEqual(sp_one.eval(), max_tf)
      self._assertSparseTensorValueEqual(sp_zero.eval(), min_tf)

      # Values at different indices.
      sp_zero = sparse_tensor.SparseTensor([[0]], [0], [7])
      sp_zero_2 = sparse_tensor.SparseTensor([[1]], [0], [7])
      expected = sparse_tensor.SparseTensor([[0], [1]], [0, 0], [7])
      max_tf = sparse_ops.sparse_maximum(sp_zero, sp_zero_2).eval()
      min_tf = sparse_ops.sparse_minimum(sp_zero, sp_zero_2).eval()
      self._assertSparseTensorValueEqual(expected.eval(), max_tf)
      self._assertSparseTensorValueEqual(expected.eval(), min_tf)

  def testRandom(self):
    np.random.seed(1618)
    shapes = [(13,), (6, 8), (1, 7, 1)]
    for shape in shapes:
      for dtype in [np.int32, np.int64, np.float16, np.float32, np.float64]:
        a_np = np.random.randn(*shape).astype(dtype)
        b_np = np.random.randn(*shape).astype(dtype)
        sp_a, unused_a_nnz = _sparsify(a_np, thresh=-.5)
        sp_b, unused_b_nnz = _sparsify(b_np, thresh=-.5)

        with self.test_session(use_gpu=False):
          maximum_tf = sparse_ops.sparse_maximum(sp_a, sp_b)
          maximum_tf_densified = sparse_ops.sparse_tensor_to_dense(
              maximum_tf).eval()
          minimum_tf = sparse_ops.sparse_minimum(sp_a, sp_b)
          minimum_tf_densified = sparse_ops.sparse_tensor_to_dense(
              minimum_tf).eval()

          a_densified = sparse_ops.sparse_tensor_to_dense(sp_a).eval()
          b_densified = sparse_ops.sparse_tensor_to_dense(sp_b).eval()

        self.assertAllEqual(
            np.maximum(a_densified, b_densified), maximum_tf_densified)
        self.assertAllEqual(
            np.minimum(a_densified, b_densified), minimum_tf_densified)

  def testMismatchedShapes(self):
    with self.test_session(use_gpu=False):
      sp_zero = sparse_tensor.SparseTensor([[0, 0]], [0], [1, 1])
      sp_one = sparse_tensor.SparseTensor([[0]], [1], [2])
      with self.assertRaisesOpError("Operands do not have the same ranks"):
        sparse_ops.sparse_maximum(sp_zero, sp_one).eval()

      sp_zero = sparse_tensor.SparseTensor([[0]], [0], [1])
      sp_one = sparse_tensor.SparseTensor([[0]], [1], [2])
      with self.assertRaisesOpError("Operands' shapes do not match"):
        sparse_ops.sparse_maximum(sp_zero, sp_one).eval()


class SparseTransposeTest(test.TestCase):

  def testTranspose(self):
    with self.test_session(use_gpu=False):
      np.random.seed(1618)
      shapes = [np.random.randint(1, 10, size=rank) for rank in range(1, 6)]
      for shape in shapes:
        for dtype in [np.int32, np.int64, np.float32, np.float64]:
          dn_input = np.random.randn(*shape).astype(dtype)
          rank = array_ops.rank(dn_input).eval()
          perm = np.random.choice(rank, rank, False)
          sp_input, unused_a_nnz = _sparsify(dn_input)
          sp_trans = sparse_ops.sparse_transpose(sp_input, perm=perm)
          dn_trans = sparse_ops.sparse_tensor_to_dense(sp_trans).eval()
          expected_trans = array_ops.transpose(dn_input, perm=perm).eval()
          self.assertAllEqual(dn_trans, expected_trans)


if __name__ == "__main__":
  googletest.main()
