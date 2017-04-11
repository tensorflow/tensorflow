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
"""Tests for tensorflow.ops.math_ops.matrix_solve."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test


def BatchMatMul(a, b):
  # A numpy implementation of tf.matmul().
  if a.ndim < 3:
    return np.dot(a, b)
  # Get the number of matrices.
  n = np.prod(a.shape[:-2])
  assert n == np.prod(b.shape[:-2])
  a_flat = np.reshape(a, tuple([n]) + a.shape[-2:])
  b_flat = np.reshape(b, tuple([n]) + b.shape[-2:])
  c_flat_shape = [n, a.shape[-2], b.shape[-1]]
  c_flat = np.empty(c_flat_shape)
  for i in range(n):
    c_flat[i, :, :] = np.dot(a_flat[i, :, :], b_flat[i, :, :])
  return np.reshape(c_flat, a.shape[:-1] + b_flat.shape[-1:])


def BatchRegularizedLeastSquares(matrices, rhss, l2_regularization=0.0):
  # A numpy implementation of regularized least squares solver using
  # the normal equations.
  matrix_dims = matrices.shape
  matrices_transposed = np.swapaxes(matrices, -2, -1)
  rows = matrix_dims[-2]
  cols = matrix_dims[-1]
  if rows >= cols:
    preconditioner = l2_regularization * np.identity(cols)
    gramian = BatchMatMul(matrices_transposed, matrices) + preconditioner
    inverse = np.linalg.inv(gramian)
    left_pseudo_inverse = BatchMatMul(inverse, matrices_transposed)
    return BatchMatMul(left_pseudo_inverse, rhss)
  else:
    preconditioner = l2_regularization * np.identity(rows)
    gramian = BatchMatMul(matrices, matrices_transposed) + preconditioner
    inverse = np.linalg.inv(gramian)
    right_pseudo_inverse = BatchMatMul(matrices_transposed, inverse)
    return BatchMatMul(right_pseudo_inverse, rhss)


class MatrixSolveLsOpTest(test.TestCase):

  def _verifySolve(self, x, y):
    for np_type in [np.float32, np.float64]:
      a = x.astype(np_type)
      b = y.astype(np_type)
      np_ans, _, _, _ = np.linalg.lstsq(a, b)
      for fast in [True, False]:
        with self.test_session():
          tf_ans = linalg_ops.matrix_solve_ls(a, b, fast=fast)
          ans = tf_ans.eval()
        self.assertEqual(np_ans.shape, tf_ans.get_shape())
        self.assertEqual(np_ans.shape, ans.shape)

        # Check residual norm.
        tf_r = b - BatchMatMul(a, ans)
        tf_r_norm = np.sum(tf_r * tf_r)
        np_r = b - BatchMatMul(a, np_ans)
        np_r_norm = np.sum(np_r * np_r)
        self.assertAllClose(np_r_norm, tf_r_norm)

        # Check solution.
        self.assertAllClose(np_ans, ans, atol=1e-5, rtol=1e-5)

  def _verifySolveBatch(self, x, y):
    # Since numpy.linalg.lsqr does not support batch solves, as opposed
    # to numpy.linalg.solve, we just perform this test for a fixed batch size
    # of 2x3.
    for np_type in [np.float32, np.float64]:
      a = np.tile(x.astype(np_type), [2, 3, 1, 1])
      b = np.tile(y.astype(np_type), [2, 3, 1, 1])
      np_ans = np.empty([2, 3, a.shape[-1], b.shape[-1]])
      for dim1 in range(2):
        for dim2 in range(3):
          np_ans[dim1, dim2, :, :], _, _, _ = np.linalg.lstsq(
              a[dim1, dim2, :, :], b[dim1, dim2, :, :])
      for fast in [True, False]:
        with self.test_session():
          tf_ans = linalg_ops.matrix_solve_ls(a, b, fast=fast).eval()
        self.assertEqual(np_ans.shape, tf_ans.shape)
        # Check residual norm.
        tf_r = b - BatchMatMul(a, tf_ans)
        tf_r_norm = np.sum(tf_r * tf_r)
        np_r = b - BatchMatMul(a, np_ans)
        np_r_norm = np.sum(np_r * np_r)
        self.assertAllClose(np_r_norm, tf_r_norm)
        # Check solution.
        if fast or a.shape[-2] >= a.shape[-1]:
          # We skip this test for the underdetermined case when using the
          # slow path, because Eigen does not return a minimum norm solution.
          # TODO(rmlarsen): Enable this check for all paths if/when we fix
          # Eigen's solver.
          self.assertAllClose(np_ans, tf_ans, atol=1e-5, rtol=1e-5)

  def _verifyRegularized(self, x, y, l2_regularizer):
    for np_type in [np.float32, np.float64]:
      # Test with a single matrix.
      a = x.astype(np_type)
      b = y.astype(np_type)
      np_ans = BatchRegularizedLeastSquares(a, b, l2_regularizer)
      with self.test_session():
        # Test matrix_solve_ls on regular matrices
        tf_ans = linalg_ops.matrix_solve_ls(
            a, b, l2_regularizer=l2_regularizer, fast=True).eval()
        self.assertAllClose(np_ans, tf_ans, atol=1e-5, rtol=1e-5)

      # Test with a 2x3 batch of matrices.
      a = np.tile(x.astype(np_type), [2, 3, 1, 1])
      b = np.tile(y.astype(np_type), [2, 3, 1, 1])
      np_ans = BatchRegularizedLeastSquares(a, b, l2_regularizer)
      with self.test_session():
        tf_ans = linalg_ops.matrix_solve_ls(
            a, b, l2_regularizer=l2_regularizer, fast=True).eval()
      self.assertAllClose(np_ans, tf_ans, atol=1e-5, rtol=1e-5)

  def testSquare(self):
    # 2x2 matrices, 2x3 right-hand sides.

    matrix = np.array([[1., 2.], [3., 4.]])
    rhs = np.array([[1., 0., 1.], [0., 1., 1.]])
    self._verifySolve(matrix, rhs)
    self._verifySolveBatch(matrix, rhs)
    self._verifyRegularized(matrix, rhs, l2_regularizer=0.1)

  def testOverdetermined(self):
    # 2x2 matrices, 2x3 right-hand sides.
    matrix = np.array([[1., 2.], [3., 4.], [5., 6.]])
    rhs = np.array([[1., 0., 1.], [0., 1., 1.], [1., 1., 0.]])
    self._verifySolve(matrix, rhs)
    self._verifySolveBatch(matrix, rhs)
    self._verifyRegularized(matrix, rhs, l2_regularizer=0.1)

  def testUnderdetermined(self):
    # 2x2 matrices, 2x3 right-hand sides.
    matrix = np.array([[1., 2., 3], [4., 5., 6.]])
    rhs = np.array([[1., 0., 1.], [0., 1., 1.]])
    self._verifySolve(matrix, rhs)
    self._verifySolveBatch(matrix, rhs)
    self._verifyRegularized(matrix, rhs, l2_regularizer=0.1)

  def testWrongDimensions(self):
    # The matrix and right-hand sides should have the same number of rows.
    with self.test_session():
      matrix = constant_op.constant([[1., 0.], [0., 1.]])
      rhs = constant_op.constant([[1., 0.]])
      with self.assertRaises(ValueError):
        linalg_ops.matrix_solve_ls(matrix, rhs)

  def testEmpty(self):
    full = np.array([[1., 2.], [3., 4.], [5., 6.]])
    empty0 = np.empty([3, 0])
    empty1 = np.empty([0, 2])
    for fast in [True, False]:
      with self.test_session():
        tf_ans = linalg_ops.matrix_solve_ls(empty0, empty0, fast=fast).eval()
        self.assertEqual(tf_ans.shape, (0, 0))
        tf_ans = linalg_ops.matrix_solve_ls(empty0, full, fast=fast).eval()
        self.assertEqual(tf_ans.shape, (0, 2))
        tf_ans = linalg_ops.matrix_solve_ls(full, empty0, fast=fast).eval()
        self.assertEqual(tf_ans.shape, (2, 0))
        tf_ans = linalg_ops.matrix_solve_ls(empty1, empty1, fast=fast).eval()
        self.assertEqual(tf_ans.shape, (2, 2))

  def testBatchResultSize(self):
    # 3x3x3 matrices, 3x3x1 right-hand sides.
    matrix = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.] * 3).reshape(3, 3, 3)
    rhs = np.array([1., 2., 3.] * 3).reshape(3, 3, 1)
    answer = linalg_ops.matrix_solve(matrix, rhs)
    ls_answer = linalg_ops.matrix_solve_ls(matrix, rhs)
    self.assertEqual(ls_answer.get_shape(), [3, 3, 1])
    self.assertEqual(answer.get_shape(), [3, 3, 1])


if __name__ == "__main__":
  test.main()
