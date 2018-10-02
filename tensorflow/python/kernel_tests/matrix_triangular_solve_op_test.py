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
"""Tests for tensorflow.ops.math_ops.matrix_triangular_solve."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test


class MatrixTriangularSolveOpTest(test.TestCase):

  def _verifySolveAllWays(self, x, y, dtypes, batch_dims=None):
    for lower in True, False:
      for adjoint in True, False:
        for use_placeholder in True, False:
          self._verifySolve(
              x,
              y,
              lower=lower,
              adjoint=adjoint,
              batch_dims=batch_dims,
              use_placeholder=use_placeholder,
              dtypes=dtypes)

  def _verifySolveAllWaysReal(self, x, y, batch_dims=None):
    self._verifySolveAllWays(x, y, (np.float32, np.float64), batch_dims)

  def _verifySolveAllWaysComplex(self, x, y, batch_dims=None):
    self._verifySolveAllWays(x, y, (np.complex64, np.complex128), batch_dims)

  def _verifySolve(self,
                   x,
                   y,
                   lower=True,
                   adjoint=False,
                   batch_dims=None,
                   use_placeholder=False,
                   dtypes=(np.float32, np.float64)):
    for np_type in dtypes:
      a = x.astype(np_type)
      b = y.astype(np_type)
      # For numpy.solve we have to explicitly zero out the strictly
      # upper or lower triangle.
      if lower and a.size > 0:
        a_np = np.tril(a)
      elif a.size > 0:
        a_np = np.triu(a)
      else:
        a_np = a
      if adjoint:
        a_np = np.conj(np.transpose(a_np))

      if batch_dims is not None:
        a = np.tile(a, batch_dims + [1, 1])
        a_np = np.tile(a_np, batch_dims + [1, 1])
        b = np.tile(b, batch_dims + [1, 1])

      with self.test_session(use_gpu=True) as sess:
        if use_placeholder:
          a_tf = array_ops.placeholder(a.dtype)
          b_tf = array_ops.placeholder(b.dtype)
          tf_ans = linalg_ops.matrix_triangular_solve(
              a_tf, b_tf, lower=lower, adjoint=adjoint)
          tf_val = sess.run(tf_ans, feed_dict={a_tf: a, b_tf: b})
          np_ans = np.linalg.solve(a_np, b)
        else:
          a_tf = constant_op.constant(a)
          b_tf = constant_op.constant(b)
          tf_ans = linalg_ops.matrix_triangular_solve(
              a_tf, b_tf, lower=lower, adjoint=adjoint)
          tf_val = tf_ans.eval()
          np_ans = np.linalg.solve(a_np, b)
          self.assertEqual(np_ans.shape, tf_ans.get_shape())
        self.assertEqual(np_ans.shape, tf_val.shape)
        self.assertAllClose(np_ans, tf_val)

  def testSolve(self):
    # 1x1 matrix, single rhs.
    matrix = np.array([[0.1]])
    rhs0 = np.array([[1.]])
    self._verifySolveAllWaysReal(matrix, rhs0)
    # 2x2 matrices, single right-hand side.
    matrix = np.array([[1., 2.], [3., 4.]])
    rhs0 = np.array([[1.], [1.]])
    self._verifySolveAllWaysReal(matrix, rhs0)
    # 2x2 matrices, 3 right-hand sides.
    rhs1 = np.array([[1., 0., 1.], [0., 1., 1.]])
    self._verifySolveAllWaysReal(matrix, rhs1)

  def testSolveComplex(self):
    # 1x1 matrix, single rhs.
    matrix = np.array([[0.1 + 1j * 0.1]])
    rhs0 = np.array([[1. + 1j]])
    self._verifySolveAllWaysComplex(matrix, rhs0)
    # 2x2 matrices, single right-hand side.
    matrix = np.array([[1., 2.], [3., 4.]]).astype(np.complex64)
    matrix += 1j * matrix
    rhs0 = np.array([[1.], [1.]]).astype(np.complex64)
    rhs0 += 1j * rhs0
    self._verifySolveAllWaysComplex(matrix, rhs0)
    # 2x2 matrices, 3 right-hand sides.
    rhs1 = np.array([[1., 0., 1.], [0., 1., 1.]]).astype(np.complex64)
    rhs1 += 1j * rhs1
    self._verifySolveAllWaysComplex(matrix, rhs1)

  def testSolveBatch(self):
    matrix = np.array([[1., 2.], [3., 4.]])
    rhs = np.array([[1., 0., 1.], [0., 1., 1.]])
    # Batch of 2x3x2x2 matrices, 2x3x2x3 right-hand sides.
    self._verifySolveAllWaysReal(matrix, rhs, batch_dims=[2, 3])
    # Batch of 3x2x2x2 matrices, 3x2x2x3 right-hand sides.
    self._verifySolveAllWaysReal(matrix, rhs, batch_dims=[3, 2])

  def testSolveBatchComplex(self):
    matrix = np.array([[1., 2.], [3., 4.]]).astype(np.complex64)
    matrix += 1j * matrix
    rhs = np.array([[1., 0., 1.], [0., 1., 1.]]).astype(np.complex64)
    rhs += 1j * rhs
    # Batch of 2x3x2x2 matrices, 2x3x2x3 right-hand sides.
    self._verifySolveAllWaysComplex(matrix, rhs, batch_dims=[2, 3])
    # Batch of 3x2x2x2 matrices, 3x2x2x3 right-hand sides.
    self._verifySolveAllWaysComplex(matrix, rhs, batch_dims=[3, 2])

  def testNonSquareMatrix(self):
    # A non-square matrix should cause an error.
    matrix = np.array([[1., 2., 3.], [3., 4., 5.]])
    with self.cached_session():
      with self.assertRaises(ValueError):
        self._verifySolve(matrix, matrix)
      with self.assertRaises(ValueError):
        self._verifySolve(matrix, matrix, batch_dims=[2, 3])

  def testWrongDimensions(self):
    # The matrix should have the same number of rows as the
    # right-hand sides.
    matrix = np.array([[1., 0.], [0., 1.]])
    rhs = np.array([[1., 0.]])
    with self.cached_session():
      with self.assertRaises(ValueError):
        self._verifySolve(matrix, rhs)
      with self.assertRaises(ValueError):
        self._verifySolve(matrix, rhs, batch_dims=[2, 3])

  def testNotInvertible(self):
    # The input should be invertible.
    # The matrix is singular because it has a zero on the diagonal.
    singular_matrix = np.array([[1., 0., -1.], [-1., 0., 1.], [0., -1., 1.]])
    with self.cached_session():
      with self.assertRaisesOpError("Input matrix is not invertible."):
        self._verifySolve(singular_matrix, singular_matrix)
      with self.assertRaisesOpError("Input matrix is not invertible."):
        self._verifySolve(singular_matrix, singular_matrix, batch_dims=[2, 3])

  def testEmpty(self):
    self._verifySolve(np.empty([0, 2, 2]), np.empty([0, 2, 2]), lower=True)
    self._verifySolve(np.empty([2, 0, 0]), np.empty([2, 0, 0]), lower=True)
    self._verifySolve(np.empty([2, 0, 0]), np.empty([2, 0, 0]), lower=False)
    self._verifySolve(
        np.empty([2, 0, 0]), np.empty([2, 0, 0]), lower=True, batch_dims=[3, 2])


if __name__ == "__main__":
  test.main()
