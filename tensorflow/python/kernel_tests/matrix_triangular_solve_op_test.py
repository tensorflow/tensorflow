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

"""Tests for tensorflow.ops.math_ops.matrix_triangular_solve."""
# pylint: disable=unused-import,g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

# pylint: enable=unused-import,g-bad-import-order


class MatrixTriangularSolveOpTest(tf.test.TestCase):

  def _verifySolve(self, x, y, lower=True):
    for np_type in [np.float32, np.float64]:
      a = x.astype(np_type)
      b = y.astype(np_type)
      with self.test_session():
        if a.ndim == 2:
          tf_ans = tf.matrix_triangular_solve(a, b, lower=lower)
        else:
          tf_ans = tf.batch_matrix_triangular_solve(a, b, lower=lower)
        out = tf_ans.eval()
      if lower:
        np_ans = np.linalg.solve(np.tril(a), b)
      else:
        np_ans = np.linalg.solve(np.triu(a), b)
      self.assertEqual(np_ans.shape, out.shape)
      self.assertAllClose(np_ans, out)

  def testBasicLower(self):
    # 2x2 matrices, 2x1 right-hand side.
    matrix0 = np.array([[1., 2.], [3., 4.]])
    rhs0 = np.array([[1.], [1.]])
    self._verifySolve(matrix0, rhs0)

    # 2x2 matrices, 2x3 right-hand sides.
    matrix1 = np.array([[1., 2.], [3., 4.]])
    matrix2 = np.array([[1., 3.], [3., 5.]])
    rhs1 = np.array([[1., 0., 1.], [0., 1., 1.]])
    rhs2 = np.array([[1., 1., 1.], [2., 2., 2.]])
    self._verifySolve(matrix1, rhs1)
    self._verifySolve(matrix2, rhs2)
    # A multidimensional batch of 2x2 matrices and 2x3 right-hand sides.
    matrix_batch = np.concatenate([np.expand_dims(matrix1, 0), np.expand_dims(
        matrix2, 0)])
    matrix_batch = np.tile(matrix_batch, [2, 3, 1, 1])
    rhs_batch = np.concatenate([np.expand_dims(rhs1, 0), np.expand_dims(rhs2, 0)
                               ])
    rhs_batch = np.tile(rhs_batch, [2, 3, 1, 1])
    self._verifySolve(matrix_batch, rhs_batch)

  def testBasicUpper(self):
    # 2x2 matrices, 2x1 right-hand side.
    matrix0 = np.array([[1., 2.], [3., 4.]])
    rhs0 = np.array([[1.], [1.]])
    self._verifySolve(matrix0, rhs0, lower=False)

    # 2x2 matrices, 2x3 right-hand sides.
    matrix1 = np.array([[1., 2.], [3., 4.]])
    matrix2 = np.array([[1., 3.], [3., 5.]])
    rhs1 = np.array([[1., 0., 1.], [0., 1., 1.]])
    rhs2 = np.array([[1., 1., 1.], [2., 2., 2.]])
    self._verifySolve(matrix1, rhs1, lower=False)
    self._verifySolve(matrix2, rhs2, lower=False)
    # A multidimensional batch of 2x2 matrices and 2x3 right-hand sides.
    matrix_batch = np.concatenate([np.expand_dims(matrix1, 0), np.expand_dims(
        matrix2, 0)])
    matrix_batch = np.tile(matrix_batch, [2, 3, 1, 1])
    rhs_batch = np.concatenate([np.expand_dims(rhs1, 0), np.expand_dims(rhs2, 0)
                               ])
    rhs_batch = np.tile(rhs_batch, [2, 3, 1, 1])
    self._verifySolve(matrix_batch, rhs_batch, lower=False)

  def testNonSquareMatrix(self):
    # When the solve of a non-square matrix is attempted we should return
    # an error
    with self.test_session():
      with self.assertRaises(ValueError):
        matrix = tf.constant([[1., 2., 3.], [3., 4., 5.]])
        tf.matrix_triangular_solve(matrix, matrix)

  def testWrongDimensions(self):
    # The matrix and rhs should have the same number of rows as the
    # right-hand sides.
    with self.test_session():
      matrix = tf.constant([[1., 0.], [0., 1.]])
      rhs = tf.constant([[1., 0.]])
      with self.assertRaises(ValueError):
        tf.matrix_triangular_solve(matrix, rhs)

  def testNotInvertible(self):
    # The input should be invertible.
    with self.test_session():
      with self.assertRaisesOpError("Input matrix is not invertible."):
        # The matrix has a zero on the diagonal.
        matrix = tf.constant([[1., 0., -1.], [-1., 0., 1.], [0., -1., 1.]])
        tf.matrix_triangular_solve(matrix, matrix).eval()

  def testEmpty(self):
    self._verifySolve(np.empty([0, 2, 2]), np.empty([0, 2, 2]))
    self._verifySolve(np.empty([2, 0, 0]), np.empty([2, 0, 0]))


if __name__ == "__main__":
  tf.test.main()
