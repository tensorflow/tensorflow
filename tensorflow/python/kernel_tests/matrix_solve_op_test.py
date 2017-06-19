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


class MatrixSolveOpTest(test.TestCase):

  def _verifySolve(self, x, y, batch_dims=None):
    for adjoint in False, True:
      for np_type in [np.float32, np.float64, np.complex64, np.complex128]:
        if np_type is [np.float32, np.float64]:
          a = x.real().astype(np_type)
          b = y.real().astype(np_type)
        else:
          a = x.astype(np_type)
          b = y.astype(np_type)
        if adjoint:
          a_np = np.conj(np.transpose(a))
        else:
          a_np = a
        if batch_dims is not None:
          a = np.tile(a, batch_dims + [1, 1])
          a_np = np.tile(a_np, batch_dims + [1, 1])
          b = np.tile(b, batch_dims + [1, 1])

        np_ans = np.linalg.solve(a_np, b)
        with self.test_session():
          tf_ans = linalg_ops.matrix_solve(a, b, adjoint=adjoint)
          out = tf_ans.eval()
          self.assertEqual(tf_ans.get_shape(), out.shape)
          self.assertEqual(np_ans.shape, out.shape)
          self.assertAllClose(np_ans, out)

  def testSolve(self):
    matrix = np.array([[1. + 5.j, 2. + 6.j], [3. + 7j, 4. + 8.j]])
    # 2x1 right-hand side.
    rhs1 = np.array([[1. + 0.j], [1. + 0.j]])
    self._verifySolve(matrix, rhs1)
    # 2x3 right-hand sides.
    rhs3 = np.array(
        [[1. + 0.j, 0. + 0.j, 1. + 0.j], [0. + 0.j, 1. + 0.j, 1. + 0.j]])
    self._verifySolve(matrix, rhs3)

  def testSolveBatch(self):
    matrix = np.array([[1. + 5.j, 2. + 6.j], [3. + 7j, 4. + 8.j]])
    rhs = np.array([[1. + 0.j], [1. + 0.j]])
    # Batch of 2x3x2x2 matrices, 2x3x2x3 right-hand sides.
    self._verifySolve(matrix, rhs, batch_dims=[2, 3])
    # Batch of 3x2x2x2 matrices, 3x2x2x3 right-hand sides.
    self._verifySolve(matrix, rhs, batch_dims=[3, 2])

  def testNonSquareMatrix(self):
    # When the solve of a non-square matrix is attempted we should return
    # an error
    with self.test_session():
      with self.assertRaises(ValueError):
        matrix = constant_op.constant([[1., 2., 3.], [3., 4., 5.]])
        linalg_ops.matrix_solve(matrix, matrix)

  def testWrongDimensions(self):
    # The matrix and right-hand sides should have the same number of rows.
    with self.test_session():
      matrix = constant_op.constant([[1., 0.], [0., 1.]])
      rhs = constant_op.constant([[1., 0.]])
      with self.assertRaises(ValueError):
        linalg_ops.matrix_solve(matrix, rhs)

  def testNotInvertible(self):
    # The input should be invertible.
    with self.test_session():
      with self.assertRaisesOpError("Input matrix is not invertible."):
        # All rows of the matrix below add to zero
        matrix = constant_op.constant(
            [[1., 0., -1.], [-1., 1., 0.], [0., -1., 1.]])
        linalg_ops.matrix_solve(matrix, matrix).eval()


if __name__ == "__main__":
  test.main()
