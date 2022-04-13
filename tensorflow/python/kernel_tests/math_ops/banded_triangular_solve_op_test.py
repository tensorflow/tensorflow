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
"""Tests for tensorflow.ops.math_ops.banded_triangular_solve."""

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test


class BandedTriangularSolveOpTest(test.TestCase):

  def _verifySolveAllWays(self, x, y, dtypes, batch_dims=None):
    for lower in (False,):
      for adjoint in (False, True):
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

      # Now we need to convert a to a dense triangular matrix.
      def make_diags(diags, lower=True):
        n = len(diags[0])
        a = np.zeros(n * n, dtype=diags.dtype)
        if lower:
          for i, diag in enumerate(diags):
            a[n * i:n * n:n + 1] = diag[i:]
        else:
          diags_flip = np.flip(diags, 0)
          for i, diag in enumerate(diags_flip):
            a[i:(n - i) * n:n + 1] = diag[:(n - i)]
        return a.reshape(n, n)

      # For numpy.solve we have to explicitly zero out the strictly
      # upper or lower triangle.
      if a.size > 0:
        a_np = make_diags(a, lower=lower)
      else:
        a_np = a
      if adjoint:
        a_np = np.conj(np.transpose(a_np))

      if batch_dims is not None:
        a = np.tile(a, batch_dims + [1, 1])
        a_np = np.tile(a_np, batch_dims + [1, 1])
        b = np.tile(b, batch_dims + [1, 1])

      with self.cached_session():
        a_tf = a
        b_tf = b
        if use_placeholder:
          a_tf = array_ops.placeholder_with_default(a_tf, shape=None)
          b_tf = array_ops.placeholder_with_default(b_tf, shape=None)
        tf_ans = linalg_ops.banded_triangular_solve(
            a_tf, b_tf, lower=lower, adjoint=adjoint)
        tf_val = self.evaluate(tf_ans)
        np_ans = np.linalg.solve(a_np, b)
        self.assertEqual(np_ans.shape, tf_val.shape)
        self.assertAllClose(np_ans, tf_val)

  @test_util.run_deprecated_v1
  def testSolve(self):
    # 1x1 matrix, single rhs.
    matrix = np.array([[0.1]])
    rhs0 = np.array([[1.]])
    self._verifySolveAllWaysReal(matrix, rhs0)
    # 2x2 matrix with 2 bands, single right-hand side.
    # Corresponds to the lower triangular
    # [[1., 0.], [3., 4.]]
    # and upper triangular
    # [[2., 1.], [0., 3.]]
    matrix = np.array([[1., 4.], [2., 3.]])
    rhs0 = np.array([[1.], [1.]])
    self._verifySolveAllWaysReal(matrix, rhs0)
    # 2x2 matrix with 2 bands, 3 right-hand sides.
    rhs1 = np.array([[1., 0., 1.], [0., 1., 1.]])
    self._verifySolveAllWaysReal(matrix, rhs1)
    # 4 x 4 matrix with 2 bands, 3 right hand sides.
    # Corresponds to the lower triangular
    # [[1.,  0., 0., 0.],
    #  [-1., 2., 0., 0.],
    #  [0., -2., 3., 0.],
    #  [0., 0., -3., 4.]]
    # and upper triangular
    # [[1.,  1., 0., 0.],
    #  [0., -1., 2., 0.],
    #  [0., 0., -2., 3.],
    #  [0., 0., 0., -3.]]
    matrix = np.array([[1., 2., 3., 4.], [1., -1., -2., -3.]])
    rhs0 = np.array([[1., 0., 1.], [0., 1., 1.], [-1., 2., 1.], [0., -1., -1.]])
    self._verifySolveAllWaysReal(matrix, rhs0)

  def testSolveBandSizeSmaller(self):
    rhs0 = np.random.randn(6, 4)

    # 6 x 6 matrix with 2 bands. Ensure all non-zero entries.
    matrix = 2. * np.random.uniform(size=[3, 6]) + 1.
    self._verifySolveAllWaysReal(matrix, rhs0)

    # 6 x 6 matrix with 3 bands. Ensure all non-zero entries.
    matrix = 2. * np.random.uniform(size=[3, 6]) + 1.
    self._verifySolveAllWaysReal(matrix, rhs0)

  @test.disable_with_predicate(
      pred=test.is_built_with_rocm,
      skip_message="ROCm does not support BLAS operations for complex types")
  @test_util.run_deprecated_v1
  def testSolveComplex(self):
    # 1x1 matrix, single rhs.
    matrix = np.array([[0.1 + 1j * 0.1]])
    rhs0 = np.array([[1. + 1j]])
    self._verifySolveAllWaysComplex(matrix, rhs0)
    # 2x2 matrix with 2 bands, single right-hand side.
    # Corresponds to
    # [[1. + 1j, 0.], [4 + 1j, 2 + 1j]]
    matrix = np.array([[1., 2.], [3., 4.]]).astype(np.complex64)
    matrix += 1j * matrix
    rhs0 = np.array([[1.], [1.]]).astype(np.complex64)
    rhs0 += 1j * rhs0
    self._verifySolveAllWaysComplex(matrix, rhs0)
    # 2x2 matrix with 2 bands, 3 right-hand sides.
    rhs1 = np.array([[1., 0., 1.], [0., 1., 1.]]).astype(np.complex64)
    rhs1 += 1j * rhs1
    self._verifySolveAllWaysComplex(matrix, rhs1)

  @test_util.run_deprecated_v1
  def testSolveBatch(self):
    matrix = np.array([[1., 2.], [3., 4.]])
    rhs = np.array([[1., 0., 1.], [0., 1., 1.]])
    # Batch of 2x3x2x2 matrices, 2x3x2x3 right-hand sides.
    self._verifySolveAllWaysReal(matrix, rhs, batch_dims=[2, 3])
    # Batch of 3x2x2x2 matrices, 3x2x2x3 right-hand sides.
    self._verifySolveAllWaysReal(matrix, rhs, batch_dims=[3, 2])

    matrix = np.array([[1., 2., 3., 4.], [-1., -2., -3., -4.],
                       [-1., 1., 2., 3.]])
    rhs = np.array([[-1., 2.], [1., 1.], [0., 1.], [2., 3.]])
    # Batch of 2x3x4x4 matrices with 3 bands, 2x3x4x2 right-hand sides.
    self._verifySolveAllWaysReal(matrix, rhs, batch_dims=[2, 3])
    # Batch of 3x2x4x4 matrices with 3 bands, 3x2x4x2 right-hand sides.
    self._verifySolveAllWaysReal(matrix, rhs, batch_dims=[3, 2])

  @test.disable_with_predicate(
      pred=test.is_built_with_rocm,
      skip_message="ROCm does not support BLAS operations for complex types")
  @test_util.run_deprecated_v1
  def testSolveBatchComplex(self):
    matrix = np.array([[1., 2.], [3., 4.]]).astype(np.complex64)
    matrix += 1j * matrix
    rhs = np.array([[1., 0., 1.], [0., 1., 1.]]).astype(np.complex64)
    rhs += 1j * rhs
    # Batch of 2x3x2x2 matrices, 2x3x2x3 right-hand sides.
    self._verifySolveAllWaysComplex(matrix, rhs, batch_dims=[2, 3])
    # Batch of 3x2x2x2 matrices, 3x2x2x3 right-hand sides.
    self._verifySolveAllWaysComplex(matrix, rhs, batch_dims=[3, 2])

  @test_util.run_deprecated_v1
  def testWrongDimensions(self):
    # The matrix should have the same number of rows as the
    # right-hand sides.
    matrix = np.array([[1., 1.], [1., 1.]])
    rhs = np.array([[1., 0.]])
    with self.cached_session():
      with self.assertRaises(ValueError):
        self._verifySolve(matrix, rhs)
      with self.assertRaises(ValueError):
        self._verifySolve(matrix, rhs, batch_dims=[2, 3])

    # Number of bands exceeds the dimension of the matrix.
    matrix = np.ones((6, 4))
    rhs = np.ones((4, 2))
    with self.cached_session():
      with self.assertRaises(ValueError):
        self._verifySolve(matrix, rhs)
      with self.assertRaises(ValueError):
        self._verifySolve(matrix, rhs, batch_dims=[2, 3])

  @test_util.run_deprecated_v1
  @test_util.disable_xla("XLA cannot throw assertion errors during a kernel.")
  def testNotInvertible(self):
    # The input should be invertible.
    # The matrix is singular because it has a zero on the diagonal.
    # FIXME(rmlarsen): The GPU kernel does not check for singularity.
    singular_matrix = np.array([[1., 0., -1.], [-1., 0., 1.], [0., -1., 1.]])
    with self.cached_session():
      with self.assertRaisesOpError("Input matrix is not invertible."):
        self._verifySolve(singular_matrix, singular_matrix)
      with self.assertRaisesOpError("Input matrix is not invertible."):
        self._verifySolve(singular_matrix, singular_matrix, batch_dims=[2, 3])


if __name__ == "__main__":
  test.main()
