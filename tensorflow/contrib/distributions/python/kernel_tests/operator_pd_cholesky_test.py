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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib import distributions
from tensorflow.contrib.distributions.python.ops import operator_pd_cholesky
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.platform import test


def softplus(x):
  return np.log(1 + np.exp(x))


class OperatorPDCholeskyTest(test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def _random_cholesky_array(self, shape):
    mat = self._rng.rand(*shape)
    chol = distribution_util.matrix_diag_transform(
        mat, transform=nn_ops.softplus)
    # Zero the upper triangle because we're using this as a true Cholesky factor
    # in our tests.
    return array_ops.matrix_band_part(chol, -1, 0).eval()

  def testLogDet(self):
    with self.test_session():
      batch_shape = ()
      for k in [1, 4]:
        chol_shape = batch_shape + (k, k)
        chol = self._random_cholesky_array(chol_shape)
        operator = operator_pd_cholesky.OperatorPDCholesky(chol)
        log_det = operator.log_det()
        expected_log_det = np.log(np.prod(np.diag(chol))**2)

        self.assertEqual(batch_shape, log_det.get_shape())
        self.assertAllClose(expected_log_det, log_det.eval())

  def testLogDetBatchMatrix(self):
    with self.test_session():
      batch_shape = (2, 3)
      for k in [1, 4]:
        chol_shape = batch_shape + (k, k)
        chol = self._random_cholesky_array(chol_shape)
        operator = operator_pd_cholesky.OperatorPDCholesky(chol)
        log_det = operator.log_det()

        self.assertEqual(batch_shape, log_det.get_shape())

        # Test the log-determinant of the [1, 1] matrix.
        chol_11 = chol[1, 1, :, :]
        expected_log_det = np.log(np.prod(np.diag(chol_11))**2)
        self.assertAllClose(expected_log_det, log_det.eval()[1, 1])

  def testSqrtMatmulSingleMatrix(self):
    with self.test_session():
      batch_shape = ()
      for k in [1, 4]:
        x_shape = batch_shape + (k, 3)
        x = self._rng.rand(*x_shape)
        chol_shape = batch_shape + (k, k)
        chol = self._random_cholesky_array(chol_shape)

        operator = operator_pd_cholesky.OperatorPDCholesky(chol)

        sqrt_operator_times_x = operator.sqrt_matmul(x)
        expected = math_ops.matmul(chol, x)

        self.assertEqual(expected.get_shape(),
                         sqrt_operator_times_x.get_shape())
        self.assertAllClose(expected.eval(), sqrt_operator_times_x.eval())

  def testSqrtMatmulBatchMatrix(self):
    with self.test_session():
      batch_shape = (2, 3)
      for k in [1, 4]:
        x_shape = batch_shape + (k, 5)
        x = self._rng.rand(*x_shape)
        chol_shape = batch_shape + (k, k)
        chol = self._random_cholesky_array(chol_shape)

        operator = operator_pd_cholesky.OperatorPDCholesky(chol)

        sqrt_operator_times_x = operator.sqrt_matmul(x)
        expected = math_ops.matmul(chol, x)

        self.assertEqual(expected.get_shape(),
                         sqrt_operator_times_x.get_shape())
        self.assertAllClose(expected.eval(), sqrt_operator_times_x.eval())

  def testSqrtMatmulBatchMatrixWithTranspose(self):
    with self.test_session():
      batch_shape = (2, 3)
      for k in [1, 4]:
        x_shape = batch_shape + (5, k)
        x = self._rng.rand(*x_shape)
        chol_shape = batch_shape + (k, k)
        chol = self._random_cholesky_array(chol_shape)

        operator = operator_pd_cholesky.OperatorPDCholesky(chol)

        sqrt_operator_times_x = operator.sqrt_matmul(x, transpose_x=True)
        # tf.batch_matmul is defined x * y, so "y" is on the right, not "x".
        expected = math_ops.matmul(chol, x, adjoint_b=True)

        self.assertEqual(expected.get_shape(),
                         sqrt_operator_times_x.get_shape())
        self.assertAllClose(expected.eval(), sqrt_operator_times_x.eval())

  def testMatmulSingleMatrix(self):
    with self.test_session():
      batch_shape = ()
      for k in [1, 4]:
        x_shape = batch_shape + (k, 5)
        x = self._rng.rand(*x_shape)
        chol_shape = batch_shape + (k, k)
        chol = self._random_cholesky_array(chol_shape)
        matrix = math_ops.matmul(chol, chol, adjoint_b=True)

        operator = operator_pd_cholesky.OperatorPDCholesky(chol)

        expected = math_ops.matmul(matrix, x)

        self.assertEqual(expected.get_shape(), operator.matmul(x).get_shape())
        self.assertAllClose(expected.eval(), operator.matmul(x).eval())

  def testMatmulBatchMatrix(self):
    with self.test_session():
      batch_shape = (2, 3)
      for k in [1, 4]:
        x_shape = batch_shape + (k, 5)
        x = self._rng.rand(*x_shape)
        chol_shape = batch_shape + (k, k)
        chol = self._random_cholesky_array(chol_shape)
        matrix = math_ops.matmul(chol, chol, adjoint_b=True)

        operator = operator_pd_cholesky.OperatorPDCholesky(chol)

        expected = math_ops.matmul(matrix, x)

        self.assertEqual(expected.get_shape(), operator.matmul(x).get_shape())
        self.assertAllClose(expected.eval(), operator.matmul(x).eval())

  def testMatmulBatchMatrixWithTranspose(self):
    with self.test_session():
      batch_shape = (2, 3)
      for k in [1, 4]:
        x_shape = batch_shape + (5, k)
        x = self._rng.rand(*x_shape)
        chol_shape = batch_shape + (k, k)
        chol = self._random_cholesky_array(chol_shape)
        matrix = math_ops.matmul(chol, chol, adjoint_b=True)

        operator = operator_pd_cholesky.OperatorPDCholesky(chol)
        operator_times_x = operator.matmul(x, transpose_x=True)

        # tf.batch_matmul is defined x * y, so "y" is on the right, not "x".
        expected = math_ops.matmul(matrix, x, adjoint_b=True)

        self.assertEqual(expected.get_shape(), operator_times_x.get_shape())
        self.assertAllClose(expected.eval(), operator_times_x.eval())

  def testShape(self):
    # All other shapes are defined by the abstractmethod shape, so we only need
    # to test this.
    with self.test_session():
      for shape in [(3, 3), (2, 3, 3), (1, 2, 3, 3)]:
        chol = self._random_cholesky_array(shape)
        operator = operator_pd_cholesky.OperatorPDCholesky(chol)
        self.assertAllEqual(shape, operator.shape().eval())

  def testToDense(self):
    with self.test_session():
      chol = self._random_cholesky_array((3, 3))
      chol_2 = chol.copy()
      chol_2[0, 2] = 1000  # Make sure upper triangular part makes no diff.
      operator = operator_pd_cholesky.OperatorPDCholesky(chol_2)
      self.assertAllClose(chol.dot(chol.T), operator.to_dense().eval())

  def testSqrtToDense(self):
    with self.test_session():
      chol = self._random_cholesky_array((2, 3, 3))
      chol_2 = chol.copy()
      chol_2[0, 0, 2] = 1000  # Make sure upper triangular part makes no diff.
      operator = operator_pd_cholesky.OperatorPDCholesky(chol_2)
      self.assertAllClose(chol, operator.sqrt_to_dense().eval())

  def testNonPositiveDefiniteMatrixRaises(self):
    # Singlular matrix with one positive eigenvalue and one zero eigenvalue.
    with self.test_session():
      lower_mat = [[1.0, 0.0], [2.0, 0.0]]
      operator = operator_pd_cholesky.OperatorPDCholesky(lower_mat)
      with self.assertRaisesOpError("x > 0 did not hold"):
        operator.to_dense().eval()

  def testNonPositiveDefiniteMatrixDoesNotRaiseIfNotVerifyPd(self):
    # Singlular matrix with one positive eigenvalue and one zero eigenvalue.
    with self.test_session():
      lower_mat = [[1.0, 0.0], [2.0, 0.0]]
      operator = operator_pd_cholesky.OperatorPDCholesky(
          lower_mat, verify_pd=False)
      operator.to_dense().eval()  # Should not raise.

  def testNotHavingTwoIdenticalLastDimsRaises(self):
    # Unless the last two dims are equal, this cannot represent a matrix, and it
    # should raise.
    with self.test_session():
      batch_vec = [[1.0], [2.0]]  # shape 2 x 1
      with self.assertRaisesOpError("x == y did not hold"):
        operator = operator_pd_cholesky.OperatorPDCholesky(batch_vec)
        operator.to_dense().eval()


class MatrixDiagTransformTest(test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(0)

  def check_off_diagonal_same(self, m1, m2):
    """Check the lower triangular part, not upper or diag."""
    self.assertAllClose(np.tril(m1, k=-1), np.tril(m2, k=-1))
    self.assertAllClose(np.triu(m1, k=1), np.triu(m2, k=1))

  def testNonBatchMatrixWithTransform(self):
    mat = self._rng.rand(4, 4)
    with self.test_session():
      chol = distributions.matrix_diag_transform(mat, transform=nn_ops.softplus)
      self.assertEqual((4, 4), chol.get_shape())

      self.check_off_diagonal_same(mat, chol.eval())
      self.assertAllClose(softplus(np.diag(mat)), np.diag(chol.eval()))

  def testNonBatchMatrixNoTransform(self):
    mat = self._rng.rand(4, 4)
    with self.test_session():
      # Default is no transform.
      chol = distributions.matrix_diag_transform(mat)
      self.assertEqual((4, 4), chol.get_shape())
      self.assertAllClose(mat, chol.eval())

  def testBatchMatrixWithTransform(self):
    mat = self._rng.rand(2, 4, 4)
    mat_0 = mat[0, :, :]
    with self.test_session():
      chol = distributions.matrix_diag_transform(mat, transform=nn_ops.softplus)

      self.assertEqual((2, 4, 4), chol.get_shape())

      chol_0 = chol.eval()[0, :, :]

      self.check_off_diagonal_same(mat_0, chol_0)
      self.assertAllClose(softplus(np.diag(mat_0)), np.diag(chol_0))

      self.check_off_diagonal_same(mat_0, chol_0)
      self.assertAllClose(softplus(np.diag(mat_0)), np.diag(chol_0))

  def testBatchMatrixNoTransform(self):
    mat = self._rng.rand(2, 4, 4)
    with self.test_session():
      # Default is no transform.
      chol = distributions.matrix_diag_transform(mat)

      self.assertEqual((2, 4, 4), chol.get_shape())
      self.assertAllClose(mat, chol.eval())


if __name__ == "__main__":
  test.main()
