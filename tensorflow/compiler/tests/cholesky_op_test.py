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
"""Tests for tensorflow.ops.tf.Cholesky."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test


class CholeskyOpTest(xla_test.XLATestCase):

  # Cholesky defined for float64, float32, complex64, complex128
  # (https://www.tensorflow.org/api_docs/python/tf/cholesky)
  @property
  def float_types(self):
    return set(super(CholeskyOpTest, self).float_types).intersection(
        (np.float64, np.float32, np.complex64, np.complex128))

  def _verifyCholeskyBase(self, sess, placeholder, x, chol, verification, atol):
    chol_np, verification_np = sess.run([chol, verification], {placeholder: x})
    self.assertAllClose(x, verification_np, atol=atol)
    self.assertShapeEqual(x, chol)
    # Check that the cholesky is lower triangular, and has positive diagonal
    # elements.
    if chol_np.shape[-1] > 0:
      chol_reshaped = np.reshape(chol_np, (-1, chol_np.shape[-2],
                                           chol_np.shape[-1]))
      for chol_matrix in chol_reshaped:
        self.assertAllClose(chol_matrix, np.tril(chol_matrix), atol=atol)
        self.assertTrue((np.diag(chol_matrix) > 0.0).all())

  def _verifyCholesky(self, x, atol=1e-6):
    # Verify that LL^T == x.
    with self.session() as sess:
      placeholder = array_ops.placeholder(
          dtypes.as_dtype(x.dtype), shape=x.shape)
      with self.test_scope():
        chol = linalg_ops.cholesky(placeholder)
      verification = test_util.matmul_without_tf32(chol, chol, adjoint_b=True)
      self._verifyCholeskyBase(sess, placeholder, x, chol, verification, atol)

  def testBasic(self):
    data = np.array([[4., -1., 2.], [-1., 6., 0], [2., 0., 5.]])
    for dtype in self.float_types:
      self._verifyCholesky(data.astype(dtype))

  def testBatch(self):
    for dtype in self.float_types:
      simple_array = np.array(
          [[[1., 0.], [0., 5.]]], dtype=dtype)  # shape (1, 2, 2)
      self._verifyCholesky(simple_array)
      self._verifyCholesky(np.vstack((simple_array, simple_array)))
      odd_sized_array = np.array(
          [[[4., -1., 2.], [-1., 6., 0], [2., 0., 5.]]], dtype=dtype)
      self._verifyCholesky(np.vstack((odd_sized_array, odd_sized_array)))

      # Generate random positive-definite matrices.
      matrices = np.random.rand(10, 5, 5).astype(dtype)
      for i in xrange(10):
        matrices[i] = np.dot(matrices[i].T, matrices[i])
      self._verifyCholesky(matrices, atol=1e-4)

  @test_util.run_v2_only
  def testNonSquareMatrixV2(self):
    for dtype in self.float_types:
      with self.assertRaises(errors.InvalidArgumentError):
        linalg_ops.cholesky(np.array([[1., 2., 3.], [3., 4., 5.]], dtype=dtype))
      with self.assertRaises(errors.InvalidArgumentError):
        linalg_ops.cholesky(
            np.array(
                [[[1., 2., 3.], [3., 4., 5.]], [[1., 2., 3.], [3., 4., 5.]]],
                dtype=dtype))

  @test_util.run_v1_only("Different error types")
  def testNonSquareMatrixV1(self):
    for dtype in self.float_types:
      with self.assertRaises(ValueError):
        linalg_ops.cholesky(np.array([[1., 2., 3.], [3., 4., 5.]], dtype=dtype))
      with self.assertRaises(ValueError):
        linalg_ops.cholesky(
            np.array(
                [[[1., 2., 3.], [3., 4., 5.]], [[1., 2., 3.], [3., 4., 5.]]],
                dtype=dtype))

  @test_util.run_v2_only
  def testWrongDimensionsV2(self):
    for dtype in self.float_types:
      tensor3 = constant_op.constant([1., 2.], dtype=dtype)
      with self.assertRaises(errors.InvalidArgumentError):
        linalg_ops.cholesky(tensor3)
      with self.assertRaises(errors.InvalidArgumentError):
        linalg_ops.cholesky(tensor3)

  @test_util.run_v1_only("Different error types")
  def testWrongDimensionsV1(self):
    for dtype in self.float_types:
      tensor3 = constant_op.constant([1., 2.], dtype=dtype)
      with self.assertRaises(ValueError):
        linalg_ops.cholesky(tensor3)
      with self.assertRaises(ValueError):
        linalg_ops.cholesky(tensor3)

  def testLarge2000x2000(self):
    n = 2000
    shape = (n, n)
    data = np.ones(shape).astype(np.float32) / (2.0 * n) + np.diag(
        np.ones(n).astype(np.float32))
    self._verifyCholesky(data, atol=1e-4)

  def testMatrixConditionNumbers(self):
    for dtype in self.float_types:
      condition_number = 1000
      size = 20

      # Generate random positive-definite symmetric matrices, and take their
      # Eigendecomposition.
      matrix = np.random.rand(size, size)
      matrix = np.dot(matrix.T, matrix)
      _, w = np.linalg.eigh(matrix)

      # Build new Eigenvalues exponentially distributed between 1 and
      # 1/condition_number
      v = np.exp(-np.log(condition_number) * np.linspace(0, size, size) / size)
      matrix = np.dot(np.dot(w, np.diag(v)), w.T).astype(dtype)
      self._verifyCholesky(matrix, atol=1e-4)

if __name__ == "__main__":
  test.main()
