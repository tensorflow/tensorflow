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

from tensorflow.contrib import linalg as linalg_lib
from tensorflow.contrib.linalg.python.ops import linear_operator_test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

linalg = linalg_lib
random_seed.set_random_seed(23)


class SquareLinearOperatorMatrixTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def _operator_and_mat_and_feed_dict(self, shape, dtype, use_placeholder):
    shape = list(shape)

    matrix = linear_operator_test_util.random_positive_definite_matrix(shape,
                                                                       dtype)

    if use_placeholder:
      matrix_ph = array_ops.placeholder(dtype=dtype)
      # Evaluate here because (i) you cannot feed a tensor, and (ii)
      # values are random and we want the same value used for both mat and
      # feed_dict.
      matrix = matrix.eval()
      operator = linalg.LinearOperatorMatrix(matrix)
      feed_dict = {matrix_ph: matrix}
    else:
      operator = linalg.LinearOperatorMatrix(matrix)
      feed_dict = None

    # Convert back to Tensor.  Needed if use_placeholder, since then we have
    # already evaluated matrix to a numpy array.
    mat = ops.convert_to_tensor(matrix)

    return operator, mat, feed_dict

  def test_is_x_flags(self):
    # Matrix with two positive eigenvalues.
    matrix = [[1., 0.], [1., 11.]]
    operator = linalg.LinearOperatorMatrix(
        matrix,
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertFalse(operator.is_self_adjoint)


class SquareLinearOperatorMatrixSymmetricPositiveDefiniteTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest.

  In this test, the operator is constructed with hints that invoke the use of
  a Cholesky decomposition for solves/determinant.
  """

  def setUp(self):
    # Increase from 1e-6 to 1e-5.  This reduction in tolerance happens,
    # presumably, because we are taking a different code path in the operator
    # and the matrix.  The operator uses a Choleksy, the matrix uses standard
    # solve.
    self._atol[dtypes.float32] = 1e-5
    self._rtol[dtypes.float32] = 1e-5
    self._atol[dtypes.float64] = 1e-10
    self._rtol[dtypes.float64] = 1e-10

  @property
  def _dtypes_to_test(self):
    return [dtypes.float32, dtypes.float64]

  def _operator_and_mat_and_feed_dict(self, shape, dtype, use_placeholder):
    shape = list(shape)

    matrix = linear_operator_test_util.random_positive_definite_matrix(
        shape, dtype, force_well_conditioned=True)

    if use_placeholder:
      matrix_ph = array_ops.placeholder(dtype=dtype)
      # Evaluate here because (i) you cannot feed a tensor, and (ii)
      # values are random and we want the same value used for both mat and
      # feed_dict.
      matrix = matrix.eval()
      operator = linalg.LinearOperatorMatrix(
          matrix, is_self_adjoint=True, is_positive_definite=True)
      feed_dict = {matrix_ph: matrix}
    else:
      operator = linalg.LinearOperatorMatrix(
          matrix, is_self_adjoint=True, is_positive_definite=True)
      feed_dict = None

    # Convert back to Tensor.  Needed if use_placeholder, since then we have
    # already evaluated matrix to a numpy array.
    mat = ops.convert_to_tensor(matrix)

    return operator, mat, feed_dict

  def test_is_x_flags(self):
    # Matrix with two positive eigenvalues.
    matrix = [[1., 0.], [0., 7.]]
    operator = linalg.LinearOperatorMatrix(
        matrix, is_positive_definite=True, is_self_adjoint=True)

    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_self_adjoint)

    # Should be auto-set
    self.assertTrue(operator.is_non_singular)
    self.assertTrue(operator._is_spd)


class NonSquareLinearOperatorMatrixTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def _operator_and_mat_and_feed_dict(self, shape, dtype, use_placeholder):
    matrix = linear_operator_test_util.random_normal(shape, dtype=dtype)
    if use_placeholder:
      matrix_ph = array_ops.placeholder(dtype=dtype)
      # Evaluate here because (i) you cannot feed a tensor, and (ii)
      # values are random and we want the same value used for both mat and
      # feed_dict.
      matrix = matrix.eval()
      operator = linalg.LinearOperatorMatrix(matrix)
      feed_dict = {matrix_ph: matrix}
    else:
      operator = linalg.LinearOperatorMatrix(matrix)
      feed_dict = None

    # Convert back to Tensor.  Needed if use_placeholder, since then we have
    # already evaluated matrix to a numpy array.
    mat = ops.convert_to_tensor(matrix)

    return operator, mat, feed_dict

  def test_is_x_flags(self):
    # Matrix with two positive eigenvalues.
    matrix = [[3., 0.], [1., 1.]]
    operator = linalg.LinearOperatorMatrix(
        matrix,
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertFalse(operator.is_self_adjoint)


if __name__ == "__main__":
  test.main()
