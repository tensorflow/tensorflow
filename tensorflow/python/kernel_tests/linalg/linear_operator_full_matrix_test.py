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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

linalg = linalg_lib


@test_util.run_all_in_graph_and_eager_modes
class SquareLinearOperatorFullMatrixTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    shape = list(build_info.shape)

    matrix = linear_operator_test_util.random_positive_definite_matrix(
        shape, dtype)

    lin_op_matrix = matrix

    if use_placeholder:
      lin_op_matrix = array_ops.placeholder_with_default(matrix, shape=None)

    # Set the hints to none to test non-symmetric PD code paths.
    operator = linalg.LinearOperatorFullMatrix(
        lin_op_matrix,
        is_square=True,
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
        is_positive_definite=True if ensure_self_adjoint_and_pd else None)

    return operator, matrix

  def test_is_x_flags(self):
    # Matrix with two positive eigenvalues.
    matrix = [[1., 0.], [1., 11.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix,
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertFalse(operator.is_self_adjoint)
    # Auto-detected.
    self.assertTrue(operator.is_square)

  def test_assert_non_singular_raises_if_cond_too_big_but_finite(self):
    with self.cached_session():
      tril = linear_operator_test_util.random_tril_matrix(
          shape=(50, 50), dtype=np.float32)
      diag = np.logspace(-2, 2, 50).astype(np.float32)
      tril = array_ops.matrix_set_diag(tril, diag)
      matrix = self.evaluate(math_ops.matmul(tril, tril, transpose_b=True))
      operator = linalg.LinearOperatorFullMatrix(matrix)
      with self.assertRaisesOpError("Singular matrix"):
        # Ensure that we have finite condition number...just HUGE.
        cond = np.linalg.cond(matrix)
        self.assertTrue(np.isfinite(cond))
        self.assertGreater(cond, 1e12)
        operator.assert_non_singular().run()

  def test_assert_non_singular_raises_if_cond_infinite(self):
    with self.cached_session():
      matrix = [[1., 1.], [1., 1.]]
      # We don't pass the is_self_adjoint hint here, which means we take the
      # generic code path.
      operator = linalg.LinearOperatorFullMatrix(matrix)
      with self.assertRaisesOpError("Singular matrix"):
        operator.assert_non_singular().run()

  def test_assert_self_adjoint(self):
    matrix = [[0., 1.], [0., 1.]]
    operator = linalg.LinearOperatorFullMatrix(matrix)
    with self.cached_session():
      with self.assertRaisesOpError("not equal to its adjoint"):
        operator.assert_self_adjoint().run()

  @test_util.disable_xla("Assert statements in kernels not supported in XLA")
  def test_assert_positive_definite(self):
    matrix = [[1., 1.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(matrix, is_self_adjoint=True)
    with self.cached_session():
      with self.assertRaises(errors.InvalidArgumentError):
        operator.assert_positive_definite().run()

  def test_tape_safe(self):
    matrix = variables_module.Variable([[2.]])
    operator = linalg.LinearOperatorFullMatrix(matrix)
    self.check_tape_safe(operator)


@test_util.run_all_in_graph_and_eager_modes
class SquareLinearOperatorFullMatrixSymmetricPositiveDefiniteTest(
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

  @staticmethod
  def dtypes_to_test():
    return [dtypes.float32, dtypes.float64]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):

    # Matrix is always symmetric and positive definite in this class.
    del ensure_self_adjoint_and_pd

    shape = list(build_info.shape)

    matrix = linear_operator_test_util.random_positive_definite_matrix(
        shape, dtype, force_well_conditioned=True)

    lin_op_matrix = matrix

    if use_placeholder:
      lin_op_matrix = array_ops.placeholder_with_default(matrix, shape=None)

    operator = linalg.LinearOperatorFullMatrix(
        lin_op_matrix,
        is_square=True,
        is_self_adjoint=True,
        is_positive_definite=True)

    return operator, matrix

  def test_is_x_flags(self):
    # Matrix with two positive eigenvalues.
    matrix = [[1., 0.], [0., 7.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix, is_positive_definite=True, is_self_adjoint=True)

    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_self_adjoint)

    # Should be auto-set
    self.assertTrue(operator.is_non_singular)
    self.assertTrue(operator._can_use_cholesky)
    self.assertTrue(operator.is_square)

  @test_util.disable_xla("Assert statements in kernels not supported in XLA")
  def test_assert_non_singular(self):
    matrix = [[1., 1.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix, is_self_adjoint=True, is_positive_definite=True)
    with self.cached_session():
      # Cholesky decomposition may fail, so the error is not specific to
      # non-singular.
      with self.assertRaisesOpError(""):
        operator.assert_non_singular().run()

  def test_assert_self_adjoint(self):
    matrix = [[0., 1.], [0., 1.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix, is_self_adjoint=True, is_positive_definite=True)
    with self.cached_session():
      with self.assertRaisesOpError("not equal to its adjoint"):
        operator.assert_self_adjoint().run()

  @test_util.disable_xla("Assert statements in kernels not supported in XLA")
  def test_assert_positive_definite(self):
    matrix = [[1., 1.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix, is_self_adjoint=True, is_positive_definite=True)
    with self.cached_session():
      # Cholesky decomposition may fail, so the error is not specific to
      # non-singular.
      with self.assertRaisesOpError(""):
        operator.assert_positive_definite().run()

  def test_tape_safe(self):
    matrix = variables_module.Variable([[2.]])
    operator = linalg.LinearOperatorFullMatrix(
        matrix, is_self_adjoint=True, is_positive_definite=True)
    self.check_tape_safe(operator)


@test_util.run_all_in_graph_and_eager_modes
class NonSquareLinearOperatorFullMatrixTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    del ensure_self_adjoint_and_pd
    shape = list(build_info.shape)
    matrix = linear_operator_test_util.random_normal(shape, dtype=dtype)

    lin_op_matrix = matrix

    if use_placeholder:
      lin_op_matrix = array_ops.placeholder_with_default(matrix, shape=None)

    operator = linalg.LinearOperatorFullMatrix(lin_op_matrix, is_square=True)

    return operator, matrix

  def test_is_x_flags(self):
    matrix = [[3., 2., 1.], [1., 1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix,
        is_self_adjoint=False)
    self.assertEqual(operator.is_positive_definite, None)
    self.assertEqual(operator.is_non_singular, None)
    self.assertFalse(operator.is_self_adjoint)
    self.assertFalse(operator.is_square)

  def test_matrix_must_have_at_least_two_dims_or_raises(self):
    with self.assertRaisesRegexp(ValueError, "at least 2 dimensions"):
      linalg.LinearOperatorFullMatrix([1.])

  def test_tape_safe(self):
    matrix = variables_module.Variable([[2., 1.]])
    operator = linalg.LinearOperatorFullMatrix(matrix)
    self.check_tape_safe(operator)


if __name__ == "__main__":
  linear_operator_test_util.add_tests(SquareLinearOperatorFullMatrixTest)
  linear_operator_test_util.add_tests(NonSquareLinearOperatorFullMatrixTest)
  linear_operator_test_util.add_tests(
      SquareLinearOperatorFullMatrixSymmetricPositiveDefiniteTest)
  test.main()
