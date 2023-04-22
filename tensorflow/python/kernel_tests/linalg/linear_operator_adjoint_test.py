# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_adjoint
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

linalg = linalg_lib

LinearOperatorAdjoint = linear_operator_adjoint.LinearOperatorAdjoint  # pylint: disable=invalid-name


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorAdjointTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def setUp(self):
    self._atol[dtypes.complex64] = 1e-5
    self._rtol[dtypes.complex64] = 1e-5

  def operator_and_matrix(self,
                          build_info,
                          dtype,
                          use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    shape = list(build_info.shape)

    if ensure_self_adjoint_and_pd:
      matrix = linear_operator_test_util.random_positive_definite_matrix(
          shape, dtype, force_well_conditioned=True)
    else:
      matrix = linear_operator_test_util.random_tril_matrix(
          shape, dtype, force_well_conditioned=True, remove_upper=True)

    lin_op_matrix = matrix

    if use_placeholder:
      lin_op_matrix = array_ops.placeholder_with_default(matrix, shape=None)

    if ensure_self_adjoint_and_pd:
      operator = LinearOperatorAdjoint(
          linalg.LinearOperatorFullMatrix(
              lin_op_matrix, is_positive_definite=True, is_self_adjoint=True))
    else:
      operator = LinearOperatorAdjoint(
          linalg.LinearOperatorLowerTriangular(lin_op_matrix))

    return operator, linalg.adjoint(matrix)

  def test_base_operator_hint_used(self):
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix,
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    operator_adjoint = LinearOperatorAdjoint(operator)
    self.assertTrue(operator_adjoint.is_positive_definite)
    self.assertTrue(operator_adjoint.is_non_singular)
    self.assertFalse(operator_adjoint.is_self_adjoint)

  def test_supplied_hint_used(self):
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(matrix)
    operator_adjoint = LinearOperatorAdjoint(
        operator,
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator_adjoint.is_positive_definite)
    self.assertTrue(operator_adjoint.is_non_singular)
    self.assertFalse(operator_adjoint.is_self_adjoint)

  def test_contradicting_hints_raise(self):
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix, is_positive_definite=False)
    with self.assertRaisesRegex(ValueError, "positive-definite"):
      LinearOperatorAdjoint(operator, is_positive_definite=True)

    operator = linalg.LinearOperatorFullMatrix(matrix, is_self_adjoint=False)
    with self.assertRaisesRegex(ValueError, "self-adjoint"):
      LinearOperatorAdjoint(operator, is_self_adjoint=True)

  def test_name(self):
    matrix = [[11., 0.], [1., 8.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix, name="my_operator", is_non_singular=True)

    operator = LinearOperatorAdjoint(operator)

    self.assertEqual("my_operator_adjoint", operator.name)

  def test_matmul_adjoint_operator(self):
    matrix1 = np.random.randn(4, 4)
    matrix2 = np.random.randn(4, 4)
    full_matrix1 = linalg.LinearOperatorFullMatrix(matrix1)
    full_matrix2 = linalg.LinearOperatorFullMatrix(matrix2)

    self.assertAllClose(
        np.matmul(matrix1, matrix2.T),
        self.evaluate(
            full_matrix1.matmul(full_matrix2, adjoint_arg=True).to_dense()))

    self.assertAllClose(
        np.matmul(matrix1.T, matrix2),
        self.evaluate(
            full_matrix1.matmul(full_matrix2, adjoint=True).to_dense()))

    self.assertAllClose(
        np.matmul(matrix1.T, matrix2.T),
        self.evaluate(
            full_matrix1.matmul(full_matrix2, adjoint=True,
                                adjoint_arg=True).to_dense()))

  def test_matmul_adjoint_complex_operator(self):
    matrix1 = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    matrix2 = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    full_matrix1 = linalg.LinearOperatorFullMatrix(matrix1)
    full_matrix2 = linalg.LinearOperatorFullMatrix(matrix2)

    self.assertAllClose(
        np.matmul(matrix1,
                  matrix2.conj().T),
        self.evaluate(
            full_matrix1.matmul(full_matrix2, adjoint_arg=True).to_dense()))

    self.assertAllClose(
        np.matmul(matrix1.conj().T, matrix2),
        self.evaluate(
            full_matrix1.matmul(full_matrix2, adjoint=True).to_dense()))

    self.assertAllClose(
        np.matmul(matrix1.conj().T,
                  matrix2.conj().T),
        self.evaluate(
            full_matrix1.matmul(full_matrix2, adjoint=True,
                                adjoint_arg=True).to_dense()))

  def test_matvec(self):
    matrix = np.array([[1., 2.], [3., 4.]])
    x = np.array([1., 2.])
    operator = linalg.LinearOperatorFullMatrix(matrix)
    self.assertAllClose(matrix.dot(x), self.evaluate(operator.matvec(x)))
    self.assertAllClose(matrix.T.dot(x), self.evaluate(operator.H.matvec(x)))

  def test_solve_adjoint_operator(self):
    matrix1 = self.evaluate(
        linear_operator_test_util.random_tril_matrix(
            [4, 4], dtype=dtypes.float64, force_well_conditioned=True))
    matrix2 = np.random.randn(4, 4)
    full_matrix1 = linalg.LinearOperatorLowerTriangular(
        matrix1, is_non_singular=True)
    full_matrix2 = linalg.LinearOperatorFullMatrix(matrix2)

    self.assertAllClose(
        self.evaluate(linalg.triangular_solve(matrix1, matrix2.T)),
        self.evaluate(
            full_matrix1.solve(full_matrix2, adjoint_arg=True).to_dense()))

    self.assertAllClose(
        self.evaluate(linalg.triangular_solve(matrix1.T, matrix2, lower=False)),
        self.evaluate(
            full_matrix1.solve(full_matrix2, adjoint=True).to_dense()))

    self.assertAllClose(
        self.evaluate(
            linalg.triangular_solve(matrix1.T, matrix2.T, lower=False)),
        self.evaluate(
            full_matrix1.solve(full_matrix2, adjoint=True,
                               adjoint_arg=True).to_dense()))

  def test_solve_adjoint_complex_operator(self):
    matrix1 = self.evaluate(
        linear_operator_test_util.random_tril_matrix(
            [4, 4], dtype=dtypes.complex128, force_well_conditioned=True) +
        1j * linear_operator_test_util.random_tril_matrix(
            [4, 4], dtype=dtypes.complex128, force_well_conditioned=True))
    matrix2 = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)

    full_matrix1 = linalg.LinearOperatorLowerTriangular(
        matrix1, is_non_singular=True)
    full_matrix2 = linalg.LinearOperatorFullMatrix(matrix2)

    self.assertAllClose(
        self.evaluate(linalg.triangular_solve(matrix1,
                                              matrix2.conj().T)),
        self.evaluate(
            full_matrix1.solve(full_matrix2, adjoint_arg=True).to_dense()))

    self.assertAllClose(
        self.evaluate(
            linalg.triangular_solve(matrix1.conj().T, matrix2, lower=False)),
        self.evaluate(
            full_matrix1.solve(full_matrix2, adjoint=True).to_dense()))

    self.assertAllClose(
        self.evaluate(
            linalg.triangular_solve(
                matrix1.conj().T, matrix2.conj().T, lower=False)),
        self.evaluate(
            full_matrix1.solve(full_matrix2, adjoint=True,
                               adjoint_arg=True).to_dense()))

  def test_solvevec(self):
    matrix = np.array([[1., 2.], [3., 4.]])
    inv_matrix = np.linalg.inv(matrix)
    x = np.array([1., 2.])
    operator = linalg.LinearOperatorFullMatrix(matrix)
    self.assertAllClose(inv_matrix.dot(x), self.evaluate(operator.solvevec(x)))
    self.assertAllClose(
        inv_matrix.T.dot(x), self.evaluate(operator.H.solvevec(x)))

  def test_tape_safe(self):
    matrix = variables_module.Variable([[1., 2.], [3., 4.]])
    operator = LinearOperatorAdjoint(linalg.LinearOperatorFullMatrix(matrix))
    self.check_tape_safe(operator)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorAdjointNonSquareTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  """Tests done in the base class NonSquareLinearOperatorDerivedClassTest."""

  def operator_and_matrix(self, build_info, dtype, use_placeholder):
    shape_before_adjoint = list(build_info.shape)
    # We need to swap the last two dimensions because we are taking the adjoint
    # of this operator
    shape_before_adjoint[-1], shape_before_adjoint[-2] = (
        shape_before_adjoint[-2], shape_before_adjoint[-1])
    matrix = linear_operator_test_util.random_normal(
        shape_before_adjoint, dtype=dtype)

    lin_op_matrix = matrix

    if use_placeholder:
      lin_op_matrix = array_ops.placeholder_with_default(matrix, shape=None)

    operator = LinearOperatorAdjoint(
        linalg.LinearOperatorFullMatrix(lin_op_matrix))

    return operator, linalg.adjoint(matrix)


if __name__ == "__main__":
  linear_operator_test_util.add_tests(LinearOperatorAdjointTest)
  test.main()
