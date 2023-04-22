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

from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

linalg = linalg_lib


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorLowerTriangularTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @staticmethod
  def skip_these_tests():
    # Cholesky does not make sense for triangular matrices.
    return ["cholesky"]

  def operator_and_matrix(self, build_info, dtype, use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    shape = list(build_info.shape)
    # Upper triangle will be nonzero, but ignored.
    # Use a diagonal that ensures this matrix is well conditioned.
    tril = linear_operator_test_util.random_tril_matrix(
        shape, dtype=dtype, force_well_conditioned=True, remove_upper=False)
    if ensure_self_adjoint_and_pd:
      # Get the diagonal and make the matrix out of it.
      tril = array_ops.matrix_diag_part(tril)
      tril = math_ops.abs(tril) + 1e-1
      tril = array_ops.matrix_diag(tril)

    lin_op_tril = tril

    if use_placeholder:
      lin_op_tril = array_ops.placeholder_with_default(lin_op_tril, shape=None)

    operator = linalg.LinearOperatorLowerTriangular(
        lin_op_tril,
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
        is_positive_definite=True if ensure_self_adjoint_and_pd else None)

    matrix = array_ops.matrix_band_part(tril, -1, 0)

    return operator, matrix

  def test_assert_non_singular(self):
    # Singular matrix with one positive eigenvalue and one zero eigenvalue.
    with self.cached_session():
      tril = [[1., 0.], [1., 0.]]
      operator = linalg.LinearOperatorLowerTriangular(tril)
      with self.assertRaisesOpError("Singular operator"):
        operator.assert_non_singular().run()

  def test_is_x_flags(self):
    # Matrix with two positive eigenvalues.
    tril = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorLowerTriangular(
        tril,
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertFalse(operator.is_self_adjoint)

  def test_tril_must_have_at_least_two_dims_or_raises(self):
    with self.assertRaisesRegex(ValueError, "at least 2 dimensions"):
      linalg.LinearOperatorLowerTriangular([1.])

  def test_triangular_diag_matmul(self):
    operator1 = linalg_lib.LinearOperatorLowerTriangular(
        [[1., 0., 0.], [2., 1., 0.], [2., 3., 3.]])
    operator2 = linalg_lib.LinearOperatorDiag([2., 2., 3.])
    operator_matmul = operator1.matmul(operator2)
    self.assertTrue(isinstance(
        operator_matmul,
        linalg_lib.LinearOperatorLowerTriangular))
    self.assertAllClose(
        math_ops.matmul(
            operator1.to_dense(),
            operator2.to_dense()),
        self.evaluate(operator_matmul.to_dense()))

    operator_matmul = operator2.matmul(operator1)
    self.assertTrue(isinstance(
        operator_matmul,
        linalg_lib.LinearOperatorLowerTriangular))
    self.assertAllClose(
        math_ops.matmul(
            operator2.to_dense(),
            operator1.to_dense()),
        self.evaluate(operator_matmul.to_dense()))

  def test_tape_safe(self):
    tril = variables_module.Variable([[1., 0.], [0., 1.]])
    operator = linalg_lib.LinearOperatorLowerTriangular(
        tril, is_non_singular=True)
    self.check_tape_safe(operator)


if __name__ == "__main__":
  linear_operator_test_util.add_tests(LinearOperatorLowerTriangularTest)
  test.main()
