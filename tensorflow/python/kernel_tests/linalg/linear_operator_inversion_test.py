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

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_inversion
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

linalg = linalg_lib

LinearOperatorInversion = linear_operator_inversion.LinearOperatorInversion  # pylint: disable=invalid-name


class LinearOperatorInversionTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def setUp(self):
    self._atol[dtypes.complex64] = 1e-5
    self._rtol[dtypes.complex64] = 1e-5

  def _operator_and_matrix(self,
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
      operator = LinearOperatorInversion(
          linalg.LinearOperatorFullMatrix(
              lin_op_matrix, is_positive_definite=True, is_self_adjoint=True))
    else:
      operator = LinearOperatorInversion(
          linalg.LinearOperatorLowerTriangular(lin_op_matrix))

    return operator, linalg.inv(matrix)

  def test_base_operator_hint_used(self):
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix,
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    operator_inv = LinearOperatorInversion(operator)
    self.assertTrue(operator_inv.is_positive_definite)
    self.assertTrue(operator_inv.is_non_singular)
    self.assertFalse(operator_inv.is_self_adjoint)

  def test_supplied_hint_used(self):
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(matrix)
    operator_inv = LinearOperatorInversion(
        operator,
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator_inv.is_positive_definite)
    self.assertTrue(operator_inv.is_non_singular)
    self.assertFalse(operator_inv.is_self_adjoint)

  def test_contradicting_hints_raise(self):
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix, is_positive_definite=False)
    with self.assertRaisesRegexp(ValueError, "positive-definite"):
      LinearOperatorInversion(operator, is_positive_definite=True)

    operator = linalg.LinearOperatorFullMatrix(matrix, is_self_adjoint=False)
    with self.assertRaisesRegexp(ValueError, "self-adjoint"):
      LinearOperatorInversion(operator, is_self_adjoint=True)

  def test_singular_raises(self):
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 1.], [1., 1.]]

    operator = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=False)
    with self.assertRaisesRegexp(ValueError, "is_non_singular"):
      LinearOperatorInversion(operator)

    operator = linalg.LinearOperatorFullMatrix(matrix)
    with self.assertRaisesRegexp(ValueError, "is_non_singular"):
      LinearOperatorInversion(operator, is_non_singular=False)

  def test_name(self):
    matrix = [[11., 0.], [1., 8.]]
    operator = linalg.LinearOperatorFullMatrix(
        matrix, name="my_operator", is_non_singular=True)

    operator = LinearOperatorInversion(operator)

    self.assertEqual("my_operator_inv", operator.name)


if __name__ == "__main__":
  test.main()
