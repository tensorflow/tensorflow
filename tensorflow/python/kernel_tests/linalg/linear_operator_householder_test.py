# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_householder as householder
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

linalg = linalg_lib
CheckTapeSafeSkipOptions = linear_operator_test_util.CheckTapeSafeSkipOptions


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorHouseholderTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @staticmethod
  def operator_shapes_infos():
    shape_info = linear_operator_test_util.OperatorShapesInfo
    return [
        shape_info((1, 1)),
        shape_info((1, 3, 3)),
        shape_info((3, 4, 4)),
        shape_info((2, 1, 4, 4))]

  @staticmethod
  def skip_these_tests():
    # This linear operator is never positive definite.
    return ["cholesky"]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    shape = list(build_info.shape)
    reflection_axis = linear_operator_test_util.random_sign_uniform(
        shape[:-1], minval=1., maxval=2., dtype=dtype)
    # Make sure unit norm.
    reflection_axis = reflection_axis / linalg_ops.norm(
        reflection_axis, axis=-1, keepdims=True)

    lin_op_reflection_axis = reflection_axis

    if use_placeholder:
      lin_op_reflection_axis = array_ops.placeholder_with_default(
          reflection_axis, shape=None)

    operator = householder.LinearOperatorHouseholder(lin_op_reflection_axis)

    mat = reflection_axis[..., array_ops.newaxis]
    matrix = -2 * math_ops.matmul(mat, mat, adjoint_b=True)
    matrix = array_ops.matrix_set_diag(
        matrix, 1. + array_ops.matrix_diag_part(matrix))

    return operator, matrix

  def test_scalar_reflection_axis_raises(self):
    with self.assertRaisesRegex(ValueError, "must have at least 1 dimension"):
      householder.LinearOperatorHouseholder(1.)

  def test_householder_adjoint_type(self):
    reflection_axis = [1., 3., 5., 8.]
    operator = householder.LinearOperatorHouseholder(reflection_axis)
    self.assertIsInstance(
        operator.adjoint(), householder.LinearOperatorHouseholder)

  def test_householder_inverse_type(self):
    reflection_axis = [1., 3., 5., 8.]
    operator = householder.LinearOperatorHouseholder(reflection_axis)
    self.assertIsInstance(
        operator.inverse(), householder.LinearOperatorHouseholder)

  def test_tape_safe(self):
    reflection_axis = variables_module.Variable([1., 3., 5., 8.])
    operator = householder.LinearOperatorHouseholder(reflection_axis)
    self.check_tape_safe(
        operator,
        skip_options=[
            # Determinant hard-coded as 1.
            CheckTapeSafeSkipOptions.DETERMINANT,
            CheckTapeSafeSkipOptions.LOG_ABS_DETERMINANT,
            # Trace hard-coded.
            CheckTapeSafeSkipOptions.TRACE,
        ])


if __name__ == "__main__":
  linear_operator_test_util.add_tests(LinearOperatorHouseholderTest)
  test.main()
