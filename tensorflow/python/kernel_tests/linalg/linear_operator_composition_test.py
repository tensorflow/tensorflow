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

import numpy as np

from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

linalg = linalg_lib
rng = np.random.RandomState(0)


class SquareLinearOperatorCompositionTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)
    # Increase from 1e-6 to 1e-4 and 2e-4.
    self._atol[dtypes.float32] = 2e-4
    self._atol[dtypes.complex64] = 1e-4
    self._rtol[dtypes.float32] = 2e-4
    self._rtol[dtypes.complex64] = 1e-4

  @staticmethod
  def skip_these_tests():
    # Cholesky not implemented.
    return ["cholesky"]

  def operator_and_matrix(self, build_info, dtype, use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    shape = list(build_info.shape)

    # Either 1 or 2 matrices, depending.
    num_operators = rng.randint(low=1, high=3)
    if ensure_self_adjoint_and_pd:
      # The random PD matrices are also symmetric. Here we are computing
      # A @ A ... @ A. Since A is symmetric and PD, so are any powers of it.
      matrices = [
          linear_operator_test_util.random_positive_definite_matrix(
              shape, dtype, force_well_conditioned=True)] * num_operators
    else:
      matrices = [
          linear_operator_test_util.random_positive_definite_matrix(
              shape, dtype, force_well_conditioned=True)
          for _ in range(num_operators)
      ]

    lin_op_matrices = matrices

    if use_placeholder:
      lin_op_matrices = [
          array_ops.placeholder_with_default(
              matrix, shape=None) for matrix in matrices]

    operator = linalg.LinearOperatorComposition(
        [linalg.LinearOperatorFullMatrix(l) for l in lin_op_matrices],
        is_positive_definite=True if ensure_self_adjoint_and_pd else None,
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
        is_square=True)

    matmul_order_list = list(reversed(matrices))
    mat = matmul_order_list[0]
    for other_mat in matmul_order_list[1:]:
      mat = math_ops.matmul(other_mat, mat)

    return operator, mat

  def test_is_x_flags(self):
    # Matrix with two positive eigenvalues, 1, and 1.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorComposition(
        [linalg.LinearOperatorFullMatrix(matrix)],
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertFalse(operator.is_self_adjoint)

  def test_is_non_singular_auto_set(self):
    # Matrix with two positive eigenvalues, 11 and 8.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[11., 0.], [1., 8.]]
    operator_1 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)
    operator_2 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)

    operator = linalg.LinearOperatorComposition(
        [operator_1, operator_2],
        is_positive_definite=False,  # No reason it HAS to be False...
        is_non_singular=None)
    self.assertFalse(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)

    with self.assertRaisesRegex(ValueError, "always non-singular"):
      linalg.LinearOperatorComposition(
          [operator_1, operator_2], is_non_singular=False)

  def test_name(self):
    matrix = [[11., 0.], [1., 8.]]
    operator_1 = linalg.LinearOperatorFullMatrix(matrix, name="left")
    operator_2 = linalg.LinearOperatorFullMatrix(matrix, name="right")

    operator = linalg.LinearOperatorComposition([operator_1, operator_2])

    self.assertEqual("left_o_right", operator.name)

  def test_different_dtypes_raises(self):
    operators = [
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3)),
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3).astype(np.float32))
    ]
    with self.assertRaisesRegex(TypeError, "same dtype"):
      linalg.LinearOperatorComposition(operators)

  def test_empty_operators_raises(self):
    with self.assertRaisesRegex(ValueError, "non-empty"):
      linalg.LinearOperatorComposition([])


class NonSquareLinearOperatorCompositionTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)
    # Increase from 1e-6 to 1e-4
    self._atol[dtypes.float32] = 1e-4
    self._atol[dtypes.complex64] = 1e-4
    self._rtol[dtypes.float32] = 1e-4
    self._rtol[dtypes.complex64] = 1e-4

  @staticmethod
  def skip_these_tests():
    # Testing the condition number fails when using XLA with cuBLASLt
    # A slight numerical difference between different matmul algorithms
    # leads to large precision issues
    return linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest.skip_these_tests(
    ) + ["cond"]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    del ensure_self_adjoint_and_pd
    shape = list(build_info.shape)

    # Create 2 matrices/operators, A1, A2, which becomes A = A1 A2.
    # Use inner dimension of 2.
    k = 2
    batch_shape = shape[:-2]
    shape_1 = batch_shape + [shape[-2], k]
    shape_2 = batch_shape + [k, shape[-1]]

    # Ensure that the matrices are well-conditioned by generating
    # random matrices whose singular values are close to 1.
    # The reason to do this is because cond(AB) <= cond(A) * cond(B).
    # By ensuring that each factor has condition number close to 1, we ensure
    # that the condition number of the product isn't too far away from 1.
    def generate_well_conditioned(shape, dtype):
      m, n = shape[-2], shape[-1]
      min_dim = min(m, n)
      # Generate singular values that are close to 1.
      d = linear_operator_test_util.random_normal(
          shape[:-2] + [min_dim],
          mean=1.,
          stddev=0.1,
          dtype=dtype)
      zeros = array_ops.zeros(shape=shape[:-2] + [m, n], dtype=dtype)
      d = linalg_lib.set_diag(zeros, d)
      u, _ = linalg_lib.qr(linear_operator_test_util.random_normal(
          shape[:-2] + [m, m], dtype=dtype))

      v, _ = linalg_lib.qr(linear_operator_test_util.random_normal(
          shape[:-2] + [n, n], dtype=dtype))
      return math_ops.matmul(u, math_ops.matmul(d, v))

    matrices = [
        generate_well_conditioned(shape_1, dtype=dtype),
        generate_well_conditioned(shape_2, dtype=dtype),
    ]

    lin_op_matrices = matrices

    if use_placeholder:
      lin_op_matrices = [
          array_ops.placeholder_with_default(
              matrix, shape=None) for matrix in matrices]

    operator = linalg.LinearOperatorComposition(
        [linalg.LinearOperatorFullMatrix(l) for l in lin_op_matrices])

    matmul_order_list = list(reversed(matrices))
    mat = matmul_order_list[0]
    for other_mat in matmul_order_list[1:]:
      mat = math_ops.matmul(other_mat, mat)

    return operator, mat

  @test_util.run_deprecated_v1
  def test_static_shapes(self):
    operators = [
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 4)),
        linalg.LinearOperatorFullMatrix(rng.rand(2, 4, 5))
    ]
    operator = linalg.LinearOperatorComposition(operators)
    self.assertAllEqual((2, 3, 5), operator.shape)

  @test_util.run_deprecated_v1
  def test_shape_tensors_when_statically_available(self):
    operators = [
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 4)),
        linalg.LinearOperatorFullMatrix(rng.rand(2, 4, 5))
    ]
    operator = linalg.LinearOperatorComposition(operators)
    with self.cached_session():
      self.assertAllEqual((2, 3, 5), operator.shape_tensor())

  @test_util.run_deprecated_v1
  def test_shape_tensors_when_only_dynamically_available(self):
    mat_1 = rng.rand(1, 2, 3, 4)
    mat_2 = rng.rand(1, 2, 4, 5)
    mat_ph_1 = array_ops.placeholder(dtypes.float64)
    mat_ph_2 = array_ops.placeholder(dtypes.float64)
    feed_dict = {mat_ph_1: mat_1, mat_ph_2: mat_2}

    operators = [
        linalg.LinearOperatorFullMatrix(mat_ph_1),
        linalg.LinearOperatorFullMatrix(mat_ph_2)
    ]
    operator = linalg.LinearOperatorComposition(operators)
    with self.cached_session():
      self.assertAllEqual(
          (1, 2, 3, 5), operator.shape_tensor().eval(feed_dict=feed_dict))


if __name__ == "__main__":
  linear_operator_test_util.add_tests(SquareLinearOperatorCompositionTest)
  linear_operator_test_util.add_tests(NonSquareLinearOperatorCompositionTest)
  test.main()
