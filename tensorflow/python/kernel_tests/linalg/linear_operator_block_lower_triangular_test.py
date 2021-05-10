# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_block_lower_triangular as block_lower_triangular
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test

linalg = linalg_lib
rng = np.random.RandomState(0)


def _block_lower_triangular_dense(expected_shape, blocks):
  """Convert a list of blocks into a dense blockwise lower-triangular matrix."""
  rows = []
  num_cols = 0
  for row_blocks in blocks:

    # Get the batch shape for the block.
    batch_row_shape = array_ops.shape(row_blocks[0])[:-1]

    num_cols += array_ops.shape(row_blocks[-1])[-1]
    zeros_to_pad_after_shape = array_ops.concat(
        [batch_row_shape, [expected_shape[-2] - num_cols]], axis=-1)
    zeros_to_pad_after = array_ops.zeros(
        zeros_to_pad_after_shape, dtype=row_blocks[-1].dtype)

    row_blocks.append(zeros_to_pad_after)
    rows.append(array_ops.concat(row_blocks, axis=-1))

  return array_ops.concat(rows, axis=-2)


@test_util.run_all_in_graph_and_eager_modes
class SquareLinearOperatorBlockLowerTriangularTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def setUp(self):
    # Increase from 1e-6 to 1e-5
    self._atol[dtypes.float32] = 1e-5
    self._atol[dtypes.complex64] = 1e-5
    self._rtol[dtypes.float32] = 1e-5
    self._rtol[dtypes.complex64] = 1e-5
    super(SquareLinearOperatorBlockLowerTriangularTest, self).setUp()

  @staticmethod
  def use_blockwise_arg():
    return True

  @staticmethod
  def skip_these_tests():
    # Skipping since `LinearOperatorBlockLowerTriangular` is in general not
    # self-adjoint.
    return ["cholesky", "eigvalsh"]

  @staticmethod
  def operator_shapes_infos():
    shape_info = linear_operator_test_util.OperatorShapesInfo
    return [
        shape_info((0, 0)),
        shape_info((1, 1)),
        shape_info((1, 3, 3)),
        shape_info((5, 5), blocks=[[(2, 2)], [(3, 2), (3, 3)]]),
        shape_info((3, 7, 7),
                   blocks=[[(1, 2, 2)], [(1, 3, 2), (3, 3, 3)],
                           [(1, 2, 2), (1, 2, 3), (1, 2, 2)]]),
        shape_info((2, 4, 6, 6),
                   blocks=[[(2, 1, 2, 2)], [(1, 4, 2), (4, 4, 4)]]),
    ]

  def operator_and_matrix(
      self, shape_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):

    expected_blocks = (
        shape_info.__dict__["blocks"] if "blocks" in shape_info.__dict__
        else [[list(shape_info.shape)]])

    matrices = []
    for i, row_shapes in enumerate(expected_blocks):
      row = []
      for j, block_shape in enumerate(row_shapes):
        if i == j:  # operator is on the diagonal
          row.append(
              linear_operator_test_util.random_positive_definite_matrix(
                  block_shape, dtype, force_well_conditioned=True))
        else:
          row.append(
              linear_operator_test_util.random_normal(block_shape, dtype=dtype))
      matrices.append(row)

    lin_op_matrices = matrices

    if use_placeholder:
      lin_op_matrices = [[
          array_ops.placeholder_with_default(
              matrix, shape=None) for matrix in row] for row in matrices]

    operator = block_lower_triangular.LinearOperatorBlockLowerTriangular(
        [[linalg.LinearOperatorFullMatrix(  # pylint:disable=g-complex-comprehension
            l,
            is_square=True,
            is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
            is_positive_definite=True if ensure_self_adjoint_and_pd else None)
          for l in row] for row in lin_op_matrices])

    # Should be auto-set.
    self.assertTrue(operator.is_square)

    # Broadcast the shapes.
    expected_shape = list(shape_info.shape)
    broadcasted_matrices = linear_operator_util.broadcast_matrix_batch_dims(
        [op for row in matrices for op in row])  # pylint: disable=g-complex-comprehension
    matrices = [broadcasted_matrices[i * (i + 1) // 2:(i + 1) * (i + 2) // 2]
                for i in range(len(matrices))]

    block_lower_triangular_dense = _block_lower_triangular_dense(
        expected_shape, matrices)

    if not use_placeholder:
      block_lower_triangular_dense.set_shape(expected_shape)

    return operator, block_lower_triangular_dense

  def test_is_x_flags(self):
    # Matrix with two positive eigenvalues, 1, and 1.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = block_lower_triangular.LinearOperatorBlockLowerTriangular(
        [[linalg.LinearOperatorFullMatrix(matrix)]],
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertFalse(operator.is_self_adjoint)

  def test_block_lower_triangular_inverse_type(self):
    matrix = [[1., 0.], [0., 1.]]
    operator = block_lower_triangular.LinearOperatorBlockLowerTriangular(
        [[linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)],
         [linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True),
          linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)]],
        is_non_singular=True,
    )
    inverse = operator.inverse()
    self.assertIsInstance(
        inverse,
        block_lower_triangular.LinearOperatorBlockLowerTriangular)
    self.assertEqual(2, len(inverse.operators))
    self.assertEqual(1, len(inverse.operators[0]))
    self.assertEqual(2, len(inverse.operators[1]))

  def test_tape_safe(self):
    operator_1 = linalg.LinearOperatorFullMatrix(
        variables_module.Variable([[1., 0.], [0., 1.]]),
        is_self_adjoint=True,
        is_positive_definite=True)
    operator_2 = linalg.LinearOperatorFullMatrix(
        variables_module.Variable([[2., 0.], [1., 0.]]))
    operator_3 = linalg.LinearOperatorFullMatrix(
        variables_module.Variable([[3., 1.], [1., 3.]]),
        is_self_adjoint=True,
        is_positive_definite=True)
    operator = block_lower_triangular.LinearOperatorBlockLowerTriangular(
        [[operator_1], [operator_2, operator_3]],
        is_self_adjoint=False,
        is_positive_definite=True)

    diagonal_grads_only = ["diag_part", "trace", "determinant",
                           "log_abs_determinant"]
    self.check_tape_safe(operator, skip_options=diagonal_grads_only)

    for y in diagonal_grads_only:
      for diag_block in [operator_1, operator_3]:
        with backprop.GradientTape() as tape:
          grads = tape.gradient(getattr(operator, y)(), diag_block.variables)
          for item in grads:
            self.assertIsNotNone(item)

  def test_is_non_singular_auto_set(self):
    # Matrix with two positive eigenvalues, 11 and 8.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[11., 0.], [1., 8.]]
    operator_1 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)
    operator_2 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)
    operator_3 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)

    operator = block_lower_triangular.LinearOperatorBlockLowerTriangular(
        [[operator_1], [operator_2, operator_3]],
        is_positive_definite=False,  # No reason it HAS to be False...
        is_non_singular=None)
    self.assertFalse(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)

    with self.assertRaisesRegex(ValueError, "always non-singular"):
      block_lower_triangular.LinearOperatorBlockLowerTriangular(
          [[operator_1], [operator_2, operator_3]], is_non_singular=False)

    operator_4 = linalg.LinearOperatorFullMatrix(
        [[1., 0.], [2., 0.]], is_non_singular=False)

    # A singular operator off of the main diagonal shouldn't raise
    block_lower_triangular.LinearOperatorBlockLowerTriangular(
        [[operator_1], [operator_4, operator_2]], is_non_singular=True)

    with self.assertRaisesRegex(ValueError, "always singular"):
      block_lower_triangular.LinearOperatorBlockLowerTriangular(
          [[operator_1], [operator_2, operator_4]], is_non_singular=True)

  def test_different_dtypes_raises(self):
    operators = [
        [linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3))],
        [linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3)),
         linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3).astype(np.float32))]
    ]
    with self.assertRaisesRegex(TypeError, "same dtype"):
      block_lower_triangular.LinearOperatorBlockLowerTriangular(operators)

  def test_non_square_operator_raises(self):
    operators = [
        [linalg.LinearOperatorFullMatrix(rng.rand(3, 4), is_square=False)],
        [linalg.LinearOperatorFullMatrix(rng.rand(4, 4)),
         linalg.LinearOperatorFullMatrix(rng.rand(4, 4))]
    ]
    with self.assertRaisesRegex(ValueError, "must be square"):
      block_lower_triangular.LinearOperatorBlockLowerTriangular(operators)

  def test_empty_operators_raises(self):
    with self.assertRaisesRegex(ValueError, "non-empty"):
      block_lower_triangular.LinearOperatorBlockLowerTriangular([])

  def test_operators_wrong_length_raises(self):
    with self.assertRaisesRegex(ValueError, "must contain `i` blocks"):
      block_lower_triangular.LinearOperatorBlockLowerTriangular([
          [linalg.LinearOperatorFullMatrix(rng.rand(2, 2))],
          [linalg.LinearOperatorFullMatrix(rng.rand(2, 2))
           for _ in range(3)]])

  def test_operators_mismatched_dimension_raises(self):
    operators = [
        [linalg.LinearOperatorFullMatrix(rng.rand(3, 3))],
        [linalg.LinearOperatorFullMatrix(rng.rand(3, 4)),
         linalg.LinearOperatorFullMatrix(rng.rand(3, 3))]
    ]
    with self.assertRaisesRegex(ValueError, "must be equal"):
      block_lower_triangular.LinearOperatorBlockLowerTriangular(operators)

  def test_incompatible_input_blocks_raises(self):
    matrix_1 = array_ops.placeholder_with_default(rng.rand(4, 4), shape=None)
    matrix_2 = array_ops.placeholder_with_default(rng.rand(3, 4), shape=None)
    matrix_3 = array_ops.placeholder_with_default(rng.rand(3, 3), shape=None)
    operators = [
        [linalg.LinearOperatorFullMatrix(matrix_1, is_square=True)],
        [linalg.LinearOperatorFullMatrix(matrix_2),
         linalg.LinearOperatorFullMatrix(matrix_3, is_square=True)]
    ]
    operator = block_lower_triangular.LinearOperatorBlockLowerTriangular(
        operators)
    x = np.random.rand(2, 4, 5).tolist()
    msg = ("dimension does not match" if context.executing_eagerly()
           else "input structure is ambiguous")
    with self.assertRaisesRegex(ValueError, msg):
      operator.matmul(x)


if __name__ == "__main__":
  linear_operator_test_util.add_tests(
      SquareLinearOperatorBlockLowerTriangularTest)
  test.main()
