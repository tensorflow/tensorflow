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

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_block_diag as block_diag
from tensorflow.python.ops.linalg import linear_operator_lower_triangular as lower_triangular
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test

linalg = linalg_lib
rng = np.random.RandomState(0)


def _block_diag_dense(expected_shape, blocks):
  """Convert a list of blocks, into a dense block diagonal matrix."""
  rows = []
  num_cols = 0
  for block in blocks:
    # Get the batch shape for the block.
    batch_row_shape = array_ops.shape(block)[:-1]

    zeros_to_pad_before_shape = array_ops.concat(
        [batch_row_shape, [num_cols]], axis=-1)
    zeros_to_pad_before = array_ops.zeros(
        shape=zeros_to_pad_before_shape, dtype=block.dtype)
    num_cols += array_ops.shape(block)[-1]
    zeros_to_pad_after_shape = array_ops.concat(
        [batch_row_shape, [expected_shape[-1] - num_cols]], axis=-1)
    zeros_to_pad_after = array_ops.zeros(
        zeros_to_pad_after_shape, dtype=block.dtype)

    rows.append(array_ops.concat(
        [zeros_to_pad_before, block, zeros_to_pad_after], axis=-1))

  return array_ops.concat(rows, axis=-2)


@test_util.run_all_in_graph_and_eager_modes
class SquareLinearOperatorBlockDiagTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def setUp(self):
    # Increase from 1e-6 to 1e-4
    self._atol[dtypes.float32] = 1e-4
    self._atol[dtypes.complex64] = 1e-4
    self._rtol[dtypes.float32] = 1e-4
    self._rtol[dtypes.complex64] = 1e-4

  @staticmethod
  def operator_shapes_infos():
    shape_info = linear_operator_test_util.OperatorShapesInfo
    return [
        shape_info((0, 0)),
        shape_info((1, 1)),
        shape_info((1, 3, 3)),
        shape_info((5, 5), blocks=[(2, 2), (3, 3)]),
        shape_info((3, 7, 7), blocks=[(1, 2, 2), (3, 2, 2), (1, 3, 3)]),
        shape_info((2, 1, 5, 5), blocks=[(2, 1, 2, 2), (1, 3, 3)]),
    ]

  @staticmethod
  def use_blockwise_arg():
    return True

  def operator_and_matrix(
      self, shape_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    shape = list(shape_info.shape)
    expected_blocks = (
        shape_info.__dict__["blocks"] if "blocks" in shape_info.__dict__
        else [shape])
    matrices = [
        linear_operator_test_util.random_positive_definite_matrix(
            block_shape, dtype, force_well_conditioned=True)
        for block_shape in expected_blocks
    ]

    lin_op_matrices = matrices

    if use_placeholder:
      lin_op_matrices = [
          array_ops.placeholder_with_default(
              matrix, shape=None) for matrix in matrices]

    operator = block_diag.LinearOperatorBlockDiag(
        [linalg.LinearOperatorFullMatrix(
            l,
            is_square=True,
            is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
            is_positive_definite=True if ensure_self_adjoint_and_pd else None)
         for l in lin_op_matrices])

    # Should be auto-set.
    self.assertTrue(operator.is_square)

    # Broadcast the shapes.
    expected_shape = list(shape_info.shape)

    matrices = linear_operator_util.broadcast_matrix_batch_dims(matrices)

    block_diag_dense = _block_diag_dense(expected_shape, matrices)

    if not use_placeholder:
      block_diag_dense.set_shape(
          expected_shape[:-2] + [expected_shape[-1], expected_shape[-1]])

    return operator, block_diag_dense

  def test_is_x_flags(self):
    # Matrix with two positive eigenvalues, 1, and 1.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = block_diag.LinearOperatorBlockDiag(
        [linalg.LinearOperatorFullMatrix(matrix)],
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertFalse(operator.is_self_adjoint)

  def test_is_x_parameters(self):
    matrix = [[1., 0.], [1., 1.]]
    sub_operator = linalg.LinearOperatorFullMatrix(matrix)
    operator = block_diag.LinearOperatorBlockDiag(
        [sub_operator],
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertEqual(
        operator.parameters,
        {
            "name": None,
            "is_square": True,
            "is_positive_definite": True,
            "is_self_adjoint": False,
            "is_non_singular": True,
            "operators": [sub_operator],
        })
    self.assertEqual(
        sub_operator.parameters,
        {
            "is_non_singular": None,
            "is_positive_definite": None,
            "is_self_adjoint": None,
            "is_square": None,
            "matrix": matrix,
            "name": "LinearOperatorFullMatrix",
        })

  def test_block_diag_adjoint_type(self):
    matrix = [[1., 0.], [0., 1.]]
    operator = block_diag.LinearOperatorBlockDiag(
        [
            linalg.LinearOperatorFullMatrix(
                matrix,
                is_non_singular=True,
            ),
            linalg.LinearOperatorFullMatrix(
                matrix,
                is_non_singular=True,
            ),
        ],
        is_non_singular=True,
    )
    adjoint = operator.adjoint()
    self.assertIsInstance(
        adjoint,
        block_diag.LinearOperatorBlockDiag)
    self.assertEqual(2, len(adjoint.operators))

  def test_block_diag_cholesky_type(self):
    matrix = [[1., 0.], [0., 1.]]
    operator = block_diag.LinearOperatorBlockDiag(
        [
            linalg.LinearOperatorFullMatrix(
                matrix,
                is_positive_definite=True,
                is_self_adjoint=True,
            ),
            linalg.LinearOperatorFullMatrix(
                matrix,
                is_positive_definite=True,
                is_self_adjoint=True,
            ),
        ],
        is_positive_definite=True,
        is_self_adjoint=True,
    )
    cholesky_factor = operator.cholesky()
    self.assertIsInstance(
        cholesky_factor,
        block_diag.LinearOperatorBlockDiag)
    self.assertEqual(2, len(cholesky_factor.operators))
    self.assertIsInstance(
        cholesky_factor.operators[0],
        lower_triangular.LinearOperatorLowerTriangular)
    self.assertIsInstance(
        cholesky_factor.operators[1],
        lower_triangular.LinearOperatorLowerTriangular
    )

  def test_block_diag_inverse_type(self):
    matrix = [[1., 0.], [0., 1.]]
    operator = block_diag.LinearOperatorBlockDiag(
        [
            linalg.LinearOperatorFullMatrix(
                matrix,
                is_non_singular=True,
            ),
            linalg.LinearOperatorFullMatrix(
                matrix,
                is_non_singular=True,
            ),
        ],
        is_non_singular=True,
    )
    inverse = operator.inverse()
    self.assertIsInstance(
        inverse,
        block_diag.LinearOperatorBlockDiag)
    self.assertEqual(2, len(inverse.operators))

  def test_block_diag_matmul_type(self):
    matrices1 = []
    matrices2 = []
    for i in range(1, 5):
      matrices1.append(linalg.LinearOperatorFullMatrix(
          linear_operator_test_util.random_normal(
              [2, i], dtype=dtypes.float32)))

      matrices2.append(linalg.LinearOperatorFullMatrix(
          linear_operator_test_util.random_normal(
              [i, 3], dtype=dtypes.float32)))

    operator1 = block_diag.LinearOperatorBlockDiag(matrices1, is_square=False)
    operator2 = block_diag.LinearOperatorBlockDiag(matrices2, is_square=False)

    expected_matrix = math_ops.matmul(
        operator1.to_dense(), operator2.to_dense())
    actual_operator = operator1.matmul(operator2)

    self.assertIsInstance(
        actual_operator, block_diag.LinearOperatorBlockDiag)
    actual_, expected_ = self.evaluate([
        actual_operator.to_dense(), expected_matrix])
    self.assertAllClose(actual_, expected_)

  def test_block_diag_matmul_raises(self):
    matrices1 = []
    for i in range(1, 5):
      matrices1.append(linalg.LinearOperatorFullMatrix(
          linear_operator_test_util.random_normal(
              [2, i], dtype=dtypes.float32)))
    operator1 = block_diag.LinearOperatorBlockDiag(matrices1, is_square=False)
    operator2 = linalg.LinearOperatorFullMatrix(
        linear_operator_test_util.random_normal(
            [15, 3], dtype=dtypes.float32))

    with self.assertRaisesRegex(ValueError, "Operators are incompatible"):
      operator1.matmul(operator2)

  def test_block_diag_solve_type(self):
    matrices1 = []
    matrices2 = []
    for i in range(1, 5):
      matrices1.append(linalg.LinearOperatorFullMatrix(
          linear_operator_test_util.random_tril_matrix(
              [i, i],
              dtype=dtypes.float32,
              force_well_conditioned=True)))

      matrices2.append(linalg.LinearOperatorFullMatrix(
          linear_operator_test_util.random_normal(
              [i, 3], dtype=dtypes.float32)))

    operator1 = block_diag.LinearOperatorBlockDiag(matrices1)
    operator2 = block_diag.LinearOperatorBlockDiag(matrices2, is_square=False)

    expected_matrix = linalg.solve(
        operator1.to_dense(), operator2.to_dense())
    actual_operator = operator1.solve(operator2)

    self.assertIsInstance(
        actual_operator, block_diag.LinearOperatorBlockDiag)
    actual_, expected_ = self.evaluate([
        actual_operator.to_dense(), expected_matrix])
    self.assertAllClose(actual_, expected_)

  def test_block_diag_solve_raises(self):
    matrices1 = []
    for i in range(1, 5):
      matrices1.append(linalg.LinearOperatorFullMatrix(
          linear_operator_test_util.random_normal(
              [i, i], dtype=dtypes.float32)))
    operator1 = block_diag.LinearOperatorBlockDiag(matrices1)
    operator2 = linalg.LinearOperatorFullMatrix(
        linear_operator_test_util.random_normal(
            [15, 3], dtype=dtypes.float32))

    with self.assertRaisesRegex(ValueError, "Operators are incompatible"):
      operator1.solve(operator2)

  def test_tape_safe(self):
    matrices = []
    for _ in range(4):
      matrices.append(variables_module.Variable(
          linear_operator_test_util.random_positive_definite_matrix(
              [2, 2], dtype=dtypes.float32, force_well_conditioned=True)))

    operator = block_diag.LinearOperatorBlockDiag(
        [linalg.LinearOperatorFullMatrix(
            matrix, is_self_adjoint=True,
            is_positive_definite=True) for matrix in matrices],
        is_self_adjoint=True,
        is_positive_definite=True,
    )
    self.check_tape_safe(operator)

  def test_is_non_singular_auto_set(self):
    # Matrix with two positive eigenvalues, 11 and 8.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[11., 0.], [1., 8.]]
    operator_1 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)
    operator_2 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)

    operator = block_diag.LinearOperatorBlockDiag(
        [operator_1, operator_2],
        is_positive_definite=False,  # No reason it HAS to be False...
        is_non_singular=None)
    self.assertFalse(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)

    with self.assertRaisesRegex(ValueError, "always non-singular"):
      block_diag.LinearOperatorBlockDiag(
          [operator_1, operator_2], is_non_singular=False)

  def test_name(self):
    matrix = [[11., 0.], [1., 8.]]
    operator_1 = linalg.LinearOperatorFullMatrix(matrix, name="left")
    operator_2 = linalg.LinearOperatorFullMatrix(matrix, name="right")

    operator = block_diag.LinearOperatorBlockDiag([operator_1, operator_2])

    self.assertEqual("left_ds_right", operator.name)

  def test_different_dtypes_raises(self):
    operators = [
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3)),
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3).astype(np.float32))
    ]
    with self.assertRaisesRegex(TypeError, "same dtype"):
      block_diag.LinearOperatorBlockDiag(operators)

  def test_empty_operators_raises(self):
    with self.assertRaisesRegex(ValueError, "non-empty"):
      block_diag.LinearOperatorBlockDiag([])

  def test_incompatible_input_blocks_raises(self):
    matrix_1 = array_ops.placeholder_with_default(rng.rand(4, 4), shape=None)
    matrix_2 = array_ops.placeholder_with_default(rng.rand(3, 3), shape=None)
    operators = [
        linalg.LinearOperatorFullMatrix(matrix_1, is_square=True),
        linalg.LinearOperatorFullMatrix(matrix_2, is_square=True)
    ]
    operator = block_diag.LinearOperatorBlockDiag(operators)
    x = np.random.rand(2, 4, 5).tolist()
    msg = ("dimension does not match" if context.executing_eagerly()
           else "input structure is ambiguous")
    with self.assertRaisesRegex(ValueError, msg):
      operator.matmul(x)


@test_util.run_all_in_graph_and_eager_modes
class NonSquareLinearOperatorBlockDiagTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def setUp(self):
    # Increase from 1e-6 to 1e-4
    self._atol[dtypes.float32] = 1e-4
    self._atol[dtypes.complex64] = 1e-4
    self._rtol[dtypes.float32] = 1e-4
    self._rtol[dtypes.complex64] = 1e-4
    super(NonSquareLinearOperatorBlockDiagTest, self).setUp()

  @staticmethod
  def operator_shapes_infos():
    shape_info = linear_operator_test_util.OperatorShapesInfo
    return [
        shape_info((1, 0)),
        shape_info((1, 2, 3)),
        shape_info((5, 3), blocks=[(2, 1), (3, 2)]),
        shape_info((3, 6, 5), blocks=[(1, 2, 1), (3, 1, 2), (1, 3, 2)]),
        shape_info((2, 1, 5, 2), blocks=[(2, 1, 2, 1), (1, 3, 1)]),
    ]

  @staticmethod
  def skip_these_tests():
    return [
        "cholesky",
        "cond",
        "det",
        "diag_part",
        "eigvalsh",
        "inverse",
        "log_abs_det",
        "solve",
        "solve_with_broadcast",
        "trace"]

  @staticmethod
  def use_blockwise_arg():
    return True

  def operator_and_matrix(
      self, shape_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    del ensure_self_adjoint_and_pd
    shape = list(shape_info.shape)
    expected_blocks = (
        shape_info.__dict__["blocks"] if "blocks" in shape_info.__dict__
        else [shape])
    matrices = [
        linear_operator_test_util.random_normal(block_shape, dtype=dtype)
        for block_shape in expected_blocks
    ]

    lin_op_matrices = matrices

    if use_placeholder:
      lin_op_matrices = [
          array_ops.placeholder_with_default(
              matrix, shape=None) for matrix in matrices]

    blocks = []
    for l in lin_op_matrices:
      blocks.append(
          linalg.LinearOperatorFullMatrix(
              l,
              is_square=False,
              is_self_adjoint=False,
              is_positive_definite=False))
    operator = block_diag.LinearOperatorBlockDiag(blocks)

    # Broadcast the shapes.
    expected_shape = list(shape_info.shape)

    matrices = linear_operator_util.broadcast_matrix_batch_dims(matrices)

    block_diag_dense = _block_diag_dense(expected_shape, matrices)

    if not use_placeholder:
      block_diag_dense.set_shape(expected_shape)

    return operator, block_diag_dense


if __name__ == "__main__":
  linear_operator_test_util.add_tests(SquareLinearOperatorBlockDiagTest)
  linear_operator_test_util.add_tests(NonSquareLinearOperatorBlockDiagTest)
  test.main()
