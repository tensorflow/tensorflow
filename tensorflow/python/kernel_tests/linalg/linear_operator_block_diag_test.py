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
        [batch_row_shape, [expected_shape[-2] - num_cols]], axis=-1)
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

    with self.assertRaisesRegexp(ValueError, "always non-singular"):
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
    with self.assertRaisesRegexp(TypeError, "same dtype"):
      block_diag.LinearOperatorBlockDiag(operators)

  def test_non_square_operator_raises(self):
    operators = [
        linalg.LinearOperatorFullMatrix(rng.rand(3, 4), is_square=False),
        linalg.LinearOperatorFullMatrix(rng.rand(3, 3))
    ]
    with self.assertRaisesRegexp(ValueError, "square matrices"):
      block_diag.LinearOperatorBlockDiag(operators)

  def test_empty_operators_raises(self):
    with self.assertRaisesRegexp(ValueError, "non-empty"):
      block_diag.LinearOperatorBlockDiag([])


if __name__ == "__main__":
  linear_operator_test_util.add_tests(SquareLinearOperatorBlockDiagTest)
  test.main()
