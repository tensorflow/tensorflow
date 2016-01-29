# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Operations for linear algebra."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_linalg_ops
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_linalg_ops import *
# pylint: enable=wildcard-import


@ops.RegisterShape("Cholesky")
def _CholeskyShape(op):
  input_shape = op.inputs[0].get_shape().with_rank(2)
  # The matrix must be square.
  input_shape[0].assert_is_compatible_with(input_shape[1])
  return [input_shape]


@ops.RegisterShape("BatchCholesky")
def _BatchCholeskyShape(op):
  input_shape = op.inputs[0].get_shape().with_rank_at_least(3)
  # The matrices in the batch must be square.
  input_shape[-1].assert_is_compatible_with(input_shape[-2])
  return [input_shape]


@ops.RegisterShape("MatrixDeterminant")
def _MatrixDeterminantShape(op):
  input_shape = op.inputs[0].get_shape().with_rank(2)
  # The matrix must be square.
  input_shape[0].assert_is_compatible_with(input_shape[1])
  if input_shape.ndims is not None:
    return [tensor_shape.scalar()]
  else:
    return [tensor_shape.unknown_shape()]


@ops.RegisterShape("BatchMatrixDeterminant")
def _BatchMatrixDeterminantShape(op):
  input_shape = op.inputs[0].get_shape().with_rank_at_least(3)
  # The matrices in the batch must be square.
  input_shape[-1].assert_is_compatible_with(input_shape[-2])
  if input_shape.ndims is not None:
    return [input_shape[:-2]]
  else:
    return [tensor_shape.unknown_shape()]


@ops.RegisterShape("MatrixInverse")
def _MatrixInverseShape(op):
  input_shape = op.inputs[0].get_shape().with_rank(2)
  # The matrix must be square.
  input_shape[0].assert_is_compatible_with(input_shape[1])
  return [input_shape]


@ops.RegisterShape("BatchMatrixInverse")
def _BatchMatrixInverseShape(op):
  input_shape = op.inputs[0].get_shape().with_rank_at_least(3)
  # The matrices in the batch must be square.
  input_shape[-1].assert_is_compatible_with(input_shape[-2])
  return [input_shape]


@ops.RegisterShape("SelfAdjointEig")
def _SelfAdjointEigShape(op):
  input_shape = op.inputs[0].get_shape().with_rank(2)
  # The matrix must be square.
  input_shape[0].assert_is_compatible_with(input_shape[1])
  d = input_shape.dims[0]
  out_shape = tensor_shape.TensorShape([d+1, d])
  return [out_shape]


@ops.RegisterShape("BatchSelfAdjointEig")
def _BatchSelfAdjointEigShape(op):
  input_shape = op.inputs[0].get_shape().with_rank_at_least(3)
  # The matrices in the batch must be square.
  input_shape[-1].assert_is_compatible_with(input_shape[-2])
  dlist = input_shape.dims
  dlist[-2] += 1
  out_shape = tensor_shape.TensorShape(dlist)
  return [out_shape]


@ops.RegisterShape("MatrixSolve")
def _MatrixSolveShape(op):
  lhs_shape = op.inputs[0].get_shape().with_rank(2)
  rhs_shape = op.inputs[1].get_shape().with_rank_at_least(2)
  # The matrix must be square.
  lhs_shape[0].assert_is_compatible_with(lhs_shape[1])
  # The matrix and right-hand side must have the same number of rows.
  lhs_shape[0].assert_is_compatible_with(rhs_shape[0])
  return [[lhs_shape[1], rhs_shape[1]]]


@ops.RegisterShape("BatchMatrixSolve")
def _BatchMatrixSolveShape(op):
  lhs_shape = op.inputs[0].get_shape().with_rank_at_least(3)
  rhs_shape = op.inputs[1].get_shape().with_rank_at_least(3)
  # The matrices must be square.
  lhs_shape[-1].assert_is_compatible_with(lhs_shape[-2])
  # The matrices and right-hand sides in the batch must have the same number of
  # rows.
  lhs_shape[-2].assert_is_compatible_with(rhs_shape[-2])
  return [lhs_shape[:-2].concatenate(rhs_shape[-1])]


@ops.RegisterShape("MatrixTriangularSolve")
def _MatrixTriangularSolveShape(op):
  lhs_shape = op.inputs[0].get_shape().with_rank(2)
  rhs_shape = op.inputs[1].get_shape().with_rank_at_least(2)
  # The matrix must be square.
  lhs_shape[0].assert_is_compatible_with(lhs_shape[1])
  # The matrix and righ-hand side must have the same number of rows.
  lhs_shape[0].assert_is_compatible_with(rhs_shape[0])
  return [rhs_shape]


@ops.RegisterShape("BatchMatrixTriangularSolve")
def _BatchMatrixTriangularSolveShape(op):
  lhs_shape = op.inputs[0].get_shape().with_rank_at_least(3)
  rhs_shape = op.inputs[1].get_shape().with_rank_at_least(3)
  # The matrices must be square.
  lhs_shape[-1].assert_is_compatible_with(lhs_shape[-2])
  # The matrices and righ-hand sides in the batch must have the same number of
  # rows.
  lhs_shape[-2].assert_is_compatible_with(rhs_shape[-2])
  return [rhs_shape]


@ops.RegisterShape("MatrixSolveLs")
def _MatrixSolveLsShape(op):
  lhs_shape = op.inputs[0].get_shape().with_rank(2)
  rhs_shape = op.inputs[1].get_shape().with_rank_at_least(2)
  # The matrix and right-hand side must have the same number of rows.
  lhs_shape[0].assert_is_compatible_with(rhs_shape[0])
  return [[lhs_shape[1], rhs_shape[1]]]


@ops.RegisterShape("BatchMatrixSolveLs")
def _BatchMatrixSolveLsShape(op):
  lhs_shape = op.inputs[0].get_shape().with_rank_at_least(3)
  rhs_shape = op.inputs[1].get_shape().with_rank_at_least(3)
  # The matrices and right-hand sides in the batch must have the same number of
  # rows.
  lhs_shape[-2].assert_is_compatible_with(rhs_shape[-2])
  return [lhs_shape[:-3].concatenate([lhs_shape[-1], rhs_shape[-1]])]


# pylint: disable=invalid-name
def matrix_solve_ls(matrix, rhs, l2_regularizer=0.0, fast=True, name=None):
  r"""Solves a linear least-squares problem.

  Below we will use the following notation
  `matrix`=\\(A \in \Re^{m \times n}\\),
  `rhs`=\\(B  \in \Re^{m \times k}\\),
  `output`=\\(X  \in \Re^{n \times k}\\),
  `l2_regularizer`=\\(\lambda\\).

  If `fast` is `True`, then the solution is computed by solving the normal
  equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
  \\(X = (A^T A + \lambda I)^{-1} A^T B\\), which solves the regularized
  least-squares problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}}
  ||A Z - B||_F^2 + \lambda ||Z||_F^2\\). If \\(m \lt n\\) then `output` is
  computed as \\(X = A^T (A A^T + \lambda I)^{-1} B\\),
  which (for \\(\lambda = 0\\)) is the minimum-norm solution to the
  under-determined linear system, i.e.
  \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||Z||_F^2 \\),
  subject to \\(A Z = B\\).
  Notice that the fast path is only numerically stable when \\(A\\) is
  numerically full rank and has a condition number
  \\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach}}}\\)
  or \\(\lambda\\) is sufficiently large.

  If `fast` is `False` then the solution is computed using the rank revealing
  QR decomposition with column pivoting. This will always compute a
  least-squares solution that minimizes the residual norm
  \\(||A X - B||_F^2 \\), even when \\(A\\) is rank deficient or
  ill-conditioned. Notice: The current version does not compute a minimum norm
  solution. If `fast` is `False` then `l2_regularizer` is ignored.

  Args:
    matrix: 2-D `Tensor` of shape `[M, N]`.
    rhs: 2-D `Tensor` of shape is `[M, K]`.
    l2_regularizer: 0-D  `double` `Tensor`. Ignored if `fast=False`.
    fast: bool. Defaults to `True`.
    name: string, optional name of the operation.

  Returns:
    output: Matrix of shape `[N, K]` containing the matrix that solves
      `matrix * output = rhs` in the least-squares sense.
  """
  return gen_linalg_ops.matrix_solve_ls(matrix,
                                        rhs,
                                        l2_regularizer,
                                        fast=fast,
                                        name=name)


def batch_matrix_solve_ls(matrix,
                          rhs,
                          l2_regularizer=0.0,
                          fast=True,
                          name=None):
  r"""Solves multiple linear least-squares problems.

  `matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
  form `M`-by-`N` matrices. Rhs is a tensor of shape `[..., M, K]` whose
  inner-most 2 dimensions form `M`-by-`K` matrices.   The computed output is a
  `Tensor` of shape `[..., N, K]` whose inner-most 2 dimensions form `M`-by-`K`
  matrices that solve the equations
  `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]` in the least squares
  sense.

  Below we will use the following notation for each pair of
  matrix and right-hand sides in the batch:

  `matrix`=\\(A \in \Re^{m \times n}\\),
  `rhs`=\\(B  \in \Re^{m \times k}\\),
  `output`=\\(X  \in \Re^{n \times k}\\),
  `l2_regularizer`=\\(\lambda\\).

  If `fast` is `True`, then the solution is computed by solving the normal
  equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
  \\(X = (A^T A + \lambda I)^{-1} A^T B\\), which solves the least-squares
  problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||A Z - B||_F^2 +
  \lambda ||Z||_F^2\\). If \\(m \lt n\\) then `output` is computed as
  \\(X = A^T (A A^T + \lambda I)^{-1} B\\), which (for \\(\lambda = 0\\)) is
  the minimum-norm solution to the under-determined linear system, i.e.
  \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||Z||_F^2 \\), subject to
  \\(A Z = B\\). Notice that the fast path is only numerically stable when
  \\(A\\) is numerically full rank and has a condition number
  \\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach}}}\\) or\\(\lambda\\)
  is sufficiently large.

  If `fast` is `False` then the solution is computed using the rank revealing
  QR decomposition with column pivoting. This will always compute a
  least-squares solution that minimizes the residual norm \\(||A X - B||_F^2\\),
  even when \\(A\\) is rank deficient or ill-conditioned. Notice: The current
  version does not compute a minimum norm solution. If `fast` is `False` then
  `l2_regularizer` is ignored.

  Args:
    matrix: `Tensor` of shape `[..., M, N]`.
    rhs: `Tensor` of shape `[..., M, K]`.
    l2_regularizer: 0-D `double` `Tensor`. Ignored if `fast=False`.
    fast: bool. Defaults to `True`.
    name: string, optional name of the operation.

  Returns:
    output: `Tensor` of shape `[..., N, K]` whose inner-most 2 dimensions form
      `M`-by-`K` matrices that solve the equations
      `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]` in the least
      squares sense.
  """
  return gen_linalg_ops.batch_matrix_solve_ls(matrix,
                                              rhs,
                                              l2_regularizer,
                                              fast=fast,
                                              name=name)


# pylint: enable=invalid-name
