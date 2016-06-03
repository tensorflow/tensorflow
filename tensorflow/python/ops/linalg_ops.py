# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_linalg_ops import *
# pylint: enable=wildcard-import


@ops.RegisterShape("Cholesky")
@ops.RegisterShape("CholeskyGrad")
@ops.RegisterShape("MatrixInverse")
def _UnchangedSquare(op):
  input_shape = op.inputs[0].get_shape().with_rank(2)
  # The matrix must be square.
  input_shape[0].assert_is_compatible_with(input_shape[1])
  return [input_shape]


@ops.RegisterShape("BatchCholesky")
@ops.RegisterShape("BatchCholeskyGrad")
@ops.RegisterShape("BatchMatrixInverse")
def _BatchUnchangedSquare(op):
  input_shape = op.inputs[0].get_shape().with_rank_at_least(2)
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
  input_shape = op.inputs[0].get_shape().with_rank_at_least(2)
  # The matrices in the batch must be square.
  input_shape[-1].assert_is_compatible_with(input_shape[-2])
  if input_shape.ndims is not None:
    return [input_shape[:-2]]
  else:
    return [tensor_shape.unknown_shape()]


@ops.RegisterShape("SelfAdjointEig")
def _SelfAdjointEigShape(op):
  input_shape = op.inputs[0].get_shape().with_rank(2)
  # The matrix must be square.
  input_shape[0].assert_is_compatible_with(input_shape[1])
  d = input_shape.dims[0]
  out_shape = tensor_shape.TensorShape([d + 1, d])
  return [out_shape]


@ops.RegisterShape("BatchSelfAdjointEig")
def _BatchSelfAdjointEigShape(op):
  input_shape = op.inputs[0].get_shape().with_rank_at_least(2)
  # The matrices in the batch must be square.
  input_shape[-1].assert_is_compatible_with(input_shape[-2])
  dlist = input_shape.dims
  dlist[-2] += 1
  out_shape = tensor_shape.TensorShape(dlist)
  return [out_shape]


@ops.RegisterShape("MatrixSolve")
@ops.RegisterShape("MatrixTriangularSolve")
def _SquareMatrixSolveShape(op):
  lhs_shape = op.inputs[0].get_shape().with_rank(2)
  rhs_shape = op.inputs[1].get_shape().with_rank_at_least(2)
  # The matrix must be square.
  lhs_shape[0].assert_is_compatible_with(lhs_shape[1])
  # The matrix and right-hand side must have the same number of rows.
  lhs_shape[0].assert_is_compatible_with(rhs_shape[0])
  return [rhs_shape]


@ops.RegisterShape("BatchMatrixSolve")
@ops.RegisterShape("BatchMatrixTriangularSolve")
def _BatchSquareMatrixSolveShape(op):
  lhs_shape = op.inputs[0].get_shape().with_rank_at_least(2)
  rhs_shape = op.inputs[1].get_shape().with_rank_at_least(2)
  # The matrices must be square.
  lhs_shape[-1].assert_is_compatible_with(lhs_shape[-2])
  # The matrices and right-hand sides in the batch must have the same number of
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
  lhs_shape = op.inputs[0].get_shape().with_rank_at_least(2)
  rhs_shape = op.inputs[1].get_shape().with_rank_at_least(2)
  # The matrices and right-hand sides in the batch must have the same number of
  # rows.
  lhs_shape[-2].assert_is_compatible_with(rhs_shape[-2])
  return [lhs_shape[:-2].concatenate([lhs_shape[-1], rhs_shape[-1]])]


# Names below are lower_case.
# pylint: disable=invalid-name


def cholesky_solve(chol, rhs, name=None):
  """Solve linear equations `A X = RHS`, given Cholesky factorization of `A`.

  ```python
  # Solve one system of linear equations (K = 1).
  A = [[3, 1], [1, 3]]
  RHS = [[2], [22]]  # shape 2 x 1
  chol = tf.cholesky(A)
  X = tf.cholesky_solve(chol, RHS)
  # tf.matmul(A, X) ~ RHS
  X[:, 0]  # Solution to the linear system A x = RHS[:, 0]

  # Solve five systems of linear equations (K = 5).
  A = [[3, 1], [1, 3]]
  RHS = [[1, 2, 3, 4, 5], [11, 22, 33, 44, 55]]  # shape 2 x 5
  ...
  X[:, 2]  # Solution to the linear system A x = RHS[:, 2]
  ```

  Args:
    chol:  A `Tensor`.  Must be `float32` or `float64`, shape is `[M, M]`.
      Cholesky factorization of `A`, e.g. `chol = tf.cholesky(A)`.  For that
      reason, only the lower triangular part (including the diagonal) of `chol`
      is used.  The strictly upper part is assumed to be zero and not accessed.
    rhs:  A `Tensor`, same type as `chol`, shape is `[M, K]`, designating `K`
      systems of linear equations.
    name:  A name to give this `Op`.  Defaults to `cholesky_solve`.

  Returns:
    Solution to `A X = RHS`, shape `[M, K]`.  The solutions to the `K` systems.
  """
  # To solve C C^* x = rhs, we
  # 1. Solve C y = rhs for y, thus y = C^* x
  # 2. Solve C^* x = y for x
  with ops.op_scope([chol, rhs], name, "cholesky_solve"):
    y = gen_linalg_ops.matrix_triangular_solve(
        chol, rhs, adjoint=False, lower=True)
    x = gen_linalg_ops.matrix_triangular_solve(
        chol, y, adjoint=True, lower=True)
    return x


def batch_cholesky_solve(chol, rhs, name=None):
  """Solve batches of linear eqns `A X = RHS`, given Cholesky factorizations.

  ```python
  # Solve one linear system (K = 1) for every member of the length 10 batch.
  A = ... # shape 10 x 2 x 2
  RHS = ... # shape 10 x 2 x 1
  chol = tf.batch_cholesky(A)  # shape 10 x 2 x 2
  X = tf.batch_cholesky_solve(chol, RHS)  # shape 10 x 2 x 1
  # tf.matmul(A, X) ~ RHS
  X[3, :, 0]  # Solution to the linear system A[3, :, :] x = RHS[3, :, 0]

  # Solve five linear systems (K = 5) for every member of the length 10 batch.
  A = ... # shape 10 x 2 x 2
  RHS = ... # shape 10 x 2 x 5
  ...
  X[3, :, 2]  # Solution to the linear system A[3, :, :] x = RHS[3, :, 2]
  ```

  Args:
    chol:  A `Tensor`.  Must be `float32` or `float64`, shape is `[..., M, M]`.
      Cholesky factorization of `A`, e.g. `chol = tf.batch_cholesky(A)`.
      For that reason, only the lower triangular parts (including the diagonal)
      of the last two dimensions of `chol` are used.  The strictly upper part is
      assumed to be zero and not accessed.
    rhs:  A `Tensor`, same type as `chol`, shape is `[..., M, K]`.
    name:  A name to give this `Op`.  Defaults to `batch_cholesky_solve`.

  Returns:
    Solution to `A x = rhs`, shape `[..., M, K]`.
  """
  # To solve C C^* x = rhs, we
  # 1. Solve C y = rhs for y, thus y = C^* x
  # 2. Solve C^* x = y for x
  with ops.op_scope([chol, rhs], name, "batch_cholesky_solve"):
    y = gen_linalg_ops.batch_matrix_triangular_solve(
        chol, rhs, adjoint=False, lower=True)
    x = gen_linalg_ops.batch_matrix_triangular_solve(
        chol, y, adjoint=True, lower=True)
    return x


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

  If `fast` is `False` an algorithm based on the numerically robust complete
  orthogonal decomposition is used. This computes the minimum-norm
  least-squares solution, even when \\(A\\) is rank deficient. This path is
  typically 6-7 times slower than the fast path. If `fast` is `False` then
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
