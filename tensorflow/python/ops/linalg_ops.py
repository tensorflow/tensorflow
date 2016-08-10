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


def _UnchangedSquareHelper(input_shape):
  """Helper for {Batch}UnchangedSquare."""
  # The matrices in the batch must be square.
  input_shape[-1].assert_is_compatible_with(input_shape[-2])
  return [input_shape]


@ops.RegisterShape("Cholesky")
@ops.RegisterShape("CholeskyGrad")
@ops.RegisterShape("MatrixInverse")
def _UnchangedSquare(op):
  """Shape function for matrix ops with output equal to input shape."""
  return _UnchangedSquareHelper(op.inputs[0].get_shape().with_rank(2))


@ops.RegisterShape("BatchCholesky")
@ops.RegisterShape("BatchCholeskyGrad")
@ops.RegisterShape("BatchMatrixInverse")
def _BatchUnchangedSquare(op):
  """Shape function for batch matrix ops with output equal to input shape."""
  return _UnchangedSquareHelper(op.inputs[0].get_shape().with_rank_at_least(2))


@ops.RegisterShape("MatrixDeterminant")
def _MatrixDeterminantShape(op):
  """Shape function for determinant op."""
  input_shape = op.inputs[0].get_shape().with_rank(2)
  # The matrix must be square.
  input_shape[0].assert_is_compatible_with(input_shape[1])
  if input_shape.ndims is not None:
    return [tensor_shape.scalar()]
  else:
    return [tensor_shape.unknown_shape()]


@ops.RegisterShape("BatchMatrixDeterminant")
def _BatchMatrixDeterminantShape(op):
  """Shape function for batch determinant op."""
  input_shape = op.inputs[0].get_shape().with_rank_at_least(2)
  # The matrices in the batch must be square.
  input_shape[-1].assert_is_compatible_with(input_shape[-2])
  if input_shape.ndims is not None:
    return [input_shape[:-2]]
  else:
    return [tensor_shape.unknown_shape()]


@ops.RegisterShape("SelfAdjointEig")
def _SelfAdjointEigShape(op):
  """Shape function for self-adjoint eigensolver op."""
  input_shape = op.inputs[0].get_shape().with_rank(2)
  # The matrix must be square.
  input_shape[0].assert_is_compatible_with(input_shape[1])
  d = input_shape.dims[0]
  out_shape = tensor_shape.TensorShape([d + 1, d])
  return [out_shape]


@ops.RegisterShape("BatchSelfAdjointEig")
def _BatchSelfAdjointEigShape(op):
  """Shape function for batch self-adjoint eigensolver op."""
  input_shape = op.inputs[0].get_shape().with_rank_at_least(2)
  # The matrices in the batch must be square.
  input_shape[-1].assert_is_compatible_with(input_shape[-2])
  dlist = input_shape.dims
  dlist[-2] += 1
  out_shape = tensor_shape.TensorShape(dlist)
  return [out_shape]


def _SelfAdjointEigV2ShapeHelper(op, input_shape):
  """Shape inference helper for {Batch}SelfAdjointEigV2."""
  batch_shape = input_shape[:-2]
  n = input_shape[-1].merge_with(input_shape[-2])
  compute_v = op.get_attr("compute_v")
  if compute_v:
    return [batch_shape.concatenate([n]), batch_shape.concatenate([n, n])]
  else:
    return [batch_shape.concatenate([n]), [0]]


@ops.RegisterShape("SelfAdjointEigV2")
def _SelfAdjointEigShapeV2(op):
  """Shape function for SelfAdjointEigV2."""
  return _SelfAdjointEigV2ShapeHelper(op, op.inputs[0].get_shape().with_rank(2))


@ops.RegisterShape("BatchSelfAdjointEigV2")
def _BatchSelfAdjointEigV2Shape(op):
  """Shape function for BatchSelfAdjointEigV2."""
  return _SelfAdjointEigV2ShapeHelper(
      op, op.inputs[0].get_shape().with_rank_at_least(2))


def _SvdShapeHelper(input_shape, op):
  """Shape inference helper for {Batch}SVD op."""
  unknown = tensor_shape.unknown_shape()
  if input_shape.ndims is not None:
    return [unknown, unknown, unknown]
  compute_uv = op.get_attr("compute_uv")
  full_matrices = op.get_attr("full_matrices")
  m = input_shape[-2]
  n = input_shape[-1]
  p = min(m, n)
  batch_shape = input_shape[:-2]
  s_shape = batch_shape.concatenate([p])
  if compute_uv:
    if full_matrices:
      u_shape = batch_shape.concatenate([m, m])
      v_shape = batch_shape.concatenate([n, n])
    else:
      u_shape = batch_shape.concatenate([m, p])
      v_shape = batch_shape.concatenate([n, p])
  else:
    u_shape = [0]
    v_shape = [0]
  return [s_shape, u_shape, v_shape]


@ops.RegisterShape("Svd")
def _SvdShape(op):
  """Shape function for SVD op."""
  return _SvdShapeHelper(op.inputs[0].get_shape().with_rank(2), op)


@ops.RegisterShape("BatchSvd")
def _BatchSvdShape(op):
  """Shape function for batch SVD op."""
  return _SvdShapeHelper(op.inputs[0].get_shape().with_rank_at_least(2), op)


def _SquareMatrixSolveShapeHelper(lhs_shape, rhs_shape):
  """Shape inference helper function for square matrix solver ops."""
  # The matrix must be square.
  lhs_shape[-1].assert_is_compatible_with(lhs_shape[-2])
  # The matrix and right-hand side must have the same number of rows.
  lhs_shape[-2].assert_is_compatible_with(rhs_shape[-2])
  return [rhs_shape]


@ops.RegisterShape("MatrixSolve")
@ops.RegisterShape("MatrixTriangularSolve")
def _SquareMatrixSolveShape(op):
  """Shape function for square matrix solver ops."""
  return _SquareMatrixSolveShapeHelper(op.inputs[0].get_shape().with_rank(2),
                                       op.inputs[1].get_shape().with_rank(2))


@ops.RegisterShape("BatchMatrixSolve")
@ops.RegisterShape("BatchMatrixTriangularSolve")
def _BatchSquareMatrixSolveShape(op):
  """Shape function for batch square matrix solver ops."""
  return _SquareMatrixSolveShapeHelper(
      op.inputs[0].get_shape().with_rank_at_least(2),
      op.inputs[1].get_shape().with_rank_at_least(2))


def _MatrixSolveLsShapeHelper(lhs_shape, rhs_shape):
  """Shape inference helper function for least squares matrix solver ops."""
  # The matrices and right-hand sides must have the same number of rows.
  lhs_shape[-2].assert_is_compatible_with(rhs_shape[-2])
  return [lhs_shape[:-2].concatenate([lhs_shape[-1], rhs_shape[-1]])]


@ops.RegisterShape("MatrixSolveLs")
def _MatrixSolveLsShape(op):
  """Shape function for least-squares matrix solver op."""
  return _MatrixSolveLsShapeHelper(op.inputs[0].get_shape().with_rank(2),
                                   op.inputs[1].get_shape().with_rank(2))


@ops.RegisterShape("BatchMatrixSolveLs")
def _BatchMatrixSolveLsShape(op):
  """Shape function for batch least-squares matrix solver op."""
  return _MatrixSolveLsShapeHelper(
      op.inputs[0].get_shape().with_rank_at_least(2),
      op.inputs[1].get_shape().with_rank_at_least(2))


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


def self_adjoint_eig(matrix, name=None):
  """Computes the eigen decomposition of a self-adjoint matrix.

  Computes the eigenvalues and eigenvectors of an N-by-N matrix `matrix` such
  that `matrix * v[:,i] = e(i) * v[:,i]`, for i=0...N-1.

  Args:
    matrix: `Tensor` of shape `[N, N]`.
    name: string, optional name of the operation.

  Returns:
    e: Eigenvalues. Shape is `[N]`.
    v: Eigenvectors. Shape is `[N, N]`. The columns contain the eigenvectors of
      `matrix`.
  """
  e, v = gen_linalg_ops.self_adjoint_eig_v2(matrix, compute_v=True, name=name)
  return e, v


def batch_self_adjoint_eig(tensor, name=None):
  """Computes the eigen decomposition of a batch of self-adjoint matrices.

  Computes the eigenvalues and eigenvectors of the innermost N-by-N matrices
  in `tensor` such that
  `tensor[...,:,:] * v[..., :,i] = e(..., i) * v[...,:,i]`, for i=0...N-1.

  Args:
    tensor: `Tensor` of shape `[..., N, N]`.
    name: string, optional name of the operation.

  Returns:
    e: Eigenvalues. Shape is `[..., N]`.
    v: Eigenvectors. Shape is `[..., N, N]`. The columns of the inner most
    matrices
      contain eigenvectors of the corresponding matrices in `tensor`
  """
  e, v = gen_linalg_ops.batch_self_adjoint_eig_v2(
      tensor, compute_v=True, name=name)
  return e, v


def self_adjoint_eigvals(matrix, name=None):
  """Computes the eigenvalues a self-adjoint  matrix.

  Args:
    matrix: `Tensor` of shape `[N, N]`.
    name: string, optional name of the operation.

  Returns:
    e: Eigenvalues of `matrix`. Shape is `[N]`.
  """
  e, _ = gen_linalg_ops.self_adjoint_eig_v2(matrix, compute_v=False, name=name)
  return e


def batch_self_adjoint_eigvals(tensor, name=None):
  """Computes the eigenvalues of a batch of self-adjoint matrices.

  Args:
    tensor: `Tensor` of shape `[..., N, N]`.
    name: string, optional name of the operation.

  Returns:
    e: Eigenvalues. Shape is `[..., N]`. The vector `e[..., :]` contains the `N`
      eigenvalues of `tensor[..., :, :]`.
  """
  e, _ = gen_linalg_ops.batch_self_adjoint_eig_v2(
      tensor, compute_v=False, name=name)
  return e


def svd(matrix, compute_uv=True, full_matrices=False, name=None):
  """Computes the singular value decomposition of a matrix.

  Computes the SVD of `matrix` such that `matrix = u * diag(s) *
  transpose(v)`

  ```prettyprint
  # a is a matrix.
  # s is a vector of singular values.
  # u is the matrix of left singular vectors.
  # v is a matrix of right singular vectors.
  s, u, v = svd(a)
  s = svd(a, compute_uv=False)
  ```

  Args:
    matrix: `Tensor` of shape `[M, N]`. Let `P` be the minimum of `M` and `N`.
    compute_uv: If `True` then left and right singular vectors will be
      computed and returned in `u` and `v`, respectively. Otherwise, only the
      singular values will be computed, which can be significantly faster.
    full_matrices: If true, compute full-sized `u` and `v`. If false
      (the default), compute only the leading `P` singular vectors.
      Ignored if `compute_uv` is `False`.
    name: string, optional name of the operation.

  Returns:
    s: Singular values. Shape is `[P]`.
    u: Right singular vectors. If `full_matrices` is `False` (default) then
      shape is `[M, P]`; if `full_matrices` is `True` then shape is
      `[M, M]`. Not returned if `compute_uv` is `False`.
    v: Left singular vectors. If `full_matrices` is `False` (default) then
      shape is `[N, P]`. If `full_matrices` is `True` then shape is
      `[N, N]`. Not returned if `compute_uv` is `False`.
  """
  s, u, v = gen_linalg_ops.svd(matrix,
                               compute_uv=compute_uv,
                               full_matrices=full_matrices)
  if compute_uv:
    return s, u, v
  else:
    return s


def batch_svd(tensor, compute_uv=True, full_matrices=False, name=None):
  """Computes the singular value decompositions of a batch of matrices.

  Computes the SVD of each inner matrix in `tensor` such that
  `tensor[..., :, :] = u[..., :, :] * diag(s[..., :, :]) * transpose(v[..., :,
  :])`

  ```prettyprint
  # a is a tensor.
  # s is a tensor of singular values.
  # u is a tensor of left singular vectors.
  # v is a tensor of right singular vectors.
  s, u, v = batch_svd(a)
  s = batch_svd(a, compute_uv=False)
  ```

  Args:
    matrix: `Tensor` of shape `[..., M, N]`. Let `P` be the minimum of `M` and
      `N`.
    compute_uv: If `True` then left and right singular vectors will be
      computed and returned in `u` and `v`, respectively. Otherwise, only the
      singular values will be computed, which can be significantly faster.
    full_matrices: If true, compute full-sized `u` and `v`. If false
      (the default), compute only the leading `P` singular vectors.
      Ignored if `compute_uv` is `False`.
    name: string, optional name of the operation.

  Returns:
    s: Singular values. Shape is `[..., P]`.
    u: Right singular vectors. If `full_matrices` is `False` (default) then
      shape is `[..., M, P]`; if `full_matrices` is `True` then shape is
      `[..., M, M]`. Not returned if `compute_uv` is `False`.
    v: Left singular vectors. If `full_matrices` is `False` (default) then
      shape is `[..., N, P]`. If `full_matrices` is `True` then shape is
      `[..., N, N]`. Not returned if `compute_uv` is `False`.
  """
  s, u, v = gen_linalg_ops.batch_svd(
      tensor, compute_uv=compute_uv, full_matrices=full_matrices)
  if compute_uv:
    return s, u, v
  else:
    return s


# pylint: enable=invalid-name
