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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import math_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_linalg_ops import *
# pylint: enable=wildcard-import

# Names below are lower_case.
# pylint: disable=invalid-name


def cholesky_solve(chol, rhs, name=None):
  """Solves systems of linear eqns `A X = RHS`, given Cholesky factorizations.

  ```python
  # Solve 10 separate 2x2 linear systems:
  A = ... # shape 10 x 2 x 2
  RHS = ... # shape 10 x 2 x 1
  chol = tf.cholesky(A)  # shape 10 x 2 x 2
  X = tf.cholesky_solve(chol, RHS)  # shape 10 x 2 x 1
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
      Cholesky factorization of `A`, e.g. `chol = tf.cholesky(A)`.
      For that reason, only the lower triangular parts (including the diagonal)
      of the last two dimensions of `chol` are used.  The strictly upper part is
      assumed to be zero and not accessed.
    rhs:  A `Tensor`, same type as `chol`, shape is `[..., M, K]`.
    name:  A name to give this `Op`.  Defaults to `cholesky_solve`.

  Returns:
    Solution to `A x = rhs`, shape `[..., M, K]`.
  """
  # To solve C C^* x = rhs, we
  # 1. Solve C y = rhs for y, thus y = C^* x
  # 2. Solve C^* x = y for x
  with ops.name_scope(name, "cholesky_solve", [chol, rhs]):
    y = gen_linalg_ops.matrix_triangular_solve(
        chol, rhs, adjoint=False, lower=True)
    x = gen_linalg_ops.matrix_triangular_solve(
        chol, y, adjoint=True, lower=True)
    return x


def eye(
    num_rows,
    num_columns=None,
    batch_shape=None,
    dtype=dtypes.float32,
    name=None):
  """Construct an identity matrix, or a batch of matrices.

  ```python
  # Construct one identity matrix.
  tf.eye(2)
  ==> [[1., 0.],
       [0., 1.]]

  # Construct a batch of 3 identity matricies, each 2 x 2.
  # batch_identity[i, :, :] is a 2 x 2 identity matrix, i = 0, 1, 2.
  batch_identity = tf.eye(2, batch_shape=[3])

  # Construct one 2 x 3 "identity" matrix
  tf.eye(2, num_columns=3)
  ==> [[ 1.,  0.,  0.],
       [ 0.,  1.,  0.]]
  ```

  Args:
    num_rows: Non-negative `int32` scalar `Tensor` giving the number of rows
      in each batch matrix.
    num_columns: Optional non-negative `int32` scalar `Tensor` giving the number
      of columns in each batch matrix.  Defaults to `num_rows`.
    batch_shape:  `int32` `Tensor`.  If provided, returned `Tensor` will have
      leading batch dimensions of this shape.
    dtype:  The type of an element in the resulting `Tensor`
    name:  A name for this `Op`.  Defaults to "eye".

  Returns:
    A `Tensor` of shape `batch_shape + [num_rows, num_columns]`
  """
  with ops.name_scope(
      name, default_name="eye", values=[num_rows, num_columns, batch_shape]):

    batch_shape = [] if batch_shape is None else batch_shape
    batch_shape = ops.convert_to_tensor(
        batch_shape, name="shape", dtype=dtypes.int32)

    if num_columns is None:
      diag_size = num_rows
    else:
      diag_size = math_ops.minimum(num_rows, num_columns)
    diag_shape = array_ops.concat(0, (batch_shape, [diag_size]))
    diag_ones = array_ops.ones(diag_shape, dtype=dtype)

    if num_columns is None:
      return array_ops.matrix_diag(diag_ones)
    else:
      shape = array_ops.concat(0, (batch_shape, [num_rows, num_columns]))
      zero_matrix = array_ops.zeros(shape, dtype=dtype)
      return array_ops.matrix_set_diag(zero_matrix, diag_ones)


def matrix_solve_ls(matrix, rhs, l2_regularizer=0.0, fast=True, name=None):
  r"""Solves one or more linear least-squares problems.

  `matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
  form `M`-by-`N` matrices. Rhs is a tensor of shape `[..., M, K]` whose
  inner-most 2 dimensions form `M`-by-`K` matrices.   The computed output is a
  `Tensor` of shape `[..., N, K]` whose inner-most 2 dimensions form `M`-by-`K`
  matrices that solve the equations
  `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]` in the least squares
  sense.

  Below we will use the following notation for each pair of matrix and
  right-hand sides in the batch:

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
  # pylint: disable=protected-access
  return gen_linalg_ops._matrix_solve_ls(
      matrix, rhs, l2_regularizer, fast=fast, name=name)


def self_adjoint_eig(tensor, name=None):
  """Computes the eigen decomposition of a batch of self-adjoint matrices.

  Computes the eigenvalues and eigenvectors of the innermost N-by-N matrices
  in `tensor` such that
  `tensor[...,:,:] * v[..., :,i] = e[..., i] * v[...,:,i]`, for i=0...N-1.

  Args:
    tensor: `Tensor` of shape `[..., N, N]`. Only the lower triangular part of
      each inner inner matrix is referenced.
    name: string, optional name of the operation.

  Returns:
    e: Eigenvalues. Shape is `[..., N]`.
    v: Eigenvectors. Shape is `[..., N, N]`. The columns of the inner most
      matrices contain eigenvectors of the corresponding matrices in `tensor`
  """
  # pylint: disable=protected-access
  e, v = gen_linalg_ops._self_adjoint_eig_v2(tensor, compute_v=True, name=name)
  return e, v


def self_adjoint_eigvals(tensor, name=None):
  """Computes the eigenvalues of one or more self-adjoint matrices.

  Args:
    tensor: `Tensor` of shape `[..., N, N]`.
    name: string, optional name of the operation.

  Returns:
    e: Eigenvalues. Shape is `[..., N]`. The vector `e[..., :]` contains the `N`
      eigenvalues of `tensor[..., :, :]`.
  """
  # pylint: disable=protected-access
  e, _ = gen_linalg_ops._self_adjoint_eig_v2(tensor, compute_v=False, name=name)
  return e


def svd(tensor, full_matrices=False, compute_uv=True, name=None):
  """Computes the singular value decompositions of one or more matrices.

  Computes the SVD of each inner matrix in `tensor` such that
  `tensor[..., :, :] = u[..., :, :] * diag(s[..., :, :]) * transpose(v[..., :,
  :])`

  ```prettyprint
  # a is a tensor.
  # s is a tensor of singular values.
  # u is a tensor of left singular vectors.
  # v is a tensor of right singular vectors.
  s, u, v = svd(a)
  s = svd(a, compute_uv=False)
  ```

  Args:
    matrix: `Tensor` of shape `[..., M, N]`. Let `P` be the minimum of `M` and
      `N`.
    full_matrices: If true, compute full-sized `u` and `v`. If false
      (the default), compute only the leading `P` singular vectors.
      Ignored if `compute_uv` is `False`.
    compute_uv: If `True` then left and right singular vectors will be
      computed and returned in `u` and `v`, respectively. Otherwise, only the
      singular values will be computed, which can be significantly faster.
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
  # pylint: disable=protected-access
  s, u, v = gen_linalg_ops._svd(
      tensor, compute_uv=compute_uv, full_matrices=full_matrices)
  if compute_uv:
    return math_ops.real(s), u, v
  else:
    return math_ops.real(s)

# pylint: enable=invalid-name
