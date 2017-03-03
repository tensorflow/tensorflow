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

import numpy as np

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
  with ops.name_scope(name, 'cholesky_solve', [chol, rhs]):
    y = gen_linalg_ops.matrix_triangular_solve(
        chol, rhs, adjoint=False, lower=True)
    x = gen_linalg_ops.matrix_triangular_solve(
        chol, y, adjoint=True, lower=True)
    return x


def eye(num_rows,
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
      name, default_name='eye', values=[num_rows, num_columns, batch_shape]):

    batch_shape = [] if batch_shape is None else batch_shape
    batch_shape = ops.convert_to_tensor(
        batch_shape, name='shape', dtype=dtypes.int32)

    if num_columns is None:
      diag_size = num_rows
    else:
      diag_size = math_ops.minimum(num_rows, num_columns)
    diag_shape = array_ops.concat((batch_shape, [diag_size]), 0)
    diag_ones = array_ops.ones(diag_shape, dtype=dtype)

    if num_columns is None:
      return array_ops.matrix_diag(diag_ones)
    else:
      shape = array_ops.concat((batch_shape, [num_rows, num_columns]), 0)
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
  #v is a tensor of right singular vectors.
  s, u, v = svd(a)
  s = svd(a, compute_uv=False)
  ```

  Args:
    tensor: `Tensor` of shape `[..., M, N]`. Let `P` be the minimum of `M` and
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

  @compatibility(numpy)
  Mostly equivalent to numpy.linalg.svd, except that the order of output
  arguments here is `s`, `u`, `v` when `compute_uv` is `True`, as opposed to
  `u`, `s`, `v` for numpy.linalg.svd.
  @end_compatibility
  """
  # pylint: disable=protected-access
  s, u, v = gen_linalg_ops._svd(
      tensor, compute_uv=compute_uv, full_matrices=full_matrices)
  # pylint: enable=protected-access
  if compute_uv:
    return math_ops.real(s), u, v
  else:
    return math_ops.real(s)


# pylint: disable=redefined-builtin
def norm(tensor, ord='euclidean', axis=None, keep_dims=False, name=None):
  r"""Computes the norm of vectors, matrices, and tensors.

  This function can compute 3 different matrix norms (Frobenius, 1-norm, and
  inf-norm) and up to 9218868437227405311 different vectors norms.

  Args:
    tensor: `Tensor` of types `float32`, `float64`, `complex64`, `complex128`
    ord: Order of the norm. Supported values are 'fro', 'euclidean', `0`,
      `1, `2`, `np.inf` and any positive real number yielding the corresponding
      p-norm. Default is 'euclidean' which is equivalent to Frobenius norm if
      `tensor` is a matrix and equivalent to 2-norm for vectors.
      Some restrictions apply,
        a) The Frobenius norm `fro` is not defined for vectors,
        b) If axis is a 2-tuple (matrix-norm), only 'euclidean', 'fro', `1`,
           `np.inf` are supported.
      See the description of `axis` on how to compute norms for a batch of
      vectors or matrices stored in a tensor.
    axis: If `axis` is `None` (the default), the input is considered a vector
      and a single vector norm is computed over the entire set of values in the
      tensor, i.e. `norm(tensor, ord=ord)` is equivalent to
      `norm(reshape(tensor, [-1]), ord=ord)`.
      If `axis` is a Python integer, the input is considered a batch of vectors,
      and `axis`t determines the axis in `tensor` over which to compute vector
      norms.
      If `axis` is a 2-tuple of Python integers it is considered a batch of
      matrices and `axis` determines the axes in `tensor` over which to compute
      a matrix norm.
      Negative indices are supported. Example: If you are passing a tensor that
      can be either a matrix or a batch of matrices at runtime, pass
      `axis=[-2,-1]` instead of `axis=None` to make sure that matrix norms are
      computed.
    keep_dims: If True, the axis indicated in `axis` are kept with size 1.
      Otherwise, the dimensions in `axis` are removed from the output shape.
    name: The name of the op.

  Returns:
    output: A `Tensor` of the same type as tensor, containing the vector or
      matrix norms. If `keep_dims` is True then the rank of output is equal to
      the rank of `tensor`. Otherwise, if `axis` is none the output is a scalar,
      if `axis` is an integer, the rank of `output` is one less than the rank
      of `tensor`, if `axis` is a 2-tuple the rank of `output` is two less
      than the rank of `tensor`.

  Raises:
    ValueError: If `ord` or `axis` is invalid.

  @compatibility(numpy)
  Mostly equivalent to numpy.linalg.norm.
  Not supported: ord <= 0, 2-norm for matrices, nuclear norm.
  Other differences:
    a) If axis is `None`, treats the flattened `tensor` as a vector
     regardless of rank.
    b) Explicitly supports 'euclidean' norm as the default, including for
     higher order tensors.
  @end_compatibility
  """

  is_matrix_norm = ((isinstance(axis, tuple) or isinstance(axis, list)) and
                    len(axis) == 2)
  if is_matrix_norm:
    axis = tuple(axis)
    if (not isinstance(axis[0], int) or not isinstance(axis[1], int) or
        axis[0] == axis[1]):
      raise ValueError(
          "'axis' must be None, an integer, or a tuple of 2 unique integers")
    # TODO(rmlarsen): Implement matrix 2-norm using tf.svd().
    supported_matrix_norms = ['euclidean', 'fro', 1, np.inf]
    if ord not in supported_matrix_norms:
      raise ValueError("'ord' must be a supported matrix norm in %s, got %s" %
                       (supported_matrix_norms, ord))
  else:
    if not (isinstance(axis, int) or axis is None):
      raise ValueError(
          "'axis' must be None, an integer, or a tuple of 2 unique integers")

    supported_vector_norms = ['euclidean', 1, 2, np.inf]
    if (not np.isreal(ord) or ord <= 0) and ord not in supported_vector_norms:
      raise ValueError("'ord' must be a supported vector norm, got %s" % ord)
    if axis is not None:
      axis = (axis,)

  with ops.name_scope(name, 'norm', [tensor]):
    tensor = ops.convert_to_tensor(tensor)
    if ord in ['fro', 'euclidean', 2, 2.0]:
      # TODO(rmlarsen): Move 2-norm to a separate clause once we support it for
      # matrices.
      result = math_ops.sqrt(
          math_ops.reduce_sum(
              math_ops.square(tensor), axis, keep_dims=True))
    else:
      result = math_ops.abs(tensor)
      if ord == 1:
        sum_axis = None if axis is None else axis[0]
        result = math_ops.reduce_sum(result, sum_axis, keep_dims=True)
        if is_matrix_norm:
          result = math_ops.reduce_max(result, axis[-1], keep_dims=True)
      elif ord == np.inf:
        if is_matrix_norm:
          result = math_ops.reduce_sum(result, axis[1], keep_dims=True)
        max_axis = None if axis is None else axis[0]
        result = math_ops.reduce_max(result, max_axis, keep_dims=True)
      else:
        # General p-norms (positive p only)
        result = math_ops.pow(math_ops.reduce_sum(
            math_ops.pow(result, ord), axis, keep_dims=True),
                              1.0 / ord)
    if not keep_dims:
      result = array_ops.squeeze(result, axis)
    return result


# pylint: enable=invalid-name,redefined-builtin
