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
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_linalg_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

# Names below are lower_case.
# pylint: disable=invalid-name


def _RegularizedGramianCholesky(matrix, l2_regularizer, first_kind):
  r"""Computes Cholesky factorization of regularized gramian matrix.

  Below we will use the following notation for each pair of matrix and
  right-hand sides in the batch:

  `matrix`=\\(A \in \Re^{m \times n}\\),
  `output`=\\(C  \in \Re^{\min(m, n) \times \min(m,n)}\\),
  `l2_regularizer`=\\(\lambda\\).

  If `first_kind` is True, returns the Cholesky factorization \\(L\\) such that
  \\(L L^H =  A^H A + \lambda I\\).
  If `first_kind` is False, returns the Cholesky factorization \\(L\\) such that
  \\(L L^H =  A A^H + \lambda I\\).

  Args:
    matrix: `Tensor` of shape `[..., M, N]`.
    l2_regularizer: 0-D `double` `Tensor`. Ignored if `fast=False`.
    first_kind: bool. Controls what gramian matrix to factor.
  Returns:
    output: `Tensor` of shape `[..., min(M,N), min(M,N)]` whose inner-most 2
      dimensions contain the Cholesky factors \\(L\\) described above.
  """

  gramian = math_ops.matmul(
      matrix, matrix, adjoint_a=first_kind, adjoint_b=not first_kind)
  if isinstance(l2_regularizer, ops.Tensor) or l2_regularizer != 0:
    matrix_shape = array_ops.shape(matrix)
    batch_shape = matrix_shape[:-2]
    if first_kind:
      small_dim = matrix_shape[-1]
    else:
      small_dim = matrix_shape[-2]
    identity = eye(small_dim, batch_shape=batch_shape, dtype=matrix.dtype)
    small_dim_static = matrix.shape[-1 if first_kind else -2]
    identity.set_shape(
        matrix.shape[:-2].concatenate([small_dim_static, small_dim_static]))
    gramian += l2_regularizer * identity
  return gen_linalg_ops.cholesky(gramian)


@tf_export(
    'linalg.cholesky_solve', v1=['linalg.cholesky_solve', 'cholesky_solve'])
@deprecation.deprecated_endpoints('cholesky_solve')
def cholesky_solve(chol, rhs, name=None):
  """Solves systems of linear eqns `A X = RHS`, given Cholesky factorizations.

  ```python
  # Solve 10 separate 2x2 linear systems:
  A = ... # shape 10 x 2 x 2
  RHS = ... # shape 10 x 2 x 1
  chol = tf.linalg.cholesky(A)  # shape 10 x 2 x 2
  X = tf.linalg.cholesky_solve(chol, RHS)  # shape 10 x 2 x 1
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
      Cholesky factorization of `A`, e.g. `chol = tf.linalg.cholesky(A)`.
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


@tf_export('eye', 'linalg.eye')
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
    batch_shape:  A list or tuple of Python integers or a 1-D `int32` `Tensor`.
      If provided, the returned `Tensor` will have leading batch dimensions of
      this shape.
    dtype:  The type of an element in the resulting `Tensor`
    name:  A name for this `Op`.  Defaults to "eye".

  Returns:
    A `Tensor` of shape `batch_shape + [num_rows, num_columns]`
  """
  return linalg_ops_impl.eye(num_rows,
                             num_columns=num_columns,
                             batch_shape=batch_shape,
                             dtype=dtype,
                             name=name)


@tf_export('linalg.lstsq', v1=['linalg.lstsq', 'matrix_solve_ls'])
@deprecation.deprecated_endpoints('matrix_solve_ls')
def matrix_solve_ls(matrix, rhs, l2_regularizer=0.0, fast=True, name=None):
  r"""Solves one or more linear least-squares problems.

  `matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
  form `M`-by-`N` matrices. Rhs is a tensor of shape `[..., M, K]` whose
  inner-most 2 dimensions form `M`-by-`K` matrices.  The computed output is a
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

  Raises:
    NotImplementedError: linalg.lstsq is currently disabled for complex128
    and l2_regularizer != 0 due to poor accuracy.
  """

  # pylint: disable=long-lambda
  def _use_composite_impl(fast, tensor_shape):
    """Determines whether to use the composite or specialized CPU kernel.

    When the total size of the tensor is larger than the cache size and the
    batch size is large compared to the smallest matrix dimension, then the
    composite implementation is inefficient since it has to read the entire
    tensor from memory multiple times. In this case we fall back to the
    original CPU kernel, which does all the computational steps on each
    matrix separately.

    Only fast mode is supported by the composite impl, so `False` is returned
    if `fast` is `False`.

    Args:
      fast: bool indicating if fast mode in the solver was requested.
      tensor_shape: The shape of the tensor.

    Returns:
      True if the composite impl should be used. False otherwise.
    """
    if fast is False:
      return False
    batch_shape = tensor_shape[:-2]
    matrix_shape = tensor_shape[-2:]
    if not tensor_shape.is_fully_defined():
      return True
    tensor_size = tensor_shape.num_elements() * matrix.dtype.size
    is_io_bound = batch_shape.num_elements() > np.min(matrix_shape)
    L2_CACHE_SIZE_GUESSTIMATE = 256000
    if tensor_size > L2_CACHE_SIZE_GUESSTIMATE and is_io_bound:
      return False
    else:
      return True

  def _overdetermined(matrix, rhs, l2_regularizer):
    """Computes (A^H*A + l2_regularizer)^{-1} * A^H * rhs."""
    chol = _RegularizedGramianCholesky(
        matrix, l2_regularizer=l2_regularizer, first_kind=True)
    return cholesky_solve(chol, math_ops.matmul(matrix, rhs, adjoint_a=True))

  def _underdetermined(matrix, rhs, l2_regularizer):
    """Computes A^H * (A*A^H + l2_regularizer)^{-1} * rhs."""
    chol = _RegularizedGramianCholesky(
        matrix, l2_regularizer=l2_regularizer, first_kind=False)
    return math_ops.matmul(matrix, cholesky_solve(chol, rhs), adjoint_a=True)

  def _composite_impl(matrix, rhs, l2_regularizer):
    """Composite implementation of matrix_solve_ls that supports GPU."""
    with ops.name_scope(name, 'matrix_solve_ls', [matrix, rhs, l2_regularizer]):
      matrix_shape = matrix.get_shape()[-2:]
      if matrix_shape.is_fully_defined():
        if matrix_shape[-2] >= matrix_shape[-1]:
          return _overdetermined(matrix, rhs, l2_regularizer)
        else:
          return _underdetermined(matrix, rhs, l2_regularizer)
      else:
        # We have to defer determining the shape to runtime and use
        # conditional execution of the appropriate graph.
        matrix_shape = array_ops.shape(matrix)[-2:]
        return control_flow_ops.cond(
            matrix_shape[-2] >= matrix_shape[-1],
            lambda: _overdetermined(matrix, rhs, l2_regularizer),
            lambda: _underdetermined(matrix, rhs, l2_regularizer))

  matrix = ops.convert_to_tensor(matrix, name='matrix')
  if matrix.dtype == dtypes.complex128 and l2_regularizer != 0:
    # TODO(rmlarsen): Investigate and fix accuracy bug.
    raise NotImplementedError('matrix_solve_ls is currently disabled for '
                              'complex128 and l2_regularizer != 0 due to '
                              'poor accuracy.')
  tensor_shape = matrix.get_shape()
  if _use_composite_impl(fast, tensor_shape):
    return _composite_impl(matrix, rhs, l2_regularizer)
  else:
    return gen_linalg_ops.matrix_solve_ls(
        matrix, rhs, l2_regularizer, fast=fast, name=name)


@tf_export('linalg.eigh', v1=['linalg.eigh', 'self_adjoint_eig'])
@deprecation.deprecated_endpoints('self_adjoint_eig')
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
    e: Eigenvalues. Shape is `[..., N]`. Sorted in non-decreasing order.
    v: Eigenvectors. Shape is `[..., N, N]`. The columns of the inner most
      matrices contain eigenvectors of the corresponding matrices in `tensor`
  """
  e, v = gen_linalg_ops.self_adjoint_eig_v2(tensor, compute_v=True, name=name)
  return e, v


@tf_export('linalg.eigvalsh', v1=['linalg.eigvalsh', 'self_adjoint_eigvals'])
@deprecation.deprecated_endpoints('self_adjoint_eigvals')
def self_adjoint_eigvals(tensor, name=None):
  """Computes the eigenvalues of one or more self-adjoint matrices.

  Note: If your program backpropagates through this function, you should replace
  it with a call to tf.linalg.eigh (possibly ignoring the second output) to
  avoid computing the eigen decomposition twice. This is because the
  eigenvectors are used to compute the gradient w.r.t. the eigenvalues. See
  _SelfAdjointEigV2Grad in linalg_grad.py.

  Args:
    tensor: `Tensor` of shape `[..., N, N]`.
    name: string, optional name of the operation.

  Returns:
    e: Eigenvalues. Shape is `[..., N]`. The vector `e[..., :]` contains the `N`
      eigenvalues of `tensor[..., :, :]`.
  """
  e, _ = gen_linalg_ops.self_adjoint_eig_v2(tensor, compute_v=False, name=name)
  return e


@tf_export('linalg.svd', v1=['linalg.svd', 'svd'])
@deprecation.deprecated_endpoints('svd')
def svd(tensor, full_matrices=False, compute_uv=True, name=None):
  r"""Computes the singular value decompositions of one or more matrices.

  Computes the SVD of each inner matrix in `tensor` such that
  `tensor[..., :, :] = u[..., :, :] * diag(s[..., :, :]) *
   transpose(conj(v[..., :, :]))`

  ```python
  # a is a tensor.
  # s is a tensor of singular values.
  # u is a tensor of left singular vectors.
  # v is a tensor of right singular vectors.
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
    s: Singular values. Shape is `[..., P]`. The values are sorted in reverse
      order of magnitude, so s[..., 0] is the largest value, s[..., 1] is the
      second largest, etc.
    u: Left singular vectors. If `full_matrices` is `False` (default) then
      shape is `[..., M, P]`; if `full_matrices` is `True` then shape is
      `[..., M, M]`. Not returned if `compute_uv` is `False`.
    v: Right singular vectors. If `full_matrices` is `False` (default) then
      shape is `[..., N, P]`. If `full_matrices` is `True` then shape is
      `[..., N, N]`. Not returned if `compute_uv` is `False`.

  @compatibility(numpy)
  Mostly equivalent to numpy.linalg.svd, except that
    * The order of output  arguments here is `s`, `u`, `v` when `compute_uv` is
      `True`, as opposed to `u`, `s`, `v` for numpy.linalg.svd.
    * full_matrices is `False` by default as opposed to `True` for
       numpy.linalg.svd.
    * tf.linalg.svd uses the standard definition of the SVD
      \\(A = U \Sigma V^H\\), such that the left singular vectors of `a` are
      the columns of `u`, while the right singular vectors of `a` are the
      columns of `v`. On the other hand, numpy.linalg.svd returns the adjoint
      \\(V^H\\) as the third output argument.
  ```python
  import tensorflow as tf
  import numpy as np
  s, u, v = tf.linalg.svd(a)
  tf_a_approx = tf.matmul(u, tf.matmul(tf.linalg.diag(s), v, adjoint_b=True))
  u, s, v_adj = np.linalg.svd(a, full_matrices=False)
  np_a_approx = np.dot(u, np.dot(np.diag(s), v_adj))
  # tf_a_approx and np_a_approx should be numerically close.
  ```
  @end_compatibility
  """
  s, u, v = gen_linalg_ops.svd(
      tensor, compute_uv=compute_uv, full_matrices=full_matrices, name=name)
  if compute_uv:
    return math_ops.real(s), u, v
  else:
    return math_ops.real(s)


# pylint: disable=redefined-builtin
@tf_export('norm', 'linalg.norm', v1=[])
def norm_v2(tensor,
            ord='euclidean',
            axis=None,
            keepdims=None,
            name=None):
  r"""Computes the norm of vectors, matrices, and tensors.

  This function can compute several different vector norms (the 1-norm, the
  Euclidean or 2-norm, the inf-norm, and in general the p-norm for p > 0) and
  matrix norms (Frobenius, 1-norm, 2-norm and inf-norm).

  Args:
    tensor: `Tensor` of types `float32`, `float64`, `complex64`, `complex128`
    ord: Order of the norm. Supported values are `'fro'`, `'euclidean'`,
      `1`, `2`, `np.inf` and any positive real number yielding the corresponding
      p-norm. Default is `'euclidean'` which is equivalent to Frobenius norm if
      `tensor` is a matrix and equivalent to 2-norm for vectors.
      Some restrictions apply:
        a) The Frobenius norm `'fro'` is not defined for vectors,
        b) If axis is a 2-tuple (matrix norm), only `'euclidean'`, '`fro'`, `1`,
           `2`, `np.inf` are supported.
      See the description of `axis` on how to compute norms for a batch of
      vectors or matrices stored in a tensor.
    axis: If `axis` is `None` (the default), the input is considered a vector
      and a single vector norm is computed over the entire set of values in the
      tensor, i.e. `norm(tensor, ord=ord)` is equivalent to
      `norm(reshape(tensor, [-1]), ord=ord)`.
      If `axis` is a Python integer, the input is considered a batch of vectors,
      and `axis` determines the axis in `tensor` over which to compute vector
      norms.
      If `axis` is a 2-tuple of Python integers it is considered a batch of
      matrices and `axis` determines the axes in `tensor` over which to compute
      a matrix norm.
      Negative indices are supported. Example: If you are passing a tensor that
      can be either a matrix or a batch of matrices at runtime, pass
      `axis=[-2,-1]` instead of `axis=None` to make sure that matrix norms are
      computed.
    keepdims: If True, the axis indicated in `axis` are kept with size 1.
      Otherwise, the dimensions in `axis` are removed from the output shape.
    name: The name of the op.

  Returns:
    output: A `Tensor` of the same type as tensor, containing the vector or
      matrix norms. If `keepdims` is True then the rank of output is equal to
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
  return norm(tensor=tensor,
              ord=ord,
              axis=axis,
              keepdims=keepdims,
              name=name)


# pylint: disable=redefined-builtin
@tf_export(v1=['norm', 'linalg.norm'])
@deprecation.deprecated_args(
    None, 'keep_dims is deprecated, use keepdims instead', 'keep_dims')
def norm(tensor,
         ord='euclidean',
         axis=None,
         keepdims=None,
         name=None,
         keep_dims=None):
  r"""Computes the norm of vectors, matrices, and tensors.

  This function can compute several different vector norms (the 1-norm, the
  Euclidean or 2-norm, the inf-norm, and in general the p-norm for p > 0) and
  matrix norms (Frobenius, 1-norm, 2-norm and inf-norm).

  Args:
    tensor: `Tensor` of types `float32`, `float64`, `complex64`, `complex128`
    ord: Order of the norm. Supported values are 'fro', 'euclidean',
      `1`, `2`, `np.inf` and any positive real number yielding the corresponding
      p-norm. Default is 'euclidean' which is equivalent to Frobenius norm if
      `tensor` is a matrix and equivalent to 2-norm for vectors.
      Some restrictions apply:
        a) The Frobenius norm `fro` is not defined for vectors,
        b) If axis is a 2-tuple (matrix norm), only 'euclidean', 'fro', `1`,
           `2`, `np.inf` are supported.
      See the description of `axis` on how to compute norms for a batch of
      vectors or matrices stored in a tensor.
    axis: If `axis` is `None` (the default), the input is considered a vector
      and a single vector norm is computed over the entire set of values in the
      tensor, i.e. `norm(tensor, ord=ord)` is equivalent to
      `norm(reshape(tensor, [-1]), ord=ord)`.
      If `axis` is a Python integer, the input is considered a batch of vectors,
      and `axis` determines the axis in `tensor` over which to compute vector
      norms.
      If `axis` is a 2-tuple of Python integers it is considered a batch of
      matrices and `axis` determines the axes in `tensor` over which to compute
      a matrix norm.
      Negative indices are supported. Example: If you are passing a tensor that
      can be either a matrix or a batch of matrices at runtime, pass
      `axis=[-2,-1]` instead of `axis=None` to make sure that matrix norms are
      computed.
    keepdims: If True, the axis indicated in `axis` are kept with size 1.
      Otherwise, the dimensions in `axis` are removed from the output shape.
    name: The name of the op.
    keep_dims: Deprecated alias for `keepdims`.

  Returns:
    output: A `Tensor` of the same type as tensor, containing the vector or
      matrix norms. If `keepdims` is True then the rank of output is equal to
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
  keepdims = deprecation.deprecated_argument_lookup('keepdims', keepdims,
                                                    'keep_dims', keep_dims)
  if keepdims is None:
    keepdims = False

  is_matrix_norm = ((isinstance(axis, tuple) or isinstance(axis, list)) and
                    len(axis) == 2)
  if is_matrix_norm:
    axis = tuple(axis)
    if (not isinstance(axis[0], int) or not isinstance(axis[1], int) or
        axis[0] == axis[1]):
      raise ValueError(
          "'axis' must be None, an integer, or a tuple of 2 unique integers")
    supported_matrix_norms = ['euclidean', 'fro', 1, 2, np.inf]
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
      if is_matrix_norm and ord in [2, 2.0]:
        rank = array_ops.rank(tensor)
        positive_axis = map_fn.map_fn(
            lambda i: control_flow_ops.cond(i >= 0, lambda: i, lambda: i + rank),
            ops.convert_to_tensor(axis))
        axes = math_ops.range(rank)
        perm_before = array_ops.concat(
            [array_ops.setdiff1d(axes, positive_axis)[0], positive_axis],
            axis=0)
        perm_after = map_fn.map_fn(
            lambda i: math_ops.cast(
                array_ops.squeeze(
                    array_ops.where(math_ops.equal(perm_before, i))),
                dtype=dtypes.int32), axes)
        permed = array_ops.transpose(tensor, perm=perm_before)
        matrix_2_norm = array_ops.expand_dims(
            math_ops.reduce_max(
                math_ops.abs(gen_linalg_ops.svd(permed, compute_uv=False)[0]),
                axis=-1,
                keepdims=True),
            axis=-1)
        result = array_ops.transpose(matrix_2_norm, perm=perm_after)
      else:
        result = math_ops.sqrt(
            math_ops.reduce_sum(
                tensor * math_ops.conj(tensor), axis, keepdims=True))
        # TODO(rmlarsen): Replace with the following, once gradients are defined
        # result = math_ops.reduce_euclidean_norm(tensor, axis, keepdims=True)
    else:
      result = math_ops.abs(tensor)
      if ord == 1:
        sum_axis = None if axis is None else axis[0]
        result = math_ops.reduce_sum(result, sum_axis, keepdims=True)
        if is_matrix_norm:
          result = math_ops.reduce_max(result, axis[-1], keepdims=True)
      elif ord == np.inf:
        if is_matrix_norm:
          result = math_ops.reduce_sum(result, axis[1], keepdims=True)
        max_axis = None if axis is None else axis[0]
        result = math_ops.reduce_max(result, max_axis, keepdims=True)
      else:
        # General p-norms (positive p only)
        result = math_ops.pow(
            math_ops.reduce_sum(math_ops.pow(result, ord), axis, keepdims=True),
            1.0 / ord)
    if not keepdims:
      result = array_ops.squeeze(result, axis)
    return result


# pylint: enable=invalid-name,redefined-builtin
