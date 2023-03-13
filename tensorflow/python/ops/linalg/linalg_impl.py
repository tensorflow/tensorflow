# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

# Linear algebra ops.
band_part = array_ops.matrix_band_part
cholesky = linalg_ops.cholesky
cholesky_solve = linalg_ops.cholesky_solve
det = linalg_ops.matrix_determinant
slogdet = gen_linalg_ops.log_matrix_determinant
tf_export('linalg.slogdet')(dispatch.add_dispatch_support(slogdet))
diag = array_ops.matrix_diag
diag_part = array_ops.matrix_diag_part
eigh = linalg_ops.self_adjoint_eig
eigvalsh = linalg_ops.self_adjoint_eigvals
einsum = special_math_ops.einsum
eye = linalg_ops.eye
inv = linalg_ops.matrix_inverse
logm = gen_linalg_ops.matrix_logarithm
lu = gen_linalg_ops.lu
tf_export('linalg.logm')(dispatch.add_dispatch_support(logm))
lstsq = linalg_ops.matrix_solve_ls
norm = linalg_ops.norm
qr = linalg_ops.qr
set_diag = array_ops.matrix_set_diag
solve = linalg_ops.matrix_solve
sqrtm = linalg_ops.matrix_square_root
svd = linalg_ops.svd
tensordot = math_ops.tensordot
trace = math_ops.trace
transpose = array_ops.matrix_transpose
triangular_solve = linalg_ops.matrix_triangular_solve


@tf_export('linalg.logdet')
@dispatch.add_dispatch_support
def logdet(matrix, name=None):
  """Computes log of the determinant of a hermitian positive definite matrix.

  ```python
  # Compute the determinant of a matrix while reducing the chance of over- or
  underflow:
  A = ... # shape 10 x 10
  det = tf.exp(tf.linalg.logdet(A))  # scalar
  ```

  Args:
    matrix:  A `Tensor`. Must be `float16`, `float32`, `float64`, `complex64`,
      or `complex128` with shape `[..., M, M]`.
    name:  A name to give this `Op`.  Defaults to `logdet`.

  Returns:
    The natural log of the determinant of `matrix`.

  @compatibility(numpy)
  Equivalent to numpy.linalg.slogdet, although no sign is returned since only
  hermitian positive definite matrices are supported.
  @end_compatibility
  """
  # This uses the property that the log det(A) = 2*sum(log(real(diag(C))))
  # where C is the cholesky decomposition of A.
  with ops.name_scope(name, 'logdet', [matrix]):
    chol = gen_linalg_ops.cholesky(matrix)
    return 2.0 * math_ops.reduce_sum(
        math_ops.log(math_ops.real(array_ops.matrix_diag_part(chol))),
        axis=[-1])


@tf_export('linalg.adjoint')
@dispatch.add_dispatch_support
def adjoint(matrix, name=None):
  """Transposes the last two dimensions of and conjugates tensor `matrix`.

  For example:

  ```python
  x = tf.constant([[1 + 1j, 2 + 2j, 3 + 3j],
                   [4 + 4j, 5 + 5j, 6 + 6j]])
  tf.linalg.adjoint(x)  # [[1 - 1j, 4 - 4j],
                        #  [2 - 2j, 5 - 5j],
                        #  [3 - 3j, 6 - 6j]]
  ```

  Args:
    matrix:  A `Tensor`. Must be `float16`, `float32`, `float64`, `complex64`,
      or `complex128` with shape `[..., M, M]`.
    name:  A name to give this `Op` (optional).

  Returns:
    The adjoint (a.k.a. Hermitian transpose a.k.a. conjugate transpose) of
    matrix.
  """
  with ops.name_scope(name, 'adjoint', [matrix]):
    matrix = ops.convert_to_tensor(matrix, name='matrix')
    return array_ops.matrix_transpose(matrix, conjugate=True)


# This section is ported nearly verbatim from Eigen's implementation:
# https://eigen.tuxfamily.org/dox/unsupported/MatrixExponential_8h_source.html
def _matrix_exp_pade3(matrix):
  """3rd-order Pade approximant for matrix exponential."""
  b = [120.0, 60.0, 12.0]
  b = [constant_op.constant(x, matrix.dtype) for x in b]
  ident = linalg_ops.eye(
      array_ops.shape(matrix)[-2],
      batch_shape=array_ops.shape(matrix)[:-2],
      dtype=matrix.dtype)
  matrix_2 = math_ops.matmul(matrix, matrix)
  tmp = matrix_2 + b[1] * ident
  matrix_u = math_ops.matmul(matrix, tmp)
  matrix_v = b[2] * matrix_2 + b[0] * ident
  return matrix_u, matrix_v


def _matrix_exp_pade5(matrix):
  """5th-order Pade approximant for matrix exponential."""
  b = [30240.0, 15120.0, 3360.0, 420.0, 30.0]
  b = [constant_op.constant(x, matrix.dtype) for x in b]
  ident = linalg_ops.eye(
      array_ops.shape(matrix)[-2],
      batch_shape=array_ops.shape(matrix)[:-2],
      dtype=matrix.dtype)
  matrix_2 = math_ops.matmul(matrix, matrix)
  matrix_4 = math_ops.matmul(matrix_2, matrix_2)
  tmp = matrix_4 + b[3] * matrix_2 + b[1] * ident
  matrix_u = math_ops.matmul(matrix, tmp)
  matrix_v = b[4] * matrix_4 + b[2] * matrix_2 + b[0] * ident
  return matrix_u, matrix_v


def _matrix_exp_pade7(matrix):
  """7th-order Pade approximant for matrix exponential."""
  b = [17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0]
  b = [constant_op.constant(x, matrix.dtype) for x in b]
  ident = linalg_ops.eye(
      array_ops.shape(matrix)[-2],
      batch_shape=array_ops.shape(matrix)[:-2],
      dtype=matrix.dtype)
  matrix_2 = math_ops.matmul(matrix, matrix)
  matrix_4 = math_ops.matmul(matrix_2, matrix_2)
  matrix_6 = math_ops.matmul(matrix_4, matrix_2)
  tmp = matrix_6 + b[5] * matrix_4 + b[3] * matrix_2 + b[1] * ident
  matrix_u = math_ops.matmul(matrix, tmp)
  matrix_v = b[6] * matrix_6 + b[4] * matrix_4 + b[2] * matrix_2 + b[0] * ident
  return matrix_u, matrix_v


def _matrix_exp_pade9(matrix):
  """9th-order Pade approximant for matrix exponential."""
  b = [
      17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0,
      2162160.0, 110880.0, 3960.0, 90.0
  ]
  b = [constant_op.constant(x, matrix.dtype) for x in b]
  ident = linalg_ops.eye(
      array_ops.shape(matrix)[-2],
      batch_shape=array_ops.shape(matrix)[:-2],
      dtype=matrix.dtype)
  matrix_2 = math_ops.matmul(matrix, matrix)
  matrix_4 = math_ops.matmul(matrix_2, matrix_2)
  matrix_6 = math_ops.matmul(matrix_4, matrix_2)
  matrix_8 = math_ops.matmul(matrix_6, matrix_2)
  tmp = (
      matrix_8 + b[7] * matrix_6 + b[5] * matrix_4 + b[3] * matrix_2 +
      b[1] * ident)
  matrix_u = math_ops.matmul(matrix, tmp)
  matrix_v = (
      b[8] * matrix_8 + b[6] * matrix_6 + b[4] * matrix_4 + b[2] * matrix_2 +
      b[0] * ident)
  return matrix_u, matrix_v


def _matrix_exp_pade13(matrix):
  """13th-order Pade approximant for matrix exponential."""
  b = [
      64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
      1187353796428800.0, 129060195264000.0, 10559470521600.0, 670442572800.0,
      33522128640.0, 1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0
  ]
  b = [constant_op.constant(x, matrix.dtype) for x in b]
  ident = linalg_ops.eye(
      array_ops.shape(matrix)[-2],
      batch_shape=array_ops.shape(matrix)[:-2],
      dtype=matrix.dtype)
  matrix_2 = math_ops.matmul(matrix, matrix)
  matrix_4 = math_ops.matmul(matrix_2, matrix_2)
  matrix_6 = math_ops.matmul(matrix_4, matrix_2)
  tmp_u = (
      math_ops.matmul(matrix_6, matrix_6 + b[11] * matrix_4 + b[9] * matrix_2) +
      b[7] * matrix_6 + b[5] * matrix_4 + b[3] * matrix_2 + b[1] * ident)
  matrix_u = math_ops.matmul(matrix, tmp_u)
  tmp_v = b[12] * matrix_6 + b[10] * matrix_4 + b[8] * matrix_2
  matrix_v = (
      math_ops.matmul(matrix_6, tmp_v) + b[6] * matrix_6 + b[4] * matrix_4 +
      b[2] * matrix_2 + b[0] * ident)
  return matrix_u, matrix_v


@tf_export('linalg.expm')
@dispatch.add_dispatch_support
def matrix_exponential(input, name=None):  # pylint: disable=redefined-builtin
  r"""Computes the matrix exponential of one or more square matrices.

  $$exp(A) = \sum_{n=0}^\infty A^n/n!$$

  The exponential is computed using a combination of the scaling and squaring
  method and the Pade approximation. Details can be found in:
  Nicholas J. Higham, "The scaling and squaring method for the matrix
  exponential revisited," SIAM J. Matrix Anal. Applic., 26:1179-1193, 2005.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. The output is a tensor of the same shape as the input
  containing the exponential for all input submatrices `[..., :, :]`.

  Args:
    input: A `Tensor`. Must be `float16`, `float32`, `float64`, `complex64`, or
      `complex128` with shape `[..., M, M]`.
    name:  A name to give this `Op` (optional).

  Returns:
    the matrix exponential of the input.

  Raises:
    ValueError: An unsupported type is provided as input.

  @compatibility(scipy)
  Equivalent to scipy.linalg.expm
  @end_compatibility
  """
  with ops.name_scope(name, 'matrix_exponential', [input]):
    matrix = ops.convert_to_tensor(input, name='input')
    if matrix.shape[-2:] == [0, 0]:
      return matrix
    batch_shape = matrix.shape[:-2]
    if not batch_shape.is_fully_defined():
      batch_shape = array_ops.shape(matrix)[:-2]

    # reshaping the batch makes the where statements work better
    matrix = array_ops.reshape(
        matrix, array_ops.concat(([-1], array_ops.shape(matrix)[-2:]), axis=0))
    l1_norm = math_ops.reduce_max(
        math_ops.reduce_sum(
            math_ops.abs(matrix),
            axis=array_ops.size(array_ops.shape(matrix)) - 2),
        axis=-1)[..., array_ops.newaxis, array_ops.newaxis]

    const = lambda x: constant_op.constant(x, l1_norm.dtype)

    def _nest_where(vals, cases):
      assert len(vals) == len(cases) - 1
      if len(vals) == 1:
        return array_ops.where_v2(
            math_ops.less(l1_norm, const(vals[0])), cases[0], cases[1])
      else:
        return array_ops.where_v2(
            math_ops.less(l1_norm, const(vals[0])), cases[0],
            _nest_where(vals[1:], cases[1:]))

    if matrix.dtype in [dtypes.float16, dtypes.float32, dtypes.complex64]:
      maxnorm = const(3.925724783138660)
      squarings = math_ops.maximum(
          math_ops.floor(
              math_ops.log(l1_norm / maxnorm) / math_ops.log(const(2.0))), 0)
      u3, v3 = _matrix_exp_pade3(matrix)
      u5, v5 = _matrix_exp_pade5(matrix)
      u7, v7 = _matrix_exp_pade7(
          matrix /
          math_ops.cast(math_ops.pow(const(2.0), squarings), matrix.dtype))
      conds = (4.258730016922831e-001, 1.880152677804762e+000)
      u = _nest_where(conds, (u3, u5, u7))
      v = _nest_where(conds, (v3, v5, v7))
    elif matrix.dtype in [dtypes.float64, dtypes.complex128]:
      maxnorm = const(5.371920351148152)
      squarings = math_ops.maximum(
          math_ops.floor(
              math_ops.log(l1_norm / maxnorm) / math_ops.log(const(2.0))), 0)
      u3, v3 = _matrix_exp_pade3(matrix)
      u5, v5 = _matrix_exp_pade5(matrix)
      u7, v7 = _matrix_exp_pade7(matrix)
      u9, v9 = _matrix_exp_pade9(matrix)
      u13, v13 = _matrix_exp_pade13(
          matrix /
          math_ops.cast(math_ops.pow(const(2.0), squarings), matrix.dtype))
      conds = (1.495585217958292e-002, 2.539398330063230e-001,
               9.504178996162932e-001, 2.097847961257068e+000)
      u = _nest_where(conds, (u3, u5, u7, u9, u13))
      v = _nest_where(conds, (v3, v5, v7, v9, v13))
    else:
      raise ValueError('tf.linalg.expm does not support matrices of type %s' %
                       matrix.dtype)

    is_finite = math_ops.is_finite(math_ops.reduce_max(l1_norm))
    nan = constant_op.constant(np.nan, matrix.dtype)
    result = control_flow_ops.cond(
        is_finite, lambda: linalg_ops.matrix_solve(-u + v, u + v),
        lambda: array_ops.fill(array_ops.shape(matrix), nan))
    max_squarings = math_ops.reduce_max(squarings)
    i = const(0.0)

    def c(i, _):
      return control_flow_ops.cond(is_finite,
                                   lambda: math_ops.less(i, max_squarings),
                                   lambda: constant_op.constant(False))

    def b(i, r):
      return i + 1, array_ops.where_v2(
          math_ops.less(i, squarings), math_ops.matmul(r, r), r)

    _, result = control_flow_ops.while_loop(c, b, [i, result])
    if not matrix.shape.is_fully_defined():
      return array_ops.reshape(
          result,
          array_ops.concat((batch_shape, array_ops.shape(result)[-2:]), axis=0))
    return array_ops.reshape(result, batch_shape.concatenate(result.shape[-2:]))


@tf_export('linalg.banded_triangular_solve', v1=[])
def banded_triangular_solve(
    bands,
    rhs,
    lower=True,
    adjoint=False,  # pylint: disable=redefined-outer-name
    name=None):
  r"""Solve triangular systems of equations with a banded solver.

  `bands` is a tensor of shape `[..., K, M]`, where `K` represents the number
  of bands stored. This corresponds to a batch of `M` by `M` matrices, whose
  `K` subdiagonals (when `lower` is `True`) are stored.

  This operator broadcasts the batch dimensions of `bands` and the batch
  dimensions of `rhs`.


  Examples:

  Storing 2 bands of a 3x3 matrix.
  Note that first element in the second row is ignored due to
  the 'LEFT_RIGHT' padding.

  >>> x = [[2., 3., 4.], [1., 2., 3.]]
  >>> x2 = [[2., 3., 4.], [10000., 2., 3.]]
  >>> y = tf.zeros([3, 3])
  >>> z = tf.linalg.set_diag(y, x, align='LEFT_RIGHT', k=(-1, 0))
  >>> z
  <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
  array([[2., 0., 0.],
         [2., 3., 0.],
         [0., 3., 4.]], dtype=float32)>
  >>> soln = tf.linalg.banded_triangular_solve(x, tf.ones([3, 1]))
  >>> soln
  <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
  array([[0.5 ],
         [0.  ],
         [0.25]], dtype=float32)>
  >>> are_equal = soln == tf.linalg.banded_triangular_solve(x2, tf.ones([3, 1]))
  >>> tf.reduce_all(are_equal).numpy()
  True
  >>> are_equal = soln == tf.linalg.triangular_solve(z, tf.ones([3, 1]))
  >>> tf.reduce_all(are_equal).numpy()
  True

  Storing 2 superdiagonals of a 4x4 matrix. Because of the 'LEFT_RIGHT' padding
  the last element of the first row is ignored.

  >>> x = [[2., 3., 4., 5.], [-1., -2., -3., -4.]]
  >>> y = tf.zeros([4, 4])
  >>> z = tf.linalg.set_diag(y, x, align='LEFT_RIGHT', k=(0, 1))
  >>> z
  <tf.Tensor: shape=(4, 4), dtype=float32, numpy=
  array([[-1.,  2.,  0.,  0.],
         [ 0., -2.,  3.,  0.],
         [ 0.,  0., -3.,  4.],
         [ 0.,  0., -0., -4.]], dtype=float32)>
  >>> soln = tf.linalg.banded_triangular_solve(x, tf.ones([4, 1]), lower=False)
  >>> soln
  <tf.Tensor: shape=(4, 1), dtype=float32, numpy=
  array([[-4.       ],
         [-1.5      ],
         [-0.6666667],
         [-0.25     ]], dtype=float32)>
  >>> are_equal = (soln == tf.linalg.triangular_solve(
  ...   z, tf.ones([4, 1]), lower=False))
  >>> tf.reduce_all(are_equal).numpy()
  True


  Args:
    bands: A `Tensor` describing the bands of the left hand side, with shape
      `[..., K, M]`. The `K` rows correspond to the diagonal to the `K - 1`-th
      diagonal (the diagonal is the top row) when `lower` is `True` and
      otherwise the `K - 1`-th superdiagonal to the diagonal (the diagonal is
      the bottom row) when `lower` is `False`. The bands are stored with
      'LEFT_RIGHT' alignment, where the superdiagonals are padded on the right
      and subdiagonals are padded on the left. This is the alignment cuSPARSE
      uses.  See  `tf.linalg.set_diag` for more details.
    rhs: A `Tensor` of shape [..., M] or [..., M, N] and with the same dtype as
      `diagonals`. Note that if the shape of `rhs` and/or `diags` isn't known
      statically, `rhs` will be treated as a matrix rather than a vector.
    lower: An optional `bool`. Defaults to `True`. Boolean indicating whether
      `bands` represents a lower or upper triangular matrix.
    adjoint: An optional `bool`. Defaults to `False`. Boolean indicating whether
      to solve with the matrix's block-wise adjoint.
    name:  A name to give this `Op` (optional).

  Returns:
    A `Tensor` of shape [..., M] or [..., M, N] containing the solutions.
  """
  with ops.name_scope(name, 'banded_triangular_solve', [bands, rhs]):
    return gen_linalg_ops.banded_triangular_solve(
        bands, rhs, lower=lower, adjoint=adjoint)


@tf_export('linalg.tridiagonal_solve')
@dispatch.add_dispatch_support
def tridiagonal_solve(diagonals,
                      rhs,
                      diagonals_format='compact',
                      transpose_rhs=False,
                      conjugate_rhs=False,
                      name=None,
                      partial_pivoting=True,
                      perturb_singular=False):
  r"""Solves tridiagonal systems of equations.

  The input can be supplied in various formats: `matrix`, `sequence` and
  `compact`, specified by the `diagonals_format` arg.

  In `matrix` format, `diagonals` must be a tensor of shape `[..., M, M]`, with
  two inner-most dimensions representing the square tridiagonal matrices.
  Elements outside of the three diagonals will be ignored.

  In `sequence` format, `diagonals` are supplied as a tuple or list of three
  tensors of shapes `[..., N]`, `[..., M]`, `[..., N]` representing
  superdiagonals, diagonals, and subdiagonals, respectively. `N` can be either
  `M-1` or `M`; in the latter case, the last element of superdiagonal and the
  first element of subdiagonal will be ignored.

  In `compact` format the three diagonals are brought together into one tensor
  of shape `[..., 3, M]`, with last two dimensions containing superdiagonals,
  diagonals, and subdiagonals, in order. Similarly to `sequence` format,
  elements `diagonals[..., 0, M-1]` and `diagonals[..., 2, 0]` are ignored.

  The `compact` format is recommended as the one with best performance. In case
  you need to cast a tensor into a compact format manually, use `tf.gather_nd`.
  An example for a tensor of shape [m, m]:

  ```python
  rhs = tf.constant([...])
  matrix = tf.constant([[...]])
  m = matrix.shape[0]
  dummy_idx = [0, 0]  # An arbitrary element to use as a dummy
  indices = [[[i, i + 1] for i in range(m - 1)] + [dummy_idx],  # Superdiagonal
           [[i, i] for i in range(m)],                          # Diagonal
           [dummy_idx] + [[i + 1, i] for i in range(m - 1)]]    # Subdiagonal
  diagonals=tf.gather_nd(matrix, indices)
  x = tf.linalg.tridiagonal_solve(diagonals, rhs)
  ```

  Regardless of the `diagonals_format`, `rhs` is a tensor of shape `[..., M]` or
  `[..., M, K]`. The latter allows to simultaneously solve K systems with the
  same left-hand sides and K different right-hand sides. If `transpose_rhs`
  is set to `True` the expected shape is `[..., M]` or `[..., K, M]`.

  The batch dimensions, denoted as `...`, must be the same in `diagonals` and
  `rhs`.

  The output is a tensor of the same shape as `rhs`: either `[..., M]` or
  `[..., M, K]`.

  The op isn't guaranteed to raise an error if the input matrix is not
  invertible. `tf.debugging.check_numerics` can be applied to the output to
  detect invertibility problems.

  **Note**: with large batch sizes, the computation on the GPU may be slow, if
  either `partial_pivoting=True` or there are multiple right-hand sides
  (`K > 1`). If this issue arises, consider if it's possible to disable pivoting
  and have `K = 1`, or, alternatively, consider using CPU.

  On CPU, solution is computed via Gaussian elimination with or without partial
  pivoting, depending on `partial_pivoting` parameter. On GPU, Nvidia's cuSPARSE
  library is used: https://docs.nvidia.com/cuda/cusparse/index.html#gtsv

  Args:
    diagonals: A `Tensor` or tuple of `Tensor`s describing left-hand sides. The
      shape depends of `diagonals_format`, see description above. Must be
      `float32`, `float64`, `complex64`, or `complex128`.
    rhs: A `Tensor` of shape [..., M] or [..., M, K] and with the same dtype as
      `diagonals`. Note that if the shape of `rhs` and/or `diags` isn't known
      statically, `rhs` will be treated as a matrix rather than a vector.
    diagonals_format: one of `matrix`, `sequence`, or `compact`. Default is
      `compact`.
    transpose_rhs: If `True`, `rhs` is transposed before solving (has no effect
      if the shape of rhs is [..., M]).
    conjugate_rhs: If `True`, `rhs` is conjugated before solving.
    name:  A name to give this `Op` (optional).
    partial_pivoting: whether to perform partial pivoting. `True` by default.
      Partial pivoting makes the procedure more stable, but slower. Partial
      pivoting is unnecessary in some cases, including diagonally dominant and
      symmetric positive definite matrices (see e.g. theorem 9.12 in [1]).
    perturb_singular: whether to perturb singular matrices to return a finite
      result. `False` by default. If true, solutions to systems involving
      a singular matrix will be computed by perturbing near-zero pivots in
      the partially pivoted LU decomposition. Specifically, tiny pivots are
      perturbed by an amount of order `eps * max_{ij} |U(i,j)|` to avoid
      overflow. Here `U` is the upper triangular part of the LU decomposition,
      and `eps` is the machine precision. This is useful for solving
      numerically singular systems when computing eigenvectors by inverse
      iteration.
      If `partial_pivoting` is `False`, `perturb_singular` must be `False` as
      well.

  Returns:
    A `Tensor` of shape [..., M] or [..., M, K] containing the solutions.
    If the input matrix is singular, the result is undefined.

  Raises:
    ValueError: Is raised if any of the following conditions hold:
      1. An unsupported type is provided as input,
      2. the input tensors have incorrect shapes,
      3. `perturb_singular` is `True` but `partial_pivoting` is not.
    UnimplementedError: Whenever `partial_pivoting` is true and the backend is
      XLA, or whenever `perturb_singular` is true and the backend is
      XLA or GPU.

  [1] Nicholas J. Higham (2002). Accuracy and Stability of Numerical Algorithms:
  Second Edition. SIAM. p. 175. ISBN 978-0-89871-802-7.

  """
  if perturb_singular and not partial_pivoting:
    raise ValueError('partial_pivoting must be True if perturb_singular is.')

  if diagonals_format == 'compact':
    return _tridiagonal_solve_compact_format(diagonals, rhs, transpose_rhs,
                                             conjugate_rhs, partial_pivoting,
                                             perturb_singular, name)

  if diagonals_format == 'sequence':
    if not isinstance(diagonals, (tuple, list)) or len(diagonals) != 3:
      raise ValueError('Expected diagonals to be a sequence of length 3.')

    superdiag, maindiag, subdiag = diagonals
    if (not subdiag.shape[:-1].is_compatible_with(maindiag.shape[:-1]) or
        not superdiag.shape[:-1].is_compatible_with(maindiag.shape[:-1])):
      raise ValueError(
          'Tensors representing the three diagonals must have the same shape,'
          'except for the last dimension, got {}, {}, {}'.format(
              subdiag.shape, maindiag.shape, superdiag.shape))

    m = tensor_shape.dimension_value(maindiag.shape[-1])

    def pad_if_necessary(t, name, last_dim_padding):
      n = tensor_shape.dimension_value(t.shape[-1])
      if not n or n == m:
        return t
      if n == m - 1:
        paddings = ([[0, 0] for _ in range(len(t.shape) - 1)] +
                    [last_dim_padding])
        return array_ops.pad(t, paddings)
      raise ValueError('Expected {} to be have length {} or {}, got {}.'.format(
          name, m, m - 1, n))

    subdiag = pad_if_necessary(subdiag, 'subdiagonal', [1, 0])
    superdiag = pad_if_necessary(superdiag, 'superdiagonal', [0, 1])

    diagonals = array_ops_stack.stack((superdiag, maindiag, subdiag), axis=-2)
    return _tridiagonal_solve_compact_format(diagonals, rhs, transpose_rhs,
                                             conjugate_rhs, partial_pivoting,
                                             perturb_singular, name)

  if diagonals_format == 'matrix':
    m1 = tensor_shape.dimension_value(diagonals.shape[-1])
    m2 = tensor_shape.dimension_value(diagonals.shape[-2])
    if m1 and m2 and m1 != m2:
      raise ValueError(
          'Expected last two dimensions of diagonals to be same, got {} and {}'
          .format(m1, m2))
    m = m1 or m2
    diagonals = array_ops.matrix_diag_part(
        diagonals, k=(-1, 1), padding_value=0., align='LEFT_RIGHT')
    return _tridiagonal_solve_compact_format(diagonals, rhs, transpose_rhs,
                                             conjugate_rhs, partial_pivoting,
                                             perturb_singular, name)

  raise ValueError('Unrecognized diagonals_format: {}'.format(diagonals_format))


def _tridiagonal_solve_compact_format(diagonals, rhs, transpose_rhs,
                                      conjugate_rhs, partial_pivoting,
                                      perturb_singular, name):
  """Helper function used after the input has been cast to compact form."""
  diags_rank, rhs_rank = diagonals.shape.rank, rhs.shape.rank

  # If we know the rank of the diagonal tensor, do some static checking.
  if diags_rank:
    if diags_rank < 2:
      raise ValueError(
          'Expected diagonals to have rank at least 2, got {}'.format(
              diags_rank))
    if rhs_rank and rhs_rank != diags_rank and rhs_rank != diags_rank - 1:
      raise ValueError('Expected the rank of rhs to be {} or {}, got {}'.format(
          diags_rank - 1, diags_rank, rhs_rank))
    if (rhs_rank and not diagonals.shape[:-2].is_compatible_with(
        rhs.shape[:diags_rank - 2])):
      raise ValueError('Batch shapes {} and {} are incompatible'.format(
          diagonals.shape[:-2], rhs.shape[:diags_rank - 2]))

  if diagonals.shape[-2] and diagonals.shape[-2] != 3:
    raise ValueError('Expected 3 diagonals got {}'.format(diagonals.shape[-2]))

  def check_num_lhs_matches_num_rhs():
    if (diagonals.shape[-1] and rhs.shape[-2] and
        diagonals.shape[-1] != rhs.shape[-2]):
      raise ValueError('Expected number of left-hand sided and right-hand '
                       'sides to be equal, got {} and {}'.format(
                           diagonals.shape[-1], rhs.shape[-2]))

  if rhs_rank and diags_rank and rhs_rank == diags_rank - 1:
    # Rhs provided as a vector, ignoring transpose_rhs
    if conjugate_rhs:
      rhs = math_ops.conj(rhs)
    rhs = array_ops.expand_dims(rhs, -1)
    check_num_lhs_matches_num_rhs()
    return array_ops.squeeze(
        linalg_ops.tridiagonal_solve(diagonals, rhs, partial_pivoting,
                                     perturb_singular, name), -1)

  if transpose_rhs:
    rhs = array_ops.matrix_transpose(rhs, conjugate=conjugate_rhs)
  elif conjugate_rhs:
    rhs = math_ops.conj(rhs)

  check_num_lhs_matches_num_rhs()
  return linalg_ops.tridiagonal_solve(diagonals, rhs, partial_pivoting,
                                      perturb_singular, name)


@tf_export('linalg.tridiagonal_matmul')
@dispatch.add_dispatch_support
def tridiagonal_matmul(diagonals, rhs, diagonals_format='compact', name=None):
  r"""Multiplies tridiagonal matrix by matrix.

  `diagonals` is representation of 3-diagonal NxN matrix, which depends on
  `diagonals_format`.

  In `matrix` format, `diagonals` must be a tensor of shape `[..., M, M]`, with
  two inner-most dimensions representing the square tridiagonal matrices.
  Elements outside of the three diagonals will be ignored.

  If `sequence` format, `diagonals` is list or tuple of three tensors:
  `[superdiag, maindiag, subdiag]`, each having shape [..., M]. Last element
  of `superdiag` first element of `subdiag` are ignored.

  In `compact` format the three diagonals are brought together into one tensor
  of shape `[..., 3, M]`, with last two dimensions containing superdiagonals,
  diagonals, and subdiagonals, in order. Similarly to `sequence` format,
  elements `diagonals[..., 0, M-1]` and `diagonals[..., 2, 0]` are ignored.

  The `sequence` format is recommended as the one with the best performance.

  `rhs` is matrix to the right of multiplication. It has shape `[..., M, N]`.

  Example:

  ```python
  superdiag = tf.constant([-1, -1, 0], dtype=tf.float64)
  maindiag = tf.constant([2, 2, 2], dtype=tf.float64)
  subdiag = tf.constant([0, -1, -1], dtype=tf.float64)
  diagonals = [superdiag, maindiag, subdiag]
  rhs = tf.constant([[1, 1], [1, 1], [1, 1]], dtype=tf.float64)
  x = tf.linalg.tridiagonal_matmul(diagonals, rhs, diagonals_format='sequence')
  ```

  Args:
    diagonals: A `Tensor` or tuple of `Tensor`s describing left-hand sides. The
      shape depends of `diagonals_format`, see description above. Must be
      `float32`, `float64`, `complex64`, or `complex128`.
    rhs: A `Tensor` of shape [..., M, N] and with the same dtype as `diagonals`.
    diagonals_format: one of `sequence`, or `compact`. Default is `compact`.
    name:  A name to give this `Op` (optional).

  Returns:
    A `Tensor` of shape [..., M, N] containing the result of multiplication.

  Raises:
    ValueError: An unsupported type is provided as input, or when the input
    tensors have incorrect shapes.
  """
  if diagonals_format == 'compact':
    superdiag = diagonals[..., 0, :]
    maindiag = diagonals[..., 1, :]
    subdiag = diagonals[..., 2, :]
  elif diagonals_format == 'sequence':
    superdiag, maindiag, subdiag = diagonals
  elif diagonals_format == 'matrix':
    m1 = tensor_shape.dimension_value(diagonals.shape[-1])
    m2 = tensor_shape.dimension_value(diagonals.shape[-2])
    if m1 and m2 and m1 != m2:
      raise ValueError(
          'Expected last two dimensions of diagonals to be same, got {} and {}'
          .format(m1, m2))
    diags = array_ops.matrix_diag_part(
        diagonals, k=(-1, 1), padding_value=0., align='LEFT_RIGHT')
    superdiag = diags[..., 0, :]
    maindiag = diags[..., 1, :]
    subdiag = diags[..., 2, :]
  else:
    raise ValueError('Unrecognized diagonals_format: %s' % diagonals_format)

  # C++ backend requires matrices.
  # Converting 1-dimensional vectors to matrices with 1 row.
  superdiag = array_ops.expand_dims(superdiag, -2)
  maindiag = array_ops.expand_dims(maindiag, -2)
  subdiag = array_ops.expand_dims(subdiag, -2)

  return linalg_ops.tridiagonal_mat_mul(superdiag, maindiag, subdiag, rhs, name)


def _maybe_validate_matrix(a, validate_args):
  """Checks that input is a `float` matrix."""
  assertions = []
  if not a.dtype.is_floating:
    raise TypeError('Input `a` must have `float`-like `dtype` '
                    '(saw {}).'.format(a.dtype.name))
  if a.shape is not None and a.shape.rank is not None:
    if a.shape.rank < 2:
      raise ValueError('Input `a` must have at least 2 dimensions '
                       '(saw: {}).'.format(a.shape.rank))
  elif validate_args:
    assertions.append(
        check_ops.assert_rank_at_least(
            a, rank=2, message='Input `a` must have at least 2 dimensions.'))
  return assertions


@tf_export('linalg.matrix_rank')
@dispatch.add_dispatch_support
def matrix_rank(a, tol=None, validate_args=False, name=None):
  """Compute the matrix rank of one or more matrices.

  Args:
    a: (Batch of) `float`-like matrix-shaped `Tensor`(s) which are to be
      pseudo-inverted.
    tol: Threshold below which the singular value is counted as 'zero'.
      Default value: `None` (i.e., `eps * max(rows, cols) * max(singular_val)`).
    validate_args: When `True`, additional assertions might be embedded in the
      graph.
      Default value: `False` (i.e., no graph assertions are added).
    name: Python `str` prefixed to ops created by this function.
      Default value: 'matrix_rank'.

  Returns:
    matrix_rank: (Batch of) `int32` scalars representing the number of non-zero
      singular values.
  """
  with ops.name_scope(name or 'matrix_rank'):
    a = ops.convert_to_tensor(a, dtype_hint=dtypes.float32, name='a')
    assertions = _maybe_validate_matrix(a, validate_args)
    if assertions:
      with ops.control_dependencies(assertions):
        a = array_ops.identity(a)
    s = svd(a, compute_uv=False)
    if tol is None:
      if (a.shape[-2:]).is_fully_defined():
        m = np.max(a.shape[-2:].as_list())
      else:
        m = math_ops.reduce_max(array_ops.shape(a)[-2:])
      eps = np.finfo(a.dtype.as_numpy_dtype).eps
      tol = (
          eps * math_ops.cast(m, a.dtype) *
          math_ops.reduce_max(s, axis=-1, keepdims=True))
    return math_ops.reduce_sum(math_ops.cast(s > tol, dtypes.int32), axis=-1)


@tf_export('linalg.pinv')
@dispatch.add_dispatch_support
def pinv(a, rcond=None, validate_args=False, name=None):
  """Compute the Moore-Penrose pseudo-inverse of one or more matrices.

  Calculate the [generalized inverse of a matrix](
  https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) using its
  singular-value decomposition (SVD) and including all large singular values.

  The pseudo-inverse of a matrix `A`, is defined as: 'the matrix that 'solves'
  [the least-squares problem] `A @ x = b`,' i.e., if `x_hat` is a solution, then
  `A_pinv` is the matrix such that `x_hat = A_pinv @ b`. It can be shown that if
  `U @ Sigma @ V.T = A` is the singular value decomposition of `A`, then
  `A_pinv = V @ inv(Sigma) U^T`. [(Strang, 1980)][1]

  This function is analogous to [`numpy.linalg.pinv`](
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html).
  It differs only in default value of `rcond`. In `numpy.linalg.pinv`, the
  default `rcond` is `1e-15`. Here the default is
  `10. * max(num_rows, num_cols) * np.finfo(dtype).eps`.

  Args:
    a: (Batch of) `float`-like matrix-shaped `Tensor`(s) which are to be
      pseudo-inverted.
    rcond: `Tensor` of small singular value cutoffs.  Singular values smaller
      (in modulus) than `rcond` * largest_singular_value (again, in modulus) are
      set to zero. Must broadcast against `tf.shape(a)[:-2]`.
      Default value: `10. * max(num_rows, num_cols) * np.finfo(a.dtype).eps`.
    validate_args: When `True`, additional assertions might be embedded in the
      graph.
      Default value: `False` (i.e., no graph assertions are added).
    name: Python `str` prefixed to ops created by this function.
      Default value: 'pinv'.

  Returns:
    a_pinv: (Batch of) pseudo-inverse of input `a`. Has same shape as `a` except
      rightmost two dimensions are transposed.

  Raises:
    TypeError: if input `a` does not have `float`-like `dtype`.
    ValueError: if input `a` has fewer than 2 dimensions.

  #### Examples

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp

  a = tf.constant([[1.,  0.4,  0.5],
                   [0.4, 0.2,  0.25],
                   [0.5, 0.25, 0.35]])
  tf.matmul(tf.linalg.pinv(a), a)
  # ==> array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]], dtype=float32)

  a = tf.constant([[1.,  0.4,  0.5,  1.],
                   [0.4, 0.2,  0.25, 2.],
                   [0.5, 0.25, 0.35, 3.]])
  tf.matmul(tf.linalg.pinv(a), a)
  # ==> array([[ 0.76,  0.37,  0.21, -0.02],
               [ 0.37,  0.43, -0.33,  0.02],
               [ 0.21, -0.33,  0.81,  0.01],
               [-0.02,  0.02,  0.01,  1.  ]], dtype=float32)
  ```

  #### References

  [1]: G. Strang. 'Linear Algebra and Its Applications, 2nd Ed.' Academic Press,
       Inc., 1980, pp. 139-142.
  """
  with ops.name_scope(name or 'pinv'):
    a = ops.convert_to_tensor(a, name='a')

    assertions = _maybe_validate_matrix(a, validate_args)
    if assertions:
      with ops.control_dependencies(assertions):
        a = array_ops.identity(a)

    dtype = a.dtype.as_numpy_dtype

    if rcond is None:

      def get_dim_size(dim):
        dim_val = tensor_shape.dimension_value(a.shape[dim])
        if dim_val is not None:
          return dim_val
        return array_ops.shape(a)[dim]

      num_rows = get_dim_size(-2)
      num_cols = get_dim_size(-1)
      if isinstance(num_rows, int) and isinstance(num_cols, int):
        max_rows_cols = float(max(num_rows, num_cols))
      else:
        max_rows_cols = math_ops.cast(
            math_ops.maximum(num_rows, num_cols), dtype)
      rcond = 10. * max_rows_cols * np.finfo(dtype).eps

    rcond = ops.convert_to_tensor(rcond, dtype=dtype, name='rcond')

    # Calculate pseudo inverse via SVD.
    # Note: if a is Hermitian then u == v. (We might observe additional
    # performance by explicitly setting `v = u` in such cases.)
    [
        singular_values,  # Sigma
        left_singular_vectors,  # U
        right_singular_vectors,  # V
    ] = svd(
        a, full_matrices=False, compute_uv=True)

    # Saturate small singular values to inf. This has the effect of make
    # `1. / s = 0.` while not resulting in `NaN` gradients.
    cutoff = rcond * math_ops.reduce_max(singular_values, axis=-1)
    singular_values = array_ops.where_v2(
        singular_values > array_ops.expand_dims_v2(cutoff, -1), singular_values,
        np.array(np.inf, dtype))

    # By the definition of the SVD, `a == u @ s @ v^H`, and the pseudo-inverse
    # is defined as `pinv(a) == v @ inv(s) @ u^H`.
    a_pinv = math_ops.matmul(
        right_singular_vectors / array_ops.expand_dims_v2(singular_values, -2),
        left_singular_vectors,
        adjoint_b=True)

    if a.shape is not None and a.shape.rank is not None:
      a_pinv.set_shape(a.shape[:-2].concatenate([a.shape[-1], a.shape[-2]]))

    return a_pinv


@tf_export('linalg.lu_solve')
@dispatch.add_dispatch_support
def lu_solve(lower_upper, perm, rhs, validate_args=False, name=None):
  """Solves systems of linear eqns `A X = RHS`, given LU factorizations.

  Note: this function does not verify the implied matrix is actually invertible
  nor is this condition checked even when `validate_args=True`.

  Args:
    lower_upper: `lu` as returned by `tf.linalg.lu`, i.e., if `matmul(P,
      matmul(L, U)) = X` then `lower_upper = L + U - eye`.
    perm: `p` as returned by `tf.linag.lu`, i.e., if `matmul(P, matmul(L, U)) =
      X` then `perm = argmax(P)`.
    rhs: Matrix-shaped float `Tensor` representing targets for which to solve;
      `A X = RHS`. To handle vector cases, use: `lu_solve(..., rhs[...,
        tf.newaxis])[..., 0]`.
    validate_args: Python `bool` indicating whether arguments should be checked
      for correctness. Note: this function does not verify the implied matrix is
        actually invertible, even when `validate_args=True`.
      Default value: `False` (i.e., don't validate arguments).
    name: Python `str` name given to ops managed by this object.
      Default value: `None` (i.e., 'lu_solve').

  Returns:
    x: The `X` in `A @ X = RHS`.

  #### Examples

  ```python
  import numpy as np
  import tensorflow as tf
  import tensorflow_probability as tfp

  x = [[[1., 2],
        [3, 4]],
       [[7, 8],
        [3, 4]]]
  inv_x = tf.linalg.lu_solve(*tf.linalg.lu(x), rhs=tf.eye(2))
  tf.assert_near(tf.matrix_inverse(x), inv_x)
  # ==> True
  ```

  """

  with ops.name_scope(name or 'lu_solve'):
    lower_upper = ops.convert_to_tensor(
        lower_upper, dtype_hint=dtypes.float32, name='lower_upper')
    perm = ops.convert_to_tensor(perm, dtype_hint=dtypes.int32, name='perm')
    rhs = ops.convert_to_tensor(rhs, dtype_hint=lower_upper.dtype, name='rhs')

    assertions = _lu_solve_assertions(lower_upper, perm, rhs, validate_args)
    if assertions:
      with ops.control_dependencies(assertions):
        lower_upper = array_ops.identity(lower_upper)
        perm = array_ops.identity(perm)
        rhs = array_ops.identity(rhs)

    if (rhs.shape.rank == 2 and perm.shape.rank == 1):
      # Both rhs and perm have scalar batch_shape.
      permuted_rhs = array_ops.gather(rhs, perm, axis=-2)
    else:
      # Either rhs or perm have non-scalar batch_shape or we can't determine
      # this information statically.
      rhs_shape = array_ops.shape(rhs)
      broadcast_batch_shape = array_ops.broadcast_dynamic_shape(
          rhs_shape[:-2],
          array_ops.shape(perm)[:-1])
      d, m = rhs_shape[-2], rhs_shape[-1]
      rhs_broadcast_shape = array_ops.concat([broadcast_batch_shape, [d, m]],
                                             axis=0)

      # Tile out rhs.
      broadcast_rhs = array_ops.broadcast_to(rhs, rhs_broadcast_shape)
      broadcast_rhs = array_ops.reshape(broadcast_rhs, [-1, d, m])

      # Tile out perm and add batch indices.
      broadcast_perm = array_ops.broadcast_to(perm, rhs_broadcast_shape[:-1])
      broadcast_perm = array_ops.reshape(broadcast_perm, [-1, d])
      broadcast_batch_size = math_ops.reduce_prod(broadcast_batch_shape)
      broadcast_batch_indices = array_ops.broadcast_to(
          math_ops.range(broadcast_batch_size)[:, array_ops.newaxis],
          [broadcast_batch_size, d])
      broadcast_perm = array_ops_stack.stack(
          [broadcast_batch_indices, broadcast_perm], axis=-1)

      permuted_rhs = array_ops.gather_nd(broadcast_rhs, broadcast_perm)
      permuted_rhs = array_ops.reshape(permuted_rhs, rhs_broadcast_shape)

    lower = set_diag(
        band_part(lower_upper, num_lower=-1, num_upper=0),
        array_ops.ones(
            array_ops.shape(lower_upper)[:-1], dtype=lower_upper.dtype))
    return triangular_solve(
        lower_upper,  # Only upper is accessed.
        triangular_solve(lower, permuted_rhs),
        lower=False)


@tf_export('linalg.lu_matrix_inverse')
@dispatch.add_dispatch_support
def lu_matrix_inverse(lower_upper, perm, validate_args=False, name=None):
  """Computes the inverse given the LU decomposition(s) of one or more matrices.

  This op is conceptually identical to,

  ```python
  inv_X = tf.lu_matrix_inverse(*tf.linalg.lu(X))
  tf.assert_near(tf.matrix_inverse(X), inv_X)
  # ==> True
  ```

  Note: this function does not verify the implied matrix is actually invertible
  nor is this condition checked even when `validate_args=True`.

  Args:
    lower_upper: `lu` as returned by `tf.linalg.lu`, i.e., if `matmul(P,
      matmul(L, U)) = X` then `lower_upper = L + U - eye`.
    perm: `p` as returned by `tf.linag.lu`, i.e., if `matmul(P, matmul(L, U)) =
      X` then `perm = argmax(P)`.
    validate_args: Python `bool` indicating whether arguments should be checked
      for correctness. Note: this function does not verify the implied matrix is
        actually invertible, even when `validate_args=True`.
      Default value: `False` (i.e., don't validate arguments).
    name: Python `str` name given to ops managed by this object.
      Default value: `None` (i.e., 'lu_matrix_inverse').

  Returns:
    inv_x: The matrix_inv, i.e.,
      `tf.matrix_inverse(tf.linalg.lu_reconstruct(lu, perm))`.

  #### Examples

  ```python
  import numpy as np
  import tensorflow as tf
  import tensorflow_probability as tfp

  x = [[[3., 4], [1, 2]],
       [[7., 8], [3, 4]]]
  inv_x = tf.linalg.lu_matrix_inverse(*tf.linalg.lu(x))
  tf.assert_near(tf.matrix_inverse(x), inv_x)
  # ==> True
  ```

  """

  with ops.name_scope(name or 'lu_matrix_inverse'):
    lower_upper = ops.convert_to_tensor(
        lower_upper, dtype_hint=dtypes.float32, name='lower_upper')
    perm = ops.convert_to_tensor(perm, dtype_hint=dtypes.int32, name='perm')
    assertions = lu_reconstruct_assertions(lower_upper, perm, validate_args)
    if assertions:
      with ops.control_dependencies(assertions):
        lower_upper = array_ops.identity(lower_upper)
        perm = array_ops.identity(perm)
    shape = array_ops.shape(lower_upper)
    return lu_solve(
        lower_upper,
        perm,
        rhs=eye(shape[-1], batch_shape=shape[:-2], dtype=lower_upper.dtype),
        validate_args=False)


@tf_export('linalg.lu_reconstruct')
@dispatch.add_dispatch_support
def lu_reconstruct(lower_upper, perm, validate_args=False, name=None):
  """The reconstruct one or more matrices from their LU decomposition(s).

  Args:
    lower_upper: `lu` as returned by `tf.linalg.lu`, i.e., if `matmul(P,
      matmul(L, U)) = X` then `lower_upper = L + U - eye`.
    perm: `p` as returned by `tf.linag.lu`, i.e., if `matmul(P, matmul(L, U)) =
      X` then `perm = argmax(P)`.
    validate_args: Python `bool` indicating whether arguments should be checked
      for correctness.
      Default value: `False` (i.e., don't validate arguments).
    name: Python `str` name given to ops managed by this object.
      Default value: `None` (i.e., 'lu_reconstruct').

  Returns:
    x: The original input to `tf.linalg.lu`, i.e., `x` as in,
      `lu_reconstruct(*tf.linalg.lu(x))`.

  #### Examples

  ```python
  import numpy as np
  import tensorflow as tf
  import tensorflow_probability as tfp

  x = [[[3., 4], [1, 2]],
       [[7., 8], [3, 4]]]
  x_reconstructed = tf.linalg.lu_reconstruct(*tf.linalg.lu(x))
  tf.assert_near(x, x_reconstructed)
  # ==> True
  ```

  """
  with ops.name_scope(name or 'lu_reconstruct'):
    lower_upper = ops.convert_to_tensor(
        lower_upper, dtype_hint=dtypes.float32, name='lower_upper')
    perm = ops.convert_to_tensor(perm, dtype_hint=dtypes.int32, name='perm')

    assertions = lu_reconstruct_assertions(lower_upper, perm, validate_args)
    if assertions:
      with ops.control_dependencies(assertions):
        lower_upper = array_ops.identity(lower_upper)
        perm = array_ops.identity(perm)

    shape = array_ops.shape(lower_upper)

    lower = set_diag(
        band_part(lower_upper, num_lower=-1, num_upper=0),
        array_ops.ones(shape[:-1], dtype=lower_upper.dtype))
    upper = band_part(lower_upper, num_lower=0, num_upper=-1)
    x = math_ops.matmul(lower, upper)

    if (lower_upper.shape is None or lower_upper.shape.rank is None or
        lower_upper.shape.rank != 2):
      # We either don't know the batch rank or there are >0 batch dims.
      batch_size = math_ops.reduce_prod(shape[:-2])
      d = shape[-1]
      x = array_ops.reshape(x, [batch_size, d, d])
      perm = array_ops.reshape(perm, [batch_size, d])
      perm = map_fn.map_fn(array_ops.invert_permutation, perm)
      batch_indices = array_ops.broadcast_to(
          math_ops.range(batch_size)[:, array_ops.newaxis], [batch_size, d])
      x = array_ops.gather_nd(
          x, array_ops_stack.stack([batch_indices, perm], axis=-1))
      x = array_ops.reshape(x, shape)
    else:
      x = array_ops.gather(x, array_ops.invert_permutation(perm))

    x.set_shape(lower_upper.shape)
    return x


def lu_reconstruct_assertions(lower_upper, perm, validate_args):
  """Returns list of assertions related to `lu_reconstruct` assumptions."""
  assertions = []

  message = 'Input `lower_upper` must have at least 2 dimensions.'
  if lower_upper.shape.rank is not None and lower_upper.shape.rank < 2:
    raise ValueError(message)
  elif validate_args:
    assertions.append(
        check_ops.assert_rank_at_least_v2(lower_upper, rank=2, message=message))

  message = '`rank(lower_upper)` must equal `rank(perm) + 1`'
  if lower_upper.shape.rank is not None and perm.shape.rank is not None:
    if lower_upper.shape.rank != perm.shape.rank + 1:
      raise ValueError(message)
  elif validate_args:
    assertions.append(
        check_ops.assert_rank(
            lower_upper, rank=array_ops.rank(perm) + 1, message=message))

  message = '`lower_upper` must be square.'
  if lower_upper.shape[:-2].is_fully_defined():
    if lower_upper.shape[-2] != lower_upper.shape[-1]:
      raise ValueError(message)
  elif validate_args:
    m, n = array_ops.split(
        array_ops.shape(lower_upper)[-2:], num_or_size_splits=2)
    assertions.append(check_ops.assert_equal(m, n, message=message))

  return assertions


def _lu_solve_assertions(lower_upper, perm, rhs, validate_args):
  """Returns list of assertions related to `lu_solve` assumptions."""
  assertions = lu_reconstruct_assertions(lower_upper, perm, validate_args)

  message = 'Input `rhs` must have at least 2 dimensions.'
  if rhs.shape.ndims is not None:
    if rhs.shape.ndims < 2:
      raise ValueError(message)
  elif validate_args:
    assertions.append(
        check_ops.assert_rank_at_least(rhs, rank=2, message=message))

  message = '`lower_upper.shape[-1]` must equal `rhs.shape[-1]`.'
  if (lower_upper.shape[-1] is not None and rhs.shape[-2] is not None):
    if lower_upper.shape[-1] != rhs.shape[-2]:
      raise ValueError(message)
  elif validate_args:
    assertions.append(
        check_ops.assert_equal(
            array_ops.shape(lower_upper)[-1],
            array_ops.shape(rhs)[-2],
            message=message))

  return assertions


@tf_export('linalg.eigh_tridiagonal')
@dispatch.add_dispatch_support
def eigh_tridiagonal(alpha,
                     beta,
                     eigvals_only=True,
                     select='a',
                     select_range=None,
                     tol=None,
                     name=None):
  """Computes the eigenvalues of a Hermitian tridiagonal matrix.

  Args:
    alpha: A real or complex tensor of shape (n), the diagonal elements of the
      matrix. NOTE: If alpha is complex, the imaginary part is ignored (assumed
        zero) to satisfy the requirement that the matrix be Hermitian.
    beta: A real or complex tensor of shape (n-1), containing the elements of
      the first super-diagonal of the matrix. If beta is complex, the first
      sub-diagonal of the matrix is assumed to be the conjugate of beta to
      satisfy the requirement that the matrix be Hermitian
    eigvals_only: If False, both eigenvalues and corresponding eigenvectors are
      computed. If True, only eigenvalues are computed. Default is True.
    select: Optional string with values in {‘a’, ‘v’, ‘i’} (default is 'a') that
      determines which eigenvalues to calculate:
        'a': all eigenvalues.
        ‘v’: eigenvalues in the interval (min, max] given by `select_range`.
        'i’: eigenvalues with indices min <= i <= max.
    select_range: Size 2 tuple or list or tensor specifying the range of
      eigenvalues to compute together with select. If select is 'a',
      select_range is ignored.
    tol: Optional scalar. The absolute tolerance to which each eigenvalue is
      required. An eigenvalue (or cluster) is considered to have converged if it
      lies in an interval of this width. If tol is None (default), the value
      eps*|T|_2 is used where eps is the machine precision, and |T|_2 is the
      2-norm of the matrix T.
    name: Optional name of the op.

  Returns:
    eig_vals: The eigenvalues of the matrix in non-decreasing order.
    eig_vectors: If `eigvals_only` is False the eigenvectors are returned in
      the second output argument.

  Raises:
     ValueError: If input values are invalid.
     NotImplemented: Computing eigenvectors for `eigvals_only` = False is
       not implemented yet.

  This op implements a subset of the functionality of
  scipy.linalg.eigh_tridiagonal.

  Note: The result is undefined if the input contains +/-inf or NaN, or if
  any value in beta has a magnitude greater than
  `numpy.sqrt(numpy.finfo(beta.dtype.as_numpy_dtype).max)`.


  TODO(b/187527398):
    Add support for outer batch dimensions.

  #### Examples

  ```python
  import numpy
  eigvals = tf.linalg.eigh_tridiagonal([0.0, 0.0, 0.0], [1.0, 1.0])
  eigvals_expected = [-numpy.sqrt(2.0), 0.0, numpy.sqrt(2.0)]
  tf.assert_near(eigvals_expected, eigvals)
  # ==> True
  ```

  """
  with ops.name_scope(name or 'eigh_tridiagonal'):

    def _compute_eigenvalues(alpha, beta):
      """Computes all eigenvalues of a Hermitian tridiagonal matrix."""

      def _sturm(alpha, beta_sq, pivmin, alpha0_perturbation, x):
        """Implements the Sturm sequence recurrence."""
        with ops.name_scope('sturm'):
          n = alpha.shape[0]
          zeros = array_ops.zeros(array_ops.shape(x), dtype=dtypes.int32)
          ones = array_ops.ones(array_ops.shape(x), dtype=dtypes.int32)

          # The first step in the Sturm sequence recurrence
          # requires special care if x is equal to alpha[0].
          def sturm_step0():
            q = alpha[0] - x
            count = array_ops.where(q < 0, ones, zeros)
            q = array_ops.where(
                math_ops.equal(alpha[0], x), alpha0_perturbation, q)
            return q, count

          # Subsequent steps all take this form:
          def sturm_step(i, q, count):
            q = alpha[i] - beta_sq[i - 1] / q - x
            count = array_ops.where(q <= pivmin, count + 1, count)
            q = array_ops.where(q <= pivmin, math_ops.minimum(q, -pivmin), q)
            return q, count

          # The first step initializes q and count.
          q, count = sturm_step0()

          # Peel off ((n-1) % blocksize) steps from the main loop, so we can run
          # the bulk of the iterations unrolled by a factor of blocksize.
          blocksize = 16
          i = 1
          peel = (n - 1) % blocksize
          unroll_cnt = peel

          def unrolled_steps(start, q, count):
            for j in range(unroll_cnt):
              q, count = sturm_step(start + j, q, count)
            return start + unroll_cnt, q, count

          i, q, count = unrolled_steps(i, q, count)

          # Run the remaining steps of the Sturm sequence using a partially
          # unrolled while loop.
          unroll_cnt = blocksize
          cond = lambda i, q, count: math_ops.less(i, n)
          _, _, count = control_flow_ops.while_loop(
              cond, unrolled_steps, [i, q, count], back_prop=False)
          return count

      with ops.name_scope('compute_eigenvalues'):
        if alpha.dtype.is_complex:
          alpha = math_ops.real(alpha)
          beta_sq = math_ops.real(math_ops.conj(beta) * beta)
          beta_abs = math_ops.sqrt(beta_sq)
        else:
          beta_sq = math_ops.square(beta)
          beta_abs = math_ops.abs(beta)

        # Estimate the largest and smallest eigenvalues of T using the
        # Gershgorin circle theorem.
        finfo = np.finfo(alpha.dtype.as_numpy_dtype)
        off_diag_abs_row_sum = array_ops.concat(
            [beta_abs[:1], beta_abs[:-1] + beta_abs[1:], beta_abs[-1:]], axis=0)
        lambda_est_max = math_ops.minimum(
            finfo.max, math_ops.reduce_max(alpha + off_diag_abs_row_sum))
        lambda_est_min = math_ops.maximum(
            finfo.min, math_ops.reduce_min(alpha - off_diag_abs_row_sum))
        # Upper bound on 2-norm of T.
        t_norm = math_ops.maximum(
            math_ops.abs(lambda_est_min), math_ops.abs(lambda_est_max))

        # Compute the smallest allowed pivot in the Sturm sequence to avoid
        # overflow.
        one = np.ones([], dtype=alpha.dtype.as_numpy_dtype)
        safemin = np.maximum(one / finfo.max, (one + finfo.eps) * finfo.tiny)
        pivmin = safemin * math_ops.maximum(one, math_ops.reduce_max(beta_sq))
        alpha0_perturbation = math_ops.square(finfo.eps * beta_abs[0])
        abs_tol = finfo.eps * t_norm
        if tol:
          abs_tol = math_ops.maximum(tol, abs_tol)
        # In the worst case, when the absolute tolerance is eps*lambda_est_max
        # and lambda_est_max = -lambda_est_min, we have to take as many
        # bisection steps as there are bits in the mantissa plus 1.
        max_it = finfo.nmant + 1

        # Determine the indices of the desired eigenvalues, based on select
        # and select_range.
        asserts = None
        if select == 'a':
          target_counts = math_ops.range(n)
        elif select == 'i':
          asserts = check_ops.assert_less_equal(
              select_range[0],
              select_range[1],
              message='Got empty index range in select_range.')
          target_counts = math_ops.range(select_range[0], select_range[1] + 1)
        elif select == 'v':
          asserts = check_ops.assert_less(
              select_range[0],
              select_range[1],
              message='Got empty interval in select_range.')
        else:
          raise ValueError("'select must have a value in {'a', 'i', 'v'}.")

        if asserts:
          with ops.control_dependencies([asserts]):
            alpha = array_ops.identity(alpha)

        # Run binary search for all desired eigenvalues in parallel, starting
        # from  an interval slightly wider than the estimated
        # [lambda_est_min, lambda_est_max].
        fudge = 2.1  # We widen starting interval the Gershgorin interval a bit.
        norm_slack = math_ops.cast(n, alpha.dtype) * fudge * finfo.eps * t_norm
        if select in {'a', 'i'}:
          lower = lambda_est_min - norm_slack - 2 * fudge * pivmin
          upper = lambda_est_max + norm_slack + fudge * pivmin
        else:
          # Count the number of eigenvalues in the given range.
          lower = select_range[0] - norm_slack - 2 * fudge * pivmin
          upper = select_range[1] + norm_slack + fudge * pivmin
          first = _sturm(alpha, beta_sq, pivmin, alpha0_perturbation, lower)
          last = _sturm(alpha, beta_sq, pivmin, alpha0_perturbation, upper)
          target_counts = math_ops.range(first, last)

        # Pre-broadcast the scalars used in the Sturm sequence for improved
        # performance.
        upper = math_ops.minimum(upper, finfo.max)
        lower = math_ops.maximum(lower, finfo.min)
        target_shape = array_ops.shape(target_counts)
        lower = array_ops.broadcast_to(lower, shape=target_shape)
        upper = array_ops.broadcast_to(upper, shape=target_shape)
        pivmin = array_ops.broadcast_to(pivmin, target_shape)
        alpha0_perturbation = array_ops.broadcast_to(alpha0_perturbation,
                                                     target_shape)

        # We compute the midpoint as 0.5*lower + 0.5*upper to avoid overflow in
        # (lower + upper) or (upper - lower) when the matrix has eigenvalues
        # with magnitude greater than finfo.max / 2.
        def midpoint(lower, upper):
          return (0.5 * lower) + (0.5 * upper)

        def continue_binary_search(i, lower, upper):
          return math_ops.logical_and(
              math_ops.less(i, max_it),
              math_ops.less(abs_tol, math_ops.reduce_max(upper - lower)))

        def binary_search_step(i, lower, upper):
          mid = midpoint(lower, upper)
          counts = _sturm(alpha, beta_sq, pivmin, alpha0_perturbation, mid)
          lower = array_ops.where(counts <= target_counts, mid, lower)
          upper = array_ops.where(counts > target_counts, mid, upper)
          return i + 1, lower, upper

        # Start parallel binary searches.
        _, lower, upper = control_flow_ops.while_loop(continue_binary_search,
                                                      binary_search_step,
                                                      [0, lower, upper])
        return midpoint(lower, upper)

    def _compute_eigenvectors(alpha, beta, eigvals):
      """Implements inverse iteration to compute eigenvectors."""
      with ops.name_scope('compute_eigenvectors'):
        k = array_ops.size(eigvals)
        n = array_ops.size(alpha)
        alpha = math_ops.cast(alpha, dtype=beta.dtype)

        # Eigenvectors corresponding to cluster of close eigenvalues are
        # not unique and need to be explicitly orthogonalized. Here we
        # identify such clusters. Note: This function assumes that
        # eigenvalues are sorted in non-decreasing order.
        gap = eigvals[1:] - eigvals[:-1]
        eps = np.finfo(eigvals.dtype.as_numpy_dtype).eps
        t_norm = math_ops.maximum(
            math_ops.abs(eigvals[0]), math_ops.abs(eigvals[-1]))
        gaptol = np.sqrt(eps) * t_norm
        # Find the beginning and end of runs of eigenvectors corresponding
        # to eigenvalues closer than "gaptol", which will need to be
        # orthogonalized against each other.
        close = math_ops.less(gap, gaptol)
        left_neighbor_close = array_ops.concat([[False], close], axis=0)
        right_neighbor_close = array_ops.concat([close, [False]], axis=0)
        ortho_interval_start = math_ops.logical_and(
            math_ops.logical_not(left_neighbor_close), right_neighbor_close)
        ortho_interval_start = array_ops.squeeze(
            array_ops.where_v2(ortho_interval_start), axis=-1)
        ortho_interval_end = math_ops.logical_and(
            left_neighbor_close, math_ops.logical_not(right_neighbor_close))
        ortho_interval_end = array_ops.squeeze(
            array_ops.where_v2(ortho_interval_end), axis=-1) + 1
        num_clusters = array_ops.size(ortho_interval_end)

        # We perform inverse iteration for all eigenvectors in parallel,
        # starting from a random set of vectors, until all have converged.
        v0 = math_ops.cast(
            stateless_random_ops.stateless_random_normal(
                shape=(k, n), seed=[7, 42]),
            dtype=beta.dtype)
        nrm_v = norm(v0, axis=1)
        v0 = v0 / nrm_v[:, array_ops.newaxis]
        zero_nrm = constant_op.constant(0, shape=nrm_v.shape, dtype=nrm_v.dtype)

        # Replicate alpha-eigvals(ik) and beta across the k eigenvectors so we
        # can solve the k systems
        #    [T - eigvals(i)*eye(n)] x_i = r_i
        # simultaneously using the batching mechanism.
        eigvals_cast = math_ops.cast(eigvals, dtype=beta.dtype)
        alpha_shifted = (
            alpha[array_ops.newaxis, :] - eigvals_cast[:, array_ops.newaxis])
        beta = array_ops.tile(beta[array_ops.newaxis, :], [k, 1])
        diags = [beta, alpha_shifted, math_ops.conj(beta)]

        def orthogonalize_close_eigenvectors(eigenvectors):
          # Eigenvectors corresponding to a cluster of close eigenvalues are not
          # uniquely defined, but the subspace they span is. To avoid numerical
          # instability, we explicitly mutually orthogonalize such eigenvectors
          # after each step of inverse iteration. It is customary to use
          # modified Gram-Schmidt for this, but this is not very efficient
          # on some platforms, so here we defer to the QR decomposition in
          # TensorFlow.
          def orthogonalize_cluster(cluster_idx, eigenvectors):
            start = ortho_interval_start[cluster_idx]
            end = ortho_interval_end[cluster_idx]
            update_indices = array_ops.expand_dims(
                math_ops.range(start, end), -1)
            vectors_in_cluster = eigenvectors[start:end, :]
            # We use the builtin QR factorization to orthonormalize the
            # vectors in the cluster.
            q, _ = qr(transpose(vectors_in_cluster))
            vectors_to_update = transpose(q)
            eigenvectors = array_ops.tensor_scatter_nd_update(
                eigenvectors, update_indices, vectors_to_update)
            return cluster_idx + 1, eigenvectors

          _, eigenvectors = control_flow_ops.while_loop(
              lambda i, ev: math_ops.less(i, num_clusters),
              orthogonalize_cluster, [0, eigenvectors])
          return eigenvectors

        def continue_iteration(i, _, nrm_v, nrm_v_old):
          max_it = 5  # Taken from LAPACK xSTEIN.
          min_norm_growth = 0.1
          norm_growth_factor = constant_op.constant(
              1 + min_norm_growth, dtype=nrm_v.dtype)
          # We stop the inverse iteration when we reach the maximum number of
          # iterations or the norm growths is less than 10%.
          return math_ops.logical_and(
              math_ops.less(i, max_it),
              math_ops.reduce_any(
                  math_ops.greater_equal(
                      math_ops.real(nrm_v),
                      math_ops.real(norm_growth_factor * nrm_v_old))))

        def inverse_iteration_step(i, v, nrm_v, nrm_v_old):
          v = tridiagonal_solve(
              diags,
              v,
              diagonals_format='sequence',
              partial_pivoting=True,
              perturb_singular=True)
          nrm_v_old = nrm_v
          nrm_v = norm(v, axis=1)
          v = v / nrm_v[:, array_ops.newaxis]
          v = orthogonalize_close_eigenvectors(v)
          return i + 1, v, nrm_v, nrm_v_old

        _, v, nrm_v, _ = control_flow_ops.while_loop(continue_iteration,
                                                     inverse_iteration_step,
                                                     [0, v0, nrm_v, zero_nrm])
        return transpose(v)

    alpha = ops.convert_to_tensor(alpha, name='alpha')
    n = alpha.shape[0]
    if n <= 1:
      return math_ops.real(alpha)
    beta = ops.convert_to_tensor(beta, name='beta')

    if alpha.dtype != beta.dtype:
      raise ValueError("'alpha' and 'beta' must have the same type.")

    eigvals = _compute_eigenvalues(alpha, beta)
    if eigvals_only:
      return eigvals

    eigvectors = _compute_eigenvectors(alpha, beta, eigvals)
    return eigvals, eigvectors
