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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.compat import compat
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

# Linear algebra ops.
band_part = array_ops.matrix_band_part
cholesky = linalg_ops.cholesky
cholesky_solve = linalg_ops.cholesky_solve
det = linalg_ops.matrix_determinant
slogdet = gen_linalg_ops.log_matrix_determinant
tf_export('linalg.slogdet')(slogdet)
diag = array_ops.matrix_diag
diag_part = array_ops.matrix_diag_part
eigh = linalg_ops.self_adjoint_eig
eigvalsh = linalg_ops.self_adjoint_eigvals
einsum = special_math_ops.einsum
eye = linalg_ops.eye
inv = linalg_ops.matrix_inverse
logm = gen_linalg_ops.matrix_logarithm
lu = gen_linalg_ops.lu
tf_export('linalg.logm')(logm)
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
def matrix_exponential(input, name=None):  # pylint: disable=redefined-builtin
  r"""Computes the matrix exponential of one or more square matrices.

  exp(A) = \sum_{n=0}^\infty A^n/n!

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
        axis=-1)
    const = lambda x: constant_op.constant(x, l1_norm.dtype)

    def _nest_where(vals, cases):
      assert len(vals) == len(cases) - 1
      if len(vals) == 1:
        return array_ops.where(
            math_ops.less(l1_norm, const(vals[0])), cases[0], cases[1])
      else:
        return array_ops.where(
            math_ops.less(l1_norm, const(vals[0])), cases[0],
            _nest_where(vals[1:], cases[1:]))

    if matrix.dtype in [dtypes.float16, dtypes.float32, dtypes.complex64]:
      maxnorm = const(3.925724783138660)
      squarings = math_ops.maximum(
          math_ops.floor(
              math_ops.log(l1_norm / maxnorm) / math_ops.log(const(2.0))), 0)
      u3, v3 = _matrix_exp_pade3(matrix)
      u5, v5 = _matrix_exp_pade5(matrix)
      u7, v7 = _matrix_exp_pade7(matrix / math_ops.cast(
          math_ops.pow(const(2.0), squarings),
          matrix.dtype)[..., array_ops.newaxis, array_ops.newaxis])
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
      u13, v13 = _matrix_exp_pade13(matrix / math_ops.cast(
          math_ops.pow(const(2.0), squarings),
          matrix.dtype)[..., array_ops.newaxis, array_ops.newaxis])
      conds = (1.495585217958292e-002, 2.539398330063230e-001,
               9.504178996162932e-001, 2.097847961257068e+000)
      u = _nest_where(conds, (u3, u5, u7, u9, u13))
      v = _nest_where(conds, (v3, v5, v7, v9, v13))
    else:
      raise ValueError('tf.linalg.expm does not support matrices of type %s' %
                       matrix.dtype)
    numer = u + v
    denom = -u + v
    result = linalg_ops.matrix_solve(denom, numer)
    max_squarings = math_ops.reduce_max(squarings)

    i = const(0.0)
    c = lambda i, r: math_ops.less(i, max_squarings)

    def b(i, r):
      return i + 1, array_ops.where(
          math_ops.less(i, squarings), math_ops.matmul(r, r), r)

    _, result = control_flow_ops.while_loop(c, b, [i, result])
    if not matrix.shape.is_fully_defined():
      return array_ops.reshape(
          result,
          array_ops.concat((batch_shape, array_ops.shape(result)[-2:]), axis=0))
    return array_ops.reshape(result, batch_shape.concatenate(result.shape[-2:]))


@tf_export('linalg.tridiagonal_solve')
def tridiagonal_solve(diagonals,
                      rhs,
                      diagonals_format='compact',
                      transpose_rhs=False,
                      conjugate_rhs=False,
                      name=None,
                      partial_pivoting=True):
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

  Returns:
    A `Tensor` of shape [..., M] or [..., M, K] containing the solutions.

  Raises:
    ValueError: An unsupported type is provided as input, or when the input
    tensors have incorrect shapes.

  [1] Nicholas J. Higham (2002). Accuracy and Stability of Numerical Algorithms:
  Second Edition. SIAM. p. 175. ISBN 978-0-89871-802-7.

  """
  if diagonals_format == 'compact':
    return _tridiagonal_solve_compact_format(diagonals, rhs, transpose_rhs,
                                             conjugate_rhs, partial_pivoting,
                                             name)

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

    diagonals = array_ops.stack((superdiag, maindiag, subdiag), axis=-2)
    return _tridiagonal_solve_compact_format(diagonals, rhs, transpose_rhs,
                                             conjugate_rhs, partial_pivoting,
                                             name)

  if diagonals_format == 'matrix':
    m1 = tensor_shape.dimension_value(diagonals.shape[-1])
    m2 = tensor_shape.dimension_value(diagonals.shape[-2])
    if m1 and m2 and m1 != m2:
      raise ValueError(
          'Expected last two dimensions of diagonals to be same, got {} and {}'
          .format(m1, m2))
    m = m1 or m2
    diagonals = gen_array_ops.matrix_diag_part_v2(
        diagonals, k=(-1, 1), padding_value=0.)
    # matrix_diag_part pads at the end. Because the subdiagonal has the
    # convention of having the padding in the front, we need to rotate the last
    # Tensor.
    superdiag, d, subdiag = array_ops.unstack(diagonals, num=3, axis=-2)
    subdiag = manip_ops.roll(subdiag, shift=1, axis=-1)
    diagonals = array_ops.stack((superdiag, d, subdiag), axis=-2)
    return _tridiagonal_solve_compact_format(
        diagonals, rhs, transpose_rhs, conjugate_rhs, partial_pivoting, name)

  raise ValueError('Unrecognized diagonals_format: {}'.format(diagonals_format))


def _tridiagonal_solve_compact_format(diagonals, rhs, transpose_rhs,
                                      conjugate_rhs, partial_pivoting, name):
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
        linalg_ops.tridiagonal_solve(diagonals, rhs, partial_pivoting, name),
        -1)

  if transpose_rhs:
    rhs = array_ops.matrix_transpose(rhs, conjugate=conjugate_rhs)
  elif conjugate_rhs:
    rhs = math_ops.conj(rhs)

  check_num_lhs_matches_num_rhs()
  result = linalg_ops.tridiagonal_solve(diagonals, rhs, partial_pivoting, name)
  if transpose_rhs and not compat.forward_compatible(2019, 10, 18):
    return array_ops.matrix_transpose(result)
  return result


@tf_export('linalg.tridiagonal_matmul')
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

    maindiag = array_ops.matrix_diag_part(diagonals)
    superdiag = gen_array_ops.matrix_diag_part_v2(
        diagonals, k=1, padding_value=0.)
    superdiag = array_ops.concat(
        [superdiag,
         array_ops.zeros_like(
             superdiag[..., 0])[..., array_ops.newaxis]],
        axis=-1)
    subdiag = gen_array_ops.matrix_diag_part_v2(
        diagonals, k=-1, padding_value=0.)
    subdiag = array_ops.concat([
        array_ops.zeros_like(subdiag[..., 0])[..., array_ops.newaxis],
        subdiag], axis=-1)
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
def matrix_rank(a, tol=None, validate_args=False, name=None):
  """Compute the matrix rank of one or more matrices.

  Arguments:
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
  tf.matmul(tf.linalg..pinv(a), a)
  # ==> array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]], dtype=float32)

  a = tf.constant([[1.,  0.4,  0.5,  1.],
                   [0.4, 0.2,  0.25, 2.],
                   [0.5, 0.25, 0.35, 3.]])
  tf.matmul(tf.linalg..pinv(a), a)
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
      broadcast_perm = array_ops.stack(
          [broadcast_batch_indices, broadcast_perm], axis=-1)

      permuted_rhs = array_ops.gather_nd(broadcast_rhs, broadcast_perm)
      permuted_rhs = array_ops.reshape(permuted_rhs, rhs_broadcast_shape)

    lower = set_diag(
        band_part(lower_upper, num_lower=-1, num_upper=0),
        array_ops.ones(
            array_ops.shape(lower_upper)[:-1], dtype=lower_upper.dtype))
    return linear_operator_util.matrix_triangular_solve_with_broadcast(
        lower_upper,  # Only upper is accessed.
        linear_operator_util.matrix_triangular_solve_with_broadcast(
            lower, permuted_rhs),
        lower=False)


@tf_export('linalg.lu_matrix_inverse')
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
      x = array_ops.gather_nd(x, array_ops.stack([batch_indices, perm],
                                                 axis=-1))
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
