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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
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
def logdet(matrix, name=None):
  """Computes log of the determinant of a hermitian positive definite matrix.

  ```python
  # Compute the determinant of a matrix while reducing the chance of over- or
  underflow:
  A = ... # shape 10 x 10
  det = tf.exp(tf.logdet(A))  # scalar
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
def adjoint(matrix, name=None):
  """Transposes the last two dimensions of and conjugates tensor `matrix`.

  For example:

  ```python
  x = tf.constant([[1 + 1j, 2 + 2j, 3 + 3j],
                   [4 + 4j, 5 + 5j, 6 + 6j]])
  tf.linalg.adjoint(x)  # [[1 - 1j, 4 - 4j],
                        #  [2 - 2j, 5 - 5j],
                        #  [3 - 3j, 6 - 6j]]

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
  ident = linalg_ops.eye(array_ops.shape(matrix)[-2],
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
  ident = linalg_ops.eye(array_ops.shape(matrix)[-2],
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
  ident = linalg_ops.eye(array_ops.shape(matrix)[-2],
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
  ident = linalg_ops.eye(array_ops.shape(matrix)[-2],
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
  ident = linalg_ops.eye(array_ops.shape(matrix)[-2],
                         batch_shape=array_ops.shape(matrix)[:-2],
                         dtype=matrix.dtype)
  matrix_2 = math_ops.matmul(matrix, matrix)
  matrix_4 = math_ops.matmul(matrix_2, matrix_2)
  matrix_6 = math_ops.matmul(matrix_4, matrix_2)
  tmp_u = (
      math_ops.matmul(matrix_6,
                      matrix_6 + b[11] * matrix_4 + b[9] * matrix_2) +
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
    input: A `Tensor`. Must be `float16`, `float32`, `float64`, `complex64`,
      or `complex128` with shape `[..., M, M]`.
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
        math_ops.reduce_sum(math_ops.abs(matrix),
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
      u7, v7 = _matrix_exp_pade7(
          matrix / math_ops.pow(
              constant_op.constant(2.0, dtype=matrix.dtype),
              math_ops.cast(squarings, matrix.dtype))[...,
                                                      array_ops.newaxis,
                                                      array_ops.newaxis])
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
          matrix / math_ops.pow(
              constant_op.constant(2.0, dtype=matrix.dtype),
              math_ops.cast(squarings, matrix.dtype))[...,
                                                      array_ops.newaxis,
                                                      array_ops.newaxis])
      conds = (1.495585217958292e-002,
               2.539398330063230e-001,
               9.504178996162932e-001,
               2.097847961257068e+000)
      u = _nest_where(conds, (u3, u5, u7, u9, u13))
      v = _nest_where(conds, (v3, v5, v7, v9, v13))
    else:
      raise ValueError(
          'tf.linalg.expm does not support matrices of type %s' % matrix.dtype)
    numer = u + v
    denom = -u + v
    result = linalg_ops.matrix_solve(denom, numer)
    max_squarings = math_ops.reduce_max(squarings)

    i = const(0.0)
    c = lambda i, r: math_ops.less(i, max_squarings)
    def b(i, r):
      return i+1, array_ops.where(math_ops.less(i, squarings),
                                  math_ops.matmul(r, r), r)
    _, result = control_flow_ops.while_loop(c, b, [i, result])
    if not matrix.shape.is_fully_defined():
      return array_ops.reshape(
          result,
          array_ops.concat((batch_shape, array_ops.shape(result)[-2:]), axis=0))
    return array_ops.reshape(result, batch_shape.concatenate(result.shape[-2:]))
