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

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
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
diag = array_ops.matrix_diag
diag_part = array_ops.matrix_diag_part
eigh = linalg_ops.self_adjoint_eig
eigvalsh = linalg_ops.self_adjoint_eigvals
einsum = special_math_ops.einsum
expm = gen_linalg_ops.matrix_exponential
eye = linalg_ops.eye
inv = linalg_ops.matrix_inverse
logm = gen_linalg_ops.matrix_logarithm
lstsq = linalg_ops.matrix_solve_ls
norm = linalg_ops.norm
qr = linalg_ops.qr
set_diag = array_ops.matrix_set_diag
solve = linalg_ops.matrix_solve
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
        reduction_indices=[-1])


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
