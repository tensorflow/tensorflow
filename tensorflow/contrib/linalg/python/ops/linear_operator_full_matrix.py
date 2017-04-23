# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""`LinearOperator` that wraps a [batch] matrix."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.linalg.python.ops import linear_operator
from tensorflow.contrib.linalg.python.ops import linear_operator_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops

__all__ = ["LinearOperatorFullMatrix"]


class LinearOperatorFullMatrix(linear_operator.LinearOperator):
  """`LinearOperator` that wraps a [batch] matrix.

  This operator wraps a [batch] matrix `A` (which is a `Tensor`) with shape
  `[B1,...,Bb, M, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `M x N` matrix.

  ```python
  # Create a 2 x 2 linear operator.
  matrix = [[1., 2.], [3., 4.]]
  operator = LinearOperatorFullMatrix(matrix)

  operator.to_dense()
  ==> [[1., 2.]
       [3., 4.]]

  operator.shape
  ==> [2, 2]

  operator.log_determinant()
  ==> scalar Tensor

  x = ... Shape [2, 4] Tensor
  operator.apply(x)
  ==> Shape [2, 4] Tensor

  # Create a [2, 3] batch of 4 x 4 linear operators.
  matrix = tf.random_normal(shape=[2, 3, 4, 4])
  operator = LinearOperatorFullMatrix(matrix)
  ```

  #### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `apply` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  with b >= 0
  x.shape =        [B1,...,Bb] + [N, R],  with R >= 0.
  ```

  #### Performance

  `LinearOperatorFullMatrix` has exactly the same performance as would be
  achieved by using standard `TensorFlow` matrix ops.  Intelligent choices are
  made based on the following initialization hints.

  * If `dtype` is real, and `is_self_adjoint` and `is_positive_definite`, a
    Cholesky factorization is used for the determinant and solve.

  In all cases, suppose `operator` is a `LinearOperatorFullMatrix` of shape
  `[M, N]`, and `x.shape = [N, R]`.  Then

  * `operator.apply(x)` is `O(M * N * R)`.
  * If `M=N`, `operator.solve(x)` is `O(N^3 * R)`.
  * If `M=N`, `operator.determinant()` is `O(N^3)`.

  If instead `operator` and `x` have shape `[B1,...,Bb, M, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite`.
  These have the following meaning
  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """

  def __init__(self,
               matrix,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               name="LinearOperatorFullMatrix"):
    r"""Initialize a `LinearOperatorFullMatrix`.

    Args:
      matrix:  Shape `[B1,...,Bb, M, N]` with `b >= 0`, `M, N >= 0`.
        Allowed dtypes: `float32`, `float64`, `complex64`, `complex128`.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix\
            #Extension_for_non_symmetric_matrices
      name: A name for this `LinearOperator`.

    Raises:
      TypeError:  If `diag.dtype` is not an allowed type.
    """

    with ops.name_scope(name, values=[matrix]):
      self._matrix = ops.convert_to_tensor(matrix, name="matrix")
      self._check_matrix(self._matrix)

      # Special treatment for (real) Symmetric Positive Definite.
      self._is_spd = (
          (not self._matrix.dtype.is_complex)
          and is_self_adjoint and is_positive_definite)
      if self._is_spd:
        self._chol = linalg_ops.cholesky(self._matrix)

      super(LinearOperatorFullMatrix, self).__init__(
          dtype=self._matrix.dtype,
          graph_parents=[self._matrix],
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          name=name)

  def _check_matrix(self, matrix):
    """Static check of the `matrix` argument."""
    allowed_dtypes = [
        dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]

    matrix = ops.convert_to_tensor(matrix, name="matrix")

    dtype = matrix.dtype
    if dtype not in allowed_dtypes:
      raise TypeError(
          "Argument matrix must have dtype in %s.  Found: %s"
          % (allowed_dtypes, dtype))

    if matrix.get_shape().ndims is not None and matrix.get_shape().ndims < 2:
      raise ValueError(
          "Argument matrix must have at least 2 dimensions.  Found: %s"
          % matrix)

  def _shape(self):
    return self._matrix.get_shape()

  def _shape_tensor(self):
    return array_ops.shape(self._matrix)

  def _apply(self, x, adjoint=False, adjoint_arg=False):
    return math_ops.matmul(
        self._matrix, x, adjoint_a=adjoint, adjoint_b=adjoint_arg)

  def _determinant(self):
    if self._is_spd:
      return math_ops.exp(self.log_abs_determinant())
    return linalg_ops.matrix_determinant(self._matrix)

  def _log_abs_determinant(self):
    if self._is_spd:
      diag = array_ops.matrix_diag_part(self._chol)
      return 2 * math_ops.reduce_sum(math_ops.log(diag), reduction_indices=[-1])
    abs_det = math_ops.abs(self.determinant())
    return math_ops.log(abs_det)

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    rhs = linear_operator_util.matrix_adjoint(rhs) if adjoint_arg else rhs
    if self._is_spd:
      return linalg_ops.cholesky_solve(self._chol, rhs)
    return linalg_ops.matrix_solve(self._matrix, rhs, adjoint=adjoint)

  def _to_dense(self):
    return self._matrix
