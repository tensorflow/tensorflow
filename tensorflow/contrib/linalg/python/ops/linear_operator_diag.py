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
"""`LinearOperator` acting like a diagonal matrix."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.linalg.python.ops import linear_operator
from tensorflow.contrib.linalg.python.ops import linear_operator_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops

__all__ = ["LinearOperatorDiag",]


class LinearOperatorDiag(linear_operator.LinearOperator):
  """`LinearOperator` acting like a [batch] square diagonal matrix.

  This operator acts like a [batch] diagonal matrix `A` with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  `LinearOperatorDiag` is initialized with a (batch) vector.

  ```python
  # Create a 2 x 2 diagonal linear operator.
  diag = [1., -1.]
  operator = LinearOperatorDiag(diag)

  operator.to_dense()
  ==> [[1.,  0.]
       [0., -1.]]

  operator.shape
  ==> [2, 2]

  operator.log_determinant()
  ==> scalar Tensor

  x = ... Shape [2, 4] Tensor
  operator.apply(x)
  ==> Shape [2, 4] Tensor

  # Create a [2, 3] batch of 4 x 4 linear operators.
  diag = tf.random_normal(shape=[2, 3, 4])
  operator = LinearOperatorDiag(diag)

  # Create a shape [2, 1, 4, 2] vector.  Note that this shape is compatible
  # since the batch dimensions, [2, 1], are brodcast to
  # operator.batch_shape = [2, 3].
  y = tf.random_normal(shape=[2, 1, 4, 2])
  x = operator.solve(y)
  ==> operator.apply(x) = y
  ```

  #### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `apply` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =   [C1,...,Cc] + [N, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
  ```

  #### Performance

  Suppose `operator` is a `LinearOperatorDiag` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then

  * `operator.apply(x)` involves `N * R` multiplications.
  * `operator.solve(x)` involves `N` divisions and `N * R` multiplications.
  * `operator.determinant()` involves a size `N` `reduce_prod`.

  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
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
               diag,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               name="LinearOperatorDiag"):
    """Initialize a `LinearOperatorDiag`.

    Args:
      diag:  Shape `[B1,...,Bb, N]` `Tensor` with `b >= 0` `N >= 0`.
        The diagonal of the operator.  Allowed dtypes: `float32`, `float64`,
          `complex64`, `complex128`.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `diag.dtype` is real, this is auto-set to `True`.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the real part of all eigenvalues is positive.  We do not require
        the operator to be self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix
            #Extension_for_non_symmetric_matrices
      name: A name for this `LinearOperator`.

    Raises:
      TypeError:  If `diag.dtype` is not an allowed type.
      ValueError:  If `diag.dtype` is real, and `is_self_adjoint` is not `True`.
    """

    allowed_dtypes = [
        dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]

    with ops.name_scope(name, values=[diag]):
      self._diag = ops.convert_to_tensor(diag, name="diag")
      dtype = self._diag.dtype
      if dtype not in allowed_dtypes:
        raise TypeError(
            "Argument diag must have dtype in %s.  Found: %s"
            % (allowed_dtypes, dtype))

      # Check and auto-set hints.
      if not dtype.is_complex:
        if is_self_adjoint is False:
          raise ValueError("A real diagonal operator is always self adjoint.")
        else:
          is_self_adjoint = True

      super(LinearOperatorDiag, self).__init__(
          dtype=dtype,
          graph_parents=[self._diag],
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          name=name)

  def _shape(self):
    # If d_shape = [5, 3], we return [5, 3, 3].
    d_shape = self._diag.get_shape()
    return d_shape.concatenate(d_shape[-1:])

  def _shape_dynamic(self):
    d_shape = array_ops.shape(self._diag)
    k = d_shape[-1]
    return array_ops.concat((d_shape, [k]), 0)

  def _assert_non_singular(self):
    return linear_operator_util.assert_no_entries_with_modulus_zero(
        self._diag,
        message="Singular operator:  Diagonal contained zero values.")

  def _assert_positive_definite(self):
    if self.dtype.is_complex:
      message = (
          "Diagonal operator had diagonal entries with non-positive real part, "
          "thus was not positive definite.")
    else:
      message = (
          "Real diagonal operator had non-positive diagonal entries, "
          "thus was not positive definite.")

    return check_ops.assert_positive(
        math_ops.real(self._diag),
        message=message)

  def _assert_self_adjoint(self):
    return linear_operator_util.assert_zero_imag_part(
        self._diag,
        message=(
            "This diagonal operator contained non-zero imaginary values.  "
            " Thus it was not self-adjoint."))

  def _apply(self, x, adjoint=False):
    diag_term = math_ops.conj(self._diag) if adjoint else self._diag
    diag_mat = array_ops.expand_dims(diag_term, -1)
    return diag_mat * x

  def _determinant(self):
    return math_ops.reduce_prod(self._diag, reduction_indices=[-1])

  def _log_abs_determinant(self):
    return math_ops.reduce_sum(
        math_ops.log(math_ops.abs(self._diag)), reduction_indices=[-1])

  def _solve(self, rhs, adjoint=False):
    diag_term = math_ops.conj(self._diag) if adjoint else self._diag
    inv_diag_mat = array_ops.expand_dims(1. / diag_term, -1)
    return rhs * inv_diag_mat

  def _to_dense(self):
    return array_ops.matrix_diag(self._diag)

  def _add_to_tensor(self, x):
    x_diag = array_ops.matrix_diag_part(x)
    new_diag = self._diag + x_diag
    return array_ops.matrix_set_diag(x, new_diag)
