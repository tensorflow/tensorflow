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
"""`LinearOperator` acting like a lower triangular matrix."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export

__all__ = [
    "LinearOperatorLowerTriangular",
]


@tf_export("linalg.LinearOperatorLowerTriangular")
class LinearOperatorLowerTriangular(linear_operator.LinearOperator):
  """`LinearOperator` acting like a [batch] square lower triangular matrix.

  This operator acts like a [batch] lower triangular matrix `A` with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.

  `LinearOperatorLowerTriangular` is initialized with a `Tensor` having
  dimensions `[B1,...,Bb, N, N]`. The upper triangle of the last two
  dimensions is ignored.

  ```python
  # Create a 2 x 2 lower-triangular linear operator.
  tril = [[1., 2.], [3., 4.]]
  operator = LinearOperatorLowerTriangular(tril)

  # The upper triangle is ignored.
  operator.to_dense()
  ==> [[1., 0.]
       [3., 4.]]

  operator.shape
  ==> [2, 2]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor

  # Create a [2, 3] batch of 4 x 4 linear operators.
  tril = tf.random.normal(shape=[2, 3, 4, 4])
  operator = LinearOperatorLowerTriangular(tril)
  ```

  #### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =        [B1,...,Bb] + [N, R],  with R >= 0.
  ```

  #### Performance

  Suppose `operator` is a `LinearOperatorLowerTriangular` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then

  * `operator.matmul(x)` involves `N^2 * R` multiplications.
  * `operator.solve(x)` involves `N * R` size `N` back-substitutions.
  * `operator.determinant()` involves a size `N` `reduce_prod`.

  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning:

  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """

  def __init__(self,
               tril,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorLowerTriangular"):
    r"""Initialize a `LinearOperatorLowerTriangular`.

    Args:
      tril:  Shape `[B1,...,Bb, N, N]` with `b >= 0`, `N >= 0`.
        The lower triangular part of `tril` defines this operator.  The strictly
        upper triangle is ignored.
      is_non_singular:  Expect that this operator is non-singular.
        This operator is non-singular if and only if its diagonal elements are
        all non-zero.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  This operator is self-adjoint only if it is diagonal with
        real-valued diagonal entries.  In this case it is advised to use
        `LinearOperatorDiag`.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`.

    Raises:
      ValueError:  If `is_square` is `False`.
    """

    if is_square is False:
      raise ValueError(
          "Only square lower triangular operators supported at this time.")
    is_square = True

    with ops.name_scope(name, values=[tril]):
      self._tril = linear_operator_util.convert_nonref_to_tensor(tril,
                                                                 name="tril")
      self._check_tril(self._tril)

      super(LinearOperatorLowerTriangular, self).__init__(
          dtype=self._tril.dtype,
          graph_parents=[self._tril],
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          name=name)

  def _check_tril(self, tril):
    """Static check of the `tril` argument."""

    if tril.shape.ndims is not None and tril.shape.ndims < 2:
      raise ValueError(
          "Argument tril must have at least 2 dimensions.  Found: %s"
          % tril)

  def _get_tril(self):
    """Gets the `tril` kwarg, with upper part zero-d out."""
    return array_ops.matrix_band_part(self._tril, -1, 0)

  def _get_diag(self):
    """Gets the diagonal part of `tril` kwarg."""
    return array_ops.matrix_diag_part(self._tril)

  def _shape(self):
    return self._tril.shape

  def _shape_tensor(self):
    return array_ops.shape(self._tril)

  def _assert_non_singular(self):
    return linear_operator_util.assert_no_entries_with_modulus_zero(
        self._get_diag(),
        message="Singular operator:  Diagonal contained zero values.")

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    return math_ops.matmul(
        self._get_tril(), x, adjoint_a=adjoint, adjoint_b=adjoint_arg)

  def _determinant(self):
    return math_ops.reduce_prod(self._get_diag(), axis=[-1])

  def _log_abs_determinant(self):
    return math_ops.reduce_sum(
        math_ops.log(math_ops.abs(self._get_diag())), axis=[-1])

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
    return linear_operator_util.matrix_triangular_solve_with_broadcast(
        self._get_tril(), rhs, lower=True, adjoint=adjoint)

  def _to_dense(self):
    return self._get_tril()
