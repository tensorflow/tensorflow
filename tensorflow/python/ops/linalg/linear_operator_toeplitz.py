# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""`LinearOperator` acting like a Toeplitz matrix."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.util.tf_export import tf_export

__all__ = ["LinearOperatorToeplitz",]


@tf_export("linalg.LinearOperatorToeplitz")
class LinearOperatorToeplitz(linear_operator.LinearOperator):
  """`LinearOperator` acting like a [batch] of toeplitz matrices.

  This operator acts like a [batch] Toeplitz matrix `A` with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  #### Description in terms of toeplitz matrices

  Toeplitz means that `A` has constant diagonals. Hence, `A` can be generated
  with two vectors. One represents the first column of the matrix, and the
  other represents the first row.

  Below is a 4 x 4 example:

  ```
  A = |a b c d|
      |e a b c|
      |f e a b|
      |g f e a|
  ```

  #### Example of a Toeplitz operator.

  ```python
  # Create a 3 x 3 Toeplitz operator.
  col = [1., 2., 3.]
  row = [1., 4., -9.]
  operator = LinearOperatorToeplitz(col, row)

  operator.to_dense()
  ==> [[1., 4., -9.],
       [2., 1., 4.],
       [3., 2., 1.]]

  operator.shape
  ==> [3, 3]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [3, 4] Tensor
  operator.matmul(x)
  ==> Shape [3, 4] Tensor

  #### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =   [C1,...,Cc] + [N, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
  ```

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
               col,
               row,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorToeplitz"):
    r"""Initialize a `LinearOperatorToeplitz`.

    Args:
      col: Shape `[B1,...,Bb, N]` `Tensor` with `b >= 0` `N >= 0`.
        The first column of the operator. Allowed dtypes: `float16`, `float32`,
          `float64`, `complex64`, `complex128`. Note that the first entry of
          `col` is assumed to be the same as the first entry of `row`.
      row: Shape `[B1,...,Bb, N]` `Tensor` with `b >= 0` `N >= 0`.
        The first row of the operator. Allowed dtypes: `float16`, `float32`,
          `float64`, `complex64`, `complex128`. Note that the first entry of
          `row` is assumed to be the same as the first entry of `col`.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `diag.dtype` is real, this is auto-set to `True`.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`.
    """

    with ops.name_scope(name, values=[row, col]):
      self._row = ops.convert_to_tensor(row, name="row")
      self._col = ops.convert_to_tensor(col, name="col")
      self._check_row_col(self._row, self._col)

      circulant_col = array_ops.concat(
          [self._col,
           array_ops.zeros_like(self._col[..., 0:1]),
           array_ops.reverse(self._row[..., 1:], axis=[-1])], axis=-1)

      # To be used for matmul.
      self._circulant = linear_operator_circulant.LinearOperatorCirculant(
          fft_ops.fft(_to_complex(circulant_col)),
          input_output_dtype=self._row.dtype)

      if is_square is False:  # pylint:disable=g-bool-id-comparison
        raise ValueError("Only square Toeplitz operators currently supported.")
      is_square = True

      super(LinearOperatorToeplitz, self).__init__(
          dtype=self._row.dtype,
          graph_parents=[self._row, self._col],
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          name=name)

  def _check_row_col(self, row, col):
    """Static check of row and column."""
    for name, tensor in [["row", row], ["col", col]]:
      if tensor.get_shape().ndims is not None and tensor.get_shape().ndims < 1:
        raise ValueError("Argument {} must have at least 1 dimension.  "
                         "Found: {}".format(name, tensor))

    if row.get_shape()[-1] is not None and col.get_shape()[-1] is not None:
      if row.get_shape()[-1] != col.get_shape()[-1]:
        raise ValueError(
            "Expected square matrix, got row and col with mismatched "
            "dimensions.")

  def _shape(self):
    # If d_shape = [5, 3], we return [5, 3, 3].
    v_shape = array_ops.broadcast_static_shape(
        self.row.shape, self.col.shape)
    return v_shape.concatenate(v_shape[-1:])

  def _shape_tensor(self):
    v_shape = array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.row),
        array_ops.shape(self.col))
    k = v_shape[-1]
    return array_ops.concat((v_shape, [k]), 0)

  def _assert_self_adjoint(self):
    return check_ops.assert_equal(
        self.row,
        self.col,
        message=("row and col are not the same, and "
                 "so this operator is not self-adjoint."))

  # TODO(srvasude): Add efficient solver and determinant calculations to this
  # class (based on Levinson recursion.)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # Given a Toeplitz matrix, we can embed it in a Circulant matrix to perform
    # efficient matrix multiplications. Given a Toeplitz matrix with first row
    # [t_0, t_1, ... t_{n-1}] and first column [t0, t_{-1}, ..., t_{-(n-1)},
    # let C by the circulant matrix with first column [t0, t_{-1}, ...,
    # t_{-(n-1)}, 0, t_{n-1}, ..., t_1]. Also adjoin to our input vector `x`
    # `n` zeros, to make it a vector of length `2n` (call it y). It can be shown
    # that if we take the first n entries of `Cy`, this is equal to the Toeplitz
    # multiplication. See:
    # http://math.mit.edu/icg/resources/teaching/18.085-spring2015/toeplitz.pdf
    # for more details.
    x = linalg.adjoint(x) if adjoint_arg else x
    expanded_x = array_ops.concat([x, array_ops.zeros_like(x)], axis=-2)
    result = self._circulant.matmul(
        expanded_x, adjoint=adjoint, adjoint_arg=False)

    return math_ops.cast(
        result[..., :self.domain_dimension_tensor(), :],
        self.dtype)

  def _trace(self):
    return math_ops.cast(
        self.domain_dimension_tensor(),
        dtype=self.dtype) * self.col[..., 0]

  def _diag_part(self):
    diag_entry = self.col[..., 0:1]
    return diag_entry * array_ops.ones(
        [self.domain_dimension_tensor()], self.dtype)

  @property
  def col(self):
    return self._col

  @property
  def row(self):
    return self._row


def _to_complex(x):
  dtype = dtypes.complex64
  if x.dtype in [dtypes.float64, dtypes.complex128]:
    dtype = dtypes.complex128
  return math_ops.cast(x, dtype)
