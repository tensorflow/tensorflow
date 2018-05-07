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
"""CholeskyOuterProduct bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.ops.distributions import util as distribution_util


__all__ = [
    "CholeskyOuterProduct",
]


class CholeskyOuterProduct(bijector.Bijector):
  """Compute `g(X) = X @ X.T`; X is lower-triangular, positive-diagonal matrix.

  Note: the upper-triangular part of X is ignored (whether or not its zero).

  The surjectivity of g as a map from  the set of n x n positive-diagonal
  lower-triangular matrices to the set of SPD matrices follows immediately from
  executing the Cholesky factorization algorithm on an SPD matrix A to produce a
  positive-diagonal lower-triangular matrix L such that `A = L @ L.T`.

  To prove the injectivity of g, suppose that L_1 and L_2 are lower-triangular
  with positive diagonals and satisfy `A = L_1 @ L_1.T = L_2 @ L_2.T`. Then
    `inv(L_1) @ A @ inv(L_1).T = [inv(L_1) @ L_2] @ [inv(L_1) @ L_2].T = I`.
  Setting `L_3 := inv(L_1) @ L_2`, that L_3 is a positive-diagonal
  lower-triangular matrix follows from `inv(L_1)` being positive-diagonal
  lower-triangular (which follows from the diagonal of a triangular matrix being
  its spectrum), and that the product of two positive-diagonal lower-triangular
  matrices is another positive-diagonal lower-triangular matrix.

  A simple inductive argument (proceeding one column of L_3 at a time) shows
  that, if `I = L_3 @ L_3.T`, with L_3 being lower-triangular with positive-
  diagonal, then `L_3 = I`. Thus, `L_1 = L_2`, proving injectivity of g.

  #### Examples

  ```python
  bijector.CholeskyOuterProduct().forward(x=[[1., 0], [2, 1]])
  # Result: [[1., 2], [2, 5]], i.e., x @ x.T

  bijector.CholeskyOuterProduct().inverse(y=[[1., 2], [2, 5]])
  # Result: [[1., 0], [2, 1]], i.e., cholesky(y).
  ```

  """

  def __init__(self, validate_args=False, name="cholesky_outer_product"):
    """Instantiates the `CholeskyOuterProduct` bijector.

    Args:
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    self._graph_parents = []
    self._name = name
    super(CholeskyOuterProduct, self).__init__(
        forward_min_event_ndims=2,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    if self.validate_args:
      is_matrix = check_ops.assert_rank_at_least(x, 2)
      shape = array_ops.shape(x)
      is_square = check_ops.assert_equal(shape[-2], shape[-1])
      x = control_flow_ops.with_dependencies([is_matrix, is_square], x)
    # For safety, explicitly zero-out the upper triangular part.
    x = array_ops.matrix_band_part(x, -1, 0)
    return math_ops.matmul(x, x, adjoint_b=True)

  def _inverse(self, y):
    return linalg_ops.cholesky(y)

  def _forward_log_det_jacobian(self, x):
    # Let Y be a symmetric, positive definite matrix and write:
    #   Y = X X.T
    # where X is lower-triangular.
    #
    # Observe that,
    #   dY[i,j]/dX[a,b]
    #   = d/dX[a,b] { X[i,:] X[j,:] }
    #   = sum_{d=1}^p { I[i=a] I[d=b] X[j,d] + I[j=a] I[d=b] X[i,d] }
    #
    # To compute the Jacobian dX/dY we must represent X,Y as vectors. Since Y is
    # symmetric and X is lower-triangular, we need vectors of dimension:
    #   d = p (p + 1) / 2
    # where X, Y are p x p matrices, p > 0. We use a row-major mapping, i.e.,
    #   k = { i (i + 1) / 2 + j   i>=j
    #       { undef               i<j
    # and assume zero-based indexes. When k is undef, the element is dropped.
    # Example:
    #           j      k
    #        0 1 2 3  /
    #    0 [ 0 . . . ]
    # i  1 [ 1 2 . . ]
    #    2 [ 3 4 5 . ]
    #    3 [ 6 7 8 9 ]
    # Write vec[.] to indicate transforming a matrix to vector via k(i,j). (With
    # slight abuse: k(i,j)=undef means the element is dropped.)
    #
    # We now show d vec[Y] / d vec[X] is lower triangular. Assuming both are
    # defined, observe that k(i,j) < k(a,b) iff (1) i<a or (2) i=a and j<b.
    # In both cases dvec[Y]/dvec[X]@[k(i,j),k(a,b)] = 0 since:
    # (1) j<=i<a thus i,j!=a.
    # (2) i=a>j  thus i,j!=a.
    #
    # Since the Jacobian is lower-triangular, we need only compute the product
    # of diagonal elements:
    #   d vec[Y] / d vec[X] @[k(i,j), k(i,j)]
    #   = X[j,j] + I[i=j] X[i,j]
    #   = 2 X[j,j].
    # Since there is a 2 X[j,j] term for every lower-triangular element of X we
    # conclude:
    #   |Jac(d vec[Y]/d vec[X])| = 2^p prod_{j=0}^{p-1} X[j,j]^{p-j}.
    diag = array_ops.matrix_diag_part(x)

    # We now ensure diag is columnar. Eg, if `diag = [1, 2, 3]` then the output
    # is `[[1], [2], [3]]` and if `diag = [[1, 2, 3], [4, 5, 6]]` then the
    # output is unchanged.
    diag = self._make_columnar(diag)

    if self.validate_args:
      is_matrix = check_ops.assert_rank_at_least(
          x, 2, message="Input must be a (batch of) matrix.")
      shape = array_ops.shape(x)
      is_square = check_ops.assert_equal(
          shape[-2], shape[-1],
          message="Input must be a (batch of) square matrix.")
      # Assuming lower-triangular means we only need check diag>0.
      is_positive_definite = check_ops.assert_positive(
          diag, message="Input must be positive definite.")
      x = control_flow_ops.with_dependencies(
          [is_matrix, is_square, is_positive_definite], x)

    # Create a vector equal to: [p, p-1, ..., 2, 1].
    if x.get_shape().ndims is None or x.get_shape()[-1].value is None:
      p_int = array_ops.shape(x)[-1]
      p_float = math_ops.cast(p_int, dtype=x.dtype)
    else:
      p_int = x.get_shape()[-1].value
      p_float = np.array(p_int, dtype=x.dtype.as_numpy_dtype)
    exponents = math_ops.linspace(p_float, 1., p_int)

    sum_weighted_log_diag = array_ops.squeeze(
        math_ops.matmul(math_ops.log(diag),
                        exponents[..., array_ops.newaxis]),
        axis=-1)
    fldj = p_float * np.log(2.) + sum_weighted_log_diag

    return fldj

  def _make_columnar(self, x):
    """Ensures non-scalar input has at least one column.

    Example:
      If `x = [1, 2, 3]` then the output is `[[1], [2], [3]]`.

      If `x = [[1, 2, 3], [4, 5, 6]]` then the output is unchanged.

      If `x = 1` then the output is unchanged.

    Args:
      x: `Tensor`.

    Returns:
      columnar_x: `Tensor` with at least two dimensions.
    """
    if x.get_shape().ndims is not None:
      if x.get_shape().ndims == 1:
        x = x[array_ops.newaxis, :]
      return x
    shape = array_ops.shape(x)
    maybe_expanded_shape = array_ops.concat([
        shape[:-1],
        distribution_util.pick_vector(
            math_ops.equal(array_ops.rank(x), 1),
            [1], np.array([], dtype=np.int32)),
        shape[-1:],
    ], 0)
    return array_ops.reshape(x, maybe_expanded_shape)
