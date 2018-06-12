# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""MatrixInverseTriL bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector


__all__ = [
    "MatrixInverseTriL",
]


class MatrixInverseTriL(bijector.Bijector):
  """Computes `g(L) = inv(L)`, where `L` is a lower-triangular matrix.

  `L` must be nonsingular; equivalently, all diagonal entries of `L` must be
  nonzero.

  The input must have `rank >= 2`.  The input is treated as a batch of matrices
  with batch shape `input.shape[:-2]`, where each matrix has dimensions
  `input.shape[-2]` by `input.shape[-1]` (hence `input.shape[-2]` must equal
  `input.shape[-1]`).

  #### Examples

  ```python
  tfd.bijectors.MatrixInverseTriL().forward(x=[[1., 0], [2, 1]])
  # Result: [[1., 0], [-2, 1]], i.e., inv(x)

  tfd.bijectors.MatrixInverseTriL().inverse(y=[[1., 0], [-2, 1]])
  # Result: [[1., 0], [2, 1]], i.e., inv(y).
  ```

  """

  def __init__(self, validate_args=False, name="matrix_inverse_tril"):
    """Instantiates the `MatrixInverseTriL` bijector.

    Args:
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    self._graph_parents = []
    self._name = name
    super(MatrixInverseTriL, self).__init__(
        forward_min_event_ndims=2,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    with ops.control_dependencies(self._assertions(x)):
      shape = array_ops.shape(x)
      return linalg_ops.matrix_triangular_solve(
          x, linalg_ops.eye(shape[-1], batch_shape=shape[:-2]), lower=True)

  def _inverse(self, y):
    return self._forward(y)

  def _forward_log_det_jacobian(self, x):
    # Calculation of the Jacobian:
    #
    # Let X = (x_{ij}), 0 <= i,j < n, be a matrix of indeterminates.  Let Z =
    # X^{-1} where Z = (z_{ij}).  Then
    #
    #     dZ/dx_{ij} = (d/dt | t=0) Y(t)^{-1},
    #
    # where Y(t) = X + t*E_{ij} and E_{ij} is the matrix with a 1 in the (i,j)
    # entry and zeros elsewhere.  By the product rule,
    #
    #     0 = d/dt [Identity matrix]
    #       = d/dt [Y Y^{-1}]
    #       = Y d/dt[Y^{-1}] + dY/dt Y^{-1}
    #
    # so
    #
    #     d/dt[Y^{-1}] = -Y^{-1} dY/dt Y^{-1}
    #                  = -Y^{-1} E_{ij} Y^{-1}.
    #
    # Evaluating at t=0,
    #
    #     dZ/dx_{ij} = -Z E_{ij} Z.
    #
    # Taking the (r,s) entry of each side,
    #
    #     dz_{rs}/dx_{ij} = -z_{ri}z_{sj}.
    #
    # Now, let J be the Jacobian dZ/dX, arranged as the n^2-by-n^2 matrix whose
    # (r*n + s, i*n + j) entry is dz_{rs}/dx_{ij}.  Considering J as an n-by-n
    # block matrix with n-by-n blocks, the above expression for dz_{rs}/dx_{ij}
    # shows that the block at position (r,i) is -z_{ri}Z.  Hence
    #
    #          J = -KroneckerProduct(Z, Z),
    #     det(J) = (-1)^(n^2) (det Z)^(2n)
    #            = (-1)^n (det X)^(-2n).
    with ops.control_dependencies(self._assertions(x)):
      return (-2. * math_ops.cast(array_ops.shape(x)[-1], x.dtype.base_dtype) *
              math_ops.reduce_sum(
                  math_ops.log(math_ops.abs(array_ops.matrix_diag_part(x))),
                  axis=-1))

  def _assertions(self, x):
    if not self.validate_args:
      return []
    shape = array_ops.shape(x)
    is_matrix = check_ops.assert_rank_at_least(
        x, 2, message="Input must have rank at least 2.")
    is_square = check_ops.assert_equal(
        shape[-2], shape[-1], message="Input must be a square matrix.")
    above_diagonal = array_ops.matrix_band_part(
        array_ops.matrix_set_diag(
            x, array_ops.zeros(shape[:-1], dtype=dtypes.float32)),
        0, -1)
    is_lower_triangular = check_ops.assert_equal(
        above_diagonal, array_ops.zeros_like(above_diagonal),
        message="Input must be lower triangular.")
    # A lower triangular matrix is nonsingular iff all its diagonal entries are
    # nonzero.
    diag_part = array_ops.matrix_diag_part(x)
    is_nonsingular = check_ops.assert_none_equal(
        diag_part, array_ops.zeros_like(diag_part),
        message="Input must have all diagonal entries nonzero.")
    return [is_matrix, is_square, is_lower_triangular, is_nonsingular]
