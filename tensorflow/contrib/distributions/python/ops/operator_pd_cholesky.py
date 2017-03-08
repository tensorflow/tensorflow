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
"""Symmetric positive definite (PD) Operator defined by a Cholesky factor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import operator_pd
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops


class OperatorPDCholesky(operator_pd.OperatorPDBase):
  """Class representing a (batch) of positive definite matrices `A`.

  This class provides access to functions of a batch of symmetric positive
  definite (PD) matrices `A` in `R^{k x k}` defined by Cholesky factor(s).
  Determinants and solves are `O(k^2)`.

  In practice, this operator represents a (batch) matrix `A` with shape
  `[N1,...,Nn, k, k]` for some `n >= 0`.  The first `n` indices designate a
  batch member.  For every batch member `(i1,...,in)`, `A[i1,...,ib, : :]` is
  a `k x k` matrix.

  Since `A` is (batch) positive definite, it has a (or several) square roots `S`
  such that `A = SS^T`.

  For example,

  ```python
  distributions = tf.contrib.distributions
  chol = [[1.0, 0.0], [1.0, 2.0]]
  operator = OperatorPDCholesky(chol)
  operator.log_det()

  # Compute the quadratic form x^T A^{-1} x for vector x.
  x = [1.0, 2.0]
  operator.inv_quadratic_form_on_vectors(x)

  # Matrix multiplication by the square root, S w.
  # If w is iid normal, S w has covariance A.
  w = [[1.0], [2.0]]
  operator.sqrt_matmul(w)
  ```

  The above three methods, `log_det`, `inv_quadratic_form_on_vectors`, and
  `sqrt_matmul` provide "all" that is necessary to use a covariance matrix
  in a multi-variate normal distribution.  See the class
  `MultivariateNormalCholesky`.
  """

  def __init__(self, chol, verify_pd=True, name="OperatorPDCholesky"):
    """Initialize an OperatorPDCholesky.

    Args:
      chol:  Shape `[N1,...,Nn, k, k]` tensor with `n >= 0`, `k >= 1`, and
        positive diagonal elements.  The strict upper triangle of `chol` is
        never used, and the user may set these elements to zero, or ignore them.
      verify_pd: Whether to check that `chol` has positive diagonal (this is
        equivalent to it being a Cholesky factor of a symmetric positive
        definite matrix.  If `verify_pd` is `False`, correct behavior is not
        guaranteed.
      name:  A name to prepend to all ops created by this class.
    """
    self._verify_pd = verify_pd
    self._name = name
    with ops.name_scope(name):
      with ops.name_scope("init", values=[chol]):
        self._chol = self._check_chol(chol)

  @property
  def verify_pd(self):
    """Whether to verify that this `Operator` is positive definite."""
    return self._verify_pd

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self._chol.dtype

  def _batch_log_det(self):
    """Log determinant of every batch member."""
    # Note that array_ops.diag_part does not seem more efficient for non-batch,
    # and would give a bad result for a batch matrix, so aways use
    # matrix_diag_part.
    diag = array_ops.matrix_diag_part(self._chol)
    det = 2.0 * math_ops.reduce_sum(math_ops.log(diag), reduction_indices=[-1])
    det.set_shape(self.get_shape()[:-2])
    return det

  @property
  def inputs(self):
    """List of tensors that were provided as initialization inputs."""
    return [self._chol]

  def _inv_quadratic_form_on_vectors(self, x):
    # This Operator is defined in terms of the square root, which is easy to
    # solve with (backsubstitution), so this is the preferred way to do
    # inv_quadratic_form_on_vectors().
    return self._iqfov_via_sqrt_solve(x)

  def _matmul(self, x, transpose_x=False):
    # tf.matmul is defined a * b.
    chol = array_ops.matrix_band_part(self._chol, -1, 0)
    chol_times_x = math_ops.matmul(
        chol, x, transpose_a=True, transpose_b=transpose_x)
    return math_ops.matmul(chol, chol_times_x)

  def _batch_matmul(self, x, transpose_x=False):
    # tf.matmul is defined x * y, so "y" is on the right, not "x".
    chol = array_ops.matrix_band_part(self._chol, -1, 0)
    chol_times_x = math_ops.matmul(
        chol, x, adjoint_a=True, adjoint_b=transpose_x)
    return math_ops.matmul(chol, chol_times_x)

  def _sqrt_matmul(self, x, transpose_x=False):
    chol = array_ops.matrix_band_part(self._chol, -1, 0)
    # tf.matmul is defined a * b
    return math_ops.matmul(chol, x, adjoint_b=transpose_x)

  def _batch_sqrt_matmul(self, x, transpose_x=False):
    chol = array_ops.matrix_band_part(self._chol, -1, 0)
    # tf.batch_matmul is defined x * y, so "y" is on the right, not "x".
    return math_ops.matmul(chol, x, adjoint_b=transpose_x)

  def _batch_solve(self, rhs):
    return linalg_ops.cholesky_solve(self._chol, rhs)

  def _batch_sqrt_solve(self, rhs):
    return linalg_ops.matrix_triangular_solve(self._chol, rhs, lower=True)

  def get_shape(self):
    """`TensorShape` giving static shape."""
    return self._chol.get_shape()

  def _shape(self):
    return array_ops.shape(self._chol)

  def _check_chol(self, chol):
    """Verify that `chol` is proper."""
    chol = ops.convert_to_tensor(chol, name="chol")
    if not self.verify_pd:
      return chol

    shape = array_ops.shape(chol)
    rank = array_ops.rank(chol)

    is_matrix = check_ops.assert_rank_at_least(chol, 2)
    is_square = check_ops.assert_equal(
        array_ops.gather(shape, rank - 2), array_ops.gather(shape, rank - 1))

    deps = [is_matrix, is_square]
    diag = array_ops.matrix_diag_part(chol)
    deps.append(check_ops.assert_positive(diag))

    return control_flow_ops.with_dependencies(deps, chol)

  def _sqrt_to_dense(self):
    chol = array_ops.matrix_band_part(self._chol, -1, 0)
    return array_ops.identity(chol)

  def _to_dense(self):
    chol = array_ops.matrix_band_part(self._chol, -1, 0)
    return math_ops.matmul(chol, chol, adjoint_b=True)
