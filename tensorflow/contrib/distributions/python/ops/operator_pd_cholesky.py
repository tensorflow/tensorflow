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

__all__ = ['OperatorPDCholesky', 'batch_matrix_diag_transform']


class OperatorPDCholesky(operator_pd.OperatorPDBase):
  """Class representing a (batch) of positive definite matrices `A`.

  This class provides access to functions of a batch of symmetric positive
  definite (PD) matrices `A` in `R^{k x k}` defined by Cholesky factor(s).
  Determinants and solves are `O(k^2)`.

  In practice, this operator represents a (batch) matrix `A` with shape
  `[N1,...,Nb, k, k]` for some `b >= 0`.  The first `b` indices designate a
  batch member.  For every batch member `(n1,...,nb)`, `A[n1,...,nb, : :]` is
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
  operator.inv_quadratic_form(x)

  # Matrix multiplication by the square root, S w.
  # If w is iid normal, S w has covariance A.
  w = [[1.0], [2.0]]
  operator.sqrt_matmul(w)
  ```

  The above three methods, `log_det`, `inv_quadratic_form`, and
  `sqrt_matmul` provide "all" that is necessary to use a covariance matrix
  in a multi-variate normal distribution.  See the class `MVNOperatorPD`.
  """

  def __init__(self, chol, verify_pd=True, name='OperatorPDCholesky'):
    """Initialize an OperatorPDCholesky.

    Args:
      chol:  Shape `[N1,...,Nb, k, k]` tensor with `b >= 0`, `k >= 1`, and
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
      with ops.op_scope([chol], 'init'):
        self._diag = array_ops.batch_matrix_diag_part(chol)
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

  def inv_quadratic_form(self, x, name='inv_quadratic_form'):
    """Compute the induced vector norm (squared): ||x||^2 := x^T A^{-1} x.

    For every batch member, this is done in `O(k^2)` complexity.  The efficiency
    depends on the shape of `x`.
    * If `x.shape = [M1,...,Mm, N1,...,Nb, k]`, `m >= 0`, and
      `self.shape = [N1,...,Nb, k, k]`, `x` will be reshaped and the
      initialization matrix `chol` does not need to be copied.
    * Otherwise, data will be broadcast and copied.

    Args:
      x: `Tensor` with shape broadcastable to `[N1,...,Nb, k]` and same `dtype`
        as self.  If the batch dimensions of `x` do not match exactly with those
        of self, `x` and/or self's Cholesky factor will broadcast to match, and
        the resultant set of linear systems are solved independently.  This may
        result in inefficient operation.
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` holding the square of the norm induced by inverse of `A`.  For
      every broadcast batch member.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([x] + self.inputs, name):
        x = ops.convert_to_tensor(x, name='x')
        # Boolean, True iff x.shape = [M1,...,Mm] + chol.shape[:-1].
        should_flip = self._should_flip(x)
        x_whitened = control_flow_ops.cond(
            should_flip,
            lambda: self._x_whitened_if_should_flip(x),
            lambda: self._x_whitened_if_no_flip(x))

        # batch version of: || L^{-1} x||^2
        x_whitened_norm = math_ops.reduce_sum(
            math_ops.square(x_whitened),
            reduction_indices=[-1])

        # Set the final shape by making a dummy tensor that will never be
        # evaluated.
        chol_without_final_dim = math_ops.reduce_sum(
            self._chol, reduction_indices=[-1])
        final_shape = (x + chol_without_final_dim).get_shape()[:-1]
        x_whitened_norm.set_shape(final_shape)

        return x_whitened_norm

  def _should_flip(self, x):
    """Return boolean tensor telling whether `x` should be flipped."""
    # We "flip" (see self._flip_front_dims_to_back) iff
    # chol.shape =         [N1,...,Nn, k, k]
    # x.shape = [M1,...,Mm, N1,...,Nn, k]
    x_shape = array_ops.shape(x)
    x_rank = array_ops.rank(x)
    # If m <= 0, we should not flip.
    m = x_rank + 1 - self.rank()
    def result_if_m_positive():
      x_shape_right = array_ops.slice(x_shape, [m], [x_rank - m])
      return math_ops.reduce_all(
          math_ops.equal(x_shape_right, self.vector_shape()))
    return control_flow_ops.cond(
        m > 0,
        result_if_m_positive,
        lambda: ops.convert_to_tensor(False))

  def _x_whitened_if_no_flip(self, x):
    """x_whitened in the event of no flip."""
    # Tensors to use if x and chol have same shape, or a shape that must be
    # broadcast to match.
    chol_bcast, x_bcast = self._get_chol_and_x_compatible_shape(x)

    # batch version of: L^{-1} x
    # Note that here x_bcast has trailing dims of (k, 1), for "1" system of k
    # linear equations.  This is the form used by the solver.
    x_whitened_expanded = linalg_ops.batch_matrix_triangular_solve(
        chol_bcast, x_bcast)

    x_whitened = array_ops.squeeze(x_whitened_expanded, squeeze_dims=[-1])
    return x_whitened

  def _x_whitened_if_should_flip(self, x):
    # Tensor to use if x.shape = [M1,...,Mm] + chol.shape[:-1],
    # which is common if x was sampled.
    x_flipped = self._flip_front_dims_to_back(x)

    # batch version of: L^{-1} x
    x_whitened_expanded = linalg_ops.batch_matrix_triangular_solve(
        self._chol, x_flipped)

    return self._unfip_back_dims_to_front(
        x_whitened_expanded,
        array_ops.shape(x),
        x.get_shape())

  def _flip_front_dims_to_back(self, x):
    """Flip x to make x.shape = chol.shape[:-1] + [M1*...*Mr]."""
    # E.g. suppose
    # chol.shape =         [N1,...,Nn, k, k]
    # x.shape = [M1,...,Mm, N1,...,Nn, k]
    # Then we want to return x_flipped where
    # x_flipped.shape = [N1,...,Nn, k, M1*...*Mm].
    x_shape = array_ops.shape(x)
    x_rank = array_ops.rank(x)
    m = x_rank + 1 - self.rank()
    x_shape_left = array_ops.slice(x_shape, [0], [m])

    # Permutation corresponding to [N1,...,Nn, k, M1,...,Mm]
    perm = array_ops.concat(
        0, (math_ops.range(m, x_rank), math_ops.range(0, m)))
    x_permuted = array_ops.transpose(x, perm=perm)

    # Now that things are ordered correctly, condense the last dimensions.
    # condensed_shape = [M1*...*Mm]
    condensed_shape = array_ops.pack([math_ops.reduce_prod(x_shape_left)])
    new_shape = array_ops.concat(0, (self.vector_shape(), condensed_shape))

    return array_ops.reshape(x_permuted, new_shape)

  def _unfip_back_dims_to_front(self, x_flipped, x_shape, x_get_shape):
    # E.g. suppose that originally
    # chol.shape =         [N1,...,Nn, k, k]
    # x.shape = [M1,...,Mm, N1,...,Nn, k]
    # Then we have flipped the dims so that
    # x_flipped.shape = [N1,...,Nn, k, M1*...*Mm].
    # We want to return x with the original shape.
    rank = array_ops.rank(x_flipped)
    # Permutation corresponding to [M1*...*Mm, N1,...,Nn, k]
    perm = array_ops.concat(
        0, (math_ops.range(rank - 1, rank), math_ops.range(0, rank - 1)))
    x_with_end_at_beginning = array_ops.transpose(x_flipped, perm=perm)
    x = array_ops.reshape(x_with_end_at_beginning, x_shape)
    return x

  def _get_chol_and_x_compatible_shape(self, x):
    """Return self.chol and x, (possibly) broadcast to compatible shape."""
    # x and chol are "compatible" if their shape matches except for the last two
    # dimensions of chol are [k, k], and the last two of x are [k, 1].
    # E.g. x.shape = [A, B, k, 1], and chol.shape = [A, B, k, k]
    # This is required for the batch_triangular_solve, which does not broadcast.

    # TODO(langmore) This broadcast replicates matrices unnecesarily!  In the
    # case where
    # x.shape = [M1,...,Mr, N1,...,Nb, k], and chol.shape = [N1,...,Nb, k, k]
    # (which is common if x was sampled), the front dimensions of x can be
    # "flipped" to the end, making
    # x_flipped.shape = [N1,...,Nb, k, M1*...*Mr],
    # and this can be handled by the linear solvers.  This is preferred, because
    # it does not replicate the matrix, or create any new data.

    # We assume x starts without the trailing singleton dimension, e.g.
    # x.shape = [B, k].
    chol = self._chol
    with ops.op_scope([x] + self.inputs, 'get_chol_and_x_compatible_shape'):
      # If we determine statically that shapes match, we're done.
      if x.get_shape() == chol.get_shape()[:-1]:
        x_expanded = array_ops.expand_dims(x, -1)
        return chol, x_expanded

      # Dynamic check if shapes match or not.
      vector_shape = self.vector_shape()  # Shape of chol minus last dim.
      are_same_rank = math_ops.equal(
          array_ops.rank(x), array_ops.rank(vector_shape))

      def shapes_match_if_same_rank():
        return math_ops.reduce_all(math_ops.equal(
            array_ops.shape(x), vector_shape))

      shapes_match = control_flow_ops.cond(are_same_rank,
                                           shapes_match_if_same_rank,
                                           lambda: ops.convert_to_tensor(False))

      # Make tensors (never instantiated) holding the broadcast shape.
      # matrix_broadcast_dummy is the shape we will broadcast chol to.
      matrix_bcast_dummy = chol + array_ops.expand_dims(x, -1)
      # vector_bcast_dummy is the shape we will bcast x to, before we expand it.
      chol_minus_last_dim = math_ops.reduce_sum(chol, reduction_indices=[-1])
      vector_bcast_dummy = x + chol_minus_last_dim

      chol_bcast = chol + array_ops.zeros_like(matrix_bcast_dummy)
      x_bcast = x + array_ops.zeros_like(vector_bcast_dummy)

      chol_result = control_flow_ops.cond(shapes_match, lambda: chol,
                                          lambda: chol_bcast)
      chol_result.set_shape(matrix_bcast_dummy.get_shape())
      x_result = control_flow_ops.cond(shapes_match, lambda: x, lambda: x_bcast)
      x_result.set_shape(vector_bcast_dummy.get_shape())

      x_expanded = array_ops.expand_dims(x_result, -1)

      return chol_result, x_expanded

  def log_det(self, name='log_det'):
    """Log determinant of every batch member."""
    with ops.name_scope(self.name):
      with ops.op_scope(self.inputs, name):
        det = 2.0 * math_ops.reduce_sum(
            math_ops.log(self._diag),
            reduction_indices=[-1])
        det.set_shape(self._chol.get_shape()[:-2])
        return det

  @property
  def inputs(self):
    """List of tensors that were provided as initialization inputs."""
    return [self._chol]

  def sqrt_matmul(self, x, name='sqrt_matmul'):
    """Left (batch) matmul `x` by a sqrt of this matrix:  `Sx` where `A = S S^T.

    Args:
      x: `Tensor` with shape broadcastable to `[N1,...,Nb, k]` and same `dtype`
        as self.
      name:  A name scope to use for ops added by this method.

    Returns:
      Shape `[N1,...,Nb, k]` `Tensor` holding the product `S x`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([x] + self.inputs, name):
        chol_lower = array_ops.batch_matrix_band_part(self._chol, -1, 0)
        return math_ops.batch_matmul(chol_lower, x)

  def get_shape(self):
    """`TensorShape` giving static shape."""
    return self._chol.get_shape()

  def shape(self, name='shape'):
    with ops.name_scope(self.name):
      with ops.op_scope(self.inputs, name):
        return array_ops.shape(self._chol)

  def _check_chol(self, chol):
    """Verify that `chol` is proper."""
    chol = ops.convert_to_tensor(chol, name='chol')
    if not self.verify_pd:
      return chol

    shape = array_ops.shape(chol)
    rank = array_ops.rank(chol)

    is_matrix = check_ops.assert_rank_at_least(chol, 2)
    is_square = check_ops.assert_equal(
        array_ops.gather(shape, rank - 2), array_ops.gather(shape, rank - 1))

    deps = [is_matrix, is_square]
    deps.append(check_ops.assert_positive(self._diag))

    return control_flow_ops.with_dependencies(deps, chol)

  def matmul(self, x, name='matmul'):
    """Left (batch) matrix multiplication of `x` by this operator."""
    chol = self._chol
    with ops.name_scope(self.name):
      with ops.op_scope(self.inputs, name):
        a_times_x = math_ops.batch_matmul(chol, x, adj_x=True)
        return math_ops.batch_matmul(chol, a_times_x)

  def to_dense_sqrt(self, name='to_dense_sqrt'):
    """Return a dense (batch) matrix representing sqrt of this covariance."""
    with ops.name_scope(self.name):
      with ops.op_scope(self.inputs, name):
        return array_ops.identity(self._chol)

  def to_dense(self, name='to_dense'):
    """Return a dense (batch) matrix representing this covariance."""
    chol = self._chol
    with ops.name_scope(self.name):
      with ops.op_scope(self.inputs, name):
        return math_ops.batch_matmul(chol, chol, adj_y=True)


def batch_matrix_diag_transform(matrix, transform=None, name=None):
  """Transform diagonal of [batch-]matrix, leave rest of matrix unchanged.

  Create a trainable covariance defined by a Cholesky factor:

  ```python
  # Transform network layer into 2 x 2 array.
  matrix_values = tf.contrib.layers.fully_connected(activations, 4)
  matrix = tf.reshape(matrix_values, (batch_size, 2, 2))

  # Make the diagonal positive.  If the upper triangle was zero, this would be a
  # valid Cholesky factor.
  chol = batch_matrix_diag_transform(matrix, transform=tf.nn.softplus)

  # OperatorPDCholesky ignores the upper triangle.
  operator = OperatorPDCholesky(chol)
  ```

  Example of heteroskedastic 2-D linear regression.

  ```python
  # Get a trainable Cholesky factor.
  matrix_values = tf.contrib.layers.fully_connected(activations, 4)
  matrix = tf.reshape(matrix_values, (batch_size, 2, 2))
  chol = batch_matrix_diag_transform(matrix, transform=tf.nn.softplus)

  # Get a trainable mean.
  mu = tf.contrib.layers.fully_connected(activations, 2)

  # This is a fully trainable multivariate normal!
  dist = tf.contrib.distributions.MVNCholesky(mu, chol)

  # Standard log loss.  Minimizing this will "train" mu and chol, and then dist
  # will be a distribution predicting labels as multivariate Gaussians.
  loss = -1 * tf.reduce_mean(dist.log_pdf(labels))
  ```

  Args:
    matrix:  Rank `R` `Tensor`, `R >= 2`, where the last two dimensions are
      equal.
    transform:  Element-wise function mapping `Tensors` to `Tensors`.  To
      be applied to the diagonal of `matrix`.  If `None`, `matrix` is returned
      unchanged.  Defaults to `None`.
    name:  A name to give created ops.
      Defaults to "batch_matrix_diag_transform".

  Returns:
    A `Tensor` with same shape and `dtype` as `matrix`.
  """
  with ops.op_scope([matrix], name, 'batch_matrix_diag_transform'):
    matrix = ops.convert_to_tensor(matrix, name='matrix')
    if transform is None:
      return matrix
    # Replace the diag with transformed diag.
    diag = array_ops.batch_matrix_diag_part(matrix)
    transformed_diag = transform(diag)
    matrix += array_ops.batch_matrix_diag(transformed_diag - diag)

  return matrix
