# Copyright 2016 Google Inc. All Rights Reserved.
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
"""The Multivariate Normal distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util  # pylint: disable=line-too-long
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


def _assert_compatible_shapes(mu, sigma):
  r_mu = array_ops.rank(mu)
  r_sigma = array_ops.rank(sigma)
  sigma_shape = array_ops.shape(sigma)
  sigma_rank = array_ops.rank(sigma)
  mu_shape = array_ops.shape(mu)
  return control_flow_ops.group(
      logging_ops.Assert(
          math_ops.equal(r_mu + 1, r_sigma),
          ["Rank of mu should be one less than rank of sigma, but saw: ",
           r_mu, " vs. ", r_sigma]),
      logging_ops.Assert(
          math_ops.equal(
              array_ops.gather(sigma_shape, sigma_rank - 2),
              array_ops.gather(sigma_shape, sigma_rank - 1)),
          ["Last two dimensions of sigma (%s) must be equal: " % sigma.name,
           sigma_shape]),
      logging_ops.Assert(
          math_ops.reduce_all(math_ops.equal(
              mu_shape,
              array_ops.slice(
                  sigma_shape, [0], array_ops.pack([sigma_rank - 1])))),
          ["mu.shape and sigma.shape[:-1] must match, but saw: ",
           mu_shape, " vs. ", sigma_shape]))


def _assert_batch_positive_definite(sigma_chol):
  """Add assertions checking that the sigmas are all Positive Definite.

  Given `sigma_chol == cholesky(sigma)`, it is sufficient to check that
  `all(diag(sigma_chol) > 0)`.  This is because to check that a matrix is PD,
  it is sufficient that its cholesky factorization is PD, and to check that a
  triangular matrix is PD, it is sufficient to check that its diagonal
  entries are positive.

  Args:
    sigma_chol: N-D.  The lower triangular cholesky decomposition of `sigma`.

  Returns:
    An assertion op to use with `control_dependencies`, verifying that
    `sigma_chol` is positive definite.
  """
  sigma_batch_diag = array_ops.batch_matrix_diag_part(sigma_chol)
  return logging_ops.Assert(
      math_ops.reduce_all(sigma_batch_diag > 0),
      ["sigma_chol is not positive definite.  batched diagonals: ",
       sigma_batch_diag, " shaped: ", array_ops.shape(sigma_batch_diag)])


def _determinant_from_sigma_chol(sigma_chol):
  det_last_dim = array_ops.rank(sigma_chol) - 2
  sigma_batch_diag = array_ops.batch_matrix_diag_part(sigma_chol)
  det = math_ops.square(math_ops.reduce_prod(
      sigma_batch_diag, reduction_indices=det_last_dim))
  det.set_shape(sigma_chol.get_shape()[:-2])
  return det


class MultivariateNormal(object):
  """The Multivariate Normal distribution on `R^k`.

  The distribution has mean and covariance parameters mu (1-D), sigma (2-D),
  or alternatively mean `mu` and factored covariance (cholesky decomposed
  `sigma`) called `sigma_chol`.

  #### Mathematical details

  The PDF of this distribution is:

  ```
  f(x) = (2*pi)^(-k/2) |det(sigma)|^(-1/2) exp(-1/2*(x-mu)^*.sigma^{-1}.(x-mu))
  ```

  where `.` denotes the inner product on `R^k` and `^*` denotes transpose.

  Alternatively, if `sigma` is positive definite, it can be represented in terms
  of its lower triangular cholesky factorization

  ```sigma = sigma_chol . sigma_chol^*```

  and the pdf above allows simpler computation:

  ```
  |det(sigma)| = reduce_prod(diag(sigma_chol))^2
  x_whitened = sigma^{-1/2} . (x - mu) = tri_solve(sigma_chol, x - mu)
  (x-mu)^* .sigma^{-1} . (x-mu) = x_whitened^* . x_whitened
  ```

  where `tri_solve()` solves a triangular system of equations.

  #### Examples

  A single multi-variate Gaussian distribution is defined by a vector of means
  of length `k`, and a covariance matrix of shape `k x k`.

  Extra leading dimensions, if provided, allow for batches.

  ```python
  # Initialize a single 3-variate Gaussian with diagonal covariance.
  mu = [1, 2, 3]
  sigma = [[1, 0, 0], [0, 3, 0], [0, 0, 2]]
  dist = tf.contrib.distributions.MultivariateNormal(mu=mu, sigma=sigma)

  # Evaluate this on an observation in R^3, returning a scalar.
  dist.pdf([-1, 0, 1])

  # Initialize a batch of two 3-variate Gaussians.
  mu = [[1, 2, 3], [11, 22, 33]]
  sigma = ...  # shape 2 x 3 x 3
  dist = tf.contrib.distributions.MultivariateNormal(mu=mu, sigma=sigma)

  # Evaluate this on a two observations, each in R^3, returning a length two
  # tensor.
  x = [[-1, 0, 1], [-11, 0, 11]]  # Shape 2 x 3.
  dist.pdf(x)
  ```

  """

  def __init__(self, mu, sigma=None, sigma_chol=None, name=None):
    """Multivariate Normal distributions on `R^k`.

    User must provide means `mu`, which are tensors of rank `N+1` (`N >= 0`)
    with the last dimension having length `k`.

    User must provide exactly one of `sigma` (the covariance matrices) or
    `sigma_chol` (the cholesky decompositions of the covariance matrices).
    `sigma` or `sigma_chol` must be of rank `N+2`.  The last two dimensions
    must both have length `k`.  The first `N` dimensions correspond to batch
    indices.

    If `sigma_chol` is not provided, the batch cholesky factorization of `sigma`
    is calculated for you.

    The shapes of `mu` and `sigma` must match for the first `N` dimensions.

    Regardless of which parameter is provided, the covariance matrices must all
    be **positive definite** (an error is raised if one of them is not).

    Args:
      mu: (N+1)-D.  `float` or `double` tensor, the means of the distributions.
      sigma: (N+2)-D.  (optional) `float` or `double` tensor, the covariances
        of the distribution(s).  The first `N+1` dimensions must match
        those of `mu`.  Must be batch-positive-definite.
      sigma_chol: (N+2)-D.  (optional) `float` or `double` tensor, a
        lower-triangular factorization of `sigma`
        (`sigma = sigma_chol . sigma_chol^*`).  The first `N+1` dimensions
        must match those of `mu`.  The tensor itself need not be batch
        lower triangular: we ignore the upper triangular part.  However,
        the batch diagonals must be positive (i.e., sigma_chol must be
        batch-positive-definite).
      name: The name to give Ops created by the initializer.

    Raises:
      ValueError: if neither sigma nor sigma_chol is provided.
      TypeError: if mu and sigma (resp. sigma_chol) are different dtypes.
    """
    if (sigma is None) == (sigma_chol is None):
      raise ValueError("Exactly one of sigma and sigma_chol must be provided")

    with ops.op_scope([mu, sigma, sigma_chol], name, "MultivariateNormal"):
      sigma_or_half = sigma_chol if sigma is None else sigma

      mu = ops.convert_to_tensor(mu)
      sigma_or_half = ops.convert_to_tensor(sigma_or_half)

      contrib_tensor_util.assert_same_float_dtype((mu, sigma_or_half))

      with ops.control_dependencies([
          _assert_compatible_shapes(mu, sigma_or_half)]):
        mu = array_ops.identity(mu, name="mu")

        # Store the dimensionality of the MVNs
        self._k = array_ops.gather(array_ops.shape(mu), array_ops.rank(mu) - 1)

        if sigma_chol is not None:
          # Ensure we only keep the lower triangular part.
          sigma_chol = array_ops.batch_matrix_band_part(
              sigma_chol, num_lower=-1, num_upper=0)
          sigma_det = _determinant_from_sigma_chol(sigma_chol)
          with ops.control_dependencies([
              _assert_batch_positive_definite(sigma_chol)]):
            self._sigma = math_ops.batch_matmul(
                sigma_chol, sigma_chol, adj_y=True, name="sigma")
            self._sigma_chol = array_ops.identity(sigma_chol, "sigma_chol")
            self._sigma_det = array_ops.identity(sigma_det, "sigma_det")
            self._mu = array_ops.identity(mu, "mu")
        else:  # sigma is not None
          sigma_chol = linalg_ops.batch_cholesky(sigma)
          sigma_det = _determinant_from_sigma_chol(sigma_chol)
          # batch_cholesky checks for PSD; so we can just use it here.
          with ops.control_dependencies([sigma_chol]):
            self._sigma = array_ops.identity(sigma, "sigma")
            self._sigma_chol = array_ops.identity(sigma_chol, "sigma_chol")
            self._sigma_det = array_ops.identity(sigma_det, "sigma_det")
            self._mu = array_ops.identity(mu, "mu")

  @property
  def dtype(self):
    return self._mu.dtype

  @property
  def mu(self):
    return self._mu

  @property
  def sigma(self):
    return self._sigma

  @property
  def mean(self):
    return self._mu

  @property
  def sigma_det(self):
    return self._sigma_det

  def log_pdf(self, x, name=None):
    """Log pdf of observations `x` given these Multivariate Normals.

    Args:
      x: tensor of dtype `dtype`, must be broadcastable with `mu`.
      name: The name to give this op.

    Returns:
      log_pdf: tensor of dtype `dtype`, the log-PDFs of `x`.
    """
    with ops.op_scope(
        [self._mu, self._sigma_chol, x], name, "MultivariateNormalLogPdf"):
      x = ops.convert_to_tensor(x)
      contrib_tensor_util.assert_same_float_dtype((self._mu, x))

      x_centered = x - self.mu

      x_rank = array_ops.rank(x_centered)
      sigma_rank = array_ops.rank(self._sigma_chol)

      x_rank_vec = array_ops.pack([x_rank])
      sigma_rank_vec = array_ops.pack([sigma_rank])
      x_shape = array_ops.shape(x_centered)

      # sigma_chol is shaped [D, E, F, ..., k, k]
      # x_centered shape is one of:
      #   [D, E, F, ..., k], or [F, ..., k], or
      #   [A, B, C, D, E, F, ..., k]
      # and we need to convert x_centered to shape:
      #   [D, E, F, ..., k, A*B*C] (or 1 if A, B, C don't exist)
      # then transpose and reshape x_whitened back to one of the shapes:
      #   [D, E, F, ..., k], or [1, 1, F, ..., k], or
      #   [A, B, C, D, E, F, ..., k]

      # This helper handles the case where rank(x_centered) < rank(sigma)
      def _broadcast_x_not_higher_rank_than_sigma():
        return array_ops.reshape(
            x_centered,
            array_ops.concat(
                # Reshape to ones(deficient x rank) + x_shape + [1]
                0, (array_ops.ones(array_ops.pack([sigma_rank - x_rank - 1]),
                                   dtype=x_rank.dtype),
                    x_shape,
                    [1])))

      # These helpers handle the case where rank(x_centered) >= rank(sigma)
      def _broadcast_x_higher_rank_than_sigma():
        x_shape_left = array_ops.slice(
            x_shape, [0], sigma_rank_vec - 1)
        x_shape_right = array_ops.slice(
            x_shape, sigma_rank_vec - 1, x_rank_vec - 1)
        x_shape_perm = array_ops.concat(
            0, (math_ops.range(sigma_rank - 1, x_rank),
                math_ops.range(0, sigma_rank - 1)))
        return array_ops.reshape(
            # Convert to [D, E, F, ..., k, B, C]
            array_ops.transpose(
                x_centered, perm=x_shape_perm),
            # Reshape to [D, E, F, ..., k, B*C]
            array_ops.concat(
                0, (x_shape_right,
                    array_ops.pack([
                        math_ops.reduce_prod(x_shape_left, 0)]))))

      def _unbroadcast_x_higher_rank_than_sigma():
        x_shape_left = array_ops.slice(
            x_shape, [0], sigma_rank_vec - 1)
        x_shape_right = array_ops.slice(
            x_shape, sigma_rank_vec - 1, x_rank_vec - 1)
        x_shape_perm = array_ops.concat(
            0, (math_ops.range(sigma_rank - 1, x_rank),
                math_ops.range(0, sigma_rank - 1)))
        return array_ops.transpose(
            # [D, E, F, ..., k, B, C] => [B, C, D, E, F, ..., k]
            array_ops.reshape(
                # convert to [D, E, F, ..., k, B, C]
                x_whitened_broadcast,
                array_ops.concat(0, (x_shape_right, x_shape_left))),
            perm=x_shape_perm)

      # Step 1: reshape x_centered
      x_centered_broadcast = control_flow_ops.cond(
          # x_centered == [D, E, F, ..., k] => [D, E, F, ..., k, 1]
          # or         == [F, ..., k] => [1, 1, F, ..., k, 1]
          x_rank <= sigma_rank - 1,
          _broadcast_x_not_higher_rank_than_sigma,
          # x_centered == [B, C, D, E, F, ..., k] => [D, E, F, ..., k, B*C]
          _broadcast_x_higher_rank_than_sigma)

      x_whitened_broadcast = linalg_ops.batch_matrix_triangular_solve(
          self._sigma_chol, x_centered_broadcast)

      # Reshape x_whitened_broadcast back to x_whitened
      x_whitened = control_flow_ops.cond(
          x_rank <= sigma_rank - 1,
          lambda: array_ops.reshape(x_whitened_broadcast, x_shape),
          _unbroadcast_x_higher_rank_than_sigma)

      x_whitened = array_ops.expand_dims(x_whitened, -1)
      # Reshape x_whitened to contain row vectors
      # Returns a batchwise scalar
      x_whitened_norm = math_ops.batch_matmul(
          x_whitened, x_whitened, adj_x=True)
      x_whitened_norm = control_flow_ops.cond(
          x_rank <= sigma_rank - 1,
          lambda: array_ops.squeeze(x_whitened_norm, [-2, -1]),
          lambda: array_ops.squeeze(x_whitened_norm, [-1]))

      log_two_pi = constant_op.constant(math.log(2 * math.pi), dtype=self.dtype)
      k = math_ops.cast(self._k, self.dtype)
      log_pdf_value = (
          -math_ops.log(self._sigma_det) -k * log_two_pi - x_whitened_norm) / 2
      final_shaped_value = control_flow_ops.cond(
          x_rank <= sigma_rank - 1,
          lambda: log_pdf_value,
          lambda: array_ops.squeeze(log_pdf_value, [-1]))

      output_static_shape = x_centered.get_shape()[:-1]
      final_shaped_value.set_shape(output_static_shape)
      return final_shaped_value

  def pdf(self, x, name=None):
    """The PDF of observations `x` under these Multivariate Normals.

    Args:
      x: tensor of dtype `dtype`, must be broadcastable with `mu` and `sigma`.
      name: The name to give this op.

    Returns:
      pdf: tensor of dtype `dtype`, the pdf values of `x`.
    """
    with ops.op_scope(
        [self._mu, self._sigma_chol, x], name, "MultivariateNormalPdf"):
      return math_ops.exp(self.log_pdf(x))

  def entropy(self, name=None):
    """The entropies of these Multivariate Normals.

    Args:
      name: The name to give this op.

    Returns:
      entropy: tensor of dtype `dtype`, the entropies.
    """
    with ops.op_scope(
        [self._mu, self._sigma_chol], name, "MultivariateNormalEntropy"):
      one_plus_log_two_pi = constant_op.constant(
          1 + math.log(2 * math.pi), dtype=self.dtype)

      # Use broadcasting rules to calculate the full broadcast sigma.
      k = math_ops.cast(self._k, dtype=self.dtype)
      entropy_value = (
          k * one_plus_log_two_pi + math_ops.log(self._sigma_det)) / 2
      entropy_value.set_shape(self._sigma_det.get_shape())
      return entropy_value

  def sample(self, n, seed=None, name=None):
    """Sample `n` observations from the Multivariate Normal Distributions.

    Args:
      n: `Scalar`, type int32, the number of observations to sample.
      seed: Python integer, the random seed.
      name: The name to give this op.

    Returns:
      samples: `[n, ...]`, a `Tensor` of `n` samples for each
        of the distributions determined by broadcasting the hyperparameters.
    """
    with ops.op_scope(
        [self._mu, self._sigma_chol, n], name, "MultivariateNormalSample"):
      # TODO(ebrevdo): Is there a better way to get broadcast_shape?
      broadcast_shape = self.mu.get_shape()
      n = ops.convert_to_tensor(n)
      sigma_shape_left = array_ops.slice(
          array_ops.shape(self._sigma_chol),
          [0], array_ops.pack([array_ops.rank(self._sigma_chol) - 2]))

      k_n = array_ops.pack([self._k, n])
      shape = array_ops.concat(0, [sigma_shape_left, k_n])
      white_samples = random_ops.random_normal(
          shape=shape, mean=0, stddev=1, dtype=self._mu.dtype, seed=seed)

      correlated_samples = math_ops.batch_matmul(
          self._sigma_chol, white_samples)

      # Move the last dimension to the front
      perm = array_ops.concat(
          0,
          (array_ops.pack([array_ops.rank(correlated_samples) - 1]),
           math_ops.range(0, array_ops.rank(correlated_samples) - 1)))

      # TODO(ebrevdo): Once we get a proper tensor contraction op,
      # perform the inner product using that instead of batch_matmul
      # and this slow transpose can go away!
      correlated_samples = array_ops.transpose(correlated_samples, perm)

      samples = correlated_samples + self.mu

      # Provide some hints to shape inference
      n_val = tensor_util.constant_value(n)
      final_shape = tensor_shape.vector(n_val).concatenate(broadcast_shape)
      samples.set_shape(final_shape)

      return samples
