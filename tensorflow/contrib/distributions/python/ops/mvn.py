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
"""Multivariate Normal distribution classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import operator_pd_cholesky
from tensorflow.contrib.distributions.python.ops import operator_pd_full
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


__all__ = [
    "MultivariateNormalCholesky",
    "MultivariateNormalFull",
]


class MultivariateNormalOperatorPD(distribution.Distribution):
  """The multivariate normal distribution on `R^k`.

  This distribution is defined by a 1-D mean `mu` and an instance of
  `OperatorPDBase`, which provides access to a symmetric positive definite
  operator, which defines the covariance.

  #### Mathematical details

  The PDF of this distribution is:

  ```
  f(x) = (2*pi)^(-k/2) |det(sigma)|^(-1/2) exp(-1/2*(x-mu)^*.sigma^{-1}.(x-mu))
  ```

  where `.` denotes the inner product on `R^k` and `^*` denotes transpose.

  #### Examples

  A single multi-variate Gaussian distribution is defined by a vector of means
  of length `k`, and a covariance matrix of shape `k x k`.

  Extra leading dimensions, if provided, allow for batches.

  ```python
  # Initialize a single 3-variate Gaussian.
  mu = [1, 2, 3]
  chol = [[1, 0, 0.], [1, 3, 0], [1, 2, 3]]
  cov = tf.contrib.distributions.OperatorPDCholesky(chol)
  dist = tf.contrib.distributions.MultivariateNormalOperatorPD(mu, cov)

  # Evaluate this on an observation in R^3, returning a scalar.
  dist.pdf([-1, 0, 1.])

  # Initialize a batch of two 3-variate Gaussians.
  mu = [[1, 2, 3], [11, 22, 33.]]
  chol = ...  # shape 2 x 3 x 3, lower triangular, positive diagonal.
  cov = tf.contrib.distributions.OperatorPDCholesky(chol)
  dist = tf.contrib.distributions.MultivariateNormalOperatorPD(mu, cov)

  # Evaluate this on a two observations, each in R^3, returning a length two
  # tensor.
  x = [[-1, 0, 1], [-11, 0, 11.]]  # Shape 2 x 3.
  dist.pdf(x)
  ```

  """

  def __init__(
      self,
      mu,
      cov,
      allow_nan=False,
      strict=True,
      strict_statistics=True,
      name="MultivariateNormalCov"):
    """Multivariate Normal distributions on `R^k`.

    User must provide means `mu`, and an instance of `OperatorPDBase`, `cov`,
    which determines the covariance.

    Args:
      mu: `float` or `double` tensor with shape `[N1,...,Nb, k]`, `b >= 0`.
      cov: `float` or `double` instance of `OperatorPDBase` with same `dtype`
        as `mu` and shape `[N1,...,Nb, k, k]`.
      allow_nan:  Boolean, default False.  If False, raise an exception if
        a statistic (e.g. mean/mode/etc...) is undefined for any batch member.
        If True, batch members with valid parameters leading to undefined
        statistics will return NaN for this statistic.
      strict: Whether to validate input with asserts.  If `strict` is `False`,
        and the inputs are invalid, correct behavior is not guaranteed.
      strict_statistics:  Boolean, default True.  If True, raise an exception if
        a statistic (e.g. mean/mode/etc...) is undefined for any batch member.
        If False, batch members with valid parameters leading to undefined
        statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      TypeError: If `mu` and `cov` are different dtypes.
    """
    self._strict_statistics = strict_statistics
    self._strict = strict
    with ops.name_scope(name):
      with ops.op_scope([mu] + cov.inputs, "init"):
        self._cov = cov
        self._mu = self._check_mu(mu)
        self._name = name

  def _check_mu(self, mu):
    """Return `mu` after validity checks and possibly with assertations."""
    mu = ops.convert_to_tensor(mu)
    cov = self._cov

    if mu.dtype != cov.dtype:
      raise TypeError(
          "mu and cov must have the same dtype.  Found mu.dtype = %s, "
          "cov.dtype = %s"
          % (mu.dtype, cov.dtype))
    if not self.strict:
      return mu
    else:
      assert_compatible_shapes = control_flow_ops.group(
          check_ops.assert_equal(
              array_ops.rank(mu) + 1,
              cov.rank(),
              data=["mu should have rank 1 less than cov.  Found: rank(mu) = ",
                    array_ops.rank(mu), " rank(cov) = ", cov.rank()],
          ),
          check_ops.assert_equal(
              array_ops.shape(mu),
              cov.vector_shape(),
              data=["mu.shape and cov.shape[:-1] should match.  "
                    "Found: shape(mu) = "
                    , array_ops.shape(mu), " shape(cov) = ", cov.shape()],
          ),
      )
      return control_flow_ops.with_dependencies([assert_compatible_shapes], mu)

  @property
  def strict(self):
    """Boolean describing behavior on invalid input."""
    return self._strict

  @property
  def strict_statistics(self):
    """Boolean describing behavior when a stat is undefined for batch member."""
    return self._strict_statistics

  @property
  def dtype(self):
    return self._mu.dtype

  def get_event_shape(self):
    """`TensorShape` available at graph construction time."""
    # Recall _check_mu ensures mu and self._cov have same batch shape.
    return self._cov.get_shape()[-1:]

  def event_shape(self, name="event_shape"):
    """Shape of a sample from a single distribution as a 1-D int32 `Tensor`."""
    # Recall _check_mu ensures mu and self._cov have same batch shape.
    with ops.name_scope(self.name):
      with ops.op_scope(self._cov.inputs, name):
        return array_ops.pack([self._cov.vector_space_dimension()])

  def batch_shape(self, name="batch_shape"):
    """Batch dimensions of this instance as a 1-D int32 `Tensor`."""
    # Recall _check_mu ensures mu and self._cov have same batch shape.
    with ops.name_scope(self.name):
      with ops.op_scope(self._cov.inputs, name):
        return self._cov.batch_shape()

  def get_batch_shape(self):
    """`TensorShape` available at graph construction time."""
    # Recall _check_mu ensures mu and self._cov have same batch shape.
    return self._cov.get_batch_shape()

  @property
  def mu(self):
    return self._mu

  @property
  def sigma(self):
    """Dense (batch) covariance matrix, if available."""
    with ops.name_scope(self.name):
      return self._cov.to_dense()

  def mean(self, name="mean"):
    """Mean of each batch member."""
    with ops.name_scope(self.name):
      with ops.op_scope([self._mu], name):
        return array_ops.identity(self._mu)

  def mode(self, name="mode"):
    """Mode of each batch member."""
    with ops.name_scope(self.name):
      with ops.op_scope([self._mu], name):
        return array_ops.identity(self._mu)

  def variance(self, name="variance"):
    """Variance of each batch member."""
    with ops.name_scope(self.name):
      return self.sigma

  def log_sigma_det(self, name="log_sigma_det"):
    """Log of determinant of covariance matrix."""
    with ops.name_scope(self.name):
      with ops.op_scope(self._cov.inputs, name):
        return self._cov.log_det()

  def sigma_det(self, name="sigma_det"):
    """Determinant of covariance matrix."""
    with ops.name_scope(self.name):
      with ops.op_scope(self._cov.inputs, name):
        return math_ops.exp(self._cov.log_det())

  def log_prob(self, x, name="log_prob"):
    """Log prob of observations `x` given these Multivariate Normals.

    `x` is a batch vector with compatible shape if `x` is a `Tensor` whose
    shape can be broadcast up to either:

    ````
    self.batch_shape + self.event_shape
    OR
    [M1,...,Mm] + self.batch_shape + self.event_shape
    ```

    Args:
      x: Compatible batch vector with same `dtype` as this distribution.
      name: The name to give this op.

    Returns:
      log_prob: tensor of dtype `dtype`, the log-PDFs of `x`.
    """
    # Q:  Why are shape requirements as stated above?
    # A:  The compatible shapes are precisely the ones that will broadcast to
    #     a shape compatible with self._cov.
    # See Operator base class for notes about shapes compatible with self._cov.
    with ops.name_scope(self.name):
      with ops.op_scope([self._mu, x] + self._cov.inputs, name):
        x = ops.convert_to_tensor(x)
        contrib_tensor_util.assert_same_float_dtype((self._mu, x))

        # _check_mu asserts that self.mu has same batch shape as self.cov.
        # so batch shape of self.mu = that of self._cov and self, and the
        # batch shape of x_centered is a broadcast version of these.  If this
        # broadcast results in a shape like
        # [M1,...,Mm] + self.batch_shape + self.event_shape
        # OR
        # self.batch_shape + self.event_shape
        # then subsequent operator calls are guaranteed to work.
        x_centered = x - self.mu

        # Compute the term x^{-1} sigma^{-1} x which appears in the exponent of
        # the pdf.
        x_whitened_norm = self._cov.inv_quadratic_form_on_vectors(x_centered)

        log_sigma_det = self.log_sigma_det()

        log_two_pi = constant_op.constant(
            math.log(2 * math.pi), dtype=self.dtype)
        k = math_ops.cast(self._cov.vector_space_dimension(), self.dtype)
        log_prob_value = -(log_sigma_det + k * log_two_pi + x_whitened_norm) / 2

        output_static_shape = x_centered.get_shape()[:-1]
        log_prob_value.set_shape(output_static_shape)
        return log_prob_value

  def prob(self, x, name="prob"):
    """The PDF of observations `x` under these Multivariate Normals.

    `x` is a batch vector with compatible shape if `x` is a `Tensor` whose
    shape can be broadcast up to either:

    ````
    self.batch_shape + self.event_shape
    OR
    [M1,...,Mm] + self.batch_shape + self.event_shape
    ```

    Args:
      x: Compatible batch vector with same `dtype` as this distribution.
      name: The name to give this op.

    Returns:
      prob: tensor of dtype `dtype`, the prob values of `x`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._mu, x] + self._cov.inputs, name):
        return math_ops.exp(self.log_prob(x))

  def entropy(self, name="entropy"):
    """The entropies of these Multivariate Normals.

    Args:
      name: The name to give this op.

    Returns:
      entropy: tensor of dtype `dtype`, the entropies.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._mu] + self._cov.inputs, name):
        log_sigma_det = self.log_sigma_det()
        one_plus_log_two_pi = constant_op.constant(1 + math.log(2 * math.pi),
                                                   dtype=self.dtype)

        # Use broadcasting rules to calculate the full broadcast sigma.
        k = math_ops.cast(self._cov.vector_space_dimension(), dtype=self.dtype)
        entropy_value = (k * one_plus_log_two_pi + log_sigma_det) / 2
        entropy_value.set_shape(log_sigma_det.get_shape())
        return entropy_value

  def sample(self, n, seed=None, name="sample"):
    """Sample `n` observations from the Multivariate Normal Distributions.

    Args:
      n: `Scalar`, type int32, the number of observations to sample.
      seed: Python integer, the random seed.
      name: The name to give this op.

    Returns:
      samples: `[n, ...]`, a `Tensor` of `n` samples for each
        of the distributions determined by broadcasting the hyperparameters.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._mu, n] + self._cov.inputs, name):
        # Recall _check_mu ensures mu and self._cov have same batch shape.
        broadcast_shape = self.mu.get_shape()
        n = ops.convert_to_tensor(n)

        shape = array_ops.concat(0, [self._cov.vector_shape(), [n]])
        white_samples = random_ops.random_normal(shape=shape,
                                                 mean=0,
                                                 stddev=1,
                                                 dtype=self.dtype,
                                                 seed=seed)

        correlated_samples = self._cov.sqrt_matmul(white_samples)

        # Move the last dimension to the front
        perm = array_ops.concat(0, (
            array_ops.pack([array_ops.rank(correlated_samples) - 1]),
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

  @property
  def is_reparameterized(self):
    return True

  @property
  def name(self):
    return self._name

  @property
  def is_continuous(self):
    return True


class MultivariateNormalCholesky(MultivariateNormalOperatorPD):
  """The multivariate normal distribution on `R^k`.

  This distribution is defined by a 1-D mean `mu` and a Cholesky factor `chol`.
  Providing the Cholesky factor allows for `O(k^2)` pdf evaluation and sampling,
  and requires `O(k^2)` storage.

  #### Mathematical details

  The PDF of this distribution is:

  ```
  f(x) = (2*pi)^(-k/2) |det(sigma)|^(-1/2) exp(-1/2*(x-mu)^*.sigma^{-1}.(x-mu))
  ```

  where `.` denotes the inner product on `R^k` and `^*` denotes transpose.

  #### Examples

  A single multi-variate Gaussian distribution is defined by a vector of means
  of length `k`, and a covariance matrix of shape `k x k`.

  Extra leading dimensions, if provided, allow for batches.

  ```python
  # Initialize a single 3-variate Gaussian with diagonal covariance.
  mu = [1, 2, 3.]
  chol = [[1, 0, 0], [0, 3, 0], [0, 0, 2]]
  dist = tf.contrib.distributions.MultivariateNormalCholesky(mu, chol)

  # Evaluate this on an observation in R^3, returning a scalar.
  dist.pdf([-1, 0, 1])

  # Initialize a batch of two 3-variate Gaussians.
  mu = [[1, 2, 3], [11, 22, 33]]
  chol = ...  # shape 2 x 3 x 3, lower triangular, positive diagonal.
  dist = tf.contrib.distributions.MultivariateNormalCholesky(mu, chol)

  # Evaluate this on a two observations, each in R^3, returning a length two
  # tensor.
  x = [[-1, 0, 1], [-11, 0, 11]]  # Shape 2 x 3.
  dist.pdf(x)
  ```

  Trainable (batch) Choesky matrices can be created with
  `tf.contrib.distributions.batch_matrix_diag_transform()`

  """

  def __init__(
      self,
      mu,
      chol,
      strict=True,
      strict_statistics=True,
      name="MultivariateNormalCholesky"):
    """Multivariate Normal distributions on `R^k`.

    User must provide means `mu` and `chol` which holds the (batch) Cholesky
    factors `S`, such that the covariance of each batch member is `S S^*`.

    Args:
      mu: `(N+1)-D`  `float` or `double` tensor with shape `[N1,...,Nb, k]`,
        `b >= 0`.
      chol: `(N+2)-D` `Tensor` with same `dtype` as `mu` and shape
        `[N1,...,Nb, k, k]`.
      strict: Whether to validate input with asserts.  If `strict` is `False`,
        and the inputs are invalid, correct behavior is not guaranteed.
      strict_statistics:  Boolean, default True.  If True, raise an exception if
        a statistic (e.g. mean/mode/etc...) is undefined for any batch member.
        If False, batch members with valid parameters leading to undefined
        statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      TypeError: If `mu` and `chol` are different dtypes.
    """
    cov = operator_pd_cholesky.OperatorPDCholesky(chol, verify_pd=strict)
    super(MultivariateNormalCholesky, self).__init__(
        mu, cov, strict_statistics=strict_statistics, strict=strict, name=name)


class MultivariateNormalFull(MultivariateNormalOperatorPD):
  """The multivariate normal distribution on `R^k`.

  This distribution is defined by a 1-D mean `mu` and covariance matrix `sigma`.
  Evaluation of the pdf, determinant, and sampling are all `O(k^3)` operations.

  #### Mathematical details

  The PDF of this distribution is:

  ```
  f(x) = (2*pi)^(-k/2) |det(sigma)|^(-1/2) exp(-1/2*(x-mu)^*.sigma^{-1}.(x-mu))
  ```

  where `.` denotes the inner product on `R^k` and `^*` denotes transpose.

  #### Examples

  A single multi-variate Gaussian distribution is defined by a vector of means
  of length `k`, and a covariance matrix of shape `k x k`.

  Extra leading dimensions, if provided, allow for batches.

  ```python
  # Initialize a single 3-variate Gaussian with diagonal covariance.
  mu = [1, 2, 3.]
  sigma = [[1, 0, 0], [0, 3, 0], [0, 0, 2.]]
  dist = tf.contrib.distributions.MultivariateNormalFull(mu, chol)

  # Evaluate this on an observation in R^3, returning a scalar.
  dist.pdf([-1, 0, 1])

  # Initialize a batch of two 3-variate Gaussians.
  mu = [[1, 2, 3], [11, 22, 33.]]
  sigma = ...  # shape 2 x 3 x 3, positive definite.
  dist = tf.contrib.distributions.MultivariateNormalFull(mu, sigma)

  # Evaluate this on a two observations, each in R^3, returning a length two
  # tensor.
  x = [[-1, 0, 1], [-11, 0, 11.]]  # Shape 2 x 3.
  dist.pdf(x)
  ```

  """

  def __init__(
      self,
      mu,
      sigma,
      strict=True,
      strict_statistics=True,
      name="MultivariateNormalFull"):
    """Multivariate Normal distributions on `R^k`.

    User must provide means `mu` and `sigma`, the mean and covariance.

    Args:
      mu: `(N+1)-D`  `float` or `double` tensor with shape `[N1,...,Nb, k]`,
        `b >= 0`.
      sigma: `(N+2)-D` `Tensor` with same `dtype` as `mu` and shape
        `[N1,...,Nb, k, k]`.
      strict: Whether to validate input with asserts.  If `strict` is `False`,
        and the inputs are invalid, correct behavior is not guaranteed.
      strict_statistics:  Boolean, default True.  If True, raise an exception if
        a statistic (e.g. mean/mode/etc...) is undefined for any batch member.
        If False, batch members with valid parameters leading to undefined
        statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      TypeError: If `mu` and `sigma` are different dtypes.
    """
    cov = operator_pd_full.OperatorPDFull(sigma, verify_pd=strict)
    super(MultivariateNormalFull, self).__init__(
        mu, cov, strict_statistics=strict_statistics, strict=strict, name=name)
