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
from tensorflow.contrib.distributions.python.ops import kullback_leibler
from tensorflow.contrib.distributions.python.ops import operator_pd_cholesky
from tensorflow.contrib.distributions.python.ops import operator_pd_diag
from tensorflow.contrib.distributions.python.ops import operator_pd_full
from tensorflow.contrib.distributions.python.ops import operator_pd_vdvt_update
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
    "MultivariateNormalDiag",
    "MultivariateNormalCholesky",
    "MultivariateNormalFull",
    "MultivariateNormalDiagPlusVDVT",
]


class MultivariateNormalOperatorPD(distribution.Distribution):
  """The multivariate normal distribution on `R^k`.

  This distribution is defined by a 1-D mean `mu` and an instance of
  `OperatorPDBase`, which provides access to a symmetric positive definite
  operator, which defines the covariance.

  #### Mathematical details

  With `C` the covariance matrix represented by the operator, the PDF of this
  distribution is:

  ```
  f(x) = (2 pi)^(-k/2) |det(C)|^(-1/2) exp(-1/2 (x - mu)^T C^{-1} (x - mu))
  ```

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

  def __init__(self,
               mu,
               cov,
               validate_args=True,
               allow_nan_stats=False,
               name="MultivariateNormalCov"):
    """Multivariate Normal distributions on `R^k`.

    User must provide means `mu`, and an instance of `OperatorPDBase`, `cov`,
    which determines the covariance.

    Args:
      mu: Floating point tensor with shape `[N1,...,Nb, k]`, `b >= 0`.
      cov: Instance of `OperatorPDBase` with same `dtype` as `mu` and shape
        `[N1,...,Nb, k, k]`.
      validate_args: Whether to validate input with asserts.  If `validate_args`
        is `False`, and the inputs are invalid, correct behavior is not
        guaranteed.
      allow_nan_stats:  `Boolean`, default `False`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      TypeError: If `mu` and `cov` are different dtypes.
    """
    self._allow_nan_stats = allow_nan_stats
    self._validate_args = validate_args
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

    # Try to validate with static checks.
    mu_shape = mu.get_shape()
    cov_shape = cov.get_shape()
    if mu_shape.is_fully_defined() and cov_shape.is_fully_defined():
      if mu_shape != cov_shape[:-1]:
        raise ValueError(
            "mu.shape and cov.shape[:-1] should match.  Found: mu.shape=%s, "
            "cov.shape=%s" % (mu_shape, cov_shape))
      else:
        return mu

    # Static checks could not be run, so possibly do dynamic checks.
    if not self.validate_args:
      return mu
    else:
      assert_same_rank = check_ops.assert_equal(
          array_ops.rank(mu) + 1,
          cov.rank(),
          data=["mu should have rank 1 less than cov.  Found: rank(mu) = ",
                array_ops.rank(mu), " rank(cov) = ", cov.rank()],
      )
      with ops.control_dependencies([assert_same_rank]):
        assert_same_shape = check_ops.assert_equal(
            array_ops.shape(mu),
            cov.vector_shape(),
            data=["mu.shape and cov.shape[:-1] should match.  "
                  "Found: shape(mu) = "
                  , array_ops.shape(mu), " shape(cov) = ", cov.shape()],
        )
        return control_flow_ops.with_dependencies([assert_same_shape], mu)

  @property
  def validate_args(self):
    """`Boolean` describing behavior on invalid input."""
    return self._validate_args

  @property
  def allow_nan_stats(self):
    """`Boolean` describing behavior when stats are undefined."""
    return self._allow_nan_stats

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

  def sample_n(self, n, seed=None, name="sample_n"):
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


class MultivariateNormalDiag(MultivariateNormalOperatorPD):
  """The multivariate normal distribution on `R^k`.

  This distribution is defined by a 1-D mean `mu` and a 1-D diagonal
  `diag_stdev`, representing the standard deviations.  This distribution
  assumes the random variables, `(X_1,...,X_k)` are independent, thus no
  non-diagonal terms of the covariance matrix are needed.

  This allows for `O(k)` pdf evaluation, sampling, and storage.

  #### Mathematical details

  The PDF of this distribution is defined in terms of the diagonal covariance
  determined by `diag_stdev`: `C_{ii} = diag_stdev[i]**2`.

  ```
  f(x) = (2 pi)^(-k/2) |det(C)|^(-1/2) exp(-1/2 (x - mu)^T C^{-1} (x - mu))
  ```

  #### Examples

  A single multi-variate Gaussian distribution is defined by a vector of means
  of length `k`, and the square roots of the (independent) random variables.

  Extra leading dimensions, if provided, allow for batches.

  ```python
  # Initialize a single 3-variate Gaussian with diagonal standard deviation.
  mu = [1, 2, 3.]
  diag_stdev = [4, 5, 6.]
  dist = tf.contrib.distributions.MultivariateNormalDiag(mu, diag_stdev)

  # Evaluate this on an observation in R^3, returning a scalar.
  dist.pdf([-1, 0, 1])

  # Initialize a batch of two 3-variate Gaussians.
  mu = [[1, 2, 3], [11, 22, 33]]  # shape 2 x 3
  diag_stdev = ...  # shape 2 x 3, positive.
  dist = tf.contrib.distributions.MultivariateNormalDiag(mu, diag_stdev)

  # Evaluate this on a two observations, each in R^3, returning a length two
  # tensor.
  x = [[-1, 0, 1], [-11, 0, 11]]  # Shape 2 x 3.
  dist.pdf(x)
  ```

  """

  def __init__(
      self,
      mu,
      diag_stdev,
      validate_args=True,
      allow_nan_stats=False,
      name="MultivariateNormalDiag"):
    """Multivariate Normal distributions on `R^k`.

    User must provide means `mu` and standard deviations `diag_stdev`.
    Each batch member represents a random vector `(X_1,...,X_k)` of independent
    random normals.
    The mean of `X_i` is `mu[i]`, and the standard deviation is `diag_stdev[i]`.

    Args:
      mu:  Rank `N + 1` floating point tensor with shape `[N1,...,Nb, k]`,
        `b >= 0`.
      diag_stdev: Rank `N + 1` `Tensor` with same `dtype` and shape as `mu`,
        representing the standard deviations.  Must be positive.
      validate_args: Whether to validate input with asserts.  If `validate_args`
        is `False`,
        and the inputs are invalid, correct behavior is not guaranteed.
      allow_nan_stats:  `Boolean`, default `False`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      TypeError: If `mu` and `diag_stdev` are different dtypes.
    """
    cov = operator_pd_diag.OperatorPDSqrtDiag(
        diag_stdev, verify_pd=validate_args)
    super(MultivariateNormalDiag, self).__init__(
        mu, cov, allow_nan_stats=allow_nan_stats, validate_args=validate_args,
        name=name)


class MultivariateNormalDiagPlusVDVT(MultivariateNormalOperatorPD):
  """The multivariate normal distribution on `R^k`.

  Every batch member of this distribution is defined by a mean and a lightweight
  covariance matrix `C`.

  #### Mathematical details

  The PDF of this distribution in terms of the mean `mu` and covariance `C` is:

  ```
  f(x) = (2 pi)^(-k/2) |det(C)|^(-1/2) exp(-1/2 (x - mu)^T C^{-1} (x - mu))
  ```

  For every batch member, this distribution represents `k` random variables
  `(X_1,...,X_k)`, with mean `E[X_i] = mu[i]`, and covariance matrix
  `C_{ij} := E[(X_i - mu[i])(X_j - mu[j])]`

  The user initializes this class by providing the mean `mu`, and a lightweight
  definition of `C`:

  ```
  C = SS^T = SS = (M + V D V^T) (M + V D V^T)
  M is diagonal (k x k)
  V = is shape (k x r), typically r << k
  D = is diagonal (r x r), optional (defaults to identity).
  ```

  This allows for `O(kr + r^3)` pdf evaluation and determinant, and `O(kr)`
  sampling and storage (per batch member).

  #### Examples

  A single multi-variate Gaussian distribution is defined by a vector of means
  of length `k`, and square root of the covariance `S = M + V D V^T`.  Extra
  leading dimensions, if provided, allow for batches.

  ```python
  # Initialize a single 3-variate Gaussian with covariance square root
  # S = M + V D V^T, where V D V^T is a matrix-rank 2 update.
  mu = [1, 2, 3.]
  diag_large = [1.1, 2.2, 3.3]
  v = ... # shape 3 x 2
  diag_small = [4., 5.]
  dist = tf.contrib.distributions.MultivariateNormalDiagPlusVDVT(
      mu, diag_large, v, diag_small=diag_small)

  # Evaluate this on an observation in R^3, returning a scalar.
  dist.pdf([-1, 0, 1])

  # Initialize a batch of two 3-variate Gaussians.  This time, don't provide
  # diag_small.  This means S = M + V V^T.
  mu = [[1, 2, 3], [11, 22, 33]]  # shape 2 x 3
  diag_large = ... # shape 2 x 3
  v = ... # shape 2 x 3 x 1, a matrix-rank 1 update.
  dist = tf.contrib.distributions.MultivariateNormalDiagPlusVDVT(
      mu, diag_large, v)

  # Evaluate this on a two observations, each in R^3, returning a length two
  # tensor.
  x = [[-1, 0, 1], [-11, 0, 11]]  # Shape 2 x 3.
  dist.pdf(x)
  ```

  """

  def __init__(
      self,
      mu,
      diag_large,
      v,
      diag_small=None,
      validate_args=True,
      allow_nan_stats=False,
      name="MultivariateNormalDiagPlusVDVT"):
    """Multivariate Normal distributions on `R^k`.

    For every batch member, this distribution represents `k` random variables
    `(X_1,...,X_k)`, with mean `E[X_i] = mu[i]`, and covariance matrix
    `C_{ij} := E[(X_i - mu[i])(X_j - mu[j])]`

    The user initializes this class by providing the mean `mu`, and a
    lightweight definition of `C`:

    ```
    C = SS^T = SS = (M + V D V^T) (M + V D V^T)
    M is diagonal (k x k)
    V = is shape (k x r), typically r << k
    D = is diagonal (r x r), optional (defaults to identity).
    ```

    Args:
      mu:  Rank `n + 1` floating point tensor with shape `[N1,...,Nn, k]`,
        `n >= 0`.  The means.
      diag_large:  Optional rank `n + 1` floating point tensor, shape
        `[N1,...,Nn, k]` `n >= 0`.  Defines the diagonal matrix `M`.
      v:  Rank `n + 1` floating point tensor, shape `[N1,...,Nn, k, r]`
        `n >= 0`.  Defines the matrix `V`.
      diag_small:  Rank `n + 1` floating point tensor, shape
        `[N1,...,Nn, k]` `n >= 0`.  Defines the diagonal matrix `D`.  Default
        is `None`, which means `D` will be the identity matrix.
      validate_args: Whether to validate input with asserts.  If `validate_args`
        is `False`,
        and the inputs are invalid, correct behavior is not guaranteed.
      allow_nan_stats:  `Boolean`, default `False`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.
    """
    m = operator_pd_diag.OperatorPDDiag(diag_large, verify_pd=validate_args)
    cov = operator_pd_vdvt_update.OperatorPDSqrtVDVTUpdate(
        m, v, diag=diag_small, verify_pd=validate_args,
        verify_shapes=validate_args)
    super(MultivariateNormalDiagPlusVDVT, self).__init__(
        mu, cov, allow_nan_stats=allow_nan_stats, validate_args=validate_args,
        name=name)


class MultivariateNormalCholesky(MultivariateNormalOperatorPD):
  """The multivariate normal distribution on `R^k`.

  This distribution is defined by a 1-D mean `mu` and a Cholesky factor `chol`.
  Providing the Cholesky factor allows for `O(k^2)` pdf evaluation and sampling,
  and requires `O(k^2)` storage.

  #### Mathematical details

  The Cholesky factor `chol` defines the covariance matrix: `C = chol chol^T`.

  The PDF of this distribution is then:

  ```
  f(x) = (2 pi)^(-k/2) |det(C)|^(-1/2) exp(-1/2 (x - mu)^T C^{-1} (x - mu))
  ```

  #### Examples

  A single multi-variate Gaussian distribution is defined by a vector of means
  of length `k`, and a covariance matrix of shape `k x k`.

  Extra leading dimensions, if provided, allow for batches.

  ```python
  # Initialize a single 3-variate Gaussian with diagonal covariance.
  # Note, this would be more efficient with MultivariateNormalDiag.
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

  def __init__(self,
               mu,
               chol,
               validate_args=True,
               allow_nan_stats=False,
               name="MultivariateNormalCholesky"):
    """Multivariate Normal distributions on `R^k`.

    User must provide means `mu` and `chol` which holds the (batch) Cholesky
    factors, such that the covariance of each batch member is `chol chol^T`.

    Args:
      mu: `(N+1)-D` floating point tensor with shape `[N1,...,Nb, k]`,
        `b >= 0`.
      chol: `(N+2)-D` `Tensor` with same `dtype` as `mu` and shape
        `[N1,...,Nb, k, k]`.  The upper triangular part is ignored (treated as
        though it is zero), and the diagonal must be positive.
      validate_args: Whether to validate input with asserts.  If `validate_args`
        is `False`, and the inputs are invalid, correct behavior is not
        guaranteed.
      allow_nan_stats:  `Boolean`, default `False`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      TypeError: If `mu` and `chol` are different dtypes.
    """
    cov = operator_pd_cholesky.OperatorPDCholesky(chol, verify_pd=validate_args)
    super(MultivariateNormalCholesky, self).__init__(
        mu,
        cov,
        allow_nan_stats=allow_nan_stats,
        validate_args=validate_args,
        name=name)


class MultivariateNormalFull(MultivariateNormalOperatorPD):
  """The multivariate normal distribution on `R^k`.

  This distribution is defined by a 1-D mean `mu` and covariance matrix `sigma`.
  Evaluation of the pdf, determinant, and sampling are all `O(k^3)` operations.

  #### Mathematical details

  With `C = sigma`, the PDF of this distribution is:

  ```
  f(x) = (2 pi)^(-k/2) |det(C)|^(-1/2) exp(-1/2 (x - mu)^T C^{-1} (x - mu))
  ```

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

  def __init__(self,
               mu,
               sigma,
               validate_args=True,
               allow_nan_stats=False,
               name="MultivariateNormalFull"):
    """Multivariate Normal distributions on `R^k`.

    User must provide means `mu` and `sigma`, the mean and covariance.

    Args:
      mu: `(N+1)-D` floating point tensor with shape `[N1,...,Nb, k]`,
        `b >= 0`.
      sigma: `(N+2)-D` `Tensor` with same `dtype` as `mu` and shape
        `[N1,...,Nb, k, k]`.  Each batch member must be positive definite.
      validate_args: Whether to validate input with asserts.  If `validate_args`
        is `False`, and the inputs are invalid, correct behavior is not
        guaranteed.
      allow_nan_stats:  `Boolean`, default `False`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      TypeError: If `mu` and `sigma` are different dtypes.
    """
    cov = operator_pd_full.OperatorPDFull(sigma, verify_pd=validate_args)
    super(MultivariateNormalFull, self).__init__(
        mu,
        cov,
        allow_nan_stats=allow_nan_stats,
        validate_args=validate_args,
        name=name)


def _kl_mvn_mvn_brute_force(mvn_a, mvn_b, name=None):
  """Batched KL divergence `KL(mvn_a || mvn_b)` for multivariate normals.

  With `X`, `Y` both multivariate normals in `R^k` with means `mu_x`, `mu_y` and
  covariance `C_x`, `C_y` respectively,

  ```
  KL(X || Y) = 0.5 * ( T + Q + - k + L ),
  T := trace(C_b^{-1} C_a),
  Q := (mu_b - mu_a)^T C_b^{-1} (mu_b - mu_a),
  L := Log[Det(C_b)] - Log[Det(C_a)]
  ```

  This `Op` computes the trace by solving `C_b^{-1} C_a`.  Although efficient
  methods for solving systems with `C_b` may be available, a dense version of
  (the square root of) `C_a` is used, so performance is `O(B s k^2)` where `B`
  is the batch size, and `s` is the cost of solving `C_b x = y` for vectors `x`
  and `y`.

  Args:
    mvn_a:  Instance of subclass of `MultivariateNormalOperatorPD`.
    mvn_b:  Instance of subclass of `MultivariateNormalOperatorPD`.
    name:  (optional) name to use for created ops.  Default "kl_mvn_mvn".

  Returns:
    Batchwise `KL(mvn_a || mvn_b)`.
  """
  # Access the "private" OperatorPD that each mvn is built from.
  cov_a = mvn_a._cov  # pylint: disable=protected-access
  cov_b = mvn_b._cov  # pylint: disable=protected-access
  mu_a = mvn_a.mu
  mu_b = mvn_b.mu
  inputs = [mu_a, mu_b] + cov_a.inputs + cov_b.inputs

  with ops.op_scope(inputs, name, "kl_mvn_mvn"):
    # If Ca = AA', Cb = BB', then
    # tr[inv(Cb) Ca] = tr[inv(B)' inv(B) A A']
    #                = tr[inv(B) A A' inv(B)']
    #                = tr[(inv(B) A) (inv(B) A)']
    #                = sum_{ik} (inv(B) A)_{ik}^2
    # The second equality follows from the cyclic permutation property.
    b_inv_a = cov_b.sqrt_solve(cov_a.sqrt_to_dense())
    t = math_ops.reduce_sum(
        math_ops.square(b_inv_a),
        reduction_indices=[-1, -2])
    q = cov_b.inv_quadratic_form_on_vectors(mu_b - mu_a)
    k = math_ops.cast(cov_a.vector_space_dimension(), mvn_a.dtype)
    one_half_l = cov_b.sqrt_log_det() - cov_a.sqrt_log_det()
    return 0.5 * (t + q - k) + one_half_l


# Register KL divergences.
kl_classes = [
    MultivariateNormalFull,
    MultivariateNormalCholesky,
    MultivariateNormalDiag,
    MultivariateNormalDiagPlusVDVT,
]


for mvn_aa in kl_classes:
  # Register when they are the same here, and do not register when they are the
  # same below because that would result in a repeated registration.
  kullback_leibler.RegisterKL(mvn_aa, mvn_aa)(_kl_mvn_mvn_brute_force)
  for mvn_bb in kl_classes:
    if mvn_bb != mvn_aa:
      kullback_leibler.RegisterKL(mvn_aa, mvn_bb)(_kl_mvn_mvn_brute_force)
