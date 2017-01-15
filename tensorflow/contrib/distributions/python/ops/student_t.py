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
"""Student's t distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops


class StudentT(distribution.Distribution):
  """Student's t distribution with degree-of-freedom parameter df.

  #### Mathematical details

  Write `sigma` for the scale and `mu` for the mean (both are scalars). The PDF
  of this distribution is:

  ```none
  f(x) = (1 + y**2 / df)**(-0.5 (df + 1)) / Z
  where,
  y(x) = (x - mu) / sigma
  Z    = abs(sigma) sqrt(df pi) Gamma(0.5 df) / Gamma(0.5 (df + 1))
  ```

  Notice that `sigma` has semantics more similar to standard deviation than
  variance.  (Recall that the variance of the Student's t-distribution is
  `sigma**2 df / (df - 2)` when `df > 2`.)

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  # Define a single scalar Student t distribution.
  single_dist = tf.contrib.distributions.StudentT(df=3)

  # Evaluate the pdf at 1, returning a scalar Tensor.
  single_dist.pdf(1.)

  # Define a batch of two scalar valued Student t's.
  # The first has degrees of freedom 2, mean 1, and scale 11.
  # The second 3, 2 and 22.
  multi_dist = tf.contrib.distributions.StudentT(df=[2, 3],
                                                 mu=[1, 2.],
                                                 sigma=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  multi_dist.pdf([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  multi_dist.sample(3)
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two Student's t distributions.
  # Both have df 2 and mean 1, but different scales.
  dist = tf.contrib.distributions.StudentT(df=2, mu=1, sigma=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.pdf(3.0)
  ```

  """

  def __init__(self,
               df,
               mu,
               sigma,
               validate_args=False,
               allow_nan_stats=True,
               name="StudentT"):
    """Construct Student's t distributions.

    The distributions have degree of freedom `df`, mean `mu`, and scale `sigma`.

    The parameters `df`, `mu`, and `sigma` must be shaped in a way that supports
    broadcasting (e.g. `df + mu + sigma` is a valid operation).

    Args:
      df: Numeric `Tensor`. The degrees of freedom of the distribution(s).
        `df` must contain only positive values.
      mu: Numeric `Tensor`. The mean(s) of the distribution(s).
      sigma: Numeric `Tensor`. The scaling factor(s) for the distribution(s).
        Note that `sigma` is not technically the standard deviation of this
        distribution but has semantics more similar to std. deviation than
        variance.
      validate_args: `Boolean`, default `False`.  Whether to assert that
        `df > 0` and `sigma > 0`. If `validate_args` is `False` and inputs are
        invalid, correct behavior is not guaranteed.
      allow_nan_stats: `Boolean`, default `True`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      TypeError: if mu and sigma are different dtypes.
    """
    parameters = locals()
    parameters.pop("self")
    with ops.name_scope(name, values=[df, mu, sigma]) as ns:
      with ops.control_dependencies([check_ops.assert_positive(df)]
                                    if validate_args else []):
        self._df = array_ops.identity(df, name="df")
        self._mu = array_ops.identity(mu, name="mu")
        self._sigma = array_ops.identity(sigma, name="sigma")
        contrib_tensor_util.assert_same_float_dtype(
            (self._df, self._mu, self._sigma))
    super(StudentT, self).__init__(
        dtype=self._sigma.dtype,
        is_continuous=True,
        is_reparameterized=True,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._df, self._mu, self._sigma],
        name=ns)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("df", "mu", "sigma"), (
            [ops.convert_to_tensor(
                sample_shape, dtype=dtypes.int32)] * 3)))

  @property
  def df(self):
    """Degrees of freedom in these Student's t distribution(s)."""
    return self._df

  @property
  def mu(self):
    """Locations of these Student's t distribution(s)."""
    return self._mu

  @property
  def sigma(self):
    """Scaling factors of these Student's t distribution(s)."""
    return self._sigma

  def _batch_shape(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.df),
        array_ops.broadcast_dynamic_shape(
            array_ops.shape(self.mu), array_ops.shape(self.sigma)))

  def _get_batch_shape(self):
    return array_ops.broadcast_static_shape(
        array_ops.broadcast_static_shape(self.df.get_shape(),
                                         self.mu.get_shape()),
        self.sigma.get_shape())

  def _event_shape(self):
    return constant_op.constant([], dtype=math_ops.int32)

  def _get_event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    # The sampling method comes from the fact that if:
    #   X ~ Normal(0, 1)
    #   Z ~ Chi2(df)
    #   Y = X / sqrt(Z / df)
    # then:
    #   Y ~ StudentT(df).
    shape = array_ops.concat([[n], self.batch_shape()], 0)
    normal_sample = random_ops.random_normal(shape, dtype=self.dtype, seed=seed)
    df = self.df * array_ops.ones(self.batch_shape(), dtype=self.dtype)
    gamma_sample = random_ops.random_gamma(
        [n],
        0.5 * df,
        beta=0.5,
        dtype=self.dtype,
        seed=distribution_util.gen_new_seed(seed, salt="student_t"))
    samples = normal_sample / math_ops.sqrt(gamma_sample / df)
    return samples * self.sigma + self.mu  # Abs(sigma) not wanted.

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _log_unnormalized_prob(self, x):
    y = (x - self.mu) / self.sigma  # Abs(sigma) superfluous.
    return -0.5 * (self.df + 1.) * math_ops.log1p(y**2. / self.df)

  def _log_normalization(self):
    return (math_ops.log(math_ops.abs(self.sigma)) +
            0.5 * math_ops.log(self.df) +
            0.5 * np.log(np.pi) +
            math_ops.lgamma(0.5 * self.df) -
            math_ops.lgamma(0.5 * (self.df + 1.)))

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _cdf(self, x):
    # Take Abs(sigma) to make subsequent where work correctly.
    y = (x - self.mu) / math_ops.abs(self.sigma)
    x_t = self.df / (y**2. + self.df)
    neg_cdf = 0.5 * math_ops.betainc(0.5 * self.df, 0.5, x_t)
    return array_ops.where(math_ops.less(y, 0.), neg_cdf, 1. - neg_cdf)

  def _entropy(self):
    v = array_ops.ones(self.batch_shape(), dtype=self.dtype)[..., None]
    u = v * self.df[..., None]
    beta_arg = array_ops.concat([u, v], -1) / 2.
    return (math_ops.log(math_ops.abs(self.sigma)) +
            0.5 * math_ops.log(self.df) +
            special_math_ops.lbeta(beta_arg) +
            0.5 * (self.df + 1.) *
            (math_ops.digamma(0.5 * (self.df + 1.)) -
             math_ops.digamma(0.5 * self.df)))

  @distribution_util.AppendDocstring(
      """The mean of Student's T equals `mu` if `df > 1`, otherwise it is `NaN`.
      If `self.allow_nan_stats=True`, then an exception will be raised rather
      than returning `NaN`.""")
  def _mean(self):
    mean = self.mu * array_ops.ones(self.batch_shape(), dtype=self.dtype)
    if self.allow_nan_stats:
      nan = np.array(np.nan, dtype=self.dtype.as_numpy_dtype())
      return array_ops.where(
          math_ops.greater(
              self.df,
              array_ops.ones(self.batch_shape(), dtype=self.dtype)),
          mean,
          array_ops.fill(self.batch_shape(), nan, name="nan"))
    else:
      return control_flow_ops.with_dependencies(
          [
              check_ops.assert_less(
                  array_ops.ones((), dtype=self.dtype),
                  self.df,
                  message="mean not defined for components of df <= 1"),
          ],
          mean)

  @distribution_util.AppendDocstring("""
      The variance for Student's T equals

      ```
      df / (df - 2), when df > 2
      infinity, when 1 < df <= 2
      NaN, when df <= 1
      ```
      """)
  def _variance(self):
    # We need to put the tf.where inside the outer tf.where to ensure we never
    # hit a NaN in the gradient.
    denom = array_ops.where(math_ops.greater(self.df, 2.),
                            self.df - 2.,
                            array_ops.ones_like(self.df))
    # Abs(sigma) superfluous.
    var = (array_ops.ones(self.batch_shape(), dtype=self.dtype) *
           math_ops.square(self.sigma) * self.df / denom)
    # When 1 < df <= 2, variance is infinite.
    inf = np.array(np.inf, dtype=self.dtype.as_numpy_dtype())
    result_where_defined = array_ops.where(
        math_ops.greater(self.df, array_ops.fill(self.batch_shape(), 2.)),
        var,
        array_ops.fill(self.batch_shape(), inf, name="inf"))

    if self.allow_nan_stats:
      nan = np.array(np.nan, dtype=self.dtype.as_numpy_dtype())
      return array_ops.where(
          math_ops.greater(
              self.df,
              array_ops.ones(self.batch_shape(), dtype=self.dtype)),
          result_where_defined,
          array_ops.fill(self.batch_shape(), nan, name="nan"))
    else:
      return control_flow_ops.with_dependencies(
          [
              check_ops.assert_less(
                  array_ops.ones((), dtype=self.dtype),
                  self.df,
                  message="variance not defined for components of df <= 1"),
          ],
          result_where_defined)

  def _std(self):
    return math_ops.sqrt(self.variance())

  def _mode(self):
    return array_ops.identity(self.mu)


class StudentTWithAbsDfSoftplusSigma(StudentT):
  """StudentT with `df = floor(abs(df))` and `sigma = softplus(sigma)`."""

  def __init__(self,
               df,
               mu,
               sigma,
               validate_args=False,
               allow_nan_stats=True,
               name="StudentTWithAbsDfSoftplusSigma"):
    parameters = locals()
    parameters.pop("self")
    with ops.name_scope(name, values=[df, sigma]) as ns:
      super(StudentTWithAbsDfSoftplusSigma, self).__init__(
          df=math_ops.floor(math_ops.abs(df)),
          mu=mu,
          sigma=nn.softplus(sigma, name="softplus_sigma"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=ns)
    self._parameters = parameters
