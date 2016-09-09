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

import math
import numpy as np

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.python.framework import common_shapes
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

  The PDF of this distribution is:

  `f(t) = gamma((df+1)/2)/sqrt(df*pi)/gamma(df/2)*(1+t^2/df)^(-(df+1)/2)`

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
      df: Floating point tensor, the degrees of freedom of the
        distribution(s). `df` must contain only positive values.
      mu: Floating point tensor, the means of the distribution(s).
      sigma: Floating point tensor, the scaling factor for the
        distribution(s). `sigma` must contain only positive values.
        Note that `sigma` is not the standard deviation of this distribution.
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
    with ops.name_scope(name, values=[df, mu, sigma]) as ns:
      with ops.control_dependencies([
          check_ops.assert_positive(df),
          check_ops.assert_positive(sigma),
      ] if validate_args else []):
        self._df = array_ops.identity(df, name="df")
        self._mu = array_ops.identity(mu, name="mu")
        self._sigma = array_ops.identity(sigma, name="sigma")
        contrib_tensor_util.assert_same_float_dtype(
            (self._df, self._mu, self._sigma))
        super(StudentT, self).__init__(
            dtype=self._sigma.dtype,
            parameters={"df": self._df, "mu": self._mu, "sigma": self._sigma},
            is_continuous=True,
            is_reparameterized=True,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=ns)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("df", "mu", "sigma"), ([ops.convert_to_tensor(
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
    return array_ops.shape(self.df + self.mu + self.sigma)

  def _get_batch_shape(self):
    return common_shapes.broadcast_shape(
        self.sigma.get_shape(),
        common_shapes.broadcast_shape(
            self.df.get_shape(),
            self.mu.get_shape()))

  def _event_shape(self):
    return constant_op.constant([], dtype=math_ops.int32)

  def _get_event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    # We use 2 uniform random floats to generate polar random variates.
    # http://dl.acm.org/citation.cfm?id=179631
    # Theorem 2. Let G, H be iid variates, uniformly distributed on [0,1].
    # Let theta = 2*pi*H, let R = sqrt(df*(G^(-2/df) - 1)) for df > 0.
    # Let X = R*cos(theta), and let Y = R*sin(theta).
    # Then X ~ t_df and Y ~ t_df.
    # The variates X and Y are not independent.
    shape = array_ops.concat(0, ([2, n], self.batch_shape()))
    uniform = random_ops.random_uniform(shape=shape,
                                        dtype=self.dtype,
                                        seed=seed)
    samples_g, samples_h = array_ops.unpack(uniform, num=2)
    theta = (2. * math.pi) * samples_h
    r = math_ops.sqrt(self.df *
                      (math_ops.pow(samples_g, -2 / self.df) - 1))
    samples = r * math_ops.cos(theta)
    return samples * self.sigma + self.mu

  def _log_prob(self, x):
    y = (x - self.mu) / self.sigma
    half_df = 0.5 * self.df
    return (math_ops.lgamma(0.5 + half_df) -
            math_ops.lgamma(half_df) -
            0.5 * math_ops.log(self.df) -
            0.5 * math.log(math.pi) -
            math_ops.log(self.sigma) -
            (0.5 + half_df) * math_ops.log(1. + math_ops.square(y) / self.df))

  def _prob(self, x):
    y = (x - self.mu) / self.sigma
    half_df = 0.5 * self.df
    return (math_ops.exp(math_ops.lgamma(0.5 + half_df) -
                         math_ops.lgamma(half_df)) /
            (math_ops.sqrt(self.df) * math.sqrt(math.pi) * self.sigma) *
            math_ops.pow(1. + math_ops.square(y) / self.df, -(0.5 + half_df)))

  def _entropy(self):
    u = array_ops.expand_dims(self.df * self._ones(), -1)
    v = array_ops.expand_dims(self._ones(), -1)
    beta_arg = array_ops.concat(len(u.get_shape()) - 1, [u, v]) / 2
    half_df = 0.5 * self.df
    return ((0.5 + half_df) * (math_ops.digamma(0.5 + half_df) -
                               math_ops.digamma(half_df)) +
            0.5 * math_ops.log(self.df) +
            special_math_ops.lbeta(beta_arg) +
            math_ops.log(self.sigma))

  def _mean(self):
    mean = self.mu * self._ones()
    if self.allow_nan_stats:
      nan = np.array(np.nan, dtype=self.dtype.as_numpy_dtype())
      return math_ops.select(
          math_ops.greater(self.df, self._ones()), mean,
          array_ops.fill(self.batch_shape(), nan, name="nan"))
    else:
      return control_flow_ops.with_dependencies([
          check_ops.assert_less(
              array_ops.ones((), dtype=self.dtype), self.df,
              message="mean not defined for components of df <= 1"),
      ], mean)

  def _variance(self):
    var = (self._ones() *
           math_ops.square(self.sigma) * self.df / (self.df - 2))
    # When 1 < df <= 2, variance is infinite.
    inf = np.array(np.inf, dtype=self.dtype.as_numpy_dtype())
    result_where_defined = math_ops.select(
        math_ops.greater(self.df, array_ops.fill(self.batch_shape(), 2.)),
        var,
        array_ops.fill(self.batch_shape(), inf, name="inf"))

    if self.allow_nan_stats:
      nan = np.array(np.nan, dtype=self.dtype.as_numpy_dtype())
      return math_ops.select(
          math_ops.greater(self.df, self._ones()),
          result_where_defined,
          array_ops.fill(self.batch_shape(), nan, name="nan"))
    else:
      return control_flow_ops.with_dependencies([
          check_ops.assert_less(
              array_ops.ones((), dtype=self.dtype), self.df,
              message="variance not defined for components of df <= 1"),
      ], result_where_defined)

  def _std(self):
    return math_ops.sqrt(self.variance())

  def _mode(self):
    return array_ops.identity(self.mu)

  def _ones(self):
    return array_ops.ones(self.batch_shape(), dtype=self.dtype)


distribution_util.append_class_fun_doc(StudentT.mean, doc_str="""

    The mean of Student's T equals `mu` if `df > 1`, otherwise it is `NaN`.  If
    `self.allow_nan_stats=True`, then an exception will be raised rather than
    returning `NaN`.
""")

distribution_util.append_class_fun_doc(StudentT.variance, doc_str="""

    Variance for Student's T equals

    ```
    df / (df - 2), when df > 2
    infinity, when 1 < df <= 2
    NaN, when df <= 1
    ```

""")


class StudentTWithAbsDfSoftplusSigma(StudentT):
  """StudentT with `df = floor(abs(df))` and `sigma = softplus(sigma)`."""

  def __init__(self,
               df,
               mu,
               sigma,
               validate_args=False,
               allow_nan_stats=True,
               name="StudentTWithAbsDfSoftplusSigma"):
    with ops.name_scope(name, values=[df, mu, sigma]) as ns:
      super(StudentTWithAbsDfSoftplusSigma, self).__init__(
          df=math_ops.floor(math_ops.abs(df)),
          mu=mu,
          sigma=nn.softplus(sigma),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=ns)
