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

from tensorflow.contrib.distributions.python.ops import distribution  # pylint: disable=line-too-long
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util  # pylint: disable=line-too-long
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
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
               validate_args=True,
               allow_nan_stats=False,
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
      validate_args: Whether to assert that `df > 0, sigma > 0`. If
        `validate_args` is `False` and inputs are invalid, correct behavior is
        not guaranteed.
      allow_nan_stats:  Boolean, default `False`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      TypeError: if mu and sigma are different dtypes.
    """
    self._allow_nan_stats = allow_nan_stats
    self._validate_args = validate_args
    with ops.op_scope([df, mu, sigma], name) as scope:
      with ops.control_dependencies([check_ops.assert_positive(
          df), check_ops.assert_positive(sigma)] if validate_args else []):
        self._df = ops.convert_to_tensor(df, name="df")
        self._mu = ops.convert_to_tensor(mu, name="mu")
        self._sigma = ops.convert_to_tensor(sigma, name="sigma")
        contrib_tensor_util.assert_same_float_dtype(
            (self._df, self._mu, self._sigma))
      self._name = scope
      self._get_batch_shape = self._ones().get_shape()
      self._get_event_shape = tensor_shape.TensorShape([])

  @property
  def allow_nan_stats(self):
    """Boolean describing behavior when a stat is undefined for batch member."""
    return self._allow_nan_stats

  @property
  def validate_args(self):
    """Boolean describing behavior on invalid input."""
    return self._validate_args

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self._df.dtype

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

  def mean(self, name="mean"):
    """Mean of the distribution.

    The mean of Student's T equals `mu` if `df > 1`, otherwise it is `NaN`.  If
    `self.allow_nan_stats=False`, then an exception will be raised rather than
    returning `NaN`.

    Args:
      name:  A name to give this op.

    Returns:
      The mean for every batch member, a `Tensor` with same `dtype` as self.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._mu], name):
        result_if_defined = self._mu * self._ones()
        if self.allow_nan_stats:
          df_gt_1 = self._df > self._ones()
          nan = np.nan + self._zeros()
          return math_ops.select(df_gt_1, result_if_defined, nan)
        else:
          one = constant_op.constant(1.0, dtype=self.dtype)
          return control_flow_ops.with_dependencies(
              [check_ops.assert_less(
                  one, self._df,
                  message="mean not defined for components of df <= 1"
              )], result_if_defined)

  def mode(self, name="mode"):
    with ops.name_scope(self.name):
      with ops.op_scope([self._mu], name):
        return array_ops.identity(self._mu)

  def variance(self, name="variance"):
    """Variance of the distribution.

    Variance for Student's T equals

    ```
    df / (df - 2), when df > 2
    infinity, when 1 < df <= 2
    NaN, when df <= 1
    ```

    The NaN state occurs because mean is undefined for `df <= 1`, and if
    `self.allow_nan_stats` is `False`, an exception will be raised if any batch
    members fall into this state.

    Args:
      name:  A name for this op.

    Returns:
      The variance for every batch member, a `Tensor` with same `dtype` as self.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._df, self._sigma], name):
        result_where_finite = (
            self._zeros()
            + math_ops.square(self._sigma) * self._df / (self._df - 2))
        # When 1 < df <= 2, variance is infinite.
        result_where_defined = math_ops.select(
            self._zeros() + self._df > 2,
            result_where_finite,
            self._zeros() + np.inf)

        if self.allow_nan_stats:
          return math_ops.select(
              (self._zeros() + self._df > 1),
              result_where_defined,
              self._zeros() + np.nan)
        else:
          one = constant_op.constant(1.0, dtype=self.dtype)
          return control_flow_ops.with_dependencies(
              [check_ops.assert_less(
                  one, self._df,
                  message="variance not defined for components of df <= 1"
              )], result_where_defined)

  def std(self, name="std"):
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        return math_ops.sqrt(self.variance())

  def batch_shape(self, name="batch_shape"):
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        return array_ops.shape(self._ones())

  def get_batch_shape(self):
    return self._get_batch_shape

  def event_shape(self, name="event_shape"):
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        return constant_op.constant([], dtype=math_ops.int32)

  def get_event_shape(self):
    return self._event_shape

  def log_prob(self, x, name="log_prob"):
    """Log prob of observations in `x` under these Student's t-distribution(s).

    Args:
      x: tensor of dtype `dtype`, must be broadcastable with `mu` and `df`.
      name: The name to give this op.

    Returns:
      log_prob: tensor of dtype `dtype`, the log-PDFs of `x`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._df, self._mu, self._sigma, x], name):
        x = ops.convert_to_tensor(x)
        if x.dtype != self.dtype:
          raise TypeError("Input x dtype does not match dtype: %s vs. %s" %
                          (x.dtype, self.dtype))
        df_2 = self._df / 2
        log_beta = (math_ops.lgamma(0.5) + math_ops.lgamma(df_2) -
                    math_ops.lgamma(0.5 + df_2))
        return (-math_ops.log(self._df) / 2 - log_beta - (self._df + 1) / 2 *
                math_ops.log(1 + math_ops.square((x - self._mu) / self._sigma) /
                             self._df) - math_ops.log(self._sigma))

  def prob(self, x, name="prob"):
    """The PDF of observations in `x` under these Student's t distribution(s).

    Args:
      x: tensor of dtype `dtype`, must be broadcastable with `df`, `mu`, and
        `sigma`.
      name: The name to give this op.

    Returns:
      prob: tensor of dtype `dtype`, the prob values of `x`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._df, self._mu, self._sigma, x], name):
        x = ops.convert_to_tensor(x)
        if x.dtype != self.dtype:
          raise TypeError("Input x dtype does not match dtype: %s vs. %s" %
                          (x.dtype, self.dtype))
        reloc_scaled = (x - self._mu) / self._sigma
        return (math_ops.exp(math_ops.lgamma((self._df + 1) / 2) -
                             math_ops.lgamma(self._df / 2)) /
                math_ops.sqrt(self._df) / math.sqrt(np.pi) *
                math_ops.pow(1 + math_ops.square(reloc_scaled) / self._df,
                             -(self._df + 1) / 2) / self.sigma)

  def entropy(self, name="entropy"):
    """The entropy of Student t distribution(s).

    Args:
      name: The name to give this op.

    Returns:
      entropy: tensor of dtype `dtype`, the entropy.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._df, self._sigma], name):
        u = array_ops.expand_dims(self._df + self._zeros(), -1)
        v = array_ops.expand_dims(self._ones(), -1)
        beta_arg = array_ops.concat(len(u.get_shape()) - 1, [u, v]) / 2
        return ((self._df + 1) / 2 * (math_ops.digamma((self._df + 1) / 2) -
                                      math_ops.digamma(self._df / 2)) +
                math_ops.log(self._df) / 2 +
                special_math_ops.lbeta(beta_arg) +
                math_ops.log(self._sigma))

  def sample_n(self, n, seed=None, name="sample_n"):
    """Sample `n` observations from the Student t Distributions.

    Args:
      n: `Scalar`, type int32, the number of observations to sample.
      seed: Python integer, the random seed.
      name: The name to give this op.

    Returns:
      samples: a `Tensor` of shape `(n,) + self.batch_shape + self.event_shape`
          with values of type `self.dtype`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._df, self._mu, self._sigma, n], name):
        n = ops.convert_to_tensor(n, name="n")
        n_val = tensor_util.constant_value(n)

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
        theta = (2 * np.pi) * samples_h
        r = math_ops.sqrt(self._df *
                          (math_ops.pow(samples_g, -2 / self._df) - 1))
        samples = r * math_ops.cos(theta)

        # Provide some hints to shape inference
        inferred_shape = tensor_shape.vector(n_val).concatenate(
            self.get_batch_shape())
        samples.set_shape(inferred_shape)

        return samples * self._sigma + self._mu

  @property
  def is_reparameterized(self):
    return True

  def _ones(self):
    return array_ops.ones_like(self._df + self._mu + self._sigma)

  def _zeros(self):
    return array_ops.zeros_like(self._df + self._mu + self._sigma)

  @property
  def is_continuous(self):
    return True
