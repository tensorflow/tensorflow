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
"""The InverseGamma distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

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
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation


__all__ = [
    "InverseGamma",
    "InverseGammaWithSoftplusConcentrationRate",
]


class InverseGamma(distribution.Distribution):
  """InverseGamma distribution.

  The `InverseGamma` distribution is defined over positive real numbers using
  parameters `concentration` (aka "alpha") and `rate` (aka "beta").

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; alpha, beta, x > 0) = x**(-alpha - 1) exp(-beta / x) / Z
  Z = Gamma(alpha) beta**-alpha
  ```

  where:

  * `concentration = alpha`,
  * `rate = beta`,
  * `Z` is the normalizing constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The cumulative density function (cdf) is,

  ```none
  cdf(x; alpha, beta, x > 0) = GammaInc(alpha, beta / x) / Gamma(alpha)
  ```

  where `GammaInc` is the [upper incomplete Gamma function](
  https://en.wikipedia.org/wiki/Incomplete_gamma_function).

  The parameters can be intuited via their relationship to mean and stddev,

  ```none
  concentration = alpha = (mean / stddev)**2
  rate = beta = mean / stddev**2
  ```

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  WARNING: This distribution may draw 0-valued samples for small concentration
  values. See note in `tf.random_gamma` docstring.

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  dist = tfd.InverseGamma(concentration=3.0, rate=2.0)
  dist2 = tfd.InverseGamma(concentration=[3.0, 4.0], rate=[2.0, 3.0])
  ```

  """

  @deprecation.deprecated(
      "2018-10-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.contrib.distributions`.",
      warn_once=True)
  def __init__(self,
               concentration,
               rate,
               validate_args=False,
               allow_nan_stats=True,
               name="InverseGamma"):
    """Construct InverseGamma with `concentration` and `rate` parameters.

    The parameters `concentration` and `rate` must be shaped in a way that
    supports broadcasting (e.g. `concentration + rate` is a valid operation).

    Args:
      concentration: Floating point tensor, the concentration params of the
        distribution(s). Must contain only positive values.
      rate: Floating point tensor, the inverse scale params of the
        distribution(s). Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.


    Raises:
      TypeError: if `concentration` and `rate` are different dtypes.
    """
    parameters = dict(locals())
    with ops.name_scope(name, values=[concentration, rate]) as name:
      with ops.control_dependencies([
          check_ops.assert_positive(concentration),
          check_ops.assert_positive(rate),
      ] if validate_args else []):
        self._concentration = array_ops.identity(
            concentration, name="concentration")
        self._rate = array_ops.identity(rate, name="rate")
        check_ops.assert_same_float_dtype(
            [self._concentration, self._rate])
    super(InverseGamma, self).__init__(
        dtype=self._concentration.dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        reparameterization_type=distribution.NOT_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=[self._concentration,
                       self._rate],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("concentration", "rate"), ([ops.convert_to_tensor(
            sample_shape, dtype=dtypes.int32)] * 2)))

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  @property
  def rate(self):
    """Rate parameter."""
    return self._rate

  def _batch_shape_tensor(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.concentration),
        array_ops.shape(self.rate))

  def _batch_shape(self):
    return array_ops.broadcast_static_shape(
        self.concentration.get_shape(),
        self.rate.get_shape())

  def _event_shape_tensor(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  @distribution_util.AppendDocstring(
      """Note: See `tf.random_gamma` docstring for sampling details and
      caveats.""")
  def _sample_n(self, n, seed=None):
    return 1. / random_ops.random_gamma(
        shape=[n],
        alpha=self.concentration,
        beta=self.rate,
        dtype=self.dtype,
        seed=seed)

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _cdf(self, x):
    x = self._maybe_assert_valid_sample(x)
    # Note that igammac returns the upper regularized incomplete gamma
    # function Q(a, x), which is what we want for the CDF.
    return math_ops.igammac(self.concentration, self.rate / x)

  def _log_unnormalized_prob(self, x):
    x = self._maybe_assert_valid_sample(x)
    return -(1. + self.concentration) * math_ops.log(x) - self.rate / x

  def _log_normalization(self):
    return (math_ops.lgamma(self.concentration)
            - self.concentration * math_ops.log(self.rate))

  def _entropy(self):
    return (self.concentration
            + math_ops.log(self.rate)
            + math_ops.lgamma(self.concentration)
            - ((1. + self.concentration) *
               math_ops.digamma(self.concentration)))

  @distribution_util.AppendDocstring(
      """The mean of an inverse gamma distribution is
      `rate / (concentration - 1)`, when `concentration > 1`, and `NaN`
      otherwise. If `self.allow_nan_stats` is `False`, an exception will be
      raised rather than returning `NaN`""")
  def _mean(self):
    mean = self.rate / (self.concentration - 1.)
    if self.allow_nan_stats:
      nan = array_ops.fill(
          self.batch_shape_tensor(),
          np.array(np.nan, dtype=self.dtype.as_numpy_dtype()),
          name="nan")
      return array_ops.where(self.concentration > 1., mean, nan)
    else:
      return control_flow_ops.with_dependencies([
          check_ops.assert_less(
              array_ops.ones([], self.dtype), self.concentration,
              message="mean undefined when any concentration <= 1"),
      ], mean)

  @distribution_util.AppendDocstring(
      """Variance for inverse gamma is defined only for `concentration > 2`. If
      `self.allow_nan_stats` is `False`, an exception will be raised rather
      than returning `NaN`.""")
  def _variance(self):
    var = (
        math_ops.square(self.rate) / math_ops.squared_difference(
            self.concentration, 1.) / (self.concentration - 2.))
    if self.allow_nan_stats:
      nan = array_ops.fill(
          self.batch_shape_tensor(),
          np.array(np.nan, dtype=self.dtype.as_numpy_dtype()),
          name="nan")
      return array_ops.where(self.concentration > 2., var, nan)
    else:
      return control_flow_ops.with_dependencies([
          check_ops.assert_less(
              constant_op.constant(2., dtype=self.dtype),
              self.concentration,
              message="variance undefined when any concentration <= 2"),
      ], var)

  @distribution_util.AppendDocstring(
      """The mode of an inverse gamma distribution is `rate / (concentration +
      1)`.""")
  def _mode(self):
    return self.rate / (1. + self.concentration)

  def _maybe_assert_valid_sample(self, x):
    check_ops.assert_same_float_dtype(
        tensors=[x], dtype=self.dtype)
    if not self.validate_args:
      return x
    return control_flow_ops.with_dependencies([
        check_ops.assert_positive(x),
    ], x)


class InverseGammaWithSoftplusConcentrationRate(InverseGamma):
  """`InverseGamma` with softplus of `concentration` and `rate`."""

  @deprecation.deprecated(
      "2018-10-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.contrib.distributions`.",
      warn_once=True)
  def __init__(self,
               concentration,
               rate,
               validate_args=False,
               allow_nan_stats=True,
               name="InverseGammaWithSoftplusConcentrationRate"):
    parameters = dict(locals())
    with ops.name_scope(name, values=[concentration, rate]) as name:
      super(InverseGammaWithSoftplusConcentrationRate, self).__init__(
          concentration=nn.softplus(concentration,
                                    name="softplus_concentration"),
          rate=nn.softplus(rate, name="softplus_rate"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
    self._parameters = parameters
