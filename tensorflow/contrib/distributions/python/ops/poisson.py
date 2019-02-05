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
"""The Poisson distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation

__all__ = [
    "Poisson",
]


_poisson_sample_note = """
The Poisson distribution is technically only defined for non-negative integer
values. When `validate_args=False`, non-integral inputs trigger an assertion.

When `validate_args=False` calculations are otherwise unchanged despite
integral or non-integral inputs.

When `validate_args=False`, evaluating the pmf at non-integral values,
corresponds to evaluations of an unnormalized distribution, that does not
correspond to evaluations of the cdf.
"""


class Poisson(distribution.Distribution):
  """Poisson distribution.

  The Poisson distribution is parameterized by an event `rate` parameter.

  #### Mathematical Details

  The probability mass function (pmf) is,

  ```none
  pmf(k; lambda, k >= 0) = (lambda^k / k!) / Z
  Z = exp(lambda).
  ```

  where `rate = lambda` and `Z` is the normalizing constant.

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
               rate=None,
               log_rate=None,
               validate_args=False,
               allow_nan_stats=True,
               name="Poisson"):
    """Initialize a batch of Poisson distributions.

    Args:
      rate: Floating point tensor, the rate parameter. `rate` must be positive.
        Must specify exactly one of `rate` and `log_rate`.
      log_rate: Floating point tensor, the log of the rate parameter.
        Must specify exactly one of `rate` and `log_rate`.
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
      ValueError: if none or both of `rate`, `log_rate` are specified.
      TypeError: if `rate` is not a float-type.
      TypeError: if `log_rate` is not a float-type.
    """
    parameters = dict(locals())
    with ops.name_scope(name, values=[rate]) as name:
      if (rate is None) == (log_rate is None):
        raise ValueError("Must specify exactly one of `rate` and `log_rate`.")
      elif log_rate is None:
        rate = ops.convert_to_tensor(rate, name="rate")
        if not rate.dtype.is_floating:
          raise TypeError("rate.dtype ({}) is a not a float-type.".format(
              rate.dtype.name))
        with ops.control_dependencies([check_ops.assert_positive(rate)] if
                                      validate_args else []):
          self._rate = array_ops.identity(rate, name="rate")
          self._log_rate = math_ops.log(rate, name="log_rate")
      else:
        log_rate = ops.convert_to_tensor(log_rate, name="log_rate")
        if not log_rate.dtype.is_floating:
          raise TypeError("log_rate.dtype ({}) is a not a float-type.".format(
              log_rate.dtype.name))
        self._rate = math_ops.exp(log_rate, name="rate")
        self._log_rate = ops.convert_to_tensor(log_rate, name="log_rate")
    super(Poisson, self).__init__(
        dtype=self._rate.dtype,
        reparameterization_type=distribution.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._rate],
        name=name)

  @property
  def rate(self):
    """Rate parameter."""
    return self._rate

  @property
  def log_rate(self):
    """Log rate parameter."""
    return self._log_rate

  def _batch_shape_tensor(self):
    return array_ops.shape(self.rate)

  def _batch_shape(self):
    return self.rate.shape

  def _event_shape_tensor(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  @distribution_util.AppendDocstring(_poisson_sample_note)
  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  @distribution_util.AppendDocstring(_poisson_sample_note)
  def _log_cdf(self, x):
    return math_ops.log(self.cdf(x))

  @distribution_util.AppendDocstring(_poisson_sample_note)
  def _cdf(self, x):
    if self.validate_args:
      x = distribution_util.embed_check_nonnegative_integer_form(x)
    return math_ops.igammac(1. + x, self.rate)

  def _log_normalization(self):
    return self.rate

  def _log_unnormalized_prob(self, x):
    if self.validate_args:
      x = distribution_util.embed_check_nonnegative_integer_form(x)
    return x * self.log_rate - math_ops.lgamma(1. + x)

  def _mean(self):
    return array_ops.identity(self.rate)

  def _variance(self):
    return array_ops.identity(self.rate)

  @distribution_util.AppendDocstring(
      """Note: when `rate` is an integer, there are actually two modes: `rate`
      and `rate - 1`. In this case we return the larger, i.e., `rate`.""")
  def _mode(self):
    return math_ops.floor(self.rate)

  def _sample_n(self, n, seed=None):
    return random_ops.random_poisson(
        self.rate, [n], dtype=self.dtype, seed=seed)
