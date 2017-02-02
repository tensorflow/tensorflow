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

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


__all__ = [
    "Poisson",
]


_poisson_sample_note = """
Note that the input value must be a non-negative floating point tensor with
dtype `dtype` and whose shape can be broadcast with `self.rate`. `x` is only
legal if it is non-negative and its components are equal to integer values.
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

  def __init__(self,
               rate,
               validate_args=False,
               allow_nan_stats=True,
               name="Poisson"):
    """Initialize a batch of Poisson distributions.

    Args:
      rate: Floating point tensor, the rate parameter of the
        distribution(s). `rate` must be positive.
      validate_args: Python `Boolean`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `Boolean`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined.  When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: `String` name prefixed to Ops created by this class.
    """
    parameters = locals()
    with ops.name_scope(name, values=[rate]) as ns:
      with ops.control_dependencies([check_ops.assert_positive(rate)] if
                                    validate_args else []):
        self._rate = array_ops.identity(rate, name="rate")
    super(Poisson, self).__init__(
        dtype=self._rate.dtype,
        is_continuous=False,
        reparameterization_type=distribution.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._rate],
        name=ns)

  @property
  def rate(self):
    """Rate parameter."""
    return self._rate

  def _batch_shape_tensor(self):
    return array_ops.shape(self.rate)

  def _batch_shape(self):
    return self.rate.get_shape()

  def _event_shape_tensor(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  @distribution_util.AppendDocstring(_poisson_sample_note)
  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  @distribution_util.AppendDocstring(_poisson_sample_note)
  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  @distribution_util.AppendDocstring(_poisson_sample_note)
  def _log_cdf(self, x):
    return math_ops.log(self.cdf(x))

  @distribution_util.AppendDocstring(_poisson_sample_note)
  def _cdf(self, x):
    x = self._assert_valid_sample(x, check_integer=False)
    return math_ops.igammac(math_ops.floor(x + 1), self.rate)

  def _log_normalization(self):
    return self.rate

  def _log_unnormalized_prob(self, x):
    x = self._assert_valid_sample(x, check_integer=True)
    return x * math_ops.log(self.rate) - math_ops.lgamma(x + 1)

  def _mean(self):
    return array_ops.identity(self.rate)

  def _variance(self):
    return array_ops.identity(self.rate)

  @distribution_util.AppendDocstring(
      """Note: when `rate` is an integer, there are actually two modes: `rate`
      and `rate - 1`. In this case we return the larger, i.e., `rate`.""")
  def _mode(self):
    return math_ops.floor(self.rate)

  def _assert_valid_sample(self, x, check_integer=True):
    if not self.validate_args:
      return x
    dependencies = [check_ops.assert_non_negative(x)]
    if check_integer:
      dependencies += [distribution_util.assert_integer_form(
          x, message="x has non-integer components.")]
    return control_flow_ops.with_dependencies(dependencies, x)
