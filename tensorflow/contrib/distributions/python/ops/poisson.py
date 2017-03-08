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


_poisson_prob_note = """
Note thet the input value must be a non-negative floating point tensor with
dtype `dtype` and whose shape can be broadcast with `self.lam`. `x` is only
legal if it is non-negative and its components are equal to integer values.
"""


class Poisson(distribution.Distribution):
  """Poisson distribution.

  The Poisson distribution is parameterized by `lam`, the rate parameter.

  The pmf of this distribution is:

  ```

  pmf(k) = e^(-lam) * lam^k / k!,  k >= 0
  ```

  """

  def __init__(self,
               lam,
               validate_args=False,
               allow_nan_stats=True,
               name="Poisson"):
    """Construct Poisson distributions.

    Args:
      lam: Floating point tensor, the rate parameter of the
        distribution(s). `lam` must be positive.
      validate_args: `Boolean`, default `False`.  Whether to assert that
        `lam > 0` as well as inputs to pmf computations are non-negative
        integers. If validate_args is `False`, then `pmf` computations might
        return `NaN`, but can be evaluated at any real value.
      allow_nan_stats: `Boolean`, default `True`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: A name for this distribution.
    """
    parameters = locals()
    parameters.pop("self")
    with ops.name_scope(name, values=[lam]) as ns:
      with ops.control_dependencies([check_ops.assert_positive(lam)] if
                                    validate_args else []):
        self._lam = array_ops.identity(lam, name="lam")
    super(Poisson, self).__init__(
        dtype=self._lam.dtype,
        is_continuous=False,
        is_reparameterized=False,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._lam],
        name=ns)

  @property
  def lam(self):
    """Rate parameter."""
    return self._lam

  def _batch_shape(self):
    return array_ops.shape(self.lam)

  def _get_batch_shape(self):
    return self.lam.get_shape()

  def _event_shape(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _get_event_shape(self):
    return tensor_shape.scalar()

  @distribution_util.AppendDocstring(_poisson_prob_note)
  def _log_prob(self, x):
    x = self._assert_valid_sample(x, check_integer=True)
    return x * math_ops.log(self.lam) - self.lam - math_ops.lgamma(x + 1)

  @distribution_util.AppendDocstring(_poisson_prob_note)
  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _log_cdf(self, x):
    return math_ops.log(self.cdf(x))

  def _cdf(self, x):
    x = self._assert_valid_sample(x, check_integer=False)
    return math_ops.igammac(math_ops.floor(x + 1), self.lam)

  def _mean(self):
    return array_ops.identity(self.lam)

  def _variance(self):
    return array_ops.identity(self.lam)

  def _std(self):
    return math_ops.sqrt(self.variance())

  @distribution_util.AppendDocstring(
      """Note that when `lam` is an integer, there are actually two modes.
      Namely, `lam` and `lam - 1` are both modes. Here we return
      only the larger of the two modes.""")
  def _mode(self):
    return math_ops.floor(self.lam)

  def _assert_valid_sample(self, x, check_integer=True):
    if not self.validate_args: return x
    with ops.name_scope("check_x", values=[x]):
      dependencies = [check_ops.assert_non_negative(x)]
      if check_integer:
        dependencies += [distribution_util.assert_integer_form(
            x, message="x has non-integer components.")]
      return control_flow_ops.with_dependencies(dependencies, x)
