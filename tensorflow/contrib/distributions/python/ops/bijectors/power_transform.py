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
"""PowerTransform bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.util import deprecation


__all__ = [
    "PowerTransform",
]


class PowerTransform(bijector.Bijector):
  """Compute `Y = g(X) = (1 + X * c)**(1 / c), X >= -1 / c`.

  The [power transform](https://en.wikipedia.org/wiki/Power_transform) maps
  inputs from `[0, inf]` to `[-1/c, inf]`; this is equivalent to the `inverse`
  of this bijector.

  This bijector is equivalent to the `Exp` bijector when `c=0`.
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
               power=0.,
               validate_args=False,
               name="power_transform"):
    """Instantiates the `PowerTransform` bijector.

    Args:
      power: Python `float` scalar indicating the transform power, i.e.,
        `Y = g(X) = (1 + X * c)**(1 / c)` where `c` is the `power`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.

    Raises:
      ValueError: if `power < 0` or is not known statically.
    """
    self._graph_parents = []
    self._name = name
    self._validate_args = validate_args
    with self._name_scope("init", values=[power]):
      power = tensor_util.constant_value(
          ops.convert_to_tensor(power, name="power"))
    if power is None or power < 0:
      raise ValueError("`power` must be a non-negative TF constant.")
    self._power = power
    super(PowerTransform, self).__init__(
        forward_min_event_ndims=0,
        validate_args=validate_args,
        name=name)

  @property
  def power(self):
    """The `c` in: `Y = g(X) = (1 + X * c)**(1 / c)`."""
    return self._power

  def _forward(self, x):
    x = self._maybe_assert_valid_x(x)
    if self.power == 0.:
      return math_ops.exp(x)
    # If large x accuracy is an issue, consider using:
    # (1. + x * self.power)**(1. / self.power) when x >> 1.
    return math_ops.exp(math_ops.log1p(x * self.power) / self.power)

  def _inverse(self, y):
    y = self._maybe_assert_valid_y(y)
    if self.power == 0.:
      return math_ops.log(y)
    # If large y accuracy is an issue, consider using:
    # (y**self.power - 1.) / self.power when y >> 1.
    return math_ops.expm1(math_ops.log(y) * self.power) / self.power

  def _inverse_log_det_jacobian(self, y):
    y = self._maybe_assert_valid_y(y)
    return (self.power - 1.) * math_ops.log(y)

  def _forward_log_det_jacobian(self, x):
    x = self._maybe_assert_valid_x(x)
    if self.power == 0.:
      return x
    return (1. / self.power - 1.) * math_ops.log1p(x * self.power)

  def _maybe_assert_valid_x(self, x):
    if not self.validate_args or self.power == 0.:
      return x
    is_valid = check_ops.assert_non_negative(
        1. + self.power * x,
        message="Forward transformation input must be at least {}.".format(
            -1. / self.power))
    return control_flow_ops.with_dependencies([is_valid], x)

  def _maybe_assert_valid_y(self, y):
    if not self.validate_args:
      return y
    is_valid = check_ops.assert_positive(
        y, message="Inverse transformation input must be greater than 0.")
    return control_flow_ops.with_dependencies([is_valid], y)
