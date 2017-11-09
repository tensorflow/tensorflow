# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Weibull bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector


__all__ = [
    "Weibull",
]


class Weibull(bijector.Bijector):
  """Compute `Y = g(X) = 1 - exp((-X / scale) ** concentration), X >= 0`.

  This bijector maps inputs from `[0, inf]` to [0, 1]`. The inverse of the
  bijector applied to a uniform random variable `X ~ U(0, 1) gives back a
  random variable with the
  [Weibull distribution](https://en.wikipedia.org/wiki/Weibull_distribution):

  ```none
  Y ~ Weibull(scale, concentration)
  pdf(y; scale, concentration, y >= 0) = (scale / concentration) * (
    scale / concentration) ** (concentration - 1) * exp(
      -(y / scale) ** concentration)
  ```
  """

  def __init__(self,
               scale=1.,
               concentration=1.,
               event_ndims=0,
               validate_args=False,
               name="weibull"):
    """Instantiates the `Weibull` bijector.

    Args:
      scale: Positive Float-type `Tensor` that is the same dtype and is
        broadcastable with `concentration`.
        This is `l` in `Y = g(X) = 1 - exp((-x / l) ** k)`.
      concentration: Positive Float-type `Tensor` that is the same dtype and is
        broadcastable with `scale`.
        This is `k` in `Y = g(X) = 1 - exp((-x / l) ** k)`.
      event_ndims: Python scalar indicating the number of dimensions associated
        with a particular draw from the distribution.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    self._graph_parents = []
    self._name = name
    self._validate_args = validate_args
    with self._name_scope("init", values=[scale, concentration]):
      self._scale = ops.convert_to_tensor(scale, name="scale")
      self._concentration = ops.convert_to_tensor(
          concentration, name="concentration")
      check_ops.assert_same_float_dtype([self._scale, self._concentration])
      if validate_args:
        self._scale = control_flow_ops.with_dependencies([
            check_ops.assert_positive(
                self._scale,
                message="Argument scale was not positive")
        ], self._scale)
        self._concentration = control_flow_ops.with_dependencies([
            check_ops.assert_positive(
                self._concentration,
                message="Argument concentration was not positive")
        ], self._concentration)

    super(Weibull, self).__init__(
        event_ndims=event_ndims,
        validate_args=validate_args,
        name=name)

  @property
  def scale(self):
    """The `l` in `Y = g(X) = 1 - exp((-x / l) ** k)`."""
    return self._scale

  @property
  def concentration(self):
    """The `k` in `Y = g(X) = 1 - exp((-x / l) ** k)`."""
    return self._concentration

  def _forward(self, x):
    x = self._maybe_assert_valid_x(x)
    return -math_ops.expm1(-((x / self.scale) ** self.concentration))

  def _inverse(self, y):
    y = self._maybe_assert_valid_y(y)
    return self.scale * (-math_ops.log1p(-y)) ** (1 / self.concentration)

  def _inverse_log_det_jacobian(self, y):
    y = self._maybe_assert_valid_y(y)
    event_dims = self._event_dims_tensor(y)
    return math_ops.reduce_sum(
        -math_ops.log1p(-y) +
        (1 / self.concentration - 1) * math_ops.log(-math_ops.log1p(-y)) +
        math_ops.log(self.scale / self.concentration),
        axis=event_dims)

  def _forward_log_det_jacobian(self, x):
    x = self._maybe_assert_valid_x(x)
    event_dims = self._event_dims_tensor(x)
    return math_ops.reduce_sum(
        -(x / self.scale) ** self.concentration +
        (self.concentration - 1) * math_ops.log(x) +
        math_ops.log(self.concentration) +
        -self.concentration * math_ops.log(self.scale),
        axis=event_dims)

  def _maybe_assert_valid_x(self, x):
    if not self.validate_args:
      return x
    is_valid = check_ops.assert_non_negative(
        x,
        message="Forward transformation input must be at least {}.".format(0))
    return control_flow_ops.with_dependencies([is_valid], x)

  def _maybe_assert_valid_y(self, y):
    if not self.validate_args:
      return y
    is_positive = check_ops.assert_non_negative(
        y, message="Inverse transformation input must be greater than 0.")
    less_than_one = check_ops.assert_less_equal(
        y, constant_op.constant(1., y.dtype),
        message="Inverse transformation input must be less than or equal to 1.")
    return control_flow_ops.with_dependencies([is_positive, less_than_one], y)
