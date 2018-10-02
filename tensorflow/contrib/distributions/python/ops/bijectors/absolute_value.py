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
"""AbsoluteValue bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.util import deprecation

__all__ = [
    "AbsoluteValue",
]


class AbsoluteValue(bijector.Bijector):
  """Computes `Y = g(X) = Abs(X)`, element-wise.

  This non-injective bijector allows for transformations of scalar distributions
  with the absolute value function, which maps `(-inf, inf)` to `[0, inf)`.

  * For `y in (0, inf)`, `AbsoluteValue.inverse(y)` returns the set inverse
    `{x in (-inf, inf) : |x| = y}` as a tuple, `-y, y`.
  * `AbsoluteValue.inverse(0)` returns `0, 0`, which is not the set inverse
    (the set inverse is the singleton `{0}`), but "works" in conjunction with
    `TransformedDistribution` to produce a left semi-continuous pdf.
  * For `y < 0`, `AbsoluteValue.inverse(y)` happily returns the
    wrong thing, `-y, y`.  This is done for efficiency.  If
    `validate_args == True`, `y < 0` will raise an exception.


  ```python
  tfd = tf.contrib.distributions

  abs = tfd.bijectors.AbsoluteValue()

  abs.forward([-1., 0., 1.])
  ==> [1., 0.,  1.]

  abs.inverse(1.)
  ==> [-1., 1.]

  # The |dX/dY| is constant, == 1.  So Log|dX/dY| == 0.
  abs.inverse_log_det_jacobian(1.)
  ==> [0., 0.]

  # Special case handling of 0.
  abs.inverse(0.)
  ==> [0., 0.]

  abs.inverse_log_det_jacobian(0.)
  ==> [0., 0.]
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
  def __init__(self, validate_args=False, name="absolute_value"):
    """Instantiates the `AbsoluteValue` bijector.

    Args:
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness, in particular whether inputs to `inverse` and
        `inverse_log_det_jacobian` are non-negative.
      name: Python `str` name given to ops managed by this object.
    """
    self._graph_parents = []
    self._name = name

    with self._name_scope("init"):
      super(AbsoluteValue, self).__init__(
          forward_min_event_ndims=0,
          is_constant_jacobian=True,
          validate_args=validate_args,
          name=name)

  def _forward(self, x):
    return math_ops.abs(x)

  def _inverse(self, y):
    if self.validate_args:
      y = control_flow_ops.with_dependencies(
          [check_ops.assert_non_negative(y, message="Argument y was negative")],
          y)
    return -y, y

  def _inverse_log_det_jacobian(self, y):
    # If event_ndims = 2,
    # F^{-1}(y) = (-y, y), so DF^{-1}(y) = (-1, 1),
    # so Log|DF^{-1}(y)| = Log[1, 1] = [0, 0].
    zeros = constant_op.constant(0., dtype=y.dtype)
    if self.validate_args:
      zeros = control_flow_ops.with_dependencies(
          [check_ops.assert_non_negative(y, message="Argument y was negative")],
          zeros)
    return zeros, zeros

  @property
  def _is_injective(self):
    return False
