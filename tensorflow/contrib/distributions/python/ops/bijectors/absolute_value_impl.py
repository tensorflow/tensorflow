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

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector

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
  abs = ds.bijectors.AbsoluteValue()

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

  def __init__(self, event_ndims=0, validate_args=False, name="absolute_value"):
    """Instantiates the `AbsoluteValue` bijector.

    Args:
      event_ndims: Python scalar indicating the number of dimensions associated
        with a particular draw from the distribution.  Currently only zero is
        supported.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness, in particular whether inputs to `inverse` and
        `inverse_log_det_jacobian` are non-negative.
      name: Python `str` name given to ops managed by this object.

    Raises:
      ValueError:  If `event_ndims` is not zero.
    """
    self._graph_parents = []
    self._name = name

    event_ndims = ops.convert_to_tensor(event_ndims, name="event_ndims")
    event_ndims_const = tensor_util.constant_value(event_ndims)
    if event_ndims_const is not None and event_ndims_const not in (0,):
      raise ValueError("event_ndims(%s) was not 0" % event_ndims_const)
    else:
      if validate_args:
        event_ndims = control_flow_ops.with_dependencies(
            [check_ops.assert_equal(
                event_ndims, 0, message="event_ndims was not 0")],
            event_ndims)

    with self._name_scope("init"):
      super(AbsoluteValue, self).__init__(
          event_ndims=event_ndims,
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
    batch_shape = array_ops.shape(y)[:array_ops.rank(y) - self.event_ndims]
    zeros = array_ops.zeros(batch_shape, dtype=y.dtype)
    if self.validate_args:
      zeros = control_flow_ops.with_dependencies(
          [check_ops.assert_non_negative(y, message="Argument y was negative")],
          zeros)
    return zeros, zeros

  @property
  def _is_injective(self):
    return False
