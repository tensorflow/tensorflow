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
"""Softplus bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.ops.distributions import util as distribution_util


__all__ = [
    "Softplus",
]


class Softplus(bijector.Bijector):
  """Bijector which computes `Y = g(X) = Log[1 + exp(X)]`.

  The softplus `Bijector` has the following two useful properties:

  * The domain is the positive real numbers
  * `softplus(x) approx x`, for large `x`, so it does not overflow as easily as
    the `Exp` `Bijector`.

  The optional nonzero `hinge_softness` parameter changes the transition at
  zero.  With `hinge_softness = c`, the bijector is:

    ```f_c(x) := c * g(x / c) = c * Log[1 + exp(x / c)].```

  For large `x >> 1`, `c * Log[1 + exp(x / c)] approx c * Log[exp(x / c)] = x`,
  so the behavior for large `x` is the same as the standard softplus.

  As `c > 0` approaches 0 from the right, `f_c(x)` becomes less and less soft,
  approaching `max(0, x)`.

  * `c = 1` is the default.
  * `c > 0` but small means `f(x) approx ReLu(x) = max(0, x)`.
  * `c < 0` flips sign and reflects around the `y-axis`: `f_{-c}(x) = -f_c(-x)`.
  * `c = 0` results in a non-bijective transformation and triggers an exception.

    Example Use:

    ```python
    # Create the Y=g(X)=softplus(X) transform which works only on Tensors with 1
    # batch ndim and 2 event ndims (i.e., vector of matrices).
    softplus = Softplus()
    x = [[[1., 2],
          [3, 4]],
         [[5, 6],
          [7, 8]]]
    log(1 + exp(x)) == softplus.forward(x)
    log(exp(x) - 1) == softplus.inverse(x)
    ```

    Note: log(.) and exp(.) are applied element-wise but the Jacobian is a
    reduction over the event space.
  """

  @distribution_util.AppendDocstring(
      kwargs_dict={
          "hinge_softness": (
              "Nonzero floating point `Tensor`.  Controls the softness of what "
              "would otherwise be a kink at the origin.  Default is 1.0")})
  def __init__(self,
               hinge_softness=None,
               validate_args=False,
               name="softplus"):
    with ops.name_scope(name, values=[hinge_softness]):
      if hinge_softness is not None:
        self._hinge_softness = ops.convert_to_tensor(
            hinge_softness, name="hinge_softness")
      else:
        self._hinge_softness = None
      if validate_args:
        nonzero_check = check_ops.assert_none_equal(
            ops.convert_to_tensor(
                0, dtype=self.hinge_softness.dtype),
            self.hinge_softness,
            message="hinge_softness must be non-zero")
        self._hinge_softness = control_flow_ops.with_dependencies(
            [nonzero_check], self.hinge_softness)

    super(Softplus, self).__init__(
        forward_min_event_ndims=0,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    if self.hinge_softness is None:
      return nn_ops.softplus(x)
    hinge_softness = math_ops.cast(self.hinge_softness, x.dtype)
    return hinge_softness * nn_ops.softplus(x / hinge_softness)

  def _inverse(self, y):
    if self.hinge_softness is None:
      return distribution_util.softplus_inverse(y)
    hinge_softness = math_ops.cast(self.hinge_softness, y.dtype)
    return hinge_softness * distribution_util.softplus_inverse(
        y / hinge_softness)

  def _inverse_log_det_jacobian(self, y):
    # Could also do:
    #   ildj = math_ops.reduce_sum(y - distribution_util.softplus_inverse(y),
    #                              axis=event_dims)
    # but the following is more numerically stable. Ie,
    # Y = Log[1 + exp{X}] ==> X = Log[exp{Y} - 1]
    # ==> dX/dY = exp{Y} / (exp{Y} - 1)
    #           = 1 / (1 - exp{-Y}),
    # which is the most stable for large Y > 0. For small Y, we use
    # 1 - exp{-Y} approx Y.
    if self.hinge_softness is not None:
      y /= math_ops.cast(self.hinge_softness, y.dtype)
    return -math_ops.log(-math_ops.expm1(-y))

  def _forward_log_det_jacobian(self, x):
    if self.hinge_softness is not None:
      x /= math_ops.cast(self.hinge_softness, x.dtype)
    return -nn_ops.softplus(-x)

  @property
  def hinge_softness(self):
    return self._hinge_softness
