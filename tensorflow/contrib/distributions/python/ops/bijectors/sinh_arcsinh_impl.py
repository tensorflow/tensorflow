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
"""SinhArcsinh bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import trig
from tensorflow.python.framework import ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector

sinh = trig.sinh
arcsinh = trig.arcsinh
cosh = trig.cosh
log_cosh = trig.log_cosh

__all__ = [
    "SinhArcsinh",
]


class SinhArcsinh(bijector.Bijector):
  """Compute `Y = g(X) = Sinh( (Arcsinh(X) + skewness) * tailweight )`.

  For `skewness in (-inf, inf)` and `tailweight in (0, inf)`, this
  transformation is a
  diffeomorphism of the real line `(-inf, inf)`.  The inverse transform is
  `X = g^{-1}(Y) = Sinh( ArcSinh(Y) / tailweight - skewness )`.

  The `SinhArcsinh` transformation of the Normal is described in
  [Sinh-arcsinh distributions](https://www.jstor.org/stable/27798865)
  This Bijector allows a similar transformation of any distribution supported on
  `(-inf, inf)`.

  #### Meaning of the parameters

  * If `skewness = 0` and `tailweight = 1`, this transform is the identity.
  * Positive (negative) `skewness` leads to positive (negative) skew.
    * positive skew means, for unimodal `X` centered at zero, the mode of `Y` is
      "tilted" to the right.
    * positive skew means positive values of `Y` become more likely, and
      negative values become less likely.
  * Larger (smaller) `tailweight` leads to fatter (thinner) tails.
    * Fatter tails mean larger values of `|Y|` become more likely.
    * If `X` is a unit Normal, `tailweight < 1` leads to a distribution that is
      "flat" around `Y = 0`, and a very steep drop-off in the tails.
    * If `X` is a unit Normal, `tailweight > 1` leads to a distribution more
      peaked at the mode with heavier tails.

  To see the argument about the tails, note that for `|X| >> 1` and
  `|X| >> (|skewness| * tailweight)**tailweight`, we have
  `Y approx 0.5 X**tailweight e**(sign(X) skewness * tailweight)`.
  """

  def __init__(self,
               skewness=0.,
               tailweight=1.,
               event_ndims=0,
               validate_args=False,
               name="sinh_arcsinh"):
    """Instantiates the `SinhArcsinh` bijector.

    Args:
      skewness:  Skewness parameter.  Float-type `Tensor`.
      tailweight:  Tailweight parameter.  Positive `Tensor` of same `dtype` as
        `skewness`
        and broadcastable `shape`.
      event_ndims: Python scalar indicating the number of dimensions associated
        with a particular draw from the distribution.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    self._graph_parents = []
    self._name = name
    self._validate_args = validate_args
    with self._name_scope("init", values=[skewness, tailweight]):
      self._skewness = ops.convert_to_tensor(skewness, name="skewness")
      self._tailweight = ops.convert_to_tensor(tailweight, name="tailweight")
      check_ops.assert_same_float_dtype([self._skewness, self._tailweight])
      if validate_args:
        self._tailweight = control_flow_ops.with_dependencies([
            check_ops.assert_positive(
                self._tailweight,
                message="Argument tailweight was not positive")
        ], self._tailweight)
    super(SinhArcsinh, self).__init__(
        event_ndims=event_ndims, validate_args=validate_args, name=name)

  @property
  def skewness(self):
    """The `skewness` in: `Y  = Sinh((Arcsinh(X) + skewness) * tailweight)`."""
    return self._skewness

  @property
  def tailweight(self):
    """The `tailweight` in: `Y = Sinh((Arcsinh(X) + skewness) * tailweight)`."""
    return self._tailweight

  def _forward(self, x):
    return sinh((arcsinh(x) + self.skewness) * self.tailweight)

  def _inverse(self, y):
    return sinh(arcsinh(y) / self.tailweight - self.skewness)

  def _inverse_log_det_jacobian(self, y):
    # x = sinh(arcsinh(y) / tailweight - skewness)
    # Using sinh' = cosh, arcsinh'(y) = 1 / sqrt(y**2 + 1),
    # dx/dy
    # = cosh(arcsinh(y) / tailweight - skewness)
    #     / (tailweight * sqrt(y**2 + 1))
    event_dims = self._event_dims_tensor(y)
    return math_ops.reduce_sum(
        log_cosh(arcsinh(y) / self.tailweight - self.skewness) -
        math_ops.log(self.tailweight) - 0.5 * math_ops.log1p(y**2),
        axis=event_dims)

  def _forward_log_det_jacobian(self, x):
    # y = sinh((arcsinh(x) + skewness) * tailweight)
    # Using sinh' = cosh, arcsinh'(x) = 1 / sqrt(x**2 + 1),
    # dy/dx
    # = cosh((arcsinh(x) + skewness) * tailweight) * tailweight / sqrt(x**2 + 1)
    event_dims = self._event_dims_tensor(x)
    return math_ops.reduce_sum(
        log_cosh((arcsinh(x) + self.skewness) * self.tailweight) +
        math_ops.log(self.tailweight) - 0.5 * math_ops.log1p(x**2),
        axis=event_dims)
