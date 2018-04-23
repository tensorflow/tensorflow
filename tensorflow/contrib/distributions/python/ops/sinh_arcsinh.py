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
"""SinhArcsinh transformation of a distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import bijectors
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.distributions import normal
from tensorflow.python.ops.distributions import transformed_distribution

__all__ = [
    "SinhArcsinh",
]


class SinhArcsinh(transformed_distribution.TransformedDistribution):
  """The SinhArcsinh transformation of a distribution on `(-inf, inf)`.

  This distribution models a random variable, making use of
  a `SinhArcsinh` transformation (which has adjustable tailweight and skew),
  a rescaling, and a shift.

  The `SinhArcsinh` transformation of the Normal is described in great depth in
  [Sinh-arcsinh distributions](https://www.jstor.org/stable/27798865).
  Here we use a slightly different parameterization, in terms of `tailweight`
  and `skewness`.  Additionally we allow for distributions other than Normal,
  and control over `scale` as well as a "shift" parameter `loc`.

  #### Mathematical Details

  Given random variable `Z`, we define the SinhArcsinh
  transformation of `Z`, `Y`, parameterized by
  `(loc, scale, skewness, tailweight)`, via the relation:

  ```
  Y := loc + scale * F(Z) * (2 / F_0(2))
  F(Z) := Sinh( (Arcsinh(Z) + skewness) * tailweight )
  F_0(Z) := Sinh( Arcsinh(Z) * tailweight )
  ```

  This distribution is similar to the location-scale transformation
  `L(Z) := loc + scale * Z` in the following ways:

  * If `skewness = 0` and `tailweight = 1` (the defaults), `F(Z) = Z`, and then
    `Y = L(Z)` exactly.
  * `loc` is used in both to shift the result by a constant factor.
  * The multiplication of `scale` by `2 / F_0(2)` ensures that if `skewness = 0`
    `P[Y - loc <= 2 * scale] = P[L(Z) - loc <= 2 * scale]`.
    Thus it can be said that the weights in the tails of `Y` and `L(Z)` beyond
    `loc + 2 * scale` are the same.

  This distribution is different than `loc + scale * Z` due to the
  reshaping done by `F`:

  * Positive (negative) `skewness` leads to positive (negative) skew.
    * positive skew means, the mode of `F(Z)` is "tilted" to the right.
    * positive skew means positive values of `F(Z)` become more likely, and
      negative values become less likely.
  * Larger (smaller) `tailweight` leads to fatter (thinner) tails.
    * Fatter tails mean larger values of `|F(Z)|` become more likely.
    * `tailweight < 1` leads to a distribution that is "flat" around `Y = loc`,
      and a very steep drop-off in the tails.
    * `tailweight > 1` leads to a distribution more peaked at the mode with
      heavier tails.

  To see the argument about the tails, note that for `|Z| >> 1` and
  `|Z| >> (|skewness| * tailweight)**tailweight`, we have
  `Y approx 0.5 Z**tailweight e**(sign(Z) skewness * tailweight)`.

  To see the argument regarding multiplying `scale` by `2 / F_0(2)`,

  ```
  P[(Y - loc) / scale <= 2] = P[F(Z) * (2 / F_0(2)) <= 2]
                            = P[F(Z) <= F_0(2)]
                            = P[Z <= 2]  (if F = F_0).
  ```
  """

  def __init__(self,
               loc,
               scale,
               skewness=None,
               tailweight=None,
               distribution=None,
               validate_args=False,
               allow_nan_stats=True,
               name="SinhArcsinh"):
    """Construct SinhArcsinh distribution on `(-inf, inf)`.

    Arguments `(loc, scale, skewness, tailweight)` must have broadcastable shape
    (indexing batch dimensions).  They must all have the same `dtype`.

    Args:
      loc: Floating-point `Tensor`.
      scale:  `Tensor` of same `dtype` as `loc`.
      skewness:  Skewness parameter.  Default is `0.0` (no skew).
      tailweight:  Tailweight parameter. Default is `1.0` (unchanged tailweight)
      distribution: `tf.Distribution`-like instance. Distribution that is
        transformed to produce this distribution.
        Default is `tf.distributions.Normal(0., 1.)`.
        Must be a scalar-batch, scalar-event distribution.  Typically
        `distribution.reparameterization_type = FULLY_REPARAMETERIZED` or it is
        a function of non-trainable parameters. WARNING: If you backprop through
        a `SinhArcsinh` sample and `distribution` is not
        `FULLY_REPARAMETERIZED` yet is a function of trainable variables, then
        the gradient will be incorrect!
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = locals()

    with ops.name_scope(name,
                        values=[loc, scale, skewness, tailweight]) as name:
      loc = ops.convert_to_tensor(loc, name="loc")
      dtype = loc.dtype
      scale = ops.convert_to_tensor(scale, name="scale", dtype=dtype)
      tailweight = 1. if tailweight is None else tailweight
      has_default_skewness = skewness is None
      skewness = 0. if skewness is None else skewness
      tailweight = ops.convert_to_tensor(
          tailweight, name="tailweight", dtype=dtype)
      skewness = ops.convert_to_tensor(skewness, name="skewness", dtype=dtype)

      batch_shape = distribution_util.get_broadcast_shape(
          loc, scale, tailweight, skewness)

      # Recall, with Z a random variable,
      #   Y := loc + C * F(Z),
      #   F(Z) := Sinh( (Arcsinh(Z) + skewness) * tailweight )
      #   F_0(Z) := Sinh( Arcsinh(Z) * tailweight )
      #   C := 2 * scale / F_0(2)
      if distribution is None:
        distribution = normal.Normal(
            loc=array_ops.zeros([], dtype=dtype),
            scale=array_ops.ones([], dtype=dtype),
            allow_nan_stats=allow_nan_stats)
      else:
        asserts = distribution_util.maybe_check_scalar_distribution(
            distribution, dtype, validate_args)
        if asserts:
          loc = control_flow_ops.with_dependencies(asserts, loc)

      # Make the SAS bijector, 'F'.
      f = bijectors.SinhArcsinh(
          skewness=skewness, tailweight=tailweight)
      if has_default_skewness:
        f_noskew = f
      else:
        f_noskew = bijectors.SinhArcsinh(
            skewness=skewness.dtype.as_numpy_dtype(0.),
            tailweight=tailweight)

      # Make the AffineScalar bijector, Z --> loc + scale * Z (2 / F_0(2))
      c = 2 * scale / f_noskew.forward(ops.convert_to_tensor(2, dtype=dtype))
      affine = bijectors.AffineScalar(
          shift=loc,
          scale=c,
          validate_args=validate_args)

      bijector = bijectors.Chain([affine, f])

      super(SinhArcsinh, self).__init__(
          distribution=distribution,
          bijector=bijector,
          batch_shape=batch_shape,
          validate_args=validate_args,
          name=name)
    self._parameters = parameters
    self._loc = loc
    self._scale = scale
    self._tailweight = tailweight
    self._skewness = skewness

  @property
  def loc(self):
    """The `loc` in `Y := loc + scale @ F(Z) * (2 / F(2))."""
    return self._loc

  @property
  def scale(self):
    """The `LinearOperator` `scale` in `Y := loc + scale @ F(Z) * (2 / F(2))."""
    return self._scale

  @property
  def tailweight(self):
    """Controls the tail decay.  `tailweight > 1` means faster than Normal."""
    return self._tailweight

  @property
  def skewness(self):
    """Controls the skewness.  `Skewness > 0` means right skew."""
    return self._skewness
