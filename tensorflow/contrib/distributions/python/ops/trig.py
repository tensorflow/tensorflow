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
"""Trigonometric functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops

__all__ = [
    "sinh",
    "arcsinh",
    "cosh",
    "log_cosh",
]


def sinh(x, name="sinh"):
  """Hyperbolic sin:  `sinh(x) = (e**x - e**-x) / 2`.

  For `x in (-inf, inf)`, `arcsinh(sinh(x)) = sinh(arcsinh(x)) = x.`

  Args:
    x:  Numeric `Tensor`.
    name:  A string name to prepend to created Ops.

  Returns:
    Numeric `Tensor` of same `shape` and `dtype` as `x`.
  """
  with ops.name_scope(name):
    # For x near zero, this could be replaced with the identity x --> x as a
    # Taylor approximation.  The Taylor approximation would be more accurate,
    # and the current implementation will equal zero "too often", specifically
    # for |x| small enough so that exp(x) = 1 (up to floating point precision).
    # Although less accurate, the present implementation is still safe for
    # antipated use.  Why?  No one would use this function and expect that zero
    # was not a possible output value.
    x = ops.convert_to_tensor(x, name="x")
    return 0.5 * (math_ops.exp(x) - math_ops.exp(-x))


def arcsinh(x, name="arcsinh"):
  """Inverse hyperbolic sin:  `arcsinh(x) = log(x + sqrt(x**2 + 1))`.

  For `x in (-inf, inf)`, `arcsinh(sinh(x)) = sinh(arcsinh(x)) = x.`

  Args:
    x:  Numeric `Tensor`.
    name:  A string name to prepend to created Ops.

  Returns:
    Numeric `Tensor` of same `shape` and `dtype` as `x`.
  """
  with ops.name_scope(name):
    x = ops.convert_to_tensor(x, name="x")
    finfo = np.finfo(x.dtype.as_numpy_dtype)

    # To compute stable arcsinh(x), we will compute various approximations of
    # z := x + sqrt(x**2 + 1), and then arcsinh(x) = log(z).
    # Different approximations are used over different values of x, then the
    # result is pieced together using tf.where.  Since NaN propagate through the
    # unselected branch during grdients of where, care is taken to ensure that
    # every approximation is finite for all input values.

    # For x near zero, the straightforward formula for z is fine.
    # This formula will have trouble once x < 0 and x**2 + 1 = x**2, since then
    # z = 0 (numerically), and then we have log(z) = log(0) = -inf
    # This formula also has trouble once x > sqrt(finfo.max), since then
    # x**2 = inf, and thus z = inf.  Therefore we clip.
    x_near_zero = clip_ops.clip_by_value(x, -1., 1.)
    x_is_near_zero = math_ops.abs(x) < 1.
    z_for_x_near_zero = x + math_ops.sqrt(x_near_zero**2 + 1)

    # Some cutoffs.
    # Important!  Keep these cutoffs in sync with the tests, which use cutoffs
    # of the exact same name.
    # very_big_cutoff**2 = finfo.max, the maximum representable value.
    # very_small_cutoff could have been defined as 1 / sqrt(eps), so that for
    # x < very_small_cutoff, 1 / x**2 + 1 = 1, which causes trouble.
    # The 0.01 was added in order to make this match numpy in 32bit
    # as much as possible.  Anything < 1 should be stable.
    very_small_cutoff = -0.01 / np.sqrt(finfo.eps)
    very_big_cutoff = np.sqrt(finfo.max)

    # For very_small_cutoff < x < -1, and 1 < x < very_big_cutoff, and
    # x != 0, we can use
    # z = sqrt(x**2 + 1) = |x| * sqrt(1 + 1 / x**2).
    # This formula has trouble if x < -sqrt(eps) since then 1 + 1 / x**2 = 1,
    # and then we get z = x + |x| = 0, thus returning log(0) = -inf.
    # This formula also has trouble if x**2 = Inf.  Therefore we clip.
    # This formula also has trouble if x = 0, since then we have 1 / 0**2 = inf.
    x_not_near_zero = array_ops.where(
        x >= 0.,
        math_ops.maximum(x, 1.),
        math_ops.minimum(x, -1.))
    x_clipped_moderate_or_big = clip_ops.clip_by_value(
        x_not_near_zero, very_small_cutoff, very_big_cutoff)
    z_for_moderate_or_big_x = x + math_ops.abs(x) * math_ops.sqrt(
        1. + 1. / x_clipped_moderate_or_big**2)

    # For x < very_small_cutoff, we use the first order Taylor series,
    # sqrt(1 + 1 / x**2) approx 1 + 1 / (2 * x**2)
    # This formula has trouble for x = 0.
    x_is_very_small = x < very_small_cutoff
    z_for_very_small_x = 1 / (2. * math_ops.abs(x_not_near_zero))

    z = array_ops.where(
        x_is_near_zero,
        z_for_x_near_zero,
        array_ops.where(
            x_is_very_small,
            z_for_very_small_x,
            z_for_moderate_or_big_x))

    return math_ops.log(z)


def cosh(x, name="cosh"):
  """Hyperbolic cosine:  `cosh(x) = (e**x + e**-x) / 2`.

  For `x in (-inf, inf)`, `arccosh(cosh(x)) = cosh(arccosh(x)) = x.`

  Args:
    x:  Numeric `Tensor`.
    name:  A string name to prepend to created Ops.

  Returns:
    Numeric `Tensor` of same `shape` and `dtype` as `x`.
  """
  with ops.name_scope(name):
    x = ops.convert_to_tensor(x, name="x")
    return 0.5 * (math_ops.exp(x) + math_ops.exp(-x))


def log_cosh(x, name="log_cosh"):
  """Logarithm of hyperbolic cosine:  `log_cosh(x) = Log[(e**x + e**-x) / 2]`.

  Args:
    x:  Numeric `Tensor`.
    name:  A string name to prepend to created Ops.

  Returns:
    Numeric `Tensor` of same `shape` and `dtype` as `x`.
  """
  # For large |x| >> 1, e**x will become Inf.  So we need to approximate
  # Log[e**x + e**-x] approx |x|.
  # We also need to ensure that large |x| is never fed to the exponential func.
  with ops.name_scope(name):
    x = ops.convert_to_tensor(x, name="x")
    large_x_value = 0.9 * np.log(np.finfo(x.dtype.as_numpy_dtype).max)
    x_capped = clip_ops.clip_by_value(x, -large_x_value, large_x_value)
    return array_ops.where(
        math_ops.abs(x) > large_x_value,
        math_ops.abs(x) - np.log(2).astype(x.dtype.as_numpy_dtype),
        math_ops.log(cosh(x_capped)))
