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

# Functions "ndtr" and "ndtri" are derived from calculations made in:
# https://root.cern.ch/doc/v608/SpecFuncCephesInv_8cxx_source.html
# In the following email exchange, the author gives his consent to redistribute
# derived works under an Apache 2.0 license.
#
# From: Stephen Moshier <steve@moshier.net>
# Date: Sat, Jun 9, 2018 at 2:36 PM
# Subject: Re: Licensing cephes under Apache (BSD-like) license.
# To: rif <rif@google.com>
#
#
#
# Hello Rif,
#
# Yes, Google may distribute Cephes files under the Apache 2 license.
#
# If clarification is needed, I do not favor BSD over other free licenses.
# I would agree that Apache 2 seems to cover the concern you mentioned
# about sublicensees.
#
# Best wishes for good luck with your projects!
# Steve Moshier
#
#
#
# On Thu, 31 May 2018, rif wrote:
#
# > Hello Steve.
# > My name is Rif. I work on machine learning software at Google.
# >
# > Your cephes software continues to be incredibly useful and widely used. I
# > was wondering whether it would be permissible for us to use the Cephes code
# > under the Apache 2.0 license, which is extremely similar in permissions to
# > the BSD license (Wikipedia comparisons). This would be quite helpful to us
# > in terms of avoiding multiple licenses on software.
# >
# > I'm sorry to bother you with this (I can imagine you're sick of hearing
# > about this by now), but I want to be absolutely clear we're on the level and
# > not misusing your important software. In former conversation with Eugene
# > Brevdo (ebrevdo@google.com), you wrote "If your licensing is similar to BSD,
# > the formal way that has been handled is simply to add a statement to the
# > effect that you are incorporating the Cephes software by permission of the
# > author." I wanted to confirm that (a) we could use the Apache license, (b)
# > that we don't need to (and probably you don't want to) keep getting
# > contacted about individual uses, because your intent is generally to allow
# > this software to be reused under "BSD-like" license, and (c) you're OK
# > letting incorporators decide whether a license is sufficiently BSD-like?
# >
# > Best,
# >
# > rif
# >
# >
# >

"""Special Math Ops."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

__all__ = [
    "erfinv",
    "ndtr",
    "ndtri",
    "log_ndtr",
    "log_cdf_laplace",
]


# log_ndtr uses different functions over the ranges
# (-infty, lower](lower, upper](upper, infty)
# Lower bound values were chosen by examining where the support of ndtr
# appears to be zero, relative to scipy's (which is always 64bit). They were
# then made more conservative just to be safe. (Conservative means use the
# expansion more than we probably need to.) See `NdtrTest` in
# special_math_test.py.
LOGNDTR_FLOAT64_LOWER = np.array(-20, np.float64)
LOGNDTR_FLOAT32_LOWER = np.array(-10, np.float32)

# Upper bound values were chosen by examining for which values of 'x'
# Log[cdf(x)] is 0, after which point we need to use the approximation
# Log[cdf(x)] = Log[1 - cdf(-x)] approx -cdf(-x). We chose a value slightly
# conservative, meaning we use the approximation earlier than needed.
LOGNDTR_FLOAT64_UPPER = np.array(8, np.float64)
LOGNDTR_FLOAT32_UPPER = np.array(5, np.float32)


def ndtr(x, name="ndtr"):
  """Normal distribution function.

  Returns the area under the Gaussian probability density function, integrated
  from minus infinity to x:

  ```
                    1       / x
     ndtr(x)  = ----------  |    exp(-0.5 t**2) dt
                sqrt(2 pi)  /-inf

              = 0.5 (1 + erf(x / sqrt(2)))
              = 0.5 erfc(x / sqrt(2))
  ```

  Args:
    x: `Tensor` of type `float32`, `float64`.
    name: Python string. A name for the operation (default="ndtr").

  Returns:
    ndtr: `Tensor` with `dtype=x.dtype`.

  Raises:
    TypeError: if `x` is not floating-type.
  """

  with ops.name_scope(name, values=[x]):
    x = ops.convert_to_tensor(x, name="x")
    if x.dtype.as_numpy_dtype not in [np.float32, np.float64]:
      raise TypeError(
          "x.dtype=%s is not handled, see docstring for supported types."
          % x.dtype)
    return _ndtr(x)


def _ndtr(x):
  """Implements ndtr core logic."""
  half_sqrt_2 = constant_op.constant(
      0.5 * np.sqrt(2.), dtype=x.dtype, name="half_sqrt_2")
  w = x * half_sqrt_2
  z = math_ops.abs(w)
  y = array_ops.where_v2(
      math_ops.less(z, half_sqrt_2), 1. + math_ops.erf(w),
      array_ops.where_v2(
          math_ops.greater(w, 0.), 2. - math_ops.erfc(z), math_ops.erfc(z)))
  return 0.5 * y


def ndtri(p, name="ndtri"):
  """The inverse of the CDF of the Normal distribution function.

  Returns x such that the area under the pdf from minus infinity to x is equal
  to p.

  A piece-wise rational approximation is done for the function.
  This is a port of the implementation in netlib.

  Args:
    p: `Tensor` of type `float32`, `float64`.
    name: Python string. A name for the operation (default="ndtri").

  Returns:
    x: `Tensor` with `dtype=p.dtype`.

  Raises:
    TypeError: if `p` is not floating-type.
  """

  with ops.name_scope(name, values=[p]):
    p = ops.convert_to_tensor(p, name="p")
    if p.dtype.as_numpy_dtype not in [np.float32, np.float64]:
      raise TypeError(
          "p.dtype=%s is not handled, see docstring for supported types."
          % p.dtype)
    return _ndtri(p)


def _ndtri(p):
  """Implements ndtri core logic."""

  # Constants used in piece-wise rational approximations. Taken from the cephes
  # library:
  # https://root.cern.ch/doc/v608/SpecFuncCephesInv_8cxx_source.html

  p0 = [
      -1.23916583867381258016E0, 1.39312609387279679503E1,
      -5.66762857469070293439E1, 9.80010754185999661536E1,
      -5.99633501014107895267E1
  ]
  q0 = [
      -1.18331621121330003142E0, 1.59056225126211695515E1,
      -8.20372256168333339912E1, 2.00260212380060660359E2,
      -2.25462687854119370527E2, 8.63602421390890590575E1,
      4.67627912898881538453E0, 1.95448858338141759834E0, 1.0
  ]
  p1 = [
      -8.57456785154685413611E-4, -3.50424626827848203418E-2,
      -1.40256079171354495875E-1, 2.18663306850790267539E0,
      1.46849561928858024014E1, 4.40805073893200834700E1,
      5.71628192246421288162E1, 3.15251094599893866154E1,
      4.05544892305962419923E0
  ]
  q1 = [
      -9.33259480895457427372E-4, -3.80806407691578277194E-2,
      -1.42182922854787788574E-1, 2.50464946208309415979E0,
      1.50425385692907503408E1, 4.13172038254672030440E1,
      4.53907635128879210584E1, 1.57799883256466749731E1, 1.0
  ]
  p2 = [
      6.23974539184983293730E-9, 2.65806974686737550832E-6,
      3.01581553508235416007E-4, 1.23716634817820021358E-2,
      2.01485389549179081538E-1, 1.33303460815807542389E0,
      3.93881025292474443415E0, 6.91522889068984211695E0,
      3.23774891776946035970E0
  ]
  q2 = [
      6.79019408009981274425E-9, 2.89247864745380683936E-6,
      3.28014464682127739104E-4, 1.34204006088543189037E-2,
      2.16236993594496635890E-1, 1.37702099489081330271E0,
      3.67983563856160859403E0, 6.02427039364742014255E0, 1.0
  ]

  def _create_polynomial(var, coeffs):
    """Compute n_th order polynomial via Horner's method."""
    coeffs = np.array(coeffs, var.dtype.as_numpy_dtype)
    if not coeffs.size:
      return array_ops.zeros_like(var)
    return coeffs[0] + _create_polynomial(var, coeffs[1:]) * var

  maybe_complement_p = array_ops.where_v2(p > -np.expm1(-2.), 1. - p, p)
  # Write in an arbitrary value in place of 0 for p since 0 will cause NaNs
  # later on. The result from the computation when p == 0 is not used so any
  # number that doesn't result in NaNs is fine.
  sanitized_mcp = array_ops.where_v2(
      maybe_complement_p <= 0.,
      array_ops.fill(array_ops.shape(p), np.array(0.5, p.dtype.as_numpy_dtype)),
      maybe_complement_p)

  # Compute x for p > exp(-2): x/sqrt(2pi) = w + w**3 P0(w**2)/Q0(w**2).
  w = sanitized_mcp - 0.5
  ww = w ** 2
  x_for_big_p = w + w * ww * (_create_polynomial(ww, p0)
                              / _create_polynomial(ww, q0))
  x_for_big_p *= -np.sqrt(2. * np.pi)

  # Compute x for p <= exp(-2): x = z - log(z)/z - (1/z) P(1/z) / Q(1/z),
  # where z = sqrt(-2. * log(p)), and P/Q are chosen between two different
  # arrays based on whether p < exp(-32).
  z = math_ops.sqrt(-2. * math_ops.log(sanitized_mcp))
  first_term = z - math_ops.log(z) / z
  second_term_small_p = (
      _create_polynomial(1. / z, p2) /
      _create_polynomial(1. / z, q2) / z)
  second_term_otherwise = (
      _create_polynomial(1. / z, p1) /
      _create_polynomial(1. / z, q1) / z)
  x_for_small_p = first_term - second_term_small_p
  x_otherwise = first_term - second_term_otherwise

  x = array_ops.where_v2(
      sanitized_mcp > np.exp(-2.), x_for_big_p,
      array_ops.where_v2(z >= 8.0, x_for_small_p, x_otherwise))

  x = array_ops.where_v2(p > 1. - np.exp(-2.), x, -x)
  infinity_scalar = constant_op.constant(np.inf, dtype=p.dtype)
  infinity = array_ops.fill(array_ops.shape(p), infinity_scalar)
  x_nan_replaced = array_ops.where_v2(p <= 0.0, -infinity,
                                      array_ops.where_v2(p >= 1.0, infinity, x))
  return x_nan_replaced


def log_ndtr(x, series_order=3, name="log_ndtr"):
  """Log Normal distribution function.

  For details of the Normal distribution function see `ndtr`.

  This function calculates `(log o ndtr)(x)` by either calling `log(ndtr(x))` or
  using an asymptotic series. Specifically:
  - For `x > upper_segment`, use the approximation `-ndtr(-x)` based on
    `log(1-x) ~= -x, x << 1`.
  - For `lower_segment < x <= upper_segment`, use the existing `ndtr` technique
    and take a log.
  - For `x <= lower_segment`, we use the series approximation of erf to compute
    the log CDF directly.

  The `lower_segment` is set based on the precision of the input:

  ```
  lower_segment = { -20,  x.dtype=float64
                  { -10,  x.dtype=float32
  upper_segment = {   8,  x.dtype=float64
                  {   5,  x.dtype=float32
  ```

  When `x < lower_segment`, the `ndtr` asymptotic series approximation is:

  ```
     ndtr(x) = scale * (1 + sum) + R_N
     scale   = exp(-0.5 x**2) / (-x sqrt(2 pi))
     sum     = Sum{(-1)^n (2n-1)!! / (x**2)^n, n=1:N}
     R_N     = O(exp(-0.5 x**2) (2N+1)!! / |x|^{2N+3})
  ```

  where `(2n-1)!! = (2n-1) (2n-3) (2n-5) ...  (3) (1)` is a
  [double-factorial](https://en.wikipedia.org/wiki/Double_factorial).


  Args:
    x: `Tensor` of type `float32`, `float64`.
    series_order: Positive Python `integer`. Maximum depth to
      evaluate the asymptotic expansion. This is the `N` above.
    name: Python string. A name for the operation (default="log_ndtr").

  Returns:
    log_ndtr: `Tensor` with `dtype=x.dtype`.

  Raises:
    TypeError: if `x.dtype` is not handled.
    TypeError: if `series_order` is a not Python `integer.`
    ValueError:  if `series_order` is not in `[0, 30]`.
  """
  if not isinstance(series_order, int):
    raise TypeError("series_order must be a Python integer.")
  if series_order < 0:
    raise ValueError("series_order must be non-negative.")
  if series_order > 30:
    raise ValueError("series_order must be <= 30.")

  with ops.name_scope(name, values=[x]):
    x = ops.convert_to_tensor(x, name="x")

    if x.dtype.as_numpy_dtype == np.float64:
      lower_segment = LOGNDTR_FLOAT64_LOWER
      upper_segment = LOGNDTR_FLOAT64_UPPER
    elif x.dtype.as_numpy_dtype == np.float32:
      lower_segment = LOGNDTR_FLOAT32_LOWER
      upper_segment = LOGNDTR_FLOAT32_UPPER
    else:
      raise TypeError("x.dtype=%s is not supported." % x.dtype)

    # The basic idea here was ported from:
    #   https://root.cern.ch/doc/v608/SpecFuncCephesInv_8cxx_source.html
    # We copy the main idea, with a few changes
    # * For x >> 1, and X ~ Normal(0, 1),
    #     Log[P[X < x]] = Log[1 - P[X < -x]] approx -P[X < -x],
    #     which extends the range of validity of this function.
    # * We use one fixed series_order for all of 'x', rather than adaptive.
    # * Our docstring properly reflects that this is an asymptotic series, not a
    #   Taylor series. We also provided a correct bound on the remainder.
    # * We need to use the max/min in the _log_ndtr_lower arg to avoid nan when
    #   x=0. This happens even though the branch is unchosen because when x=0
    #   the gradient of a select involves the calculation 1*dy+0*(-inf)=nan
    #   regardless of whether dy is finite. Note that the minimum is a NOP if
    #   the branch is chosen.
    return array_ops.where_v2(
        math_ops.greater(x, upper_segment),
        -_ndtr(-x),  # log(1-x) ~= -x, x << 1  # pylint: disable=invalid-unary-operand-type
        array_ops.where_v2(
            math_ops.greater(x, lower_segment),
            math_ops.log(_ndtr(math_ops.maximum(x, lower_segment))),
            _log_ndtr_lower(math_ops.minimum(x, lower_segment), series_order)))


def _log_ndtr_lower(x, series_order):
  """Asymptotic expansion version of `Log[cdf(x)]`, appropriate for `x<<-1`."""
  x_2 = math_ops.square(x)
  # Log of the term multiplying (1 + sum)
  log_scale = -0.5 * x_2 - math_ops.log(-x) - 0.5 * np.log(2. * np.pi)
  return log_scale + math_ops.log(_log_ndtr_asymptotic_series(x, series_order))


def _log_ndtr_asymptotic_series(x, series_order):
  """Calculates the asymptotic series used in log_ndtr."""
  dtype = x.dtype.as_numpy_dtype
  if series_order <= 0:
    return np.array(1, dtype)
  x_2 = math_ops.square(x)
  even_sum = array_ops.zeros_like(x)
  odd_sum = array_ops.zeros_like(x)
  x_2n = x_2  # Start with x^{2*1} = x^{2*n} with n = 1.
  for n in range(1, series_order + 1):
    y = np.array(_double_factorial(2 * n - 1), dtype) / x_2n
    if n % 2:
      odd_sum += y
    else:
      even_sum += y
    x_2n *= x_2
  return 1. + even_sum - odd_sum


def erfinv(x, name="erfinv"):
  """The inverse function for erf, the error function.

  Args:
    x: `Tensor` of type `float32`, `float64`.
    name: Python string. A name for the operation (default="erfinv").

  Returns:
    x: `Tensor` with `dtype=x.dtype`.

  Raises:
    TypeError: if `x` is not floating-type.
  """

  with ops.name_scope(name, values=[x]):
    x = ops.convert_to_tensor(x, name="x")
    if x.dtype.as_numpy_dtype not in [np.float32, np.float64]:
      raise TypeError(
          "x.dtype=%s is not handled, see docstring for supported types."
          % x.dtype)
    return ndtri((x + 1.0) / 2.0) / np.sqrt(2)


def _double_factorial(n):
  """The double factorial function for small Python integer `n`."""
  return np.prod(np.arange(n, 1, -2))


def log_cdf_laplace(x, name="log_cdf_laplace"):
  """Log Laplace distribution function.

  This function calculates `Log[L(x)]`, where `L(x)` is the cumulative
  distribution function of the Laplace distribution, i.e.

  ```L(x) := 0.5 * int_{-infty}^x e^{-|t|} dt```

  For numerical accuracy, `L(x)` is computed in different ways depending on `x`,

  ```
  x <= 0:
    Log[L(x)] = Log[0.5] + x, which is exact

  0 < x:
    Log[L(x)] = Log[1 - 0.5 * e^{-x}], which is exact
  ```

  Args:
    x: `Tensor` of type `float32`, `float64`.
    name: Python string. A name for the operation (default="log_ndtr").

  Returns:
    `Tensor` with `dtype=x.dtype`.

  Raises:
    TypeError: if `x.dtype` is not handled.
  """

  with ops.name_scope(name, values=[x]):
    x = ops.convert_to_tensor(x, name="x")

    # For x < 0, L(x) = 0.5 * exp{x} exactly, so Log[L(x)] = log(0.5) + x.
    lower_solution = -np.log(2.) + x

    # safe_exp_neg_x = exp{-x} for x > 0, but is
    # bounded above by 1, which avoids
    #   log[1 - 1] = -inf for x = log(1/2), AND
    #   exp{-x} --> inf, for x << -1
    safe_exp_neg_x = math_ops.exp(-math_ops.abs(x))

    # log1p(z) = log(1 + z) approx z for |z| << 1. This approximation is used
    # internally by log1p, rather than being done explicitly here.
    upper_solution = math_ops.log1p(-0.5 * safe_exp_neg_x)

    return array_ops.where_v2(x < 0., lower_solution, upper_solution)
