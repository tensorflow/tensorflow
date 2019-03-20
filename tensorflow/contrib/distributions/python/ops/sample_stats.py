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
"""Functions for computing statistics of samples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import util
from tensorflow.python.ops.signal import fft_ops

__all__ = [
    "auto_correlation",
    "percentile",
]


# TODO(langmore) Write separate versions of this for real/complex dtype, taking
# advantage of optimized real-fft ops.
def auto_correlation(
    x,
    axis=-1,
    max_lags=None,
    center=True,
    normalize=True,
    name="auto_correlation"):
  """Auto correlation along one axis.

  Given a `1-D` wide sense stationary (WSS) sequence `X`, the auto correlation
  `RXX` may be defined as  (with `E` expectation and `Conj` complex conjugate)

  ```
  RXX[m] := E{ W[m] Conj(W[0]) } = E{ W[0] Conj(W[-m]) },
  W[n]   := (X[n] - MU) / S,
  MU     := E{ X[0] },
  S**2   := E{ (X[0] - MU) Conj(X[0] - MU) }.
  ```

  This function takes the viewpoint that `x` is (along one axis) a finite
  sub-sequence of a realization of (WSS) `X`, and then uses `x` to produce an
  estimate of `RXX[m]` as follows:

  After extending `x` from length `L` to `inf` by zero padding, the auto
  correlation estimate `rxx[m]` is computed for `m = 0, 1, ..., max_lags` as

  ```
  rxx[m] := (L - m)**-1 sum_n w[n + m] Conj(w[n]),
  w[n]   := (x[n] - mu) / s,
  mu     := L**-1 sum_n x[n],
  s**2   := L**-1 sum_n (x[n] - mu) Conj(x[n] - mu)
  ```

  The error in this estimate is proportional to `1 / sqrt(len(x) - m)`, so users
  often set `max_lags` small enough so that the entire output is meaningful.

  Note that since `mu` is an imperfect estimate of `E{ X[0] }`, and we divide by
  `len(x) - m` rather than `len(x) - m - 1`, our estimate of auto correlation
  contains a slight bias, which goes to zero as `len(x) - m --> infinity`.

  Args:
    x:  `float32` or `complex64` `Tensor`.
    axis:  Python `int`. The axis number along which to compute correlation.
      Other dimensions index different batch members.
    max_lags:  Positive `int` tensor.  The maximum value of `m` to consider
      (in equation above).  If `max_lags >= x.shape[axis]`, we effectively
      re-set `max_lags` to `x.shape[axis] - 1`.
    center:  Python `bool`.  If `False`, do not subtract the mean estimate `mu`
      from `x[n]` when forming `w[n]`.
    normalize:  Python `bool`.  If `False`, do not divide by the variance
      estimate `s**2` when forming `w[n]`.
    name:  `String` name to prepend to created ops.

  Returns:
    `rxx`: `Tensor` of same `dtype` as `x`.  `rxx.shape[i] = x.shape[i]` for
      `i != axis`, and `rxx.shape[axis] = max_lags + 1`.

  Raises:
    TypeError:  If `x` is not a supported type.
  """
  # Implementation details:
  # Extend length N / 2 1-D array x to length N by zero padding onto the end.
  # Then, set
  #   F[x]_k := sum_n x_n exp{-i 2 pi k n / N }.
  # It is not hard to see that
  #   F[x]_k Conj(F[x]_k) = F[R]_k, where
  #   R_m := sum_n x_n Conj(x_{(n - m) mod N}).
  # One can also check that R_m / (N / 2 - m) is an unbiased estimate of RXX[m].

  # Since F[x] is the DFT of x, this leads us to a zero-padding and FFT/IFFT
  # based version of estimating RXX.
  # Note that this is a special case of the Wiener-Khinchin Theorem.
  with ops.name_scope(name, values=[x]):
    x = ops.convert_to_tensor(x, name="x")

    # Rotate dimensions of x in order to put axis at the rightmost dim.
    # FFT op requires this.
    rank = util.prefer_static_rank(x)
    if axis < 0:
      axis = rank + axis
    shift = rank - 1 - axis
    # Suppose x.shape[axis] = T, so there are T "time" steps.
    #   ==> x_rotated.shape = B + [T],
    # where B is x_rotated's batch shape.
    x_rotated = util.rotate_transpose(x, shift)

    if center:
      x_rotated -= math_ops.reduce_mean(x_rotated, axis=-1, keepdims=True)

    # x_len = N / 2 from above explanation.  The length of x along axis.
    # Get a value for x_len that works in all cases.
    x_len = util.prefer_static_shape(x_rotated)[-1]

    # TODO(langmore) Investigate whether this zero padding helps or hurts.  At
    # the moment is necessary so that all FFT implementations work.
    # Zero pad to the next power of 2 greater than 2 * x_len, which equals
    # 2**(ceil(Log_2(2 * x_len))).  Note: Log_2(X) = Log_e(X) / Log_e(2).
    x_len_float64 = math_ops.cast(x_len, np.float64)
    target_length = math_ops.pow(
        np.float64(2.),
        math_ops.ceil(math_ops.log(x_len_float64 * 2) / np.log(2.)))
    pad_length = math_ops.cast(target_length - x_len_float64, np.int32)

    # We should have:
    # x_rotated_pad.shape = x_rotated.shape[:-1] + [T + pad_length]
    #                     = B + [T + pad_length]
    x_rotated_pad = util.pad(x_rotated, axis=-1, back=True, count=pad_length)

    dtype = x.dtype
    if not dtype.is_complex:
      if not dtype.is_floating:
        raise TypeError("Argument x must have either float or complex dtype"
                        " found: {}".format(dtype))
      x_rotated_pad = math_ops.complex(x_rotated_pad,
                                       dtype.real_dtype.as_numpy_dtype(0.))

    # Autocorrelation is IFFT of power-spectral density (up to some scaling).
    fft_x_rotated_pad = fft_ops.fft(x_rotated_pad)
    spectral_density = fft_x_rotated_pad * math_ops.conj(fft_x_rotated_pad)
    # shifted_product is R[m] from above detailed explanation.
    # It is the inner product sum_n X[n] * Conj(X[n - m]).
    shifted_product = fft_ops.ifft(spectral_density)

    # Cast back to real-valued if x was real to begin with.
    shifted_product = math_ops.cast(shifted_product, dtype)

    # Figure out if we can deduce the final static shape, and set max_lags.
    # Use x_rotated as a reference, because it has the time dimension in the far
    # right, and was created before we performed all sorts of crazy shape
    # manipulations.
    know_static_shape = True
    if not x_rotated.shape.is_fully_defined():
      know_static_shape = False
    if max_lags is None:
      max_lags = x_len - 1
    else:
      max_lags = ops.convert_to_tensor(max_lags, name="max_lags")
      max_lags_ = tensor_util.constant_value(max_lags)
      if max_lags_ is None or not know_static_shape:
        know_static_shape = False
        max_lags = math_ops.minimum(x_len - 1, max_lags)
      else:
        max_lags = min(x_len - 1, max_lags_)

    # Chop off the padding.
    # We allow users to provide a huge max_lags, but cut it off here.
    # shifted_product_chopped.shape = x_rotated.shape[:-1] + [max_lags]
    shifted_product_chopped = shifted_product[..., :max_lags + 1]

    # If possible, set shape.
    if know_static_shape:
      chopped_shape = x_rotated.shape.as_list()
      chopped_shape[-1] = min(x_len, max_lags + 1)
      shifted_product_chopped.set_shape(chopped_shape)

    # Recall R[m] is a sum of N / 2 - m nonzero terms x[n] Conj(x[n - m]).  The
    # other terms were zeros arising only due to zero padding.
    # `denominator = (N / 2 - m)` (defined below) is the proper term to
    # divide by to make this an unbiased estimate of the expectation
    # E[X[n] Conj(X[n - m])].
    x_len = math_ops.cast(x_len, dtype.real_dtype)
    max_lags = math_ops.cast(max_lags, dtype.real_dtype)
    denominator = x_len - math_ops.range(0., max_lags + 1.)
    denominator = math_ops.cast(denominator, dtype)
    shifted_product_rotated = shifted_product_chopped / denominator

    if normalize:
      shifted_product_rotated /= shifted_product_rotated[..., :1]

    # Transpose dimensions back to those of x.
    return util.rotate_transpose(shifted_product_rotated, -shift)


# TODO(langmore) To make equivalent to numpy.percentile:
#  Make work with a sequence of floats or single float for 'q'.
#  Make work with "linear", "midpoint" interpolation. (linear should be default)
def percentile(x,
               q,
               axis=None,
               interpolation=None,
               keep_dims=False,
               validate_args=False,
               name=None):
  """Compute the `q`-th percentile of `x`.

  Given a vector `x`, the `q`-th percentile of `x` is the value `q / 100` of the
  way from the minimum to the maximum in a sorted copy of `x`.

  The values and distances of the two nearest neighbors as well as the
  `interpolation` parameter will determine the percentile if the normalized
  ranking does not match the location of `q` exactly.

  This function is the same as the median if `q = 50`, the same as the minimum
  if `q = 0` and the same as the maximum if `q = 100`.


  ```python
  # Get 30th percentile with default ('nearest') interpolation.
  x = [1., 2., 3., 4.]
  percentile(x, q=30.)
  ==> 2.0

  # Get 30th percentile with 'lower' interpolation
  x = [1., 2., 3., 4.]
  percentile(x, q=30., interpolation='lower')
  ==> 1.0

  # Get 100th percentile (maximum).  By default, this is computed over every dim
  x = [[1., 2.]
       [3., 4.]]
  percentile(x, q=100.)
  ==> 4.0

  # Treat the leading dim as indexing samples, and find the 100th quantile (max)
  # over all such samples.
  x = [[1., 2.]
       [3., 4.]]
  percentile(x, q=100., axis=[0])
  ==> [3., 4.]
  ```

  Compare to `numpy.percentile`.

  Args:
    x:  Floating point `N-D` `Tensor` with `N > 0`.  If `axis` is not `None`,
      `x` must have statically known number of dimensions.
    q:  Scalar `Tensor` in `[0, 100]`. The percentile.
    axis:  Optional `0-D` or `1-D` integer `Tensor` with constant values.
      The axis that hold independent samples over which to return the desired
      percentile.  If `None` (the default), treat every dimension as a sample
      dimension, returning a scalar.
    interpolation : {"lower", "higher", "nearest"}.  Default: "nearest"
      This optional parameter specifies the interpolation method to
      use when the desired quantile lies between two data points `i < j`:
        * lower: `i`.
        * higher: `j`.
        * nearest: `i` or `j`, whichever is nearest.
    keep_dims:  Python `bool`. If `True`, the last dimension is kept with size 1
      If `False`, the last dimension is removed from the output shape.
    validate_args:  Whether to add runtime checks of argument validity.
      If False, and arguments are incorrect, correct behavior is not guaranteed.
    name:  A Python string name to give this `Op`.  Default is "percentile"

  Returns:
    A `(N - len(axis))` dimensional `Tensor` of same dtype as `x`, or, if
      `axis` is `None`, a scalar.

  Raises:
    ValueError:  If argument 'interpolation' is not an allowed type.
  """
  name = name or "percentile"
  allowed_interpolations = {"lower", "higher", "nearest"}

  if interpolation is None:
    interpolation = "nearest"
  else:
    if interpolation not in allowed_interpolations:
      raise ValueError("Argument 'interpolation' must be in %s.  Found %s" %
                       (allowed_interpolations, interpolation))

  with ops.name_scope(name, values=[x, q]):
    x = ops.convert_to_tensor(x, name="x")
    # Double is needed here and below, else we get the wrong index if the array
    # is huge along axis.
    q = math_ops.cast(q, dtypes.float64, name="q")
    _get_static_ndims(q, expect_ndims=0)

    if validate_args:
      q = control_flow_ops.with_dependencies([
          check_ops.assert_rank(q, 0),
          check_ops.assert_greater_equal(q, math_ops.cast(0., dtypes.float64)),
          check_ops.assert_less_equal(q, math_ops.cast(100., dtypes.float64))
      ], q)

    if axis is None:
      y = array_ops.reshape(x, [-1])
    else:
      axis = ops.convert_to_tensor(axis, name="axis")
      check_ops.assert_integer(axis)
      axis_ndims = _get_static_ndims(
          axis, expect_static=True, expect_ndims_no_more_than=1)
      axis_const = tensor_util.constant_value(axis)
      if axis_const is None:
        raise ValueError(
            "Expected argument 'axis' to be statically available.  Found: %s" %
            axis)
      axis = axis_const
      if axis_ndims == 0:
        axis = [axis]
      axis = [int(a) for a in axis]
      x_ndims = _get_static_ndims(
          x, expect_static=True, expect_ndims_at_least=1)
      axis = _make_static_axis_non_negative(axis, x_ndims)
      y = _move_dims_to_flat_end(x, axis, x_ndims)

    frac_at_q_or_above = 1. - q / 100.
    d = math_ops.cast(array_ops.shape(y)[-1], dtypes.float64)

    if interpolation == "lower":
      index = math_ops.ceil((d - 1) * frac_at_q_or_above)
    elif interpolation == "higher":
      index = math_ops.floor((d - 1) * frac_at_q_or_above)
    elif interpolation == "nearest":
      index = math_ops.round((d - 1) * frac_at_q_or_above)

    # If d is gigantic, then we would have d == d - 1, even in double... So
    # let's use max/min to avoid out of bounds errors.
    d = array_ops.shape(y)[-1]
    # d - 1 will be distinct from d in int32.
    index = clip_ops.clip_by_value(math_ops.cast(index, dtypes.int32), 0, d - 1)

    # Sort everything, not just the top 'k' entries, which allows multiple calls
    # to sort only once (under the hood) and use CSE.
    sorted_y = _sort_tensor(y)

    # result.shape = B
    result = sorted_y[..., index]
    result.set_shape(y.get_shape()[:-1])

    if keep_dims:
      if axis is None:
        # ones_vec = [1, 1,..., 1], total length = len(S) + len(B).
        ones_vec = array_ops.ones(
            shape=[_get_best_effort_ndims(x)], dtype=dtypes.int32)
        result *= array_ops.ones(ones_vec, dtype=x.dtype)
      else:
        result = _insert_back_keep_dims(result, axis)

    return result


def _get_static_ndims(x,
                      expect_static=False,
                      expect_ndims=None,
                      expect_ndims_no_more_than=None,
                      expect_ndims_at_least=None):
  """Get static number of dimensions and assert that some expectations are met.

  This function returns the number of dimensions "ndims" of x, as a Python int.

  The optional expect arguments are used to check the ndims of x, but this is
  only done if the static ndims of x is not None.

  Args:
    x:  A Tensor.
    expect_static:  Expect `x` to have statically defined `ndims`.
    expect_ndims:  Optional Python integer.  If provided, assert that x has
      number of dimensions equal to this.
    expect_ndims_no_more_than:  Optional Python integer.  If provided, assert
      that x has no more than this many dimensions.
    expect_ndims_at_least:  Optional Python integer.  If provided, assert that
      x has at least this many dimensions.

  Returns:
    ndims:  A Python integer.

  Raises:
    ValueError:  If any of the expectations above are violated.
  """
  ndims = x.get_shape().ndims
  if ndims is None:
    shape_const = tensor_util.constant_value(array_ops.shape(x))
    if shape_const is not None:
      ndims = shape_const.ndim

  if ndims is None:
    if expect_static:
      raise ValueError(
          "Expected argument 'x' to have statically defined 'ndims'.  Found: " %
          x)
    return

  if expect_ndims is not None:
    ndims_message = ("Expected argument 'x' to have ndims %s.  Found tensor %s"
                     % (expect_ndims, x))
    if ndims != expect_ndims:
      raise ValueError(ndims_message)

  if expect_ndims_at_least is not None:
    ndims_at_least_message = (
        "Expected argument 'x' to have ndims >= %d.  Found tensor %s" % (
            expect_ndims_at_least, x))
    if ndims < expect_ndims_at_least:
      raise ValueError(ndims_at_least_message)

  if expect_ndims_no_more_than is not None:
    ndims_no_more_than_message = (
        "Expected argument 'x' to have ndims <= %d.  Found tensor %s" % (
            expect_ndims_no_more_than, x))
    if ndims > expect_ndims_no_more_than:
      raise ValueError(ndims_no_more_than_message)

  return ndims


def _get_best_effort_ndims(x,
                           expect_ndims=None,
                           expect_ndims_at_least=None,
                           expect_ndims_no_more_than=None):
  """Get static ndims if possible.  Fallback on `tf.rank(x)`."""
  ndims_static = _get_static_ndims(
      x,
      expect_ndims=expect_ndims,
      expect_ndims_at_least=expect_ndims_at_least,
      expect_ndims_no_more_than=expect_ndims_no_more_than)
  if ndims_static is not None:
    return ndims_static
  return array_ops.rank(x)


def _insert_back_keep_dims(x, axis):
  """Insert the dims in `axis` back as singletons after being removed.

  Args:
    x:  `Tensor`.
    axis:  Python list of integers.

  Returns:
    `Tensor` with same values as `x`, but additional singleton dimensions.
  """
  for i in sorted(axis):
    x = array_ops.expand_dims(x, axis=i)
  return x


def _make_static_axis_non_negative(axis, ndims):
  """Convert possibly negatively indexed axis to non-negative.

  Args:
    axis:  Iterable over Python integers.
    ndims:  Number of dimensions into which axis indexes.

  Returns:
    A list of non-negative Python integers.

  Raises:
    ValueError: If values in `axis` are too big/small to index into `ndims`.
  """
  non_negative_axis = []
  for d in axis:
    if d >= 0:
      if d >= ndims:
        raise ValueError("dim %d not in the interval [0, %d]." % (d, ndims - 1))
      non_negative_axis.append(d)
    else:
      if d < -1 * ndims:
        raise ValueError(
            "Negatively indexed dim %d not in the interval [-%d, -1]" % (d,
                                                                         ndims))
      non_negative_axis.append(ndims + d)
  return non_negative_axis


def _move_dims_to_flat_end(x, axis, x_ndims):
  """Move dims corresponding to `axis` in `x` to the end, then flatten.

  Args:
    x: `Tensor` with shape `[B0,B1,...,Bb]`.
    axis:  Python list of indices into dimensions of `x`.
    x_ndims:  Python integer holding number of dimensions in `x`.

  Returns:
    `Tensor` with value from `x` and dims in `axis` moved to end into one single
      dimension.
  """
  # Suppose x.shape = [a, b, c, d]
  # Suppose axis = [1, 3]

  # front_dims = [0, 2] in example above.
  front_dims = sorted(set(range(x_ndims)).difference(axis))
  # x_permed.shape = [a, c, b, d]
  x_permed = array_ops.transpose(x, perm=front_dims + list(axis))

  if x.get_shape().is_fully_defined():
    x_shape = x.get_shape().as_list()
    # front_shape = [a, c], end_shape = [b * d]
    front_shape = [x_shape[i] for i in front_dims]
    end_shape = [np.prod([x_shape[i] for i in axis])]
    full_shape = front_shape + end_shape
  else:
    front_shape = array_ops.shape(x_permed)[:x_ndims - len(axis)]
    end_shape = [-1]
    full_shape = array_ops.concat([front_shape, end_shape], axis=0)
  return array_ops.reshape(x_permed, shape=full_shape)


def _sort_tensor(tensor):
  """Use `top_k` to sort a `Tensor` along the last dimension."""
  sorted_, _ = nn_ops.top_k(tensor, k=array_ops.shape(tensor)[-1])
  return sorted_
