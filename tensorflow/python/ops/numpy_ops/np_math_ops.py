# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Mathematical operations."""
# pylint: disable=g-direct-tensorflow-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import six

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils


@np_utils.np_doc_only(np.dot)
def dot(a, b):  # pylint: disable=missing-docstring

  def f(a, b):  # pylint: disable=missing-docstring
    return np_utils.cond(
        np_utils.logical_or(
            math_ops.equal(array_ops.rank(a), 0),
            math_ops.equal(array_ops.rank(b), 0)),
        lambda: a * b,
        lambda: np_utils.cond(  # pylint: disable=g-long-lambda
            math_ops.equal(array_ops.rank(b), 1), lambda: math_ops.tensordot(
                a, b, axes=[[-1], [-1]]), lambda: math_ops.tensordot(
                    a, b, axes=[[-1], [-2]])))

  return _bin_op(f, a, b)


# TODO(wangpeng): Make element-wise ops `ufunc`s
def _bin_op(tf_fun, a, b, promote=True):
  if promote:
    a, b = np_array_ops._promote_dtype(a, b)  # pylint: disable=protected-access
  else:
    a = np_array_ops.array(a)
    b = np_array_ops.array(b)
  return np_utils.tensor_to_ndarray(tf_fun(a.data, b.data))


@np_utils.np_doc(np.add)
def add(x1, x2):

  def add_or_or(x1, x2):
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      return math_ops.logical_or(x1, x2)
    return math_ops.add(x1, x2)

  return _bin_op(add_or_or, x1, x2)


@np_utils.np_doc(np.subtract)
def subtract(x1, x2):
  return _bin_op(math_ops.subtract, x1, x2)


@np_utils.np_doc(np.multiply)
def multiply(x1, x2):

  def mul_or_and(x1, x2):
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      return math_ops.logical_and(x1, x2)
    return math_ops.multiply(x1, x2)

  return _bin_op(mul_or_and, x1, x2)


@np_utils.np_doc(np.true_divide)
def true_divide(x1, x2):  # pylint: disable=missing-function-docstring

  def _avoid_float64(x1, x2):
    if x1.dtype == x2.dtype and x1.dtype in (dtypes.int32, dtypes.int64):
      x1 = math_ops.cast(x1, dtype=dtypes.float32)
      x2 = math_ops.cast(x2, dtype=dtypes.float32)
    return x1, x2

  def f(x1, x2):
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      float_ = np_dtypes.default_float_type()
      x1 = math_ops.cast(x1, float_)
      x2 = math_ops.cast(x2, float_)
    if not np_dtypes.is_allow_float64():
      # math_ops.truediv in Python3 produces float64 when both inputs are int32
      # or int64. We want to avoid that when is_allow_float64() is False.
      x1, x2 = _avoid_float64(x1, x2)
    return math_ops.truediv(x1, x2)

  return _bin_op(f, x1, x2)


divide = true_divide


@np_utils.np_doc(np.floor_divide)
def floor_divide(x1, x2):  # pylint: disable=missing-function-docstring

  def f(x1, x2):
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      x1 = math_ops.cast(x1, dtypes.int8)
      x2 = math_ops.cast(x2, dtypes.int8)
    return math_ops.floordiv(x1, x2)

  return _bin_op(f, x1, x2)


@np_utils.np_doc(np.mod)
def mod(x1, x2):  # pylint: disable=missing-function-docstring

  def f(x1, x2):
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      x1 = math_ops.cast(x1, dtypes.int8)
      x2 = math_ops.cast(x2, dtypes.int8)
    return math_ops.mod(x1, x2)

  return _bin_op(f, x1, x2)


remainder = mod


@np_utils.np_doc(np.divmod)
def divmod(x1, x2):  # pylint: disable=redefined-builtin
  return floor_divide(x1, x2), mod(x1, x2)


@np_utils.np_doc(np.maximum)
def maximum(x1, x2):

  def max_or_or(x1, x2):
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      return math_ops.logical_or(x1, x2)
    return math_ops.maximum(x1, x2)

  return _bin_op(max_or_or, x1, x2)


@np_utils.np_doc(np.minimum)
def minimum(x1, x2):

  def min_or_and(x1, x2):
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      return math_ops.logical_and(x1, x2)
    return math_ops.minimum(x1, x2)

  return _bin_op(min_or_and, x1, x2)


@np_utils.np_doc(np.clip)
def clip(a, a_min, a_max):  # pylint: disable=missing-docstring
  if a_min is None and a_max is None:
    raise ValueError('Not more than one of `a_min` and `a_max` may be `None`.')
  if a_min is None:
    return minimum(a, a_max)
  elif a_max is None:
    return maximum(a, a_min)
  else:
    a, a_min, a_max = np_array_ops._promote_dtype(a, a_min, a_max)  # pylint: disable=protected-access
    return np_utils.tensor_to_ndarray(
        clip_ops.clip_by_value(
            *np_utils.tf_broadcast(a.data, a_min.data, a_max.data)))


@np_utils.np_doc(np.matmul)
def matmul(x1, x2):  # pylint: disable=missing-docstring

  def f(x1, x2):
    try:
      return np_utils.cond(
          math_ops.equal(array_ops.rank(x2), 1),
          lambda: math_ops.tensordot(x1, x2, axes=1),
          lambda: np_utils.cond(
              math_ops.equal(array_ops.rank(x1), 1),  # pylint: disable=g-long-lambda
              lambda: math_ops.tensordot(  # pylint: disable=g-long-lambda
                  x1, x2, axes=[[0], [-2]]),
              lambda: math_ops.matmul(x1, x2)))
    except errors.InvalidArgumentError as err:
      six.reraise(ValueError, ValueError(str(err)), sys.exc_info()[2])

  return _bin_op(f, x1, x2)


@np_utils.np_doc(np.tensordot)
def tensordot(a, b, axes=2):
  return _bin_op(lambda a, b: math_ops.tensordot(a, b, axes=axes), a, b)


@np_utils.np_doc_only(np.inner)
def inner(a, b):  # pylint: disable=missing-function-docstring

  def f(a, b):
    return np_utils.cond(
        np_utils.logical_or(
            math_ops.equal(array_ops.rank(a), 0),
            math_ops.equal(array_ops.rank(b), 0)), lambda: a * b,
        lambda: math_ops.tensordot(a, b, axes=[[-1], [-1]]))

  return _bin_op(f, a, b)


@np_utils.np_doc(np.cross)
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):  # pylint: disable=missing-docstring

  def f(a, b):  # pylint: disable=missing-docstring
    # We can't assign to captured variable `axisa`, so make a new variable
    axis_a = axisa
    axis_b = axisb
    axis_c = axisc
    if axis is not None:
      axis_a = axis
      axis_b = axis
      axis_c = axis
    if axis_a < 0:
      axis_a = np_utils.add(axis_a, array_ops.rank(a))
    if axis_b < 0:
      axis_b = np_utils.add(axis_b, array_ops.rank(b))

    def maybe_move_axis_to_last(a, axis):

      def move_axis_to_last(a, axis):
        return array_ops.transpose(
            a,
            array_ops.concat([
                math_ops.range(axis),
                math_ops.range(axis + 1, array_ops.rank(a)), [axis]
            ],
                             axis=0))

      return np_utils.cond(axis == np_utils.subtract(array_ops.rank(a), 1),
                           lambda: a, lambda: move_axis_to_last(a, axis))

    a = maybe_move_axis_to_last(a, axis_a)
    b = maybe_move_axis_to_last(b, axis_b)
    a_dim = np_utils.getitem(array_ops.shape(a), -1)
    b_dim = np_utils.getitem(array_ops.shape(b), -1)

    def maybe_pad_0(a, size_of_last_dim):

      def pad_0(a):
        return array_ops.pad(
            a,
            array_ops.concat([
                array_ops.zeros([array_ops.rank(a) - 1, 2], dtypes.int32),
                constant_op.constant([[0, 1]], dtypes.int32)
            ],
                             axis=0))

      return np_utils.cond(
          math_ops.equal(size_of_last_dim, 2), lambda: pad_0(a), lambda: a)

    a = maybe_pad_0(a, a_dim)
    b = maybe_pad_0(b, b_dim)
    c = math_ops.cross(*np_utils.tf_broadcast(a, b))
    if axis_c < 0:
      axis_c = np_utils.add(axis_c, array_ops.rank(c))

    def move_last_to_axis(a, axis):
      r = array_ops.rank(a)
      return array_ops.transpose(
          a,
          array_ops.concat(
              [math_ops.range(axis), [r - 1],
               math_ops.range(axis, r - 1)],
              axis=0))

    c = np_utils.cond(
        (a_dim == 2) & (b_dim == 2),
        lambda: c[..., 2],
        lambda: np_utils.cond(  # pylint: disable=g-long-lambda
            axis_c == np_utils.subtract(array_ops.rank(c), 1), lambda: c,
            lambda: move_last_to_axis(c, axis_c)))
    return c

  return _bin_op(f, a, b)


@np_utils.np_doc(np.power)
def power(x1, x2):
  return _bin_op(math_ops.pow, x1, x2)


@np_utils.np_doc(np.float_power)
def float_power(x1, x2):
  return power(x1, x2)


@np_utils.np_doc(np.arctan2)
def arctan2(x1, x2):
  return _bin_op(math_ops.atan2, x1, x2)


@np_utils.np_doc(np.nextafter)
def nextafter(x1, x2):
  return _bin_op(math_ops.nextafter, x1, x2)


@np_utils.np_doc(np.heaviside)
def heaviside(x1, x2):  # pylint: disable=missing-function-docstring

  def f(x1, x2):
    return array_ops.where_v2(
        x1 < 0, constant_op.constant(0, dtype=x2.dtype),
        array_ops.where_v2(x1 > 0, constant_op.constant(1, dtype=x2.dtype), x2))

  y = _bin_op(f, x1, x2)
  if not np.issubdtype(y.dtype, np.inexact):
    y = y.astype(np_dtypes.default_float_type())
  return y


@np_utils.np_doc(np.hypot)
def hypot(x1, x2):
  return sqrt(square(x1) + square(x2))


@np_utils.np_doc(np.kron)
def kron(a, b):  # pylint: disable=missing-function-docstring
  # pylint: disable=protected-access,g-complex-comprehension
  a, b = np_array_ops._promote_dtype(a, b)
  ndim = max(a.ndim, b.ndim)
  if a.ndim < ndim:
    a = np_array_ops.reshape(a, np_array_ops._pad_left_to(ndim, a.shape))
  if b.ndim < ndim:
    b = np_array_ops.reshape(b, np_array_ops._pad_left_to(ndim, b.shape))
  a_reshaped = np_array_ops.reshape(a, [i for d in a.shape for i in (d, 1)])
  b_reshaped = np_array_ops.reshape(b, [i for d in b.shape for i in (1, d)])
  out_shape = tuple(np.multiply(a.shape, b.shape))
  return np_array_ops.reshape(a_reshaped * b_reshaped, out_shape)


@np_utils.np_doc(np.outer)
def outer(a, b):

  def f(a, b):
    return array_ops.reshape(a, [-1, 1]) * array_ops.reshape(b, [-1])

  return _bin_op(f, a, b)


# This can also be implemented via tf.reduce_logsumexp
@np_utils.np_doc(np.logaddexp)
def logaddexp(x1, x2):
  amax = maximum(x1, x2)
  delta = x1 - x2
  return np_array_ops.where(
      isnan(delta),
      x1 + x2,  # NaNs or infinities of the same sign.
      amax + log1p(exp(-abs(delta))))


@np_utils.np_doc(np.logaddexp2)
def logaddexp2(x1, x2):
  amax = maximum(x1, x2)
  delta = x1 - x2
  return np_array_ops.where(
      isnan(delta),
      x1 + x2,  # NaNs or infinities of the same sign.
      amax + log1p(exp2(-abs(delta))) / np.log(2))


@np_utils.np_doc(np.polyval)
def polyval(p, x):  # pylint: disable=missing-function-docstring

  def f(p, x):
    if p.shape.rank == 0:
      p = array_ops.reshape(p, [1])
    p = array_ops.unstack(p)
    # TODO(wangpeng): Make tf version take a tensor for p instead of a list.
    y = math_ops.polyval(p, x)
    # If the polynomial is 0-order, numpy requires the result to be broadcast to
    # `x`'s shape.
    if len(p) == 1:
      y = array_ops.broadcast_to(y, x.shape)
    return y

  return _bin_op(f, p, x)


@np_utils.np_doc(np.isclose)
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):  # pylint: disable=missing-docstring

  def f(a, b):  # pylint: disable=missing-docstring
    dtype = a.dtype
    if np.issubdtype(dtype.as_numpy_dtype, np.inexact):
      rtol_ = ops.convert_to_tensor(rtol, dtype.real_dtype)
      atol_ = ops.convert_to_tensor(atol, dtype.real_dtype)
      result = (math_ops.abs(a - b) <= atol_ + rtol_ * math_ops.abs(b))
      if equal_nan:
        result = result | (math_ops.is_nan(a) & math_ops.is_nan(b))
      return result
    else:
      return a == b

  return _bin_op(f, a, b)


@np_utils.np_doc(np.allclose)
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
  return np_array_ops.all(
      isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))


def _tf_gcd(x1, x2):  # pylint: disable=missing-function-docstring

  def _gcd_cond_fn(_, x2):
    return math_ops.reduce_any(x2 != 0)

  def _gcd_body_fn(x1, x2):
    # math_ops.mod will raise an error when any element of x2 is 0. To avoid
    # that, we change those zeros to ones. Their values don't matter because
    # they won't be used.
    x2_safe = array_ops.where_v2(x2 != 0, x2, constant_op.constant(1, x2.dtype))
    x1, x2 = (array_ops.where_v2(x2 != 0, x2, x1),
              array_ops.where_v2(x2 != 0, math_ops.mod(x1, x2_safe),
                                 constant_op.constant(0, x2.dtype)))
    return (array_ops.where_v2(x1 < x2, x2,
                               x1), array_ops.where_v2(x1 < x2, x1, x2))

  if (not np.issubdtype(x1.dtype.as_numpy_dtype, np.integer) or
      not np.issubdtype(x2.dtype.as_numpy_dtype, np.integer)):
    raise ValueError('Arguments to gcd must be integers.')
  shape = array_ops.broadcast_static_shape(x1.shape, x2.shape)
  x1 = array_ops.broadcast_to(x1, shape)
  x2 = array_ops.broadcast_to(x2, shape)
  value, _ = control_flow_ops.while_loop(_gcd_cond_fn, _gcd_body_fn,
                                         (math_ops.abs(x1), math_ops.abs(x2)))
  return value


# Note that np.gcd may not be present in some supported versions of numpy.
@np_utils.np_doc(None, np_fun_name='gcd')
def gcd(x1, x2):
  return _bin_op(_tf_gcd, x1, x2)


# Note that np.lcm may not be present in some supported versions of numpy.
@np_utils.np_doc(None, np_fun_name='lcm')
def lcm(x1, x2):  # pylint: disable=missing-function-docstring

  def f(x1, x2):
    d = _tf_gcd(x1, x2)
    # Same as the `x2_safe` trick above
    d_safe = array_ops.where_v2(
        math_ops.equal(d, 0), constant_op.constant(1, d.dtype), d)
    return array_ops.where_v2(
        math_ops.equal(d, 0), constant_op.constant(0, d.dtype),
        math_ops.abs(x1 * x2) // d_safe)

  return _bin_op(f, x1, x2)


def _bitwise_binary_op(tf_fn, x1, x2):  # pylint: disable=missing-function-docstring

  def f(x1, x2):
    is_bool = (x1.dtype == dtypes.bool)
    if is_bool:
      assert x2.dtype == dtypes.bool
      x1 = math_ops.cast(x1, dtypes.int8)
      x2 = math_ops.cast(x2, dtypes.int8)
    r = tf_fn(x1, x2)
    if is_bool:
      r = math_ops.cast(r, dtypes.bool)
    return r

  return _bin_op(f, x1, x2)


@np_utils.np_doc(np.bitwise_and)
def bitwise_and(x1, x2):
  return _bitwise_binary_op(bitwise_ops.bitwise_and, x1, x2)


@np_utils.np_doc(np.bitwise_or)
def bitwise_or(x1, x2):
  return _bitwise_binary_op(bitwise_ops.bitwise_or, x1, x2)


@np_utils.np_doc(np.bitwise_xor)
def bitwise_xor(x1, x2):
  return _bitwise_binary_op(bitwise_ops.bitwise_xor, x1, x2)


@np_utils.np_doc(np.bitwise_not)
def bitwise_not(x):

  def f(x):
    if x.dtype == dtypes.bool:
      return math_ops.logical_not(x)
    return bitwise_ops.invert(x)

  return _scalar(f, x)


def _scalar(tf_fn, x, promote_to_float=False):
  """Computes the tf_fn(x) for each element in `x`.

  Args:
    tf_fn: function that takes a single Tensor argument.
    x: array_like. Could be an ndarray, a Tensor or any object that can be
      converted to a Tensor using `ops.convert_to_tensor`.
    promote_to_float: whether to cast the argument to a float dtype
      (`np_dtypes.default_float_type`) if it is not already.

  Returns:
    An ndarray with the same shape as `x`. The default output dtype is
    determined by `np_dtypes.default_float_type`, unless x is an ndarray with a
    floating point type, in which case the output type is same as x.dtype.
  """
  x = np_array_ops.asarray(x)
  if promote_to_float and not np.issubdtype(x.dtype, np.inexact):
    x = x.astype(np_dtypes.default_float_type())
  return np_utils.tensor_to_ndarray(tf_fn(x.data))


@np_utils.np_doc(np.log)
def log(x):
  return _scalar(math_ops.log, x, True)


@np_utils.np_doc(np.exp)
def exp(x):
  return _scalar(math_ops.exp, x, True)


@np_utils.np_doc(np.sqrt)
def sqrt(x):
  return _scalar(math_ops.sqrt, x, True)


@np_utils.np_doc(np.abs)
def abs(x):  # pylint: disable=redefined-builtin
  return _scalar(math_ops.abs, x)


@np_utils.np_doc(np.absolute)
def absolute(x):
  return abs(x)


@np_utils.np_doc(np.fabs)
def fabs(x):
  return abs(x)


@np_utils.np_doc(np.ceil)
def ceil(x):
  return _scalar(math_ops.ceil, x, True)


@np_utils.np_doc(np.floor)
def floor(x):
  return _scalar(math_ops.floor, x, True)


@np_utils.np_doc(np.conj)
def conj(x):
  return _scalar(math_ops.conj, x)


@np_utils.np_doc(np.negative)
def negative(x):
  return _scalar(math_ops.negative, x)


@np_utils.np_doc(np.reciprocal)
def reciprocal(x):
  return _scalar(math_ops.reciprocal, x)


@np_utils.np_doc(np.signbit)
def signbit(x):

  def f(x):
    if x.dtype == dtypes.bool:
      return array_ops.fill(x.shape, False)
    return x < 0

  return _scalar(f, x)


@np_utils.np_doc(np.sin)
def sin(x):
  return _scalar(math_ops.sin, x, True)


@np_utils.np_doc(np.cos)
def cos(x):
  return _scalar(math_ops.cos, x, True)


@np_utils.np_doc(np.tan)
def tan(x):
  return _scalar(math_ops.tan, x, True)


@np_utils.np_doc(np.sinh)
def sinh(x):
  return _scalar(math_ops.sinh, x, True)


@np_utils.np_doc(np.cosh)
def cosh(x):
  return _scalar(math_ops.cosh, x, True)


@np_utils.np_doc(np.tanh)
def tanh(x):
  return _scalar(math_ops.tanh, x, True)


@np_utils.np_doc(np.arcsin)
def arcsin(x):
  return _scalar(math_ops.asin, x, True)


@np_utils.np_doc(np.arccos)
def arccos(x):
  return _scalar(math_ops.acos, x, True)


@np_utils.np_doc(np.arctan)
def arctan(x):
  return _scalar(math_ops.atan, x, True)


@np_utils.np_doc(np.arcsinh)
def arcsinh(x):
  return _scalar(math_ops.asinh, x, True)


@np_utils.np_doc(np.arccosh)
def arccosh(x):
  return _scalar(math_ops.acosh, x, True)


@np_utils.np_doc(np.arctanh)
def arctanh(x):
  return _scalar(math_ops.atanh, x, True)


@np_utils.np_doc(np.deg2rad)
def deg2rad(x):

  def f(x):
    return x * (np.pi / 180.0)

  return _scalar(f, x, True)


@np_utils.np_doc(np.rad2deg)
def rad2deg(x):
  return x * (180.0 / np.pi)


_tf_float_types = [
    dtypes.bfloat16, dtypes.float16, dtypes.float32, dtypes.float64
]


@np_utils.np_doc(np.angle)
def angle(z, deg=False):  # pylint: disable=missing-function-docstring

  def f(x):
    if x.dtype in _tf_float_types:
      # Workaround for b/147515503
      return array_ops.where_v2(x < 0, np.pi, 0)
    else:
      return math_ops.angle(x)

  y = _scalar(f, z, True)
  if deg:
    y = rad2deg(y)
  return y


@np_utils.np_doc(np.cbrt)
def cbrt(x):

  def f(x):
    # __pow__ can't handle negative base, so we use `abs` here.
    rt = math_ops.abs(x)**(1.0 / 3)
    return array_ops.where_v2(x < 0, -rt, rt)

  return _scalar(f, x, True)


@np_utils.np_doc(np.conjugate)
def conjugate(x):
  return _scalar(math_ops.conj, x)


@np_utils.np_doc(np.exp2)
def exp2(x):

  def f(x):
    return 2**x

  return _scalar(f, x, True)


@np_utils.np_doc(np.expm1)
def expm1(x):
  return _scalar(math_ops.expm1, x, True)


@np_utils.np_doc(np.fix)
def fix(x):

  def f(x):
    return array_ops.where_v2(x < 0, math_ops.ceil(x), math_ops.floor(x))

  return _scalar(f, x, True)


@np_utils.np_doc(np.iscomplex)
def iscomplex(x):
  return np_array_ops.imag(x) != 0


@np_utils.np_doc(np.isreal)
def isreal(x):
  return np_array_ops.imag(x) == 0


@np_utils.np_doc(np.iscomplexobj)
def iscomplexobj(x):
  x = np_array_ops.array(x)
  return np.issubdtype(x.dtype, np.complexfloating)


@np_utils.np_doc(np.isrealobj)
def isrealobj(x):
  return not iscomplexobj(x)


@np_utils.np_doc(np.isnan)
def isnan(x):
  return _scalar(math_ops.is_nan, x, True)


def _make_nan_reduction(onp_reduction, reduction, init_val):
  """Helper to generate nan* functions."""

  @np_utils.np_doc(onp_reduction)
  def nan_reduction(a, axis=None, dtype=None, keepdims=False):
    a = np_array_ops.array(a)
    v = np_array_ops.array(init_val, dtype=a.dtype)
    return reduction(
        np_array_ops.where(isnan(a), v, a),
        axis=axis,
        dtype=dtype,
        keepdims=keepdims)

  return nan_reduction


nansum = _make_nan_reduction(np.nansum, np_array_ops.sum, 0)
nanprod = _make_nan_reduction(np.nanprod, np_array_ops.prod, 1)


@np_utils.np_doc(np.nanmean)
def nanmean(a, axis=None, dtype=None, keepdims=None):  # pylint: disable=missing-docstring
  a = np_array_ops.array(a)
  if np.issubdtype(a.dtype, np.bool_) or np.issubdtype(a.dtype, np.integer):
    return np_array_ops.mean(a, axis=axis, dtype=dtype, keepdims=keepdims)
  nan_mask = logical_not(isnan(a))
  if dtype is None:
    dtype = a.dtype
  normalizer = np_array_ops.sum(
      nan_mask, axis=axis, dtype=dtype, keepdims=keepdims)
  return nansum(a, axis=axis, dtype=dtype, keepdims=keepdims) / normalizer


@np_utils.np_doc(np.isfinite)
def isfinite(x):
  return _scalar(math_ops.is_finite, x, True)


@np_utils.np_doc(np.isinf)
def isinf(x):
  return _scalar(math_ops.is_inf, x, True)


@np_utils.np_doc(np.isneginf)
def isneginf(x):
  return x == np_array_ops.full_like(x, -np.inf)


@np_utils.np_doc(np.isposinf)
def isposinf(x):
  return x == np_array_ops.full_like(x, np.inf)


@np_utils.np_doc(np.log2)
def log2(x):
  return log(x) / np.log(2)


@np_utils.np_doc(np.log10)
def log10(x):
  return log(x) / np.log(10)


@np_utils.np_doc(np.log1p)
def log1p(x):
  return _scalar(math_ops.log1p, x, True)


@np_utils.np_doc(np.positive)
def positive(x):
  return _scalar(lambda x: x, x)


@np_utils.np_doc(np.sinc)
def sinc(x):

  def f(x):
    pi_x = x * np.pi
    return array_ops.where_v2(x == 0, array_ops.ones_like(x),
                              math_ops.sin(pi_x) / pi_x)

  return _scalar(f, x, True)


@np_utils.np_doc(np.square)
def square(x):
  return _scalar(math_ops.square, x)


@np_utils.np_doc(np.diff)
def diff(a, n=1, axis=-1):  # pylint: disable=missing-function-docstring

  def f(a):
    nd = a.shape.rank
    if (axis + nd if axis < 0 else axis) >= nd:
      raise ValueError('axis %s is out of bounds for array of dimension %s' %
                       (axis, nd))
    if n < 0:
      raise ValueError('order must be non-negative but got %s' % n)
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    op = math_ops.not_equal if a.dtype == dtypes.bool else math_ops.subtract
    for _ in range(n):
      a = op(a[slice1], a[slice2])
    return a

  return _scalar(f, a)


def _flip_args(f):
  def _f(a, b):
    return f(b, a)
  return _f


setattr(np_arrays.ndarray, '__abs__', absolute)
setattr(np_arrays.ndarray, '__floordiv__', floor_divide)
setattr(np_arrays.ndarray, '__rfloordiv__', _flip_args(floor_divide))
setattr(np_arrays.ndarray, '__mod__', mod)
setattr(np_arrays.ndarray, '__rmod__', _flip_args(mod))
setattr(np_arrays.ndarray, '__add__', add)
setattr(np_arrays.ndarray, '__radd__', _flip_args(add))
setattr(np_arrays.ndarray, '__sub__', subtract)
setattr(np_arrays.ndarray, '__rsub__', _flip_args(subtract))
setattr(np_arrays.ndarray, '__mul__', multiply)
setattr(np_arrays.ndarray, '__rmul__', _flip_args(multiply))
setattr(np_arrays.ndarray, '__pow__', power)
setattr(np_arrays.ndarray, '__rpow__', _flip_args(power))
setattr(np_arrays.ndarray, '__truediv__', true_divide)
setattr(np_arrays.ndarray, '__rtruediv__', _flip_args(true_divide))


def _comparison(tf_fun, x1, x2, cast_bool_to_int=False):
  dtype = np_utils.result_type(x1, x2)
  # Cast x1 and x2 to the result_type if needed.
  x1 = np_array_ops.array(x1, dtype=dtype)
  x2 = np_array_ops.array(x2, dtype=dtype)
  x1 = x1.data
  x2 = x2.data
  if cast_bool_to_int and x1.dtype == dtypes.bool:
    x1 = math_ops.cast(x1, dtypes.int32)
    x2 = math_ops.cast(x2, dtypes.int32)
  return np_utils.tensor_to_ndarray(tf_fun(x1, x2))


@np_utils.np_doc(np.equal)
def equal(x1, x2):
  return _comparison(math_ops.equal, x1, x2)


@np_utils.np_doc(np.not_equal)
def not_equal(x1, x2):
  return _comparison(math_ops.not_equal, x1, x2)


@np_utils.np_doc(np.greater)
def greater(x1, x2):
  return _comparison(math_ops.greater, x1, x2, True)


@np_utils.np_doc(np.greater_equal)
def greater_equal(x1, x2):
  return _comparison(math_ops.greater_equal, x1, x2, True)


@np_utils.np_doc(np.less)
def less(x1, x2):
  return _comparison(math_ops.less, x1, x2, True)


@np_utils.np_doc(np.less_equal)
def less_equal(x1, x2):
  return _comparison(math_ops.less_equal, x1, x2, True)


@np_utils.np_doc(np.array_equal)
def array_equal(a1, a2):

  def f(a1, a2):
    if a1.shape != a2.shape:
      return constant_op.constant(False)
    return math_ops.reduce_all(math_ops.equal(a1, a2))

  return _comparison(f, a1, a2)


def _logical_binary_op(tf_fun, x1, x2):
  x1 = np_array_ops.array(x1, dtype=np.bool_)
  x2 = np_array_ops.array(x2, dtype=np.bool_)
  return np_utils.tensor_to_ndarray(tf_fun(x1.data, x2.data))


@np_utils.np_doc(np.logical_and)
def logical_and(x1, x2):
  return _logical_binary_op(math_ops.logical_and, x1, x2)


@np_utils.np_doc(np.logical_or)
def logical_or(x1, x2):
  return _logical_binary_op(math_ops.logical_or, x1, x2)


@np_utils.np_doc(np.logical_xor)
def logical_xor(x1, x2):
  return _logical_binary_op(math_ops.logical_xor, x1, x2)


@np_utils.np_doc(np.logical_not)
def logical_not(x):
  x = np_array_ops.array(x, dtype=np.bool_)
  return np_utils.tensor_to_ndarray(math_ops.logical_not(x.data))


setattr(np_arrays.ndarray, '__invert__', logical_not)
setattr(np_arrays.ndarray, '__lt__', less)
setattr(np_arrays.ndarray, '__le__', less_equal)
setattr(np_arrays.ndarray, '__gt__', greater)
setattr(np_arrays.ndarray, '__ge__', greater_equal)
setattr(np_arrays.ndarray, '__eq__', equal)
setattr(np_arrays.ndarray, '__ne__', not_equal)


@np_utils.np_doc(np.linspace)
def linspace(  # pylint: disable=missing-docstring
    start, stop, num=50, endpoint=True, retstep=False, dtype=float, axis=0):
  if dtype:
    dtype = np_utils.result_type(dtype)
  start = np_array_ops.array(start, dtype=dtype).data
  stop = np_array_ops.array(stop, dtype=dtype).data
  if num < 0:
    raise ValueError('Number of samples {} must be non-negative.'.format(num))
  step = ops.convert_to_tensor(np.nan)
  if endpoint:
    result = math_ops.linspace(start, stop, num, axis=axis)
    if num > 1:
      step = (stop - start) / (num - 1)
  else:
    # math_ops.linspace does not support endpoint=False so we manually handle it
    # here.
    if num > 1:
      step = ((stop - start) / num)
      new_stop = math_ops.cast(stop, step.dtype) - step
      start = math_ops.cast(start, new_stop.dtype)
      result = math_ops.linspace(start, new_stop, num, axis=axis)
    else:
      result = math_ops.linspace(start, stop, num, axis=axis)
  if dtype:
    result = math_ops.cast(result, dtype)
  if retstep:
    return (np_arrays.tensor_to_ndarray(result),
            np_arrays.tensor_to_ndarray(step))
  else:
    return np_arrays.tensor_to_ndarray(result)


@np_utils.np_doc(np.logspace)
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
  dtype = np_utils.result_type(start, stop, dtype)
  result = linspace(
      start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis).data
  result = math_ops.pow(math_ops.cast(base, result.dtype), result)
  if dtype:
    result = math_ops.cast(result, dtype)
  return np_arrays.tensor_to_ndarray(result)


@np_utils.np_doc(np.ptp)
def ptp(a, axis=None, keepdims=None):
  return (np_array_ops.amax(a, axis=axis, keepdims=keepdims) -
          np_array_ops.amin(a, axis=axis, keepdims=keepdims))


@np_utils.np_doc_only(np.concatenate)
def concatenate(arys, axis=0):
  if not isinstance(arys, (list, tuple)):
    arys = [arys]
  if not arys:
    raise ValueError('Need at least one array to concatenate.')
  dtype = np_utils.result_type(*arys)
  arys = [np_array_ops.array(array, dtype=dtype).data for array in arys]
  return np_arrays.tensor_to_ndarray(array_ops.concat(arys, axis))


@np_utils.np_doc_only(np.tile)
def tile(a, reps):  # pylint: disable=missing-function-docstring
  a = np_array_ops.array(a).data
  reps = np_array_ops.array(reps, dtype=dtypes.int32).reshape([-1]).data

  a_rank = array_ops.rank(a)
  reps_size = array_ops.size(reps)
  reps = array_ops.pad(
      reps, [[math_ops.maximum(a_rank - reps_size, 0), 0]], constant_values=1)
  a_shape = array_ops.pad(
      array_ops.shape(a), [[math_ops.maximum(reps_size - a_rank, 0), 0]],
      constant_values=1)
  a = array_ops.reshape(a, a_shape)

  return np_arrays.tensor_to_ndarray(array_ops.tile(a, reps))


@np_utils.np_doc(np.count_nonzero)
def count_nonzero(a, axis=None):
  return np_arrays.tensor_to_ndarray(
      math_ops.count_nonzero(np_array_ops.array(a).data, axis))


@np_utils.np_doc(np.argsort)
def argsort(a, axis=-1, kind='quicksort', order=None):  # pylint: disable=missing-docstring
  # TODO(nareshmodi): make string tensors also work.
  if kind not in ('quicksort', 'stable'):
    raise ValueError("Only 'quicksort' and 'stable' arguments are supported.")
  if order is not None:
    raise ValueError("'order' argument to sort is not supported.")
  stable = (kind == 'stable')

  a = np_array_ops.array(a).data

  def _argsort(a, axis, stable):
    if axis is None:
      a = array_ops.reshape(a, [-1])
      axis = 0

    return sort_ops.argsort(a, axis, stable=stable)

  tf_ans = control_flow_ops.cond(
      math_ops.equal(array_ops.rank(a), 0), lambda: constant_op.constant([0]),
      lambda: _argsort(a, axis, stable))

  return np_array_ops.array(tf_ans, dtype=np.intp)


@np_utils.np_doc(np.sort)
def sort(a, axis=-1, kind='quicksort', order=None):  # pylint: disable=missing-docstring
  if kind != 'quicksort':
    raise ValueError("Only 'quicksort' is supported.")
  if order is not None:
    raise ValueError("'order' argument to sort is not supported.")

  a = np_array_ops.array(a)

  if axis is None:
    result_t = sort_ops.sort(array_ops.reshape(a.data, [-1]), 0)
    return np_utils.tensor_to_ndarray(result_t)
  else:
    return np_utils.tensor_to_ndarray(sort_ops.sort(a.data, axis))


def _argminmax(fn, a, axis=None):
  a = np_array_ops.array(a)
  if axis is None:
    # When axis is None numpy flattens the array.
    a_t = array_ops.reshape(a.data, [-1])
  else:
    a_t = np_array_ops.atleast_1d(a).data
  return np_utils.tensor_to_ndarray(fn(input=a_t, axis=axis))


@np_utils.np_doc(np.argmax)
def argmax(a, axis=None):
  return _argminmax(math_ops.argmax, a, axis)


@np_utils.np_doc(np.argmin)
def argmin(a, axis=None):
  return _argminmax(math_ops.argmin, a, axis)


@np_utils.np_doc(np.append)
def append(arr, values, axis=None):
  if axis is None:
    return concatenate([np_array_ops.ravel(arr), np_array_ops.ravel(values)], 0)
  else:
    return concatenate([arr, values], axis=axis)


@np_utils.np_doc(np.average)
def average(a, axis=None, weights=None, returned=False):  # pylint: disable=missing-docstring
  if axis is not None and not isinstance(axis, six.integer_types):
    # TODO(wangpeng): Support tuple of ints as `axis`
    raise ValueError('`axis` must be an integer. Tuple of ints is not '
                     'supported yet. Got type: %s' % type(axis))
  a = np_array_ops.array(a)
  if weights is None:  # Treat all weights as 1
    if not np.issubdtype(a.dtype, np.inexact):
      a = a.astype(
          np_utils.result_type(a.dtype, np_dtypes.default_float_type()))
    avg = math_ops.reduce_mean(a.data, axis=axis)
    if returned:
      if axis is None:
        weights_sum = array_ops.size(a.data)
      else:
        weights_sum = array_ops.shape(a.data)[axis]
      weights_sum = math_ops.cast(weights_sum, a.data.dtype)
  else:
    if np.issubdtype(a.dtype, np.inexact):
      out_dtype = np_utils.result_type(a.dtype, weights)
    else:
      out_dtype = np_utils.result_type(a.dtype, weights,
                                       np_dtypes.default_float_type())
    a = np_array_ops.array(a, out_dtype).data
    weights = np_array_ops.array(weights, out_dtype).data

    def rank_equal_case():
      control_flow_ops.Assert(
          math_ops.reduce_all(array_ops.shape(a) == array_ops.shape(weights)),
          [array_ops.shape(a), array_ops.shape(weights)])
      weights_sum = math_ops.reduce_sum(weights, axis=axis)
      avg = math_ops.reduce_sum(a * weights, axis=axis) / weights_sum
      return avg, weights_sum

    if axis is None:
      avg, weights_sum = rank_equal_case()
    else:

      def rank_not_equal_case():
        control_flow_ops.Assert(
            array_ops.rank(weights) == 1, [array_ops.rank(weights)])
        weights_sum = math_ops.reduce_sum(weights)
        axes = ops.convert_to_tensor([[axis], [0]])
        avg = math_ops.tensordot(a, weights, axes) / weights_sum
        return avg, weights_sum

      # We condition on rank rather than shape equality, because if we do the
      # latter, when the shapes are partially unknown but the ranks are known
      # and different, np_utils.cond will run shape checking on the true branch,
      # which will raise a shape-checking error.
      avg, weights_sum = np_utils.cond(
          math_ops.equal(array_ops.rank(a), array_ops.rank(weights)),
          rank_equal_case, rank_not_equal_case)

  avg = np_array_ops.array(avg)
  if returned:
    weights_sum = np_array_ops.broadcast_to(weights_sum,
                                            array_ops.shape(avg.data))
    return avg, weights_sum
  return avg


@np_utils.np_doc(np.trace)
def trace(a, offset=0, axis1=0, axis2=1, dtype=None):  # pylint: disable=missing-docstring
  if dtype:
    dtype = np_utils.result_type(dtype)
  a = np_array_ops.asarray(a, dtype).data

  if offset == 0:
    a_shape = a.shape
    if a_shape.rank is not None:
      rank = len(a_shape)
      if (axis1 == -2 or axis1 == rank - 2) and (axis2 == -1 or
                                                 axis2 == rank - 1):
        return np_utils.tensor_to_ndarray(math_ops.trace(a))

  a = np_array_ops.diagonal(a, offset, axis1, axis2)
  return np_array_ops.sum(a, -1, dtype)


@np_utils.np_doc(np.meshgrid)
def meshgrid(*xi, **kwargs):
  """This currently requires copy=True and sparse=False."""
  sparse = kwargs.get('sparse', False)
  if sparse:
    raise ValueError('meshgrid doesnt support returning sparse arrays yet')

  copy = kwargs.get('copy', True)
  if not copy:
    raise ValueError('meshgrid only supports copy=True')

  indexing = kwargs.get('indexing', 'xy')

  xi = [np_array_ops.asarray(arg).data for arg in xi]
  kwargs = {'indexing': indexing}

  outputs = array_ops.meshgrid(*xi, **kwargs)
  outputs = [np_utils.tensor_to_ndarray(output) for output in outputs]

  return outputs
