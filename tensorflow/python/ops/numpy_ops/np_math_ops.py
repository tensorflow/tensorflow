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

import numbers
import sys

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export


pi = np.pi
tf_export.tf_export('experimental.numpy.pi', v1=[]).export_constant(
    __name__, 'pi'
)
e = np.e
tf_export.tf_export('experimental.numpy.e', v1=[]).export_constant(
    __name__, 'e'
)
inf = np.inf
tf_export.tf_export('experimental.numpy.inf', v1=[]).export_constant(
    __name__, 'inf'
)


@tf_export.tf_export('experimental.numpy.dot', v1=[])
@np_utils.np_doc_only('dot')
def dot(a, b):  # pylint: disable=missing-docstring
  def f(a, b):  # pylint: disable=missing-docstring
    return np_utils.cond(
        np_utils.logical_or(
            math_ops.equal(array_ops.rank(a), 0),
            math_ops.equal(array_ops.rank(b), 0),
        ),
        lambda: a * b,
        lambda: np_utils.cond(  # pylint: disable=g-long-lambda
            math_ops.equal(array_ops.rank(b), 1),
            lambda: math_ops.tensordot(a, b, axes=[[-1], [-1]]),
            lambda: math_ops.tensordot(a, b, axes=[[-1], [-2]]),
        ),
    )

  return _bin_op(f, a, b)


# TODO(wangpeng): Make element-wise ops `ufunc`s
def _bin_op(tf_fun, a, b, promote=True):
  if promote:
    a, b = np_array_ops._promote_dtype_binary(a, b)  # pylint: disable=protected-access
  else:
    a = np_array_ops.array(a)
    b = np_array_ops.array(b)
  return tf_fun(a, b)


@tf_export.tf_export('experimental.numpy.add', v1=[])
@np_utils.np_doc('add')
def add(x1, x2):
  def add_or_or(x1, x2):
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      return math_ops.logical_or(x1, x2)
    return math_ops.add(x1, x2)

  return _bin_op(add_or_or, x1, x2)


@tf_export.tf_export('experimental.numpy.subtract', v1=[])
@np_utils.np_doc('subtract')
def subtract(x1, x2):
  return _bin_op(math_ops.subtract, x1, x2)


@tf_export.tf_export('experimental.numpy.multiply', v1=[])
@np_utils.np_doc('multiply')
def multiply(x1, x2):
  def mul_or_and(x1, x2):
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      return math_ops.logical_and(x1, x2)
    return math_ops.multiply(x1, x2)

  return _bin_op(mul_or_and, x1, x2)


@tf_export.tf_export('experimental.numpy.true_divide', v1=[])
@np_utils.np_doc('true_divide')
def true_divide(x1, x2):  # pylint: disable=missing-function-docstring
  def _avoid_float64(x1, x2):
    if x1.dtype == x2.dtype and x1.dtype in (dtypes.int32, dtypes.int64):
      x1 = math_ops.cast(x1, dtype=dtypes.float32)
      x2 = math_ops.cast(x2, dtype=dtypes.float32)
    return x1, x2

  def f(x1, x2):
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      float_ = np_utils.result_type(float)
      x1 = math_ops.cast(x1, float_)
      x2 = math_ops.cast(x2, float_)
    if not np_dtypes.is_allow_float64():
      # math_ops.truediv in Python3 produces float64 when both inputs are int32
      # or int64. We want to avoid that when is_allow_float64() is False.
      x1, x2 = _avoid_float64(x1, x2)
    return math_ops.truediv(x1, x2)

  return _bin_op(f, x1, x2)


@tf_export.tf_export('experimental.numpy.divide', v1=[])
@np_utils.np_doc('divide')
def divide(x1, x2):  # pylint: disable=missing-function-docstring
  return true_divide(x1, x2)


@tf_export.tf_export('experimental.numpy.floor_divide', v1=[])
@np_utils.np_doc('floor_divide')
def floor_divide(x1, x2):  # pylint: disable=missing-function-docstring
  def f(x1, x2):
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      x1 = math_ops.cast(x1, dtypes.int8)
      x2 = math_ops.cast(x2, dtypes.int8)
    return math_ops.floordiv(x1, x2)

  return _bin_op(f, x1, x2)


@tf_export.tf_export('experimental.numpy.mod', v1=[])
@np_utils.np_doc('mod')
def mod(x1, x2):  # pylint: disable=missing-function-docstring
  def f(x1, x2):
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      x1 = math_ops.cast(x1, dtypes.int8)
      x2 = math_ops.cast(x2, dtypes.int8)
    return math_ops.mod(x1, x2)

  return _bin_op(f, x1, x2)


@tf_export.tf_export('experimental.numpy.remainder', v1=[])
@np_utils.np_doc('remainder')
def remainder(x1, x2):  # pylint: disable=missing-function-docstring
  return mod(x1, x2)


@tf_export.tf_export('experimental.numpy.divmod', v1=[])
@np_utils.np_doc('divmod')
def divmod(x1, x2):  # pylint: disable=redefined-builtin
  return floor_divide(x1, x2), mod(x1, x2)


@tf_export.tf_export('experimental.numpy.maximum', v1=[])
@np_utils.np_doc('maximum')
def maximum(x1, x2):  # pylint: disable=missing-function-docstring
  # Fast path for when maximum is used as relu.
  if (
      isinstance(x2, numbers.Real)
      and not isinstance(x2, bool)
      and x2 == 0
      and isinstance(x1, np_arrays.ndarray)
      and x1.dtype != dtypes.bool
  ):
    return nn_ops.relu(np_array_ops.asarray(x1))

  def max_or_or(x1, x2):
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      return math_ops.logical_or(x1, x2)
    return math_ops.maximum(x1, x2)

  return _bin_op(max_or_or, x1, x2)


@tf_export.tf_export('experimental.numpy.minimum', v1=[])
@np_utils.np_doc('minimum')
def minimum(x1, x2):
  def min_or_and(x1, x2):
    if x1.dtype == dtypes.bool:
      assert x2.dtype == dtypes.bool
      return math_ops.logical_and(x1, x2)
    return math_ops.minimum(x1, x2)

  return _bin_op(min_or_and, x1, x2)


@tf_export.tf_export('experimental.numpy.clip', v1=[])
@np_utils.np_doc('clip')
def clip(a, a_min, a_max):  # pylint: disable=missing-docstring
  if a_min is None and a_max is None:
    raise ValueError('Not more than one of `a_min` and `a_max` may be `None`.')
  if a_min is None:
    return minimum(a, a_max)
  elif a_max is None:
    return maximum(a, a_min)
  else:
    a, a_min, a_max = np_array_ops._promote_dtype(a, a_min, a_max)  # pylint: disable=protected-access
    return clip_ops.clip_by_value(*np_utils.tf_broadcast(a, a_min, a_max))


@tf_export.tf_export('experimental.numpy.matmul', v1=[])
@np_utils.np_doc('matmul')
def matmul(x1, x2):  # pylint: disable=missing-docstring
  def f(x1, x2):
    try:
      if x1._rank() == 2 and x2._rank() == 2:  # pylint: disable=protected-access
        # Fast path for known ranks.
        return gen_math_ops.mat_mul(x1, x2)
      return np_utils.cond(
          math_ops.equal(np_utils.tf_rank(x2), 1),
          lambda: math_ops.tensordot(x1, x2, axes=1),
          lambda: np_utils.cond(  # pylint: disable=g-long-lambda
              math_ops.equal(np_utils.tf_rank(x1), 1),
              lambda: math_ops.tensordot(  # pylint: disable=g-long-lambda
                  x1, x2, axes=[[0], [-2]]
              ),
              lambda: math_ops.matmul(x1, x2),
          ),
      )
    except errors.InvalidArgumentError as err:
      raise ValueError(str(err)).with_traceback(sys.exc_info()[2])

  return _bin_op(f, x1, x2)


# Exported so it can be called from Tensor.__matmul__. NumPy's matmul handles
# batched matmul as well, so simply including promotion in TF's current
# __matmul__ implementation was not sufficient.
setattr(np_arrays.ndarray, '_matmul', matmul)


@tf_export.tf_export('experimental.numpy.tensordot', v1=[])
@np_utils.np_doc('tensordot')
def tensordot(a, b, axes=2):
  return _bin_op(lambda a, b: math_ops.tensordot(a, b, axes=axes), a, b)


@tf_export.tf_export('experimental.numpy.inner', v1=[])
@np_utils.np_doc_only('inner')
def inner(a, b):  # pylint: disable=missing-function-docstring
  def f(a, b):
    return np_utils.cond(
        np_utils.logical_or(
            math_ops.equal(array_ops.rank(a), 0),
            math_ops.equal(array_ops.rank(b), 0),
        ),
        lambda: a * b,
        lambda: math_ops.tensordot(a, b, axes=[[-1], [-1]]),
    )

  return _bin_op(f, a, b)


@tf_export.tf_export('experimental.numpy.cross', v1=[])
@np_utils.np_doc('cross')
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):  # pylint: disable=missing-docstring
  def f(a, b):  # pylint: disable=missing-docstring
    # We can't assign to captured variable `axisa`, so make a new variable
    if axis is None:
      axis_a = axisa
      axis_b = axisb
      axis_c = axisc
    else:
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
            array_ops.concat(
                [
                    math_ops.range(axis),
                    math_ops.range(axis + 1, array_ops.rank(a)),
                    [axis],
                ],
                axis=0,
            ),
        )

      return np_utils.cond(
          axis == np_utils.subtract(array_ops.rank(a), 1),
          lambda: a,
          lambda: move_axis_to_last(a, axis),
      )

    a = maybe_move_axis_to_last(a, axis_a)
    b = maybe_move_axis_to_last(b, axis_b)
    a_dim = np_utils.getitem(array_ops.shape(a), -1)
    b_dim = np_utils.getitem(array_ops.shape(b), -1)

    def maybe_pad_0(a, size_of_last_dim):
      def pad_0(a):
        return array_ops.pad(
            a,
            array_ops.concat(
                [
                    array_ops.zeros([array_ops.rank(a) - 1, 2], dtypes.int32),
                    constant_op.constant([[0, 1]], dtypes.int32),
                ],
                axis=0,
            ),
        )

      return np_utils.cond(
          math_ops.equal(size_of_last_dim, 2), lambda: pad_0(a), lambda: a
      )

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
              [math_ops.range(axis), [r - 1], math_ops.range(axis, r - 1)],
              axis=0,
          ),
      )

    c = np_utils.cond(
        (a_dim == 2) & (b_dim == 2),
        lambda: c[..., 2],
        lambda: np_utils.cond(  # pylint: disable=g-long-lambda
            axis_c == np_utils.subtract(array_ops.rank(c), 1),
            lambda: c,
            lambda: move_last_to_axis(c, axis_c),
        ),
    )
    return c

  return _bin_op(f, a, b)


@tf_export.tf_export('experimental.numpy.vdot', v1=[])
@np_utils.np_doc_only('vdot')
def vdot(a, b):  # pylint: disable=missing-docstring
  a, b = np_array_ops._promote_dtype(a, b)  # pylint: disable=protected-access
  a = np_array_ops.reshape(a, [-1])
  b = np_array_ops.reshape(b, [-1])
  if a.dtype == np_dtypes.complex128 or a.dtype == np_dtypes.complex64:
    a = conj(a)
  return dot(a, b)


@tf_export.tf_export('experimental.numpy.power', v1=[])
@np_utils.np_doc('power')
def power(x1, x2):
  return _bin_op(math_ops.pow, x1, x2)


@tf_export.tf_export('experimental.numpy.float_power', v1=[])
@np_utils.np_doc('float_power')
def float_power(x1, x2):
  return power(x1, x2)


@tf_export.tf_export('experimental.numpy.arctan2', v1=[])
@np_utils.np_doc('arctan2')
def arctan2(x1, x2):
  return _bin_op(math_ops.atan2, x1, x2)


@tf_export.tf_export('experimental.numpy.nextafter', v1=[])
@np_utils.np_doc('nextafter')
def nextafter(x1, x2):
  return _bin_op(math_ops.nextafter, x1, x2)


@tf_export.tf_export('experimental.numpy.heaviside', v1=[])
@np_utils.np_doc('heaviside')
def heaviside(x1, x2):  # pylint: disable=missing-function-docstring
  def f(x1, x2):
    return array_ops.where_v2(
        x1 < 0,
        constant_op.constant(0, dtype=x2.dtype),
        array_ops.where_v2(x1 > 0, constant_op.constant(1, dtype=x2.dtype), x2),
    )

  y = _bin_op(f, x1, x2)
  if not np.issubdtype(y.dtype.as_numpy_dtype, np.inexact):
    y = y.astype(np_utils.result_type(float))
  return y


@tf_export.tf_export('experimental.numpy.hypot', v1=[])
@np_utils.np_doc('hypot')
def hypot(x1, x2):
  return sqrt(square(x1) + square(x2))


@tf_export.tf_export('experimental.numpy.kron', v1=[])
@np_utils.np_doc('kron')
def kron(a, b):  # pylint: disable=missing-function-docstring
  # pylint: disable=protected-access,g-complex-comprehension
  a, b = np_array_ops._promote_dtype(a, b)
  t_a = np_utils.cond(
      a.shape.rank < b.shape.rank,
      lambda: np_array_ops.reshape(  # pylint: disable=g-long-lambda
          a, np_array_ops._pad_left_to(b.shape.rank, a.shape)
      ),
      lambda: a,
  )
  t_b = np_utils.cond(
      b.shape.rank < a.shape.rank,
      lambda: np_array_ops.reshape(  # pylint: disable=g-long-lambda
          b, np_array_ops._pad_left_to(a.shape.rank, b.shape)
      ),
      lambda: b,
  )

  def _make_shape(shape, prepend):
    ones = array_ops.ones_like(shape)
    if prepend:
      shapes = [ones, shape]
    else:
      shapes = [shape, ones]
    return array_ops.reshape(array_ops_stack.stack(shapes, axis=1), [-1])

  a_shape = array_ops.shape(t_a)
  b_shape = array_ops.shape(t_b)
  a_reshaped = np_array_ops.reshape(t_a, _make_shape(a_shape, False))
  b_reshaped = np_array_ops.reshape(t_b, _make_shape(b_shape, True))
  out_shape = a_shape * b_shape
  return np_array_ops.reshape(a_reshaped * b_reshaped, out_shape)


@tf_export.tf_export('experimental.numpy.outer', v1=[])
@np_utils.np_doc('outer')
def outer(a, b):
  def f(a, b):
    return array_ops.reshape(a, [-1, 1]) * array_ops.reshape(b, [-1])

  return _bin_op(f, a, b)


# This can also be implemented via tf.reduce_logsumexp
@tf_export.tf_export('experimental.numpy.logaddexp', v1=[])
@np_utils.np_doc('logaddexp')
def logaddexp(x1, x2):
  amax = maximum(x1, x2)
  delta = x1 - x2
  return np_array_ops.where(
      isnan(delta),
      x1 + x2,  # NaNs or infinities of the same sign.
      amax + log1p(exp(-abs(delta))),
  )


@tf_export.tf_export('experimental.numpy.logaddexp2', v1=[])
@np_utils.np_doc('logaddexp2')
def logaddexp2(x1, x2):
  amax = maximum(x1, x2)
  delta = x1 - x2
  return np_array_ops.where(
      isnan(delta),
      x1 + x2,  # NaNs or infinities of the same sign.
      amax + log1p(exp2(-abs(delta))) / np.log(2),
  )


@tf_export.tf_export('experimental.numpy.polyval', v1=[])
@np_utils.np_doc('polyval')
def polyval(p, x):  # pylint: disable=missing-function-docstring
  def f(p, x):
    if p.shape.rank == 0:
      p = array_ops.reshape(p, [1])
    p = array_ops_stack.unstack(p)
    # TODO(wangpeng): Make tf version take a tensor for p instead of a list.
    y = math_ops.polyval(p, x)
    # If the polynomial is 0-order, numpy requires the result to be broadcast to
    # `x`'s shape.
    if len(p) == 1:
      y = array_ops.broadcast_to(y, x.shape)
    return y

  return _bin_op(f, p, x)


@tf_export.tf_export('experimental.numpy.isclose', v1=[])
@np_utils.np_doc('isclose')
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):  # pylint: disable=missing-docstring
  def f(a, b):  # pylint: disable=missing-docstring
    dtype = a.dtype
    if np.issubdtype(dtype.as_numpy_dtype, np.inexact):
      rtol_ = ops.convert_to_tensor(rtol, dtype.real_dtype)
      atol_ = ops.convert_to_tensor(atol, dtype.real_dtype)
      result = math_ops.abs(a - b) <= atol_ + rtol_ * math_ops.abs(b)
      if equal_nan:
        result = result | (math_ops.is_nan(a) & math_ops.is_nan(b))
      return result
    else:
      return a == b

  return _bin_op(f, a, b)


@tf_export.tf_export('experimental.numpy.allclose', v1=[])
@np_utils.np_doc('allclose')
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
  return np_array_ops.all(
      isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
  )


def _tf_gcd(x1, x2):  # pylint: disable=missing-function-docstring
  def _gcd_cond_fn(_, x2):
    return math_ops.reduce_any(x2 != 0)

  def _gcd_body_fn(x1, x2):
    # math_ops.mod will raise an error when any element of x2 is 0. To avoid
    # that, we change those zeros to ones. Their values don't matter because
    # they won't be used.
    x2_safe = array_ops.where_v2(x2 != 0, x2, constant_op.constant(1, x2.dtype))
    x1, x2 = (
        array_ops.where_v2(x2 != 0, x2, x1),
        array_ops.where_v2(
            x2 != 0,
            math_ops.mod(x1, x2_safe),
            constant_op.constant(0, x2.dtype),
        ),
    )
    return (
        array_ops.where_v2(x1 < x2, x2, x1),
        array_ops.where_v2(x1 < x2, x1, x2),
    )

  if not np.issubdtype(
      x1.dtype.as_numpy_dtype, np.integer
  ) or not np.issubdtype(x2.dtype.as_numpy_dtype, np.integer):
    raise ValueError('Arguments to gcd must be integers.')
  shape = array_ops.broadcast_dynamic_shape(
      array_ops.shape(x1), array_ops.shape(x2)
  )
  x1 = array_ops.broadcast_to(x1, shape)
  x2 = array_ops.broadcast_to(x2, shape)
  value, _ = while_loop.while_loop(
      _gcd_cond_fn, _gcd_body_fn, (math_ops.abs(x1), math_ops.abs(x2))
  )
  return value


# Note that np.gcd may not be present in some supported versions of numpy.
@tf_export.tf_export('experimental.numpy.gcd', v1=[])
@np_utils.np_doc('gcd')
def gcd(x1, x2):
  return _bin_op(_tf_gcd, x1, x2)


# Note that np.lcm may not be present in some supported versions of numpy.
@tf_export.tf_export('experimental.numpy.lcm', v1=[])
@np_utils.np_doc('lcm')
def lcm(x1, x2):  # pylint: disable=missing-function-docstring
  def f(x1, x2):
    d = _tf_gcd(x1, x2)
    # Same as the `x2_safe` trick above
    d_safe = array_ops.where_v2(
        math_ops.equal(d, 0), constant_op.constant(1, d.dtype), d
    )
    x1 = math_ops.abs(x1)
    x2 = math_ops.abs(x2)
    return array_ops.where_v2(
        math_ops.equal(d, 0),
        constant_op.constant(0, d.dtype),
        x1 * (x2 // d_safe),
    )

  return _bin_op(f, x1, x2)


def _bitwise_binary_op(tf_fn, x1, x2):  # pylint: disable=missing-function-docstring
  def f(x1, x2):
    is_bool = x1.dtype == dtypes.bool
    if is_bool:
      assert x2.dtype == dtypes.bool
      x1 = math_ops.cast(x1, dtypes.int8)
      x2 = math_ops.cast(x2, dtypes.int8)
    r = tf_fn(x1, x2)
    if is_bool:
      r = math_ops.cast(r, dtypes.bool)
    return r

  return _bin_op(f, x1, x2)


@tf_export.tf_export('experimental.numpy.bitwise_and', v1=[])
@np_utils.np_doc('bitwise_and')
def bitwise_and(x1, x2):
  return _bitwise_binary_op(bitwise_ops.bitwise_and, x1, x2)


@tf_export.tf_export('experimental.numpy.bitwise_or', v1=[])
@np_utils.np_doc('bitwise_or')
def bitwise_or(x1, x2):
  return _bitwise_binary_op(bitwise_ops.bitwise_or, x1, x2)


@tf_export.tf_export('experimental.numpy.bitwise_xor', v1=[])
@np_utils.np_doc('bitwise_xor')
def bitwise_xor(x1, x2):
  return _bitwise_binary_op(bitwise_ops.bitwise_xor, x1, x2)


@tf_export.tf_export('experimental.numpy.bitwise_not', v1=[])
@np_utils.np_doc('bitwise_not', link=np_utils.AliasOf('invert'))
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
    promote_to_float: whether to cast the argument to a float dtype if it is not
      already.

  Returns:
    An ndarray with the same shape as `x`. The default output dtype is
    determined by `np_utils.result_type(float)`, unless x is an ndarray with a
    floating point type, in which case the output type is same as x.dtype.
  """
  x = np_array_ops.asarray(x)
  if promote_to_float and not np.issubdtype(x.dtype.as_numpy_dtype, np.inexact):
    x = x.astype(np_utils.result_type(float))
  return tf_fn(x)


@tf_export.tf_export('experimental.numpy.log', v1=[])
@np_utils.np_doc('log')
def log(x):
  return _scalar(math_ops.log, x, True)


@tf_export.tf_export('experimental.numpy.exp', v1=[])
@np_utils.np_doc('exp')
def exp(x):
  return _scalar(math_ops.exp, x, True)


@tf_export.tf_export('experimental.numpy.sqrt', v1=[])
@np_utils.np_doc('sqrt')
def sqrt(x):
  return _scalar(math_ops.sqrt, x, True)


@tf_export.tf_export('experimental.numpy.abs', v1=[])
@np_utils.np_doc('abs', link=np_utils.AliasOf('absolute'))
def abs(x):  # pylint: disable=redefined-builtin
  return _scalar(math_ops.abs, x)


@tf_export.tf_export('experimental.numpy.absolute', v1=[])
@np_utils.np_doc('absolute')
def absolute(x):
  return abs(x)


@tf_export.tf_export('experimental.numpy.fabs', v1=[])
@np_utils.np_doc('fabs')
def fabs(x):
  return abs(x)


@tf_export.tf_export('experimental.numpy.ceil', v1=[])
@np_utils.np_doc('ceil')
def ceil(x):
  return _scalar(math_ops.ceil, x, True)


@tf_export.tf_export('experimental.numpy.floor', v1=[])
@np_utils.np_doc('floor')
def floor(x):
  return _scalar(math_ops.floor, x, True)


@tf_export.tf_export('experimental.numpy.conj', v1=[])
@np_utils.np_doc('conj')
def conj(x):
  return _scalar(math_ops.conj, x)


@tf_export.tf_export('experimental.numpy.negative', v1=[])
@np_utils.np_doc('negative')
def negative(x):
  return _scalar(math_ops.negative, x)


@tf_export.tf_export('experimental.numpy.reciprocal', v1=[])
@np_utils.np_doc('reciprocal')
def reciprocal(x):
  return _scalar(math_ops.reciprocal, x)


@tf_export.tf_export('experimental.numpy.signbit', v1=[])
@np_utils.np_doc('signbit')
def signbit(x):
  def f(x):
    if x.dtype == dtypes.bool:
      return array_ops.fill(array_ops.shape(x), False)
    return x < 0

  return _scalar(f, x)


@tf_export.tf_export('experimental.numpy.sin', v1=[])
@np_utils.np_doc('sin')
def sin(x):
  return _scalar(math_ops.sin, x, True)


@tf_export.tf_export('experimental.numpy.cos', v1=[])
@np_utils.np_doc('cos')
def cos(x):
  return _scalar(math_ops.cos, x, True)


@tf_export.tf_export('experimental.numpy.tan', v1=[])
@np_utils.np_doc('tan')
def tan(x):
  return _scalar(math_ops.tan, x, True)


@tf_export.tf_export('experimental.numpy.sinh', v1=[])
@np_utils.np_doc('sinh')
def sinh(x):
  return _scalar(math_ops.sinh, x, True)


@tf_export.tf_export('experimental.numpy.cosh', v1=[])
@np_utils.np_doc('cosh')
def cosh(x):
  return _scalar(math_ops.cosh, x, True)


@tf_export.tf_export('experimental.numpy.tanh', v1=[])
@np_utils.np_doc('tanh')
def tanh(x):
  return _scalar(math_ops.tanh, x, True)


@tf_export.tf_export('experimental.numpy.arcsin', v1=[])
@np_utils.np_doc('arcsin')
def arcsin(x):
  return _scalar(math_ops.asin, x, True)


@tf_export.tf_export('experimental.numpy.arccos', v1=[])
@np_utils.np_doc('arccos')
def arccos(x):
  return _scalar(math_ops.acos, x, True)


@tf_export.tf_export('experimental.numpy.arctan', v1=[])
@np_utils.np_doc('arctan')
def arctan(x):
  return _scalar(math_ops.atan, x, True)


@tf_export.tf_export('experimental.numpy.arcsinh', v1=[])
@np_utils.np_doc('arcsinh')
def arcsinh(x):
  return _scalar(math_ops.asinh, x, True)


@tf_export.tf_export('experimental.numpy.arccosh', v1=[])
@np_utils.np_doc('arccosh')
def arccosh(x):
  return _scalar(math_ops.acosh, x, True)


@tf_export.tf_export('experimental.numpy.arctanh', v1=[])
@np_utils.np_doc('arctanh')
def arctanh(x):
  return _scalar(math_ops.atanh, x, True)


@tf_export.tf_export('experimental.numpy.deg2rad', v1=[])
@np_utils.np_doc('deg2rad')
def deg2rad(x):
  def f(x):
    return x * (np.pi / 180.0)

  return _scalar(f, x, True)


@tf_export.tf_export('experimental.numpy.rad2deg', v1=[])
@np_utils.np_doc('rad2deg')
def rad2deg(x):
  return x * (180.0 / np.pi)


_tf_float_types = [
    dtypes.bfloat16,
    dtypes.float16,
    dtypes.float32,
    dtypes.float64,
]


@tf_export.tf_export('experimental.numpy.angle', v1=[])
@np_utils.np_doc('angle')
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


@tf_export.tf_export('experimental.numpy.cbrt', v1=[])
@np_utils.np_doc('cbrt')
def cbrt(x):
  def f(x):
    # __pow__ can't handle negative base, so we use `abs` here.
    rt = math_ops.abs(x) ** (1.0 / 3)
    return array_ops.where_v2(x < 0, -rt, rt)

  return _scalar(f, x, True)


@tf_export.tf_export('experimental.numpy.conjugate', v1=[])
@np_utils.np_doc('conjugate', link=np_utils.AliasOf('conj'))
def conjugate(x):
  return _scalar(math_ops.conj, x)


@tf_export.tf_export('experimental.numpy.exp2', v1=[])
@np_utils.np_doc('exp2')
def exp2(x):
  def f(x):
    return 2**x

  return _scalar(f, x, True)


@tf_export.tf_export('experimental.numpy.expm1', v1=[])
@np_utils.np_doc('expm1')
def expm1(x):
  return _scalar(math_ops.expm1, x, True)


@tf_export.tf_export('experimental.numpy.fix', v1=[])
@np_utils.np_doc('fix')
def fix(x):
  def f(x):
    return array_ops.where_v2(x < 0, math_ops.ceil(x), math_ops.floor(x))

  return _scalar(f, x, True)


@tf_export.tf_export('experimental.numpy.iscomplex', v1=[])
@np_utils.np_doc('iscomplex')
def iscomplex(x):
  return np_array_ops.imag(x) != 0


@tf_export.tf_export('experimental.numpy.isreal', v1=[])
@np_utils.np_doc('isreal')
def isreal(x):
  return np_array_ops.imag(x) == 0


@tf_export.tf_export('experimental.numpy.iscomplexobj', v1=[])
@np_utils.np_doc('iscomplexobj')
def iscomplexobj(x):
  x = np_array_ops.array(x)
  return np.issubdtype(x.dtype.as_numpy_dtype, np.complexfloating)


@tf_export.tf_export('experimental.numpy.isrealobj', v1=[])
@np_utils.np_doc('isrealobj')
def isrealobj(x):
  return not iscomplexobj(x)


@tf_export.tf_export('experimental.numpy.isnan', v1=[])
@np_utils.np_doc('isnan')
def isnan(x):
  return _scalar(math_ops.is_nan, x, True)


def _make_nan_reduction(np_fun_name, reduction, init_val):
  """Helper to generate nan* functions."""

  @np_utils.np_doc(np_fun_name)
  def nan_reduction(a, axis=None, dtype=None, keepdims=False):
    a = np_array_ops.array(a)
    v = np_array_ops.array(init_val, dtype=a.dtype)
    return reduction(
        np_array_ops.where(isnan(a), v, a),
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
    )

  return nan_reduction


nansum = tf_export.tf_export('experimental.numpy.nansum', v1=[])(
    _make_nan_reduction('nansum', np_array_ops.sum, 0)
)
nanprod = tf_export.tf_export('experimental.numpy.nanprod', v1=[])(
    _make_nan_reduction('nanprod', np_array_ops.prod, 1)
)


@tf_export.tf_export('experimental.numpy.nanmean', v1=[])
@np_utils.np_doc('nanmean')
def nanmean(a, axis=None, dtype=None, keepdims=None):  # pylint: disable=missing-docstring
  a = np_array_ops.array(a)
  if np.issubdtype(a.dtype.as_numpy_dtype, np.bool_) or np.issubdtype(
      a.dtype.as_numpy_dtype, np.integer
  ):
    return np_array_ops.mean(a, axis=axis, dtype=dtype, keepdims=keepdims)
  nan_mask = logical_not(isnan(a))
  if dtype is None:
    dtype = a.dtype.as_numpy_dtype
  normalizer = np_array_ops.sum(
      nan_mask, axis=axis, dtype=dtype, keepdims=keepdims
  )
  return nansum(a, axis=axis, dtype=dtype, keepdims=keepdims) / normalizer


@tf_export.tf_export('experimental.numpy.isfinite', v1=[])
@np_utils.np_doc('isfinite')
def isfinite(x):
  return _scalar(math_ops.is_finite, x, True)


@tf_export.tf_export('experimental.numpy.isinf', v1=[])
@np_utils.np_doc('isinf')
def isinf(x):
  if x.dtype.is_floating:
    return _scalar(math_ops.is_inf, x, True)
  return False


@tf_export.tf_export('experimental.numpy.isneginf', v1=[])
@np_utils.np_doc('isneginf')
def isneginf(x):
  if x.dtype.is_floating:
    return x == np_array_ops.full_like(x, -np.inf)
  return False


@tf_export.tf_export('experimental.numpy.isposinf', v1=[])
@np_utils.np_doc('isposinf')
def isposinf(x):
  if x.dtype.is_floating:
    return x == np_array_ops.full_like(x, np.inf)
  return False


@tf_export.tf_export('experimental.numpy.log2', v1=[])
@np_utils.np_doc('log2')
def log2(x):
  return log(x) / np.log(2)


@tf_export.tf_export('experimental.numpy.log10', v1=[])
@np_utils.np_doc('log10')
def log10(x):
  return log(x) / np.log(10)


@tf_export.tf_export('experimental.numpy.log1p', v1=[])
@np_utils.np_doc('log1p')
def log1p(x):
  return _scalar(math_ops.log1p, x, True)


@tf_export.tf_export('experimental.numpy.positive', v1=[])
@np_utils.np_doc('positive')
def positive(x):
  return _scalar(lambda x: x, x)


@tf_export.tf_export('experimental.numpy.sinc', v1=[])
@np_utils.np_doc('sinc')
def sinc(x):
  def f(x):
    pi_x = x * np.pi
    return array_ops.where_v2(
        x == 0, array_ops.ones_like(x), math_ops.sin(pi_x) / pi_x
    )

  return _scalar(f, x, True)


@tf_export.tf_export('experimental.numpy.square', v1=[])
@np_utils.np_doc('square')
def square(x):
  return _scalar(math_ops.square, x)


@tf_export.tf_export('experimental.numpy.diff', v1=[])
@np_utils.np_doc('diff')
def diff(a, n=1, axis=-1):  # pylint: disable=missing-function-docstring
  def f(a):
    # TODO(agarwal): transpose and reshape to N, H, 1 and do a 1D convolution
    # TODO(agarwal): avoid depending on static rank.
    nd = a.shape.rank
    if nd is None:
      raise ValueError(
          'Function `diff` currently requires a known rank for input `a`. '
          f'Received: a={a} (unknown rank)'
      )
    if (axis + nd if axis < 0 else axis) >= nd:
      raise ValueError(
          f'Argument `axis` (received axis={axis}) is out of bounds '
          f'for input {a} of rank {nd}.'
      )
    if n < 0:
      raise ValueError(
          f'Argument `order` must be a non-negative integer. Received: axis={n}'
      )
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


def _wrap(f, reverse=False):
  """Wraps binary ops so they can be added as operator overloads on ndarray."""

  def _f(a, b):
    if reverse:
      a, b = b, a

    if (
        getattr(b, '__array_priority__', 0)
        > np_arrays.ndarray.__array_priority__
    ):
      return NotImplemented

    return f(a, b)

  return _f


def _comparison(tf_fun, x1, x2, cast_bool_to_int=False):
  """Helper function for comparision."""
  dtype = np_utils.result_type(x1, x2)
  # Cast x1 and x2 to the result_type if needed.
  x1 = np_array_ops.array(x1, dtype=dtype)
  x2 = np_array_ops.array(x2, dtype=dtype)
  if cast_bool_to_int and x1.dtype == dtypes.bool:
    x1 = math_ops.cast(x1, dtypes.int32)
    x2 = math_ops.cast(x2, dtypes.int32)
  return tf_fun(x1, x2)


@tf_export.tf_export('experimental.numpy.equal', v1=[])
@np_utils.np_doc('equal')
def equal(x1, x2):
  return _comparison(math_ops.equal, x1, x2)


@tf_export.tf_export('experimental.numpy.not_equal', v1=[])
@np_utils.np_doc('not_equal')
def not_equal(x1, x2):
  return _comparison(math_ops.not_equal, x1, x2)


@tf_export.tf_export('experimental.numpy.greater', v1=[])
@np_utils.np_doc('greater')
def greater(x1, x2):
  return _comparison(math_ops.greater, x1, x2, True)


@tf_export.tf_export('experimental.numpy.greater_equal', v1=[])
@np_utils.np_doc('greater_equal')
def greater_equal(x1, x2):
  return _comparison(math_ops.greater_equal, x1, x2, True)


@tf_export.tf_export('experimental.numpy.less', v1=[])
@np_utils.np_doc('less')
def less(x1, x2):
  return _comparison(math_ops.less, x1, x2, True)


@tf_export.tf_export('experimental.numpy.less_equal', v1=[])
@np_utils.np_doc('less_equal')
def less_equal(x1, x2):
  return _comparison(math_ops.less_equal, x1, x2, True)


@tf_export.tf_export('experimental.numpy.array_equal', v1=[])
@np_utils.np_doc('array_equal')
def array_equal(a1, a2):  # pylint: disable=missing-function-docstring
  def f(x1, x2):
    return np_utils.cond(
        math_ops.equal(array_ops.rank(x1), array_ops.rank(x2)),
        lambda: np_utils.cond(  # pylint: disable=g-long-lambda
            np_utils.reduce_all(
                math_ops.equal(array_ops.shape(x1), array_ops.shape(x2))
            ),
            lambda: math_ops.reduce_all(math_ops.equal(x1, x2)),
            lambda: constant_op.constant(False),
        ),
        lambda: constant_op.constant(False),
    )

  return _comparison(f, a1, a2)


def _logical_binary_op(tf_fun, x1, x2):
  x1 = np_array_ops.array(x1, dtype=np.bool_)
  x2 = np_array_ops.array(x2, dtype=np.bool_)
  return tf_fun(x1, x2)


@tf_export.tf_export('experimental.numpy.logical_and', v1=[])
@np_utils.np_doc('logical_and')
def logical_and(x1, x2):
  return _logical_binary_op(math_ops.logical_and, x1, x2)


@tf_export.tf_export('experimental.numpy.logical_or', v1=[])
@np_utils.np_doc('logical_or')
def logical_or(x1, x2):
  return _logical_binary_op(math_ops.logical_or, x1, x2)


@tf_export.tf_export('experimental.numpy.logical_xor', v1=[])
@np_utils.np_doc('logical_xor')
def logical_xor(x1, x2):
  return _logical_binary_op(math_ops.logical_xor, x1, x2)


@tf_export.tf_export('experimental.numpy.logical_not', v1=[])
@np_utils.np_doc('logical_not')
def logical_not(x):
  x = np_array_ops.array(x, dtype=np.bool_)
  return math_ops.logical_not(x)


@tf_export.tf_export('experimental.numpy.linspace', v1=[])
@np_utils.np_doc('linspace')
def linspace(  # pylint: disable=missing-docstring
    start, stop, num=50, endpoint=True, retstep=False, dtype=float, axis=0
):
  if dtype:
    dtype = np_utils.result_type(dtype)
  start = np_array_ops.array(start, dtype=dtype)
  stop = np_array_ops.array(stop, dtype=dtype)
  if num < 0:
    raise ValueError(
        'Argument `num` (number of samples) must be a non-negative integer. '
        f'Received: num={num}'
    )
  step = ops.convert_to_tensor(np.nan)
  if endpoint:
    result = math_ops.linspace(start, stop, num, axis=axis)
    if num > 1:
      step = (stop - start) / (num - 1)
  else:
    # math_ops.linspace does not support endpoint=False so we manually handle it
    # here.
    if num > 0:
      step = (stop - start) / num
    if num > 1:
      new_stop = math_ops.cast(stop, step.dtype) - step
      start = math_ops.cast(start, new_stop.dtype)
      result = math_ops.linspace(start, new_stop, num, axis=axis)
    else:
      result = math_ops.linspace(start, stop, num, axis=axis)
  if dtype:
    if dtype.is_integer:
      # Since numpy 1.20, linspace's rounding is towards -inf instead of 0
      result = math_ops.floor(result)
    result = math_ops.cast(result, dtype)
  if retstep:
    return (result, step)
  else:
    return result


@tf_export.tf_export('experimental.numpy.logspace', v1=[])
@np_utils.np_doc('logspace')
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
  dtype = np_utils.result_type(start, stop, dtype)
  result = linspace(
      start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis
  )
  result = math_ops.pow(math_ops.cast(base, result.dtype), result)
  if dtype:
    result = math_ops.cast(result, dtype)
  return result


@tf_export.tf_export('experimental.numpy.geomspace', v1=[])
@np_utils.np_doc('geomspace')
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):  # pylint: disable=missing-docstring
  dtype = (
      dtypes.as_dtype(dtype)  # pylint: disable=g-long-ternary
      if dtype
      else np_utils.result_type(
          start, stop, float(num), np_array_ops.zeros((), dtype)
      )
  )
  computation_dtype = np.promote_types(dtype.as_numpy_dtype, np.float32)
  start = np_array_ops.asarray(start, dtype=computation_dtype)
  stop = np_array_ops.asarray(stop, dtype=computation_dtype)
  # follow the numpy geomspace convention for negative and complex endpoints
  start_sign = 1 - np_array_ops.sign(np_array_ops.real(start))
  stop_sign = 1 - np_array_ops.sign(np_array_ops.real(stop))
  signflip = 1 - start_sign * stop_sign // 2
  res = signflip * logspace(
      log10(signflip * start),
      log10(signflip * stop),
      num,
      endpoint=endpoint,
      base=10.0,
      dtype=computation_dtype,
      axis=0,
  )
  if axis != 0:
    res = np_array_ops.moveaxis(res, 0, axis)
  return math_ops.cast(res, dtype)


@tf_export.tf_export('experimental.numpy.ptp', v1=[])
@np_utils.np_doc('ptp')
def ptp(a, axis=None, keepdims=None):
  return np_array_ops.amax(a, axis=axis, keepdims=keepdims) - np_array_ops.amin(
      a, axis=axis, keepdims=keepdims
  )


@tf_export.tf_export('experimental.numpy.concatenate', v1=[])
@np_utils.np_doc_only('concatenate')
def concatenate(arys, axis=0):  # pylint: disable=missing-function-docstring
  if not isinstance(arys, (list, tuple)):
    arys = [arys]
  if not arys:
    raise ValueError(
        'Need at least one array to concatenate. Received empty '
        f'input: arys={arys}'
    )
  dtype = np_utils.result_type(*arys)
  arys = [np_array_ops.array(array, dtype=dtype) for array in arys]
  return array_ops.concat(arys, axis)


@tf_export.tf_export('experimental.numpy.tile', v1=[])
@np_utils.np_doc_only('tile')
def tile(a, reps):  # pylint: disable=missing-function-docstring
  a = np_array_ops.array(a)
  reps = array_ops.reshape(np_array_ops.array(reps, dtype=dtypes.int32), [-1])

  a_rank = array_ops.rank(a)
  reps_size = array_ops.size(reps)
  reps = array_ops.pad(
      reps, [[math_ops.maximum(a_rank - reps_size, 0), 0]], constant_values=1
  )
  a_shape = array_ops.pad(
      array_ops.shape(a),
      [[math_ops.maximum(reps_size - a_rank, 0), 0]],
      constant_values=1,
  )
  a = array_ops.reshape(a, a_shape)

  return array_ops.tile(a, reps)


@tf_export.tf_export('experimental.numpy.count_nonzero', v1=[])
@np_utils.np_doc('count_nonzero')
def count_nonzero(a, axis=None):
  return math_ops.count_nonzero(np_array_ops.array(a), axis)


@tf_export.tf_export('experimental.numpy.argsort', v1=[])
@np_utils.np_doc('argsort')
def argsort(a, axis=-1, kind='quicksort', order=None):  # pylint: disable=missing-docstring
  # TODO(nareshmodi): make string tensors also work.
  if kind not in ('quicksort', 'stable'):
    raise ValueError(
        'Invalid value for argument `kind`. '
        'Only kind="quicksort" and kind="stable" are supported. '
        f'Received: kind={kind}'
    )
  if order is not None:
    raise ValueError('The `order` argument is not supported. Pass order=None')
  stable = kind == 'stable'

  a = np_array_ops.array(a)

  def _argsort(a, axis, stable):
    if axis is None:
      a = array_ops.reshape(a, [-1])
      axis = 0

    return sort_ops.argsort(a, axis, stable=stable)

  tf_ans = np_utils.cond(
      math_ops.equal(array_ops.rank(a), 0),
      lambda: constant_op.constant([0]),
      lambda: _argsort(a, axis, stable),
  )

  if ops.is_auto_dtype_conversion_enabled():
    return np_array_ops.array(tf_ans, dtype=int)
  else:
    return np_array_ops.array(tf_ans, dtype=np.intp)


@tf_export.tf_export('experimental.numpy.sort', v1=[])
@np_utils.np_doc('sort')
def sort(a, axis=-1, kind='quicksort', order=None):  # pylint: disable=missing-docstring
  if kind != 'quicksort':
    raise ValueError(
        'Invalid value for argument `kind`. '
        'Only kind="quicksort" is supported. '
        f'Received: kind={kind}'
    )
  if order is not None:
    raise ValueError('The `order` argument is not supported. Pass order=None')

  a = np_array_ops.array(a)

  if axis is None:
    return sort_ops.sort(array_ops.reshape(a, [-1]), 0)
  else:
    return sort_ops.sort(a, axis)


def _argminmax(fn, a, axis=None):
  a = np_array_ops.array(a)
  if axis is None:
    # When axis is None numpy flattens the array.
    a_t = array_ops.reshape(a, [-1])
  else:
    a_t = np_array_ops.atleast_1d(a)
  return fn(input=a_t, axis=axis)


@tf_export.tf_export('experimental.numpy.argmax', v1=[])
@np_utils.np_doc('argmax')
def argmax(a, axis=None):
  return _argminmax(math_ops.argmax, a, axis)


@tf_export.tf_export('experimental.numpy.argmin', v1=[])
@np_utils.np_doc('argmin')
def argmin(a, axis=None):
  return _argminmax(math_ops.argmin, a, axis)


@tf_export.tf_export('experimental.numpy.append', v1=[])
@np_utils.np_doc('append')
def append(arr, values, axis=None):
  if axis is None:
    return concatenate([np_array_ops.ravel(arr), np_array_ops.ravel(values)], 0)
  else:
    return concatenate([arr, values], axis=axis)


@tf_export.tf_export('experimental.numpy.average', v1=[])
@np_utils.np_doc('average')
def average(a, axis=None, weights=None, returned=False):  # pylint: disable=missing-docstring
  if axis is not None and not isinstance(axis, int):
    # TODO(wangpeng): Support tuple of ints as `axis`
    raise ValueError(
        'Argument `axis` must be an integer. '
        f'Received axis={axis} (of type {type(axis)})'
    )
  a = np_array_ops.array(a)
  default_float_type = np_utils.result_type(float)
  if weights is None:  # Treat all weights as 1
    if not np.issubdtype(a.dtype.as_numpy_dtype, np.inexact):
      a = a.astype(np_utils.result_type(a.dtype, default_float_type))
    avg = math_ops.reduce_mean(a, axis=axis)
    if returned:
      if axis is None:
        weights_sum = array_ops.size(a)
      else:
        weights_sum = array_ops.shape(a)[axis]
      weights_sum = math_ops.cast(weights_sum, a.dtype)
  else:
    if np.issubdtype(a.dtype.as_numpy_dtype, np.inexact):
      out_dtype = np_utils.result_type(a.dtype, weights)
    else:
      out_dtype = np_utils.result_type(a.dtype, weights, default_float_type)
    a = np_array_ops.array(a, out_dtype)
    weights = np_array_ops.array(weights, out_dtype)

    def rank_equal_case():
      control_flow_assert.Assert(
          math_ops.reduce_all(array_ops.shape(a) == array_ops.shape(weights)),
          [array_ops.shape(a), array_ops.shape(weights)],
      )
      weights_sum = math_ops.reduce_sum(weights, axis=axis)
      avg = math_ops.reduce_sum(a * weights, axis=axis) / weights_sum
      return avg, weights_sum

    if axis is None:
      avg, weights_sum = rank_equal_case()
    else:

      def rank_not_equal_case():
        control_flow_assert.Assert(
            array_ops.rank(weights) == 1, [array_ops.rank(weights)]
        )
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
          rank_equal_case,
          rank_not_equal_case,
      )

  avg = np_array_ops.array(avg)
  if returned:
    weights_sum = np_array_ops.broadcast_to(weights_sum, array_ops.shape(avg))
    return avg, weights_sum
  return avg


@tf_export.tf_export('experimental.numpy.trace', v1=[])
@np_utils.np_doc('trace')
def trace(a, offset=0, axis1=0, axis2=1, dtype=None):  # pylint: disable=missing-docstring
  if dtype:
    dtype = np_utils.result_type(dtype)
  a = np_array_ops.asarray(a, dtype)

  if offset == 0:
    a_shape = a.shape
    if a_shape.rank is not None:
      rank = len(a_shape)
      if (axis1 == -2 or axis1 == rank - 2) and (
          axis2 == -1 or axis2 == rank - 1
      ):
        return math_ops.trace(a)

  a = np_array_ops.diagonal(a, offset, axis1, axis2)
  return np_array_ops.sum(a, -1, dtype)


@tf_export.tf_export('experimental.numpy.meshgrid', v1=[])
@np_utils.np_doc('meshgrid')
def meshgrid(*xi, **kwargs):
  """This currently requires copy=True and sparse=False."""
  sparse = kwargs.get('sparse', False)
  if sparse:
    raise ValueError(
        'Function `meshgrid` does not support returning sparse arrays yet. '
        f'Received: sparse={sparse}'
    )

  copy = kwargs.get('copy', True)
  if not copy:
    raise ValueError(
        f'Function `meshgrid` only supports copy=True. Received: copy={copy}'
    )

  indexing = kwargs.get('indexing', 'xy')

  xi = [np_array_ops.asarray(arg) for arg in xi]
  kwargs = {'indexing': indexing}

  outputs = array_ops.meshgrid(*xi, **kwargs)

  return outputs


# Uses np_doc_only here because np.einsum (in 1.16) doesn't have argument
# `subscripts`, even though the doc says it has.
@tf_export.tf_export('experimental.numpy.einsum', v1=[])
@np_utils.np_doc_only('einsum')
def einsum(subscripts, *operands, **kwargs):  # pylint: disable=missing-docstring
  casting = kwargs.get('casting', 'safe')
  optimize = kwargs.get('optimize', False)
  if casting == 'safe':
    operands = np_array_ops._promote_dtype(*operands)  # pylint: disable=protected-access
  elif casting == 'no':
    operands = [np_array_ops.asarray(x) for x in operands]
  else:
    raise ValueError(
        'Invalid value for argument `casting`. '
        f'Expected casting="safe" or casting="no". Received: casting={casting}'
    )
  if not optimize:
    # TF doesn't have a "no optimization" option.
    # TODO(wangpeng): Print a warning that np and tf use different
    #   optimizations.
    tf_optimize = 'greedy'
  elif optimize == True:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
    tf_optimize = 'greedy'
  elif optimize == 'greedy':
    tf_optimize = 'greedy'
  elif optimize == 'optimal':
    tf_optimize = 'optimal'
  else:
    raise ValueError(
        'Invalid value for argument `optimize`. '
        'Expected one of {True, "greedy", "optimal"}. '
        f'Received: optimize={optimize}'
    )

  res = special_math_ops.einsum(subscripts, *operands, optimize=tf_optimize)
  return res


def _tensor_t(self):
  """Returns a Tensor which is the transpose of this Tensor."""
  return self.transpose()


def _tensor_ndim(self):
  """Returns the rank of the Tensor."""
  return self.shape.ndims


def _tensor_pos(self):
  """Returns self, for unary operator `+`."""
  return self


def _tensor_size(self):
  """Returns the number of elements in this Tensor, if fully known."""
  if not self.shape.is_fully_defined():
    return None
  return np.prod(self.shape.as_list())


def _tensor_tolist(self):
  if ops.is_symbolic_tensor(self):
    raise ValueError('Symbolic Tensors do not support the tolist API.')

  return self._numpy().tolist()  # pylint: disable=protected-access


def _enable_numpy_methods(tensor_class):
  """A helper method for adding additional NumPy methods."""
  t = property(_tensor_t)
  setattr(tensor_class, 'T', t)

  ndim = property(_tensor_ndim)
  setattr(tensor_class, 'ndim', ndim)

  size = property(_tensor_size)
  setattr(tensor_class, 'size', size)

  setattr(tensor_class, '__pos__', _tensor_pos)
  setattr(tensor_class, 'tolist', _tensor_tolist)

  # TODO(b/178540516): Make a custom `setattr` that changes the method's
  #   docstring to the TF one.
  setattr(tensor_class, 'transpose', np_array_ops.transpose)
  setattr(tensor_class, 'flatten', np_array_ops.flatten)
  setattr(tensor_class, 'reshape', np_array_ops._reshape_method_wrapper)  # pylint: disable=protected-access
  setattr(tensor_class, 'ravel', np_array_ops.ravel)
  setattr(tensor_class, 'clip', clip)
  setattr(tensor_class, 'astype', math_ops.cast)
  setattr(tensor_class, '__round__', np_array_ops.around)
  setattr(tensor_class, 'max', np_array_ops.amax)
  setattr(tensor_class, 'mean', np_array_ops.mean)
  setattr(tensor_class, 'min', np_array_ops.amin)

  # TODO(wangpeng): Remove `data` when all uses of it are removed
  data = property(lambda self: self)
  setattr(tensor_class, 'data', data)


def enable_numpy_methods_on_tensor():
  """Adds additional NumPy methods on tf.Tensor class."""
  _enable_numpy_methods(tensor.Tensor)
