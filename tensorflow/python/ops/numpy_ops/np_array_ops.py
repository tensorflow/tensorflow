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
"""Common array methods."""
# pylint: disable=g-direct-tensorflow-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import six

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import nest


def empty(shape, dtype=float):  # pylint: disable=redefined-outer-name
  """Returns an empty array with the specified shape and dtype.

  Args:
    shape: A fully defined shape. Could be - NumPy array or a python scalar,
      list or tuple of integers, - TensorFlow tensor/ndarray of integer type and
      rank <=1.
    dtype: Optional, defaults to float. The type of the resulting ndarray. Could
      be a python type, a NumPy type or a TensorFlow `DType`.

  Returns:
    An ndarray.
  """
  return zeros(shape, dtype)


def empty_like(a, dtype=None):
  """Returns an empty array with the shape and possibly type of the input array.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can be
      converted to a Tensor using `tf.convert_to_tensor`.
    dtype: Optional, defaults to dtype of the input array. The type of the
      resulting ndarray. Could be a python type, a NumPy type or a TensorFlow
      `DType`.

  Returns:
    An ndarray.
  """
  return zeros_like(a, dtype)


def zeros(shape, dtype=float):  # pylint: disable=redefined-outer-name
  """Returns an ndarray with the given shape and type filled with zeros.

  Args:
    shape: A fully defined shape. Could be - NumPy array or a python scalar,
      list or tuple of integers, - TensorFlow tensor/ndarray of integer type and
      rank <=1.
    dtype: Optional, defaults to float. The type of the resulting ndarray. Could
      be a python type, a NumPy type or a TensorFlow `DType`.

  Returns:
    An ndarray.
  """
  if dtype:
    dtype = np_utils.result_type(dtype)
  if isinstance(shape, np_arrays.ndarray):
    shape = shape.data
  return np_arrays.tensor_to_ndarray(array_ops.zeros(shape, dtype=dtype))


def zeros_like(a, dtype=None):
  """Returns an array of zeros with the shape and type of the input array.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can be
      converted to a Tensor using `tf.convert_to_tensor`.
    dtype: Optional, defaults to dtype of the input array. The type of the
      resulting ndarray. Could be a python type, a NumPy type or a TensorFlow
      `DType`.

  Returns:
    An ndarray.
  """
  if isinstance(a, np_arrays.ndarray):
    a = a.data
  if dtype is None:
    # We need to let np_utils.result_type decide the dtype, not tf.zeros_like
    dtype = np_utils.result_type(a)
  else:
    # TF and numpy has different interpretations of Python types such as
    # `float`, so we let `np_utils.result_type` decide.
    dtype = np_utils.result_type(dtype)
  dtype = dtypes.as_dtype(dtype)  # Work around b/149877262
  return np_arrays.tensor_to_ndarray(array_ops.zeros_like(a, dtype))


def ones(shape, dtype=float):  # pylint: disable=redefined-outer-name
  """Returns an ndarray with the given shape and type filled with ones.

  Args:
    shape: A fully defined shape. Could be - NumPy array or a python scalar,
      list or tuple of integers, - TensorFlow tensor/ndarray of integer type and
      rank <=1.
    dtype: Optional, defaults to float. The type of the resulting ndarray. Could
      be a python type, a NumPy type or a TensorFlow `DType`.

  Returns:
    An ndarray.
  """
  if dtype:
    dtype = np_utils.result_type(dtype)
  if isinstance(shape, np_arrays.ndarray):
    shape = shape.data
  return np_arrays.tensor_to_ndarray(array_ops.ones(shape, dtype=dtype))


def ones_like(a, dtype=None):
  """Returns an array of ones with the shape and type of the input array.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can be
      converted to a Tensor using `tf.convert_to_tensor`.
    dtype: Optional, defaults to dtype of the input array. The type of the
      resulting ndarray. Could be a python type, a NumPy type or a TensorFlow
      `DType`.

  Returns:
    An ndarray.
  """
  if isinstance(a, np_arrays.ndarray):
    a = a.data
  if dtype is None:
    dtype = np_utils.result_type(a)
  else:
    dtype = np_utils.result_type(dtype)
  return np_arrays.tensor_to_ndarray(array_ops.ones_like(a, dtype))


@np_utils.np_doc(np.eye)
def eye(N, M=None, k=0, dtype=float):  # pylint: disable=invalid-name,missing-docstring
  if dtype:
    dtype = np_utils.result_type(dtype)
  if not M:
    M = N
  # Making sure N, M and k are `int`
  N = int(N)
  M = int(M)
  k = int(k)
  if k >= M or -k >= N:
    # tf.linalg.diag will raise an error in this case
    return zeros([N, M], dtype=dtype)
  if k == 0:
    return np_arrays.tensor_to_ndarray(linalg_ops.eye(N, M, dtype=dtype))
  # We need the precise length, otherwise tf.linalg.diag will raise an error
  diag_len = min(N, M)
  if k > 0:
    if N >= M:
      diag_len -= k
    elif N + k > M:
      diag_len = M - k
  elif k <= 0:
    if M >= N:
      diag_len += k
    elif M - k > N:
      diag_len = N + k
  diagonal_ = array_ops.ones([diag_len], dtype=dtype)
  return np_arrays.tensor_to_ndarray(
      array_ops.matrix_diag(diagonal=diagonal_, num_rows=N, num_cols=M, k=k))


def identity(n, dtype=float):
  """Returns a square array with ones on the main diagonal and zeros elsewhere.

  Args:
    n: number of rows/cols.
    dtype: Optional, defaults to float. The type of the resulting ndarray. Could
      be a python type, a NumPy type or a TensorFlow `DType`.

  Returns:
    An ndarray of shape (n, n) and requested type.
  """
  return eye(N=n, M=n, dtype=dtype)


def full(shape, fill_value, dtype=None):  # pylint: disable=redefined-outer-name
  """Returns an array with given shape and dtype filled with `fill_value`.

  Args:
    shape: A valid shape object. Could be a native python object or an object of
      type ndarray, numpy.ndarray or tf.TensorShape.
    fill_value: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    dtype: Optional, defaults to dtype of the `fill_value`. The type of the
      resulting ndarray. Could be a python type, a NumPy type or a TensorFlow
      `DType`.

  Returns:
    An ndarray.

  Raises:
    ValueError: if `fill_value` can not be broadcast to shape `shape`.
  """
  fill_value = asarray(fill_value, dtype=dtype)
  if np_utils.isscalar(shape):
    shape = array_ops.reshape(shape, [1])
  return np_arrays.tensor_to_ndarray(
      array_ops.broadcast_to(fill_value.data, shape))


# Using doc only here since np full_like signature doesn't seem to have the
# shape argument (even though it exists in the documentation online).
@np_utils.np_doc_only(np.full_like)
def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None):  # pylint: disable=missing-docstring,redefined-outer-name
  """order, subok and shape arguments mustn't be changed."""
  if order != 'K':
    raise ValueError('Non-standard orders are not supported.')
  if not subok:
    raise ValueError('subok being False is not supported.')
  if shape:
    raise ValueError('Overriding the shape is not supported.')

  a = asarray(a).data
  dtype = dtype or np_utils.result_type(a)
  fill_value = asarray(fill_value, dtype=dtype)
  return np_arrays.tensor_to_ndarray(
      array_ops.broadcast_to(fill_value.data, array_ops.shape(a)))


# TODO(wangpeng): investigate whether we can make `copy` default to False.
# TODO(wangpeng): np_utils.np_doc can't handle np.array because np.array is a
#   builtin function. Make np_utils.np_doc support builtin functions.
def array(val, dtype=None, copy=True, ndmin=0):  # pylint: disable=redefined-outer-name
  """Creates an ndarray with the contents of val.

  Args:
    val: array_like. Could be an ndarray, a Tensor or any object that can be
      converted to a Tensor using `tf.convert_to_tensor`.
    dtype: Optional, defaults to dtype of the `val`. The type of the resulting
      ndarray. Could be a python type, a NumPy type or a TensorFlow `DType`.
    copy: Determines whether to create a copy of the backing buffer. Since
      Tensors are immutable, a copy is made only if val is placed on a different
      device than the current one. Even if `copy` is False, a new Tensor may
      need to be built to satisfy `dtype` and `ndim`. This is used only if `val`
      is an ndarray or a Tensor.
    ndmin: The minimum rank of the returned array.

  Returns:
    An ndarray.
  """
  if dtype:
    dtype = np_utils.result_type(dtype)
  if isinstance(val, np_arrays.ndarray):
    result_t = val.data
  else:
    result_t = val

  if copy and isinstance(result_t, ops.Tensor):
    # Note: In eager mode, a copy of `result_t` is made only if it is not on
    # the context device.
    result_t = array_ops.identity(result_t)

  if not isinstance(result_t, ops.Tensor):
    if not dtype:
      dtype = np_utils.result_type(result_t)
    # We can't call `convert_to_tensor(result_t, dtype=dtype)` here because
    # convert_to_tensor doesn't allow incompatible arguments such as (5.5, int)
    # while np.array allows them. We need to convert-then-cast.
    def maybe_data(x):
      if isinstance(x, np_arrays.ndarray):
        return x.data
      return x

    # Handles lists of ndarrays
    result_t = nest.map_structure(maybe_data, result_t)
    # EagerTensor conversion complains about "mixed types" when converting
    # tensors with no dtype information. This is because it infers types based
    # on one selected item in the list. So e.g. when converting [2., 2j]
    # to a tensor, it will select float32 as the inferred type and not be able
    # to convert the list to a float 32 tensor.
    # Since we have some information about the final dtype we care about, we
    # supply that information so that convert_to_tensor will do best-effort
    # conversion to that dtype first.
    result_t = np_arrays.convert_to_tensor(result_t, dtype_hint=dtype)
    result_t = math_ops.cast(result_t, dtype=dtype)
  elif dtype:
    result_t = math_ops.cast(result_t, dtype)
  ndims = array_ops.rank(result_t)

  def true_fn():
    old_shape = array_ops.shape(result_t)
    new_shape = array_ops.concat(
        [array_ops.ones(ndmin - ndims, dtypes.int32), old_shape], axis=0)
    return array_ops.reshape(result_t, new_shape)

  result_t = np_utils.cond(
      np_utils.greater(ndmin, ndims), true_fn, lambda: result_t)
  return np_arrays.tensor_to_ndarray(result_t)


@np_utils.np_doc(np.asarray)
def asarray(a, dtype=None):
  if dtype:
    dtype = np_utils.result_type(dtype)
  if isinstance(a, np_arrays.ndarray) and (not dtype or dtype == a.dtype):
    return a
  return array(a, dtype, copy=False)


@np_utils.np_doc(np.asanyarray)
def asanyarray(a, dtype=None):
  return asarray(a, dtype)


@np_utils.np_doc(np.ascontiguousarray)
def ascontiguousarray(a, dtype=None):
  return array(a, dtype, ndmin=1)


# Numerical ranges.
def arange(start, stop=None, step=1, dtype=None):
  """Returns `step`-separated values in the range [start, stop).

  Args:
    start: Start of the interval. Included in the range.
    stop: End of the interval. If not specified, `start` is treated as 0 and
      `start` value is used as `stop`. If specified, it is not included in the
      range if `step` is integer. When `step` is floating point, it may or may
      not be included.
    step: The difference between 2 consecutive values in the output range. It is
      recommended to use `linspace` instead of using non-integer values for
      `step`.
    dtype: Optional. Type of the resulting ndarray. Could be a python type, a
      NumPy type or a TensorFlow `DType`. If not provided, the largest type of
      `start`, `stop`, `step` is used.

  Raises:
    ValueError: If step is zero.
  """
  if not step:
    raise ValueError('step must be non-zero.')
  if dtype:
    dtype = np_utils.result_type(dtype)
  else:
    if stop is None:
      dtype = np_utils.result_type(start, step)
    else:
      dtype = np_utils.result_type(start, step, stop)
  if step > 0 and ((stop is not None and start > stop) or
                   (stop is None and start < 0)):
    return array([], dtype=dtype)
  if step < 0 and ((stop is not None and start < stop) or
                   (stop is None and start > 0)):
    return array([], dtype=dtype)
  # TODO(srbs): There are some bugs when start or stop is float type and dtype
  # is integer type.
  return np_arrays.tensor_to_ndarray(
      math_ops.cast(math_ops.range(start, limit=stop, delta=step), dtype=dtype))


@np_utils.np_doc(np.geomspace)
def geomspace(start, stop, num=50, endpoint=True, dtype=float):  # pylint: disable=missing-docstring
  if dtype:
    dtype = np_utils.result_type(dtype)
  if num < 0:
    raise ValueError('Number of samples {} must be non-negative.'.format(num))
  if not num:
    return empty([0])
  step = 1.
  if endpoint:
    if num > 1:
      step = math_ops.pow((stop / start), 1 / (num - 1))
  else:
    step = math_ops.pow((stop / start), 1 / num)
  result = math_ops.cast(math_ops.range(num), step.dtype)
  result = math_ops.pow(step, result)
  result = math_ops.multiply(result, start)
  if dtype:
    result = math_ops.cast(result, dtype=dtype)
  return np_arrays.tensor_to_ndarray(result)


# Building matrices.
@np_utils.np_doc(np.diag)
def diag(v, k=0):  # pylint: disable=missing-docstring
  """Raises an error if input is not 1- or 2-d."""
  v = asarray(v).data
  v_rank = array_ops.rank(v)

  v.shape.with_rank_at_most(2)

  # TODO(nareshmodi): Consider a np_utils.Assert version that will fail during
  # tracing time if the shape is known.
  control_flow_ops.Assert(
      np_utils.logical_or(math_ops.equal(v_rank, 1), math_ops.equal(v_rank, 2)),
      [v_rank])

  def _diag(v, k):
    return np_utils.cond(
        math_ops.equal(array_ops.size(v), 0),
        lambda: array_ops.zeros([abs(k), abs(k)], dtype=v.dtype),
        lambda: array_ops.matrix_diag(v, k=k))

  def _diag_part(v, k):
    v_shape = array_ops.shape(v)
    v, k = np_utils.cond(
        np_utils.logical_or(
            np_utils.less_equal(k, -1 * np_utils.getitem(v_shape, 0)),
            np_utils.greater_equal(k, np_utils.getitem(v_shape, 1)),
        ), lambda: (array_ops.zeros([0, 0], dtype=v.dtype), 0), lambda: (v, k))
    result = array_ops.matrix_diag_part(v, k=k)
    return result

  result = np_utils.cond(
      math_ops.equal(v_rank, 1), lambda: _diag(v, k), lambda: _diag_part(v, k))
  return np_utils.tensor_to_ndarray(result)


@np_utils.np_doc(np.diagonal)
def diagonal(a, offset=0, axis1=0, axis2=1):  # pylint: disable=missing-docstring
  a = asarray(a).data

  maybe_rank = a.shape.rank
  if maybe_rank is not None and offset == 0 and (
      axis1 == maybe_rank - 2 or axis1 == -2) and (axis2 == maybe_rank - 1 or
                                                   axis2 == -1):
    return np_utils.tensor_to_ndarray(array_ops.matrix_diag_part(a))

  a = moveaxis(np_utils.tensor_to_ndarray(a), (axis1, axis2), (-2, -1)).data

  a_shape = array_ops.shape(a)

  def _zeros():  # pylint: disable=missing-docstring
    return (array_ops.zeros(
        array_ops.concat([a_shape[:-1], [0]], 0), dtype=a.dtype), 0)

  # All zeros since diag_part doesn't handle all possible k (aka offset).
  # Written this way since cond will run shape inference on both branches,
  # and diag_part shape inference will fail when offset is out of bounds.
  a, offset = np_utils.cond(
      np_utils.logical_or(
          np_utils.less_equal(offset, -1 * np_utils.getitem(a_shape, -2)),
          np_utils.greater_equal(offset, np_utils.getitem(a_shape, -1)),
      ), _zeros, lambda: (a, offset))

  a = np_utils.tensor_to_ndarray(array_ops.matrix_diag_part(a, k=offset))
  return a


def diagflat(v, k=0):
  """Returns a 2-d array with flattened `v` as diagonal.

  Args:
    v: array_like of any rank. Gets flattened when setting as diagonal. Could be
      an ndarray, a Tensor or any object that can be converted to a Tensor using
      `tf.convert_to_tensor`.
    k: Position of the diagonal. Defaults to 0, the main diagonal. Positive
      values refer to diagonals shifted right, negative values refer to
      diagonals shifted left.

  Returns:
    2-d ndarray.
  """
  v = asarray(v)
  return diag(array_ops.reshape(v.data, [-1]), k)


def _promote_dtype(*arrays):
  dtype = np_utils.result_type(*arrays)
  return [asarray(a, dtype=dtype) for a in arrays]


def all(a, axis=None, keepdims=None):  # pylint: disable=redefined-builtin
  """Whether all array elements or those along an axis evaluate to true.

  Casts the array to bool type if it is not already and uses `tf.reduce_all` to
  compute the result.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can be
      converted to a Tensor using `tf.convert_to_tensor`.
    axis: Optional. Could be an int or a tuple of integers. If not specified,
      the reduction is performed over all array indices.
    keepdims: If true, retains reduced dimensions with length 1.

  Returns:
    An ndarray. Note that unlike NumPy this does not return a scalar bool if
    `axis` is None.
  """
  a = asarray(a, dtype=bool)
  return np_utils.tensor_to_ndarray(
      math_ops.reduce_all(input_tensor=a.data, axis=axis, keepdims=keepdims))


def any(a, axis=None, keepdims=None):  # pylint: disable=redefined-builtin
  """Whether any element in the entire array or in an axis evaluates to true.

  Casts the array to bool type if it is not already and uses `tf.reduce_any` to
  compute the result.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can be
      converted to a Tensor using `tf.convert_to_tensor`.
    axis: Optional. Could be an int or a tuple of integers. If not specified,
      the reduction is performed over all array indices.
    keepdims: If true, retains reduced dimensions with length 1.

  Returns:
    An ndarray. Note that unlike NumPy this does not return a scalar bool if
    `axis` is None.
  """
  a = asarray(a, dtype=bool)
  return np_utils.tensor_to_ndarray(
      math_ops.reduce_any(input_tensor=a.data, axis=axis, keepdims=keepdims))


def compress(condition, a, axis=None):
  """Compresses `a` by selecting values along `axis` with `condition` true.

  Uses `tf.boolean_mask`.

  Args:
    condition: 1-d array of bools. If `condition` is shorter than the array axis
      (or the flattened array if axis is None), it is padded with False.
    a: array_like. Could be an ndarray, a Tensor or any object that can be
      converted to a Tensor using `tf.convert_to_tensor`.
    axis: Optional. Axis along which to select elements. If None, `condition` is
      applied on flattened array.

  Returns:
    An ndarray.

  Raises:
    ValueError: if `condition` is not of rank 1.
  """
  condition = asarray(condition, dtype=bool)
  a = asarray(a)

  if condition.ndim != 1:
    raise ValueError('condition must be a 1-d array.')
  # `np.compress` treats scalars as 1-d arrays.
  if a.ndim == 0:
    a = ravel(a)

  if axis is None:
    a = ravel(a)
    axis = 0

  if axis < 0:
    axis += a.ndim

  assert axis >= 0 and axis < a.ndim

  # `tf.boolean_mask` requires the first dimensions of array and condition to
  # match. `np.compress` pads condition with False when it is shorter.
  condition_t = condition.data
  a_t = a.data
  if condition.shape[0] < a.shape[axis]:
    padding = array_ops.fill([a.shape[axis] - condition.shape[0]], False)
    condition_t = array_ops.concat([condition_t, padding], axis=0)
  return np_utils.tensor_to_ndarray(
      array_ops.boolean_mask(tensor=a_t, mask=condition_t, axis=axis))


def copy(a):
  """Returns a copy of the array."""
  return array(a, copy=True)


def _maybe_promote_to_int(a):
  if dtypes.as_dtype(a.dtype).is_integer:
    # If a is an integer type and its precision is less than that of `int`,
    # the output type will be `int`.
    output_type = np.promote_types(a.dtype, int)
    if output_type != a.dtype:
      a = asarray(a, dtype=output_type)

  return a


@np_utils.np_doc(np.cumprod)
def cumprod(a, axis=None, dtype=None):  # pylint: disable=missing-docstring
  a = asarray(a, dtype=dtype)

  if dtype is None:
    a = _maybe_promote_to_int(a)

  # If axis is None, the input is flattened.
  if axis is None:
    a = ravel(a)
    axis = 0
  elif axis < 0:
    axis += array_ops.rank(a.data)
  return np_utils.tensor_to_ndarray(math_ops.cumprod(a.data, axis))


@np_utils.np_doc(np.cumsum)
def cumsum(a, axis=None, dtype=None):  # pylint: disable=missing-docstring
  a = asarray(a, dtype=dtype)

  if dtype is None:
    a = _maybe_promote_to_int(a)

  # If axis is None, the input is flattened.
  if axis is None:
    a = ravel(a)
    axis = 0
  elif axis < 0:
    axis += array_ops.rank(a.data)
  return np_utils.tensor_to_ndarray(math_ops.cumsum(a.data, axis))


def imag(a):
  """Returns imaginary parts of all elements in `a`.

  Uses `tf.imag`.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can be
      converted to a Tensor using `tf.convert_to_tensor`.

  Returns:
    An ndarray with the same shape as `a`.
  """
  a = asarray(a)
  # TODO(srbs): np.imag returns a scalar if a is a scalar, whereas we always
  # return an ndarray.
  return np_utils.tensor_to_ndarray(math_ops.imag(a.data))


_TO_INT_ = 0
_TO_FLOAT = 1


def _reduce(tf_fn,
            a,
            axis=None,
            dtype=None,
            keepdims=None,
            promote_int=_TO_INT_,
            tf_bool_fn=None,
            preserve_bool=False):
  """A general reduction function.

  Args:
    tf_fn: the TF reduction function.
    a: the array to be reduced.
    axis: (optional) the axis along which to do the reduction. If None, all
      dimensions are reduced.
    dtype: (optional) the dtype of the result.
    keepdims: (optional) whether to keep the reduced dimension(s).
    promote_int: how to promote integer and bool inputs. There are three
      choices. (1) `_TO_INT_` always promotes them to np.int_ or np.uint; (2)
      `_TO_FLOAT` always promotes them to a float type (determined by
      dtypes.default_float_type); (3) None: don't promote.
    tf_bool_fn: (optional) the TF reduction function for bool inputs. It will
      only be used if `dtype` is explicitly set to `np.bool_` or if `a`'s dtype
      is `np.bool_` and `preserve_bool` is True.
    preserve_bool: a flag to control whether to use `tf_bool_fn` if `a`'s dtype
      is `np.bool_` (some reductions such as np.sum convert bools to integers,
      while others such as np.max preserve bools.

  Returns:
    An ndarray.
  """
  if dtype:
    dtype = np_utils.result_type(dtype)
  if keepdims is None:
    keepdims = False
  a = asarray(a, dtype=dtype)
  if ((dtype == np.bool_ or preserve_bool and a.dtype == np.bool_) and
      tf_bool_fn is not None):
    return np_utils.tensor_to_ndarray(
        tf_bool_fn(input_tensor=a.data, axis=axis, keepdims=keepdims))
  if dtype is None:
    dtype = a.dtype
    if np.issubdtype(dtype, np.integer) or dtype == np.bool_:
      if promote_int == _TO_INT_:
        # If a is an integer/bool type and whose bit width is less than np.int_,
        # numpy up-casts it to np.int_ based on the documentation at
        # https://numpy.org/doc/1.18/reference/generated/numpy.sum.html
        if dtype == np.bool_:
          is_signed = True
          width = 8  # We can use any number here that is less than 64
        else:
          is_signed = np.issubdtype(dtype, np.signedinteger)
          width = np.iinfo(dtype).bits
        # Numpy int_ and uint are defined as 'long' and 'unsigned long', so
        # should have the same bit width.
        if width < np.iinfo(np.int_).bits:
          if is_signed:
            dtype = np.int_
          else:
            dtype = np.uint
          a = a.astype(dtype)
      elif promote_int == _TO_FLOAT:
        a = a.astype(np_dtypes.default_float_type())

  return np_utils.tensor_to_ndarray(
      tf_fn(input_tensor=a.data, axis=axis, keepdims=keepdims))


@np_utils.np_doc(np.sum)
def sum(a, axis=None, dtype=None, keepdims=None):  # pylint: disable=redefined-builtin
  return _reduce(
      math_ops.reduce_sum,
      a,
      axis=axis,
      dtype=dtype,
      keepdims=keepdims,
      tf_bool_fn=math_ops.reduce_any)


@np_utils.np_doc(np.prod)
def prod(a, axis=None, dtype=None, keepdims=None):
  return _reduce(
      math_ops.reduce_prod,
      a,
      axis=axis,
      dtype=dtype,
      keepdims=keepdims,
      tf_bool_fn=math_ops.reduce_all)


@np_utils.np_doc(np.mean)
def mean(a, axis=None, dtype=None, keepdims=None):
  return _reduce(
      math_ops.reduce_mean,
      a,
      axis=axis,
      dtype=dtype,
      keepdims=keepdims,
      promote_int=_TO_FLOAT)


@np_utils.np_doc(np.amax)
def amax(a, axis=None, keepdims=None):
  return _reduce(
      math_ops.reduce_max,
      a,
      axis=axis,
      dtype=None,
      keepdims=keepdims,
      promote_int=None,
      tf_bool_fn=math_ops.reduce_any,
      preserve_bool=True)


@np_utils.np_doc(np.amin)
def amin(a, axis=None, keepdims=None):
  return _reduce(
      math_ops.reduce_min,
      a,
      axis=axis,
      dtype=None,
      keepdims=keepdims,
      promote_int=None,
      tf_bool_fn=math_ops.reduce_all,
      preserve_bool=True)


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=None):  # pylint: disable=missing-docstring
  if dtype:
    working_dtype = np_utils.result_type(a, dtype)
  else:
    working_dtype = None
  if out is not None:
    raise ValueError('Setting out is not supported.')
  if ddof != 0:
    # TF reduce_variance doesn't support ddof, so calculate it using raw ops.
    def reduce_fn(input_tensor, axis, keepdims):
      means = math_ops.reduce_mean(input_tensor, axis=axis, keepdims=True)
      centered = input_tensor - means
      if input_tensor.dtype in (dtypes.complex64, dtypes.complex128):
        centered = math_ops.cast(
            math_ops.real(centered * math_ops.conj(centered)),
            input_tensor.dtype)
      else:
        centered = math_ops.square(centered)
      squared_deviations = math_ops.reduce_sum(
          centered, axis=axis, keepdims=keepdims)

      if axis is None:
        n = array_ops.size(input_tensor)
      else:
        if axis < 0:
          axis += array_ops.rank(input_tensor)
        n = math_ops.reduce_prod(
            array_ops.gather(array_ops.shape(input_tensor), axis))
      n = math_ops.cast(n - ddof, input_tensor.dtype)

      return math_ops.cast(math_ops.divide(squared_deviations, n), dtype)
  else:
    reduce_fn = math_ops.reduce_variance

  result = _reduce(
      reduce_fn,
      a,
      axis=axis,
      dtype=working_dtype,
      keepdims=keepdims,
      promote_int=_TO_FLOAT).data
  if dtype:
    result = math_ops.cast(result, dtype)
  return np_utils.tensor_to_ndarray(result)


@np_utils.np_doc(np.std)
def std(a, axis=None, keepdims=None):  # pylint: disable=missing-function-docstring
  return _reduce(
      math_ops.reduce_std, a, axis=axis, dtype=None, keepdims=keepdims,
      promote_int=_TO_FLOAT)


@np_utils.np_doc(np.ravel)
def ravel(a):  # pylint: disable=missing-docstring
  a = asarray(a)
  if a.ndim == 1:
    return a
  return np_utils.tensor_to_ndarray(array_ops.reshape(a.data, [-1]))


setattr(np_arrays.ndarray, 'ravel', ravel)


def real(val):
  """Returns real parts of all elements in `a`.

  Uses `tf.real`.

  Args:
    val: array_like. Could be an ndarray, a Tensor or any object that can be
      converted to a Tensor using `tf.convert_to_tensor`.

  Returns:
    An ndarray with the same shape as `a`.
  """
  val = asarray(val)
  # TODO(srbs): np.real returns a scalar if val is a scalar, whereas we always
  # return an ndarray.
  return np_utils.tensor_to_ndarray(math_ops.real(val.data))


@np_utils.np_doc(np.repeat)
def repeat(a, repeats, axis=None):  # pylint: disable=missing-docstring
  a = asarray(a).data
  original_shape = a._shape_as_list()  # pylint: disable=protected-access
  # Best effort recovery of the shape.
  if original_shape is not None and None not in original_shape:
    if not original_shape:
      original_shape = (repeats,)
    else:
      repeats_np = np.ravel(np.array(repeats))
      if repeats_np.size == 1:
        repeats_np = repeats_np.item()
        if axis is None:
          original_shape = (repeats_np * np.prod(original_shape),)
        else:
          original_shape[axis] = repeats_np * original_shape[axis]
      else:
        if axis is None:
          original_shape = (repeats_np.sum(),)
        else:
          original_shape[axis] = repeats_np.sum()

  repeats = asarray(repeats).data
  result = array_ops.repeat(a, repeats, axis)
  result.set_shape(original_shape)

  return np_utils.tensor_to_ndarray(result)


@np_utils.np_doc(np.around)
def around(a, decimals=0):  # pylint: disable=missing-docstring
  a = asarray(a)
  dtype = a.dtype
  factor = math.pow(10, decimals)
  if np.issubdtype(dtype, np.inexact):
    factor = math_ops.cast(factor, dtype)
  else:
    # Use float as the working dtype when a.dtype is exact (e.g. integer),
    # because `decimals` can be negative.
    float_dtype = np_dtypes.default_float_type()
    a = a.astype(float_dtype).data
    factor = math_ops.cast(factor, float_dtype)
  a = math_ops.multiply(a, factor)
  a = math_ops.round(a)
  a = math_ops.divide(a, factor)
  return np_utils.tensor_to_ndarray(a).astype(dtype)


round_ = around
setattr(np_arrays.ndarray, '__round__', around)


@np_utils.np_doc(np.reshape)
def reshape(a, newshape, order='C'):
  """order argument can only b 'C' or 'F'."""
  if order not in {'C', 'F'}:
    raise ValueError('Unsupported order argument {}'.format(order))

  a = asarray(a)
  if isinstance(newshape, np_arrays.ndarray):
    newshape = newshape.data
  if isinstance(newshape, int):
    newshape = [newshape]

  if order == 'F':
    r = array_ops.transpose(
        array_ops.reshape(array_ops.transpose(a.data), newshape[::-1]))
  else:
    r = array_ops.reshape(a.data, newshape)

  return np_utils.tensor_to_ndarray(r)


def _reshape_method_wrapper(a, *newshape, **kwargs):
  order = kwargs.pop('order', 'C')
  if kwargs:
    raise ValueError('Unsupported arguments: {}'.format(kwargs.keys()))

  if len(newshape) == 1 and not isinstance(newshape[0], int):
    newshape = newshape[0]

  return reshape(a, newshape, order=order)


def expand_dims(a, axis):
  """Expand the shape of an array.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can be
      converted to a Tensor using `tf.convert_to_tensor`.
    axis: int. axis on which to expand the shape.

  Returns:
    An ndarray with the contents and dtype of `a` and shape expanded on axis.
  """
  a = asarray(a)
  return np_utils.tensor_to_ndarray(array_ops.expand_dims(a.data, axis=axis))


def squeeze(a, axis=None):
  """Removes single-element axes from the array.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can be
      converted to a Tensor using `tf.convert_to_tensor`.
    axis: scalar or list/tuple of ints.
  TODO(srbs): tf.squeeze throws error when axis is a Tensor eager execution is
    enabled. So we cannot allow axis to be array_like here. Fix.

  Returns:
    An ndarray.
  """
  a = asarray(a)
  return np_utils.tensor_to_ndarray(array_ops.squeeze(a, axis))


def transpose(a, axes=None):
  """Permutes dimensions of the array.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can be
      converted to a Tensor using `tf.convert_to_tensor`.
    axes: array_like. A list of ints with length rank(a) or None specifying the
      order of permutation. The i'th dimension of the output array corresponds
      to axes[i]'th dimension of the `a`. If None, the axes are reversed.

  Returns:
    An ndarray.
  """
  a = asarray(a)
  if axes is not None:
    axes = asarray(axes)
  return np_utils.tensor_to_ndarray(array_ops.transpose(a=a.data, perm=axes))


@np_utils.np_doc(np.swapaxes)
def swapaxes(a, axis1, axis2):  # pylint: disable=missing-docstring
  a = asarray(a)

  a_rank = array_ops.rank(a)
  if axis1 < 0:
    axis1 += a_rank
  if axis2 < 0:
    axis2 += a_rank

  perm = math_ops.range(a_rank)
  perm = array_ops.tensor_scatter_update(perm, [[axis1], [axis2]],
                                         [axis2, axis1])
  a = array_ops.transpose(a, perm)

  return np_utils.tensor_to_ndarray(a)


@np_utils.np_doc(np.moveaxis)
def moveaxis(a, source, destination):  # pylint: disable=missing-docstring
  """Raises ValueError if source, destination not in (-ndim(a), ndim(a))."""
  if not source and not destination:
    return a

  a = asarray(a).data

  if isinstance(source, int):
    source = (source,)
  if isinstance(destination, int):
    destination = (destination,)

  a_rank = np_utils._maybe_static(array_ops.rank(a))  # pylint: disable=protected-access

  def _correct_axis(axis, rank):
    if axis < 0:
      return axis + rank
    return axis

  source = tuple(_correct_axis(axis, a_rank) for axis in source)
  destination = tuple(_correct_axis(axis, a_rank) for axis in destination)

  if a.shape.rank is not None:
    perm = [i for i in range(a_rank) if i not in source]
    for dest, src in sorted(zip(destination, source)):
      assert dest <= len(perm)
      perm.insert(dest, src)
  else:
    r = math_ops.range(a_rank)

    def _remove_indices(a, b):
      """Remove indices (`b`) from `a`."""
      items = array_ops.unstack(sort_ops.sort(array_ops.stack(b)), num=len(b))

      i = 0
      result = []

      for item in items:
        result.append(a[i:item])
        i = item + 1

      result.append(a[i:])

      return array_ops.concat(result, 0)

    minus_sources = _remove_indices(r, source)
    minus_dest = _remove_indices(r, destination)

    perm = array_ops.scatter_nd(
        array_ops.expand_dims(minus_dest, 1), minus_sources, [a_rank])
    perm = array_ops.tensor_scatter_update(
        perm, array_ops.expand_dims(destination, 1), source)
  a = array_ops.transpose(a, perm)

  return np_utils.tensor_to_ndarray(a)


def _setitem(arr, index, value):
  """Sets the `value` at `index` in the array `arr`.

  This works by replacing the slice at `index` in the tensor with `value`.
  Since tensors are immutable, this builds a new tensor using the `tf.concat`
  op. Currently, only 0-d and 1-d indices are supported.

  Note that this may break gradients e.g.

  a = tf_np.array([1, 2, 3])
  old_a_t = a.data

  with tf.GradientTape(persistent=True) as g:
    g.watch(a.data)
    b = a * 2
    a[0] = 5
  g.gradient(b.data, [a.data])  # [None]
  g.gradient(b.data, [old_a_t])  # [[2., 2., 2.]]

  Here `d_b / d_a` is `[None]` since a.data no longer points to the same
  tensor.

  Args:
    arr: array_like.
    index: scalar or 1-d integer array.
    value: value to set at index.

  Returns:
    ndarray

  Raises:
    ValueError: if `index` is not a scalar or 1-d array.
  """
  # TODO(srbs): Figure out a solution to the gradient problem.
  arr = asarray(arr)
  index = asarray(index)
  if index.ndim == 0:
    index = ravel(index)
  elif index.ndim > 1:
    raise ValueError('index must be a scalar or a 1-d array.')
  value = asarray(value, dtype=arr.dtype)
  if arr.shape[len(index):] != value.shape:
    value = full(arr.shape[len(index):], value)
  prefix_t = arr.data[:index.data[0]]
  postfix_t = arr.data[index.data[0] + 1:]
  if len(index) == 1:
    arr._data = array_ops.concat(  # pylint: disable=protected-access
        [prefix_t, array_ops.expand_dims(value.data, 0), postfix_t], 0)
  else:
    subarray = arr[index.data[0]]
    _setitem(subarray, index[1:], value)
    arr._data = array_ops.concat(  # pylint: disable=protected-access
        [prefix_t, array_ops.expand_dims(subarray.data, 0), postfix_t], 0)


setattr(np_arrays.ndarray, 'transpose', transpose)
setattr(np_arrays.ndarray, 'reshape', _reshape_method_wrapper)
setattr(np_arrays.ndarray, '__setitem__', _setitem)


def pad(ary, pad_width, mode, constant_values=0):
  """Pads an array.

  Args:
    ary: array_like of rank N. Input array.
    pad_width: {sequence, array_like, int}. Number of values padded to the edges
      of each axis. ((before_1, after_1), ... (before_N, after_N)) unique pad
      widths for each axis. ((before, after),) yields same before and after pad
      for each axis. (pad,) or int is a shortcut for before = after = pad width
      for all axes.
    mode: string. One of the following string values: 'constant' Pads with a
      constant value. 'reflect' Pads with the reflection of the vector mirrored
      on the first and last values of the vector along each axis. 'symmetric'
      Pads with the reflection of the vector mirrored along the edge of the
      array.
      **NOTE**: The supported list of `mode` does not match that of numpy's.
    constant_values: scalar with same dtype as `array`. Used in 'constant' mode
      as the pad value.  Default is 0.

  Returns:
    An ndarray padded array of rank equal to `array` with shape increased
    according to `pad_width`.

  Raises:
    ValueError if `mode` is not supported.
  """
  if not (mode == 'constant' or mode == 'reflect' or mode == 'symmetric'):
    raise ValueError('Unsupported padding mode: ' + mode)
  mode = mode.upper()
  ary = asarray(ary)
  pad_width = asarray(pad_width, dtype=dtypes.int32)
  return np_utils.tensor_to_ndarray(
      array_ops.pad(
          tensor=ary.data,
          paddings=pad_width.data,
          mode=mode,
          constant_values=constant_values))


@np_utils.np_doc(np.take)
def take(a, indices, axis=None, out=None, mode='clip'):
  """out argument is not supported, and default mode is clip."""
  if out is not None:
    raise ValueError('out argument is not supported in take.')

  if mode not in {'raise', 'clip', 'wrap'}:
    raise ValueError("Invalid mode '{}' for take".format(mode))

  a = asarray(a).data
  indices = asarray(indices).data

  if axis is None:
    a = array_ops.reshape(a, [-1])
    axis = 0

  axis_size = array_ops.shape(a, out_type=indices.dtype)[axis]
  if mode == 'clip':
    indices = clip_ops.clip_by_value(indices, 0, axis_size - 1)
  elif mode == 'wrap':
    indices = math_ops.floormod(indices, axis_size)
  else:
    raise ValueError("The 'raise' mode to take is not supported.")

  return np_utils.tensor_to_ndarray(array_ops.gather(a, indices, axis=axis))


@np_utils.np_doc_only(np.where)
def where(condition, x=None, y=None):
  """Raises ValueError if exactly one of x or y is not None."""
  condition = asarray(condition, dtype=np.bool_)
  if x is None and y is None:
    return nonzero(condition)
  elif x is not None and y is not None:
    x, y = _promote_dtype(x, y)
    return np_utils.tensor_to_ndarray(
        array_ops.where_v2(condition.data, x.data, y.data))
  raise ValueError('Both x and y must be ndarrays, or both must be None.')


@np_utils.np_doc(np.select)
def select(condlist, choicelist, default=0):  # pylint: disable=missing-docstring
  if len(condlist) != len(choicelist):
    msg = 'condlist must have length equal to choicelist ({} vs {})'
    raise ValueError(msg.format(len(condlist), len(choicelist)))
  if not condlist:
    raise ValueError('condlist must be non-empty')
  choices = _promote_dtype(default, *choicelist)
  choicelist = choices[1:]
  output = choices[0]
  # The traversal is in reverse order so we can return the first value in
  # choicelist where condlist is True.
  for cond, choice in zip(condlist[::-1], choicelist[::-1]):
    output = where(cond, choice, output)
  return output


def shape(a):
  """Return the shape of an array.

  Args:
    a: array_like. Input array.

  Returns:
    Tuple of ints.
  """
  a = asarray(a)
  return a.shape


def ndim(a):
  a = asarray(a)
  return a.ndim


def isscalar(a):
  return ndim(a) == 0


def _boundaries_to_sizes(a, boundaries, axis):
  """Converting boundaries of splits to sizes of splits.

  Args:
    a: the array to be split.
    boundaries: the boundaries, as in np.split.
    axis: the axis along which to split.

  Returns:
    A list of sizes of the splits, as in tf.split.
  """
  if axis >= len(a.shape):
    raise ValueError('axis %s is out of bound for shape %s' % (axis, a.shape))
  total_size = a.shape[axis]
  sizes = []
  sizes_sum = 0
  prev = 0
  for i, b in enumerate(boundaries):
    size = b - prev
    if size < 0:
      raise ValueError('The %s-th boundary %s is smaller than the previous '
                       'boundary %s' % (i, b, prev))
    size = min(size, max(0, total_size - sizes_sum))
    sizes.append(size)
    sizes_sum += size
    prev = b
  sizes.append(max(0, total_size - sizes_sum))
  return sizes


@np_utils.np_doc(np.split)
def split(ary, indices_or_sections, axis=0):
  ary = asarray(ary)
  if not isinstance(indices_or_sections, six.integer_types):
    indices_or_sections = _boundaries_to_sizes(ary, indices_or_sections, axis)
  result = array_ops.split(ary.data, indices_or_sections, axis=axis)
  return [np_utils.tensor_to_ndarray(a) for a in result]


def _split_on_axis(np_fun, axis):

  @np_utils.np_doc(np_fun)
  def f(ary, indices_or_sections):
    return split(ary, indices_or_sections, axis=axis)

  return f


vsplit = _split_on_axis(np.vsplit, axis=0)
hsplit = _split_on_axis(np.hsplit, axis=1)
dsplit = _split_on_axis(np.dsplit, axis=2)


@np_utils.np_doc(np.broadcast_to)
def broadcast_to(array, shape):  # pylint: disable=redefined-outer-name
  return full(shape, array)


@np_utils.np_doc(np.stack)
def stack(arrays, axis=0):
  arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a.data if isinstance(a, np_arrays.ndarray) else a for a in arrays
  ]
  return asarray(array_ops.stack(unwrapped_arrays, axis))


@np_utils.np_doc(np.hstack)
def hstack(tup):
  arrays = [atleast_1d(a) for a in tup]
  arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a.data if isinstance(a, np_arrays.ndarray) else a for a in arrays
  ]
  rank = array_ops.rank(unwrapped_arrays[0])
  return np_utils.cond(
      math_ops.equal(rank,
                     1), lambda: array_ops.concat(unwrapped_arrays, axis=0),
      lambda: array_ops.concat(unwrapped_arrays, axis=1))


@np_utils.np_doc(np.vstack)
def vstack(tup):
  arrays = [atleast_2d(a) for a in tup]
  arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a.data if isinstance(a, np_arrays.ndarray) else a for a in arrays
  ]
  return array_ops.concat(unwrapped_arrays, axis=0)


@np_utils.np_doc(np.dstack)
def dstack(tup):
  arrays = [atleast_3d(a) for a in tup]
  arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a.data if isinstance(a, np_arrays.ndarray) else a for a in arrays
  ]
  return array_ops.concat(unwrapped_arrays, axis=2)


def _pad_left_to(n, old_shape):
  old_shape = asarray(old_shape, dtype=np.int32).data
  new_shape = array_ops.pad(
      old_shape, [[math_ops.maximum(n - array_ops.size(old_shape), 0), 0]],
      constant_values=1)
  return asarray(new_shape)


def _atleast_nd(n, new_shape, *arys):
  """Reshape arrays to be at least `n`-dimensional.

  Args:
    n: The minimal rank.
    new_shape: a function that takes `n` and the old shape and returns the
      desired new shape.
    *arys: ndarray(s) to be reshaped.

  Returns:
    The reshaped array(s).
  """

  def f(x):
    # pylint: disable=g-long-lambda
    x = asarray(x)
    return asarray(
        np_utils.cond(
            np_utils.greater(n, array_ops.rank(x)),
            lambda: reshape(x, new_shape(n, array_ops.shape(x.data))).data,
            lambda: x.data))

  arys = list(map(f, arys))
  if len(arys) == 1:
    return arys[0]
  else:
    return arys


@np_utils.np_doc(np.atleast_1d)
def atleast_1d(*arys):
  return _atleast_nd(1, _pad_left_to, *arys)


@np_utils.np_doc(np.atleast_2d)
def atleast_2d(*arys):
  return _atleast_nd(2, _pad_left_to, *arys)


@np_utils.np_doc(np.atleast_3d)
def atleast_3d(*arys):  # pylint: disable=missing-docstring

  def new_shape(_, old_shape):
    # pylint: disable=g-long-lambda
    ndim_ = array_ops.size(old_shape)
    return np_utils.cond(
        math_ops.equal(ndim_, 0),
        lambda: constant_op.constant([1, 1, 1], dtype=dtypes.int32),
        lambda: np_utils.cond(
            math_ops.equal(ndim_, 1), lambda: array_ops.pad(
                old_shape, [[1, 1]], constant_values=1), lambda: array_ops.pad(
                    old_shape, [[0, 1]], constant_values=1)))

  return _atleast_nd(3, new_shape, *arys)


@np_utils.np_doc(np.nonzero)
def nonzero(a):
  a = atleast_1d(a).data
  if a.shape.rank is None:
    raise ValueError("The rank of `a` is unknown, so we can't decide how many "
                     'arrays to return.')
  return nest.map_structure(
      np_arrays.tensor_to_ndarray,
      array_ops.unstack(
          array_ops.where_v2(math_ops.cast(a, dtypes.bool)),
          a.shape.rank,
          axis=1))


@np_utils.np_doc(np.diag_indices)
def diag_indices(n, ndim=2):  # pylint: disable=missing-docstring,redefined-outer-name
  if n < 0:
    raise ValueError(
        'n argument to diag_indices must be nonnegative, got {}'.format(n))
  if ndim < 0:
    raise ValueError(
        'ndim argument to diag_indices must be nonnegative, got {}'.format(
            ndim))

  return (math_ops.range(n),) * ndim


@np_utils.np_doc(np.tri)
def tri(N, M=None, k=0, dtype=None):  # pylint: disable=invalid-name,missing-docstring
  M = M if M is not None else N
  if dtype is not None:
    dtype = np_utils.result_type(dtype)
  else:
    dtype = np_dtypes.default_float_type()

  if k < 0:
    lower = -k - 1
    if lower > N:
      r = array_ops.zeros([N, M], dtype)
    else:
      # Keep as tf bool, since we create an upper triangular matrix and invert
      # it.
      o = array_ops.ones([N, M], dtype=dtypes.bool)
      r = math_ops.cast(
          math_ops.logical_not(array_ops.matrix_band_part(o, lower, -1)), dtype)
  else:
    o = array_ops.ones([N, M], dtype)
    if k > M:
      r = o
    else:
      r = array_ops.matrix_band_part(o, -1, k)
  return np_utils.tensor_to_ndarray(r)


@np_utils.np_doc(np.tril)
def tril(m, k=0):  # pylint: disable=missing-docstring
  m = asarray(m).data
  m_shape = m.shape.as_list()

  if len(m_shape) < 2:
    raise ValueError('Argument to tril must have rank at least 2')

  if m_shape[-1] is None or m_shape[-2] is None:
    raise ValueError('Currently, the last two dimensions of the input array '
                     'need to be known.')

  z = constant_op.constant(0, m.dtype)

  mask = tri(*m_shape[-2:], k=k, dtype=bool)
  return np_utils.tensor_to_ndarray(
      array_ops.where_v2(
          array_ops.broadcast_to(mask, array_ops.shape(m)), m, z))


@np_utils.np_doc(np.triu)
def triu(m, k=0):  # pylint: disable=missing-docstring
  m = asarray(m).data
  m_shape = m.shape.as_list()

  if len(m_shape) < 2:
    raise ValueError('Argument to triu must have rank at least 2')

  if m_shape[-1] is None or m_shape[-2] is None:
    raise ValueError('Currently, the last two dimensions of the input array '
                     'need to be known.')

  z = constant_op.constant(0, m.dtype)

  mask = tri(*m_shape[-2:], k=k - 1, dtype=bool)
  return np_utils.tensor_to_ndarray(
      array_ops.where_v2(
          array_ops.broadcast_to(mask, array_ops.shape(m)), z, m))


@np_utils.np_doc(np.flip)
def flip(m, axis=None):  # pylint: disable=missing-docstring
  m = asarray(m).data

  if axis is None:
    return np_utils.tensor_to_ndarray(
        array_ops.reverse(m, math_ops.range(array_ops.rank(m))))

  axis = np_utils._canonicalize_axis(axis, array_ops.rank(m))  # pylint: disable=protected-access

  return np_utils.tensor_to_ndarray(array_ops.reverse(m, [axis]))


@np_utils.np_doc(np.flipud)
def flipud(m):  # pylint: disable=missing-docstring
  return flip(m, 0)


@np_utils.np_doc(np.fliplr)
def fliplr(m):  # pylint: disable=missing-docstring
  return flip(m, 1)


@np_utils.np_doc(np.roll)
def roll(a, shift, axis=None):  # pylint: disable=missing-docstring
  a = asarray(a).data

  if axis is not None:
    return np_utils.tensor_to_ndarray(manip_ops.roll(a, shift, axis))

  # If axis is None, the roll happens as a 1-d tensor.
  original_shape = array_ops.shape(a)
  a = manip_ops.roll(array_ops.reshape(a, [-1]), shift, 0)
  return np_utils.tensor_to_ndarray(array_ops.reshape(a, original_shape))


@np_utils.np_doc(np.rot90)
def rot90(m, k=1, axes=(0, 1)):  # pylint: disable=missing-docstring
  m_rank = array_ops.rank(m)
  ax1, ax2 = np_utils._canonicalize_axes(axes, m_rank)  # pylint: disable=protected-access

  k = k % 4
  if k == 0:
    return m
  elif k == 2:
    return flip(flip(m, ax1), ax2)
  else:
    perm = math_ops.range(m_rank)
    perm = array_ops.tensor_scatter_update(perm, [[ax1], [ax2]], [ax2, ax1])

    if k == 1:
      return transpose(flip(m, ax2), perm)
    else:
      return flip(transpose(m, perm), ax2)


@np_utils.np_doc(np.vander)
def vander(x, N=None, increasing=False):  # pylint: disable=missing-docstring,invalid-name
  x = asarray(x).data

  x_shape = array_ops.shape(x)
  N = N or x_shape[0]

  N_temp = np_utils.get_static_value(N)  # pylint: disable=invalid-name
  if N_temp is not None:
    N = N_temp
    if N < 0:
      raise ValueError('N must be nonnegative')
  else:
    control_flow_ops.Assert(N >= 0, [N])

  rank = array_ops.rank(x)
  rank_temp = np_utils.get_static_value(rank)
  if rank_temp is not None:
    rank = rank_temp
    if rank != 1:
      raise ValueError('x must be a one-dimensional array')
  else:
    control_flow_ops.Assert(math_ops.equal(rank, 1), [rank])

  if increasing:
    start = 0
    limit = N
    delta = 1
  else:
    start = N - 1
    limit = -1
    delta = -1

  x = array_ops.expand_dims(x, -1)
  return np_utils.tensor_to_ndarray(
      math_ops.pow(
          x, math_ops.cast(math_ops.range(start, limit, delta), dtype=x.dtype)))


@np_utils.np_doc(np.ix_)
def ix_(*args):  # pylint: disable=missing-docstring
  n = len(args)
  output = []
  for i, a in enumerate(args):
    a = asarray(a).data
    a_rank = array_ops.rank(a)
    a_rank_temp = np_utils.get_static_value(a_rank)
    if a_rank_temp is not None:
      a_rank = a_rank_temp
      if a_rank != 1:
        raise ValueError('Arguments must be 1-d, got arg {} of rank {}'.format(
            i, a_rank))
    else:
      control_flow_ops.Assert(math_ops.equal(a_rank, 1), [a_rank])

    new_shape = [1] * n
    new_shape[i] = -1
    dtype = a.dtype
    if dtype == dtypes.bool:
      output.append(
          np_utils.tensor_to_ndarray(
              array_ops.reshape(nonzero(a)[0].data, new_shape)))
    elif dtype.is_integer:
      output.append(np_utils.tensor_to_ndarray(array_ops.reshape(a, new_shape)))
    else:
      raise ValueError(
          'Only integer and bool dtypes are supported, got {}'.format(dtype))

  return output
