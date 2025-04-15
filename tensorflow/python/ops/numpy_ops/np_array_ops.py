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

import builtins
import enum
import functools
import math
import numbers

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export


newaxis = np.newaxis
tf_export.tf_export('experimental.numpy.newaxis', v1=[]).export_constant(
    __name__, 'newaxis'
)


@tf_export.tf_export('experimental.numpy.empty', v1=[])
@np_utils.np_doc('empty')
def empty(shape, dtype=float):  # pylint: disable=redefined-outer-name
  return zeros(shape, dtype)


@tf_export.tf_export('experimental.numpy.empty_like', v1=[])
@np_utils.np_doc('empty_like')
def empty_like(a, dtype=None):
  return zeros_like(a, dtype)


@tf_export.tf_export('experimental.numpy.zeros', v1=[])
@np_utils.np_doc('zeros')
def zeros(shape, dtype=float):  # pylint: disable=redefined-outer-name
  dtype = (
      np_utils.result_type(dtype) if dtype else np_dtypes.default_float_type()
  )
  return array_ops.zeros(shape, dtype=dtype)


@tf_export.tf_export('experimental.numpy.zeros_like', v1=[])
@np_utils.np_doc('zeros_like')
def zeros_like(a, dtype=None):  # pylint: disable=missing-docstring
  dtype = np_utils.result_type_unary(a, dtype)

  dtype = dtypes.as_dtype(dtype)  # Work around b/149877262
  return array_ops.zeros_like(a, dtype)


@tf_export.tf_export('experimental.numpy.ones', v1=[])
@np_utils.np_doc('ones')
def ones(shape, dtype=float):  # pylint: disable=redefined-outer-name
  if dtype:
    dtype = np_utils.result_type(dtype)
  return array_ops.ones(shape, dtype=dtype)


@tf_export.tf_export('experimental.numpy.ones_like', v1=[])
@np_utils.np_doc('ones_like')
def ones_like(a, dtype=None):
  dtype = np_utils.result_type_unary(a, dtype)
  return array_ops.ones_like(a, dtype)


@tf_export.tf_export('experimental.numpy.eye', v1=[])
@np_utils.np_doc('eye')
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
    return linalg_ops.eye(N, M, dtype=dtype)
  # We need the precise length, otherwise tf.linalg.diag will raise an error
  diag_len = builtins.min(N, M)
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
  return array_ops.matrix_diag(diagonal=diagonal_, num_rows=N, num_cols=M, k=k)


@tf_export.tf_export('experimental.numpy.identity', v1=[])
@np_utils.np_doc('identity')
def identity(n, dtype=float):
  return eye(N=n, M=n, dtype=dtype)


@tf_export.tf_export('experimental.numpy.full', v1=[])
@np_utils.np_doc('full')
def full(shape, fill_value, dtype=None):  # pylint: disable=redefined-outer-name
  if not isinstance(shape, np_arrays.ndarray):
    shape = asarray(np_arrays.convert_to_tensor(shape, dtype_hint=np.int32))
  shape = atleast_1d(shape)
  fill_value = asarray(fill_value, dtype=dtype)
  return array_ops.broadcast_to(fill_value, shape)


# Using doc only here since np full_like signature doesn't seem to have the
# shape argument (even though it exists in the documentation online).
@tf_export.tf_export('experimental.numpy.full_like', v1=[])
@np_utils.np_doc_only('full_like')
def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None):  # pylint: disable=missing-docstring,redefined-outer-name
  """order, subok and shape arguments mustn't be changed."""
  if order != 'K':
    raise ValueError('Non-standard orders are not supported.')
  if not subok:
    raise ValueError('subok being False is not supported.')
  if shape:
    raise ValueError('Overriding the shape is not supported.')

  a = asarray(a)
  dtype = dtype or np_utils.result_type(a)
  fill_value = asarray(fill_value, dtype=dtype)
  return array_ops.broadcast_to(fill_value, array_ops.shape(a))


def _array_internal(val, dtype=None, copy=True, ndmin=0):  # pylint: disable=redefined-outer-name
  """Main implementation of np.array()."""
  result_t = val

  if not isinstance(result_t, tensor_lib.Tensor):
    dtype = np_utils.result_type_unary(result_t, dtype)
    # We can't call `convert_to_tensor(result_t, dtype=dtype)` here because
    # convert_to_tensor doesn't allow incompatible arguments such as (5.5, int)
    # while np.array allows them. We need to convert-then-cast.

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

  if copy:
    result_t = array_ops.identity(result_t)

  max_ndmin = 32
  if ndmin > max_ndmin:
    raise ValueError(
        f'ndmin bigger than allowable number of dimensions: {max_ndmin}.'
    )

  if ndmin == 0:
    return result_t

  ndims = array_ops.rank(result_t)

  def true_fn():
    old_shape = array_ops.shape(result_t)
    new_shape = array_ops.concat(
        [array_ops.ones(ndmin - ndims, dtypes.int32), old_shape], axis=0
    )
    return array_ops.reshape(result_t, new_shape)

  result_t = np_utils.cond(
      np_utils.greater(ndmin, ndims), true_fn, lambda: result_t
  )
  return result_t


# TODO(wangpeng): investigate whether we can make `copy` default to False.
# pylint: disable=g-short-docstring-punctuation,g-no-space-after-docstring-summary,g-doc-return-or-yield,g-doc-args
@tf_export.tf_export('experimental.numpy.array', v1=[])
@np_utils.np_doc_only('array')
def array(val, dtype=None, copy=True, ndmin=0):  # pylint: disable=redefined-outer-name
  """Since Tensors are immutable, a copy is made only if val is placed on a

  different device than the current one. Even if `copy` is False, a new Tensor
  may need to be built to satisfy `dtype` and `ndim`. This is used only if `val`
  is an ndarray or a Tensor.
  """  # pylint:disable=g-docstring-missing-newline
  if dtype:
    dtype = np_utils.result_type(dtype)
  return _array_internal(val, dtype, copy, ndmin)


# pylint: enable=g-short-docstring-punctuation,g-no-space-after-docstring-summary,g-doc-return-or-yield,g-doc-args


@tf_export.tf_export('experimental.numpy.asarray', v1=[])
@np_utils.np_doc('asarray')
def asarray(a, dtype=None):
  if dtype:
    dtype = np_utils.result_type(dtype)
  if isinstance(a, np_arrays.ndarray) and (
      not dtype or dtype == a.dtype.as_numpy_dtype
  ):
    return a
  return array(a, dtype, copy=False)


@tf_export.tf_export('experimental.numpy.asanyarray', v1=[])
@np_utils.np_doc('asanyarray')
def asanyarray(a, dtype=None):
  return asarray(a, dtype)


@tf_export.tf_export('experimental.numpy.ascontiguousarray', v1=[])
@np_utils.np_doc('ascontiguousarray')
def ascontiguousarray(a, dtype=None):
  return array(a, dtype, ndmin=1)


# Numerical ranges.
@tf_export.tf_export('experimental.numpy.arange', v1=[])
@np_utils.np_doc('arange')
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
  if step > 0 and (
      (stop is not None and start > stop) or (stop is None and start < 0)
  ):
    return array([], dtype=dtype)
  if step < 0 and (
      (stop is not None and start < stop) or (stop is None and start > 0)
  ):
    return array([], dtype=dtype)
  # TODO(srbs): There are some bugs when start or stop is float type and dtype
  # is integer type.
  return math_ops.cast(
      math_ops.range(start, limit=stop, delta=step), dtype=dtype
  )


# Building matrices.
@tf_export.tf_export('experimental.numpy.diag', v1=[])
@np_utils.np_doc('diag')
def diag(v, k=0):  # pylint: disable=missing-docstring
  """Raises an error if input is not 1- or 2-d."""
  v = asarray(v)
  v_rank = array_ops.rank(v)

  v.shape.with_rank_at_most(2)

  # TODO(nareshmodi): Consider a np_utils.Assert version that will fail during
  # tracing time if the shape is known.
  control_flow_assert.Assert(
      np_utils.logical_or(math_ops.equal(v_rank, 1), math_ops.equal(v_rank, 2)),
      [v_rank],
  )

  def _diag(v, k):
    return np_utils.cond(
        math_ops.equal(array_ops.size(v), 0),
        lambda: array_ops.zeros([abs(k), abs(k)], dtype=v.dtype),
        lambda: array_ops.matrix_diag(v, k=k),
    )

  def _diag_part(v, k):
    v_shape = array_ops.shape(v)
    v, k = np_utils.cond(
        np_utils.logical_or(
            np_utils.less_equal(k, -1 * np_utils.getitem(v_shape, 0)),
            np_utils.greater_equal(k, np_utils.getitem(v_shape, 1)),
        ),
        lambda: (array_ops.zeros([0, 0], dtype=v.dtype), 0),
        lambda: (v, k),
    )
    result = array_ops.matrix_diag_part(v, k=k)
    return result

  result = np_utils.cond(
      math_ops.equal(v_rank, 1), lambda: _diag(v, k), lambda: _diag_part(v, k)
  )
  return result


@tf_export.tf_export('experimental.numpy.diagonal', v1=[])
@np_utils.np_doc('diagonal')
def diagonal(a, offset=0, axis1=0, axis2=1):  # pylint: disable=missing-docstring
  a = asarray(a)

  maybe_rank = a.shape.rank
  if (
      maybe_rank is not None
      and offset == 0
      and (axis1 == maybe_rank - 2 or axis1 == -2)
      and (axis2 == maybe_rank - 1 or axis2 == -1)
  ):
    return array_ops.matrix_diag_part(a)

  a = moveaxis(a, (axis1, axis2), (-2, -1))

  a_shape = array_ops.shape(a)

  def _zeros():  # pylint: disable=missing-docstring
    return (
        array_ops.zeros(
            array_ops.concat([a_shape[:-1], [0]], 0), dtype=a.dtype
        ),
        0,
    )

  # All zeros since diag_part doesn't handle all possible k (aka offset).
  # Written this way since cond will run shape inference on both branches,
  # and diag_part shape inference will fail when offset is out of bounds.
  a, offset = np_utils.cond(
      np_utils.logical_or(
          np_utils.less_equal(offset, -1 * np_utils.getitem(a_shape, -2)),
          np_utils.greater_equal(offset, np_utils.getitem(a_shape, -1)),
      ),
      _zeros,
      lambda: (a, offset),
  )

  a = array_ops.matrix_diag_part(a, k=offset)
  return a


@tf_export.tf_export('experimental.numpy.diagflat', v1=[])
@np_utils.np_doc('diagflat')
def diagflat(v, k=0):
  v = asarray(v)
  return diag(array_ops.reshape(v, [-1]), k)


def _promote_dtype(*arrays):
  dtype = np_utils.result_type(*arrays)

  def _fast_asarray(a):
    if isinstance(a, np_arrays.ndarray) and dtype == a.dtype.as_numpy_dtype:
      return a
    return _array_internal(a, dtype=dtype, copy=False)

  return [_fast_asarray(a) for a in arrays]


def _promote_dtype_binary(t1, t2):
  dtype = np_utils._result_type_binary(t1, t2)  # pylint: disable=protected-access
  if not (
      isinstance(t1, np_arrays.ndarray) and dtype == t1.dtype.as_numpy_dtype
  ):
    t1 = _array_internal(t1, dtype=dtype, copy=False)
  if not (
      isinstance(t2, np_arrays.ndarray) and dtype == t2.dtype.as_numpy_dtype
  ):
    t2 = _array_internal(t2, dtype=dtype, copy=False)
  return t1, t2


@tf_export.tf_export('experimental.numpy.all', v1=[])
@np_utils.np_doc('all')
def all(a, axis=None, keepdims=None):  # pylint: disable=redefined-builtin
  a = asarray(a, dtype=bool)
  return math_ops.reduce_all(input_tensor=a, axis=axis, keepdims=keepdims)


@tf_export.tf_export('experimental.numpy.any', v1=[])
@np_utils.np_doc('any')
def any(a, axis=None, keepdims=None):  # pylint: disable=redefined-builtin
  a = asarray(a, dtype=bool)
  return math_ops.reduce_any(input_tensor=a, axis=axis, keepdims=keepdims)


@tf_export.tf_export('experimental.numpy.compress', v1=[])
@np_utils.np_doc('compress')
def compress(condition, a, axis=None):  # pylint: disable=redefined-outer-name,missing-function-docstring
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
  condition_t = condition
  a_t = a
  if condition.shape[0] < a.shape[axis]:
    padding = array_ops.fill([a.shape[axis] - condition.shape[0]], False)
    condition_t = array_ops.concat([condition_t, padding], axis=0)
  return array_ops.boolean_mask(tensor=a_t, mask=condition_t, axis=axis)


@tf_export.tf_export('experimental.numpy.copy', v1=[])
@np_utils.np_doc('copy')
def copy(a):
  return array(a, copy=True)


def _maybe_promote_to_int(a):
  if dtypes.as_dtype(a.dtype).is_integer:
    # If a is an integer type and its precision is less than that of `int`,
    # the output type will be `int`.
    a_numpy_dtype = a.dtype.as_numpy_dtype
    output_type = np.promote_types(a_numpy_dtype, int)
    if output_type != a_numpy_dtype:
      a = asarray(a, dtype=output_type)

  return a


@tf_export.tf_export('experimental.numpy.cumprod', v1=[])
@np_utils.np_doc('cumprod')
def cumprod(a, axis=None, dtype=None):  # pylint: disable=missing-docstring
  a = asarray(a, dtype=dtype)

  if dtype is None:
    a = _maybe_promote_to_int(a)

  # If axis is None, the input is flattened.
  if axis is None:
    a = ravel(a)
    axis = 0
  elif axis < 0:
    axis += array_ops.rank(a)
  return math_ops.cumprod(a, axis)


@tf_export.tf_export('experimental.numpy.cumsum', v1=[])
@np_utils.np_doc('cumsum')
def cumsum(a, axis=None, dtype=None):  # pylint: disable=missing-docstring
  a = asarray(a, dtype=dtype)

  if dtype is None:
    a = _maybe_promote_to_int(a)

  # If axis is None, the input is flattened.
  if axis is None:
    a = ravel(a)
    axis = 0
  elif axis < 0:
    axis += array_ops.rank(a)
  return math_ops.cumsum(a, axis)


@tf_export.tf_export('experimental.numpy.imag', v1=[])
@np_utils.np_doc('imag')
def imag(val):
  val = asarray(val)
  # TODO(srbs): np.imag returns a scalar if `val` is a scalar, whereas we always
  # return an ndarray.
  return math_ops.imag(val)


_TO_INT_ = 0
_TO_FLOAT = 1


def _reduce(
    tf_fn,
    a,
    axis=None,
    dtype=None,
    keepdims=None,
    promote_int=_TO_INT_,
    tf_bool_fn=None,
    preserve_bool=False,
):
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
  if (
      dtype == np.bool_ or preserve_bool and a.dtype == np.bool_
  ) and tf_bool_fn is not None:
    return tf_bool_fn(input_tensor=a, axis=axis, keepdims=keepdims)
  if dtype is None:
    dtype = a.dtype.as_numpy_dtype
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
        if ops.is_auto_dtype_conversion_enabled():
          # We default to 32 bits when using auto dtype conversion semantics.
          if width < np.iinfo(np.int32).bits:
            if is_signed:
              dtype = np.int32
            else:
              dtype = np.uint32
        else:
          if width < np.iinfo(np.int_).bits:
            if is_signed:
              dtype = np.int_
            else:
              dtype = np.uint
        a = math_ops.cast(a, dtype)
      elif promote_int == _TO_FLOAT:
        # Use a default float type.
        a = math_ops.cast(a, np_utils.result_type(float))

  if isinstance(axis, tensor_lib.Tensor) and axis.dtype not in (
      dtypes.int32,
      dtypes.int64,
  ):
    axis = math_ops.cast(axis, dtypes.int64)

  return tf_fn(input_tensor=a, axis=axis, keepdims=keepdims)


# TODO (DarrenZhang01): Add `axis` support to the `size` API.
@tf_export.tf_export('experimental.numpy.size', v1=[])
@np_utils.np_doc('size')
def size(x, axis=None):  # pylint: disable=missing-docstring
  if axis is not None:
    raise NotImplementedError(
        'axis argument is not supported in the current `np.size` implementation'
    )
  if isinstance(x, (int, float, np.int32, np.int64, np.float32, np.float64)):
    return 1
  x = asarray(x)
  if x.shape.is_fully_defined():
    return np.prod(x.shape.as_list(), dtype=int)
  else:
    return array_ops.size_v2(x)


@tf_export.tf_export('experimental.numpy.sum', v1=[])
@np_utils.np_doc('sum')
def sum(a, axis=None, dtype=None, keepdims=None):  # pylint: disable=redefined-builtin
  return _reduce(
      math_ops.reduce_sum,
      a,
      axis=axis,
      dtype=dtype,
      keepdims=keepdims,
      tf_bool_fn=math_ops.reduce_any,
  )


@tf_export.tf_export('experimental.numpy.prod', v1=[])
@np_utils.np_doc('prod')
def prod(a, axis=None, dtype=None, keepdims=None):
  return _reduce(
      math_ops.reduce_prod,
      a,
      axis=axis,
      dtype=dtype,
      keepdims=keepdims,
      tf_bool_fn=math_ops.reduce_all,
  )


@tf_export.tf_export('experimental.numpy.mean', v1=[])
@np_utils.np_doc('mean', unsupported_params=['out'])
def mean(a, axis=None, dtype=None, out=None, keepdims=None):
  if out is not None:
    raise ValueError('Setting out is not supported.')
  return _reduce(
      math_ops.reduce_mean,
      a,
      axis=axis,
      dtype=dtype,
      keepdims=keepdims,
      promote_int=_TO_FLOAT,
  )


@tf_export.tf_export('experimental.numpy.amax', v1=[])
@np_utils.np_doc('amax', unsupported_params=['out'])
def amax(a, axis=None, out=None, keepdims=None):
  if out is not None:
    raise ValueError('Setting out is not supported.')
  return _reduce(
      math_ops.reduce_max,
      a,
      axis=axis,
      dtype=None,
      keepdims=keepdims,
      promote_int=None,
      tf_bool_fn=math_ops.reduce_any,
      preserve_bool=True,
  )


@tf_export.tf_export('experimental.numpy.amin', v1=[])
@np_utils.np_doc('amin', unsupported_params=['out'])
def amin(a, axis=None, out=None, keepdims=None):
  if out is not None:
    raise ValueError('Setting out is not supported.')
  return _reduce(
      math_ops.reduce_min,
      a,
      axis=axis,
      dtype=None,
      keepdims=keepdims,
      promote_int=None,
      tf_bool_fn=math_ops.reduce_all,
      preserve_bool=True,
  )


@tf_export.tf_export('experimental.numpy.var', v1=[])
@np_utils.np_doc('var')
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
            input_tensor.dtype,
        )
      else:
        centered = math_ops.square(centered)
      squared_deviations = math_ops.reduce_sum(
          centered, axis=axis, keepdims=keepdims
      )

      if axis is None:
        n = array_ops.size(input_tensor)
      else:
        if axis < 0:
          axis += array_ops.rank(input_tensor)
        n = math_ops.reduce_prod(
            array_ops.gather(array_ops.shape(input_tensor), axis)
        )
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
      promote_int=_TO_FLOAT,
  )
  if dtype:
    result = math_ops.cast(result, dtype)
  return result


@tf_export.tf_export('experimental.numpy.std', v1=[])
@np_utils.np_doc('std')
def std(a, axis=None, keepdims=None):  # pylint: disable=missing-function-docstring
  return _reduce(
      math_ops.reduce_std,
      a,
      axis=axis,
      dtype=None,
      keepdims=keepdims,
      promote_int=_TO_FLOAT,
  )


@tf_export.tf_export('experimental.numpy.ravel', v1=[])
@np_utils.np_doc('ravel')
def ravel(a):  # pylint: disable=missing-docstring
  a = asarray(a)
  return array_ops.reshape(a, [-1])


@tf_export.tf_export('experimental.numpy.real', v1=[])
@np_utils.np_doc('real')
def real(val):
  val = asarray(val)
  # TODO(srbs): np.real returns a scalar if val is a scalar, whereas we always
  # return an ndarray.
  return math_ops.real(val)


@tf_export.tf_export('experimental.numpy.repeat', v1=[])
@np_utils.np_doc('repeat')
def repeat(a, repeats, axis=None):  # pylint: disable=missing-docstring
  a = asarray(a)
  original_shape = a._shape_as_list()  # pylint: disable=protected-access
  # Best effort recovery of the shape.
  known_shape = original_shape is not None and None not in original_shape
  if known_shape:
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

  repeats = asarray(repeats)
  result = array_ops.repeat(a, repeats, axis)
  if known_shape:
    result.set_shape(original_shape)

  return result


@tf_export.tf_export('experimental.numpy.around', v1=[])
@np_utils.np_doc('around')
def around(a, decimals=0):  # pylint: disable=missing-docstring
  a = asarray(a)
  dtype = a.dtype.as_numpy_dtype
  factor = math.pow(10, decimals)
  if np.issubdtype(dtype, np.inexact):
    factor = math_ops.cast(factor, dtype)
  else:
    # Use float as the working dtype when a.dtype is exact (e.g. integer),
    # because `decimals` can be negative.
    float_dtype = np_utils.result_type(float)
    a = a.astype(float_dtype)
    factor = math_ops.cast(factor, float_dtype)
  a = math_ops.multiply(a, factor)
  a = math_ops.round(a)
  a = math_ops.divide(a, factor)
  return a.astype(dtype)


setattr(np_arrays.ndarray, '__round__', around)


@tf_export.tf_export('experimental.numpy.reshape', v1=[])
@np_utils.np_doc('reshape')
def reshape(a, newshape, order='C'):
  """order argument can only b 'C' or 'F'."""
  if order not in {'C', 'F'}:
    raise ValueError('Unsupported order argument {}'.format(order))

  a = asarray(a)
  if isinstance(newshape, int):
    newshape = [newshape]

  if order == 'F':
    r = array_ops.transpose(
        array_ops.reshape(array_ops.transpose(a), newshape[::-1])
    )
  else:
    r = array_ops.reshape(a, newshape)

  return r


def _reshape_method_wrapper(a, *newshape, **kwargs):
  order = kwargs.pop('order', 'C')
  if kwargs:
    raise ValueError('Unsupported arguments: {}'.format(kwargs.keys()))

  if len(newshape) == 1 and not isinstance(newshape[0], int):
    newshape = newshape[0]

  return reshape(a, newshape, order=order)


@tf_export.tf_export('experimental.numpy.expand_dims', v1=[])
@np_utils.np_doc('expand_dims')
def expand_dims(a, axis):
  a = asarray(a)
  return array_ops.expand_dims(a, axis=axis)


@tf_export.tf_export('experimental.numpy.squeeze', v1=[])
@np_utils.np_doc('squeeze')
def squeeze(a, axis=None):
  a = asarray(a)
  return array_ops.squeeze(a, axis)


@tf_export.tf_export('experimental.numpy.flatten', v1=[])
@np_utils.np_doc('flatten', link=np_utils.NoLink())
def flatten(a, order='C'):
  a = asarray(a)
  if order == 'C' or order == 'A' or order == 'K':
    # Row major.
    return array_ops.reshape(a, [-1])
  elif order == 'F':
    # Column major
    return array_ops.reshape(array_ops.transpose(a), [-1])
  else:
    raise ValueError(
        'order can only be C, A, K (all row major) or F (column major).'
    )


@tf_export.tf_export('experimental.numpy.transpose', v1=[])
@np_utils.np_doc('transpose')
def transpose(a, axes=None):
  a = asarray(a)
  if axes is not None:
    axes = asarray(axes)
  return array_ops.transpose(a=a, perm=axes)


@tf_export.tf_export('experimental.numpy.swapaxes', v1=[])
@np_utils.np_doc('swapaxes')
def swapaxes(a, axis1, axis2):  # pylint: disable=missing-docstring
  a = asarray(a)

  def adjust_axes(axes, rank):
    def f(x):
      if isinstance(x, int):
        if x < 0:
          x = x + rank
      else:
        x = array_ops.where_v2(x < 0, np_utils.add(x, a_rank), x)
      return x

    return nest.map_structure(f, axes)

  if (
      a.shape.rank is not None
      and isinstance(axis1, int)
      and isinstance(axis2, int)
  ):
    # This branch makes sure `perm` is statically known, to avoid a
    # not-compile-time-constant XLA error.
    a_rank = a.shape.rank
    axis1, axis2 = adjust_axes((axis1, axis2), a_rank)
    perm = list(range(a_rank))
    perm[axis1] = axis2
    perm[axis2] = axis1
  else:
    a_rank = array_ops.rank(a)
    axis1, axis2 = adjust_axes((axis1, axis2), a_rank)
    perm = math_ops.range(a_rank)
    perm = array_ops.tensor_scatter_update(
        perm, [[axis1], [axis2]], [axis2, axis1]
    )
  a = array_ops.transpose(a, perm)
  return a


@tf_export.tf_export('experimental.numpy.moveaxis', v1=[])
@np_utils.np_doc('moveaxis')
def moveaxis(a, source, destination):  # pylint: disable=missing-docstring
  """Raises ValueError if source, destination not in (-ndim(a), ndim(a))."""
  if not source and not destination:
    return a

  a = asarray(a)

  if isinstance(source, int):
    source = (source,)
  if isinstance(destination, int):
    destination = (destination,)
  if len(source) != len(destination):
    raise ValueError('The lengths of source and destination must equal')

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
      items = array_ops_stack.unstack(
          sort_ops.sort(array_ops_stack.stack(b)), num=len(b)
      )

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
        array_ops.expand_dims(minus_dest, 1), minus_sources, [a_rank]
    )
    perm = array_ops.tensor_scatter_update(
        perm, array_ops.expand_dims(destination, 1), source
    )
  a = array_ops.transpose(a, perm)

  return a


@tf_export.tf_export('experimental.numpy.pad', v1=[])
@np_utils.np_doc('pad')
def pad(array, pad_width, mode, **kwargs):  # pylint: disable=redefined-outer-name
  """Only supports modes 'constant', 'reflect' and 'symmetric' currently."""
  constant_values = kwargs.get('constant_values', 0)
  if not (mode == 'constant' or mode == 'reflect' or mode == 'symmetric'):
    raise ValueError('Unsupported padding mode: ' + mode)
  mode = mode.upper()
  array = asarray(array)
  pad_width = asarray(pad_width, dtype=dtypes.int32)
  return array_ops.pad(
      tensor=array,
      paddings=pad_width,
      mode=mode,
      constant_values=constant_values,
  )


@tf_export.tf_export('experimental.numpy.take', v1=[])
@np_utils.np_doc('take')
def take(a, indices, axis=None, out=None, mode='clip'):
  """out argument is not supported, and default mode is clip."""
  if out is not None:
    raise ValueError('out argument is not supported in take.')

  if mode not in {'raise', 'clip', 'wrap'}:
    raise ValueError("Invalid mode '{}' for take".format(mode))

  a = asarray(a)
  indices = asarray(indices)

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

  return array_ops.gather(a, indices, axis=axis)


@tf_export.tf_export('experimental.numpy.where', v1=[])
@np_utils.np_doc_only('where')
def where(condition, x=None, y=None):
  """Raises ValueError if exactly one of x or y is not None."""
  condition = asarray(condition, dtype=np.bool_)
  if x is None and y is None:
    return nonzero(condition)
  elif x is not None and y is not None:
    x, y = _promote_dtype(x, y)
    return array_ops.where_v2(condition, x, y)
  raise ValueError('Both x and y must be ndarrays, or both must be None.')


@tf_export.tf_export('experimental.numpy.select', v1=[])
@np_utils.np_doc('select')
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


@tf_export.tf_export('experimental.numpy.shape', v1=[])
@np_utils.np_doc(
    'shape',
    link=np_utils.Link(
        'https://numpy.org/doc/1.18/reference/generated/numpy.shape.html'
    ),
)
def shape(a):
  a = asarray(a)
  return a.shape


@tf_export.tf_export('experimental.numpy.ndim', v1=[])
@np_utils.np_doc('ndim', link=np_utils.NoLink())
def ndim(a):
  a = asarray(a)
  return a.ndim


@tf_export.tf_export('experimental.numpy.isscalar', v1=[])
@np_utils.np_doc('isscalar')
def isscalar(num):
  return ndim(num) == 0


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
      raise ValueError(
          'The %s-th boundary %s is smaller than the previous boundary %s'
          % (i, b, prev)
      )
    size = builtins.min(size, builtins.max(0, total_size - sizes_sum))
    sizes.append(size)
    sizes_sum += size
    prev = b
  sizes.append(builtins.max(0, total_size - sizes_sum))
  return sizes


@tf_export.tf_export('experimental.numpy.split', v1=[])
@np_utils.np_doc('split')
def split(ary, indices_or_sections, axis=0):
  ary = asarray(ary)
  if not isinstance(indices_or_sections, int):
    indices_or_sections = _boundaries_to_sizes(ary, indices_or_sections, axis)
  return array_ops.split(ary, indices_or_sections, axis=axis)


def _split_on_axis(np_fun_name, axis):  # pylint: disable=missing-function-docstring
  @np_utils.np_doc(np_fun_name)
  def f(ary, indices_or_sections):
    # for 1-D array, hsplit becomes vsplit
    new_axis = np_utils.cond(
        math_ops.equal(axis, 1),
        lambda: np_utils.cond(  # pylint: disable=g-long-lambda
            math_ops.equal(array_ops.rank(ary), 1), lambda: 0, lambda: axis
        ),
        lambda: axis,
    )
    if isinstance(indices_or_sections, int):
      ary_shape = ary.shape[new_axis]
      if ary_shape is not None and ary_shape % indices_or_sections:
        raise ValueError('array split does not result in an equal division')
    return split(ary, indices_or_sections, axis=new_axis)

  return f


vsplit = tf_export.tf_export('experimental.numpy.vsplit', v1=[])(
    _split_on_axis('vsplit', axis=0)
)
hsplit = tf_export.tf_export('experimental.numpy.hsplit', v1=[])(
    _split_on_axis('hsplit', axis=1)
)
dsplit = tf_export.tf_export('experimental.numpy.dsplit', v1=[])(
    _split_on_axis('dsplit', axis=2)
)


@tf_export.tf_export('experimental.numpy.broadcast_to', v1=[])
@np_utils.np_doc('broadcast_to')
def broadcast_to(array, shape):  # pylint: disable=redefined-outer-name
  return full(shape, array)


@tf_export.tf_export('experimental.numpy.stack', v1=[])
@np_utils.np_doc('stack')
def stack(arrays, axis=0):  # pylint: disable=missing-function-docstring
  if isinstance(arrays, (np_arrays.ndarray, tensor_lib.Tensor)):
    arrays = asarray(arrays)
    if axis == 0:
      return arrays
    else:
      return swapaxes(arrays, 0, axis)
  arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a if isinstance(a, np_arrays.ndarray) else a for a in arrays
  ]
  return asarray(array_ops_stack.stack(unwrapped_arrays, axis))


@tf_export.tf_export('experimental.numpy.hstack', v1=[])
@np_utils.np_doc('hstack')
def hstack(tup):
  arrays = [atleast_1d(a) for a in tup]
  arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a if isinstance(a, np_arrays.ndarray) else a for a in arrays
  ]
  rank = array_ops.rank(unwrapped_arrays[0])
  return np_utils.cond(
      math_ops.equal(rank, 1),
      lambda: array_ops.concat(unwrapped_arrays, axis=0),
      lambda: array_ops.concat(unwrapped_arrays, axis=1),
  )


@tf_export.tf_export('experimental.numpy.vstack', v1=[])
@np_utils.np_doc('vstack')
def vstack(tup):
  arrays = [atleast_2d(a) for a in tup]
  arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a if isinstance(a, np_arrays.ndarray) else a for a in arrays
  ]
  return array_ops.concat(unwrapped_arrays, axis=0)


@tf_export.tf_export('experimental.numpy.dstack', v1=[])
@np_utils.np_doc('dstack')
def dstack(tup):
  arrays = [atleast_3d(a) for a in tup]
  arrays = _promote_dtype(*arrays)  # pylint: disable=protected-access
  unwrapped_arrays = [
      a if isinstance(a, np_arrays.ndarray) else a for a in arrays
  ]
  return array_ops.concat(unwrapped_arrays, axis=2)


def _pad_left_to(n, old_shape):
  old_shape = asarray(old_shape, dtype=np.int32)
  new_shape = array_ops.pad(
      old_shape,
      [[math_ops.maximum(n - array_ops.size(old_shape), 0), 0]],
      constant_values=1,
  )
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
            lambda: reshape(x, new_shape(n, array_ops.shape(x))),
            lambda: x,
        )
    )

  arys = list(map(f, arys))
  if len(arys) == 1:
    return arys[0]
  else:
    return arys


@tf_export.tf_export('experimental.numpy.atleast_1d', v1=[])
@np_utils.np_doc('atleast_1d')
def atleast_1d(*arys):
  return _atleast_nd(1, _pad_left_to, *arys)


@tf_export.tf_export('experimental.numpy.atleast_2d', v1=[])
@np_utils.np_doc('atleast_2d')
def atleast_2d(*arys):
  return _atleast_nd(2, _pad_left_to, *arys)


@tf_export.tf_export('experimental.numpy.atleast_3d', v1=[])
@np_utils.np_doc('atleast_3d')
def atleast_3d(*arys):  # pylint: disable=missing-docstring
  def new_shape(_, old_shape):
    # pylint: disable=g-long-lambda
    ndim_ = array_ops.size(old_shape)
    return np_utils.cond(
        math_ops.equal(ndim_, 0),
        lambda: constant_op.constant([1, 1, 1], dtype=dtypes.int32),
        lambda: np_utils.cond(
            math_ops.equal(ndim_, 1),
            lambda: array_ops.pad(old_shape, [[1, 1]], constant_values=1),
            lambda: array_ops.pad(old_shape, [[0, 1]], constant_values=1),
        ),
    )

  return _atleast_nd(3, new_shape, *arys)


@tf_export.tf_export('experimental.numpy.nonzero', v1=[])
@np_utils.np_doc('nonzero')
def nonzero(a):
  a = atleast_1d(a)
  if a.shape.rank is None:
    raise ValueError(
        "The rank of `a` is unknown, so we can't decide how many "
        'arrays to return.'
    )
  return array_ops_stack.unstack(
      array_ops.where_v2(math_ops.cast(a, dtypes.bool)), a.shape.rank, axis=1
  )


@tf_export.tf_export('experimental.numpy.diag_indices', v1=[])
@np_utils.np_doc('diag_indices')
def diag_indices(n, ndim=2):  # pylint: disable=missing-docstring,redefined-outer-name
  if n < 0:
    raise ValueError(
        'n argument to diag_indices must be nonnegative, got {}'.format(n)
    )
  if ndim < 0:
    raise ValueError(
        'ndim argument to diag_indices must be nonnegative, got {}'.format(ndim)
    )

  return (math_ops.range(n),) * ndim


@tf_export.tf_export('experimental.numpy.tri', v1=[])
@np_utils.np_doc('tri')
def tri(N, M=None, k=0, dtype=None):  # pylint: disable=invalid-name,missing-docstring
  M = M if M is not None else N
  if dtype is not None:
    dtype = np_utils.result_type(dtype)
  else:
    # Use a default float type.
    dtype = np_utils.result_type(float)

  if k < 0:
    lower = -k - 1
    if lower > N:
      r = array_ops.zeros([N, M], dtype)
    else:
      # Keep as tf bool, since we create an upper triangular matrix and invert
      # it.
      o = array_ops.ones([N, M], dtype=dtypes.bool)
      r = math_ops.cast(
          math_ops.logical_not(array_ops.matrix_band_part(o, lower, -1)), dtype
      )
  else:
    o = array_ops.ones([N, M], dtype)
    if k > M:
      r = o
    else:
      r = array_ops.matrix_band_part(o, -1, k)
  return r


@tf_export.tf_export('experimental.numpy.tril', v1=[])
@np_utils.np_doc('tril')
def tril(m, k=0):  # pylint: disable=missing-docstring
  m = asarray(m)
  if m.shape.ndims is None:
    raise ValueError('Argument to tril should have known rank')
  m_shape = m.shape.as_list()

  if len(m_shape) < 2:
    raise ValueError('Argument to tril must have rank at least 2')

  if m_shape[-1] is None or m_shape[-2] is None:
    raise ValueError(
        'Currently, the last two dimensions of the input array '
        'need to be known.'
    )

  z = constant_op.constant(0, m.dtype)

  mask = tri(*m_shape[-2:], k=k, dtype=bool)
  return array_ops.where_v2(
      array_ops.broadcast_to(mask, array_ops.shape(m)), m, z
  )


@tf_export.tf_export('experimental.numpy.triu', v1=[])
@np_utils.np_doc('triu')
def triu(m, k=0):  # pylint: disable=missing-docstring
  m = asarray(m)
  if m.shape.ndims is None:
    raise ValueError('Argument to triu should have known rank')
  m_shape = m.shape.as_list()

  if len(m_shape) < 2:
    raise ValueError('Argument to triu must have rank at least 2')

  if m_shape[-1] is None or m_shape[-2] is None:
    raise ValueError(
        'Currently, the last two dimensions of the input array '
        'need to be known.'
    )

  z = constant_op.constant(0, m.dtype)

  mask = tri(*m_shape[-2:], k=k - 1, dtype=bool)
  return array_ops.where_v2(
      array_ops.broadcast_to(mask, array_ops.shape(m)), z, m
  )


@tf_export.tf_export('experimental.numpy.flip', v1=[])
@np_utils.np_doc('flip')
def flip(m, axis=None):  # pylint: disable=missing-docstring
  m = asarray(m)

  if axis is None:
    return array_ops.reverse(m, math_ops.range(array_ops.rank(m)))

  axis = np_utils._canonicalize_axis(axis, array_ops.rank(m))  # pylint: disable=protected-access

  return array_ops.reverse(m, [axis])


@tf_export.tf_export('experimental.numpy.flipud', v1=[])
@np_utils.np_doc('flipud')
def flipud(m):  # pylint: disable=missing-docstring
  return flip(m, 0)


@tf_export.tf_export('experimental.numpy.fliplr', v1=[])
@np_utils.np_doc('fliplr')
def fliplr(m):  # pylint: disable=missing-docstring
  return flip(m, 1)


@tf_export.tf_export('experimental.numpy.roll', v1=[])
@np_utils.np_doc('roll')
def roll(a, shift, axis=None):  # pylint: disable=missing-docstring
  a = asarray(a)

  if axis is not None:
    return manip_ops.roll(a, shift, axis)

  # If axis is None, the roll happens as a 1-d tensor.
  original_shape = array_ops.shape(a)
  a = manip_ops.roll(array_ops.reshape(a, [-1]), shift, 0)
  return array_ops.reshape(a, original_shape)


@tf_export.tf_export('experimental.numpy.rot90', v1=[])
@np_utils.np_doc('rot90')
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


@tf_export.tf_export('experimental.numpy.vander', v1=[])
@np_utils.np_doc('vander')
def vander(x, N=None, increasing=False):  # pylint: disable=missing-docstring,invalid-name
  x = asarray(x)

  x_shape = array_ops.shape(x)
  if N is None:
    N = x_shape[0]

  N_temp = np_utils.get_static_value(N)  # pylint: disable=invalid-name
  if N_temp is not None:
    N = N_temp
    if N < 0:
      raise ValueError('N must be nonnegative')
  else:
    control_flow_assert.Assert(N >= 0, [N])

  rank = array_ops.rank(x)
  rank_temp = np_utils.get_static_value(rank)
  if rank_temp is not None:
    rank = rank_temp
    if rank != 1:
      raise ValueError('x must be a one-dimensional array')
  else:
    control_flow_assert.Assert(math_ops.equal(rank, 1), [rank])

  if increasing:
    start = 0
    limit = N
    delta = 1
  else:
    start = N - 1
    limit = -1
    delta = -1

  x = array_ops.expand_dims(x, -1)
  return math_ops.pow(
      x, math_ops.cast(math_ops.range(start, limit, delta), dtype=x.dtype)
  )


@tf_export.tf_export('experimental.numpy.ix_', v1=[])
@np_utils.np_doc('ix_')
def ix_(*args):  # pylint: disable=missing-docstring
  n = len(args)
  output = []
  for i, a in enumerate(args):
    a = asarray(a)
    a_rank = array_ops.rank(a)
    a_rank_temp = np_utils.get_static_value(a_rank)
    if a_rank_temp is not None:
      a_rank = a_rank_temp
      if a_rank != 1:
        raise ValueError(
            'Arguments must be 1-d, got arg {} of rank {}'.format(i, a_rank)
        )
    else:
      control_flow_assert.Assert(math_ops.equal(a_rank, 1), [a_rank])

    new_shape = [1] * n
    new_shape[i] = -1
    dtype = a.dtype
    if dtype == dtypes.bool:
      output.append(array_ops.reshape(nonzero(a)[0], new_shape))
    elif dtype.is_integer:
      output.append(array_ops.reshape(a, new_shape))
    else:
      raise ValueError(
          'Only integer and bool dtypes are supported, got {}'.format(dtype)
      )

  return output


@tf_export.tf_export('experimental.numpy.broadcast_arrays', v1=[])
@np_utils.np_doc('broadcast_arrays')
def broadcast_arrays(*args, **kwargs):  # pylint: disable=missing-docstring
  subok = kwargs.pop('subok', False)
  if subok:
    raise ValueError('subok=True is not supported.')
  if kwargs:
    raise ValueError('Received unsupported arguments {}'.format(kwargs.keys()))

  args = [asarray(arg) for arg in args]
  return np_utils.tf_broadcast(*args)


@tf_export.tf_export('experimental.numpy.sign', v1=[])
@np_utils.np_doc_only('sign')
def sign(x, out=None, where=None, **kwargs):  # pylint: disable=missing-docstring,redefined-outer-name
  if out:
    raise ValueError('tf.numpy doesnt support setting out.')
  if where:
    raise ValueError('tf.numpy doesnt support setting where.')
  if kwargs:
    raise ValueError('tf.numpy doesnt support setting {}'.format(kwargs.keys()))

  x = asarray(x)

  # Numpy 2.x and later uses the same definition of sign.
  if np.lib.NumpyVersion(np.__version__) >= '2.0.0.dev0':
    return math_ops.sign(x)

  dtype = x.dtype.as_numpy_dtype
  if np.issubdtype(dtype, np.complexfloating):
    result = math_ops.cast(math_ops.sign(math_ops.real(x)), dtype)
  else:
    result = math_ops.sign(x)

  return result


# Note that np.take_along_axis may not be present in some supported versions of
# numpy.
@tf_export.tf_export('experimental.numpy.take_along_axis', v1=[])
@np_utils.np_doc('take_along_axis')
def take_along_axis(arr, indices, axis):  # pylint: disable=missing-docstring
  arr = asarray(arr)
  indices = asarray(indices)

  if axis is None:
    return take_along_axis(arr.ravel(), indices, 0)

  rank = array_ops.rank(arr)
  axis = axis + rank if axis < 0 else axis

  # Broadcast shapes to match, ensure that the axis of interest is not
  # broadcast.
  arr_shape_original = array_ops.shape(arr, out_type=indices.dtype)
  indices_shape_original = array_ops.shape(indices, out_type=indices.dtype)
  arr_shape = array_ops.tensor_scatter_update(arr_shape_original, [[axis]], [1])
  indices_shape = array_ops.tensor_scatter_update(
      indices_shape_original, [[axis]], [1]
  )
  broadcasted_shape = array_ops.broadcast_dynamic_shape(
      arr_shape, indices_shape
  )
  arr_shape = array_ops.tensor_scatter_update(
      broadcasted_shape, [[axis]], [arr_shape_original[axis]]
  )
  indices_shape = array_ops.tensor_scatter_update(
      broadcasted_shape, [[axis]], [indices_shape_original[axis]]
  )
  arr = array_ops.broadcast_to(arr, arr_shape)
  indices = array_ops.broadcast_to(indices, indices_shape)

  # Save indices shape so we can restore it later.
  possible_result_shape = indices.shape

  # Correct indices since gather doesn't correctly handle negative indices.
  indices = array_ops.where_v2(indices < 0, indices + arr_shape[axis], indices)

  swapaxes_ = lambda t: swapaxes(t, axis, -1)

  dont_move_axis_to_end = math_ops.equal(axis, np_utils.subtract(rank, 1))
  arr = np_utils.cond(
      dont_move_axis_to_end, lambda: arr, lambda: swapaxes_(arr)
  )
  indices = np_utils.cond(
      dont_move_axis_to_end, lambda: indices, lambda: swapaxes_(indices)
  )

  arr_shape = array_ops.shape(arr)
  arr = array_ops.reshape(arr, [-1, arr_shape[-1]])

  indices_shape = array_ops.shape(indices)
  indices = array_ops.reshape(indices, [-1, indices_shape[-1]])

  result = array_ops.gather(arr, indices, batch_dims=1)
  result = array_ops.reshape(result, indices_shape)
  result = np_utils.cond(
      dont_move_axis_to_end, lambda: result, lambda: swapaxes_(result)
  )
  result.set_shape(possible_result_shape)

  return result


# pylint: disable=redefined-builtin,undefined-variable
@tf_export.tf_export('experimental.numpy.max', v1=[])
@np_utils.np_doc('max', link=np_utils.AliasOf('amax'))
def max(a, axis=None, keepdims=None):
  return amax(a, axis=axis, keepdims=keepdims)


@tf_export.tf_export('experimental.numpy.min', v1=[])
@np_utils.np_doc('min', link=np_utils.AliasOf('amin'))
def min(a, axis=None, keepdims=None):
  return amin(a, axis=axis, keepdims=keepdims)


@tf_export.tf_export('experimental.numpy.round', v1=[])
@np_utils.np_doc('round', link=np_utils.AliasOf('around'))
def round(a, decimals=0):
  return around(a, decimals=decimals)


# pylint: enable=redefined-builtin,undefined-variable


_SLICE_ERROR = (
    'only integers, slices (`:`), ellipsis (`...`), '
    'numpy.newaxis (`None`) and integer or boolean arrays are valid indices'
)


def _as_index(idx, need_scalar=True):
  """Helper function to parse idx as an index.

  Args:
    idx: index
    need_scalar: If idx needs to be a scalar value.

  Returns:
    A pair, (indx, bool). First one is the parsed index and can be a tensor,
    or scalar integer / Dimension. Second one is True if rank is known to be 0.

  Raises:
    IndexError: For incorrect indices.
  """
  if isinstance(idx, (numbers.Integral, tensor_shape.Dimension)):
    return idx, True
  data = asarray(idx)
  if data.dtype == dtypes.bool:
    if data.shape.ndims != 1:
      # TODO(agarwal): handle higher rank boolean masks.
      raise NotImplementedError('Need rank 1 for bool index %s' % idx)
    data = array_ops.where_v2(data)
    data = array_ops.reshape(data, [-1])
  if need_scalar and data.shape.rank not in (None, 0):
    raise IndexError(_SLICE_ERROR + ', got {!r}'.format(idx))
  np_dtype = data.dtype.as_numpy_dtype
  if not np.issubdtype(np_dtype, np.integer):
    raise IndexError(_SLICE_ERROR + ', got {!r}'.format(idx))
  if data.dtype not in (dtypes.int64, dtypes.int32):
    # TF slicing can only handle int32/int64. So we need to cast.
    promoted_dtype = np.promote_types(np.int32, np_dtype)
    if promoted_dtype == np.int32:
      data = math_ops.cast(data, dtypes.int32)
    elif promoted_dtype == np.int64:
      data = math_ops.cast(data, dtypes.int64)
    else:
      raise IndexError(_SLICE_ERROR + ', got {!r}'.format(idx))
  return data, data.shape.rank == 0


class _UpdateMethod(enum.Enum):
  UPDATE = 0
  ADD = 1
  MIN = 2
  MAX = 3


def _slice_helper(tensor, slice_spec, update_method=None, updates=None):
  """Helper function for __getitem__ and _with_index_update_helper.

  This function collects the indices in `slice_spec` into two buckets, which we
  can call "idx1" and "idx2" here. idx1 is intended for `strided_slice`, idx2
  `gather`.  They also correspond to "basic indices" and "advanced indices" in
  numpy.  This function supports both reading and writing at the indices. The
  reading path can be summarized as `gather(stride_slice(tensor, idx1),
  idx2)`. The writing path can be summarized as `strided_slice_update(tensor,
  idx1, scatter(strided_slice(tensor, idx1), idx2, updates))`.  (`gather` here
  means `tf.gather` or `tf.gather_nd`; `scatter` here means
  `tf.tensor_scatter_update`.)  The writing path is inefficient because it needs
  to first read out a portion (probably much larger than `updates`) of `tensor`
  using `strided_slice`, update it, and then write the portion back. An
  alternative approach is to only use `scatter`, which amounts to using the
  indexing mechanism of gather/scatter to implement
  strided_slice/strided_slice_update. This is feasible for XLA Gather/Scatter
  because they support spans (e.g. `2:5`) in indices (as begin/end pairs), but
  not TF gather/scatter because they don't support spans (except those that
  cover entire dimensions, i.e. `:`).  If we materialize spans into individual
  indices, the size of the index tensor would explode.  (Note that XLA
  Gather/Scatter have a similar problem for stride > 1 because they don't
  support strides.  Indices such as `1:2:8` will need to be materialized into
  individual indices such as [1, 3, 5, 7].)

  Args:
    tensor: the tensor to be read from or write into.
    slice_spec: the indices.
    update_method: (optional) a member of `_UpdateMethod`, indicating how to
      update the values (replacement, add, etc.). `None` indicates just reading.
    updates: (optional) the new values to write into `tensor`. It must have the
      same dtype as `tensor`.

  Returns:
    The result of reading (if `update_method` is `None`) or the updated `tensor`
    after writing.
  """
  begin, end, strides = [], [], []
  new_axis_mask, shrink_axis_mask = 0, 0
  begin_mask, end_mask = 0, 0
  ellipsis_mask = 0
  advanced_indices = []
  shrink_indices = []
  for index, s in enumerate(slice_spec):
    if isinstance(s, slice):
      if s.start is not None:
        begin.append(_as_index(s.start)[0])
      else:
        begin.append(0)
        begin_mask |= 1 << index
      if s.stop is not None:
        end.append(_as_index(s.stop)[0])
      else:
        end.append(0)
        end_mask |= 1 << index
      if s.step is not None:
        strides.append(_as_index(s.step)[0])
      else:
        strides.append(1)
    elif s is Ellipsis:
      begin.append(0)
      end.append(0)
      strides.append(1)
      ellipsis_mask |= 1 << index
    elif s is array_ops.newaxis:
      begin.append(0)
      end.append(0)
      strides.append(1)
      new_axis_mask |= 1 << index
    else:
      s, is_scalar = _as_index(s, False)
      if is_scalar:
        begin.append(s)
        end.append(s + 1)
        strides.append(1)
        shrink_axis_mask |= 1 << index
        shrink_indices.append(index)
      else:
        begin.append(0)
        end.append(0)
        strides.append(1)
        begin_mask |= 1 << index
        end_mask |= 1 << index
        advanced_indices.append((index, s, ellipsis_mask != 0))

  # stack possibly involves no tensors, so we must use op_scope correct graph.
  with ops.name_scope(
      None,
      'strided_slice',
      [tensor] + begin + end + strides,
      skip_on_eager=False,
  ) as name:
    if begin:
      packed_begin, packed_end, packed_strides = (
          array_ops_stack.stack(begin),
          array_ops_stack.stack(end),
          array_ops_stack.stack(strides),
      )
      if (
          packed_begin.dtype == dtypes.int64
          or packed_end.dtype == dtypes.int64
          or packed_strides.dtype == dtypes.int64
      ):
        if packed_begin.dtype != dtypes.int64:
          packed_begin = math_ops.cast(packed_begin, dtypes.int64)
        if packed_end.dtype != dtypes.int64:
          packed_end = math_ops.cast(packed_end, dtypes.int64)
        if packed_strides.dtype != dtypes.int64:
          packed_strides = math_ops.cast(packed_strides, dtypes.int64)
    else:
      var_empty = constant_op.constant([], dtype=dtypes.int32)
      packed_begin = packed_end = packed_strides = var_empty
    if update_method == _UpdateMethod.UPDATE and not advanced_indices:
      return array_ops.tensor_strided_slice_update(
          tensor,
          packed_begin,
          packed_end,
          packed_strides,
          updates,
          begin_mask=begin_mask,
          end_mask=end_mask,
          shrink_axis_mask=shrink_axis_mask,
          new_axis_mask=new_axis_mask,
          ellipsis_mask=ellipsis_mask,
          name=name,
      )
    else:
      # TODO(b/164251540): Find a better way to support update that does not
      #   involve one read + two writes.
      if updates is not None:
        original_tensor = tensor
      # TODO(agarwal): set_shape on tensor to set rank.
      tensor = array_ops.strided_slice(
          tensor,
          packed_begin,
          packed_end,
          packed_strides,
          begin_mask=begin_mask,
          end_mask=end_mask,
          shrink_axis_mask=shrink_axis_mask,
          new_axis_mask=new_axis_mask,
          ellipsis_mask=ellipsis_mask,
          name=name,
      )
    if not advanced_indices:
      if update_method is None:
        return tensor
      assert update_method != _UpdateMethod.UPDATE
      # TF lacks TensorStridedSliceAdd and alike, so we need to do
      # read+add+update.
      if update_method == _UpdateMethod.ADD:
        update_op = math_ops.add
      elif update_method == _UpdateMethod.MIN:
        update_op = math_ops.minimum
      elif update_method == _UpdateMethod.MAX:
        update_op = math_ops.maximum
      return array_ops.tensor_strided_slice_update(
          original_tensor,
          packed_begin,
          packed_end,
          packed_strides,
          update_op(tensor, updates),
          begin_mask=begin_mask,
          end_mask=end_mask,
          shrink_axis_mask=shrink_axis_mask,
          new_axis_mask=new_axis_mask,
          ellipsis_mask=ellipsis_mask,
          name=name + '_2',
      )
    advanced_indices_map = {}
    for index, data, had_ellipsis in advanced_indices:
      if had_ellipsis:
        num_shrink = len([x for x in shrink_indices if x > index])
        dim = index - len(slice_spec) + num_shrink
      else:
        num_shrink = len([x for x in shrink_indices if x < index])
        dim = index - num_shrink
      advanced_indices_map[dim] = data
    dims = sorted(advanced_indices_map.keys())
    dims_contiguous = True
    if len(dims) > 1:
      if dims[0] < 0 and dims[-1] >= 0:  # not all same sign
        dims_contiguous = False
      else:
        for i in range(len(dims) - 1):
          if dims[i] + 1 != dims[i + 1]:
            dims_contiguous = False
            break
    indices = [advanced_indices_map[x] for x in dims]
    indices = _promote_dtype(*indices)
    indices = np_utils.tf_broadcast(*indices)
    stacked_indices = array_ops_stack.stack(indices, axis=-1)
    # Skip the contiguous-dims optimization for update because there is no
    # tf.*scatter* op that supports the `axis` argument.
    if not dims_contiguous or updates is not None:
      if range(len(dims)) != dims:
        tensor = moveaxis(tensor, dims, range(len(dims)))
      tensor_shape_prefix = array_ops.shape(
          tensor, out_type=stacked_indices.dtype
      )[: len(dims)]
      stacked_indices = array_ops.where_v2(
          stacked_indices < 0,
          stacked_indices + tensor_shape_prefix,
          stacked_indices,
      )
      if updates is None:
        return array_ops.gather_nd(tensor, stacked_indices)
      else:
        # We only need to move-axis `updates` in the contiguous case becausce
        # only in this case the result dimensions of advanced indexing are in
        # the middle of `updates`. In the non-contiguous case, those dimensions
        # are always at the front.
        if dims_contiguous:
          # TODO(wangpeng): Support unknown rank (e.g. by partially flattening
          #   `updates`)
          if stacked_indices.shape.rank is None:
            raise NotImplementedError(
                'Rank of the advanced indices must currently be known'
            )
          batch_size = stacked_indices.shape.rank - 1
          batch_start = dims[0]
          if batch_start < 0:
            batch_start += len(dims) - batch_size

          def range_(start, length):
            return range(start, start + length)

          updates = moveaxis(
              updates, range_(batch_start, batch_size), range(batch_size)
          )
        if update_method == _UpdateMethod.UPDATE:
          update_op = array_ops.tensor_scatter_update
        elif update_method == _UpdateMethod.ADD:
          update_op = array_ops.tensor_scatter_add
        elif update_method == _UpdateMethod.MIN:
          update_op = array_ops.tensor_scatter_min
        elif update_method == _UpdateMethod.MAX:
          update_op = array_ops.tensor_scatter_max
        tensor = update_op(tensor, stacked_indices, updates)
        if range(len(dims)) != dims:
          tensor = moveaxis(tensor, range(len(dims)), dims)
        return array_ops.tensor_strided_slice_update(
            original_tensor,
            packed_begin,
            packed_end,
            packed_strides,
            tensor,
            begin_mask=begin_mask,
            end_mask=end_mask,
            shrink_axis_mask=shrink_axis_mask,
            new_axis_mask=new_axis_mask,
            ellipsis_mask=ellipsis_mask,
            name=name + '_2',
        )
    # Note that gather_nd does not support gathering from inside the array.
    # To avoid shuffling data back and forth, we transform the indices and
    # do a gather instead.
    rank = np_utils._maybe_static(array_ops.rank(tensor))  # pylint: disable=protected-access
    dims = [(x + rank if x < 0 else x) for x in dims]
    shape_tensor = array_ops.shape(tensor)
    dim_sizes = array_ops.gather(shape_tensor, dims)
    if len(dims) == 1:
      stacked_indices = indices[0]
    stacked_indices = math_ops.cast(stacked_indices, dtypes.int32)
    stacked_indices = array_ops.where_v2(
        stacked_indices < 0, stacked_indices + dim_sizes, stacked_indices
    )
    axis = dims[0]
    if len(dims) > 1:
      index_scaling = math_ops.cumprod(dim_sizes, reverse=True, exclusive=True)

      def _tensordot(a, b):
        # TODO(b/168657656): This function should be replaced by
        # tensordot(axis=1) once MatMul has int32 XLA kernel.
        b = array_ops.broadcast_to(b, array_ops.shape(a))
        return math_ops.reduce_sum(a * b, axis=-1)

      stacked_indices = _tensordot(stacked_indices, index_scaling)
      flat_shape = array_ops.concat(
          [shape_tensor[:axis], [-1], shape_tensor[axis + len(dims) :]], axis=0
      )
      tensor = array_ops.reshape(tensor, flat_shape)

    return array_ops.gather(tensor, stacked_indices, axis=axis)


def _as_spec_tuple(slice_spec):
  """Convert slice_spec to tuple."""
  if isinstance(slice_spec, (list, tuple)) and not isinstance(
      slice_spec, np.ndarray
  ):
    is_index = True
    for s in slice_spec:
      if s is None or s is Ellipsis or isinstance(s, (list, tuple, slice)):
        is_index = False
        break
      elif isinstance(s, (np_arrays.ndarray, np.ndarray)) and s.ndim != 0:
        is_index = False
        break
    if not is_index:
      return tuple(slice_spec)
  return (slice_spec,)


def _getitem(self, slice_spec):
  """Implementation of ndarray.__getitem__."""
  if (
      isinstance(slice_spec, bool)
      or (
          isinstance(slice_spec, core_tf_types.Tensor)
          and slice_spec.dtype == dtypes.bool
      )
      or (
          isinstance(slice_spec, (np.ndarray, np_arrays.ndarray))
          and slice_spec.dtype == np.bool_
      )
  ):
    return array_ops.boolean_mask(tensor=self, mask=slice_spec)

  if not isinstance(slice_spec, tuple):
    slice_spec = _as_spec_tuple(slice_spec)

  result_t = _slice_helper(self, slice_spec)
  return result_t


def _with_index_update_helper(update_method, a, slice_spec, updates):
  """Implementation of ndarray._with_index_*."""
  if (
      isinstance(slice_spec, bool)
      or (
          isinstance(slice_spec, core_tf_types.Tensor)
          and slice_spec.dtype == dtypes.bool
      )
      or (
          isinstance(slice_spec, (np.ndarray, np_arrays.ndarray))
          and slice_spec.dtype == np.bool_
      )
  ):
    slice_spec = nonzero(slice_spec)

  if not isinstance(slice_spec, tuple):
    slice_spec = _as_spec_tuple(slice_spec)

  a_dtype = a.dtype
  a, updates = _promote_dtype_binary(a, updates)
  result_t = _slice_helper(a, slice_spec, update_method, updates)
  return result_t.astype(a_dtype)


setattr(np_arrays.ndarray, '_numpy_style_getitem', _getitem)
setattr(
    np_arrays.ndarray,
    '_with_index_update',
    functools.partial(_with_index_update_helper, _UpdateMethod.UPDATE),
)
setattr(
    np_arrays.ndarray,
    '_with_index_add',
    functools.partial(_with_index_update_helper, _UpdateMethod.ADD),
)
setattr(
    np_arrays.ndarray,
    '_with_index_min',
    functools.partial(_with_index_update_helper, _UpdateMethod.MIN),
)
setattr(
    np_arrays.ndarray,
    '_with_index_max',
    functools.partial(_with_index_update_helper, _UpdateMethod.MAX),
)
