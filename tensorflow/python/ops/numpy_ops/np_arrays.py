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
"""ndarray class."""

# pylint: disable=g-direct-tensorflow-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
import numpy as np
import six

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.util import nest


_SLICE_TYPE_ERROR = (
    'Only integers, slices (`:`), ellipsis (`...`), '
    'tf.newaxis (`None`) and scalar tf.int32/tf.int64 tensors are valid '
    'indices')

_SUPPORTED_SLICE_DTYPES = (dtypes.int32, dtypes.int32_ref, dtypes.int64,
                           dtypes.int64_ref)


def _check_index(idx):
  """Check if a given value is a valid index into a tensor."""
  if isinstance(idx, (numbers.Integral, tensor_shape.Dimension)):
    return

  # Optimistic check. Assumptions:
  # * any object with a dtype is supported
  # * any object with a dtype has a sizeable shape attribute.
  dtype = getattr(idx, 'dtype', None)
  if (dtype is None or dtypes.as_dtype(dtype) not in _SUPPORTED_SLICE_DTYPES or
      idx.shape and len(idx.shape) == 1):
    # TODO(slebedev): IndexError seems more appropriate here, but it
    # will break `_slice_helper` contract.
    raise TypeError(_SLICE_TYPE_ERROR + ', got {!r}'.format(idx))


def _is_undefined_dimension(d):
  return isinstance(d, tensor_shape.Dimension) and d.value is None


def _slice_helper(tensor, slice_spec, var=None):
  """Copied from array_ops._slice_helper, will be merged back later."""
  if isinstance(slice_spec, bool) or \
  (isinstance(slice_spec, ops.Tensor) and slice_spec.dtype == dtypes.bool) or \
  (isinstance(slice_spec, np.ndarray) and slice_spec.dtype == bool):
    return array_ops.boolean_mask(tensor=tensor, mask=slice_spec)

  if not isinstance(slice_spec, (list, tuple)):
    slice_spec = [slice_spec]

  begin, end, strides = [], [], []
  index = 0

  new_axis_mask, shrink_axis_mask = 0, 0
  begin_mask, end_mask = 0, 0
  ellipsis_mask = 0
  for s in slice_spec:
    if isinstance(s, slice):
      if s.start is not None and not _is_undefined_dimension(s.start):
        _check_index(s.start)
        begin.append(s.start)
      else:
        begin.append(0)
        begin_mask |= (1 << index)
      if s.stop is not None and not _is_undefined_dimension(s.stop):
        _check_index(s.stop)
        end.append(s.stop)
      else:
        end.append(0)
        end_mask |= (1 << index)
      if s.step is not None and not _is_undefined_dimension(s.step):
        _check_index(s.step)
        strides.append(s.step)
      else:
        strides.append(1)
    elif s is Ellipsis:
      begin.append(0)
      end.append(0)
      strides.append(1)
      ellipsis_mask |= (1 << index)
    elif s is array_ops.newaxis:
      begin.append(0)
      end.append(0)
      strides.append(1)
      new_axis_mask |= (1 << index)
    else:
      _check_index(s)
      begin.append(s)
      end.append(s + 1)
      strides.append(1)
      shrink_axis_mask |= (1 << index)
    index += 1

  # stack possibly involves no tensors, so we must use op_scope correct graph.
  with ops.name_scope(
      None,
      'strided_slice', [tensor] + begin + end + strides,
      skip_on_eager=False) as name:
    if begin:
      packed_begin, packed_end, packed_strides = (array_ops.stack(begin),
                                                  array_ops.stack(end),
                                                  array_ops.stack(strides))
      if (packed_begin.dtype == dtypes.int64 or
          packed_end.dtype == dtypes.int64 or
          packed_strides.dtype == dtypes.int64):
        if packed_begin.dtype != dtypes.int64:
          packed_begin = math_ops.cast(packed_begin, dtypes.int64)
        if packed_end.dtype != dtypes.int64:
          packed_end = math_ops.cast(packed_end, dtypes.int64)
        if packed_strides.dtype != dtypes.int64:
          packed_strides = math_ops.cast(packed_strides, dtypes.int64)
    else:
      var_empty = constant_op.constant([], dtype=dtypes.int32)
      packed_begin = packed_end = packed_strides = var_empty
    return array_ops.strided_slice(
        tensor,
        packed_begin,
        packed_end,
        packed_strides,
        begin_mask=begin_mask,
        end_mask=end_mask,
        shrink_axis_mask=shrink_axis_mask,
        new_axis_mask=new_axis_mask,
        ellipsis_mask=ellipsis_mask,
        var=var,
        name=name)


def convert_to_tensor(value, dtype=None, dtype_hint=None):
  """Wrapper over `tf.convert_to_tensor`.

  Args:
    value: value to convert
    dtype: (optional) the type we would like it to be converted to.
    dtype_hint: (optional) soft preference for the type we would like it to be
      converted to. `tf.convert_to_tensor` will attempt to convert value to this
      type first, but will not fail if conversion is not possible falling back
      to inferring the type instead.

  Returns:
    Value converted to tf.Tensor.
  """
  # A safer version of `tf.convert_to_tensor` to work around b/149876037.
  # TODO(wangpeng): Remove this function once the bug is fixed.
  if (dtype is None and isinstance(value, six.integer_types) and
      value >= 2**63):
    dtype = dtypes.uint64
  elif (dtype is None and isinstance(value, float)):
    dtype = np_dtypes.default_float_type()
  return ops.convert_to_tensor(value, dtype=dtype, dtype_hint=dtype_hint)


class ndarray(object):  # pylint: disable=invalid-name
  """Equivalent of numpy.ndarray backed by TensorFlow tensors.

  This does not support all features of NumPy ndarrays e.g. strides and
  memory order since, unlike NumPy, the backing storage is not a raw memory
  buffer.

  TODO(srbs): Clearly specify which attributes and methods are not supported
  or if there are any differences in behavior.
  """

  def __init__(self, shape, dtype=float, buffer=None):  # pylint: disable=redefined-builtin
    """Initializes an ndarray.

    This is a low level interface for building ndarrays and should be avoided.
    Users should instead use methods in array_creation.py.

    This class provides a numpy.ndarray like interface for a TF Tensor with a
    fully-defined shape. Note that, unlike the backing buffer of np.ndarray,
    Tensors are immutable. So, operations like `__setitem__` are performed by
    replacing the Tensor. This restricts the ability to implement NumPy `view`
    semantics.

    Compared to numpy.ndarray, this does not support `offset`, `strides`
    and `order` arguments.

    Args:
      shape: The shape of the array. Must be a scalar, an iterable of integers
        or a `TensorShape` object.
      dtype: Optional. The dtype of the array. Must be a python type, a numpy
        type or a tensorflow `DType` object.
      buffer: Optional. The backing buffer of the array. Must have shape
        `shape`. Must be a `ndarray`, `np.ndarray` or a `Tensor`.

    Raises:
      ValueError: If `buffer` is specified and its shape does not match
       `shape`.
    """
    if dtype and not isinstance(dtype, dtypes.DType):
      dtype = dtypes.as_dtype(np.dtype(dtype))
    if buffer is None:
      buffer = array_ops.zeros(shape, dtype=dtype)
    else:
      if isinstance(buffer, ndarray):
        buffer = buffer.data
      elif isinstance(buffer, np.ndarray):
        # If `buffer` is a np.ndarray, the Tensor will share the underlying
        # storage of the array.
        buffer = convert_to_tensor(value=buffer, dtype=dtype)
      elif not isinstance(buffer, ops.Tensor):
        raise ValueError('Unexpected type for `buffer` {}. Must be an ndarray,'
                         ' Tensor or np.ndarray.'.format(type(buffer)))

      if shape is not None and tuple(shape) != buffer._shape_tuple():  # pylint: disable=protected-access
        # TODO(srbs): NumPy allows this. Investigate if/how to support this.
        raise ValueError('shape arg must match buffer.shape.')

    assert isinstance(buffer, ops.Tensor)
    if dtype and dtype != buffer.dtype:
      buffer = array_ops.bitcast(buffer, dtype)
    self._data = buffer
    self.base = None

  @property
  def data(self):
    """Tensor object containing the array data.

    This has a few key differences from the Python buffer object used in
    NumPy arrays.
    1. Tensors are immutable. So operations requiring in-place edit, e.g.
       __setitem__, are performed by replacing the underlying buffer with a new
       one.
    2. Tensors do not provide access to their raw buffer.

    Returns:
      A Tensor.
    """
    return self._data

  @property
  def shape(self):
    """Returns a tuple or tf.Tensor of array dimensions."""
    shape = self.data.shape
    if shape.is_fully_defined():
      return tuple(shape.as_list())
    else:
      return array_ops.shape(self.data)

  @property
  def dtype(self):
    return np.dtype(self.data.dtype.as_numpy_dtype)

  @property
  def ndim(self):
    ndims = self.data.shape.ndims
    if ndims is None:
      return array_ops.rank(self.data)
    else:
      return ndims

  @property
  def size(self):
    """Returns the number of elements in the array."""
    shape = self.shape
    if isinstance(shape, ops.Tensor):
      return array_ops.size(self.data)
    else:
      return np.prod(self.shape)

  @property
  def T(self):  # pylint: disable=invalid-name
    return self.transpose()

  def __len__(self):
    shape = self.shape
    if isinstance(shape, ops.Tensor):
      raise TypeError('len() of symbolic tensor undefined')
    elif shape:
      return self.shape[0]
    else:
      raise TypeError('len() of unsized object.')

  def astype(self, dtype):
    if self.dtype == dtype:
      return self
    else:
      return tensor_to_ndarray(math_ops.cast(self.data, dtype))

  # Unary operations
  def __neg__(self):
    return tensor_to_ndarray(-self.data)  # pylint: disable=invalid-unary-operand-type

  def __pos__(self):
    return self

  __hash__ = None

  def __int__(self):
    return int(self.data)

  def __float__(self):
    return float(self.data)

  def __nonzero__(self):
    return bool(self.data)

  def __bool__(self):
    return self.__nonzero__()

  def __getitem__(self, slice_spec):
    # TODO(srbs): Need to support better indexing.
    def _gettensor(x):
      if isinstance(x, ndarray):
        x = x.data
      if isinstance(x, ops.Tensor) and x.dtype not in (
          dtypes.int32, dtypes.int64):
        # Currently _slice_helper will only work with int32/int64 tensors, but
        # type inference by numpy can create {u,}int{8,16}, so just cast.
        x = math_ops.cast(x, dtypes.int32)
      return x
    slice_spec = nest.map_structure(_gettensor, slice_spec)

    result_t = _slice_helper(self.data, slice_spec)
    return tensor_to_ndarray(result_t)

  def __iter__(self):
    if not isinstance(self.data, ops.EagerTensor):
      raise TypeError('Iteration over symbolic tensor is not allowed')
    for i in range(self.shape[0]):
      result_t = self.data[i]
      yield tensor_to_ndarray(result_t)
    return

  def __array__(self, dtype=None):
    """Returns a NumPy ndarray.

    This allows instances of this class to be directly used in NumPy routines.
    However, doing that may force a copy to CPU.

    Args:
      dtype: A NumPy compatible type.

    Returns:
      A NumPy ndarray.
    """
    return np.asarray(self.data, dtype)

  __array_priority__ = 110

  def __index__(self):
    """Returns a python scalar.

    This allows using an instance of this class as an array index.
    Note that only arrays of integer types with size 1 can be used as array
    indices.

    Returns:
      A Python scalar.

    Raises:
      TypeError: If the array is not of an integer type.
      ValueError: If the array does not have size 1.
    """
    # TODO(wangpeng): Handle graph mode
    if not isinstance(self.data, ops.EagerTensor):
      raise TypeError('Indexing using symbolic tensor is not allowed')
    return np.asscalar(self.data.numpy())

  def tolist(self):
    return self.data.numpy().tolist()

  def __str__(self):
    return 'ndarray<{}>'.format(self.data.__str__())

  def __repr__(self):
    return 'ndarray<{}>'.format(self.data.__repr__())


def tensor_to_ndarray(tensor):
  return ndarray(tensor._shape_tuple(), dtype=tensor.dtype, buffer=tensor)  # pylint: disable=protected-access


def ndarray_to_tensor(arr, dtype=None, name=None, as_ref=False):
  if as_ref:
    raise ValueError('as_ref is not supported.')
  if dtype and dtypes.as_dtype(arr.dtype) != dtype:
    return math_ops.cast(arr.data, dtype)
  result_t = arr.data
  if name:
    result_t = array_ops.identity(result_t, name=name)
  return result_t


ops.register_tensor_conversion_function(ndarray, ndarray_to_tensor)
