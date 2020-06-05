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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.numpy_ops import np_dtypes


def convert_to_tensor(value, dtype=None):
  # A safer version of `tf.convert_to_tensor` to work around b/149876037.
  # TODO(wangpeng): Remove this function once the bug is fixed.
  if (dtype is None and isinstance(value, six.integer_types) and
      value >= 2**63):
    dtype = dtypes.uint64
  elif (dtype is None and isinstance(value, float)):
    dtype = np_dtypes.default_float_type()
  return ops.convert_to_tensor(value, dtype=dtype)


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
    """Returns a tuple of array dimensions."""
    return self.data._shape_tuple()  # pylint: disable=protected-access

  @property
  def dtype(self):
    return np.dtype(self.data.dtype.as_numpy_dtype)

  @property
  def ndim(self):
    return self.data.shape.ndims

  @property
  def size(self):
    """Returns the number of elements in the array."""
    return np.prod(self.shape)

  @property
  def T(self):  # pylint: disable=invalid-name
    return self.transpose()

  def __len__(self):
    if self.shape:
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
    result_t = self.data.__getitem__(slice_spec)
    return tensor_to_ndarray(result_t)

  def __iter__(self):
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


# Don't use a namedtuple since nest considers that a tuple and unflattens and
# flattens it.
class ShardedNdArray(object):
  """Wrapper over ndarray that can contain tensors on multiple devices.

    This is returned by extensions.pmap, and contains the individual tensors on
    different devices.
  """

  def __init__(self, tensors):
    """Initializes the ShardedNdArray.

    Note that the tensors should be ordered in the way the pmap producing these
    tensors is run.

    Args:
      tensors: list or tuple of eager tensors, one for each device.
    """

    if not isinstance(tensors, (list, tuple)) or not tensors:
      raise ValueError(
          'Unable to create a ShardedNdArray without a list of tensors.')
    self.tensors = tensors
    self.n_devices = len(tensors)

  def __getitem__(self, i):
    return self.tensors[i]

  @property
  def shape(self):
    return (self.n_devices,) + self.tensors[0]._shape_tuple()  # pylint: disable=protected-access

  @property
  def dtype(self):
    return self.tensors[0].dtype


def convert_sharded_tensor_to_eager_tensor(value, *args, **kwargs):
  del args, kwargs
  # TODO(nareshmodi): Consider a collective op to gather the tensors from the
  # various devices for performance reasons.
  return array_ops.stack(value.tensors)


ops.register_tensor_conversion_function(ShardedNdArray,
                                        convert_sharded_tensor_to_eager_tensor)
