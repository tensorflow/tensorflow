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

import numpy as np
import six

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_export


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
  elif dtype is None and dtype_hint is None and isinstance(value, float):
    dtype = np_dtypes.default_float_type()
  return ops.convert_to_tensor(value, dtype=dtype, dtype_hint=dtype_hint)


class NdarraySpec(type_spec.BatchableTypeSpec):
  """Type specification for a `tf.experiemntal.numpy.ndarray`."""

  value_type = property(lambda self: ndarray)

  def __init__(self, data_spec):
    if not isinstance(data_spec, tensor_spec.TensorSpec):
      raise ValueError('NdarraySpec.__init__ was expecting a tf.TypeSpec, '
                       'but got a {} instead.'.format(type(data_spec)))
    self._data_spec = data_spec
    self._hash = None

  @property
  def _component_specs(self):
    return self._data_spec

  def _to_components(self, value):
    return value.data

  def _from_components(self, data):
    return tensor_to_ndarray(data)

  def _serialize(self):
    return (self._data_spec,)

  def _batch(self, batch_size):
    return NdarraySpec(self._data_spec._batch(batch_size))  # pylint: disable=protected-access

  def _unbatch(self):
    return NdarraySpec(self._data_spec._unbatch())  # pylint: disable=protected-access

  def __hash__(self):
    if self._hash is None:
      self._hash = hash((type(self), self._data_spec))
    return self._hash


@np_export.np_export('ndarray')  # pylint: disable=invalid-name
class ndarray(composite_tensor.CompositeTensor):
  """Equivalent of numpy.ndarray backed by TensorFlow tensors.

  This does not support all features of NumPy ndarrays e.g. strides and
  memory order since, unlike NumPy, the backing storage is not a raw memory
  buffer.

  TODO(srbs): Clearly specify which attributes and methods are not supported
  or if there are any differences in behavior.
  """

  __slots__ = ['_data', '_dtype', '_type_spec_internal']

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

      if shape is not None:
        buffer.set_shape(shape)

    assert isinstance(buffer, ops.Tensor)
    if dtype and dtype != buffer.dtype:
      buffer = math_ops.cast(buffer, dtype)
    self._data = buffer
    self._type_spec_internal = None
    self._dtype = None

  @classmethod
  def from_tensor(cls, tensor):
    o = cls.__new__(cls, None)
    # pylint: disable=protected-access
    o._data = tensor
    o._dtype = None
    o._type_spec_internal = None
    # pylint: enable=protected-access
    return o

  @property
  def _type_spec(self):
    if self._type_spec_internal is None:
      self._type_spec_internal = NdarraySpec(
          type_spec.type_spec_from_value(self._data))
    return self._type_spec_internal

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
    if self._dtype is None:
      self._dtype = np_dtypes._get_cached_dtype(self._data.dtype)  # pylint: disable=protected-access
    return self._dtype

  def _is_boolean(self):
    return self._data.dtype == dtypes.bool

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

  def __bool__(self):
    return bool(self.data)

  def __nonzero__(self):
    return self.__bool__()

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

  # NOTE: we currently prefer interop with TF to allow TF to take precedence.
  __array_priority__ = 90

  def __array_module__(self, types):
    # Experimental support for NumPy's module dispatch with NEP-37:
    # https://numpy.org/neps/nep-0037-array-module.html
    # Currently requires https://github.com/seberg/numpy-dispatch

    # pylint: disable=g-import-not-at-top
    import tensorflow.compat.v2 as tf

    if all(issubclass(t, (ndarray, np.ndarray)) for t in types):
      return tf.experimental.numpy
    else:
      return NotImplemented

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
    return self.data.numpy().item()

  def tolist(self):
    return self.data.numpy().tolist()

  def __str__(self):
    return 'ndarray<{}>'.format(self.data.__str__())

  def __repr__(self):
    return 'ndarray<{}>'.format(self.data.__repr__())


def tensor_to_ndarray(tensor):
  return ndarray.from_tensor(tensor)


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
