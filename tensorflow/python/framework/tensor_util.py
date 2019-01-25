# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities to create TensorProtos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import compat

# Fallback in case fast_tensor_util is not properly compiled.
# pylint: disable=g-import-not-at-top
try:
  from tensorflow.python.framework import fast_tensor_util
  _FAST_TENSOR_UTIL_AVAILABLE = True
except ImportError:
  _FAST_TENSOR_UTIL_AVAILABLE = False

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export

# pylint: enable=g-import-not-at-top


def ExtractBitsFromFloat16(x):
  return np.asarray(x, dtype=np.float16).view(np.uint16).item()


def SlowAppendFloat16ArrayToTensorProto(tensor_proto, proto_values):
  tensor_proto.half_val.extend(
      [ExtractBitsFromFloat16(x) for x in proto_values])


def _MediumAppendFloat16ArrayToTensorProto(tensor_proto, proto_values):
  # TODO: Remove the conversion if cython supports np.float16_t
  fast_tensor_util.AppendFloat16ArrayToTensorProto(
      tensor_proto,
      np.asarray(proto_values, dtype=np.float16).view(np.uint16))


def ExtractBitsFromBFloat16(x):
  return np.asarray(
      x, dtype=dtypes.bfloat16.as_numpy_dtype).view(np.uint16).item()


def SlowAppendBFloat16ArrayToTensorProto(tensor_proto, proto_values):
  tensor_proto.half_val.extend(
      [ExtractBitsFromBFloat16(x) for x in proto_values])


def FastAppendBFloat16ArrayToTensorProto(tensor_proto, proto_values):
  fast_tensor_util.AppendBFloat16ArrayToTensorProto(
      tensor_proto, np.asarray(
          proto_values, dtype=dtypes.bfloat16.as_numpy_dtype).view(np.uint16))


if _FAST_TENSOR_UTIL_AVAILABLE:
  _NP_TO_APPEND_FN = {
      dtypes.bfloat16.as_numpy_dtype:
          FastAppendBFloat16ArrayToTensorProto,
      np.float16:
          _MediumAppendFloat16ArrayToTensorProto,
      np.float32:
          fast_tensor_util.AppendFloat32ArrayToTensorProto,
      np.float64:
          fast_tensor_util.AppendFloat64ArrayToTensorProto,
      np.int32:
          fast_tensor_util.AppendInt32ArrayToTensorProto,
      np.int64:
          fast_tensor_util.AppendInt64ArrayToTensorProto,
      np.uint8:
          fast_tensor_util.AppendUInt8ArrayToTensorProto,
      np.uint16:
          fast_tensor_util.AppendUInt16ArrayToTensorProto,
      np.uint32:
          fast_tensor_util.AppendUInt32ArrayToTensorProto,
      np.uint64:
          fast_tensor_util.AppendUInt64ArrayToTensorProto,
      np.int8:
          fast_tensor_util.AppendInt8ArrayToTensorProto,
      np.int16:
          fast_tensor_util.AppendInt16ArrayToTensorProto,
      np.complex64:
          fast_tensor_util.AppendComplex64ArrayToTensorProto,
      np.complex128:
          fast_tensor_util.AppendComplex128ArrayToTensorProto,
      np.object:
          fast_tensor_util.AppendObjectArrayToTensorProto,
      np.bool:
          fast_tensor_util.AppendBoolArrayToTensorProto,
      dtypes.qint8.as_numpy_dtype:
          fast_tensor_util.AppendInt8ArrayToTensorProto,
      dtypes.quint8.as_numpy_dtype:
          fast_tensor_util.AppendUInt8ArrayToTensorProto,
      dtypes.qint16.as_numpy_dtype:
          fast_tensor_util.AppendInt8ArrayToTensorProto,
      dtypes.quint16.as_numpy_dtype:
          fast_tensor_util.AppendUInt8ArrayToTensorProto,
      dtypes.qint32.as_numpy_dtype:
          fast_tensor_util.AppendInt32ArrayToTensorProto,
      # NOTE(touts): Intentionally no way to feed a DT_BFLOAT16.
  }
else:

  def SlowAppendFloat32ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.float_val.extend([x.item() for x in proto_values])

  def SlowAppendFloat64ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.double_val.extend([x.item() for x in proto_values])

  def SlowAppendIntArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.int_val.extend([x.item() for x in proto_values])

  def SlowAppendInt64ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.int64_val.extend([x.item() for x in proto_values])

  def SlowAppendQIntArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.int_val.extend([x.item(0) for x in proto_values])

  def SlowAppendUInt32ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.uint32_val.extend([x.item() for x in proto_values])

  def SlowAppendUInt64ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.uint64_val.extend([x.item() for x in proto_values])

  def SlowAppendComplex64ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.scomplex_val.extend(
        [v.item() for x in proto_values for v in [x.real, x.imag]])

  def SlowAppendComplex128ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.dcomplex_val.extend(
        [v.item() for x in proto_values for v in [x.real, x.imag]])

  def SlowAppendObjectArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.string_val.extend([compat.as_bytes(x) for x in proto_values])

  def SlowAppendBoolArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.bool_val.extend([x.item() for x in proto_values])

  _NP_TO_APPEND_FN = {
      dtypes.bfloat16.as_numpy_dtype: SlowAppendBFloat16ArrayToTensorProto,
      np.float16: SlowAppendFloat16ArrayToTensorProto,
      np.float32: SlowAppendFloat32ArrayToTensorProto,
      np.float64: SlowAppendFloat64ArrayToTensorProto,
      np.int32: SlowAppendIntArrayToTensorProto,
      np.int64: SlowAppendInt64ArrayToTensorProto,
      np.uint8: SlowAppendIntArrayToTensorProto,
      np.uint16: SlowAppendIntArrayToTensorProto,
      np.uint32: SlowAppendUInt32ArrayToTensorProto,
      np.uint64: SlowAppendUInt64ArrayToTensorProto,
      np.int8: SlowAppendIntArrayToTensorProto,
      np.int16: SlowAppendIntArrayToTensorProto,
      np.complex64: SlowAppendComplex64ArrayToTensorProto,
      np.complex128: SlowAppendComplex128ArrayToTensorProto,
      np.object: SlowAppendObjectArrayToTensorProto,
      np.bool: SlowAppendBoolArrayToTensorProto,
      dtypes.qint8.as_numpy_dtype: SlowAppendQIntArrayToTensorProto,
      dtypes.quint8.as_numpy_dtype: SlowAppendQIntArrayToTensorProto,
      dtypes.qint16.as_numpy_dtype: SlowAppendQIntArrayToTensorProto,
      dtypes.quint16.as_numpy_dtype: SlowAppendQIntArrayToTensorProto,
      dtypes.qint32.as_numpy_dtype: SlowAppendQIntArrayToTensorProto,
      # NOTE(touts): Intentionally no way to feed a DT_BFLOAT16.
  }


def GetFromNumpyDTypeDict(dtype_dict, dtype):
  # NOTE: dtype_dict.get(dtype) always returns None.
  for key, val in six.iteritems(dtype_dict):
    if key == dtype:
      return val
  return None


def GetNumpyAppendFn(dtype):
  # numpy dtype for strings are variable length. We can not compare
  # dtype with a single constant (np.string does not exist) to decide
  # dtype is a "string" type. We need to compare the dtype.type to be
  # sure it's a string type.
  if dtype.type == np.string_ or dtype.type == np.unicode_:
    if _FAST_TENSOR_UTIL_AVAILABLE:
      return fast_tensor_util.AppendObjectArrayToTensorProto
    else:
      return SlowAppendObjectArrayToTensorProto
  return GetFromNumpyDTypeDict(_NP_TO_APPEND_FN, dtype)


def TensorShapeProtoToList(shape):
  """Convert a TensorShape to a list.

  Args:
    shape: A TensorShapeProto.

  Returns:
    List of integers representing the dimensions of the tensor.
  """
  return [dim.size for dim in shape.dim]


def _GetDenseDimensions(list_of_lists):
  """Returns the inferred dense dimensions of a list of lists."""
  if not isinstance(list_of_lists, (list, tuple)):
    return []
  elif not list_of_lists:
    return [0]
  else:
    return [len(list_of_lists)] + _GetDenseDimensions(list_of_lists[0])


def _FlattenToStrings(nested_strings):
  if isinstance(nested_strings, (list, tuple)):
    for inner in nested_strings:
      for flattened_string in _FlattenToStrings(inner):
        yield flattened_string
  else:
    yield nested_strings


_TENSOR_CONTENT_TYPES = frozenset([
    dtypes.float32, dtypes.float64, dtypes.int32, dtypes.uint8, dtypes.int16,
    dtypes.int8, dtypes.int64, dtypes.qint8, dtypes.quint8, dtypes.qint16,
    dtypes.quint16, dtypes.qint32, dtypes.uint32, dtypes.uint64
])


class _Message(object):

  def __init__(self, message):
    self._message = message

  def __repr__(self):
    return self._message


def _FirstNotNone(l):
  for x in l:
    if x is not None:
      if isinstance(x, ops.Tensor):
        return _Message("list containing Tensors")
      else:
        return x
  return None


def _NotNone(v):
  if v is None:
    return _Message("None")
  else:
    return v


def _FilterTuple(v):
  if not isinstance(v, (list, tuple)):
    return v
  if isinstance(v, tuple):
    if not any(isinstance(x, (list, tuple)) for x in v):
      return None
  if isinstance(v, list):
    if not any(isinstance(x, (list, tuple)) for x in v):
      return _FirstNotNone(
          [None if isinstance(x, (list, tuple)) else x for x in v])
  return _FirstNotNone([_FilterTuple(x) for x in v])


def _FilterInt(v):
  if isinstance(v, (list, tuple)):
    return _FirstNotNone([_FilterInt(x) for x in v])
  return None if isinstance(
      v, (compat.integral_types, tensor_shape.Dimension)) else _NotNone(v)


def _FilterFloat(v):
  if isinstance(v, (list, tuple)):
    return _FirstNotNone([_FilterFloat(x) for x in v])
  return None if isinstance(v, compat.real_types) else _NotNone(v)


def _FilterComplex(v):
  if isinstance(v, (list, tuple)):
    return _FirstNotNone([_FilterComplex(x) for x in v])
  return None if isinstance(v, compat.complex_types) else _NotNone(v)


def _FilterStr(v):
  if isinstance(v, (list, tuple)):
    return _FirstNotNone([_FilterStr(x) for x in v])
  if isinstance(v, compat.bytes_or_text_types):
    return None
  else:
    return _NotNone(v)


def _FilterBool(v):
  if isinstance(v, (list, tuple)):
    return _FirstNotNone([_FilterBool(x) for x in v])
  return None if isinstance(v, bool) else _NotNone(v)


def _FilterNotTensor(v):
  if isinstance(v, (list, tuple)):
    return _FirstNotNone([_FilterNotTensor(x) for x in v])
  return str(v) if isinstance(v, ops.Tensor) else None


_TF_TO_IS_OK = {
    dtypes.bool: [_FilterBool],
    dtypes.complex128: [_FilterComplex],
    dtypes.complex64: [_FilterComplex],
    dtypes.float16: [_FilterFloat],
    dtypes.float32: [_FilterFloat],
    dtypes.float64: [_FilterFloat],
    dtypes.int16: [_FilterInt],
    dtypes.int32: [_FilterInt],
    dtypes.int64: [_FilterInt],
    dtypes.int8: [_FilterInt],
    dtypes.qint16: [_FilterInt, _FilterTuple],
    dtypes.qint32: [_FilterInt, _FilterTuple],
    dtypes.qint8: [_FilterInt, _FilterTuple],
    dtypes.quint16: [_FilterInt, _FilterTuple],
    dtypes.quint8: [_FilterInt, _FilterTuple],
    dtypes.string: [_FilterStr],
    dtypes.uint16: [_FilterInt],
    dtypes.uint8: [_FilterInt],
    dtypes.uint32: [_FilterInt],
    dtypes.uint64: [_FilterInt],
}


def _AssertCompatible(values, dtype):
  if dtype is None:
    fn_list = [_FilterNotTensor]
  else:
    try:
      fn_list = _TF_TO_IS_OK[dtype]
    except KeyError:
      # There isn't a specific fn_list, so we try to do the best possible.
      if dtype.is_integer:
        fn_list = [_FilterInt]
      elif dtype.is_floating:
        fn_list = [_FilterFloat]
      elif dtype.is_complex:
        fn_list = [_FilterComplex]
      elif dtype.is_quantized:
        fn_list = [_FilterInt, _FilterTuple]
      else:
        fn_list = [_FilterNotTensor]
  mismatch = _FirstNotNone([fn(values) for fn in fn_list])
  if mismatch is not None:
    if dtype is None:
      raise TypeError("List of Tensors when single Tensor expected")
    else:
      raise TypeError("Expected %s, got %s of type '%s' instead." %
                      (dtype.name, repr(mismatch), type(mismatch).__name__))


# pylint: disable=invalid-name
@tf_export(v1=["make_tensor_proto"])
def make_tensor_proto(values, dtype=None, shape=None, verify_shape=False,
                      allow_broadcast=False):
  """Create a TensorProto.

  Args:
    values:         Values to put in the TensorProto.
    dtype:          Optional tensor_pb2 DataType value.
    shape:          List of integers representing the dimensions of tensor.
    verify_shape:   Boolean that enables verification of a shape of values.
    allow_broadcast:Boolean that enables allowing scalars and 1 length vector
        broadcasting. Cannot be true when verify_shape is true.

  Returns:
    A `TensorProto`. Depending on the type, it may contain data in the
    "tensor_content" attribute, which is not directly useful to Python programs.
    To access the values you should convert the proto back to a numpy ndarray
    with `tf.make_ndarray(proto)`.

    If `values` is a `TensorProto`, it is immediately returned; `dtype` and
    `shape` are ignored.

  Raises:
    TypeError:  if unsupported types are provided.
    ValueError: if arguments have inappropriate values or if verify_shape is
     True and shape of values is not equals to a shape from the argument.

  make_tensor_proto accepts "values" of a python scalar, a python list, a
  numpy ndarray, or a numpy scalar.

  If "values" is a python scalar or a python list, make_tensor_proto
  first convert it to numpy ndarray. If dtype is None, the
  conversion tries its best to infer the right numpy data
  type. Otherwise, the resulting numpy array has a compatible data
  type with the given dtype.

  In either case above, the numpy ndarray (either the caller provided
  or the auto converted) must have the compatible type with dtype.

  make_tensor_proto then converts the numpy array to a tensor proto.

  If "shape" is None, the resulting tensor proto represents the numpy
  array precisely.

  Otherwise, "shape" specifies the tensor's shape and the numpy array
  can not have more elements than what "shape" specifies.

  """
  if allow_broadcast and verify_shape:
    raise ValueError("allow_broadcast and verify_shape are not both allowed.")
  if isinstance(values, tensor_pb2.TensorProto):
    return values

  if dtype:
    dtype = dtypes.as_dtype(dtype)

  is_quantized = (
      dtype in [
          dtypes.qint8, dtypes.quint8, dtypes.qint16, dtypes.quint16,
          dtypes.qint32
      ])

  # We first convert value to a numpy array or scalar.
  if isinstance(values, (np.ndarray, np.generic)):
    if dtype:
      nparray = values.astype(dtype.as_numpy_dtype)
    else:
      nparray = values
  elif callable(getattr(values, "__array__", None)) or isinstance(
      getattr(values, "__array_interface__", None), dict):
    # If a class has the __array__ method, or __array_interface__ dict, then it
    # is possible to convert to numpy array.
    nparray = np.asarray(values, dtype=dtype)

    # This is the preferred way to create an array from the object, so replace
    # the `values` with the array so that _FlattenToStrings is not run.
    values = nparray
  else:
    if values is None:
      raise ValueError("None values not supported.")
    # if dtype is provided, forces numpy array to be the type
    # provided if possible.
    if dtype and dtype.is_numpy_compatible:
      np_dt = dtype.as_numpy_dtype
    else:
      np_dt = None
    # If shape is None, numpy.prod returns None when dtype is not set, but raises
    # exception when dtype is set to np.int64
    if shape is not None and np.prod(shape, dtype=np.int64) == 0:
      nparray = np.empty(shape, dtype=np_dt)
    else:
      _AssertCompatible(values, dtype)
      nparray = np.array(values, dtype=np_dt)
      # check to them.
      # We need to pass in quantized values as tuples, so don't apply the shape
      if (list(nparray.shape) != _GetDenseDimensions(values) and
          not is_quantized):
        raise ValueError("""Argument must be a dense tensor: %s"""
                         """ - got shape %s, but wanted %s.""" %
                         (values, list(nparray.shape),
                          _GetDenseDimensions(values)))

    # python/numpy default float type is float64. We prefer float32 instead.
    if (nparray.dtype == np.float64) and dtype is None:
      nparray = nparray.astype(np.float32)
    # python/numpy default int type is int64. We prefer int32 instead.
    elif (nparray.dtype == np.int64) and dtype is None:
      downcasted_array = nparray.astype(np.int32)
      # Do not down cast if it leads to precision loss.
      if np.array_equal(downcasted_array, nparray):
        nparray = downcasted_array

  # if dtype is provided, it must be compatible with what numpy
  # conversion says.
  numpy_dtype = dtypes.as_dtype(nparray.dtype)
  if numpy_dtype is None:
    raise TypeError("Unrecognized data type: %s" % nparray.dtype)

  # If dtype was specified and is a quantized type, we convert
  # numpy_dtype back into the quantized version.
  if is_quantized:
    numpy_dtype = dtype

  if dtype is not None and (not hasattr(dtype, "base_dtype") or
                            dtype.base_dtype != numpy_dtype.base_dtype):
    raise TypeError("Incompatible types: %s vs. %s. Value is %s" %
                    (dtype, nparray.dtype, values))

  # If shape is not given, get the shape from the numpy array.
  if shape is None:
    shape = nparray.shape
    is_same_size = True
    shape_size = nparray.size
  else:
    shape = [int(dim) for dim in shape]
    shape_size = np.prod(shape, dtype=np.int64)
    is_same_size = shape_size == nparray.size

    if allow_broadcast:
      if nparray.shape == (1,) or nparray.shape == tuple():
        pass
      elif nparray.size != shape_size:
        raise TypeError("Expected Tensor's shape: %s, got %s." %
                        (tuple(shape), nparray.shape))

    else:
      if verify_shape and nparray.shape != tuple(shape):
        raise TypeError("Expected Tensor's shape: %s, got %s." %
                        (tuple(shape), nparray.shape))

      if nparray.size > shape_size:
        raise ValueError(
            "Too many elements provided. Needed at most %d, but received %d" %
            (shape_size, nparray.size))

  tensor_proto = tensor_pb2.TensorProto(
      dtype=numpy_dtype.as_datatype_enum,
      tensor_shape=tensor_shape.as_shape(shape).as_proto())

  if is_same_size and numpy_dtype in _TENSOR_CONTENT_TYPES and shape_size > 1:
    if nparray.size * nparray.itemsize >= (1 << 31):
      raise ValueError(
          "Cannot create a tensor proto whose content is larger than 2GB.")
    tensor_proto.tensor_content = nparray.tostring()
    return tensor_proto

  # If we were not given values as a numpy array, compute the proto_values
  # from the given values directly, to avoid numpy trimming nulls from the
  # strings. Since values could be a list of strings, or a multi-dimensional
  # list of lists that might or might not correspond to the given shape,
  # we flatten it conservatively.
  if numpy_dtype == dtypes.string and not isinstance(values, np.ndarray):
    proto_values = _FlattenToStrings(values)

    # At this point, values may be a list of objects that we could not
    # identify a common type for (hence it was inferred as
    # np.object/dtypes.string).  If we are unable to convert it to a
    # string, we raise a more helpful error message.
    #
    # Ideally, we'd be able to convert the elements of the list to a
    # common type, but this type inference requires some thinking and
    # so we defer it for now.
    try:
      str_values = [compat.as_bytes(x) for x in proto_values]
    except TypeError:
      raise TypeError("Failed to convert object of type %s to Tensor. "
                      "Contents: %s. Consider casting elements to a "
                      "supported type." % (type(values), values))
    tensor_proto.string_val.extend(str_values)
    return tensor_proto

  # TensorFlow expects C order (a.k.a., eigen row major).
  proto_values = nparray.ravel()

  append_fn = GetNumpyAppendFn(proto_values.dtype)
  if append_fn is None:
    raise TypeError(
        "Element type not supported in TensorProto: %s" % numpy_dtype.name)
  append_fn(tensor_proto, proto_values)

  return tensor_proto
# pylint: enable=invalid-name


@tf_export("make_ndarray")
def MakeNdarray(tensor):
  """Create a numpy ndarray from a tensor.

  Create a numpy ndarray with the same shape and data as the tensor.

  Args:
    tensor: A TensorProto.

  Returns:
    A numpy array with the tensor contents.

  Raises:
    TypeError: if tensor has unsupported type.

  """
  shape = [d.size for d in tensor.tensor_shape.dim]
  num_elements = np.prod(shape, dtype=np.int64)
  tensor_dtype = dtypes.as_dtype(tensor.dtype)
  dtype = tensor_dtype.as_numpy_dtype

  if tensor.tensor_content:
    return (np.frombuffer(tensor.tensor_content,
                          dtype=dtype).copy().reshape(shape))

  if tensor_dtype == dtypes.string:
    # np.pad throws on these arrays of type np.object.
    values = list(tensor.string_val)
    padding = num_elements - len(values)
    if padding > 0:
      last = values[-1] if values else ""
      values.extend([last] * padding)
    return np.array(values, dtype=dtype).reshape(shape)

  if tensor_dtype == dtypes.float16 or tensor_dtype == dtypes.bfloat16:
    # the half_val field of the TensorProto stores the binary representation
    # of the fp16: we need to reinterpret this as a proper float16
    values = np.fromiter(tensor.half_val, dtype=np.uint16)
    values.dtype = tensor_dtype.as_numpy_dtype
  elif tensor_dtype == dtypes.float32:
    values = np.fromiter(tensor.float_val, dtype=dtype)
  elif tensor_dtype == dtypes.float64:
    values = np.fromiter(tensor.double_val, dtype=dtype)
  elif tensor_dtype in [
      dtypes.int32, dtypes.uint8, dtypes.uint16, dtypes.int16, dtypes.int8,
      dtypes.qint32, dtypes.quint8, dtypes.qint8, dtypes.qint16, dtypes.quint16
  ]:
    values = np.fromiter(tensor.int_val, dtype=dtype)
  elif tensor_dtype == dtypes.int64:
    values = np.fromiter(tensor.int64_val, dtype=dtype)
  elif tensor_dtype == dtypes.complex64:
    it = iter(tensor.scomplex_val)
    values = np.array([complex(x[0], x[1]) for x in zip(it, it)], dtype=dtype)
  elif tensor_dtype == dtypes.complex128:
    it = iter(tensor.dcomplex_val)
    values = np.array([complex(x[0], x[1]) for x in zip(it, it)], dtype=dtype)
  elif tensor_dtype == dtypes.bool:
    values = np.fromiter(tensor.bool_val, dtype=dtype)
  else:
    raise TypeError("Unsupported tensor type: %s" % tensor.dtype)

  if values.size == 0:
    return np.zeros(shape, dtype)

  if values.size != num_elements:
    values = np.pad(values, (0, num_elements - values.size), "edge")

  return values.reshape(shape)


def ShapeEquals(tensor_proto, shape):
  """Returns True if "tensor_proto" has the given "shape".

  Args:
    tensor_proto: A TensorProto.
    shape: A tensor shape, expressed as a TensorShape, list, or tuple.

  Returns:
    True if "tensor_proto" has the given "shape", otherwise False.

  Raises:
    TypeError: If "tensor_proto" is not a TensorProto, or shape is not a
      TensorShape, list, or tuple.
  """
  if not isinstance(tensor_proto, tensor_pb2.TensorProto):
    raise TypeError("tensor_proto is not a tensor_pb2.TensorProto object")
  if isinstance(shape, tensor_shape_pb2.TensorShapeProto):
    shape = [d.size for d in shape.dim]
  elif not isinstance(shape, (list, tuple)):
    raise TypeError("shape is not a list or tuple")
  tensor_shape_list = [d.size for d in tensor_proto.tensor_shape.dim]
  return all(x == y for x, y in zip(tensor_shape_list, shape))


def _ConstantValue(tensor, partial):
  # TODO(touts): Support Variables?
  if not isinstance(tensor, ops.Tensor):
    raise TypeError("%r is not a Tensor, has type %s" % (tensor, type(tensor)))
  if tensor.op.type == "Const":
    return MakeNdarray(tensor.op.get_attr("value"))
  elif tensor.op.type == "Shape":
    input_shape = tensor.op.inputs[0].get_shape()
    if input_shape.is_fully_defined():
      return np.array(
          [dim.value for dim in input_shape.dims],
          dtype=tensor.dtype.as_numpy_dtype)
    else:
      return None
  elif tensor.op.type == "Size":
    input_shape = tensor.op.inputs[0].get_shape()
    if input_shape.is_fully_defined():
      return np.prod([dim.value for dim in input_shape.dims], dtype=np.int32)
    else:
      return None
  elif tensor.op.type == "Rank":
    input_shape = tensor.op.inputs[0].get_shape()
    if input_shape.ndims is not None:
      return np.ndarray(
          shape=(),
          buffer=np.array([input_shape.ndims], dtype=np.int32),
          dtype=np.int32)
    else:
      return None
  elif tensor.op.type == "Range":
    start = constant_value(tensor.op.inputs[0])
    if start is None:
      return None
    limit = constant_value(tensor.op.inputs[1])
    if limit is None:
      return None
    delta = constant_value(tensor.op.inputs[2])
    if delta is None:
      return None
    return np.arange(start, limit, delta, dtype=tensor.dtype.as_numpy_dtype)
  elif tensor.op.type == "Cast":
    pre_cast = constant_value(tensor.op.inputs[0])
    if pre_cast is None:
      return None
    cast_dtype = dtypes.as_dtype(tensor.op.get_attr("DstT"))
    return pre_cast.astype(cast_dtype.as_numpy_dtype)
  elif tensor.op.type == "Concat":
    dim = constant_value(tensor.op.inputs[0])
    if dim is None:
      return None
    values = []
    for x in tensor.op.inputs[1:]:
      value = constant_value(x)
      if value is None:
        return None
      values.append(value)
    return np.concatenate(values, axis=dim)
  elif tensor.op.type == "ConcatV2":
    dim = constant_value(tensor.op.inputs[-1])
    if dim is None:
      return None
    values = []
    for x in tensor.op.inputs[:-1]:
      value = constant_value(x)
      if value is None:
        return None
      values.append(value)
    return np.concatenate(values, axis=dim)
  elif tensor.op.type == "Pack":
    values = []
    # Some imported GraphDefs have Pack ops with zero inputs. Those are invalid
    # and shouldn't be produced, but to deal sensibly with them here we check
    # and return None.
    if not tensor.op.inputs:
      return None
    # We can't handle axis != 0 Packs at the moment.
    if tensor.op.get_attr("axis") != 0:
      return None
    for x in tensor.op.inputs:
      value = constant_value(x, partial)
      if value is None and not partial:
        return None
      values.append(value)
    return np.array(values)
  elif tensor.op.type == "Fill":
    fill_shape = tensor.shape
    fill_value = constant_value(tensor.op.inputs[1])
    if fill_shape.is_fully_defined() and fill_value is not None:
      return np.full(fill_shape.as_list(), fill_value, dtype=fill_value.dtype)
    else:
      return None
  elif tensor.op.type == "Equal":
    value1 = constant_value(tensor.op.inputs[0])
    if value1 is None:
      return None
    value2 = constant_value(tensor.op.inputs[1])
    if value2 is None:
      return None
    return np.equal(value1, value2)
  elif tensor.op.type == "NotEqual":
    value1 = constant_value(tensor.op.inputs[0])
    if value1 is None:
      return None
    value2 = constant_value(tensor.op.inputs[1])
    if value2 is None:
      return None
    return np.not_equal(value1, value2)
  else:
    return None


def constant_value(tensor, partial=False):  # pylint: disable=invalid-name
  """Returns the constant value of the given tensor, if efficiently calculable.

  This function attempts to partially evaluate the given tensor, and
  returns its value as a numpy ndarray if this succeeds.

  TODO(mrry): Consider whether this function should use a registration
  mechanism like gradients and ShapeFunctions, so that it is easily
  extensible.

  NOTE: If `constant_value(tensor)` returns a non-`None` result, it will no
  longer be possible to feed a different value for `tensor`. This allows the
  result of this function to influence the graph that is constructed, and
  permits static shape optimizations.

  Args:
    tensor: The Tensor to be evaluated.
    partial: If True, the returned numpy array is allowed to have partially
      evaluated values. Values that can't be evaluated will be None.

  Returns:
    A numpy ndarray containing the constant value of the given `tensor`,
    or None if it cannot be calculated.

  Raises:
    TypeError: if tensor is not an ops.Tensor.
  """
  if isinstance(tensor, ops.EagerTensor):
    return tensor.numpy()
  ret = _ConstantValue(tensor, partial)
  if ret is not None:
    # The caller may now depend on the constant value of `tensor`, so we
    # conservatively prevent it from being fed.
    tensor.graph.prevent_feeding(tensor)
  return ret


def constant_value_as_shape(tensor):  # pylint: disable=invalid-name
  """A version of `constant_value()` that returns a `TensorShape`.

  This version should be used when a constant tensor value is
  interpreted as a (possibly partial) shape, e.g. in the shape
  function for `tf.reshape()`. By explicitly requesting a
  `TensorShape` as the return value, it is possible to represent
  unknown dimensions; by contrast, `constant_value()` is
  all-or-nothing.

  Args:
    tensor: The rank-0 or rank-1 Tensor to be evaluated.

  Returns:
    A `TensorShape` based on the constant value of the given `tensor`.

  Raises:
    ValueError: If the shape is rank-0 and is not statically known to be -1.
  """
  if isinstance(tensor, ops.EagerTensor):
    return tensor_shape.as_shape(
        [dim if dim != -1 else None for dim in tensor.numpy()])

  if tensor.get_shape().ndims == 0:
    value = constant_value(tensor)
    if value is None:
      raise ValueError(
          "Received a scalar with unknown value as shape; require a statically "
          "known scalar with value '-1' to describe an unknown shape.")
    if value != -1:
      raise ValueError(
          "Received a scalar value '%s' as shape; require a statically known "
          "scalar with value '-1' to describe an unknown shape." % value)
    return tensor_shape.unknown_shape()

  shape = tensor.get_shape().with_rank(1)
  if shape == [0]:
    return tensor_shape.scalar()
  elif tensor.op.type == "Shape":
    return tensor.op.inputs[0].get_shape()
  elif tensor.op.type == "Pack":
    ret = tensor_shape.scalar()  # Empty list.
    # Since we expect rank 1 inputs, Pack's axis must be zero, otherwise it
    # would not be rank 1.
    assert tensor.op.get_attr("axis") == 0
    for pack_input in tensor.op.inputs:
      # `pack_input` must be a scalar. Attempt to evaluate it, and append it
      # to `ret`.
      pack_input_val = constant_value(pack_input)
      if pack_input_val is None or pack_input_val < 0:
        new_dim = tensor_shape.Dimension(None)
      else:
        new_dim = tensor_shape.Dimension(pack_input_val)
      ret = ret.concatenate([new_dim])
    return ret
  elif tensor.op.type == "Concat":
    # We assume that `tensor.op.inputs[0]` evaluates to 0, as this is
    # the only legal value when concatenating vectors, and it will
    # have been checked by a previous shape function.
    ret = tensor_shape.scalar()  # Empty list.
    for concat_input in tensor.op.inputs[1:]:
      # `concat_input` must be a vector. Attempt to evaluate it as a shape,
      # and concatenate it with `ret`.
      ret = ret.concatenate(constant_value_as_shape(concat_input))
    return ret
  elif tensor.op.type == "ConcatV2":
    # We assume that `tensor.op.inputs[-1]` evaluates to 0, as this is
    # the only legal value when concatenating vectors, and it will
    # have been checked by a previous shape function.
    ret = tensor_shape.scalar()  # Empty list.
    for concat_input in tensor.op.inputs[:-1]:
      # `concat_input` must be a vector. Attempt to evaluate it as a shape,
      # and concatenate it with `ret`.
      ret = ret.concatenate(constant_value_as_shape(concat_input))
    return ret
  elif tensor.op.type == "StridedSlice":
    try:
      begin = constant_value(tensor.op.inputs[1])
      end = constant_value(tensor.op.inputs[2])
      strides = constant_value(tensor.op.inputs[3])
      if begin is not None and end is not None and strides is not None:
        begin = begin[0]
        end = end[0]
        strides = strides[0]
        begin_mask = tensor.op.get_attr("begin_mask")
        if begin_mask == 1:
          begin = None
        end_mask = tensor.op.get_attr("end_mask")
        if end_mask == 1:
          end = None

        ellipsis_mask = tensor.op.get_attr("ellipsis_mask")
        new_axis_mask = tensor.op.get_attr("new_axis_mask")
        shrink_axis_mask = tensor.op.get_attr("shrink_axis_mask")
        valid_attributes = (not ellipsis_mask and not new_axis_mask and
                            not shrink_axis_mask and (not begin_mask or
                                                      (begin_mask == 1)) and
                            (not end_mask or (end_mask == 1)))
        if valid_attributes:  # additional inputs not supported
          prev = constant_value_as_shape(tensor.op.inputs[0])
          prev = prev[begin:end:strides]
          ret = tensor_shape.TensorShape(prev)
          return ret

    except ValueError:  # Could come from get_attr or slicing prev.
      pass
    except TypeError:  # Could come from slicing prev.
      pass

  ret = tensor_shape.unknown_shape(shape.dims[0].value)
  value = constant_value(tensor)
  if value is not None:
    ret = ret.merge_with(
        tensor_shape.TensorShape([d if d >= 0 else None for d in value]))
  return ret


def is_tensor(x):  # pylint: disable=invalid-name
  """Check whether `x` is of tensor type.

  Check whether an object is a tensor. This check is equivalent to calling
  `isinstance(x, (tf.Tensor, tf.SparseTensor, tf.Variable))` and also checks
  if all the component variables of a MirroredVariable or a ReplicaLocalVariable
  are tensors.

  Args:
    x: A python object to check.

  Returns:
    `True` if `x` is a tensor, `False` if not.
  """
  return (isinstance(x, ops._TensorLike) or ops.is_dense_tensor_like(x) or  # pylint: disable=protected-access
          (hasattr(x, "is_tensor_like") and x.is_tensor_like))
