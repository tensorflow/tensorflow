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
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import compat

# TODO(opensource): Add support for pyx_library in the open-source build.
# For now, we use the slow versions that fast_tensor_util replaces.
# pylint: disable=g-import-not-at-top
try:
  from tensorflow.python.framework import fast_tensor_util
  _FAST_TENSOR_UTIL_AVAILABLE = True
except ImportError:
  _FAST_TENSOR_UTIL_AVAILABLE = False

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
# pylint: enable=g-import-not-at-top


def ExtractBitsFromFloat16(x):
  return np.asscalar(np.asarray(x, dtype=np.float16).view(np.uint16))


def SlowAppendFloat16ArrayToTensorProto(tensor_proto, proto_values):
  tensor_proto.half_val.extend([
      ExtractBitsFromFloat16(x) for x in proto_values])

if _FAST_TENSOR_UTIL_AVAILABLE:
  _NP_TO_APPEND_FN = {
      # TODO(sesse): We should have a
      # fast_tensor_util.AppendFloat16ArrayToTensorProto,
      # but it seems np.float16_t doesn't exist?
      np.float16: SlowAppendFloat16ArrayToTensorProto,
      np.float32: fast_tensor_util.AppendFloat32ArrayToTensorProto,
      np.float64: fast_tensor_util.AppendFloat64ArrayToTensorProto,
      np.int32: fast_tensor_util.AppendInt32ArrayToTensorProto,
      np.int64: fast_tensor_util.AppendInt64ArrayToTensorProto,
      np.uint8: fast_tensor_util.AppendUInt8ArrayToTensorProto,
      np.uint16: fast_tensor_util.AppendUInt16ArrayToTensorProto,
      np.int8: fast_tensor_util.AppendInt8ArrayToTensorProto,
      np.int16: fast_tensor_util.AppendInt16ArrayToTensorProto,
      np.complex64: fast_tensor_util.AppendComplex64ArrayToTensorProto,
      np.complex128: fast_tensor_util.AppendComplex128ArrayToTensorProto,
      np.object: fast_tensor_util.AppendObjectArrayToTensorProto,
      np.bool: fast_tensor_util.AppendBoolArrayToTensorProto,
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
    tensor_proto.float_val.extend([np.asscalar(x) for x in proto_values])

  def SlowAppendFloat64ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.double_val.extend([np.asscalar(x) for x in proto_values])

  def SlowAppendIntArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.int_val.extend([np.asscalar(x) for x in proto_values])

  def SlowAppendQIntArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.int_val.extend([np.asscalar(x[0]) for x in proto_values])

  def SlowAppendInt64ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.int64_val.extend([np.asscalar(x) for x in proto_values])

  def SlowAppendComplex64ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.scomplex_val.extend([np.asscalar(v)
                                      for x in proto_values
                                      for v in [x.real, x.imag]])

  def SlowAppendComplex128ArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.dcomplex_val.extend([np.asscalar(v)
                                      for x in proto_values
                                      for v in [x.real, x.imag]])

  def SlowAppendObjectArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.string_val.extend([compat.as_bytes(x) for x in proto_values])

  def SlowAppendBoolArrayToTensorProto(tensor_proto, proto_values):
    tensor_proto.bool_val.extend([np.asscalar(x) for x in proto_values])

  _NP_TO_APPEND_FN = {
      np.float16: SlowAppendFloat16ArrayToTensorProto,
      np.float32: SlowAppendFloat32ArrayToTensorProto,
      np.float64: SlowAppendFloat64ArrayToTensorProto,
      np.int32: SlowAppendIntArrayToTensorProto,
      np.int64: SlowAppendInt64ArrayToTensorProto,
      np.uint8: SlowAppendIntArrayToTensorProto,
      np.uint16: SlowAppendIntArrayToTensorProto,
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
  if isinstance(nested_strings, list):
    for inner in nested_strings:
      for flattened_string in _FlattenToStrings(inner):
        yield flattened_string
  else:
    yield nested_strings


_TENSOR_CONTENT_TYPES = frozenset([
    dtypes.float32, dtypes.float64, dtypes.int32, dtypes.uint8, dtypes.int16,
    dtypes.int8, dtypes.int64, dtypes.qint8, dtypes.quint8, dtypes.qint16,
    dtypes.quint16, dtypes.qint32,
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


def _FilterInt(v):
  if isinstance(v, (list, tuple)):
    return _FirstNotNone([_FilterInt(x) for x in v])
  return None if isinstance(v, compat.integral_types) else _NotNone(v)


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
    dtypes.bool: _FilterBool,
    dtypes.complex128: _FilterComplex,
    dtypes.complex64: _FilterComplex,
    dtypes.float32: _FilterFloat,
    dtypes.float64: _FilterFloat,
    dtypes.int16: _FilterInt,
    dtypes.int32: _FilterInt,
    dtypes.int64: _FilterInt,
    dtypes.int8: _FilterInt,
    dtypes.qint16: _FilterInt,
    dtypes.qint32: _FilterInt,
    dtypes.qint8: _FilterInt,
    dtypes.quint16: _FilterInt,
    dtypes.quint8: _FilterInt,
    dtypes.string: _FilterStr,
    dtypes.uint16: _FilterInt,
    dtypes.uint8: _FilterInt,
}


def _AssertCompatible(values, dtype):
  fn = _TF_TO_IS_OK.get(dtype, _FilterNotTensor)
  mismatch = fn(values)
  if mismatch is not None:
    if dtype is None:
      raise TypeError("List of Tensors when single Tensor expected")
    else:
      raise TypeError("Expected %s, got %s of type '%s' instead." %
                      (dtype.name, repr(mismatch), type(mismatch).__name__))


def make_tensor_proto(values, dtype=None, shape=None):
  """Create a TensorProto.

  Args:
    values:    Values to put in the TensorProto.
    dtype:     Optional tensor_pb2 DataType value.
    shape:     List of integers representing the dimensions of tensor.

  Returns:
    A TensorProto. Depending on the type, it may contain data in the
    "tensor_content" attribute, which is not directly useful to Python programs.
    To access the values you should convert the proto back to a numpy ndarray
    with tensor_util.MakeNdarray(proto).

  Raises:
    TypeError:  if unsupported types are provided.
    ValueError: if arguments have inappropriate values.

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
  if dtype:
    dtype = dtypes.as_dtype(dtype)

  is_quantized = (dtype in [dtypes.qint8, dtypes.quint8, dtypes.qint16,
                            dtypes.quint16, dtypes.qint32])

  # We first convert value to a numpy array or scalar.
  if isinstance(values, (np.ndarray, np.generic)):
    if dtype:
      nparray = values.astype(dtype.as_numpy_dtype)
    else:
      nparray = values
  else:
    if values is None:
      raise ValueError("None values not supported.")
    # if dtype is provided, forces numpy array to be the type
    # provided if possible.
    np_dt = dtype.as_numpy_dtype if dtype else None
    if np.prod(shape) == 0:
      nparray = np.empty(shape, dtype=np_dt)
    else:
      _AssertCompatible(values, dtype)
      nparray = np.array(values, dtype=np_dt)
      # check to them.
      # We need to pass in quantized values as tuples, so don't apply the shape
      if (list(nparray.shape) != _GetDenseDimensions(values) and
          not is_quantized):
        raise ValueError("""Argument must be a dense tensor: %s"""
                         """ - got shape %s, but wanted %s.""" % (
                             values, list(nparray.shape),
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
    raise TypeError("Incompatible types: %s vs. %s" % (dtype, nparray.dtype))

  # If shape is not given, get the shape from the numpy array.
  if shape is None:
    shape = nparray.shape
    is_same_size = True
    shape_size = nparray.size
  else:
    shape = [int(dim) for dim in shape]
    shape_size = np.prod(shape)
    is_same_size = shape_size == nparray.size

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
    tensor_proto.string_val.extend([compat.as_bytes(x) for x in proto_values])
    return tensor_proto

  # TensorFlow expects C order (a.k.a., eigen row major).
  proto_values = nparray.ravel()

  append_fn = GetNumpyAppendFn(proto_values.dtype)
  if append_fn is None:
    raise TypeError("Element type not supported in TensorProto: %s" %
                    numpy_dtype.name)
  append_fn(tensor_proto, proto_values)

  return tensor_proto


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
  num_elements = np.prod(shape)
  tensor_dtype = dtypes.as_dtype(tensor.dtype)
  dtype = tensor_dtype.as_numpy_dtype

  if tensor.tensor_content:
    return np.fromstring(tensor.tensor_content, dtype=dtype).reshape(shape)
  elif tensor_dtype == dtypes.float32:
    if len(tensor.float_val) == 1:
      return np.repeat(np.array(tensor.float_val[0], dtype=dtype),
                       num_elements).reshape(shape)
    else:
      return np.fromiter(tensor.float_val, dtype=dtype).reshape(shape)
  elif tensor_dtype == dtypes.float64:
    if len(tensor.double_val) == 1:
      return np.repeat(np.array(tensor.double_val[0], dtype=dtype),
                       num_elements).reshape(shape)
    else:
      return np.fromiter(tensor.double_val, dtype=dtype).reshape(shape)
  elif tensor_dtype in [dtypes.int32, dtypes.uint8, dtypes.uint16, dtypes.int16,
                        dtypes.int8, dtypes.qint32, dtypes.quint8, dtypes.qint8,
                        dtypes.qint16, dtypes.quint16, dtypes.bfloat16]:
    if len(tensor.int_val) == 1:
      return np.repeat(np.array(tensor.int_val[0], dtype=dtype),
                       num_elements).reshape(shape)
    else:
      return np.fromiter(tensor.int_val, dtype=dtype).reshape(shape)
  elif tensor_dtype == dtypes.int64:
    if len(tensor.int64_val) == 1:
      return np.repeat(np.array(tensor.int64_val[0], dtype=dtype),
                       num_elements).reshape(shape)
    else:
      return np.fromiter(tensor.int64_val, dtype=dtype).reshape(shape)
  elif tensor_dtype == dtypes.string:
    if len(tensor.string_val) == 1:
      return np.repeat(np.array(tensor.string_val[0], dtype=dtype),
                       num_elements).reshape(shape)
    else:
      return np.array([x for x in tensor.string_val],
                      dtype=dtype).reshape(shape)
  elif tensor_dtype == dtypes.complex64:
    it = iter(tensor.scomplex_val)
    if len(tensor.scomplex_val) == 2:
      return np.repeat(np.array(complex(tensor.scomplex_val[0],
                                        tensor.scomplex_val[1]), dtype=dtype),
                       num_elements).reshape(shape)
    else:
      return np.array([complex(x[0], x[1]) for x in zip(it, it)],
                      dtype=dtype).reshape(shape)
  elif tensor_dtype == dtypes.complex128:
    it = iter(tensor.dcomplex_val)
    if len(tensor.dcomplex_val) == 2:
      return np.repeat(np.array(complex(tensor.dcomplex_val[0],
                                        tensor.dcomplex_val[1]), dtype=dtype),
                       num_elements).reshape(shape)
    else:
      return np.array([complex(x[0], x[1]) for x in zip(it, it)],
                      dtype=dtype).reshape(shape)
  elif tensor_dtype == dtypes.bool:
    if len(tensor.bool_val) == 1:
      return np.repeat(np.array(tensor.bool_val[0], dtype=dtype),
                       num_elements).reshape(shape)
    else:
      return np.fromiter(tensor.bool_val, dtype=dtype).reshape(shape)
  else:
    raise TypeError("Unsupported tensor type: %s" % tensor.dtype)


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


def _ConstantValue(tensor):
  # TODO(touts): Support Variables?
  if not isinstance(tensor, ops.Tensor):
    raise TypeError("tensor is not a Tensor")
  if tensor.op.type == "Const":
    return MakeNdarray(tensor.op.get_attr("value"))
  elif tensor.op.type == "Shape":
    input_shape = tensor.op.inputs[0].get_shape()
    if input_shape.is_fully_defined():
      return np.array([dim.value for dim in input_shape.dims],
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
      return input_shape.ndims
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
  else:
    return None


def constant_value(tensor):
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

  Returns:
    A numpy ndarray containing the constant value of the given `tensor`,
    or None if it cannot be calculated.

  Raises:
    TypeError: if tensor is not an ops.Tensor.
  """
  ret = _ConstantValue(tensor)
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
    tensor: The rank-1 Tensor to be evaluated.

  Returns:
    A `TensorShape` based on the constant value of the given `tensor`.
  """
  shape = tensor.get_shape().with_rank(1)
  if tensor.get_shape() == [0]:
    return tensor_shape.scalar()
  elif tensor.op.type == "Shape":
    return tensor.op.inputs[0].get_shape()
  elif tensor.op.type == "Pack":
    ret = tensor_shape.scalar()  # Empty list.
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
  else:
    ret = tensor_shape.unknown_shape(shape[0].value)
    value = constant_value(tensor)
    if value is not None:
      ret = ret.merge_with(tensor_shape.TensorShape(
          [d if d != -1 else None for d in value]))
    return ret
