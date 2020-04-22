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
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

# Fallback in case fast_tensor_util is not properly compiled.
# pylint: disable=g-import-not-at-top
try:
  from tensorflow.python.framework import fast_tensor_util
  _FAST_TENSOR_UTIL_AVAILABLE = True
except ImportError:
  _FAST_TENSOR_UTIL_AVAILABLE = False
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
    tensor_proto.int_val.extend([x.item()[0] for x in proto_values])

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
    dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32, dtypes.uint8,
    dtypes.int16, dtypes.int8, dtypes.int64, dtypes.qint8, dtypes.quint8,
    dtypes.qint16, dtypes.quint16, dtypes.qint32, dtypes.uint32, dtypes.uint64
])


# pylint: disable=invalid-name
def _check_failed(v):
  # NB. none of the _check_* functions could raise a ValueError, so
  # it is safe to use here.
  raise ValueError(v)


def _check_quantized(values):
  # Cannot rely on `nest` because the leaves are tuples.
  if not isinstance(values, (list, tuple)):
    _check_failed(values)
  if isinstance(values, tuple):
    _ = [_check_int(v) for v in values]
  else:
    _ = [_check_quantized(v) for v in values]


def _generate_isinstance_check(expected_types):
  def inner(values):
    _ = [_check_failed(v) for v in nest.flatten(values)
         if not isinstance(v, expected_types)]
  return inner

_check_int = _generate_isinstance_check(
    (compat.integral_types, tensor_shape.Dimension))
_check_float = _generate_isinstance_check(compat.real_types)
_check_complex = _generate_isinstance_check(compat.complex_types)
_check_str = _generate_isinstance_check(compat.bytes_or_text_types)
_check_bool = _generate_isinstance_check(bool)


def _check_not_tensor(values):
  _ = [_check_failed(v) for v in nest.flatten(values)
       if isinstance(v, ops.Tensor)]
# pylint: enable=invalid-name

_TF_TO_IS_OK = {
    dtypes.bool: _check_bool,
    dtypes.complex128: _check_complex,
    dtypes.complex64: _check_complex,
    dtypes.float16: _check_float,
    dtypes.float32: _check_float,
    dtypes.float64: _check_float,
    dtypes.int16: _check_int,
    dtypes.int32: _check_int,
    dtypes.int64: _check_int,
    dtypes.int8: _check_int,
    dtypes.qint16: _check_quantized,
    dtypes.qint32: _check_quantized,
    dtypes.qint8: _check_quantized,
    dtypes.quint16: _check_quantized,
    dtypes.quint8: _check_quantized,
    dtypes.string: _check_str,
    dtypes.uint16: _check_int,
    dtypes.uint8: _check_int,
    dtypes.uint32: _check_int,
    dtypes.uint64: _check_int,
}


def _AssertCompatible(values, dtype):
  if dtype is None:
    fn = _check_not_tensor
  else:
    try:
      fn = _TF_TO_IS_OK[dtype]
    except KeyError:
      # There isn't a specific fn, so we try to do the best possible.
      if dtype.is_integer:
        fn = _check_int
      elif dtype.is_floating:
        fn = _check_float
      elif dtype.is_complex:
        fn = _check_complex
      elif dtype.is_quantized:
        fn = _check_quantized
      else:
        fn = _check_not_tensor

  try:
    fn(values)
  except ValueError as e:
    [mismatch] = e.args
    if dtype is None:
      raise TypeError("Expected any non-tensor type, got a tensor instead.")
    else:
      raise TypeError("Expected %s, got %s of type '%s' instead." %
                      (dtype.name, repr(mismatch), type(mismatch).__name__))


def _is_array_like(obj):  # pylint: disable=invalid-name
  """Check if a given object is array-like."""
  if isinstance(obj, ops.Tensor) and not isinstance(obj, ops._EagerTensorBase):  # pylint: disable=protected-access
    # Tensor implements __array__ only so it can inform the user that it is not
    # a valid array.
    return False

  # TODO(slebedev): an object could also implement C-level array interface.
  if (callable(getattr(obj, "__array__", None)) or
      isinstance(getattr(obj, "__array_interface__", None), dict)):
    return True

  try:
    memoryview(obj)
  except TypeError:
    return False
  else:
    return not isinstance(obj, bytes)


# pylint: disable=invalid-name
@tf_export("make_tensor_proto")
def make_tensor_proto(values, dtype=None, shape=None, verify_shape=False,
                      allow_broadcast=False):
  """Create a TensorProto.

  In TensorFlow 2.0, representing tensors as protos should no longer be a
  common workflow. That said, this utility function is still useful for
  generating TF Serving request protos:

  ```python
    request = tensorflow_serving.apis.predict_pb2.PredictRequest()
    request.model_spec.name = "my_model"
    request.model_spec.signature_name = "serving_default"
    request.inputs["images"].CopyFrom(tf.make_tensor_proto(X_new))
  ```

  `make_tensor_proto` accepts "values" of a python scalar, a python list, a
  numpy ndarray, or a numpy scalar.

  If "values" is a python scalar or a python list, make_tensor_proto
  first convert it to numpy ndarray. If dtype is None, the
  conversion tries its best to infer the right numpy data
  type. Otherwise, the resulting numpy array has a compatible data
  type with the given dtype.

  In either case above, the numpy ndarray (either the caller provided
  or the auto-converted) must have the compatible type with dtype.

  `make_tensor_proto` then converts the numpy array to a tensor proto.

  If "shape" is None, the resulting tensor proto represents the numpy
  array precisely.

  Otherwise, "shape" specifies the tensor's shape and the numpy array
  can not have more elements than what "shape" specifies.

  Args:
    values:         Values to put in the TensorProto.
    dtype:          Optional tensor_pb2 DataType value.
    shape:          List of integers representing the dimensions of tensor.
    verify_shape:   Boolean that enables verification of a shape of values.
    allow_broadcast:  Boolean that enables allowing scalars and 1 length vector
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

  if _is_array_like(values):
    values = np.asarray(values)

  # We first convert value to a numpy array or scalar.
  if isinstance(values, (np.ndarray, np.generic)):
    if dtype and dtype.is_numpy_compatible:
      nparray = values.astype(dtype.as_numpy_dtype)
    else:
      nparray = values
  else:
    if values is None:
      raise ValueError("None values not supported.")
    # if dtype is provided, forces numpy array to be the type
    # provided if possible.
    if dtype and dtype.is_numpy_compatible:
      np_dt = dtype.as_numpy_dtype
    else:
      np_dt = None
    # If shape is None, numpy.prod returns None when dtype is not set, but
    # raises exception when dtype is set to np.int64
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

  For example:

  ```python
  # Tensor a has shape (2,3)
  a = tf.constant([[1,2,3],[4,5,6]])
  proto_tensor = tf.make_tensor_proto(a)  # convert `tensor a` to a proto tensor
  tf.make_ndarray(proto_tensor) # output: array([[1, 2, 3],
  #                                              [4, 5, 6]], dtype=int32)
  # output has shape (2,3)
  ```

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
  elif tensor.op.type == "Unpack":
    # We can't handle axis != 0 Unpacks at the moment.
    if tensor.op.get_attr("axis") != 0:
      return None
    value = constant_value(tensor.op.inputs[0], partial)
    if value is None:
      return None
    return value[tensor.value_index]
  elif tensor.op.type == "Split":
    dim = constant_value(tensor.op.inputs[0])
    value = constant_value(tensor.op.inputs[1], partial)
    if value is None or dim is None:
      return None
    split = np.split(value, tensor.op.get_attr("num_split"), dim)
    return split[tensor.value_index]
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
  elif tensor.op.type == "StopGradient":
    return constant_value(tensor.op.inputs[0], partial)
  elif tensor.op.type == "Identity":
    return constant_value(tensor.op.inputs[0], partial)
  elif tensor.op.type in ("CheckNumericsV2", "DebugIdentityV2"):
    return constant_value(tensor.op.inputs[0], partial)
  else:
    return None


@tf_export("get_static_value")
def constant_value(tensor, partial=False):  # pylint: disable=invalid-name
  """Returns the constant value of the given tensor, if efficiently calculable.

  This function attempts to partially evaluate the given tensor, and
  returns its value as a numpy ndarray if this succeeds.

  Compatibility(V1): If `constant_value(tensor)` returns a non-`None` result, it
  will no longer be possible to feed a different value for `tensor`. This allows
  the result of this function to influence the graph that is constructed, and
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
  if not is_tensor(tensor):
    return tensor
  if not isinstance(tensor, ops.Tensor):
    return None
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
    return tensor_shape.TensorShape([])
  elif tensor.op.type == "Cast":
    pre_cast = constant_value_as_shape(tensor.op.inputs[0])
    if pre_cast.dims is None:
      # the input to cast has a totally undefined shape; just return that.
      return pre_cast
    cast_dtype = dtypes.as_dtype(tensor.op.get_attr("DstT"))
    if cast_dtype not in (dtypes.int32, dtypes.int64):
      return tensor_shape.unknown_shape(shape.dims[0].value)
    dest_dtype_shape_array = np.array(
        [x if x is not None else -1 for x in pre_cast.as_list()]).astype(
            cast_dtype.as_numpy_dtype)
    return tensor_shape.TensorShape([
        x if x >= 0 else None
        for x in dest_dtype_shape_array])
  elif tensor.op.type == "Shape":
    return tensor.op.inputs[0].get_shape()
  elif tensor.op.type == "Pack":
    ret = tensor_shape.TensorShape([])  # Empty list.
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
    ret = tensor_shape.TensorShape([])  # Empty list.
    for concat_input in tensor.op.inputs[1:]:
      # `concat_input` must be a vector. Attempt to evaluate it as a shape,
      # and concatenate it with `ret`.
      ret = ret.concatenate(constant_value_as_shape(concat_input))
    return ret
  elif tensor.op.type == "ConcatV2":
    # We assume that `tensor.op.inputs[-1]` evaluates to 0, as this is
    # the only legal value when concatenating vectors, and it will
    # have been checked by a previous shape function.
    ret = tensor_shape.TensorShape([])  # Empty list.
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
  elif (tensor.op.type == "Placeholder" and
        tensor.op.graph.building_function and
        hasattr(tensor.op.graph, "internal_captures")):
    # If we are inside a FuncGraph try to lookup the constant value of the
    # corresponding external capture. Note that we only look at captures and
    # not the fed inputs because those can be fed different values in different
    # instantiations of the function call or different iterations of a
    # tf.while_loop.
    for i, capture in enumerate(tensor.op.graph.internal_captures):
      if capture is tensor:
        external_capture = tensor.op.graph.external_captures[i]
        return constant_value_as_shape(external_capture)

  ret = tensor_shape.unknown_shape(shape.dims[0].value)
  value = constant_value(tensor)
  if value is not None:
    ret = ret.merge_with(
        tensor_shape.TensorShape([d if d >= 0 else None for d in value]))
  return ret


# TODO(mdan): Deprecate in favor of more static-friendly types.
@tf_export("is_tensor")
def is_tensor(x):  # pylint: disable=invalid-name
  """Checks whether `x` is a TF-native type that can be passed to many TF ops.

  Use is_tensor to differentiate types that can ingested by TensorFlow ops
  without any conversion (e.g., `tf.Tensor`, `tf.SparseTensor`, and
  `tf.RaggedTensor`) from types that need to be converted into tensors before
  they are ingested (e.g., numpy `ndarray` and Python scalars).

  For example, in the following code block:

  ```python
  if not tf.is_tensor(t):
    t = tf.convert_to_tensor(t)
  return t.dtype
  ```

  we check to make sure that `t` is a tensor (and convert it if not) before
  accessing its `shape` and `dtype`.

  Args:
    x: A python object to check.

  Returns:
    `True` if `x` is a tensor or "tensor-like", `False` if not.
  """
  return (isinstance(x, internal.NativeObject) or
          ops.is_dense_tensor_like(x) or
          getattr(x, "is_tensor_like", False))


def shape_tensor(shape):  # pylint: disable=invalid-name
  """Convert to an int32 or int64 tensor, defaulting to int32 if empty."""
  dtype = None
  if isinstance(shape, (tuple, list)):
    if not shape:
      dtype = dtypes.int32
    else:
      # If there are Dimension objects in the shape, unwrap them. This can be a
      # problem if v1 and v2 TensorShape objects get mixed up in partial
      # conversions, leading to shapes such as (1, 2, Dimension(5)), which are
      # not convertible to Tensors because of mixed content.
      shape = tuple(map(tensor_shape.dimension_value, shape))
  return ops.convert_to_tensor(shape, dtype=dtype, name="shape")


# DO NOT USE: For testing only.
_ENABLE_MAYBE_SET_STATIC_SHAPE = True


def maybe_set_static_shape(tensor, shape):  # pylint: disable=invalid-name
  """Sets the shape of `tensor` to the `shape`'s constant value, if inferrable.

  This is a temporary workaround to fix shape inference across functional op
  boundaries. E.g.

  ```python
  shape = tf.constant([3])
  @tf.function
  def f():
    u = tf.random_uniform(shape)
    return u
  ```

  If we were to rely solely on C++ shape inference, the shape of `u` inside
  `f` would be unknown because C++ shape inference is not aware of the outer
  graph and all it sees is a Placeholder node when backtracing the captured
  tensor for `shape`. `maybe_set_static_shape` computes the static shape value
  of `shape` by traversing the `FuncGraph` boundaries and sets the correct
  shape.

  A longer term solution would be to fix C++ shape inference.

  Args:
    tensor: A tensor.
    shape: A shape tensor.
  """
  if (_ENABLE_MAYBE_SET_STATIC_SHAPE and not context.executing_eagerly() and
      ops.get_default_graph().building_function and
      not tensor.shape.is_fully_defined() and is_tensor(shape)):
    shape = shape_tensor(shape)
    const_shape = constant_value_as_shape(shape)
    tensor.set_shape(const_shape)
