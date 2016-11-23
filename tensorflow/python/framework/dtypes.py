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

"""Library of dtypes (Tensor element types)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.framework import types_pb2


class DType(object):
  """Represents the type of the elements in a `Tensor`.

  The following `DType` objects are defined:

  * `tf.float16`: 16-bit half-precision floating-point.
  * `tf.float32`: 32-bit single-precision floating-point.
  * `tf.float64`: 64-bit double-precision floating-point.
  * `tf.bfloat16`: 16-bit truncated floating-point.
  * `tf.complex64`: 64-bit single-precision complex.
  * `tf.complex128`: 128-bit double-precision complex.
  * `tf.int8`: 8-bit signed integer.
  * `tf.uint8`: 8-bit unsigned integer.
  * `tf.uint16`: 16-bit unsigned integer.
  * `tf.int16`: 16-bit signed integer.
  * `tf.int32`: 32-bit signed integer.
  * `tf.int64`: 64-bit signed integer.
  * `tf.bool`: Boolean.
  * `tf.string`: String.
  * `tf.qint8`: Quantized 8-bit signed integer.
  * `tf.quint8`: Quantized 8-bit unsigned integer.
  * `tf.qint16`: Quantized 16-bit signed integer.
  * `tf.quint16`: Quantized 16-bit unsigned integer.
  * `tf.qint32`: Quantized 32-bit signed integer.
  * `tf.resource`: Handle to a mutable resource.

  In addition, variants of these types with the `_ref` suffix are
  defined for reference-typed tensors.

  The `tf.as_dtype()` function converts numpy types and string type
  names to a `DType` object.

  @@is_compatible_with
  @@name
  @@base_dtype
  @@real_dtype
  @@is_floating
  @@is_complex
  @@is_integer
  @@is_quantized
  @@is_unsigned

  @@as_numpy_dtype
  @@as_datatype_enum

  @@limits
  """

  def __init__(self, type_enum):
    """Creates a new `DataType`.

    NOTE(mrry): In normal circumstances, you should not need to
    construct a `DataType` object directly. Instead, use the
    `tf.as_dtype()` function.

    Args:
      type_enum: A `types_pb2.DataType` enum value.

    Raises:
      TypeError: If `type_enum` is not a value `types_pb2.DataType`.

    """
    # TODO(mrry): Make the necessary changes (using __new__) to ensure
    # that calling this returns one of the interned values.
    type_enum = int(type_enum)
    if (type_enum not in types_pb2.DataType.values()
        or type_enum == types_pb2.DT_INVALID):
      raise TypeError(
          "type_enum is not a valid types_pb2.DataType: %s" % type_enum)
    self._type_enum = type_enum

  @property
  def _is_ref_dtype(self):
    """Returns `True` if this `DType` represents a reference type."""
    return self._type_enum > 100

  @property
  def _as_ref(self):
    """Returns a reference `DType` based on this `DType`."""
    if self._is_ref_dtype:
      return self
    else:
      return _INTERN_TABLE[self._type_enum + 100]

  @property
  def base_dtype(self):
    """Returns a non-reference `DType` based on this `DType`."""
    if self._is_ref_dtype:
      return _INTERN_TABLE[self._type_enum - 100]
    else:
      return self

  @property
  def real_dtype(self):
    """Returns the dtype correspond to this dtype's real part."""
    base = self.base_dtype
    if base == complex64:
      return float32
    elif base == complex128:
      return float64
    else:
      return self

  @property
  def is_numpy_compatible(self):
    return (self._type_enum != types_pb2.DT_RESOURCE and
            self._type_enum != types_pb2.DT_RESOURCE_REF)

  @property
  def as_numpy_dtype(self):
    """Returns a `numpy.dtype` based on this `DType`."""
    return _TF_TO_NP[self._type_enum]

  @property
  def as_datatype_enum(self):
    """Returns a `types_pb2.DataType` enum value based on this `DType`."""
    return self._type_enum

  @property
  def is_integer(self):
    """Returns whether this is a (non-quantized) integer type."""
    return (self.is_numpy_compatible and not self.is_quantized and
            issubclass(self.as_numpy_dtype, np.integer))

  @property
  def is_floating(self):
    """Returns whether this is a (non-quantized, real) floating point type."""
    return self.is_numpy_compatible and issubclass(self.as_numpy_dtype,
                                                   np.floating)

  @property
  def is_complex(self):
    """Returns whether this is a complex floating point type."""
    return self.base_dtype in (complex64, complex128)

  @property
  def is_quantized(self):
    """Returns whether this is a quantized data type."""
    return self.base_dtype in [qint8, quint8, qint16, quint16, qint32, bfloat16]

  @property
  def is_unsigned(self):
    """Returns whether this type is unsigned.

    Non-numeric, unordered, and quantized types are not considered unsigned, and
    this function returns `False`.

    Returns:
      Whether a `DType` is unsigned.
    """
    try:
      return self.min == 0
    except TypeError:
      return False

  @property
  def min(self):
    """Returns the minimum representable value in this data type.

    Raises:
      TypeError: if this is a non-numeric, unordered, or quantized type.

    """
    if (self.is_quantized or self.base_dtype in
        (bool, string, complex64, complex128)):
      raise TypeError("Cannot find minimum value of %s." % self)

    # there is no simple way to get the min value of a dtype, we have to check
    # float and int types separately
    try:
      return np.finfo(self.as_numpy_dtype()).min
    except:  # bare except as possible raises by finfo not documented
      try:
        return np.iinfo(self.as_numpy_dtype()).min
      except:
        raise TypeError("Cannot find minimum value of %s." % self)

  @property
  def max(self):
    """Returns the maximum representable value in this data type.

    Raises:
      TypeError: if this is a non-numeric, unordered, or quantized type.

    """
    if (self.is_quantized or self.base_dtype in
        (bool, string, complex64, complex128)):
      raise TypeError("Cannot find maximum value of %s." % self)

    # there is no simple way to get the max value of a dtype, we have to check
    # float and int types separately
    try:
      return np.finfo(self.as_numpy_dtype()).max
    except:  # bare except as possible raises by finfo not documented
      try:
        return np.iinfo(self.as_numpy_dtype()).max
      except:
        raise TypeError("Cannot find maximum value of %s." % self)

  @property
  def limits(self, clip_negative=True):
    """Return intensity limits, i.e. (min, max) tuple, of the dtype.
    Args:
      clip_negative : bool, optional
          If True, clip the negative range (i.e. return 0 for min intensity)
          even if the image dtype allows negative values.
    Returns
      min, max : tuple
        Lower and upper intensity limits.
    """
    min, max = dtype_range[self.as_numpy_dtype]
    if clip_negative:
      min = 0
    return min, max

  def is_compatible_with(self, other):
    """Returns True if the `other` DType will be converted to this DType.

    The conversion rules are as follows:

    ```python
    DType(T)       .is_compatible_with(DType(T))        == True
    DType(T)       .is_compatible_with(DType(T).as_ref) == True
    DType(T).as_ref.is_compatible_with(DType(T))        == False
    DType(T).as_ref.is_compatible_with(DType(T).as_ref) == True
    ```

    Args:
      other: A `DType` (or object that may be converted to a `DType`).

    Returns:
      True if a Tensor of the `other` `DType` will be implicitly converted to
      this `DType`.
    """
    other = as_dtype(other)
    return self._type_enum in (
        other.as_datatype_enum, other.base_dtype.as_datatype_enum)

  def __eq__(self, other):
    """Returns True iff this DType refers to the same type as `other`."""
    if other is None:
      return False
    try:
      dtype = as_dtype(other).as_datatype_enum
      return self._type_enum == dtype  # pylint: disable=protected-access
    except TypeError:
      return False

  def __ne__(self, other):
    """Returns True iff self != other."""
    return not self.__eq__(other)

  @property
  def name(self):
    """Returns the string name for this `DType`."""
    return _TYPE_TO_STRING[self._type_enum]

  def __str__(self):
    return "<dtype: %r>" % self.name

  def __repr__(self):
    return "tf." + self.name

  def __hash__(self):
    return self._type_enum

  @property
  def size(self):
    return np.dtype(self.as_numpy_dtype).itemsize

# Define data type range of numpy dtype
dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.uint8: (0, 255),
               np.uint16: (0, 65535),
               np.int8: (-128, 127),
               np.int16: (-32768, 32767),
               np.int64: (-2**63, 2**63 - 1),
               np.uint64: (0, 2**64 - 1),
               np.int32: (-2**31, 2**31 - 1),
               np.uint32: (0, 2**32 - 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}

# Define standard wrappers for the types_pb2.DataType enum.
resource = DType(types_pb2.DT_RESOURCE)
float16 = DType(types_pb2.DT_HALF)
half = float16
float32 = DType(types_pb2.DT_FLOAT)
float64 = DType(types_pb2.DT_DOUBLE)
double = float64
int32 = DType(types_pb2.DT_INT32)
uint8 = DType(types_pb2.DT_UINT8)
uint16 = DType(types_pb2.DT_UINT16)
int16 = DType(types_pb2.DT_INT16)
int8 = DType(types_pb2.DT_INT8)
string = DType(types_pb2.DT_STRING)
complex64 = DType(types_pb2.DT_COMPLEX64)
complex128 = DType(types_pb2.DT_COMPLEX128)
int64 = DType(types_pb2.DT_INT64)
bool = DType(types_pb2.DT_BOOL)
qint8 = DType(types_pb2.DT_QINT8)
quint8 = DType(types_pb2.DT_QUINT8)
qint16 = DType(types_pb2.DT_QINT16)
quint16 = DType(types_pb2.DT_QUINT16)
qint32 = DType(types_pb2.DT_QINT32)
resource_ref = DType(types_pb2.DT_RESOURCE_REF)
bfloat16 = DType(types_pb2.DT_BFLOAT16)
float16_ref = DType(types_pb2.DT_HALF_REF)
half_ref = float16_ref
float32_ref = DType(types_pb2.DT_FLOAT_REF)
float64_ref = DType(types_pb2.DT_DOUBLE_REF)
double_ref = float64_ref
int32_ref = DType(types_pb2.DT_INT32_REF)
uint8_ref = DType(types_pb2.DT_UINT8_REF)
uint16_ref = DType(types_pb2.DT_UINT16_REF)
int16_ref = DType(types_pb2.DT_INT16_REF)
int8_ref = DType(types_pb2.DT_INT8_REF)
string_ref = DType(types_pb2.DT_STRING_REF)
complex64_ref = DType(types_pb2.DT_COMPLEX64_REF)
complex128_ref = DType(types_pb2.DT_COMPLEX128_REF)
int64_ref = DType(types_pb2.DT_INT64_REF)
bool_ref = DType(types_pb2.DT_BOOL_REF)
qint8_ref = DType(types_pb2.DT_QINT8_REF)
quint8_ref = DType(types_pb2.DT_QUINT8_REF)
qint16_ref = DType(types_pb2.DT_QINT16_REF)
quint16_ref = DType(types_pb2.DT_QUINT16_REF)
qint32_ref = DType(types_pb2.DT_QINT32_REF)
bfloat16_ref = DType(types_pb2.DT_BFLOAT16_REF)


# Maintain an intern table so that we don't have to create a large
# number of small objects.
_INTERN_TABLE = {
    types_pb2.DT_HALF: float16,
    types_pb2.DT_FLOAT: float32,
    types_pb2.DT_DOUBLE: float64,
    types_pb2.DT_INT32: int32,
    types_pb2.DT_UINT8: uint8,
    types_pb2.DT_UINT16: uint16,
    types_pb2.DT_INT16: int16,
    types_pb2.DT_INT8: int8,
    types_pb2.DT_STRING: string,
    types_pb2.DT_COMPLEX64: complex64,
    types_pb2.DT_COMPLEX128: complex128,
    types_pb2.DT_INT64: int64,
    types_pb2.DT_BOOL: bool,
    types_pb2.DT_QINT8: qint8,
    types_pb2.DT_QUINT8: quint8,
    types_pb2.DT_QINT16: qint16,
    types_pb2.DT_QUINT16: quint16,
    types_pb2.DT_QINT32: qint32,
    types_pb2.DT_BFLOAT16: bfloat16,
    types_pb2.DT_RESOURCE: resource,
    types_pb2.DT_HALF_REF: float16_ref,
    types_pb2.DT_FLOAT_REF: float32_ref,
    types_pb2.DT_DOUBLE_REF: float64_ref,
    types_pb2.DT_INT32_REF: int32_ref,
    types_pb2.DT_UINT8_REF: uint8_ref,
    types_pb2.DT_UINT16_REF: uint16_ref,
    types_pb2.DT_INT16_REF: int16_ref,
    types_pb2.DT_INT8_REF: int8_ref,
    types_pb2.DT_STRING_REF: string_ref,
    types_pb2.DT_COMPLEX64_REF: complex64_ref,
    types_pb2.DT_COMPLEX128_REF: complex128_ref,
    types_pb2.DT_INT64_REF: int64_ref,
    types_pb2.DT_BOOL_REF: bool_ref,
    types_pb2.DT_QINT8_REF: qint8_ref,
    types_pb2.DT_QUINT8_REF: quint8_ref,
    types_pb2.DT_QINT16_REF: qint16_ref,
    types_pb2.DT_QUINT16_REF: quint16_ref,
    types_pb2.DT_QINT32_REF: qint32_ref,
    types_pb2.DT_BFLOAT16_REF: bfloat16_ref,
    types_pb2.DT_RESOURCE_REF: resource_ref,
}


# Standard mappings between types_pb2.DataType values and string names.
_TYPE_TO_STRING = {
    types_pb2.DT_HALF: "float16",
    types_pb2.DT_FLOAT: "float32",
    types_pb2.DT_DOUBLE: "float64",
    types_pb2.DT_INT32: "int32",
    types_pb2.DT_UINT8: "uint8",
    types_pb2.DT_UINT16: "uint16",
    types_pb2.DT_INT16: "int16",
    types_pb2.DT_INT8: "int8",
    types_pb2.DT_STRING: "string",
    types_pb2.DT_COMPLEX64: "complex64",
    types_pb2.DT_COMPLEX128: "complex128",
    types_pb2.DT_INT64: "int64",
    types_pb2.DT_BOOL: "bool",
    types_pb2.DT_QINT8: "qint8",
    types_pb2.DT_QUINT8: "quint8",
    types_pb2.DT_QINT16: "qint16",
    types_pb2.DT_QUINT16: "quint16",
    types_pb2.DT_QINT32: "qint32",
    types_pb2.DT_BFLOAT16: "bfloat16",
    types_pb2.DT_RESOURCE: "resource",
    types_pb2.DT_HALF_REF: "float16_ref",
    types_pb2.DT_FLOAT_REF: "float32_ref",
    types_pb2.DT_DOUBLE_REF: "float64_ref",
    types_pb2.DT_INT32_REF: "int32_ref",
    types_pb2.DT_UINT8_REF: "uint8_ref",
    types_pb2.DT_UINT16_REF: "uint16_ref",
    types_pb2.DT_INT16_REF: "int16_ref",
    types_pb2.DT_INT8_REF: "int8_ref",
    types_pb2.DT_STRING_REF: "string_ref",
    types_pb2.DT_COMPLEX64_REF: "complex64_ref",
    types_pb2.DT_COMPLEX128_REF: "complex128_ref",
    types_pb2.DT_INT64_REF: "int64_ref",
    types_pb2.DT_BOOL_REF: "bool_ref",
    types_pb2.DT_QINT8_REF: "qint8_ref",
    types_pb2.DT_QUINT8_REF: "quint8_ref",
    types_pb2.DT_QINT16_REF: "qint16_ref",
    types_pb2.DT_QUINT16_REF: "quint16_ref",
    types_pb2.DT_QINT32_REF: "qint32_ref",
    types_pb2.DT_BFLOAT16_REF: "bfloat16_ref",
    types_pb2.DT_RESOURCE_REF: "resource_ref",
}
_STRING_TO_TF = {value: _INTERN_TABLE[key]
                 for key, value in _TYPE_TO_STRING.items()}
# Add non-canonical aliases.
_STRING_TO_TF["half"] = float16
_STRING_TO_TF["half_ref"] = float16_ref
_STRING_TO_TF["float"] = float32
_STRING_TO_TF["float_ref"] = float32_ref
_STRING_TO_TF["double"] = float64
_STRING_TO_TF["double_ref"] = float64_ref


# Numpy representation for quantized dtypes.
#
# These are magic strings that are used in the swig wrapper to identify
# quantized types.
# TODO(mrry,keveman): Investigate Numpy type registration to replace this
# hard-coding of names.
_np_qint8 = np.dtype([("qint8", np.int8, 1)])
_np_quint8 = np.dtype([("quint8", np.uint8, 1)])
_np_qint16 = np.dtype([("qint16", np.int16, 1)])
_np_quint16 = np.dtype([("quint16", np.uint16, 1)])
_np_qint32 = np.dtype([("qint32", np.int32, 1)])

# Standard mappings between types_pb2.DataType values and numpy.dtypes.
_NP_TO_TF = frozenset([
    (np.float16, float16),
    (np.float32, float32),
    (np.float64, float64),
    (np.int32, int32),
    (np.int64, int64),
    (np.uint8, uint8),
    (np.uint16, uint16),
    (np.int16, int16),
    (np.int8, int8),
    (np.complex64, complex64),
    (np.complex128, complex128),
    (np.object, string),
    (np.bool, bool),
    (_np_qint8, qint8),
    (_np_quint8, quint8),
    (_np_qint16, qint16),
    (_np_quint16, quint16),
    (_np_qint32, qint32),
    # NOTE(touts): Intentionally no way to feed a DT_BFLOAT16.
])
_TF_TO_NP = {
    types_pb2.DT_HALF: np.float16,
    types_pb2.DT_FLOAT: np.float32,
    types_pb2.DT_DOUBLE: np.float64,
    types_pb2.DT_INT32: np.int32,
    types_pb2.DT_UINT8: np.uint8,
    types_pb2.DT_UINT16: np.uint16,
    types_pb2.DT_INT16: np.int16,
    types_pb2.DT_INT8: np.int8,
    # NOTE(touts): For strings we use np.object as it supports variable length
    # strings.
    types_pb2.DT_STRING: np.object,
    types_pb2.DT_COMPLEX64: np.complex64,
    types_pb2.DT_COMPLEX128: np.complex128,
    types_pb2.DT_INT64: np.int64,
    types_pb2.DT_BOOL: np.bool,
    types_pb2.DT_QINT8: _np_qint8,
    types_pb2.DT_QUINT8: _np_quint8,
    types_pb2.DT_QINT16: _np_qint16,
    types_pb2.DT_QUINT16: _np_quint16,
    types_pb2.DT_QINT32: _np_qint32,
    types_pb2.DT_BFLOAT16: np.uint16,

    # Ref types
    types_pb2.DT_HALF_REF: np.float16,
    types_pb2.DT_FLOAT_REF: np.float32,
    types_pb2.DT_DOUBLE_REF: np.float64,
    types_pb2.DT_INT32_REF: np.int32,
    types_pb2.DT_UINT8_REF: np.uint8,
    types_pb2.DT_UINT16_REF: np.uint16,
    types_pb2.DT_INT16_REF: np.int16,
    types_pb2.DT_INT8_REF: np.int8,
    types_pb2.DT_STRING_REF: np.object,
    types_pb2.DT_COMPLEX64_REF: np.complex64,
    types_pb2.DT_COMPLEX128_REF: np.complex128,
    types_pb2.DT_INT64_REF: np.int64,
    types_pb2.DT_BOOL_REF: np.bool,
    types_pb2.DT_QINT8_REF: _np_qint8,
    types_pb2.DT_QUINT8_REF: _np_quint8,
    types_pb2.DT_QINT16_REF: _np_qint16,
    types_pb2.DT_QUINT16_REF: _np_quint16,
    types_pb2.DT_QINT32_REF: _np_qint32,
    types_pb2.DT_BFLOAT16_REF: np.uint16,
}


QUANTIZED_DTYPES = frozenset(
    [qint8, quint8, qint16, quint16, qint32, qint8_ref, quint8_ref, qint16_ref,
     quint16_ref, qint32_ref])


def as_dtype(type_value):
  """Converts the given `type_value` to a `DType`.

  Args:
    type_value: A value that can be converted to a `tf.DType`
      object. This may currently be a `tf.DType` object, a
      [`DataType` enum](https://www.tensorflow.org/code/tensorflow/core/framework/types.proto),
      a string type name, or a `numpy.dtype`.

  Returns:
    A `DType` corresponding to `type_value`.

  Raises:
    TypeError: If `type_value` cannot be converted to a `DType`.
  """
  if isinstance(type_value, DType):
    return type_value

  try:
    return _INTERN_TABLE[type_value]
  except KeyError:
    pass

  try:
    return _STRING_TO_TF[type_value]
  except KeyError:
    pass

  if isinstance(type_value, np.dtype):
    # The numpy dtype for strings is variable length. We can not compare
    # dtype with a single constant (np.string does not exist) to decide
    # dtype is a "string" type. We need to compare the dtype.type to be
    # sure it's a string type.
    if type_value.type == np.string_ or type_value.type == np.unicode_:
      return string

  for key, val in _NP_TO_TF:
    try:
      if key == type_value:
        return val
    except TypeError as e:
      raise TypeError("Cannot convert {} to a dtype. {}".format(type_value, e))

  raise TypeError(
      "Cannot convert value %r to a TensorFlow DType." % type_value)
