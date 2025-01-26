# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Library of types used for type annotation."""
from tensorflow.python.framework import dtypes as _dtypes


class DTypeAnnotation:
  pass


def _create_dtype_wrapper(name, underlying_dtype: _dtypes.DType):
  return type(name, (DTypeAnnotation,), {"underlying_dtype": underlying_dtype})


BFloat16 = _create_dtype_wrapper("BFloat16", _dtypes.bfloat16)
Bool = _create_dtype_wrapper("Bool", _dtypes.bool)
Complex128 = _create_dtype_wrapper("Complex128", _dtypes.complex128)
Complex64 = _create_dtype_wrapper("Complex64", _dtypes.complex64)
Float8e4m3fn = _create_dtype_wrapper("Float8e4m3fn", _dtypes.float8_e4m3fn)
Float8e5m2 = _create_dtype_wrapper("Float8e5m2", _dtypes.float8_e5m2)
Float8e4m3fnuz = _create_dtype_wrapper(
    "Float8e4m3fnuz", _dtypes.float8_e4m3fnuz
)
Float8e4m3b11fnuz = _create_dtype_wrapper(
    "Float8e4m3b11fnuz", _dtypes.float8_e4m3b11fnuz
)
Float8e5m2fnuz = _create_dtype_wrapper(
    "Float8e5m2fnuz", _dtypes.float8_e5m2fnuz
)
Float16 = _create_dtype_wrapper("Float16", _dtypes.float16)
Float32 = _create_dtype_wrapper("Float32", _dtypes.float32)
Float64 = _create_dtype_wrapper("Float64", _dtypes.float64)
Half = _create_dtype_wrapper("Half", _dtypes.float16)
Int4 = _create_dtype_wrapper("Int4", _dtypes.int4)
Int8 = _create_dtype_wrapper("Int8", _dtypes.int8)
Int16 = _create_dtype_wrapper("Int16", _dtypes.int16)
Int32 = _create_dtype_wrapper("Int32", _dtypes.int32)
Int64 = _create_dtype_wrapper("Int64", _dtypes.int64)
UInt4 = _create_dtype_wrapper("UInt4", _dtypes.uint4)
UInt8 = _create_dtype_wrapper("UInt8", _dtypes.uint8)
UInt16 = _create_dtype_wrapper("UInt16", _dtypes.uint16)
UInt32 = _create_dtype_wrapper("UInt32", _dtypes.uint32)
UInt64 = _create_dtype_wrapper("UInt64", _dtypes.uint64)
QInt8 = _create_dtype_wrapper("QInt8", _dtypes.qint8)
QInt16 = _create_dtype_wrapper("QInt16", _dtypes.qint16)
QInt32 = _create_dtype_wrapper("QInt32", _dtypes.qint32)
QUInt16 = _create_dtype_wrapper("QUInt16", _dtypes.quint16)
QUInt8 = _create_dtype_wrapper("QUInt8", _dtypes.quint8)
Resource = _create_dtype_wrapper("Resource", _dtypes.resource)
String = _create_dtype_wrapper("String", _dtypes.string)
Variant = _create_dtype_wrapper("Variant", _dtypes.variant)
