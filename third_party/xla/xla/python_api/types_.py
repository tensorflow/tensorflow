# Copyright 2018 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================
"""Utilities for XLA-specific Python types."""

import collections

import ml_dtypes as _md  # Avoids becoming a part of public Tensorflow API.
import numpy as _np  # Avoids becoming a part of public Tensorflow API.

from local_xla.xla import xla_data_pb2

# Records correspondence between a XLA primitive type and Python/Numpy types.
#
# primitive_type: value of type xla_data_pb2.PrimitiveType
# numpy_dtype: corresponding Numpy "dtype" (like np.float32)
# literal_field_name: name of the field in the LiteralProto message elements
# of this type go into.
# literal_field_type: type of the field named 'literal_field_name'.
#
# TODO(eliben): figure out how to avoid knowing the extra Python type and the
# astype cast when writing into Literals.
TypeConversionRecord = collections.namedtuple('TypeConversionRecord', [
    'primitive_type', 'numpy_dtype', 'literal_field_name', 'literal_field_type'
])

# Maps from XLA primitive types to TypeConversionRecord.
MAP_XLA_TYPE_TO_RECORD = {
    xla_data_pb2.BF16: TypeConversionRecord(
        primitive_type=xla_data_pb2.BF16,
        numpy_dtype=_md.bfloat16,
        literal_field_name='bf16s',
        literal_field_type=float,
    ),
    xla_data_pb2.F16: TypeConversionRecord(
        primitive_type=xla_data_pb2.F16,
        numpy_dtype=_np.float16,
        literal_field_name='f16s',
        literal_field_type=float,
    ),
    xla_data_pb2.F32: TypeConversionRecord(
        primitive_type=xla_data_pb2.F32,
        numpy_dtype=_np.float32,
        literal_field_name='f32s',
        literal_field_type=float,
    ),
    xla_data_pb2.F64: TypeConversionRecord(
        primitive_type=xla_data_pb2.F64,
        numpy_dtype=_np.float64,
        literal_field_name='f64s',
        literal_field_type=float,
    ),
    xla_data_pb2.S8: TypeConversionRecord(
        primitive_type=xla_data_pb2.S8,
        numpy_dtype=_np.int8,
        literal_field_name='s8s',
        literal_field_type=int,
    ),
    xla_data_pb2.S16: TypeConversionRecord(
        primitive_type=xla_data_pb2.S16,
        numpy_dtype=_np.int16,
        literal_field_name='s16s',
        literal_field_type=int,
    ),
    xla_data_pb2.S32: TypeConversionRecord(
        primitive_type=xla_data_pb2.S32,
        numpy_dtype=_np.int32,
        literal_field_name='s32s',
        literal_field_type=int,
    ),
    xla_data_pb2.S64: TypeConversionRecord(
        primitive_type=xla_data_pb2.S64,
        numpy_dtype=_np.int64,
        literal_field_name='s64s',
        literal_field_type=int,
    ),
    xla_data_pb2.U8: TypeConversionRecord(
        primitive_type=xla_data_pb2.U8,
        numpy_dtype=_np.uint8,
        literal_field_name='s8s',
        literal_field_type=int,
    ),
    xla_data_pb2.U16: TypeConversionRecord(
        primitive_type=xla_data_pb2.U16,
        numpy_dtype=_np.uint16,
        literal_field_name='s16s',
        literal_field_type=int,
    ),
    xla_data_pb2.U32: TypeConversionRecord(
        primitive_type=xla_data_pb2.U32,
        numpy_dtype=_np.uint32,
        literal_field_name='s32s',
        literal_field_type=int,
    ),
    xla_data_pb2.U64: TypeConversionRecord(
        primitive_type=xla_data_pb2.U64,
        numpy_dtype=_np.uint64,
        literal_field_name='s64s',
        literal_field_type=int,
    ),
    xla_data_pb2.PRED: TypeConversionRecord(
        primitive_type=xla_data_pb2.PRED,
        numpy_dtype=_np.bool_,
        literal_field_name='preds',
        literal_field_type=bool,
    ),
}

# Maps from Numpy dtypes to TypeConversionRecord.
# Note the conversion on the key. Numpy has a known issue wherein dtype hashing
# doesn't work as expected (https://github.com/numpy/numpy/issues/7242). Thus,
# when keying by dtype in this dict, we use the string form of dtypes.
MAP_DTYPE_TO_RECORD = {
    str(_np.dtype(record.numpy_dtype)): record
    for record in MAP_XLA_TYPE_TO_RECORD.values()
}
