/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_KEY_VALUE_SORT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_KEY_VALUE_SORT_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/types.h"

extern "C" {

// 'keys' represents a 3-dimensional shape with dimensions [a, b, c]. The 'b'
// dimension of 'keys' is sorted into ascending order. If 'values_count' is <=
// 0, 'values' and 'values_primitive_type_size_in_bytes' can be nullptr.
// If 'values_count' > 0, they contain exactly 'values_count' many elements.
// Each element of 'values' also represents a 3-dimensional shape with
// dimensions [a, b, c], and the size of the primitive type of the i-th shape
// has exactly 'values_primitive_type_size_in_bytes[i]' bytes. The elements in
// each 'values' shape are reordered in such a way that if the element at index
// 'i' in 'keys' was moved to index 'j', the element at index 'i' in a 'values'
// shape is also moved to index 'j' (which means that the same elements
// correspond to each other as before).
extern void __xla_cpu_runtime_KeyValueSortPRED(
    bool* keys, tensorflow::int64 a, tensorflow::int64 b, tensorflow::int64 c,
    char** values, tensorflow::int32 values_count,
    tensorflow::int32* values_primitive_type_size_in_bytes);

extern void __xla_cpu_runtime_KeyValueSortS8(
    tensorflow::int8* keys, tensorflow::int64 a, tensorflow::int64 b,
    tensorflow::int64 c, char** values, tensorflow::int32 values_count,
    tensorflow::int32* values_primitive_type_size_in_bytes);

extern void __xla_cpu_runtime_KeyValueSortU8(
    tensorflow::uint8* keys, tensorflow::int64 a, tensorflow::int64 b,
    tensorflow::int64 c, char** values, tensorflow::int32 values_count,
    tensorflow::int32* values_primitive_type_size_in_bytes);

extern void __xla_cpu_runtime_KeyValueSortS16(
    tensorflow::int16* keys, tensorflow::int64 a, tensorflow::int64 b,
    tensorflow::int64 c, char** values, tensorflow::int32 values_count,
    tensorflow::int32* values_primitive_type_size_in_bytes);

extern void __xla_cpu_runtime_KeyValueSortU16(
    tensorflow::uint16* keys, tensorflow::int64 a, tensorflow::int64 b,
    tensorflow::int64 c, char** values, tensorflow::int32 values_count,
    tensorflow::int32* values_primitive_type_size_in_bytes);

extern void __xla_cpu_runtime_KeyValueSortF16(
    Eigen::half* keys, tensorflow::int64 a, tensorflow::int64 b,
    tensorflow::int64 c, char** values, tensorflow::int32 values_count,
    tensorflow::int32* values_primitive_type_size_in_bytes);

extern void __xla_cpu_runtime_KeyValueSortS32(
    tensorflow::int32* keys, tensorflow::int64 a, tensorflow::int64 b,
    tensorflow::int64 c, char** values, tensorflow::int32 values_count,
    tensorflow::int32* values_primitive_type_size_in_bytes);

extern void __xla_cpu_runtime_KeyValueSortU32(
    tensorflow::uint32* keys, tensorflow::int64 a, tensorflow::int64 b,
    tensorflow::int64 c, char** values, tensorflow::int32 values_count,
    tensorflow::int32* values_primitive_type_size_in_bytes);

extern void __xla_cpu_runtime_KeyValueSortF32(
    float* keys, tensorflow::int64 a, tensorflow::int64 b, tensorflow::int64 c,
    char** values, tensorflow::int32 values_count,
    tensorflow::int32* values_primitive_type_size_in_bytes);

extern void __xla_cpu_runtime_KeyValueSortS64(
    tensorflow::int64* keys, tensorflow::int64 a, tensorflow::int64 b,
    tensorflow::int64 c, char** values, tensorflow::int32 values_count,
    tensorflow::int32* values_primitive_type_size_in_bytes);

extern void __xla_cpu_runtime_KeyValueSortU64(
    tensorflow::uint64* keys, tensorflow::int64 a, tensorflow::int64 b,
    tensorflow::int64 c, char** values, tensorflow::int32 values_count,
    tensorflow::int32* values_primitive_type_size_in_bytes);

extern void __xla_cpu_runtime_KeyValueSortF64(
    double* keys, tensorflow::int64 a, tensorflow::int64 b, tensorflow::int64 c,
    char** values, tensorflow::int32 values_count,
    tensorflow::int32* values_primitive_type_size_in_bytes);
}

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_KEY_VALUE_SORT_H_
