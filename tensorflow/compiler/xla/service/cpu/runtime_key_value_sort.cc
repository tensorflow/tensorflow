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
#include "tensorflow/compiler/xla/service/cpu/runtime_key_value_sort.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace {
using tensorflow::int16;
using tensorflow::int32;
using tensorflow::int64;
using tensorflow::int8;
using tensorflow::uint16;
using tensorflow::uint32;
using tensorflow::uint64;
using tensorflow::uint8;

template <typename KeyType>
void KeyValueSort(std::pair<KeyType, int64>* row_to_sort, int64 num_elements) {
  std::sort(row_to_sort, row_to_sort + num_elements);
}

// For floating point numbers, we want a total order comparator. -NaN and NaN
// should appear at the beginning and end of the ordering, and -0.0 should
// appear before 0.0. Also we want to have a stable sort, so if the keys are the
// same, we compare the index values.
template <typename KeyType>
bool LessThan(KeyType lhs, int64 lhs_index, KeyType rhs, int64 rhs_index) {
  bool lhs_is_negative = std::signbit(lhs);
  bool rhs_is_negative = std::signbit(rhs);
  // If the signs are different, we can just compare the signs.
  if (lhs_is_negative != rhs_is_negative) {
    return lhs_is_negative && !rhs_is_negative;
  }
  bool lhs_nan = std::isnan(lhs);
  bool rhs_nan = std::isnan(rhs);
  // Exactly one number is nan?
  if (lhs_nan != rhs_nan) {
    if (lhs_nan) {
      return lhs_is_negative;
    }
    return !rhs_is_negative;
  }
  if (lhs != rhs) {
    return lhs < rhs;
  }
  return lhs_index < rhs_index;
}

template <>
void KeyValueSort(std::pair<double, int64>* row_to_sort, int64 num_elements) {
  std::sort(row_to_sort, row_to_sort + num_elements,
            [](const std::pair<double, int64>& lhs,
               const std::pair<double, int64>& rhs) -> bool {
              return LessThan(lhs.first, lhs.second, rhs.first, rhs.second);
            });
}

template <>
void KeyValueSort(std::pair<float, int64>* row_to_sort, int64 num_elements) {
  std::sort(row_to_sort, row_to_sort + num_elements,
            [](const std::pair<float, int64>& lhs,
               const std::pair<float, int64>& rhs) -> bool {
              return LessThan(lhs.first, lhs.second, rhs.first, rhs.second);
            });
}

template <>
void KeyValueSort(std::pair<Eigen::half, int64>* row_to_sort,
                  int64 num_elements) {
  std::sort(row_to_sort, row_to_sort + num_elements,
            [](const std::pair<Eigen::half, int64>& lhs,
               const std::pair<Eigen::half, int64>& rhs) -> bool {
              return LessThan(
                  Eigen::half_impl::half_to_float(lhs.first), lhs.second,
                  Eigen::half_impl::half_to_float(rhs.first), rhs.second);
            });
}

template <typename KeyType>
void KeyValueSortImpl(KeyType* keys, int64 a, int64 b, int64 c, char** values,
                      int32 values_count,
                      int32* values_primitive_type_size_in_bytes) {
  // 'values' and 'values_primitive_type_size_in_bytes' are managed by the JIT
  // code, so msan can't tell they are initialized.
  TF_ANNOTATE_MEMORY_IS_INITIALIZED(values, values_count * sizeof(char*));
  TF_ANNOTATE_MEMORY_IS_INITIALIZED(values_primitive_type_size_in_bytes,
                                    values_count * sizeof(int32));

  // High-level idea of the iteration/sorting logic:
  // Conceptually we have a 3-dimensional shape [a, b, c]. b corresponds to the
  // dimension to sort, c is the product of the more minor dimensions (set to 1
  // if b is the most minor dimension), and a is the product of the more major
  // dimensions (set to 1 if b is the most major dimension). There are a * c
  // many rows that we need to sort. We iterate through these, calculate a
  // 'base_offset' value which points to the first element in that row, and add
  // i * c for accessing the 'i'-th element in that row.

  int64 sort_dimension_elements = b;
  int64 num_iteration_elements = a * c;
  int64 sort_dimension_offset = c;

  std::unique_ptr<std::pair<KeyType, int64>[]> row_to_sort(
      new std::pair<KeyType, int64>[sort_dimension_elements]);
  std::unique_ptr<std::string[]> reordered_values(
      new std::string[sort_dimension_elements]);
  for (int64 index = 0; index < num_iteration_elements; ++index) {
    // 'index' can be split into two values which index into the 'c' dimension
    // and the 'a' dimension, respectively. 'index' % 'c' is the index into the
    // 'c' dimension, 'index' / 'c' is the index into the 'a' dimension. When
    // calculating the base offset, we need to multiply the index into the 'a'
    // dimension with 'b' * 'c'.
    // 'index' / 'c' * 'c' * 'b' = ('index' - 'index' % 'c') * 'b'.
    int64 base_offset =
        index % sort_dimension_offset +
        (index - index % sort_dimension_offset) * sort_dimension_elements;
    // TODO(b/26783907): We could define a custom iterator class that references
    // all arrays. Then we could avoid the intermediate copy. However this
    // would become more complicated, and it is not clear if the benefit is high
    // enough.
    for (int64 i = 0; i < sort_dimension_elements; ++i) {
      row_to_sort[i] =
          std::make_pair(keys[base_offset + i * sort_dimension_offset], i);
    }
    KeyValueSort(row_to_sort.get(), sort_dimension_elements);
    for (int64 i = 0; i < sort_dimension_elements; ++i) {
      keys[base_offset + i * sort_dimension_offset] = row_to_sort[i].first;
    }

    // Reorder the values according to the order defined by the keys.
    for (int32 idx = 0; idx < values_count; ++idx) {
      for (int64 i = 0; i < sort_dimension_elements; ++i) {
        int64 memory_index =
            (base_offset + row_to_sort[i].second * sort_dimension_offset) *
            values_primitive_type_size_in_bytes[idx];

        reordered_values[i] =
            std::string(values[idx] + memory_index,
                        values_primitive_type_size_in_bytes[idx]);
      }
      for (int64 i = 0; i < sort_dimension_elements; ++i) {
        int64 memory_index = (base_offset + i * sort_dimension_offset) *
                             values_primitive_type_size_in_bytes[idx];
        memcpy(values[idx] + memory_index, reordered_values[i].c_str(),
               values_primitive_type_size_in_bytes[idx]);
      }
    }
  }
}
}  // namespace

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_KeyValueSortPRED(
    bool* keys, int64 a, int64 b, int64 c, char** values, int32 values_count,
    int32* values_primitive_type_size_in_bytes) {
  KeyValueSortImpl(keys, a, b, c, values, values_count,
                   values_primitive_type_size_in_bytes);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_KeyValueSortS8(
    int8* keys, int64 a, int64 b, int64 c, char** values, int32 values_count,
    int32* values_primitive_type_size_in_bytes) {
  KeyValueSortImpl(keys, a, b, c, values, values_count,
                   values_primitive_type_size_in_bytes);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_KeyValueSortU8(
    uint8* keys, int64 a, int64 b, int64 c, char** values, int32 values_count,
    int32* values_primitive_type_size_in_bytes) {
  KeyValueSortImpl(keys, a, b, c, values, values_count,
                   values_primitive_type_size_in_bytes);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_KeyValueSortS16(
    int16* keys, int64 a, int64 b, int64 c, char** values, int32 values_count,
    int32* values_primitive_type_size_in_bytes) {
  KeyValueSortImpl(keys, a, b, c, values, values_count,
                   values_primitive_type_size_in_bytes);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_KeyValueSortU16(
    uint16* keys, int64 a, int64 b, int64 c, char** values, int32 values_count,
    int32* values_primitive_type_size_in_bytes) {
  KeyValueSortImpl(keys, a, b, c, values, values_count,
                   values_primitive_type_size_in_bytes);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_KeyValueSortF16(
    Eigen::half* keys, int64 a, int64 b, int64 c, char** values,
    int32 values_count, int32* values_primitive_type_size_in_bytes) {
  KeyValueSortImpl(keys, a, b, c, values, values_count,
                   values_primitive_type_size_in_bytes);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_KeyValueSortS32(
    int32* keys, int64 a, int64 b, int64 c, char** values, int32 values_count,
    int32* values_primitive_type_size_in_bytes) {
  KeyValueSortImpl(keys, a, b, c, values, values_count,
                   values_primitive_type_size_in_bytes);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_KeyValueSortU32(
    uint32* keys, int64 a, int64 b, int64 c, char** values, int32 values_count,
    int32* values_primitive_type_size_in_bytes) {
  KeyValueSortImpl(keys, a, b, c, values, values_count,
                   values_primitive_type_size_in_bytes);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_KeyValueSortF32(
    float* keys, int64 a, int64 b, int64 c, char** values, int32 values_count,
    int32* values_primitive_type_size_in_bytes) {
  KeyValueSortImpl(keys, a, b, c, values, values_count,
                   values_primitive_type_size_in_bytes);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_KeyValueSortS64(
    int64* keys, int64 a, int64 b, int64 c, char** values, int32 values_count,
    int32* values_primitive_type_size_in_bytes) {
  KeyValueSortImpl(keys, a, b, c, values, values_count,
                   values_primitive_type_size_in_bytes);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_KeyValueSortU64(
    uint64* keys, int64 a, int64 b, int64 c, char** values, int32 values_count,
    int32* values_primitive_type_size_in_bytes) {
  KeyValueSortImpl(keys, a, b, c, values, values_count,
                   values_primitive_type_size_in_bytes);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_KeyValueSortF64(
    double* keys, int64 a, int64 b, int64 c, char** values, int32 values_count,
    int32* values_primitive_type_size_in_bytes) {
  KeyValueSortImpl(keys, a, b, c, values, values_count,
                   values_primitive_type_size_in_bytes);
}
