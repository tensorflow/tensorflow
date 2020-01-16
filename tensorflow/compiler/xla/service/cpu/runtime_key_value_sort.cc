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
#include <cstring>
#include <memory>
#include <numeric>
#include <string>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace {
using tensorflow::int32;
using tensorflow::int64;
}  // namespace

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_KeyValueSort(
    int64 a, int64 b, int64 c, char** values, int32 values_count,
    int32* values_primitive_type_size_in_bytes, bool is_stable,
    char* run_options, int64* prof_counters,
    void (*less_than)(char*, char*, char**, char**, tensorflow::int64*)) {
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

  std::unique_ptr<int64[]> indices(new int64[sort_dimension_elements]);
  std::unique_ptr<char*[]> comparison_values(new char*[2 * values_count]);
  std::iota(indices.get(), indices.get() + sort_dimension_elements, 0);
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
    auto compare_function = [&](int64 a, int64 b) -> bool {
      for (int32 i = 0; i < values_count; ++i) {
        int64 memory_index_lhs = (base_offset + a * sort_dimension_offset) *
                                 values_primitive_type_size_in_bytes[i];
        int64 memory_index_rhs = (base_offset + b * sort_dimension_offset) *
                                 values_primitive_type_size_in_bytes[i];
        comparison_values[i * 2] = values[i] + memory_index_lhs;
        comparison_values[i * 2 + 1] = values[i] + memory_index_rhs;
      }
      char result = 0;  // Overwritten by less_than.
      less_than(&result, run_options, comparison_values.get(), nullptr,
                prof_counters);
      return result != 0u;
    };
    if (is_stable) {
      std::stable_sort(indices.get(), indices.get() + sort_dimension_elements,
                       compare_function);
    } else {
      std::sort(indices.get(), indices.get() + sort_dimension_elements,
                compare_function);
    }

    // Reorder the values according to the order defined by 'indices'.
    for (int32 idx = 0; idx < values_count; ++idx) {
      for (int64 i = 0; i < sort_dimension_elements; ++i) {
        int64 memory_index =
            (base_offset + indices[i] * sort_dimension_offset) *
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
