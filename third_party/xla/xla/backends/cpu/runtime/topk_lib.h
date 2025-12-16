/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_TOPK_LIB_H_
#define XLA_BACKENDS_CPU_RUNTIME_TOPK_LIB_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/base/dynamic_annotations.h"

namespace xla::cpu::internal {

template <typename T>
void TopK(int64_t batch_size, int64_t input_size, int64_t k, const T* values,
          T* out_values, int32_t* out_indices) {
  // values is managed by the JIT code, so msan can't tell they are initialized.
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(values,
                                      input_size * batch_size * sizeof(T));

  auto convert_to_int = [](T value) {
    uint32_t x = absl::bit_cast<uint32_t>(value);
    return static_cast<int32_t>(x) < 0 ? std::numeric_limits<int32_t>::max() - x
                                       : x;
  };

  std::vector<int32_t> temp_indices(input_size);
  for (int64_t batch = 0; batch != batch_size; ++batch) {
    absl::c_iota(temp_indices, 0);

    const T* values_batch = values + batch * input_size;

    auto kth_element = temp_indices.begin() + k;
    std::partial_sort(temp_indices.begin(), kth_element, temp_indices.end(),
                      [&](size_t i1, size_t i2) {
                        // Do the comparison in integers to enforce a total
                        // order of -NaN < -Inf < -0 < +0 < +Inf < +NaN.
                        int32_t v1 = convert_to_int(values_batch[i1]);
                        int32_t v2 = convert_to_int(values_batch[i2]);
                        if (v1 == v2) {
                          return i1 < i2;  // Stabilize sorting.
                        }
                        return v1 > v2;
                      });

    T* out_values_batch = out_values + batch * k;
    int32_t* out_indices_batch = out_indices + batch * k;
    std::copy(temp_indices.begin(), kth_element, out_indices_batch);
    for (int64_t i = 0; i < k; i++) {
      out_values_batch[i] = values_batch[temp_indices[i]];
    }
  }
}

}  // namespace xla::cpu::internal

#endif  // XLA_BACKENDS_CPU_RUNTIME_TOPK_LIB_H_
