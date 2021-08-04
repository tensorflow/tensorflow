/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/runtime_topk.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/macros.h"

template <typename T>
static void TopK(int64_t batch_size, int64_t input_size, int64_t k,
                 const T* values, T* out_values,
                 tensorflow::int32* out_indices) {
  // 'values' is managed by the JIT code, so msan can't tell they are
  // initialized.
  TF_ANNOTATE_MEMORY_IS_INITIALIZED(values,
                                    input_size * batch_size * sizeof(T));

  std::vector<tensorflow::int32> temp_indices(input_size);
  for (int64_t batch = 0; batch != batch_size; ++batch) {
    std::iota(temp_indices.begin(), temp_indices.end(), 0);

    const T* values_batch = values + batch * input_size;

    auto convert_to_int = [](T value) {
      tensorflow::uint32 x;
      std::memcpy(&x, &value, sizeof(x));
      return static_cast<tensorflow::int32>(x) < 0
                 ? std::numeric_limits<tensorflow::int32>::max() - x
                 : x;
    };

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
    tensorflow::int32* out_indices_batch = out_indices + batch * k;
    std::copy(temp_indices.begin(), kth_element, out_indices_batch);
    for (int64_t i = 0; i < k; i++) {
      out_values_batch[i] = values_batch[temp_indices[i]];
    }
  }
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_TopKF32(
    int64_t batch_size, int64_t input_size, int64_t k, const float* values,
    float* out_values, tensorflow::int32* out_indices) {
  TopK(batch_size, input_size, k, values, out_values, out_indices);
}
