/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REVERSE_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REVERSE_H_

#include <algorithm>
#include <array>
#include <cstdint>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/runtime_shape.h"

namespace tflite {
namespace reference_ops {

template <typename Scalar>
void Reverse(std::array<int32_t, 8>& axes, int num_axes,
             const RuntimeShape& input_shape, const Scalar* input_data,
             Scalar* output_data) {
  ruy::profiler::ScopeLabel label("Reverse");
  bool is_upper = (axes[num_axes - 1] == input_shape.DimensionsCount() - 1);
  bool is_lower = (axes[0] == 0);
  int rank = input_shape.DimensionsCount();
  if (is_upper && is_lower) {
    std::reverse_copy(input_data, input_data + input_shape.FlatSize(),
                      output_data);
    return;
  } else {
    int32_t min_dim = axes[0];
    int32_t max_dim = axes[num_axes - 1];
    int upper_size = 1;
    for (int i = 0; i < min_dim; ++i) {
      upper_size *= input_shape.Dims(i);
    }
    int lower_size = 1;
    for (int i = max_dim + 1; i < rank; ++i) {
      lower_size *= input_shape.Dims(i);
    }
    int middle_size = 1;
    for (int i = min_dim; i <= max_dim; ++i) {
      middle_size *= input_shape.Dims(i);
    }

    if (lower_size > 1) {
      for (int i = 0; i < upper_size; ++i) {
        for (int j = 0; j < middle_size; ++j) {
          Scalar* src =
              (Scalar*)input_data + (i * (middle_size) + j) * lower_size;
          Scalar* dst =
              (Scalar*)output_data +
              (i * (middle_size) + (middle_size - j - 1)) * lower_size;
          memcpy(dst, src, lower_size * sizeof(Scalar));
        }
      }
    } else {
      for (int i = 0; i < upper_size; ++i) {
        std::reverse_copy(input_data + i * (middle_size),
                          input_data + i * middle_size + middle_size,
                          output_data + i * (middle_size));
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REVERSE_H_
