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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_

#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_ops {

inline void LeakyRelu(const tflite::LeakyReluParams& params,
                      const RuntimeShape& input_shape, const float* input_data,
                      const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    // Note that alpha might be > 1 or < 0, so we don't use std::max here.
    output_data[i] = val > 0 ? val : val * params.alpha;
  }
}

template <typename T>
inline void QuantizeLeakyRelu(const LeakyReluParams& params,
                              const RuntimeShape& input_shape,
                              const T* input_data,
                              const RuntimeShape& output_shape,
                              T* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  static const int32_t quantized_min = std::numeric_limits<T>::min();
  static const int32_t quantized_max = std::numeric_limits<T>::max();

  // Extract the sign and create a safely positive multiplier outside the loop.
  // This supports negative alpha values (matching float execution behavior)
  // while preventing assertion failures, as MultiplyByQuantizedMultiplier
  // strictly requires a non-negative multiplier.
  const bool is_alpha_negative = params.output_multiplier_alpha < 0;
  const int32_t safe_alpha_multiplier = is_alpha_negative
                                            ? -params.output_multiplier_alpha
                                            : params.output_multiplier_alpha;

  for (int i = 0; i < flat_size; ++i) {
    const int32_t input_value = input_data[i] - params.input_offset;

    int32_t unclamped_output = params.output_offset;
    if (input_value >= 0) {
      unclamped_output += MultiplyByQuantizedMultiplier(
          input_value, params.output_multiplier_identity,
          params.output_shift_identity);
    } else {
      int32_t scaled_alpha_value = MultiplyByQuantizedMultiplier(
          input_value, safe_alpha_multiplier, params.output_shift_alpha);
      unclamped_output +=
          is_alpha_negative ? -scaled_alpha_value : scaled_alpha_value;
    }

    const T clamped_output =
        std::min(quantized_max, std::max(quantized_min, unclamped_output));
    output_data[i] = static_cast<T>(clamped_output);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_
