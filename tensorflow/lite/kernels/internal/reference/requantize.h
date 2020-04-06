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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REQUANTIZE_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REQUANTIZE_H_

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

template <typename input_type, typename output_type>
inline void Requantize(const input_type* input_data, int32_t size,
                       int32_t effective_scale_multiplier,
                       int32_t effective_scale_shift, int32_t input_zeropoint,
                       int32_t output_zeropoint, output_type* output_data) {
  ruy::profiler::ScopeLabel label("Requantize");
  const bool same_scale =
      (effective_scale_multiplier == 1 << 30 && effective_scale_shift == 1);
  if (same_scale) {
    const bool mixed_type_int8_uint8 =
        std::is_same<input_type, int8_t>::value &&
        std::is_same<output_type, uint8_t>::value;
    const bool mixed_type_uint8_int8 =
        std::is_same<input_type, uint8_t>::value &&
        std::is_same<output_type, int8_t>::value;
    const int32_t zero_point_diff = input_zeropoint - output_zeropoint;
    // Fast path to do requantization for the case when just a shift of 128 is
    // needed.
    if ((mixed_type_int8_uint8 && zero_point_diff == -128) ||
        (mixed_type_uint8_int8 && zero_point_diff == 128)) {
      for (int i = 0; i < size; ++i) {
        output_data[i] = input_data[i] ^ 0x80;
      }
    }
  }
  static constexpr int32_t kMinOutput = std::numeric_limits<output_type>::min();
  static constexpr int32_t kMaxOutput = std::numeric_limits<output_type>::max();
  for (int i = 0; i < size; ++i) {
    const int32_t input = input_data[i] - input_zeropoint;
    const int32_t output =
        MultiplyByQuantizedMultiplier(input, effective_scale_multiplier,
                                      effective_scale_shift) +
        output_zeropoint;
    const int32_t clamped_output =
        std::max(std::min(output, kMaxOutput), kMinOutput);
    output_data[i] = static_cast<output_type>(clamped_output);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REQUANTIZE_H_
