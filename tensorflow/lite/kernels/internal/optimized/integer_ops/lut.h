/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_LUT_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_LUT_H_

#include <cstdint>

#if __aarch64__ && __clang__
#include <arm_neon.h>
#endif

#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"

namespace tflite {
namespace optimized_integer_ops {

inline void LookupTable(const uint8_t* input_data, int num_elements,
                        const uint8_t* lut, uint8_t* output_data) {
  int i = 0;
#if __aarch64__ && __clang__
  // This code uses ARM64-only instructions.
  // TODO(b/143709993): Port to ARMv7

  // Load the tables into registers. (4*4 128-bit registers)
  uint8x16x4_t table[4];
  table[0] = vld1q_u8_x4(lut + 16 * 4 * 0);
  table[1] = vld1q_u8_x4(lut + 16 * 4 * 1);
  table[2] = vld1q_u8_x4(lut + 16 * 4 * 2);
  table[3] = vld1q_u8_x4(lut + 16 * 4 * 3);

  // Vectorized loop; process uint8x16_t (16 elements) at a time.
  constexpr int vectorized_16_loop_step = 16;
  const int vectorized_16_loop_end =
      num_elements / vectorized_16_loop_step * vectorized_16_loop_step;
  for (; i < vectorized_16_loop_end; i += vectorized_16_loop_step) {
    uint8x16_t input = vld1q_u8(input_data + i);
    uint8x16_t output = optimized_ops::aarch64_lookup_vector(table, input);
    vst1q_u8(output_data + i, output);
  }
  // Postamble and non-ARM64 code: simple for loop.
#endif
  for (; i < num_elements; ++i) {
    output_data[i] = lut[input_data[i]];
  }
}

// LUTPopulate<int8_t> has ordered the LUT so that indexing it with an
// int8_t is just done by casting it to an uint8_t. We can thus reuse the uint8
// LookupTable function.
inline void LookupTable(const int8_t* input_data, int num_elements,
                        const int8_t* lut, int8_t* output_data) {
  LookupTable(reinterpret_cast<const uint8_t*>(input_data), num_elements,
              reinterpret_cast<const uint8_t*>(lut),
              reinterpret_cast<uint8_t*>(output_data));
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_LUT_H_
