/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_4BIT_FULLY_CONNECTED_COMMON_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_4BIT_FULLY_CONNECTED_COMMON_H_
#include <cstdint>

namespace tflite {
namespace optimized_4bit {

// Since we need to convert int4 to int8 with shifts, it is faster if we
// can use unsigned int4, so just subtract zero_point_4bit from all values.
// Fold input * zero_point into quantization since we need to quantize
// each input and multiply by zero_point_4bit to convert back to signed int.
constexpr int zero_point_4bit = -7;

inline int8_t upper(int8_t value) { return value >> 4; }

inline int8_t lower(int8_t value) {
  uint8_t sign_y = UINT8_C(256) - (value & UINT8_C(8));
  return (value & UINT8_C(7)) | sign_y;
}

inline int8_t merge(int8_t upper, int8_t lower) {
  const auto to_int4 = [](int8_t v) -> uint8_t {
    int32_t x = v + 7;
    return static_cast<uint8_t>(x);
  };
  return (to_int4(upper) << 4) | to_int4(lower);
}

}  // namespace optimized_4bit
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_4BIT_FULLY_CONNECTED_COMMON_H_
