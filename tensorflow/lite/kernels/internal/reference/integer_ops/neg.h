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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_NEG_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_NEG_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_integer_ops {

// Quantized Negate with int8 input and output, input and output must have an
// equal scale
// [zero_point_sum] represents the sum of the input and output zero points
inline void Negate(const RuntimeShape& input_shape, const int8_t* input_data,
                   const RuntimeShape& output_shape, int8_t* output_data,
                   int16_t zero_point_sum) {
  // where: output, output_zero_point, input_zero_point ∈ [-128, 127] : int8
  // zero_point_sum = (input_zero_point + output_zero_point)
  // equation: output = zero_point_sum - input
  // highest possible value for zero_point_sum = 127 + 127 = 254
  // lowest possible value for zero_point_sum = -128 + (-128) = -256
  // lowest possible neg value = lowest zero_point_sum - 127 = -256 - 127 =
  // -383
  // highest possible neg value = highest zero_point_sum - (-128) = 254 + 128
  // = 382
  // thus, accumulate on int16 [-383, 382]

  constexpr int16_t kI8Min =
      static_cast<int16_t>(std::numeric_limits<int8_t>::min());
  constexpr int16_t kI8Max =
      static_cast<int16_t>(std::numeric_limits<int8_t>::max());

  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; ++i) {
    // all operations accumulated on int16
    const int16_t neg = zero_point_sum - static_cast<int16_t>(input_data[i]);
    const auto clamped_neg = std::min(std::max(neg, kI8Min), kI8Max);
    output_data[i] = static_cast<int8_t>(clamped_neg);
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_NEG_H_