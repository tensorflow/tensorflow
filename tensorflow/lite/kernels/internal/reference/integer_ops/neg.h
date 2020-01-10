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

// Quantized Negate with int8 input and output, input and output must have the
// equal scale
inline void Negate(const RuntimeShape& input_shape, const int8_t* input_data,
                   int32_t input_zero_point, const RuntimeShape& output_shape,
                   int8_t* output_data, int32_t output_zero_point) {
  // equation: out =  in_zp + out_zp - in
  // where out, out_zp, in_zp ∈ [-128, 127] : int8
  // highest possible value: 127 + 127 - (-128) = 382
  // lowest possible value: (-128) + (-128) - (127)  = -383
  // accumulate on int16

  constexpr int16_t kI8Max =
      static_cast<int16_t>(std::numeric_limits<int8_t>::max());
  constexpr int16_t kI8Min =
      static_cast<int16_t>(std::numeric_limits<int8_t>::min());

  // within: [-128, 127]
  TFLITE_DCHECK_GE(input_zero_point, static_cast<int32_t>(kI8Min));
  TFLITE_DCHECK_LE(input_zero_point, static_cast<int32_t>(kI8Max));

  // within: [-128, 127]
  TFLITE_DCHECK_GE(output_zero_point, static_cast<int32_t>(kI8Min));
  TFLITE_DCHECK_LE(output_zero_point, static_cast<int32_t>(kI8Max));

  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  // already within int8 range, stored in int32
  const auto prior = static_cast<int16_t>(input_zero_point + output_zero_point);

  for (int i = 0; i < flat_size; ++i) {
    // all operations performed on int16
    const int16_t neg_accum = prior - static_cast<int16_t>(input_data[i]);
    auto clamped_accum = neg_accum > kI8Max ? kI8Max : neg_accum;
    clamped_accum = clamped_accum < kI8Min ? kI8Min : clamped_accum;
    output_data[i] = static_cast<int8_t>(clamped_accum);
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_NEG_H_