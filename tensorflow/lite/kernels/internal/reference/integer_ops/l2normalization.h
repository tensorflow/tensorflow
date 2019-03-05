/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_L2NORMALIZATION_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_L2NORMALIZATION_H_

#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_integer_ops {

inline void L2Normalization(int32_t input_zero_point, int32_t outer_size,
                            int32_t depth, const int8* input_data,
                            int8* output_data) {
  static constexpr int8_t kMinInt8 = std::numeric_limits<int8_t>::min();
  static constexpr int8_t kMaxInt8 = std::numeric_limits<int8_t>::max();
  // The output scale must be in sync with Prepare().
  // Output is in 1/128 scale so the actual output range is nudged from [-1, 1]
  // to [-1, 127/128].
  static constexpr int32_t kOutputScale = 7;
  for (int outer_index = 0; outer_index < outer_size; ++outer_index) {
    // int32 = (int8 - int8) ^ 2.
    // ([-128, 127] - [-128, 127]) ^ 2 = [0, (2^8 - 1)^2] so the accumulator is
    // safe from overflowing in at least 2^16 steps.
    int32_t acc = 0;
    for (int inner_index = 0; inner_index < depth; ++inner_index) {
      int32_t input =
          input_data[depth * outer_index + inner_index] - input_zero_point;
      acc += input * input;
    }
    int32_t inv_l2norm_multiplier;
    int inv_l2norm_shift;
    GetInvSqrtQuantizedMultiplierExp(acc, /*reverse_shift*/ -1,
                                     &inv_l2norm_multiplier, &inv_l2norm_shift);

    for (int inner_index = 0; inner_index < depth; ++inner_index) {
      int32_t input =
          input_data[depth * outer_index + inner_index] - input_zero_point;

      // Rescale and downcast. Rescale is folded into the division.
      int32_t output_in_q24 = MultiplyByQuantizedMultiplier(
          input, inv_l2norm_multiplier, inv_l2norm_shift + kOutputScale);
      output_in_q24 =
          std::min(static_cast<int32_t>(kMaxInt8),
                   std::max(static_cast<int32_t>(kMinInt8), output_in_q24));
      output_data[depth * outer_index + inner_index] =
          static_cast<int8>(output_in_q24);
    }
  }
}
}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_L2NORMALIZATION_H_
