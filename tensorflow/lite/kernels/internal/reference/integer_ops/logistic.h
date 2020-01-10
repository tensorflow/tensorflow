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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_LOGISTIC_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_LOGISTIC_H_

#include <limits>
#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_integer_ops {

inline void Logistic(int32_t input_zero_point, int32_t input_range_radius,
                     int32_t input_multiplier, int32_t input_left_shift,
                     int32_t input_size, const int8_t* input_data,
                     int8_t* output_data) {
  // Integer bits must be in sync with Prepare() function.
  static constexpr int32_t kInputIntegerBits = 4;
  static constexpr int32_t kOutputIntegerBits = 8;
  static constexpr int8_t kMinInt8 = std::numeric_limits<int8_t>::min();
  static constexpr int8_t kMaxInt8 = std::numeric_limits<int8_t>::max();
  static constexpr int32_t kOutputZeroPoint = -128;

  for (int i = 0; i < input_size; ++i) {
    const int32_t input =
        static_cast<int32_t>(input_data[i]) - input_zero_point;
    if (input <= -input_range_radius) {
      output_data[i] = kMinInt8;
    } else if (input >= input_range_radius) {
      output_data[i] = kMaxInt8;
    } else {
      const int32_t input_in_q4 = MultiplyByQuantizedMultiplier(
          input, input_multiplier, input_left_shift);
      using FixedPoint4 = gemmlowp::FixedPoint<int32_t, kInputIntegerBits>;
      const int32_t output_in_q0 =
          gemmlowp::logistic(FixedPoint4::FromRaw(input_in_q4)).raw();

      // Rescale and downcast.
      using gemmlowp::RoundingDivideByPOT;
      int32_t output_in_q23 =
          RoundingDivideByPOT(output_in_q0, 31 - kOutputIntegerBits);
      output_in_q23 = std::min(std::max(output_in_q23 + kOutputZeroPoint,
                                        static_cast<int32_t>(kMinInt8)),
                               static_cast<int32_t>(kMaxInt8));
      output_data[i] = static_cast<int8_t>(output_in_q23);
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_LOGISTIC_H_
