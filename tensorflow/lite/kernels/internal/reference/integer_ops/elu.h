/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *     ==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_ELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_ELU_H_

#include "public/gemmlowp.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_integer_ops {

inline void QElu(const EluParams& params, const RuntimeShape& input_shape,
                 const int8* input_data, const RuntimeShape& output_shape,
                 int8* output_data) {
  const int32 input_zero_point = params.input_zero_point;
  const int32 input_range_radius = params.input_range_radius;
  const int32 input_multiplier = params.input_multiplier;
  const int input_left_shift = params.input_left_shift;
  const int size = MatchingFlatSize(input_shape, output_shape);

  int c = 0;
  // Leftover loop: handle one value at a time with scalar code.
  for (; c < size; ++c) {
    const int8 input_val_u8 = input_data[c];
    const int32 input_val_centered =
        static_cast<int32>(input_val_u8) - input_zero_point;
    int8 output_val;
    if (input_val_centered < -input_range_radius) {
      output_val = std::numeric_limits<int8_t>::min();
    } else if (input_val_centered > input_range_radius) {
      output_val = std::numeric_limits<int8_t>::max();
    } else if (input_val_centered >= 0) {
      output_val = input_val_u8;
    } else {
      // For all negative values of x
      const int32 input_val_rescaled =
          MultiplyByQuantizedMultiplierGreaterThanOne(
              input_val_centered, input_multiplier, input_left_shift);

      using FixedPoint4 = gemmlowp::FixedPoint<int32, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 =
          gemmlowp::exp_on_negative_values(input_val_f4);

      int32 output = output_val_f0.raw() - FixedPoint0::One().raw();
      const FixedPoint0 final_val_f4 = FixedPoint0::FromRaw(output);

      using gemmlowp::RoundingDivideByPOT;
      int32 output_val_s32 =
          RoundingDivideByPOT(final_val_f4.raw(), input_left_shift + 3);

      if (output_val_s32 >= std::numeric_limits<int8_t>::max()) {
        output_val_s32 = std::numeric_limits<int8_t>::max();
      } else if (output_val_s32 <= std::numeric_limits<int8_t>::min()) {
        output_val_s32 = std::numeric_limits<int8_t>::min();
      }
      output_val = static_cast<int8>(output_val_s32);
    }
    output_data[c] = output_val;
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif
