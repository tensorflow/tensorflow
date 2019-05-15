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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_DIV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_DIV_H_

#include "public/gemmlowp.h"
#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_integer_ops {

inline void DivElementwise(int size, const ArithmeticParams& params,
                           const int8_t* input1_data, const int8_t* input2_data,
                           int8_t* output_data) {
  for (int i = 0; i < size; ++i) {
    const int32 input1_val = params.input1_offset + input1_data[i];
    int32 input2_val = params.input2_offset + input2_data[i];
    using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;
    int32 sign_multiplier = 1;
    if (input2_val < 0) {
      sign_multiplier = -1;
      input2_val = sign_multiplier * input2_val;
    }

    const int32 input2_diff_rescaled = MultiplyByQuantizedMultiplier(
        input2_val * (1 << params.left_shift), params.input2_multiplier,
        params.input2_shift);

    int num_bits_over_unit;
    FixedPoint0 shifted_scale = FixedPoint0::FromRaw(
        GetReciprocal(input2_diff_rescaled, 0, &num_bits_over_unit));

    const int32 unsat_output = gemmlowp::RoundingDivideByPOT(
        shifted_scale.raw(), num_bits_over_unit + 31 - 9);

    const int32 input2_scaled = MultiplyByQuantizedMultiplier(
        unsat_output, params.input1_multiplier, params.input1_shift);

    const int32 unclamped_result =
        params.output_offset + MultiplyByQuantizedMultiplier(
                                   sign_multiplier * input2_scaled * input1_val,
                                   params.output_multiplier,
                                   params.output_shift);
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<int8_t>(clamped_output);
  }
}

inline void Div(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int8_t* input1_data,
                const RuntimeShape& input2_shape, const int8_t* input2_data,
                const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  gemmlowp::ScopedProfilingLabel label("Div/8bit");
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);

  DivElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void BroadcastDiv4DSlow(const ArithmeticParams& params,
                               const RuntimeShape& input1_shape,
                               const int8_t* input1_data,
                               const RuntimeShape& input2_shape,
                               const int8_t* input2_data,
                               const RuntimeShape& output_shape,
                               int8_t* output_data) {
  gemmlowp::ScopedProfilingLabel label("BroadcastDiv4DSlow/8bit");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  // The input shapes are extended as part of NdArrayDesc initialization.
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);

  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          const int32 input1_val =
              params.input1_offset +
              input1_data[SubscriptToIndex(desc1, b, y, x, c)];
          int32 input2_val = params.input2_offset +
                             input2_data[SubscriptToIndex(desc2, b, y, x, c)];
          using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;

          int32 sign_multiplier = 1;
          if (input2_val < 0) {
            sign_multiplier = -1;
            input2_val = sign_multiplier * input2_val;
          }

          const int32 input2_diff_rescaled = MultiplyByQuantizedMultiplier(
              input2_val * (1 << params.left_shift), params.input2_multiplier,
              params.input2_shift);
          int num_bits_over_unit;
          FixedPoint0 shifted_scale = FixedPoint0::FromRaw(
              GetReciprocal(input2_diff_rescaled, 0, &num_bits_over_unit));

          int32 unsat_output = gemmlowp::RoundingDivideByPOT(
              shifted_scale.raw(), num_bits_over_unit + 31 - 9);

          const int32 input2_scaled = MultiplyByQuantizedMultiplier(
              unsat_output, params.input1_multiplier, params.input1_shift);

          const int32 unclamped_result =
              params.output_offset +
              MultiplyByQuantizedMultiplier(
                  sign_multiplier * input2_scaled * input1_val,
                  params.output_multiplier, params.output_shift);
          const int32 clamped_output = std::min(
              params.quantized_activation_max,
              std::max(params.quantized_activation_min, unclamped_result));
          output_data[Offset(extended_output_shape, b, y, x, c)] =
              static_cast<int8_t>(clamped_output);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_DIV_H_