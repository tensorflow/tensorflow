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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_ACTIVATIONS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_ACTIVATIONS_H_

#include "tensorflow/lite/kernels/cpu_backend_gemm.h"

namespace tflite {
namespace optimized_ops {

inline void Tanh16bitPrecision(const TanhParams& params,
                               const RuntimeShape& input_shape,
                               const uint8* input_data,
                               const RuntimeShape& output_shape,
                               uint8* output_data) {
  // Note that this is almost the exact same code as in Logistic().
  ruy::profiler::ScopeLabel label("Tanh/Uint8");
  const int32 input_zero_point = params.input_zero_point;
  const int32 input_range_radius = params.input_range_radius;
  const int16 input_multiplier = static_cast<int16>(params.input_multiplier);
  const int16 input_left_shift = static_cast<int16>(params.input_left_shift);
  const int size = MatchingFlatSize(input_shape, output_shape);

  int c = 0;
  int16_t output_zero_point = 128;

// TODO(b/139252020): Replace GEMMLOWP_NEON with USE_NEON when the bug is fixed.
// The converted versions of gemmlowp::tanh and gemmlowp::logistic, done by
// arm_sse_2_neon.h, produce incorrect results with int16x8_t data types.
#ifdef GEMMLOWP_NEON
  const int16x8_t range_radius_dup = vdupq_n_s16(input_range_radius);
  const int16x8_t neg_range_radius_dup = vdupq_n_s16(-input_range_radius);
  const int16x8_t output_zero_point_s16 = vdupq_n_s16(output_zero_point);

  // Handle 32 values at a time
  for (; c <= size - 32; c += 32) {
    // Read input uint8 values, cast to int16 and subtract input_zero_point
    using cpu_backend_gemm::detail::Load16AndSubtractZeroPoint;
    const int16x8x2_t input_val_centered_0_1 =
        Load16AndSubtractZeroPoint(input_data + c, input_zero_point);
    const int16x8x2_t input_val_centered_2_3 =
        Load16AndSubtractZeroPoint(input_data + c + 16, input_zero_point);

    // Prepare the bit masks that we will use at the end to implement the logic
    // that was expressed in the scalar code with branching:
    //   if (input_val_centered < -input_range_radius) {
    //     output_val = 0;
    //   } else if (input_val_centered > input_range_radius) {
    //     output_val = 255;
    //   } else {
    //     ...
    uint8x16x2_t masks_clamp_0_1 = CalculateUnsignedClampingWithRangeBitMasks(
        input_val_centered_0_1, range_radius_dup, neg_range_radius_dup);
    uint8x16x2_t masks_clamp_2_3 = CalculateUnsignedClampingWithRangeBitMasks(
        input_val_centered_2_3, range_radius_dup, neg_range_radius_dup);

    int16x8x4_t input_val_rescaled = SaturatingRounding(
        input_val_centered_0_1.val[0], input_val_centered_0_1.val[1],
        input_val_centered_2_3.val[0], input_val_centered_2_3.val[1],
        input_left_shift, input_multiplier);

    int16x8x4_t output_val_s16 = FixedPoint4Tanh(input_val_rescaled);

    // Add the output zero point
    output_val_s16.val[0] =
        vaddq_s16(output_val_s16.val[0], output_zero_point_s16);
    output_val_s16.val[1] =
        vaddq_s16(output_val_s16.val[1], output_zero_point_s16);
    output_val_s16.val[2] =
        vaddq_s16(output_val_s16.val[2], output_zero_point_s16);
    output_val_s16.val[3] =
        vaddq_s16(output_val_s16.val[3], output_zero_point_s16);

    // Cast output values to uint8, saturating
    uint8x16_t output_val_u8_0_1 = vcombine_u8(
        vqmovun_s16(output_val_s16.val[0]), vqmovun_s16(output_val_s16.val[1]));
    uint8x16_t output_val_u8_2_3 = vcombine_u8(
        vqmovun_s16(output_val_s16.val[2]), vqmovun_s16(output_val_s16.val[3]));

    ClampWithRangeAndStore(output_data + c, output_val_u8_0_1, masks_clamp_0_1);
    ClampWithRangeAndStore(output_data + c + 16, output_val_u8_2_3,
                           masks_clamp_2_3);
  }
#endif  // GEMMLOWP_NEON
  // Leftover loop: handle one value at a time with scalar code.
  for (; c < size; ++c) {
    const uint8 input_val_u8 = input_data[c];
    const int16 input_val_centered =
        static_cast<int16>(input_val_u8) - input_zero_point;
    uint8 output_val;
    if (input_val_centered < -input_range_radius) {
      output_val = 0;
    } else if (input_val_centered > input_range_radius) {
      output_val = 255;
    } else {
      using gemmlowp::SaturatingRoundingDoublingHighMul;
      const int16 input_val_rescaled = SaturatingRoundingDoublingHighMul(
          static_cast<int16>(input_val_centered * (1 << input_left_shift)),
          static_cast<int16>(input_multiplier));
      using FixedPoint4 = gemmlowp::FixedPoint<int16, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int16, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::tanh(input_val_f4);
      using gemmlowp::RoundingDivideByPOT;
      int16 output_val_s16 = RoundingDivideByPOT(output_val_f0.raw(), 8);
      output_val_s16 += output_zero_point;
      if (output_val_s16 == 256) {
        output_val_s16 = 255;
      }
      TFLITE_DCHECK_GE(output_val_s16, 0);
      TFLITE_DCHECK_LE(output_val_s16, 255);
      output_val = static_cast<uint8>(output_val_s16);
    }
    output_data[c] = output_val;
  }
}

inline void Tanh16bitPrecision(const TanhParams& params,
                               const RuntimeShape& input_shape,
                               const int8* input_data,
                               const RuntimeShape& output_shape,
                               int8* output_data) {
  // Note that this is almost the exact same code as in Logistic().
  ruy::profiler::ScopeLabel label("Tanh/Int8");
  const int32 input_zero_point = params.input_zero_point;
  const int32 input_range_radius = params.input_range_radius;
  const int16 input_multiplier = static_cast<int16>(params.input_multiplier);
  const int16 input_left_shift = static_cast<int16>(params.input_left_shift);
  const int size = MatchingFlatSize(input_shape, output_shape);

  int c = 0;
// TODO(b/139252020): Replace GEMMLOWP_NEON with USE_NEON when the bug is fixed.
// The converted versions of gemmlowp::tanh and gemmlowp::logistic, done by
// arm_sse_2_neon.h, produce incorrect results with int16x8_t data types.
#ifdef GEMMLOWP_NEON
  const int16x8_t range_radius_dup = vdupq_n_s16(input_range_radius);
  const int16x8_t neg_range_radius_dup = vdupq_n_s16(-input_range_radius);

  // Handle 32 values at a time
  for (; c <= size - 32; c += 32) {
    // Read input int8 values, cast to int16 and subtract input_zero_point
    using cpu_backend_gemm::detail::Load16AndSubtractZeroPoint;
    const int16x8x2_t input_val_centered_0_1 =
        Load16AndSubtractZeroPoint(input_data + c, input_zero_point);
    const int16x8x2_t input_val_centered_2_3 =
        Load16AndSubtractZeroPoint(input_data + c + 16, input_zero_point);

    // Prepare the bit masks that we will use at the end to implement the logic
    // that was expressed in the scalar code with branching:
    //   if (input_val_centered < -input_range_radius) {
    //     output_val = -128;
    //   } else if (input_val_centered > input_range_radius) {
    //     output_val = 127;
    //   } else {
    //     ...
    uint8x16x2_t masks_clamp_0_1 = CalculateSignedClampingWithRangeBitMasks(
        input_val_centered_0_1, range_radius_dup, neg_range_radius_dup);
    uint8x16x2_t masks_clamp_2_3 = CalculateSignedClampingWithRangeBitMasks(
        input_val_centered_2_3, range_radius_dup, neg_range_radius_dup);

    int16x8x4_t input_val_rescaled = SaturatingRounding(
        input_val_centered_0_1.val[0], input_val_centered_0_1.val[1],
        input_val_centered_2_3.val[0], input_val_centered_2_3.val[1],
        input_left_shift, input_multiplier);

    int16x8x4_t output_val_s16 = FixedPoint4Tanh(input_val_rescaled);

    // Cast output values to uint8, saturating
    int8x16_t output_val_s8_0_1 = vcombine_s8(
        vqmovn_s16(output_val_s16.val[0]), vqmovn_s16(output_val_s16.val[1]));
    int8x16_t output_val_s8_2_3 = vcombine_s8(
        vqmovn_s16(output_val_s16.val[2]), vqmovn_s16(output_val_s16.val[3]));

    ClampWithRangeAndStore(output_data + c, output_val_s8_0_1, masks_clamp_0_1);
    ClampWithRangeAndStore(output_data + c + 16, output_val_s8_2_3,
                           masks_clamp_2_3);
  }
#endif  // GEMMLOWP_NEON
  // Leftover loop: handle one value at a time with scalar code.
  for (; c < size; ++c) {
    const int8 input_val_s8 = input_data[c];
    const int16 input_val_centered =
        static_cast<int16>(input_val_s8) - input_zero_point;
    int8 output_val;
    if (input_val_centered <= -input_range_radius) {
      output_val = -128;
    } else if (input_val_centered >= input_range_radius) {
      output_val = 127;
    } else {
      using gemmlowp::SaturatingRoundingDoublingHighMul;
      const int16 input_val_rescaled = SaturatingRoundingDoublingHighMul(
          static_cast<int16>(input_val_centered * (1 << input_left_shift)),
          static_cast<int16>(input_multiplier));
      using FixedPoint4 = gemmlowp::FixedPoint<int16, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int16, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::tanh(input_val_f4);
      using gemmlowp::RoundingDivideByPOT;
      int16 output_val_s16 = RoundingDivideByPOT(output_val_f0.raw(), 8);
      if (output_val_s16 == 128) {
        output_val_s16 = 127;
      }
      TFLITE_DCHECK_GE(output_val_s16, -128);
      TFLITE_DCHECK_LE(output_val_s16, 127);
      output_val = static_cast<int8>(output_val_s16);
    }
    output_data[c] = output_val;
  }
}

}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_ACTIVATIONS_H_
