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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_SOFTMAX_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_SOFTMAX_H_

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/experimental/ruy/profiler/instrumentation.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"

namespace tflite {
namespace optimized_integer_ops {

// Quantized softmax with int8 input and output.
inline void Softmax(const SoftmaxParams& params,
                    const RuntimeShape& input_shape, const int8* input_data,
                    const RuntimeShape& output_shape, int8* output_data) {
  const int32 input_beta_multiplier = params.input_multiplier;
  const int32 input_beta_left_shift = params.input_left_shift;
  const int diff_min = params.diff_min;
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large as
  // -32 before multiplying by input_beta_multiplier, and therefore as large as
  // -16 afterwards.  Note that exp(-8) is definitely not insignificant to
  // accumulation, but exp(-16) definitely is.
  static const int kScaledDiffIntegerBits = 5;
  static const int kAccumulationIntegerBits = 12;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32, kScaledDiffIntegerBits>;
  using FixedPointAccum = gemmlowp::FixedPoint<int32, kAccumulationIntegerBits>;
  using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;

  ruy::profiler::ScopeLabel label("SoftmaxInt8/8bit");
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int b = 0; b < outer_size; ++b) {
    const int8* input_data_ptr = input_data + b * depth;
    int8* output_data_ptr = output_data + b * depth;

    // Determine the largest entry in the current row
    int8_t max_in_row = -128;
    {
      int c = 0;
#ifdef USE_NEON
      int8x16_t max16_0 = vdupq_n_s8(-128);
      int8x16_t max16_1 = vdupq_n_s8(-128);
      for (; c <= depth - 32; c += 32) {
        max16_0 = vmaxq_s8(max16_0, vld1q_s8(input_data_ptr + c + 0));
        max16_1 = vmaxq_s8(max16_1, vld1q_s8(input_data_ptr + c + 16));
      }
      int8x16_t max16 = vmaxq_s8(max16_0, max16_1);
      if (c <= depth - 16) {
        max16 = vmaxq_s8(max16, vld1q_s8(input_data_ptr + c));
        c += 16;
      }
      int8x8_t max8 = vmax_s8(vget_low_s8(max16), vget_high_s8(max16));
      if (c <= depth - 8) {
        max8 = vmax_s8(max8, vld1_s8(input_data_ptr + c));
        c += 8;
      }
      int8x8_t max4 = vmax_s8(max8, vext_s8(max8, max8, 4));
      int8x8_t max2 = vmax_s8(max4, vext_s8(max4, max4, 2));
      int8x8_t max1 = vpmax_s8(max2, max2);
      max_in_row = vget_lane_s8(max1, 0);
#endif
      for (; c < depth; ++c) {
        max_in_row = std::max(max_in_row, input_data_ptr[c]);
      }
    }

#ifdef USE_NEON
    using FixedPointAccumInt32x4 =
        gemmlowp::FixedPoint<int32x4_t, kAccumulationIntegerBits>;
    using FixedPointScaledDiffInt32x4 =
        gemmlowp::FixedPoint<int32x4_t, kScaledDiffIntegerBits>;
    using FixedPoint0Int32x4 = gemmlowp::FixedPoint<int32x4_t, 0>;
    FixedPoint0Int32x4 input_beta_multiplier_f0 =
        FixedPoint0Int32x4::FromScalarRaw(input_beta_multiplier);
    int16x8_t max_in_row_s16 = vdupq_n_s16(max_in_row);
#endif

    // Compute the sum of exponentials of the differences of entries in the
    // current row from the largest entry in the current row.
    FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
    {
      int c = 0;
#ifdef USE_NEON
      int32x4_t diff_min_s32 = vdupq_n_s32(diff_min);
      FixedPointAccumInt32x4 sum_of_exps_0 = FixedPointAccumInt32x4::Zero();
      FixedPointAccumInt32x4 sum_of_exps_1 = FixedPointAccumInt32x4::Zero();
      FixedPointAccumInt32x4 zeros = FixedPointAccumInt32x4::Zero();
      for (; c <= depth - 8; c += 8) {
        int16x8_t input_s16 = vmovl_s8(vld1_s8(input_data_ptr + c));
        int16x8_t input_diff_s16 = vsubq_s16(input_s16, max_in_row_s16);
        int32x4_t input_diff_s32_0 = vmovl_s16(vget_low_s16(input_diff_s16));
        int32x4_t input_diff_s32_1 = vmovl_s16(vget_high_s16(input_diff_s16));
        int32x4_t mask_0 =
            gemmlowp::MaskIfGreaterThanOrEqual(input_diff_s32_0, diff_min_s32);
        int32x4_t mask_1 =
            gemmlowp::MaskIfGreaterThanOrEqual(input_diff_s32_1, diff_min_s32);
        FixedPointScaledDiffInt32x4 scaled_diff_0 =
            input_beta_multiplier_f0 *
            FixedPointScaledDiffInt32x4::FromRaw(
                gemmlowp::ShiftLeft(input_diff_s32_0, input_beta_left_shift));
        FixedPointScaledDiffInt32x4 scaled_diff_1 =
            input_beta_multiplier_f0 *
            FixedPointScaledDiffInt32x4::FromRaw(
                gemmlowp::ShiftLeft(input_diff_s32_1, input_beta_left_shift));
        FixedPointAccumInt32x4 exps_0 =
            gemmlowp::Rescale<kAccumulationIntegerBits>(
                exp_on_negative_values(scaled_diff_0));
        FixedPointAccumInt32x4 exps_1 =
            gemmlowp::Rescale<kAccumulationIntegerBits>(
                exp_on_negative_values(scaled_diff_1));
        FixedPointAccumInt32x4 masked_exps_0 =
            SelectUsingMask(mask_0, exps_0, zeros);
        FixedPointAccumInt32x4 masked_exps_1 =
            SelectUsingMask(mask_1, exps_1, zeros);
        sum_of_exps_0 = sum_of_exps_0 + masked_exps_0;
        sum_of_exps_1 = sum_of_exps_1 + masked_exps_1;
      }
      int32x4_t sum_of_exps_reduced_4 = (sum_of_exps_0 + sum_of_exps_1).raw();
      int32x2_t sum_of_exps_reduced_2 =
          vadd_s32(vget_low_s32(sum_of_exps_reduced_4),
                   vget_high_s32(sum_of_exps_reduced_4));
      int32x2_t sum_of_exps_reduced_1 =
          vpadd_s32(sum_of_exps_reduced_2, sum_of_exps_reduced_2);
      sum_of_exps =
          FixedPointAccum::FromRaw(vget_lane_s32(sum_of_exps_reduced_1, 0));
#endif
      for (; c < depth; ++c) {
        int32 input_diff = static_cast<int32>(input_data_ptr[c]) - max_in_row;
        if (input_diff >= diff_min) {
          const int32 input_diff_rescaled =
              MultiplyByQuantizedMultiplierGreaterThanOne(
                  input_diff, input_beta_multiplier, input_beta_left_shift);
          const FixedPointScaledDiff scaled_diff_f8 =
              FixedPointScaledDiff::FromRaw(input_diff_rescaled);
          sum_of_exps =
              sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                exp_on_negative_values(scaled_diff_f8));
        }
      }
    }

    // Compute the fixed-point multiplier and shift that we need to apply to
    // perform a division by the above-computed sum-of-exponentials.
    int num_bits_over_unit;
    FixedPoint0 shifted_scale = FixedPoint0::FromRaw(GetReciprocal(
        sum_of_exps.raw(), kAccumulationIntegerBits, &num_bits_over_unit));

    // Compute the quotients of exponentials of differences of entries in the
    // current row from the largest entry, over the previously-computed sum of
    // exponentials.
    {
      int c = 0;
#ifdef USE_NEON
      int16x8_t diff_min_s16 = vdupq_n_s16(diff_min);
      for (; c <= depth - 8; c += 8) {
        int16x8_t input_s16 = vmovl_s8(vld1_s8(input_data_ptr + c));
        int16x8_t input_diff_s16 = vsubq_s16(input_s16, max_in_row_s16);
        int32x4_t input_diff_s32_0 = vmovl_s16(vget_low_s16(input_diff_s16));
        int32x4_t input_diff_s32_1 = vmovl_s16(vget_high_s16(input_diff_s16));
        uint8x8_t mask = vmovn_u16(vcgeq_s16(input_diff_s16, diff_min_s16));
        FixedPointScaledDiffInt32x4 scaled_diff_0 =
            input_beta_multiplier_f0 *
            FixedPointScaledDiffInt32x4::FromRaw(
                gemmlowp::ShiftLeft(input_diff_s32_0, input_beta_left_shift));
        FixedPointScaledDiffInt32x4 scaled_diff_1 =
            input_beta_multiplier_f0 *
            FixedPointScaledDiffInt32x4::FromRaw(
                gemmlowp::ShiftLeft(input_diff_s32_1, input_beta_left_shift));
        FixedPoint0Int32x4 exp_0 = exp_on_negative_values(scaled_diff_0);
        FixedPoint0Int32x4 exp_1 = exp_on_negative_values(scaled_diff_1);
        int32x4_t output_zero_point_s32 = vdupq_n_s32(-128);
        int32x4_t output_s32_0 =
            vaddq_s32(gemmlowp::RoundingDivideByPOT(
                          vqrdmulhq_n_s32(exp_0.raw(), shifted_scale.raw()),
                          num_bits_over_unit + 31 - 8),
                      output_zero_point_s32);
        int32x4_t output_s32_1 =
            vaddq_s32(gemmlowp::RoundingDivideByPOT(
                          vqrdmulhq_n_s32(exp_1.raw(), shifted_scale.raw()),
                          num_bits_over_unit + 31 - 8),
                      output_zero_point_s32);
        int16x8_t output_s16 =
            vcombine_s16(vqmovn_s32(output_s32_0), vqmovn_s32(output_s32_1));
        int8x8_t output_s8 = vqmovn_s16(output_s16);
        int8x8_t masked_output = vbsl_s8(mask, output_s8, vdup_n_s8(-128));
        vst1_s8(output_data_ptr + c, masked_output);
      }
#endif
      for (; c < depth; ++c) {
        int32 input_diff = static_cast<int32>(input_data_ptr[c]) - max_in_row;
        if (input_diff >= diff_min) {
          const int32 input_diff_rescaled =
              MultiplyByQuantizedMultiplierGreaterThanOne(
                  input_diff, input_beta_multiplier, input_beta_left_shift);
          const FixedPointScaledDiff scaled_diff_f8 =
              FixedPointScaledDiff::FromRaw(input_diff_rescaled);

          FixedPoint0 exp_in_0 = exp_on_negative_values(scaled_diff_f8);
          const int32 unsat_output = gemmlowp::RoundingDivideByPOT(
              (shifted_scale * exp_in_0).raw(), num_bits_over_unit + 31 - 8);
          const int32 shifted_output = unsat_output - 128;

          output_data_ptr[c] =
              static_cast<int8>(std::max(std::min(shifted_output, 127), -128));

        } else {
          output_data_ptr[c] = -128;
        }
      }
    }
  }
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_SOFTMAX_H_
