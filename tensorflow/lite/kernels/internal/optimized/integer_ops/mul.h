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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MUL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MUL_H_

#include "tensorflow/lite/experimental/ruy/profiler/instrumentation.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mul.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_integer_ops {

// Element-wise mul that can often be used for inner loop of broadcast Mul as
// well as the non-broadcast Mul.
inline void MulElementwise(int size, const ArithmeticParams& params,
                           const int8* input1_data, const int8* input2_data,
                           int8* output_data) {
  ruy::profiler::ScopeLabel label("MulElementwiseInt8/8bit");
  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  TFLITE_DCHECK_GT(params.output_offset, -256);
  TFLITE_DCHECK_LT(params.output_offset, 256);
#ifdef USE_NEON
  const auto input1_offset_vector = vdupq_n_s16(params.input1_offset);
  const auto input2_offset_vector = vdupq_n_s16(params.input2_offset);
  const auto output_offset_vector = vdupq_n_s16(params.output_offset);
  const auto output_activation_min_vector =
      vdup_n_s8(params.quantized_activation_min);
  const auto output_activation_max_vector =
      vdup_n_s8(params.quantized_activation_max);
  const int left_shift = std::max(0, params.output_shift);
  const int right_shift = std::max(0, -params.output_shift);
  const int32x4_t left_shift_vec = vdupq_n_s32(left_shift);
  for (; i <= size - 8; i += 8) {
    // We load / store 8 at a time, multiplying as two sets of 4 int32s.
    const auto input1_val_original = vld1_s8(input1_data + i);
    const auto input2_val_original = vld1_s8(input2_data + i);
    const auto input1_val_s16 = vmovl_s8(input1_val_original);
    const auto input2_val_s16 = vmovl_s8(input2_val_original);
    const auto input1_val = vaddq_s16(input1_val_s16, input1_offset_vector);
    const auto input2_val = vaddq_s16(input2_val_s16, input2_offset_vector);

    const auto input1_val_low = vget_low_s16(input1_val);
    const auto input1_val_high = vget_high_s16(input1_val);
    const auto input2_val_low = vget_low_s16(input2_val);
    const auto input2_val_high = vget_high_s16(input2_val);

    auto p1 = vmull_s16(input2_val_low, input1_val_low);
    auto p2 = vmull_s16(input2_val_high, input1_val_high);

    p1 = vshlq_s32(p1, left_shift_vec);
    p2 = vshlq_s32(p2, left_shift_vec);
    p1 = vqrdmulhq_n_s32(p1, params.output_multiplier);
    p2 = vqrdmulhq_n_s32(p2, params.output_multiplier);
    using gemmlowp::RoundingDivideByPOT;
    p1 = RoundingDivideByPOT(p1, right_shift);
    p2 = RoundingDivideByPOT(p2, right_shift);

    const auto p1_narrowed = vqmovn_s32(p1);
    const auto p2_narrowed = vqmovn_s32(p2);
    const auto p =
        vaddq_s16(vcombine_s16(p1_narrowed, p2_narrowed), output_offset_vector);
    const auto clamped =
        vmax_s8(output_activation_min_vector,
                vmin_s8(output_activation_max_vector, vqmovn_s16(p)));
    vst1_s8(output_data + i, clamped);
  }
#endif  // NEON

  for (; i < size; ++i) {
    const int32 input1_val = params.input1_offset + input1_data[i];
    const int32 input2_val = params.input2_offset + input2_data[i];
    const int32 unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                      params.output_multiplier,
                                      params.output_shift);
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<int8>(clamped_output);
  }
}

// Broadcast mul that can often be used for inner loop of broadcast Mul.
inline void MulSimpleBroadcast(int size, const ArithmeticParams& params,
                               const int8 broadcast_value,
                               const int8* input2_data, int8* output_data) {
  ruy::profiler::ScopeLabel label("BroadMulSimpleBroadcastInt8/8bit");
  const int16 input1_val = params.input1_offset + broadcast_value;

  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  TFLITE_DCHECK_GT(params.output_offset, -256);
  TFLITE_DCHECK_LT(params.output_offset, 256);
#ifdef USE_NEON
  const auto input2_offset_vector = vdupq_n_s16(params.input2_offset);
  const auto output_offset_vector = vdupq_n_s16(params.output_offset);
  const auto output_activation_min_vector =
      vdup_n_s8(params.quantized_activation_min);
  const auto output_activation_max_vector =
      vdup_n_s8(params.quantized_activation_max);
  const int left_shift = std::max(0, params.output_shift);
  const int right_shift = std::max(0, -params.output_shift);
  const int32x4_t left_shift_vec = vdupq_n_s32(left_shift);
  for (; i <= size - 8; i += 8) {
    // We load / store 8 at a time, multiplying as two sets of 4 int32s.
    const auto input2_val_original = vld1_s8(input2_data + i);
    const auto input2_val_s16 = vmovl_s8(input2_val_original);
    const auto input2_val = vaddq_s16(input2_val_s16, input2_offset_vector);

    const auto input2_val_low = vget_low_s16(input2_val);
    const auto input2_val_high = vget_high_s16(input2_val);

    auto p1 = vmull_n_s16(input2_val_low, input1_val);
    auto p2 = vmull_n_s16(input2_val_high, input1_val);

    p1 = vshlq_s32(p1, left_shift_vec);
    p2 = vshlq_s32(p2, left_shift_vec);
    p1 = vqrdmulhq_n_s32(p1, params.output_multiplier);
    p2 = vqrdmulhq_n_s32(p2, params.output_multiplier);
    using gemmlowp::RoundingDivideByPOT;
    p1 = RoundingDivideByPOT(p1, right_shift);
    p2 = RoundingDivideByPOT(p2, right_shift);

    const auto p1_narrowed = vqmovn_s32(p1);
    const auto p2_narrowed = vqmovn_s32(p2);
    const auto p =
        vaddq_s16(vcombine_s16(p1_narrowed, p2_narrowed), output_offset_vector);
    const auto clamped =
        vmax_s8(output_activation_min_vector,
                vmin_s8(output_activation_max_vector, vqmovn_s16(p)));
    vst1_s8(output_data + i, clamped);
  }
#endif  // NEON

  for (; i < size; ++i) {
    const int32 input2_val = params.input2_offset + input2_data[i];
    const int32 unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                      params.output_multiplier,
                                      params.output_shift);
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<int8>(clamped_output);
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int8* input1_data,
                const RuntimeShape& input2_shape, const int8* input2_data,
                const RuntimeShape& output_shape, int8* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  ruy::profiler::ScopeLabel label("MulInt8/8bit");
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  MulElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void BroadcastMulFivefold(const ArithmeticParams& unswitched_params,
                                 const RuntimeShape& unswitched_input1_shape,
                                 const int8* unswitched_input1_data,
                                 const RuntimeShape& unswitched_input2_shape,
                                 const int8* unswitched_input2_data,
                                 const RuntimeShape& output_shape,
                                 int8* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastMulFivefoldInt8/8bit");

  ArithmeticParams switched_params = unswitched_params;
  switched_params.input1_offset = unswitched_params.input2_offset;
  switched_params.input2_offset = unswitched_params.input1_offset;

  const bool use_unswitched =
      unswitched_params.broadcast_category ==
      tflite::BroadcastableOpCategory::kFirstInputBroadcastsFast;

  const ArithmeticParams& params =
      use_unswitched ? unswitched_params : switched_params;
  const int8* input1_data =
      use_unswitched ? unswitched_input1_data : unswitched_input2_data;
  const int8* input2_data =
      use_unswitched ? unswitched_input2_data : unswitched_input1_data;

  // Fivefold nested loops. The second input resets its position for each
  // iteration of the second loop. The first input resets its position at the
  // beginning of the fourth loop. The innermost loop is an elementwise Mul of
  // sections of the arrays.
  int8* output_data_ptr = output_data;
  const int8* input1_data_ptr = input1_data;
  const int8* input2_data_reset = input2_data;
  int y0 = params.broadcast_shape[0];
  int y1 = params.broadcast_shape[1];
  int y2 = params.broadcast_shape[2];
  int y3 = params.broadcast_shape[3];
  int y4 = params.broadcast_shape[4];
  if (y4 > 1) {
    for (int i0 = 0; i0 < y0; ++i0) {
      const int8* input2_data_ptr = nullptr;
      for (int i1 = 0; i1 < y1; ++i1) {
        input2_data_ptr = input2_data_reset;
        for (int i2 = 0; i2 < y2; ++i2) {
          for (int i3 = 0; i3 < y3; ++i3) {
            MulElementwise(y4, params, input1_data_ptr, input2_data_ptr,
                           output_data_ptr);
            input2_data_ptr += y4;
            output_data_ptr += y4;
          }
          input1_data_ptr += y4;
        }
      }
      input2_data_reset = input2_data_ptr;
    }
  } else {
    for (int i0 = 0; i0 < y0; ++i0) {
      const int8* input2_data_ptr = nullptr;
      for (int i1 = 0; i1 < y1; ++i1) {
        input2_data_ptr = input2_data_reset;
        for (int i2 = 0; i2 < y2; ++i2) {
          MulSimpleBroadcast(y3, params, *input1_data_ptr, input2_data_ptr,
                             output_data_ptr);
          input2_data_ptr += y3;
          output_data_ptr += y3;
          ++input1_data_ptr;
        }
      }
      input2_data_reset = input2_data_ptr;
    }
  }
}

inline void BroadcastMulDispatch(const ArithmeticParams& params,
                                 const RuntimeShape& input1_shape,
                                 const int8* input1_data,
                                 const RuntimeShape& input2_shape,
                                 const int8* input2_data,
                                 const RuntimeShape& output_shape,
                                 int8* output_data) {
  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast) {
    return reference_integer_ops::BroadcastMul4DSlow(
        params, input1_shape, input1_data, input2_shape, input2_data,
        output_shape, output_data);
  }

  BroadcastMulFivefold(params, input1_shape, input1_data, input2_shape,
                       input2_data, output_shape, output_data);
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MUL_H_
