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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_DIV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_DIV_H_

#include "public/gemmlowp.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_integer_ops {

// Element-wise Div that can often be used for inner loop of broadcast Div as
// well as the non-broadcast Div.
inline void DivElementwise(bool use_unswitched, int size,
                           const ArithmeticParams& params,
                           const int8* input1_data, const int8* input2_data,
                           int8* output_data) {
  gemmlowp::ScopedProfilingLabel label("DivElementwiseInt8/8bit");
  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  TFLITE_DCHECK_GT(params.output_offset, -256);
  TFLITE_DCHECK_LT(params.output_offset, 256);

  for (; i < size; ++i) {
    const int32 input1_val = params.input1_offset + input1_data[i];
    int32 input2_val = params.input2_offset + input2_data[i];
    int32 mul1, mul2;
    using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;

    if (use_unswitched) {
      mul1 = input1_val;
    } else {
      mul1 = input2_val;
      input2_val = input1_val;
    }

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

    mul2 = MultiplyByQuantizedMultiplier(unsat_output, params.input1_multiplier,
                                         params.input1_shift);

    const int32 unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplier(sign_multiplier * mul1 * mul2,
                                      params.output_multiplier,
                                      params.output_shift);

    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));

    output_data[i] = static_cast<int8>(clamped_output);
  }
}

// Broadcast Div that can often be used for inner loop of broadcast Div.
inline void DivSimpleBroadcast(bool use_unswitched, int size,
                               const ArithmeticParams& params,
                               const int8 broadcast_value,
                               const int8* input2_data, int8* output_data) {
  gemmlowp::ScopedProfilingLabel label("DivSimpleBroadcastInt8/8bit");
  const int16 input1_val = params.input1_offset + broadcast_value;

  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  TFLITE_DCHECK_GT(params.output_offset, -256);
  TFLITE_DCHECK_LT(params.output_offset, 256);

  for (; i < size; ++i) {
    int32 input2_val = params.input2_offset + input2_data[i];
    int32 mul1, mul2;
    using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;

    if (use_unswitched) {
      mul1 = input1_val;
    } else {
      mul1 = input2_val;
      input2_val = input1_val;
    }

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

    mul2 = MultiplyByQuantizedMultiplier(unsat_output, params.input1_multiplier,
                                         params.input1_shift);

    const int32 unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplier(sign_multiplier * mul1 * mul2,
                                      params.output_multiplier,
                                      params.output_shift);
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<int8>(clamped_output);
  }
}

inline void Div(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int8* input1_data,
                const RuntimeShape& input2_shape, const int8* input2_data,
                const RuntimeShape& output_shape, int8* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  gemmlowp::ScopedProfilingLabel label("DivInt8/8bit");
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);

  DivElementwise(true, flat_size, params, input1_data, input2_data,
                 output_data);
}

inline void BroadcastDivFivefold(const ArithmeticParams& unswitched_params,
                                 const RuntimeShape& unswitched_input1_shape,
                                 const int8* unswitched_input1_data,
                                 const RuntimeShape& unswitched_input2_shape,
                                 const int8* unswitched_input2_data,
                                 const RuntimeShape& output_shape,
                                 int8* output_data) {
  gemmlowp::ScopedProfilingLabel label("BroadcastDivFivefoldInt8/8bit");

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
            DivElementwise(use_unswitched, y4, params, input1_data_ptr,
                           input2_data_ptr, output_data_ptr);
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
          DivSimpleBroadcast(use_unswitched, y3, params, *input1_data_ptr,
                             input2_data_ptr, output_data_ptr);
          input2_data_ptr += y3;
          output_data_ptr += y3;
          ++input1_data_ptr;
        }
      }
      input2_data_reset = input2_data_ptr;
    }
  }
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_DIV_H_
