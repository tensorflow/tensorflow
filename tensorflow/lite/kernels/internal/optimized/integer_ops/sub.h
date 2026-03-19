/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_SUB_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_SUB_H_

#include <algorithm>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/avx2_quantization_utils.h"
#include "tensorflow/lite/kernels/internal/reference/sub.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_integer_ops {

inline void SubElementwiseInt16(int size, const ArithmeticParams& params,
                                const int16* input1_data,
                                const int16* input2_data, int16* output_data) {
  ruy::profiler::ScopeLabel label("SubElementwiseInt16/16bit");
  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -32768);
  TFLITE_DCHECK_GT(params.input2_offset, -32768);
  TFLITE_DCHECK_LT(params.input1_offset, 32768);
  TFLITE_DCHECK_LT(params.input2_offset, 32768);

#ifdef __AVX2__
  const int32_t input1_left_shift = params.left_shift + params.input1_shift;
  const int32_t input2_left_shift = params.left_shift + params.input2_shift;
  const __m256i input1_offset = _mm256_set1_epi32(params.input1_offset);
  const __m256i input2_offset = _mm256_set1_epi32(params.input2_offset);
  const __m256i output_offset = _mm256_set1_epi32(params.output_offset);
  const __m256i clamp_max_v =
      _mm256_set1_epi32(params.quantized_activation_max);
  const __m256i clamp_min_v =
      _mm256_set1_epi32(params.quantized_activation_min);

  for (; i <= size - 16; i += 16) {
    const __m256i input1_val_original =
        _mm256_loadu_si256(reinterpret_cast<__m256i const*>(input1_data + i));
    const __m256i input2_val_original =
        _mm256_loadu_si256(reinterpret_cast<__m256i const*>(input2_data + i));

    __m256i s11 =
        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(input1_val_original));
    __m256i s12 =
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(input1_val_original, 1));
    __m256i s21 =
        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(input2_val_original));
    __m256i s22 =
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(input2_val_original, 1));

    s11 = _mm256_add_epi32(s11, input1_offset);
    s12 = _mm256_add_epi32(s12, input1_offset);
    s21 = _mm256_add_epi32(s21, input2_offset);
    s22 = _mm256_add_epi32(s22, input2_offset);

    s11 = avx2_utils::MultiplyByQuantizedMultiplier(
        s11, params.input1_multiplier, input1_left_shift);
    s12 = avx2_utils::MultiplyByQuantizedMultiplier(
        s12, params.input1_multiplier, input1_left_shift);
    s21 = avx2_utils::MultiplyByQuantizedMultiplier(
        s21, params.input2_multiplier, input2_left_shift);
    s22 = avx2_utils::MultiplyByQuantizedMultiplier(
        s22, params.input2_multiplier, input2_left_shift);

    __m256i s1 = _mm256_sub_epi32(s11, s21);
    __m256i s2 = _mm256_sub_epi32(s12, s22);

    s1 = avx2_utils::MultiplyByQuantizedMultiplier(s1, params.output_multiplier,
                                                   params.output_shift);
    s2 = avx2_utils::MultiplyByQuantizedMultiplier(s2, params.output_multiplier,
                                                   params.output_shift);

    s1 = _mm256_add_epi32(s1, output_offset);
    s2 = _mm256_add_epi32(s2, output_offset);

    s1 = _mm256_min_epi32(s1, clamp_max_v);
    s1 = _mm256_max_epi32(s1, clamp_min_v);
    s2 = _mm256_min_epi32(s2, clamp_max_v);
    s2 = _mm256_max_epi32(s2, clamp_min_v);

    avx2_utils::CastInt32ToInt16AndStore(output_data + i, s1);
    avx2_utils::CastInt32ToInt16AndStore(output_data + i + 8, s2);
  }
#endif  // __AVX2__

  for (; i < size; ++i) {
    const int32_t input1_val = params.input1_offset + input1_data[i];
    const int32_t input2_val = params.input2_offset + input2_data[i];
    const int32_t shifted_input1_val = input1_val * (1 << params.left_shift);
    const int32_t shifted_input2_val = input2_val * (1 << params.left_shift);
    const int32_t scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, params.input1_multiplier, params.input1_shift);
    const int32_t scaled_input2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2_val, params.input2_multiplier, params.input2_shift);
    const int32_t raw_sum = scaled_input1_val - scaled_input2_val;
    const int32_t raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sum, params.output_multiplier, params.output_shift) +
        params.output_offset;
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output));
    output_data[i] = static_cast<int16>(clamped_output);
  }
}

inline void BroadcastSubFiveFold(const ArithmeticParams& unswitched_params,
                                 const RuntimeShape& input1_shape,
                                 const int16* unswitched_input1_data,
                                 const RuntimeShape& input2_shape,
                                 const int16* unswitched_input2_data,
                                 const RuntimeShape& output_shape,
                                 int16* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSubFiveFold/16bit");

  ArithmeticParams switched_params = unswitched_params;
  switched_params.input1_offset = unswitched_params.input2_offset;
  switched_params.input1_multiplier = unswitched_params.input2_multiplier;
  switched_params.input1_shift = unswitched_params.input2_shift;
  switched_params.input2_offset = unswitched_params.input1_offset;
  switched_params.input2_multiplier = unswitched_params.input1_multiplier;
  switched_params.input2_shift = unswitched_params.input1_shift;

  const bool use_unswitched =
      unswitched_params.broadcast_category ==
      tflite::BroadcastableOpCategory::kFirstInputBroadcastsFast;

  const ArithmeticParams& params =
      use_unswitched ? unswitched_params : switched_params;
  const int16_t* input1_data =
      use_unswitched ? unswitched_input1_data : unswitched_input2_data;
  const int16_t* input2_data =
      use_unswitched ? unswitched_input2_data : unswitched_input1_data;

  int16_t* output_data_ptr = output_data;
  const int16_t* input1_data_ptr = input1_data;
  const int16_t* input2_data_reset = input2_data;
  // In the fivefold pattern, y0, y2 and y4 are not broadcast, and so shared
  // between input shapes. y3 for input 1 is always broadcast, and so the
  // dimension there is 1, whereas optionally y1 might be broadcast for input 2.
  // The flatsize for each inputs are as below.
  // input1.shape.FlatSize = y0 * y1 * y2 * y4,
  // input2.shape.FlatSize = y0 * y2 * y3 * y4.
  const int y0 = params.broadcast_shape[0];
  const int y1 = params.broadcast_shape[1];
  const int y2 = params.broadcast_shape[2];
  const int y3 = params.broadcast_shape[3];
  const int y4 = params.broadcast_shape[4];
  for (int i0 = 0; i0 < y0; ++i0) {
    const int16_t* input2_data_ptr = nullptr;
    for (int i1 = 0; i1 < y1; ++i1) {
      input2_data_ptr = input2_data_reset;
      for (int i2 = 0; i2 < y2; ++i2) {
        for (int i3 = 0; i3 < y3; ++i3) {
          if (use_unswitched) {
            SubElementwiseInt16(y4, params, input1_data_ptr, input2_data_ptr,
                                output_data_ptr);
          } else {
            // When input1 and input2 are switched, calculate (input2 - input1)
            // and use unswitched_params as we switch the switched input here.
            SubElementwiseInt16(y4, unswitched_params, input2_data_ptr,
                                input1_data_ptr, output_data_ptr);
          }
          input2_data_ptr += y4;
          output_data_ptr += y4;
        }
        // We have broadcast y4 of input1 data y3 times, and now move on.
        input1_data_ptr += y4;
      }
    }
    // We have broadcast y2*y3*y4 of input2 data y1 times, and now move on.
    input2_data_reset = input2_data_ptr;
  }
}

inline void Sub(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16* input1_data,
                const RuntimeShape& input2_shape, const int16* input2_data,
                const RuntimeShape& output_shape, int16* output_data) {
  ruy::profiler::ScopeLabel label("SubInt16/16bit");
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_GT(params.input1_offset, -32768);
  TFLITE_DCHECK_GT(params.input2_offset, -32768);
  TFLITE_DCHECK_LT(params.input1_offset, 32768);
  TFLITE_DCHECK_LT(params.input2_offset, 32768);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  SubElementwiseInt16(flat_size, params, input1_data, input2_data, output_data);
}

inline void BroadcastSubDispatch(const ArithmeticParams& params,
                                 const RuntimeShape& input1_shape,
                                 const int16* input1_data,
                                 const RuntimeShape& input2_shape,
                                 const int16* input2_data,
                                 const RuntimeShape& output_shape,
                                 int16* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSubDispatchInt16/16bit");
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_GT(params.input1_offset, -32768);
  TFLITE_DCHECK_GT(params.input2_offset, -32768);
  TFLITE_DCHECK_LT(params.input1_offset, 32768);
  TFLITE_DCHECK_LT(params.input2_offset, 32768);

  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast) {
    return reference_ops::BroadcastQuantSubSlow(
        params, input1_shape, input1_data, input2_shape, input2_data,
        output_shape, output_data);
  }

  BroadcastSubFiveFold(params, input1_shape, input1_data, input2_shape,
                       input2_data, output_shape, output_data);
}
}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_SUB_H_
