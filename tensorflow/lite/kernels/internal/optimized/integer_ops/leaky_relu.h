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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_LEAKY_RELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_LEAKY_RELU_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/optimized/avx2_quantization_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_integer_ops {

inline void QuantizeLeakyRelu(const LeakyReluParams& params,
                              const RuntimeShape& input_shape,
                              const int16* input_data,
                              const RuntimeShape& output_shape,
                              int16* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  const int32_t quantized_min = std::numeric_limits<int16>::min();
  const int32_t quantized_max = std::numeric_limits<int16>::max();
  int i = 0;

#ifdef __AVX2__
  const __m256i input_offset = _mm256_set1_epi32(params.input_offset);
  const __m256i output_offset = _mm256_set1_epi32(params.output_offset);
  const __m256i output_muliplier_identity =
      _mm256_set1_epi32(params.output_multiplier_identity);
  const __m256i output_shift_identity =
      _mm256_set1_epi32(params.output_shift_identity);
  const __m256i output_multiplier_alpha =
      _mm256_set1_epi32(params.output_multiplier_alpha);
  const __m256i output_shift_alpha =
      _mm256_set1_epi32(params.output_shift_alpha);
  const __m256i clamp_max_v = _mm256_set1_epi32(quantized_max);
  const __m256i clamp_min_v = _mm256_set1_epi32(quantized_min);

  for (; i <= flat_size - 16; i += 16) {
    const __m256i input =
        _mm256_loadu_si256(reinterpret_cast<__m256i const*>(input_data + i));
    __m256i input_low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(input));
    __m256i input_high =
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(input, 1));
    input_low = _mm256_sub_epi32(input_low, input_offset);
    input_high = _mm256_sub_epi32(input_high, input_offset);

    const __m256i zeros = _mm256_setzero_si256();
    const __m256i input_low_mask = _mm256_cmpgt_epi32(input_low, zeros);
    const __m256i input_high_mask = _mm256_cmpgt_epi32(input_high, zeros);
    const __m256i input_low_output_multiplier = avx2_utils::mm256_blendv_epi32(
        output_multiplier_alpha, output_muliplier_identity, input_low_mask);
    const __m256i input_low_output_shift = avx2_utils::mm256_blendv_epi32(
        output_shift_alpha, output_shift_identity, input_low_mask);
    const __m256i input_high_output_multiplier = avx2_utils::mm256_blendv_epi32(
        output_multiplier_alpha, output_muliplier_identity, input_high_mask);
    const __m256i input_high_output_shift = avx2_utils::mm256_blendv_epi32(
        output_shift_alpha, output_shift_identity, input_high_mask);

    input_low = avx2_utils::MultiplyByQuantizedMultiplier(
        input_low, input_low_output_multiplier, input_low_output_shift);
    input_high = avx2_utils::MultiplyByQuantizedMultiplier(
        input_high, input_high_output_multiplier, input_high_output_shift);

    input_low = _mm256_add_epi32(input_low, output_offset);
    input_high = _mm256_add_epi32(input_high, output_offset);

    input_low = _mm256_min_epi32(input_low, clamp_max_v);
    input_low = _mm256_max_epi32(input_low, clamp_min_v);
    input_high = _mm256_min_epi32(input_high, clamp_max_v);
    input_high = _mm256_max_epi32(input_high, clamp_min_v);

    avx2_utils::CastInt32ToInt16AndStore(output_data + i, input_low);
    avx2_utils::CastInt32ToInt16AndStore(output_data + i + 8, input_high);
  }
#endif  // __AVX2__

  for (; i < flat_size; ++i) {
    const int32_t input_value = input_data[i] - params.input_offset;
    int32_t unclamped_output;
    if (input_value >= 0) {
      unclamped_output = params.output_offset +
                         MultiplyByQuantizedMultiplier(
                             input_value, params.output_multiplier_identity,
                             params.output_shift_identity);
    } else {
      unclamped_output = params.output_offset +
                         MultiplyByQuantizedMultiplier(
                             input_value, params.output_multiplier_alpha,
                             params.output_shift_alpha);
    }
    const int16 clamped_output =
        std::min(quantized_max, std::max(quantized_min, unclamped_output));
    output_data[i] = static_cast<int16>(clamped_output);
  }
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_LEAKY_RELU_H_
