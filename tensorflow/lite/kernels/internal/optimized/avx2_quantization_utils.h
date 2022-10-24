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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_AVX2_QUANTIZATION_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_AVX2_QUANTIZATION_UTILS_H_
#ifdef __AVX2__

#include <immintrin.h>

#include <limits>

#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {
namespace avx2_utils {

static inline void mm_storeu_si64(void *dst, __m128i v) {
#if (defined __clang__) || (defined _MSC_VER)
  _mm_storeu_si64(dst, v);
#else
  // GCC 9 lacks support for _mm_storeu_si64.
  *static_cast<std::int64_t *>(dst) = _mm_extract_epi64(v, 0);
#endif
}

static inline __m256i mm256_blendv_epi32(const __m256i &a, const __m256i &b,
                                         const __m256i &mask) {
  __m256 result =
      _mm256_blendv_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b),
                       _mm256_castsi256_ps(mask));
  return _mm256_castps_si256(result);
}

static inline __m256i rounding_right_shift(const __m256i &value,
                                           int32_t right_shift) {
  TFLITE_DCHECK_GT(right_shift, 0);
  const int32_t one_shift_exp_minus1 = 1 << (right_shift - 1);
  __m256i nudge = _mm256_set1_epi32(one_shift_exp_minus1);
  const __m256i r_plus_nudge = _mm256_add_epi32(value, nudge);
  const __m256i shifted_sum =
      _mm256_srav_epi32(r_plus_nudge, _mm256_set1_epi32(right_shift));

  // Identify overflow in each lane and create mask.
  const __m256i mask_num_plus_nudge_overflow = _mm256_cmpgt_epi32(
      value, _mm256_set1_epi32(0x7fffffff - one_shift_exp_minus1));
  // Fill results with either (value + nudge) >> exponent or
  // std::numeric_limits<std::int32_t>::max() in the case of overflow.
  return mm256_blendv_epi32(
      shifted_sum, _mm256_set1_epi32(std::numeric_limits<std::int32_t>::max()),
      mask_num_plus_nudge_overflow);
}

static inline __m256i rounding_right_shift(const __m256i &value,
                                           const __m256i right_shift) {
  const __m256i zeros = _mm256_setzero_si256();
  const __m256i mask_rightshift_gtz = _mm256_cmpgt_epi32(right_shift, zeros);
  const __m256i one_shift_exp_minus1 =
      _mm256_sllv_epi32(_mm256_set1_epi32(1),
                        _mm256_sub_epi32(right_shift, _mm256_set1_epi32(1)));
  __m256i nudge =
      mm256_blendv_epi32(zeros, one_shift_exp_minus1, mask_rightshift_gtz);
  const __m256i r_plus_nudge = _mm256_add_epi32(value, nudge);
  const __m256i shifted_sum = _mm256_srav_epi32(r_plus_nudge, right_shift);

  // Identify overflow in each lane and create mask.
  const __m256i mask_num_plus_nudge_overflow = _mm256_cmpgt_epi32(
      value, _mm256_sub_epi32(_mm256_set1_epi32(0x7fffffff), nudge));
  // Fill results with either (value + nudge) >> exponent or
  // std::numeric_limits<std::int32_t>::max() in the case of overflow.
  return mm256_blendv_epi32(
      shifted_sum, _mm256_set1_epi32(std::numeric_limits<std::int32_t>::max()),
      mask_num_plus_nudge_overflow);
}

inline void CastInt32ToInt16AndStore(int16 *dst, const __m256i v) {
  // As _mm256_cvtepi32_epi16 is not supported in AVX2, use the below repack.
  // Select bytes 0, 1, 4, 5, 8, 9, 12, 13 within each lane, effectively
  // truncating each 16-bit integer.
  const __m256i repack_perm = _mm256_set1_epi64x(0x0d0c090805040100);
  const __m256i shuffled_v = _mm256_shuffle_epi8(v, repack_perm);
  mm_storeu_si64(dst, _mm256_extracti128_si256(shuffled_v, 0));
  mm_storeu_si64(dst + 4, _mm256_extracti128_si256(shuffled_v, 1));
}

inline __m256i MultiplyByQuantizedMultiplier(const __m256i &value,
                                             const int32_t multiplier,
                                             const int32_t left_shift) {
  const __m256i repack_perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
  const __m256i shifted_value =
      left_shift > 0 ? _mm256_sllv_epi32(value, _mm256_set1_epi32(left_shift))
                     : value;

  __m256i scaled_v_low = _mm256_mul_epi32(
      _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_value, 0)),
      _mm256_set1_epi64x(multiplier));
  __m256i scaled_v_high = _mm256_mul_epi32(
      _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_value, 1)),
      _mm256_set1_epi64x(multiplier));

  scaled_v_low = _mm256_srlv_epi64(scaled_v_low, _mm256_set1_epi64x(31));
  scaled_v_high = _mm256_srlv_epi64(scaled_v_high, _mm256_set1_epi64x(31));
  // As _mm256_cvtepi64_epi32 is not supported in AVX2, use the below permute.
  scaled_v_high = _mm256_slli_epi64(scaled_v_high, 32);
  __m256i result = _mm256_blend_epi32(scaled_v_low, scaled_v_high, 0xaa);
  result = _mm256_permutevar8x32_epi32(result, repack_perm);
  if (left_shift >= 0) {
    return result;
  }
  return rounding_right_shift(result, -left_shift);
}

inline __m256i MultiplyByQuantizedMultiplier(const __m256i &value,
                                             const __m256i multiplier,
                                             const __m256i left_shift) {
  const __m256i zero_vector = _mm256_setzero_si256();
  const __m256i positive_left_shift = _mm256_max_epi32(left_shift, zero_vector);
  const __m256i positive_right_shift =
      _mm256_max_epi32(_mm256_sub_epi32(zero_vector, left_shift), zero_vector);

  const __m256i repack_perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
  const __m256i shifted_value = _mm256_sllv_epi32(value, positive_left_shift);

  const __m256i multiplier_low =
      _mm256_cvtepi32_epi64(_mm256_extracti128_si256(multiplier, 0));
  const __m256i multiplier_high =
      _mm256_cvtepi32_epi64(_mm256_extracti128_si256(multiplier, 1));

  __m256i scaled_v_low = _mm256_mul_epi32(
      _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_value, 0)),
      multiplier_low);
  __m256i scaled_v_high = _mm256_mul_epi32(
      _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_value, 1)),
      multiplier_high);

  scaled_v_low = _mm256_srlv_epi64(scaled_v_low, _mm256_set1_epi64x(31));
  scaled_v_high = _mm256_srlv_epi64(scaled_v_high, _mm256_set1_epi64x(31));
  // As _mm256_cvtepi64_epi32 is not supported in AVX2, use the below permute.
  scaled_v_high = _mm256_slli_epi64(scaled_v_high, 32);
  __m256i result = _mm256_blend_epi32(scaled_v_low, scaled_v_high, 0xaa);
  result = _mm256_permutevar8x32_epi32(result, repack_perm);

  return rounding_right_shift(result, positive_right_shift);
}
}  // namespace avx2_utils
}  // namespace tflite

#endif  // __AVX2__
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_AVX2_QUANTIZATION_UTILS_H_
