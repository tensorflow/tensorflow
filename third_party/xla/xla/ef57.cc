/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/ef57.h"

#include <limits>
#include <tuple>

#include "absl/types/span.h"
#include "xla/compiler_macros.h"
#include "tsl/platform/logging.h"

#ifdef XLA_HAS_SSE2
#include <immintrin.h>  // IWYU pragma: keep
#endif

#if defined(XLA_HAS_ARM_NEON) && defined(XLA_HAS_ARM64)
#include <arm_neon.h>  // IWYU pragma: keep
#endif

namespace xla {

void ConvertF64ToEf57(absl::Span<const double> input,
                      absl::Span<float> output) {
  DCHECK_EQ(input.size() * 2, output.size());
#ifdef __AVX__
  constexpr int kDoublesPerAvxIteration = sizeof(__m256d) / sizeof(double);
  constexpr int kFloatsPerSseRegister = sizeof(__m128) / sizeof(float);
  while (input.size() >= kDoublesPerAvxIteration) {
    __m256d x = _mm256_loadu_pd(input.data());

    __m128 x_hi_f32 = _mm256_cvtpd_ps(x);
    __m256d x_hi_f64 = _mm256_cvtps_pd(x_hi_f32);
    __m256d x_lo_f64 = _mm256_sub_pd(x, x_hi_f64);
    __m128 x_lo_f32 = _mm256_cvtpd_ps(x_lo_f64);

    const __m128 inf = _mm_set1_ps(std::numeric_limits<float>::infinity());
    __m128 x_hi_exponent = _mm_and_ps(x_hi_f32, inf);
    __m128 x_is_finite = _mm_cmplt_ps(x_hi_exponent, inf);
    x_lo_f32 = _mm_and_ps(x_lo_f32, x_is_finite);

    _mm_storeu_ps(output.data(), _mm_unpacklo_ps(x_hi_f32, x_lo_f32));
    output.remove_prefix(kFloatsPerSseRegister);
    _mm_storeu_ps(output.data(), _mm_unpackhi_ps(x_hi_f32, x_lo_f32));
    output.remove_prefix(kFloatsPerSseRegister);

    input.remove_prefix(kDoublesPerAvxIteration);
  }
#endif
#ifdef XLA_HAS_SSE2
  constexpr int kDoublesPerSseIteration = sizeof(__m128d) / sizeof(double);
  constexpr int kFloatsPerSseIteration = sizeof(__m128) / sizeof(float);
  while (input.size() >= kDoublesPerSseIteration) {
    __m128d x = _mm_loadu_pd(input.data());
    __m128 x_hi_f32 = _mm_cvtpd_ps(x);
    __m128d x_hi_f64 = _mm_cvtps_pd(x_hi_f32);
    __m128d x_lo_f64 = _mm_sub_pd(x, x_hi_f64);
    __m128 x_lo_f32 = _mm_cvtpd_ps(x_lo_f64);

    const __m128 inf = _mm_set1_ps(std::numeric_limits<float>::infinity());
    __m128 x_hi_exponent = _mm_and_ps(x_hi_f32, inf);
    __m128 x_is_finite = _mm_cmplt_ps(x_hi_exponent, inf);
    x_lo_f32 = _mm_and_ps(x_lo_f32, x_is_finite);

    __m128 to_store = _mm_unpacklo_ps(x_hi_f32, x_lo_f32);
    _mm_storeu_ps(output.data(), to_store);

    input.remove_prefix(kDoublesPerSseIteration);
    output.remove_prefix(kFloatsPerSseIteration);
  }
#endif
#if defined(XLA_HAS_ARM_NEON) && defined(XLA_HAS_ARM64)
  constexpr int kDoublesPerNeonIteration = sizeof(float64x2_t) / sizeof(double);
  constexpr int kFloatsPerNeonIteration = sizeof(float32x2x2_t) / sizeof(float);
  while (input.size() >= kDoublesPerNeonIteration) {
    float64x2_t x = vld1q_f64(input.data());
    float32x2_t x_hi_f32 = vcvt_f32_f64(x);
    float64x2_t x_hi_f64 = vcvt_f64_f32(x_hi_f32);
    float64x2_t x_lo_f64 = vsubq_f64(x, x_hi_f64);
    float32x2_t x_lo_f32 = vcvt_f32_f64(x_lo_f64);

    uint32x2_t x_is_finite =
        vcalt_f32(x_hi_f32, vdup_n_f32(std::numeric_limits<float>::infinity()));
    x_lo_f32 = vreinterpret_f32_u32(
        vand_u32(vreinterpret_u32_f32(x_lo_f32), x_is_finite));

    float32x2x2_t to_store;
    to_store.val[0] = x_hi_f32;
    to_store.val[1] = x_lo_f32;
    vst2_f32(output.data(), to_store);

    input.remove_prefix(kDoublesPerNeonIteration);
    output.remove_prefix(kFloatsPerNeonIteration);
  }
#endif

  while (input.size() >= 1) {
    std::tie(output[0], output[1]) = SplitF64ToF32(input.front());
    input.remove_prefix(1);
    output.remove_prefix(2);
  }
}

}  // namespace xla
