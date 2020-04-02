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
#include "tensorflow/lite/kernels/internal/optimized/sse_tensor_utils_impl.h"

#ifdef __SSSE3__

#include <emmintrin.h>  // SSE2
#include <tmmintrin.h>  // SSSE3
#ifdef __SSE4_1__
#include <smmintrin.h>  // SSE4.1
#endif

#include <cstdint>

#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {
namespace tensor_utils {
namespace {

// Dot product of four int8 vectors of 4 elements packed into a XMM register.
// Result is four int32 scalars packed into a XMM register.
// int8x4x4 · int8x4x4 => int32x4
static inline __m128i DotProdInt8x4x4(__m128i a_8x16, __m128i b_8x16) {
  // Transfer sign from 'a' to 'b', as _mm_maddubs_epi16 treats 'a' unsigned.
  b_8x16 = _mm_sign_epi8(b_8x16, a_8x16);
  a_8x16 = _mm_abs_epi8(a_8x16);
  // sumprod[i] = a[2*i]*b[2*i] + a[2*i+1]*b[2*i+1] (i = 0..7)
  __m128i sumprod_16x8 = _mm_maddubs_epi16(a_8x16, b_8x16);
  // sumprod[i] = sumprod[2*i]*1 + sumprod[2*i+1]*1 (i = 0..3)
  return _mm_madd_epi16(sumprod_16x8, _mm_set1_epi16(1));
}

// Horizontally add 4 int32 values stored in a single XMM register to int32_t.
static inline int32_t ReduceInt32x4(__m128i acc) {
  // Shuffle to contain high half of acc (both in high and low halfs).
  __m128i shuffle = _mm_unpackhi_epi64(acc, acc);
  // Add shuffle and acc; low half is sums of twos (high half is ignored).
  acc = _mm_add_epi32(acc, shuffle);
  // Shuffle the two elements in low half (ignore high half).
  shuffle = _mm_shuffle_epi32(acc, _MM_SHUFFLE(2, 3, 0, 1));
  // Add shuffle and acc; lowest element is sum of all 4 input.
  acc = _mm_add_epi32(acc, shuffle);
  // Return lowest element as int32_t.
  return _mm_cvtsi128_si32(acc);
}

// Horizontally add each of 4 XMM registers with 4 int32 values, pack result
// into a single XMM register. Similar to ReduceInt32x4, but with 4x inputs.
static inline __m128i ReduceInt32x4x4(__m128i a, __m128i b, __m128i c,
                                      __m128i d) {
  // Assuming x = [x0, x1, x2, x3]
  const __m128i a_b_lo_half = _mm_unpacklo_epi32(a, b);  // [a0, b0, a1, b1]
  const __m128i a_b_hi_half = _mm_unpackhi_epi32(a, b);  // [a2, b2, a3, b3]
  const __m128i a_plus_b =
      _mm_add_epi32(a_b_lo_half, a_b_hi_half);  // [a0+a2, b0+b2, a1+a3, b1+b3]
  const __m128i c_d_lo_half = _mm_unpacklo_epi32(c, d);  // [c0, d0, c1, d1]
  const __m128i c_d_hi_half = _mm_unpackhi_epi32(c, d);  // [c2, d2, c3, d3]
  const __m128i c_plus_d =
      _mm_add_epi32(c_d_lo_half, c_d_hi_half);  // [c0+c2, d0+d2, c1+c3, d1+d3]
  const __m128i all_evns =
      _mm_unpacklo_epi64(a_plus_b, c_plus_d);  // [a02, b02, c02, d02]
  const __m128i all_odds =
      _mm_unpackhi_epi64(a_plus_b, c_plus_d);  // [a13, b13, c13, d13]
  return _mm_add_epi32(all_evns, all_odds);    // [a0123, b0123, c0123, d0123]
}

// Returns the ith element of a XMM register holding float numbers.
template <int i>
float GetFloatVectorElement(__m128 v) {
  static_assert(i >= 0 && i < 4, "The index must be 0 <= i < 4.");
  // Note, _mm_extract_ps returns int, so we can't use it here.
  // These lines will be optimized to extractps anyway.
  v = _mm_shuffle_ps(v, v, _MM_SHUFFLE(i, i, i, i));
  return _mm_cvtss_f32(v);
}

}  // namespace

void SseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors,
    const float* __restrict__ scaling_factors, int n_batch,
    float* __restrict__ result) {
  for (std::intptr_t batch = 0; batch < n_batch; ++batch) {
    const float batch_scaling_factor = scaling_factors[batch];
    // Compute dot-product for every column.
    for (std::intptr_t row = 0; row < m_rows; ++row) {
      // Get the address of the first element of the row.
      const int8_t* __restrict__ row_ptr = matrix + row * m_cols;

      // Initialize the dot product sum for the row to 0.
      __m128i dotprod_32x4 = _mm_setzero_si128();
      std::intptr_t col = 0;
      // For every block of 16x 8-bit inputs.
      while (col < (m_cols & ~15)) {
        const __m128i vec_8x16 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(vectors + col));
        const __m128i row_8x16 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_ptr + col));
        // dotprod += vec · row
        dotprod_32x4 =
            _mm_add_epi32(dotprod_32x4, DotProdInt8x4x4(vec_8x16, row_8x16));
        col += 16;
      }
#ifdef __SSE4_1__
      // Postamble for 8x 8-bit inputs.
      if (col < (m_cols & ~7)) {
        const __m128i vec_16x8 = _mm_cvtepi8_epi16(
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(vectors + col)));
        const __m128i row_16x8 = _mm_cvtepi8_epi16(
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row_ptr + col)));
        // dotprod += vec · row
        dotprod_32x4 =
            _mm_add_epi32(dotprod_32x4, _mm_madd_epi16(vec_16x8, row_16x8));
        col += 8;
      }
      // Postamble for 4x 8-bit inputs.
      if (col < (m_cols & ~3)) {
        const __m128i vec_32x4 = _mm_cvtepi8_epi32(
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(vectors + col)));
        const __m128i row_32x4 = _mm_cvtepi8_epi32(
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row_ptr + col)));
        // dotprod += vec · row
        dotprod_32x4 =
            _mm_add_epi32(dotprod_32x4, _mm_mullo_epi32(vec_32x4, row_32x4));
        col += 4;
      }
#endif

      // Horizontally add the 4 intermediate sum values to get the final
      // dot-prod value for this row.
      int32_t sum = ReduceInt32x4(dotprod_32x4);

#if defined(__SSE4_1__) && defined(__clang__)
      // SSE 4.1: Don't try to unroll and vectorize this, already done above.
#pragma clang loop unroll(disable) vectorize(disable)
#endif
      // Postamble loop for <4x (<16x without SSE 4.1) remaining 8-bit inputs.
      for (; col < m_cols; ++col) {
        sum += row_ptr[col] * vectors[col];
      }  // for col

      *result += sum * batch_scaling_factor;
      ++result;
    }  // for row

    vectors += m_cols;
  }  // for batch
}

void SseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors,
    const float* __restrict__ scaling_factors, int n_batch,
    float* __restrict__ result, const float* __restrict__ per_channel_scale,
    const int32_t* __restrict__ input_offset) {
  static constexpr std::intptr_t kBlockSize = 16;
  for (std::intptr_t batch = 0; batch < n_batch; ++batch) {
    const float batch_scaling_factor = scaling_factors[batch];
    for (std::intptr_t row = 0; row < m_rows; ++row) {
      const int8_t* __restrict__ row_ptr = matrix + row * m_cols;
      float scale = batch_scaling_factor;
      if (per_channel_scale != nullptr) {
        scale *= per_channel_scale[row];
      }
      __m128i dotprod_32x4 = _mm_setzero_si128();
      __m128i row_sum_16x8 = _mm_setzero_si128();
      std::intptr_t col = 0;
      for (; col < (m_cols & ~(kBlockSize - 1)); col += kBlockSize) {
        const __m128i vec_8x16 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(vectors + col));
        const __m128i row_8x16 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_ptr + col));
        // dotprod += vec · row
        dotprod_32x4 =
            _mm_add_epi32(dotprod_32x4, DotProdInt8x4x4(vec_8x16, row_8x16));

        // Pairwise add 16x 8-bit values; equivalently, multipy-add with 1.
        // Result is 8x 16-bit values.
        const __m128i row_16x8 = _mm_maddubs_epi16(_mm_set1_epi8(1), row_8x16);
        row_sum_16x8 = _mm_add_epi16(row_sum_16x8, row_16x8);
      }  // for col
      // Pairwise add 8x 16-bit values; equivalently, multipy-add with 1.
      // Result is 4x 32-bit values.
      const __m128i row_sum_32x4 =
          _mm_madd_epi16(row_sum_16x8, _mm_set1_epi16(1));
      int32_t sum = ReduceInt32x4(dotprod_32x4);
      int32_t row_sum = ReduceInt32x4(row_sum_32x4);
      // Postamble loop.
      for (; col < m_cols; ++col) {
        sum += row_ptr[col] * vectors[col];
        row_sum += row_ptr[col];
      }  // for col
      sum -= row_sum * input_offset[batch];
      *result += sum * scale;
      ++result;
    }  // for row
    vectors += m_cols;
  }  // for batch
}

namespace {

// Implements sparse-matrix - vector multiply-accumulate.
inline void SseSparseMatrixVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    const int m_rows, const int m_cols, const int8_t* __restrict__ vector,
    const float scaling_factor, float* __restrict__ result) {
  static const std::intptr_t kBlockSize = 16;
  TFLITE_DCHECK_EQ(m_cols % kBlockSize, 0);
  const uint8_t* __restrict__ ledger_ptr = ledger;
  for (std::intptr_t row = 0; row < m_rows; ++row) {
    // Initialize the dot product sum for the row to 0.
    __m128i dotprod_32x4 = _mm_setzero_si128();
    std::intptr_t num_nonzero_blocks = *ledger_ptr++;
    for (std::intptr_t i = 0; i < num_nonzero_blocks; i++) {
      const std::intptr_t col_index = *ledger_ptr++ * kBlockSize;
      const __m128i vec_8x16 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(vector + col_index));
      const __m128i row_8x16 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(matrix));
      // dotprod += vec · row
      dotprod_32x4 =
          _mm_add_epi32(dotprod_32x4, DotProdInt8x4x4(vec_8x16, row_8x16));
      matrix += kBlockSize;
    }  // for col
    // Horizontally add the 4 intermediate sum values to get the final
    // dot-prod value for this row.
    int32_t dotprod = ReduceInt32x4(dotprod_32x4);

    result[row] += dotprod * scaling_factor;
  }  // for row
}

// Implements sparse-matrix - batch-of-4-vectors multiply-accumulate.
// The stride between vectors and results must be equal to m_cols.
// Parameter 'batch' is the index of the first batch, must be a multiple of 4.
inline void SseSparseMatrix4VectorsMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    const int m_rows, const int m_cols,
    const int8_t* __restrict__ const vectors, const __m128 scaling_factors_fx4,
    float* __restrict__ const results) {
  static const std::intptr_t kBlockSize = 16;
  TFLITE_DCHECK_EQ(m_cols % kBlockSize, 0);

  const int8_t* __restrict__ vector0 = vectors + 0 * m_cols;
  const int8_t* __restrict__ vector1 = vectors + 1 * m_cols;
  const int8_t* __restrict__ vector2 = vectors + 2 * m_cols;
  const int8_t* __restrict__ vector3 = vectors + 3 * m_cols;
  float* __restrict__ result0 = results + 0 * m_rows;
  float* __restrict__ result1 = results + 1 * m_rows;
  float* __restrict__ result2 = results + 2 * m_rows;
  float* __restrict__ result3 = results + 3 * m_rows;

  for (std::intptr_t row = 0; row < m_rows; ++row) {
    // Initialize the dot product sum for the row to 0.
    __m128i dp0_32x4 = _mm_setzero_si128();
    __m128i dp1_32x4 = _mm_setzero_si128();
    __m128i dp2_32x4 = _mm_setzero_si128();
    __m128i dp3_32x4 = _mm_setzero_si128();

    std::intptr_t num_nonzero_blocks = *ledger++;
    for (std::intptr_t i = 0; i < num_nonzero_blocks; i++) {
      const std::intptr_t col_index = *ledger++ * kBlockSize;
      // vecN are for different batches
      const __m128i vec0_8x16 = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(vector0 + col_index));
      const __m128i vec1_8x16 = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(vector1 + col_index));
      const __m128i vec2_8x16 = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(vector2 + col_index));
      const __m128i vec3_8x16 = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(vector3 + col_index));
      const __m128i row_8x16 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(matrix));
      // dp += vec · row
      // dpN are for different batches
      dp0_32x4 = _mm_add_epi32(dp0_32x4, DotProdInt8x4x4(row_8x16, vec0_8x16));
      dp1_32x4 = _mm_add_epi32(dp1_32x4, DotProdInt8x4x4(row_8x16, vec1_8x16));
      dp2_32x4 = _mm_add_epi32(dp2_32x4, DotProdInt8x4x4(row_8x16, vec2_8x16));
      dp3_32x4 = _mm_add_epi32(dp3_32x4, DotProdInt8x4x4(row_8x16, vec3_8x16));
      matrix += kBlockSize;
    }  // for col

    // Horizontally add the 4 intermediate values.
    const __m128i dp_32x4 =
        ReduceInt32x4x4(dp0_32x4, dp1_32x4, dp2_32x4, dp3_32x4);
    // Convert to float
    const __m128 dp_fx4 = _mm_cvtepi32_ps(dp_32x4);
    // Load the results (This is an Accumulate function..)
    __m128 result_fx4 =
        _mm_set_ps(result3[row], result2[row], result1[row], result0[row]);
    // result += dp .* scaling
    result_fx4 =
        _mm_add_ps(result_fx4, _mm_mul_ps(dp_fx4, scaling_factors_fx4));
    // Save the results
    result0[row] = GetFloatVectorElement<0>(result_fx4);
    result1[row] = GetFloatVectorElement<1>(result_fx4);
    result2[row] = GetFloatVectorElement<2>(result_fx4);
    result3[row] = GetFloatVectorElement<3>(result_fx4);
  }  // for row
}

}  // namespace

void SseSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    const int m_rows, const int m_cols, const int8_t* __restrict__ vectors,
    const float* __restrict__ scaling_factors, int n_batch,
    float* __restrict__ results) {
  int batch = 0;
  const int kBatchSize4 = 4;
  const int n_batch_rounddown_to_batchsize_4 = n_batch & ~(kBatchSize4 - 1);
  while (batch < n_batch_rounddown_to_batchsize_4) {
    const __m128 scaling_factors_fx4 = _mm_loadu_ps(scaling_factors + batch);
    SseSparseMatrix4VectorsMultiplyAccumulate(
        matrix, ledger, m_rows, m_cols, vectors, scaling_factors_fx4, results);
    batch += kBatchSize4;
    vectors += kBatchSize4 * m_cols;
    results += kBatchSize4 * m_rows;
  }  // for batch
  while (batch < n_batch) {
    SseSparseMatrixVectorMultiplyAccumulate(matrix, ledger, m_rows, m_cols,
                                            vectors, scaling_factors[batch],
                                            results);
    ++batch;
    vectors += m_cols;
    results += m_rows;
  }  // for batch
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // __SSSE3__
