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

#ifdef __SSE4_1__

#include <emmintrin.h>  // SSE2
#include <smmintrin.h>  // SSE4.1
#include <tmmintrin.h>  // SSSE3

#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {
namespace tensor_utils {
namespace {

// Elementwise multiply two i8x8 vectors to i16x8, add elements pairwise and
// accumulate result to a i32x4 accumulator.
//
// Shared by the inner loop of MatrixBatchVectorMultiplyAccumulate(int8) and
// SparseMatrixBatchVectorMultiplyAccumulate(int8).
//
// x86 SSE has no i8*i8 instruction (only a u8*i8), so we need to do sign
// extension to 16 bit and do i16*i16 multiplications. There is an instruction
// to sign-extend i8x8 => i16x8 from the lower half of the register (used here),
// but there is no direct way to sign-extend the high half, only multiple
// instructions (see _mm_cmpgt_epi8 and _mm_unpackhi_epi8). Bottom line is, it
// is actually cheaper to only to process 8 elements = 64b at a time.
static inline __m128i MatrixBatchVectorMultiplyAccumulateLoopBodySse(
    __m128i dotprod, __m128i a_8x8, __m128i b_8x8) {
  // Sign extend i8 => i16
  __m128i a_16x8 = _mm_cvtepi8_epi16(a_8x8);  // SSE4.1
  __m128i b_16x8 = _mm_cvtepi8_epi16(b_8x8);  // SSE4.1
  // sumprod[i] = a[2*i]*b[2*i] + a[2*i+1]*b[2*i+1] (i = 0..3)
  __m128i sumprod_32x4 = _mm_madd_epi16(a_16x8, b_16x8);  // SSE2
  // i32x4 + i32x4
  return _mm_add_epi32(dotprod, sumprod_32x4);  // SSE2
}

// Horizontally add 4 int32 values stored in a single XMM register to int32_t.
static inline int32_t ReduceInt32x4(__m128i acc) {
  acc = _mm_hadd_epi32(acc, acc);  // SSSE3
  // This second hadd could be only 64 bit, but 64 and 128 bit hadd has same
  // latency on most CPUs, and it costs more to move. (Moving can be no-op, but
  // nevertheless is an extra instruction occupying the decoder and I cache.)
  acc = _mm_hadd_epi32(acc, acc);  // SSSE3
  // SSE4.1 instrinsic, but actually translated to SSE2 instruction (due to
  // moving from 0th element).
  return _mm_extract_epi32(acc, 0);
}

}  // namespace

void SseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, int result_stride) {
  static constexpr int kBlockSize = 8;
  for (int batch = 0; batch < n_batch; ++batch) {
    const float batch_scaling_factor = scaling_factors[batch];
    // Compute dot-product for every column.
    for (int row = 0; row < m_rows; ++row, result += result_stride) {
      // Get the address of the first element of the row.
      const int8_t* row_ptr = matrix + row * m_cols;

      // Initialize the dot product sum for the row to 0.
      __m128i dotprod_32x4 = _mm_setzero_si128();  // SSE2
      // For every block of kBlockSize 8-bit elements.
      int col = 0;
      for (; col < (m_cols & ~(kBlockSize - 1)); col += kBlockSize) {
        // See comment at MatrixBatchVectorMultiplyAccumulateLoopBodySse why to
        // load only 64 bits. _mm_loadl_epi64 requires SSE2.
        const __m128i vec_8x8 =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(vectors + col));
        const __m128i row_8x8 =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row_ptr + col));
        dotprod_32x4 = MatrixBatchVectorMultiplyAccumulateLoopBodySse(
            dotprod_32x4, vec_8x8, row_8x8);
      }  // for col
      // Horizontally add the 4 intermediate sum values to get the final
      // dot-prod value for this row.
      int32_t sum = ReduceInt32x4(dotprod_32x4);

      // Postamble loop.
      for (; col < m_cols; ++col) {
        sum += row_ptr[col] * vectors[col];
      }  // for col

      *result += sum * batch_scaling_factor;
    }  // for row

    vectors += m_cols;
  }  // for batch
}

void SseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, int result_stride,
    const float* per_channel_scale, const int32_t* input_offset) {
  static constexpr int kBlockSize = 8;
  for (int batch = 0; batch < n_batch; ++batch) {
    const float batch_scaling_factor = scaling_factors[batch];
    for (int row = 0; row < m_rows; ++row, result += result_stride) {
      const int8_t* row_ptr = matrix + row * m_cols;
      __m128i dotprod_32x4 = _mm_setzero_si128();  // SSE2
      __m128i row_sum_16x8 = _mm_setzero_si128();
      int col = 0;
      for (; col < (m_cols & ~(kBlockSize - 1)); col += kBlockSize) {
        const __m128i vec_8x8 =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(vectors + col));
        const __m128i row_8x8 =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row_ptr + col));
        dotprod_32x4 = MatrixBatchVectorMultiplyAccumulateLoopBodySse(
            dotprod_32x4, vec_8x8, row_8x8);
        __m128i row_16x8 = _mm_cvtepi8_epi16(row_8x8);
        row_sum_16x8 = _mm_add_epi16(row_sum_16x8, row_16x8);
      }  // for col
      row_sum_16x8 = _mm_hadd_epi16(row_sum_16x8, row_sum_16x8);
      __m128i row_sum_32x4 = _mm_cvtepi16_epi32(row_sum_16x8);
      int32_t sum = ReduceInt32x4(dotprod_32x4);
      int32_t row_sum = ReduceInt32x4(row_sum_32x4);
      // Postamble loop.
      for (; col < m_cols; ++col) {
        sum += row_ptr[col] * vectors[col];
        row_sum += row_ptr[col];
      }  // for col
      sum -= row_sum * input_offset[batch];
      *result += sum * batch_scaling_factor * per_channel_scale[row];
    }  // for row
    vectors += m_cols;
  }  // for batch
}

void SseSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result,
    int result_stride) {
  static const int kBlockSize = 16;
  TFLITE_DCHECK_EQ(m_cols % kBlockSize, 0);

  for (int batch = 0; batch < n_batch; ++batch, vectors += m_cols) {
    const float batch_scaling_factor = scaling_factors[batch];
    const uint8_t* ledger_ptr = ledger;
    const int8_t* row_ptr = matrix;
    for (int row = 0; row < m_rows; ++row, result += result_stride) {
      // Initialize the dot product sum for the row to 0.
      __m128i dotprod_32x4 = _mm_setzero_si128();
      int num_nonzero_blocks = *ledger_ptr++;
      for (int i = 0; i < num_nonzero_blocks; i++) {
        const int col_index = *ledger_ptr++ * kBlockSize;
        // With sparse models, we assume the block size is 16, we can't change
        // it to 8 here to better fit SSE (see dense version). Instead, do the
        // int8x8_t computation twice.
        __m128i vec_8x8 = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(vectors + col_index));
        __m128i row_8x8 =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row_ptr));
        dotprod_32x4 = MatrixBatchVectorMultiplyAccumulateLoopBodySse(
            dotprod_32x4, vec_8x8, row_8x8);
        vec_8x8 = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(vectors + col_index + 8));
        row_8x8 =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row_ptr + 8));
        dotprod_32x4 = MatrixBatchVectorMultiplyAccumulateLoopBodySse(
            dotprod_32x4, vec_8x8, row_8x8);
        row_ptr += kBlockSize;
      }
      // Horizontally add the 4 intermediate sum values to get the final
      // dot-prod value for this row.
      int32_t dotprod = ReduceInt32x4(dotprod_32x4);

      *result += dotprod * batch_scaling_factor;
    }  // for row
  }    // for batch
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // __SSE4_1__
