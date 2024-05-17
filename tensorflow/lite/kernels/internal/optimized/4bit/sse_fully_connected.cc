/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#if defined(FC_4BIT_SSE) && defined(__SSSE3__)

#include <stdint.h>
#include <stdlib.h>

// NOLINTBEGIN
#include <tmmintrin.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/optimized/4bit/fully_connected_common.h"
#include "tensorflow/lite/kernels/internal/optimized/4bit/sse_fully_connected_impl.h"

namespace tflite {
namespace optimized_4bit {
#define is_aligned(ptr, bytes) ((((size_t)(ptr)) & (bytes - 1)) == 0)

void SsePackInner(const int8_t* src, uint8_t* box, int src_rows, int src_cols,
                  int outer_row, int outer_col, int outer_rows, int outer_cols,
                  int inner_rows, int inner_cols) {
  const int width = inner_rows;
  const int depth = inner_cols;
  const int real_depth = depth / 2;
  const int real_src_cols = src_cols / 2;
  const int row = outer_row * inner_rows;
  const int col = outer_col * inner_cols;
  int src_width = std::min(width, src_rows - row);
  int src_depth = std::min(depth, src_cols - col);
  int real_col = col / 2;
  const int8_t* src_data = src + row * real_src_cols + real_col;
  int real_src_depth = src_depth / 2;
  const __m128i bitmask_upper = _mm_set1_epi16(255U << 8);
  const __m128i bitmask_lower = _mm_set1_epi16(255U);
  const __m128i seven = _mm_set1_epi8(7);
  for (int m = 0; m < src_width; ++m) {
    int i = 0;
    int k = 0;
    for (; i < (real_src_depth & (~15)); i += 16) {
      const __m128i values_128i = _mm_loadu_si128((__m128i*)(src_data + i));
      // sign extend uv1
      __m128i uv1 = _mm_srai_epi16(values_128i, 4);
      uv1 = _mm_add_epi8(uv1, seven);
      uv1 = _mm_and_si128(uv1, bitmask_upper);
      __m128i uv2 = _mm_slli_epi16(values_128i, 8);
      uv2 = _mm_srai_epi16(uv2, 12);
      uv2 = _mm_add_epi8(uv2, seven);
      uv2 = _mm_and_si128(uv2, bitmask_lower);
      uv1 = _mm_or_si128(uv1, uv2);

      __m128i lv1 = _mm_slli_epi16(values_128i, 4);
      lv1 = _mm_srai_epi16(lv1, 4);
      lv1 = _mm_add_epi8(lv1, seven);
      lv1 = _mm_and_si128(lv1, bitmask_upper);
      __m128i lv2 = _mm_slli_epi16(values_128i, 12);
      lv2 = _mm_srai_epi16(lv2, 12);
      lv2 = _mm_add_epi8(lv2, seven);
      lv2 = _mm_and_si128(lv2, bitmask_lower);

      lv1 = _mm_or_si128(lv1, lv2);
      __m128i u = _mm_or_si128(_mm_slli_epi16(uv1, 4),
                               _mm_unpackhi_epi64(uv1, _mm_setzero_si128()));
      __m128i l = _mm_or_si128(_mm_slli_epi16(lv1, 4),
                               _mm_unpackhi_epi64(lv1, _mm_setzero_si128()));
      __m128i v = _mm_unpacklo_epi8(l, u);
      _mm_store_si128((__m128i*)(box + k), v);
      k += 16;
    }
    // Handle remaining values -- if greater than or equal to
    // 16 values remaining, do the shuffle.
    if (i < real_src_depth) {
      int remaining = 8;
      remaining =
          remaining < (real_src_depth - i) ? remaining : real_src_depth - i;
      for (int j = 0; j < remaining; j++) {
        const int8_t v1 = (int8_t)src_data[i + j];
        int8_t uv1 = upper(v1);
        int8_t lv1 = lower(v1);
        int8_t uv2 = 0;
        int8_t lv2 = 0;
        if ((i + j + 8) < real_src_depth) {
          const int8_t v2 = (int8_t)src_data[i + j + 8];
          uv2 = upper(v2);
          lv2 = lower(v2);
        }
        box[k] = merge(lv1, lv2);
        box[k + 1] = merge(uv1, uv2);
        k += 2;
      }
    }
    box += real_depth;
    src_data += real_src_cols;
  }
}

void SsePrepack(uint8_t* dest, const int8_t* tensor, int layout_rows,
                int layout_cols, int src_rows, int src_cols, int width,
                int depth) {
  size_t size = layout_rows * layout_cols / 2;
  memset(dest, static_cast<uint8_t>(119), sizeof(uint8_t) * size);
  int outer_cols = layout_cols / depth;
  int outer_rows = layout_rows / width;
  int inner_cols = depth;
  int inner_rows = width;
  for (int outer_row = 0; outer_row < outer_rows; ++outer_row) {
    for (int outer_col = 0; outer_col < outer_cols; ++outer_col) {
      const int cluster_index = outer_row * outer_cols + outer_col;
      const int real_depth = inner_cols / 2;
      uint8_t* box = dest + cluster_index * real_depth * inner_rows;
      SsePackInner(tensor, box, src_rows, src_cols, outer_row, outer_col,
                   outer_rows, outer_cols, inner_rows, inner_cols);
    }
  }
}

void SseBatchQuantizeFloats4Bit(const float* float_data_ptr, int n_batch,
                                int n_data, int8_t* quantized_data_ptr,
                                float* scaling_factors, int width, int depth,
                                int32_t* input_offsets) {
  const int rows = n_batch;
  const int cols = n_data;
  // depth is always cols
  const int layout_rows = (rows + (width - 1)) & ~(width - 1);
  const int layout_cols = (cols + (depth - 1)) & ~(depth - 1);
  const int size = layout_rows * layout_cols;
  int8_t* data = quantized_data_ptr;
  memset(data, 0, sizeof(int8_t) * size);
  memset(input_offsets, 0, sizeof(int32_t) * layout_rows);
  const float* tensor_data = float_data_ptr;
  // basically, we need to make a new 4D matrix
  // [rows / width, cols / depth, width, depth] in depth-first
  const int outer_cols = layout_cols / depth;
  const int outer_rows = layout_rows / width;
  float* scaling_factors_ptr = scaling_factors;
  for (int outer_row = 0; outer_row < outer_rows; outer_row++) {
    std::vector<float> scale(width);
    const int row = width * outer_row;
    scaling_factors_ptr = scaling_factors + row;
    for (int w = 0; w < width; ++w) {
      if ((row + w) >= rows) {
        continue;
      }
      const float* start = tensor_data + (row + w) * cols;
      int c = 0;
      float scale_denom = 0;
      for (; c < cols; ++c) {
        scale_denom = std::max(scale_denom, std::abs(*(start++)));
      }
      if (scale_denom == 0) {
        scale_denom = 127.0;
      }
      scale[w] = 127.0 / scale_denom;
      scaling_factors_ptr[w] = scale_denom / 127.0;
    }
    for (int outer_col = 0; outer_col < outer_cols; ++outer_col) {
      const int col = depth * outer_col;
      const int src_width = std::min(width, rows - row);
      const int src_depth = std::min(depth, cols - col);
      const int cluster_index = outer_row * outer_cols + outer_col;
      int8_t* box = data + cluster_index * depth * width;
      for (int w = 0; w < src_width; ++w) {
        const float* float_data = tensor_data + (row + w) * cols + col;
        int d = 0;
        for (; d < src_depth; ++d) {
          int8_t q = static_cast<int8_t>(TfLiteRound(float_data[d] * scale[w]));
          box[w * depth + d] = q;
          input_offsets[row + w] += q;
        }
      }
    }
  }
  for (int r = 0; r < layout_rows; ++r) {
    input_offsets[r] = input_offsets[r] * zero_point_4bit;
  }
}

void SseAssignBiasAndComputeOffsets(const int32_t* input_offsets,
                                    const float* batch_scales,
                                    const float* filter_scales,
                                    const float* bias_ptr, float* output_ptr,
                                    int output_depth, int batch_size) {
  if (bias_ptr) {
    for (int b = 0; b < batch_size; ++b) {
      const float val = *input_offsets++ * *batch_scales++;
      const float* filter_scales_ptr = filter_scales;
      const float* bias_ptr_tmp = bias_ptr;
      int i = 0;
      for (; i < output_depth; i++) {
        *output_ptr++ = (val * *filter_scales_ptr++) + *bias_ptr_tmp++;
      }
    }
    return;
  }
  for (int b = 0; b < batch_size; ++b) {
    const float val = *input_offsets++ * *batch_scales++;
    const float* filter_scales_ptr = filter_scales;
    int i = 0;
    for (; i < output_depth; i++) {
      *output_ptr++ = (val * *filter_scales_ptr++);
    }
  }
}

template <int Depth, int Width>
void SseUnpack(float* output_ptr, const int32_t* dst, int batch_size,
               int num_units, const float* scaling_factors,
               const float* filter_scales, int dst_layout_rows,
               int dst_layout_cols) {
  if (Width == 1) {
    const int outer_rows = dst_layout_rows / Width;
    const int outer_cols = dst_layout_cols / Depth;
    const int32_t* dst_ptr = dst;
    int unit = 0;
    for (int outer_col = 0; outer_col < outer_cols;
         ++outer_col, unit += Depth) {
      float* tmp_output_ptr = output_ptr + unit;
      int len = num_units - unit < Depth ? num_units - unit : Depth;
      int cond = len & ~3;
      const float* scaling_factors_ptr = scaling_factors;
      for (int outer_row = 0; outer_row < outer_rows; ++outer_row) {
        const float scale = *scaling_factors_ptr;
        const float* filter_scales_ptr = filter_scales + unit;
        int i = 0;
        for (; i < cond; i += 4) {
          *(tmp_output_ptr++) += *(dst_ptr++) * scale * (*filter_scales_ptr++);
          *(tmp_output_ptr++) += *(dst_ptr++) * scale * (*filter_scales_ptr++);
          *(tmp_output_ptr++) += *(dst_ptr++) * scale * (*filter_scales_ptr++);
          *(tmp_output_ptr++) += *(dst_ptr++) * scale * (*filter_scales_ptr++);
        }
        for (; i < len; ++i) {
          *(tmp_output_ptr++) += *(dst_ptr++) * scale * (*filter_scales_ptr++);
        }
        dst_ptr += (Depth - len);
        scaling_factors_ptr += Width;
        tmp_output_ptr += (num_units - len);
      }
    }
    return;
  }
  const int outer_rows = dst_layout_rows / Width;
  const int outer_cols = dst_layout_cols / Depth;
  for (int outer_col = 0; outer_col < outer_cols; ++outer_col) {
    const int unit = outer_col * Depth;
    const int remaining_units = std::min(num_units - unit, Depth);
    const int depth_offset = Depth - remaining_units;
    const int width_offset = num_units - remaining_units;
    int outer_row = 0;
    for (; outer_row < outer_rows; ++outer_row) {
      const int batch = outer_row * Width;
      const int remaining_width = std::min(batch_size - batch, Width);
      const int cluster_index = outer_col * outer_rows + outer_row;
      const int32_t* dst_ptr = dst + cluster_index * Depth * Width;
      float* tmp_output_ptr = output_ptr + batch * num_units + unit;
      const float* scale = scaling_factors + batch;
      int w = remaining_width;
      for (; w > 0; --w, scale++) {
        int d = remaining_units;
        const float* filter_scales_ptr = filter_scales + unit;
        for (; d > 0; --d) {
          *tmp_output_ptr++ += *dst_ptr++ * (*scale) * (*filter_scales_ptr++);
        }
        dst_ptr += depth_offset;
        tmp_output_ptr += width_offset;
      }
    }
  }
}

inline __m128i DotProdInt8x4x4(__m128i acc_32x4, __m128i a_8x16,
                               __m128i b_8x16) {
  b_8x16 = _mm_sign_epi8(b_8x16, a_8x16);
  a_8x16 = _mm_abs_epi8(a_8x16);
  __m128i sumprod_16x8 = _mm_maddubs_epi16(a_8x16, b_8x16);
  return _mm_add_epi32(acc_32x4,
                       _mm_madd_epi16(sumprod_16x8, _mm_set1_epi16(1)));
}

inline __m128i ReduceInt32x4x4(__m128i a, __m128i b, __m128i c, __m128i d) {
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

template <int RowsLeft, int RowsRight, int Cols>
void SseRunKernel(const uint8_t* lhs, const int8_t* rhs, int32_t* dst,
                  int lhs_layout_rows, int lhs_layout_cols, int rhs_layout_rows,
                  int rhs_layout_cols, int dst_layout_rows,
                  int dst_layout_cols) {
  const int start_row = 0;
  const int start_col = 0;
  const int end_row = lhs_layout_rows;
  const int end_col = rhs_layout_rows;
  const int clamped_end_row = std::min(end_row, dst_layout_cols);
  const int clamped_end_col = std::min(end_col, dst_layout_rows);
  int32_t* elementPtr = dst;
  const int outer_rows = (clamped_end_row + RowsLeft - 1) / RowsLeft;
  const int outer_cols = (clamped_end_col + RowsRight - 1) / RowsRight;
  const int depth = std::min(lhs_layout_cols / Cols, rhs_layout_cols / Cols);
  const __m128i bitmask = _mm_set1_epi8(15);
  const uintptr_t padding = 15;
  std::vector<uint8_t> lhs_vec_data((RowsLeft * lhs_layout_cols / 2) + padding);
  uint8_t* lhs_vec = lhs_vec_data.data();
  for (int i = start_row; i < outer_rows; ++i) {
    int left_index = i * RowsLeft * lhs_layout_cols / 2;
    const uint8_t* lhs_val_data = lhs + left_index;
    if (!is_aligned(lhs_val_data, 16)) {
      size_t size = RowsLeft * lhs_layout_cols / 2;
      uintptr_t aligned =
          (reinterpret_cast<uintptr_t>(lhs_vec) + padding) & ~(padding);
      lhs_vec = reinterpret_cast<uint8_t*>(aligned);
      memcpy(lhs_vec, lhs_val_data, size);
      lhs_val_data = lhs_vec;
    }
    for (int j = start_col; j < outer_cols; ++j) {
      const uint8_t* lhs_val = lhs_val_data;
      int right_index = j * RowsRight * rhs_layout_cols;
      const int8_t* rhs_val = rhs + right_index;
      __m128i accum[RowsRight * RowsLeft];
      for (int m = 0; m < (RowsLeft * RowsRight); ++m) {
        accum[m] = _mm_set1_epi8(0);
      }
      for (int k = 0; k < depth; ++k) {
        __m128i lhs_row[RowsLeft];
        for (int m = 0; m < RowsLeft; ++m) {
          lhs_row[m] = _mm_load_si128((__m128i*)(lhs_val));
          lhs_val += 16;
        }
        __m128i rhs[RowsRight][2];
        for (int m = 0; m < RowsRight; ++m) {
          for (int n = 0; n < 2; ++n) {
            rhs[m][n] = _mm_loadu_si128((__m128i*)(rhs_val));
            rhs_val += 16;
          }
        }
        __m128i lhs_row_8[RowsLeft][2];
        for (int m = 0; m < RowsLeft; ++m) {
          lhs_row_8[m][0] = _mm_srli_epi16(lhs_row[m], 4);
          lhs_row_8[m][1] = _mm_and_si128(lhs_row[m], bitmask);
        }
        for (int m = 0; m < RowsLeft; ++m) {
          lhs_row_8[m][0] = _mm_and_si128(lhs_row_8[m][0], bitmask);
        }
        for (int i = 0; i < 2; ++i) {
          for (int r = 0; r < RowsRight; ++r) {
            for (int l = 0; l < RowsLeft; ++l) {
              accum[r * RowsLeft + l] = DotProdInt8x4x4(
                  accum[r * RowsLeft + l], lhs_row_8[l][i], rhs[r][i]);
            }
          }
        }
      }
      for (int r = 0; r < RowsRight; ++r) {
        __m128i sum =
            ReduceInt32x4x4(accum[r * RowsLeft], accum[r * RowsLeft + 1],
                            accum[r * RowsLeft + 2], accum[r * RowsLeft + 3]);
        _mm_storeu_si128((__m128i*)elementPtr, sum);
        elementPtr += 4;
      }
    }
  }
}
// NOLINTEND

template void SseUnpack<4, 1>(float* output_ptr, const int32_t* dst,
                              int batch_size, int num_units,
                              const float* scaling_factors,
                              const float* filter_scales, int dst_layout_rows,
                              int dst_layout_cols);

template void SseUnpack<4, 2>(float* output_ptr, const int32_t* dst,
                              int batch_size, int num_units,
                              const float* scaling_factors,
                              const float* filter_scales, int dst_layout_rows,
                              int dst_layout_cols);

template void SseUnpack<4, 4>(float* output_ptr, const int32_t* dst,
                              int batch_size, int num_units,
                              const float* scaling_factors,
                              const float* filter_scales, int dst_layout_rows,
                              int dst_layout_cols);

template void SseRunKernel<4, 1, 32>(const uint8_t* lhs, const int8_t* rhs,
                                     int32_t* dst, int lhs_layout_rows,
                                     int lhs_layout_cols, int rhs_layout_rows,
                                     int rhs_layout_cols, int dst_layout_rows,
                                     int dst_layout_cols);

template void SseRunKernel<4, 2, 32>(const uint8_t* lhs, const int8_t* rhs,
                                     int32_t* dst, int lhs_layout_rows,
                                     int lhs_layout_cols, int rhs_layout_rows,
                                     int rhs_layout_cols, int dst_layout_rows,
                                     int dst_layout_cols);

template void SseRunKernel<4, 4, 32>(const uint8_t* lhs, const int8_t* rhs,
                                     int32_t* dst, int lhs_layout_rows,
                                     int lhs_layout_cols, int rhs_layout_rows,
                                     int rhs_layout_cols, int dst_layout_rows,
                                     int dst_layout_cols);

}  // namespace optimized_4bit
}  // namespace tflite

#endif  // defined(FC_4BIT_SSE) && defined(__SSSE3__)
