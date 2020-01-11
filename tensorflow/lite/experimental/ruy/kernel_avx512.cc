/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include <algorithm>
#include <cstdint>

#include "profiling/instrumentation.h"
#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/kernel.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/platform.h"

#if RUY_PLATFORM(AVX512) && RUY_OPT_ENABLED(RUY_OPT_ASM)
#include <immintrin.h>  // IWYU pragma: keep
#endif

namespace ruy {

#if !(RUY_PLATFORM(AVX512) && RUY_OPT_ENABLED(RUY_OPT_ASM))

void Kernel8bitAvx512(const KernelParams8bit<16, 16>& params) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

void Kernel8bitAvx512SingleCol(const KernelParams8bit<16, 16>& params) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

void KernelFloatAvx512(const KernelParamsFloat<16, 16>& params) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

void KernelFloatAvx512SingleCol(const KernelParamsFloat<16, 16>& params) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

#else  // RUY_PLATFORM(AVX512) && RUY_OPT_ENABLED(RUY_OPT_ASM)

void Kernel8bitAvx512(const KernelParams8bit<16, 16>& params) {
  gemmlowp::ScopedProfilingLabel label("Kernel kAvx512 8-bit");

  std::int32_t dst_stride;
  if ((params.dst_type_id == DstTypeId<std::int8_t>::kValue) ||
      (params.dst_type_id == DstTypeId<std::uint8_t>::kValue)) {
    dst_stride = params.dst_stride;
  } else if (params.dst_type_id == DstTypeId<std::int16_t>::kValue) {
    dst_stride = params.dst_stride / sizeof(std::int16_t);
  } else if (params.dst_type_id == DstTypeId<std::int32_t>::kValue) {
    dst_stride = params.dst_stride / sizeof(std::int32_t);
  } else {
    RUY_DCHECK(false);
  }

  int bias_ptr_block_increment = params.flags & RUY_ASM_FLAG_HAS_BIAS ? 16 : 0;

  const std::int8_t* rhs_col_ptr = params.rhs_base_ptr;
  void* dst_col_ptr = params.dst_base_ptr;
  const std::int32_t* bias_col_ptr = params.bias;
  if (params.flags & RUY_ASM_FLAG_HAS_BIAS) {
    bias_col_ptr += params.start_row;
  }

  for (int col = params.start_col; col <= params.last_col; col += 16) {
    const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
    void* dst_ptr = dst_col_ptr;
    const std::int32_t* bias_ptr = bias_col_ptr;

    const std::int32_t lhs_zero_point = params.lhs_zero_point;
    const bool has_rhs_sums_offsets =
        (params.flags & RUY_ASM_FLAG_HAS_RHS_SUMS) && lhs_zero_point;
    std::int32_t rhs_sums_offsets[16];
    if (has_rhs_sums_offsets) {
      const __m512i rhs_sums_offset_v =
          _mm512_mullo_epi32(_mm512_set1_epi32(lhs_zero_point),
                             _mm512_loadu_epi32(&params.rhs_sums[col]));
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(rhs_sums_offsets),
                          rhs_sums_offset_v);
    }

    for (int row = params.start_row; row <= params.last_row; row += 16) {
      const int residual_rows = std::min(params.dst_rows - row, 16);
      const int residual_cols = std::min(params.dst_cols - col, 16);

      __m512i accum_data_v0;
      __m512i accum_data_v1;
      __m512i accum_data_v2;
      __m512i accum_data_v3;
      __m512i accum_data_v4;
      __m512i accum_data_v5;
      __m512i accum_data_v6;
      __m512i accum_data_v7;
      __m512i accum_data_v8;
      __m512i accum_data_v9;
      __m512i accum_data_va;
      __m512i accum_data_vb;
      __m512i accum_data_vc;
      __m512i accum_data_vd;
      __m512i accum_data_ve;
      __m512i accum_data_vf;

      // Initialize with bias.
      const __mmask16 row_mask =
          (static_cast<std::uint32_t>(1) << residual_rows) - 1;
      __m512i initial_accum_data = _mm512_maskz_loadu_epi32(row_mask, bias_ptr);
      bias_ptr += bias_ptr_block_increment;

      const std::int32_t rhs_zero_point = params.rhs_zero_point;
      if ((params.flags & RUY_ASM_FLAG_HAS_LHS_SUMS) && rhs_zero_point) {
        const __m512i lhs_sums_offset =
            _mm512_mullo_epi32(_mm512_set1_epi32(rhs_zero_point),
                               _mm512_loadu_epi32(&params.lhs_sums[row]));
        initial_accum_data =
            _mm512_sub_epi32(initial_accum_data, lhs_sums_offset);
      }

      const std::int32_t prod_zp_depth = params.prod_zp_depth;
      if (prod_zp_depth != 0) {
        initial_accum_data = _mm512_add_epi32(initial_accum_data,
                                              _mm512_set1_epi32(prod_zp_depth));
      }

      // Adjustments differing across columns.
      if (has_rhs_sums_offsets) {
        accum_data_v0 = _mm512_sub_epi32(
            initial_accum_data, _mm512_set1_epi32(rhs_sums_offsets[0]));
        accum_data_v1 = _mm512_sub_epi32(
            initial_accum_data, _mm512_set1_epi32(rhs_sums_offsets[1]));
        accum_data_v2 = _mm512_sub_epi32(
            initial_accum_data, _mm512_set1_epi32(rhs_sums_offsets[2]));
        accum_data_v3 = _mm512_sub_epi32(
            initial_accum_data, _mm512_set1_epi32(rhs_sums_offsets[3]));
        accum_data_v4 = _mm512_sub_epi32(
            initial_accum_data, _mm512_set1_epi32(rhs_sums_offsets[4]));
        accum_data_v5 = _mm512_sub_epi32(
            initial_accum_data, _mm512_set1_epi32(rhs_sums_offsets[5]));
        accum_data_v6 = _mm512_sub_epi32(
            initial_accum_data, _mm512_set1_epi32(rhs_sums_offsets[6]));
        accum_data_v7 = _mm512_sub_epi32(
            initial_accum_data, _mm512_set1_epi32(rhs_sums_offsets[7]));
        accum_data_v8 = _mm512_sub_epi32(
            initial_accum_data, _mm512_set1_epi32(rhs_sums_offsets[8]));
        accum_data_v9 = _mm512_sub_epi32(
            initial_accum_data, _mm512_set1_epi32(rhs_sums_offsets[9]));
        accum_data_va = _mm512_sub_epi32(
            initial_accum_data, _mm512_set1_epi32(rhs_sums_offsets[10]));
        accum_data_vb = _mm512_sub_epi32(
            initial_accum_data, _mm512_set1_epi32(rhs_sums_offsets[11]));
        accum_data_vc = _mm512_sub_epi32(
            initial_accum_data, _mm512_set1_epi32(rhs_sums_offsets[12]));
        accum_data_vd = _mm512_sub_epi32(
            initial_accum_data, _mm512_set1_epi32(rhs_sums_offsets[13]));
        accum_data_ve = _mm512_sub_epi32(
            initial_accum_data, _mm512_set1_epi32(rhs_sums_offsets[14]));
        accum_data_vf = _mm512_sub_epi32(
            initial_accum_data, _mm512_set1_epi32(rhs_sums_offsets[15]));
      } else {
        accum_data_v0 = initial_accum_data;
        accum_data_v1 = initial_accum_data;
        accum_data_v2 = initial_accum_data;
        accum_data_v3 = initial_accum_data;
        accum_data_v4 = initial_accum_data;
        accum_data_v5 = initial_accum_data;
        accum_data_v6 = initial_accum_data;
        accum_data_v7 = initial_accum_data;
        accum_data_v8 = initial_accum_data;
        accum_data_v9 = initial_accum_data;
        accum_data_va = initial_accum_data;
        accum_data_vb = initial_accum_data;
        accum_data_vc = initial_accum_data;
        accum_data_vd = initial_accum_data;
        accum_data_ve = initial_accum_data;
        accum_data_vf = initial_accum_data;
      }

      const std::int8_t* lhs_ptr = lhs_col_ptr;
      const std::int8_t* rhs_ptr = rhs_col_ptr;
      for (int d = 0; d < params.depth; d += 4) {
        const __m512i lhs_data = _mm512_loadu_epi8(lhs_ptr);
        __m512i rhs_data_8bit = _mm512_loadu_epi8(rhs_ptr);

        // Each "int32" is two 16-bit RHS values, sign extended from 8-bit.
        std::int32_t rhs_data[32];
        const __m256i rhs_data_bottom_lane =
            _mm512_castsi512_si256(rhs_data_8bit);
        const __m256i rhs_data_top_lane =
            _mm512_extracti32x8_epi32(rhs_data_8bit, 1);
        const __m512i rhs_16_bit_dup_low =
            _mm512_cvtepi8_epi16(rhs_data_bottom_lane);
        const __m512i rhs_16_bit_dup_high =
            _mm512_cvtepi8_epi16(rhs_data_top_lane);
        // Now that we have cast the RHS data, we store it so that each value
        // can be separately loaded in the accumulation loop.
        _mm512_storeu_si512(reinterpret_cast<__m256i*>(rhs_data),
                            rhs_16_bit_dup_low);
        _mm512_storeu_si512(reinterpret_cast<__m256i*>(rhs_data + 16),
                            rhs_16_bit_dup_high);

        // Take bytes 0, 1, 4, 5, 8, 9, ... and expand to 16-bit.
        const __m512i lhs_16_bit_low =
            _mm512_cvtepi8_epi16(_mm512_cvtepi32_epi16(lhs_data));
        // Take bytes 2, 3, 6, 7, 10, 11, ... and expand to 16-bit.
        const __m512i lhs_16_bit_high = _mm512_cvtepi8_epi16(
            _mm512_cvtepi32_epi16(_mm512_srli_epi32(lhs_data, 16)));

        // Process column 0.
        {
          __m512i accum_v = accum_data_v0;
          constexpr int index = 0;

          const __m512i rhs_16_bit_dup_low = _mm512_set1_epi32(rhs_data[index]);
          const __m512i rhs_16_bit_dup_high =
              _mm512_set1_epi32(rhs_data[index + 1]);

          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
          accum_data_v0 = accum_v;
        }
        // Process column 1.
        {
          __m512i accum_v = accum_data_v1;
          constexpr int index = 2;

          const __m512i rhs_16_bit_dup_low = _mm512_set1_epi32(rhs_data[index]);
          const __m512i rhs_16_bit_dup_high =
              _mm512_set1_epi32(rhs_data[index + 1]);

          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
          accum_data_v1 = accum_v;
        }
        // Process column 2.
        {
          __m512i accum_v = accum_data_v2;
          constexpr int index = 4;

          const __m512i rhs_16_bit_dup_low = _mm512_set1_epi32(rhs_data[index]);
          const __m512i rhs_16_bit_dup_high =
              _mm512_set1_epi32(rhs_data[index + 1]);

          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
          accum_data_v2 = accum_v;
        }
        // Process column 3.
        {
          __m512i accum_v = accum_data_v3;
          constexpr int index = 6;

          const __m512i rhs_16_bit_dup_low = _mm512_set1_epi32(rhs_data[index]);
          const __m512i rhs_16_bit_dup_high =
              _mm512_set1_epi32(rhs_data[index + 1]);

          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
          accum_data_v3 = accum_v;
        }
        // Process column 4.
        {
          __m512i accum_v = accum_data_v4;
          constexpr int index = 8;

          const __m512i rhs_16_bit_dup_low = _mm512_set1_epi32(rhs_data[index]);
          const __m512i rhs_16_bit_dup_high =
              _mm512_set1_epi32(rhs_data[index + 1]);

          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
          accum_data_v4 = accum_v;
        }
        // Process column 5.
        {
          __m512i accum_v = accum_data_v5;
          constexpr int index = 10;

          const __m512i rhs_16_bit_dup_low = _mm512_set1_epi32(rhs_data[index]);
          const __m512i rhs_16_bit_dup_high =
              _mm512_set1_epi32(rhs_data[index + 1]);

          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
          accum_data_v5 = accum_v;
        }
        // Process column 6.
        {
          __m512i accum_v = accum_data_v6;
          constexpr int index = 12;

          const __m512i rhs_16_bit_dup_low = _mm512_set1_epi32(rhs_data[index]);
          const __m512i rhs_16_bit_dup_high =
              _mm512_set1_epi32(rhs_data[index + 1]);

          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
          accum_data_v6 = accum_v;
        }
        // Process column 7.
        {
          __m512i accum_v = accum_data_v7;
          constexpr int index = 14;

          const __m512i rhs_16_bit_dup_low = _mm512_set1_epi32(rhs_data[index]);
          const __m512i rhs_16_bit_dup_high =
              _mm512_set1_epi32(rhs_data[index + 1]);

          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
          accum_data_v7 = accum_v;
        }
        // Process column 8.
        {
          __m512i accum_v = accum_data_v8;
          constexpr int index = 16;

          const __m512i rhs_16_bit_dup_low = _mm512_set1_epi32(rhs_data[index]);
          const __m512i rhs_16_bit_dup_high =
              _mm512_set1_epi32(rhs_data[index + 1]);

          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
          accum_data_v8 = accum_v;
        }
        // Process column 9.
        {
          __m512i accum_v = accum_data_v9;
          constexpr int index = 18;

          const __m512i rhs_16_bit_dup_low = _mm512_set1_epi32(rhs_data[index]);
          const __m512i rhs_16_bit_dup_high =
              _mm512_set1_epi32(rhs_data[index + 1]);

          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
          accum_data_v9 = accum_v;
        }
        // Process column 10.
        {
          __m512i accum_v = accum_data_va;
          constexpr int index = 20;

          const __m512i rhs_16_bit_dup_low = _mm512_set1_epi32(rhs_data[index]);
          const __m512i rhs_16_bit_dup_high =
              _mm512_set1_epi32(rhs_data[index + 1]);

          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
          accum_data_va = accum_v;
        }
        // Process column 11.
        {
          __m512i accum_v = accum_data_vb;
          constexpr int index = 22;

          const __m512i rhs_16_bit_dup_low = _mm512_set1_epi32(rhs_data[index]);
          const __m512i rhs_16_bit_dup_high =
              _mm512_set1_epi32(rhs_data[index + 1]);

          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
          accum_data_vb = accum_v;
        }
        // Process column 12.
        {
          __m512i accum_v = accum_data_vc;
          constexpr int index = 24;

          const __m512i rhs_16_bit_dup_low = _mm512_set1_epi32(rhs_data[index]);
          const __m512i rhs_16_bit_dup_high =
              _mm512_set1_epi32(rhs_data[index + 1]);

          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
          accum_data_vc = accum_v;
        }
        // Process column 13.
        {
          __m512i accum_v = accum_data_vd;
          constexpr int index = 26;

          const __m512i rhs_16_bit_dup_low = _mm512_set1_epi32(rhs_data[index]);
          const __m512i rhs_16_bit_dup_high =
              _mm512_set1_epi32(rhs_data[index + 1]);

          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
          accum_data_vd = accum_v;
        }
        // Process column 14.
        {
          __m512i accum_v = accum_data_ve;
          constexpr int index = 28;

          const __m512i rhs_16_bit_dup_low = _mm512_set1_epi32(rhs_data[index]);
          const __m512i rhs_16_bit_dup_high =
              _mm512_set1_epi32(rhs_data[index + 1]);

          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
          accum_data_ve = accum_v;
        }
        // Process column 15.
        {
          __m512i accum_v = accum_data_vf;
          constexpr int index = 30;

          const __m512i rhs_16_bit_dup_low = _mm512_set1_epi32(rhs_data[index]);
          const __m512i rhs_16_bit_dup_high =
              _mm512_set1_epi32(rhs_data[index + 1]);

          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_v = _mm512_add_epi32(
              accum_v, _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
          accum_data_vf = accum_v;
        }

        lhs_ptr += 16 * 4;
        rhs_ptr += 16 * 4;
      }

      if (params.dst_type_id != DstTypeId<std::int32_t>::kValue) {
        __m512i m_vector;
        __m512i e_vector;
        // Does not make use of RUY_ASM_FLAG_NEEDS_LEFT_SHIFT.
        if (params.flags & RUY_ASM_FLAG_HAS_PERCHANNEL) {
          m_vector = _mm512_maskz_loadu_epi32(
              row_mask, &params.multiplier_fixedpoint[row]);
          e_vector = _mm512_maskz_loadu_epi32(row_mask,
                                              &params.multiplier_exponent[row]);
        } else {
          // These arrays have size LhsCols, and are pre-filled.
          m_vector = _mm512_set1_epi32(params.multiplier_fixedpoint[0]);
          e_vector = _mm512_set1_epi32(params.multiplier_exponent[0]);
        }

        const __m512i m_64bit_low =
            _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(m_vector, 0));
        const __m512i m_64bit_high =
            _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(m_vector, 1));

        const __m512i zero_vector = _mm512_setzero_epi32();
        const __m512i left_shift = _mm512_max_epi32(e_vector, zero_vector);
        const __m512i neg_e_vector = _mm512_sub_epi32(zero_vector, e_vector);
        const __m512i right_shift = _mm512_max_epi32(neg_e_vector, zero_vector);
        const __m512i final_right_shift =
            _mm512_add_epi32(right_shift, _mm512_set1_epi32(31));
        const __m512i final_right_shift_low = _mm512_cvtepi32_epi64(
            _mm512_extracti32x8_epi32(final_right_shift, 0));
        const __m512i final_right_shift_high = _mm512_cvtepi32_epi64(
            _mm512_extracti32x8_epi32(final_right_shift, 1));

        const __m512i offset_vector =
            _mm512_slli_epi64(_mm512_set1_epi64(1), 30);
        // Really these should be shifted by neg_e_vector, but tests pass when
        // using right_shift.
        const __m512i offset_vector_low = _mm512_sllv_epi64(
            offset_vector,
            _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(right_shift, 0)));
        const __m512i offset_vector_high = _mm512_sllv_epi64(
            offset_vector,
            _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(right_shift, 1)));

        // Shift and round column 0.
        {
          accum_data_v0 = _mm512_sllv_epi32(accum_data_v0, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m512i scaled_v_low =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v0, 0)),
                               m_64bit_low);
          __m512i scaled_v_high =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v0, 1)),
                               m_64bit_high);

          scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

          accum_data_v0 =
              _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
          accum_data_v0 = _mm512_inserti32x8(
              accum_data_v0, _mm512_cvtepi64_epi32(scaled_v_high), 1);
        }
        // Shift and round column 1.
        {
          accum_data_v1 = _mm512_sllv_epi32(accum_data_v1, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m512i scaled_v_low =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v1, 0)),
                               m_64bit_low);
          __m512i scaled_v_high =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v1, 1)),
                               m_64bit_high);

          scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

          accum_data_v1 =
              _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
          accum_data_v1 = _mm512_inserti32x8(
              accum_data_v1, _mm512_cvtepi64_epi32(scaled_v_high), 1);
        }
        // Shift and round column 2.
        {
          accum_data_v2 = _mm512_sllv_epi32(accum_data_v2, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m512i scaled_v_low =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v2, 0)),
                               m_64bit_low);
          __m512i scaled_v_high =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v2, 1)),
                               m_64bit_high);

          scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

          accum_data_v2 =
              _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
          accum_data_v2 = _mm512_inserti32x8(
              accum_data_v2, _mm512_cvtepi64_epi32(scaled_v_high), 1);
        }
        // Shift and round column 3.
        {
          accum_data_v3 = _mm512_sllv_epi32(accum_data_v3, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m512i scaled_v_low =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v3, 0)),
                               m_64bit_low);
          __m512i scaled_v_high =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v3, 1)),
                               m_64bit_high);

          scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

          accum_data_v3 =
              _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
          accum_data_v3 = _mm512_inserti32x8(
              accum_data_v3, _mm512_cvtepi64_epi32(scaled_v_high), 1);
        }
        // Shift and round column 4.
        {
          accum_data_v4 = _mm512_sllv_epi32(accum_data_v4, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m512i scaled_v_low =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v4, 0)),
                               m_64bit_low);
          __m512i scaled_v_high =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v4, 1)),
                               m_64bit_high);

          scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

          accum_data_v4 =
              _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
          accum_data_v4 = _mm512_inserti32x8(
              accum_data_v4, _mm512_cvtepi64_epi32(scaled_v_high), 1);
        }
        // Shift and round column 5.
        {
          accum_data_v5 = _mm512_sllv_epi32(accum_data_v5, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m512i scaled_v_low =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v5, 0)),
                               m_64bit_low);
          __m512i scaled_v_high =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v5, 1)),
                               m_64bit_high);

          scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

          accum_data_v5 =
              _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
          accum_data_v5 = _mm512_inserti32x8(
              accum_data_v5, _mm512_cvtepi64_epi32(scaled_v_high), 1);
        }
        // Shift and round column 6.
        {
          accum_data_v6 = _mm512_sllv_epi32(accum_data_v6, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m512i scaled_v_low =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v6, 0)),
                               m_64bit_low);
          __m512i scaled_v_high =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v6, 1)),
                               m_64bit_high);

          scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

          accum_data_v6 =
              _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
          accum_data_v6 = _mm512_inserti32x8(
              accum_data_v6, _mm512_cvtepi64_epi32(scaled_v_high), 1);
        }
        // Shift and round column 7.
        {
          accum_data_v7 = _mm512_sllv_epi32(accum_data_v7, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m512i scaled_v_low =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v7, 0)),
                               m_64bit_low);
          __m512i scaled_v_high =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v7, 1)),
                               m_64bit_high);

          scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

          accum_data_v7 =
              _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
          accum_data_v7 = _mm512_inserti32x8(
              accum_data_v7, _mm512_cvtepi64_epi32(scaled_v_high), 1);
        }
        // Shift and round column 8.
        {
          accum_data_v8 = _mm512_sllv_epi32(accum_data_v8, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m512i scaled_v_low =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v8, 0)),
                               m_64bit_low);
          __m512i scaled_v_high =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v8, 1)),
                               m_64bit_high);

          scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

          accum_data_v8 =
              _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
          accum_data_v8 = _mm512_inserti32x8(
              accum_data_v8, _mm512_cvtepi64_epi32(scaled_v_high), 1);
        }
        // Shift and round column 9.
        {
          accum_data_v9 = _mm512_sllv_epi32(accum_data_v9, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m512i scaled_v_low =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v9, 0)),
                               m_64bit_low);
          __m512i scaled_v_high =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_v9, 1)),
                               m_64bit_high);

          scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

          accum_data_v9 =
              _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
          accum_data_v9 = _mm512_inserti32x8(
              accum_data_v9, _mm512_cvtepi64_epi32(scaled_v_high), 1);
        }
        // Shift and round column 10.
        {
          accum_data_va = _mm512_sllv_epi32(accum_data_va, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m512i scaled_v_low =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_va, 0)),
                               m_64bit_low);
          __m512i scaled_v_high =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_va, 1)),
                               m_64bit_high);

          scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

          accum_data_va =
              _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
          accum_data_va = _mm512_inserti32x8(
              accum_data_va, _mm512_cvtepi64_epi32(scaled_v_high), 1);
        }
        // Shift and round column 11.
        {
          accum_data_vb = _mm512_sllv_epi32(accum_data_vb, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m512i scaled_v_low =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_vb, 0)),
                               m_64bit_low);
          __m512i scaled_v_high =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_vb, 1)),
                               m_64bit_high);

          scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

          accum_data_vb =
              _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
          accum_data_vb = _mm512_inserti32x8(
              accum_data_vb, _mm512_cvtepi64_epi32(scaled_v_high), 1);
        }
        // Shift and round column 12.
        {
          accum_data_vc = _mm512_sllv_epi32(accum_data_vc, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m512i scaled_v_low =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_vc, 0)),
                               m_64bit_low);
          __m512i scaled_v_high =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_vc, 1)),
                               m_64bit_high);

          scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

          accum_data_vc =
              _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
          accum_data_vc = _mm512_inserti32x8(
              accum_data_vc, _mm512_cvtepi64_epi32(scaled_v_high), 1);
        }
        // Shift and round column 13.
        {
          accum_data_vd = _mm512_sllv_epi32(accum_data_vd, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m512i scaled_v_low =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_vd, 0)),
                               m_64bit_low);
          __m512i scaled_v_high =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_vd, 1)),
                               m_64bit_high);

          scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

          accum_data_vd =
              _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
          accum_data_vd = _mm512_inserti32x8(
              accum_data_vd, _mm512_cvtepi64_epi32(scaled_v_high), 1);
        }
        // Shift and round column 14.
        {
          accum_data_ve = _mm512_sllv_epi32(accum_data_ve, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m512i scaled_v_low =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_ve, 0)),
                               m_64bit_low);
          __m512i scaled_v_high =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_ve, 1)),
                               m_64bit_high);

          scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

          accum_data_ve =
              _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
          accum_data_ve = _mm512_inserti32x8(
              accum_data_ve, _mm512_cvtepi64_epi32(scaled_v_high), 1);
        }
        // Shift and round column 15.
        {
          accum_data_vf = _mm512_sllv_epi32(accum_data_vf, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m512i scaled_v_low =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_vf, 0)),
                               m_64bit_low);
          __m512i scaled_v_high =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(
                                   _mm512_extracti32x8_epi32(accum_data_vf, 1)),
                               m_64bit_high);

          scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

          accum_data_vf =
              _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
          accum_data_vf = _mm512_inserti32x8(
              accum_data_vf, _mm512_cvtepi64_epi32(scaled_v_high), 1);
        }
#if !RUY_OPT_ENABLED(RUY_OPT_NATIVE_ROUNDING)
        RUY_DCHECK(false);
#endif

        if (params.dst_zero_point != 0) {
          __m512i dst_zero_point = _mm512_set1_epi32(params.dst_zero_point);
          accum_data_v0 = _mm512_add_epi32(accum_data_v0, dst_zero_point);
          accum_data_v1 = _mm512_add_epi32(accum_data_v1, dst_zero_point);
          accum_data_v2 = _mm512_add_epi32(accum_data_v2, dst_zero_point);
          accum_data_v3 = _mm512_add_epi32(accum_data_v3, dst_zero_point);
          accum_data_v4 = _mm512_add_epi32(accum_data_v4, dst_zero_point);
          accum_data_v5 = _mm512_add_epi32(accum_data_v5, dst_zero_point);
          accum_data_v6 = _mm512_add_epi32(accum_data_v6, dst_zero_point);
          accum_data_v7 = _mm512_add_epi32(accum_data_v7, dst_zero_point);
          accum_data_v8 = _mm512_add_epi32(accum_data_v8, dst_zero_point);
          accum_data_v9 = _mm512_add_epi32(accum_data_v9, dst_zero_point);
          accum_data_va = _mm512_add_epi32(accum_data_va, dst_zero_point);
          accum_data_vb = _mm512_add_epi32(accum_data_vb, dst_zero_point);
          accum_data_vc = _mm512_add_epi32(accum_data_vc, dst_zero_point);
          accum_data_vd = _mm512_add_epi32(accum_data_vd, dst_zero_point);
          accum_data_ve = _mm512_add_epi32(accum_data_ve, dst_zero_point);
          accum_data_vf = _mm512_add_epi32(accum_data_vf, dst_zero_point);
        }
      }

      const __m512i clamp_max_v = _mm512_set1_epi32(params.clamp_max);
      const __m512i clamp_min_v = _mm512_set1_epi32(params.clamp_min);

      const bool store_full_block =
          (residual_rows == 16) && (residual_cols == 16);

      __m512i accum_data_v[16];

      // In most cases we would make this conditional on (!store_full_block) and
      // unwind the clamp-and-store loop, but the benefit appears small.
      {
        accum_data_v[0] = accum_data_v0;
        accum_data_v[1] = accum_data_v1;
        accum_data_v[2] = accum_data_v2;
        accum_data_v[3] = accum_data_v3;
        accum_data_v[4] = accum_data_v4;
        accum_data_v[5] = accum_data_v5;
        accum_data_v[6] = accum_data_v6;
        accum_data_v[7] = accum_data_v7;
        accum_data_v[8] = accum_data_v8;
        accum_data_v[9] = accum_data_v9;
        accum_data_v[10] = accum_data_va;
        accum_data_v[11] = accum_data_vb;
        accum_data_v[12] = accum_data_vc;
        accum_data_v[13] = accum_data_vd;
        accum_data_v[14] = accum_data_ve;
        accum_data_v[15] = accum_data_vf;
      }

      if (params.dst_type_id == DstTypeId<std::int8_t>::kValue) {
        std::int8_t* tmp_ptr = static_cast<std::int8_t*>(dst_ptr);
        const int block_col_offset = dst_stride;
        if (store_full_block) {
          for (int j = 0; j < 16; ++j) {
            __m512i result = accum_data_v[j];
            result = _mm512_min_epi32(result, clamp_max_v);
            result = _mm512_max_epi32(result, clamp_min_v);
            _mm_storeu_epi8(tmp_ptr + j * block_col_offset,
                            _mm512_cvtepi32_epi8(result));
          }
        } else {
          for (int j = 0; j < residual_cols; ++j) {
            __m512i result = accum_data_v[j];
            result = _mm512_min_epi32(result, clamp_max_v);
            result = _mm512_max_epi32(result, clamp_min_v);
            _mm_mask_storeu_epi8(tmp_ptr + j * block_col_offset, row_mask,
                                 _mm512_cvtepi32_epi8(result));
          }
        }
        dst_ptr = static_cast<void*>(static_cast<std::int8_t*>(dst_ptr) + 16);
      } else if (params.dst_type_id == DstTypeId<std::uint8_t>::kValue) {
        std::uint8_t* tmp_ptr = static_cast<std::uint8_t*>(dst_ptr);
        const int block_col_offset = dst_stride;
        if (store_full_block) {
          for (int j = 0; j < residual_cols; ++j) {
            __m512i result = accum_data_v[j];
            result = _mm512_min_epi32(result, clamp_max_v);
            result = _mm512_max_epi32(result, clamp_min_v);
            _mm_storeu_epi8(tmp_ptr + j * block_col_offset,
                            _mm512_cvtepi32_epi8(result));
          }
        } else {
          for (int j = 0; j < residual_cols; ++j) {
            __m512i result = accum_data_v[j];
            result = _mm512_min_epi32(result, clamp_max_v);
            result = _mm512_max_epi32(result, clamp_min_v);
            _mm_mask_storeu_epi8(tmp_ptr + j * block_col_offset, row_mask,
                                 _mm512_cvtepi32_epi8(result));
          }
        }
        dst_ptr = static_cast<void*>(static_cast<std::uint8_t*>(dst_ptr) + 16);
      } else if (params.dst_type_id == DstTypeId<std::int16_t>::kValue) {
        std::int16_t* tmp_ptr = static_cast<std::int16_t*>(dst_ptr);
        const int block_col_offset = dst_stride;
        if (store_full_block) {
          for (int j = 0; j < 16; ++j) {
            __m512i result = accum_data_v[j];
            result = _mm512_min_epi32(result, clamp_max_v);
            result = _mm512_max_epi32(result, clamp_min_v);
            _mm256_storeu_epi16(tmp_ptr + j * block_col_offset,
                                _mm512_cvtepi32_epi16(result));
          }
        } else {
          for (int j = 0; j < residual_cols; ++j) {
            __m512i result = accum_data_v[j];
            result = _mm512_min_epi32(result, clamp_max_v);
            result = _mm512_max_epi32(result, clamp_min_v);
            _mm256_mask_storeu_epi16(tmp_ptr + j * block_col_offset, row_mask,
                                     _mm512_cvtepi32_epi16(result));
          }
        }
        dst_ptr = static_cast<void*>(static_cast<std::int16_t*>(dst_ptr) + 16);
      } else if (params.dst_type_id == DstTypeId<std::int32_t>::kValue) {
        if (store_full_block) {
          std::int32_t* tmp_ptr = static_cast<std::int32_t*>(dst_ptr);
          for (int j = 0; j < 16; ++j) {
            _mm512_storeu_epi32(tmp_ptr + j * dst_stride, accum_data_v[j]);
          }
        } else {
          std::int32_t* tmp_ptr = static_cast<std::int32_t*>(dst_ptr);
          for (int j = 0; j < residual_cols; ++j) {
            _mm512_mask_storeu_epi32(tmp_ptr + j * dst_stride, row_mask,
                                     accum_data_v[j]);
          }
        }
        dst_ptr = static_cast<void*>(static_cast<std::int32_t*>(dst_ptr) + 16);
      } else {
        RUY_DCHECK(false);
      }

      lhs_col_ptr += 16 * params.lhs_stride;
    }  // End row-block loop.

    dst_col_ptr = static_cast<void*>(static_cast<char*>(dst_col_ptr) +
                                     16 * params.dst_stride);
    rhs_col_ptr += 16 * params.rhs_stride;
  }  // End col-block loop.
}  // NOLINT(readability/fn_size)

void Kernel8bitAvx512SingleCol(const KernelParams8bit<16, 16>& params) {
  gemmlowp::ScopedProfilingLabel label("Kernel kAvx512 8-bit GEMV");

  RUY_DCHECK_EQ(params.dst_cols, 1);
  RUY_DCHECK_EQ(params.last_col, 0);
  RUY_DCHECK_EQ(params.start_col, 0);

  std::int32_t dst_stride;
  if ((params.dst_type_id == DstTypeId<std::int8_t>::kValue) ||
      (params.dst_type_id == DstTypeId<std::uint8_t>::kValue)) {
    dst_stride = params.dst_stride;
  } else if (params.dst_type_id == DstTypeId<std::int16_t>::kValue) {
    dst_stride = params.dst_stride / sizeof(std::int16_t);
  } else if (params.dst_type_id == DstTypeId<std::int32_t>::kValue) {
    dst_stride = params.dst_stride / sizeof(std::int32_t);
  } else {
    RUY_DCHECK(false);
  }

  int bias_ptr_block_increment = params.flags & RUY_ASM_FLAG_HAS_BIAS ? 16 : 0;

  const std::int8_t* rhs_col_ptr = params.rhs_base_ptr;
  void* dst_col_ptr = params.dst_base_ptr;
  const std::int32_t* bias_col_ptr = params.bias;
  if (params.flags & RUY_ASM_FLAG_HAS_BIAS) {
    bias_col_ptr += params.start_row;
  }

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  void* dst_ptr = dst_col_ptr;
  const std::int32_t* bias_ptr = bias_col_ptr;

  const std::int32_t lhs_zero_point = params.lhs_zero_point;
  const bool has_rhs_sums_offsets =
      (params.flags & RUY_ASM_FLAG_HAS_RHS_SUMS) && lhs_zero_point;
  std::int32_t rhs_sums_offsets[16];
  if (has_rhs_sums_offsets) {
    const __m512i rhs_sums_offset_v =
        _mm512_mullo_epi32(_mm512_set1_epi32(lhs_zero_point),
                           _mm512_loadu_epi32(&params.rhs_sums[0]));
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(rhs_sums_offsets),
                        rhs_sums_offset_v);
  }

  for (int row = params.start_row; row <= params.last_row; row += 16) {
    const int residual_rows = std::min(params.dst_rows - row, 16);

    __m512i accum_data_v0;

    // Initialize with bias.
    const __mmask16 row_mask =
        (static_cast<std::uint32_t>(1) << residual_rows) - 1;
    __m512i initial_accum_data = _mm512_maskz_loadu_epi32(row_mask, bias_ptr);
    bias_ptr += bias_ptr_block_increment;

    const std::int32_t rhs_zero_point = params.rhs_zero_point;
    if ((params.flags & RUY_ASM_FLAG_HAS_LHS_SUMS) && rhs_zero_point) {
      const __m512i lhs_sums_offset =
          _mm512_mullo_epi32(_mm512_set1_epi32(rhs_zero_point),
                             _mm512_loadu_epi32(&params.lhs_sums[row]));
      initial_accum_data =
          _mm512_sub_epi32(initial_accum_data, lhs_sums_offset);
    }

    const std::int32_t prod_zp_depth = params.prod_zp_depth;
    if (prod_zp_depth != 0) {
      initial_accum_data = _mm512_add_epi32(initial_accum_data,
                                            _mm512_set1_epi32(prod_zp_depth));
    }

    // Adjustments differing across columns.
    if (has_rhs_sums_offsets) {
      accum_data_v0 = _mm512_sub_epi32(initial_accum_data,
                                       _mm512_set1_epi32(rhs_sums_offsets[0]));
    } else {
      accum_data_v0 = initial_accum_data;
    }

    const std::int8_t* lhs_ptr = lhs_col_ptr;
    const std::int8_t* rhs_ptr = rhs_col_ptr;
    for (int d = 0; d < params.depth; d += 4) {
      const __m512i lhs_data = _mm512_loadu_epi8(lhs_ptr);
      const __m128i rhs_data_8bit = _mm_loadu_epi8(rhs_ptr);

      // Each "int32" is two 16-bit RHS values, sign extended from 8-bit.
      // For simplicity we load 4x the data that we need and process twice the
      // data  that we need  and store only the data we need.
      std::int32_t rhs_data[2];
      const __m128i rhs_16_bit_dup = _mm_cvtepi8_epi16(rhs_data_8bit);
      // Now that we have cast the RHS data, we store it so that each value
      // can be separately loaded in the accumulation loop.
      _mm_storeu_si64(reinterpret_cast<__m128i*>(rhs_data), rhs_16_bit_dup);

      // Take bytes 0, 1, 4, 5, 8, 9, ... and expand to 16-bit.
      const __m512i lhs_16_bit_low =
          _mm512_cvtepi8_epi16(_mm512_cvtepi32_epi16(lhs_data));
      // Take bytes 2, 3, 6, 7, 10, 11, ... and expand to 16-bit.
      const __m512i lhs_16_bit_high = _mm512_cvtepi8_epi16(
          _mm512_cvtepi32_epi16(_mm512_srli_epi32(lhs_data, 16)));

      // Process column 0.
      __m512i accum_v = accum_data_v0;
      constexpr int index = 0;

      const __m512i rhs_16_bit_dup_low = _mm512_set1_epi32(rhs_data[index]);
      const __m512i rhs_16_bit_dup_high =
          _mm512_set1_epi32(rhs_data[index + 1]);

      accum_v = _mm512_add_epi32(
          accum_v, _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
      accum_v = _mm512_add_epi32(
          accum_v, _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
      accum_data_v0 = accum_v;

      lhs_ptr += 16 * 4;
      rhs_ptr += 16 * 4;
    }

    if (params.dst_type_id != DstTypeId<std::int32_t>::kValue) {
      __m512i m_vector;
      __m512i e_vector;
      // Does not make use of RUY_ASM_FLAG_NEEDS_LEFT_SHIFT.
      if (params.flags & RUY_ASM_FLAG_HAS_PERCHANNEL) {
        m_vector = _mm512_maskz_loadu_epi32(row_mask,
                                            &params.multiplier_fixedpoint[row]);
        e_vector = _mm512_maskz_loadu_epi32(row_mask,
                                            &params.multiplier_exponent[row]);
      } else {
        // These arrays have size LhsCols, and are pre-filled.
        m_vector = _mm512_set1_epi32(params.multiplier_fixedpoint[0]);
        e_vector = _mm512_set1_epi32(params.multiplier_exponent[0]);
      }

      const __m512i m_64bit_low =
          _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(m_vector, 0));
      const __m512i m_64bit_high =
          _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(m_vector, 1));

      const __m512i zero_vector = _mm512_setzero_epi32();
      const __m512i left_shift = _mm512_max_epi32(e_vector, zero_vector);
      const __m512i neg_e_vector = _mm512_sub_epi32(zero_vector, e_vector);
      const __m512i right_shift = _mm512_max_epi32(neg_e_vector, zero_vector);
      const __m512i final_right_shift =
          _mm512_add_epi32(right_shift, _mm512_set1_epi32(31));
      const __m512i final_right_shift_low = _mm512_cvtepi32_epi64(
          _mm512_extracti32x8_epi32(final_right_shift, 0));
      const __m512i final_right_shift_high = _mm512_cvtepi32_epi64(
          _mm512_extracti32x8_epi32(final_right_shift, 1));

      const __m512i offset_vector = _mm512_slli_epi64(_mm512_set1_epi64(1), 30);
      // Really these should be shifted by neg_e_vector, but tests pass when
      // using right_shift.
      const __m512i offset_vector_low = _mm512_sllv_epi64(
          offset_vector,
          _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(right_shift, 0)));
      const __m512i offset_vector_high = _mm512_sllv_epi64(
          offset_vector,
          _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(right_shift, 1)));

      // Shift and round column 0.
      accum_data_v0 = _mm512_sllv_epi32(accum_data_v0, left_shift);
      // Apply the fixed-point part of the multiplier.
      __m512i scaled_v_low = _mm512_mul_epi32(
          _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(accum_data_v0, 0)),
          m_64bit_low);
      __m512i scaled_v_high = _mm512_mul_epi32(
          _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(accum_data_v0, 1)),
          m_64bit_high);

      scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
      scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

      scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
      scaled_v_high = _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

      accum_data_v0 =
          _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
      accum_data_v0 = _mm512_inserti32x8(
          accum_data_v0, _mm512_cvtepi64_epi32(scaled_v_high), 1);
#if !RUY_OPT_ENABLED(RUY_OPT_NATIVE_ROUNDING)
      RUY_DCHECK(false);
#endif

      if (params.dst_zero_point != 0) {
        __m512i dst_zero_point = _mm512_set1_epi32(params.dst_zero_point);
        accum_data_v0 = _mm512_add_epi32(accum_data_v0, dst_zero_point);
      }
    }

    const __m512i clamp_max_v = _mm512_set1_epi32(params.clamp_max);
    const __m512i clamp_min_v = _mm512_set1_epi32(params.clamp_min);

    if (params.dst_type_id == DstTypeId<std::int8_t>::kValue) {
      std::int8_t* tmp_ptr = static_cast<std::int8_t*>(dst_ptr);
      __m512i result = accum_data_v0;
      result = _mm512_min_epi32(result, clamp_max_v);
      result = _mm512_max_epi32(result, clamp_min_v);
      _mm_mask_storeu_epi8(tmp_ptr, row_mask, _mm512_cvtepi32_epi8(result));
      dst_ptr = static_cast<void*>(static_cast<std::int8_t*>(dst_ptr) + 16);
    } else if (params.dst_type_id == DstTypeId<std::uint8_t>::kValue) {
      std::uint8_t* tmp_ptr = static_cast<std::uint8_t*>(dst_ptr);
      __m512i result = accum_data_v0;
      result = _mm512_min_epi32(result, clamp_max_v);
      result = _mm512_max_epi32(result, clamp_min_v);
      _mm_mask_storeu_epi8(tmp_ptr, row_mask, _mm512_cvtepi32_epi8(result));
      dst_ptr = static_cast<void*>(static_cast<std::uint8_t*>(dst_ptr) + 16);
    } else if (params.dst_type_id == DstTypeId<std::int16_t>::kValue) {
      std::int16_t* tmp_ptr = static_cast<std::int16_t*>(dst_ptr);
      __m512i result = accum_data_v0;
      result = _mm512_min_epi32(result, clamp_max_v);
      result = _mm512_max_epi32(result, clamp_min_v);
      _mm256_mask_storeu_epi16(tmp_ptr, row_mask,
                               _mm512_cvtepi32_epi16(result));
      dst_ptr = static_cast<void*>(static_cast<std::int16_t*>(dst_ptr) + 16);
    } else if (params.dst_type_id == DstTypeId<std::int32_t>::kValue) {
      std::int32_t* tmp_ptr = static_cast<std::int32_t*>(dst_ptr);
      _mm512_mask_storeu_epi32(tmp_ptr, row_mask, accum_data_v0);
      dst_ptr = static_cast<void*>(static_cast<std::int32_t*>(dst_ptr) + 16);
    } else {
      RUY_DCHECK(false);
    }

    lhs_col_ptr += 16 * params.lhs_stride;
  }  // End row-block loop.
}  // NOLINT(readability/fn_size)

void KernelFloatAvx512(const KernelParamsFloat<16, 16>& params) {
  gemmlowp::ScopedProfilingLabel label("Kernel kAvx512 float");

  // As parameters are defined, we need to scale by sizeof(float).
  const std::int64_t lhs_stride = params.lhs_stride >> 2;
  const std::int64_t dst_stride = params.dst_stride >> 2;
  const std::int64_t rhs_stride = params.rhs_stride >> 2;

  int bias_ptr_block_increment = params.flags & RUY_ASM_FLAG_HAS_BIAS ? 1 : 0;
  const int end_row = std::min(params.dst_rows, params.last_row + 16);
  const int end_col = std::min(params.dst_cols, params.last_col + 16);

  const float* adj_rhs_col_ptr =
      params.rhs_base_ptr - params.start_col * rhs_stride;
  float* adj_dst_col_ptr =
      params.dst_base_ptr - params.start_col * dst_stride - params.start_row;
  const float* adj_lhs_col_ptr =
      params.lhs_base_ptr - params.start_row * lhs_stride;
  const float* bias_col_ptr = params.bias;

  const __m512 clamp_max_v = _mm512_set1_ps(params.clamp_max);
  const __m512 clamp_min_v = _mm512_set1_ps(params.clamp_min);

  int col = params.start_col;
  for (; col <= end_col - 16; col += 16) {
    const float* rhs_col_ptr = adj_rhs_col_ptr + col * rhs_stride;
    float* dst_col_ptr = adj_dst_col_ptr + col * dst_stride;

    int row = params.start_row;
    for (; row <= end_row - 16; row += 16) {
      const float* lhs_col_ptr = adj_lhs_col_ptr + row * lhs_stride;
      float* dst_ptr = dst_col_ptr + row;
      const float* bias_ptr = bias_col_ptr + row * bias_ptr_block_increment;

      // Initialize with bias.
      const __m512 initial_accum_data = _mm512_loadu_ps(bias_ptr);

      // Process block in two halves, split by columns.
      {
        constexpr int mmm = 0;

        __m512 accum_data_v0 = initial_accum_data;
        __m512 accum_data_v1 = initial_accum_data;
        __m512 accum_data_v2 = initial_accum_data;
        __m512 accum_data_v3 = initial_accum_data;
        __m512 accum_data_v4 = initial_accum_data;
        __m512 accum_data_v5 = initial_accum_data;
        __m512 accum_data_v6 = initial_accum_data;
        __m512 accum_data_v7 = initial_accum_data;

        const float* lhs_ptr = lhs_col_ptr;
        const float* rhs_ptr = rhs_col_ptr + 8 * mmm;
        for (int d = 0; d < (params.depth - 1); ++d) {
          const __m512 lhs_data = _mm512_loadu_ps(lhs_ptr);
          // In this version RHS values are loaded individually rather than
          // first loading together and then extract with broadcasting. This is
          // because AVX flavours and instrinsics and compilers in combination
          // do not handle this pattern of extraction very well.
          const float* rhs_data = rhs_ptr;
          lhs_ptr += 16;
          rhs_ptr += 16;

          {
            const __m512 dup_rhs_element_j0 = _mm512_set1_ps(rhs_data[0]);
            accum_data_v0 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j0, accum_data_v0);
            const __m512 dup_rhs_element_j1 = _mm512_set1_ps(rhs_data[1]);
            accum_data_v1 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j1, accum_data_v1);
            const __m512 dup_rhs_element_j2 = _mm512_set1_ps(rhs_data[2]);
            accum_data_v2 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j2, accum_data_v2);
            const __m512 dup_rhs_element_j3 = _mm512_set1_ps(rhs_data[3]);
            accum_data_v3 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j3, accum_data_v3);
            const __m512 dup_rhs_element_j4 = _mm512_set1_ps(rhs_data[4]);
            accum_data_v4 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j4, accum_data_v4);
            const __m512 dup_rhs_element_j5 = _mm512_set1_ps(rhs_data[5]);
            accum_data_v5 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j5, accum_data_v5);
            const __m512 dup_rhs_element_j6 = _mm512_set1_ps(rhs_data[6]);
            accum_data_v6 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j6, accum_data_v6);
            const __m512 dup_rhs_element_j7 = _mm512_set1_ps(rhs_data[7]);
            accum_data_v7 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j7, accum_data_v7);
          }
        }
        {
          const __m512 lhs_data = _mm512_loadu_ps(lhs_ptr);
          const float* rhs_data = rhs_ptr;
          {
            const __m512 dup_rhs_element_j0 = _mm512_set1_ps(rhs_data[0]);
            accum_data_v0 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j0, accum_data_v0);
            const __m512 dup_rhs_element_j1 = _mm512_set1_ps(rhs_data[1]);
            accum_data_v1 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j1, accum_data_v1);
            const __m512 dup_rhs_element_j2 = _mm512_set1_ps(rhs_data[2]);
            accum_data_v2 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j2, accum_data_v2);
            const __m512 dup_rhs_element_j3 = _mm512_set1_ps(rhs_data[3]);
            accum_data_v3 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j3, accum_data_v3);
            const __m512 dup_rhs_element_j4 = _mm512_set1_ps(rhs_data[4]);
            accum_data_v4 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j4, accum_data_v4);
            const __m512 dup_rhs_element_j5 = _mm512_set1_ps(rhs_data[5]);
            accum_data_v5 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j5, accum_data_v5);
            const __m512 dup_rhs_element_j6 = _mm512_set1_ps(rhs_data[6]);
            accum_data_v6 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j6, accum_data_v6);
            const __m512 dup_rhs_element_j7 = _mm512_set1_ps(rhs_data[7]);
            accum_data_v7 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j7, accum_data_v7);
          }
          {
            float* block_ptr = dst_ptr + (mmm * 8 + 0) * dst_stride;
            accum_data_v0 = _mm512_min_ps(accum_data_v0, clamp_max_v);
            accum_data_v0 = _mm512_max_ps(accum_data_v0, clamp_min_v);
            _mm512_storeu_ps(block_ptr + 0 * dst_stride, accum_data_v0);
            accum_data_v1 = _mm512_min_ps(accum_data_v1, clamp_max_v);
            accum_data_v1 = _mm512_max_ps(accum_data_v1, clamp_min_v);
            _mm512_storeu_ps(block_ptr + 1 * dst_stride, accum_data_v1);
            accum_data_v2 = _mm512_min_ps(accum_data_v2, clamp_max_v);
            accum_data_v2 = _mm512_max_ps(accum_data_v2, clamp_min_v);
            _mm512_storeu_ps(block_ptr + 2 * dst_stride, accum_data_v2);
            accum_data_v3 = _mm512_min_ps(accum_data_v3, clamp_max_v);
            accum_data_v3 = _mm512_max_ps(accum_data_v3, clamp_min_v);
            _mm512_storeu_ps(block_ptr + 3 * dst_stride, accum_data_v3);
            accum_data_v4 = _mm512_min_ps(accum_data_v4, clamp_max_v);
            accum_data_v4 = _mm512_max_ps(accum_data_v4, clamp_min_v);
            _mm512_storeu_ps(block_ptr + 4 * dst_stride, accum_data_v4);
            accum_data_v5 = _mm512_min_ps(accum_data_v5, clamp_max_v);
            accum_data_v5 = _mm512_max_ps(accum_data_v5, clamp_min_v);
            _mm512_storeu_ps(block_ptr + 5 * dst_stride, accum_data_v5);
            accum_data_v6 = _mm512_min_ps(accum_data_v6, clamp_max_v);
            accum_data_v6 = _mm512_max_ps(accum_data_v6, clamp_min_v);
            _mm512_storeu_ps(block_ptr + 6 * dst_stride, accum_data_v6);
            accum_data_v7 = _mm512_min_ps(accum_data_v7, clamp_max_v);
            accum_data_v7 = _mm512_max_ps(accum_data_v7, clamp_min_v);
            _mm512_storeu_ps(block_ptr + 7 * dst_stride, accum_data_v7);
          }
        }
      }  // Inner half-block loop, unrolled, first iteration.
      {
        constexpr int mmm = 1;

        __m512 accum_data_v0 = initial_accum_data;
        __m512 accum_data_v1 = initial_accum_data;
        __m512 accum_data_v2 = initial_accum_data;
        __m512 accum_data_v3 = initial_accum_data;
        __m512 accum_data_v4 = initial_accum_data;
        __m512 accum_data_v5 = initial_accum_data;
        __m512 accum_data_v6 = initial_accum_data;
        __m512 accum_data_v7 = initial_accum_data;

        const float* lhs_ptr = lhs_col_ptr;
        const float* rhs_ptr = rhs_col_ptr + 8 * mmm;
        for (int d = 0; d < (params.depth - 1); ++d) {
          const __m512 lhs_data = _mm512_loadu_ps(lhs_ptr);
          const float* rhs_data = rhs_ptr;
          lhs_ptr += 16;
          rhs_ptr += 16;
          {
            const __m512 dup_rhs_element_j0 = _mm512_set1_ps(rhs_data[0]);
            accum_data_v0 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j0, accum_data_v0);
            const __m512 dup_rhs_element_j1 = _mm512_set1_ps(rhs_data[1]);
            accum_data_v1 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j1, accum_data_v1);
            const __m512 dup_rhs_element_j2 = _mm512_set1_ps(rhs_data[2]);
            accum_data_v2 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j2, accum_data_v2);
            const __m512 dup_rhs_element_j3 = _mm512_set1_ps(rhs_data[3]);
            accum_data_v3 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j3, accum_data_v3);
            const __m512 dup_rhs_element_j4 = _mm512_set1_ps(rhs_data[4]);
            accum_data_v4 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j4, accum_data_v4);
            const __m512 dup_rhs_element_j5 = _mm512_set1_ps(rhs_data[5]);
            accum_data_v5 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j5, accum_data_v5);
            const __m512 dup_rhs_element_j6 = _mm512_set1_ps(rhs_data[6]);
            accum_data_v6 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j6, accum_data_v6);
            const __m512 dup_rhs_element_j7 = _mm512_set1_ps(rhs_data[7]);
            accum_data_v7 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j7, accum_data_v7);
          }
        }
        {
          const __m512 lhs_data = _mm512_loadu_ps(lhs_ptr);
          const float* rhs_data = rhs_ptr;
          {
            const __m512 dup_rhs_element_j0 = _mm512_set1_ps(rhs_data[0]);
            accum_data_v0 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j0, accum_data_v0);
            const __m512 dup_rhs_element_j1 = _mm512_set1_ps(rhs_data[1]);
            accum_data_v1 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j1, accum_data_v1);
            const __m512 dup_rhs_element_j2 = _mm512_set1_ps(rhs_data[2]);
            accum_data_v2 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j2, accum_data_v2);
            const __m512 dup_rhs_element_j3 = _mm512_set1_ps(rhs_data[3]);
            accum_data_v3 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j3, accum_data_v3);
            const __m512 dup_rhs_element_j4 = _mm512_set1_ps(rhs_data[4]);
            accum_data_v4 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j4, accum_data_v4);
            const __m512 dup_rhs_element_j5 = _mm512_set1_ps(rhs_data[5]);
            accum_data_v5 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j5, accum_data_v5);
            const __m512 dup_rhs_element_j6 = _mm512_set1_ps(rhs_data[6]);
            accum_data_v6 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j6, accum_data_v6);
            const __m512 dup_rhs_element_j7 = _mm512_set1_ps(rhs_data[7]);
            accum_data_v7 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j7, accum_data_v7);
          }
          {
            float* block_ptr = dst_ptr + (mmm * 8 + 0) * dst_stride;
            accum_data_v0 = _mm512_min_ps(accum_data_v0, clamp_max_v);
            accum_data_v0 = _mm512_max_ps(accum_data_v0, clamp_min_v);
            _mm512_storeu_ps(block_ptr + 0 * dst_stride, accum_data_v0);
            accum_data_v1 = _mm512_min_ps(accum_data_v1, clamp_max_v);
            accum_data_v1 = _mm512_max_ps(accum_data_v1, clamp_min_v);
            _mm512_storeu_ps(block_ptr + 1 * dst_stride, accum_data_v1);
            accum_data_v2 = _mm512_min_ps(accum_data_v2, clamp_max_v);
            accum_data_v2 = _mm512_max_ps(accum_data_v2, clamp_min_v);
            _mm512_storeu_ps(block_ptr + 2 * dst_stride, accum_data_v2);
            accum_data_v3 = _mm512_min_ps(accum_data_v3, clamp_max_v);
            accum_data_v3 = _mm512_max_ps(accum_data_v3, clamp_min_v);
            _mm512_storeu_ps(block_ptr + 3 * dst_stride, accum_data_v3);
            accum_data_v4 = _mm512_min_ps(accum_data_v4, clamp_max_v);
            accum_data_v4 = _mm512_max_ps(accum_data_v4, clamp_min_v);
            _mm512_storeu_ps(block_ptr + 4 * dst_stride, accum_data_v4);
            accum_data_v5 = _mm512_min_ps(accum_data_v5, clamp_max_v);
            accum_data_v5 = _mm512_max_ps(accum_data_v5, clamp_min_v);
            _mm512_storeu_ps(block_ptr + 5 * dst_stride, accum_data_v5);
            accum_data_v6 = _mm512_min_ps(accum_data_v6, clamp_max_v);
            accum_data_v6 = _mm512_max_ps(accum_data_v6, clamp_min_v);
            _mm512_storeu_ps(block_ptr + 6 * dst_stride, accum_data_v6);
            accum_data_v7 = _mm512_min_ps(accum_data_v7, clamp_max_v);
            accum_data_v7 = _mm512_max_ps(accum_data_v7, clamp_min_v);
            _mm512_storeu_ps(block_ptr + 7 * dst_stride, accum_data_v7);
          }
        }
      }  // Inner half-block loop, unrolled, second iteration.
    }    // End row-block loop.

    // The unrolling within this conditional may be somewhat pointless. It
    // depends on the kinds of models.
    if (row < end_row) {
      const int residual_rows = end_row - row;

      const float* lhs_col_ptr = adj_lhs_col_ptr + row * lhs_stride;
      float* dst_ptr = dst_col_ptr + row;
      const float* bias_ptr = bias_col_ptr + row * bias_ptr_block_increment;

      // Initialize with bias.
      const __mmask16 row_mask =
          (static_cast<std::uint32_t>(1) << residual_rows) - 1;
      const __m512 initial_accum_data =
          _mm512_maskz_loadu_ps(row_mask, bias_ptr);

      // Process block in two halves, split by columns.
      for (int mmm = 0; mmm < 2; ++mmm) {
        __m512 accum_data_v0 = initial_accum_data;
        __m512 accum_data_v1 = initial_accum_data;
        __m512 accum_data_v2 = initial_accum_data;
        __m512 accum_data_v3 = initial_accum_data;
        __m512 accum_data_v4 = initial_accum_data;
        __m512 accum_data_v5 = initial_accum_data;
        __m512 accum_data_v6 = initial_accum_data;
        __m512 accum_data_v7 = initial_accum_data;

        const float* lhs_ptr = lhs_col_ptr;
        const float* rhs_ptr = rhs_col_ptr + 8 * mmm;
        for (int d = 0; d < (params.depth - 1); ++d) {
          const __m512 lhs_data = _mm512_loadu_ps(lhs_ptr);
          const float* rhs_data = rhs_ptr;
          lhs_ptr += 16;
          rhs_ptr += 16;
          {
            const __m512 dup_rhs_element_j0 = _mm512_set1_ps(rhs_data[0]);
            accum_data_v0 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j0, accum_data_v0);
            const __m512 dup_rhs_element_j1 = _mm512_set1_ps(rhs_data[1]);
            accum_data_v1 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j1, accum_data_v1);
            const __m512 dup_rhs_element_j2 = _mm512_set1_ps(rhs_data[2]);
            accum_data_v2 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j2, accum_data_v2);
            const __m512 dup_rhs_element_j3 = _mm512_set1_ps(rhs_data[3]);
            accum_data_v3 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j3, accum_data_v3);
            const __m512 dup_rhs_element_j4 = _mm512_set1_ps(rhs_data[4]);
            accum_data_v4 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j4, accum_data_v4);
            const __m512 dup_rhs_element_j5 = _mm512_set1_ps(rhs_data[5]);
            accum_data_v5 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j5, accum_data_v5);
            const __m512 dup_rhs_element_j6 = _mm512_set1_ps(rhs_data[6]);
            accum_data_v6 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j6, accum_data_v6);
            const __m512 dup_rhs_element_j7 = _mm512_set1_ps(rhs_data[7]);
            accum_data_v7 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j7, accum_data_v7);
          }
        }
        {
          const __m512 lhs_data = _mm512_loadu_ps(lhs_ptr);
          const float* rhs_data = rhs_ptr;
          {
            const __m512 dup_rhs_element_j0 = _mm512_set1_ps(rhs_data[0]);
            accum_data_v0 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j0, accum_data_v0);
            const __m512 dup_rhs_element_j1 = _mm512_set1_ps(rhs_data[1]);
            accum_data_v1 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j1, accum_data_v1);
            const __m512 dup_rhs_element_j2 = _mm512_set1_ps(rhs_data[2]);
            accum_data_v2 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j2, accum_data_v2);
            const __m512 dup_rhs_element_j3 = _mm512_set1_ps(rhs_data[3]);
            accum_data_v3 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j3, accum_data_v3);
            const __m512 dup_rhs_element_j4 = _mm512_set1_ps(rhs_data[4]);
            accum_data_v4 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j4, accum_data_v4);
            const __m512 dup_rhs_element_j5 = _mm512_set1_ps(rhs_data[5]);
            accum_data_v5 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j5, accum_data_v5);
            const __m512 dup_rhs_element_j6 = _mm512_set1_ps(rhs_data[6]);
            accum_data_v6 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j6, accum_data_v6);
            const __m512 dup_rhs_element_j7 = _mm512_set1_ps(rhs_data[7]);
            accum_data_v7 =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j7, accum_data_v7);
          }
          {
            float* block_ptr = dst_ptr + (mmm * 8 + 0) * dst_stride;
            accum_data_v0 = _mm512_min_ps(accum_data_v0, clamp_max_v);
            accum_data_v0 = _mm512_max_ps(accum_data_v0, clamp_min_v);
            _mm512_mask_storeu_ps(block_ptr + 0 * dst_stride, row_mask,
                                  accum_data_v0);
            accum_data_v1 = _mm512_min_ps(accum_data_v1, clamp_max_v);
            accum_data_v1 = _mm512_max_ps(accum_data_v1, clamp_min_v);
            _mm512_mask_storeu_ps(block_ptr + 1 * dst_stride, row_mask,
                                  accum_data_v1);
            accum_data_v2 = _mm512_min_ps(accum_data_v2, clamp_max_v);
            accum_data_v2 = _mm512_max_ps(accum_data_v2, clamp_min_v);
            _mm512_mask_storeu_ps(block_ptr + 2 * dst_stride, row_mask,
                                  accum_data_v2);
            accum_data_v3 = _mm512_min_ps(accum_data_v3, clamp_max_v);
            accum_data_v3 = _mm512_max_ps(accum_data_v3, clamp_min_v);
            _mm512_mask_storeu_ps(block_ptr + 3 * dst_stride, row_mask,
                                  accum_data_v3);
            accum_data_v4 = _mm512_min_ps(accum_data_v4, clamp_max_v);
            accum_data_v4 = _mm512_max_ps(accum_data_v4, clamp_min_v);
            _mm512_mask_storeu_ps(block_ptr + 4 * dst_stride, row_mask,
                                  accum_data_v4);
            accum_data_v5 = _mm512_min_ps(accum_data_v5, clamp_max_v);
            accum_data_v5 = _mm512_max_ps(accum_data_v5, clamp_min_v);
            _mm512_mask_storeu_ps(block_ptr + 5 * dst_stride, row_mask,
                                  accum_data_v5);
            accum_data_v6 = _mm512_min_ps(accum_data_v6, clamp_max_v);
            accum_data_v6 = _mm512_max_ps(accum_data_v6, clamp_min_v);
            _mm512_mask_storeu_ps(block_ptr + 6 * dst_stride, row_mask,
                                  accum_data_v6);
            accum_data_v7 = _mm512_min_ps(accum_data_v7, clamp_max_v);
            accum_data_v7 = _mm512_max_ps(accum_data_v7, clamp_min_v);
            _mm512_mask_storeu_ps(block_ptr + 7 * dst_stride, row_mask,
                                  accum_data_v7);
          }
        }
      }  // Inner half-block loop.
    }    // Residual rows, main col-block loop.
  }      // End col-block loop.

  if (col < end_col) {
    RUY_DCHECK_GE(end_col - col, 0);
    RUY_DCHECK_LT(end_col - col, 16);

    __m512 accum_data_v[8];

    const float* rhs_col_ptr = adj_rhs_col_ptr + col * rhs_stride;
    float* dst_col_ptr = adj_dst_col_ptr + col * dst_stride;

    for (int row = params.start_row; row < end_row; row += 16) {
      const int residual_rows = std::min(end_row - row, 16);

      const float* lhs_col_ptr = adj_lhs_col_ptr + row * lhs_stride;
      float* dst_ptr = dst_col_ptr + row;
      const float* bias_ptr = bias_col_ptr + row * bias_ptr_block_increment;

      // Initialize with bias.
      const __mmask16 row_mask =
          (static_cast<std::uint32_t>(1) << residual_rows) - 1;
      const __m512 initial_accum_data =
          _mm512_maskz_loadu_ps(row_mask, bias_ptr);

      // Process block in two halves, split by columns.
      for (int mmm = 0; mmm < 2; ++mmm) {
        for (int j = 0; j < 8; ++j) {
          accum_data_v[j] = initial_accum_data;
        }

        const float* lhs_ptr = lhs_col_ptr;
        const float* rhs_ptr = rhs_col_ptr + 8 * mmm;
        for (int d = 0; d < params.depth; ++d) {
          const __m512 lhs_data = _mm512_loadu_ps(lhs_ptr);
          const float* rhs_data = rhs_ptr;

          for (int j = 0; j < 8; ++j) {
            const __m512 dup_rhs_element_j = _mm512_set1_ps(rhs_data[j]);
            accum_data_v[j] =
                _mm512_fmadd_ps(lhs_data, dup_rhs_element_j, accum_data_v[j]);
          }
          lhs_ptr += 16;
          rhs_ptr += 16;
        }

        const int residual_cols = std::min(end_col - col - 8 * mmm, 8);

        if (residual_rows == 16) {
          if (residual_cols == 8) {
            for (int j = 0; j < 8; ++j) {
              float* block_ptr = dst_ptr + (mmm * 8 + j) * dst_stride;
              accum_data_v[j] = _mm512_min_ps(accum_data_v[j], clamp_max_v);
              accum_data_v[j] = _mm512_max_ps(accum_data_v[j], clamp_min_v);
              _mm512_storeu_ps(block_ptr, accum_data_v[j]);
            }
          } else {
            for (int j = 0; j < residual_cols; ++j) {
              float* block_ptr = dst_ptr + (mmm * 8 + j) * dst_stride;
              accum_data_v[j] = _mm512_min_ps(accum_data_v[j], clamp_max_v);
              accum_data_v[j] = _mm512_max_ps(accum_data_v[j], clamp_min_v);
              _mm512_storeu_ps(block_ptr, accum_data_v[j]);
            }
          }
        } else {
          for (int j = 0; j < residual_cols; ++j) {
            float* block_ptr = dst_ptr + (mmm * 8 + j) * dst_stride;
            accum_data_v[j] = _mm512_min_ps(accum_data_v[j], clamp_max_v);
            accum_data_v[j] = _mm512_max_ps(accum_data_v[j], clamp_min_v);
            _mm512_mask_storeu_ps(block_ptr, row_mask, accum_data_v[j]);
          }
        }
      }  // Inner half-block loop.
    }    // End row-block loop.
  }      // Residual cols.
}

void KernelFloatAvx512SingleCol(const KernelParamsFloat<16, 16>& params) {
  gemmlowp::ScopedProfilingLabel label("Kernel kAvx512 float GEMV");

  RUY_DCHECK_EQ(params.dst_cols, 1);
  RUY_DCHECK_EQ(params.last_col, 0);
  RUY_DCHECK_EQ(params.start_col, 0);

  // As parameters are defined, we need to scale by sizeof(float).
  const std::int64_t lhs_stride = params.lhs_stride >> 2;

  int bias_ptr_block_increment = params.flags & RUY_ASM_FLAG_HAS_BIAS ? 1 : 0;
  const int end_row = std::min(params.dst_rows, params.last_row + 16);

  float* adj_dst_col_ptr = params.dst_base_ptr - params.start_row;
  const float* adj_lhs_col_ptr =
      params.lhs_base_ptr - params.start_row * lhs_stride;
  const float* bias_col_ptr = params.bias;

  const __m512 clamp_max_v = _mm512_set1_ps(params.clamp_max);
  const __m512 clamp_min_v = _mm512_set1_ps(params.clamp_min);

  __m512 accum_data_v;

  const float* rhs_col_ptr = params.rhs_base_ptr;
  float* dst_col_ptr = adj_dst_col_ptr;

  int row = params.start_row;
  for (; row <= end_row - 16; row += 16) {
    const float* lhs_col_ptr = adj_lhs_col_ptr + row * lhs_stride;
    float* dst_ptr = dst_col_ptr + row;
    const float* bias_ptr = bias_col_ptr + row * bias_ptr_block_increment;

    // Initialize with bias.
    accum_data_v = _mm512_loadu_ps(bias_ptr);

    const float* lhs_ptr = lhs_col_ptr;
    const float* rhs_ptr = rhs_col_ptr;
    for (int d = 0; d < params.depth; ++d) {
      const __m512 lhs_data = _mm512_loadu_ps(lhs_ptr);
      const float rhs_data = *rhs_ptr;

      const __m512 dup_rhs_element_j = _mm512_set1_ps(rhs_data);
      accum_data_v = _mm512_fmadd_ps(lhs_data, dup_rhs_element_j, accum_data_v);
      lhs_ptr += 16;
      rhs_ptr += 16;
    }

    accum_data_v = _mm512_min_ps(accum_data_v, clamp_max_v);
    accum_data_v = _mm512_max_ps(accum_data_v, clamp_min_v);
    _mm512_storeu_ps(dst_ptr, accum_data_v);
  }  // End row-block loop.

  if (row < end_row) {
    const int residual_rows = end_row - row;
    RUY_CHECK_GE(residual_rows, 1);
    RUY_CHECK_LT(residual_rows, 16);

    const float* lhs_col_ptr = adj_lhs_col_ptr + row * lhs_stride;
    float* dst_ptr = dst_col_ptr + row;
    const float* bias_ptr = bias_col_ptr + row * bias_ptr_block_increment;

    // Initialize with bias.
    const __mmask16 row_mask =
        (static_cast<std::uint32_t>(1) << residual_rows) - 1;
    accum_data_v = _mm512_maskz_loadu_ps(row_mask, bias_ptr);

    const float* lhs_ptr = lhs_col_ptr;
    const float* rhs_ptr = rhs_col_ptr;
    for (int d = 0; d < params.depth; ++d) {
      const __m512 lhs_data = _mm512_loadu_ps(lhs_ptr);
      const float rhs_data = *rhs_ptr;

      const __m512 dup_rhs_element_j = _mm512_set1_ps(rhs_data);
      accum_data_v = _mm512_fmadd_ps(lhs_data, dup_rhs_element_j, accum_data_v);
      lhs_ptr += 16;
      rhs_ptr += 16;
    }

    accum_data_v = _mm512_min_ps(accum_data_v, clamp_max_v);
    accum_data_v = _mm512_max_ps(accum_data_v, clamp_min_v);
    _mm512_mask_storeu_ps(dst_ptr, row_mask, accum_data_v);
  }  // End handling of residual rows.
}

#endif  //  RUY_PLATFORM(AVX512) && RUY_OPT_ENABLED(RUY_OPT_ASM)

}  // namespace ruy
