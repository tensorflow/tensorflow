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

void KernelFloatAvx512(const KernelParamsFloat<16, 16>& params) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

#else  // RUY_PLATFORM(AVX512) && RUY_OPT_ENABLED(RUY_OPT_ASM)

inline std::int32_t mm512_get1_epi32(const __m512i v, int i) {
  __m256i a =
      i < 8 ? _mm512_extracti32x8_epi32(v, 0) : _mm512_extracti32x8_epi32(v, 1);
  switch (i & ~8) {
    case 0:
      return _mm256_extract_epi32(a, 0);
    case 1:
      return _mm256_extract_epi32(a, 1);
    case 2:
      return _mm256_extract_epi32(a, 2);
    case 3:
      return _mm256_extract_epi32(a, 3);
    case 4:
      return _mm256_extract_epi32(a, 4);
    case 5:
      return _mm256_extract_epi32(a, 5);
    case 6:
      return _mm256_extract_epi32(a, 6);
    case 7:
      return _mm256_extract_epi32(a, 7);
    default:
      RUY_DCHECK_LT(i, 16);
      return 0;
  }
}

inline __m512i mm512_set1_epi32(__m512i* v, int i, std::int32_t x) {
  return *v = _mm512_mask_set1_epi32(*v, 1 << i, x);
}

void Kernel8bitAvx512(const KernelParams8bit<16, 16>& params) {
  gemmlowp::ScopedProfilingLabel label("Kernel kAvx512");

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

    for (int row = params.start_row; row <= params.last_row; row += 16) {
      const int residual_rows = std::min(params.dst_rows - row, 16);
      const int residual_cols = std::min(params.dst_cols - col, 16);

      __m512i accum_data_v[16];

      // Initialize with bias.
      const __mmask16 row_mask =
          (static_cast<std::uint32_t>(1) << residual_rows) - 1;
      const __m512i initial_accum_data =
          _mm512_maskz_loadu_epi32(row_mask, bias_ptr);
      bias_ptr += bias_ptr_block_increment;

      for (int j = 0; j < 16; ++j) {
        accum_data_v[j] = initial_accum_data;
      }

      const std::int8_t* lhs_ptr = lhs_col_ptr;
      const std::int8_t* rhs_ptr = rhs_col_ptr;
      for (int d = 0; d < params.depth; d += 4) {
        const __m512i lhs_data = _mm512_loadu_epi8(lhs_ptr);
        __m512i rhs_data = _mm512_loadu_epi8(rhs_ptr);

        // Take bytes 0, 1, 4, 5, 8, 9, ... and expand to 16-bit.
        __m512i lhs_16_bit_low =
            _mm512_cvtepi8_epi16(_mm512_cvtepi32_epi16(lhs_data));
        // Take bytes 2, 3, 6, 7, 10, 11, ... and expand to 16-bit.
        __m512i lhs_16_bit_high = _mm512_cvtepi8_epi16(
            _mm512_cvtepi32_epi16(_mm512_srli_epi32(lhs_data, 16)));

        for (int j = 0; j < 16; ++j) {
          // Mask that drops the 0th element.
          static constexpr std::uint16_t shift_mask = 0xfffe;
          const __m256i dup_rhs_element_low =
              _mm256_broadcastw_epi16(_mm512_castsi512_si128(rhs_data));
          // Shift rhs_data, moving next element into 0 position.
          const __m256i dup_rhs_element_high = _mm256_set1_epi16(
              _mm_extract_epi16(_mm512_castsi512_si128(rhs_data), 1));
          // Shift rhs_data, moving next element into 0 position.
          rhs_data = _mm512_maskz_compress_epi32(shift_mask, rhs_data);

          __m512i rhs_16_bit_dup_low =
              _mm512_cvtepi8_epi16(dup_rhs_element_low);
          __m512i rhs_16_bit_dup_high =
              _mm512_cvtepi8_epi16(dup_rhs_element_high);

          accum_data_v[j] = _mm512_add_epi32(
              accum_data_v[j],
              _mm512_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_data_v[j] = _mm512_add_epi32(
              accum_data_v[j],
              _mm512_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
        }

        lhs_ptr += 16 * 4;
        rhs_ptr += 16 * 4;
      }

      // Move most of this up to bias, or even outside row loop.

      const std::int32_t lhs_zero_point = params.lhs_zero_point;
      const std::int32_t rhs_zero_point = params.rhs_zero_point;
      const std::int32_t prod_zp_depth = params.prod_zp_depth;
      if ((params.flags & RUY_ASM_FLAG_HAS_LHS_SUMS) && rhs_zero_point) {
        const __m512i lhs_sums_offset =
            _mm512_mullo_epi32(_mm512_set1_epi32(rhs_zero_point),
                               _mm512_loadu_epi32(&params.lhs_sums[row]));
        for (int j = 0; j < 16; ++j) {
          accum_data_v[j] = _mm512_sub_epi32(accum_data_v[j], lhs_sums_offset);
        }
      }
      if (((params.flags & RUY_ASM_FLAG_HAS_RHS_SUMS) && lhs_zero_point) ||
          prod_zp_depth) {
        __m512i non_lhs_sums_offset =
            _mm512_mullo_epi32(_mm512_set1_epi32(lhs_zero_point),
                               _mm512_loadu_epi32(&params.rhs_sums[col]));
        non_lhs_sums_offset = _mm512_sub_epi32(
            non_lhs_sums_offset, _mm512_set1_epi32(prod_zp_depth));

        for (int j = 0; j < 16; ++j) {
          accum_data_v[j] = _mm512_sub_epi32(
              accum_data_v[j],
              _mm512_set1_epi32(mm512_get1_epi32(non_lhs_sums_offset, j)));
        }
      }

      //

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
          m_vector =
              _mm512_maskz_loadu_epi32(row_mask, params.multiplier_fixedpoint);
          e_vector =
              _mm512_maskz_loadu_epi32(row_mask, params.multiplier_exponent);
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

        for (int j = 0; j < 16; ++j) {
          accum_data_v[j] = _mm512_sllv_epi32(accum_data_v[j], left_shift);
          // Apply the fixed-point part of the multiplier.
          __m512i scaled_v_low =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(
                                   accum_data_v[j], 0)),
                               m_64bit_low);
          __m512i scaled_v_high =
              _mm512_mul_epi32(_mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(
                                   accum_data_v[j], 1)),
                               m_64bit_high);

          scaled_v_low = _mm512_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm512_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm512_srav_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm512_srav_epi64(scaled_v_high, final_right_shift_high);

          accum_data_v[j] =
              _mm512_castsi256_si512(_mm512_cvtepi64_epi32(scaled_v_low));
          accum_data_v[j] = _mm512_inserti32x8(
              accum_data_v[j], _mm512_cvtepi64_epi32(scaled_v_high), 1);

#if !RUY_OPT_ENABLED(RUY_OPT_NATIVE_ROUNDING)
          RUY_DCHECK(false);
#endif
        }

        if (params.dst_zero_point) {
          __m512i dst_zero_point = _mm512_set1_epi32(params.dst_zero_point);
          for (int j = 0; j < 16; ++j) {
            accum_data_v[j] = _mm512_add_epi32(accum_data_v[j], dst_zero_point);
          }
        }
        __m512i clamp_max_v = _mm512_set1_epi32(params.clamp_max);
        __m512i clamp_min_v = _mm512_set1_epi32(params.clamp_min);
        for (int j = 0; j < 16; ++j) {
          accum_data_v[j] = _mm512_min_epi32(accum_data_v[j], clamp_max_v);
          accum_data_v[j] = _mm512_max_epi32(accum_data_v[j], clamp_min_v);
        }
      }
      const bool store_full_block =
          (residual_rows == 16) && (residual_cols == 16);

      if (params.dst_type_id == DstTypeId<std::int8_t>::kValue) {
        std::int8_t* tmp_ptr = static_cast<std::int8_t*>(dst_ptr);
        const int block_col_offset = dst_stride;
        if (store_full_block) {
          for (int j = 0; j < 16; ++j) {
            _mm_storeu_epi8(tmp_ptr, _mm512_cvtepi32_epi8(accum_data_v[j]));
            tmp_ptr += block_col_offset;
          }
        } else {
          for (int j = 0; j < residual_cols; ++j) {
            _mm_mask_storeu_epi8(tmp_ptr, row_mask,
                                 _mm512_cvtepi32_epi8(accum_data_v[j]));
            tmp_ptr += block_col_offset;
          }
        }
        dst_ptr = static_cast<void*>(static_cast<std::int8_t*>(dst_ptr) + 16);
      } else if (params.dst_type_id == DstTypeId<std::uint8_t>::kValue) {
        std::uint8_t* tmp_ptr = static_cast<std::uint8_t*>(dst_ptr);
        const int block_col_offset = dst_stride;
        if (store_full_block) {
          for (int j = 0; j < 16; ++j) {
            _mm_storeu_epi8(tmp_ptr, _mm512_cvtepi32_epi8(accum_data_v[j]));
            tmp_ptr += block_col_offset;
          }
        } else {
          for (int j = 0; j < residual_cols; ++j) {
            _mm_mask_storeu_epi8(tmp_ptr, row_mask,
                                 _mm512_cvtepi32_epi8(accum_data_v[j]));
            tmp_ptr += block_col_offset;
          }
        }
        dst_ptr = static_cast<void*>(static_cast<std::uint8_t*>(dst_ptr) + 16);
      } else if (params.dst_type_id == DstTypeId<std::int16_t>::kValue) {
        std::int16_t* tmp_ptr = static_cast<std::int16_t*>(dst_ptr);
        const int block_col_offset = dst_stride;
        if (store_full_block) {
          for (int j = 0; j < 16; ++j) {
            _mm256_storeu_epi16(tmp_ptr,
                                _mm512_cvtepi32_epi16(accum_data_v[j]));
            tmp_ptr += block_col_offset;
          }
        } else {
          for (int j = 0; j < residual_cols; ++j) {
            _mm256_mask_storeu_epi16(tmp_ptr, row_mask,
                                     _mm512_cvtepi32_epi16(accum_data_v[j]));
            tmp_ptr += block_col_offset;
          }
        }
        dst_ptr = static_cast<void*>(static_cast<std::int16_t*>(dst_ptr) + 16);
      } else if (params.dst_type_id == DstTypeId<std::int32_t>::kValue) {
        if (store_full_block) {
          std::int32_t* tmp_ptr = static_cast<std::int32_t*>(dst_ptr);
          const int block_col_offset = dst_stride;
          for (int j = 0; j < 16; ++j) {
            _mm512_storeu_epi32(tmp_ptr, accum_data_v[j]);
            tmp_ptr += block_col_offset;
          }
        } else {
          std::int32_t* dst_block_ptr = static_cast<std::int32_t*>(dst_ptr);
          for (int j = 0; j < residual_cols; ++j) {
            _mm512_mask_storeu_epi32(dst_block_ptr, row_mask, accum_data_v[j]);
            dst_block_ptr += dst_stride;
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
}

void KernelFloatAvx512(const KernelParamsFloat<16, 16>& params) {
  gemmlowp::ScopedProfilingLabel label("Kernel kAvx512");

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
          const __m256 rhs_data = _mm256_loadu_ps(rhs_ptr);
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
          const __m256 rhs_data = _mm256_loadu_ps(rhs_ptr);
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
          const __m256 rhs_data = _mm256_loadu_ps(rhs_ptr);
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
          const __m256 rhs_data = _mm256_loadu_ps(rhs_ptr);
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
          const __m256 rhs_data = _mm256_loadu_ps(rhs_ptr);
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
          const __m256 rhs_data = _mm256_loadu_ps(rhs_ptr);
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
          const __m256 rhs_data = _mm256_loadu_ps(rhs_ptr);

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

#endif  //  RUY_PLATFORM(AVX512) && RUY_OPT_ENABLED(RUY_OPT_ASM)

}  // namespace ruy
