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

#if RUY_PLATFORM(AVX2) && RUY_OPT_ENABLED(RUY_OPT_ASM)
#include <immintrin.h>  // IWYU pragma: keep
#endif

namespace ruy {

#if !(RUY_PLATFORM(AVX2) && RUY_OPT_ENABLED(RUY_OPT_ASM))

void Kernel8bitAvx2(const KernelParams8bit<8, 8>& params) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

void Kernel8bitAvx2SingleCol(const KernelParams8bit<8, 8>& params) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

void KernelFloatAvx2(const KernelParamsFloat<8, 8>& params) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

void KernelFloatAvx2SingleCol(const KernelParamsFloat<8, 8>& params) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

#else  // RUY_PLATFORM(AVX2) && RUY_OPT_ENABLED(RUY_OPT_ASM)

static constexpr int kAvx8bitBlockSize = 8;
static constexpr int kAvx8bitInnerSize = 4;

namespace {
namespace intrin_utils {

inline __m256 mm256_n_loadu_epi32(int n, const std::int32_t* src) {
  switch (n) {
    case 0:
      return _mm256_setzero_si256();
    case 1:
      return _mm256_setr_m128(_mm_setr_epi32(src[0], 0, 0, 0),
                              _mm_setzero_si128());
    case 2:
      return _mm256_setr_m128(_mm_setr_epi32(src[0], src[1], 0, 0),
                              _mm_setzero_si128());
    case 3:
      return _mm256_setr_m128(_mm_setr_epi32(src[0], src[1], src[2], 0),
                              _mm_setzero_si128());
    case 4:
      return _mm256_castsi128_si256(
          _mm_loadu_si128(reinterpret_cast<__m128i const*>(src)));
    case 5:
      return _mm256_setr_epi32(src[0], src[1], src[2], src[3], src[4], 0, 0, 0);
    case 6:
      return _mm256_setr_epi32(src[0], src[1], src[2], src[3], src[4], src[5],
                               0, 0);
    case 7:
      return _mm256_setr_epi32(src[0], src[1], src[2], src[3], src[4], src[5],
                               src[6], 0);
    case 8:
      return _mm256_loadu_si256(reinterpret_cast<__m256i const*>(src));
    default:
      RUY_DCHECK_LT(n, 9);
      return _mm256_setzero_si256();
  }
}

inline void mm256_n_storeu_cvtepi32_epi8(std::uint8_t* dst, int residual_rows,
                                         const __m256 v) {
  // Select bytes 0, 4, 8, 12 within each lane, effectively truncating.
  const __m256i repack_perm = _mm256_set1_epi32(0x0c080400);
  __m256i shuffled_v;
  if (residual_rows > 1) {
    // This selects 0, 4, 8, 12, 0, 4, 8, 12, ..., but we only use the first 4
    // in each 128-bit lane.
    shuffled_v = _mm256_shuffle_epi8(v, repack_perm);
  }
  switch (residual_rows) {
    case 0:
      break;
    case 1:
      dst[0] = _mm256_extract_epi8(v, 0);
      break;
    case 2:
      _mm_storeu_si16(dst, _mm256_extracti128_si256(shuffled_v, 0));
      break;
    case 3: {
      __m128i trailing_packed = _mm256_extracti128_si256(shuffled_v, 0);
      _mm_storeu_si16(dst, trailing_packed);
      dst[2] = _mm_extract_epi8(trailing_packed, 2);
      break;
    }
    case 4:
      _mm_storeu_si32(dst, _mm256_extracti128_si256(shuffled_v, 0));
      break;
    case 5:
      _mm_storeu_si32(dst, _mm256_extracti128_si256(shuffled_v, 0));
      dst[4] = _mm256_extract_epi8(shuffled_v, 16);
      break;
    case 6:
      _mm_storeu_si32(dst, _mm256_extracti128_si256(shuffled_v, 0));
      _mm_storeu_si16(dst + 4, _mm256_extracti128_si256(shuffled_v, 1));
      break;
    case 7: {
      _mm_storeu_si32(dst, _mm256_extracti128_si256(shuffled_v, 0));
      __m128i trailing_packed = _mm256_extracti128_si256(shuffled_v, 1);
      _mm_storeu_si16(dst + 4, trailing_packed);
      dst[6] = _mm_extract_epi8(trailing_packed, 2);
      break;
    }
    case 8:
      _mm_storeu_si32(dst, _mm256_extracti128_si256(shuffled_v, 0));
      _mm_storeu_si32(dst + 4, _mm256_extracti128_si256(shuffled_v, 1));
      break;
    default:
      RUY_DCHECK_LE(residual_rows, 8);
      break;
  }
}

inline void mm256_storeu_cvtepi32_epi8(std::uint8_t* dst, const __m256 v) {
  // Select bytes 0, 4, 8, 12 within each lane, effectively truncating.
  const __m256i repack_perm = _mm256_set1_epi32(0x0c080400);
  const __m256i shuffled_v = _mm256_shuffle_epi8(v, repack_perm);
  _mm_storeu_si32(dst, _mm256_extracti128_si256(shuffled_v, 0));
  _mm_storeu_si32(dst + 4, _mm256_extracti128_si256(shuffled_v, 1));
}

inline void mm256_n_storeu_cvtepi32_epi8(std::int8_t* dst, int residual_rows,
                                         const __m256 v) {
  intrin_utils::mm256_n_storeu_cvtepi32_epi8(
      reinterpret_cast<std::uint8_t*>(dst), residual_rows, v);
}

inline void mm256_storeu_cvtepi32_epi8(std::int8_t* dst, const __m256 v) {
  // Select bytes 0, 4, 8, 12 within each lane, effectively truncating.
  const __m256i repack_perm = _mm256_set1_epi32(0x0c080400);
  const __m256i shuffled_v = _mm256_shuffle_epi8(v, repack_perm);
  _mm_storeu_si32(dst, _mm256_extracti128_si256(shuffled_v, 0));
  _mm_storeu_si32(dst + 4, _mm256_extracti128_si256(shuffled_v, 1));
}

inline void mm256_n_storeu_cvtepi32_epi16(std::int16_t* dst, int residual_rows,
                                          const __m256 v) {
  // Select bytes 0, 1, 4, 5, 8, 9, 12, 13 within each lane, effectively
  // truncating each 16-bit integer.
  const __m256i repack_perm = _mm256_set1_epi64x(0x0d0c090805040100);
  __m256i shuffled_v;
  __m128i shuffled_v_low;
  if (residual_rows > 1) {
    shuffled_v = _mm256_shuffle_epi8(v, repack_perm);
    shuffled_v_low = _mm256_extracti128_si256(shuffled_v, 0);
  } else {
    shuffled_v_low = _mm256_extracti128_si256(v, 0);
  }
  switch (residual_rows) {
    case 0:
      break;
    case 1:
      _mm_storeu_si16(dst, shuffled_v_low);
      break;
    case 2:
      _mm_storeu_si32(dst, shuffled_v_low);
      break;
    case 3: {
      _mm_storeu_si32(dst, shuffled_v_low);
      dst[2] = _mm_extract_epi16(shuffled_v_low, 2);
      break;
    }
    case 4:
      _mm_storeu_si64(dst, shuffled_v_low);
      break;
    case 5:
      _mm_storeu_si64(dst, shuffled_v_low);
      dst[4] = _mm256_extract_epi16(shuffled_v, 8);
      break;
    case 6:
      _mm_storeu_si64(dst, shuffled_v_low);
      _mm_storeu_si32(dst + 4, _mm256_extracti128_si256(shuffled_v, 1));
      break;
    case 7: {
      _mm_storeu_si64(dst, shuffled_v_low);
      __m128i trailing_packed = _mm256_extracti128_si256(shuffled_v, 1);
      _mm_storeu_si32(dst + 4, trailing_packed);
      dst[6] = _mm_extract_epi16(trailing_packed, 2);
      break;
    }
    case 8:
      _mm_storeu_si64(dst, _mm256_extracti128_si256(shuffled_v, 0));
      _mm_storeu_si64(dst + 4, _mm256_extracti128_si256(shuffled_v, 1));
      break;
    default:
      RUY_DCHECK_LE(residual_rows, 8);
      break;
  }
}

inline void mm256_storeu_cvtepi32_epi16(std::int16_t* dst, const __m256 v) {
  // Select bytes 0, 1, 4, 5, 8, 9, 12, 13 within each lane, effectively
  // truncating each 16-bit integer.
  const __m256i repack_perm = _mm256_set1_epi64x(0x0d0c090805040100);
  const __m256i shuffled_v = _mm256_shuffle_epi8(v, repack_perm);
  _mm_storeu_si64(dst, _mm256_extracti128_si256(shuffled_v, 0));
  _mm_storeu_si64(dst + 4, _mm256_extracti128_si256(shuffled_v, 1));
}

inline void mm256_n_storeu_epi32(std::int32_t* dst, int residual_rows,
                                 const __m256 v) {
  const __m128i v_low = _mm256_extracti128_si256(v, 0);
  switch (residual_rows) {
    case 0:
      break;
    case 1:
      _mm_storeu_si32(dst, v_low);
      break;
    case 2:
      _mm_storeu_si64(dst, v_low);
      break;
    case 3: {
      __m128i trailing_packed = v_low;
      _mm_storeu_si64(dst, trailing_packed);
      dst[2] = _mm_extract_epi32(trailing_packed, 2);
      break;
    }
    case 4:
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), v_low);
      break;
    case 5:
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), v_low);
      dst[4] = _mm256_extract_epi32(v, 4);
      break;
    case 6:
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), v_low);
      _mm_storeu_si64(dst + 4, _mm256_extracti128_si256(v, 1));
      break;
    case 7: {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), v_low);
      __m128i trailing_packed = _mm256_extracti128_si256(v, 1);
      _mm_storeu_si64(dst + 4, trailing_packed);
      dst[6] = _mm_extract_epi32(trailing_packed, 2);
      break;
    }
    case 8:
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), v);
      break;
    default:
      RUY_DCHECK_LE(residual_rows, 8);
      break;
  }
}

inline void mm256_storeu_epi32(std::int32_t* dst, const __m256 v) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), v);
}

inline float mm256_get1_ps(const __m256 a, int i) {
  __m256i ai = _mm256_castps_si256(a);
  int float_val_as_int;
  switch (i) {
    case 0:
      float_val_as_int = _mm256_extract_epi32(ai, 0);
      break;
    case 1:
      float_val_as_int = _mm256_extract_epi32(ai, 1);
      break;
    case 2:
      float_val_as_int = _mm256_extract_epi32(ai, 2);
      break;
    case 3:
      float_val_as_int = _mm256_extract_epi32(ai, 3);
      break;
    case 4:
      float_val_as_int = _mm256_extract_epi32(ai, 4);
      break;
    case 5:
      float_val_as_int = _mm256_extract_epi32(ai, 5);
      break;
    case 6:
      float_val_as_int = _mm256_extract_epi32(ai, 6);
      break;
    case 7:
      float_val_as_int = _mm256_extract_epi32(ai, 7);
      break;
    default:
      RUY_DCHECK_LT(i, 8);
      return .0f;
  }
  return reinterpret_cast<float&>(float_val_as_int);
}

inline __m256 mm256_n_loadu_ps(int i, const float* src) {
  switch (i) {
    case 0:
      return _mm256_setzero_ps();
    case 1:
      return _mm256_setr_m128(_mm_setr_ps(src[0], .0f, .0f, .0f),
                              _mm_setzero_ps());
    case 2:
      return _mm256_setr_m128(_mm_setr_ps(src[0], src[1], .0f, .0f),
                              _mm_setzero_ps());
    case 3:
      return _mm256_setr_m128(_mm_setr_ps(src[0], src[1], src[2], .0f),
                              _mm_setzero_ps());
    case 4:
      return _mm256_setr_m128(_mm_setr_ps(src[0], src[1], src[2], src[3]),
                              _mm_setzero_ps());
    case 5:
      return _mm256_setr_ps(src[0], src[1], src[2], src[3], src[4], .0f, .0f,
                            .0f);
    case 6:
      return _mm256_setr_ps(src[0], src[1], src[2], src[3], src[4], src[5], .0f,
                            .0f);
    case 7:
      return _mm256_setr_ps(src[0], src[1], src[2], src[3], src[4], src[5],
                            src[6], .0f);
    case 8:
      return _mm256_loadu_ps(src);
    default:
      RUY_DCHECK_LT(i, 9);
      return _mm256_setzero_ps();
  }
}

inline void mm256_n_storeu_ps(float* dst, int residual_rows, const __m256 v) {
  for (int i = 0; i < residual_rows; ++i) {
    dst[i] = intrin_utils::mm256_get1_ps(v, i);
  }
}
}  // namespace intrin_utils
}  // namespace

void Kernel8bitAvx2(const KernelParams8bit<8, 8>& params) {
  gemmlowp::ScopedProfilingLabel label("Kernel kAvx2 8-bit");

  const std::int8_t splitter_idx_data[32] = {
      0, 1, 4, 5, 8,  9,  12, 13,  //
      2, 3, 6, 7, 10, 11, 14, 15,  //
      0, 1, 4, 5, 8,  9,  12, 13,  //
      2, 3, 6, 7, 10, 11, 14, 15   //
  };

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

  int bias_ptr_block_increment =
      params.flags & RUY_ASM_FLAG_HAS_BIAS ? kAvx8bitBlockSize : 0;

  const std::int8_t* rhs_col_ptr = params.rhs_base_ptr;
  void* dst_col_ptr = params.dst_base_ptr;
  const std::int32_t* bias_col_ptr = params.bias;
  if (params.flags & RUY_ASM_FLAG_HAS_BIAS) {
    bias_col_ptr += params.start_row;
  }

  for (int col = params.start_col; col <= params.last_col;
       col += kAvx8bitBlockSize) {
    const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
    void* dst_ptr = dst_col_ptr;
    const std::int32_t* bias_ptr = bias_col_ptr;

    const std::int32_t lhs_zero_point = params.lhs_zero_point;
    const bool has_rhs_sums_offsets =
        (params.flags & RUY_ASM_FLAG_HAS_RHS_SUMS) && lhs_zero_point;
    std::int32_t rhs_sums_offsets[8];
    if (has_rhs_sums_offsets) {
      const __m256i rhs_sums_offset_v = _mm256_mullo_epi32(
          _mm256_set1_epi32(lhs_zero_point),
          _mm256_loadu_si256(
              reinterpret_cast<__m256i const*>(&params.rhs_sums[col])));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(rhs_sums_offsets),
                          rhs_sums_offset_v);
    }

    for (int row = params.start_row; row <= params.last_row;
         row += kAvx8bitBlockSize) {
      const int residual_rows =
          std::min(params.dst_rows - row, kAvx8bitBlockSize);
      const int residual_cols =
          std::min(params.dst_cols - col, kAvx8bitBlockSize);

      const __m256i splitter_idx = _mm256_loadu_si256(
          reinterpret_cast<__m256i const*>(splitter_idx_data));

      __m256i accum_data_v0;
      __m256i accum_data_v1;
      __m256i accum_data_v2;
      __m256i accum_data_v3;
      __m256i accum_data_v4;
      __m256i accum_data_v5;
      __m256i accum_data_v6;
      __m256i accum_data_v7;

      // Initialize with bias.
      __m256i initial_accum_data =
          intrin_utils::mm256_n_loadu_epi32(residual_rows, bias_ptr);
      bias_ptr += bias_ptr_block_increment;

      // Adjustments common across columns.
      const std::int32_t rhs_zero_point = params.rhs_zero_point;
      if ((params.flags & RUY_ASM_FLAG_HAS_LHS_SUMS) && rhs_zero_point) {
        const __m256i lhs_sums_offset = _mm256_mullo_epi32(
            _mm256_set1_epi32(rhs_zero_point),
            _mm256_loadu_si256(
                reinterpret_cast<__m256i const*>(&params.lhs_sums[row])));
        initial_accum_data =
            _mm256_sub_epi32(initial_accum_data, lhs_sums_offset);
      }
      const std::int32_t prod_zp_depth = params.prod_zp_depth;
      if (prod_zp_depth) {
        initial_accum_data = _mm256_add_epi32(initial_accum_data,
                                              _mm256_set1_epi32(prod_zp_depth));
      }

      // Adjustments differing across columns.
      if (has_rhs_sums_offsets) {
        accum_data_v0 = _mm256_sub_epi32(
            initial_accum_data, _mm256_set1_epi32(rhs_sums_offsets[0]));
        accum_data_v1 = _mm256_sub_epi32(
            initial_accum_data, _mm256_set1_epi32(rhs_sums_offsets[1]));
        accum_data_v2 = _mm256_sub_epi32(
            initial_accum_data, _mm256_set1_epi32(rhs_sums_offsets[2]));
        accum_data_v3 = _mm256_sub_epi32(
            initial_accum_data, _mm256_set1_epi32(rhs_sums_offsets[3]));
        accum_data_v4 = _mm256_sub_epi32(
            initial_accum_data, _mm256_set1_epi32(rhs_sums_offsets[4]));
        accum_data_v5 = _mm256_sub_epi32(
            initial_accum_data, _mm256_set1_epi32(rhs_sums_offsets[5]));
        accum_data_v6 = _mm256_sub_epi32(
            initial_accum_data, _mm256_set1_epi32(rhs_sums_offsets[6]));
        accum_data_v7 = _mm256_sub_epi32(
            initial_accum_data, _mm256_set1_epi32(rhs_sums_offsets[7]));
      } else {
        accum_data_v0 = initial_accum_data;
        accum_data_v1 = initial_accum_data;
        accum_data_v2 = initial_accum_data;
        accum_data_v3 = initial_accum_data;
        accum_data_v4 = initial_accum_data;
        accum_data_v5 = initial_accum_data;
        accum_data_v6 = initial_accum_data;
        accum_data_v7 = initial_accum_data;
      }

      const std::int8_t* lhs_ptr = lhs_col_ptr;
      const std::int8_t* rhs_ptr = rhs_col_ptr;
      for (int d = 0; d < params.depth; d += kAvx8bitInnerSize) {
        const __m256i lhs_data =
            _mm256_load_si256(reinterpret_cast<const __m256i*>(lhs_ptr));
        const __m256i rhs_data_8bit =
            _mm256_load_si256(reinterpret_cast<const __m256i*>(rhs_ptr));

        // Each "int32" is two 16-bit RHS values, sign extended from 8-bit.
        std::int32_t rhs_data[16];
        const __m128i rhs_data_bottom_lane =
            _mm256_castsi256_si128(rhs_data_8bit);
        const __m128i rhs_data_top_lane =
            _mm256_extracti128_si256(rhs_data_8bit, 1);
        const __m256i rhs_16_bit_dup_low =
            _mm256_cvtepi8_epi16(rhs_data_bottom_lane);
        const __m256i rhs_16_bit_dup_high =
            _mm256_cvtepi8_epi16(rhs_data_top_lane);
        // Now that we have cast the RHS data, we store it so that each value
        // can be separately loaded in the accumulation loop.
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(rhs_data),
                            rhs_16_bit_dup_low);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(rhs_data + 8),
                            rhs_16_bit_dup_high);

        const __m256i lhs_data_split =
            _mm256_shuffle_epi8(lhs_data, splitter_idx);
        const __m256i lhs_data_split_expand_bottom =
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(lhs_data_split, 0));
        const __m256i lhs_data_split_expand_top =
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(lhs_data_split, 1));

        // Take bytes 0, 1, 4, 5, 8, 9, ... expanded to 16-bit.
        const __m256i lhs_16_bit_low = _mm256_permute2x128_si256(
            lhs_data_split_expand_bottom, lhs_data_split_expand_top, 0x20);
        // Take bytes 2, 3, 6, 7, 10, 11, ... expanded to 16-bit.
        const __m256i lhs_16_bit_high = _mm256_permute2x128_si256(
            lhs_data_split_expand_bottom, lhs_data_split_expand_top, 0x31);
        // Accumulate for column 0.
        {
          const std::int32_t low_rhs_value = rhs_data[0];
          const std::int32_t high_rhs_value = rhs_data[1];

          const __m256i rhs_16_bit_dup_low = _mm256_set1_epi32(low_rhs_value);
          const __m256i rhs_16_bit_dup_high = _mm256_set1_epi32(high_rhs_value);

          accum_data_v0 = _mm256_add_epi32(
              accum_data_v0,
              _mm256_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_data_v0 = _mm256_add_epi32(
              accum_data_v0,
              _mm256_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
        }
        // Accumulate for column 1.
        {
          const std::int32_t low_rhs_value = rhs_data[2];
          const std::int32_t high_rhs_value = rhs_data[3];

          const __m256i rhs_16_bit_dup_low = _mm256_set1_epi32(low_rhs_value);
          const __m256i rhs_16_bit_dup_high = _mm256_set1_epi32(high_rhs_value);

          accum_data_v1 = _mm256_add_epi32(
              accum_data_v1,
              _mm256_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_data_v1 = _mm256_add_epi32(
              accum_data_v1,
              _mm256_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
        }
        // Accumulate for column 2.
        {
          const std::int32_t low_rhs_value = rhs_data[4];
          const std::int32_t high_rhs_value = rhs_data[5];

          const __m256i rhs_16_bit_dup_low = _mm256_set1_epi32(low_rhs_value);
          const __m256i rhs_16_bit_dup_high = _mm256_set1_epi32(high_rhs_value);

          accum_data_v2 = _mm256_add_epi32(
              accum_data_v2,
              _mm256_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_data_v2 = _mm256_add_epi32(
              accum_data_v2,
              _mm256_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
        }
        // Accumulate for column 3.
        {
          const std::int32_t low_rhs_value = rhs_data[6];
          const std::int32_t high_rhs_value = rhs_data[7];

          const __m256i rhs_16_bit_dup_low = _mm256_set1_epi32(low_rhs_value);
          const __m256i rhs_16_bit_dup_high = _mm256_set1_epi32(high_rhs_value);

          accum_data_v3 = _mm256_add_epi32(
              accum_data_v3,
              _mm256_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_data_v3 = _mm256_add_epi32(
              accum_data_v3,
              _mm256_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
        }
        // Accumulate for column 4.
        {
          const std::int32_t low_rhs_value = rhs_data[8];
          const std::int32_t high_rhs_value = rhs_data[9];

          const __m256i rhs_16_bit_dup_low = _mm256_set1_epi32(low_rhs_value);
          const __m256i rhs_16_bit_dup_high = _mm256_set1_epi32(high_rhs_value);

          accum_data_v4 = _mm256_add_epi32(
              accum_data_v4,
              _mm256_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_data_v4 = _mm256_add_epi32(
              accum_data_v4,
              _mm256_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
        }
        // Accumulate for column 5.
        {
          const std::int32_t low_rhs_value = rhs_data[10];
          const std::int32_t high_rhs_value = rhs_data[11];

          const __m256i rhs_16_bit_dup_low = _mm256_set1_epi32(low_rhs_value);
          const __m256i rhs_16_bit_dup_high = _mm256_set1_epi32(high_rhs_value);

          accum_data_v5 = _mm256_add_epi32(
              accum_data_v5,
              _mm256_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_data_v5 = _mm256_add_epi32(
              accum_data_v5,
              _mm256_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
        }
        // Accumulate for column 6.
        {
          const std::int32_t low_rhs_value = rhs_data[12];
          const std::int32_t high_rhs_value = rhs_data[13];

          const __m256i rhs_16_bit_dup_low = _mm256_set1_epi32(low_rhs_value);
          const __m256i rhs_16_bit_dup_high = _mm256_set1_epi32(high_rhs_value);

          accum_data_v6 = _mm256_add_epi32(
              accum_data_v6,
              _mm256_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_data_v6 = _mm256_add_epi32(
              accum_data_v6,
              _mm256_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
        }
        // Accumulate for column 7.
        {
          const std::int32_t low_rhs_value = rhs_data[14];
          const std::int32_t high_rhs_value = rhs_data[15];

          const __m256i rhs_16_bit_dup_low = _mm256_set1_epi32(low_rhs_value);
          const __m256i rhs_16_bit_dup_high = _mm256_set1_epi32(high_rhs_value);

          accum_data_v7 = _mm256_add_epi32(
              accum_data_v7,
              _mm256_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
          accum_data_v7 = _mm256_add_epi32(
              accum_data_v7,
              _mm256_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));
        }

        lhs_ptr += kAvx8bitBlockSize * kAvx8bitInnerSize;
        rhs_ptr += kAvx8bitBlockSize * kAvx8bitInnerSize;
      }

      if (params.dst_type_id != DstTypeId<std::int32_t>::kValue) {
        __m256i m_vector;
        __m256i e_vector;
        // Does not make use of RUY_ASM_FLAG_NEEDS_LEFT_SHIFT.
        if (params.flags & RUY_ASM_FLAG_HAS_PERCHANNEL) {
          m_vector = intrin_utils::mm256_n_loadu_epi32(
              residual_rows, &params.multiplier_fixedpoint[row]);
          e_vector = intrin_utils::mm256_n_loadu_epi32(
              residual_rows, &params.multiplier_exponent[row]);
        } else {
          // These arrays have size LhsCols, and are pre-filled.
          m_vector = _mm256_set1_epi32(params.multiplier_fixedpoint[0]);
          e_vector = _mm256_set1_epi32(params.multiplier_exponent[0]);
        }

        const __m256i m_64bit_low =
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256(m_vector, 0));
        const __m256i m_64bit_high =
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256(m_vector, 1));

        const __m256i zero_vector = _mm256_setzero_si256();
        const __m256i left_shift = _mm256_max_epi32(e_vector, zero_vector);
        const __m256i neg_e_vector = _mm256_sub_epi32(zero_vector, e_vector);
        const __m256i right_shift = _mm256_max_epi32(neg_e_vector, zero_vector);
        const __m256i final_right_shift =
            _mm256_add_epi32(right_shift, _mm256_set1_epi32(31));
        const __m256i final_right_shift_low = _mm256_cvtepi32_epi64(
            _mm256_extracti128_si256(final_right_shift, 0));
        const __m256i final_right_shift_high = _mm256_cvtepi32_epi64(
            _mm256_extracti128_si256(final_right_shift, 1));
        // Really we want 0x100000000, but use half to avoid overflowing.
        const __m256i convert_to_signed_halved =
            _mm256_srlv_epi32(_mm256_set1_epi32(0x80000000), right_shift);
        const __m256i convert_to_unsigned_64 =
            _mm256_set1_epi64x(0x8000000000000000);

        __m256i post_scaling_offset = _mm256_add_epi32(
            convert_to_signed_halved, convert_to_signed_halved);

        const __m256i offset_vector =
            _mm256_slli_epi64(_mm256_set1_epi64x(1), 30);
        // Really these should be shifted by neg_e_vector, but tests pass when
        // using right_shift.
        const __m256i offset_vector_low = _mm256_add_epi64(
            _mm256_sllv_epi64(offset_vector,
                              _mm256_cvtepi32_epi64(
                                  _mm256_extracti128_si256(right_shift, 0))),
            convert_to_unsigned_64);
        const __m256i offset_vector_high = _mm256_add_epi64(
            _mm256_sllv_epi64(offset_vector,
                              _mm256_cvtepi32_epi64(
                                  _mm256_extracti128_si256(right_shift, 1))),
            convert_to_unsigned_64);

        if (params.dst_zero_point) {
          const __m256i dst_zero_point =
              _mm256_set1_epi32(params.dst_zero_point);
          // The post-scaling offset is subtracted later, so this has the effect
          // of adding the zero point.
          post_scaling_offset =
              _mm256_sub_epi32(post_scaling_offset, dst_zero_point);
        }

#if !RUY_OPT_ENABLED(RUY_OPT_NATIVE_ROUNDING)
        RUY_DCHECK(false);
#endif
        const __m256i repack_perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

        // We cannot do
        //
        // scaled_v_low =
        //     _mm256_srav_epi64(scaled_v_low, final_right_shift_low);
        // scaled_v_high =
        //     _mm256_srav_epi64(scaled_v_high, final_right_shift_high);
        //
        // since this instruction is not in AVX2. Instead we use
        // _mm256_srlv_epi64, but this is an unsigned shift, so we applied
        // offsets before (convert_to_unsigned_64) and after
        // (convert_to_signed_halved).
        //
        // The overall process is, for 64-bit scaled accumulator:
        // unsigned_accum = signed_accum + 1 << 63;
        // unsigned_accum = (unsigned_accum >> right_shift) >> 31;
        // signed_accum = unsigned_accum - ((1 << 32) >> right_shift) / 2 * 2;

        // There are various ways to repack the results, in the absence of
        // _mm256_cvtepi64_epi32() or anything like it.
        // A.
        // accum_data_v[j] =
        //     _mm256_set_epi32(_mm256_extract_epi32(scaled_v_high, 6),
        //                      _mm256_extract_epi32(scaled_v_high, 4),
        //                      _mm256_extract_epi32(scaled_v_high, 2),
        //                      _mm256_extract_epi32(scaled_v_high, 0),
        //                      _mm256_extract_epi32(scaled_v_low, 6),
        //                      _mm256_extract_epi32(scaled_v_low, 4),
        //                      _mm256_extract_epi32(scaled_v_low, 2),
        //                      _mm256_extract_epi32(scaled_v_low, 0));
        // B.
        // scaled_v_low = _mm256_shuffle_epi32(scaled_v_low, 0xd8);
        // scaled_v_high = _mm256_shuffle_epi32(scaled_v_high, 0xd8);
        // accum_data_v[j] =
        //     _mm256_set_epi64x(_mm256_extract_epi64(scaled_v_high, 2),
        //                       _mm256_extract_epi64(scaled_v_high, 0),
        //                       _mm256_extract_epi64(scaled_v_low, 2),
        //                       _mm256_extract_epi64(scaled_v_low, 0));
        // C.
        // scaled_v_low =
        //     _mm256_permutevar8x32_epi32(scaled_v_low, repack_perm);
        // scaled_v_high =
        //     _mm256_permutevar8x32_epi32(scaled_v_high, repack_perm);
        // accum_data_v[j] =
        //     _mm256_permute2x128_si256(scaled_v_low, scaled_v_high, 0x20);
        //
        // However, we choose the following because it uses two lighter
        // instructions. The permutation does have a longer latency, but this
        // loop can be unrolled.
        // D.
        // scaled_v_high = _mm256_slli_epi64(scaled_v_high, 32);
        // __m256i results =
        //     _mm256_blend_epi32(scaled_v_low, scaled_v_high, 0xaa);
        // results = _mm256_permutevar8x32_epi32(results, repack_perm);
        // accum_data_v[j] = _mm256_sub_epi32(results, post_scaling_offset);
        {
          __m256i shifted_accum = _mm256_sllv_epi32(accum_data_v0, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m256i scaled_v_low = _mm256_mul_epi32(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 0)),
              m_64bit_low);
          __m256i scaled_v_high = _mm256_mul_epi32(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 1)),
              m_64bit_high);

          scaled_v_low = _mm256_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm256_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm256_srlv_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm256_srlv_epi64(scaled_v_high, final_right_shift_high);

          scaled_v_high = _mm256_slli_epi64(scaled_v_high, 32);
          __m256i results =
              _mm256_blend_epi32(scaled_v_low, scaled_v_high, 0xaa);
          results = _mm256_permutevar8x32_epi32(results, repack_perm);

          accum_data_v0 = _mm256_sub_epi32(results, post_scaling_offset);
        }
        {
          __m256i shifted_accum = _mm256_sllv_epi32(accum_data_v1, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m256i scaled_v_low = _mm256_mul_epi32(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 0)),
              m_64bit_low);
          __m256i scaled_v_high = _mm256_mul_epi32(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 1)),
              m_64bit_high);

          scaled_v_low = _mm256_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm256_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm256_srlv_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm256_srlv_epi64(scaled_v_high, final_right_shift_high);

          scaled_v_high = _mm256_slli_epi64(scaled_v_high, 32);
          __m256i results =
              _mm256_blend_epi32(scaled_v_low, scaled_v_high, 0xaa);
          results = _mm256_permutevar8x32_epi32(results, repack_perm);

          accum_data_v1 = _mm256_sub_epi32(results, post_scaling_offset);
        }
        {
          __m256i shifted_accum = _mm256_sllv_epi32(accum_data_v2, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m256i scaled_v_low = _mm256_mul_epi32(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 0)),
              m_64bit_low);
          __m256i scaled_v_high = _mm256_mul_epi32(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 1)),
              m_64bit_high);

          scaled_v_low = _mm256_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm256_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm256_srlv_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm256_srlv_epi64(scaled_v_high, final_right_shift_high);

          scaled_v_high = _mm256_slli_epi64(scaled_v_high, 32);
          __m256i results =
              _mm256_blend_epi32(scaled_v_low, scaled_v_high, 0xaa);
          results = _mm256_permutevar8x32_epi32(results, repack_perm);

          accum_data_v2 = _mm256_sub_epi32(results, post_scaling_offset);
        }
        {
          __m256i shifted_accum = _mm256_sllv_epi32(accum_data_v3, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m256i scaled_v_low = _mm256_mul_epi32(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 0)),
              m_64bit_low);
          __m256i scaled_v_high = _mm256_mul_epi32(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 1)),
              m_64bit_high);

          scaled_v_low = _mm256_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm256_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm256_srlv_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm256_srlv_epi64(scaled_v_high, final_right_shift_high);

          scaled_v_high = _mm256_slli_epi64(scaled_v_high, 32);
          __m256i results =
              _mm256_blend_epi32(scaled_v_low, scaled_v_high, 0xaa);
          results = _mm256_permutevar8x32_epi32(results, repack_perm);

          accum_data_v3 = _mm256_sub_epi32(results, post_scaling_offset);
        }
        {
          __m256i shifted_accum = _mm256_sllv_epi32(accum_data_v4, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m256i scaled_v_low = _mm256_mul_epi32(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 0)),
              m_64bit_low);
          __m256i scaled_v_high = _mm256_mul_epi32(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 1)),
              m_64bit_high);

          scaled_v_low = _mm256_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm256_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm256_srlv_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm256_srlv_epi64(scaled_v_high, final_right_shift_high);

          scaled_v_high = _mm256_slli_epi64(scaled_v_high, 32);
          __m256i results =
              _mm256_blend_epi32(scaled_v_low, scaled_v_high, 0xaa);
          results = _mm256_permutevar8x32_epi32(results, repack_perm);

          accum_data_v4 = _mm256_sub_epi32(results, post_scaling_offset);
        }
        {
          __m256i shifted_accum = _mm256_sllv_epi32(accum_data_v5, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m256i scaled_v_low = _mm256_mul_epi32(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 0)),
              m_64bit_low);
          __m256i scaled_v_high = _mm256_mul_epi32(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 1)),
              m_64bit_high);

          scaled_v_low = _mm256_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm256_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm256_srlv_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm256_srlv_epi64(scaled_v_high, final_right_shift_high);

          scaled_v_high = _mm256_slli_epi64(scaled_v_high, 32);
          __m256i results =
              _mm256_blend_epi32(scaled_v_low, scaled_v_high, 0xaa);
          results = _mm256_permutevar8x32_epi32(results, repack_perm);

          accum_data_v5 = _mm256_sub_epi32(results, post_scaling_offset);
        }
        {
          __m256i shifted_accum = _mm256_sllv_epi32(accum_data_v6, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m256i scaled_v_low = _mm256_mul_epi32(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 0)),
              m_64bit_low);
          __m256i scaled_v_high = _mm256_mul_epi32(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 1)),
              m_64bit_high);

          scaled_v_low = _mm256_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm256_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm256_srlv_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm256_srlv_epi64(scaled_v_high, final_right_shift_high);

          scaled_v_high = _mm256_slli_epi64(scaled_v_high, 32);
          __m256i results =
              _mm256_blend_epi32(scaled_v_low, scaled_v_high, 0xaa);
          results = _mm256_permutevar8x32_epi32(results, repack_perm);

          accum_data_v6 = _mm256_sub_epi32(results, post_scaling_offset);
        }
        {
          __m256i shifted_accum = _mm256_sllv_epi32(accum_data_v7, left_shift);
          // Apply the fixed-point part of the multiplier.
          __m256i scaled_v_low = _mm256_mul_epi32(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 0)),
              m_64bit_low);
          __m256i scaled_v_high = _mm256_mul_epi32(
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 1)),
              m_64bit_high);

          scaled_v_low = _mm256_add_epi64(scaled_v_low, offset_vector_low);
          scaled_v_high = _mm256_add_epi64(scaled_v_high, offset_vector_high);

          scaled_v_low = _mm256_srlv_epi64(scaled_v_low, final_right_shift_low);
          scaled_v_high =
              _mm256_srlv_epi64(scaled_v_high, final_right_shift_high);

          scaled_v_high = _mm256_slli_epi64(scaled_v_high, 32);
          __m256i results =
              _mm256_blend_epi32(scaled_v_low, scaled_v_high, 0xaa);
          results = _mm256_permutevar8x32_epi32(results, repack_perm);

          accum_data_v7 = _mm256_sub_epi32(results, post_scaling_offset);
        }
      }
      const __m256i clamp_max_v = _mm256_set1_epi32(params.clamp_max);
      const __m256i clamp_min_v = _mm256_set1_epi32(params.clamp_min);
      const bool store_full_block = (residual_rows == kAvx8bitBlockSize) &&
                                    (residual_cols == kAvx8bitBlockSize);

      __m256i accum_data_v[kAvx8bitBlockSize];
      if (!store_full_block) {
        accum_data_v[0] = accum_data_v0;
        accum_data_v[1] = accum_data_v1;
        accum_data_v[2] = accum_data_v2;
        accum_data_v[3] = accum_data_v3;
        accum_data_v[4] = accum_data_v4;
        accum_data_v[5] = accum_data_v5;
        accum_data_v[6] = accum_data_v6;
        accum_data_v[7] = accum_data_v7;
      }

      if (params.dst_type_id == DstTypeId<std::int8_t>::kValue) {
        std::int8_t* tmp_ptr = static_cast<std::int8_t*>(dst_ptr);
        if (store_full_block) {
          accum_data_v0 = _mm256_min_epi32(accum_data_v0, clamp_max_v);
          accum_data_v0 = _mm256_max_epi32(accum_data_v0, clamp_min_v);
          accum_data_v1 = _mm256_min_epi32(accum_data_v1, clamp_max_v);
          accum_data_v1 = _mm256_max_epi32(accum_data_v1, clamp_min_v);
          accum_data_v2 = _mm256_min_epi32(accum_data_v2, clamp_max_v);
          accum_data_v2 = _mm256_max_epi32(accum_data_v2, clamp_min_v);
          accum_data_v3 = _mm256_min_epi32(accum_data_v3, clamp_max_v);
          accum_data_v3 = _mm256_max_epi32(accum_data_v3, clamp_min_v);
          accum_data_v4 = _mm256_min_epi32(accum_data_v4, clamp_max_v);
          accum_data_v4 = _mm256_max_epi32(accum_data_v4, clamp_min_v);
          accum_data_v5 = _mm256_min_epi32(accum_data_v5, clamp_max_v);
          accum_data_v5 = _mm256_max_epi32(accum_data_v5, clamp_min_v);
          accum_data_v6 = _mm256_min_epi32(accum_data_v6, clamp_max_v);
          accum_data_v6 = _mm256_max_epi32(accum_data_v6, clamp_min_v);
          accum_data_v7 = _mm256_min_epi32(accum_data_v7, clamp_max_v);
          accum_data_v7 = _mm256_max_epi32(accum_data_v7, clamp_min_v);
          intrin_utils::mm256_storeu_cvtepi32_epi8(&tmp_ptr[0 * dst_stride],
                                                   accum_data_v0);
          intrin_utils::mm256_storeu_cvtepi32_epi8(&tmp_ptr[1 * dst_stride],
                                                   accum_data_v1);
          intrin_utils::mm256_storeu_cvtepi32_epi8(&tmp_ptr[2 * dst_stride],
                                                   accum_data_v2);
          intrin_utils::mm256_storeu_cvtepi32_epi8(&tmp_ptr[3 * dst_stride],
                                                   accum_data_v3);
          intrin_utils::mm256_storeu_cvtepi32_epi8(&tmp_ptr[4 * dst_stride],
                                                   accum_data_v4);
          intrin_utils::mm256_storeu_cvtepi32_epi8(&tmp_ptr[5 * dst_stride],
                                                   accum_data_v5);
          intrin_utils::mm256_storeu_cvtepi32_epi8(&tmp_ptr[6 * dst_stride],
                                                   accum_data_v6);
          intrin_utils::mm256_storeu_cvtepi32_epi8(&tmp_ptr[7 * dst_stride],
                                                   accum_data_v7);
        } else {
          for (int j = 0; j < residual_cols; ++j) {
            __m256 result = accum_data_v[j];
            result = _mm256_min_epi32(result, clamp_max_v);
            result = _mm256_max_epi32(result, clamp_min_v);
            intrin_utils::mm256_n_storeu_cvtepi32_epi8(tmp_ptr, residual_rows,
                                                       result);
            tmp_ptr += dst_stride;
          }
        }
        dst_ptr = static_cast<void*>(static_cast<std::int8_t*>(dst_ptr) +
                                     kAvx8bitBlockSize);
      } else if (params.dst_type_id == DstTypeId<std::uint8_t>::kValue) {
        std::uint8_t* tmp_ptr = static_cast<std::uint8_t*>(dst_ptr);
        if (store_full_block) {
          accum_data_v0 = _mm256_min_epi32(accum_data_v0, clamp_max_v);
          accum_data_v0 = _mm256_max_epi32(accum_data_v0, clamp_min_v);
          accum_data_v1 = _mm256_min_epi32(accum_data_v1, clamp_max_v);
          accum_data_v1 = _mm256_max_epi32(accum_data_v1, clamp_min_v);
          accum_data_v2 = _mm256_min_epi32(accum_data_v2, clamp_max_v);
          accum_data_v2 = _mm256_max_epi32(accum_data_v2, clamp_min_v);
          accum_data_v3 = _mm256_min_epi32(accum_data_v3, clamp_max_v);
          accum_data_v3 = _mm256_max_epi32(accum_data_v3, clamp_min_v);
          accum_data_v4 = _mm256_min_epi32(accum_data_v4, clamp_max_v);
          accum_data_v4 = _mm256_max_epi32(accum_data_v4, clamp_min_v);
          accum_data_v5 = _mm256_min_epi32(accum_data_v5, clamp_max_v);
          accum_data_v5 = _mm256_max_epi32(accum_data_v5, clamp_min_v);
          accum_data_v6 = _mm256_min_epi32(accum_data_v6, clamp_max_v);
          accum_data_v6 = _mm256_max_epi32(accum_data_v6, clamp_min_v);
          accum_data_v7 = _mm256_min_epi32(accum_data_v7, clamp_max_v);
          accum_data_v7 = _mm256_max_epi32(accum_data_v7, clamp_min_v);
          intrin_utils::mm256_storeu_cvtepi32_epi8(&tmp_ptr[0], accum_data_v0);
          intrin_utils::mm256_storeu_cvtepi32_epi8(&tmp_ptr[dst_stride],
                                                   accum_data_v1);
          intrin_utils::mm256_storeu_cvtepi32_epi8(&tmp_ptr[2 * dst_stride],
                                                   accum_data_v2);
          intrin_utils::mm256_storeu_cvtepi32_epi8(&tmp_ptr[3 * dst_stride],
                                                   accum_data_v3);
          intrin_utils::mm256_storeu_cvtepi32_epi8(&tmp_ptr[4 * dst_stride],
                                                   accum_data_v4);
          intrin_utils::mm256_storeu_cvtepi32_epi8(&tmp_ptr[5 * dst_stride],
                                                   accum_data_v5);
          intrin_utils::mm256_storeu_cvtepi32_epi8(&tmp_ptr[6 * dst_stride],
                                                   accum_data_v6);
          intrin_utils::mm256_storeu_cvtepi32_epi8(&tmp_ptr[7 * dst_stride],
                                                   accum_data_v7);
        } else {
          for (int j = 0; j < residual_cols; ++j) {
            __m256 result = accum_data_v[j];
            result = _mm256_min_epi32(result, clamp_max_v);
            result = _mm256_max_epi32(result, clamp_min_v);
            intrin_utils::mm256_n_storeu_cvtepi32_epi8(tmp_ptr, residual_rows,
                                                       result);
            tmp_ptr += dst_stride;
          }
        }
        dst_ptr = static_cast<void*>(static_cast<std::uint8_t*>(dst_ptr) +
                                     kAvx8bitBlockSize);
      } else if (params.dst_type_id == DstTypeId<std::int16_t>::kValue) {
        std::int16_t* tmp_ptr = static_cast<std::int16_t*>(dst_ptr);
        if (store_full_block) {
          accum_data_v0 = _mm256_min_epi32(accum_data_v0, clamp_max_v);
          accum_data_v0 = _mm256_max_epi32(accum_data_v0, clamp_min_v);
          accum_data_v1 = _mm256_min_epi32(accum_data_v1, clamp_max_v);
          accum_data_v1 = _mm256_max_epi32(accum_data_v1, clamp_min_v);
          accum_data_v2 = _mm256_min_epi32(accum_data_v2, clamp_max_v);
          accum_data_v2 = _mm256_max_epi32(accum_data_v2, clamp_min_v);
          accum_data_v3 = _mm256_min_epi32(accum_data_v3, clamp_max_v);
          accum_data_v3 = _mm256_max_epi32(accum_data_v3, clamp_min_v);
          accum_data_v4 = _mm256_min_epi32(accum_data_v4, clamp_max_v);
          accum_data_v4 = _mm256_max_epi32(accum_data_v4, clamp_min_v);
          accum_data_v5 = _mm256_min_epi32(accum_data_v5, clamp_max_v);
          accum_data_v5 = _mm256_max_epi32(accum_data_v5, clamp_min_v);
          accum_data_v6 = _mm256_min_epi32(accum_data_v6, clamp_max_v);
          accum_data_v6 = _mm256_max_epi32(accum_data_v6, clamp_min_v);
          accum_data_v7 = _mm256_min_epi32(accum_data_v7, clamp_max_v);
          accum_data_v7 = _mm256_max_epi32(accum_data_v7, clamp_min_v);
          intrin_utils::mm256_storeu_cvtepi32_epi16(&tmp_ptr[0], accum_data_v0);
          intrin_utils::mm256_storeu_cvtepi32_epi16(&tmp_ptr[dst_stride],
                                                    accum_data_v1);
          intrin_utils::mm256_storeu_cvtepi32_epi16(&tmp_ptr[2 * dst_stride],
                                                    accum_data_v2);
          intrin_utils::mm256_storeu_cvtepi32_epi16(&tmp_ptr[3 * dst_stride],
                                                    accum_data_v3);
          intrin_utils::mm256_storeu_cvtepi32_epi16(&tmp_ptr[4 * dst_stride],
                                                    accum_data_v4);
          intrin_utils::mm256_storeu_cvtepi32_epi16(&tmp_ptr[5 * dst_stride],
                                                    accum_data_v5);
          intrin_utils::mm256_storeu_cvtepi32_epi16(&tmp_ptr[6 * dst_stride],
                                                    accum_data_v6);
          intrin_utils::mm256_storeu_cvtepi32_epi16(&tmp_ptr[7 * dst_stride],
                                                    accum_data_v7);
        } else {
          for (int j = 0; j < residual_cols; ++j) {
            __m256 result = accum_data_v[j];
            result = _mm256_min_epi32(result, clamp_max_v);
            result = _mm256_max_epi32(result, clamp_min_v);
            intrin_utils::mm256_n_storeu_cvtepi32_epi16(tmp_ptr, residual_rows,
                                                        result);
            tmp_ptr += dst_stride;
          }
        }
        dst_ptr = static_cast<void*>(static_cast<std::int16_t*>(dst_ptr) +
                                     kAvx8bitBlockSize);
      } else if (params.dst_type_id == DstTypeId<std::int32_t>::kValue) {
        if (store_full_block) {
          std::int32_t* tmp_ptr = static_cast<std::int32_t*>(dst_ptr);
          intrin_utils::mm256_storeu_epi32(&tmp_ptr[0], accum_data_v0);
          intrin_utils::mm256_storeu_epi32(&tmp_ptr[dst_stride], accum_data_v1);
          intrin_utils::mm256_storeu_epi32(&tmp_ptr[2 * dst_stride],
                                           accum_data_v2);
          intrin_utils::mm256_storeu_epi32(&tmp_ptr[3 * dst_stride],
                                           accum_data_v3);
          intrin_utils::mm256_storeu_epi32(&tmp_ptr[4 * dst_stride],
                                           accum_data_v4);
          intrin_utils::mm256_storeu_epi32(&tmp_ptr[5 * dst_stride],
                                           accum_data_v5);
          intrin_utils::mm256_storeu_epi32(&tmp_ptr[6 * dst_stride],
                                           accum_data_v6);
          intrin_utils::mm256_storeu_epi32(&tmp_ptr[7 * dst_stride],
                                           accum_data_v7);
        } else {
          std::int32_t* dst_block_ptr = static_cast<std::int32_t*>(dst_ptr);
          for (int j = 0; j < residual_cols; ++j) {
            intrin_utils::mm256_n_storeu_epi32(dst_block_ptr, residual_rows,
                                               accum_data_v[j]);
            dst_block_ptr += dst_stride;
          }
        }
        dst_ptr = static_cast<void*>(static_cast<std::int32_t*>(dst_ptr) +
                                     kAvx8bitBlockSize);
      } else {
        RUY_DCHECK(false);
      }

      lhs_col_ptr += kAvx8bitBlockSize * params.lhs_stride;
    }  // End row-block loop.

    dst_col_ptr = static_cast<void*>(static_cast<char*>(dst_col_ptr) +
                                     kAvx8bitBlockSize * params.dst_stride);
    rhs_col_ptr += kAvx8bitBlockSize * params.rhs_stride;
  }  // End col-block loop.
}  // NOLINT(readability/fn_size)

void Kernel8bitAvx2SingleCol(const KernelParams8bit<8, 8>& params) {
  gemmlowp::ScopedProfilingLabel label("Kernel kAvx2 8-bit GEMV");

  RUY_DCHECK_EQ(params.dst_cols, 1);
  RUY_DCHECK_EQ(params.last_col, 0);
  RUY_DCHECK_EQ(params.start_col, 0);

  const std::int8_t splitter_idx_data[32] = {
      0, 1, 4, 5, 8,  9,  12, 13,  //
      2, 3, 6, 7, 10, 11, 14, 15,  //
      0, 1, 4, 5, 8,  9,  12, 13,  //
      2, 3, 6, 7, 10, 11, 14, 15   //
  };

  int bias_ptr_block_increment =
      params.flags & RUY_ASM_FLAG_HAS_BIAS ? kAvx8bitBlockSize : 0;

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
  std::int32_t rhs_sums_offsets[8];
  if (has_rhs_sums_offsets) {
    const __m256i rhs_sums_offset_v = _mm256_mullo_epi32(
        _mm256_set1_epi32(lhs_zero_point),
        _mm256_loadu_si256(
            reinterpret_cast<__m256i const*>(&params.rhs_sums[0])));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(rhs_sums_offsets),
                        rhs_sums_offset_v);
  }

  for (int row = params.start_row; row <= params.last_row;
       row += kAvx8bitBlockSize) {
    const int residual_rows =
        std::min(params.dst_rows - row, kAvx8bitBlockSize);

    const __m256i splitter_idx =
        _mm256_loadu_si256(reinterpret_cast<__m256i const*>(splitter_idx_data));

    __m256i accum_data_v0;

    // Initialize with bias.
    __m256i initial_accum_data =
        intrin_utils::mm256_n_loadu_epi32(residual_rows, bias_ptr);
    bias_ptr += bias_ptr_block_increment;

    // Adjustments common across columns.
    const std::int32_t rhs_zero_point = params.rhs_zero_point;
    if ((params.flags & RUY_ASM_FLAG_HAS_LHS_SUMS) && rhs_zero_point) {
      const __m256i lhs_sums_offset = _mm256_mullo_epi32(
          _mm256_set1_epi32(rhs_zero_point),
          _mm256_loadu_si256(
              reinterpret_cast<__m256i const*>(&params.lhs_sums[row])));
      initial_accum_data =
          _mm256_sub_epi32(initial_accum_data, lhs_sums_offset);
    }
    const std::int32_t prod_zp_depth = params.prod_zp_depth;
    if (prod_zp_depth) {
      initial_accum_data = _mm256_add_epi32(initial_accum_data,
                                            _mm256_set1_epi32(prod_zp_depth));
    }

    // Adjustments differing across columns.
    if (has_rhs_sums_offsets) {
      accum_data_v0 = _mm256_sub_epi32(initial_accum_data,
                                       _mm256_set1_epi32(rhs_sums_offsets[0]));
    } else {
      accum_data_v0 = initial_accum_data;
    }

    const std::int8_t* lhs_ptr = lhs_col_ptr;
    const std::int8_t* rhs_ptr = rhs_col_ptr;
    for (int d = 0; d < params.depth; d += kAvx8bitInnerSize) {
      const __m256i lhs_data =
          _mm256_load_si256(reinterpret_cast<const __m256i*>(lhs_ptr));
      const __m128i rhs_data_8bit = _mm_loadu_si32(rhs_ptr);

      // Each "int32" is two 16-bit RHS values, sign extended from 8-bit.
      // For simplicity we load 4x the data that we need and process twice the
      // data  that we need  and store only the data we need.
      std::int32_t rhs_data[2];
      const __m128i rhs_16_bit_dup = _mm_cvtepi8_epi16(rhs_data_8bit);
      // Now that we have cast the RHS data, we store it so that each value
      // can be separately loaded in the accumulation loop.
      _mm_storeu_si64(reinterpret_cast<__m128i*>(rhs_data), rhs_16_bit_dup);

      const __m256i lhs_data_split =
          _mm256_shuffle_epi8(lhs_data, splitter_idx);
      const __m256i lhs_data_split_expand_bottom =
          _mm256_cvtepi8_epi16(_mm256_extracti128_si256(lhs_data_split, 0));
      const __m256i lhs_data_split_expand_top =
          _mm256_cvtepi8_epi16(_mm256_extracti128_si256(lhs_data_split, 1));

      // Take bytes 0, 1, 4, 5, 8, 9, ... expanded to 16-bit.
      const __m256i lhs_16_bit_low = _mm256_permute2x128_si256(
          lhs_data_split_expand_bottom, lhs_data_split_expand_top, 0x20);
      // Take bytes 2, 3, 6, 7, 10, 11, ... expanded to 16-bit.
      const __m256i lhs_16_bit_high = _mm256_permute2x128_si256(
          lhs_data_split_expand_bottom, lhs_data_split_expand_top, 0x31);
      // Accumulate for column 0.
      const std::int32_t low_rhs_value = rhs_data[0];
      const std::int32_t high_rhs_value = rhs_data[1];

      const __m256i rhs_16_bit_dup_low = _mm256_set1_epi32(low_rhs_value);
      const __m256i rhs_16_bit_dup_high = _mm256_set1_epi32(high_rhs_value);

      accum_data_v0 = _mm256_add_epi32(
          accum_data_v0, _mm256_madd_epi16(lhs_16_bit_low, rhs_16_bit_dup_low));
      accum_data_v0 = _mm256_add_epi32(
          accum_data_v0,
          _mm256_madd_epi16(lhs_16_bit_high, rhs_16_bit_dup_high));

      lhs_ptr += kAvx8bitBlockSize * kAvx8bitInnerSize;
      rhs_ptr += kAvx8bitBlockSize * kAvx8bitInnerSize;
    }

    if (params.dst_type_id != DstTypeId<std::int32_t>::kValue) {
      __m256i m_vector;
      __m256i e_vector;
      // Does not make use of RUY_ASM_FLAG_NEEDS_LEFT_SHIFT.
      if (params.flags & RUY_ASM_FLAG_HAS_PERCHANNEL) {
        m_vector = intrin_utils::mm256_n_loadu_epi32(
            residual_rows, &params.multiplier_fixedpoint[row]);
        e_vector = intrin_utils::mm256_n_loadu_epi32(
            residual_rows, &params.multiplier_exponent[row]);
      } else {
        // These arrays have size LhsCols, and are pre-filled.
        m_vector = _mm256_set1_epi32(params.multiplier_fixedpoint[0]);
        e_vector = _mm256_set1_epi32(params.multiplier_exponent[0]);
      }

      const __m256i m_64bit_low =
          _mm256_cvtepi32_epi64(_mm256_extracti128_si256(m_vector, 0));
      const __m256i m_64bit_high =
          _mm256_cvtepi32_epi64(_mm256_extracti128_si256(m_vector, 1));

      const __m256i zero_vector = _mm256_setzero_si256();
      const __m256i left_shift = _mm256_max_epi32(e_vector, zero_vector);
      const __m256i neg_e_vector = _mm256_sub_epi32(zero_vector, e_vector);
      const __m256i right_shift = _mm256_max_epi32(neg_e_vector, zero_vector);
      const __m256i final_right_shift =
          _mm256_add_epi32(right_shift, _mm256_set1_epi32(31));
      const __m256i final_right_shift_low =
          _mm256_cvtepi32_epi64(_mm256_extracti128_si256(final_right_shift, 0));
      const __m256i final_right_shift_high =
          _mm256_cvtepi32_epi64(_mm256_extracti128_si256(final_right_shift, 1));
      // Really we want 0x100000000, but use half to avoid overflowing.
      const __m256i convert_to_signed_halved =
          _mm256_srlv_epi32(_mm256_set1_epi32(0x80000000), right_shift);
      const __m256i convert_to_unsigned_64 =
          _mm256_set1_epi64x(0x8000000000000000);

      __m256i post_scaling_offset =
          _mm256_add_epi32(convert_to_signed_halved, convert_to_signed_halved);

      const __m256i offset_vector =
          _mm256_slli_epi64(_mm256_set1_epi64x(1), 30);
      // Really these should be shifted by neg_e_vector, but tests pass when
      // using right_shift.
      const __m256i offset_vector_low = _mm256_add_epi64(
          _mm256_sllv_epi64(
              offset_vector,
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(right_shift, 0))),
          convert_to_unsigned_64);
      const __m256i offset_vector_high = _mm256_add_epi64(
          _mm256_sllv_epi64(
              offset_vector,
              _mm256_cvtepi32_epi64(_mm256_extracti128_si256(right_shift, 1))),
          convert_to_unsigned_64);

      if (params.dst_zero_point) {
        const __m256i dst_zero_point = _mm256_set1_epi32(params.dst_zero_point);
        // The post-scaling offset is subtracted later, so this has the effect
        // of adding the zero point.
        post_scaling_offset =
            _mm256_sub_epi32(post_scaling_offset, dst_zero_point);
      }

#if !RUY_OPT_ENABLED(RUY_OPT_NATIVE_ROUNDING)
      RUY_DCHECK(false);
#endif
      const __m256i repack_perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

      // See GEMM version for details of this process.
      {
        __m256i shifted_accum = _mm256_sllv_epi32(accum_data_v0, left_shift);
        // Apply the fixed-point part of the multiplier.
        __m256i scaled_v_low = _mm256_mul_epi32(
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 0)),
            m_64bit_low);
        __m256i scaled_v_high = _mm256_mul_epi32(
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shifted_accum, 1)),
            m_64bit_high);

        scaled_v_low = _mm256_add_epi64(scaled_v_low, offset_vector_low);
        scaled_v_high = _mm256_add_epi64(scaled_v_high, offset_vector_high);

        scaled_v_low = _mm256_srlv_epi64(scaled_v_low, final_right_shift_low);
        scaled_v_high =
            _mm256_srlv_epi64(scaled_v_high, final_right_shift_high);

        scaled_v_high = _mm256_slli_epi64(scaled_v_high, 32);
        __m256i results = _mm256_blend_epi32(scaled_v_low, scaled_v_high, 0xaa);
        results = _mm256_permutevar8x32_epi32(results, repack_perm);

        accum_data_v0 = _mm256_sub_epi32(results, post_scaling_offset);
      }
    }
    const __m256i clamp_max_v = _mm256_set1_epi32(params.clamp_max);
    const __m256i clamp_min_v = _mm256_set1_epi32(params.clamp_min);

    if (params.dst_type_id == DstTypeId<std::int8_t>::kValue) {
      std::int8_t* tmp_ptr = static_cast<std::int8_t*>(dst_ptr);
      __m256 result = accum_data_v0;
      result = _mm256_min_epi32(result, clamp_max_v);
      result = _mm256_max_epi32(result, clamp_min_v);
      intrin_utils::mm256_n_storeu_cvtepi32_epi8(tmp_ptr, residual_rows,
                                                 result);
      dst_ptr = static_cast<void*>(static_cast<std::int8_t*>(dst_ptr) +
                                   kAvx8bitBlockSize);
    } else if (params.dst_type_id == DstTypeId<std::uint8_t>::kValue) {
      std::uint8_t* tmp_ptr = static_cast<std::uint8_t*>(dst_ptr);
      __m256 result = accum_data_v0;
      result = _mm256_min_epi32(result, clamp_max_v);
      result = _mm256_max_epi32(result, clamp_min_v);
      intrin_utils::mm256_n_storeu_cvtepi32_epi8(tmp_ptr, residual_rows,
                                                 result);
      dst_ptr = static_cast<void*>(static_cast<std::uint8_t*>(dst_ptr) +
                                   kAvx8bitBlockSize);
    } else if (params.dst_type_id == DstTypeId<std::int16_t>::kValue) {
      std::int16_t* tmp_ptr = static_cast<std::int16_t*>(dst_ptr);
      __m256 result = accum_data_v0;
      result = _mm256_min_epi32(result, clamp_max_v);
      result = _mm256_max_epi32(result, clamp_min_v);
      intrin_utils::mm256_n_storeu_cvtepi32_epi16(tmp_ptr, residual_rows,
                                                  result);
      dst_ptr = static_cast<void*>(static_cast<std::int16_t*>(dst_ptr) +
                                   kAvx8bitBlockSize);
    } else if (params.dst_type_id == DstTypeId<std::int32_t>::kValue) {
      std::int32_t* dst_block_ptr = static_cast<std::int32_t*>(dst_ptr);
      intrin_utils::mm256_n_storeu_epi32(dst_block_ptr, residual_rows,
                                         accum_data_v0);
      dst_ptr = static_cast<void*>(static_cast<std::int32_t*>(dst_ptr) +
                                   kAvx8bitBlockSize);
    } else {
      RUY_DCHECK(false);
    }

    lhs_col_ptr += kAvx8bitBlockSize * params.lhs_stride;
  }  // End row-block loop.

  dst_col_ptr = static_cast<void*>(static_cast<char*>(dst_col_ptr) +
                                   kAvx8bitBlockSize * params.dst_stride);
  rhs_col_ptr += kAvx8bitBlockSize * params.rhs_stride;
}  // NOLINT(readability/fn_size)

void KernelFloatAvx2(const KernelParamsFloat<8, 8>& params) {
  gemmlowp::ScopedProfilingLabel label("Kernel kAvx2 float");

  // As parameters are defined, we need to scale by sizeof(float).
  const std::int64_t lhs_stride = params.lhs_stride >> 2;
  const std::int64_t dst_stride = params.dst_stride >> 2;
  const std::int64_t rhs_stride = params.rhs_stride >> 2;
  //
  int bias_ptr_block_increment = params.flags & RUY_ASM_FLAG_HAS_BIAS ? 1 : 0;
  // AVX2 float block size = 8.
  const int end_row = std::min(params.dst_rows, params.last_row + 8);
  const int end_col = std::min(params.dst_cols, params.last_col + 8);
  //
  const float* adj_rhs_col_ptr =
      params.rhs_base_ptr - params.start_col * rhs_stride;
  float* adj_dst_col_ptr =
      params.dst_base_ptr - params.start_col * dst_stride - params.start_row;
  const float* adj_lhs_col_ptr =
      params.lhs_base_ptr - params.start_row * lhs_stride;
  const float* bias_col_ptr = params.bias;

  const __m256 clamp_max_v = _mm256_set1_ps(params.clamp_max);
  const __m256 clamp_min_v = _mm256_set1_ps(params.clamp_min);

  int col = params.start_col;
  // Loop through cols by float block size, leaving incomplete remainder
  for (; col <= end_col - 8; col += 8) {
    __m256 accum_data_v[8];

    const float* rhs_col_ptr = adj_rhs_col_ptr + col * rhs_stride;
    float* dst_col_ptr = adj_dst_col_ptr + col * dst_stride;

    for (int row = params.start_row; row < end_row; row += 8) {
      const int residual_rows = std::min(end_row - row, 8);

      const float* lhs_col_ptr = adj_lhs_col_ptr + row * lhs_stride;
      float* dst_ptr = dst_col_ptr + row;
      const float* bias_ptr = bias_col_ptr + row * bias_ptr_block_increment;

      // Initialize with bias.
      const __m256 initial_accum_data =
          intrin_utils::mm256_n_loadu_ps(residual_rows, bias_ptr);

      for (int j = 0; j < 8; ++j) {
        accum_data_v[j] = initial_accum_data;
      }

      const float* lhs_ptr = lhs_col_ptr;
      const float* rhs_ptr = rhs_col_ptr;
      for (int d = 0; d < params.depth; ++d) {
        const __m256 lhs_data = _mm256_loadu_ps(lhs_ptr);
        // In this version RHS values are loaded individually rather than first
        // loading together and then extract with broadcasting. This is because
        // AVX flavours and instrinsics and compilers in combination do not
        // handle this pattern of extraction very well.
        const float* rhs_data = rhs_ptr;

        for (int j = 0; j < 8; ++j) {
          const __m256 dup_rhs_element_j = _mm256_set1_ps(rhs_data[j]);
          accum_data_v[j] =
              _mm256_fmadd_ps(lhs_data, dup_rhs_element_j, accum_data_v[j]);
        }
        lhs_ptr += 8;
        rhs_ptr += 8;
      }

      if (residual_rows == 8) {
        for (int j = 0; j < 8; ++j) {
          float* block_ptr = dst_ptr + j * dst_stride;
          accum_data_v[j] = _mm256_min_ps(accum_data_v[j], clamp_max_v);
          accum_data_v[j] = _mm256_max_ps(accum_data_v[j], clamp_min_v);
          _mm256_storeu_ps(block_ptr, accum_data_v[j]);
        }
      } else {
        for (int j = 0; j < 8; ++j) {
          float* block_ptr = dst_ptr + j * dst_stride;
          accum_data_v[j] = _mm256_min_ps(accum_data_v[j], clamp_max_v);
          accum_data_v[j] = _mm256_max_ps(accum_data_v[j], clamp_min_v);
          intrin_utils::mm256_n_storeu_ps(block_ptr, residual_rows,
                                          accum_data_v[j]);
        }
      }
    }  // End row-block loop.
  }    // End col-block loop.

  if (col < end_col) {
    // Remaining cols in [0, float block size).
    RUY_DCHECK_GE(end_col - col, 0);
    RUY_DCHECK_LT(end_col - col, 8);

    __m256 accum_data_v[8];

    const float* rhs_col_ptr = adj_rhs_col_ptr + col * rhs_stride;
    float* dst_col_ptr = adj_dst_col_ptr + col * dst_stride;
    const int residual_cols = std::min(end_col - col, 8);

    for (int row = params.start_row; row < end_row; row += 8) {
      const int residual_rows = std::min(end_row - row, 8);

      const float* lhs_col_ptr = adj_lhs_col_ptr + row * lhs_stride;
      float* dst_ptr = dst_col_ptr + row;
      const float* bias_ptr = bias_col_ptr + row * bias_ptr_block_increment;

      // Initialize with bias.
      const __m256 initial_accum_data =
          intrin_utils::mm256_n_loadu_ps(residual_rows, bias_ptr);

      for (int j = 0; j < 8; ++j) {
        accum_data_v[j] = initial_accum_data;
      }

      const float* lhs_ptr = lhs_col_ptr;
      const float* rhs_ptr = rhs_col_ptr;
      for (int d = 0; d < params.depth; ++d) {
        const __m256 lhs_data = _mm256_loadu_ps(lhs_ptr);
        const float* rhs_data = rhs_ptr;

        for (int j = 0; j < 8; ++j) {
          const __m256 dup_rhs_element_j = _mm256_set1_ps(rhs_data[j]);
          accum_data_v[j] =
              _mm256_fmadd_ps(lhs_data, dup_rhs_element_j, accum_data_v[j]);
        }
        lhs_ptr += 8;
        rhs_ptr += 8;
      }

      for (int j = 0; j < residual_cols; ++j) {
        float* block_ptr = dst_ptr + j * dst_stride;
        accum_data_v[j] = _mm256_min_ps(accum_data_v[j], clamp_max_v);
        accum_data_v[j] = _mm256_max_ps(accum_data_v[j], clamp_min_v);
        intrin_utils::mm256_n_storeu_ps(block_ptr, residual_rows,
                                        accum_data_v[j]);
      }
    }  // End row-block loop.
  }    // End col-block terminal conditional.
}

void KernelFloatAvx2SingleCol(const KernelParamsFloat<8, 8>& params) {
  gemmlowp::ScopedProfilingLabel label("Kernel kAvx2 float GEMV");

  RUY_DCHECK_EQ(params.dst_cols, 1);
  RUY_DCHECK_EQ(params.last_col, 0);
  RUY_DCHECK_EQ(params.start_col, 0);

  // As parameters are defined, we need to scale by sizeof(float).
  const std::int64_t lhs_stride = params.lhs_stride >> 2;
  //
  int bias_ptr_block_increment = params.flags & RUY_ASM_FLAG_HAS_BIAS ? 1 : 0;
  // AVX2 float block size = 8.
  const int end_row = std::min(params.dst_rows, params.last_row + 8);

  float* adj_dst_col_ptr = params.dst_base_ptr - params.start_row;
  const float* adj_lhs_col_ptr =
      params.lhs_base_ptr - params.start_row * lhs_stride;
  const float* bias_col_ptr = params.bias;

  const __m256 clamp_max_v = _mm256_set1_ps(params.clamp_max);
  const __m256 clamp_min_v = _mm256_set1_ps(params.clamp_min);

  __m256 accum_data_v;

  const float* rhs_col_ptr = params.rhs_base_ptr;
  float* dst_col_ptr = adj_dst_col_ptr;

  int row = params.start_row;
  for (; row <= end_row - 8; row += 8) {
    const float* lhs_col_ptr = adj_lhs_col_ptr + row * lhs_stride;
    float* dst_ptr = dst_col_ptr + row;
    const float* bias_ptr = bias_col_ptr + row * bias_ptr_block_increment;

    // Initialize with bias.
    accum_data_v = _mm256_loadu_ps(bias_ptr);

    const float* lhs_ptr = lhs_col_ptr;
    const float* rhs_ptr = rhs_col_ptr;
    int d = 0;
    for (; d <= params.depth - 4; d += 4) {
      const __m256 lhs_data_0 = _mm256_loadu_ps(lhs_ptr);
      const __m256 dup_rhs_element_0 = _mm256_set1_ps(rhs_ptr[0]);
      accum_data_v =
          _mm256_fmadd_ps(lhs_data_0, dup_rhs_element_0, accum_data_v);
      const __m256 dup_rhs_element_1 = _mm256_set1_ps(rhs_ptr[8]);
      const __m256 lhs_data_1 = _mm256_loadu_ps(lhs_ptr + 8);
      accum_data_v =
          _mm256_fmadd_ps(lhs_data_1, dup_rhs_element_1, accum_data_v);

      const __m256 lhs_data_2 = _mm256_loadu_ps(lhs_ptr + 16);
      const __m256 dup_rhs_element_2 = _mm256_set1_ps(rhs_ptr[16]);
      accum_data_v =
          _mm256_fmadd_ps(lhs_data_2, dup_rhs_element_2, accum_data_v);
      const __m256 dup_rhs_element_3 = _mm256_set1_ps(rhs_ptr[24]);
      const __m256 lhs_data_3 = _mm256_loadu_ps(lhs_ptr + 24);
      accum_data_v =
          _mm256_fmadd_ps(lhs_data_3, dup_rhs_element_3, accum_data_v);
      lhs_ptr += 32;  // Loaded 8 * 4 floats.
      rhs_ptr += 32;
    }
    for (; d < params.depth; ++d) {
      const __m256 lhs_data = _mm256_loadu_ps(lhs_ptr);
      const float* rhs_data = rhs_ptr;

      const __m256 dup_rhs_element_j = _mm256_set1_ps(rhs_data[0]);
      accum_data_v = _mm256_fmadd_ps(lhs_data, dup_rhs_element_j, accum_data_v);
      lhs_ptr += 8;
      rhs_ptr += 8;
    }

    accum_data_v = _mm256_min_ps(accum_data_v, clamp_max_v);
    accum_data_v = _mm256_max_ps(accum_data_v, clamp_min_v);
    _mm256_storeu_ps(dst_ptr, accum_data_v);
  }  // End row-block loop.

  if (row < end_row) {
    const int residual_rows = end_row - row;
    RUY_CHECK_GE(residual_rows, 1);
    RUY_CHECK_LT(residual_rows, 8);

    const float* lhs_col_ptr = adj_lhs_col_ptr + row * lhs_stride;
    float* dst_ptr = dst_col_ptr + row;
    const float* bias_ptr = bias_col_ptr + row * bias_ptr_block_increment;

    // Initialize with bias.
    accum_data_v = intrin_utils::mm256_n_loadu_ps(residual_rows, bias_ptr);

    const float* lhs_ptr = lhs_col_ptr;
    const float* rhs_ptr = rhs_col_ptr;
    for (int d = 0; d < params.depth; ++d) {
      const __m256 lhs_data = _mm256_loadu_ps(lhs_ptr);
      const float* rhs_data = rhs_ptr;

      const __m256 dup_rhs_element_j = _mm256_set1_ps(rhs_data[0]);
      accum_data_v = _mm256_fmadd_ps(lhs_data, dup_rhs_element_j, accum_data_v);
      lhs_ptr += 8;
      rhs_ptr += 8;
    }

    accum_data_v = _mm256_min_ps(accum_data_v, clamp_max_v);
    accum_data_v = _mm256_max_ps(accum_data_v, clamp_min_v);
    intrin_utils::mm256_n_storeu_ps(dst_ptr, residual_rows, accum_data_v);
  }  // End handling of residual rows.
}

#endif  //  RUY_PLATFORM(AVX2) && RUY_OPT_ENABLED(RUY_OPT_ASM)

}  // namespace ruy
