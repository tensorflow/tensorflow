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

#include <cstdint>
#include <cstring>

#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/matrix.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/pack.h"
#include "tensorflow/lite/experimental/ruy/path.h"
#include "tensorflow/lite/experimental/ruy/platform.h"
#include "tensorflow/lite/experimental/ruy/profiler/instrumentation.h"

#if RUY_PLATFORM(AVX512) && RUY_OPT_ENABLED(RUY_OPT_INTRINSICS)
#include <immintrin.h>  // IWYU pragma: keep
#endif

namespace ruy {

#if !(RUY_PLATFORM(AVX512) && RUY_OPT_ENABLED(RUY_OPT_ASM))

void Pack8bitAvx512(const std::int8_t* src_ptr, std::int8_t input_xor,
                    const std::int8_t* zerobuf, int src_stride,
                    int remaining_src_cols, int src_rows,
                    std::int8_t* packed_ptr, std::int32_t* sums_ptr) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

void PackFloatAvx512(const float* src_ptr, const float* zerobuf, int src_stride,
                     int remaining_src_cols, int src_rows, float* packed_ptr) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

#else  // RUY_PLATFORM(AVX512) && RUY_OPT_ENABLED(RUY_OPT_ASM)

// The first int8_t template parameter is arbitrary: this routine is common to
// all 8-bit source matrix types.
using PackImpl8bitAvx512 =
    PackImpl<Path::kAvx512, FixedKernelLayout<Order::kColMajor, 4, 16>,
             std::int8_t, std::int8_t, std::int32_t>;

namespace {

inline void ZeroHalf8bitAvx512(int src_rows, std::int8_t packed_zero_point,
                               std::int8_t* packed_ptr) {
  using Layout = PackImpl8bitAvx512::Layout;
  static constexpr int kHalfLayoutCols =
      PackImpl8bitAvx512::kHalfLayoutCols;  // Half the number of cols in a
                                            // block.
  RUY_DCHECK_EQ(kHalfLayoutCols, 8);
  RUY_DCHECK_EQ(Layout::kCols, 16);
  RUY_DCHECK_EQ(Layout::kRows, 4);

  const int non_trailing_blocks = (src_rows & ~31) >> 2;
  // This routine fills half blocks, and typically fills the second halves.
  // Thus packed_ptr is already offset by 8 * 4.
  for (int k = 0; k < non_trailing_blocks; ++k) {
    for (int j = 0; j < (kHalfLayoutCols * Layout::kRows); ++j) {
      packed_ptr[Layout::kCols * Layout::kRows * k + j] = packed_zero_point;
    }
  }
}

inline __m512i LoaduTwo(const std::int8_t* addr_lo,
                        const std::int8_t* addr_hi) {
  __m512i lower_filled = _mm512_castsi256_si512(_mm256_loadu_epi8(addr_lo));
  return _mm512_inserti32x8(lower_filled, _mm256_loadu_epi8(addr_hi), 1);
}

inline __m512i MaskLoaduTwo(__mmask32 row_mask, const __m256i default_value_v,
                            const std::int8_t* addr_lo,
                            const std::int8_t* addr_hi) {
  const __m512i lower_filled = _mm512_castsi256_si512(
      _mm256_mask_loadu_epi8(default_value_v, row_mask, addr_lo));
  return _mm512_inserti32x8(
      lower_filled, _mm256_mask_loadu_epi8(default_value_v, row_mask, addr_hi),
      1);
}

inline void HalfPack8bitAvx512(const std::int8_t* src_ptr,
                               std::int8_t input_xor,
                               const std::int8_t* zerobuf, int src_stride,
                               int remaining_src_cols, int src_rows,
                               std::int8_t* packed_ptr, std::int32_t* sums_ptr,
                               std::int8_t* trailing_buf) {
  using Layout = PackImpl8bitAvx512::Layout;
  RUY_DCHECK_EQ(Layout::kCols, 16);
  RUY_DCHECK_EQ(Layout::kRows, 4);
  // Each Layout::Rows is 4 contiguous input, contiguous packed elements.
  // We process 8 of these chunks at a time, padding short input chunks.
  constexpr int kNumRowChunks = 8;
  constexpr int kNumChunkedSrcRows = kNumRowChunks * Layout::kRows;

  const std::int8_t* src_ptr0 = src_ptr;
  const std::int8_t* src_ptr1 = src_ptr0 + src_stride;
  const std::int8_t* src_ptr2 = src_ptr1 + src_stride;
  const std::int8_t* src_ptr3 = src_ptr2 + src_stride;
  const std::int8_t* src_ptr4 = src_ptr3 + src_stride;
  const std::int8_t* src_ptr5 = src_ptr4 + src_stride;
  const std::int8_t* src_ptr6 = src_ptr5 + src_stride;
  const std::int8_t* src_ptr7 = src_ptr6 + src_stride;
  std::int64_t src_inc0 = kNumChunkedSrcRows;
  std::int64_t src_inc1 = kNumChunkedSrcRows;
  std::int64_t src_inc2 = kNumChunkedSrcRows;
  std::int64_t src_inc3 = kNumChunkedSrcRows;
  std::int64_t src_inc4 = kNumChunkedSrcRows;
  std::int64_t src_inc5 = kNumChunkedSrcRows;
  std::int64_t src_inc6 = kNumChunkedSrcRows;
  std::int64_t src_inc7 = kNumChunkedSrcRows;
  // Handle cases where source does not have kHalfLayoutCols (8) columns.
  if (remaining_src_cols < 8) {
    if (remaining_src_cols <= 0) {
      src_ptr0 = zerobuf;
      src_inc0 = 0;
    }
    if (remaining_src_cols <= 1) {
      src_ptr1 = zerobuf;
      src_inc1 = 0;
    }
    if (remaining_src_cols <= 2) {
      src_ptr2 = zerobuf;
      src_inc2 = 0;
    }
    if (remaining_src_cols <= 3) {
      src_ptr3 = zerobuf;
      src_inc3 = 0;
    }
    if (remaining_src_cols <= 4) {
      src_ptr4 = zerobuf;
      src_inc4 = 0;
    }
    if (remaining_src_cols <= 5) {
      src_ptr5 = zerobuf;
      src_inc5 = 0;
    }
    if (remaining_src_cols <= 6) {
      src_ptr6 = zerobuf;
      src_inc6 = 0;
    }
    src_ptr7 = zerobuf;
    src_inc7 = 0;
  }

  const std::int8_t zero_point = zerobuf[0];

  if (sums_ptr) {
    // i: kHalfLayoutCols.
    for (int i = 0; i < 8; ++i) {
      sums_ptr[i] = 0;
    }
  }
  std::int32_t sums_adjustment = 0;
  const __m512i ones_16bit = _mm512_set1_epi16(1);
  __m512i sums_8x2_32bit = _mm512_set1_epi32(0);

  // The overall packing effectively pads the source rows to
  // (src_rows + 63) & ~63. The iteration over k may skip when m=1, and then we
  // only pack for (src_rows + 31) & ~31. When there is an incomplete
  // destination block, this is stored into trailing_buf instead of packed_ptr.
  for (int k = 0; k < src_rows; k += 2 * kNumChunkedSrcRows) {
    // m: {0, 1} for 2 chunks of rows.
    for (int m = 0; m < 2; ++m) {
      // Available source rows.
      // If this is less than 0 (for m=1), we skip, having filled trailing
      // buffer for m=0. Also, if source rows is zero on m=1, then we filled
      // exactly to the end of the column in the packed buffer.
      const int available_src_rows = src_rows - k - m * kNumChunkedSrcRows;
      // Effectively,
      // available rows = std::max(0, std::min(8, src_rows - k - 8 * 4 * m));
      // treat each case separately.
      if (available_src_rows >= kNumChunkedSrcRows) {
        // i: chunks, s: Layout::Rows.
        if (sums_ptr) {
          __m512i t0, t1, t2, t3;
          __m512i r0, r1, r2, r3;
          const __m512i input_xor_v = _mm512_set1_epi8(input_xor);

          t0 = LoaduTwo(src_ptr0, src_ptr4);
          t1 = LoaduTwo(src_ptr1, src_ptr5);
          t2 = LoaduTwo(src_ptr2, src_ptr6);
          t3 = LoaduTwo(src_ptr3, src_ptr7);

          r0 = _mm512_unpacklo_epi32(t0, t1);
          r2 = _mm512_unpackhi_epi32(t0, t1);
          r1 = _mm512_unpacklo_epi32(t2, t3);
          r3 = _mm512_unpackhi_epi32(t2, t3);

          t0 = _mm512_unpacklo_epi64(r0, r1);
          t2 = _mm512_unpackhi_epi64(r0, r1);
          t1 = _mm512_unpacklo_epi64(r2, r3);
          t3 = _mm512_unpackhi_epi64(r2, r3);

          r0 = _mm512_shuffle_i32x4(t0, t1, 0x88);
          r1 = _mm512_shuffle_i32x4(t0, t1, 0xdd);
          r2 = _mm512_shuffle_i32x4(t2, t3, 0x88);
          r3 = _mm512_shuffle_i32x4(t2, t3, 0xdd);

          r0 = _mm512_xor_si512(r0, input_xor_v);
          r1 = _mm512_xor_si512(r1, input_xor_v);
          r2 = _mm512_xor_si512(r2, input_xor_v);
          r3 = _mm512_xor_si512(r3, input_xor_v);

          const __m256i r0_0 = _mm512_castsi512_si256(r0);
          const __m256i r0_1 = _mm512_extracti32x8_epi32(r0, 1);
          const __m256i r1_0 = _mm512_castsi512_si256(r1);
          const __m256i r1_1 = _mm512_extracti32x8_epi32(r1, 1);
          const __m256i r2_0 = _mm512_castsi512_si256(r2);
          const __m256i r2_1 = _mm512_extracti32x8_epi32(r2, 1);
          const __m256i r3_0 = _mm512_castsi512_si256(r3);
          const __m256i r3_1 = _mm512_extracti32x8_epi32(r3, 1);

          __m512i sums_8x4_16bit;
          sums_8x4_16bit = _mm512_cvtepi8_epi16(r0_0);
          sums_8x4_16bit =
              _mm512_add_epi16(sums_8x4_16bit, _mm512_cvtepi8_epi16(r0_1));
          sums_8x4_16bit =
              _mm512_add_epi16(sums_8x4_16bit, _mm512_cvtepi8_epi16(r1_0));
          sums_8x4_16bit =
              _mm512_add_epi16(sums_8x4_16bit, _mm512_cvtepi8_epi16(r1_1));
          sums_8x4_16bit =
              _mm512_add_epi16(sums_8x4_16bit, _mm512_cvtepi8_epi16(r2_0));
          sums_8x4_16bit =
              _mm512_add_epi16(sums_8x4_16bit, _mm512_cvtepi8_epi16(r2_1));
          sums_8x4_16bit =
              _mm512_add_epi16(sums_8x4_16bit, _mm512_cvtepi8_epi16(r3_0));
          sums_8x4_16bit =
              _mm512_add_epi16(sums_8x4_16bit, _mm512_cvtepi8_epi16(r3_1));
          // The sums have been performed across columns, and now we have
          // 4x16-bit sums packed together. We use madd for pairwise 32-bit
          // sums.
          const __m512i sums_8x2_32bit_new =
              _mm512_madd_epi16(sums_8x4_16bit, ones_16bit);
          sums_8x2_32bit = _mm512_add_epi32(sums_8x2_32bit, sums_8x2_32bit_new);

          _mm256_storeu_epi8(packed_ptr + 0 * 16 * 4, r0_0);
          _mm256_storeu_epi8(packed_ptr + 2 * 16 * 4, r0_1);
          _mm256_storeu_epi8(packed_ptr + 4 * 16 * 4, r1_0);
          _mm256_storeu_epi8(packed_ptr + 6 * 16 * 4, r1_1);
          _mm256_storeu_epi8(packed_ptr + 1 * 16 * 4, r2_0);
          _mm256_storeu_epi8(packed_ptr + 3 * 16 * 4, r2_1);
          _mm256_storeu_epi8(packed_ptr + 5 * 16 * 4, r3_0);
          _mm256_storeu_epi8(packed_ptr + 7 * 16 * 4, r3_1);
        } else {
          __m512i t0, t1, t2, t3;
          __m512i r0, r1, r2, r3;
          const __m512i input_xor_v = _mm512_set1_epi8(input_xor);

          t0 = LoaduTwo(src_ptr0, src_ptr4);
          t1 = LoaduTwo(src_ptr1, src_ptr5);
          t2 = LoaduTwo(src_ptr2, src_ptr6);
          t3 = LoaduTwo(src_ptr3, src_ptr7);

          r0 = _mm512_unpacklo_epi32(t0, t1);
          r2 = _mm512_unpackhi_epi32(t0, t1);
          r1 = _mm512_unpacklo_epi32(t2, t3);
          r3 = _mm512_unpackhi_epi32(t2, t3);

          t0 = _mm512_unpacklo_epi64(r0, r1);
          t2 = _mm512_unpackhi_epi64(r0, r1);
          t1 = _mm512_unpacklo_epi64(r2, r3);
          t3 = _mm512_unpackhi_epi64(r2, r3);

          r0 = _mm512_shuffle_i32x4(t0, t1, 0x88);
          r1 = _mm512_shuffle_i32x4(t0, t1, 0xdd);
          r2 = _mm512_shuffle_i32x4(t2, t3, 0x88);
          r3 = _mm512_shuffle_i32x4(t2, t3, 0xdd);

          r0 = _mm512_xor_si512(r0, input_xor_v);
          r1 = _mm512_xor_si512(r1, input_xor_v);
          r2 = _mm512_xor_si512(r2, input_xor_v);
          r3 = _mm512_xor_si512(r3, input_xor_v);

          const __m256i r0_0 = _mm512_castsi512_si256(r0);
          const __m256i r0_1 = _mm512_extracti32x8_epi32(r0, 1);
          const __m256i r1_0 = _mm512_castsi512_si256(r1);
          const __m256i r1_1 = _mm512_extracti32x8_epi32(r1, 1);
          const __m256i r2_0 = _mm512_castsi512_si256(r2);
          const __m256i r2_1 = _mm512_extracti32x8_epi32(r2, 1);
          const __m256i r3_0 = _mm512_castsi512_si256(r3);
          const __m256i r3_1 = _mm512_extracti32x8_epi32(r3, 1);
          _mm256_storeu_epi8(packed_ptr + 0 * 16 * 4, r0_0);
          _mm256_storeu_epi8(packed_ptr + 2 * 16 * 4, r0_1);
          _mm256_storeu_epi8(packed_ptr + 4 * 16 * 4, r1_0);
          _mm256_storeu_epi8(packed_ptr + 6 * 16 * 4, r1_1);
          _mm256_storeu_epi8(packed_ptr + 1 * 16 * 4, r2_0);
          _mm256_storeu_epi8(packed_ptr + 3 * 16 * 4, r2_1);
          _mm256_storeu_epi8(packed_ptr + 5 * 16 * 4, r3_0);
          _mm256_storeu_epi8(packed_ptr + 7 * 16 * 4, r3_1);
        }
      } else if (available_src_rows > 0) {
        RUY_DCHECK_LT(available_src_rows >> 2, kNumChunkedSrcRows);
        const __mmask32 row_mask =
            (static_cast<std::uint64_t>(1) << available_src_rows) - 1;

        // We do not care what goes into the trailing buffer, but we want
        // in_data[...] ^ input_xor == 0 for irrelevant values in the summation.
        //
        // We compensate for padding-with-zero_point by initializing the
        // summations with the compensating offset, effectively
        // ((input_xor ^ input_xor) - (zero_point ^ input_xor)) *
        //                         4 * (8 - ((available_src_rows + 3) >> 2)).
        //
        // Note that (zero_point ^ input_xor) is performed in 8-bits and then
        // cast.
        sums_adjustment += -(zero_point ^ input_xor) * 4 *
                           (8 - ((available_src_rows + 3) >> 2));

        __m512i t0, t1, t2, t3;
        __m512i r0, r1, r2, r3;
        const __m512i input_xor_v = _mm512_set1_epi8(input_xor);
        const __m256i zero_point_v = _mm256_set1_epi8(zero_point);

        t0 = MaskLoaduTwo(row_mask, zero_point_v, src_ptr0, src_ptr4);
        t1 = MaskLoaduTwo(row_mask, zero_point_v, src_ptr1, src_ptr5);
        t2 = MaskLoaduTwo(row_mask, zero_point_v, src_ptr2, src_ptr6);
        t3 = MaskLoaduTwo(row_mask, zero_point_v, src_ptr3, src_ptr7);

        r0 = _mm512_unpacklo_epi32(t0, t1);
        r2 = _mm512_unpackhi_epi32(t0, t1);
        r1 = _mm512_unpacklo_epi32(t2, t3);
        r3 = _mm512_unpackhi_epi32(t2, t3);

        t0 = _mm512_unpacklo_epi64(r0, r1);
        t2 = _mm512_unpackhi_epi64(r0, r1);
        t1 = _mm512_unpacklo_epi64(r2, r3);
        t3 = _mm512_unpackhi_epi64(r2, r3);

        r0 = _mm512_shuffle_i32x4(t0, t1, 0x88);
        r1 = _mm512_shuffle_i32x4(t0, t1, 0xdd);
        r2 = _mm512_shuffle_i32x4(t2, t3, 0x88);
        r3 = _mm512_shuffle_i32x4(t2, t3, 0xdd);

        r0 = _mm512_xor_si512(r0, input_xor_v);
        r1 = _mm512_xor_si512(r1, input_xor_v);
        r2 = _mm512_xor_si512(r2, input_xor_v);
        r3 = _mm512_xor_si512(r3, input_xor_v);

        const __m256i r0_0 = _mm512_castsi512_si256(r0);
        const __m256i r0_1 = _mm512_extracti32x8_epi32(r0, 1);
        const __m256i r1_0 = _mm512_castsi512_si256(r1);
        const __m256i r1_1 = _mm512_extracti32x8_epi32(r1, 1);
        const __m256i r2_0 = _mm512_castsi512_si256(r2);
        const __m256i r2_1 = _mm512_extracti32x8_epi32(r2, 1);
        const __m256i r3_0 = _mm512_castsi512_si256(r3);
        const __m256i r3_1 = _mm512_extracti32x8_epi32(r3, 1);

        __m512i sums_8x4_16bit;
        sums_8x4_16bit = _mm512_cvtepi8_epi16(r0_0);
        sums_8x4_16bit =
            _mm512_add_epi16(sums_8x4_16bit, _mm512_cvtepi8_epi16(r0_1));
        sums_8x4_16bit =
            _mm512_add_epi16(sums_8x4_16bit, _mm512_cvtepi8_epi16(r1_0));
        sums_8x4_16bit =
            _mm512_add_epi16(sums_8x4_16bit, _mm512_cvtepi8_epi16(r1_1));
        sums_8x4_16bit =
            _mm512_add_epi16(sums_8x4_16bit, _mm512_cvtepi8_epi16(r2_0));
        sums_8x4_16bit =
            _mm512_add_epi16(sums_8x4_16bit, _mm512_cvtepi8_epi16(r2_1));
        sums_8x4_16bit =
            _mm512_add_epi16(sums_8x4_16bit, _mm512_cvtepi8_epi16(r3_0));
        sums_8x4_16bit =
            _mm512_add_epi16(sums_8x4_16bit, _mm512_cvtepi8_epi16(r3_1));
        // The sums have been performed across columns, and now we have
        // 4x16-bit sums packed together. We use madd for pairwise 32-bit
        // sums.
        const __m512i sums_8x2_32bit_new =
            _mm512_madd_epi16(sums_8x4_16bit, ones_16bit);
        sums_8x2_32bit = _mm512_add_epi32(sums_8x2_32bit, sums_8x2_32bit_new);

        _mm256_storeu_epi8(trailing_buf + 0 * 16 * 4, r0_0);
        _mm256_storeu_epi8(trailing_buf + 2 * 16 * 4, r0_1);
        _mm256_storeu_epi8(trailing_buf + 4 * 16 * 4, r1_0);
        _mm256_storeu_epi8(trailing_buf + 6 * 16 * 4, r1_1);
        _mm256_storeu_epi8(trailing_buf + 1 * 16 * 4, r2_0);
        _mm256_storeu_epi8(trailing_buf + 3 * 16 * 4, r2_1);
        _mm256_storeu_epi8(trailing_buf + 5 * 16 * 4, r3_0);
        _mm256_storeu_epi8(trailing_buf + 7 * 16 * 4, r3_1);
      }

      packed_ptr += 16 * kNumChunkedSrcRows;
      src_ptr0 += src_inc0;
      src_ptr1 += src_inc1;
      src_ptr2 += src_inc2;
      src_ptr3 += src_inc3;
      src_ptr4 += src_inc4;
      src_ptr5 += src_inc5;
      src_ptr6 += src_inc6;
      src_ptr7 += src_inc7;
    }
  }

  if (sums_ptr) {
    const __m256i sums_adjustment_v = _mm256_set1_epi32(sums_adjustment);

    __m256i sums = _mm256_loadu_epi32(sums_ptr);
    const __m512i idx =
        _mm512_set_epi32(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);

    // We earlier used madd for pairwise 32-bit sums, and now we deinterlace the
    // neighbours, finshing up by adding them to the stored accumulated sums.
    const __m512i sums_2x8_32bit =
        _mm512_permutexvar_epi32(idx, sums_8x2_32bit);
    sums = _mm256_add_epi32(sums, sums_adjustment_v);
    sums = _mm256_add_epi32(sums, _mm512_castsi512_si256(sums_2x8_32bit));
    sums = _mm256_add_epi32(sums, _mm512_extracti32x8_epi32(sums_2x8_32bit, 1));

    _mm256_storeu_epi32(sums_ptr, sums);
  }
}

inline __m512 LoaduTwo(const float* addr_lo, const float* addr_hi) {
  const __m512 lower_filled = _mm512_castps256_ps512(_mm256_loadu_ps(addr_lo));
  return _mm512_insertf32x8(lower_filled, _mm256_loadu_ps(addr_hi), 1);
}

inline __m512 MaskLoaduTwo(__mmask8 row_mask, const float* addr_lo,
                           const float* addr_hi) {
  const __m512 lower_filled =
      _mm512_castps256_ps512(_mm256_maskz_loadu_ps(row_mask, addr_lo));
  return _mm512_insertf32x8(lower_filled,
                            _mm256_maskz_loadu_ps(row_mask, addr_hi), 1);
}

inline __m512 Mm512UnpackloPsx2(const __m512 a, const __m512 b) {
  return _mm512_castpd_ps(
      _mm512_unpacklo_pd(_mm512_castps_pd(a), _mm512_castps_pd(b)));
}

inline __m512 Mm512UnpackhiPsx2(const __m512 a, const __m512 b) {
  return _mm512_castpd_ps(
      _mm512_unpackhi_pd(_mm512_castps_pd(a), _mm512_castps_pd(b)));
}

inline void HalfPackFloatAvx512(const float* src_ptr, const float* zerobuf,
                                int src_stride, int remaining_src_cols,
                                int src_rows, float* packed_ptr,
                                float* trailing_buf) {
  const float* src_ptr0 = src_ptr;
  const float* src_ptr1 = src_ptr0 + src_stride;
  const float* src_ptr2 = src_ptr1 + src_stride;
  const float* src_ptr3 = src_ptr2 + src_stride;
  const float* src_ptr4 = src_ptr3 + src_stride;
  const float* src_ptr5 = src_ptr4 + src_stride;
  const float* src_ptr6 = src_ptr5 + src_stride;
  const float* src_ptr7 = src_ptr6 + src_stride;
  std::int64_t src_inc0 = 8;
  std::int64_t src_inc1 = 8;
  std::int64_t src_inc2 = 8;
  std::int64_t src_inc3 = 8;
  std::int64_t src_inc4 = 8;
  std::int64_t src_inc5 = 8;
  std::int64_t src_inc6 = 8;
  std::int64_t src_inc7 = 8;
  if (remaining_src_cols < 8) {
    if (remaining_src_cols <= 0) {
      src_ptr0 = zerobuf;
      src_inc0 = 0;
    }
    if (remaining_src_cols <= 1) {
      src_ptr1 = zerobuf;
      src_inc1 = 0;
    }
    if (remaining_src_cols <= 2) {
      src_ptr2 = zerobuf;
      src_inc2 = 0;
    }
    if (remaining_src_cols <= 3) {
      src_ptr3 = zerobuf;
      src_inc3 = 0;
    }
    if (remaining_src_cols <= 4) {
      src_ptr4 = zerobuf;
      src_inc4 = 0;
    }
    if (remaining_src_cols <= 5) {
      src_ptr5 = zerobuf;
      src_inc5 = 0;
    }
    if (remaining_src_cols <= 6) {
      src_ptr6 = zerobuf;
      src_inc6 = 0;
    }
    src_ptr7 = zerobuf;
    src_inc7 = 0;
  }

  for (int k = 0; k < src_rows; k += 16) {
    for (int m = 0; m < 2; ++m) {
      const int available_src_rows = src_rows - k - 8 * m;
      // Effectively,
      // available_src_rows = std::max(0, std::min(8, src_rows - k - 8 * m));
      // but treat each case separately.
      if (available_src_rows > 7) {
        __m512 t0, t1, t2, t3;
        __m512 r0, r1, r2, r3;

        t0 = LoaduTwo(src_ptr0, src_ptr4);
        t1 = LoaduTwo(src_ptr1, src_ptr5);
        t2 = LoaduTwo(src_ptr2, src_ptr6);
        t3 = LoaduTwo(src_ptr3, src_ptr7);

        r0 = _mm512_unpacklo_ps(t0, t1);
        r2 = _mm512_unpackhi_ps(t0, t1);
        r1 = _mm512_unpacklo_ps(t2, t3);
        r3 = _mm512_unpackhi_ps(t2, t3);

        t0 = Mm512UnpackloPsx2(r0, r1);
        t2 = Mm512UnpackhiPsx2(r0, r1);
        t1 = Mm512UnpackloPsx2(r2, r3);
        t3 = Mm512UnpackhiPsx2(r2, r3);

        r0 = _mm512_shuffle_f32x4(t0, t1, 0x88);
        r1 = _mm512_shuffle_f32x4(t0, t1, 0xdd);
        r2 = _mm512_shuffle_f32x4(t2, t3, 0x88);
        r3 = _mm512_shuffle_f32x4(t2, t3, 0xdd);

        _mm256_storeu_ps(packed_ptr + 0 * 16, _mm512_castps512_ps256(r0));
        _mm256_storeu_ps(packed_ptr + 2 * 16, _mm512_extractf32x8_ps(r0, 1));
        _mm256_storeu_ps(packed_ptr + 4 * 16, _mm512_castps512_ps256(r1));
        _mm256_storeu_ps(packed_ptr + 6 * 16, _mm512_extractf32x8_ps(r1, 1));
        _mm256_storeu_ps(packed_ptr + 1 * 16, _mm512_castps512_ps256(r2));
        _mm256_storeu_ps(packed_ptr + 3 * 16, _mm512_extractf32x8_ps(r2, 1));
        _mm256_storeu_ps(packed_ptr + 5 * 16, _mm512_castps512_ps256(r3));
        _mm256_storeu_ps(packed_ptr + 7 * 16, _mm512_extractf32x8_ps(r3, 1));
      } else if (available_src_rows > 0) {
        const __mmask8 row_mask =
            (static_cast<std::uint32_t>(1) << available_src_rows) - 1;

        __m512 t0, t1, t2, t3;
        __m512 r0, r1, r2, r3;

        t0 = MaskLoaduTwo(row_mask, src_ptr0, src_ptr4);
        t1 = MaskLoaduTwo(row_mask, src_ptr1, src_ptr5);
        t2 = MaskLoaduTwo(row_mask, src_ptr2, src_ptr6);
        t3 = MaskLoaduTwo(row_mask, src_ptr3, src_ptr7);

        r0 = _mm512_unpacklo_ps(t0, t1);
        r2 = _mm512_unpackhi_ps(t0, t1);
        r1 = _mm512_unpacklo_ps(t2, t3);
        r3 = _mm512_unpackhi_ps(t2, t3);

        t0 = Mm512UnpackloPsx2(r0, r1);
        t2 = Mm512UnpackhiPsx2(r0, r1);
        t1 = Mm512UnpackloPsx2(r2, r3);
        t3 = Mm512UnpackhiPsx2(r2, r3);

        r0 = _mm512_shuffle_f32x4(t0, t1, 0x88);
        r1 = _mm512_shuffle_f32x4(t0, t1, 0xdd);
        r2 = _mm512_shuffle_f32x4(t2, t3, 0x88);
        r3 = _mm512_shuffle_f32x4(t2, t3, 0xdd);

        _mm256_storeu_ps(trailing_buf + 0 * 16, _mm512_castps512_ps256(r0));
        _mm256_storeu_ps(trailing_buf + 2 * 16, _mm512_extractf32x8_ps(r0, 1));
        _mm256_storeu_ps(trailing_buf + 4 * 16, _mm512_castps512_ps256(r1));
        _mm256_storeu_ps(trailing_buf + 6 * 16, _mm512_extractf32x8_ps(r1, 1));
        _mm256_storeu_ps(trailing_buf + 1 * 16, _mm512_castps512_ps256(r2));
        _mm256_storeu_ps(trailing_buf + 3 * 16, _mm512_extractf32x8_ps(r2, 1));
        _mm256_storeu_ps(trailing_buf + 5 * 16, _mm512_castps512_ps256(r3));
        // Do not store _mm512_extractf32x8_ps(r3, 1).
      }

      packed_ptr += 16 * 8;
      src_ptr0 += src_inc0;
      src_ptr1 += src_inc1;
      src_ptr2 += src_inc2;
      src_ptr3 += src_inc3;
      src_ptr4 += src_inc4;
      src_ptr5 += src_inc5;
      src_ptr6 += src_inc6;
      src_ptr7 += src_inc7;
    }
  }
}

inline void ZeroHalfFloatAvx512(int src_rows, float* packed_ptr) {
  const int non_trailing_rows = src_rows & ~7;
  for (int k = 0; k < non_trailing_rows; ++k) {
    for (int j = 0; j < 8; ++j) {
      packed_ptr[j] = 0.0f;
    }
    packed_ptr += 16;
  }
}

}  // namespace.

void Pack8bitAvx512(const std::int8_t* src_ptr, std::int8_t input_xor,
                    const std::int8_t* zerobuf, int src_stride,
                    int remaining_src_cols, int src_rows,
                    std::int8_t* packed_ptr, std::int32_t* sums_ptr) {
  profiler::ScopeLabel label("Pack kAvx512 8bit");

  using Layout = PackImpl8bitAvx512::Layout;
  constexpr int kHalfBlockOffset = 32;
  RUY_DCHECK_EQ(kHalfBlockOffset * 2, Layout::kRows * Layout::kCols);
  static constexpr int kHalfLayoutCols =
      PackImpl8bitAvx512::kHalfLayoutCols;  // Half the number of cols in a
                                            // block.
  RUY_DCHECK_EQ(kHalfLayoutCols, 8);
  RUY_DCHECK_EQ(Layout::kCols, 16);
  RUY_DCHECK_EQ(Layout::kRows, 4);

  // Each Layout::Rows is 4 contiguous input, contiguous packed elements.
  // We process 8 of these chunks at a time, padding short input chunks.
  constexpr int kNumRowChunks = 8;

  // Each packed block is 4*16, and there are normally 8. The trailing block is
  // only slightly shorter.
  constexpr int kTrailingBufSize =
      kNumRowChunks * Layout::kCols * Layout::kRows;
  std::int8_t trailing_buf[kTrailingBufSize];
  memset(trailing_buf, 0, kTrailingBufSize * sizeof(std::int8_t));

  std::int32_t* second_sums_ptr =
      sums_ptr ? sums_ptr + kHalfLayoutCols : nullptr;
  if (remaining_src_cols > kHalfLayoutCols) {
    HalfPack8bitAvx512(src_ptr, input_xor, zerobuf, src_stride,
                       remaining_src_cols, src_rows, packed_ptr, sums_ptr,
                       trailing_buf);
    HalfPack8bitAvx512(src_ptr + src_stride * kHalfLayoutCols, input_xor,
                       zerobuf, src_stride,
                       remaining_src_cols - kHalfLayoutCols, src_rows,
                       packed_ptr + kHalfBlockOffset, second_sums_ptr,
                       trailing_buf + kHalfBlockOffset);
  } else {
    HalfPack8bitAvx512(src_ptr, input_xor, zerobuf, src_stride,
                       remaining_src_cols, src_rows, packed_ptr, sums_ptr,
                       trailing_buf);
    ZeroHalf8bitAvx512(src_rows, zerobuf[0] ^ input_xor,
                       packed_ptr + kHalfBlockOffset);
    // The kernel may not need the second half-blocks sums to be set.
    if (second_sums_ptr) {
      for (int i = 0; i < kHalfLayoutCols; ++i) {
        second_sums_ptr[i] = (zerobuf[0] ^ input_xor) * ((src_rows + 3) & ~3);
      }
    }
  }
  constexpr int kChunkedRowMask = kNumRowChunks * Layout::kRows - 1;
  const bool trailing_data = (src_rows & kChunkedRowMask) > 0;
  // If the number of source rows is not a multiple of kChunkedRowMask, there
  // will be data in the trailing buffer,
  if (trailing_data > 0) {
    const int non_trailing_rows = src_rows & ~kChunkedRowMask;
    // Destination "rows" are padded to next highest multiple of Layout::kRows.
    const int dst_rows = (src_rows + 3) & ~3;
    const int trailing_rows = dst_rows - non_trailing_rows;
    memcpy(packed_ptr + Layout::kCols * non_trailing_rows, trailing_buf,
           Layout::kCols * trailing_rows * sizeof(std::int8_t));
  }
}

void PackFloatAvx512(const float* src_ptr, const float* zerobuf, int src_stride,
                     int remaining_src_cols, int src_rows, float* packed_ptr) {
  profiler::ScopeLabel label("Pack kAvx512 float");
  float trailing_buf[7 * 16];
  if (remaining_src_cols > 8) {
    HalfPackFloatAvx512(src_ptr, zerobuf, src_stride, remaining_src_cols,
                        src_rows, packed_ptr, trailing_buf);
    HalfPackFloatAvx512(src_ptr + src_stride * 8, zerobuf, src_stride,
                        remaining_src_cols - 8, src_rows, packed_ptr + 8,
                        trailing_buf + 8);
  } else {
    memset(trailing_buf, 0, sizeof(trailing_buf));
    HalfPackFloatAvx512(src_ptr, zerobuf, src_stride, remaining_src_cols,
                        src_rows, packed_ptr, trailing_buf);
    ZeroHalfFloatAvx512(src_rows, packed_ptr + 8);
  }
  const int trailing_rows = src_rows & 7;
  if (trailing_rows > 0) {
    const int non_trailing_rows = src_rows & ~7;
    memcpy(packed_ptr + 16 * non_trailing_rows, trailing_buf,
           16 * trailing_rows * sizeof(float));
  }
}

#endif  // RUY_PLATFORM(AVX512) && RUY_OPT_ENABLED(RUY_OPT_INTRINSICS)

}  // namespace ruy
