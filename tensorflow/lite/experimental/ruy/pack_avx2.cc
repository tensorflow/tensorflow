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

#if RUY_PLATFORM(AVX2) && RUY_OPT_ENABLED(RUY_OPT_INTRINSICS)
#include <immintrin.h>  // IWYU pragma: keep
#endif

namespace ruy {

#if !(RUY_PLATFORM(AVX2) && RUY_OPT_ENABLED(RUY_OPT_ASM))

void Pack8bitAvx2(const std::int8_t* src_ptr, std::int8_t input_xor,
                  const std::int8_t* zerobuf, int src_stride,
                  int remaining_src_cols, int src_rows, std::int8_t* packed_ptr,
                  std::int32_t* sums_ptr) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

void PackFloatAvx2(const float* src_ptr, const float* zerobuf, int src_stride,
                   int remaining_src_cols, int src_rows, float* packed_ptr) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

#else  // RUY_PLATFORM(AVX2) && RUY_OPT_ENABLED(RUY_OPT_ASM)

// The first int8_t template parameter is arbitrary: this routine is common to
// all 8-bit source matrix types.
using PackImpl8bitAvx2 =
    PackImpl<Path::kAvx2, FixedKernelLayout<Order::kColMajor, 4, 8>,
             std::int8_t, std::int8_t, std::int32_t>;

using PackImplFloatAvx2 =
    PackImpl<Path::kAvx2, FixedKernelLayout<Order::kRowMajor, 1, 8>, float,
             float, float>;

namespace {

inline __m256i MaskLoadu(int available_src_rows, std::int8_t zero_point,
                         const std::int8_t* addr) {
  RUY_DCHECK_LT(available_src_rows, 32);
  __m256i padded_data;

  if (available_src_rows >= 16) {
    __m128i load_hi = _mm_set1_epi8(zero_point);
    __m128i load_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(addr));
    memcpy(&load_hi, addr + 16, available_src_rows - 16);
    padded_data = _mm256_set_m128i(load_hi, load_lo);
  } else {
    __m128i load_hi = _mm_set1_epi8(zero_point);
    __m128i load_lo = load_hi;
    memcpy(&load_lo, addr, available_src_rows);
    padded_data = _mm256_set_m128i(load_hi, load_lo);
  }
  return padded_data;
}

inline void Pack8bitAvx2Packer(const std::int8_t* src_ptr,
                               std::int8_t input_xor,
                               const std::int8_t* zerobuf, int src_stride,
                               int remaining_src_cols, int src_rows,
                               std::int8_t* packed_ptr, std::int32_t* sums_ptr,
                               std::int8_t* trailing_buf) {
  using Layout = PackImpl8bitAvx2::Layout;
  RUY_DCHECK_EQ(Layout::kCols, 8);
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
  // Handle cases where source does not have Layout::kCols (8) columns.
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
    // i: Layout::kCols.
    for (int i = 0; i < 8; ++i) {
      sums_ptr[i] = 0;
    }
  }
  std::int32_t sums_adjustment = 0;
  const __m256i ones_16bit = _mm256_set1_epi16(1);
  __m256i sums_4x2_32bit_lo = _mm256_set1_epi32(0);
  __m256i sums_4x2_32bit_hi = _mm256_set1_epi32(0);

  // The overall packing effectively pads the source rows to
  // (src_rows + 63) & ~63. The iteration over k may skip when m=1, and then we
  // only pack for (src_rows + 31) & ~31. When there is an incomplete
  // destination block, this is stored into trailing_buf instead of packed_ptr.
  for (int k = 0; k < src_rows; k += kNumChunkedSrcRows) {
    // Available source rows.
    // If this is less than 0 (for m=1), we skip, having filled trailing
    // buffer for m=0. Also, if source rows is zero on m=1, then we filled
    // exactly to the end of the column in the packed buffer.
    const int available_src_rows = src_rows - k;
    // Effectively,
    // available rows = std::max(0, std::min(8, src_rows - k));
    // treat each case separately.
    if (available_src_rows >= kNumChunkedSrcRows) {
      if (sums_ptr) {
        __m256i t0, t1, t2, t3, t4, t5, t6, t7;
        __m256i r0, r1, r2, r3, r4, r5, r6, r7;
        const __m256i input_xor_v = _mm256_set1_epi8(input_xor);

        t0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr0));
        t4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr4));
        t1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr1));
        t5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr5));
        t2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr2));
        t6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr6));
        t3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr3));
        t7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr7));

        r0 = _mm256_unpacklo_epi32(t0, t1);
        r4 = _mm256_unpacklo_epi32(t4, t5);
        r2 = _mm256_unpackhi_epi32(t0, t1);
        r6 = _mm256_unpackhi_epi32(t4, t5);
        r1 = _mm256_unpacklo_epi32(t2, t3);
        r5 = _mm256_unpacklo_epi32(t6, t7);
        r3 = _mm256_unpackhi_epi32(t2, t3);
        r7 = _mm256_unpackhi_epi32(t6, t7);

        t0 = _mm256_unpacklo_epi64(r0, r1);
        t4 = _mm256_unpacklo_epi64(r4, r5);
        t2 = _mm256_unpackhi_epi64(r0, r1);
        t6 = _mm256_unpackhi_epi64(r4, r5);
        t1 = _mm256_unpacklo_epi64(r2, r3);
        t5 = _mm256_unpacklo_epi64(r6, r7);
        t3 = _mm256_unpackhi_epi64(r2, r3);
        t7 = _mm256_unpackhi_epi64(r6, r7);

        // The preceding sets of rearrangement operations interleaved by 4 bytes
        // and then by 8 bytes *within* lanes. The following set interleave by
        // 16 bytes (128-bit), operating *between* AVX lanes. For instance (t0,
        // t4) are interleaved to create (r0, r1). This complexity follows from
        // the way that AVX is centered around MM 128-bit lanes.
        r0 = _mm256_permute2x128_si256(t0, t4, 0x20);
        r4 = _mm256_permute2x128_si256(t1, t5, 0x20);
        r1 = _mm256_permute2x128_si256(t0, t4, 0x31);
        r5 = _mm256_permute2x128_si256(t1, t5, 0x31);
        r2 = _mm256_permute2x128_si256(t2, t6, 0x20);
        r6 = _mm256_permute2x128_si256(t3, t7, 0x20);
        r3 = _mm256_permute2x128_si256(t2, t6, 0x31);
        r7 = _mm256_permute2x128_si256(t3, t7, 0x31);

        r0 = _mm256_xor_si256(r0, input_xor_v);
        r1 = _mm256_xor_si256(r1, input_xor_v);
        r2 = _mm256_xor_si256(r2, input_xor_v);
        r3 = _mm256_xor_si256(r3, input_xor_v);
        r4 = _mm256_xor_si256(r4, input_xor_v);
        r5 = _mm256_xor_si256(r5, input_xor_v);
        r6 = _mm256_xor_si256(r6, input_xor_v);
        r7 = _mm256_xor_si256(r7, input_xor_v);

        __m256i sums_4x4_16bit_lo;
        sums_4x4_16bit_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r0));
        sums_4x4_16bit_lo =
            _mm256_add_epi16(sums_4x4_16bit_lo,
                             _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r1)));
        sums_4x4_16bit_lo =
            _mm256_add_epi16(sums_4x4_16bit_lo,
                             _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r2)));
        sums_4x4_16bit_lo =
            _mm256_add_epi16(sums_4x4_16bit_lo,
                             _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r3)));
        sums_4x4_16bit_lo =
            _mm256_add_epi16(sums_4x4_16bit_lo,
                             _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r4)));
        sums_4x4_16bit_lo =
            _mm256_add_epi16(sums_4x4_16bit_lo,
                             _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r5)));
        sums_4x4_16bit_lo =
            _mm256_add_epi16(sums_4x4_16bit_lo,
                             _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r6)));
        sums_4x4_16bit_lo =
            _mm256_add_epi16(sums_4x4_16bit_lo,
                             _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r7)));

        // The sums have been performed across columns, and now we have 4x16-bit
        // sums packed together. We use madd for pairwise 32-bit sums.
        const __m256i sums_4x2_32bit_lo_new =
            _mm256_madd_epi16(sums_4x4_16bit_lo, ones_16bit);
        sums_4x2_32bit_lo =
            _mm256_add_epi32(sums_4x2_32bit_lo, sums_4x2_32bit_lo_new);

        __m256i sums_4x4_16bit_hi;
        sums_4x4_16bit_hi =
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r0, 1));
        sums_4x4_16bit_hi = _mm256_add_epi16(
            sums_4x4_16bit_hi,
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r1, 1)));
        sums_4x4_16bit_hi = _mm256_add_epi16(
            sums_4x4_16bit_hi,
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r2, 1)));
        sums_4x4_16bit_hi = _mm256_add_epi16(
            sums_4x4_16bit_hi,
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r3, 1)));
        sums_4x4_16bit_hi = _mm256_add_epi16(
            sums_4x4_16bit_hi,
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r4, 1)));
        sums_4x4_16bit_hi = _mm256_add_epi16(
            sums_4x4_16bit_hi,
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r5, 1)));
        sums_4x4_16bit_hi = _mm256_add_epi16(
            sums_4x4_16bit_hi,
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r6, 1)));
        sums_4x4_16bit_hi = _mm256_add_epi16(
            sums_4x4_16bit_hi,
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r7, 1)));

        const __m256i sums_4x2_32bit_hi_new =
            _mm256_madd_epi16(sums_4x4_16bit_hi, ones_16bit);
        sums_4x2_32bit_hi =
            _mm256_add_epi32(sums_4x2_32bit_hi, sums_4x2_32bit_hi_new);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed_ptr + 0 * 8 * 4),
                            r0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed_ptr + 2 * 8 * 4),
                            r4);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed_ptr + 4 * 8 * 4),
                            r1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed_ptr + 6 * 8 * 4),
                            r5);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed_ptr + 1 * 8 * 4),
                            r2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed_ptr + 3 * 8 * 4),
                            r6);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed_ptr + 5 * 8 * 4),
                            r3);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed_ptr + 7 * 8 * 4),
                            r7);
      } else {
        __m256i t0, t1, t2, t3, t4, t5, t6, t7;
        __m256i r0, r1, r2, r3, r4, r5, r6, r7;
        const __m256i input_xor_v = _mm256_set1_epi8(input_xor);

        t0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr0));
        t4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr4));
        t1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr1));
        t5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr5));
        t2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr2));
        t6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr6));
        t3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr3));
        t7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr7));

        r0 = _mm256_unpacklo_epi32(t0, t1);
        r4 = _mm256_unpacklo_epi32(t4, t5);
        r2 = _mm256_unpackhi_epi32(t0, t1);
        r6 = _mm256_unpackhi_epi32(t4, t5);
        r1 = _mm256_unpacklo_epi32(t2, t3);
        r5 = _mm256_unpacklo_epi32(t6, t7);
        r3 = _mm256_unpackhi_epi32(t2, t3);
        r7 = _mm256_unpackhi_epi32(t6, t7);

        t0 = _mm256_unpacklo_epi64(r0, r1);
        t4 = _mm256_unpacklo_epi64(r4, r5);
        t2 = _mm256_unpackhi_epi64(r0, r1);
        t6 = _mm256_unpackhi_epi64(r4, r5);
        t1 = _mm256_unpacklo_epi64(r2, r3);
        t5 = _mm256_unpacklo_epi64(r6, r7);
        t3 = _mm256_unpackhi_epi64(r2, r3);
        t7 = _mm256_unpackhi_epi64(r6, r7);

        // The preceding sets of rearrangement operations interleaved by 4 bytes
        // and then by 8 bytes *within* lanes. The following set interleave by
        // 16 bytes (128-bit), operating *between* AVX lanes. For instance (t0,
        // t4) are interleaved to create (r0, r1). This complexity follows from
        // the way that AVX is centered around MM 128-bit lanes.
        r0 = _mm256_permute2x128_si256(t0, t4, 0x20);
        r4 = _mm256_permute2x128_si256(t1, t5, 0x20);
        r1 = _mm256_permute2x128_si256(t0, t4, 0x31);
        r5 = _mm256_permute2x128_si256(t1, t5, 0x31);
        r2 = _mm256_permute2x128_si256(t2, t6, 0x20);
        r6 = _mm256_permute2x128_si256(t3, t7, 0x20);
        r3 = _mm256_permute2x128_si256(t2, t6, 0x31);
        r7 = _mm256_permute2x128_si256(t3, t7, 0x31);

        r0 = _mm256_xor_si256(r0, input_xor_v);
        r1 = _mm256_xor_si256(r1, input_xor_v);
        r2 = _mm256_xor_si256(r2, input_xor_v);
        r3 = _mm256_xor_si256(r3, input_xor_v);
        r4 = _mm256_xor_si256(r4, input_xor_v);
        r5 = _mm256_xor_si256(r5, input_xor_v);
        r6 = _mm256_xor_si256(r6, input_xor_v);
        r7 = _mm256_xor_si256(r7, input_xor_v);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed_ptr + 0 * 8 * 4),
                            r0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed_ptr + 2 * 8 * 4),
                            r4);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed_ptr + 4 * 8 * 4),
                            r1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed_ptr + 6 * 8 * 4),
                            r5);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed_ptr + 1 * 8 * 4),
                            r2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed_ptr + 3 * 8 * 4),
                            r6);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed_ptr + 5 * 8 * 4),
                            r3);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed_ptr + 7 * 8 * 4),
                            r7);
      }
    } else if (available_src_rows > 0) {
      RUY_DCHECK_LT(available_src_rows, kNumChunkedSrcRows);
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
      sums_adjustment +=
          -(zero_point ^ input_xor) * 4 * (8 - ((available_src_rows + 3) >> 2));

      __m256i t0, t1, t2, t3, t4, t5, t6, t7;
      __m256i r0, r1, r2, r3, r4, r5, r6, r7;
      const __m256i input_xor_v = _mm256_set1_epi8(input_xor);

      t0 = MaskLoadu(available_src_rows, zero_point, src_ptr0);
      t4 = MaskLoadu(available_src_rows, zero_point, src_ptr4);
      t1 = MaskLoadu(available_src_rows, zero_point, src_ptr1);
      t5 = MaskLoadu(available_src_rows, zero_point, src_ptr5);
      t2 = MaskLoadu(available_src_rows, zero_point, src_ptr2);
      t6 = MaskLoadu(available_src_rows, zero_point, src_ptr6);
      t3 = MaskLoadu(available_src_rows, zero_point, src_ptr3);
      t7 = MaskLoadu(available_src_rows, zero_point, src_ptr7);

      r0 = _mm256_unpacklo_epi32(t0, t1);
      r4 = _mm256_unpacklo_epi32(t4, t5);
      r2 = _mm256_unpackhi_epi32(t0, t1);
      r6 = _mm256_unpackhi_epi32(t4, t5);
      r1 = _mm256_unpacklo_epi32(t2, t3);
      r5 = _mm256_unpacklo_epi32(t6, t7);
      r3 = _mm256_unpackhi_epi32(t2, t3);
      r7 = _mm256_unpackhi_epi32(t6, t7);

      t0 = _mm256_unpacklo_epi64(r0, r1);
      t4 = _mm256_unpacklo_epi64(r4, r5);
      t2 = _mm256_unpackhi_epi64(r0, r1);
      t6 = _mm256_unpackhi_epi64(r4, r5);
      t1 = _mm256_unpacklo_epi64(r2, r3);
      t5 = _mm256_unpacklo_epi64(r6, r7);
      t3 = _mm256_unpackhi_epi64(r2, r3);
      t7 = _mm256_unpackhi_epi64(r6, r7);

      // The preceding sets of rearrangement operations interleaved by 4 bytes
      // and then by 8 bytes *within* lanes. The following set interleave by
      // 16 bytes (128-bit), operating *between* AVX lanes. For instance (t0,
      // t4) are interleaved to create (r0, r1). This complexity follows from
      // the way that AVX is centered around MM 128-bit lanes.
      r0 = _mm256_permute2x128_si256(t0, t4, 0x20);
      r4 = _mm256_permute2x128_si256(t1, t5, 0x20);
      r1 = _mm256_permute2x128_si256(t0, t4, 0x31);
      r5 = _mm256_permute2x128_si256(t1, t5, 0x31);
      r2 = _mm256_permute2x128_si256(t2, t6, 0x20);
      r6 = _mm256_permute2x128_si256(t3, t7, 0x20);
      r3 = _mm256_permute2x128_si256(t2, t6, 0x31);
      r7 = _mm256_permute2x128_si256(t3, t7, 0x31);

      r0 = _mm256_xor_si256(r0, input_xor_v);
      r1 = _mm256_xor_si256(r1, input_xor_v);
      r2 = _mm256_xor_si256(r2, input_xor_v);
      r3 = _mm256_xor_si256(r3, input_xor_v);
      r4 = _mm256_xor_si256(r4, input_xor_v);
      r5 = _mm256_xor_si256(r5, input_xor_v);
      r6 = _mm256_xor_si256(r6, input_xor_v);
      r7 = _mm256_xor_si256(r7, input_xor_v);

      __m256i sums_4x4_16bit_lo;
      sums_4x4_16bit_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r0));
      sums_4x4_16bit_lo = _mm256_add_epi16(
          sums_4x4_16bit_lo, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r1)));
      sums_4x4_16bit_lo = _mm256_add_epi16(
          sums_4x4_16bit_lo, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r2)));
      sums_4x4_16bit_lo = _mm256_add_epi16(
          sums_4x4_16bit_lo, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r3)));
      sums_4x4_16bit_lo = _mm256_add_epi16(
          sums_4x4_16bit_lo, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r4)));
      sums_4x4_16bit_lo = _mm256_add_epi16(
          sums_4x4_16bit_lo, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r5)));
      sums_4x4_16bit_lo = _mm256_add_epi16(
          sums_4x4_16bit_lo, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r6)));
      sums_4x4_16bit_lo = _mm256_add_epi16(
          sums_4x4_16bit_lo, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r7)));

      // The sums have been performed across columns, and now we have 4x16-bit
      // sums packed together. We use madd for pairwise 32-bit sums.
      const __m256i sums_4x2_32bit_lo_new =
          _mm256_madd_epi16(sums_4x4_16bit_lo, ones_16bit);
      sums_4x2_32bit_lo =
          _mm256_add_epi32(sums_4x2_32bit_lo, sums_4x2_32bit_lo_new);

      __m256i sums_4x4_16bit_hi;
      sums_4x4_16bit_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r0, 1));
      sums_4x4_16bit_hi = _mm256_add_epi16(
          sums_4x4_16bit_hi,
          _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r1, 1)));
      sums_4x4_16bit_hi = _mm256_add_epi16(
          sums_4x4_16bit_hi,
          _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r2, 1)));
      sums_4x4_16bit_hi = _mm256_add_epi16(
          sums_4x4_16bit_hi,
          _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r3, 1)));
      sums_4x4_16bit_hi = _mm256_add_epi16(
          sums_4x4_16bit_hi,
          _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r4, 1)));
      sums_4x4_16bit_hi = _mm256_add_epi16(
          sums_4x4_16bit_hi,
          _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r5, 1)));
      sums_4x4_16bit_hi = _mm256_add_epi16(
          sums_4x4_16bit_hi,
          _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r6, 1)));
      sums_4x4_16bit_hi = _mm256_add_epi16(
          sums_4x4_16bit_hi,
          _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r7, 1)));

      const __m256i sums_4x2_32bit_hi_new =
          _mm256_madd_epi16(sums_4x4_16bit_hi, ones_16bit);
      sums_4x2_32bit_hi =
          _mm256_add_epi32(sums_4x2_32bit_hi, sums_4x2_32bit_hi_new);

      _mm256_storeu_si256(reinterpret_cast<__m256i*>(trailing_buf + 0 * 8 * 4),
                          r0);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(trailing_buf + 2 * 8 * 4),
                          r4);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(trailing_buf + 4 * 8 * 4),
                          r1);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(trailing_buf + 6 * 8 * 4),
                          r5);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(trailing_buf + 1 * 8 * 4),
                          r2);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(trailing_buf + 3 * 8 * 4),
                          r6);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(trailing_buf + 5 * 8 * 4),
                          r3);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(trailing_buf + 7 * 8 * 4),
                          r7);
    }

    packed_ptr += 8 * kNumChunkedSrcRows;
    src_ptr0 += src_inc0;
    src_ptr1 += src_inc1;
    src_ptr2 += src_inc2;
    src_ptr3 += src_inc3;
    src_ptr4 += src_inc4;
    src_ptr5 += src_inc5;
    src_ptr6 += src_inc6;
    src_ptr7 += src_inc7;
  }

  if (sums_ptr) {
    const __m256i sums_adjustment_v = _mm256_set1_epi32(sums_adjustment);

    __m256i sums =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sums_ptr));
    const __m256i idx = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    // We earlier used madd for pairwise 32-bit sums, and now we deinterlace the
    // neighbours, finshing up by adding them to the stored accumulated sums.
    const __m256i sums_2x4_32bit_lo =
        _mm256_permutevar8x32_epi32(sums_4x2_32bit_lo, idx);
    const __m256i sums_2x4_32bit_hi =
        _mm256_permutevar8x32_epi32(sums_4x2_32bit_hi, idx);
    const __m256i sums_2x4_32bit_a =
        _mm256_permute2x128_si256(sums_2x4_32bit_lo, sums_2x4_32bit_hi, 0x20);
    const __m256i sums_2x4_32bit_b =
        _mm256_permute2x128_si256(sums_2x4_32bit_lo, sums_2x4_32bit_hi, 0x31);
    sums = _mm256_add_epi32(sums, sums_adjustment_v);
    sums = _mm256_add_epi32(sums, sums_2x4_32bit_a);
    sums = _mm256_add_epi32(sums, sums_2x4_32bit_b);

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(sums_ptr), sums);
  }
}

inline __m256 Mm256UnpackloPsx2(const __m256 a, const __m256 b) {
  return _mm256_castpd_ps(
      _mm256_unpacklo_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)));
}

inline __m256 Mm256UnpackhiPsx2(const __m256 a, const __m256 b) {
  return _mm256_castpd_ps(
      _mm256_unpackhi_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)));
}

inline void PackFloatAvx2Packer(const float* src_ptr, const float* zerobuf,
                                int src_stride, int remaining_src_cols,
                                int src_rows, float* packed_ptr,
                                float* trailing_buf) {
  RUY_DCHECK_EQ(PackImplFloatAvx2::Layout::kCols, 8);
  RUY_DCHECK_EQ(PackImplFloatAvx2::Layout::kRows, 1);

  // This packing amounts to tranposition of 8x8 blocks.
  static constexpr int kPackCols = 8;  // Source cols packed together.
  static constexpr int kPackRows = 8;  // Short input is padded.

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
  // Handle cases where source does not have kPackDim (8) columns.
  if (remaining_src_cols < kPackCols) {
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

  for (int k = 0; k < src_rows; k += kPackRows) {
    const int available_src_rows = src_rows - k;
    // Effectively,
    // available_src_rows = std::max(0, std::min(kPackDim, src_rows - k));
    // but treat each case separately.
    if (available_src_rows >= kPackRows) {
      __m256 t0, t1, t2, t3, t4, t5, t6, t7;
      __m256 r0, r1, r2, r3, r4, r5, r6, r7;

      t0 = _mm256_loadu_ps(src_ptr0);
      t4 = _mm256_loadu_ps(src_ptr4);
      t1 = _mm256_loadu_ps(src_ptr1);
      t5 = _mm256_loadu_ps(src_ptr5);
      t2 = _mm256_loadu_ps(src_ptr2);
      t6 = _mm256_loadu_ps(src_ptr6);
      t3 = _mm256_loadu_ps(src_ptr3);
      t7 = _mm256_loadu_ps(src_ptr7);

      r0 = _mm256_unpacklo_ps(t0, t1);
      r4 = _mm256_unpacklo_ps(t4, t5);
      r2 = _mm256_unpackhi_ps(t0, t1);
      r6 = _mm256_unpackhi_ps(t4, t5);
      r1 = _mm256_unpacklo_ps(t2, t3);
      r5 = _mm256_unpacklo_ps(t6, t7);
      r3 = _mm256_unpackhi_ps(t2, t3);
      r7 = _mm256_unpackhi_ps(t6, t7);

      t0 = Mm256UnpackloPsx2(r0, r1);
      t4 = Mm256UnpackloPsx2(r4, r5);
      t2 = Mm256UnpackhiPsx2(r0, r1);
      t6 = Mm256UnpackhiPsx2(r4, r5);
      t1 = Mm256UnpackloPsx2(r2, r3);
      t5 = Mm256UnpackloPsx2(r6, r7);
      t3 = Mm256UnpackhiPsx2(r2, r3);
      t7 = Mm256UnpackhiPsx2(r6, r7);

      // The preceding sets of rearrangement operations interleaved by 4 bytes
      // and then by 8 bytes *within* lanes. The following set interleave by 16
      // bytes (128-bit), operating *between* AVX lanes. For instance (t0, t4)
      // are interleaved to create (r0, r1). This complexity follows from the
      // way that AVX is centered around MM 128-bit lanes.
      r0 = _mm256_permute2f128_ps(t0, t4, 0x20);
      r4 = _mm256_permute2f128_ps(t1, t5, 0x20);
      r1 = _mm256_permute2f128_ps(t0, t4, 0x31);
      r5 = _mm256_permute2f128_ps(t1, t5, 0x31);
      r2 = _mm256_permute2f128_ps(t2, t6, 0x20);
      r6 = _mm256_permute2f128_ps(t3, t7, 0x20);
      r3 = _mm256_permute2f128_ps(t2, t6, 0x31);
      r7 = _mm256_permute2f128_ps(t3, t7, 0x31);

      _mm256_storeu_ps(packed_ptr + 0 * 8, r0);
      _mm256_storeu_ps(packed_ptr + 2 * 8, r4);
      _mm256_storeu_ps(packed_ptr + 4 * 8, r1);
      _mm256_storeu_ps(packed_ptr + 6 * 8, r5);
      _mm256_storeu_ps(packed_ptr + 1 * 8, r2);
      _mm256_storeu_ps(packed_ptr + 3 * 8, r6);
      _mm256_storeu_ps(packed_ptr + 5 * 8, r3);
      _mm256_storeu_ps(packed_ptr + 7 * 8, r7);
    } else if (available_src_rows > 0) {
      const __m256i series = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
      const __m256i row_mask_v =
          _mm256_cmpgt_epi32(_mm256_set1_epi32(available_src_rows), series);

      __m256 t0, t1, t2, t3, t4, t5, t6, t7;
      __m256 r0, r1, r2, r3, r4, r5, r6, r7;

      t0 = _mm256_maskload_ps(src_ptr0, row_mask_v);
      t4 = _mm256_maskload_ps(src_ptr4, row_mask_v);
      t1 = _mm256_maskload_ps(src_ptr1, row_mask_v);
      t5 = _mm256_maskload_ps(src_ptr5, row_mask_v);
      t2 = _mm256_maskload_ps(src_ptr2, row_mask_v);
      t6 = _mm256_maskload_ps(src_ptr6, row_mask_v);
      t3 = _mm256_maskload_ps(src_ptr3, row_mask_v);
      t7 = _mm256_maskload_ps(src_ptr7, row_mask_v);

      r0 = _mm256_unpacklo_ps(t0, t1);
      r4 = _mm256_unpacklo_ps(t4, t5);
      r2 = _mm256_unpackhi_ps(t0, t1);
      r6 = _mm256_unpackhi_ps(t4, t5);
      r1 = _mm256_unpacklo_ps(t2, t3);
      r5 = _mm256_unpacklo_ps(t6, t7);
      r3 = _mm256_unpackhi_ps(t2, t3);
      r7 = _mm256_unpackhi_ps(t6, t7);

      t0 = Mm256UnpackloPsx2(r0, r1);
      t4 = Mm256UnpackloPsx2(r4, r5);
      t2 = Mm256UnpackhiPsx2(r0, r1);
      t6 = Mm256UnpackhiPsx2(r4, r5);
      t1 = Mm256UnpackloPsx2(r2, r3);
      t5 = Mm256UnpackloPsx2(r6, r7);
      t3 = Mm256UnpackhiPsx2(r2, r3);
      t7 = Mm256UnpackhiPsx2(r6, r7);

      // The preceding sets of rearrangement operations interleaved by 4 bytes
      // and then by 8 bytes *within* lanes. The following set interleave by 16
      // bytes (128-bit), operating *between* AVX lanes. For instance (t0, t4)
      // are interleaved to create (r0, r1). This complexity follows from the
      // way that AVX is centered around MM 128-bit lanes.
      r0 = _mm256_permute2f128_ps(t0, t4, 0x20);
      r4 = _mm256_permute2f128_ps(t1, t5, 0x20);
      r1 = _mm256_permute2f128_ps(t0, t4, 0x31);
      r5 = _mm256_permute2f128_ps(t1, t5, 0x31);
      r2 = _mm256_permute2f128_ps(t2, t6, 0x20);
      r6 = _mm256_permute2f128_ps(t3, t7, 0x20);
      r3 = _mm256_permute2f128_ps(t2, t6, 0x31);
      // r7 no longer needed.

      _mm256_storeu_ps(trailing_buf + 0 * 8, r0);
      _mm256_storeu_ps(trailing_buf + 2 * 8, r4);
      _mm256_storeu_ps(trailing_buf + 4 * 8, r1);
      _mm256_storeu_ps(trailing_buf + 6 * 8, r5);
      _mm256_storeu_ps(trailing_buf + 1 * 8, r2);
      _mm256_storeu_ps(trailing_buf + 3 * 8, r6);
      _mm256_storeu_ps(trailing_buf + 5 * 8, r3);
      // No store to (trailing_buf + 7 * 8), space not allocated.
    }

    packed_ptr += kPackRows * kPackCols;
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

}  // namespace.

void Pack8bitAvx2(const std::int8_t* src_ptr, std::int8_t input_xor,
                  const std::int8_t* zerobuf, int src_stride,
                  int remaining_src_cols, int src_rows, std::int8_t* packed_ptr,
                  std::int32_t* sums_ptr) {
  profiler::ScopeLabel label("Pack kAvx2 8bit");

  using Layout = PackImpl8bitAvx2::Layout;
  RUY_DCHECK_EQ(Layout::kCols, 8);
  RUY_DCHECK_EQ(Layout::kRows, 4);

  // Each Layout::Rows is 4 contiguous input, contiguous packed elements.
  // We process 8 of these chunks at a time, padding short input chunks.
  static constexpr int kNumRowChunks = 8;  // Short input is padded.

  // Each packed block is 4*8, and there are normally 8. The trailing block is
  // only slightly shorter.
  constexpr int kTrailingBufSize =
      kNumRowChunks * Layout::kCols * Layout::kRows;
  std::int8_t trailing_buf[kTrailingBufSize];
  memset(trailing_buf, 0, kTrailingBufSize * sizeof(std::int8_t));

  Pack8bitAvx2Packer(src_ptr, input_xor, zerobuf, src_stride,
                     remaining_src_cols, src_rows, packed_ptr, sums_ptr,
                     trailing_buf);

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

void PackFloatAvx2(const float* src_ptr, const float* zerobuf, int src_stride,
                   int remaining_src_cols, int src_rows, float* packed_ptr) {
  profiler::ScopeLabel label("Pack kAvx2 float");
  static constexpr int kPackCols = 8;  // Source cols packed together.
  static constexpr int kPackRows = 8;  // Short input is padded.
  float trailing_buf[(kPackRows - 1) * kPackCols];
  if (remaining_src_cols < 8) {
    memset(trailing_buf, 0, sizeof(trailing_buf));
  }
  PackFloatAvx2Packer(src_ptr, zerobuf, src_stride, remaining_src_cols,
                      src_rows, packed_ptr, trailing_buf);

  const int trailing_rows = src_rows & (kPackRows - 1);
  if (trailing_rows > 0) {
    const int non_trailing_rows = src_rows & ~(kPackRows - 1);
    memcpy(packed_ptr + kPackCols * non_trailing_rows, trailing_buf,
           kPackCols * trailing_rows * sizeof(float));
  }
}

#endif  // RUY_PLATFORM(AVX2) && RUY_OPT_ENABLED(RUY_OPT_INTRINSICS)

}  // namespace ruy
