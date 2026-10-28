/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_TILED_LINEARIZER_HIGHWAY_H_
#define XLA_PJRT_TILED_LINEARIZER_HIGHWAY_H_

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "absl/numeric/bits.h"
#include "hwy/base.h"
#include "hwy/highway.h"
#include "hwy/print-inl.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"

// SIMD code, compiled once per target.
HWY_BEFORE_NAMESPACE();

namespace xla::HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename D>
HWY_INLINE void FastLoadInterleaved4(
    D d, const hn::TFromD<D>* HWY_RESTRICT unaligned, hn::Vec<D>& v0,
    hn::Vec<D>& v1, hn::Vec<D>& v2, hn::Vec<D>& v3) {
#if HWY_ARCH_ARM
  const hn::RepartitionToWide<decltype(d)> dw;
  using VW = hn::Vec<decltype(dw)>;
  const size_t dw_lanes = hn::Lanes(dw);
  // `hn::LoadInterleaved4` (LD4) is expensive in ARM neoverse-v2. Search
  // `LD4` in
  // https://developer.arm.com/documentation/pjdoc466751330-593177/0002.
  // SSE codegen is the same, but AVX+ codegen is poor and much
  // slower. Hence, the implementation is only specialized for NEON.
  VW v01_lower;
  VW v23_lower;
  hn::LoadInterleaved2(
      dw,
      tsl::safe_reinterpret_cast<const hn::TFromD<decltype(dw)>*>(unaligned) +
          0 * dw_lanes,
      v01_lower, v23_lower);

  VW v01_upper;
  VW v23_upper;
  hn::LoadInterleaved2(
      dw,
      tsl::safe_reinterpret_cast<const hn::TFromD<decltype(dw)>*>(unaligned) +
          2 * dw_lanes,
      v01_upper, v23_upper);

  v0 = hn::ConcatEven(d, hn::BitCast(d, v01_upper), hn::BitCast(d, v01_lower));
  v1 = hn::ConcatOdd(d, hn::BitCast(d, v01_upper), hn::BitCast(d, v01_lower));
  v2 = hn::ConcatEven(d, hn::BitCast(d, v23_upper), hn::BitCast(d, v23_lower));
  v3 = hn::ConcatOdd(d, hn::BitCast(d, v23_upper), hn::BitCast(d, v23_lower));
#else
  hn::LoadInterleaved4(d, unaligned, v0, v1, v2, v3);
#endif
}

template <typename D>
HWY_INLINE void FastStoreInterleaved4(const hn::Vec<D>& v0,
                                      const hn::Vec<D>& v1,
                                      const hn::Vec<D>& v2,
                                      const hn::Vec<D>& v3, D d,
                                      hn::TFromD<D>* HWY_RESTRICT unaligned) {
#if HWY_ARCH_ARM
  // `hn::StoreInterleaved4` (ST4) is expensive in ARM neoverse-v2. Search
  // `ST4` in
  // https://developer.arm.com/documentation/pjdoc466751330-593177/0002.
  // SSE codegen is the same, but AVX+ codegen is poor and much
  // slower. Hence, the implementation is only specialized for NEON.
  const hn::RepartitionToWide<decltype(d)> dw;
  using VW = hn::Vec<decltype(dw)>;
  const size_t dw_lanes = hn::Lanes(dw);
  const VW v01_lower = hn::BitCast(dw, hn::InterleaveWholeLower(d, v0, v1));
  const VW v01_upper = hn::BitCast(dw, hn::InterleaveWholeUpper(d, v0, v1));
  const VW v23_lower = hn::BitCast(dw, hn::InterleaveWholeLower(d, v2, v3));
  const VW v23_upper = hn::BitCast(dw, hn::InterleaveWholeUpper(d, v2, v3));
  hn::StoreInterleaved2(
      v01_lower, v23_lower, dw,
      tsl::safe_reinterpret_cast<hn::TFromD<decltype(dw)>*>(unaligned) +
          0 * dw_lanes);
  hn::StoreInterleaved2(
      v01_upper, v23_upper, dw,
      tsl::safe_reinterpret_cast<hn::TFromD<decltype(dw)>*>(unaligned) +
          2 * dw_lanes);
#else
  hn::StoreInterleaved4(v0, v1, v2, v3, d, unaligned);
#endif
}

template <int64_t kTpuLaneCount>
HWY_INLINE void RepackToHardwareLayoutSublaneHighwayPackingFactor2(
    const uint32_t* HWY_RESTRICT source_buffer,
    uint32_t* HWY_RESTRICT target_buffer) {
  constexpr int kNumInterleavedStores = sizeof(uint16_t);
  constexpr int kStride = kTpuLaneCount / kNumInterleavedStores;
  const hn::CappedTag<uint32_t, kStride> d32;
  const hn::RepartitionToNarrow<decltype(d32)> d16;
  const size_t kCpuVectorLanes = hn::Lanes(d32);
  using V16 = hn::Vec<decltype(d16)>;
  for (int i = 0; i < kStride; i += kCpuVectorLanes) {
    const V16 lower =
        hn::BitCast(d16, hn::LoadU(d32, source_buffer + i + 0 * kStride));
    const V16 upper =
        hn::BitCast(d16, hn::LoadU(d32, source_buffer + i + 1 * kStride));
    hn::StoreInterleaved2(
        lower, upper, d16,
        reinterpret_cast<  // REINTERPRET_CAST_OK=TODO(b/408916056)
            uint16_t * HWY_RESTRICT>(target_buffer +
                                     i * kNumInterleavedStores));
  }
}

template <int64_t kTpuLaneCount>
HWY_INLINE void RepackToHardwareLayoutSublaneHighwayPackingFactor4(
    const uint32_t* HWY_RESTRICT source_buffer,
    uint32_t* HWY_RESTRICT target_buffer) {
  constexpr int kNumInterleavedStores = sizeof(uint32_t);
  constexpr int kStride = kTpuLaneCount / kNumInterleavedStores;
  const hn::CappedTag<uint32_t, kStride> d32;
  const hn::RepartitionToNarrow<decltype(d32)> d16;
  const hn::RepartitionToNarrow<decltype(d16)> d8;
  const size_t kCpuVectorLanes = hn::Lanes(d32);
  using V8 = hn::Vec<decltype(d8)>;
  for (int i = 0; i < kStride; i += kCpuVectorLanes) {
    const V8 x0 =
        hn::BitCast(d8, hn::LoadU(d32, source_buffer + i + 0 * kStride));
    const V8 x1 =
        hn::BitCast(d8, hn::LoadU(d32, source_buffer + i + 1 * kStride));
    const V8 x2 =
        hn::BitCast(d8, hn::LoadU(d32, source_buffer + i + 2 * kStride));
    const V8 x3 =
        hn::BitCast(d8, hn::LoadU(d32, source_buffer + i + 3 * kStride));

    FastStoreInterleaved4(x0, x1, x2, x3, d8,
                          tsl::safe_reinterpret_cast<uint8_t * HWY_RESTRICT>(
                              target_buffer + i * kNumInterleavedStores));
  }
}

template <int kSrcGroup, int kDstGroup, int kBitsPerGroup, typename D,
          HWY_IF_UI8_D(D)>
HWY_INLINE hn::Vec<D> ExtractBits(D d, const hn::Vec<D>& v) {
  static_assert(kBitsPerGroup > 0 && kBitsPerGroup <= 8);
  static_assert(8 % kBitsPerGroup == 0);
  constexpr int kMaxGroups = 8 / kBitsPerGroup;
  static_assert(kSrcGroup >= 0 && kSrcGroup < kMaxGroups);
  static_assert(kDstGroup >= 0 && kDstGroup < kMaxGroups);
  constexpr int kSrcBitOffset = kBitsPerGroup * kSrcGroup;
  constexpr int kDstBitOffset = kBitsPerGroup * kDstGroup;
  constexpr uint8_t kLowMask{(1 << kBitsPerGroup) - 1};
  constexpr uint8_t kMask{kLowMask << kDstBitOffset};
  hn::Vec<D> to_mask = v;
  if constexpr (kSrcBitOffset > kDstBitOffset) {
    to_mask = hn::ShiftRight<kSrcBitOffset - kDstBitOffset>(to_mask);
  } else if constexpr (kSrcBitOffset < kDstBitOffset) {
    to_mask = hn::ShiftLeft<kDstBitOffset - kSrcBitOffset>(to_mask);
  }
  return hn::And(to_mask, hn::Set(d, kMask));
}

template <int kSrcNibble, int kDstNibble, typename D>
HWY_INLINE hn::Vec<D> ExtractNibble(D d, const hn::Vec<D>& v) {
  return ExtractBits</*kSrcGroup=*/kSrcNibble, /*kDstGroup=*/kDstNibble,
                     /*kBitsPerGroup=*/4>(d, v);
}

template <int64_t kTpuLaneCount>
HWY_INLINE HWY_FLATTEN void RepackToHardwareLayoutSublaneHighwayPackingFactor8(
    const uint32_t* HWY_RESTRICT source_buffer,
    uint32_t* HWY_RESTRICT target_buffer) {
  constexpr int kNumInterleavedStores = sizeof(uint32_t);
  constexpr int kStride = kTpuLaneCount / kNumInterleavedStores;
  const hn::CappedTag<uint32_t, kStride> d32;
  const hn::RepartitionToNarrow<decltype(d32)> d16;
  hn::RepartitionToNarrow<decltype(d16)> d8;
  const size_t kCpuVectorLanes = hn::Lanes(d32);
  using V8 = hn::Vec<decltype(d8)>;
  for (int i = 0; i < kStride; i += kCpuVectorLanes) {
    auto make_byte = [&](int load_index) {
      V8 low = hn::BitCast(
          d8, hn::LoadU(d32, source_buffer + i + (load_index + 0) * kStride));
      low = ExtractNibble</*kSrcNibble=*/0, /*kDstNibble=*/0>(d8, low);
      V8 high = hn::BitCast(
          d8, hn::LoadU(d32, source_buffer + i + (load_index + 1) * kStride));
      high = ExtractNibble</*kSrcNibble=*/0, /*kDstNibble=*/1>(d8, high);
      return hn::Or(low, high);
    };

    const V8 x0 = make_byte(0);
    const V8 x1 = make_byte(2);
    const V8 x2 = make_byte(4);
    const V8 x3 = make_byte(6);

    FastStoreInterleaved4(x0, x1, x2, x3, d8,
                          tsl::safe_reinterpret_cast<uint8_t * HWY_RESTRICT>(
                              target_buffer + i * kNumInterleavedStores));
  }
}

template <int kSrcCrumb, int kDstCrumb, typename D>
HWY_INLINE hn::Vec<D> ExtractCrumb(D d, const hn::Vec<D>& v) {
  return ExtractBits</*kSrcGroup=*/kSrcCrumb, /*kDstGroup=*/kDstCrumb,
                     /*kBitsPerGroup=*/2>(d, v);
}

template <int64_t kTpuLaneCount>
HWY_INLINE HWY_FLATTEN void RepackToHardwareLayoutSublaneHighwayPackingFactor16(
    const uint32_t* HWY_RESTRICT source_buffer,
    uint32_t* HWY_RESTRICT target_buffer) {
  constexpr int kNumInterleavedStores = sizeof(uint32_t);
  constexpr int kStride = kTpuLaneCount / kNumInterleavedStores;
  const hn::CappedTag<uint32_t, kStride> d32;
  const hn::RepartitionToNarrow<decltype(d32)> d16;
  hn::RepartitionToNarrow<decltype(d16)> d8;
  const size_t kCpuVectorLanes = hn::Lanes(d32);
  using V8 = hn::Vec<decltype(d8)>;
  for (int i = 0; i < kStride; i += kCpuVectorLanes) {
    auto make_byte = [&](int load_index) {
      V8 result0 = ExtractCrumb<0, 0>(
          d8, hn::BitCast(d8, hn::LoadU(d32, source_buffer + i +
                                                 (load_index + 0) * kStride)));
      V8 result1 = ExtractCrumb<0, 1>(
          d8, hn::BitCast(d8, hn::LoadU(d32, source_buffer + i +
                                                 (load_index + 1) * kStride)));
      V8 result_low = hn::Or(result0, result1);
      V8 result2 = ExtractCrumb<0, 2>(
          d8, hn::BitCast(d8, hn::LoadU(d32, source_buffer + i +
                                                 (load_index + 2) * kStride)));
      V8 result3 = ExtractCrumb<0, 3>(
          d8, hn::BitCast(d8, hn::LoadU(d32, source_buffer + i +
                                                 (load_index + 3) * kStride)));
      V8 result_high = hn::Or(result2, result3);
      return hn::Or(result_low, result_high);
    };

    const V8 x0 = make_byte(0);
    const V8 x1 = make_byte(4);
    const V8 x2 = make_byte(8);
    const V8 x3 = make_byte(12);

    FastStoreInterleaved4(x0, x1, x2, x3, d8,
                          tsl::safe_reinterpret_cast<uint8_t * HWY_RESTRICT>(
                              target_buffer + i * kNumInterleavedStores));
  }
}

template <int kSrcBit, int kDstBit, typename D>
HWY_INLINE hn::Vec<D> ExtractBit(D d, const hn::Vec<D>& v) {
  return ExtractBits</*kSrcGroup=*/kSrcBit, /*kDstGroup=*/kDstBit,
                     /*kBitsPerGroup=*/1>(d, v);
}

template <int64_t kTpuLaneCount>
HWY_INLINE HWY_FLATTEN void RepackToHardwareLayoutSublaneHighwayPackingFactor32(
    const uint32_t* HWY_RESTRICT source_buffer,
    uint32_t* HWY_RESTRICT target_buffer) {
  constexpr int kNumInterleavedStores = sizeof(uint32_t);
  constexpr int kStride = kTpuLaneCount / kNumInterleavedStores;
  const hn::CappedTag<uint32_t, kStride> d32;
  const hn::RepartitionToNarrow<decltype(d32)> d16;
  hn::RepartitionToNarrow<decltype(d16)> d8;
  const size_t kCpuVectorLanes = hn::Lanes(d32);
  using V8 = hn::Vec<decltype(d8)>;
  for (int i = 0; i < kStride; i += kCpuVectorLanes) {
    auto make_byte = [&](int load_index) {
      V8 result0 = ExtractBit<0, 0>(
          d8, hn::BitCast(d8, hn::LoadU(d32, source_buffer + i +
                                                 (load_index + 0) * kStride)));
      V8 result1 = ExtractBit<0, 1>(
          d8, hn::BitCast(d8, hn::LoadU(d32, source_buffer + i +
                                                 (load_index + 1) * kStride)));
      V8 result01 = hn::Or(result0, result1);
      V8 result2 = ExtractBit<0, 2>(
          d8, hn::BitCast(d8, hn::LoadU(d32, source_buffer + i +
                                                 (load_index + 2) * kStride)));
      V8 result3 = ExtractBit<0, 3>(
          d8, hn::BitCast(d8, hn::LoadU(d32, source_buffer + i +
                                                 (load_index + 3) * kStride)));
      V8 result0123 = hn::Or3(result01, result2, result3);
      V8 result4 = ExtractBit<0, 4>(
          d8, hn::BitCast(d8, hn::LoadU(d32, source_buffer + i +
                                                 (load_index + 4) * kStride)));
      V8 result5 = ExtractBit<0, 5>(
          d8, hn::BitCast(d8, hn::LoadU(d32, source_buffer + i +
                                                 (load_index + 5) * kStride)));
      V8 result45 = hn::Or(result4, result5);
      V8 result6 = ExtractBit<0, 6>(
          d8, hn::BitCast(d8, hn::LoadU(d32, source_buffer + i +
                                                 (load_index + 6) * kStride)));
      V8 result7 = ExtractBit<0, 7>(
          d8, hn::BitCast(d8, hn::LoadU(d32, source_buffer + i +
                                                 (load_index + 7) * kStride)));
      V8 result4567 = hn::Or3(result45, result6, result7);
      return hn::Or(result0123, result4567);
    };

    const V8 x0 = make_byte(0);
    const V8 x1 = make_byte(8);
    const V8 x2 = make_byte(16);
    const V8 x3 = make_byte(24);

    FastStoreInterleaved4(x0, x1, x2, x3, d8,
                          tsl::safe_reinterpret_cast<uint8_t * HWY_RESTRICT>(
                              target_buffer + i * kNumInterleavedStores));
  }
}

template <int64_t kTpuLaneCount>
HWY_INLINE void RepackToHostLayoutSublaneHighwayPackingFactor2(
    const uint32_t* HWY_RESTRICT source_buffer,
    uint32_t* HWY_RESTRICT target_buffer) {
  constexpr int kNumInterleavedLoads = sizeof(uint16_t);
  constexpr int kStride = kTpuLaneCount / kNumInterleavedLoads;
  const hn::CappedTag<uint32_t, kStride> d32;
  const hn::RepartitionToNarrow<decltype(d32)> d16;
  const size_t kCpuVectorLanes = hn::Lanes(d32);
  using V16 = hn::Vec<decltype(d16)>;
  for (int i = 0; i < kStride; i += kCpuVectorLanes) {
    V16 lower, upper;
    hn::LoadInterleaved2(
        d16,
        reinterpret_cast<  // REINTERPRET_CAST_OK=TODO(b/408916056)
            const uint16_t * HWY_RESTRICT>(source_buffer +
                                           i * kNumInterleavedLoads),
        lower, upper);
    hn::StoreU(hn::BitCast(d32, lower), d32, target_buffer + i + 0 * kStride);
    hn::StoreU(hn::BitCast(d32, upper), d32, target_buffer + i + 1 * kStride);
  }
}

template <int64_t kTpuLaneCount>
HWY_INLINE void RepackToHostLayoutSublaneHighwayPackingFactor4(
    const uint32_t* HWY_RESTRICT source_buffer,
    uint32_t* HWY_RESTRICT target_buffer) {
  constexpr int kNumInterleavedLoads = sizeof(uint32_t);
  constexpr int kStride = kTpuLaneCount / kNumInterleavedLoads;
  const hn::CappedTag<uint32_t, kStride> d32;
  const hn::RepartitionToNarrow<decltype(d32)> d16;
  const hn::RepartitionToNarrow<decltype(d16)> d8;
  const size_t kCpuVectorLanes = hn::Lanes(d32);
  using V8 = hn::Vec<decltype(d8)>;
  for (int i = 0; i < kStride; i += kCpuVectorLanes) {
    V8 x0, x1, x2, x3;
    FastLoadInterleaved4(
        d8,
        tsl::safe_reinterpret_cast<const uint8_t * HWY_RESTRICT>(
            source_buffer + i * kNumInterleavedLoads),
        x0, x1, x2, x3);

    hn::StoreU(
        x0, d8,
        tsl::safe_reinterpret_cast<uint8_t*>(target_buffer + i + 0 * kStride));
    hn::StoreU(
        x1, d8,
        tsl::safe_reinterpret_cast<uint8_t*>(target_buffer + i + 1 * kStride));
    hn::StoreU(
        x2, d8,
        tsl::safe_reinterpret_cast<uint8_t*>(target_buffer + i + 2 * kStride));
    hn::StoreU(
        x3, d8,
        tsl::safe_reinterpret_cast<uint8_t*>(target_buffer + i + 3 * kStride));
  }
}

template <int64_t kTpuLaneCount>
HWY_INLINE HWY_FLATTEN void RepackToHostLayoutSublaneHighwayPackingFactor8(
    const uint32_t* HWY_RESTRICT source_buffer,
    uint32_t* HWY_RESTRICT target_buffer) {
  constexpr int kNumInterleavedLoads = sizeof(uint32_t);
  constexpr int kStride = kTpuLaneCount / kNumInterleavedLoads;
  const hn::CappedTag<uint32_t, kStride> d32;
  const hn::RepartitionToNarrow<decltype(d32)> d16;
  hn::RepartitionToNarrow<decltype(d16)> d8;
  const size_t kCpuVectorLanes = hn::Lanes(d32);
  using V8 = hn::Vec<decltype(d8)>;
  for (int i = 0; i < kStride; i += kCpuVectorLanes) {
    V8 x0, x1, x2, x3;
    FastLoadInterleaved4(
        d8,
        tsl::safe_reinterpret_cast<const uint8_t * HWY_RESTRICT>(
            source_buffer + i * kNumInterleavedLoads),
        x0, x1, x2, x3);

    auto store_results = [&](const V8& v0, int store_index) {
      const V8 result0 =
          ExtractNibble</*kSrcNibble=*/0, /*kDstNibble=*/0>(d8, v0);
      hn::StoreU(result0, d8,
                 tsl::safe_reinterpret_cast<uint8_t*>(
                     target_buffer + i + (store_index + 0) * kStride));

      const V8 result1 =
          ExtractNibble</*kSrcNibble=*/1, /*kDstNibble=*/0>(d8, v0);
      hn::StoreU(result1, d8,
                 tsl::safe_reinterpret_cast<uint8_t*>(
                     target_buffer + i + (store_index + 1) * kStride));
    };

    store_results(x0, 0);
    store_results(x1, 2);
    store_results(x2, 4);
    store_results(x3, 6);
  }
}

template <int64_t kTpuLaneCount>
HWY_INLINE HWY_FLATTEN void RepackToHostLayoutSublaneHighwayPackingFactor16(
    const uint32_t* HWY_RESTRICT source_buffer,
    uint32_t* HWY_RESTRICT target_buffer) {
  constexpr int kNumInterleavedLoads = sizeof(uint32_t);
  constexpr int kStride = kTpuLaneCount / kNumInterleavedLoads;
  const hn::CappedTag<uint32_t, kStride> d32;
  const hn::RepartitionToNarrow<decltype(d32)> d16;
  hn::RepartitionToNarrow<decltype(d16)> d8;
  const size_t kCpuVectorLanes = hn::Lanes(d32);
  using V8 = hn::Vec<decltype(d8)>;
  for (int i = 0; i < kStride; i += kCpuVectorLanes) {
    V8 x0, x1, x2, x3;
    FastLoadInterleaved4(
        d8,
        tsl::safe_reinterpret_cast<const uint8_t * HWY_RESTRICT>(
            source_buffer + i * kNumInterleavedLoads),
        x0, x1, x2, x3);
    auto store_results = [&](const V8& v, size_t store_index) {
      auto result0 = ExtractCrumb<0, 0>(d8, v);
      hn::StoreU(result0, d8,
                 tsl::safe_reinterpret_cast<uint8_t*>(
                     target_buffer + i + (store_index + 0) * kStride));
      auto result1 = ExtractCrumb<1, 0>(d8, v);
      hn::StoreU(result1, d8,
                 tsl::safe_reinterpret_cast<uint8_t*>(
                     target_buffer + i + (store_index + 1) * kStride));
      auto result2 = ExtractCrumb<2, 0>(d8, v);
      hn::StoreU(result2, d8,
                 tsl::safe_reinterpret_cast<uint8_t*>(
                     target_buffer + i + (store_index + 2) * kStride));
      auto result3 = ExtractCrumb<3, 0>(d8, v);
      hn::StoreU(result3, d8,
                 tsl::safe_reinterpret_cast<uint8_t*>(
                     target_buffer + i + (store_index + 3) * kStride));
    };
    store_results(x0, 0);
    store_results(x1, 4);
    store_results(x2, 8);
    store_results(x3, 12);
  }
}

template <int64_t kTpuLaneCount>
HWY_INLINE HWY_FLATTEN void RepackToHostLayoutSublaneHighwayPackingFactor32(
    const uint32_t* HWY_RESTRICT source_buffer,
    uint32_t* HWY_RESTRICT target_buffer) {
  constexpr int kNumInterleavedLoads = sizeof(uint32_t);
  constexpr int kStride = kTpuLaneCount / kNumInterleavedLoads;
  const hn::CappedTag<uint32_t, kStride> d32;
  const hn::RepartitionToNarrow<decltype(d32)> d16;
  hn::RepartitionToNarrow<decltype(d16)> d8;
  const size_t kCpuVectorLanes = hn::Lanes(d32);
  using V8 = hn::Vec<decltype(d8)>;
  for (int i = 0; i < kStride; i += kCpuVectorLanes) {
    V8 x0, x1, x2, x3;
    FastLoadInterleaved4(
        d8,
        tsl::safe_reinterpret_cast<const uint8_t * HWY_RESTRICT>(
            source_buffer + i * kNumInterleavedLoads),
        x0, x1, x2, x3);
    auto store_results = [&](const V8& v, size_t store_index) {
      auto result0 = ExtractBit<0, 0>(d8, v);
      hn::StoreU(result0, d8,
                 tsl::safe_reinterpret_cast<uint8_t*>(
                     target_buffer + i + (store_index + 0) * kStride));
      auto result1 = ExtractBit<1, 0>(d8, v);
      hn::StoreU(result1, d8,
                 tsl::safe_reinterpret_cast<uint8_t*>(
                     target_buffer + i + (store_index + 1) * kStride));
      auto result2 = ExtractBit<2, 0>(d8, v);
      hn::StoreU(result2, d8,
                 tsl::safe_reinterpret_cast<uint8_t*>(
                     target_buffer + i + (store_index + 2) * kStride));
      auto result3 = ExtractBit<3, 0>(d8, v);
      hn::StoreU(result3, d8,
                 tsl::safe_reinterpret_cast<uint8_t*>(
                     target_buffer + i + (store_index + 3) * kStride));
      auto result4 = ExtractBit<4, 0>(d8, v);
      hn::StoreU(result4, d8,
                 tsl::safe_reinterpret_cast<uint8_t*>(
                     target_buffer + i + (store_index + 4) * kStride));
      auto result5 = ExtractBit<5, 0>(d8, v);
      hn::StoreU(result5, d8,
                 tsl::safe_reinterpret_cast<uint8_t*>(
                     target_buffer + i + (store_index + 5) * kStride));
      auto result6 = ExtractBit<6, 0>(d8, v);
      hn::StoreU(result6, d8,
                 tsl::safe_reinterpret_cast<uint8_t*>(
                     target_buffer + i + (store_index + 6) * kStride));
      auto result7 = ExtractBit<7, 0>(d8, v);
      hn::StoreU(result7, d8,
                 tsl::safe_reinterpret_cast<uint8_t*>(
                     target_buffer + i + (store_index + 7) * kStride));
    };
    store_results(x0, 0);
    store_results(x1, 8);
    store_results(x2, 16);
    store_results(x3, 24);
  }
}

}  // namespace xla::HWY_NAMESPACE

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace xla {

// Highway implementation of `RepackToHardwareLayoutImpl`. See
// tiled_linearizer.cc for more details.
template <int64_t kPackingFactor, int64_t kTpuLaneCount>
HWY_INLINE void RepackToHardwareLayoutSublaneHighway(
    const uint32_t* HWY_RESTRICT source_buffer,
    uint32_t* HWY_RESTRICT target_buffer) {
  static_assert(HWY_IS_LITTLE_ENDIAN, "Must be little endian");
  static_assert(kTpuLaneCount % kPackingFactor == 0,
                "kTpuLaneCount not a multiple of kPackingFactor");
  constexpr int64_t kNumInterleavedStores =
      std::min(kPackingFactor, int64_t{4});
  static_assert(kTpuLaneCount % kNumInterleavedStores == 0);
  constexpr int64_t kStride = kTpuLaneCount / kNumInterleavedStores;
  static_assert(kStride > 0, "Bad kStride");
  static_assert(absl::has_single_bit<uint64_t>(kStride),
                "kStride must be a power of two");
  if constexpr (kPackingFactor == 2) {
    HWY_STATIC_DISPATCH(
        RepackToHardwareLayoutSublaneHighwayPackingFactor2<kTpuLaneCount>)
    (source_buffer, target_buffer);
    return;
  } else if constexpr (kPackingFactor == 4) {
    HWY_STATIC_DISPATCH(
        RepackToHardwareLayoutSublaneHighwayPackingFactor4<kTpuLaneCount>)
    (source_buffer, target_buffer);
    return;
  } else if constexpr (kPackingFactor == 8) {
    HWY_STATIC_DISPATCH(
        RepackToHardwareLayoutSublaneHighwayPackingFactor8<kTpuLaneCount>)
    (source_buffer, target_buffer);
    return;
  } else if constexpr (kPackingFactor == 16) {
    HWY_STATIC_DISPATCH(
        RepackToHardwareLayoutSublaneHighwayPackingFactor16<kTpuLaneCount>)
    (source_buffer, target_buffer);
    return;
  } else if constexpr (kPackingFactor == 32) {
    HWY_STATIC_DISPATCH(
        RepackToHardwareLayoutSublaneHighwayPackingFactor32<kTpuLaneCount>)
    (source_buffer, target_buffer);
    return;
  } else {
    static_assert(false, "Unsupported packing factor");
  }
}

template <int64_t kPackingFactor, int64_t kTpuLaneCount>
HWY_INLINE void RepackToHostLayoutSublaneHighway(
    const uint32_t* HWY_RESTRICT source_buffer,
    uint32_t* HWY_RESTRICT target_buffer) {
  static_assert(HWY_IS_LITTLE_ENDIAN, "Must be little endian");
  static_assert(kTpuLaneCount % kPackingFactor == 0,
                "kTpuLaneCount not a multiple of kPackingFactor");
  constexpr int64_t kNumInterleavedLoads = std::min(kPackingFactor, int64_t{4});
  static_assert(kTpuLaneCount % kNumInterleavedLoads == 0);
  constexpr int64_t kStride = kTpuLaneCount / kNumInterleavedLoads;
  static_assert(kStride > 0, "Bad kStride");
  static_assert(absl::has_single_bit<uint64_t>(kStride),
                "kStride must be a power of two");
  if constexpr (kPackingFactor == 2) {
    HWY_STATIC_DISPATCH(
        RepackToHostLayoutSublaneHighwayPackingFactor2<kTpuLaneCount>)
    (source_buffer, target_buffer);
    return;
  } else if constexpr (kPackingFactor == 4) {
    HWY_STATIC_DISPATCH(
        RepackToHostLayoutSublaneHighwayPackingFactor4<kTpuLaneCount>)
    (source_buffer, target_buffer);
    return;
  } else if constexpr (kPackingFactor == 8) {
    HWY_STATIC_DISPATCH(
        RepackToHostLayoutSublaneHighwayPackingFactor8<kTpuLaneCount>)
    (source_buffer, target_buffer);
    return;
  } else if constexpr (kPackingFactor == 16) {
    HWY_STATIC_DISPATCH(
        RepackToHostLayoutSublaneHighwayPackingFactor16<kTpuLaneCount>)
    (source_buffer, target_buffer);
    return;
  } else if constexpr (kPackingFactor == 32) {
    HWY_STATIC_DISPATCH(
        RepackToHostLayoutSublaneHighwayPackingFactor32<kTpuLaneCount>)
    (source_buffer, target_buffer);
    return;
  } else {
    static_assert(false, "Unsupported packing factor");
  }
}

}  // namespace xla

#endif  // HWY_ONCE

#endif  // XLA_PJRT_TILED_LINEARIZER_HIGHWAY_H_
