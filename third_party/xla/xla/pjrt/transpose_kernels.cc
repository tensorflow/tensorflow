/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/pjrt/transpose_kernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "xla/compiler_macros.h"
#include "xla/pjrt/transpose_kernels_internal.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "xla/pjrt/transpose_kernels.cc"  // NOLINT(whitespace/line_length)
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/targets.h"

HWY_BEFORE_NAMESPACE();
namespace xla {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Highway's model: Vectors are composed of 128-bit blocks, blocks hold lanes.
// Interleave operations work within blocks. Moving data between blocks needs
// cross-block ops such as Combine, UpperHalf, or Per4LaneBlockShuffle.
constexpr size_t kBlockBytes = 16;   // 128-bit blocks
constexpr size_t kMaxLaneBytes = 8;  // Lanes are at most 64 bits

// Helpers that produce several vectors use continuation-passing style (CPS):
// instead of returning, they call `cont(outputs...)` exactly once, in tail
// position. This is how they "return" a pack: C++ can't return one, and
// Highway vectors can't be stored in arrays, tuples, or structs (they are
// sizeless types on SVE/RVV). The helpers are HWY_INLINE and the kernels
// HWY_FLATTEN, so the continuations compile away.
//
// Lambdas need an explicit HWY_ATTR before their opening brace:
// https://github.com/google/highway/blob/master/g3doc/quick_reference.md

// Which half of each 128-bit block Unpack interleaves.
enum class BlockHalf { kLower, kUpper };

// Interleaves the element_size-byte lanes from the lower or upper half of each
// 128-bit block of `a` and `b`, starting with `a` (x86 punpckl*/punpckh*).
template <size_t element_size, BlockHalf half, class V>
HWY_INLINE V Unpack(V a, V b) {
  static_assert(element_size <= kMaxLaneBytes);
  using D = hn::DFromV<V>;
  using ElementT = hwy::UnsignedFromSize<element_size>;
  const hn::Repartition<ElementT, D> d_elem;
  auto a_elem = hn::BitCast(d_elem, a);
  auto b_elem = hn::BitCast(d_elem, b);
  if constexpr (half == BlockHalf::kLower) {
    return hn::BitCast(D(), hn::InterleaveLower(d_elem, a_elem, b_elem));
  } else {
    return hn::BitCast(D(), hn::InterleaveUpper(d_elem, a_elem, b_elem));
  }
}

// Load/Store

// Loads `bytes` bytes from `p` into the low bytes of a D vector. The remaining
// bytes are unspecified (see hn::ResizeBitCast).
template <class D, size_t bytes>
HWY_INLINE hn::Vec<D> LoadBytes(D d, const void* p) {
  const hn::FixedTag<uint8_t, bytes> d_fixed;
  return hn::ResizeBitCast(
      d, hn::LoadU(d_fixed, reinterpret_cast<const uint8_t*>(p)));
}

// Stores lane `lane` of `v` to `p`, treating `v` as lanes of `bytes` bytes.
template <class D, size_t bytes, size_t lane>
HWY_INLINE void StoreLane(D d, void* p, hn::Vec<D> v) {
  if constexpr (bytes <= kMaxLaneBytes && lane > 0) {
    using ElementT = hwy::UnsignedFromSize<bytes>;
    const hn::Repartition<ElementT, D> d_elem;
    const ElementT scalar = hn::ExtractLane(hn::BitCast(d_elem, v), lane);
    hwy::CopyBytes(&scalar, reinterpret_cast<ElementT*>(p), bytes);
  } else {
    const hn::Repartition<uint8_t, D> d_u8;
    auto v_u8 = hn::BitCast(d_u8, v);
    if constexpr (lane > 0) {
      v_u8 = hn::SlideDownLanes(d_u8, v_u8, lane * bytes);
    }
    const hn::FixedTag<uint8_t, bytes> d_fixed;
    hn::StoreU(hn::ResizeBitCast(d_fixed, v_u8), d_fixed,
               reinterpret_cast<uint8_t*>(p));
  }
}

// Stores lane k of `v` to row i + k of `b` for each k in `lane...`.
template <class D, size_t bytes, size_t... lane>
HWY_INLINE void StoreLanes(D d, char* b, int64_t ldb, hn::Vec<D> v, int64_t i,
                           std::index_sequence<lane...>) {
  (StoreLane<D, bytes, lane>(d, b + ldb * (i + static_cast<int64_t>(lane)), v),
   ...);
}

// Extracts the I-th element from a parameter pack (0-indexed).
template <size_t I, class T, class... Ts>
HWY_INLINE decltype(auto) GetFromPack(T&& t, Ts&&... ts) {
  static_assert(I < 1 + sizeof...(Ts), "Index out of bounds");
  if constexpr (I == 0) {
    return std::forward<T>(t);
  } else {
    return GetFromPack<I - 1>(std::forward<Ts>(ts)...);
  }
}

// Calls f(std::integral_constant<int64_t, I>{}, ts_I) for each element of the
// pack, in order. Signed indices make `ptr + i * stride` safe when `stride` is
// negative.
template <class F, size_t... I, class... Ts>
HWY_INLINE void ForEachIndexedImpl(F&& f, std::index_sequence<I...>,
                                   Ts&&... ts) {
  (f(std::integral_constant<int64_t, I>{}, std::forward<Ts>(ts)), ...);
}

template <class F, class... Ts>
HWY_INLINE void ForEachIndexed(F&& f, Ts&&... ts) {
  ForEachIndexedImpl(std::forward<F>(f),
                     std::make_index_sequence<sizeof...(Ts)>{},
                     std::forward<Ts>(ts)...);
}

// Calls `f(std::integral_constant<int64_t, 0>{}, ..., kCount - 1)`.
template <class F, size_t... I>
HWY_INLINE void WithIndicesImpl(F&& f, std::index_sequence<I...>) {
  std::forward<F>(f)(std::integral_constant<int64_t, I>{}...);
}

template <size_t kCount, class F>
HWY_INLINE void WithIndices(F&& f) {
  WithIndicesImpl(std::forward<F>(f), std::make_index_sequence<kCount>{});
}

// Returns output `out_i` of UnpackStepCPS. Within each group of 2 * step_size
// inputs starting at `base`, outputs base + 2j and base + 2j + 1 are the lower
// and upper interleaves of inputs base + j and base + j + step_size, at a
// granularity of element_size * step_size bytes.
template <size_t out_i, size_t element_size, size_t step_size, class... Vs>
HWY_INLINE auto UnpackOutFromPack(Vs&... vs) {
  constexpr size_t group = 2 * step_size;
  constexpr size_t base = (out_i / group) * group;
  constexpr size_t p = out_i - base;
  constexpr size_t j = p / 2;
  constexpr bool is_lo = (p % 2) == 0;
  constexpr size_t ia = base + j;
  constexpr size_t ib = base + j + step_size;
  auto& a = GetFromPack<ia>(vs...);
  auto& b = GetFromPack<ib>(vs...);
  if constexpr (is_lo) {
    return Unpack<element_size * step_size, BlockHalf::kLower>(a, b);
  } else {
    return Unpack<element_size * step_size, BlockHalf::kUpper>(a, b);
  }
}

template <size_t element_size, size_t step_size, class Cont, size_t... I,
          class... Vs>
HWY_INLINE void UnpackStepCPSImpl(Cont&& cont, std::index_sequence<I...>,
                                  Vs&... vs) {
  std::forward<Cont>(cont)(
      UnpackOutFromPack<I, element_size, step_size>(vs...)...);
}

// One stage of the in-block transpose network (pack -> pack): within each
// group of 2 * step_size inputs, pairs input j with input j + step_size and
// interleaves them at element_size * step_size bytes (see UnpackOutFromPack).
// Inputs are lvalue references because each one feeds two outputs.
template <size_t element_size, size_t step_size, class Cont, class... Vs>
HWY_INLINE void UnpackStepCPS(Cont&& cont, Vs&... in) {
  constexpr size_t N = sizeof...(Vs);
  static_assert(N % (step_size * 2) == 0);
  static_assert(element_size * step_size <= kMaxLaneBytes);
  UnpackStepCPSImpl<element_size, step_size>(
      std::forward<Cont>(cont), std::make_index_sequence<N>{}, in...);
}

// Runs UnpackStepCPS at granularities element_size, 2 * element_size, ...,
// while the granularity is below unpack_limit and at most kMaxLaneBytes.
// With unpack_limit == kBlockBytes this is a full in-block transpose: for each
// run of n = kBlockBytes / element_size consecutive vectors, the n x n matrix
// of element_size-byte elements in each 128-bit block position is transposed.
// A smaller unpack_limit skips the final stages, leaving unpack_limit-byte
// pieces of output rows for the caller to store or shuffle into place.
template <size_t element_size, size_t step_size, size_t unpack_limit,
          class Cont, class... Vs>
HWY_INLINE void InterleaveWithinBlocksCPS(Cont&& cont, Vs... in) {
  if constexpr (element_size * step_size < unpack_limit &&
                element_size * step_size <= kMaxLaneBytes) {
    UnpackStepCPS<element_size, step_size>(
        [&](auto... out) HWY_ATTR {
          InterleaveWithinBlocksCPS<element_size, step_size * 2, unpack_limit>(
              std::forward<Cont>(cont), out...);
        },
        in...);  // Lvalues here, as UnpackStepCPS requires.
  } else {
    std::forward<Cont>(cont)(in...);
  }
}

template <size_t element_size, class Cont, size_t... I, class... Vs>
HWY_INLINE void CombinePairsCPSImpl(Cont&& cont, std::index_sequence<I...>,
                                    Vs&&... vs) {
  std::forward<Cont>(cont)(Unpack<element_size, BlockHalf::kLower>(
      GetFromPack<2 * I>(std::forward<Vs>(vs)...),
      GetFromPack<2 * I + 1>(std::forward<Vs>(vs)...))...);
}

// Zips adjacent pairs of vectors (pack -> half-size pack): output i is
// Unpack<element_size, BlockHalf::kLower> of inputs 2i and 2i + 1, so only the
// lower half of each 128-bit block is used. The number of inputs must be even.
template <size_t element_size, class Cont, class... Vs>
HWY_INLINE void CombinePairsCPS(Cont&& cont, Vs&&... in) {
  CombinePairsCPSImpl<element_size>(
      std::forward<Cont>(cont), std::make_index_sequence<sizeof...(Vs) / 2>{},
      std::forward<Vs>(in)...);
}

// Packs rows narrower than a vector together (pack -> smaller pack). While a
// vector holds fewer than vector_bytes bytes of row data (element_size * bs),
// zips adjacent vectors with CombinePairsCPS and doubles element_size. Before
// each stage the row data fits in the lower half of the block, so the lower
// interleave consumes all of it and the unspecified bytes from LoadBytes are
// never stored. Assumes single-block vectors.
template <size_t element_size, size_t bs, size_t vector_bytes, class Cont,
          class... Vs>
HWY_INLINE void CombineRowsCPS(Cont&& cont, Vs... in) {
  constexpr size_t N = sizeof...(Vs);
  if constexpr (N > 1 && element_size * bs < vector_bytes) {
    static_assert(N % 2 == 0);
    CombinePairsCPS<element_size>(
        [&](auto... paired) HWY_ATTR {
          CombineRowsCPS<element_size * 2, bs, vector_bytes>(
              std::forward<Cont>(cont), paired...);
        },
        in...);
  } else {
    std::forward<Cont>(cont)(in...);
  }
}

// Loads bs rows of row_bytes bytes each (stride lda) and passes them to `cont`
// as one pack. Recurses on I; `vs...` accumulates the rows loaded so far.
template <int I, typename T, int bs, class D, size_t row_bytes, class Cont,
          class... Vs>
HWY_INLINE void LoadRowsCPS(D d, const char* a, int64_t lda, Cont&& cont,
                            Vs... vs) {
  if constexpr (I == bs) {
    std::forward<Cont>(cont)(vs...);
  } else {
    auto v = LoadBytes<D, row_bytes>(d, a + lda * I);
    LoadRowsCPS<I + 1, T, bs, D, row_bytes>(d, a, lda, std::forward<Cont>(cont),
                                            vs..., v);
  }
}

// Stores the transposed vectors: vector i holds output rows
// [i * rows_per_vec, (i + 1) * rows_per_vec), each row_bytes wide.
template <class D, size_t row_bytes, size_t rows_per_vec, class... Vs>
HWY_INLINE void StoreVecs128(D d, char* b, int64_t ldb, Vs... v) {
  ForEachIndexed(
      [&](auto idx_c, auto&& vec) HWY_ATTR {
        constexpr int64_t i = decltype(idx_c)::value;
        StoreLanes<D, row_bytes>(d, b, ldb, vec,
                                 i * static_cast<int64_t>(rows_per_vec),
                                 std::make_index_sequence<rows_per_vec>{});
      },
      v...);
}

// 128-bit (single block) transpose of a bs x bs block whose rows fit in one
// block. Rows narrower than a block are first packed together by
// CombineRowsCPS.
template <typename T, int bs>
HWY_FLATTEN void TransposeMicroKernel128(const char* HWY_RESTRICT a,
                                         int64_t lda, char* HWY_RESTRICT b,
                                         int64_t ldb) {
  using D = hn::Full128<uint8_t>;
  static constexpr size_t element_size = sizeof(T);
  static constexpr size_t row_bytes = element_size * bs;
  static_assert(row_bytes <= kBlockBytes);
  const D d;
  LoadRowsCPS<0, T, bs, D, row_bytes>(d, a, lda, [&](auto... loads) HWY_ATTR {
    CombineRowsCPS<element_size, bs, kBlockBytes>(
        [&](auto... vecs) HWY_ATTR {
          // CombineRowsCPS packed kRowsPerVec rows into each vector, so the
          // transpose resumes at kCombinedElementSize-byte granularity and
          // leaves kRowsPerVec output rows in each vector.
          constexpr size_t kNumVecs = sizeof...(vecs);
          constexpr size_t kRowsPerVec = bs / kNumVecs;
          constexpr size_t kCombinedElementSize = element_size * kRowsPerVec;
          InterleaveWithinBlocksCPS<kCombinedElementSize, 1, row_bytes>(
              [&](auto... transposed) HWY_ATTR {
                StoreVecs128<D, row_bytes, kRowsPerVec>(d, b, ldb,
                                                        transposed...);
              },
              vecs...);
        },
        loads...);
  });
}

// 256 bit and 512 bit multi block transposes
#if HWY_MIN_BYTES >= 32
// Stores vector i to output row i.
template <class D, class... Vs>
HWY_INLINE void StoreRows(D d, char* b, int64_t ldb, Vs... v) {
  ForEachIndexed(
      [&](auto idx_c, auto&& vec) HWY_ATTR {
        constexpr int64_t i = decltype(idx_c)::value;
        hn::StoreU(vec, d, reinterpret_cast<hn::TFromD<D>*>(b + ldb * i));
      },
      v...);
}

// Loads `d.MaxBytes() / kPartBytes` slices of `kPartBytes` bytes spaced
// `stride` bytes apart and concatenates them into one `D` vector.
template <size_t kPartBytes, class D>
HWY_INLINE hn::Vec<D> LoadParts(D d, const char* p, int64_t stride) {
  if constexpr (d.MaxBytes() == kPartBytes) {
    return hn::LoadU(d, reinterpret_cast<const hn::TFromD<D>*>(p));
  } else {
    const hn::Half<D> d_half;
    constexpr int64_t kHalfParts = d_half.MaxBytes() / kPartBytes;
    const auto lo = LoadParts<kPartBytes>(d_half, p, stride);
    const auto hi =
        LoadParts<kPartBytes>(d_half, p + kHalfParts * stride, stride);
    return hn::Combine(d, hi, lo);
  }
}

// Stores consecutive `kPartBytes` slices of `v` to addresses spaced `stride`
// bytes apart (for 128 bit slices on x86, vmovdqu plus vextract* memory ops).
template <size_t kPartBytes, class D>
HWY_INLINE void StoreParts(D d, hn::Vec<D> v, char* p, int64_t stride) {
  if constexpr (d.MaxBytes() == kPartBytes) {
    hn::StoreU(v, d, reinterpret_cast<hn::TFromD<D>*>(p));
  } else {
    const hn::Half<D> d_half;
    constexpr int64_t kHalfParts = d_half.MaxBytes() / kPartBytes;
    const auto lo = hn::LowerHalf(d_half, v);
    const auto hi = hn::UpperHalf(d_half, v);
    StoreParts<kPartBytes>(d_half, lo, p, stride);
    StoreParts<kPartBytes>(d_half, hi, p + kHalfParts * stride, stride);
  }
}

// Calls `cont` with `kCount` vectors of `D`, where vector `i` gathers
// `d.MaxBytes() / kPartBytes` slices from rows `i, i + kCount, ...`.
template <size_t kCount, size_t kPartBytes, class D, class Cont>
HWY_INLINE void WithGatheredParts(D d, const char* a, int64_t lda,
                                  Cont&& cont) {
  const int64_t part_stride = static_cast<int64_t>(kCount) * lda;
  WithIndices<kCount>([&](auto... i) HWY_ATTR {
    std::forward<Cont>(cont)(
        LoadParts<kPartBytes>(d, a + i * lda, part_stride)...);
  });
}

// Concatenates the even (0, 2, ...) or odd (1, 3, ...) 128 bit blocks of `lo`
// (in the lower half) and `hi` (in the upper half) for a fixed size tag `D`
// with a power of two block count, matching Highway's `ConcatEven`/`ConcatOdd`
// parameter order.
template <bool kOdd, class D>
HWY_INLINE hn::VFromD<D> ConcatEvenOddBlocks(D d, hn::VFromD<D> hi,
                                             hn::VFromD<D> lo) {
  constexpr size_t kBytes = d.MaxBytes();
  static_assert(kBytes >= 2 * kBlockBytes && (kBytes & (kBytes - 1)) == 0);
  // Require `Lanes(d) == MaxLanes(d)` so scalable tags whose runtime lane count
  // can differ from `MaxLanes(d)` fail at compile time.
  constexpr int kGroupPow2 =
      std::min(d.Pow2(), (HWY_TARGET == HWY_RVV) ? HWY_MAX_POW2 : 0);
  if constexpr (kGroupPow2 >= 0) {
    static_assert((kBytes >> kGroupPow2) <= HWY_MIN_BYTES);
  } else {
    static_assert((kBytes << (-kGroupPow2)) <= HWY_MIN_BYTES);
  }
  if constexpr (kBytes == 2 * kBlockBytes) {
    if constexpr (kOdd) {
      return hn::ConcatUpperUpper(d, hi, lo);
    } else {
      return hn::ConcatLowerLower(d, hi, lo);
    }
  } else {
#if HWY_MIN_BYTES >= 64 && HWY_ARCH_X86
    if constexpr (kBytes == 4 * kBlockBytes) {
      // Emit vshufi32x4 directly: LLVM expands the recursive Half fallback into
      // extra vpermt2q, vextracti64x4, and vperm2i128 shuffles.
      constexpr int kImm = kOdd ? 0xDD : 0x88;
      return hn::VFromD<D>{_mm512_shuffle_i32x4(lo.raw, hi.raw, kImm)};
    }
#endif
    const hn::Half<D> d_half;
    const auto lo_lo = hn::LowerHalf(d_half, lo);
    const auto lo_hi = hn::UpperHalf(d_half, lo);
    const auto hi_lo = hn::LowerHalf(d_half, hi);
    const auto hi_hi = hn::UpperHalf(d_half, hi);
    const auto lo_blocks = ConcatEvenOddBlocks<kOdd>(d_half, lo_hi, lo_lo);
    const auto hi_blocks = ConcatEvenOddBlocks<kOdd>(d_half, hi_hi, hi_lo);
    return hn::Combine(d, hi_blocks, lo_blocks);
  }
}

// Transposes a bs x bs block with `kVecBytes` byte rows (32 or 64) as vertical
// column strips of width `kStripBytes` (16 or 32), each emitting full
// `kVecBytes` stores in ascending row order. Covers 256 bit square `(32, 32)`,
// 256 bit column half `(32, 16)`, 512 bit column strip `(64, 16)`, and 512 bit
// column half `(64, 32)` kernels.
template <size_t kVecBytes, size_t kStripBytes, typename T, int bs>
HWY_FLATTEN void TransposeColStrips(const char* HWY_RESTRICT a, int64_t lda,
                                    char* HWY_RESTRICT b, int64_t ldb) {
  using D = hn::FixedTag<uint8_t, kVecBytes>;
  static_assert(sizeof(T) * bs == kVecBytes);
  constexpr int kStrips = kVecBytes / kStripBytes;
  constexpr int kOutRows = bs / kStrips;
  constexpr size_t kGroup = kBlockBytes / sizeof(T);
  constexpr bool kRoll = kOutRows > 8;
  const D d;
  XLA_UNROLL_COUNT(kRoll ? 1 : kStrips)
  for (int s = 0; s < kStrips; ++s) {
    const char* a_s = a + s * static_cast<int64_t>(kStripBytes);
    char* b_s = b + s * kOutRows * ldb;
    if constexpr (kRoll) {
      XLA_HIDE_POINTER_FROM_OPTIMIZER(a_s);
      XLA_HIDE_POINTER_FROM_OPTIMIZER(b_s);
    }
    auto finish = [&](auto... vecs) HWY_ATTR {
      InterleaveWithinBlocksCPS<sizeof(T), 1, kBlockBytes>(
          [&](auto... transposed)
              HWY_ATTR { StoreRows(d, b_s, ldb, transposed...); },
          vecs...);
    };
    if constexpr (kStripBytes == kBlockBytes) {
      WithGatheredParts<kGroup, kBlockBytes>(d, a_s, lda, finish);
    } else {
      static_assert(kStripBytes == 2 * kBlockBytes);
      constexpr int64_t kPairRows = (kVecBytes / kStripBytes) * kGroup;
      WithGatheredParts<kGroup, kStripBytes>(
          d, a_s, lda, [&](auto... p0) HWY_ATTR {
            WithGatheredParts<kGroup, kStripBytes>(
                d, a_s + kPairRows * lda, lda, [&](auto... p1) HWY_ATTR {
                  finish(ConcatEvenOddBlocks</*kOdd=*/false>(d, p1, p0)...,
                         ConcatEvenOddBlocks</*kOdd=*/true>(d, p1, p0)...);
                });
          });
    }
  }
}

// Permutes 64-bit lanes across 128-bit blocks in a 256-bit vector.
// Specifically, reorders the 4 64-bit lanes from [0, 1, 2, 3] to [0, 2, 1, 3].
template <class V>
HWY_INLINE V ShuffleAcrossBlocks_3120(V v) {
  const hn::DFromV<V> d;
  const hn::Repartition<uint64_t, decltype(d)> d64;
  return hn::BitCast(d,
                     hn::Per4LaneBlockShuffle<3, 1, 2, 0>(hn::BitCast(d64, v)));
}

// Transposes a strip of kRows = 8 / sizeof(T) rows of a bs x bs block with
// 32-byte rows into bytes [0, 8) of every output row. Each row is one vector;
// after the in-block unpack stages, each 64-bit lane holds kRows elements of
// one output row. Lanes 0 and 1 of vector i are output rows 2i and 2i + 1, and
// lanes 2 and 3 (the upper block) are output rows 2i + bs / 2 and
// 2i + bs / 2 + 1. The strip's helpers are flattened into it, but the strip
// itself is out of line so that TransposeMicroKernel256Strips' 4 calls can't
// be interleaved into more live vectors than there are registers.
template <typename T, int bs>
HWY_NOINLINE HWY_FLATTEN void TransposeStrip256(const char* HWY_RESTRICT a,
                                                int64_t lda,
                                                char* HWY_RESTRICT b,
                                                int64_t ldb) {
  static_assert(sizeof(T) * bs == 2 * kBlockBytes);
  static_assert(sizeof(T) <= sizeof(uint64_t));
  constexpr int kRows = sizeof(uint64_t) / sizeof(T);
  using D = hn::FixedTag<uint8_t, 32>;
  const D d;
  using D64 = hn::Repartition<uint64_t, D>;
  const D64 d64;
  LoadRowsCPS<0, T, kRows, D, 32>(d, a, lda, [&](auto... loads) HWY_ATTR {
    InterleaveWithinBlocksCPS<sizeof(T), 1, sizeof(uint64_t)>(
        [&](auto... interleaved) HWY_ATTR {
          ForEachIndexed(
              [&](auto idx_c, auto&& vec) HWY_ATTR {
                constexpr int64_t i = decltype(idx_c)::value;
                constexpr int64_t lo_row = 2 * i;
                constexpr int64_t hi_row = 2 * i + bs / 2;
                auto v64 = hn::BitCast(d64, vec);
                StoreLane<D64, sizeof(uint64_t), 0>(d64, b + lo_row * ldb, v64);
                StoreLane<D64, sizeof(uint64_t), 1>(d64, b + (lo_row + 1) * ldb,
                                                    v64);
                StoreLane<D64, sizeof(uint64_t), 2>(d64, b + hi_row * ldb, v64);
                StoreLane<D64, sizeof(uint64_t), 3>(d64, b + (hi_row + 1) * ldb,
                                                    v64);
              },
              interleaved...);
        },
        loads...);
  });
}

// Transposes a bs x bs block with 32-byte rows as 4 strips of
// kRows = 8 / sizeof(T) rows (see TransposeStrip256): strip s, input rows
// [s * kRows, (s + 1) * kRows), becomes output columns
// [s * kRows, (s + 1) * kRows). A strip only has kRows vectors live, so
// nothing spills.
template <typename T, int bs>
HWY_FLATTEN void TransposeMicroKernel256Strips(const char* HWY_RESTRICT a,
                                               int64_t lda,
                                               char* HWY_RESTRICT b,
                                               int64_t ldb) {
  constexpr int kRows = sizeof(uint64_t) / sizeof(T);
  static_assert(4 * kRows == bs);
  const int64_t a_stride = kRows * lda;            // kRows input rows
  constexpr int64_t b_stride = kRows * sizeof(T);  // kRows output columns
  TransposeStrip256<T, bs>(a, lda, b, ldb);
  TransposeStrip256<T, bs>(a + a_stride, lda, b + b_stride, ldb);
  TransposeStrip256<T, bs>(a + 2 * a_stride, lda, b + 2 * b_stride, ldb);
  TransposeStrip256<T, bs>(a + 3 * a_stride, lda, b + 3 * b_stride, ldb);
}

// 256-bit transpose of a bs x bs block with 16-byte rows. Rows I and
// I + bs / 2 share a vector, so the in-block network stops before its 8-byte
// stage, leaving vector i with the 64-bit lanes
// [r[2i].lo, r[2i+1].lo, r[2i].hi, r[2i+1].hi], where r[k].lo and r[k].hi are
// the first and last 8 bytes of output row k. That stage would pair lanes
// across blocks, so ShuffleAcrossBlocks_3120 does it instead, and StoreParts
// stores rows 2i and 2i + 1.
template <typename T, int bs>
HWY_FLATTEN void TransposeMicroKernel256Rect(const char* HWY_RESTRICT a,
                                             int64_t lda, char* HWY_RESTRICT b,
                                             int64_t ldb) {
  using D = hn::FixedTag<uint8_t, 32>;
  static constexpr size_t element_size = sizeof(T);
  static constexpr size_t row_bytes = element_size * bs;
  static_assert(row_bytes == kBlockBytes);
  static_assert(bs % 2 == 0);
  const D d;
  WithGatheredParts<bs / 2, kBlockBytes>(d, a, lda, [&](auto... vecs) HWY_ATTR {
    InterleaveWithinBlocksCPS<element_size, 1, kBlockBytes / 2>(
        [&](auto... interleaved) HWY_ATTR {
          ForEachIndexed(
              [&](auto i, auto&& vec) HWY_ATTR {
                const auto shuffled = ShuffleAcrossBlocks_3120(vec);
                StoreParts<kBlockBytes>(d, shuffled, b + 2 * i * ldb, ldb);
              },
              interleaved...);
        },
        vecs...);
  });
}
#endif  // HWY_MIN_BYTES >= 32

#if HWY_MIN_BYTES >= 64
// 512 bit transpose of a bs x bs block with 64 byte rows as 4 horizontal strips
// of `bs / 4` rows: each strip does full 64 byte loads in ascending row order,
// transposes each 128 bit block in place, and writes 16 byte slices across all
// `bs` output rows at stride `(bs / 4) * ldb`.
template <typename T, int bs>
HWY_FLATTEN void TransposeMicroKernel512RowStrips(const char* HWY_RESTRICT a,
                                                  int64_t lda,
                                                  char* HWY_RESTRICT b,
                                                  int64_t ldb) {
  using D = hn::FixedTag<uint8_t, 64>;
  static constexpr size_t row_bytes = 4 * kBlockBytes;
  static_assert(sizeof(T) * bs == row_bytes);
  constexpr int kRows = bs / 4;
  const D d;
  XLA_UNROLL_COUNT(1)
  for (int s = 0; s < 4; ++s) {
    const char* a_s = a + s * kRows * lda;
    char* b_s = b + s * static_cast<int64_t>(kBlockBytes);
    XLA_HIDE_POINTER_FROM_OPTIMIZER(a_s);
    XLA_HIDE_POINTER_FROM_OPTIMIZER(b_s);
    LoadRowsCPS<0, T, kRows, D, row_bytes>(
        d, a_s, lda, [&](auto... loads) HWY_ATTR {
          InterleaveWithinBlocksCPS<sizeof(T), 1, kBlockBytes>(
              [&](auto... interleaved) HWY_ATTR {
                ForEachIndexed(
                    [&](auto idx_c, auto&& vec) HWY_ATTR {
                      constexpr int64_t i = decltype(idx_c)::value;
                      StoreParts<kBlockBytes>(d, vec, b_s + i * ldb,
                                              kRows * ldb);
                    },
                    interleaved...);
              },
              loads...);
        });
  }
}

// 512 bit transpose of a bs x bs block with 32 byte rows: packs rows i and
// i + bs / 2 into each vector, interleaves within blocks up to 8 bytes, then
// pairs `(u_k, u_{k + bs/4})` to emit the fused 64 bit interleave plus
// `ConcatEvenBlocks` (rows 0 .. bs / 2 - 1) and `ConcatOddBlocks` (rows
// bs / 2 .. bs - 1) via `TwoTablesLookupLanes` (a single vpermt2q per pair on
// AVX-512; LLVM does not fold the two use unfused unpacks on its own).
template <typename T, int bs>
HWY_FLATTEN void TransposeMicroKernel512Rect256(const char* HWY_RESTRICT a,
                                                int64_t lda,
                                                char* HWY_RESTRICT b,
                                                int64_t ldb) {
  using D = hn::FixedTag<uint8_t, 64>;
  static constexpr size_t element_size = sizeof(T);
  static constexpr size_t row_bytes = 2 * kBlockBytes;
  static_assert(element_size * bs == row_bytes);
  static_assert(bs % 2 == 0);
  const D d;
  WithGatheredParts<bs / 2, row_bytes>(d, a, lda, [&](auto... vecs) HWY_ATTR {
    if constexpr (bs == 2) {
      const hn::Half<D> d_half;
      const auto row0 = hn::LowerHalf(
          d_half, ConcatEvenOddBlocks</*kOdd=*/false>(d, vecs..., vecs...));
      const auto row1 = hn::LowerHalf(
          d_half, ConcatEvenOddBlocks</*kOdd=*/true>(d, vecs..., vecs...));
      StoreRows(d_half, b, ldb, row0, row1);
    } else {
      using D64 = hn::Repartition<uint64_t, D>;
      HWY_ALIGN static constexpr int64_t kEvenIdx[8] = {0, 8, 4, 12,
                                                        1, 9, 5, 13};
      HWY_ALIGN static constexpr int64_t kOddIdx[8] = {2, 10, 6, 14,
                                                       3, 11, 7, 15};
      const D64 d64;
      const auto even_idx = hn::SetTableIndices(d64, kEvenIdx);
      const auto odd_idx = hn::SetTableIndices(d64, kOddIdx);
      InterleaveWithinBlocksCPS<element_size, 1, kMaxLaneBytes>(
          [&](auto... vs) HWY_ATTR {
            auto store_pair = [&](auto k, auto idx, int64_t row) HWY_ATTR {
              const auto a64 = hn::BitCast(d64, GetFromPack<k>(vs...));
              const auto b64 = hn::BitCast(d64, GetFromPack<k + bs / 4>(vs...));
              const auto permuted =
                  hn::TwoTablesLookupLanes(d64, a64, b64, idx);
              StoreParts<row_bytes>(d64, permuted, b + row * ldb, ldb);
            };
            WithIndices<bs / 4>([&](auto... k) HWY_ATTR {
              (store_pair(k, even_idx, 2 * k), ...);
              (store_pair(k, odd_idx, bs / 2 + 2 * k), ...);
            });
          },
          vecs...);
    }
  });
}

// Full 512 bit transpose of a bs x bs block with 64 byte rows using one 64 byte
// load and one 64 byte store per row in strictly ascending row order.
template <typename T, int bs>
HWY_FLATTEN void TransposeMicroKernel512Square(const char* HWY_RESTRICT a,
                                               int64_t lda,
                                               char* HWY_RESTRICT b,
                                               int64_t ldb) {
  using D = hn::FixedTag<uint8_t, 64>;
  static constexpr size_t element_size = sizeof(T);
  static constexpr size_t row_bytes = 4 * kBlockBytes;
  static_assert(element_size * bs == row_bytes);
  constexpr size_t kGroup = bs / 4;
  constexpr int64_t kGroupRows = kGroup;
  const D d;
  LoadRowsCPS<0, T, bs, D, row_bytes>(d, a, lda, [&](auto... loads) HWY_ATTR {
    InterleaveWithinBlocksCPS<element_size, 1, kBlockBytes>(
        [&](auto... vs) HWY_ATTR {
          auto store_half = [&](auto concat_halves, int64_t row_base) HWY_ATTR {
            auto pair01 = [&](auto k) HWY_ATTR {
              const auto& v0 = GetFromPack<k>(vs...);
              const auto& v1 = GetFromPack<k + kGroup>(vs...);
              return concat_halves(d, v1, v0);
            };
            auto pair23 = [&](auto k) HWY_ATTR {
              const auto& v2 = GetFromPack<k + 2 * kGroup>(vs...);
              const auto& v3 = GetFromPack<k + 3 * kGroup>(vs...);
              return concat_halves(d, v3, v2);
            };
            WithIndices<kGroup>([&](auto... k) HWY_ATTR {
              StoreRows(d, b + row_base * ldb, ldb,
                        ConcatEvenOddBlocks</*kOdd=*/false>(d, pair23(k),
                                                            pair01(k))...);
              StoreRows(d, b + (row_base + kGroupRows) * ldb, ldb,
                        ConcatEvenOddBlocks</*kOdd=*/true>(d, pair23(k),
                                                           pair01(k))...);
            });
          };
          auto concat_lo = [](auto tag, auto hi, auto lo) HWY_ATTR {
            return hn::ConcatLowerLower(tag, hi, lo);
          };
          auto concat_hi = [](auto tag, auto hi, auto lo) HWY_ATTR {
            return hn::ConcatUpperUpper(tag, hi, lo);
          };
          store_half(concat_lo, 0);
          store_half(concat_hi, 2 * kGroupRows);
        },
        loads...);
  });
}

// 512 bit transpose of a bs x bs block with 16 byte rows (bs >= 4), packing
// four rows per 512 bit vector and completing the 4 x 4 matrix of 32 bit chunks
// with a single 32 bit lane permute.
template <typename T, int bs>
HWY_FLATTEN void TransposeMicroKernel512Rect128(const char* HWY_RESTRICT a,
                                                int64_t lda,
                                                char* HWY_RESTRICT b,
                                                int64_t ldb) {
  using D = hn::FixedTag<uint8_t, 64>;
  using D32 = hn::Repartition<uint32_t, D>;
  static constexpr size_t element_size = sizeof(T);
  static_assert(element_size * bs == kBlockBytes);
  static_assert(bs >= 4 && bs % 4 == 0);
  HWY_ALIGN static constexpr int32_t kTranspose4x4Idx[16] = {
      0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
  const D d;
  const D32 d32;
  auto idx = hn::SetTableIndices(d32, kTranspose4x4Idx);
#if HWY_ARCH_X86 && (defined(__GNUC__) || defined(__clang__))
  // Prevent LLVM from cloning vpermd per extracted 128 bit block instead of
  // issuing one vpermd followed by vextracti32x4 memory stores.
  asm("" : "+v"(idx.raw));
#endif
  WithGatheredParts<bs / 4, kBlockBytes>(d, a, lda, [&](auto... vecs) HWY_ATTR {
    InterleaveWithinBlocksCPS<element_size, 1, sizeof(uint32_t)>(
        [&](auto... interleaved) HWY_ATTR {
          ForEachIndexed(
              [&](auto k, auto&& vec) HWY_ATTR {
                const auto v32 = hn::BitCast(d32, vec);
                const auto permuted = hn::TableLookupLanes(v32, idx);
                StoreParts<kBlockBytes>(d32, permuted, b + 4 * k * ldb, ldb);
              },
              interleaved...);
        },
        vecs...);
  });
}
#endif  // HWY_MIN_BYTES >= 64

// Scalar fallback
template <typename T, int bs>
HWY_FLATTEN void TransposeMicroKernelScalar(const char* HWY_RESTRICT a,
                                            int64_t lda, char* HWY_RESTRICT b,
                                            int64_t ldb) {
  for (int i = 0; i < bs; ++i) {
    for (int j = 0; j < bs; ++j) {
      std::memcpy(b + i * ldb + j * sizeof(T), a + j * lda + i * sizeof(T),
                  sizeof(T));
    }
  }
}

// Kernel selection for bs x bs blocks with 64, 32, and 16 byte rows. Each is
// defined only for targets whose vectors are wide enough to call it. They are
// HWY_FLATTEN so the kernels they call are inlined as if TransposeMicroKernel
// called them directly.
template <typename T, int bs>
HWY_INLINE HWY_FLATTEN void TransposeRows64(const char* HWY_RESTRICT a,
                                            int64_t lda, char* HWY_RESTRICT b,
                                            int64_t ldb);
template <typename T, int bs>
HWY_INLINE HWY_FLATTEN void TransposeRows32(const char* HWY_RESTRICT a,
                                            int64_t lda, char* HWY_RESTRICT b,
                                            int64_t ldb);
template <typename T, int bs>
HWY_INLINE HWY_FLATTEN void TransposeRows16(const char* HWY_RESTRICT a,
                                            int64_t lda, char* HWY_RESTRICT b,
                                            int64_t ldb);

#if HWY_MIN_BYTES >= 64
constexpr bool kIsX86 = HWY_ARCH_X86 != 0;
constexpr bool kIsZen4 = HWY_TARGET == HWY_AVX3_ZEN4;
constexpr bool kIsSpr = HWY_TARGET == HWY_AVX3_SPR;

// Returns whether 64 output rows `abs_ldb` bytes apart overflow 8 way and 12
// way x86 L1D sets: `abs_ldb` a multiple of 1024, or of 512 outside Zen 4. A
// 4096 byte aligned `abs_lda` aliases all 64 input rows into one L1D set, which
// cancels the conflict except for multiples of 1024 outside Zen 4.
HWY_INLINE bool HasX86L1dConflict512(int64_t abs_lda, int64_t abs_ldb) {
  constexpr int64_t kSetMask = kIsZen4 ? 1023 : 511;
  return kIsX86 && abs_ldb > 0 &&
         ((!kIsZen4 && (abs_ldb & 1023) == 0) ||
          ((abs_ldb & kSetMask) == 0 && (abs_lda & 4095) != 0));
}

template <typename T, int bs>
HWY_INLINE HWY_FLATTEN void TransposeRows64(const char* HWY_RESTRICT a,
                                            int64_t lda, char* HWY_RESTRICT b,
                                            int64_t ldb) {
  const int64_t abs_lda = std::abs(lda);
  const int64_t abs_ldb = std::abs(ldb);
  if constexpr (sizeof(T) == 1) {
    // Emit full 64 byte output rows via ColStrips when scattering or when the
    // output rows conflict in L1D.
    const bool l1d_conflict = HasX86L1dConflict512(abs_lda, abs_ldb);
    if (abs_ldb > abs_lda || l1d_conflict) {
      TransposeColStrips<64, 16, T, bs>(a, lda, b, ldb);
    } else {
      TransposeMicroKernel512RowStrips<T, bs>(a, lda, b, ldb);
    }
  } else if constexpr (sizeof(T) == 2) {
    TransposeColStrips<64, 32, T, bs>(a, lda, b, ldb);
  } else if constexpr (sizeof(T) == 4 && kIsX86 && !kIsZen4) {
    if (abs_lda <= kMaxOuterBlockSizeBytes &&
        abs_ldb <= kMaxOuterBlockSizeBytes) {
      TransposeColStrips<64, 32, T, bs>(a, lda, b, ldb);
    } else {
      TransposeMicroKernel512Square<T, bs>(a, lda, b, ldb);
    }
  } else {
    TransposeMicroKernel512Square<T, bs>(a, lda, b, ldb);
  }
}
#endif  // HWY_MIN_BYTES >= 64

#if HWY_MIN_BYTES >= 32
template <typename T, int bs>
HWY_INLINE HWY_FLATTEN void TransposeRows32(const char* HWY_RESTRICT a,
                                            int64_t lda, char* HWY_RESTRICT b,
                                            int64_t ldb) {
#if HWY_MIN_BYTES >= 64
  // Zen 4 double pumps 512 bit cross block permutes, so narrow elements (and
  // wide stride gathers on Zen 4 and SPR) run faster in 256 bit vectors across
  // 32 EVEX YMM registers.
  constexpr bool kAlways256 = kIsZen4 && sizeof(T) <= 2;
  constexpr bool kGather256 = (kIsZen4 && sizeof(T) == 8) ||
                              (kIsSpr && sizeof(T) >= 2 && sizeof(T) <= 8);
  if (kAlways256 || (kGather256 && std::abs(lda) > std::abs(ldb))) {
    constexpr size_t kStripBytes = sizeof(T) == 1 ? 16 : 32;
    TransposeColStrips<32, kStripBytes, T, bs>(a, lda, b, ldb);
  } else {
    TransposeMicroKernel512Rect256<T, bs>(a, lda, b, ldb);
  }
#else
  if constexpr (sizeof(T) == 1) {
    TransposeMicroKernel256Strips<T, bs>(a, lda, b, ldb);
  } else {
    TransposeColStrips<32, 32, T, bs>(a, lda, b, ldb);
  }
#endif
}

template <typename T, int bs>
HWY_INLINE HWY_FLATTEN void TransposeRows16(const char* HWY_RESTRICT a,
                                            int64_t lda, char* HWY_RESTRICT b,
                                            int64_t ldb) {
  const int64_t abs_lda = std::abs(lda);
  const int64_t abs_ldb = std::abs(ldb);
  // The 128 bit square kernel is preferred over the 256 bit rectangular one
  // when gathering (`abs_lda >= abs_ldb`) at a small input stride.
  const bool prefer_square_128 =
      abs_lda >= abs_ldb && abs_lda <= kMaxSquare128StrideBytes;
#if HWY_MIN_BYTES >= 64
  if constexpr (bs >= 4) {
    if (!kIsZen4 &&
        (HWY_TARGET != HWY_AVX3 || sizeof(T) <= 2 || prefer_square_128)) {
      TransposeMicroKernel512Rect128<T, bs>(a, lda, b, ldb);
    } else {
      TransposeMicroKernel256Rect<T, bs>(a, lda, b, ldb);
    }
    return;
  }
#endif
  if (prefer_square_128) {
    TransposeMicroKernel128<T, bs>(a, lda, b, ldb);
  } else {
    TransposeMicroKernel256Rect<T, bs>(a, lda, b, ldb);
  }
}
#endif  // HWY_MIN_BYTES >= 32

// Transposes one bs x bs block of T, picking a microkernel based on the row
// size, the target's vector width, and the stride magnitudes (`lda` and `ldb`
// may be negative or zero via `TransposePlan` input striding).
template <typename T, int bs>
HWY_FLATTEN void TransposeMicroKernel(const char* HWY_RESTRICT a, int64_t lda,
                                      char* HWY_RESTRICT b, int64_t ldb) {
  constexpr size_t row_bytes = sizeof(T) * bs;
  if constexpr (bs % 2 != 0) {
    TransposeMicroKernelScalar<T, bs>(a, lda, b, ldb);
  } else if constexpr (HWY_MIN_BYTES >= 64 && row_bytes == 4 * kBlockBytes) {
    TransposeRows64<T, bs>(a, lda, b, ldb);
  } else if constexpr (HWY_MIN_BYTES >= 32 && row_bytes == 2 * kBlockBytes) {
    TransposeRows32<T, bs>(a, lda, b, ldb);
  } else if constexpr (HWY_MIN_BYTES >= 32 && row_bytes == kBlockBytes) {
    TransposeRows16<T, bs>(a, lda, b, ldb);
  } else if constexpr (row_bytes <= kBlockBytes) {
    TransposeMicroKernel128<T, bs>(a, lda, b, ldb);
  } else {
    TransposeMicroKernelScalar<T, bs>(a, lda, b, ldb);
  }
}

template <typename T, int bs>
HWY_FLATTEN void TransposeMacroKernel(const char* HWY_RESTRICT a, int64_t lda,
                                      int outer_bs_a, char* HWY_RESTRICT b,
                                      int64_t ldb, int outer_bs_b) {
  // Keep macrokernel loops rolled for rows of at least 16 bytes (where outer_bs
  // is at most 4) to avoid runtime unroll remainder overhead; unroll j by 2
  // only for sub 16 byte bs == 2 kernels whose outer trip count is 8 to 32.
  constexpr int kInnerUnroll =
      (bs == 2 && sizeof(T) * bs < kBlockBytes) ? 2 : 1;
  XLA_UNROLL_COUNT(1)
  for (int i = 0; i < outer_bs_a; ++i) {
    XLA_UNROLL_COUNT(kInnerUnroll)
    for (int j = 0; j < outer_bs_b; ++j) {
      TransposeMicroKernel<T, bs>(a + bs * j * lda + i * bs * sizeof(T), lda,
                                  b + bs * i * ldb + j * bs * sizeof(T), ldb);
    }
  }
}

// Transposes `op->count` `[op->b_elems, op->a_elems]` tiles from `a + k*op->da`
// (row stride `op->lda` bytes) to `[op->a_elems, op->b_elems]` in
// `b + k*op->db` (row stride `op->ldb` bytes), where both `op->a_elems` and
// `op->b_elems` are exact multiples of `bs`. `packed` is
// `element_size | (bs << 16)`: both are packed into one template argument
// because `HWY_EXPORT_T` takes the function as a macro argument, which a comma
// in `F<a, b>` would split.
template <int packed>
ABSL_ATTRIBUTE_FUNC_ALIGN(64)
HWY_FLATTEN
    void OpFullTranspose2DHwy(const TransposeOp* HWY_RESTRICT op,
                              const char* HWY_RESTRICT a, char* HWY_RESTRICT b,
                              char* HWY_RESTRICT /*scratch*/) {
  constexpr size_t element_size = packed & 0xFFFF;
  constexpr int bs = packed >> 16;
  using T = hwy::UnsignedFromSize<element_size>;
  const int64_t count = op->count;
  const int64_t da = op->da;
  const int64_t db = op->db;
  const int64_t lda = op->lda;
  const int64_t ldb = op->ldb;
  const int outer_bs_a = op->a_elems / bs;
  const int outer_bs_b = op->b_elems / bs;
  if (count <= 0 || outer_bs_a <= 0 || outer_bs_b <= 0) {
    return;
  }
  for (int64_t k = 0;;) {
    const char* a_k = a;
    char* b_k = b;
    XLA_HIDE_POINTER_FROM_OPTIMIZER(a_k);
    XLA_HIDE_POINTER_FROM_OPTIMIZER(b_k);
    TransposeMacroKernel<T, bs>(a_k, lda, outer_bs_a, b_k, ldb, outer_bs_b);
    if (++k == count) {
      break;
    }
    a += da;
    b += db;
  }
}

template <int kBits, class D>
HWY_INLINE void PackSubbyteLoop(D d8, const uint8_t* HWY_RESTRICT in,
                                uint8_t* HWY_RESTRICT out, int64_t& i,
                                int64_t n) {
  const int64_t lanes = static_cast<int64_t>(hn::Lanes(d8));
  if constexpr (kBits == 4) {
    const auto mask = hn::Set(d8, 0x0F);
    const int64_t stride = 2 * lanes;
    for (; i + stride <= n; i += stride) {
      hn::Vec<D> v0, v1;
      hn::LoadInterleaved2(d8, in + i, v0, v1);
      auto packed = hn::Or(hn::And(v0, mask), hn::ShiftLeft<4>(v1));
      hn::StoreU(packed, d8, out + i / 2);
    }
  } else if constexpr (kBits == 2) {
    const auto mask = hn::Set(d8, 0x03);
    const int64_t stride = 4 * lanes;
    for (; i + stride <= n; i += stride) {
      hn::Vec<D> v0, v1, v2, v3;
      hn::LoadInterleaved4(d8, in + i, v0, v1, v2, v3);
      auto p01 = hn::Or(hn::And(v0, mask), hn::ShiftLeft<2>(hn::And(v1, mask)));
      auto p23 =
          hn::Or(hn::ShiftLeft<4>(hn::And(v2, mask)), hn::ShiftLeft<6>(v3));
      hn::StoreU(hn::Or(p01, p23), d8, out + i / 4);
    }
  } else {
    static_assert(kBits == 1);
    if (lanes >= 8 && lanes % 8 == 0) {
      const auto mask = hn::Set(d8, 0x01);
      for (; i + lanes <= n; i += lanes) {
        auto v = hn::LoadU(d8, in + i);
        auto m = hn::TestBit(v, mask);
        hn::StoreMaskBits(d8, m, out + i / 8);
      }
    }
  }
}

// Packs `op->count` rows of `op->lda` 1-byte elements from `a + r*op->da` into
// `kBits`-bit packed bytes at `b + r*op->db`, taking the low `kBits` bits of
// each input byte.
template <int kBits>
ABSL_ATTRIBUTE_FUNC_ALIGN(64)
HWY_FLATTEN
    void OpPackSubbyteHwy(const TransposeOp* HWY_RESTRICT op,
                          const char* HWY_RESTRICT a, char* HWY_RESTRICT b,
                          char* HWY_RESTRICT /*scratch*/) {
  constexpr int kElemsPerByte = 8 / kBits;
  constexpr uint8_t kMask = (1u << kBits) - 1u;
  const int64_t rows = op->count;
  const int64_t da = op->da;
  const int64_t db = op->db;
  const int64_t n = op->lda;
  const int64_t full_bytes = n / kElemsPerByte;
  const int64_t rem = n % kElemsPerByte;
  for (int64_t r = 0; r < rows; ++r) {
    const auto* in = reinterpret_cast<const uint8_t*>(a + r * da);
    auto* out = reinterpret_cast<uint8_t*>(b + r * db);
    int64_t i = 0;
    PackSubbyteLoop<kBits>(hn::ScalableTag<uint8_t>(), in, out, i, n);
    if (i == n) {
      continue;
    }
#if HWY_MIN_BYTES > 32
    PackSubbyteLoop<kBits>(hn::FixedTag<uint8_t, 32>(), in, out, i, n);
#endif
#if HWY_MIN_BYTES > 16 || HWY_HAVE_SCALABLE
    PackSubbyteLoop<kBits>(hn::Full128<uint8_t>(), in, out, i, n);
#endif
    for (int64_t byte_i = i / kElemsPerByte; byte_i < full_bytes; ++byte_i) {
      uint8_t byte = 0;
      for (int j = 0; j < kElemsPerByte; ++j) {
        byte |= static_cast<uint8_t>((in[byte_i * kElemsPerByte + j] & kMask)
                                     << (j * kBits));
      }
      out[byte_i] = byte;
    }
    if (rem != 0) {
      uint8_t byte = 0;
      for (int64_t j = 0; j < rem; ++j) {
        byte |= static_cast<uint8_t>(
            (in[full_bytes * kElemsPerByte + j] & kMask) << (j * kBits));
      }
      out[full_bytes] = byte;
    }
  }
}

int MaxTransposeInnerBlockSizeBytes(int elem_size_in_bytes) {
  if (elem_size_in_bytes >= 16) {
    return 16;
  }
  return std::clamp(HWY_MIN_BYTES, 16, 64);
}

}  // namespace HWY_NAMESPACE
}  // namespace xla
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace xla {

HWY_EXPORT(MaxTransposeInnerBlockSizeBytes);

void EnsureHighwayTargetInitialized() {
  hwy::ChosenTarget& chosen_target = hwy::GetChosenTarget();
  if (!chosen_target.IsInitialized()) {
    chosen_target.Update(hwy::SupportedTargets());
  }
}

namespace {

template <int kElemSize, int kBs = 2>
TransposeOp::Fn LookupTransposeMacroKernelOp(int bs) {
  if constexpr (kElemSize * kBs <= kMaxOuterBlockSizeBytes) {
    if (bs == kBs) {
      HWY_EXPORT_T(Table, OpFullTranspose2DHwy<kElemSize | (kBs << 16)>);
      return HWY_DYNAMIC_POINTER_T(Table);
    }
    return LookupTransposeMacroKernelOp<kElemSize, kBs * 2>(bs);
  }
  return nullptr;
}

}  // namespace

TransposeOp::Fn GetTransposeMacroKernelOp(int elem_size_in_bytes, int bs) {
  EnsureHighwayTargetInitialized();
  switch (elem_size_in_bytes) {
    case 1:
      return LookupTransposeMacroKernelOp<1>(bs);
    case 2:
      return LookupTransposeMacroKernelOp<2>(bs);
    case 4:
      return LookupTransposeMacroKernelOp<4>(bs);
    case 8:
      return LookupTransposeMacroKernelOp<8>(bs);
    case 16:
      return LookupTransposeMacroKernelOp<16>(bs);
    default:
      return nullptr;
  }
}

template <int kBits>
TransposeOp::Fn GetPackSubbyteOp() {
  EnsureHighwayTargetInitialized();
  HWY_EXPORT_T(Table, OpPackSubbyteHwy<kBits>);
  return HWY_DYNAMIC_POINTER_T(Table);
}

template TransposeOp::Fn GetPackSubbyteOp<1>();
template TransposeOp::Fn GetPackSubbyteOp<2>();
template TransposeOp::Fn GetPackSubbyteOp<4>();

int GetMaxTransposeInnerBlockSizeBytes(int elem_size_in_bytes) {
  return HWY_DYNAMIC_DISPATCH(MaxTransposeInnerBlockSizeBytes)(
      elem_size_in_bytes);
}

}  // namespace xla
#endif
