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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "xla/pjrt/transpose_kernels.cc"  // NOLINT(whitespace/line_length)
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

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
HWY_INLINE void StoreLanes(D d, char* b, int64_t ldb, hn::Vec<D> v, size_t i,
                           std::index_sequence<lane...>) {
  (StoreLane<D, bytes, lane>(d, b + ldb * (i + lane), v), ...);
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

// Calls f(std::integral_constant<size_t, I>{}, ts_I) for each element of the
// pack, in order.
template <class F, size_t... I, class... Ts>
HWY_INLINE void ForEachIndexedImpl(F&& f, std::index_sequence<I...>,
                                   Ts&&... ts) {
  (std::forward<F>(f)(std::integral_constant<size_t, I>{},
                      std::forward<Ts>(ts)),
   ...);
}

template <class F, class... Ts>
HWY_INLINE void ForEachIndexed(F&& f, Ts&&... ts) {
  ForEachIndexedImpl(std::forward<F>(f),
                     std::make_index_sequence<sizeof...(Ts)>{},
                     std::forward<Ts>(ts)...);
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
        constexpr size_t i = decltype(idx_c)::value;
        StoreLanes<D, row_bytes>(d, b, ldb, vec, i * rows_per_vec,
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

// 256-bit (two block) transposes
#if HWY_MIN_BYTES >= 32
// Stores vector i to output row i.
template <class D, class... Vs>
HWY_INLINE void StoreRows256(D d, char* b, int64_t ldb, Vs... v) {
  ForEachIndexed(
      [&](auto idx_c, auto&& vec) HWY_ATTR {
        constexpr size_t i = decltype(idx_c)::value;
        hn::StoreU(vec, d, reinterpret_cast<hn::TFromD<D>*>(b + ldb * i));
      },
      v...);
}

// Calls `cont` with the even-indexed elements of the pack followed by the
// odd-indexed ones. The pack size must be even.
template <size_t... I, class Cont, class... Vs>
HWY_INLINE void CallContWithSeparatedImpl(Cont&& cont,
                                          std::index_sequence<I...>,
                                          Vs&&... vs) {
  std::forward<Cont>(cont)(GetFromPack<2 * I>(std::forward<Vs>(vs)...)...,
                           GetFromPack<2 * I + 1>(std::forward<Vs>(vs)...)...);
}

template <class Cont, class... Vs>
HWY_INLINE void CallContWithSeparated(Cont&& cont, Vs&&... vs) {
  CallContWithSeparatedImpl(std::forward<Cont>(cont),
                            std::make_index_sequence<sizeof...(Vs) / 2>{},
                            std::forward<Vs>(vs)...);
}

// Loads a bs x bs block whose rows span two 128-bit blocks. For I < bs / 2,
// L_I holds the left 16 bytes of rows I and I + bs / 2 (in its lower and upper
// block) and R_I holds the right 16 bytes. The continuation receives
// L_0, ..., L_{bs/2-1}, R_0, ..., R_{bs/2-1}: in this order, the in-block
// transpose leaves output row i in vector i. `vs...` accumulates the
// (L_I, R_I) pairs, which CallContWithSeparated reorders.
template <int I, typename T, int bs, class D, class DHalf, class Cont,
          class... Vs>
HWY_INLINE void BuildVecs256SquareCPS(D d, DHalf d_half, const char* a,
                                      int64_t lda, Cont& cont, Vs... vs) {
  if constexpr (I == bs / 2) {
    CallContWithSeparated(cont, vs...);
  } else {
    auto* row0 = reinterpret_cast<const hn::TFromD<DHalf>*>(a + lda * I);
    auto row1 =
        reinterpret_cast<const hn::TFromD<DHalf>*>(a + lda * (I + bs / 2));
    auto row0_lo = hn::LoadU(d_half, row0);
    auto row0_hi = hn::LoadU(d_half, row0 + kBlockBytes / sizeof(*row0));
    auto row1_lo = hn::LoadU(d_half, row1);
    auto row1_hi = hn::LoadU(d_half, row1 + kBlockBytes / sizeof(*row0));
    auto v_lo = hn::Combine(d, row1_lo, row0_lo);
    auto v_hi = hn::Combine(d, row1_hi, row0_hi);
    BuildVecs256SquareCPS<I + 1, T, bs>(d, d_half, a, lda, cont, vs..., v_lo,
                                        v_hi);
  }
}

template <typename T, int bs, class D, class DHalf, class Cont>
HWY_INLINE void BuildVecs256SquareCPS(D d, DHalf d_half, const char* a,
                                      int64_t lda, Cont& cont) {
  BuildVecs256SquareCPS<0, T, bs>(d, d_half, a, lda, cont);
}

template <typename T, int bs>
HWY_FLATTEN void TransposeMicroKernel256Square(const char* HWY_RESTRICT a,
                                               int64_t lda,
                                               char* HWY_RESTRICT b,
                                               int64_t ldb) {
  using D = hn::FixedTag<uint8_t, 32>;
  using DHalf = hn::Half<D>;
  static constexpr size_t element_size = sizeof(T);
  static constexpr size_t row_bytes = element_size * bs;
  static_assert(row_bytes == 2 * kBlockBytes);
  static_assert(bs % 2 == 0);
  const D d;
  const DHalf d_half;
  auto cont = [&](auto... vecs) HWY_ATTR {
    InterleaveWithinBlocksCPS<element_size, 1, kBlockBytes>(
        [&](auto... transposed)
            HWY_ATTR { StoreRows256(d, b, ldb, transposed...); },
        vecs...);
  };
  BuildVecs256SquareCPS<T, bs>(d, d_half, a, lda, cont);
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

// Stores the lower and upper 128-bit halves of vector i to output rows 2i and
// 2i + 1.
template <class DHalf, class... Vs>
HWY_INLINE void StoreRectPairs(DHalf d_half, char* b, int64_t ldb, Vs... v) {
  ForEachIndexed(
      [&](auto idx_c, auto&& vec) HWY_ATTR {
        constexpr size_t i = decltype(idx_c)::value;
        const auto lo = hn::LowerHalf(d_half, vec);
        const auto hi = hn::UpperHalf(d_half, vec);
        hn::StoreU(lo, d_half,
                   reinterpret_cast<hn::TFromD<DHalf>*>(b + ldb * (i * 2)));
        hn::StoreU(hi, d_half,
                   reinterpret_cast<hn::TFromD<DHalf>*>(b + ldb * (i * 2 + 1)));
      },
      v...);
}

// Loads a bs x bs block whose rows are one 128-bit block wide, two rows per
// vector: vector I holds row I in its lower block and row I + bs / 2 in its
// upper block. Calls `cont` with the bs / 2 vectors; `vs...` accumulates the
// vectors built so far.
template <int I, typename T, int bs, class D, class DHalf, class Cont,
          class... Vs>
HWY_INLINE void BuildVecs256RectCPS(D d, DHalf d_half, const char* a,
                                    int64_t lda, Cont& cont, Vs... vs) {
  if constexpr (I == bs / 2) {
    cont(vs...);
  } else {
    auto lo = hn::LoadU(
        d_half, reinterpret_cast<const hn::TFromD<DHalf>*>(a + lda * I));
    auto hi = hn::LoadU(d_half, reinterpret_cast<const hn::TFromD<DHalf>*>(
                                    a + lda * (I + bs / 2)));
    auto v = hn::Combine(d, hi, lo);
    BuildVecs256RectCPS<I + 1, T, bs>(d, d_half, a, lda, cont, vs..., v);
  }
}

template <typename T, int bs, class D, class DHalf, class Cont>
HWY_INLINE void BuildVecs256RectCPS(D d, DHalf d_half, const char* a,
                                    int64_t lda, Cont& cont) {
  BuildVecs256RectCPS<0, T, bs>(d, d_half, a, lda, cont);
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
                constexpr size_t i = decltype(idx_c)::value;
                constexpr size_t lo_row = 2 * i;
                constexpr size_t hi_row = 2 * i + bs / 2;
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
// I + bs / 2 share a vector (see BuildVecs256RectCPS), so the in-block network
// stops before its 8-byte stage, leaving vector i with the 64-bit lanes
// [r[2i].lo, r[2i+1].lo, r[2i].hi, r[2i+1].hi], where r[k].lo and r[k].hi are
// the first and last 8 bytes of output row k. That stage would pair lanes
// across blocks, so ShuffleAcrossBlocks_3120 does it instead, and
// StoreRectPairs stores rows 2i and 2i + 1.
template <typename T, int bs>
HWY_FLATTEN void TransposeMicroKernel256Rect(const char* HWY_RESTRICT a,
                                             int64_t lda, char* HWY_RESTRICT b,
                                             int64_t ldb) {
  using D = hn::FixedTag<uint8_t, 32>;
  using DHalf = hn::Half<D>;
  static constexpr size_t element_size = sizeof(T);
  static constexpr size_t row_bytes = element_size * bs;
  static_assert(row_bytes == kBlockBytes);
  static_assert(bs % 2 == 0);
  const D d;
  const DHalf d_half;
  auto cont = [&](auto... vecs) HWY_ATTR {
    InterleaveWithinBlocksCPS<element_size, 1, kBlockBytes / 2>(
        [&](auto... interleaved) HWY_ATTR {
          StoreRectPairs(d_half, b, ldb,
                         ShuffleAcrossBlocks_3120(interleaved)...);
        },
        vecs...);
  };
  BuildVecs256RectCPS<T, bs>(d, d_half, a, lda, cont);
}
#endif  // HWY_MIN_BYTES >= 32

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

// Transposes one bs x bs block of T, picking a microkernel based on the row
// size, the target's vector width, and (for 16-byte rows) the strides.
template <typename T, int bs>
HWY_FLATTEN void TransposeMicroKernel(const char* HWY_RESTRICT a, int64_t lda,
                                      char* HWY_RESTRICT b, int64_t ldb) {
  constexpr size_t row_bytes = sizeof(T) * bs;
  if constexpr (bs % 2 != 0) {
    TransposeMicroKernelScalar<T, bs>(a, lda, b, ldb);
  } else {
#if HWY_MIN_BYTES >= 32
    if constexpr (sizeof(T) == 1 && bs == 32) {
      TransposeMicroKernel256Strips<T, bs>(a, lda, b, ldb);
      return;
    } else if constexpr (row_bytes == 2 * kBlockBytes) {
      TransposeMicroKernel256Square<T, bs>(a, lda, b, ldb);
      return;
    } else if constexpr (row_bytes == kBlockBytes) {
      if (lda >= ldb && lda <= kMaxSquare128StrideBytes) {
        TransposeMicroKernel128<T, bs>(a, lda, b, ldb);
      } else {
        TransposeMicroKernel256Rect<T, bs>(a, lda, b, ldb);
      }
      return;
    }
#endif
    if constexpr (row_bytes <= kBlockBytes) {
      TransposeMicroKernel128<T, bs>(a, lda, b, ldb);
    } else {
      TransposeMicroKernelScalar<T, bs>(a, lda, b, ldb);
    }
  }
}

template <typename T, int bs>
HWY_FLATTEN void TransposeMacroKernel(const char* HWY_RESTRICT a, int64_t lda,
                                      int outer_bs_a, char* HWY_RESTRICT b,
                                      int64_t ldb, int outer_bs_b) {
  for (int i = 0; i < outer_bs_a; ++i) {
    for (int j = 0; j < outer_bs_b; ++j) {
      TransposeMicroKernel<T, bs>(a + bs * j * lda + i * bs * sizeof(T), lda,
                                  b + bs * i * ldb + j * bs * sizeof(T), ldb);
    }
  }
}

// `packed` is element_size | (bs << 16). Both are packed into one template
// argument because HWY_EXPORT_T takes the function as a macro argument, which
// a comma in `F<a, b>` would split.
template <int packed>
HWY_FLATTEN void TransposeMacroKernelPacked(const char* HWY_RESTRICT a,
                                            int64_t lda, int outer_bs_a,
                                            char* HWY_RESTRICT b, int64_t ldb,
                                            int outer_bs_b) {
  constexpr size_t element_size = packed & 0xFFFF;
  constexpr int bs = packed >> 16;
  using T = hwy::UnsignedFromSize<element_size>;
  TransposeMacroKernel<T, bs>(a, lda, outer_bs_a, b, ldb, outer_bs_b);
}

}  // namespace HWY_NAMESPACE
}  // namespace xla
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace xla {

template <int packed>
TransposeMacroKernelFn GetTransposeMacroKernelPacked() {
  HWY_EXPORT_T(Table, TransposeMacroKernelPacked<packed>);
  return HWY_DYNAMIC_POINTER_T(Table);
}

#define EXPLICIT_INSTANTIATION_PACKED(packed) \
  template TransposeMacroKernelFn GetTransposeMacroKernelPacked<packed>()

#define EXPLICIT_INSTANTIATION_FOR_SIZE(bs)      \
  EXPLICIT_INSTANTIATION_PACKED(1 | (bs << 16)); \
  EXPLICIT_INSTANTIATION_PACKED(2 | (bs << 16)); \
  EXPLICIT_INSTANTIATION_PACKED(4 | (bs << 16)); \
  EXPLICIT_INSTANTIATION_PACKED(8 | (bs << 16)); \
  EXPLICIT_INSTANTIATION_PACKED(16 | (bs << 16))

EXPLICIT_INSTANTIATION_FOR_SIZE(1);
EXPLICIT_INSTANTIATION_FOR_SIZE(2);
EXPLICIT_INSTANTIATION_FOR_SIZE(4);
EXPLICIT_INSTANTIATION_FOR_SIZE(8);
EXPLICIT_INSTANTIATION_FOR_SIZE(16);
EXPLICIT_INSTANTIATION_FOR_SIZE(32);
EXPLICIT_INSTANTIATION_FOR_SIZE(64);

#undef EXPLICIT_INSTANTIATION_FOR_SIZE
#undef EXPLICIT_INSTANTIATION_PACKED

}  // namespace xla
#endif
