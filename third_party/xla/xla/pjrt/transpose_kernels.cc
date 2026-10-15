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
#include "hwy//foreach_target.h"  // IWYU pragma: keep
#include "hwy//highway.h"

HWY_BEFORE_NAMESPACE();
namespace xla {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Highway's model: Vectors are composed of 128-bit blocks, blocks hold lanes.
// Interleave operations work within blocks. Cross-block movement requires
// block-level shuffles.
constexpr size_t kBlockBytes = 16;   // 128-bit blocks
constexpr size_t kMaxLaneBytes = 8;  // Lanes are at most 64 bits

enum class Extract { kLo, kHi };

// Unpack: Interleave lanes within each 128-bit block
template <size_t element_size, Extract extract, class V>
HWY_INLINE V Unpack(V a, V b) {
  static_assert(element_size <= kMaxLaneBytes);
  using D = hn::DFromV<V>;
  using ElementT = hwy::UnsignedFromSize<element_size>;
  const hn::Repartition<ElementT, D> d_elem;
  auto a_elem = hn::BitCast(d_elem, a);
  auto b_elem = hn::BitCast(d_elem, b);
  if constexpr (extract == Extract::kLo) {
    return hn::BitCast(D(), hn::InterleaveLower(d_elem, a_elem, b_elem));
  } else {
    return hn::BitCast(D(), hn::InterleaveUpper(d_elem, a_elem, b_elem));
  }
}

// Load/Store
template <class D, size_t bytes>
HWY_INLINE hn::Vec<D> LoadBytes(D d, const void* p) {
  const hn::FixedTag<uint8_t, bytes> d_fixed;
  return hn::ResizeBitCast(
      d, hn::LoadU(d_fixed, reinterpret_cast<const uint8_t*>(p)));
}

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

// Pack iteration helper (pack -> stream): ForEachIndexed
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

// CPS: UnpackStep (pack -> pack), lvalue-only inputs
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
    return Unpack<element_size * step_size, Extract::kLo>(a, b);
  } else {
    return Unpack<element_size * step_size, Extract::kHi>(a, b);
  }
}

template <size_t element_size, size_t step_size, class Cont, size_t... I,
          class... Vs>
HWY_INLINE void UnpackStepCPSImpl(Cont&& cont, std::index_sequence<I...>,
                                  Vs&... vs) {
  std::forward<Cont>(cont)(
      UnpackOutFromPack<I, element_size, step_size>(vs...)...);
}

template <size_t element_size, size_t step_size, class Cont, class... Vs>
HWY_INLINE void UnpackStepCPS(Cont&& cont, Vs&... in) {
  constexpr size_t N = sizeof...(Vs);
  static_assert(N % (step_size * 2) == 0);
  static_assert(element_size * step_size <= kMaxLaneBytes);
  UnpackStepCPSImpl<element_size, step_size>(
      std::forward<Cont>(cont), std::make_index_sequence<N>{}, in...);
}

// CPS: InterleaveWithinBlocks (pack -> pack).
// Lambdas need explicit HWY_ATTR:
// https://github.com/google/highway/blob/85cb4a5a266f7cc998ee4a270bdcd8f09926f63e/hwy/base.h#L206-L208
template <size_t element_size, size_t step_size, size_t unpack_limit,
          class Cont, class... Vs>
HWY_INLINE void InterleaveWithinBlocksCPS(Cont&& cont, Vs... in) {
  if constexpr (element_size * step_size < unpack_limit &&
                element_size * step_size <= kMaxLaneBytes) {
    UnpackStepCPS<element_size, step_size>(
        [&](auto... out) HWY_ATTR {
          // out... are by-value parameters; do not forward as rvalues.
          InterleaveWithinBlocksCPS<element_size, step_size * 2, unpack_limit>(
              std::forward<Cont>(cont), out...);
        },
        in...);  // lvalues inside this function
  } else {
    std::forward<Cont>(cont)(in...);
  }
}

// Helper template that applies the pairwise interleaving reduction.
template <size_t element_size, class Cont, size_t... I, class... Vs>
HWY_INLINE void CombinePairsAccImpl(Cont&& cont, std::index_sequence<I...>,
                                    Vs&&... vs) {
  std::forward<Cont>(cont)(Unpack<element_size, Extract::kLo>(
      GetFromPack<2 * I>(std::forward<Vs>(vs)...),
      GetFromPack<2 * I + 1>(std::forward<Vs>(vs)...))...);
}

// Applies a pairwise interleaving reduction across an arbitrary sequence of
// vectors. Adjacent vectors are combined using their lower halves, and the
// resulting halved-length sequence is forwarded to the provided continuation.
template <size_t element_size, class Cont, class... Vs>
HWY_INLINE void CombinePairsAcc(Cont&& cont, Vs&&... in) {
  CombinePairsAccImpl<element_size>(
      std::forward<Cont>(cont), std::make_index_sequence<sizeof...(Vs) / 2>{},
      std::forward<Vs>(in)...);
}

template <size_t element_size, size_t bs, size_t vector_bytes, class Cont,
          class... Vs>
HWY_INLINE void CombineRowsCPS(Cont&& cont, Vs... in) {
  constexpr size_t N = sizeof...(Vs);
  if constexpr (N > 1 && element_size * bs < vector_bytes) {
    static_assert(N % 2 == 0);
    CombinePairsAcc<element_size>(
        [&](auto... paired) HWY_ATTR {
          CombineRowsCPS<element_size * 2, bs, vector_bytes>(
              std::forward<Cont>(cont), paired...);
        },
        in...);
  } else {
    std::forward<Cont>(cont)(in...);
  }
}

// Stream -> Pack adapter for loads
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

// Pack -> Stream adapter for 128-bit stores
template <class D, size_t row_bytes, size_t stores_per_row, class... Vs>
HWY_INLINE void StoreVecs128(D d, char* b, int64_t ldb, Vs... v) {
  ForEachIndexed(
      [&](auto idx_c, auto&& vec) HWY_ATTR {
        constexpr size_t i = decltype(idx_c)::value;
        StoreLanes<D, row_bytes>(d, b, ldb, vec, i * stores_per_row,
                                 std::make_index_sequence<stores_per_row>{});
      },
      v...);
}

// 128-bit (single block) transpose
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
          constexpr size_t kNumVecs = sizeof...(vecs);
          constexpr size_t kBytesInMatrix = element_size * bs * bs;
          constexpr size_t kStoresPerRow = bs / kNumVecs;
          constexpr size_t kElementsPerVec = kBytesInMatrix / (bs * kNumVecs);
          InterleaveWithinBlocksCPS<kElementsPerVec, 1, row_bytes>(
              [&](auto... transposed) HWY_ATTR {
                StoreVecs128<D, row_bytes, kStoresPerRow>(d, b, ldb,
                                                          transposed...);
              },
              vecs...);
        },
        loads...);
  });
}

// 256-bit (two block) transposes
#if HWY_MIN_BYTES >= 32
template <class D, class... Vs>
HWY_INLINE void StoreRows256(D d, char* b, int64_t ldb, Vs... v) {
  ForEachIndexed(
      [&](auto idx_c, auto&& vec) HWY_ATTR {
        constexpr size_t i = decltype(idx_c)::value;
        hn::StoreU(vec, d, reinterpret_cast<hn::TFromD<D>*>(b + ldb * i));
      },
      v...);
}

// Invokes a continuation with an explicit parameter pack, rearranging the
// arguments such that all even-indexed elements precede all odd-indexed
// elements.
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

// Loads a square block of memory into 256-bit vectors, partitioning the
// data spatially. Invokes the continuation with the vectors corresponding
// to the left half of the block followed by the right half.
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

// Loads a rectangular block of memory into 256-bit vectors. Invokes the
// continuation with the constructed sequence of vectors.
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

// Entry point
template <typename T, int bs>
HWY_FLATTEN void TransposeMicroKernel(const char* HWY_RESTRICT a, int64_t lda,
                                      char* HWY_RESTRICT b, int64_t ldb) {
  constexpr size_t row_bytes = sizeof(T) * bs;
  if constexpr (bs % 2 != 0) {
    TransposeMicroKernelScalar<T, bs>(a, lda, b, ldb);
  } else {
#if HWY_MIN_BYTES >= 32
    if constexpr (row_bytes == 2 * kBlockBytes) {
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
