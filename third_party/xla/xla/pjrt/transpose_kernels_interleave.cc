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

// On Arm CPUs with SVE/SVE2, Highway picks SVE2 over NEON by default. For
// deinterleave, SVE2 is slower here: its 3-way loads are slower than NEON's,
// and its stores cannot be paired into 32-byte writes.
//
// Disable SVE only in this file (before any header can pull in Highway) so
// deinterleave uses NEON on Arm64 while the 2D transpose and sub-byte packing
// code in transpose_kernels.cc stays untouched. Also skip `HWY_NEON_BF16` and
// `HWY_AVX3_SPR` (which only add bfloat16/FP16 over NEON/AVX3_DL) since nothing
// here uses those types.
#undef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_ALL_SVE | HWY_NEON_BF16 | HWY_AVX3_SPR)

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_TRANSPOSE_KERNELS_INTERLEAVE_ONCE_
#define TENSORFLOW_COMPILER_XLA_PJRT_TRANSPOSE_KERNELS_INTERLEAVE_ONCE_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "absl/base/attributes.h"
#include "xla/pjrt/transpose_kernels.h"
#include "xla/pjrt/transpose_kernels_internal.h"

namespace xla {
namespace {

// Plain fallback loop used for K = 5, 6, 7, unaligned buffers, and rows
// narrower than 8 bytes.
template <typename T, int K>
ABSL_ATTRIBUTE_NOINLINE void DeinterleaveScalar(int64_t count, int64_t da,
                                                int64_t db, int64_t ldb,
                                                int64_t n,
                                                const char* __restrict a,
                                                char* __restrict b) {
  for (int64_t r = 0; r < count; ++r) {
    const char* a_r = a + r * da;
    char* b_r = b + r * db;
    for (int64_t i = 0; i < n; ++i) {
      for (int k = 0; k < K; ++k) {
        std::memcpy(b_r + k * ldb + i * sizeof(T),
                    a_r + (i * K + k) * sizeof(T), sizeof(T));
      }
    }
  }
}

}  // namespace
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_TRANSPOSE_KERNELS_INTERLEAVE_ONCE_

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "xla/pjrt/transpose_kernels_interleave.cc"  // NOLINT(whitespace/line_length)
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/targets.h"

HWY_BEFORE_NAMESPACE();
namespace xla {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Loads 4 interleaved streams (`[a0,b0,c0,d0, a1,b1,c1,d1, ...]`) into
// `v0..v3`.
//
// On 64-bit Arm NEON, the 4-way `ld4` instruction is much slower than 2-way
// `ld2` or regular loads:
//   - For 1- to 4-byte elements, two `T`-typed `ld2` loads split the stream
//     into `(a, c)` and `(b, d)` lanes, and `ConcatEven`/`ConcatOdd` separate
//     `a` from `c` and `b` from `d`.
//   - For 8-byte elements on 128-bit NEON (2 elements per vector), do four
//     plain loads and rearrange them in registers.
template <class D>
HWY_INLINE void FastLoadInterleaved4(
    D d, const hn::TFromD<D>* HWY_RESTRICT unaligned, hn::Vec<D>& v0,
    hn::Vec<D>& v1, hn::Vec<D>& v2, hn::Vec<D>& v3) {
#if HWY_ARCH_ARM_A64 && (HWY_TARGET & HWY_ALL_NEON)
  if constexpr (sizeof(hn::TFromD<D>) <= 4 && D().MaxLanes() >= 2) {
    const size_t lanes = hn::Lanes(d);
    hn::Vec<D> ac_lo, bd_lo, ac_hi, bd_hi;
    hn::LoadInterleaved2(d, unaligned, ac_lo, bd_lo);
    hn::LoadInterleaved2(d, unaligned + 2 * lanes, ac_hi, bd_hi);
    v0 = hn::ConcatEven(d, ac_hi, ac_lo);
    v1 = hn::ConcatEven(d, bd_hi, bd_lo);
    v2 = hn::ConcatOdd(d, ac_hi, ac_lo);
    v3 = hn::ConcatOdd(d, bd_hi, bd_lo);
    return;
  }
  if constexpr (sizeof(hn::TFromD<D>) == 8 && D().MaxLanes() == 2) {
    auto x0 = hn::LoadU(d, unaligned + 0);        // [a0, b0]
    const auto x1 = hn::LoadU(d, unaligned + 2);  // [c0, d0]
    const auto x2 = hn::LoadU(d, unaligned + 4);  // [a1, b1]
    const auto x3 = hn::LoadU(d, unaligned + 6);  // [c1, d1]
#if defined(__GNUC__) || defined(__clang__)
    // Stop the compiler from combining these back into a slow `ld4`.
    asm("" : "+w"(x0.raw));
#endif
    v0 = hn::InterleaveLower(d, x0, x2);  // [a0, a1]
    v1 = hn::InterleaveUpper(d, x0, x2);  // [b0, b1]
    v2 = hn::InterleaveLower(d, x1, x3);  // [c0, c1]
    v3 = hn::InterleaveUpper(d, x1, x3);  // [d0, d1]
    return;
  }
#endif
  hn::LoadInterleaved4(d, unaligned, v0, v1, v2, v3);
}

template <typename T, int K>
HWY_INLINE auto MainDeinterleaveTag() {
#if HWY_MIN_BYTES > 32
  if constexpr (HWY_TARGET == HWY_AVX3_ZEN4 || (sizeof(T) == 8 && K == 4)) {
    // HWY_AVX3_ZEN4 splits 512-bit ops into 256-bit halves, so 256-bit avoids
    // extra shuffles. On AVX-512, 8-byte K = 4 with 512-bit ops hits heavy L1
    // cache misses across 4 widely spaced rows; 256-bit avoids that.
    return hn::FixedTag<T, 32 / sizeof(T)>();
  } else {
    return hn::ScalableTag<T>();
  }
#else
  return hn::ScalableTag<T>();
#endif
}

// Number of vectors written to each output row per main-loop step. Writing a
// full 64-byte cache line per row avoids cache conflicts when `ldb` is large,
// and on 128-bit Arm NEON lets the compiler pair 16-byte stores into 32-byte
// `stp` instructions.
template <int K, class D>
constexpr int BatchVectors() {
#if HWY_ARCH_ARM_A64 && (HWY_TARGET & HWY_ALL_NEON)
  if constexpr (D().MaxBytes() == 16) {
    return 4;
  }
#elif HWY_ARCH_X86 && HWY_MIN_BYTES >= 32
  if constexpr (K == 4 && sizeof(hn::TFromD<D>) == 8 && D().MaxBytes() == 32) {
    return 2;
  }
#endif
  return 1;
}

// Loads `M` consecutive groups of `K` interleaved vectors from `in` and writes
// `M` adjacent vectors to each of the `K` output rows starting at `b`.
template <int K, int M, class D>
HWY_INLINE void DeinterleaveBatch(D d, const hn::TFromD<D>* in, char* b,
                                  int64_t ldb) {
  static_assert(K >= 2 && K <= 4);
  static_assert(M == 1 || M == 2 || M == 4);
  using T = hn::TFromD<D>;
  using V = hn::Vec<D>;
#if HWY_HAVE_SCALABLE
  static_assert(M == 1);
  if constexpr (K == 2) {
    V v0, v1;
    hn::LoadInterleaved2(d, in, v0, v1);
    hn::StoreU(v0, d, reinterpret_cast<T*>(b));
    hn::StoreU(v1, d, reinterpret_cast<T*>(b + ldb));
  } else if constexpr (K == 3) {
    V v0, v1, v2;
    hn::LoadInterleaved3(d, in, v0, v1, v2);
    hn::StoreU(v0, d, reinterpret_cast<T*>(b));
    hn::StoreU(v1, d, reinterpret_cast<T*>(b + ldb));
    hn::StoreU(v2, d, reinterpret_cast<T*>(b + 2 * ldb));
  } else {
    V v0, v1, v2, v3;
    FastLoadInterleaved4(d, in, v0, v1, v2, v3);
    hn::StoreU(v0, d, reinterpret_cast<T*>(b));
    hn::StoreU(v1, d, reinterpret_cast<T*>(b + ldb));
    hn::StoreU(v2, d, reinterpret_cast<T*>(b + 2 * ldb));
    hn::StoreU(v3, d, reinterpret_cast<T*>(b + 3 * ldb));
  }
#else
  const size_t lanes = hn::Lanes(d);
  V v[M][K];
  for (int m = 0; m < M; ++m) {
    const T* p = in + m * K * lanes;
    if constexpr (K == 2) {
      hn::LoadInterleaved2(d, p, v[m][0], v[m][1]);
    } else if constexpr (K == 3) {
      hn::LoadInterleaved3(d, p, v[m][0], v[m][1], v[m][2]);
    } else {
      FastLoadInterleaved4(d, p, v[m][0], v[m][1], v[m][2], v[m][3]);
    }
  }
  for (int k = 0; k < K; ++k) {
    T* out = reinterpret_cast<T*>(b + k * ldb);
    for (int m = 0; m < M; ++m) {
      hn::StoreU(v[m][k], d, out + m * lanes);
    }
  }
#endif
}

template <typename Body>
HWY_INLINE void ForEachRow(int64_t count, int64_t da, int64_t db, const char* a,
                           char* b, Body body) {
  for (int64_t r = 0;;) {
    const char* a_r = a;
    char* b_r = b;
    XLA_HIDE_POINTER_FROM_OPTIMIZER(a_r);
    XLA_HIDE_POINTER_FROM_OPTIMIZER(b_r);
    body(a_r, b_r);
    if (++r == count) {
      break;
    }
    a += da;
    b += db;
  }
}

// Unzips `count` slices of `n * K` interleaved elements in `a` into `K` rows of
// `n` elements in `b` (`ldb` bytes between rows): `b[k][i] = a[i * K + k]`,
// using vector tag `d` (requires `n >= Lanes(d)`). Any leftover elements after
// the full-vector steps are handled with one overlapping vector starting at
// `n - Lanes(d)`.
template <int K, class D>
HWY_INLINE void DeinterleaveRows(D d, int64_t count, int64_t da, int64_t db,
                                 int64_t ldb, int64_t n, const char* a,
                                 char* b) {
  using T = hn::TFromD<D>;
  constexpr int kBatch = BatchVectors<K, D>();
  constexpr int64_t kElemBytes = static_cast<int64_t>(sizeof(T));
  const int64_t lanes = static_cast<int64_t>(hn::Lanes(d));
  const int64_t step = kBatch * lanes;
  const int64_t chunks = n / step;
  const int64_t done = chunks * step;

  auto run_chunks = [&](const char* a_r, char* b_r) HWY_ATTR {
    const T* in = reinterpret_cast<const T*>(a_r);
    char* b_cur = b_r;
#if HWY_ARCH_ARM_A64 && (HWY_TARGET & HWY_ALL_NEON) && defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for (int64_t c = 0; c < chunks;
         ++c, in += K * step, b_cur += step * kElemBytes) {
      DeinterleaveBatch<K, kBatch>(d, in, b_cur, ldb);
    }
  };
  auto run_tail = [&](const char* a_r, char* b_r) HWY_ATTR {
    auto step_at = [&](auto batch_tag, int64_t i) HWY_ATTR {
      DeinterleaveBatch<K, decltype(batch_tag)::value>(
          d, reinterpret_cast<const T*>(a_r + i * K * kElemBytes),
          b_r + i * kElemBytes, ldb);
    };
    int64_t i = done;
    if constexpr (kBatch >= 4) {
      if (i + 2 * lanes <= n) {
        step_at(std::integral_constant<int, 2>{}, i);
        i += 2 * lanes;
      }
    }
    if constexpr (kBatch >= 2) {
      if (i + lanes <= n) {
        step_at(std::integral_constant<int, 1>{}, i);
        i += lanes;
      }
    }
    if (i < n) {
      step_at(std::integral_constant<int, 1>{}, n - lanes);
    }
  };

  if (done == n) {
    ForEachRow(count, da, db, a, b, run_chunks);
  } else if (chunks == 0) {
    ForEachRow(count, da, db, a, b, run_tail);
  } else {
    ForEachRow(count, da, db, a, b, [&](const char* a_r, char* b_r) HWY_ATTR {
      run_chunks(a_r, b_r);
      run_tail(a_r, b_r);
    });
  }
}

// Variant of `DeinterleaveRows` for rows with `Lanes(d) <= n < 2 * Lanes(d)`.
template <int K, class D>
HWY_INLINE void DeinterleaveOneOrTwo(D d, int64_t count, int64_t da, int64_t db,
                                     int64_t ldb, int64_t n,
                                     const char* HWY_RESTRICT a,
                                     char* HWY_RESTRICT b) {
  using T = hn::TFromD<D>;
  constexpr int64_t kElemBytes = static_cast<int64_t>(sizeof(T));
  const int64_t rem = n - static_cast<int64_t>(hn::Lanes(d));
  if (rem == 0) {
    ForEachRow(count, da, db, a, b, [&](const char* a_r, char* b_r) HWY_ATTR {
      DeinterleaveBatch<K, 1>(d, reinterpret_cast<const T*>(a_r), b_r, ldb);
    });
  } else {
    ForEachRow(count, da, db, a, b, [&](const char* a_r, char* b_r) HWY_ATTR {
      DeinterleaveBatch<K, 1>(d, reinterpret_cast<const T*>(a_r), b_r, ldb);
      DeinterleaveBatch<K, 1>(
          d, reinterpret_cast<const T*>(a_r + rem * K * kElemBytes),
          b_r + rem * kElemBytes, ldb);
    });
  }
}

#if HWY_TARGET != HWY_SCALAR
// Handles rows shorter than `Lanes(main_d)` out of line so the main loop does
// not spill registers or load shuffle tables for narrower vector widths.
template <typename T, int K, size_t kMainBytes>
HWY_NOINLINE HWY_FLATTEN void DeinterleaveNarrowRows(int64_t count, int64_t da,
                                                     int64_t db, int64_t ldb,
                                                     int64_t n,
                                                     const char* HWY_RESTRICT a,
                                                     char* HWY_RESTRICT b) {
#if HWY_MIN_BYTES > 32
  if constexpr (kMainBytes > 32) {
    if (n >= static_cast<int64_t>(32 / sizeof(T))) {
      DeinterleaveOneOrTwo<K>(hn::FixedTag<T, 32 / sizeof(T)>(), count, da, db,
                              ldb, n, a, b);
      return;
    }
  }
#endif
#if HWY_MIN_BYTES > 16 || HWY_HAVE_SCALABLE
  if constexpr (kMainBytes > 16) {
    if (n >= static_cast<int64_t>(16 / sizeof(T))) {
#if HWY_HAVE_SCALABLE
      DeinterleaveRows<K>(hn::Full128<T>(), count, da, db, ldb, n, a, b);
#else
      DeinterleaveOneOrTwo<K>(hn::Full128<T>(), count, da, db, ldb, n, a, b);
#endif
      return;
    }
  }
#endif
  if constexpr (sizeof(T) <= 4) {
    if (n >= static_cast<int64_t>(8 / sizeof(T))) {
      DeinterleaveOneOrTwo<K>(hn::FixedTag<T, 8 / sizeof(T)>(), count, da, db,
                              ldb, n, a, b);
      return;
    }
  }
  DeinterleaveScalar<T, K>(count, da, db, ldb, n, a, b);
}
#endif

// Splits `op->count` contiguous runs of `op->b_elems * K` elements in
// `a + i*op->da` into `K` rows of `op->b_elems` elements spaced `op->ldb` bytes
// apart in `b + i*op->db`, computing `b[k][j] = a[K * j + k]`.
//
// Takes a single `Spec` type instead of `<elem_size, k>` because Highway 1.4.0
// passes the function name through `HWY_EXPORT_T` as one macro argument.
template <typename Spec>
ABSL_ATTRIBUTE_FUNC_ALIGN(64)
HWY_FLATTEN void OpDeinterleaveHwy(const TransposeOp* HWY_RESTRICT op,
                                   const char* a, char* b, char* /*scratch*/) {
  using T = hwy::UnsignedFromSize<Spec::kElemSize>;
  constexpr int k = Spec::kRows;
  const int64_t count = op->count;
  const int64_t da = op->da;
  const int64_t db = op->db;
  const int64_t ldb = op->ldb;
  const int b_elems = op->b_elems;
  if (count <= 0 || b_elems <= 0) {
    return;
  }
  if constexpr (sizeof(T) > 1) {
    const uintptr_t stride_bits =
        static_cast<uintptr_t>(ldb) |
        (count > 1 ? (static_cast<uintptr_t>(da) | static_cast<uintptr_t>(db))
                   : uintptr_t{0});
    if (HWY_UNLIKELY(((reinterpret_cast<uintptr_t>(a) |
                       reinterpret_cast<uintptr_t>(b) | stride_bits) &
                      (sizeof(T) - 1)) != 0)) {
      DeinterleaveScalar<T, k>(count, da, db, ldb, b_elems, a, b);
      return;
    }
  }
  const auto main_d = MainDeinterleaveTag<T, k>();
#if HWY_TARGET != HWY_SCALAR
  if (HWY_UNLIKELY(b_elems < static_cast<int64_t>(hn::Lanes(main_d)))) {
    DeinterleaveNarrowRows<T, k, decltype(main_d)().MaxBytes()>(
        count, da, db, ldb, b_elems, a, b);
    return;
  }
#endif
  DeinterleaveRows<k>(main_d, count, da, db, ldb, b_elems, a, b);
}

}  // namespace HWY_NAMESPACE
}  // namespace xla
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace xla {
namespace {

// Bundles element size and row count into one type so `HWY_EXPORT_T` receives a
// single macro argument without a comma (required for Highway 1.4.0).
template <size_t elem_size, int k>
struct DeinterleaveSpec {
  static constexpr size_t kElemSize = elem_size;
  static constexpr int kRows = k;
};

template <size_t elem_size, int k>
TransposeOp::Fn GetDeinterleaveHwyOp() {
  EnsureHighwayTargetInitialized();
  using Spec = DeinterleaveSpec<elem_size, k>;
  HWY_EXPORT_T(Table, OpDeinterleaveHwy<Spec>);
  return HWY_DYNAMIC_POINTER_T(Table);
}

template <typename T, int K>
ABSL_ATTRIBUTE_FUNC_ALIGN(64)
void OpDeinterleaveScalar(const TransposeOp* __restrict op,
                          const char* __restrict a, char* __restrict b,
                          char* __restrict /*scratch*/) {
  if (op->count <= 0 || op->b_elems <= 0) {
    return;
  }
  DeinterleaveScalar<T, K>(op->count, op->da, op->db, op->ldb, op->b_elems, a,
                           b);
}

template <size_t elem_size, int k>
TransposeOp::Fn PickDeinterleaveOp() {
  if constexpr (DeinterleaveKernelIsFaster(elem_size, k)) {
    if constexpr (k <= 4) {
      return GetDeinterleaveHwyOp<elem_size, k>();
    } else {
      return &OpDeinterleaveScalar<hwy::UnsignedFromSize<elem_size>, k>;
    }
  }
  return nullptr;
}

}  // namespace

TransposeOp::Fn GetDeinterleaveOp(size_t elem_size, int64_t k) {
  auto dispatch_k = [&](auto elem_tag) -> TransposeOp::Fn {
    constexpr size_t kSize = decltype(elem_tag)::value;
    switch (k) {
      case 2:
        return PickDeinterleaveOp<kSize, 2>();
      case 3:
        return PickDeinterleaveOp<kSize, 3>();
      case 4:
        return PickDeinterleaveOp<kSize, 4>();
      case 5:
        return PickDeinterleaveOp<kSize, 5>();
      case 6:
        return PickDeinterleaveOp<kSize, 6>();
      case 7:
        return PickDeinterleaveOp<kSize, 7>();
      default:
        return nullptr;
    }
  };
  switch (elem_size) {
    case 1:
      return dispatch_k(std::integral_constant<size_t, 1>{});
    case 2:
      return dispatch_k(std::integral_constant<size_t, 2>{});
    case 4:
      return dispatch_k(std::integral_constant<size_t, 4>{});
    case 8:
      return dispatch_k(std::integral_constant<size_t, 8>{});
    default:
      return nullptr;
  }
}

}  // namespace xla
#endif
