/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_PJRT_TRANSPOSE_KERNELS_H_
#define XLA_PJRT_TRANSPOSE_KERNELS_H_

#if (defined(__GNUC__) || defined(__clang__)) && defined(__SSE2__)
#define XLA_HAS_SSE2
#elif defined(_MSC_VER) && !defined(_M_ARM64EC) && defined(_M_X64)
#define XLA_HAS_SSE2
#elif defined(_MSC_VER) && !defined(_M_ARM64EC) && \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define XLA_HAS_SSE2
#elif defined(__AVX__)
#define XLA_HAS_SSE2
#endif

#if defined(__ARM_NEON) && !defined(__ARM_BIG_ENDIAN)
#define XLA_HAS_ARM_NEON
#endif

#ifdef XLA_HAS_SSE2
#include <immintrin.h>  // IWYU pragma: keep
#endif

#ifdef XLA_HAS_ARM_NEON
#include <arm_neon.h>
#endif

#if defined(XLA_HAS_SSE2) || defined(XLA_HAS_ARM_NEON)
#define XLA_HAS_VEC128
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace xla {

// Generic transpose kernel.
//
// All of the kernels that follow in this file are optimized versions of this
// generic kernel, specialized to particular block sizes and data types.
//
// The transpose kernel requires its input to be contiguous in one of the two
// dimensions being transposed, and the output to be contiguous in the other
// dimension.
//
// lda, ldb are strides in bytes.
template <typename T, int bs>
struct TransposeMicroKernel {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    for (int i = 0; i < bs; ++i) {
      for (int j = 0; j < bs; ++j) {
        *reinterpret_cast<T*>(b + i * ldb + j * sizeof(T)) =
            *reinterpret_cast<T const*>(a + j * lda + i * sizeof(T));
      }
    }
  }
};

#pragma push_macro("XLA_UNROLL")
#if defined(__clang__)
#define XLA_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#define XLA_UNROLL _Pragma("GCC unroll 128")
#else
#define XLA_UNROLL
#endif

#pragma push_macro("XLA_FLATTEN")
#if defined(__GNUC__) || defined(__clang__)
#define XLA_FLATTEN __attribute__((flatten))
#elif defined(_MSC_VER)
#define XLA_FLATTEN [[msvc::flatten]]
#else
#define XLA_FLATTEN
#endif

// The transpose microkernels use a general approach of zipping elements from
// different rows together. We start zipping together elements of size 1, size 2
// and so-on until we have achieved our transpose. As we increase the number of
// consecutive elements we are zipping together, we also increase the distance
// between the rows which are getting zipped.
//
// For example, let's say we want to transpose a 4x4 matrix:
//
// row 0: w0 w1 w2 w3
// row 1: x0 x1 x2 x3
// row 2: y0 y1 y2 y3
// row 3: z0 z1 z2 z3
//
// We first zip groups of single elements; we will zip row 0 and row 1 together
// and row 2 and row 3 together:
//
// row 0: w0 x0 w1 x1
// row 1: w2 x2 x3 x3
// row 2: y0 z0 y1 z1
// row 3: y2 z2 y3 z3
//
// Then, we zip groups of two elements; we will zip row 0 and row 2 together and
// row 1 and row 3 together:
//
// row 0: w0 x0 y0 z0
// row 1: w1 x1 y1 z1
// row 2: w2 x2 y2 z2
// row 3: w3 x3 y3 z3
//
// Note that as we double the number of elements we are zipping, we are also
// doubling the distance between rows which get zipped together.
//
// This standard algorithm gets slightly tweaked for what we will call
// "rectangular" transposes. Such transposes are trying to use vectors larger
// than the row length to speed up the transpose. This is accomplished by doing
// the outermost transpose first via loads and stores. If we go back to our
// original example, we would have:
//
// row 0: w0 w1 w2 w3 y0 y1 y2 y3
// row 1: x0 x1 x2 x3 z0 z1 z2 z3
//
// Now, we do a half zip of our elements:
//
// row 0: w0 x0 w1 x1 y0 z0 y1 z1
// row 1: w2 x2 w3 x3 y2 z2 y3 z3
//
// We can see that we have w{0-3} and y{0-3} in row 0 and x{0-3} and z{0-3} in
// row 1 but they are not in the right order. We need to shuffle them once to
// get them in the right order:
//
// row 0: w0 x0 y0 z0 w1 x1 y1 z1
// row 1: w1 x1 y1 z1 w2 x2 y2 z2
//
// Now, we can extract two rows of 4 elements from row 0 and two rows of 4
// elements from row 1 to store into memory.

enum class Extract { kLo, kHi };

#ifdef __AVX__
template <size_t element_size, Extract>
__m256i Unpack(__m256i a, __m256i b);

#if defined(__AVX2__)
template <>
inline __m256i Unpack<1, Extract::kLo>(__m256i a, __m256i b) {
  return _mm256_unpacklo_epi8(a, b);
}
template <>
inline __m256i Unpack<1, Extract::kHi>(__m256i a, __m256i b) {
  return _mm256_unpackhi_epi8(a, b);
}

template <>
inline __m256i Unpack<2, Extract::kLo>(__m256i a, __m256i b) {
  return _mm256_unpacklo_epi16(a, b);
}
template <>
inline __m256i Unpack<2, Extract::kHi>(__m256i a, __m256i b) {
  return _mm256_unpackhi_epi16(a, b);
}

template <>
inline __m256i Unpack<4, Extract::kLo>(__m256i a, __m256i b) {
  return _mm256_unpacklo_epi32(a, b);
}
template <>
inline __m256i Unpack<4, Extract::kHi>(__m256i a, __m256i b) {
  return _mm256_unpackhi_epi32(a, b);
}

template <>
inline __m256i Unpack<8, Extract::kLo>(__m256i a, __m256i b) {
  return _mm256_unpacklo_epi64(a, b);
}
template <>
inline __m256i Unpack<8, Extract::kHi>(__m256i a, __m256i b) {
  return _mm256_unpackhi_epi64(a, b);
}
#else
template <>
inline __m256i Unpack<1, Extract::kLo>(__m256i a, __m256i b) {
  __m128i a_hi = _mm256_extractf128_si256(a, 1);
  __m128i b_hi = _mm256_extractf128_si256(b, 1);
  __m128i a_lo = _mm256_castsi256_si128(a);
  __m128i b_lo = _mm256_castsi256_si128(b);
  __m128i hi = _mm_unpacklo_epi8(a_hi, b_hi);
  __m128i lo = _mm_unpacklo_epi8(a_lo, b_lo);
  return _mm256_set_m128i(hi, lo);
}
template <>
inline __m256i Unpack<1, Extract::kHi>(__m256i a, __m256i b) {
  __m128i a_hi = _mm256_extractf128_si256(a, 1);
  __m128i b_hi = _mm256_extractf128_si256(b, 1);
  __m128i a_lo = _mm256_castsi256_si128(a);
  __m128i b_lo = _mm256_castsi256_si128(b);
  __m128i hi = _mm_unpackhi_epi8(a_hi, b_hi);
  __m128i lo = _mm_unpackhi_epi8(a_lo, b_lo);
  return _mm256_set_m128i(hi, lo);
}

template <>
inline __m256i Unpack<2, Extract::kLo>(__m256i a, __m256i b) {
  __m128i a_hi = _mm256_extractf128_si256(a, 1);
  __m128i b_hi = _mm256_extractf128_si256(b, 1);
  __m128i a_lo = _mm256_castsi256_si128(a);
  __m128i b_lo = _mm256_castsi256_si128(b);
  __m128i hi = _mm_unpacklo_epi16(a_hi, b_hi);
  __m128i lo = _mm_unpacklo_epi16(a_lo, b_lo);
  return _mm256_set_m128i(hi, lo);
}
template <>
inline __m256i Unpack<2, Extract::kHi>(__m256i a, __m256i b) {
  __m128i a_hi = _mm256_extractf128_si256(a, 1);
  __m128i b_hi = _mm256_extractf128_si256(b, 1);
  __m128i a_lo = _mm256_castsi256_si128(a);
  __m128i b_lo = _mm256_castsi256_si128(b);
  __m128i hi = _mm_unpackhi_epi16(a_hi, b_hi);
  __m128i lo = _mm_unpackhi_epi16(a_lo, b_lo);
  return _mm256_set_m128i(hi, lo);
}

template <>
inline __m256i Unpack<4, Extract::kLo>(__m256i a, __m256i b) {
  return _mm256_castps_si256(
      _mm256_unpacklo_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
}
template <>
inline __m256i Unpack<4, Extract::kHi>(__m256i a, __m256i b) {
  return _mm256_castps_si256(
      _mm256_unpackhi_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
}

template <>
inline __m256i Unpack<8, Extract::kLo>(__m256i a, __m256i b) {
  return _mm256_castpd_si256(
      _mm256_unpacklo_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b)));
}
template <>
inline __m256i Unpack<8, Extract::kHi>(__m256i a, __m256i b) {
  return _mm256_castpd_si256(
      _mm256_unpackhi_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b)));
}
#endif
#endif

#ifdef XLA_HAS_SSE2
template <size_t element_size, Extract>
__m128i Unpack(__m128i a, __m128i b);

template <>
inline __m128i Unpack<1, Extract::kLo>(__m128i a, __m128i b) {
  return _mm_unpacklo_epi8(a, b);
}
template <>
inline __m128i Unpack<1, Extract::kHi>(__m128i a, __m128i b) {
  return _mm_unpackhi_epi8(a, b);
}

template <>
inline __m128i Unpack<2, Extract::kLo>(__m128i a, __m128i b) {
  return _mm_unpacklo_epi16(a, b);
}
template <>
inline __m128i Unpack<2, Extract::kHi>(__m128i a, __m128i b) {
  return _mm_unpackhi_epi16(a, b);
}

template <>
inline __m128i Unpack<4, Extract::kLo>(__m128i a, __m128i b) {
  return _mm_unpacklo_epi32(a, b);
}
template <>
inline __m128i Unpack<4, Extract::kHi>(__m128i a, __m128i b) {
  return _mm_unpackhi_epi32(a, b);
}

template <>
inline __m128i Unpack<8, Extract::kLo>(__m128i a, __m128i b) {
  return _mm_unpacklo_epi64(a, b);
}
template <>
inline __m128i Unpack<8, Extract::kHi>(__m128i a, __m128i b) {
  return _mm_unpackhi_epi64(a, b);
}

using Vec128 = __m128i;

template <typename T>
__m128i LoadElementIntoVec128(const void* p);

template <>
inline __m128i LoadElementIntoVec128<uint32_t>(const void* p) {
  // Note: We would ideally use `_mm_loadu_si32` here but older compilers do
  // not support it. However, we can replicate it using a sequence such that
  // even older compilers will turn this into a single movd instruction.
  // memcpy is used because `p` is not guaranteed to be aligned to a 4-byte
  // address.
  int load;
  memcpy(&load, p, sizeof(load));
  return _mm_cvtsi32_si128(load);
}

template <>
inline __m128i LoadElementIntoVec128<uint64_t>(const void* p) {
  return _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
}

template <>
inline __m128i LoadElementIntoVec128<__m128i>(const void* p) {
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}

template <typename T, int lane>
inline void StoreElementFromVec128(void* p, __m128i v) {
  if constexpr (sizeof(T) * lane == sizeof(Vec128) / 2) {
    return StoreElementFromVec128<T, 0>(p,
                                        Unpack<sizeof(T), Extract::kHi>(v, v));
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    if constexpr (lane != 0) {
      v = _mm_shuffle_epi32(v, _MM_SHUFFLE(lane, lane, lane, lane));
    }
    // Note: We would ideally use `_mm_storeu_si32` here but older compilers do
    // not support it. However, we can replicate it using a sequence such that
    // even older compilers will turn this into a single movd instruction.
    // memcpy is used because `p` is not guaranteed to be aligned to a 4-byte
    // address.
    memcpy(p, &v, sizeof(uint32_t));
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    static_assert(lane == 0);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(p), v);
  } else if constexpr (std::is_same_v<T, __m128i>) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
  } else {
    static_assert(sizeof(T) == 0);
  }
}
#endif

#ifdef XLA_HAS_ARM_NEON
template <size_t element_size, Extract>
uint64x2_t Unpack(uint64x2_t a, uint64x2_t b);

template <>
inline uint64x2_t Unpack<1, Extract::kLo>(uint64x2_t a, uint64x2_t b) {
  return vreinterpretq_u64_u8(
      vzipq_u8(vreinterpretq_u8_u64(a), vreinterpretq_u8_u64(b)).val[0]);
}
template <>
inline uint64x2_t Unpack<1, Extract::kHi>(uint64x2_t a, uint64x2_t b) {
  return vreinterpretq_u64_u8(
      vzipq_u8(vreinterpretq_u8_u64(a), vreinterpretq_u8_u64(b)).val[1]);
}

template <>
inline uint64x2_t Unpack<2, Extract::kLo>(uint64x2_t a, uint64x2_t b) {
  return vreinterpretq_u64_u16(
      vzipq_u16(vreinterpretq_u16_u64(a), vreinterpretq_u16_u64(b)).val[0]);
}
template <>
inline uint64x2_t Unpack<2, Extract::kHi>(uint64x2_t a, uint64x2_t b) {
  return vreinterpretq_u64_u16(
      vzipq_u16(vreinterpretq_u16_u64(a), vreinterpretq_u16_u64(b)).val[1]);
}

template <>
inline uint64x2_t Unpack<4, Extract::kLo>(uint64x2_t a, uint64x2_t b) {
  return vreinterpretq_u64_u32(
      vzipq_u32(vreinterpretq_u32_u64(a), vreinterpretq_u32_u64(b)).val[0]);
}
template <>
inline uint64x2_t Unpack<4, Extract::kHi>(uint64x2_t a, uint64x2_t b) {
  return vreinterpretq_u64_u32(
      vzipq_u32(vreinterpretq_u32_u64(a), vreinterpretq_u32_u64(b)).val[1]);
}

template <>
inline uint64x2_t Unpack<8, Extract::kLo>(uint64x2_t a, uint64x2_t b) {
  uint64x1_t a_lo = vget_low_u64(a);
  uint64x1_t b_lo = vget_low_u64(b);
  return vcombine_u64(a_lo, b_lo);
}
template <>
inline uint64x2_t Unpack<8, Extract::kHi>(uint64x2_t a, uint64x2_t b) {
  uint64x1_t a_hi = vget_high_u64(a);
  uint64x1_t b_hi = vget_high_u64(b);
  return vcombine_u64(a_hi, b_hi);
}

using Vec128 = uint64x2_t;

template <typename T>
uint64x2_t LoadElementIntoVec128(const void* p);

template <>
inline uint64x2_t LoadElementIntoVec128<uint32_t>(const void* p) {
  // Ideally, we would use `vld1q_lane_u32` but it assumes that its input is
  // aligned to a 32-bit boundary. We can only promise 8-bit aligned. That said,
  // this sequence will compile to `ldr St, [Xn]` but without an alignment hint.
  uint32_t x;
  memcpy(&x, p, sizeof(x));
  return vreinterpretq_u64_u32(vsetq_lane_u32(x, vdupq_n_u32(0), 0));
}

template <>
inline uint64x2_t LoadElementIntoVec128<uint64_t>(const void* p) {
  // Ideally, we would use `vld1q_lane_u64` but it assumes that its input is
  // aligned to a 64-bit boundary. We can only promise 8-bit aligned. That said,
  // this sequence will compile to `ldr Dt, [Xn]` but without an alignment hint.
  return vreinterpretq_u64_u8(
      vcombine_u8(vld1_u8(reinterpret_cast<const uint8_t*>(p)), vdup_n_u8(0)));
}

template <>
inline uint64x2_t LoadElementIntoVec128<uint64x2_t>(const void* p) {
  return vreinterpretq_u64_u8(vld1q_u8(reinterpret_cast<const uint8_t*>(p)));
}

template <typename T, int lane>
inline void StoreElementFromVec128(void* p, uint64x2_t v) {
  static_assert(sizeof(T) * (lane + 1) <= sizeof(uint64x2_t));
  if constexpr (std::is_same_v<T, uint64x2_t>) {
    vst1q_u8(reinterpret_cast<uint8_t*>(p), vreinterpretq_u8_u64(v));
  } else {
    T extracted;
    if constexpr (std::is_same_v<T, uint64_t>) {
      // Ideally, we would use `vst1q_lane_u64` but it assumes that its input is
      // aligned to a 64-bit boundary. We can only promise 8-bit aligned. That
      // said, this sequence will compile to `st1 {vt.d}[lane], [xn]` but
      // without an alignment hint.
      extracted = vgetq_lane_u64(v, lane);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      // Ideally, we would use `vst1q_lane_u32` but it assumes that its input is
      // aligned to a 32-bit boundary. We can only promise 8-bit aligned. That
      // said, this sequence will compile to `st1 {vt.s}[lane], [xn]` but
      // without an alignment hint.
      extracted = vgetq_lane_u32(vreinterpretq_u32_u64(v), lane);
    } else {
      static_assert(sizeof(T) == 0);
    }
    memcpy(p, &extracted, sizeof(extracted));
  }
}
#endif

#ifdef XLA_HAS_VEC128
template <size_t element_size, size_t step_size, typename T, size_t N>
inline std::array<T, N> UnpackStep(const std::array<T, N>& last_transpose) {
  static_assert(N % (step_size * 2) == 0);
  std::array<T, N> unpack;
  XLA_UNROLL
  for (int i = 0; i < N; i += step_size * 2) {
    XLA_UNROLL
    for (int j = 0; j < step_size; ++j) {
      unpack[i + 2 * j + 0] = Unpack<element_size * step_size, Extract::kLo>(
          last_transpose[i + j], last_transpose[i + j + step_size]);
      unpack[i + 2 * j + 1] = Unpack<element_size * step_size, Extract::kHi>(
          last_transpose[i + j], last_transpose[i + j + step_size]);
    }
  }
  return unpack;
}

template <size_t element_size, size_t step_size, size_t max_step_size,
          typename T, size_t N>
inline std::array<T, N> UnpackSequence(const std::array<T, N>& last_transpose) {
  if constexpr (element_size * step_size <= max_step_size) {
    std::array<T, N> unpack =
        UnpackStep<element_size, step_size>(last_transpose);
    return UnpackSequence<element_size, step_size * 2, max_step_size>(unpack);
  }
  return last_transpose;
}

template <typename T, int bs>
struct Vec128SquareTransposeMicroKernelImpl {
  XLA_FLATTEN static void Apply(const char* __restrict a, int64_t lda,
                                char* __restrict b, int64_t ldb) {
    constexpr size_t element_size = sizeof(T);
    static_assert(element_size <= 16);
    static_assert(16 % element_size == 0);
    static_assert(bs * element_size == sizeof(Vec128));
    std::array<Vec128, bs> last_transpose;
    XLA_UNROLL
    for (int i = 0; i < bs; ++i) {
      last_transpose[i] = LoadElementIntoVec128<Vec128>(a + lda * i);
    }

    last_transpose =
        UnpackSequence<element_size, /*step_size=*/1, /*max_step_size=*/8>(
            last_transpose);

    XLA_UNROLL
    for (int i = 0; i < bs; ++i) {
      StoreElementFromVec128<Vec128, 0>(b + ldb * i, last_transpose[i]);
    }
  }
};
#endif

#ifdef __AVX__
template <typename T, int bs>
struct AvxSquareTransposeMicroKernelImpl {
  XLA_FLATTEN static void Apply(const char* __restrict a, int64_t lda,
                                char* __restrict b, int64_t ldb) {
    constexpr size_t element_size = sizeof(T);
    static_assert(element_size <= 16);
    static_assert(16 % element_size == 0);
    static_assert(bs * element_size == sizeof(__m256i));
    std::array<__m256i, bs> last_transpose;
    XLA_UNROLL
    for (int i = 0; i < bs / 2; ++i) {
      auto* row0_low = reinterpret_cast<const __m128i*>(a + lda * (i + 0));
      auto* row0_high = row0_low + 1;
      auto* row1_low = reinterpret_cast<const __m128i*>(a + lda * (i + bs / 2));
      auto* row1_high = row1_low + 1;

      last_transpose[i] = _mm256_set_m128i(_mm_loadu_si128(row1_low),
                                           _mm_loadu_si128(row0_low));
      last_transpose[i + bs / 2] = _mm256_set_m128i(_mm_loadu_si128(row1_high),
                                                    _mm_loadu_si128(row0_high));
    }

    last_transpose =
        UnpackSequence<element_size, /*step_size=*/1, /*max_step_size=*/8>(
            last_transpose);

    XLA_UNROLL
    for (int i = 0; i < bs; ++i) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(b + ldb * i),
                          last_transpose[i]);
    }
  }
};
#endif

#ifdef __AVX__
template <typename T, int bs>
struct AvxRectangularTransposeMicroKernelImpl {
  XLA_FLATTEN static void Apply(const char* __restrict a, int64_t lda,
                                char* __restrict b, int64_t ldb) {
    constexpr size_t element_size = sizeof(T);
    static_assert(element_size <= 16);
    static_assert(16 % element_size == 0);
    static_assert(bs * element_size * 2 == sizeof(__m256i));
    std::array<__m256i, bs / 2> last_transpose;
    XLA_UNROLL
    for (int i = 0; i < bs / 2; ++i) {
      auto* lo = reinterpret_cast<const __m128i*>(a + lda * (i + 0));
      auto* hi = reinterpret_cast<const __m128i*>(a + lda * (i + bs / 2));
      last_transpose[i] =
          _mm256_set_m128i(_mm_loadu_si128(hi), _mm_loadu_si128(lo));
    }

    last_transpose =
        UnpackSequence<element_size, /*step_size=*/1, /*max_step_size=*/4>(
            last_transpose);

    if constexpr (element_size <= 8) {
      XLA_UNROLL
      for (int i = 0; i < bs / 2; ++i) {
#if defined(__AVX2__)
        last_transpose[i] = _mm256_permute4x64_epi64(last_transpose[i],
                                                     _MM_SHUFFLE(3, 1, 2, 0));
#else
        auto a = last_transpose[i];
        auto hi = _mm256_permute2f128_si256(a, a, 0b0001'0001);
        auto lo = _mm256_insertf128_si256(a, _mm256_castsi256_si128(a), 1);
        last_transpose[i] = _mm256_castpd_si256(_mm256_shuffle_pd(
            _mm256_castsi256_pd(lo), _mm256_castsi256_pd(hi), 0b1100));
#endif
      }
    }

    XLA_UNROLL
    for (int i = 0; i < bs / 2; ++i) {
      auto* lo = reinterpret_cast<__m128i*>(b + ldb * (i * 2 + 0));
      auto* hi = reinterpret_cast<__m128i*>(b + ldb * (i * 2 + 1));

      _mm_storeu_si128(lo, _mm256_castsi256_si128(last_transpose[i]));
      _mm_storeu_si128(hi, _mm256_extractf128_si256(last_transpose[i], 1));
    }
  }
};
#endif

#ifdef XLA_HAS_VEC128
template <>
struct TransposeMicroKernel<uint8_t, /*bs=*/4> {
  XLA_FLATTEN XLA_FLATTEN static void Apply(const char* __restrict a,
                                            int64_t lda, char* __restrict b,
                                            int64_t ldb) {
    using T = uint8_t;
    constexpr int bs = 4;
    constexpr size_t element_size = sizeof(T);
    std::array<Vec128, bs> loads;
    // [  0,  1,  2,  3 ]
    // [  4,  5,  6,  7 ]
    // [  8,  9, 10, 11 ]
    // [ 12, 13, 14, 15 ]
    XLA_UNROLL
    for (int i = 0; i < bs; ++i) {
      loads[i] = LoadElementIntoVec128<uint32_t>(a + lda * i);
    }
    // [  0,  4,  1,  5,  2,  6,  3,  7 ]
    Vec128 x_0_1 = Unpack<element_size, Extract::kLo>(loads[0], loads[1]);
    // [  8, 12,  9, 13, 10, 14, 11, 15 ]
    Vec128 x_2_3 = Unpack<element_size, Extract::kLo>(loads[2], loads[3]);
    // [  0,  4,  8, 12,  1,  5,  9, 13,  2,  6, 10, 14,  3,  7, 11, 15 ]
    Vec128 x = Unpack<element_size * 2, Extract::kLo>(x_0_1, x_2_3);

    // [  0,  4,  8, 12 ]
    StoreElementFromVec128<uint32_t, 0>(b + ldb * 0, x);
    // [  1,  5,  9, 13 ]
    StoreElementFromVec128<uint32_t, 1>(b + ldb * 1, x);
    // [  2,  6, 10, 14 ]
    StoreElementFromVec128<uint32_t, 2>(b + ldb * 2, x);
    // [  3,  7, 11, 15 ]
    StoreElementFromVec128<uint32_t, 3>(b + ldb * 3, x);
  }
};
#endif

#ifdef XLA_HAS_VEC128
template <>
struct TransposeMicroKernel<uint8_t, /*bs=*/8> {
  XLA_FLATTEN XLA_FLATTEN static void Apply(const char* __restrict a,
                                            int64_t lda, char* __restrict b,
                                            int64_t ldb) {
    using T = uint8_t;
    constexpr int bs = 8;
    constexpr size_t element_size = sizeof(T);
    // To help understand each step, let's show the contents of our SIMD
    // vectors.
    // The numbers shown are in octal and represent the source position from the
    // input in (row, column) format.
    //
    // [00, 01, 02, 03, 04, 05, 06, 07],
    // [10, 11, 12, 13, 14, 15, 16, 17],
    // [20, 21, 22, 23, 24, 25, 26, 27],
    // [30, 31, 32, 33, 34, 35, 36, 37],
    // [40, 41, 42, 43, 44, 45, 46, 47],
    // [50, 51, 52, 53, 54, 55, 56, 57],
    // [60, 61, 62, 63, 64, 65, 66, 67],
    // [70, 71, 72, 73, 74, 75, 76, 77],
    std::array<Vec128, bs> loads;
    XLA_UNROLL
    for (int i = 0; i < bs; ++i) {
      loads[i] = LoadElementIntoVec128<uint64_t>(a + lda * i);
    }

    // Pack adjacent loads together into one SIMD vector by interleaving the
    // lanes.
    //
    // [00, 10, 01, 11, 02, 12, 03, 13, 04, 14, 05, 15, 06, 16, 07, 17],
    // [20, 30, 21, 31, 22, 32, 23, 33, 24, 34, 25, 35, 26, 36, 27, 37],
    // [40, 50, 41, 51, 42, 52, 43, 53, 44, 54, 45, 55, 46, 56, 47, 57],
    // [60, 70, 61, 71, 62, 72, 63, 73, 64, 74, 65, 75, 66, 76, 67, 77],
    // In effect, we are splitting each SIMD vector into two blocks of 8
    // elements, then interleaving the elements.
    std::array<Vec128, bs / 2> last_transpose;
    XLA_UNROLL
    for (int i = 0; i < bs / 2; ++i) {
      // There is no need for `Unpack<1, Extract::kHi>` as the high half of the
      // two vectors contains zeros.
      last_transpose[i] =
          Unpack<element_size, Extract::kLo>(loads[i * 2], loads[i * 2 + 1]);
    }

    // [00, 10, 20, 30, 40, 50, 60, 70, 01, 11, 21, 31, 41, 51, 61, 71],
    // [02, 12, 22, 32, 42, 52, 62, 72, 03, 13, 23, 33, 43, 53, 63, 73],
    // [04, 14, 24, 34, 44, 54, 64, 74, 05, 15, 25, 35, 45, 55, 65, 75],
    // [06, 16, 26, 36, 46, 56, 66, 76, 07, 17, 27, 37, 47, 57, 67, 77],
    last_transpose =
        UnpackSequence<element_size * 2, /*step_size=*/1, /*max_step_size=*/4>(
            last_transpose);

    // We have two rows stored in our 128-bit SIMD vector but our block size
    // is 64-bit, unpack and do two stores.
    //
    // [00, 10, 20, 30, 40, 50, 60, 70],
    // [01, 11, 21, 31, 41, 51, 61, 71],
    // [02, 12, 22, 32, 42, 52, 62, 72],
    // [03, 13, 23, 33, 43, 53, 63, 73],
    // [04, 14, 24, 34, 44, 54, 64, 74],
    // [05, 15, 25, 35, 45, 55, 65, 75],
    // [06, 16, 26, 36, 46, 56, 66, 76],
    // [07, 17, 27, 37, 47, 57, 67, 77],
    XLA_UNROLL
    for (int i = 0; i < bs; i += 2) {
      StoreElementFromVec128<uint64_t, 0>(b + ldb * (i + 0),
                                          last_transpose[i / 2]);
      StoreElementFromVec128<uint64_t, 1>(b + ldb * (i + 1),
                                          last_transpose[i / 2]);
    }
  }
};
#endif

#ifdef __AVX__
template <>
struct TransposeMicroKernel<uint8_t, /*bs=*/16> {
  XLA_FLATTEN XLA_FLATTEN static void Apply(const char* __restrict a,
                                            int64_t lda, char* __restrict b,
                                            int64_t ldb) {
    AvxRectangularTransposeMicroKernelImpl<uint8_t, 16>::Apply(a, lda, b, ldb);
  }
};
#elif defined(XLA_HAS_VEC128)
template <>
struct TransposeMicroKernel<uint8_t, /*bs=*/16> {
  XLA_FLATTEN XLA_FLATTEN static void Apply(const char* __restrict a,
                                            int64_t lda, char* __restrict b,
                                            int64_t ldb) {
    Vec128SquareTransposeMicroKernelImpl<uint8_t, /*bs=*/16>::Apply(a, lda, b,
                                                                    ldb);
  }
};
#endif

#ifdef XLA_HAS_VEC128
template <>
struct TransposeMicroKernel<uint16_t, /*bs=*/4> {
  XLA_FLATTEN XLA_FLATTEN static void Apply(const char* __restrict a,
                                            int64_t lda, char* __restrict b,
                                            int64_t ldb) {
    using T = uint16_t;
    constexpr int bs = 4;
    constexpr size_t element_size = sizeof(T);
    // Note, Vec128 vectors can hold 8 uint16_t elements but our block size
    // is 4. We need to issue 4 loads to properly handle strides.
    //
    // [ 0,  1,  2,  3],
    // [ 4,  5,  6,  7],
    // [ 8,  9, 10, 11],
    // [12, 13, 14, 15],
    std::array<Vec128, bs> loads;
    XLA_UNROLL
    for (int i = 0; i < bs; ++i) {
      loads[i] = LoadElementIntoVec128<uint64_t>(a + lda * i);
    }

    // [ 0,  4,  1,  5,  2,  6,  3,  7 ]
    auto unpack_16_0 = Unpack<element_size, Extract::kLo>(loads[0], loads[1]);
    // [ 8, 12,  9, 13, 10, 14, 11, 15 ]
    auto unpack_16_1 = Unpack<element_size, Extract::kLo>(loads[2], loads[3]);

    // Note: there is no need for `Unpack<2, Extract::kHi>` as we only populate
    // the bottom 4 lanes.

    // [ 0,  4,  8, 12,  1,  5,  9, 13 ]
    auto unpack_32_0 =
        Unpack<element_size * 2, Extract::kLo>(unpack_16_0, unpack_16_1);
    // [ 2,  6, 10, 14,  3,  7, 11, 15 ]
    auto unpack_32_1 =
        Unpack<element_size * 2, Extract::kHi>(unpack_16_0, unpack_16_1);

    // [ 0,  4,  8, 12 ]
    StoreElementFromVec128<uint64_t, 0>(b + ldb * 0, unpack_32_0);
    // [ 1,  5,  9, 13 ]
    StoreElementFromVec128<uint64_t, 1>(b + ldb * 1, unpack_32_0);
    // [ 2,  6, 10, 14 ]
    StoreElementFromVec128<uint64_t, 0>(b + ldb * 2, unpack_32_1);
    // [ 3,  7, 11, 15 ]
    StoreElementFromVec128<uint64_t, 1>(b + ldb * 3, unpack_32_1);
  }
};
#endif

#if defined(__AVX__)
template <>
struct TransposeMicroKernel<uint16_t, /*bs=*/8> {
  XLA_FLATTEN XLA_FLATTEN static void Apply(const char* __restrict a,
                                            int64_t lda, char* __restrict b,
                                            int64_t ldb) {
    AvxRectangularTransposeMicroKernelImpl<uint16_t, 8>::Apply(a, lda, b, ldb);
  }
};
#elif defined(XLA_HAS_VEC128)
template <>
struct TransposeMicroKernel<uint16_t, /*bs=*/8> {
  XLA_FLATTEN XLA_FLATTEN static void Apply(const char* __restrict a,
                                            int64_t lda, char* __restrict b,
                                            int64_t ldb) {
    Vec128SquareTransposeMicroKernelImpl<uint16_t, /*bs=*/8>::Apply(a, lda, b,
                                                                    ldb);
  }
};
#endif

#ifdef __AVX__
template <>
struct TransposeMicroKernel<uint16_t, /*bs=*/16> {
  XLA_FLATTEN XLA_FLATTEN static void Apply(const char* __restrict a,
                                            int64_t lda, char* __restrict b,
                                            int64_t ldb) {
    AvxSquareTransposeMicroKernelImpl<uint16_t, /*bs=*/16>::Apply(a, lda, b,
                                                                  ldb);
  }
};
#endif

#ifdef __AVX__
template <>
struct TransposeMicroKernel<uint32_t, /*bs=*/4> {
  XLA_FLATTEN XLA_FLATTEN static void Apply(const char* __restrict a,
                                            int64_t lda, char* __restrict b,
                                            int64_t ldb) {
    AvxRectangularTransposeMicroKernelImpl<uint32_t, 4>::Apply(a, lda, b, ldb);
  }
};
#elif defined(XLA_HAS_VEC128)
template <>
struct TransposeMicroKernel<uint32_t, /*bs=*/4> {
  XLA_FLATTEN XLA_FLATTEN static void Apply(const char* __restrict a,
                                            int64_t lda, char* __restrict b,
                                            int64_t ldb) {
    Vec128SquareTransposeMicroKernelImpl<uint32_t, /*bs=*/4>::Apply(a, lda, b,
                                                                    ldb);
  }
};
#endif

#ifdef __AVX__
template <>
struct TransposeMicroKernel<uint32_t, /*bs=*/8> {
  XLA_FLATTEN XLA_FLATTEN static void Apply(const char* __restrict a,
                                            int64_t lda, char* __restrict b,
                                            int64_t ldb) {
    AvxSquareTransposeMicroKernelImpl<uint32_t, /*bs=*/8>::Apply(a, lda, b,
                                                                 ldb);
  }
};
#endif

#ifdef XLA_HAS_VEC128
template <>
struct TransposeMicroKernel<uint64_t, /*bs=*/2> {
  XLA_FLATTEN XLA_FLATTEN static void Apply(const char* __restrict a,
                                            int64_t lda, char* __restrict b,
                                            int64_t ldb) {
    Vec128SquareTransposeMicroKernelImpl<uint64_t, /*bs=*/2>::Apply(a, lda, b,
                                                                    ldb);
  }
};
#endif

#ifdef __AVX__
template <>
struct TransposeMicroKernel<uint64_t, /*bs=*/4> {
  XLA_FLATTEN XLA_FLATTEN static void Apply(const char* __restrict a,
                                            int64_t lda, char* __restrict b,
                                            int64_t ldb) {
    AvxSquareTransposeMicroKernelImpl<uint64_t, /*bs=*/4>::Apply(a, lda, b,
                                                                 ldb);
  }
};
#endif  // __AVX__

#pragma pop_macro("XLA_FLATTEN")
#pragma pop_macro("XLA_UNROLL")

}  // namespace xla

#endif  // XLA_PJRT_TRANSPOSE_KERNELS_H_
