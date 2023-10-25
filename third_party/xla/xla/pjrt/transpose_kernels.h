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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

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

namespace xla {

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

template <size_t bytes>
__m128i LoadElementIntoVec128(const void* p);

template <>
inline __m128i LoadElementIntoVec128</*bytes=*/sizeof(uint16_t)>(
    const void* p) {
  // Note: We would ideally use `_mm_loadu_si16` here but older compilers do
  // not support it. However, we can replicate it using a sequence such that
  // even older compilers will turn this into a single movd instruction.
  // memcpy is used because `p` is not guaranteed to be aligned to a 2-byte
  // address.
  uint16_t load;
  memcpy(&load, p, sizeof(load));
  return _mm_cvtsi32_si128(load);
}

template <>
inline __m128i LoadElementIntoVec128</*bytes=*/sizeof(uint32_t)>(
    const void* p) {
  // Note: We would ideally use `_mm_loadu_si32` here but older compilers do
  // not support it. However, we can replicate it using a sequence such that
  // even older compilers will turn this into a single movd instruction.
  // memcpy is used because `p` is not guaranteed to be aligned to a 4-byte
  // address.
  uint32_t load;
  memcpy(&load, p, sizeof(load));
  return _mm_cvtsi32_si128(load);
}

template <>
inline __m128i LoadElementIntoVec128</*bytes=*/sizeof(uint64_t)>(
    const void* p) {
  // Note: We would ideally use `_mm_loadu_si64` here but older compilers do
  // not support it. However, we can replicate it using a sequence such that
  // even older compilers will turn this into a single movd instruction.
  // memcpy is used because `p` is not guaranteed to be aligned to a 8-byte
  // address.
  uint64_t load;
  memcpy(&load, p, sizeof(load));
  return _mm_cvtsi64_si128(load);
}

template <>
inline __m128i LoadElementIntoVec128</*bytes=*/sizeof(__m128i)>(const void* p) {
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}

template <size_t bytes, int lane>
inline void StoreElementFromVec128(void* p, __m128i v) {
  constexpr size_t element_start = bytes * lane;
  constexpr size_t element_end = element_start + bytes;
  static_assert(element_start >= 0);
  static_assert(element_end <= sizeof(Vec128));
  constexpr bool halfway = element_start == sizeof(Vec128) / 2;
  if constexpr (bytes == sizeof(uint16_t)) {
    // Note: We would ideally use `_mm_storeu_si16` here but older compilers do
    // not support it. However, we can replicate it using a sequence such that
    // even older compilers will turn this into a single movd instruction.
    // memcpy is used because `p` is not guaranteed to be aligned to a 4-byte
    // address.
    const uint16_t scalar = _mm_extract_epi16(v, lane);
    memcpy(p, &scalar, bytes);
  } else if constexpr (bytes == sizeof(uint32_t)) {
    if constexpr (halfway) {
      v = Unpack<bytes, Extract::kHi>(v, v);
    } else if constexpr (lane != 0) {
      v = _mm_shuffle_epi32(v, _MM_SHUFFLE(lane, lane, lane, lane));
    }
    // Note: We would ideally use `_mm_storeu_si32` here but older compilers do
    // not support it. However, we can replicate it using a sequence such that
    // even older compilers will turn this into a single movd instruction.
    // memcpy is used because `p` is not guaranteed to be aligned to a 4-byte
    // address.
    const uint32_t scalar = _mm_cvtsi128_si32(v);
    memcpy(p, &scalar, bytes);
  } else if constexpr (bytes == sizeof(uint64_t)) {
    if constexpr (halfway) {
      v = Unpack<bytes, Extract::kHi>(v, v);
    } else {
      static_assert(lane == 0);
    }
    // Note: We would ideally use `_mm_storeu_si64` here but older compilers do
    // not support it. However, we can replicate it using a sequence such that
    // even older compilers will turn this into a single movd instruction.
    // memcpy is used because `p` is not guaranteed to be aligned to a 8-byte
    // address.
    const uint64_t scalar = _mm_cvtsi128_si64(v);
    memcpy(p, &scalar, bytes);
  } else if constexpr (bytes == sizeof(__m128i)) {
    static_assert(lane == 0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
  } else {
    static_assert(bytes == 0);
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

template <size_t>
uint64x2_t LoadElementIntoVec128(const void* p);

template <>
inline uint64x2_t LoadElementIntoVec128</*bytes=*/sizeof(uint16_t)>(
    const void* p) {
  // Ideally, we would use `vld1q_lane_u16` but it assumes that its input is
  // aligned to a 16-bit boundary. We can only promise 8-bit aligned. That said,
  // this sequence will compile to `ldr St, [Xn]` but without an alignment hint.
  uint16_t x;
  memcpy(&x, p, sizeof(x));
  return vreinterpretq_u64_u16(vsetq_lane_u16(x, vdupq_n_u16(0), 0));
}

template <>
inline uint64x2_t LoadElementIntoVec128</*bytes=*/sizeof(uint32_t)>(
    const void* p) {
  // Ideally, we would use `vld1q_lane_u32` but it assumes that its input is
  // aligned to a 32-bit boundary. We can only promise 8-bit aligned. That said,
  // this sequence will compile to `ldr St, [Xn]` but without an alignment hint.
  uint32_t x;
  memcpy(&x, p, sizeof(x));
  return vreinterpretq_u64_u32(vsetq_lane_u32(x, vdupq_n_u32(0), 0));
}

template <>
inline uint64x2_t LoadElementIntoVec128</*bytes=*/sizeof(uint64_t)>(
    const void* p) {
  // Ideally, we would use `vld1q_lane_u64` but it assumes that its input is
  // aligned to a 64-bit boundary. We can only promise 8-bit aligned. That said,
  // this sequence will compile to `ldr Dt, [Xn]` but without an alignment hint.
  return vreinterpretq_u64_u8(
      vcombine_u8(vld1_u8(reinterpret_cast<const uint8_t*>(p)), vdup_n_u8(0)));
}

template <>
inline uint64x2_t LoadElementIntoVec128</*bytes=*/sizeof(uint64x2_t)>(
    const void* p) {
  return vreinterpretq_u64_u8(vld1q_u8(reinterpret_cast<const uint8_t*>(p)));
}

template <size_t bytes, int lane>
inline void StoreElementFromVec128(void* p, uint64x2_t v) {
  static_assert(bytes * (lane + 1) <= sizeof(uint64x2_t));
  if constexpr (bytes == sizeof(uint64x2_t)) {
    static_assert(lane == 0);
    vst1q_u8(reinterpret_cast<uint8_t*>(p), vreinterpretq_u8_u64(v));
  } else {
    if constexpr (bytes == sizeof(uint64_t)) {
      // Ideally, we would use `vst1q_lane_u64` but it assumes that its input is
      // aligned to a 64-bit boundary. We can only promise 8-bit aligned. That
      // said, this sequence will compile to `st1 {vt.d}[lane], [xn]` but
      // without an alignment hint.
      uint64_t extracted = vgetq_lane_u64(v, lane);
      memcpy(p, &extracted, sizeof(extracted));
    } else if constexpr (bytes == sizeof(uint32_t)) {
      // Ideally, we would use `vst1q_lane_u32` but it assumes that its input is
      // aligned to a 32-bit boundary. We can only promise 8-bit aligned. That
      // said, this sequence will compile to `st1 {vt.s}[lane], [xn]` but
      // without an alignment hint.
      uint32_t extracted = vgetq_lane_u32(vreinterpretq_u32_u64(v), lane);
      memcpy(p, &extracted, sizeof(extracted));
    } else if constexpr (bytes == sizeof(uint16_t)) {
      // Ideally, we would use `vst1q_lane_u16` but it assumes that its input is
      // aligned to a 16-bit boundary. We can only promise 8-bit aligned. That
      // said, this sequence will compile to `st1 {vt.h}[lane], [xn]` but
      // without an alignment hint.
      uint16_t extracted = vgetq_lane_u16(vreinterpretq_u16_u64(v), lane);
      memcpy(p, &extracted, sizeof(extracted));
    } else {
      static_assert(bytes == 0);
    }
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

template <size_t element_size, size_t step_size, size_t unpack_limit,
          typename T, size_t N>
inline std::array<T, N> UnpackSequence(const std::array<T, N>& last_transpose) {
  if constexpr (element_size * step_size < unpack_limit) {
    std::array<T, N> unpack =
        UnpackStep<element_size, step_size>(last_transpose);
    return UnpackSequence<element_size, step_size * 2, unpack_limit>(unpack);
  }
  return last_transpose;
}

template <size_t element_size, size_t bs, typename T, size_t N>
inline auto UnpackLowSequence(const std::array<T, N>& last_transpose) {
  if constexpr (N > 1 && element_size * bs < sizeof(T)) {
    static_assert(N % 2 == 0);
    std::array<T, N / 2> unpack;
    for (int i = 0; i < N; i += 2) {
      unpack[i / 2] = Unpack<element_size, Extract::kLo>(last_transpose[i],
                                                         last_transpose[i + 1]);
    }
    return UnpackLowSequence<element_size * 2, bs>(unpack);
  } else {
    return last_transpose;
  }
}

template <size_t bytes, size_t... lane>
inline void StoreElementsFromVec128(char* b, int64_t ldb, Vec128 x, size_t i,
                                    std::index_sequence<lane...>) {
  (StoreElementFromVec128</*bytes=*/bytes, lane>(b + ldb * (i + lane), x), ...);
}

template <typename T, int bs>
struct Vec128RectangularTransposeMicroKernelImpl {
  XLA_FLATTEN static void Apply(const char* __restrict a, int64_t lda,
                                char* __restrict b, int64_t ldb) {
    constexpr size_t element_size = sizeof(T);
    static_assert(sizeof(Vec128) % element_size == 0);
    std::array<Vec128, bs> loads;
    // `loads`:
    // [  0,  1,  2,  3 ]
    // [  4,  5,  6,  7 ]
    // [  8,  9, 10, 11 ]
    // [ 12, 13, 14, 15 ]
    XLA_UNROLL
    for (int i = 0; i < bs; ++i) {
      loads[i] = LoadElementIntoVec128<element_size * bs>(a + lda * i);
    }

    // Each load may not have filled the entire vector. Combine rows to get a
    // fully populated vector.
    auto last_transpose = UnpackLowSequence<element_size, bs>(loads);

    constexpr int kBytesInMatrix = element_size * bs * bs;
    static_assert(kBytesInMatrix <= sizeof(Vec128) * last_transpose.size());
    static_assert(bs % last_transpose.size() == 0);
    constexpr int kStoresPerCombinedRow = bs / last_transpose.size();

    // Each row of the matrix originally occupied some fraction of a vector.
    // We may need to finish the transpose if `last_transpose.size() > 1`.
    // `last_transpose`:
    // [ 0,  4,  1,  5,  2,  6,  3,  7 ]
    // [ 8, 12,  9, 13, 10, 14, 11, 15 ]
    last_transpose =
        UnpackSequence<kBytesInMatrix / (bs * last_transpose.size()),
                       /*step_size=*/1,
                       /*unpack_limit=*/kBytesInMatrix / bs>(last_transpose);
    // `last_transpose`:
    // [ 0,  4,  8, 12,  1,  5,  9, 13 ]
    // [ 2,  6, 10, 14,  3,  7, 11, 15 ]
    XLA_UNROLL
    for (size_t i = 0; i < last_transpose.size(); ++i) {
      StoreElementsFromVec128<element_size * bs>(
          b, ldb, last_transpose[i], i * kStoresPerCombinedRow,
          std::make_index_sequence<kStoresPerCombinedRow>{});
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
    static_assert(element_size <= sizeof(__m128i));
    static_assert(sizeof(__m128i) % element_size == 0);
    static_assert(bs % 2 == 0);
    static_assert(element_size * bs == sizeof(__m256i));
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
        UnpackSequence<element_size, /*step_size=*/1,
                       /*unpack_limit=*/sizeof(__m128i)>(last_transpose);

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
    static_assert(element_size <= sizeof(__m128i));
    static_assert(sizeof(__m128i) % element_size == 0);
    static_assert(bs % 2 == 0);
    static_assert(element_size * bs * 2 == sizeof(__m256i));
    std::array<__m256i, bs / 2> last_transpose;
    XLA_UNROLL
    for (int i = 0; i < bs / 2; ++i) {
      auto* lo = reinterpret_cast<const __m128i*>(a + lda * (i + 0));
      auto* hi = reinterpret_cast<const __m128i*>(a + lda * (i + bs / 2));
      last_transpose[i] =
          _mm256_set_m128i(_mm_loadu_si128(hi), _mm_loadu_si128(lo));
    }

    last_transpose =
        UnpackSequence<element_size, /*step_size=*/1,
                       /*unpack_limit=*/sizeof(__m128i) / 2>(last_transpose);

    if constexpr (element_size <= sizeof(__m128i) / 2) {
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

// The transpose kernel requires its input to be contiguous in one of the two
// dimensions being transposed, and the output to be contiguous in the other
// dimension.
//
// lda, ldb are strides in bytes.
template <typename T, int bs>
struct TransposeMicroKernel {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    if constexpr (bs % 2 == 0) {
#ifdef __AVX__
      if constexpr (sizeof(T) * bs == sizeof(__m256i)) {
        return AvxSquareTransposeMicroKernelImpl<T, bs>::Apply(a, lda, b, ldb);
      } else if constexpr (sizeof(T) * bs == sizeof(__m128i)) {
        return AvxRectangularTransposeMicroKernelImpl<T, bs>::Apply(a, lda, b,
                                                                    ldb);
      }
#endif
#ifdef XLA_HAS_VEC128
      if constexpr (sizeof(T) * bs <= sizeof(Vec128)) {
        return Vec128RectangularTransposeMicroKernelImpl<T, bs>::Apply(a, lda,
                                                                       b, ldb);
      }
#endif
    }
    for (int i = 0; i < bs; ++i) {
      for (int j = 0; j < bs; ++j) {
        *reinterpret_cast<T*>(b + i * ldb + j * sizeof(T)) =
            *reinterpret_cast<T const*>(a + j * lda + i * sizeof(T));
      }
    }
  }
};

#pragma pop_macro("XLA_FLATTEN")
#pragma pop_macro("XLA_UNROLL")

}  // namespace xla

#endif  // XLA_PJRT_TRANSPOSE_KERNELS_H_
