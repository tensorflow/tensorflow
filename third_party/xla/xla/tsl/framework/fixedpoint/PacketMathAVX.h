/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_FRAMEWORK_FIXEDPOINT_PACKETMATHAVX_H_
#define XLA_TSL_FRAMEWORK_FIXEDPOINT_PACKETMATHAVX_H_
#ifdef _MSC_VER

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

#endif

namespace Eigen {
namespace internal {

typedef eigen_packet_wrapper<__m256i, 10> Packet32q8i;
typedef eigen_packet_wrapper<__m128i, 11> Packet16q8i;

template <>
struct packet_traits<QInt8> : default_packet_traits {
  typedef Packet32q8i type;
  typedef Packet16q8i half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 32,
  };
  enum {
    HasAdd = 0,
    HasSub = 0,
    HasMul = 0,
    HasNegate = 0,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 0,
    HasMax = 0,
    HasConj = 0,
    HasSetLinear = 0
  };
};

template <>
struct unpacket_traits<Packet32q8i> {
  typedef QInt8 type;
  typedef Packet16q8i half;
  enum {
    size = 32,
    alignment = Aligned32,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
struct unpacket_traits<Packet16q8i> {
  typedef QInt8 type;
  typedef Packet16q8i half;
  enum {
    size = 16,
    alignment = Aligned32,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
EIGEN_STRONG_INLINE Packet32q8i pset1<Packet32q8i>(const QInt8& from) {
  return _mm256_set1_epi8(from.value);
}
template <>
EIGEN_STRONG_INLINE Packet32q8i ploadu<Packet32q8i>(const QInt8* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet16q8i ploadu<Packet16q8i>(const QInt8* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm_loadu_si128(
      reinterpret_cast<const __m128i*>(from));
}

template <>
EIGEN_STRONG_INLINE Packet32q8i pload<Packet32q8i>(const QInt8* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm256_load_si256(
      reinterpret_cast<const __m256i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet16q8i pload<Packet16q8i>(const QInt8* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm_load_si128(
      reinterpret_cast<const __m128i*>(from));
}

template <>
EIGEN_STRONG_INLINE void pstoreu<QInt8>(QInt8* to, const Packet32q8i& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(to), from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt8>(QInt8* to, const Packet16q8i& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm_storeu_si128(reinterpret_cast<__m128i*>(to),
                                               from.m_val);
}

template <>
EIGEN_STRONG_INLINE void pstore<QInt8>(QInt8* to, const Packet32q8i& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm256_store_si256(reinterpret_cast<__m256i*>(to),
                                               from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QInt8>(QInt8* to, const Packet16q8i& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm_store_si128(reinterpret_cast<__m128i*>(to),
                                            from.m_val);
}

typedef __m256 Packet8f;

template <>
struct type_casting_traits<float, QInt8> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 4, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet32q8i
pcast<Packet8f, Packet32q8i>(const Packet8f& a, const Packet8f& b,
                             const Packet8f& c, const Packet8f& d) {
  const __m256i a_conv = _mm256_cvtps_epi32(a);
  const __m256i b_conv = _mm256_cvtps_epi32(b);
  const __m256i c_conv = _mm256_cvtps_epi32(c);
  const __m256i d_conv = _mm256_cvtps_epi32(d);
  __m128i low = _mm256_castsi256_si128(a_conv);
  __m128i high = _mm256_extractf128_si256(a_conv, 1);
  __m128i tmp = _mm_packs_epi32(low, high);
  __m128i low2 = _mm256_castsi256_si128(b_conv);
  __m128i high2 = _mm256_extractf128_si256(b_conv, 1);
  __m128i tmp2 = _mm_packs_epi32(low2, high2);
  __m128i converted_low = _mm_packs_epi16(tmp, tmp2);
  low = _mm256_castsi256_si128(c_conv);
  high = _mm256_extractf128_si256(c_conv, 1);
  tmp = _mm_packs_epi32(low, high);
  low2 = _mm256_castsi256_si128(d_conv);
  high2 = _mm256_extractf128_si256(d_conv, 1);
  tmp2 = _mm_packs_epi32(low2, high2);
  __m128i converted_high = _mm_packs_epi16(tmp, tmp2);
  return _mm256_insertf128_si256(_mm256_castsi128_si256(converted_low),
                                 converted_high, 1);
}

}  // end namespace internal
}  // end namespace Eigen

#endif  // XLA_TSL_FRAMEWORK_FIXEDPOINT_PACKETMATHAVX_H_
