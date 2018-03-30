/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_SPARSE_MATMUL_OP_H_
#define TENSORFLOW_KERNELS_SPARSE_MATMUL_OP_H_

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/platform/types.h"

#if defined(PLATFORM_WINDOWS)
#include "tensorflow/core/platform/windows/cpu_info.h"
#include "tensorflow/core/platform/windows/intrinsics_port.h"
#endif

namespace Eigen {
namespace internal {

// Return the float representation of the bfloat16 value
// in the lower 16-bits of input
template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pexpand_bf16_l(const Packet& from) {
  tensorflow::uint32 tmp;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  tmp = (reinterpret_cast<const tensorflow::uint32&>(from)) & 0xffff0000;
#else
  tmp = (reinterpret_cast<const tensorflow::uint32&>(from) << 16) & 0xffff0000;
#endif
  return reinterpret_cast<const float&>(tmp);
}

// Return the float representation of the bfloat16 value
// in the upper 16-bits of input
template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pexpand_bf16_u(const Packet& from) {
  tensorflow::uint32 tmp;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  tmp = (reinterpret_cast<const tensorflow::uint32&>(from) << 16) & 0xffff0000;
#else
  tmp = (reinterpret_cast<const tensorflow::uint32&>(from)) & 0xffff0000;
#endif
  return reinterpret_cast<const float&>(tmp);
}

// Specialization non-scalar version on non-sse.
// Enable vectorization on z13 and higher
#if defined(EIGEN_VECTORIZE_ALTIVEC) || defined(EIGEN_VECTORIZE_VSX) || \
    defined(EIGEN_VECTORIZE_NEON) || defined(EIGEN_VECTORIZE_ZVECTOR)
template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet4f pexpand_bf16_l(const Packet4f& from) {
  float r[4];
  tensorflow::uint32 p[4];
  pstoreu(r, from);
  tensorflow::uint32* ir = reinterpret_cast<tensorflow::uint32*>(r);
  p[0] = (ir[0] << 16) & 0xffff0000;
  p[1] = ir[0] & 0xffff0000;
  p[2] = (ir[1] << 16) & 0xffff0000;
  p[3] = ir[1] & 0xffff0000;
  return ploadu<Packet4f>(reinterpret_cast<float*>(p));
}

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet4f pexpand_bf16_u(const Packet4f& from) {
  float r[4];
  tensorflow::uint32 p[4];
  pstoreu(r, from);
  tensorflow::uint32* ir = reinterpret_cast<tensorflow::uint32*>(r);
  p[0] = (ir[2] << 16) & 0xffff0000;
  p[1] = ir[2] & 0xffff0000;
  p[2] = (ir[3] << 16) & 0xffff0000;
  p[3] = ir[3] & 0xffff0000;
  return ploadu<Packet4f>(reinterpret_cast<float*>(p));
}
#endif

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pinterleave4x64(const Packet& from) {
  return from;
}

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pbroadcast_first(const Packet& a) {
  return a;
}

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pbroadcast_second(const Packet& a) {
  assert(false && "Not applicable to Scalar Values");
  return a;
}

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pbroadcast_third(const Packet& a) {
  assert(false && "Not applicable to Scalar Values");
  return a;
}

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pbroadcast_fourth(const Packet& a) {
  assert(false && "Not applicable to Scalar Values");
  return a;
}

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pload4bf16(
    const typename unpacket_traits<Packet>::type* from) {
  assert(false && "Not applicable to Scalar Values");
  return Packet();
}

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pload2bf16(
    const typename unpacket_traits<Packet>::type* from) {
  assert(false && "Not applicable to Scalar Values");
  return Packet();
}

// Specialization for pload4bf16 and pload2bf16 for non-sse.
// Enable vectorization on z13 and higher.
#if defined(EIGEN_VECTORIZE_ALTIVEC) || defined(EIGEN_VECTORIZE_VSX) || \
    defined(EIGEN_VECTORIZE_NEON) || defined(EIGEN_VECTORIZE_ZVECTOR)
template <>
EIGEN_STRONG_INLINE Packet4f pload4bf16<Packet4f>(const float* from) {
  tensorflow::uint32 p[4];
  const tensorflow::uint32* ir =
      reinterpret_cast<const tensorflow::uint32*>(from);
  p[0] = (ir[0] << 16) & 0xffff0000;
  p[1] = ir[0] & 0xffff0000;
  p[2] = (ir[1] << 16) & 0xffff0000;
  p[3] = ir[1] & 0xffff0000;
  return ploadu<Packet4f>(reinterpret_cast<float*>(p));
}

template <>
EIGEN_STRONG_INLINE Packet4f pload2bf16<Packet4f>(const float* from) {
  tensorflow::uint32 p[4];
  const tensorflow::uint32* ir =
      reinterpret_cast<const tensorflow::uint32*>(from);
  p[0] = (ir[0] << 16) & 0xffff0000;
  p[1] = ir[0] & 0xffff0000;
  p[2] = (ir[0] << 16) & 0xffff0000;
  p[3] = ir[0] & 0xffff0000;
  return ploadu<Packet4f>(reinterpret_cast<float*>(p));
}
#endif

#if defined(EIGEN_VECTORIZE_ALTIVEC) || defined(EIGEN_VECTORIZE_VSX)
// Return a packet with the first value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_first<Packet4f>(const Packet4f& a) {
  return vec_splat(a, 0);
}

// Return a packet with the second value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_second<Packet4f>(const Packet4f& a) {
  return vec_splat(a, 1);
}

// Return a packet with the third value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_third<Packet4f>(const Packet4f& a) {
  return vec_splat(a, 2);
}

// Return a packet with the fourth value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_fourth<Packet4f>(const Packet4f& a) {
  return vec_splat(a, 3);
}
#endif

#ifdef EIGEN_VECTORIZE_SSE2
// For PacketSize of 4 floats the Packet is not modified
template <>
EIGEN_STRONG_INLINE Packet4f pinterleave4x64<Packet4f>(const Packet4f& from) {
  return from;
}

// Return a Packet with 4 floats loaded from 4 bfloat16 values
template <>
EIGEN_STRONG_INLINE Packet4f pload4bf16<Packet4f>(const float* from) {
  __m128i zero = _mm_setzero_si128();
  __m128i tmp = _mm_castpd_si128(_mm_load_pd1((const double*)from));
  return _mm_castsi128_ps(_mm_unpacklo_epi16(zero, tmp));
}

// Return a Packet with 2 floats loaded from 2 bfloat16 values
template <>
EIGEN_STRONG_INLINE Packet4f pload2bf16<Packet4f>(const float* from) {
  __m128i zero = _mm_setzero_si128();
  __m128i tmp = _mm_castps_si128(_mm_load_ps1(from));
  return _mm_castsi128_ps(_mm_unpacklo_epi16(zero, tmp));
}

// Return a Packet with 4 floats expanded from 4 bfloat16 values
// in the lower half of the 128-bit lane
template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet4f pexpand_bf16_l(const Packet4f& from) {
  __m128i zero = _mm_setzero_si128();
  __m128i tmp = _mm_castps_si128(from);
  return _mm_castsi128_ps(_mm_unpacklo_epi16(zero, tmp));
}

// Return a Packet with 4 floats expanded from 4 bfloat16 values
// in the upper half of the 128-bit lane
template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet4f pexpand_bf16_u(const Packet4f& from) {
  __m128i zero = _mm_setzero_si128();
  __m128i tmp = _mm_castps_si128(from);
  return _mm_castsi128_ps(_mm_unpackhi_epi16(zero, tmp));
}

// Return a packet with the first value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_first<Packet4f>(const Packet4f& a) {
  return _mm_set1_ps(pfirst<Packet4f>(a));
}

// Return a packet with the second value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_second<Packet4f>(const Packet4f& a) {
  return _mm_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(a, a, 1)));
}

// Return a packet with the third value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_third<Packet4f>(const Packet4f& a) {
  return _mm_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(a, a, 2)));
}

// Return a packet with the fourth value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_fourth<Packet4f>(const Packet4f& a) {
  return _mm_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(a, a, 3)));
}

#endif

#ifdef EIGEN_VECTORIZE_AVX512
template <>
EIGEN_STRONG_INLINE Packet16f
pbroadcast_first<Packet16f>(const Packet16f& a_in) {
  Packet4f a = _mm512_castps512_ps128(a_in);
  return _mm512_broadcastss_ps(a);
}
template <>
EIGEN_STRONG_INLINE Packet16f
pbroadcast_second<Packet16f>(const Packet16f& a_in) {
  Packet4f a = _mm512_castps512_ps128(a_in);
  return _mm512_broadcastss_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1)));
}
template <>
EIGEN_STRONG_INLINE Packet16f
pbroadcast_third<Packet16f>(const Packet16f& a_in) {
  Packet4f a = _mm512_castps512_ps128(a_in);
  return _mm512_broadcastss_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 2, 2, 2)));
}
template <>
EIGEN_STRONG_INLINE Packet16f
pbroadcast_fourth<Packet16f>(const Packet16f& a_in) {
  Packet4f a = _mm512_castps512_ps128(a_in);
  return _mm512_broadcastss_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 3, 3)));
}
template <>
EIGEN_STRONG_INLINE Packet8d pbroadcast_first<Packet8d>(const Packet8d& a_in) {
  Packet2d a = _mm512_castpd512_pd128(a_in);
  return _mm512_broadcastsd_pd(a);
}
template <>
EIGEN_STRONG_INLINE Packet8d pbroadcast_second<Packet8d>(const Packet8d& a_in) {
  Packet2d a = _mm_permute_pd(_mm512_castpd512_pd128(a_in), 3);
  return _mm512_broadcastsd_pd(a);
}
template <>
EIGEN_STRONG_INLINE Packet8d pbroadcast_third<Packet8d>(const Packet8d& a_in) {
  Packet2d a = _mm256_extractf128_pd(_mm512_castpd512_pd256(a_in), 1);
  return _mm512_broadcastsd_pd(a);
}
template <>
EIGEN_STRONG_INLINE Packet8d pbroadcast_fourth<Packet8d>(const Packet8d& a_in) {
  Packet2d a =
      _mm_permute_pd(_mm256_extractf128_pd(_mm512_castpd512_pd256(a_in), 1), 3);
  return _mm512_broadcastsd_pd(a);
}
template <>
EIGEN_STRONG_INLINE Packet16i
pbroadcast_first<Packet16i>(const Packet16i& a_in) {
  Packet4i a = _mm512_castsi512_si128(a_in);
  return _mm512_broadcastd_epi32(a);
}
template <>
EIGEN_STRONG_INLINE Packet16i
pbroadcast_second<Packet16i>(const Packet16i& a_in) {
  Packet4i a = _mm512_castsi512_si128(a_in);
  return _mm512_broadcastd_epi32(_mm_shuffle_epi32(a, _MM_SHUFFLE(1, 1, 1, 1)));
}
template <>
EIGEN_STRONG_INLINE Packet16i
pbroadcast_third<Packet16i>(const Packet16i& a_in) {
  Packet4i a = _mm512_castsi512_si128(a_in);
  return _mm512_broadcastd_epi32(_mm_shuffle_epi32(a, _MM_SHUFFLE(2, 2, 2, 2)));
}
template <>
EIGEN_STRONG_INLINE Packet16i
pbroadcast_fourth<Packet16i>(const Packet16i& a_in) {
  Packet4i a = _mm512_castsi512_si128(a_in);
  return _mm512_broadcastd_epi32(_mm_shuffle_epi32(a, _MM_SHUFFLE(3, 3, 3, 3)));
}
#endif

#ifdef EIGEN_VECTORIZE_AVX
// For a Packet of Size 8 floats(256-bits), swap the 2nd and 3rd quadwords
template <>
EIGEN_STRONG_INLINE Packet8f pinterleave4x64<Packet8f>(const Packet8f& from) {
#ifdef EIGEN_VECTORIZE_AVX2
  return _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(from),
                                                      _MM_SHUFFLE(3, 1, 2, 0)));
#else
  auto tmp1 = _mm256_extract_epi32(_mm256_castps_si256(from), 2);
  auto tmp2 = _mm256_extract_epi32(_mm256_castps_si256(from), 3);
  auto tmp3 = _mm256_extract_epi32(_mm256_castps_si256(from), 4);
  auto tmp4 = _mm256_extract_epi32(_mm256_castps_si256(from), 5);
  auto tmp5 = _mm256_insert_epi32(_mm256_castps_si256(from), tmp1, 4);
  tmp5 = _mm256_insert_epi32(tmp5, tmp2, 5);
  tmp5 = _mm256_insert_epi32(tmp5, tmp3, 2);
  tmp5 = _mm256_insert_epi32(tmp5, tmp4, 3);
  return _mm256_castsi256_ps(tmp5);
#endif
}
// Return a Packet with 4 floats loaded from 4 bfloat16 values
template <>
EIGEN_STRONG_INLINE Packet8f pload4bf16<Packet8f>(const float* from) {
  __m128i zero = _mm_setzero_si128();
  __m128i tmp = _mm_castpd_si128(_mm_load_pd1((const double*)from));
  return _mm256_castps128_ps256(
      _mm_castsi128_ps(_mm_unpacklo_epi16(zero, tmp)));
}
// Return a Packet with 2 floats loaded from 2 bfloat16 values
template <>
EIGEN_STRONG_INLINE Packet8f pload2bf16<Packet8f>(const float* from) {
  __m128i zero = _mm_setzero_si128();
  __m128i tmp = _mm_castps_si128(_mm_load_ps1(from));
  return _mm256_castps128_ps256(
      _mm_castsi128_ps(_mm_unpacklo_epi16(zero, tmp)));
}

#ifdef EIGEN_VECTORIZE_AVX512
// Return a Packet with 4 floats loaded from 4 bfloat16 values
template <>
EIGEN_STRONG_INLINE Packet16f pload4bf16<Packet16f>(const float* from) {
  __m128i zero = _mm_setzero_si128();
  __m128i tmp = _mm_castpd_si128(_mm_load_pd1((const double*)from));
  return _mm512_castps128_ps512(
      _mm_castsi128_ps(_mm_unpacklo_epi16(zero, tmp)));
}
// Return a Packet with 2 floats loaded from 2 bfloat16 values
template <>
EIGEN_STRONG_INLINE Packet16f pload2bf16<Packet16f>(const float* from) {
  __m128i zero = _mm_setzero_si128();
  __m128i tmp = _mm_castps_si128(_mm_load_ps1(from));
  return _mm512_castps128_ps512(
      _mm_castsi128_ps(_mm_unpacklo_epi16(zero, tmp)));
}
#endif

// For each 128-bit lane convert 4 bfloat to 4 float values from the lower half
// of the 128-bit lane
template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet8f pexpand_bf16_l(const Packet8f& from) {
#ifdef EIGEN_VECTORIZE_AVX2
  __m256i zero = _mm256_setzero_si256();
  __m256i tmp = _mm256_castps_si256(from);
  return _mm256_castsi256_ps(_mm256_unpacklo_epi16(zero, tmp));
#else
  __m128i zero = _mm_setzero_si128();
  __m128i low = _mm_castps_si128(_mm256_extractf128_ps(from, 0));
  __m128i res_l = _mm_unpacklo_epi16(zero, low);
  __m128i high = _mm_castps_si128(_mm256_extractf128_ps(from, 1));
  __m128i res_h = _mm_unpacklo_epi16(zero, high);
  __m256 res = _mm256_castps128_ps256(_mm_castsi128_ps(res_l));
  res = _mm256_insertf128_ps(res, _mm_castsi128_ps(res_h), 1);
  return res;
#endif
}

// For each 128-bit lane convert 4 bfloat to 4 float values from the upper half
// of the 128-bit lane
template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet8f pexpand_bf16_u(const Packet8f& from) {
#ifdef EIGEN_VECTORIZE_AVX2
  __m256i zero = _mm256_setzero_si256();
  __m256i tmp = _mm256_castps_si256(from);
  return _mm256_castsi256_ps(_mm256_unpackhi_epi16(zero, tmp));
#else
  __m128i zero = _mm_setzero_si128();
  __m128i low = _mm_castps_si128(_mm256_extractf128_ps(from, 0));
  __m128i res_l = _mm_unpackhi_epi16(zero, low);
  __m128i high = _mm_castps_si128(_mm256_extractf128_ps(from, 1));
  __m128i res_h = _mm_unpackhi_epi16(zero, high);
  __m256 res = _mm256_castps128_ps256(_mm_castsi128_ps(res_l));
  res = _mm256_insertf128_ps(res, _mm_castsi128_ps(res_h), 1);
  return res;
#endif
}

// Return a packet with the first value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet8f pbroadcast_first<Packet8f>(const Packet8f& a) {
  return _mm256_set1_ps(pfirst<Packet8f>(a));
}

// Return a packet with the second value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet8f pbroadcast_second<Packet8f>(const Packet8f& a) {
  return _mm256_set1_ps(
      _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_permute_ps(a, 1))));
}

// Return a packet with the third value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet8f pbroadcast_third<Packet8f>(const Packet8f& a) {
  return _mm256_set1_ps(
      _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_permute_ps(a, 2))));
}

// Return a packet with the fourth value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet8f pbroadcast_fourth<Packet8f>(const Packet8f& a) {
  return _mm256_set1_ps(
      _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_permute_ps(a, 3))));
}

#endif

#ifdef EIGEN_VECTORIZE_AVX512

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet16f pexpand_bf16_l(const Packet16f& from) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(_mm512_castsi512_si256(_mm512_castps_si512(from))),
      16));
}

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet16f pexpand_bf16_u(const Packet16f& from) {
  Packet16i tmp = _mm512_castps_si512(from);
  Packet16i tmp2 = _mm512_alignr_epi32(tmp, tmp, 8);
  return _mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(_mm512_castsi512_si256(tmp2)), 16));
}

#endif
}  // namespace internal
}  // namespace Eigen
#endif
