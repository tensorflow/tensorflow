#ifndef CXX11_SRC_FIXEDPOINT_PACKETMATHAVX512_H_
#define CXX11_SRC_FIXEDPOINT_PACKETMATHAVX512_H_

#include "PacketMathAVX2.h"

namespace Eigen {
namespace internal {

typedef eigen_packet_wrapper<__m512i, 30> Packet64q8i;
typedef eigen_packet_wrapper<__m512i, 31> Packet32q16i;
typedef eigen_packet_wrapper<__m512i, 32> Packet64q8u;
typedef eigen_packet_wrapper<__m512i, 33> Packet16q32i;

template <>
struct packet_traits<QInt8> : default_packet_traits {
  typedef Packet64q8i type;
  typedef Packet32q8i half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 64,
  };
  enum {
    HasAdd = 0,
    HasSub = 0,
    HasMul = 0,
    HasNegate = 0,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 1,
    HasMax = 1,
    HasConj = 0,
    HasSetLinear = 0
  };
};
template <>
struct packet_traits<QUInt8> : default_packet_traits {
  typedef Packet64q8u type;
  typedef Packet32q8u half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 64,
  };
  enum {
    HasAdd = 0,
    HasSub = 0,
    HasMul = 0,
    HasNegate = 0,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 1,
    HasMax = 1,
    HasConj = 0,
    HasSetLinear = 0
  };
};
template <>
struct packet_traits<QInt16> : default_packet_traits {
  typedef Packet32q16i type;
  typedef Packet16q16i half;
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
    HasMin = 1,
    HasMax = 1,
    HasConj = 0,
    HasSetLinear = 0
  };
};
template <>
struct packet_traits<QInt32> : default_packet_traits {
  typedef Packet16q32i type;
  typedef Packet8q32i half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 16,
  };
  enum {
    HasAdd = 1,
    HasSub = 1,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 0,
    HasAbs2 = 0,
    HasMin = 1,
    HasMax = 1,
    HasConj = 0,
    HasSetLinear = 0
  };
};

template <>
struct unpacket_traits<Packet64q8i> {
  typedef QInt8 type;
  typedef Packet32q8i half;
  enum {
    size = 64,
    alignment = Aligned64,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet32q16i> {
  typedef QInt16 type;
  typedef Packet16q16i half;
  enum {
    size = 32,
    alignment = Aligned64,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet64q8u> {
  typedef QUInt8 type;
  typedef Packet32q8u half;
  enum {
    size = 64,
    alignment = Aligned64,
    masked_load_available = false,
    masked_store_available = false
  };
};
template <>
struct unpacket_traits<Packet16q32i> {
  typedef QInt32 type;
  typedef Packet8q32i half;
  enum {
    size = 16,
    alignment = Aligned64,
    masked_load_available = false,
    masked_store_available = false
  };
};

// Unaligned load
template <>
EIGEN_STRONG_INLINE Packet64q8i ploadu<Packet64q8i>(const QInt8* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm512_loadu_si512(
      reinterpret_cast<const __m512i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet32q16i ploadu<Packet32q16i>(const QInt16* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm512_loadu_si512(
      reinterpret_cast<const __m512i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet64q8u ploadu<Packet64q8u>(const QUInt8* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm512_loadu_si512(
      reinterpret_cast<const __m512i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet16q32i ploadu<Packet16q32i>(const QInt32* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm512_loadu_si512(
      reinterpret_cast<const __m512i*>(from));
}

// Aligned load
template <>
EIGEN_STRONG_INLINE Packet64q8i pload<Packet64q8i>(const QInt8* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm512_load_si512(
      reinterpret_cast<const __m512i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet32q16i pload<Packet32q16i>(const QInt16* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm512_load_si512(
      reinterpret_cast<const __m512i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet64q8u pload<Packet64q8u>(const QUInt8* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm512_load_si512(
      reinterpret_cast<const __m512i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet16q32i pload<Packet16q32i>(const QInt32* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm512_load_si512(
      reinterpret_cast<const __m512i*>(from));
}

// Unaligned store
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt8>(QInt8* to, const Packet64q8i& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm512_storeu_si512(
      reinterpret_cast<__m512i*>(to), from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt16>(QInt16* to, const Packet32q16i& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm512_storeu_si512(
      reinterpret_cast<__m512i*>(to), from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QUInt8>(QUInt8* to, const Packet64q8u& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm512_storeu_si512(
      reinterpret_cast<__m512i*>(to), from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt32>(QInt32* to, const Packet16q32i& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm512_storeu_si512(
      reinterpret_cast<__m512i*>(to), from.m_val);
}

// Aligned store
template <>
EIGEN_STRONG_INLINE void pstore<QInt32>(QInt32* to, const Packet16q32i& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm512_store_si512(reinterpret_cast<__m512i*>(to),
                                               from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QUInt8>(QUInt8* to, const Packet64q8u& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm512_store_si512(reinterpret_cast<__m512i*>(to),
                                               from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QInt8>(QInt8* to, const Packet64q8i& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm512_store_si512(reinterpret_cast<__m512i*>(to),
                                               from.m_val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QInt16>(QInt16* to, const Packet32q16i& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm512_store_si512(reinterpret_cast<__m512i*>(to),
                                               from.m_val);
}

// Extract first element.
template <>
EIGEN_STRONG_INLINE QInt32 pfirst<Packet16q32i>(const Packet16q32i& a) {
  return _mm_cvtsi128_si32(_mm512_extracti32x4_epi32(a, 0));
}
template <>
EIGEN_STRONG_INLINE QUInt8 pfirst<Packet64q8u>(const Packet64q8u& a) {
  return static_cast<uint8_t>(
      _mm_extract_epi8(_mm512_extracti32x4_epi32(a.m_val, 0), 0));
}
template <>
EIGEN_STRONG_INLINE QInt8 pfirst<Packet64q8i>(const Packet64q8i& a) {
  return _mm_extract_epi8(_mm512_extracti32x4_epi32(a.m_val, 0), 0);
}
template <>
EIGEN_STRONG_INLINE QInt16 pfirst<Packet32q16i>(const Packet32q16i& a) {
  return _mm_extract_epi16(_mm512_extracti32x4_epi32(a.m_val, 0), 0);
}

// Initialize to constant value.
template <>
EIGEN_STRONG_INLINE Packet64q8i pset1<Packet64q8i>(const QInt8& from) {
  return _mm512_set1_epi8(from.value);
}
template <>
EIGEN_STRONG_INLINE Packet32q16i pset1<Packet32q16i>(const QInt16& from) {
  return _mm512_set1_epi16(from.value);
}
template <>
EIGEN_STRONG_INLINE Packet64q8u pset1<Packet64q8u>(const QUInt8& from) {
  return _mm512_set1_epi8(static_cast<uint8_t>(from.value));
}
template <>
EIGEN_STRONG_INLINE Packet16q32i pset1<Packet16q32i>(const QInt32& from) {
  return _mm512_set1_epi32(from.value);
}

// Basic arithmetic packet ops for QInt32.
template <>
EIGEN_STRONG_INLINE Packet16q32i padd<Packet16q32i>(const Packet16q32i& a,
                                                    const Packet16q32i& b) {
  return _mm512_add_epi32(a.m_val, b.m_val);
}
template <>
EIGEN_STRONG_INLINE Packet16q32i psub<Packet16q32i>(const Packet16q32i& a,
                                                    const Packet16q32i& b) {
  return _mm512_sub_epi32(a.m_val, b.m_val);
}
// Note: mullo truncates the result to 32 bits.
template <>
EIGEN_STRONG_INLINE Packet16q32i pmul<Packet16q32i>(const Packet16q32i& a,
                                                    const Packet16q32i& b) {
  return _mm512_mullo_epi32(a.m_val, b.m_val);
}
template <>
EIGEN_STRONG_INLINE Packet16q32i pnegate<Packet16q32i>(const Packet16q32i& a) {
  return _mm512_sub_epi32(_mm512_setzero_si512(), a.m_val);
}

// Min and max.
template <>
EIGEN_STRONG_INLINE Packet16q32i pmin<Packet16q32i>(const Packet16q32i& a,
                                                    const Packet16q32i& b) {
  return _mm512_min_epi32(a.m_val, b.m_val);
}
template <>
EIGEN_STRONG_INLINE Packet16q32i pmax<Packet16q32i>(const Packet16q32i& a,
                                                    const Packet16q32i& b) {
  return _mm512_max_epi32(a.m_val, b.m_val);
}

template <>
EIGEN_STRONG_INLINE Packet64q8u pmin<Packet64q8u>(const Packet64q8u& a,
                                                  const Packet64q8u& b) {
#ifdef EIGEN_VECTORIZE_AVX512BW
  return _mm512_min_epu8(a.m_val, b.m_val);
#else
  __m256i ap0 = _mm512_extracti32x8_epi32(a.m_val, 0);
  __m256i ap1 = _mm512_extracti32x8_epi32(a.m_val, 1);
  __m256i bp0 = _mm512_extracti32x8_epi32(b.m_val, 0);
  __m256i bp1 = _mm512_extracti32x8_epi32(b.m_val, 1);
  __m256i r0 = _mm256_min_epu8(ap0, bp0);
  __m256i r1 = _mm256_min_epu8(ap1, bp1);
  return _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
#endif
}
template <>
EIGEN_STRONG_INLINE Packet64q8u pmax<Packet64q8u>(const Packet64q8u& a,
                                                  const Packet64q8u& b) {
#ifdef EIGEN_VECTORIZE_AVX512BW
  return _mm512_max_epu8(a.m_val, b.m_val);
#else
  __m256i ap0 = _mm512_extracti32x8_epi32(a.m_val, 0);
  __m256i ap1 = _mm512_extracti32x8_epi32(a.m_val, 1);
  __m256i bp0 = _mm512_extracti32x8_epi32(b.m_val, 0);
  __m256i bp1 = _mm512_extracti32x8_epi32(b.m_val, 1);
  __m256i r0 = _mm256_max_epu8(ap0, bp0);
  __m256i r1 = _mm256_max_epu8(ap1, bp1);
  return _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
#endif
}

template <>
EIGEN_STRONG_INLINE Packet64q8i pmin<Packet64q8i>(const Packet64q8i& a,
                                                  const Packet64q8i& b) {
#ifdef EIGEN_VECTORIZE_AVX512BW
  return _mm512_min_epi8(a.m_val, b.m_val);
#else
  __m256i ap0 = _mm512_extracti32x8_epi32(a.m_val, 0);
  __m256i ap1 = _mm512_extracti32x8_epi32(a.m_val, 1);
  __m256i bp0 = _mm512_extracti32x8_epi32(b.m_val, 0);
  __m256i bp1 = _mm512_extracti32x8_epi32(b.m_val, 1);
  __m256i r0 = _mm256_min_epi8(ap0, bp0);
  __m256i r1 = _mm256_min_epi8(ap1, bp1);
  return _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
#endif
}
template <>
EIGEN_STRONG_INLINE Packet32q16i pmin<Packet32q16i>(const Packet32q16i& a,
                                                    const Packet32q16i& b) {
#ifdef EIGEN_VECTORIZE_AVX512BW
  return _mm512_min_epi16(a.m_val, b.m_val);
#else
  __m256i ap0 = _mm512_extracti32x8_epi32(a.m_val, 0);
  __m256i ap1 = _mm512_extracti32x8_epi32(a.m_val, 1);
  __m256i bp0 = _mm512_extracti32x8_epi32(b.m_val, 0);
  __m256i bp1 = _mm512_extracti32x8_epi32(b.m_val, 1);
  __m256i r0 = _mm256_min_epi16(ap0, bp0);
  __m256i r1 = _mm256_min_epi16(ap1, bp1);
  return _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
#endif
}
template <>
EIGEN_STRONG_INLINE Packet64q8i pmax<Packet64q8i>(const Packet64q8i& a,
                                                  const Packet64q8i& b) {
#ifdef EIGEN_VECTORIZE_AVX512BW
  return _mm512_max_epi8(a.m_val, b.m_val);
#else
  __m256i ap0 = _mm512_extracti32x8_epi32(a.m_val, 0);
  __m256i ap1 = _mm512_extracti32x8_epi32(a.m_val, 1);
  __m256i bp0 = _mm512_extracti32x8_epi32(b.m_val, 0);
  __m256i bp1 = _mm512_extracti32x8_epi32(b.m_val, 1);
  __m256i r0 = _mm256_max_epi8(ap0, bp0);
  __m256i r1 = _mm256_max_epi8(ap1, bp1);
  return _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
#endif
}
template <>
EIGEN_STRONG_INLINE Packet32q16i pmax<Packet32q16i>(const Packet32q16i& a,
                                                    const Packet32q16i& b) {
#ifdef EIGEN_VECTORIZE_AVX512BW
  return _mm512_max_epi16(a.m_val, b.m_val);
#else
  __m256i ap0 = _mm512_extracti32x8_epi32(a.m_val, 0);
  __m256i ap1 = _mm512_extracti32x8_epi32(a.m_val, 1);
  __m256i bp0 = _mm512_extracti32x8_epi32(b.m_val, 0);
  __m256i bp1 = _mm512_extracti32x8_epi32(b.m_val, 1);
  __m256i r0 = _mm256_max_epi16(ap0, bp0);
  __m256i r1 = _mm256_max_epi16(ap1, bp1);
  return _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
#endif
}

// Reductions.
template <>
EIGEN_STRONG_INLINE QInt32 predux_min<Packet16q32i>(const Packet16q32i& a) {
  Packet4i lane0 = _mm512_extracti32x4_epi32(a.m_val, 0);
  Packet4i lane1 = _mm512_extracti32x4_epi32(a.m_val, 1);
  Packet4i lane2 = _mm512_extracti32x4_epi32(a.m_val, 2);
  Packet4i lane3 = _mm512_extracti32x4_epi32(a.m_val, 3);
  Packet4i res =
      _mm_min_epi32(_mm_min_epi32(lane0, lane1), _mm_min_epi32(lane2, lane3));
  res = _mm_min_epi32(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 3, 2)));
  res = _mm_min_epi32(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 0, 1)));
  return pfirst(res);
}
template <>
EIGEN_STRONG_INLINE QInt32 predux_max<Packet16q32i>(const Packet16q32i& a) {
  Packet4i lane0 = _mm512_extracti32x4_epi32(a.m_val, 0);
  Packet4i lane1 = _mm512_extracti32x4_epi32(a.m_val, 1);
  Packet4i lane2 = _mm512_extracti32x4_epi32(a.m_val, 2);
  Packet4i lane3 = _mm512_extracti32x4_epi32(a.m_val, 3);
  Packet4i res =
      _mm_max_epi32(_mm_max_epi32(lane0, lane1), _mm_max_epi32(lane2, lane3));
  res = _mm_max_epi32(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 3, 2)));
  res = _mm_max_epi32(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 0, 1)));
  return pfirst(res);
}
template <>
EIGEN_STRONG_INLINE QInt16 predux_min<Packet32q16i>(const Packet32q16i& a) {
  Packet4i lane0 = _mm512_extracti32x4_epi32(a.m_val, 0);
  Packet4i lane1 = _mm512_extracti32x4_epi32(a.m_val, 1);
  Packet4i lane2 = _mm512_extracti32x4_epi32(a.m_val, 2);
  Packet4i lane3 = _mm512_extracti32x4_epi32(a.m_val, 3);
  Packet4i res =
      _mm_min_epi16(_mm_min_epi16(lane0, lane1), _mm_min_epi16(lane2, lane3));
  res = _mm_min_epi16(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 3, 2)));
  res = _mm_min_epi16(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 0, 1)));
  std::uint32_t w = pfirst(res);
  return std::min(
      {static_cast<std::int16_t>(w >> 16), static_cast<std::int16_t>(w)});
}
template <>
EIGEN_STRONG_INLINE QInt16 predux_max<Packet32q16i>(const Packet32q16i& a) {
  Packet4i lane0 = _mm512_extracti32x4_epi32(a.m_val, 0);
  Packet4i lane1 = _mm512_extracti32x4_epi32(a.m_val, 1);
  Packet4i lane2 = _mm512_extracti32x4_epi32(a.m_val, 2);
  Packet4i lane3 = _mm512_extracti32x4_epi32(a.m_val, 3);
  Packet4i res =
      _mm_max_epi16(_mm_max_epi16(lane0, lane1), _mm_max_epi16(lane2, lane3));
  res = _mm_max_epi16(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 3, 2)));
  res = _mm_max_epi16(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 0, 1)));
  std::uint32_t w = pfirst(res);
  return std::max(
      {static_cast<std::int16_t>(w >> 16), static_cast<std::int16_t>(w)});
}
template <>
EIGEN_STRONG_INLINE QUInt8 predux_min<Packet64q8u>(const Packet64q8u& a) {
  Packet4i lane0 = _mm512_extracti32x4_epi32(a.m_val, 0);
  Packet4i lane1 = _mm512_extracti32x4_epi32(a.m_val, 1);
  Packet4i lane2 = _mm512_extracti32x4_epi32(a.m_val, 2);
  Packet4i lane3 = _mm512_extracti32x4_epi32(a.m_val, 3);
  Packet4i res =
      _mm_min_epu8(_mm_min_epu8(lane0, lane1), _mm_min_epu8(lane2, lane3));
  res = _mm_min_epu8(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 3, 2)));
  res = _mm_min_epu8(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 0, 1)));
  std::uint32_t w = pfirst(res);
  return std::min(
      {static_cast<std::uint8_t>(w >> 24), static_cast<std::uint8_t>(w >> 16),
       static_cast<std::uint8_t>(w >> 8), static_cast<std::uint8_t>(w)});
}
template <>
EIGEN_STRONG_INLINE QUInt8 predux_max<Packet64q8u>(const Packet64q8u& a) {
  Packet4i lane0 = _mm512_extracti32x4_epi32(a.m_val, 0);
  Packet4i lane1 = _mm512_extracti32x4_epi32(a.m_val, 1);
  Packet4i lane2 = _mm512_extracti32x4_epi32(a.m_val, 2);
  Packet4i lane3 = _mm512_extracti32x4_epi32(a.m_val, 3);
  Packet4i res =
      _mm_max_epu8(_mm_max_epu8(lane0, lane1), _mm_max_epu8(lane2, lane3));
  res = _mm_max_epu8(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 3, 2)));
  res = _mm_max_epu8(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 0, 1)));
  std::uint32_t w = pfirst(res);
  return std::max(
      {static_cast<std::uint8_t>(w >> 24), static_cast<std::uint8_t>(w >> 16),
       static_cast<std::uint8_t>(w >> 8), static_cast<std::uint8_t>(w)});
}
template <>
EIGEN_STRONG_INLINE QInt8 predux_min<Packet64q8i>(const Packet64q8i& a) {
  Packet4i lane0 = _mm512_extracti32x4_epi32(a.m_val, 0);
  Packet4i lane1 = _mm512_extracti32x4_epi32(a.m_val, 1);
  Packet4i lane2 = _mm512_extracti32x4_epi32(a.m_val, 2);
  Packet4i lane3 = _mm512_extracti32x4_epi32(a.m_val, 3);
  Packet4i res =
      _mm_min_epi8(_mm_min_epi8(lane0, lane1), _mm_min_epi8(lane2, lane3));
  res = _mm_min_epi8(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 3, 2)));
  res = _mm_min_epi8(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 0, 1)));
  std::uint32_t w = pfirst(res);
  return std::min(
      {static_cast<std::int8_t>(w >> 24), static_cast<std::int8_t>(w >> 16),
       static_cast<std::int8_t>(w >> 8), static_cast<std::int8_t>(w)});
}
template <>
EIGEN_STRONG_INLINE QInt8 predux_max<Packet64q8i>(const Packet64q8i& a) {
  Packet4i lane0 = _mm512_extracti32x4_epi32(a.m_val, 0);
  Packet4i lane1 = _mm512_extracti32x4_epi32(a.m_val, 1);
  Packet4i lane2 = _mm512_extracti32x4_epi32(a.m_val, 2);
  Packet4i lane3 = _mm512_extracti32x4_epi32(a.m_val, 3);
  Packet4i res =
      _mm_max_epi8(_mm_max_epi8(lane0, lane1), _mm_max_epi8(lane2, lane3));
  res = _mm_max_epi8(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 3, 2)));
  res = _mm_max_epi8(res, _mm_shuffle_epi32(res, _MM_SHUFFLE(0, 0, 0, 1)));
  std::uint32_t w = pfirst(res);
  return std::min(
      {static_cast<std::int8_t>(w >> 24), static_cast<std::int8_t>(w >> 16),
       static_cast<std::int8_t>(w >> 8), static_cast<std::int8_t>(w)});
}

}  // end namespace internal
}  // end namespace Eigen

#endif  // CXX11_SRC_FIXEDPOINT_PACKETMATHAVX512_H_
