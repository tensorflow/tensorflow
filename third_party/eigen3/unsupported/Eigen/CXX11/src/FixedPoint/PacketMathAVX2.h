#ifndef CXX11_SRC_FIXEDPOINT_PACKETMATHAVX2_H_
#define CXX11_SRC_FIXEDPOINT_PACKETMATHAVX2_H_
#ifdef _MSC_VER

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

#endif

inline int _mm256_extract_epi16_N0(const __m256i X) {
  return _mm_extract_epi16(_mm256_extractf128_si256(X, 0 >> 3), 0 % 8);
}

inline int _mm256_extract_epi16_N1(const __m256i X) {
  return _mm_extract_epi16(_mm256_extractf128_si256(X, 1 >> 3), 1 % 8);
}

inline int _mm256_extract_epi8_N0(const __m256i X) {
  return _mm_extract_epi8(_mm256_extractf128_si256((X), 0 >> 4), 0 % 16);
}

inline int _mm256_extract_epi8_N1(const __m256i X) {
  return _mm_extract_epi8(_mm256_extractf128_si256((X), 1 >> 4), 1 % 16);
}

namespace Eigen {
namespace internal {

typedef struct Packet32q8i {
  __m256i val;
  operator __m256i() const { return val; }
  Packet32q8i() : val(_mm256_setzero_si256()){};
  Packet32q8i(__m256i val) : val(val) {}
} Packet32q8i;

typedef struct Packet16q16i {
  __m256i val;
  operator __m256i() const { return val; }
  Packet16q16i() : val(_mm256_setzero_si256()){};
  Packet16q16i(__m256i val) : val(val) {}
} Packet16q16i;

typedef struct Packet32q8u {
  __m256i val;
  operator __m256i() const { return val; }
  Packet32q8u() : val(_mm256_setzero_si256()){};
  Packet32q8u(__m256i val) : val(val) {}
} Packet32q8u;

typedef struct Packet16q8i {
  __m128i val;
  operator __m128i() const { return val; }
  Packet16q8i() : val(_mm_setzero_si128()) {}
  Packet16q8i(__m128i val) : val(val) {}
} Packet16q8i;

typedef struct Packet16q8u {
  __m128i val;
  operator __m128i() const { return val; }
  Packet16q8u() : val(_mm_setzero_si128()) {}
  Packet16q8u(__m128i val) : val(val) {}
} Packet16q8u;

typedef struct Packet8q16i {
  __m128i val;
  operator __m128i() const { return val; }
  Packet8q16i() : val(_mm_setzero_si128()) {}
  Packet8q16i(__m128i val) : val(val) {}
} Packet8q16i;

typedef struct Packet8q32i {
  __m256i val;
  operator __m256i() const { return val; }
  Packet8q32i() : val(_mm256_setzero_si256()){};
  Packet8q32i(__m256i val) : val(val) {}
} Packet8q32i;

typedef struct Packet4q32i {
  __m128i val;
  operator __m128i() const { return val; }
  Packet4q32i() : val(_mm_setzero_si128()) {}
  Packet4q32i(__m128i val) : val(val) {}
} Packet4q32i;

#ifndef EIGEN_VECTORIZE_AVX512
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
    HasMin = 1,
    HasMax = 1,
    HasConj = 0,
    HasSetLinear = 0
  };
};
template <>
struct packet_traits<QUInt8> : default_packet_traits {
  typedef Packet32q8u type;
  typedef Packet16q8u half;
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
struct packet_traits<QInt16> : default_packet_traits {
  typedef Packet16q16i type;
  typedef Packet8q16i half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 16,
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
  typedef Packet8q32i type;
  typedef Packet4q32i half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 8,
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
#endif

template <>
struct unpacket_traits<Packet32q8i> {
  typedef QInt8 type;
  typedef Packet16q8i half;
  enum {
    size = 32,
    alignment = Aligned32,
    vectorizable = true,
    masked_load_available = false
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
    masked_load_available = false
  };
};
template <>
struct unpacket_traits<Packet16q16i> {
  typedef QInt16 type;
  typedef Packet8q16i half;
  enum {
    size = 16,
    alignment = Aligned32,
    vectorizable = true,
    masked_load_available = false
  };
};
template <>
struct unpacket_traits<Packet8q16i> {
  typedef QInt16 type;
  typedef Packet8q16i half;
  enum {
    size = 8,
    alignment = Aligned32,
    vectorizable = true,
    masked_load_available = false
  };
};
template <>
struct unpacket_traits<Packet32q8u> {
  typedef QUInt8 type;
  typedef Packet16q8u half;
  enum {
    size = 32,
    alignment = Aligned32,
    vectorizable = true,
    masked_load_available = false
  };
};
template <>
struct unpacket_traits<Packet8q32i> {
  typedef QInt32 type;
  typedef Packet4q32i half;
  enum {
    size = 8,
    alignment = Aligned32,
    vectorizable = true,
    masked_load_available = false
  };
};

// Unaligned load
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
EIGEN_STRONG_INLINE Packet32q8u ploadu<Packet32q8u>(const QUInt8* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet16q16i ploadu<Packet16q16i>(const QInt16* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet8q16i ploadu<Packet8q16i>(const QInt16* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm_loadu_si128(
      reinterpret_cast<const __m128i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet8q32i ploadu<Packet8q32i>(const QInt32* from) {
  EIGEN_DEBUG_UNALIGNED_LOAD return _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(from));
}

// Aligned load
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
EIGEN_STRONG_INLINE Packet32q8u pload<Packet32q8u>(const QUInt8* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm256_load_si256(
      reinterpret_cast<const __m256i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet16q16i pload<Packet16q16i>(const QInt16* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm256_load_si256(
      reinterpret_cast<const __m256i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet8q16i pload<Packet8q16i>(const QInt16* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm_load_si128(
      reinterpret_cast<const __m128i*>(from));
}
template <>
EIGEN_STRONG_INLINE Packet8q32i pload<Packet8q32i>(const QInt32* from) {
  EIGEN_DEBUG_ALIGNED_LOAD return _mm256_load_si256(
      reinterpret_cast<const __m256i*>(from));
}

// Unaligned store
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt8>(QInt8* to, const Packet32q8i& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(to), from.val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt8>(QInt8* to, const Packet16q8i& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm_storeu_si128(reinterpret_cast<__m128i*>(to),
                                               from.val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QUInt8>(QUInt8* to, const Packet32q8u& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(to), from.val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt16>(QInt16* to, const Packet16q16i& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(to), from.val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt16>(QInt16* to, const Packet8q16i& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm_storeu_si128(reinterpret_cast<__m128i*>(to),
                                               from.val);
}
template <>
EIGEN_STRONG_INLINE void pstoreu<QInt32>(QInt32* to, const Packet8q32i& from) {
  EIGEN_DEBUG_UNALIGNED_STORE _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(to), from.val);
}

// Aligned store
template <>
EIGEN_STRONG_INLINE void pstore<QInt32>(QInt32* to, const Packet8q32i& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm256_store_si256(reinterpret_cast<__m256i*>(to),
                                               from.val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QInt16>(QInt16* to, const Packet16q16i& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm256_store_si256(reinterpret_cast<__m256i*>(to),
                                               from.val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QInt16>(QInt16* to, const Packet8q16i& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm_store_si128(reinterpret_cast<__m128i*>(to),
                                            from.val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QUInt8>(QUInt8* to, const Packet32q8u& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm256_store_si256(reinterpret_cast<__m256i*>(to),
                                               from.val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QInt8>(QInt8* to, const Packet32q8i& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm256_store_si256(reinterpret_cast<__m256i*>(to),
                                               from.val);
}
template <>
EIGEN_STRONG_INLINE void pstore<QInt8>(QInt8* to, const Packet16q8i& from) {
  EIGEN_DEBUG_ALIGNED_STORE _mm_store_si128(reinterpret_cast<__m128i*>(to),
                                            from.val);
}

// Extract first element.
template <>
EIGEN_STRONG_INLINE QInt32 pfirst<Packet8q32i>(const Packet8q32i& a) {
  return _mm_cvtsi128_si32(_mm256_castsi256_si128(a));
}
template <>
EIGEN_STRONG_INLINE QInt16 pfirst<Packet16q16i>(const Packet16q16i& a) {
  return _mm256_extract_epi16_N0(a.val);
}
template <>
EIGEN_STRONG_INLINE QUInt8 pfirst<Packet32q8u>(const Packet32q8u& a) {
  return static_cast<uint8_t>(_mm256_extract_epi8_N0(a.val));
}
template <>
EIGEN_STRONG_INLINE QInt8 pfirst<Packet32q8i>(const Packet32q8i& a) {
  return _mm256_extract_epi8_N0(a.val);
}

// Initialize to constant value.
template <>
EIGEN_STRONG_INLINE Packet32q8i pset1<Packet32q8i>(const QInt8& from) {
  return _mm256_set1_epi8(from.value);
}
template <>
EIGEN_STRONG_INLINE Packet32q8u pset1<Packet32q8u>(const QUInt8& from) {
  return _mm256_set1_epi8(static_cast<uint8_t>(from.value));
}
template <>
EIGEN_STRONG_INLINE Packet8q32i pset1<Packet8q32i>(const QInt32& from) {
  return _mm256_set1_epi32(from.value);
}

// Basic arithmetic packet ops for QInt32.
template <>
EIGEN_STRONG_INLINE Packet8q32i padd<Packet8q32i>(const Packet8q32i& a,
                                                  const Packet8q32i& b) {
  return _mm256_add_epi32(a.val, b.val);
}
template <>
EIGEN_STRONG_INLINE Packet16q16i pset1<Packet16q16i>(const QInt16& from) {
  return _mm256_set1_epi16(from.value);
}
template <>
EIGEN_STRONG_INLINE Packet8q32i psub<Packet8q32i>(const Packet8q32i& a,
                                                  const Packet8q32i& b) {
  return _mm256_sub_epi32(a.val, b.val);
}
// Note: mullo truncates the result to 32 bits.
template <>
EIGEN_STRONG_INLINE Packet8q32i pmul<Packet8q32i>(const Packet8q32i& a,
                                                  const Packet8q32i& b) {
  return _mm256_mullo_epi32(a.val, b.val);
}
template <>
EIGEN_STRONG_INLINE Packet8q32i pnegate<Packet8q32i>(const Packet8q32i& a) {
  return _mm256_sub_epi32(_mm256_setzero_si256(), a.val);
}

// Min and max.
template <>
EIGEN_STRONG_INLINE Packet8q32i pmin<Packet8q32i>(const Packet8q32i& a,
                                                  const Packet8q32i& b) {
  return _mm256_min_epi32(a.val, b.val);
}
template <>
EIGEN_STRONG_INLINE Packet8q32i pmax<Packet8q32i>(const Packet8q32i& a,
                                                  const Packet8q32i& b) {
  return _mm256_max_epi32(a.val, b.val);
}

template <>
EIGEN_STRONG_INLINE Packet16q16i pmin<Packet16q16i>(const Packet16q16i& a,
                                                    const Packet16q16i& b) {
  return _mm256_min_epi16(a.val, b.val);
}
template <>
EIGEN_STRONG_INLINE Packet16q16i pmax<Packet16q16i>(const Packet16q16i& a,
                                                    const Packet16q16i& b) {
  return _mm256_max_epi16(a.val, b.val);
}

template <>
EIGEN_STRONG_INLINE Packet32q8u pmin<Packet32q8u>(const Packet32q8u& a,
                                                  const Packet32q8u& b) {
  return _mm256_min_epu8(a.val, b.val);
}
template <>
EIGEN_STRONG_INLINE Packet32q8u pmax<Packet32q8u>(const Packet32q8u& a,
                                                  const Packet32q8u& b) {
  return _mm256_max_epu8(a.val, b.val);
}

template <>
EIGEN_STRONG_INLINE Packet32q8i pmin<Packet32q8i>(const Packet32q8i& a,
                                                  const Packet32q8i& b) {
  return _mm256_min_epi8(a.val, b.val);
}
template <>
EIGEN_STRONG_INLINE Packet32q8i pmax<Packet32q8i>(const Packet32q8i& a,
                                                  const Packet32q8i& b) {
  return _mm256_max_epi8(a.val, b.val);
}

// Reductions.
template <>
EIGEN_STRONG_INLINE QInt32 predux_min<Packet8q32i>(const Packet8q32i& a) {
  __m256i tmp = _mm256_min_epi32(a, _mm256_permute2f128_si256(a, a, 1));
  tmp =
      _mm256_min_epi32(tmp, _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  return pfirst<Packet8q32i>(
      _mm256_min_epi32(tmp, _mm256_shuffle_epi32(tmp, 1)));
}
template <>
EIGEN_STRONG_INLINE QInt32 predux_max<Packet8q32i>(const Packet8q32i& a) {
  __m256i tmp = _mm256_max_epi32(a, _mm256_permute2f128_si256(a, a, 1));
  tmp =
      _mm256_max_epi32(tmp, _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  return pfirst<Packet8q32i>(
      _mm256_max_epi32(tmp, _mm256_shuffle_epi32(tmp, 1)));
}

template <>
EIGEN_STRONG_INLINE QInt16 predux_min<Packet16q16i>(const Packet16q16i& a) {
  __m256i tmp = _mm256_min_epi16(a, _mm256_permute2f128_si256(a, a, 1));
  tmp =
      _mm256_min_epi16(tmp, _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  tmp = _mm256_min_epi16(tmp, _mm256_shuffle_epi32(tmp, 1));
  return std::min(_mm256_extract_epi16_N0(tmp), _mm256_extract_epi16_N1(tmp));
}
template <>
EIGEN_STRONG_INLINE QInt16 predux_max<Packet16q16i>(const Packet16q16i& a) {
  __m256i tmp = _mm256_max_epi16(a, _mm256_permute2f128_si256(a, a, 1));
  tmp =
      _mm256_max_epi16(tmp, _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  tmp = _mm256_max_epi16(tmp, _mm256_shuffle_epi32(tmp, 1));
  return std::max(_mm256_extract_epi16_N0(tmp), _mm256_extract_epi16_N1(tmp));
}

template <>
EIGEN_STRONG_INLINE QUInt8 predux_min<Packet32q8u>(const Packet32q8u& a) {
  __m256i tmp = _mm256_min_epu8(a, _mm256_permute2f128_si256(a, a, 1));
  tmp =
      _mm256_min_epu8(tmp, _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  tmp = _mm256_min_epu8(tmp, _mm256_shuffle_epi32(tmp, 1));
  tmp = _mm256_min_epu8(tmp,
                        _mm256_shufflelo_epi16(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  return std::min(static_cast<uint8_t>(_mm256_extract_epi8_N0(tmp)),
                  static_cast<uint8_t>(_mm256_extract_epi8_N1(tmp)));
}
template <>
EIGEN_STRONG_INLINE QUInt8 predux_max<Packet32q8u>(const Packet32q8u& a) {
  __m256i tmp = _mm256_max_epu8(a, _mm256_permute2f128_si256(a, a, 1));
  tmp =
      _mm256_max_epu8(tmp, _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  tmp = _mm256_max_epu8(tmp, _mm256_shuffle_epi32(tmp, 1));
  tmp = _mm256_max_epu8(tmp,
                        _mm256_shufflelo_epi16(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  return std::max(static_cast<uint8_t>(_mm256_extract_epi8_N0(tmp)),
                  static_cast<uint8_t>(_mm256_extract_epi8_N1(tmp)));
}

template <>
EIGEN_STRONG_INLINE QInt8 predux_min<Packet32q8i>(const Packet32q8i& a) {
  __m256i tmp = _mm256_min_epi8(a, _mm256_permute2f128_si256(a, a, 1));
  tmp =
      _mm256_min_epi8(tmp, _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  tmp = _mm256_min_epi8(tmp, _mm256_shuffle_epi32(tmp, 1));
  tmp = _mm256_min_epi8(tmp,
                        _mm256_shufflelo_epi16(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  return std::min(_mm256_extract_epi8_N0(tmp), _mm256_extract_epi8_N1(tmp));
}
template <>
EIGEN_STRONG_INLINE QInt8 predux_max<Packet32q8i>(const Packet32q8i& a) {
  __m256i tmp = _mm256_max_epi8(a, _mm256_permute2f128_si256(a, a, 1));
  tmp =
      _mm256_max_epi8(tmp, _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  tmp = _mm256_max_epi8(tmp, _mm256_shuffle_epi32(tmp, 1));
  tmp = _mm256_max_epi8(tmp,
                        _mm256_shufflelo_epi16(tmp, _MM_SHUFFLE(1, 0, 3, 2)));
  return std::max(_mm256_extract_epi8_N0(tmp), _mm256_extract_epi8_N1(tmp));
}

// Vectorized scaling of Packet32q8i by float.
template <>
struct scalar_product_op<QInt32, double> : binary_op_base<QInt32, double> {
  typedef typename ScalarBinaryOpTraits<QInt32, double>::ReturnType result_type;
#ifndef EIGEN_SCALAR_BINARY_OP_PLUGIN
  EIGEN_EMPTY_STRUCT_CTOR(scalar_product_op)
#else
  scalar_product_op() { EIGEN_SCALAR_BINARY_OP_PLUGIN }
#endif
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type
  operator()(const QInt32& a, const double& b) const {
    return a * b;
  }

  EIGEN_STRONG_INLINE const Packet8q32i packetOp(const Packet8q32i& a,
                                                 const double& b) const {
    __m256d scale = _mm256_set1_pd(b);
    __m256d a_lo = _mm256_cvtepi32_pd(_mm256_castsi256_si128(a));
    __m128i result_lo = _mm256_cvtpd_epi32(_mm256_mul_pd(scale, a_lo));
    __m256d a_hi = _mm256_cvtepi32_pd(_mm256_extracti128_si256(a, 1));
    __m128i result_hi = _mm256_cvtpd_epi32(_mm256_mul_pd(scale, a_hi));
    return _mm256_insertf128_si256(_mm256_castsi128_si256(result_lo), result_hi,
                                   1);
  }
};

template <>
struct functor_traits<scalar_product_op<QInt32, double>> {
  enum { Cost = 4 * NumTraits<float>::MulCost, PacketAccess = true };
};

}  // end namespace internal
}  // end namespace Eigen

#endif  // CXX11_SRC_FIXEDPOINT_PACKETMATHAVX2_H_
