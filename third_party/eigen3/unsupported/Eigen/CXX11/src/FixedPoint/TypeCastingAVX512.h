#ifndef EIGEN3_UNSUPPORTED_EIGEN_CXX11_SRC_FIXEDPOINT_TYPECASTINGAVX512_H_
#define EIGEN3_UNSUPPORTED_EIGEN_CXX11_SRC_FIXEDPOINT_TYPECASTINGAVX512_H_

namespace Eigen {
namespace internal {

typedef __m512 Packet16f;
typedef __m512i Packet16i;

template <>
struct type_casting_traits<QInt32, float> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet16f pcast<Packet16q32i>(const Packet16q32i& a) {
  return _mm512_cvtepi32_ps(a.val);
}

template <>
struct type_casting_traits<float, QInt32> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet16q32i pcast<Packet16f>(const Packet16f& a) {
  return _mm512_cvtps_epi32(a);
}

template <>
struct type_casting_traits<float, QInt16> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 2, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet32q16i
pcast<Packet16f>(const Packet16f& a, const Packet16f& b) {
  Packet16i a_int = _mm512_cvtps_epi32(a);
  Packet16i b_int = _mm512_cvtps_epi32(b);
#ifdef EIGEN_VECTORIZE_AVX512BW
  return _mm512_packs_epi32(a_int, b_int);
#else
  Packet8i ab_int16_low =
      _mm256_permute4x64_epi64(
        _mm256_packs_epi32(
          _mm512_castsi512_si256(a_int),
          _mm512_castsi512_si256(b_int)),
        _MM_SHUFFLE(0, 2, 1, 3));
  Packet8i ab_int16_high =
      _mm256_permute4x64_epi64(
        _mm256_packs_epi32(
          _mm512_extracti32x8_epi32(a_int, 1),
          _mm512_extracti32x8_epi32(b_int, 1)),
        _MM_SHUFFLE(0, 2, 1, 3));
  return _mm512_inserti32x8(
           _mm512_castsi256_si512(ab_int16_low),
           ab_int16_high, 1);
#endif
}

template <>
struct type_casting_traits<float, QInt8> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 4, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet64q8i
pcast<Packet16f>(const Packet16f& a,
                 const Packet16f& b,
                 const Packet16f& c,
                 const Packet16f& d) {
  Packet16i a_int = _mm512_cvtps_epi32(a);
  Packet16i b_int = _mm512_cvtps_epi32(b);
  Packet16i c_int = _mm512_cvtps_epi32(c);
  Packet16i d_int = _mm512_cvtps_epi32(d);
#ifdef EIGEN_VECTORIZE_AVX512BW
  return _mm512_packs_epi16(
           _mm512_packs_epi32(a_int, b_int),
           _mm512_packs_epi32(c_int, d_int));
#else
  Packet8i ab_int16_low =
      _mm256_permute4x64_epi64(
        _mm256_packs_epi32(
          _mm512_castsi512_si256(a_int),
          _mm512_castsi512_si256(b_int)),
        _MM_SHUFFLE(0, 2, 1, 3));
  Packet8i cd_int16_low =
      _mm256_permute4x64_epi64(
        _mm256_packs_epi32(
          _mm512_castsi512_si256(c_int),
          _mm512_castsi512_si256(d_int)),
        _MM_SHUFFLE(0, 2, 1, 3));
  Packet8i ab_int16_high =
      _mm256_permute4x64_epi64(
        _mm256_packs_epi32(
          _mm512_extracti32x8_epi32(a_int, 1),
          _mm512_extracti32x8_epi32(b_int, 1)),
        _MM_SHUFFLE(0, 2, 1, 3));
  Packet8i cd_int16_high =
      _mm256_permute4x64_epi64(
        _mm256_packs_epi32(
          _mm512_extracti32x8_epi32(c_int, 1),
          _mm512_extracti32x8_epi32(d_int, 1)),
        _MM_SHUFFLE(0, 2, 1, 3));
  Packet8i abcd_int8_low =
      _mm256_permute4x64_epi64(
        _mm256_packs_epi16(ab_int16_low, cd_int16_low),
        _MM_SHUFFLE(0, 2, 1, 3));
  Packet8i abcd_int8_high =
      _mm256_permute4x64_epi64(
        _mm256_packs_epi16(ab_int16_high, cd_int16_high),
        _MM_SHUFFLE(0, 2, 1, 3));
  return _mm512_inserti32x8(
           _mm512_castsi256_si512(abcd_int8_low),
           abcd_int8_high, 1);
#endif
}

template <>
struct type_casting_traits<QInt32, QInt8> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 4, TgtCoeffRatio = 1 };
};

template <>
struct type_casting_traits<QInt32, QInt16> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 2, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet64q8i
pcast<Packet16q32i, Packet64q8i>(const Packet16q32i& a,
                                 const Packet16q32i& b,
                                 const Packet16q32i& c,
                                 const Packet16q32i& d) {
  __m512i converted = _mm512_packs_epi16(_mm512_packs_epi32(a.val, b.val),
                                         _mm512_packs_epi32(c.val, d.val));
  return converted;
}

template <>
EIGEN_STRONG_INLINE Packet32q16i
pcast<Packet16q32i, Packet32q16i>(const Packet16q32i& a,
                                  const Packet16q32i& b) {
  __m512i converted = _mm512_packs_epi32(a.val, b.val);
  return converted;
}

template <>
struct type_casting_traits<QInt32, QUInt8> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 4, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet64q8u
pcast<Packet16q32i, Packet64q8u>(const Packet16q32i& a, const Packet16q32i& b,
                                 const Packet16q32i& c, const Packet16q32i& d) {
  const __m512i converted = _mm512_packus_epi16(
      _mm512_packus_epi32(a.val, b.val), _mm512_packus_epi32(c.val, d.val));
  return converted;
}

template <>
struct type_casting_traits<QInt32, QUInt16> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 2, TgtCoeffRatio = 1 };
};

#if 0
template <>
EIGEN_STRONG_INLINE Packet32q16u
pcast<Packet16q32i, Packet32q16u>(const Packet16q32i& a,
                                  const Packet16q32i& b) {
  const __m512i converted = _mm512_packus_epi32(a.val, b.val);
  return converted;
}
#endif

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN3_UNSUPPORTED_EIGEN_CXX11_SRC_FIXEDPOINT_TYPECASTINGAVX512_H_
