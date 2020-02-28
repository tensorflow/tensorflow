#ifndef CXX11_SRC_FIXEDPOINT_TYPECASTINGAVX2_H_
#define CXX11_SRC_FIXEDPOINT_TYPECASTINGAVX2_H_

namespace Eigen {
namespace internal {

typedef __m256 Packet8f;

template <>
struct type_casting_traits<QInt32, float> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet8f pcast<Packet8q32i>(const Packet8q32i& a) {
  return _mm256_cvtepi32_ps(a.val);
}

template <>
struct type_casting_traits<float, QInt32> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet8q32i pcast<Packet8f>(const Packet8f& a) {
  return _mm256_cvtps_epi32(a);
}

template <>
struct type_casting_traits<QInt32, QInt8> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 4, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet32q8i
pcast<Packet8q32i, Packet32q8i>(const Packet8q32i& a, const Packet8q32i& b,
                                const Packet8q32i& c, const Packet8q32i& d) {
  __m256i converted = _mm256_packs_epi16(_mm256_packs_epi32(a.val, b.val),
                                         _mm256_packs_epi32(c.val, d.val));
  // Since packs does not cross 128 bit lane boundaries,
  // we have to permute to properly order the final result.
  const __m256i permute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
  return _mm256_permutevar8x32_epi32(converted, permute_mask);
}

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
  __m256i converted = _mm256_packs_epi16(_mm256_packs_epi32(a_conv, b_conv),
                                         _mm256_packs_epi32(c_conv, d_conv));
  const __m256i permute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
  return _mm256_permutevar8x32_epi32(converted, permute_mask);
}

template <>
struct type_casting_traits<QInt32, QUInt8> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 4, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet32q8u
pcast<Packet8q32i, Packet32q8u>(const Packet8q32i& a, const Packet8q32i& b,
                                const Packet8q32i& c, const Packet8q32i& d) {
  // _mm256_packus_epi32 trims negative numbers to 0 but we can't allow numbers
  // that are too large because _mm256_packus_epi16 expects signed input
  // (example of problem input: 0x11111111, which saturates to 0xffff = -1,
  // which saturates to 0).
  const __m256i a_clip = _mm256_min_epi32(a, _mm256_set1_epi32(255));
  const __m256i b_clip = _mm256_min_epi32(b, _mm256_set1_epi32(255));
  const __m256i c_clip = _mm256_min_epi32(c, _mm256_set1_epi32(255));
  const __m256i d_clip = _mm256_min_epi32(d, _mm256_set1_epi32(255));
  const __m256i converted = _mm256_packus_epi16(
      _mm256_packus_epi32(a_clip, b_clip), _mm256_packus_epi32(c_clip, d_clip));
  // Since packus does not cross 128 bit lane boundaries,
  // we have to permute to properly order the final result.
  const __m256i permute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
  return _mm256_permutevar8x32_epi32(converted, permute_mask);
}

}  // end namespace internal
}  // end namespace Eigen

#endif  // CXX11_SRC_FIXEDPOINT_TYPECASTINGAVX2_H_
