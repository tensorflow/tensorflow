#ifndef EIGEN3_UNSUPPORTED_EIGEN_CXX11_SRC_FIXEDPOINT_TYPECASTINGAVX2_H_
#define EIGEN3_UNSUPPORTED_EIGEN_CXX11_SRC_FIXEDPOINT_TYPECASTINGAVX2_H_

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
struct type_casting_traits<QInt32, QUInt8> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 4, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet32q8u
pcast<Packet8q32i, Packet32q8u>(const Packet8q32i& a, const Packet8q32i& b,
                                const Packet8q32i& c, const Packet8q32i& d) {
  const __m256i converted = _mm256_packus_epi16(
      _mm256_packs_epi32(a.val, b.val), _mm256_packs_epi32(c.val, d.val));
  // Since packus does not cross 128 bit lane boundaries,
  // we have to permute to properly order the final result.
  const __m256i permute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
  return _mm256_permutevar8x32_epi32(converted, permute_mask);
}

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN3_UNSUPPORTED_EIGEN_CXX11_SRC_FIXEDPOINT_TYPECASTINGAVX2_H_
