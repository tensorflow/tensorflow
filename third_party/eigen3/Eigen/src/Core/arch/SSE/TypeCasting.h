// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TYPE_CASTING_SSE_H
#define EIGEN_TYPE_CASTING_SSE_H

namespace Eigen {

namespace internal {

template <>
struct type_casting_traits<float, int> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 1,
    TgtCoeffRatio = 1
  };
};

template<> EIGEN_STRONG_INLINE Packet4i pcast<Packet4f, Packet4i>(const Packet4f& a) {
  return _mm_cvttps_epi32(a);
}


template <>
struct type_casting_traits<int, float> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 1,
    TgtCoeffRatio = 1
  };
};

template<> EIGEN_STRONG_INLINE Packet4f pcast<Packet4i, Packet4f>(const Packet4i& a) {
  return _mm_cvtepi32_ps(a);
}


template <>
struct type_casting_traits<double, float> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 2,
    TgtCoeffRatio = 1
  };
};

template<> EIGEN_STRONG_INLINE Packet4f pcast<Packet2d, Packet4f>(const Packet2d& a, const Packet2d& b) {
  return _mm_shuffle_ps(_mm_cvtpd_ps(a), _mm_cvtpd_ps(b), (1 << 2) | (1 << 6));
}

template <>
struct type_casting_traits<float, double> {
  enum {
    VectorizedCast = 1,
    SrcCoeffRatio = 1,
    TgtCoeffRatio = 2
  };
};

template<> EIGEN_STRONG_INLINE Packet2d pcast<Packet4f, Packet2d>(const Packet4f& a) {
  // Simply discard the second half of the input
  return _mm_cvtps_pd(a);
}


} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_TYPE_CASTING_SSE_H
