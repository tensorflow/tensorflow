// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TYPE_CASTING_AVX_H
#define EIGEN_TYPE_CASTING_AVX_H

namespace Eigen {

namespace internal {

// For now we use SSE to handle integers, so we can't use AVX instructions to cast
// from int to float
template <>
struct type_casting_traits<float, int> {
  enum {
    VectorizedCast = 0,
    SrcCoeffRatio = 1,
    TgtCoeffRatio = 1
  };
};

template <>
struct type_casting_traits<int, float> {
  enum {
    VectorizedCast = 0,
    SrcCoeffRatio = 1,
    TgtCoeffRatio = 1
  };
};



template<> EIGEN_STRONG_INLINE Packet8i pcast<Packet8f, Packet8i>(const Packet8f& a) {
  return _mm256_cvtps_epi32(a);
}

template<> EIGEN_STRONG_INLINE Packet8f pcast<Packet8i, Packet8f>(const Packet8i& a) {
  return _mm256_cvtepi32_ps(a);
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_TYPE_CASTING_AVX_H
