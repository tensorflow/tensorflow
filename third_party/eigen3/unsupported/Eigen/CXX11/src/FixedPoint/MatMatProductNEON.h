// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
// Copyright (C) 2015 Benoit Jacob <benoitjacob@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CXX11_SRC_FIXEDPOINT_MATMATPRODUCTNEON_H_
#define CXX11_SRC_FIXEDPOINT_MATMATPRODUCTNEON_H_

namespace Eigen {
namespace internal {


// AVX2 optimized implementation of the case where the lhs is encoded using signed 8bit
// integers and the rhs using unsigned 8bit integers.
#ifdef EIGEN_USE_OPTIMIZED_INT8_UINT8_MAT_MAT_PRODUCT

template<bool _ConjLhs, bool _ConjRhs>
class gebp_traits<QInt8, QUInt8, _ConjLhs, _ConjRhs>
{
public:
  typedef QInt8 LhsScalar;
  typedef QUInt8 RhsScalar;
  typedef QInt32 ResScalar;

  enum {
    // register block size along the M and N directions
    // One for the current implementation
    nr = 1,
    mr = 1,
    // Progress made at each iteration of the product loop
    // also 1 for the current implementation
    LhsProgress = 1,
    RhsProgress = 1
  };
};

// Mat-Mat product of a signed 8bit lhs with an unsigned 8bit rhs
template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<QInt8, QUInt8, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
{
  EIGEN_DONT_INLINE
  void operator()(const DataMapper& res, const QInt8* blockA, const QUInt8* blockB,
                  Index rows, Index depth, Index cols, QInt32 alpha,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0);
};

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
EIGEN_DONT_INLINE
void gebp_kernel<QInt8, QUInt8, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
::operator()(const DataMapper& res, const QInt8* blockA, const QUInt8* blockB,
             Index rows, Index depth, Index cols, QInt32 alpha,
             Index strideA, Index strideB, Index offsetA, Index offsetB)
{
  EIGEN_STATIC_ASSERT(!ConjugateLhs, YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT(!ConjugateRhs, YOU_MADE_A_PROGRAMMING_MISTAKE);

  eigen_assert(alpha.value == 1);
  eigen_assert(strideA == -1);
  eigen_assert(strideB == -1);
  eigen_assert(offsetA == 0);
  eigen_assert(offsetB == 0);

  eigen_assert(rows > 0);
  eigen_assert(cols > 0);
  eigen_assert(depth > 0);
  eigen_assert(blockA);
  eigen_assert(blockB);

  for (Index j = 0; j < cols; ++j) {
    Index startB = j * depth;

    for (Index i = 0; i < rows; ++i) {
      Index startA = i * depth;

      for (Index k = 0; k < depth; ++k) {
        res(i, j) += blockA[startA + k] * blockB[startB + k];
      }
    }
  }
}
#endif


}  // namespace internal
}  // namespace Eigen

#endif  // CXX11_SRC_FIXEDPOINT_MATMATPRODUCTNEON_H_
