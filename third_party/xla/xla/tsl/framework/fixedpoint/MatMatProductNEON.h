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

#ifndef XLA_TSL_FRAMEWORK_FIXEDPOINT_MATMATPRODUCTNEON_H_
#define XLA_TSL_FRAMEWORK_FIXEDPOINT_MATMATPRODUCTNEON_H_

namespace Eigen {
namespace internal {

// Neon optimized implementation where both lhs and rhs are encoded using
// signed 8bit integers
#ifdef EIGEN_USE_OPTIMIZED_INT8_INT8_MAT_MAT_PRODUCT

template <bool _ConjLhs, bool _ConjRhs>
class gebp_traits<QInt8, QInt8, _ConjLhs, _ConjRhs> {
 public:
  typedef QInt8 LhsScalar;
  typedef QInt8 RhsScalar;
  typedef QInt32 ResScalar;

  enum {
    // register block size along the M and N directions
    // One for the current implementation
    nr = 4,
    mr = 1,
    // Progress made at each iteration of the product loop
    // also 1 for the current implementation
    LhsProgress = 1,
    RhsProgress = 1
  };
};

// The signed 8bit Mat-Mat product itself.
template <typename Index, typename DataMapper, int mr, int nr,
          bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<QInt8, QInt8, Index, DataMapper, mr, nr, ConjugateLhs,
                   ConjugateRhs> {
  EIGEN_DONT_INLINE
  void operator()(const DataMapper& res, const QInt8* blockA,
                  const QInt8* blockB, Index rows, Index depth, Index cols,
                  QInt32 alpha, Index strideA = -1, Index strideB = -1,
                  Index offsetA = 0, Index offsetB = 0);
};

template <typename Index, typename DataMapper, int mr, int nr,
          bool ConjugateLhs, bool ConjugateRhs>
EIGEN_DONT_INLINE void
gebp_kernel<QInt8, QInt8, Index, DataMapper, mr, nr, ConjugateLhs,
            ConjugateRhs>::operator()(const DataMapper& res,
                                      const QInt8* blockA, const QInt8* blockB,
                                      Index rows, Index depth, Index cols,
                                      QInt32 alpha, Index strideA,
                                      Index strideB, Index offsetA,
                                      Index offsetB) {
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

// Neon optimized implementation of the case where the lhs is encoded using
// signed 8bit integers and the rhs using unsigned 8bit integers.
#ifdef EIGEN_USE_OPTIMIZED_INT8_UINT8_MAT_MAT_PRODUCT

template <bool _ConjLhs, bool _ConjRhs>
class gebp_traits<QInt8, QUInt8, _ConjLhs, _ConjRhs> {
 public:
  typedef QInt8 LhsScalar;
  typedef QUInt8 RhsScalar;
  typedef QInt32 ResScalar;

  enum {
    // register block size along the M and N directions
    nr = 4,
    mr = 1,
    // Progress made at each iteration of the product loop
    // 1 for the current implementation
    LhsProgress = 1,
    RhsProgress = 1
  };
};

// Mat-Mat product of a signed 8bit lhs with an unsigned 8bit rhs
template <typename Index, typename DataMapper, int mr, int nr,
          bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<QInt8, QUInt8, Index, DataMapper, mr, nr, ConjugateLhs,
                   ConjugateRhs> {
  EIGEN_DONT_INLINE
  void operator()(const DataMapper& res, const QInt8* blockA,
                  const QUInt8* blockB, Index rows, Index depth, Index cols,
                  QInt32 alpha, Index strideA = -1, Index strideB = -1,
                  Index offsetA = 0, Index offsetB = 0);
};

template <typename Index, typename DataMapper, int mr, int nr,
          bool ConjugateLhs, bool ConjugateRhs>
EIGEN_DONT_INLINE void
gebp_kernel<QInt8, QUInt8, Index, DataMapper, mr, nr, ConjugateLhs,
            ConjugateRhs>::operator()(const DataMapper& res,
                                      const QInt8* blockA, const QUInt8* blockB,
                                      Index rows, Index depth, Index cols,
                                      QInt32 alpha, Index strideA,
                                      Index strideB, Index offsetA,
                                      Index offsetB) {
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

// Neon optimized implementation where the lhs is encoded using unsigned 8bit
// integers and the rhs using signed 8bit integers.
#ifdef EIGEN_USE_OPTIMIZED_UINT8_INT8_MAT_MAT_PRODUCT
template <bool _ConjLhs, bool _ConjRhs>
class gebp_traits<QUInt8, QInt8, _ConjLhs, _ConjRhs> {
 public:
  typedef QUInt8 LhsScalar;
  typedef QInt8 RhsScalar;
  typedef QInt32 ResScalar;

  enum {
    // register block size along the M and N directions
    nr = 4,
    mr = 1,
    // Progress made at each iteration of the product loop
    // 1 for the current implementation
    LhsProgress = 1,
    RhsProgress = 1
  };
};

// Mat-Mat product of an unsigned 8bit lhs with a signed 8bit rhs
template <typename Index, typename DataMapper, int mr, int nr,
          bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<QUInt8, QInt8, Index, DataMapper, mr, nr, ConjugateLhs,
                   ConjugateRhs> {
  EIGEN_DONT_INLINE
  void operator()(const DataMapper& res, const QUInt8* blockA,
                  const QInt8* blockB, Index rows, Index depth, Index cols,
                  QInt32 alpha, Index strideA = -1, Index strideB = -1,
                  Index offsetA = 0, Index offsetB = 0);
};

template <typename Index, typename DataMapper, int mr, int nr,
          bool ConjugateLhs, bool ConjugateRhs>
EIGEN_DONT_INLINE void
gebp_kernel<QUInt8, QInt8, Index, DataMapper, mr, nr, ConjugateLhs,
            ConjugateRhs>::operator()(const DataMapper& res,
                                      const QUInt8* blockA, const QInt8* blockB,
                                      Index rows, Index depth, Index cols,
                                      QInt32 alpha, Index strideA,
                                      Index strideB, Index offsetA,
                                      Index offsetB) {
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

#ifdef EIGEN_USE_OPTIMIZED_INT16_INT16_MAT_MAT_PRODUCT

template <bool _ConjLhs, bool _ConjRhs>
class gebp_traits<QInt16, QInt16, _ConjLhs, _ConjRhs> {
 public:
  typedef QInt16 LhsScalar;
  typedef QInt16 RhsScalar;
  typedef QInt32 ResScalar;

  enum {
    // register block size along the M and N directions
    // One for the current implementation
    nr = 4,
    mr = 1,
    // Progress made at each iteration of the product loop
    // also 1 for the current implementation
    LhsProgress = 1,
    RhsProgress = 1
  };
};

// The signed 16bit Mat-Mat product itself.
template <typename Index, typename DataMapper, int mr, int nr,
          bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<QInt16, QInt16, Index, DataMapper, mr, nr, ConjugateLhs,
                   ConjugateRhs> {
  EIGEN_DONT_INLINE
  void operator()(const DataMapper& res, const QInt16* blockA,
                  const QInt16* blockB, Index rows, Index depth, Index cols,
                  QInt32 alpha, Index strideA = -1, Index strideB = -1,
                  Index offsetA = 0, Index offsetB = 0);
};

template <typename Index, typename DataMapper, int mr, int nr,
          bool ConjugateLhs, bool ConjugateRhs>
EIGEN_DONT_INLINE void
gebp_kernel<QInt16, QInt16, Index, DataMapper, mr, nr, ConjugateLhs,
            ConjugateRhs>::operator()(const DataMapper& res,
                                      const QInt16* blockA,
                                      const QInt16* blockB, Index rows,
                                      Index depth, Index cols, QInt32 alpha,
                                      Index strideA, Index strideB,
                                      Index offsetA, Index offsetB) {
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

#endif  // XLA_TSL_FRAMEWORK_FIXEDPOINT_MATMATPRODUCTNEON_H_
