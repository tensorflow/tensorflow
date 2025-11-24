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

#ifndef XLA_TSL_FRAMEWORK_FIXEDPOINT_MATVECPRODUCT_H_
#define XLA_TSL_FRAMEWORK_FIXEDPOINT_MATVECPRODUCT_H_

namespace Eigen {
namespace internal {

// Mat-Vec product
// Both lhs and rhs are encoded as 8bit signed integers
template <typename Index, typename LhsMapper, bool ConjugateLhs,
          typename RhsMapper, bool ConjugateRhs, int Version>
struct general_matrix_vector_product<Index, QInt8, LhsMapper, ColMajor,
                                     ConjugateLhs, QInt8, RhsMapper,
                                     ConjugateRhs, Version> {
  EIGEN_DONT_INLINE static void run(Index rows, Index cols,
                                    const LhsMapper& lhs, const RhsMapper& rhs,
                                    QInt32* res, Index resIncr, QInt8 alpha);
};

template <typename Index, typename LhsMapper, bool ConjugateLhs,
          typename RhsMapper, bool ConjugateRhs, int Version>
EIGEN_DONT_INLINE void general_matrix_vector_product<
    Index, QInt8, LhsMapper, ColMajor, ConjugateLhs, QInt8, RhsMapper,
    ConjugateRhs, Version>::run(Index rows, Index cols, const LhsMapper& lhs,
                                const RhsMapper& rhs, QInt32* res,
                                Index resIncr, QInt8 alpha) {
  eigen_assert(alpha.value == 1);
  eigen_assert(resIncr == 1);
  eigen_assert(rows > 0);
  eigen_assert(cols > 0);

  for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
      res[i] += lhs(i, j) * rhs(j, 0);
    }
  }
}

template <typename Index, typename LhsMapper, bool ConjugateLhs,
          typename RhsMapper, bool ConjugateRhs, int Version>
struct general_matrix_vector_product<Index, int8_t, LhsMapper, ColMajor,
                                     ConjugateLhs, int8_t, RhsMapper,
                                     ConjugateRhs, Version> {
  EIGEN_DONT_INLINE static void run(Index rows, Index cols,
                                    const LhsMapper& lhs, const RhsMapper& rhs,
                                    int32_t* res, Index resIncr, int8_t alpha);
};

template <typename Index, typename LhsMapper, bool ConjugateLhs,
          typename RhsMapper, bool ConjugateRhs, int Version>
EIGEN_DONT_INLINE void general_matrix_vector_product<
    Index, int8_t, LhsMapper, ColMajor, ConjugateLhs, int8_t, RhsMapper,
    ConjugateRhs, Version>::run(Index rows, Index cols, const LhsMapper& lhs,
                                const RhsMapper& rhs, int32_t* res,
                                Index resIncr, int8_t alpha) {
  eigen_assert(alpha == 1);
  eigen_assert(resIncr == 1);
  eigen_assert(rows > 0);
  eigen_assert(cols > 0);

  for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
      res[i] += lhs(i, j) * rhs(j, 0);
    }
  }
}

// Mat-Vec product
// Both lhs and rhs are encoded as 16bit signed integers
template <typename Index, typename LhsMapper, bool ConjugateLhs,
          typename RhsMapper, bool ConjugateRhs, int Version>
struct general_matrix_vector_product<Index, QInt16, LhsMapper, ColMajor,
                                     ConjugateLhs, QInt16, RhsMapper,
                                     ConjugateRhs, Version> {
  EIGEN_DONT_INLINE static void run(Index rows, Index cols,
                                    const LhsMapper& lhs, const RhsMapper& rhs,
                                    QInt32* res, Index resIncr, QInt16 alpha);
};

template <typename Index, typename LhsMapper, bool ConjugateLhs,
          typename RhsMapper, bool ConjugateRhs, int Version>
EIGEN_DONT_INLINE void general_matrix_vector_product<
    Index, QInt16, LhsMapper, ColMajor, ConjugateLhs, QInt16, RhsMapper,
    ConjugateRhs, Version>::run(Index rows, Index cols, const LhsMapper& lhs,
                                const RhsMapper& rhs, QInt32* res,
                                Index resIncr, QInt16 alpha) {
  eigen_assert(alpha.value == 1);
  eigen_assert(resIncr == 1);
  eigen_assert(rows > 0);
  eigen_assert(cols > 0);

  for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
      res[i] += lhs(i, j) * rhs(j, 0);
    }
  }
}

// Mat-Vec product
// The lhs is encoded using 8bit signed integers, the rhs using 8bit unsigned
// integers
template <typename Index, typename LhsMapper, bool ConjugateLhs,
          typename RhsMapper, bool ConjugateRhs, int Version>
struct general_matrix_vector_product<Index, QInt8, LhsMapper, ColMajor,
                                     ConjugateLhs, QUInt8, RhsMapper,
                                     ConjugateRhs, Version> {
  EIGEN_DONT_INLINE static void run(Index rows, Index cols,
                                    const LhsMapper& lhs, const RhsMapper& rhs,
                                    QInt32* res, Index resIncr, QUInt8 alpha);
};

template <typename Index, typename LhsMapper, bool ConjugateLhs,
          typename RhsMapper, bool ConjugateRhs, int Version>
EIGEN_DONT_INLINE void general_matrix_vector_product<
    Index, QInt8, LhsMapper, ColMajor, ConjugateLhs, QUInt8, RhsMapper,
    ConjugateRhs, Version>::run(Index rows, Index cols, const LhsMapper& lhs,
                                const RhsMapper& rhs, QInt32* res,
                                Index resIncr, QUInt8 alpha) {
  eigen_assert(alpha.value == 1);
  eigen_assert(resIncr == 1);
  eigen_assert(rows > 0);
  eigen_assert(cols > 0);

  for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
      res[i] += lhs(i, j) * rhs(j, 0);
    }
  }
}

// Mat-Vec product
// The lhs is encoded using bit unsigned integers, the rhs using 8bit signed
// integers
template <typename Index, typename LhsMapper, bool ConjugateLhs,
          typename RhsMapper, bool ConjugateRhs, int Version>
struct general_matrix_vector_product<Index, QUInt8, LhsMapper, ColMajor,
                                     ConjugateLhs, QInt8, RhsMapper,
                                     ConjugateRhs, Version> {
  EIGEN_DONT_INLINE static void run(Index rows, Index cols,
                                    const LhsMapper& lhs, const RhsMapper& rhs,
                                    QInt32* res, Index resIncr, QInt8 alpha);
};

template <typename Index, typename LhsMapper, bool ConjugateLhs,
          typename RhsMapper, bool ConjugateRhs, int Version>
EIGEN_DONT_INLINE void general_matrix_vector_product<
    Index, QUInt8, LhsMapper, ColMajor, ConjugateLhs, QInt8, RhsMapper,
    ConjugateRhs, Version>::run(Index rows, Index cols, const LhsMapper& lhs,
                                const RhsMapper& rhs, QInt32* res,
                                Index resIncr, QInt8 alpha) {
  eigen_assert(alpha.value == 1);
  eigen_assert(resIncr == 1);
  eigen_assert(rows > 0);
  eigen_assert(cols > 0);

  for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
      res[i] += lhs(i, j) * rhs(j, 0);
    }
  }
}

}  // namespace internal
}  // namespace Eigen

#endif  // XLA_TSL_FRAMEWORK_FIXEDPOINT_MATVECPRODUCT_H_
