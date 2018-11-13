// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CXX11_SRC_FIXEDPOINT_MATVECPRODUCT_H_
#define CXX11_SRC_FIXEDPOINT_MATVECPRODUCT_H_

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

#endif  // CXX11_SRC_FIXEDPOINT_MATVECPRODUCT_H_
