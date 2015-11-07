/*
 Copyright (c) 2011, Intel Corporation. All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of Intel Corporation nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ********************************************************************************
 *   Content : Eigen bindings to Intel(R) MKL
 *     LLt decomposition based on LAPACKE_?potrf function.
 ********************************************************************************
*/

#ifndef EIGEN_LLT_MKL_H
#define EIGEN_LLT_MKL_H

#include "Eigen/src/Core/util/MKL_support.h"
#include <iostream>

namespace Eigen { 

namespace internal {

template<typename Scalar> struct mkl_llt;

#define EIGEN_MKL_LLT(EIGTYPE, MKLTYPE, MKLPREFIX) \
template<> struct mkl_llt<EIGTYPE> \
{ \
  template<typename MatrixType> \
  static inline typename MatrixType::Index potrf(MatrixType& m, char uplo) \
  { \
    lapack_int matrix_order; \
    lapack_int size, lda, info, StorageOrder; \
    EIGTYPE* a; \
    eigen_assert(m.rows()==m.cols()); \
    /* Set up parameters for ?potrf */ \
    size = m.rows(); \
    StorageOrder = MatrixType::Flags&RowMajorBit?RowMajor:ColMajor; \
    matrix_order = StorageOrder==RowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR; \
    a = &(m.coeffRef(0,0)); \
    lda = m.outerStride(); \
\
    info = LAPACKE_##MKLPREFIX##potrf( matrix_order, uplo, size, (MKLTYPE*)a, lda ); \
    info = (info==0) ? Success : NumericalIssue; \
    return info; \
  } \
}; \
template<> struct llt_inplace<EIGTYPE, Lower> \
{ \
  template<typename MatrixType> \
  static typename MatrixType::Index blocked(MatrixType& m) \
  { \
    return mkl_llt<EIGTYPE>::potrf(m, 'L'); \
  } \
  template<typename MatrixType, typename VectorType> \
  static typename MatrixType::Index rankUpdate(MatrixType& mat, const VectorType& vec, const typename MatrixType::RealScalar& sigma) \
  { return Eigen::internal::llt_rank_update_lower(mat, vec, sigma); } \
}; \
template<> struct llt_inplace<EIGTYPE, Upper> \
{ \
  template<typename MatrixType> \
  static typename MatrixType::Index blocked(MatrixType& m) \
  { \
    return mkl_llt<EIGTYPE>::potrf(m, 'U'); \
  } \
  template<typename MatrixType, typename VectorType> \
  static typename MatrixType::Index rankUpdate(MatrixType& mat, const VectorType& vec, const typename MatrixType::RealScalar& sigma) \
  { \
    Transpose<MatrixType> matt(mat); \
    return llt_inplace<EIGTYPE, Lower>::rankUpdate(matt, vec.conjugate(), sigma); \
  } \
};

EIGEN_MKL_LLT(double, double, d)
EIGEN_MKL_LLT(float, float, s)
EIGEN_MKL_LLT(dcomplex, MKL_Complex16, z)
EIGEN_MKL_LLT(scomplex, MKL_Complex8, c)

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_LLT_MKL_H
