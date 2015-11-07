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
 *    Householder QR decomposition of a matrix with column pivoting based on
 *    LAPACKE_?geqp3 function.
 ********************************************************************************
*/

#ifndef EIGEN_COLPIVOTINGHOUSEHOLDERQR_MKL_H
#define EIGEN_COLPIVOTINGHOUSEHOLDERQR_MKL_H

#include "Eigen/src/Core/util/MKL_support.h"

namespace Eigen { 

/** \internal Specialization for the data types supported by MKL */

#define EIGEN_MKL_QR_COLPIV(EIGTYPE, MKLTYPE, MKLPREFIX, EIGCOLROW, MKLCOLROW) \
template<> inline \
ColPivHouseholderQR<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic> >& \
ColPivHouseholderQR<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic> >::compute( \
              const Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic>& matrix) \
\
{ \
  using std::abs; \
  typedef Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic> MatrixType; \
  typedef MatrixType::Scalar Scalar; \
  typedef MatrixType::RealScalar RealScalar; \
  Index rows = matrix.rows();\
  Index cols = matrix.cols();\
  Index size = matrix.diagonalSize();\
\
  m_qr = matrix;\
  m_hCoeffs.resize(size);\
\
  m_colsTranspositions.resize(cols);\
  /*Index number_of_transpositions = 0;*/ \
\
  m_nonzero_pivots = 0; \
  m_maxpivot = RealScalar(0);\
  m_colsPermutation.resize(cols); \
  m_colsPermutation.indices().setZero(); \
\
  lapack_int lda = m_qr.outerStride(), i; \
  lapack_int matrix_order = MKLCOLROW; \
  LAPACKE_##MKLPREFIX##geqp3( matrix_order, rows, cols, (MKLTYPE*)m_qr.data(), lda, (lapack_int*)m_colsPermutation.indices().data(), (MKLTYPE*)m_hCoeffs.data()); \
  m_isInitialized = true; \
  m_maxpivot=m_qr.diagonal().cwiseAbs().maxCoeff(); \
  m_hCoeffs.adjointInPlace(); \
  RealScalar premultiplied_threshold = abs(m_maxpivot) * threshold(); \
  lapack_int *perm = m_colsPermutation.indices().data(); \
  for(i=0;i<size;i++) { \
    m_nonzero_pivots += (abs(m_qr.coeff(i,i)) > premultiplied_threshold);\
  } \
  for(i=0;i<cols;i++) perm[i]--;\
\
  /*m_det_pq = (number_of_transpositions%2) ? -1 : 1;  // TODO: It's not needed now; fix upon availability in Eigen */ \
\
  return *this; \
}

EIGEN_MKL_QR_COLPIV(double,   double,        d, ColMajor, LAPACK_COL_MAJOR)
EIGEN_MKL_QR_COLPIV(float,    float,         s, ColMajor, LAPACK_COL_MAJOR)
EIGEN_MKL_QR_COLPIV(dcomplex, MKL_Complex16, z, ColMajor, LAPACK_COL_MAJOR)
EIGEN_MKL_QR_COLPIV(scomplex, MKL_Complex8,  c, ColMajor, LAPACK_COL_MAJOR)

EIGEN_MKL_QR_COLPIV(double,   double,        d, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_MKL_QR_COLPIV(float,    float,         s, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_MKL_QR_COLPIV(dcomplex, MKL_Complex16, z, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_MKL_QR_COLPIV(scomplex, MKL_Complex8,  c, RowMajor, LAPACK_ROW_MAJOR)

} // end namespace Eigen

#endif // EIGEN_COLPIVOTINGHOUSEHOLDERQR_MKL_H
