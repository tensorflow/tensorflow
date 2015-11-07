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
 *    Complex Schur needed to complex unsymmetrical eigenvalues/eigenvectors.
 ********************************************************************************
*/

#ifndef EIGEN_COMPLEX_SCHUR_MKL_H
#define EIGEN_COMPLEX_SCHUR_MKL_H

#include "Eigen/src/Core/util/MKL_support.h"

namespace Eigen { 

/** \internal Specialization for the data types supported by MKL */

#define EIGEN_MKL_SCHUR_COMPLEX(EIGTYPE, MKLTYPE, MKLPREFIX, MKLPREFIX_U, EIGCOLROW, MKLCOLROW) \
template<> inline \
ComplexSchur<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW> >& \
ComplexSchur<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW> >::compute(const Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW>& matrix, bool computeU) \
{ \
  typedef Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW> MatrixType; \
  typedef MatrixType::Scalar Scalar; \
  typedef MatrixType::RealScalar RealScalar; \
  typedef std::complex<RealScalar> ComplexScalar; \
\
  eigen_assert(matrix.cols() == matrix.rows()); \
\
  m_matUisUptodate = false; \
  if(matrix.cols() == 1) \
  { \
    m_matT = matrix.cast<ComplexScalar>(); \
    if(computeU)  m_matU = ComplexMatrixType::Identity(1,1); \
      m_info = Success; \
      m_isInitialized = true; \
      m_matUisUptodate = computeU; \
      return *this; \
  } \
  lapack_int n = matrix.cols(), sdim, info; \
  lapack_int lda = matrix.outerStride(); \
  lapack_int matrix_order = MKLCOLROW; \
  char jobvs, sort='N'; \
  LAPACK_##MKLPREFIX_U##_SELECT1 select = 0; \
  jobvs = (computeU) ? 'V' : 'N'; \
  m_matU.resize(n, n); \
  lapack_int ldvs  = m_matU.outerStride(); \
  m_matT = matrix; \
  Matrix<EIGTYPE, Dynamic, Dynamic> w; \
  w.resize(n, 1);\
  info = LAPACKE_##MKLPREFIX##gees( matrix_order, jobvs, sort, select, n, (MKLTYPE*)m_matT.data(), lda, &sdim, (MKLTYPE*)w.data(), (MKLTYPE*)m_matU.data(), ldvs ); \
  if(info == 0) \
    m_info = Success; \
  else \
    m_info = NoConvergence; \
\
  m_isInitialized = true; \
  m_matUisUptodate = computeU; \
  return *this; \
\
}

EIGEN_MKL_SCHUR_COMPLEX(dcomplex, MKL_Complex16, z, Z, ColMajor, LAPACK_COL_MAJOR)
EIGEN_MKL_SCHUR_COMPLEX(scomplex, MKL_Complex8,  c, C, ColMajor, LAPACK_COL_MAJOR)
EIGEN_MKL_SCHUR_COMPLEX(dcomplex, MKL_Complex16, z, Z, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_MKL_SCHUR_COMPLEX(scomplex, MKL_Complex8,  c, C, RowMajor, LAPACK_ROW_MAJOR)

} // end namespace Eigen

#endif // EIGEN_COMPLEX_SCHUR_MKL_H
