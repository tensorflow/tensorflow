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
 *    Self-adjoint eigenvalues/eigenvectors.
 ********************************************************************************
*/

#ifndef EIGEN_SAEIGENSOLVER_MKL_H
#define EIGEN_SAEIGENSOLVER_MKL_H

#include "Eigen/src/Core/util/MKL_support.h"

namespace Eigen { 

/** \internal Specialization for the data types supported by MKL */

#define EIGEN_MKL_EIG_SELFADJ(EIGTYPE, MKLTYPE, MKLRTYPE, MKLNAME, EIGCOLROW, MKLCOLROW ) \
template<> inline \
SelfAdjointEigenSolver<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW> >& \
SelfAdjointEigenSolver<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW> >::compute(const Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW>& matrix, int options) \
{ \
  eigen_assert(matrix.cols() == matrix.rows()); \
  eigen_assert((options&~(EigVecMask|GenEigMask))==0 \
          && (options&EigVecMask)!=EigVecMask \
          && "invalid option parameter"); \
  bool computeEigenvectors = (options&ComputeEigenvectors)==ComputeEigenvectors; \
  lapack_int n = matrix.cols(), lda, matrix_order, info; \
  m_eivalues.resize(n,1); \
  m_subdiag.resize(n-1); \
  m_eivec = matrix; \
\
  if(n==1) \
  { \
    m_eivalues.coeffRef(0,0) = numext::real(matrix.coeff(0,0)); \
    if(computeEigenvectors) m_eivec.setOnes(n,n); \
    m_info = Success; \
    m_isInitialized = true; \
    m_eigenvectorsOk = computeEigenvectors; \
    return *this; \
  } \
\
  lda = matrix.outerStride(); \
  matrix_order=MKLCOLROW; \
  char jobz, uplo='L'/*, range='A'*/; \
  jobz = computeEigenvectors ? 'V' : 'N'; \
\
  info = LAPACKE_##MKLNAME( matrix_order, jobz, uplo, n, (MKLTYPE*)m_eivec.data(), lda, (MKLRTYPE*)m_eivalues.data() ); \
  m_info = (info==0) ? Success : NoConvergence; \
  m_isInitialized = true; \
  m_eigenvectorsOk = computeEigenvectors; \
  return *this; \
}


EIGEN_MKL_EIG_SELFADJ(double,   double,        double, dsyev, ColMajor, LAPACK_COL_MAJOR)
EIGEN_MKL_EIG_SELFADJ(float,    float,         float,  ssyev, ColMajor, LAPACK_COL_MAJOR)
EIGEN_MKL_EIG_SELFADJ(dcomplex, MKL_Complex16, double, zheev, ColMajor, LAPACK_COL_MAJOR)
EIGEN_MKL_EIG_SELFADJ(scomplex, MKL_Complex8,  float,  cheev, ColMajor, LAPACK_COL_MAJOR)

EIGEN_MKL_EIG_SELFADJ(double,   double,        double, dsyev, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_MKL_EIG_SELFADJ(float,    float,         float,  ssyev, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_MKL_EIG_SELFADJ(dcomplex, MKL_Complex16, double, zheev, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_MKL_EIG_SELFADJ(scomplex, MKL_Complex8,  float,  cheev, RowMajor, LAPACK_ROW_MAJOR)

} // end namespace Eigen

#endif // EIGEN_SAEIGENSOLVER_H
