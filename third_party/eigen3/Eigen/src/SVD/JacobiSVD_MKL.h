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
 *    Singular Value Decomposition - SVD.
 ********************************************************************************
*/

#ifndef EIGEN_JACOBISVD_MKL_H
#define EIGEN_JACOBISVD_MKL_H

#include "Eigen/src/Core/util/MKL_support.h"

namespace Eigen { 

/** \internal Specialization for the data types supported by MKL */

#define EIGEN_MKL_SVD(EIGTYPE, MKLTYPE, MKLRTYPE, MKLPREFIX, EIGCOLROW, MKLCOLROW) \
template<> inline \
JacobiSVD<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic>, ColPivHouseholderQRPreconditioner>& \
JacobiSVD<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic>, ColPivHouseholderQRPreconditioner>::compute(const Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic>& matrix, unsigned int computationOptions) \
{ \
  typedef Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic> MatrixType; \
  typedef MatrixType::Scalar Scalar; \
  typedef MatrixType::RealScalar RealScalar; \
  allocate(matrix.rows(), matrix.cols(), computationOptions); \
\
  /*const RealScalar precision = RealScalar(2) * NumTraits<Scalar>::epsilon();*/ \
  m_nonzeroSingularValues = m_diagSize; \
\
  lapack_int lda = matrix.outerStride(), ldu, ldvt; \
  lapack_int matrix_order = MKLCOLROW; \
  char jobu, jobvt; \
  MKLTYPE *u, *vt, dummy; \
  jobu  = (m_computeFullU) ? 'A' : (m_computeThinU) ? 'S' : 'N'; \
  jobvt = (m_computeFullV) ? 'A' : (m_computeThinV) ? 'S' : 'N'; \
  if (computeU()) { \
    ldu  = m_matrixU.outerStride(); \
    u    = (MKLTYPE*)m_matrixU.data(); \
  } else { ldu=1; u=&dummy; }\
  MatrixType localV; \
  ldvt = (m_computeFullV) ? m_cols : (m_computeThinV) ? m_diagSize : 1; \
  if (computeV()) { \
    localV.resize(ldvt, m_cols); \
    vt   = (MKLTYPE*)localV.data(); \
  } else { ldvt=1; vt=&dummy; }\
  Matrix<MKLRTYPE, Dynamic, Dynamic> superb; superb.resize(m_diagSize, 1); \
  MatrixType m_temp; m_temp = matrix; \
  LAPACKE_##MKLPREFIX##gesvd( matrix_order, jobu, jobvt, m_rows, m_cols, (MKLTYPE*)m_temp.data(), lda, (MKLRTYPE*)m_singularValues.data(), u, ldu, vt, ldvt, superb.data()); \
  if (computeV()) m_matrixV = localV.adjoint(); \
 /* for(int i=0;i<m_diagSize;i++) if (m_singularValues.coeffRef(i) < precision) { m_nonzeroSingularValues--; m_singularValues.coeffRef(i)=RealScalar(0);}*/ \
  m_isInitialized = true; \
  return *this; \
}

EIGEN_MKL_SVD(double,   double,        double, d, ColMajor, LAPACK_COL_MAJOR)
EIGEN_MKL_SVD(float,    float,         float , s, ColMajor, LAPACK_COL_MAJOR)
EIGEN_MKL_SVD(dcomplex, MKL_Complex16, double, z, ColMajor, LAPACK_COL_MAJOR)
EIGEN_MKL_SVD(scomplex, MKL_Complex8,  float , c, ColMajor, LAPACK_COL_MAJOR)

EIGEN_MKL_SVD(double,   double,        double, d, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_MKL_SVD(float,    float,         float , s, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_MKL_SVD(dcomplex, MKL_Complex16, double, z, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_MKL_SVD(scomplex, MKL_Complex8,  float , c, RowMajor, LAPACK_ROW_MAJOR)

} // end namespace Eigen

#endif // EIGEN_JACOBISVD_MKL_H
