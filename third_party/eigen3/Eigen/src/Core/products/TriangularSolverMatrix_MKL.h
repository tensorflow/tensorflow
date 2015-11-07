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
 *   Triangular matrix * matrix product functionality based on ?TRMM.
 ********************************************************************************
*/

#ifndef EIGEN_TRIANGULAR_SOLVER_MATRIX_MKL_H
#define EIGEN_TRIANGULAR_SOLVER_MATRIX_MKL_H

namespace Eigen {

namespace internal {

// implements LeftSide op(triangular)^-1 * general
#define EIGEN_MKL_TRSM_L(EIGTYPE, MKLTYPE, MKLPREFIX) \
template <typename Index, int Mode, bool Conjugate, int TriStorageOrder> \
struct triangular_solve_matrix<EIGTYPE,Index,OnTheLeft,Mode,Conjugate,TriStorageOrder,ColMajor> \
{ \
  enum { \
    IsLower = (Mode&Lower) == Lower, \
    IsUnitDiag  = (Mode&UnitDiag) ? 1 : 0, \
    IsZeroDiag  = (Mode&ZeroDiag) ? 1 : 0, \
    conjA = ((TriStorageOrder==ColMajor) && Conjugate) ? 1 : 0 \
  }; \
  static void run( \
      Index size, Index otherSize, \
      const EIGTYPE* _tri, Index triStride, \
      EIGTYPE* _other, Index otherStride, level3_blocking<EIGTYPE,EIGTYPE>& /*blocking*/) \
  { \
   MKL_INT m = size, n = otherSize, lda, ldb; \
   char side = 'L', uplo, diag='N', transa; \
   /* Set alpha_ */ \
   MKLTYPE alpha; \
   EIGTYPE myone(1); \
   assign_scalar_eig2mkl(alpha, myone); \
   ldb = otherStride;\
\
   const EIGTYPE *a; \
/* Set trans */ \
   transa = (TriStorageOrder==RowMajor) ? ((Conjugate) ? 'C' : 'T') : 'N'; \
/* Set uplo */ \
   uplo = IsLower ? 'L' : 'U'; \
   if (TriStorageOrder==RowMajor) uplo = (uplo == 'L') ? 'U' : 'L'; \
/* Set a, lda */ \
   typedef Matrix<EIGTYPE, Dynamic, Dynamic, TriStorageOrder> MatrixTri; \
   Map<const MatrixTri, 0, OuterStride<> > tri(_tri,size,size,OuterStride<>(triStride)); \
   MatrixTri a_tmp; \
\
   if (conjA) { \
     a_tmp = tri.conjugate(); \
     a = a_tmp.data(); \
     lda = a_tmp.outerStride(); \
   } else { \
     a = _tri; \
     lda = triStride; \
   } \
   if (IsUnitDiag) diag='U'; \
/* call ?trsm*/ \
   MKLPREFIX##trsm(&side, &uplo, &transa, &diag, &m, &n, &alpha, (const MKLTYPE*)a, &lda, (MKLTYPE*)_other, &ldb); \
 } \
};

EIGEN_MKL_TRSM_L(double, double, d)
EIGEN_MKL_TRSM_L(dcomplex, MKL_Complex16, z)
EIGEN_MKL_TRSM_L(float, float, s)
EIGEN_MKL_TRSM_L(scomplex, MKL_Complex8, c)


// implements RightSide general * op(triangular)^-1
#define EIGEN_MKL_TRSM_R(EIGTYPE, MKLTYPE, MKLPREFIX) \
template <typename Index, int Mode, bool Conjugate, int TriStorageOrder> \
struct triangular_solve_matrix<EIGTYPE,Index,OnTheRight,Mode,Conjugate,TriStorageOrder,ColMajor> \
{ \
  enum { \
    IsLower = (Mode&Lower) == Lower, \
    IsUnitDiag  = (Mode&UnitDiag) ? 1 : 0, \
    IsZeroDiag  = (Mode&ZeroDiag) ? 1 : 0, \
    conjA = ((TriStorageOrder==ColMajor) && Conjugate) ? 1 : 0 \
  }; \
  static void run( \
      Index size, Index otherSize, \
      const EIGTYPE* _tri, Index triStride, \
      EIGTYPE* _other, Index otherStride, level3_blocking<EIGTYPE,EIGTYPE>& /*blocking*/) \
  { \
   MKL_INT m = otherSize, n = size, lda, ldb; \
   char side = 'R', uplo, diag='N', transa; \
   /* Set alpha_ */ \
   MKLTYPE alpha; \
   EIGTYPE myone(1); \
   assign_scalar_eig2mkl(alpha, myone); \
   ldb = otherStride;\
\
   const EIGTYPE *a; \
/* Set trans */ \
   transa = (TriStorageOrder==RowMajor) ? ((Conjugate) ? 'C' : 'T') : 'N'; \
/* Set uplo */ \
   uplo = IsLower ? 'L' : 'U'; \
   if (TriStorageOrder==RowMajor) uplo = (uplo == 'L') ? 'U' : 'L'; \
/* Set a, lda */ \
   typedef Matrix<EIGTYPE, Dynamic, Dynamic, TriStorageOrder> MatrixTri; \
   Map<const MatrixTri, 0, OuterStride<> > tri(_tri,size,size,OuterStride<>(triStride)); \
   MatrixTri a_tmp; \
\
   if (conjA) { \
     a_tmp = tri.conjugate(); \
     a = a_tmp.data(); \
     lda = a_tmp.outerStride(); \
   } else { \
     a = _tri; \
     lda = triStride; \
   } \
   if (IsUnitDiag) diag='U'; \
/* call ?trsm*/ \
   MKLPREFIX##trsm(&side, &uplo, &transa, &diag, &m, &n, &alpha, (const MKLTYPE*)a, &lda, (MKLTYPE*)_other, &ldb); \
   /*std::cout << "TRMS_L specialization!\n";*/ \
 } \
};

EIGEN_MKL_TRSM_R(double, double, d)
EIGEN_MKL_TRSM_R(dcomplex, MKL_Complex16, z)
EIGEN_MKL_TRSM_R(float, float, s)
EIGEN_MKL_TRSM_R(scomplex, MKL_Complex8, c)


} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_TRIANGULAR_SOLVER_MATRIX_MKL_H
