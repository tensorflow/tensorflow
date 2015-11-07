// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2001 Intel Corporation
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// The SSE code for the 4x4 float and double matrix inverse in this file
// comes from the following Intel's library:
// http://software.intel.com/en-us/articles/optimized-matrix-library-for-use-with-the-intel-pentiumr-4-processors-sse2-instructions/
//
// Here is the respective copyright and license statement:
//
//   Copyright (c) 2001 Intel Corporation.
//
// Permition is granted to use, copy, distribute and prepare derivative works
// of this library for any purpose and without fee, provided, that the above
// copyright notice and this statement appear in all copies.
// Intel makes no representations about the suitability of this software for
// any purpose, and specifically disclaims all warranties.
// See LEGAL.TXT for all the legal information.

#ifndef EIGEN_INVERSE_SSE_H
#define EIGEN_INVERSE_SSE_H

namespace Eigen { 

namespace internal {

template<typename MatrixType, typename ResultType>
struct compute_inverse_size4<Architecture::SSE, float, MatrixType, ResultType>
{
  enum {
    MatrixAlignment     = bool(MatrixType::Flags&AlignedBit),
    ResultAlignment     = bool(ResultType::Flags&AlignedBit),
    StorageOrdersMatch  = (MatrixType::Flags&RowMajorBit) == (ResultType::Flags&RowMajorBit)
  };
  
  static void run(const MatrixType& matrix, ResultType& result)
  {
    EIGEN_ALIGN16 const unsigned int _Sign_PNNP[4] = { 0x00000000, 0x80000000, 0x80000000, 0x00000000 };

    // Load the full matrix into registers
    __m128 _L1 = matrix.template packet<MatrixAlignment>( 0);
    __m128 _L2 = matrix.template packet<MatrixAlignment>( 4);
    __m128 _L3 = matrix.template packet<MatrixAlignment>( 8);
    __m128 _L4 = matrix.template packet<MatrixAlignment>(12);

    // The inverse is calculated using "Divide and Conquer" technique. The
    // original matrix is divide into four 2x2 sub-matrices. Since each
    // register holds four matrix element, the smaller matrices are
    // represented as a registers. Hence we get a better locality of the
    // calculations.

    __m128 A, B, C, D; // the four sub-matrices
    if(!StorageOrdersMatch)
    {
      A = _mm_unpacklo_ps(_L1, _L2);
      B = _mm_unpacklo_ps(_L3, _L4);
      C = _mm_unpackhi_ps(_L1, _L2);
      D = _mm_unpackhi_ps(_L3, _L4);
    }
    else
    {
      A = _mm_movelh_ps(_L1, _L2);
      B = _mm_movehl_ps(_L2, _L1);
      C = _mm_movelh_ps(_L3, _L4);
      D = _mm_movehl_ps(_L4, _L3);
    }

    __m128 iA, iB, iC, iD,                 // partial inverse of the sub-matrices
            DC, AB;
    __m128 dA, dB, dC, dD;                 // determinant of the sub-matrices
    __m128 det, d, d1, d2;
    __m128 rd;                             // reciprocal of the determinant

    //  AB = A# * B
    AB = _mm_mul_ps(_mm_shuffle_ps(A,A,0x0F), B);
    AB = _mm_sub_ps(AB,_mm_mul_ps(_mm_shuffle_ps(A,A,0xA5), _mm_shuffle_ps(B,B,0x4E)));
    //  DC = D# * C
    DC = _mm_mul_ps(_mm_shuffle_ps(D,D,0x0F), C);
    DC = _mm_sub_ps(DC,_mm_mul_ps(_mm_shuffle_ps(D,D,0xA5), _mm_shuffle_ps(C,C,0x4E)));

    //  dA = |A|
    dA = _mm_mul_ps(_mm_shuffle_ps(A, A, 0x5F),A);
    dA = _mm_sub_ss(dA, _mm_movehl_ps(dA,dA));
    //  dB = |B|
    dB = _mm_mul_ps(_mm_shuffle_ps(B, B, 0x5F),B);
    dB = _mm_sub_ss(dB, _mm_movehl_ps(dB,dB));

    //  dC = |C|
    dC = _mm_mul_ps(_mm_shuffle_ps(C, C, 0x5F),C);
    dC = _mm_sub_ss(dC, _mm_movehl_ps(dC,dC));
    //  dD = |D|
    dD = _mm_mul_ps(_mm_shuffle_ps(D, D, 0x5F),D);
    dD = _mm_sub_ss(dD, _mm_movehl_ps(dD,dD));

    //  d = trace(AB*DC) = trace(A#*B*D#*C)
    d = _mm_mul_ps(_mm_shuffle_ps(DC,DC,0xD8),AB);

    //  iD = C*A#*B
    iD = _mm_mul_ps(_mm_shuffle_ps(C,C,0xA0), _mm_movelh_ps(AB,AB));
    iD = _mm_add_ps(iD,_mm_mul_ps(_mm_shuffle_ps(C,C,0xF5), _mm_movehl_ps(AB,AB)));
    //  iA = B*D#*C
    iA = _mm_mul_ps(_mm_shuffle_ps(B,B,0xA0), _mm_movelh_ps(DC,DC));
    iA = _mm_add_ps(iA,_mm_mul_ps(_mm_shuffle_ps(B,B,0xF5), _mm_movehl_ps(DC,DC)));

    //  d = trace(AB*DC) = trace(A#*B*D#*C) [continue]
    d  = _mm_add_ps(d, _mm_movehl_ps(d, d));
    d  = _mm_add_ss(d, _mm_shuffle_ps(d, d, 1));
    d1 = _mm_mul_ss(dA,dD);
    d2 = _mm_mul_ss(dB,dC);

    //  iD = D*|A| - C*A#*B
    iD = _mm_sub_ps(_mm_mul_ps(D,_mm_shuffle_ps(dA,dA,0)), iD);

    //  iA = A*|D| - B*D#*C;
    iA = _mm_sub_ps(_mm_mul_ps(A,_mm_shuffle_ps(dD,dD,0)), iA);

    //  det = |A|*|D| + |B|*|C| - trace(A#*B*D#*C)
    det = _mm_sub_ss(_mm_add_ss(d1,d2),d);
    rd  = _mm_div_ss(_mm_set_ss(1.0f), det);

//     #ifdef ZERO_SINGULAR
//         rd = _mm_and_ps(_mm_cmpneq_ss(det,_mm_setzero_ps()), rd);
//     #endif

    //  iB = D * (A#B)# = D*B#*A
    iB = _mm_mul_ps(D, _mm_shuffle_ps(AB,AB,0x33));
    iB = _mm_sub_ps(iB, _mm_mul_ps(_mm_shuffle_ps(D,D,0xB1), _mm_shuffle_ps(AB,AB,0x66)));
    //  iC = A * (D#C)# = A*C#*D
    iC = _mm_mul_ps(A, _mm_shuffle_ps(DC,DC,0x33));
    iC = _mm_sub_ps(iC, _mm_mul_ps(_mm_shuffle_ps(A,A,0xB1), _mm_shuffle_ps(DC,DC,0x66)));

    rd = _mm_shuffle_ps(rd,rd,0);
    rd = _mm_xor_ps(rd, _mm_load_ps((float*)_Sign_PNNP));

    //  iB = C*|B| - D*B#*A
    iB = _mm_sub_ps(_mm_mul_ps(C,_mm_shuffle_ps(dB,dB,0)), iB);

    //  iC = B*|C| - A*C#*D;
    iC = _mm_sub_ps(_mm_mul_ps(B,_mm_shuffle_ps(dC,dC,0)), iC);

    //  iX = iX / det
    iA = _mm_mul_ps(rd,iA);
    iB = _mm_mul_ps(rd,iB);
    iC = _mm_mul_ps(rd,iC);
    iD = _mm_mul_ps(rd,iD);

    result.template writePacket<ResultAlignment>( 0, _mm_shuffle_ps(iA,iB,0x77));
    result.template writePacket<ResultAlignment>( 4, _mm_shuffle_ps(iA,iB,0x22));
    result.template writePacket<ResultAlignment>( 8, _mm_shuffle_ps(iC,iD,0x77));
    result.template writePacket<ResultAlignment>(12, _mm_shuffle_ps(iC,iD,0x22));
  }

};

template<typename MatrixType, typename ResultType>
struct compute_inverse_size4<Architecture::SSE, double, MatrixType, ResultType>
{
  enum {
    MatrixAlignment = bool(MatrixType::Flags&AlignedBit),
    ResultAlignment = bool(ResultType::Flags&AlignedBit),
    StorageOrdersMatch  = (MatrixType::Flags&RowMajorBit) == (ResultType::Flags&RowMajorBit)
  };
  static void run(const MatrixType& matrix, ResultType& result)
  {
    const __m128d _Sign_NP = _mm_castsi128_pd(_mm_set_epi32(0x0,0x0,0x80000000,0x0));
    const __m128d _Sign_PN = _mm_castsi128_pd(_mm_set_epi32(0x80000000,0x0,0x0,0x0));

    // The inverse is calculated using "Divide and Conquer" technique. The
    // original matrix is divide into four 2x2 sub-matrices. Since each
    // register of the matrix holds two element, the smaller matrices are
    // consisted of two registers. Hence we get a better locality of the
    // calculations.

    // the four sub-matrices
    __m128d A1, A2, B1, B2, C1, C2, D1, D2;
    
    if(StorageOrdersMatch)
    {
      A1 = matrix.template packet<MatrixAlignment>( 0); B1 = matrix.template packet<MatrixAlignment>( 2);
      A2 = matrix.template packet<MatrixAlignment>( 4); B2 = matrix.template packet<MatrixAlignment>( 6);
      C1 = matrix.template packet<MatrixAlignment>( 8); D1 = matrix.template packet<MatrixAlignment>(10);
      C2 = matrix.template packet<MatrixAlignment>(12); D2 = matrix.template packet<MatrixAlignment>(14);
    }
    else
    {
      __m128d tmp;
      A1 = matrix.template packet<MatrixAlignment>( 0); C1 = matrix.template packet<MatrixAlignment>( 2);
      A2 = matrix.template packet<MatrixAlignment>( 4); C2 = matrix.template packet<MatrixAlignment>( 6);
      tmp = A1;
      A1 = _mm_unpacklo_pd(A1,A2);
      A2 = _mm_unpackhi_pd(tmp,A2);
      tmp = C1;
      C1 = _mm_unpacklo_pd(C1,C2);
      C2 = _mm_unpackhi_pd(tmp,C2);
      
      B1 = matrix.template packet<MatrixAlignment>( 8); D1 = matrix.template packet<MatrixAlignment>(10);
      B2 = matrix.template packet<MatrixAlignment>(12); D2 = matrix.template packet<MatrixAlignment>(14);
      tmp = B1;
      B1 = _mm_unpacklo_pd(B1,B2);
      B2 = _mm_unpackhi_pd(tmp,B2);
      tmp = D1;
      D1 = _mm_unpacklo_pd(D1,D2);
      D2 = _mm_unpackhi_pd(tmp,D2);
    }
    
    __m128d iA1, iA2, iB1, iB2, iC1, iC2, iD1, iD2,     // partial invese of the sub-matrices
            DC1, DC2, AB1, AB2;
    __m128d dA, dB, dC, dD;     // determinant of the sub-matrices
    __m128d det, d1, d2, rd;

    //  dA = |A|
    dA = _mm_shuffle_pd(A2, A2, 1);
    dA = _mm_mul_pd(A1, dA);
    dA = _mm_sub_sd(dA, _mm_shuffle_pd(dA,dA,3));
    //  dB = |B|
    dB = _mm_shuffle_pd(B2, B2, 1);
    dB = _mm_mul_pd(B1, dB);
    dB = _mm_sub_sd(dB, _mm_shuffle_pd(dB,dB,3));

    //  AB = A# * B
    AB1 = _mm_mul_pd(B1, _mm_shuffle_pd(A2,A2,3));
    AB2 = _mm_mul_pd(B2, _mm_shuffle_pd(A1,A1,0));
    AB1 = _mm_sub_pd(AB1, _mm_mul_pd(B2, _mm_shuffle_pd(A1,A1,3)));
    AB2 = _mm_sub_pd(AB2, _mm_mul_pd(B1, _mm_shuffle_pd(A2,A2,0)));

    //  dC = |C|
    dC = _mm_shuffle_pd(C2, C2, 1);
    dC = _mm_mul_pd(C1, dC);
    dC = _mm_sub_sd(dC, _mm_shuffle_pd(dC,dC,3));
    //  dD = |D|
    dD = _mm_shuffle_pd(D2, D2, 1);
    dD = _mm_mul_pd(D1, dD);
    dD = _mm_sub_sd(dD, _mm_shuffle_pd(dD,dD,3));

    //  DC = D# * C
    DC1 = _mm_mul_pd(C1, _mm_shuffle_pd(D2,D2,3));
    DC2 = _mm_mul_pd(C2, _mm_shuffle_pd(D1,D1,0));
    DC1 = _mm_sub_pd(DC1, _mm_mul_pd(C2, _mm_shuffle_pd(D1,D1,3)));
    DC2 = _mm_sub_pd(DC2, _mm_mul_pd(C1, _mm_shuffle_pd(D2,D2,0)));

    //  rd = trace(AB*DC) = trace(A#*B*D#*C)
    d1 = _mm_mul_pd(AB1, _mm_shuffle_pd(DC1, DC2, 0));
    d2 = _mm_mul_pd(AB2, _mm_shuffle_pd(DC1, DC2, 3));
    rd = _mm_add_pd(d1, d2);
    rd = _mm_add_sd(rd, _mm_shuffle_pd(rd, rd,3));

    //  iD = C*A#*B
    iD1 = _mm_mul_pd(AB1, _mm_shuffle_pd(C1,C1,0));
    iD2 = _mm_mul_pd(AB1, _mm_shuffle_pd(C2,C2,0));
    iD1 = _mm_add_pd(iD1, _mm_mul_pd(AB2, _mm_shuffle_pd(C1,C1,3)));
    iD2 = _mm_add_pd(iD2, _mm_mul_pd(AB2, _mm_shuffle_pd(C2,C2,3)));

    //  iA = B*D#*C
    iA1 = _mm_mul_pd(DC1, _mm_shuffle_pd(B1,B1,0));
    iA2 = _mm_mul_pd(DC1, _mm_shuffle_pd(B2,B2,0));
    iA1 = _mm_add_pd(iA1, _mm_mul_pd(DC2, _mm_shuffle_pd(B1,B1,3)));
    iA2 = _mm_add_pd(iA2, _mm_mul_pd(DC2, _mm_shuffle_pd(B2,B2,3)));

    //  iD = D*|A| - C*A#*B
    dA = _mm_shuffle_pd(dA,dA,0);
    iD1 = _mm_sub_pd(_mm_mul_pd(D1, dA), iD1);
    iD2 = _mm_sub_pd(_mm_mul_pd(D2, dA), iD2);

    //  iA = A*|D| - B*D#*C;
    dD = _mm_shuffle_pd(dD,dD,0);
    iA1 = _mm_sub_pd(_mm_mul_pd(A1, dD), iA1);
    iA2 = _mm_sub_pd(_mm_mul_pd(A2, dD), iA2);

    d1 = _mm_mul_sd(dA, dD);
    d2 = _mm_mul_sd(dB, dC);

    //  iB = D * (A#B)# = D*B#*A
    iB1 = _mm_mul_pd(D1, _mm_shuffle_pd(AB2,AB1,1));
    iB2 = _mm_mul_pd(D2, _mm_shuffle_pd(AB2,AB1,1));
    iB1 = _mm_sub_pd(iB1, _mm_mul_pd(_mm_shuffle_pd(D1,D1,1), _mm_shuffle_pd(AB2,AB1,2)));
    iB2 = _mm_sub_pd(iB2, _mm_mul_pd(_mm_shuffle_pd(D2,D2,1), _mm_shuffle_pd(AB2,AB1,2)));

    //  det = |A|*|D| + |B|*|C| - trace(A#*B*D#*C)
    det = _mm_add_sd(d1, d2);
    det = _mm_sub_sd(det, rd);

    //  iC = A * (D#C)# = A*C#*D
    iC1 = _mm_mul_pd(A1, _mm_shuffle_pd(DC2,DC1,1));
    iC2 = _mm_mul_pd(A2, _mm_shuffle_pd(DC2,DC1,1));
    iC1 = _mm_sub_pd(iC1, _mm_mul_pd(_mm_shuffle_pd(A1,A1,1), _mm_shuffle_pd(DC2,DC1,2)));
    iC2 = _mm_sub_pd(iC2, _mm_mul_pd(_mm_shuffle_pd(A2,A2,1), _mm_shuffle_pd(DC2,DC1,2)));

    rd = _mm_div_sd(_mm_set_sd(1.0), det);
//     #ifdef ZERO_SINGULAR
//         rd = _mm_and_pd(_mm_cmpneq_sd(det,_mm_setzero_pd()), rd);
//     #endif
    rd = _mm_shuffle_pd(rd,rd,0);

    //  iB = C*|B| - D*B#*A
    dB = _mm_shuffle_pd(dB,dB,0);
    iB1 = _mm_sub_pd(_mm_mul_pd(C1, dB), iB1);
    iB2 = _mm_sub_pd(_mm_mul_pd(C2, dB), iB2);

    d1 = _mm_xor_pd(rd, _Sign_PN);
    d2 = _mm_xor_pd(rd, _Sign_NP);

    //  iC = B*|C| - A*C#*D;
    dC = _mm_shuffle_pd(dC,dC,0);
    iC1 = _mm_sub_pd(_mm_mul_pd(B1, dC), iC1);
    iC2 = _mm_sub_pd(_mm_mul_pd(B2, dC), iC2);

    result.template writePacket<ResultAlignment>( 0, _mm_mul_pd(_mm_shuffle_pd(iA2, iA1, 3), d1));     // iA# / det
    result.template writePacket<ResultAlignment>( 4, _mm_mul_pd(_mm_shuffle_pd(iA2, iA1, 0), d2));
    result.template writePacket<ResultAlignment>( 2, _mm_mul_pd(_mm_shuffle_pd(iB2, iB1, 3), d1));     // iB# / det
    result.template writePacket<ResultAlignment>( 6, _mm_mul_pd(_mm_shuffle_pd(iB2, iB1, 0), d2));
    result.template writePacket<ResultAlignment>( 8, _mm_mul_pd(_mm_shuffle_pd(iC2, iC1, 3), d1));     // iC# / det
    result.template writePacket<ResultAlignment>(12, _mm_mul_pd(_mm_shuffle_pd(iC2, iC1, 0), d2));
    result.template writePacket<ResultAlignment>(10, _mm_mul_pd(_mm_shuffle_pd(iD2, iD1, 3), d1));     // iD# / det
    result.template writePacket<ResultAlignment>(14, _mm_mul_pd(_mm_shuffle_pd(iD2, iD1, 0), d2));
  }
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_INVERSE_SSE_H
