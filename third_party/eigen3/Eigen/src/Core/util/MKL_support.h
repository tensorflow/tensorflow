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
 *   Include file with common MKL declarations
 ********************************************************************************
*/

#ifndef EIGEN_MKL_SUPPORT_H
#define EIGEN_MKL_SUPPORT_H

#ifdef EIGEN_USE_MKL_ALL
  #ifndef EIGEN_USE_BLAS
    #define EIGEN_USE_BLAS
  #endif
  #ifndef EIGEN_USE_LAPACKE
    #define EIGEN_USE_LAPACKE
  #endif
  #ifndef EIGEN_USE_MKL_VML
    #define EIGEN_USE_MKL_VML
  #endif
#endif

#ifdef EIGEN_USE_LAPACKE_STRICT
  #define EIGEN_USE_LAPACKE
#endif

#if defined(EIGEN_USE_BLAS) || defined(EIGEN_USE_LAPACKE) || defined(EIGEN_USE_MKL_VML)
  #define EIGEN_USE_MKL
#endif

#if defined EIGEN_USE_MKL
#   include <mkl.h> 
/*Check IMKL version for compatibility: < 10.3 is not usable with Eigen*/
#   ifndef INTEL_MKL_VERSION
#       undef EIGEN_USE_MKL /* INTEL_MKL_VERSION is not even defined on older versions */
#   elif INTEL_MKL_VERSION < 100305    /* the intel-mkl-103-release-notes say this was when the lapacke.h interface was added*/
#       undef EIGEN_USE_MKL
#   endif
#   ifndef EIGEN_USE_MKL
    /*If the MKL version is too old, undef everything*/
#       undef   EIGEN_USE_MKL_ALL
#       undef   EIGEN_USE_BLAS
#       undef   EIGEN_USE_LAPACKE
#       undef   EIGEN_USE_MKL_VML
#       undef   EIGEN_USE_LAPACKE_STRICT
#       undef   EIGEN_USE_LAPACKE
#   endif
#endif

#if defined EIGEN_USE_MKL
#include <mkl_lapacke.h>
#define EIGEN_MKL_VML_THRESHOLD 128

namespace Eigen {

typedef std::complex<double> dcomplex;
typedef std::complex<float>  scomplex;

namespace internal {

template<typename MKLType, typename EigenType>
static inline void assign_scalar_eig2mkl(MKLType& mklScalar, const EigenType& eigenScalar) {
  mklScalar=eigenScalar;
}

template<typename MKLType, typename EigenType>
static inline void assign_conj_scalar_eig2mkl(MKLType& mklScalar, const EigenType& eigenScalar) {
  mklScalar=eigenScalar;
}

template <>
inline void assign_scalar_eig2mkl<MKL_Complex16,dcomplex>(MKL_Complex16& mklScalar, const dcomplex& eigenScalar) {
  mklScalar.real=eigenScalar.real();
  mklScalar.imag=eigenScalar.imag();
}

template <>
inline void assign_scalar_eig2mkl<MKL_Complex8,scomplex>(MKL_Complex8& mklScalar, const scomplex& eigenScalar) {
  mklScalar.real=eigenScalar.real();
  mklScalar.imag=eigenScalar.imag();
}

template <>
inline void assign_conj_scalar_eig2mkl<MKL_Complex16,dcomplex>(MKL_Complex16& mklScalar, const dcomplex& eigenScalar) {
  mklScalar.real=eigenScalar.real();
  mklScalar.imag=-eigenScalar.imag();
}

template <>
inline void assign_conj_scalar_eig2mkl<MKL_Complex8,scomplex>(MKL_Complex8& mklScalar, const scomplex& eigenScalar) {
  mklScalar.real=eigenScalar.real();
  mklScalar.imag=-eigenScalar.imag();
}

} // end namespace internal

} // end namespace Eigen

#endif

#endif // EIGEN_MKL_SUPPORT_H
