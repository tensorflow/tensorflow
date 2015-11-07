// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BLASUTIL_H
#define EIGEN_BLASUTIL_H

// This file contains many lightweight helper classes used to
// implement and control fast level 2 and level 3 BLAS-like routines.

namespace Eigen {

namespace internal {

// forward declarations
template<typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs=false, bool ConjugateRhs=false>
struct gebp_kernel;

template<typename Scalar, typename Index, typename DataMapper, int nr, int StorageOrder, bool Conjugate = false, bool PanelMode=false>
struct gemm_pack_rhs;

template<typename Scalar, typename Index, typename DataMapper, int Pack1, int Pack2, int StorageOrder, bool Conjugate = false, bool PanelMode = false>
struct gemm_pack_lhs;

template<
  typename Index,
  typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
  typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs,
  int ResStorageOrder>
struct general_matrix_matrix_product;

template<typename Index, typename LhsScalar, typename LhsMapper, int LhsStorageOrder, bool ConjugateLhs, typename RhsScalar, typename RhsMapper, bool ConjugateRhs, int Version=Specialized>
struct general_matrix_vector_product;


template<bool Conjugate> struct conj_if;

template<> struct conj_if<true> {
  template<typename T>
  inline T operator()(const T& x) { return numext::conj(x); }
  template<typename T>
  inline T pconj(const T& x) { return internal::pconj(x); }
};

template<> struct conj_if<false> {
  template<typename T>
  inline const T& operator()(const T& x) { return x; }
  template<typename T>
  inline const T& pconj(const T& x) { return x; }
};

template<typename Scalar> struct conj_helper<Scalar,Scalar,false,false>
{
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar pmadd(const Scalar& x, const Scalar& y, const Scalar& c) const { return internal::pmadd(x,y,c); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar pmul(const Scalar& x, const Scalar& y) const { return internal::pmul(x,y); }
};

template<typename RealScalar> struct conj_helper<std::complex<RealScalar>, std::complex<RealScalar>, false,true>
{
  typedef std::complex<RealScalar> Scalar;
  EIGEN_STRONG_INLINE Scalar pmadd(const Scalar& x, const Scalar& y, const Scalar& c) const
  { return c + pmul(x,y); }

  EIGEN_STRONG_INLINE Scalar pmul(const Scalar& x, const Scalar& y) const
  { return Scalar(numext::real(x)*numext::real(y) + numext::imag(x)*numext::imag(y), numext::imag(x)*numext::real(y) - numext::real(x)*numext::imag(y)); }
};

template<typename RealScalar> struct conj_helper<std::complex<RealScalar>, std::complex<RealScalar>, true,false>
{
  typedef std::complex<RealScalar> Scalar;
  EIGEN_STRONG_INLINE Scalar pmadd(const Scalar& x, const Scalar& y, const Scalar& c) const
  { return c + pmul(x,y); }

  EIGEN_STRONG_INLINE Scalar pmul(const Scalar& x, const Scalar& y) const
  { return Scalar(numext::real(x)*numext::real(y) + numext::imag(x)*numext::imag(y), numext::real(x)*numext::imag(y) - numext::imag(x)*numext::real(y)); }
};

template<typename RealScalar> struct conj_helper<std::complex<RealScalar>, std::complex<RealScalar>, true,true>
{
  typedef std::complex<RealScalar> Scalar;
  EIGEN_STRONG_INLINE Scalar pmadd(const Scalar& x, const Scalar& y, const Scalar& c) const
  { return c + pmul(x,y); }

  EIGEN_STRONG_INLINE Scalar pmul(const Scalar& x, const Scalar& y) const
  { return Scalar(numext::real(x)*numext::real(y) - numext::imag(x)*numext::imag(y), - numext::real(x)*numext::imag(y) - numext::imag(x)*numext::real(y)); }
};

template<typename RealScalar,bool Conj> struct conj_helper<std::complex<RealScalar>, RealScalar, Conj,false>
{
  typedef std::complex<RealScalar> Scalar;
  EIGEN_STRONG_INLINE Scalar pmadd(const Scalar& x, const RealScalar& y, const Scalar& c) const
  { return padd(c, pmul(x,y)); }
  EIGEN_STRONG_INLINE Scalar pmul(const Scalar& x, const RealScalar& y) const
  { return conj_if<Conj>()(x)*y; }
};

template<typename RealScalar,bool Conj> struct conj_helper<RealScalar, std::complex<RealScalar>, false,Conj>
{
  typedef std::complex<RealScalar> Scalar;
  EIGEN_STRONG_INLINE Scalar pmadd(const RealScalar& x, const Scalar& y, const Scalar& c) const
  { return padd(c, pmul(x,y)); }
  EIGEN_STRONG_INLINE Scalar pmul(const RealScalar& x, const Scalar& y) const
  { return x*conj_if<Conj>()(y); }
};

template<typename From,typename To> struct get_factor {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE To run(const From& x) { return x; }
};

template<typename Scalar> struct get_factor<Scalar,typename NumTraits<Scalar>::Real> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE typename NumTraits<Scalar>::Real run(const Scalar& x) { return numext::real(x); }
};


/* Helper class to analyze the factors of a Product expression.
 * In particular it allows to pop out operator-, scalar multiples,
 * and conjugate */
template<typename XprType> struct blas_traits
{
  typedef typename traits<XprType>::Scalar Scalar;
  typedef const XprType& ExtractType;
  typedef XprType _ExtractType;
  enum {
    IsComplex = NumTraits<Scalar>::IsComplex,
    IsTransposed = false,
    NeedToConjugate = false,
    HasUsableDirectAccess = (    (int(XprType::Flags)&DirectAccessBit)
                              && (   bool(XprType::IsVectorAtCompileTime)
                                  || int(inner_stride_at_compile_time<XprType>::ret) == 1)
                             ) ?  1 : 0
  };
  typedef typename conditional<bool(HasUsableDirectAccess),
    ExtractType,
    typename _ExtractType::PlainObject
    >::type DirectLinearAccessType;
  static inline ExtractType extract(const XprType& x) { return x; }
  static inline const Scalar extractScalarFactor(const XprType&) { return Scalar(1); }
};

// pop conjugate
template<typename Scalar, typename NestedXpr>
struct blas_traits<CwiseUnaryOp<scalar_conjugate_op<Scalar>, NestedXpr> >
 : blas_traits<NestedXpr>
{
  typedef blas_traits<NestedXpr> Base;
  typedef CwiseUnaryOp<scalar_conjugate_op<Scalar>, NestedXpr> XprType;
  typedef typename Base::ExtractType ExtractType;

  enum {
    IsComplex = NumTraits<Scalar>::IsComplex,
    NeedToConjugate = Base::NeedToConjugate ? 0 : IsComplex
  };
  static inline ExtractType extract(const XprType& x) { return Base::extract(x.nestedExpression()); }
  static inline Scalar extractScalarFactor(const XprType& x) { return conj(Base::extractScalarFactor(x.nestedExpression())); }
};

// pop scalar multiple
template<typename Scalar, typename NestedXpr>
struct blas_traits<CwiseUnaryOp<scalar_multiple_op<Scalar>, NestedXpr> >
 : blas_traits<NestedXpr>
{
  typedef blas_traits<NestedXpr> Base;
  typedef CwiseUnaryOp<scalar_multiple_op<Scalar>, NestedXpr> XprType;
  typedef typename Base::ExtractType ExtractType;
  static inline ExtractType extract(const XprType& x) { return Base::extract(x.nestedExpression()); }
  static inline Scalar extractScalarFactor(const XprType& x)
  { return x.functor().m_other * Base::extractScalarFactor(x.nestedExpression()); }
};

// pop opposite
template<typename Scalar, typename NestedXpr>
struct blas_traits<CwiseUnaryOp<scalar_opposite_op<Scalar>, NestedXpr> >
 : blas_traits<NestedXpr>
{
  typedef blas_traits<NestedXpr> Base;
  typedef CwiseUnaryOp<scalar_opposite_op<Scalar>, NestedXpr> XprType;
  typedef typename Base::ExtractType ExtractType;
  static inline ExtractType extract(const XprType& x) { return Base::extract(x.nestedExpression()); }
  static inline Scalar extractScalarFactor(const XprType& x)
  { return - Base::extractScalarFactor(x.nestedExpression()); }
};

// pop/push transpose
template<typename NestedXpr>
struct blas_traits<Transpose<NestedXpr> >
 : blas_traits<NestedXpr>
{
  typedef typename NestedXpr::Scalar Scalar;
  typedef blas_traits<NestedXpr> Base;
  typedef Transpose<NestedXpr> XprType;
  typedef Transpose<const typename Base::_ExtractType>  ExtractType; // const to get rid of a compile error; anyway blas traits are only used on the RHS
  typedef Transpose<const typename Base::_ExtractType> _ExtractType;
  typedef typename conditional<bool(Base::HasUsableDirectAccess),
    ExtractType,
    typename ExtractType::PlainObject
    >::type DirectLinearAccessType;
  enum {
    IsTransposed = Base::IsTransposed ? 0 : 1
  };
  static inline ExtractType extract(const XprType& x) { return Base::extract(x.nestedExpression()); }
  static inline Scalar extractScalarFactor(const XprType& x) { return Base::extractScalarFactor(x.nestedExpression()); }
};

template<typename T>
struct blas_traits<const T>
     : blas_traits<T>
{};

template<typename T, bool HasUsableDirectAccess=blas_traits<T>::HasUsableDirectAccess>
struct extract_data_selector {
  static const typename T::Scalar* run(const T& m)
  {
    return blas_traits<T>::extract(m).data();
  }
};

template<typename T>
struct extract_data_selector<T,false> {
  static typename T::Scalar* run(const T&) { return 0; }
};

template<typename T> const typename T::Scalar* extract_data(const T& m)
{
  return extract_data_selector<T>::run(m);
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_BLASUTIL_H
