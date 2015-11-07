// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_FUNCTORS_H
#define EIGEN_FUNCTORS_H

namespace Eigen {

namespace internal {

// associative functors:

/** \internal
  * \brief Template functor to compute the sum of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator+, class VectorwiseOp, MatrixBase::sum()
  */
template<typename Scalar> struct scalar_sum_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sum_op)
  EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& a, const Scalar& b) const { return a + b; }
  template<typename Packet>
  EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const
  { return internal::padd(a,b); }
  template<typename Packet>
  EIGEN_STRONG_INLINE const Scalar predux(const Packet& a) const
  { return internal::predux(a); }
};
template<typename Scalar>
struct functor_traits<scalar_sum_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasAdd
  };
};

/** \internal
  * \brief Template functor to compute the product of two scalars
  *
  * \sa class CwiseBinaryOp, Cwise::operator*(), class VectorwiseOp, MatrixBase::redux()
  */
template<typename LhsScalar,typename RhsScalar> struct scalar_product_op {
  enum {
    // TODO vectorize mixed product
    Vectorizable = is_same<LhsScalar,RhsScalar>::value && packet_traits<LhsScalar>::HasMul && packet_traits<RhsScalar>::HasMul
  };
  typedef typename scalar_product_traits<LhsScalar,RhsScalar>::ReturnType result_type;
  EIGEN_EMPTY_STRUCT_CTOR(scalar_product_op)
  EIGEN_STRONG_INLINE const result_type operator() (const LhsScalar& a, const RhsScalar& b) const { return a * b; }
  template<typename Packet>
  EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const
  { return internal::pmul(a,b); }
  template<typename Packet>
  EIGEN_STRONG_INLINE const result_type predux(const Packet& a) const
  { return internal::predux_mul(a); }
};
template<typename LhsScalar,typename RhsScalar>
struct functor_traits<scalar_product_op<LhsScalar,RhsScalar> > {
  enum {
    Cost = (NumTraits<LhsScalar>::MulCost + NumTraits<RhsScalar>::MulCost)/2, // rough estimate!
    PacketAccess = scalar_product_op<LhsScalar,RhsScalar>::Vectorizable
  };
};

/** \internal
  * \brief Template functor to compute the conjugate product of two scalars
  *
  * This is a short cut for conj(x) * y which is needed for optimization purpose; in Eigen2 support mode, this becomes x * conj(y)
  */
template<typename LhsScalar,typename RhsScalar> struct scalar_conj_product_op {

  enum {
    Conj = NumTraits<LhsScalar>::IsComplex
  };

  typedef typename scalar_product_traits<LhsScalar,RhsScalar>::ReturnType result_type;

  EIGEN_EMPTY_STRUCT_CTOR(scalar_conj_product_op)
  EIGEN_STRONG_INLINE const result_type operator() (const LhsScalar& a, const RhsScalar& b) const
  { return conj_helper<LhsScalar,RhsScalar,Conj,false>().pmul(a,b); }

  template<typename Packet>
  EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const
  { return conj_helper<Packet,Packet,Conj,false>().pmul(a,b); }
};
template<typename LhsScalar,typename RhsScalar>
struct functor_traits<scalar_conj_product_op<LhsScalar,RhsScalar> > {
  enum {
    Cost = NumTraits<LhsScalar>::MulCost,
    PacketAccess = internal::is_same<LhsScalar, RhsScalar>::value && packet_traits<LhsScalar>::HasMul
  };
};

/** \internal
  * \brief Template functor to compute the min of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::cwiseMin, class VectorwiseOp, MatrixBase::minCoeff()
  */
template<typename Scalar> struct scalar_min_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_min_op)
  EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& a, const Scalar& b) const { using std::min; return (min)(a, b); }
  template<typename Packet>
  EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const
  { return internal::pmin(a,b); }
  template<typename Packet>
  EIGEN_STRONG_INLINE const Scalar predux(const Packet& a) const
  { return internal::predux_min(a); }
};
template<typename Scalar>
struct functor_traits<scalar_min_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasMin
  };
};

/** \internal
  * \brief Template functor to compute the max of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::cwiseMax, class VectorwiseOp, MatrixBase::maxCoeff()
  */
template<typename Scalar> struct scalar_max_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_max_op)
  EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& a, const Scalar& b) const { using std::max; return (max)(a, b); }
  template<typename Packet>
  EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const
  { return internal::pmax(a,b); }
  template<typename Packet>
  EIGEN_STRONG_INLINE const Scalar predux(const Packet& a) const
  { return internal::predux_max(a); }
};
template<typename Scalar>
struct functor_traits<scalar_max_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasMax
  };
};

/** \internal
  * \brief Template functor to compute the hypot of two scalars
  *
  * \sa MatrixBase::stableNorm(), class Redux
  */
template<typename Scalar> struct scalar_hypot_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_hypot_op)
//   typedef typename NumTraits<Scalar>::Real result_type;
  EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x, const Scalar& _y) const
  {
    using std::max;
    using std::min;
    using std::sqrt;
    Scalar p = (max)(_x, _y);
    Scalar q = (min)(_x, _y);
    Scalar qp = q/p;
    return p * sqrt(Scalar(1) + qp*qp);
  }
};
template<typename Scalar>
struct functor_traits<scalar_hypot_op<Scalar> > {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess=0 };
};

/** \internal
  * \brief Template functor to compute the pow of two scalars
  */
template<typename Scalar, typename OtherScalar> struct scalar_binary_pow_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_binary_pow_op)
  inline Scalar operator() (const Scalar& a, const OtherScalar& b) const { return numext::pow(a, b); }
};
template<typename Scalar, typename OtherScalar>
struct functor_traits<scalar_binary_pow_op<Scalar,OtherScalar> > {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = false };
};

// other binary functors:

/** \internal
  * \brief Template functor to compute the difference of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator-
  */
template<typename Scalar> struct scalar_difference_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_difference_op)
  EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& a, const Scalar& b) const { return a - b; }
  template<typename Packet>
  EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const
  { return internal::psub(a,b); }
};
template<typename Scalar>
struct functor_traits<scalar_difference_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasSub
  };
};

/** \internal
  * \brief Template functor to compute the quotient of two scalars
  *
  * \sa class CwiseBinaryOp, Cwise::operator/()
  */
template<typename LhsScalar,typename RhsScalar> struct scalar_quotient_op {
  enum {
    // TODO vectorize mixed product
    Vectorizable = is_same<LhsScalar,RhsScalar>::value && packet_traits<LhsScalar>::HasDiv && packet_traits<RhsScalar>::HasDiv
  };
  typedef typename scalar_product_traits<LhsScalar,RhsScalar>::ReturnType result_type;
  EIGEN_EMPTY_STRUCT_CTOR(scalar_quotient_op)
  EIGEN_STRONG_INLINE const result_type operator() (const LhsScalar& a, const RhsScalar& b) const { return a / b; }
  template<typename Packet>
  EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const
  { return internal::pdiv(a,b); }
};
template<typename LhsScalar,typename RhsScalar>
struct functor_traits<scalar_quotient_op<LhsScalar,RhsScalar> > {
  enum {
    Cost = (NumTraits<LhsScalar>::MulCost + NumTraits<RhsScalar>::MulCost), // rough estimate!
    PacketAccess = scalar_quotient_op<LhsScalar,RhsScalar>::Vectorizable
  };
};



/** \internal
  * \brief Template functor to compute the and of two booleans
  *
  * \sa class CwiseBinaryOp, ArrayBase::operator&&
  */
struct scalar_boolean_and_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_boolean_and_op)
  EIGEN_STRONG_INLINE bool operator() (const bool& a, const bool& b) const { return a && b; }
};
template<> struct functor_traits<scalar_boolean_and_op> {
  enum {
    Cost = NumTraits<bool>::AddCost,
    PacketAccess = false
  };
};

/** \internal
  * \brief Template functor to compute the or of two booleans
  *
  * \sa class CwiseBinaryOp, ArrayBase::operator||
  */
struct scalar_boolean_or_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_boolean_or_op)
  EIGEN_STRONG_INLINE bool operator() (const bool& a, const bool& b) const { return a || b; }
};
template<> struct functor_traits<scalar_boolean_or_op> {
  enum {
    Cost = NumTraits<bool>::AddCost,
    PacketAccess = false
  };
};

/** \internal
  * \brief Template functor to compute the xor of two booleans
  *
  * \sa class CwiseBinaryOp, ArrayBase::operator^
  */
struct scalar_boolean_xor_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_boolean_xor_op)
  EIGEN_STRONG_INLINE bool operator() (const bool& a, const bool& b) const { return a ^ b; }
};
template<> struct functor_traits<scalar_boolean_xor_op> {
  enum {
    Cost = NumTraits<bool>::AddCost,
    PacketAccess = false
  };
};

// unary functors:

/** \internal
  * \brief Template functor to compute the opposite of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::operator-
  */
template<typename Scalar> struct scalar_opposite_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_opposite_op)
  EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& a) const { return -a; }
  template<typename Packet>
  EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const
  { return internal::pnegate(a); }
};
template<typename Scalar>
struct functor_traits<scalar_opposite_op<Scalar> >
{ enum {
    Cost = NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasNegate };
};

/** \internal
  * \brief Template functor to compute the absolute value of a scalar
  *
  * \sa class CwiseUnaryOp, Cwise::abs
  */
template<typename Scalar> struct scalar_abs_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_abs_op)
  typedef typename NumTraits<Scalar>::Real result_type;
  EIGEN_STRONG_INLINE const result_type operator() (const Scalar& a) const { using std::abs; return abs(a); }
  template<typename Packet>
  EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const
  { return internal::pabs(a); }
};
template<typename Scalar>
struct functor_traits<scalar_abs_op<Scalar> >
{
  enum {
    Cost = NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasAbs
  };
};

/** \internal
  * \brief Template functor to compute the squared absolute value of a scalar
  *
  * \sa class CwiseUnaryOp, Cwise::abs2
  */
template<typename Scalar> struct scalar_abs2_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_abs2_op)
  typedef typename NumTraits<Scalar>::Real result_type;
  EIGEN_STRONG_INLINE const result_type operator() (const Scalar& a) const { return numext::abs2(a); }
  template<typename Packet>
  EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const
  { return internal::pmul(a,a); }
};
template<typename Scalar>
struct functor_traits<scalar_abs2_op<Scalar> >
{ enum { Cost = NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasAbs2 }; };

/** \internal
  * \brief Template functor to compute the conjugate of a complex value
  *
  * \sa class CwiseUnaryOp, MatrixBase::conjugate()
  */
template<typename Scalar> struct scalar_conjugate_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_conjugate_op)
  EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& a) const { using numext::conj; return conj(a); }
  template<typename Packet>
  EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const { return internal::pconj(a); }
};
template<typename Scalar>
struct functor_traits<scalar_conjugate_op<Scalar> >
{
  enum {
    Cost = NumTraits<Scalar>::IsComplex ? NumTraits<Scalar>::AddCost : 0,
    PacketAccess = packet_traits<Scalar>::HasConj
  };
};

/** \internal
  * \brief Template functor to cast a scalar to another type
  *
  * \sa class CwiseUnaryOp, MatrixBase::cast()
  */
template<typename Scalar, typename NewType>
struct scalar_cast_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
  typedef NewType result_type;
  EIGEN_STRONG_INLINE const NewType operator() (const Scalar& a) const { return cast<Scalar, NewType>(a); }
};
template<typename Scalar, typename NewType>
struct functor_traits<scalar_cast_op<Scalar,NewType> >
{ enum { Cost = is_same<Scalar, NewType>::value ? 0 : NumTraits<NewType>::AddCost, PacketAccess = false }; };

/** \internal
  * \brief Template functor to convert a scalar to another type using a custom functor.
  *
  * \sa class CwiseUnaryOp, MatrixBase::convert()
  */
template<typename Scalar, typename NewType, typename ConvertOp>
struct scalar_convert_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_convert_op)
  typedef NewType result_type;
  EIGEN_STRONG_INLINE const NewType operator() (const Scalar& a) const { return ConvertOp()(a); }
};
template<typename Scalar, typename NewType, typename ConvertOp>
struct functor_traits<scalar_convert_op<Scalar,NewType,ConvertOp> >
{ enum { Cost = is_same<Scalar, NewType>::value ? 0 : NumTraits<NewType>::AddCost, PacketAccess = false }; };

/** \internal
  * \brief Template functor to extract the real part of a complex
  *
  * \sa class CwiseUnaryOp, MatrixBase::real()
  */
template<typename Scalar>
struct scalar_real_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_real_op)
  typedef typename NumTraits<Scalar>::Real result_type;
  EIGEN_STRONG_INLINE result_type operator() (const Scalar& a) const { return numext::real(a); }
};
template<typename Scalar>
struct functor_traits<scalar_real_op<Scalar> >
{ enum { Cost = 0, PacketAccess = false }; };

/** \internal
  * \brief Template functor to extract the imaginary part of a complex
  *
  * \sa class CwiseUnaryOp, MatrixBase::imag()
  */
template<typename Scalar>
struct scalar_imag_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_imag_op)
  typedef typename NumTraits<Scalar>::Real result_type;
  EIGEN_STRONG_INLINE result_type operator() (const Scalar& a) const { return numext::imag(a); }
};
template<typename Scalar>
struct functor_traits<scalar_imag_op<Scalar> >
{ enum { Cost = 0, PacketAccess = false }; };

/** \internal
  * \brief Template functor to extract the real part of a complex as a reference
  *
  * \sa class CwiseUnaryOp, MatrixBase::real()
  */
template<typename Scalar>
struct scalar_real_ref_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_real_ref_op)
  typedef typename NumTraits<Scalar>::Real result_type;
  EIGEN_STRONG_INLINE result_type& operator() (const Scalar& a) const { return numext::real_ref(*const_cast<Scalar*>(&a)); }
};
template<typename Scalar>
struct functor_traits<scalar_real_ref_op<Scalar> >
{ enum { Cost = 0, PacketAccess = false }; };

/** \internal
  * \brief Template functor to extract the imaginary part of a complex as a reference
  *
  * \sa class CwiseUnaryOp, MatrixBase::imag()
  */
template<typename Scalar>
struct scalar_imag_ref_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_imag_ref_op)
  typedef typename NumTraits<Scalar>::Real result_type;
  EIGEN_STRONG_INLINE result_type& operator() (const Scalar& a) const { return numext::imag_ref(*const_cast<Scalar*>(&a)); }
};
template<typename Scalar>
struct functor_traits<scalar_imag_ref_op<Scalar> >
{ enum { Cost = 0, PacketAccess = false }; };

/** \internal
  *
  * \brief Template functor to compute the exponential of a scalar
  *
  * \sa class CwiseUnaryOp, Cwise::exp()
  */
template<typename Scalar> struct scalar_exp_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_exp_op)
  inline const Scalar operator() (const Scalar& a) const { using std::exp; return exp(a); }
  typedef typename packet_traits<Scalar>::type Packet;
  inline Packet packetOp(const Packet& a) const { return internal::pexp(a); }
};
template<typename Scalar>
struct functor_traits<scalar_exp_op<Scalar> >
{ enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasExp }; };

/** \internal
  *
  * \brief Template functor to compute the logarithm of a scalar
  *
  * \sa class CwiseUnaryOp, Cwise::log()
  */
template<typename Scalar> struct scalar_log_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_log_op)
  inline const Scalar operator() (const Scalar& a) const { using std::log; return log(a); }
  typedef typename packet_traits<Scalar>::type Packet;
  inline Packet packetOp(const Packet& a) const { return internal::plog(a); }
};
template<typename Scalar>
struct functor_traits<scalar_log_op<Scalar> >
{ enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasLog }; };

/** \internal
  * \brief Template functor to multiply a scalar by a fixed other one
  *
  * \sa class CwiseUnaryOp, MatrixBase::operator*, MatrixBase::operator/
  */
/* NOTE why doing the pset1() in packetOp *is* an optimization ?
 * indeed it seems better to declare m_other as a Packet and do the pset1() once
 * in the constructor. However, in practice:
 *  - GCC does not like m_other as a Packet and generate a load every time it needs it
 *  - on the other hand GCC is able to moves the pset1() outside the loop :)
 *  - simpler code ;)
 * (ICC and gcc 4.4 seems to perform well in both cases, the issue is visible with y = a*x + b*y)
 */
template<typename Scalar>
struct scalar_multiple_op {
  typedef typename packet_traits<Scalar>::type Packet;
  // FIXME default copy constructors seems bugged with std::complex<>
  EIGEN_STRONG_INLINE scalar_multiple_op(const scalar_multiple_op& other) : m_other(other.m_other) { }
  EIGEN_STRONG_INLINE scalar_multiple_op(const Scalar& other) : m_other(other) { }
  EIGEN_STRONG_INLINE Scalar operator() (const Scalar& a) const { return a * m_other; }
  EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const
  { return internal::pmul(a, pset1<Packet>(m_other)); }
  typename add_const_on_value_type<typename NumTraits<Scalar>::Nested>::type m_other;
};
template<typename Scalar>
struct functor_traits<scalar_multiple_op<Scalar> >
{ enum { Cost = NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasMul }; };

template<typename Scalar1, typename Scalar2>
struct scalar_multiple2_op {
  typedef typename packet_traits<Scalar1>::type Packet1;
  typedef typename scalar_product_traits<Scalar1,Scalar2>::ReturnType result_type;
  typedef typename packet_traits<result_type>::type packet_result_type;
  EIGEN_STRONG_INLINE scalar_multiple2_op(const scalar_multiple2_op& other) : m_other(other.m_other) { }
  EIGEN_STRONG_INLINE scalar_multiple2_op(const Scalar2& other) : m_other(other) { }
  EIGEN_STRONG_INLINE result_type operator() (const Scalar1& a) const { return a * m_other; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const packet_result_type packetOp(const Packet1& a) const
  { eigen_assert("packetOp is not defined"); }
  typename add_const_on_value_type<typename NumTraits<Scalar2>::Nested>::type m_other;
};
template<typename Scalar1,typename Scalar2>
struct functor_traits<scalar_multiple2_op<Scalar1,Scalar2> >
{ enum { Cost = NumTraits<Scalar1>::MulCost, PacketAccess = false }; };

/** \internal
  * \brief Template functor to divide a scalar by a fixed other one
  *
  * This functor is used to implement the quotient of a matrix by
  * a scalar where the scalar type is not necessarily a floating point type.
  *
  * \sa class CwiseUnaryOp, MatrixBase::operator/
  */
template<typename Scalar>
struct scalar_quotient1_op {
  typedef typename packet_traits<Scalar>::type Packet;
  // FIXME default copy constructors seems bugged with std::complex<>
  EIGEN_STRONG_INLINE scalar_quotient1_op(const scalar_quotient1_op& other) : m_other(other.m_other) { }
  EIGEN_STRONG_INLINE scalar_quotient1_op(const Scalar& other) : m_other(other) {}
  EIGEN_STRONG_INLINE Scalar operator() (const Scalar& a) const { return a / m_other; }
  EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const
  { return internal::pdiv(a, pset1<Packet>(m_other)); }
  typename add_const_on_value_type<typename NumTraits<Scalar>::Nested>::type m_other;
};
template<typename Scalar>
struct functor_traits<scalar_quotient1_op<Scalar> >
{ enum { Cost = 2 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasDiv }; };

// nullary functors

template<typename Scalar>
struct scalar_constant_op {
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE scalar_constant_op(const scalar_constant_op& other) : m_other(other.m_other) { }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE scalar_constant_op(const Scalar& other) : m_other(other) { }
  template<typename Index>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (Index, Index = 0) const { return m_other; }
  template<typename Index>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(Index, Index = 0) const { return internal::pset1<Packet>(m_other); }
  const Scalar m_other;
};
template<typename Scalar>
struct functor_traits<scalar_constant_op<Scalar> >
// FIXME replace this packet test by a safe one
{ enum { Cost = 1, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = true }; };

template<typename Scalar> struct scalar_identity_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_identity_op)
  template<typename Index>
  EIGEN_STRONG_INLINE const Scalar operator() (Index row, Index col) const { return row==col ? Scalar(1) : Scalar(0); }
};
template<typename Scalar>
struct functor_traits<scalar_identity_op<Scalar> >
{ enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = false, IsRepeatable = true }; };

template <typename Scalar, bool RandomAccess> struct linspaced_op_impl;

// linear access for packet ops:
// 1) initialization
//   base = [low, ..., low] + ([step, ..., step] * [-size, ..., 0])
// 2) each step (where size is 1 for coeff access or PacketSize for packet access)
//   base += [size*step, ..., size*step]
//
// TODO: Perhaps it's better to initialize lazily (so not in the constructor but in packetOp)
//       in order to avoid the padd() in operator() ?
template <typename Scalar>
struct linspaced_op_impl<Scalar,false>
{
  typedef typename packet_traits<Scalar>::type Packet;

  linspaced_op_impl(const Scalar& low, const Scalar& step) :
  m_low(low), m_step(step),
  m_packetStep(pset1<Packet>(packet_traits<Scalar>::size*step)),
  m_base(padd(pset1<Packet>(low), pmul(pset1<Packet>(step),plset<Scalar>(-packet_traits<Scalar>::size)))) {}

  template<typename Index>
  EIGEN_STRONG_INLINE const Scalar operator() (Index i) const
  {
    m_base = padd(m_base, pset1<Packet>(m_step));
    return m_low+Scalar(i)*m_step;
  }

  template<typename Index>
  EIGEN_STRONG_INLINE const Packet packetOp(Index) const { return m_base = padd(m_base,m_packetStep); }

  const Scalar m_low;
  const Scalar m_step;
  const Packet m_packetStep;
  mutable Packet m_base;
};

// random access for packet ops:
// 1) each step
//   [low, ..., low] + ( [step, ..., step] * ( [i, ..., i] + [0, ..., size] ) )
template <typename Scalar>
struct linspaced_op_impl<Scalar,true>
{
  typedef typename packet_traits<Scalar>::type Packet;

  linspaced_op_impl(const Scalar& low, const Scalar& step) :
  m_low(low), m_step(step),
  m_lowPacket(pset1<Packet>(m_low)), m_stepPacket(pset1<Packet>(m_step)), m_interPacket(plset<Scalar>(0)) {}

  template<typename Index>
  EIGEN_STRONG_INLINE const Scalar operator() (Index i) const { return m_low+i*m_step; }

  template<typename Index>
  EIGEN_STRONG_INLINE const Packet packetOp(Index i) const
  { return internal::padd(m_lowPacket, pmul(m_stepPacket, padd(pset1<Packet>(i),m_interPacket))); }

  const Scalar m_low;
  const Scalar m_step;
  const Packet m_lowPacket;
  const Packet m_stepPacket;
  const Packet m_interPacket;
};

// ----- Linspace functor ----------------------------------------------------------------

// Forward declaration (we default to random access which does not really give
// us a speed gain when using packet access but it allows to use the functor in
// nested expressions).
template <typename Scalar, bool RandomAccess = true> struct linspaced_op;
template <typename Scalar, bool RandomAccess> struct functor_traits< linspaced_op<Scalar,RandomAccess> >
{ enum { Cost = 1, PacketAccess = packet_traits<Scalar>::HasSetLinear, IsRepeatable = true }; };
template <typename Scalar, bool RandomAccess> struct linspaced_op
{
  typedef typename packet_traits<Scalar>::type Packet;
  linspaced_op(const Scalar& low, const Scalar& high, DenseIndex num_steps) : impl((num_steps==1 ? high : low), (num_steps==1 ? Scalar() : (high-low)/(num_steps-1))) {}

  template<typename Index>
  EIGEN_STRONG_INLINE const Scalar operator() (Index i) const { return impl(i); }

  // We need this function when assigning e.g. a RowVectorXd to a MatrixXd since
  // there row==0 and col is used for the actual iteration.
  template<typename Index>
  EIGEN_STRONG_INLINE const Scalar operator() (Index row, Index col) const
  {
    eigen_assert(col==0 || row==0);
    return impl(col + row);
  }

  template<typename Index>
  EIGEN_STRONG_INLINE const Packet packetOp(Index i) const { return impl.packetOp(i); }

  // We need this function when assigning e.g. a RowVectorXd to a MatrixXd since
  // there row==0 and col is used for the actual iteration.
  template<typename Index>
  EIGEN_STRONG_INLINE const Packet packetOp(Index row, Index col) const
  {
    eigen_assert(col==0 || row==0);
    return impl.packetOp(col + row);
  }

  // This proxy object handles the actual required temporaries, the different
  // implementations (random vs. sequential access) as well as the
  // correct piping to size 2/4 packet operations.
  const linspaced_op_impl<Scalar,RandomAccess> impl;
};

// all functors allow linear access, except scalar_identity_op. So we fix here a quick meta
// to indicate whether a functor allows linear access, just always answering 'yes' except for
// scalar_identity_op.
// FIXME move this to functor_traits adding a functor_default
template<typename Functor> struct functor_has_linear_access { enum { ret = 1 }; };
template<typename Scalar> struct functor_has_linear_access<scalar_identity_op<Scalar> > { enum { ret = 0 }; };

// In Eigen, any binary op (Product, CwiseBinaryOp) require the Lhs and Rhs to have the same scalar type, except for multiplication
// where the mixing of different types is handled by scalar_product_traits
// In particular, real * complex<real> is allowed.
// FIXME move this to functor_traits adding a functor_default
template<typename Functor> struct functor_is_product_like { enum { ret = 0 }; };
template<typename LhsScalar,typename RhsScalar> struct functor_is_product_like<scalar_product_op<LhsScalar,RhsScalar> > { enum { ret = 1 }; };
template<typename LhsScalar,typename RhsScalar> struct functor_is_product_like<scalar_conj_product_op<LhsScalar,RhsScalar> > { enum { ret = 1 }; };
template<typename LhsScalar,typename RhsScalar> struct functor_is_product_like<scalar_quotient_op<LhsScalar,RhsScalar> > { enum { ret = 1 }; };


/** \internal
  * \brief Template functor to add a scalar to a fixed other one
  * \sa class CwiseUnaryOp, Array::operator+
  */
/* If you wonder why doing the pset1() in packetOp() is an optimization check scalar_multiple_op */
template<typename Scalar>
struct scalar_add_op {
  typedef typename packet_traits<Scalar>::type Packet;
  // FIXME default copy constructors seems bugged with std::complex<>
  inline scalar_add_op(const scalar_add_op& other) : m_other(other.m_other) { }
  inline scalar_add_op(const Scalar& other) : m_other(other) { }
  inline Scalar operator() (const Scalar& a) const { return a + m_other; }
  inline const Packet packetOp(const Packet& a) const
  { return internal::padd(a, pset1<Packet>(m_other)); }
  const Scalar m_other;
};
template<typename Scalar>
struct functor_traits<scalar_add_op<Scalar> >
{ enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = packet_traits<Scalar>::HasAdd }; };

/** \internal
  * \brief Template functor to compute the square root of a scalar
  * \sa class CwiseUnaryOp, Cwise::sqrt()
  */
template<typename Scalar> struct scalar_sqrt_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sqrt_op)
  inline const Scalar operator() (const Scalar& a) const { using std::sqrt; return sqrt(a); }
  typedef typename packet_traits<Scalar>::type Packet;
  inline Packet packetOp(const Packet& a) const { return internal::psqrt(a); }
};
template<typename Scalar>
struct functor_traits<scalar_sqrt_op<Scalar> >
{ enum {
    Cost = 5 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasSqrt
  };
};

/** \internal
  * \brief Template functor to compute the cosine of a scalar
  * \sa class CwiseUnaryOp, ArrayBase::cos()
  */
template<typename Scalar> struct scalar_cos_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_cos_op)
  inline Scalar operator() (const Scalar& a) const { using std::cos; return cos(a); }
  typedef typename packet_traits<Scalar>::type Packet;
  inline Packet packetOp(const Packet& a) const { return internal::pcos(a); }
};
template<typename Scalar>
struct functor_traits<scalar_cos_op<Scalar> >
{
  enum {
    Cost = 5 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasCos
  };
};

/** \internal
  * \brief Template functor to compute the sine of a scalar
  * \sa class CwiseUnaryOp, ArrayBase::sin()
  */
template<typename Scalar> struct scalar_sin_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sin_op)
  inline const Scalar operator() (const Scalar& a) const { using std::sin; return sin(a); }
  typedef typename packet_traits<Scalar>::type Packet;
  inline Packet packetOp(const Packet& a) const { return internal::psin(a); }
};
template<typename Scalar>
struct functor_traits<scalar_sin_op<Scalar> >
{
  enum {
    Cost = 5 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasSin
  };
};


/** \internal
  * \brief Template functor to compute the tan of a scalar
  * \sa class CwiseUnaryOp, ArrayBase::tan()
  */
template<typename Scalar> struct scalar_tan_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_tan_op)
  inline const Scalar operator() (const Scalar& a) const { using std::tan; return tan(a); }
  typedef typename packet_traits<Scalar>::type Packet;
  inline Packet packetOp(const Packet& a) const { return internal::ptan(a); }
};
template<typename Scalar>
struct functor_traits<scalar_tan_op<Scalar> >
{
  enum {
    Cost = 5 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasTan
  };
};

/** \internal
  * \brief Template functor to compute the arc cosine of a scalar
  * \sa class CwiseUnaryOp, ArrayBase::acos()
  */
template<typename Scalar> struct scalar_acos_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_acos_op)
  inline const Scalar operator() (const Scalar& a) const { using std::acos; return acos(a); }
  typedef typename packet_traits<Scalar>::type Packet;
  inline Packet packetOp(const Packet& a) const { return internal::pacos(a); }
};
template<typename Scalar>
struct functor_traits<scalar_acos_op<Scalar> >
{
  enum {
    Cost = 5 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasACos
  };
};

/** \internal
  * \brief Template functor to compute the arc sine of a scalar
  * \sa class CwiseUnaryOp, ArrayBase::asin()
  */
template<typename Scalar> struct scalar_asin_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_asin_op)
  inline const Scalar operator() (const Scalar& a) const { using std::asin; return asin(a); }
  typedef typename packet_traits<Scalar>::type Packet;
  inline Packet packetOp(const Packet& a) const { return internal::pasin(a); }
};
template<typename Scalar>
struct functor_traits<scalar_asin_op<Scalar> >
{
  enum {
    Cost = 5 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasASin
  };
};

/** \internal
  * \brief Template functor to raise a scalar to a power
  * \sa class CwiseUnaryOp, Cwise::pow
  */
template<typename Scalar>
struct scalar_pow_op {
  // FIXME default copy constructors seems bugged with std::complex<>
  inline scalar_pow_op(const scalar_pow_op& other) : m_exponent(other.m_exponent) { }
  inline scalar_pow_op(const Scalar& exponent) : m_exponent(exponent) {}
  inline Scalar operator() (const Scalar& a) const { return numext::pow(a, m_exponent); }
  const Scalar m_exponent;
};
template<typename Scalar>
struct functor_traits<scalar_pow_op<Scalar> >
{ enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = false }; };

/** \internal
  * \brief Template functor to compute the quotient between a scalar and array entries.
  * \sa class CwiseUnaryOp, Cwise::inverse()
  */
template<typename Scalar>
struct scalar_inverse_mult_op {
  scalar_inverse_mult_op(const Scalar& other) : m_other(other) {}
  inline Scalar operator() (const Scalar& a) const { return m_other / a; }
  template<typename Packet>
  inline const Packet packetOp(const Packet& a) const
  { return internal::pdiv(pset1<Packet>(m_other),a); }
  Scalar m_other;
};

/** \internal
  * \brief Template functor to compute the inverse of a scalar
  * \sa class CwiseUnaryOp, Cwise::inverse()
  */
template<typename Scalar>
struct scalar_inverse_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_inverse_op)
  inline Scalar operator() (const Scalar& a) const { return Scalar(1)/a; }
  template<typename Packet>
  inline const Packet packetOp(const Packet& a) const
  { return internal::pdiv(pset1<Packet>(Scalar(1)),a); }
};
template<typename Scalar>
struct functor_traits<scalar_inverse_op<Scalar> >
{ enum { Cost = NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasDiv }; };

/** \internal
  * \brief Template functor to compute the square of a scalar
  * \sa class CwiseUnaryOp, Cwise::square()
  */
template<typename Scalar>
struct scalar_square_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_square_op)
  inline Scalar operator() (const Scalar& a) const { return a*a; }
  template<typename Packet>
  inline const Packet packetOp(const Packet& a) const
  { return internal::pmul(a,a); }
};
template<typename Scalar>
struct functor_traits<scalar_square_op<Scalar> >
{ enum { Cost = NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasMul }; };

/** \internal
  * \brief Template functor to compute the cube of a scalar
  * \sa class CwiseUnaryOp, Cwise::cube()
  */
template<typename Scalar>
struct scalar_cube_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_cube_op)
  inline Scalar operator() (const Scalar& a) const { return a*a*a; }
  template<typename Packet>
  inline const Packet packetOp(const Packet& a) const
  { return internal::pmul(a,pmul(a,a)); }
};
template<typename Scalar>
struct functor_traits<scalar_cube_op<Scalar> >
{ enum { Cost = 2*NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasMul }; };

// default functor traits for STL functors:

template<typename T>
struct functor_traits<std::multiplies<T> >
{ enum { Cost = NumTraits<T>::MulCost, PacketAccess = false }; };

template<typename T>
struct functor_traits<std::divides<T> >
{ enum { Cost = NumTraits<T>::MulCost, PacketAccess = false }; };

template<typename T>
struct functor_traits<std::plus<T> >
{ enum { Cost = NumTraits<T>::AddCost, PacketAccess = false }; };

template<typename T>
struct functor_traits<std::minus<T> >
{ enum { Cost = NumTraits<T>::AddCost, PacketAccess = false }; };

template<typename T>
struct functor_traits<std::negate<T> >
{ enum { Cost = NumTraits<T>::AddCost, PacketAccess = false }; };

template<typename T>
struct functor_traits<std::logical_or<T> >
{ enum { Cost = 1, PacketAccess = false }; };

template<typename T>
struct functor_traits<std::logical_and<T> >
{ enum { Cost = 1, PacketAccess = false }; };

template<typename T>
struct functor_traits<std::logical_not<T> >
{ enum { Cost = 1, PacketAccess = false }; };

template<typename T>
struct functor_traits<std::greater<T> >
{ enum { Cost = 1, PacketAccess = false }; };

template<typename T>
struct functor_traits<std::less<T> >
{ enum { Cost = 1, PacketAccess = false }; };

template<typename T>
struct functor_traits<std::greater_equal<T> >
{ enum { Cost = 1, PacketAccess = false }; };

template<typename T>
struct functor_traits<std::less_equal<T> >
{ enum { Cost = 1, PacketAccess = false }; };

template<typename T>
struct functor_traits<std::equal_to<T> >
{ enum { Cost = 1, PacketAccess = false }; };

template<typename T>
struct functor_traits<std::not_equal_to<T> >
{ enum { Cost = 1, PacketAccess = false }; };

template<typename T>
struct functor_traits<std::binder2nd<T> >
{ enum { Cost = functor_traits<T>::Cost, PacketAccess = false }; };

template<typename T>
struct functor_traits<std::binder1st<T> >
{ enum { Cost = functor_traits<T>::Cost, PacketAccess = false }; };

template<typename T>
struct functor_traits<std::unary_negate<T> >
{ enum { Cost = 1 + functor_traits<T>::Cost, PacketAccess = false }; };

template<typename T>
struct functor_traits<std::binary_negate<T> >
{ enum { Cost = 1 + functor_traits<T>::Cost, PacketAccess = false }; };

#ifdef EIGEN_STDEXT_SUPPORT

template<typename T0,typename T1>
struct functor_traits<std::project1st<T0,T1> >
{ enum { Cost = 0, PacketAccess = false }; };

template<typename T0,typename T1>
struct functor_traits<std::project2nd<T0,T1> >
{ enum { Cost = 0, PacketAccess = false }; };

template<typename T0,typename T1>
struct functor_traits<std::select2nd<std::pair<T0,T1> > >
{ enum { Cost = 0, PacketAccess = false }; };

template<typename T0,typename T1>
struct functor_traits<std::select1st<std::pair<T0,T1> > >
{ enum { Cost = 0, PacketAccess = false }; };

template<typename T0,typename T1>
struct functor_traits<std::unary_compose<T0,T1> >
{ enum { Cost = functor_traits<T0>::Cost + functor_traits<T1>::Cost, PacketAccess = false }; };

template<typename T0,typename T1,typename T2>
struct functor_traits<std::binary_compose<T0,T1,T2> >
{ enum { Cost = functor_traits<T0>::Cost + functor_traits<T1>::Cost + functor_traits<T2>::Cost, PacketAccess = false }; };

#endif // EIGEN_STDEXT_SUPPORT

// allow to add new functors and specializations of functor_traits from outside Eigen.
// this macro is really needed because functor_traits must be specialized after it is declared but before it is used...
#ifdef EIGEN_FUNCTORS_PLUGIN
#include EIGEN_FUNCTORS_PLUGIN
#endif

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_FUNCTORS_H
