// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BINARY_FUNCTORS_H
#define EIGEN_BINARY_FUNCTORS_H

// clang-format off

namespace Eigen {

namespace internal {

//---------- associative binary functors ----------

/** \internal
  * \brief Template functor to compute the sum of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator+, class VectorwiseOp, DenseBase::sum()
  */
template<typename Scalar> struct scalar_sum_op {
//   typedef Scalar result_type;
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sum_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& a, const Scalar& b) const { return a + b; }
  template<typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const
  { return internal::padd(a,b); }
  template<typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar predux(const Packet& a) const
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
  * \brief Template specialization to deprecate the summation of boolean expressions.
  * This is required to solve Bug 426.
  * \sa DenseBase::count(), DenseBase::any(), ArrayBase::cast(), MatrixBase::cast()
  */
template<> struct scalar_sum_op<bool> : scalar_sum_op<int> {
  EIGEN_DEPRECATED
  scalar_sum_op() {}
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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type operator() (const LhsScalar& a, const RhsScalar& b) const { return a * b; }
  template<typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const
  { return internal::pmul(a,b); }
  template<typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type predux(const Packet& a) const
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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type operator() (const LhsScalar& a, const RhsScalar& b) const
  { return conj_helper<LhsScalar,RhsScalar,Conj,false>().pmul(a,b); }

  template<typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const
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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& a, const Scalar& b) const { return numext::mini(a, b); }
  template<typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const
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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& a, const Scalar& b) const  { return numext::maxi(a, b); }
  template<typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const
  { return internal::pmax(a,b); }
  template<typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar predux(const Packet& a) const
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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& _x, const Scalar& _y) const
  {
    using std::sqrt;
    Scalar p = numext::maxi(_x, _y);
    Scalar q = numext::mini(_x, _y);
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
  EIGEN_DEVICE_FUNC inline Scalar operator() (const Scalar& a, const OtherScalar& b) const { return numext::pow(a, b); }
};
template<typename Scalar, typename OtherScalar>
struct functor_traits<scalar_binary_pow_op<Scalar,OtherScalar> > {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = false };
};



//---------- non associative binary functors ----------

/** \internal
  * \brief Template functor to compute the difference of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator-
  */
template<typename Scalar> struct scalar_difference_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_difference_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& a, const Scalar& b) const { return a - b; }
  template<typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const
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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type operator() (const LhsScalar& a, const RhsScalar& b) const { return a / b; }
  template<typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const
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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator() (const bool& a, const bool& b) const { return a && b; }
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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator() (const bool& a, const bool& b) const { return a || b; }
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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator() (const bool& a, const bool& b) const { return a ^ b; }
};
template<> struct functor_traits<scalar_boolean_xor_op> {
  enum {
    Cost = NumTraits<bool>::AddCost,
    PacketAccess = false
  };
};



//---------- binary functors bound to a constant, thus appearing as a unary functor ----------

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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE scalar_multiple_op(const scalar_multiple_op& other) : m_other(other.m_other) { }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE scalar_multiple_op(const Scalar& other) : m_other(other) { }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator() (const Scalar& a) const { return a * m_other; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const
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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE scalar_multiple2_op(const scalar_multiple2_op& other) : m_other(other.m_other) { }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE scalar_multiple2_op(const Scalar2& other) : m_other(other) { }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator() (const Scalar1& a) const { return a * m_other; }
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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE scalar_quotient1_op(const scalar_quotient1_op& other) : m_other(other.m_other) { }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE scalar_quotient1_op(const Scalar& other) : m_other(other) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator() (const Scalar& a) const { return a / m_other; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const
  { return internal::pdiv(a, pset1<Packet>(m_other)); }
  typename add_const_on_value_type<typename NumTraits<Scalar>::Nested>::type m_other;
};
template<typename Scalar>
struct functor_traits<scalar_quotient1_op<Scalar> >
{ enum { Cost = 2 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasDiv }; };

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
  EIGEN_DEVICE_FUNC inline scalar_add_op(const scalar_add_op& other) : m_other(other.m_other) { }
  EIGEN_DEVICE_FUNC inline scalar_add_op(const Scalar& other) : m_other(other) { }
  EIGEN_DEVICE_FUNC inline Scalar operator() (const Scalar& a) const { return a + m_other; }
  EIGEN_DEVICE_FUNC inline const Packet packetOp(const Packet& a) const
  { return internal::padd(a, pset1<Packet>(m_other)); }
  const Scalar m_other;
};
template<typename Scalar>
struct functor_traits<scalar_add_op<Scalar> >
{ enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = packet_traits<Scalar>::HasAdd }; };

/** \internal
  * \brief Template functor to subtract a fixed scalar to another one
  * \sa class CwiseUnaryOp, Array::operator-, struct scalar_add_op, struct scalar_rsub_op
  */
template<typename Scalar>
struct scalar_sub_op {
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC inline scalar_sub_op(const scalar_sub_op& other) : m_other(other.m_other) { }
  EIGEN_DEVICE_FUNC inline scalar_sub_op(const Scalar& other) : m_other(other) { }
  EIGEN_DEVICE_FUNC inline Scalar operator() (const Scalar& a) const { return a - m_other; }
  EIGEN_DEVICE_FUNC inline const Packet packetOp(const Packet& a) const
  { return internal::psub(a, pset1<Packet>(m_other)); }
  const Scalar m_other;
};
template<typename Scalar>
struct functor_traits<scalar_sub_op<Scalar> >
{ enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = packet_traits<Scalar>::HasAdd }; };

/** \internal
  * \brief Template functor to subtract a scalar to fixed another one
  * \sa class CwiseUnaryOp, Array::operator-, struct scalar_add_op, struct scalar_sub_op
  */
template<typename Scalar>
struct scalar_rsub_op {
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC inline scalar_rsub_op(const scalar_rsub_op& other) : m_other(other.m_other) { }
  EIGEN_DEVICE_FUNC inline scalar_rsub_op(const Scalar& other) : m_other(other) { }
  EIGEN_DEVICE_FUNC inline Scalar operator() (const Scalar& a) const { return m_other - a; }
  EIGEN_DEVICE_FUNC inline const Packet packetOp(const Packet& a) const
  { return internal::psub(pset1<Packet>(m_other), a); }
  const Scalar m_other;
};
template<typename Scalar>
struct functor_traits<scalar_rsub_op<Scalar> >
{ enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = packet_traits<Scalar>::HasAdd }; };

/** \internal
  * \brief Template functor to raise a scalar to a power
  * \sa class CwiseUnaryOp, Cwise::pow
  */
template<typename Scalar>
struct scalar_pow_op {
  // FIXME default copy constructors seems bugged with std::complex<>
  EIGEN_DEVICE_FUNC inline scalar_pow_op(const scalar_pow_op& other) : m_exponent(other.m_exponent) { }
  EIGEN_DEVICE_FUNC inline scalar_pow_op(const Scalar& exponent) : m_exponent(exponent) {}
  EIGEN_DEVICE_FUNC inline Scalar operator() (const Scalar& a) const { return numext::pow(a, m_exponent); }
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
  EIGEN_DEVICE_FUNC scalar_inverse_mult_op(const Scalar& other) : m_other(other) {}
  EIGEN_DEVICE_FUNC inline Scalar operator() (const Scalar& a) const { return m_other / a; }
  template<typename Packet>
  EIGEN_DEVICE_FUNC inline const Packet packetOp(const Packet& a) const
  { return internal::pdiv(pset1<Packet>(m_other),a); }
  Scalar m_other;
};
template<typename Scalar>
struct functor_traits<scalar_inverse_mult_op<Scalar> >
{ enum { Cost = 2 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasDiv }; };

/** \internal
 * \brief Template functor to compute the modulo between an array and a scalar.
 */
template <typename Scalar>
struct scalar_mod_op {
  EIGEN_DEVICE_FUNC scalar_mod_op(const Scalar& divisor) : m_divisor(divisor) {}
  EIGEN_DEVICE_FUNC inline Scalar operator() (const Scalar& a) const { return a % m_divisor; }
  const Scalar m_divisor;
};
template <typename Scalar>
struct functor_traits<scalar_mod_op<Scalar> >
{ enum { Cost = 2 * NumTraits<Scalar>::MulCost, PacketAccess = false }; };

/** \internal
 * \brief Template functor to compute the float modulo between an array and a scalar.
 */
template <typename Scalar>
struct scalar_fmod_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_fmod_op);
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& a, const Scalar& b) const {
    EIGEN_USING_STD_MATH(fmod);
    return (fmod)(a, b);
  }
};

template <typename Scalar>
struct functor_traits<scalar_fmod_op<Scalar> > {
  enum { Cost = 2 * NumTraits<Scalar>::MulCost, PacketAccess = false };
};


} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_BINARY_FUNCTORS_H
