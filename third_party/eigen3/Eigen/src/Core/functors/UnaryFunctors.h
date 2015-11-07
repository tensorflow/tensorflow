// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_UNARY_FUNCTORS_H
#define EIGEN_UNARY_FUNCTORS_H

namespace Eigen {

namespace internal {

#if defined(__NVCC__) || !defined(__CUDA_ARCH__)
using std::abs;
using std::exp;
using std::log;
using std::min;
using std::sqrt;
using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
#endif

/** \internal
  * \brief Template functor to compute the opposite of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::operator-
  */
template<typename Scalar> struct scalar_opposite_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_opposite_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& a) const { return -a; }
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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type operator() (const Scalar& a) const { return abs(a); }
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
  EIGEN_DEVICE_FUNC
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
  EIGEN_DEVICE_FUNC
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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const NewType operator() (const Scalar& a) const { return cast<Scalar, NewType>(a); }
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
  EIGEN_DEVICE_FUNC
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
  EIGEN_DEVICE_FUNC
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
  EIGEN_DEVICE_FUNC
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
  EIGEN_DEVICE_FUNC
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
  EIGEN_DEVICE_FUNC inline const Scalar operator() (const Scalar& a) const { return exp(a); }
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
  EIGEN_DEVICE_FUNC inline const Scalar operator() (const Scalar& a) const { return log(a); }
  typedef typename packet_traits<Scalar>::type Packet;
  inline Packet packetOp(const Packet& a) const { return internal::plog(a); }
};
template<typename Scalar>
struct functor_traits<scalar_log_op<Scalar> >
{ enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasLog }; };


/** \internal
  * \brief Template functor to compute the square root of a scalar
  * \sa class CwiseUnaryOp, Cwise::sqrt()
  */
template<typename Scalar> struct scalar_sqrt_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sqrt_op)
  EIGEN_DEVICE_FUNC inline const Scalar operator() (const Scalar& a) const { return sqrt(a); }
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
  * \brief Template functor to compute the reciprocal square root of a scalar
  * \sa class CwiseUnaryOp, Cwise::rsqrt()
  */
template<typename Scalar> struct scalar_rsqrt_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_rsqrt_op)
  EIGEN_DEVICE_FUNC inline const Scalar operator() (const Scalar& a) const { return Scalar(1)/sqrt(a); }
  typedef typename packet_traits<Scalar>::type Packet;
  inline Packet packetOp(const Packet& a) const { return internal::prsqrt(a); }
};

template<typename Scalar>
struct functor_traits<scalar_rsqrt_op<Scalar> >
{ enum {
    Cost = 5 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasRsqrt
  };
};


/** \internal
  * \brief Template functor to compute the cosine of a scalar
  * \sa class CwiseUnaryOp, ArrayBase::cos()
  */
template<typename Scalar> struct scalar_cos_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_cos_op)
  EIGEN_DEVICE_FUNC inline Scalar operator() (const Scalar& a) const { return cos(a); }
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
  EIGEN_DEVICE_FUNC inline const Scalar operator() (const Scalar& a) const { return sin(a); }
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
  EIGEN_DEVICE_FUNC inline const Scalar operator() (const Scalar& a) const { return tan(a); }
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
  EIGEN_DEVICE_FUNC inline const Scalar operator() (const Scalar& a) const { return acos(a); }
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
  EIGEN_DEVICE_FUNC inline const Scalar operator() (const Scalar& a) const { return asin(a); }
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
  * \brief Template functor to compute the atan of a scalar
  * \sa class CwiseUnaryOp, ArrayBase::atan()
  */
template<typename Scalar> struct scalar_atan_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_atan_op)
  inline const Scalar operator() (const Scalar& a) const { return atan(a); }
  typedef typename packet_traits<Scalar>::type Packet;
  inline Packet packetOp(const Packet& a) const { return internal::patan(a); }
};
template<typename Scalar>
struct functor_traits<scalar_atan_op<Scalar> >
{
  enum {
    Cost = 5 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasATan
  };
};

 /** \internal
  * \brief Template functor to compute the tanh of a scalar
  * \sa class CwiseUnaryOp, ArrayBase::tanh()
  */
template<typename Scalar> struct scalar_tanh_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_tanh_op)
  EIGEN_DEVICE_FUNC inline const Scalar operator() (const Scalar& a) const { using std::tanh; return tanh(a); }
  typedef typename packet_traits<Scalar>::type Packet;
  inline Packet packetOp(const Packet& a) const { return internal::ptanh(a); }
};
template<typename Scalar>
struct functor_traits<scalar_tanh_op<Scalar> >
{
  enum {
    Cost = 6 * NumTraits<Scalar>::MulCost + 4 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasTanH
  };
};

 /** \internal
  * \brief Template functor to compute the sigmoid of a scalar
  * \sa class CwiseUnaryOp, ArrayBase::sigmoid()
  */
template <typename T>
struct scalar_sigmoid_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sigmoid_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x) const {
    const T one = T(1);
    return one / (one + std::exp(-x));
  }

  template <typename Packet>
  inline Packet packetOp(const Packet& x) const {
    const Packet one = pset1<Packet>(1);
    return pdiv(one, padd(one, pexp(pnegate(x))));
  }
};

template <typename T>
struct functor_traits<scalar_sigmoid_op<T> > {
  enum {
    Cost = NumTraits<T>::AddCost * 2 + NumTraits<T>::MulCost * 6,
    PacketAccess = packet_traits<T>::HasAdd && packet_traits<T>::HasDiv &&
                   packet_traits<T>::HasNegate && packet_traits<T>::HasExp
  };
};

/** \internal
  * \brief Template functor to compute the inverse of a scalar
  * \sa class CwiseUnaryOp, Cwise::inverse()
  */
template<typename Scalar>
struct scalar_inverse_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_inverse_op)
  EIGEN_DEVICE_FUNC inline Scalar operator() (const Scalar& a) const { return Scalar(1)/a; }
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
  EIGEN_DEVICE_FUNC inline Scalar operator() (const Scalar& a) const { return a*a; }
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
  EIGEN_DEVICE_FUNC inline Scalar operator() (const Scalar& a) const { return a*a*a; }
  template<typename Packet>
  inline const Packet packetOp(const Packet& a) const
  { return internal::pmul(a,pmul(a,a)); }
};
template<typename Scalar>
struct functor_traits<scalar_cube_op<Scalar> >
{ enum { Cost = 2*NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasMul }; };


} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_FUNCTORS_H
