// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_ASSIGNMENT_FUNCTORS_H
#define EIGEN_ASSIGNMENT_FUNCTORS_H

namespace Eigen {

namespace internal {
  
/** \internal
  * \brief Template functor for scalar/packet assignment
  *
  */
template<typename Scalar> struct assign_op {

  EIGEN_EMPTY_STRUCT_CTOR(assign_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void assignCoeff(Scalar& a, const Scalar& b) const { a = b; }
  
  template<int Alignment, typename Packet>
  EIGEN_STRONG_INLINE void assignPacket(Scalar* a, const Packet& b) const
  { internal::pstoret<Scalar,Packet,Alignment>(a,b); }
};
template<typename Scalar>
struct functor_traits<assign_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::ReadCost,
    PacketAccess = packet_traits<Scalar>::IsVectorized
  };
};

/** \internal
  * \brief Template functor for scalar/packet assignment with addition
  *
  */
template<typename Scalar> struct add_assign_op {

  EIGEN_EMPTY_STRUCT_CTOR(add_assign_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void assignCoeff(Scalar& a, const Scalar& b) const { a += b; }
  
  template<int Alignment, typename Packet>
  EIGEN_STRONG_INLINE void assignPacket(Scalar* a, const Packet& b) const
  { internal::pstoret<Scalar,Packet,Alignment>(a,internal::padd(internal::ploadt<Packet,Alignment>(a),b)); }
};
template<typename Scalar>
struct functor_traits<add_assign_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::ReadCost + NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasAdd
  };
};

/** \internal
  * \brief Template functor for scalar/packet assignment with subtraction
  *
  */
template<typename Scalar> struct sub_assign_op {

  EIGEN_EMPTY_STRUCT_CTOR(sub_assign_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void assignCoeff(Scalar& a, const Scalar& b) const { a -= b; }
  
  template<int Alignment, typename Packet>
  EIGEN_STRONG_INLINE void assignPacket(Scalar* a, const Packet& b) const
  { internal::pstoret<Scalar,Packet,Alignment>(a,internal::psub(internal::ploadt<Packet,Alignment>(a),b)); }
};
template<typename Scalar>
struct functor_traits<sub_assign_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::ReadCost + NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasAdd
  };
};

/** \internal
  * \brief Template functor for scalar/packet assignment with multiplication
  *
  */
template<typename Scalar> struct mul_assign_op {

  EIGEN_EMPTY_STRUCT_CTOR(mul_assign_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void assignCoeff(Scalar& a, const Scalar& b) const { a *= b; }
  
  template<int Alignment, typename Packet>
  EIGEN_STRONG_INLINE void assignPacket(Scalar* a, const Packet& b) const
  { internal::pstoret<Scalar,Packet,Alignment>(a,internal::pmul(internal::ploadt<Packet,Alignment>(a),b)); }
};
template<typename Scalar>
struct functor_traits<mul_assign_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::ReadCost + NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasMul
  };
};

/** \internal
  * \brief Template functor for scalar/packet assignment with diviving
  *
  */
template<typename Scalar> struct div_assign_op {

  EIGEN_EMPTY_STRUCT_CTOR(div_assign_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void assignCoeff(Scalar& a, const Scalar& b) const { a /= b; }
  
  template<int Alignment, typename Packet>
  EIGEN_STRONG_INLINE void assignPacket(Scalar* a, const Packet& b) const
  { internal::pstoret<Scalar,Packet,Alignment>(a,internal::pdiv(internal::ploadt<Packet,Alignment>(a),b)); }
};
template<typename Scalar>
struct functor_traits<div_assign_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::ReadCost + NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasMul
  };
};


/** \internal
  * \brief Template functor for scalar/packet assignment with swaping
  *
  * It works as follow. For a non-vectorized evaluation loop, we have:
  *   for(i) func(A.coeffRef(i), B.coeff(i));
  * where B is a SwapWrapper expression. The trick is to make SwapWrapper::coeff behaves like a non-const coeffRef.
  * Actually, SwapWrapper might not even be needed since even if B is a plain expression, since it has to be writable
  * B.coeff already returns a const reference to the underlying scalar value.
  * 
  * The case of a vectorized loop is more tricky:
  *   for(i,j) func.assignPacket<A_Align>(&A.coeffRef(i,j), B.packet<B_Align>(i,j));
  * Here, B must be a SwapWrapper whose packet function actually returns a proxy object holding a Scalar*,
  * the actual alignment and Packet type.
  *
  */
template<typename Scalar> struct swap_assign_op {

  EIGEN_EMPTY_STRUCT_CTOR(swap_assign_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void assignCoeff(Scalar& a, const Scalar& b) const
  {
    using std::swap;
    swap(a,const_cast<Scalar&>(b));
  }
  
  template<int LhsAlignment, int RhsAlignment, typename Packet>
  EIGEN_STRONG_INLINE void swapPacket(Scalar* a, Scalar* b) const
  {
    Packet tmp = internal::ploadt<Packet,RhsAlignment>(b);
    internal::pstoret<Scalar,Packet,RhsAlignment>(b, internal::ploadt<Packet,LhsAlignment>(a));
    internal::pstoret<Scalar,Packet,LhsAlignment>(a, tmp);
  }
};
template<typename Scalar>
struct functor_traits<swap_assign_op<Scalar> > {
  enum {
    Cost = 3 * NumTraits<Scalar>::ReadCost,
    PacketAccess = packet_traits<Scalar>::IsVectorized
  };
};

} // namespace internal

} // namespace Eigen

#endif // EIGEN_ASSIGNMENT_FUNCTORS_H
