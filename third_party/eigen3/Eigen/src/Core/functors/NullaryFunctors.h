// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_NULLARY_FUNCTORS_H
#define EIGEN_NULLARY_FUNCTORS_H

namespace Eigen {

namespace internal {

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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (Index row, Index col) const { return row==col ? Scalar(1) : Scalar(0); }
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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (Index i) const
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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (Index i) const { return m_low+i*m_step; }

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
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (Index i) const { return impl(i); }

  // We need this function when assigning e.g. a RowVectorXd to a MatrixXd since
  // there row==0 and col is used for the actual iteration.
  template<typename Index>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (Index row, Index col) const
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

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_NULLARY_FUNCTORS_H
