// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CWISE_UNARY_OP_H
#define EIGEN_CWISE_UNARY_OP_H

namespace Eigen { 

/** \class CwiseUnaryOp
  * \ingroup Core_Module
  *
  * \brief Generic expression where a coefficient-wise unary operator is applied to an expression
  *
  * \param UnaryOp template functor implementing the operator
  * \param XprType the type of the expression to which we are applying the unary operator
  *
  * This class represents an expression where a unary operator is applied to an expression.
  * It is the return type of all operations taking exactly 1 input expression, regardless of the
  * presence of other inputs such as scalars. For example, the operator* in the expression 3*matrix
  * is considered unary, because only the right-hand side is an expression, and its
  * return type is a specialization of CwiseUnaryOp.
  *
  * Most of the time, this is the only way that it is used, so you typically don't have to name
  * CwiseUnaryOp types explicitly.
  *
  * \sa MatrixBase::unaryExpr(const CustomUnaryOp &) const, class CwiseBinaryOp, class CwiseNullaryOp
  */

namespace internal {
template<typename UnaryOp, typename XprType>
struct traits<CwiseUnaryOp<UnaryOp, XprType> >
 : traits<XprType>
{
  typedef typename result_of<
                     UnaryOp(typename XprType::Scalar)
                   >::type Scalar;
  typedef typename XprType::Nested XprTypeNested;
  typedef typename remove_reference<XprTypeNested>::type _XprTypeNested;
  enum {
    Flags = _XprTypeNested::Flags & (
      HereditaryBits | LinearAccessBit | AlignedBit
      | (functor_traits<UnaryOp>::PacketAccess ? PacketAccessBit : 0)),
    CoeffReadCost = _XprTypeNested::CoeffReadCost + functor_traits<UnaryOp>::Cost
  };
};
}

template<typename UnaryOp, typename XprType, typename StorageKind>
class CwiseUnaryOpImpl;

template<typename UnaryOp, typename XprType>
class CwiseUnaryOp : internal::no_assignment_operator,
  public CwiseUnaryOpImpl<UnaryOp, XprType, typename internal::traits<XprType>::StorageKind>
{
  public:

    typedef typename CwiseUnaryOpImpl<UnaryOp, XprType,typename internal::traits<XprType>::StorageKind>::Base Base;
    EIGEN_GENERIC_PUBLIC_INTERFACE(CwiseUnaryOp)

    EIGEN_DEVICE_FUNC
    inline CwiseUnaryOp(const XprType& xpr, const UnaryOp& func = UnaryOp())
      : m_xpr(xpr), m_functor(func) {}

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Index rows() const { return m_xpr.rows(); }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Index cols() const { return m_xpr.cols(); }

    /** \returns the functor representing the unary operation */
    EIGEN_DEVICE_FUNC
    const UnaryOp& functor() const { return m_functor; }

    /** \returns the nested expression */
    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    nestedExpression() const { return m_xpr; }

    /** \returns the nested expression */
    EIGEN_DEVICE_FUNC
    typename internal::remove_all<typename XprType::Nested>::type&
    nestedExpression() { return m_xpr.const_cast_derived(); }

  protected:
    typename XprType::Nested m_xpr;
    const UnaryOp m_functor;
};

// This is the generic implementation for dense storage.
// It can be used for any expression types implementing the dense concept.
template<typename UnaryOp, typename XprType>
class CwiseUnaryOpImpl<UnaryOp,XprType,Dense>
  : public internal::dense_xpr_base<CwiseUnaryOp<UnaryOp, XprType> >::type
{
  public:

    typedef CwiseUnaryOp<UnaryOp, XprType> Derived;
    typedef typename internal::dense_xpr_base<CwiseUnaryOp<UnaryOp, XprType> >::type Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(Derived)

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const Scalar coeff(Index rowId, Index colId) const
    {
      return derived().functor()(derived().nestedExpression().coeff(rowId, colId));
    }

    template<int LoadMode>
    EIGEN_STRONG_INLINE PacketScalar packet(Index rowId, Index colId) const
    {
      return derived().functor().packetOp(derived().nestedExpression().template packet<LoadMode>(rowId, colId));
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const Scalar coeff(Index index) const
    {
      return derived().functor()(derived().nestedExpression().coeff(index));
    }

    template<int LoadMode>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE PacketScalar packet(Index index) const
    {
      return derived().functor().packetOp(derived().nestedExpression().template packet<LoadMode>(index));
    }
};

} // end namespace Eigen

#endif // EIGEN_CWISE_UNARY_OP_H
