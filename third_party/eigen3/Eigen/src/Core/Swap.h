// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SWAP_H
#define EIGEN_SWAP_H

namespace Eigen { 

/** \class SwapWrapper
  * \ingroup Core_Module
  *
  * \internal
  *
  * \brief Internal helper class for swapping two expressions
  */
namespace internal {
template<typename ExpressionType>
struct traits<SwapWrapper<ExpressionType> > : traits<ExpressionType> {};
}

template<typename ExpressionType> class SwapWrapper
  : public internal::dense_xpr_base<SwapWrapper<ExpressionType> >::type
{
  public:

    typedef typename internal::dense_xpr_base<SwapWrapper>::type Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(SwapWrapper)
    typedef typename internal::packet_traits<Scalar>::type Packet;

    EIGEN_DEVICE_FUNC
    inline SwapWrapper(ExpressionType& xpr) : m_expression(xpr) {}

    EIGEN_DEVICE_FUNC
    inline Index rows() const { return m_expression.rows(); }
    EIGEN_DEVICE_FUNC
    inline Index cols() const { return m_expression.cols(); }
    EIGEN_DEVICE_FUNC
    inline Index outerStride() const { return m_expression.outerStride(); }
    EIGEN_DEVICE_FUNC
    inline Index innerStride() const { return m_expression.innerStride(); }
    
    typedef typename internal::conditional<
                       internal::is_lvalue<ExpressionType>::value,
                       Scalar,
                       const Scalar
                     >::type ScalarWithConstIfNotLvalue;
                     
    EIGEN_DEVICE_FUNC
    inline ScalarWithConstIfNotLvalue* data() { return m_expression.data(); }
    EIGEN_DEVICE_FUNC
    inline const Scalar* data() const { return m_expression.data(); }

    EIGEN_DEVICE_FUNC
    inline Scalar& coeffRef(Index rowId, Index colId)
    {
      return m_expression.const_cast_derived().coeffRef(rowId, colId);
    }

    EIGEN_DEVICE_FUNC
    inline Scalar& coeffRef(Index index)
    {
      return m_expression.const_cast_derived().coeffRef(index);
    }

    EIGEN_DEVICE_FUNC
    inline Scalar& coeffRef(Index rowId, Index colId) const
    {
      return m_expression.coeffRef(rowId, colId);
    }

    EIGEN_DEVICE_FUNC
    inline Scalar& coeffRef(Index index) const
    {
      return m_expression.coeffRef(index);
    }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    void copyCoeff(Index rowId, Index colId, const DenseBase<OtherDerived>& other)
    {
      OtherDerived& _other = other.const_cast_derived();
      eigen_internal_assert(rowId >= 0 && rowId < rows()
                         && colId >= 0 && colId < cols());
      Scalar tmp = m_expression.coeff(rowId, colId);
      m_expression.coeffRef(rowId, colId) = _other.coeff(rowId, colId);
      _other.coeffRef(rowId, colId) = tmp;
    }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    void copyCoeff(Index index, const DenseBase<OtherDerived>& other)
    {
      OtherDerived& _other = other.const_cast_derived();
      eigen_internal_assert(index >= 0 && index < m_expression.size());
      Scalar tmp = m_expression.coeff(index);
      m_expression.coeffRef(index) = _other.coeff(index);
      _other.coeffRef(index) = tmp;
    }

    template<typename OtherDerived, int StoreMode, int LoadMode>
    void copyPacket(Index rowId, Index colId, const DenseBase<OtherDerived>& other)
    {
      OtherDerived& _other = other.const_cast_derived();
      eigen_internal_assert(rowId >= 0 && rowId < rows()
                        && colId >= 0 && colId < cols());
      Packet tmp = m_expression.template packet<StoreMode>(rowId, colId);
      m_expression.template writePacket<StoreMode>(rowId, colId,
        _other.template packet<LoadMode>(rowId, colId)
      );
      _other.template writePacket<LoadMode>(rowId, colId, tmp);
    }

    template<typename OtherDerived, int StoreMode, int LoadMode>
    void copyPacket(Index index, const DenseBase<OtherDerived>& other)
    {
      OtherDerived& _other = other.const_cast_derived();
      eigen_internal_assert(index >= 0 && index < m_expression.size());
      Packet tmp = m_expression.template packet<StoreMode>(index);
      m_expression.template writePacket<StoreMode>(index,
        _other.template packet<LoadMode>(index)
      );
      _other.template writePacket<LoadMode>(index, tmp);
    }

    EIGEN_DEVICE_FUNC
    ExpressionType& expression() const { return m_expression; }

  protected:
    ExpressionType& m_expression;
};

} // end namespace Eigen

#endif // EIGEN_SWAP_H
