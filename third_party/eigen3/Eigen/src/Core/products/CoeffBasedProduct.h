// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COEFFBASED_PRODUCT_H
#define EIGEN_COEFFBASED_PRODUCT_H

namespace Eigen { 

namespace internal {

/*********************************************************************************
*  Coefficient based product implementation.
*  It is designed for the following use cases:
*  - small fixed sizes
*  - lazy products
*********************************************************************************/

/* Since the all the dimensions of the product are small, here we can rely
 * on the generic Assign mechanism to evaluate the product per coeff (or packet).
 *
 * Note that here the inner-loops should always be unrolled.
 */

template<int Traversal, int UnrollingIndex, typename Lhs, typename Rhs, typename RetScalar>
struct product_coeff_impl;

template<int StorageOrder, int UnrollingIndex, typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct product_packet_impl;

template<typename LhsNested, typename RhsNested, int NestingFlags>
struct traits<CoeffBasedProduct<LhsNested,RhsNested,NestingFlags> >
{
  typedef MatrixXpr XprKind;
  typedef typename remove_all<LhsNested>::type _LhsNested;
  typedef typename remove_all<RhsNested>::type _RhsNested;
  typedef typename scalar_product_traits<typename _LhsNested::Scalar, typename _RhsNested::Scalar>::ReturnType Scalar;
  typedef typename promote_storage_type<typename traits<_LhsNested>::StorageKind,
                                           typename traits<_RhsNested>::StorageKind>::ret StorageKind;
  typedef typename promote_index_type<typename traits<_LhsNested>::Index,
                                         typename traits<_RhsNested>::Index>::type Index;

  enum {
      LhsCoeffReadCost = _LhsNested::CoeffReadCost,
      RhsCoeffReadCost = _RhsNested::CoeffReadCost,
      LhsFlags = _LhsNested::Flags,
      RhsFlags = _RhsNested::Flags,

      RowsAtCompileTime = _LhsNested::RowsAtCompileTime,
      ColsAtCompileTime = _RhsNested::ColsAtCompileTime,
      InnerSize = EIGEN_SIZE_MIN_PREFER_FIXED(_LhsNested::ColsAtCompileTime, _RhsNested::RowsAtCompileTime),

      MaxRowsAtCompileTime = _LhsNested::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = _RhsNested::MaxColsAtCompileTime,

      LhsRowMajor = LhsFlags & RowMajorBit,
      RhsRowMajor = RhsFlags & RowMajorBit,

      SameType = is_same<typename _LhsNested::Scalar,typename _RhsNested::Scalar>::value,

      CanVectorizeRhs = RhsRowMajor && (RhsFlags & PacketAccessBit)
                      && (ColsAtCompileTime == Dynamic
                          || ( (ColsAtCompileTime % packet_traits<Scalar>::size) == 0
                              && (RhsFlags&AlignedBit)
                             )
                         ),

      CanVectorizeLhs = (!LhsRowMajor) && (LhsFlags & PacketAccessBit)
                      && (RowsAtCompileTime == Dynamic
                          || ( (RowsAtCompileTime % packet_traits<Scalar>::size) == 0
                              && (LhsFlags&AlignedBit)
                             )
                         ),

      EvalToRowMajor = (MaxRowsAtCompileTime==1&&MaxColsAtCompileTime!=1) ? 1
                     : (MaxColsAtCompileTime==1&&MaxRowsAtCompileTime!=1) ? 0
                     : (RhsRowMajor && !CanVectorizeLhs),

      Flags = ((unsigned int)(LhsFlags | RhsFlags) & HereditaryBits & ~RowMajorBit)
            | (EvalToRowMajor ? RowMajorBit : 0)
            | NestingFlags
            | (CanVectorizeLhs ? (LhsFlags & AlignedBit) : 0)
            | (CanVectorizeRhs ? (RhsFlags & AlignedBit) : 0)
            // TODO enable vectorization for mixed types
            | (SameType && (CanVectorizeLhs || CanVectorizeRhs) ? PacketAccessBit : 0),

      CoeffReadCost = InnerSize == Dynamic ? Dynamic
                    : InnerSize * (NumTraits<Scalar>::MulCost + LhsCoeffReadCost + RhsCoeffReadCost)
                      + (InnerSize - 1) * NumTraits<Scalar>::AddCost,

      /* CanVectorizeInner deserves special explanation. It does not affect the product flags. It is not used outside
      * of Product. If the Product itself is not a packet-access expression, there is still a chance that the inner
      * loop of the product might be vectorized. This is the meaning of CanVectorizeInner. Since it doesn't affect
      * the Flags, it is safe to make this value depend on ActualPacketAccessBit, that doesn't affect the ABI.
      */
      CanVectorizeInner =    SameType
                          && LhsRowMajor
                          && (!RhsRowMajor)
                          && (LhsFlags & RhsFlags & ActualPacketAccessBit)
                          && (LhsFlags & RhsFlags & AlignedBit)
                          && (InnerSize % packet_traits<Scalar>::size == 0)
    };
};

} // end namespace internal

template<typename LhsNested, typename RhsNested, int NestingFlags>
class CoeffBasedProduct
  : internal::no_assignment_operator,
    public MatrixBase<CoeffBasedProduct<LhsNested, RhsNested, NestingFlags> >
{
  public:

    typedef MatrixBase<CoeffBasedProduct> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(CoeffBasedProduct)
    typedef typename Base::PlainObject PlainObject;

  private:

    typedef typename internal::traits<CoeffBasedProduct>::_LhsNested _LhsNested;
    typedef typename internal::traits<CoeffBasedProduct>::_RhsNested _RhsNested;

    enum {
      PacketSize = internal::packet_traits<Scalar>::size,
      InnerSize  = internal::traits<CoeffBasedProduct>::InnerSize,
      Unroll = CoeffReadCost != Dynamic && CoeffReadCost <= EIGEN_UNROLLING_LIMIT,
      CanVectorizeInner = internal::traits<CoeffBasedProduct>::CanVectorizeInner
    };

    typedef internal::product_coeff_impl<CanVectorizeInner ? InnerVectorizedTraversal : DefaultTraversal,
                                   Unroll ? InnerSize-1 : Dynamic,
                                   _LhsNested, _RhsNested, Scalar> ScalarCoeffImpl;

    typedef CoeffBasedProduct<LhsNested,RhsNested,NestByRefBit> LazyCoeffBasedProductType;

  public:

    EIGEN_DEVICE_FUNC
    inline CoeffBasedProduct(const CoeffBasedProduct& other)
      : Base(), m_lhs(other.m_lhs), m_rhs(other.m_rhs)
    {}

    template<typename Lhs, typename Rhs>
    EIGEN_DEVICE_FUNC 
    inline CoeffBasedProduct(const Lhs& lhs, const Rhs& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      // we don't allow taking products of matrices of different real types, as that wouldn't be vectorizable.
      // We still allow to mix T and complex<T>.
      EIGEN_STATIC_ASSERT((internal::scalar_product_traits<typename Lhs::RealScalar, typename Rhs::RealScalar>::Defined),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
      eigen_assert(lhs.cols() == rhs.rows()
        && "invalid matrix product"
        && "if you wanted a coeff-wise or a dot product use the respective explicit functions");
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index rows() const { return m_lhs.rows(); }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index cols() const { return m_rhs.cols(); }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const Scalar coeff(Index row, Index col) const
    {
      Scalar res;
      ScalarCoeffImpl::run(row, col, m_lhs, m_rhs, res);
      return res;
    }

    /* Allow index-based non-packet access. It is impossible though to allow index-based packed access,
     * which is why we don't set the LinearAccessBit.
     */
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const Scalar coeff(Index index) const
    {
      Scalar res;
      const Index row = RowsAtCompileTime == 1 ? 0 : index;
      const Index col = RowsAtCompileTime == 1 ? index : 0;
      ScalarCoeffImpl::run(row, col, m_lhs, m_rhs, res);
      return res;
    }

    template<int LoadMode>
    EIGEN_STRONG_INLINE const PacketScalar packet(Index row, Index col) const
    {
      PacketScalar res;
      internal::product_packet_impl<Flags&RowMajorBit ? RowMajor : ColMajor,
                              Unroll ? InnerSize-1 : Dynamic,
                              _LhsNested, _RhsNested, PacketScalar, LoadMode>
        ::run(row, col, m_lhs, m_rhs, res);
      return res;
    }

    // Implicit conversion to the nested type (trigger the evaluation of the product)
    EIGEN_DEVICE_FUNC 
    EIGEN_STRONG_INLINE operator const PlainObject& () const
    {
      m_result.lazyAssign(*this);
      return m_result;
    }

    EIGEN_DEVICE_FUNC const _LhsNested& lhs() const { return m_lhs; }
    EIGEN_DEVICE_FUNC const _RhsNested& rhs() const { return m_rhs; }

    EIGEN_DEVICE_FUNC
    const Diagonal<const LazyCoeffBasedProductType,0> diagonal() const
    { return reinterpret_cast<const LazyCoeffBasedProductType&>(*this); }

    template<int DiagonalIndex>
    EIGEN_DEVICE_FUNC 
    const Diagonal<const LazyCoeffBasedProductType,DiagonalIndex> diagonal() const
    { return reinterpret_cast<const LazyCoeffBasedProductType&>(*this); }

    EIGEN_DEVICE_FUNC
    const Diagonal<const LazyCoeffBasedProductType, DynamicIndex> diagonal(Index index) const {
      return Diagonal<const LazyCoeffBasedProductType, DynamicIndex>(
          reinterpret_cast<const LazyCoeffBasedProductType&>(*this), index);
    }

  protected:
    typename internal::add_const_on_value_type<LhsNested>::type m_lhs;
    typename internal::add_const_on_value_type<RhsNested>::type m_rhs;

    mutable PlainObject m_result;
};

namespace internal {

// here we need to overload the nested rule for products
// such that the nested type is a const reference to a plain matrix
template<typename Lhs, typename Rhs, int N, typename PlainObject>
struct nested<CoeffBasedProduct<Lhs,Rhs,EvalBeforeNestingBit|EvalBeforeAssigningBit>, N, PlainObject>
{
  typedef PlainObject const& type;
};

/***************************************************************************
* Normal product .coeff() implementation (with meta-unrolling)
***************************************************************************/

/**************************************
*** Scalar path  - no vectorization ***
**************************************/

template<int UnrollingIndex, typename Lhs, typename Rhs, typename RetScalar>
struct product_coeff_impl<DefaultTraversal, UnrollingIndex, Lhs, Rhs, RetScalar>
{
  typedef typename Lhs::Index Index;
  EIGEN_DEVICE_FUNC 
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, RetScalar &res)
  {
    product_coeff_impl<DefaultTraversal, UnrollingIndex-1, Lhs, Rhs, RetScalar>::run(row, col, lhs, rhs, res);
    res += lhs.coeff(row, UnrollingIndex) * rhs.coeff(UnrollingIndex, col);
  }
};

template<typename Lhs, typename Rhs, typename RetScalar>
struct product_coeff_impl<DefaultTraversal, 0, Lhs, Rhs, RetScalar>
{
  typedef typename Lhs::Index Index;
  EIGEN_DEVICE_FUNC 
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, RetScalar &res)
  {
    res = lhs.coeff(row, 0) * rhs.coeff(0, col);
  }
};

template<typename Lhs, typename Rhs, typename RetScalar>
struct product_coeff_impl<DefaultTraversal, Dynamic, Lhs, Rhs, RetScalar>
{
  typedef typename Lhs::Index Index;
  EIGEN_DEVICE_FUNC 
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, RetScalar& res)
  {
    eigen_assert(lhs.cols()>0 && "you are using a non initialized matrix");
    res = lhs.coeff(row, 0) * rhs.coeff(0, col);
      for(Index i = 1; i < lhs.cols(); ++i)
        res += lhs.coeff(row, i) * rhs.coeff(i, col);
  }
};

/*******************************************
*** Scalar path with inner vectorization ***
*******************************************/

template<int UnrollingIndex, typename Lhs, typename Rhs, typename Packet>
struct product_coeff_vectorized_unroller
{
  typedef typename Lhs::Index Index;
  enum { PacketSize = packet_traits<typename Lhs::Scalar>::size };
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, typename Lhs::PacketScalar &pres)
  {
    product_coeff_vectorized_unroller<UnrollingIndex-PacketSize, Lhs, Rhs, Packet>::run(row, col, lhs, rhs, pres);
    pres = padd(pres, pmul( lhs.template packet<Aligned>(row, UnrollingIndex) , rhs.template packet<Aligned>(UnrollingIndex, col) ));
  }
};

template<typename Lhs, typename Rhs, typename Packet>
struct product_coeff_vectorized_unroller<0, Lhs, Rhs, Packet>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, typename Lhs::PacketScalar &pres)
  {
    pres = pmul(lhs.template packet<Aligned>(row, 0) , rhs.template packet<Aligned>(0, col));
  }
};

template<int UnrollingIndex, typename Lhs, typename Rhs, typename RetScalar>
struct product_coeff_impl<InnerVectorizedTraversal, UnrollingIndex, Lhs, Rhs, RetScalar>
{
  typedef typename Lhs::PacketScalar Packet;
  typedef typename Lhs::Index Index;
  enum { PacketSize = packet_traits<typename Lhs::Scalar>::size };
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, RetScalar &res)
  {
    Packet pres;
    product_coeff_vectorized_unroller<UnrollingIndex+1-PacketSize, Lhs, Rhs, Packet>::run(row, col, lhs, rhs, pres);
    res = predux(pres);
  }
};

template<typename Lhs, typename Rhs, int LhsRows = Lhs::RowsAtCompileTime, int RhsCols = Rhs::ColsAtCompileTime>
struct product_coeff_vectorized_dyn_selector
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, typename Lhs::Scalar &res)
  {
    res = lhs.row(row).transpose().cwiseProduct(rhs.col(col)).sum();
  }
};

// NOTE the 3 following specializations are because taking .col(0) on a vector is a bit slower
// NOTE maybe they are now useless since we have a specialization for Block<Matrix>
template<typename Lhs, typename Rhs, int RhsCols>
struct product_coeff_vectorized_dyn_selector<Lhs,Rhs,1,RhsCols>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index /*row*/, Index col, const Lhs& lhs, const Rhs& rhs, typename Lhs::Scalar &res)
  {
    res = lhs.transpose().cwiseProduct(rhs.col(col)).sum();
  }
};

template<typename Lhs, typename Rhs, int LhsRows>
struct product_coeff_vectorized_dyn_selector<Lhs,Rhs,LhsRows,1>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index /*col*/, const Lhs& lhs, const Rhs& rhs, typename Lhs::Scalar &res)
  {
    res = lhs.row(row).transpose().cwiseProduct(rhs).sum();
  }
};

template<typename Lhs, typename Rhs>
struct product_coeff_vectorized_dyn_selector<Lhs,Rhs,1,1>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index /*row*/, Index /*col*/, const Lhs& lhs, const Rhs& rhs, typename Lhs::Scalar &res)
  {
    res = lhs.transpose().cwiseProduct(rhs).sum();
  }
};

template<typename Lhs, typename Rhs, typename RetScalar>
struct product_coeff_impl<InnerVectorizedTraversal, Dynamic, Lhs, Rhs, RetScalar>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, typename Lhs::Scalar &res)
  {
    product_coeff_vectorized_dyn_selector<Lhs,Rhs>::run(row, col, lhs, rhs, res);
  }
};

/*******************
*** Packet path  ***
*******************/

template<int UnrollingIndex, typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct product_packet_impl<RowMajor, UnrollingIndex, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Packet &res)
  {
    product_packet_impl<RowMajor, UnrollingIndex-1, Lhs, Rhs, Packet, LoadMode>::run(row, col, lhs, rhs, res);
    res =  pmadd(pset1<Packet>(lhs.coeff(row, UnrollingIndex)), rhs.template packet<LoadMode>(UnrollingIndex, col), res);
  }
};

template<int UnrollingIndex, typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct product_packet_impl<ColMajor, UnrollingIndex, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Packet &res)
  {
    product_packet_impl<ColMajor, UnrollingIndex-1, Lhs, Rhs, Packet, LoadMode>::run(row, col, lhs, rhs, res);
    res =  pmadd(lhs.template packet<LoadMode>(row, UnrollingIndex), pset1<Packet>(rhs.coeff(UnrollingIndex, col)), res);
  }
};

template<typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct product_packet_impl<RowMajor, 0, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Packet &res)
  {
    res = pmul(pset1<Packet>(lhs.coeff(row, 0)),rhs.template packet<LoadMode>(0, col));
  }
};

template<typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct product_packet_impl<ColMajor, 0, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Packet &res)
  {
    res = pmul(lhs.template packet<LoadMode>(row, 0), pset1<Packet>(rhs.coeff(0, col)));
  }
};

template<typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct product_packet_impl<RowMajor, Dynamic, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Packet& res)
  {
    eigen_assert(lhs.cols()>0 && "you are using a non initialized matrix");
    res = pmul(pset1<Packet>(lhs.coeff(row, 0)),rhs.template packet<LoadMode>(0, col));
      for(Index i = 1; i < lhs.cols(); ++i)
        res =  pmadd(pset1<Packet>(lhs.coeff(row, i)), rhs.template packet<LoadMode>(i, col), res);
  }
};

template<typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct product_packet_impl<ColMajor, Dynamic, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Packet& res)
  {
    eigen_assert(lhs.cols()>0 && "you are using a non initialized matrix");
    res = pmul(lhs.template packet<LoadMode>(row, 0), pset1<Packet>(rhs.coeff(0, col)));
      for(Index i = 1; i < lhs.cols(); ++i)
        res =  pmadd(lhs.template packet<LoadMode>(row, i), pset1<Packet>(rhs.coeff(i, col)), res);
  }
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_COEFFBASED_PRODUCT_H
