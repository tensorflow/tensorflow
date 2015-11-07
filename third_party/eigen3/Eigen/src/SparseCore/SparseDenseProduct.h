// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSEDENSEPRODUCT_H
#define EIGEN_SPARSEDENSEPRODUCT_H

namespace Eigen { 

template<typename Lhs, typename Rhs, int InnerSize> struct SparseDenseProductReturnType
{
  typedef SparseTimeDenseProduct<Lhs,Rhs> Type;
};

template<typename Lhs, typename Rhs> struct SparseDenseProductReturnType<Lhs,Rhs,1>
{
  typedef SparseDenseOuterProduct<Lhs,Rhs,false> Type;
};

template<typename Lhs, typename Rhs, int InnerSize> struct DenseSparseProductReturnType
{
  typedef DenseTimeSparseProduct<Lhs,Rhs> Type;
};

template<typename Lhs, typename Rhs> struct DenseSparseProductReturnType<Lhs,Rhs,1>
{
  typedef SparseDenseOuterProduct<Rhs,Lhs,true> Type;
};

namespace internal {

template<typename Lhs, typename Rhs, bool Tr>
struct traits<SparseDenseOuterProduct<Lhs,Rhs,Tr> >
{
  typedef Sparse StorageKind;
  typedef typename scalar_product_traits<typename traits<Lhs>::Scalar,
                                         typename traits<Rhs>::Scalar>::ReturnType Scalar;
  typedef typename Lhs::Index Index;
  typedef typename Lhs::Nested LhsNested;
  typedef typename Rhs::Nested RhsNested;
  typedef typename remove_all<LhsNested>::type _LhsNested;
  typedef typename remove_all<RhsNested>::type _RhsNested;

  enum {
    LhsCoeffReadCost = traits<_LhsNested>::CoeffReadCost,
    RhsCoeffReadCost = traits<_RhsNested>::CoeffReadCost,

    RowsAtCompileTime    = Tr ? int(traits<Rhs>::RowsAtCompileTime)     : int(traits<Lhs>::RowsAtCompileTime),
    ColsAtCompileTime    = Tr ? int(traits<Lhs>::ColsAtCompileTime)     : int(traits<Rhs>::ColsAtCompileTime),
    MaxRowsAtCompileTime = Tr ? int(traits<Rhs>::MaxRowsAtCompileTime)  : int(traits<Lhs>::MaxRowsAtCompileTime),
    MaxColsAtCompileTime = Tr ? int(traits<Lhs>::MaxColsAtCompileTime)  : int(traits<Rhs>::MaxColsAtCompileTime),

    Flags = Tr ? RowMajorBit : 0,

    CoeffReadCost = LhsCoeffReadCost + RhsCoeffReadCost + NumTraits<Scalar>::MulCost
  };
};

} // end namespace internal

template<typename Lhs, typename Rhs, bool Tr>
class SparseDenseOuterProduct
 : public SparseMatrixBase<SparseDenseOuterProduct<Lhs,Rhs,Tr> >
{
  public:

    typedef SparseMatrixBase<SparseDenseOuterProduct> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(SparseDenseOuterProduct)
    typedef internal::traits<SparseDenseOuterProduct> Traits;

  private:

    typedef typename Traits::LhsNested LhsNested;
    typedef typename Traits::RhsNested RhsNested;
    typedef typename Traits::_LhsNested _LhsNested;
    typedef typename Traits::_RhsNested _RhsNested;

  public:

    class InnerIterator;

    EIGEN_STRONG_INLINE SparseDenseOuterProduct(const Lhs& lhs, const Rhs& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      EIGEN_STATIC_ASSERT(!Tr,YOU_MADE_A_PROGRAMMING_MISTAKE);
    }

    EIGEN_STRONG_INLINE SparseDenseOuterProduct(const Rhs& rhs, const Lhs& lhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      EIGEN_STATIC_ASSERT(Tr,YOU_MADE_A_PROGRAMMING_MISTAKE);
    }

    EIGEN_STRONG_INLINE Index rows() const { return Tr ? m_rhs.rows() : m_lhs.rows(); }
    EIGEN_STRONG_INLINE Index cols() const { return Tr ? m_lhs.cols() : m_rhs.cols(); }

    EIGEN_STRONG_INLINE const _LhsNested& lhs() const { return m_lhs; }
    EIGEN_STRONG_INLINE const _RhsNested& rhs() const { return m_rhs; }

  protected:
    LhsNested m_lhs;
    RhsNested m_rhs;
};

template<typename Lhs, typename Rhs, bool Transpose>
class SparseDenseOuterProduct<Lhs,Rhs,Transpose>::InnerIterator : public _LhsNested::InnerIterator
{
    typedef typename _LhsNested::InnerIterator Base;
    typedef typename SparseDenseOuterProduct::Index Index;
  public:
    EIGEN_STRONG_INLINE InnerIterator(const SparseDenseOuterProduct& prod, Index outer)
      : Base(prod.lhs(), 0), m_outer(outer), m_factor(prod.rhs().coeff(outer))
    {
    }

    inline Index outer() const { return m_outer; }
    inline Index row() const { return Transpose ? Base::row() : m_outer; }
    inline Index col() const { return Transpose ? m_outer : Base::row(); }

    inline Scalar value() const { return Base::value() * m_factor; }

  protected:
    Index m_outer;
    Scalar m_factor;
};

namespace internal {
template<typename Lhs, typename Rhs>
struct traits<SparseTimeDenseProduct<Lhs,Rhs> >
 : traits<ProductBase<SparseTimeDenseProduct<Lhs,Rhs>, Lhs, Rhs> >
{
  typedef Dense StorageKind;
  typedef MatrixXpr XprKind;
};

template<typename SparseLhsType, typename DenseRhsType, typename DenseResType,
         typename AlphaType,
         int LhsStorageOrder = ((SparseLhsType::Flags&RowMajorBit)==RowMajorBit) ? RowMajor : ColMajor,
         bool ColPerCol = ((DenseRhsType::Flags&RowMajorBit)==0) || DenseRhsType::ColsAtCompileTime==1>
struct sparse_time_dense_product_impl;

template<typename SparseLhsType, typename DenseRhsType, typename DenseResType>
struct sparse_time_dense_product_impl<SparseLhsType,DenseRhsType,DenseResType, typename DenseResType::Scalar, RowMajor, true>
{
  typedef typename internal::remove_all<SparseLhsType>::type Lhs;
  typedef typename internal::remove_all<DenseRhsType>::type Rhs;
  typedef typename internal::remove_all<DenseResType>::type Res;
  typedef typename Lhs::Index Index;
  typedef typename Lhs::InnerIterator LhsInnerIterator;
  static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const typename Res::Scalar& alpha)
  {
    for(Index c=0; c<rhs.cols(); ++c)
    {
      Index n = lhs.outerSize();
      for(Index j=0; j<n; ++j)
      {
        typename Res::Scalar tmp(0);
        for(LhsInnerIterator it(lhs,j); it ;++it)
          tmp += it.value() * rhs.coeff(it.index(),c);
        res.coeffRef(j,c) = alpha * tmp;
      }
    }
  }
};

template<typename T1, typename T2/*, int _Options, typename _StrideType*/>
struct scalar_product_traits<T1, Ref<T2/*, _Options, _StrideType*/> >
{
  enum {
    Defined = 1
  };
  typedef typename CwiseUnaryOp<scalar_multiple2_op<T1, typename T2::Scalar>, T2>::PlainObject ReturnType;
};
template<typename SparseLhsType, typename DenseRhsType, typename DenseResType, typename AlphaType>
struct sparse_time_dense_product_impl<SparseLhsType,DenseRhsType,DenseResType, AlphaType, ColMajor, true>
{
  typedef typename internal::remove_all<SparseLhsType>::type Lhs;
  typedef typename internal::remove_all<DenseRhsType>::type Rhs;
  typedef typename internal::remove_all<DenseResType>::type Res;
  typedef typename Lhs::InnerIterator LhsInnerIterator;
  typedef typename Lhs::Index Index;
  static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const AlphaType& alpha)
  {
    for(Index c=0; c<rhs.cols(); ++c)
    {
      for(Index j=0; j<lhs.outerSize(); ++j)
      {
//        typename Res::Scalar rhs_j = alpha * rhs.coeff(j,c);
        typename internal::scalar_product_traits<AlphaType, typename Rhs::Scalar>::ReturnType rhs_j(alpha * rhs.coeff(j,c));
        for(LhsInnerIterator it(lhs,j); it ;++it)
          res.coeffRef(it.index(),c) += it.value() * rhs_j;
      }
    }
  }
};

template<typename SparseLhsType, typename DenseRhsType, typename DenseResType>
struct sparse_time_dense_product_impl<SparseLhsType,DenseRhsType,DenseResType, typename DenseResType::Scalar, RowMajor, false>
{
  typedef typename internal::remove_all<SparseLhsType>::type Lhs;
  typedef typename internal::remove_all<DenseRhsType>::type Rhs;
  typedef typename internal::remove_all<DenseResType>::type Res;
  typedef typename Lhs::InnerIterator LhsInnerIterator;
  typedef typename Lhs::Index Index;
  static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const typename Res::Scalar& alpha)
  {
    for(Index j=0; j<lhs.outerSize(); ++j)
    {
      typename Res::RowXpr res_j(res.row(j));
      for(LhsInnerIterator it(lhs,j); it ;++it)
        res_j += (alpha*it.value()) * rhs.row(it.index());
    }
  }
};

template<typename SparseLhsType, typename DenseRhsType, typename DenseResType>
struct sparse_time_dense_product_impl<SparseLhsType,DenseRhsType,DenseResType, typename DenseResType::Scalar, ColMajor, false>
{
  typedef typename internal::remove_all<SparseLhsType>::type Lhs;
  typedef typename internal::remove_all<DenseRhsType>::type Rhs;
  typedef typename internal::remove_all<DenseResType>::type Res;
  typedef typename Lhs::InnerIterator LhsInnerIterator;
  typedef typename Lhs::Index Index;
  static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const typename Res::Scalar& alpha)
  {
    for(Index j=0; j<lhs.outerSize(); ++j)
    {
      typename Rhs::ConstRowXpr rhs_j(rhs.row(j));
      for(LhsInnerIterator it(lhs,j); it ;++it)
        res.row(it.index()) += (alpha*it.value()) * rhs_j;
    }
  }
};

template<typename SparseLhsType, typename DenseRhsType, typename DenseResType,typename AlphaType>
inline void sparse_time_dense_product(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const AlphaType& alpha)
{
  sparse_time_dense_product_impl<SparseLhsType,DenseRhsType,DenseResType, AlphaType>::run(lhs, rhs, res, alpha);
}

} // end namespace internal

template<typename Lhs, typename Rhs>
class SparseTimeDenseProduct
  : public ProductBase<SparseTimeDenseProduct<Lhs,Rhs>, Lhs, Rhs>
{
  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(SparseTimeDenseProduct)

    SparseTimeDenseProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs)
    {}

    template<typename Dest> void scaleAndAddTo(Dest& dest, const Scalar& alpha) const
    {
      internal::sparse_time_dense_product(m_lhs, m_rhs, dest, alpha);
    }

  private:
    SparseTimeDenseProduct& operator=(const SparseTimeDenseProduct&);
};


// dense = dense * sparse
namespace internal {
template<typename Lhs, typename Rhs>
struct traits<DenseTimeSparseProduct<Lhs,Rhs> >
 : traits<ProductBase<DenseTimeSparseProduct<Lhs,Rhs>, Lhs, Rhs> >
{
  typedef Dense StorageKind;
};
} // end namespace internal

template<typename Lhs, typename Rhs>
class DenseTimeSparseProduct
  : public ProductBase<DenseTimeSparseProduct<Lhs,Rhs>, Lhs, Rhs>
{
  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(DenseTimeSparseProduct)

    DenseTimeSparseProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs)
    {}

    template<typename Dest> void scaleAndAddTo(Dest& dest, const Scalar& alpha) const
    {
      Transpose<const _LhsNested> lhs_t(m_lhs);
      Transpose<const _RhsNested> rhs_t(m_rhs);
      Transpose<Dest> dest_t(dest);
      internal::sparse_time_dense_product(rhs_t, lhs_t, dest_t, alpha);
    }

  private:
    DenseTimeSparseProduct& operator=(const DenseTimeSparseProduct&);
};

// sparse * dense
template<typename Derived>
template<typename OtherDerived>
inline const typename SparseDenseProductReturnType<Derived,OtherDerived>::Type
SparseMatrixBase<Derived>::operator*(const MatrixBase<OtherDerived> &other) const
{
  return typename SparseDenseProductReturnType<Derived,OtherDerived>::Type(derived(), other.derived());
}

} // end namespace Eigen

#endif // EIGEN_SPARSEDENSEPRODUCT_H
