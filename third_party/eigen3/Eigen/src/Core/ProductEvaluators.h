// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2011 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef EIGEN_PRODUCTEVALUATORS_H
#define EIGEN_PRODUCTEVALUATORS_H

namespace Eigen {
  
namespace internal {
  
// We can evaluate the product either all at once, like GeneralProduct and its evalTo() function, or
// traverse the matrix coefficient by coefficient, like CoeffBasedProduct.  Use the existing logic
// in ProductReturnType to decide.

template<typename XprType, typename ProductType>
struct product_evaluator_dispatcher;

template<typename Lhs, typename Rhs>
struct evaluator_impl<Product<Lhs, Rhs> >
  : product_evaluator_dispatcher<Product<Lhs, Rhs>, typename ProductReturnType<Lhs, Rhs>::Type> 
{
  typedef Product<Lhs, Rhs> XprType;
  typedef product_evaluator_dispatcher<XprType, typename ProductReturnType<Lhs, Rhs>::Type> Base;

  evaluator_impl(const XprType& xpr) : Base(xpr) 
  { }
};

template<typename XprType, typename ProductType>
struct product_evaluator_traits_dispatcher;

template<typename Lhs, typename Rhs>
struct evaluator_traits<Product<Lhs, Rhs> >
  : product_evaluator_traits_dispatcher<Product<Lhs, Rhs>, typename ProductReturnType<Lhs, Rhs>::Type> 
{ 
  static const int AssumeAliasing = 1;
};

// Case 1: Evaluate all at once
//
// We can view the GeneralProduct class as a part of the product evaluator. 
// Four sub-cases: InnerProduct, OuterProduct, GemmProduct and GemvProduct.
// InnerProduct is special because GeneralProduct does not have an evalTo() method in this case.

template<typename Lhs, typename Rhs>
struct product_evaluator_traits_dispatcher<Product<Lhs, Rhs>, GeneralProduct<Lhs, Rhs, InnerProduct> > 
{
  static const int HasEvalTo = 0;
};

template<typename Lhs, typename Rhs>
struct product_evaluator_dispatcher<Product<Lhs, Rhs>, GeneralProduct<Lhs, Rhs, InnerProduct> > 
  : public evaluator<typename Product<Lhs, Rhs>::PlainObject>::type
{
  typedef Product<Lhs, Rhs> XprType;
  typedef typename XprType::PlainObject PlainObject;
  typedef typename evaluator<PlainObject>::type evaluator_base;

  // TODO: Computation is too early (?)
  product_evaluator_dispatcher(const XprType& xpr) : evaluator_base(m_result)
  {
    m_result.coeffRef(0,0) = (xpr.lhs().transpose().cwiseProduct(xpr.rhs())).sum();
  }
  
protected:  
  PlainObject m_result;
};

// For the other three subcases, simply call the evalTo() method of GeneralProduct
// TODO: GeneralProduct should take evaluators, not expression objects.

template<typename Lhs, typename Rhs, int ProductType>
struct product_evaluator_traits_dispatcher<Product<Lhs, Rhs>, GeneralProduct<Lhs, Rhs, ProductType> > 
{
  static const int HasEvalTo = 1;
};

template<typename Lhs, typename Rhs, int ProductType>
struct product_evaluator_dispatcher<Product<Lhs, Rhs>, GeneralProduct<Lhs, Rhs, ProductType> > 
{
  typedef Product<Lhs, Rhs> XprType;
  typedef typename XprType::PlainObject PlainObject;
  typedef typename evaluator<PlainObject>::type evaluator_base;
  
  product_evaluator_dispatcher(const XprType& xpr) : m_xpr(xpr)
  { }
  
  template<typename DstEvaluatorType, typename DstXprType>
  void evalTo(DstEvaluatorType /* not used */, DstXprType& dst) const
  {
    dst.resize(m_xpr.rows(), m_xpr.cols());
    GeneralProduct<Lhs, Rhs, ProductType>(m_xpr.lhs(), m_xpr.rhs()).evalTo(dst);
  }
  
protected: 
  const XprType& m_xpr;
};

// Case 2: Evaluate coeff by coeff
//
// This is mostly taken from CoeffBasedProduct.h
// The main difference is that we add an extra argument to the etor_product_*_impl::run() function
// for the inner dimension of the product, because evaluator object do not know their size.

template<int Traversal, int UnrollingIndex, typename Lhs, typename Rhs, typename RetScalar>
struct etor_product_coeff_impl;

template<int StorageOrder, int UnrollingIndex, typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct etor_product_packet_impl;

template<typename Lhs, typename Rhs, typename LhsNested, typename RhsNested, int Flags>
struct product_evaluator_traits_dispatcher<Product<Lhs, Rhs>, CoeffBasedProduct<LhsNested, RhsNested, Flags> >
{
  static const int HasEvalTo = 0;
};

template<typename Lhs, typename Rhs, typename LhsNested, typename RhsNested, int Flags>
struct product_evaluator_dispatcher<Product<Lhs, Rhs>, CoeffBasedProduct<LhsNested, RhsNested, Flags> >
  : evaluator_impl_base<Product<Lhs, Rhs> >
{
  typedef Product<Lhs, Rhs> XprType;
  typedef CoeffBasedProduct<LhsNested, RhsNested, Flags> CoeffBasedProductType;

  product_evaluator_dispatcher(const XprType& xpr) 
    : m_lhsImpl(xpr.lhs()), 
      m_rhsImpl(xpr.rhs()),  
      m_innerDim(xpr.lhs().cols())
  { }

  typedef typename XprType::Index Index;
  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketScalar PacketScalar;
  typedef typename XprType::PacketReturnType PacketReturnType;

  // Everything below here is taken from CoeffBasedProduct.h

  enum {
    RowsAtCompileTime = traits<CoeffBasedProductType>::RowsAtCompileTime,
    PacketSize = packet_traits<Scalar>::size,
    InnerSize  = traits<CoeffBasedProductType>::InnerSize,
    CoeffReadCost = traits<CoeffBasedProductType>::CoeffReadCost,
    Unroll = CoeffReadCost != Dynamic && CoeffReadCost <= EIGEN_UNROLLING_LIMIT,
    CanVectorizeInner = traits<CoeffBasedProductType>::CanVectorizeInner
  };

  typedef typename evaluator<Lhs>::type LhsEtorType;
  typedef typename evaluator<Rhs>::type RhsEtorType;
  typedef etor_product_coeff_impl<CanVectorizeInner ? InnerVectorizedTraversal : DefaultTraversal,
                                  Unroll ? InnerSize-1 : Dynamic,
                                  LhsEtorType, RhsEtorType, Scalar> CoeffImpl;

  const CoeffReturnType coeff(Index row, Index col) const
  {
    Scalar res;
    CoeffImpl::run(row, col, m_lhsImpl, m_rhsImpl, m_innerDim, res);
    return res;
  }

  /* Allow index-based non-packet access. It is impossible though to allow index-based packed access,
   * which is why we don't set the LinearAccessBit.
   */
  const CoeffReturnType coeff(Index index) const
  {
    Scalar res;
    const Index row = RowsAtCompileTime == 1 ? 0 : index;
    const Index col = RowsAtCompileTime == 1 ? index : 0;
    CoeffImpl::run(row, col, m_lhsImpl, m_rhsImpl, m_innerDim, res);
    return res;
  }

  template<int LoadMode>
  const PacketReturnType packet(Index row, Index col) const
  {
    PacketScalar res;
    typedef etor_product_packet_impl<Flags&RowMajorBit ? RowMajor : ColMajor,
				     Unroll ? InnerSize-1 : Dynamic,
				     LhsEtorType, RhsEtorType, PacketScalar, LoadMode> PacketImpl;
    PacketImpl::run(row, col, m_lhsImpl, m_rhsImpl, m_innerDim, res);
    return res;
  }

protected:
  typename evaluator<Lhs>::type m_lhsImpl;
  typename evaluator<Rhs>::type m_rhsImpl;

  // TODO: Get rid of m_innerDim if known at compile time
  Index m_innerDim;
};

/***************************************************************************
* Normal product .coeff() implementation (with meta-unrolling)
***************************************************************************/

/**************************************
*** Scalar path  - no vectorization ***
**************************************/

template<int UnrollingIndex, typename Lhs, typename Rhs, typename RetScalar>
struct etor_product_coeff_impl<DefaultTraversal, UnrollingIndex, Lhs, Rhs, RetScalar>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index innerDim, RetScalar &res)
  {
    etor_product_coeff_impl<DefaultTraversal, UnrollingIndex-1, Lhs, Rhs, RetScalar>::run(row, col, lhs, rhs, innerDim, res);
    res += lhs.coeff(row, UnrollingIndex) * rhs.coeff(UnrollingIndex, col);
  }
};

template<typename Lhs, typename Rhs, typename RetScalar>
struct etor_product_coeff_impl<DefaultTraversal, 0, Lhs, Rhs, RetScalar>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index /*innerDim*/, RetScalar &res)
  {
    res = lhs.coeff(row, 0) * rhs.coeff(0, col);
  }
};

template<typename Lhs, typename Rhs, typename RetScalar>
struct etor_product_coeff_impl<DefaultTraversal, Dynamic, Lhs, Rhs, RetScalar>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index innerDim, RetScalar& res)
  {
    eigen_assert(innerDim>0 && "you are using a non initialized matrix");
    res = lhs.coeff(row, 0) * rhs.coeff(0, col);
    for(Index i = 1; i < innerDim; ++i)
      res += lhs.coeff(row, i) * rhs.coeff(i, col);
  }
};

/*******************************************
*** Scalar path with inner vectorization ***
*******************************************/

template<int UnrollingIndex, typename Lhs, typename Rhs, typename Packet>
struct etor_product_coeff_vectorized_unroller
{
  typedef typename Lhs::Index Index;
  enum { PacketSize = packet_traits<typename Lhs::Scalar>::size };
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index innerDim, typename Lhs::PacketScalar &pres)
  {
    etor_product_coeff_vectorized_unroller<UnrollingIndex-PacketSize, Lhs, Rhs, Packet>::run(row, col, lhs, rhs, innerDim, pres);
    pres = padd(pres, pmul( lhs.template packet<Aligned>(row, UnrollingIndex) , rhs.template packet<Aligned>(UnrollingIndex, col) ));
  }
};

template<typename Lhs, typename Rhs, typename Packet>
struct etor_product_coeff_vectorized_unroller<0, Lhs, Rhs, Packet>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index /*innerDim*/, typename Lhs::PacketScalar &pres)
  {
    pres = pmul(lhs.template packet<Aligned>(row, 0) , rhs.template packet<Aligned>(0, col));
  }
};

template<int UnrollingIndex, typename Lhs, typename Rhs, typename RetScalar>
struct etor_product_coeff_impl<InnerVectorizedTraversal, UnrollingIndex, Lhs, Rhs, RetScalar>
{
  typedef typename Lhs::PacketScalar Packet;
  typedef typename Lhs::Index Index;
  enum { PacketSize = packet_traits<typename Lhs::Scalar>::size };
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index innerDim, RetScalar &res)
  {
    Packet pres;
    etor_product_coeff_vectorized_unroller<UnrollingIndex+1-PacketSize, Lhs, Rhs, Packet>::run(row, col, lhs, rhs, innerDim, pres);
    etor_product_coeff_impl<DefaultTraversal,UnrollingIndex,Lhs,Rhs,RetScalar>::run(row, col, lhs, rhs, innerDim, res);
    res = predux(pres);
  }
};

template<typename Lhs, typename Rhs, int LhsRows = Lhs::RowsAtCompileTime, int RhsCols = Rhs::ColsAtCompileTime>
struct etor_product_coeff_vectorized_dyn_selector
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index /*innerDim*/, typename Lhs::Scalar &res)
  {
    res = lhs.row(row).transpose().cwiseProduct(rhs.col(col)).sum();
  }
};

// NOTE the 3 following specializations are because taking .col(0) on a vector is a bit slower
// NOTE maybe they are now useless since we have a specialization for Block<Matrix>
template<typename Lhs, typename Rhs, int RhsCols>
struct etor_product_coeff_vectorized_dyn_selector<Lhs,Rhs,1,RhsCols>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index /*row*/, Index col, const Lhs& lhs, const Rhs& rhs, Index /*innerDim*/, typename Lhs::Scalar &res)
  {
    res = lhs.transpose().cwiseProduct(rhs.col(col)).sum();
  }
};

template<typename Lhs, typename Rhs, int LhsRows>
struct etor_product_coeff_vectorized_dyn_selector<Lhs,Rhs,LhsRows,1>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index /*col*/, const Lhs& lhs, const Rhs& rhs, Index /*innerDim*/, typename Lhs::Scalar &res)
  {
    res = lhs.row(row).transpose().cwiseProduct(rhs).sum();
  }
};

template<typename Lhs, typename Rhs>
struct etor_product_coeff_vectorized_dyn_selector<Lhs,Rhs,1,1>
{
  typedef typename Lhs::Index Index;
  EIGEN_STRONG_INLINE void run(Index /*row*/, Index /*col*/, const Lhs& lhs, const Rhs& rhs, Index /*innerDim*/, typename Lhs::Scalar &res)
  {
    res = lhs.transpose().cwiseProduct(rhs).sum();
  }
};

template<typename Lhs, typename Rhs, typename RetScalar>
struct etor_product_coeff_impl<InnerVectorizedTraversal, Dynamic, Lhs, Rhs, RetScalar>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index innerDim, typename Lhs::Scalar &res)
  {
    etor_product_coeff_vectorized_dyn_selector<Lhs,Rhs>::run(row, col, lhs, rhs, innerDim, res);
  }
};

/*******************
*** Packet path  ***
*******************/

template<int UnrollingIndex, typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct etor_product_packet_impl<RowMajor, UnrollingIndex, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index innerDim, Packet &res)
  {
    etor_product_packet_impl<RowMajor, UnrollingIndex-1, Lhs, Rhs, Packet, LoadMode>::run(row, col, lhs, rhs, innerDim, res);
    res =  pmadd(pset1<Packet>(lhs.coeff(row, UnrollingIndex)), rhs.template packet<LoadMode>(UnrollingIndex, col), res);
  }
};

template<int UnrollingIndex, typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct etor_product_packet_impl<ColMajor, UnrollingIndex, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index innerDim, Packet &res)
  {
    etor_product_packet_impl<ColMajor, UnrollingIndex-1, Lhs, Rhs, Packet, LoadMode>::run(row, col, lhs, rhs, innerDim, res);
    res =  pmadd(lhs.template packet<LoadMode>(row, UnrollingIndex), pset1<Packet>(rhs.coeff(UnrollingIndex, col)), res);
  }
};

template<typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct etor_product_packet_impl<RowMajor, 0, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index /*innerDim*/, Packet &res)
  {
    res = pmul(pset1<Packet>(lhs.coeff(row, 0)),rhs.template packet<LoadMode>(0, col));
  }
};

template<typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct etor_product_packet_impl<ColMajor, 0, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index /*innerDim*/, Packet &res)
  {
    res = pmul(lhs.template packet<LoadMode>(row, 0), pset1<Packet>(rhs.coeff(0, col)));
  }
};

template<typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct etor_product_packet_impl<RowMajor, Dynamic, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index innerDim, Packet& res)
  {
    eigen_assert(innerDim>0 && "you are using a non initialized matrix");
    res = pmul(pset1<Packet>(lhs.coeff(row, 0)),rhs.template packet<LoadMode>(0, col));
    for(Index i = 1; i < innerDim; ++i)
      res =  pmadd(pset1<Packet>(lhs.coeff(row, i)), rhs.template packet<LoadMode>(i, col), res);
  }
};

template<typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct etor_product_packet_impl<ColMajor, Dynamic, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index innerDim, Packet& res)
  {
    eigen_assert(innerDim>0 && "you are using a non initialized matrix");
    res = pmul(lhs.template packet<LoadMode>(row, 0), pset1<Packet>(rhs.coeff(0, col)));
    for(Index i = 1; i < innerDim; ++i)
      res =  pmadd(lhs.template packet<LoadMode>(row, i), pset1<Packet>(rhs.coeff(i, col)), res);
  }
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_PRODUCT_EVALUATORS_H
