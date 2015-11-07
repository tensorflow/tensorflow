// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SELFADJOINT_MATRIX_VECTOR_H
#define EIGEN_SELFADJOINT_MATRIX_VECTOR_H

namespace Eigen { 

namespace internal {

/* Optimized selfadjoint matrix * vector product:
 * This algorithm processes 2 columns at onces that allows to both reduce
 * the number of load/stores of the result by a factor 2 and to reduce
 * the instruction dependency.
 */

template<typename Scalar, typename Index, int StorageOrder, int UpLo, bool ConjugateLhs, bool ConjugateRhs, int Version=Specialized>
struct selfadjoint_matrix_vector_product;

template<typename Scalar, typename Index, int StorageOrder, int UpLo, bool ConjugateLhs, bool ConjugateRhs, int Version>
struct selfadjoint_matrix_vector_product

{
static EIGEN_DONT_INLINE void run(
  Index size,
  const Scalar*  lhs, Index lhsStride,
  const Scalar* _rhs, Index rhsIncr,
  Scalar* res,
  Scalar alpha);
};

template<typename Scalar, typename Index, int StorageOrder, int UpLo, bool ConjugateLhs, bool ConjugateRhs, int Version>
EIGEN_DONT_INLINE void selfadjoint_matrix_vector_product<Scalar,Index,StorageOrder,UpLo,ConjugateLhs,ConjugateRhs,Version>::run(
  Index size,
  const Scalar*  lhs, Index lhsStride,
  const Scalar* _rhs, Index rhsIncr,
  Scalar* res,
  Scalar alpha)
{
  typedef typename packet_traits<Scalar>::type Packet;
  const Index PacketSize = sizeof(Packet)/sizeof(Scalar);

  enum {
    IsRowMajor = StorageOrder==RowMajor ? 1 : 0,
    IsLower = UpLo == Lower ? 1 : 0,
    FirstTriangular = IsRowMajor == IsLower
  };

  conj_helper<Scalar,Scalar,NumTraits<Scalar>::IsComplex && EIGEN_LOGICAL_XOR(ConjugateLhs,  IsRowMajor), ConjugateRhs> cj0;
  conj_helper<Scalar,Scalar,NumTraits<Scalar>::IsComplex && EIGEN_LOGICAL_XOR(ConjugateLhs, !IsRowMajor), ConjugateRhs> cj1;
  conj_helper<Scalar,Scalar,NumTraits<Scalar>::IsComplex, ConjugateRhs> cjd;

  conj_helper<Packet,Packet,NumTraits<Scalar>::IsComplex && EIGEN_LOGICAL_XOR(ConjugateLhs,  IsRowMajor), ConjugateRhs> pcj0;
  conj_helper<Packet,Packet,NumTraits<Scalar>::IsComplex && EIGEN_LOGICAL_XOR(ConjugateLhs, !IsRowMajor), ConjugateRhs> pcj1;

  Scalar cjAlpha = ConjugateRhs ? numext::conj(alpha) : alpha;

  // FIXME this copy is now handled outside product_selfadjoint_vector, so it could probably be removed.
  // if the rhs is not sequentially stored in memory we copy it to a temporary buffer,
  // this is because we need to extract packets
  ei_declare_aligned_stack_constructed_variable(Scalar,rhs,size,rhsIncr==1 ? const_cast<Scalar*>(_rhs) : 0);  
  if (rhsIncr!=1)
  {
    const Scalar* it = _rhs;
    for (Index i=0; i<size; ++i, it+=rhsIncr)
      rhs[i] = *it;
  }

  Index bound = (std::max)(Index(0),size-8) & 0xfffffffe;
  if (FirstTriangular)
    bound = size - bound;

  for (Index j=FirstTriangular ? bound : 0;
       j<(FirstTriangular ? size : bound);j+=2)
  {
    const Scalar* EIGEN_RESTRICT A0 = lhs + j*lhsStride;
    const Scalar* EIGEN_RESTRICT A1 = lhs + (j+1)*lhsStride;

    Scalar t0 = cjAlpha * rhs[j];
    Packet ptmp0 = pset1<Packet>(t0);
    Scalar t1 = cjAlpha * rhs[j+1];
    Packet ptmp1 = pset1<Packet>(t1);

    Scalar t2(0);
    Packet ptmp2 = pset1<Packet>(t2);
    Scalar t3(0);
    Packet ptmp3 = pset1<Packet>(t3);

    size_t starti = FirstTriangular ? 0 : j+2;
    size_t endi   = FirstTriangular ? j : size;
    size_t alignedStart = (starti) + internal::first_aligned(&res[starti], endi-starti);
    size_t alignedEnd = alignedStart + ((endi-alignedStart)/(PacketSize))*(PacketSize);

    // TODO make sure this product is a real * complex and that the rhs is properly conjugated if needed
    res[j]   += cjd.pmul(numext::real(A0[j]), t0);
    res[j+1] += cjd.pmul(numext::real(A1[j+1]), t1);
    if(FirstTriangular)
    {
      res[j]   += cj0.pmul(A1[j],   t1);
      t3       += cj1.pmul(A1[j],   rhs[j]);
    }
    else
    {
      res[j+1] += cj0.pmul(A0[j+1],t0);
      t2 += cj1.pmul(A0[j+1], rhs[j+1]);
    }

    for (size_t i=starti; i<alignedStart; ++i)
    {
      res[i] += cj0.pmul(A0[i], t0) + cj0.pmul(A1[i],t1);
      t2 += cj1.pmul(A0[i], rhs[i]);
      t3 += cj1.pmul(A1[i], rhs[i]);
    }
    // Yes this an optimization for gcc 4.3 and 4.4 (=> huge speed up)
    // gcc 4.2 does this optimization automatically.
    const Scalar* EIGEN_RESTRICT a0It  = A0  + alignedStart;
    const Scalar* EIGEN_RESTRICT a1It  = A1  + alignedStart;
    const Scalar* EIGEN_RESTRICT rhsIt = rhs + alignedStart;
          Scalar* EIGEN_RESTRICT resIt = res + alignedStart;
    for (size_t i=alignedStart; i<alignedEnd; i+=PacketSize)
    {
      Packet A0i = ploadu<Packet>(a0It);  a0It  += PacketSize;
      Packet A1i = ploadu<Packet>(a1It);  a1It  += PacketSize;
      Packet Bi  = ploadu<Packet>(rhsIt); rhsIt += PacketSize; // FIXME should be aligned in most cases
      Packet Xi  = pload <Packet>(resIt);

      Xi    = pcj0.pmadd(A0i,ptmp0, pcj0.pmadd(A1i,ptmp1,Xi));
      ptmp2 = pcj1.pmadd(A0i,  Bi, ptmp2);
      ptmp3 = pcj1.pmadd(A1i,  Bi, ptmp3);
      pstore(resIt,Xi); resIt += PacketSize;
    }
    for (size_t i=alignedEnd; i<endi; i++)
    {
      res[i] += cj0.pmul(A0[i], t0) + cj0.pmul(A1[i],t1);
      t2 += cj1.pmul(A0[i], rhs[i]);
      t3 += cj1.pmul(A1[i], rhs[i]);
    }

    res[j]   += alpha * (t2 + predux(ptmp2));
    res[j+1] += alpha * (t3 + predux(ptmp3));
  }
  for (Index j=FirstTriangular ? 0 : bound;j<(FirstTriangular ? bound : size);j++)
  {
    const Scalar* EIGEN_RESTRICT A0 = lhs + j*lhsStride;

    Scalar t1 = cjAlpha * rhs[j];
    Scalar t2(0);
    // TODO make sure this product is a real * complex and that the rhs is properly conjugated if needed
    res[j] += cjd.pmul(numext::real(A0[j]), t1);
    for (Index i=FirstTriangular ? 0 : j+1; i<(FirstTriangular ? j : size); i++)
    {
      res[i] += cj0.pmul(A0[i], t1);
      t2 += cj1.pmul(A0[i], rhs[i]);
    }
    res[j] += alpha * t2;
  }
}

} // end namespace internal 

/***************************************************************************
* Wrapper to product_selfadjoint_vector
***************************************************************************/

namespace internal {
template<typename Lhs, int LhsMode, typename Rhs>
struct traits<SelfadjointProductMatrix<Lhs,LhsMode,false,Rhs,0,true> >
  : traits<ProductBase<SelfadjointProductMatrix<Lhs,LhsMode,false,Rhs,0,true>, Lhs, Rhs> >
{};
}

template<typename Lhs, int LhsMode, typename Rhs>
struct SelfadjointProductMatrix<Lhs,LhsMode,false,Rhs,0,true>
  : public ProductBase<SelfadjointProductMatrix<Lhs,LhsMode,false,Rhs,0,true>, Lhs, Rhs >
{
  EIGEN_PRODUCT_PUBLIC_INTERFACE(SelfadjointProductMatrix)

  enum {
    LhsUpLo = LhsMode&(Upper|Lower)
  };

  SelfadjointProductMatrix(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs) {}

  template<typename Dest> void scaleAndAddTo(Dest& dest, const Scalar& alpha) const
  {
    typedef typename Dest::Scalar ResScalar;
    typedef typename Base::RhsScalar RhsScalar;
    typedef Map<Matrix<ResScalar,Dynamic,1>, Aligned> MappedDest;
    
    eigen_assert(dest.rows()==m_lhs.rows() && dest.cols()==m_rhs.cols());

    typename internal::add_const_on_value_type<ActualLhsType>::type lhs = LhsBlasTraits::extract(m_lhs);
    typename internal::add_const_on_value_type<ActualRhsType>::type rhs = RhsBlasTraits::extract(m_rhs);

    Scalar actualAlpha = alpha * LhsBlasTraits::extractScalarFactor(m_lhs)
                               * RhsBlasTraits::extractScalarFactor(m_rhs);

    enum {
      EvalToDest = (Dest::InnerStrideAtCompileTime==1),
      UseRhs = (_ActualRhsType::InnerStrideAtCompileTime==1)
    };
    
    internal::gemv_static_vector_if<ResScalar,Dest::SizeAtCompileTime,Dest::MaxSizeAtCompileTime,!EvalToDest> static_dest;
    internal::gemv_static_vector_if<RhsScalar,_ActualRhsType::SizeAtCompileTime,_ActualRhsType::MaxSizeAtCompileTime,!UseRhs> static_rhs;

    ei_declare_aligned_stack_constructed_variable(ResScalar,actualDestPtr,dest.size(),
                                                  EvalToDest ? dest.data() : static_dest.data());
                                                  
    ei_declare_aligned_stack_constructed_variable(RhsScalar,actualRhsPtr,rhs.size(),
        UseRhs ? const_cast<RhsScalar*>(rhs.data()) : static_rhs.data());
    
    if(!EvalToDest)
    {
      #ifdef EIGEN_DENSE_STORAGE_CTOR_PLUGIN
      int size = dest.size();
      EIGEN_DENSE_STORAGE_CTOR_PLUGIN
      #endif
      MappedDest(actualDestPtr, dest.size()) = dest;
    }
      
    if(!UseRhs)
    {
      #ifdef EIGEN_DENSE_STORAGE_CTOR_PLUGIN
      int size = rhs.size();
      EIGEN_DENSE_STORAGE_CTOR_PLUGIN
      #endif
      Map<typename _ActualRhsType::PlainObject>(actualRhsPtr, rhs.size()) = rhs;
    }
      
      
    internal::selfadjoint_matrix_vector_product<Scalar, Index, (internal::traits<_ActualLhsType>::Flags&RowMajorBit) ? RowMajor : ColMajor, int(LhsUpLo), bool(LhsBlasTraits::NeedToConjugate), bool(RhsBlasTraits::NeedToConjugate)>::run
      (
        lhs.rows(),                             // size
        &lhs.coeffRef(0,0),  lhs.outerStride(), // lhs info
        actualRhsPtr, 1,                        // rhs info
        actualDestPtr,                          // result info
        actualAlpha                             // scale factor
      );
    
    if(!EvalToDest)
      dest = MappedDest(actualDestPtr, dest.size());
  }
};

namespace internal {
template<typename Lhs, typename Rhs, int RhsMode>
struct traits<SelfadjointProductMatrix<Lhs,0,true,Rhs,RhsMode,false> >
  : traits<ProductBase<SelfadjointProductMatrix<Lhs,0,true,Rhs,RhsMode,false>, Lhs, Rhs> >
{};
}

template<typename Lhs, typename Rhs, int RhsMode>
struct SelfadjointProductMatrix<Lhs,0,true,Rhs,RhsMode,false>
  : public ProductBase<SelfadjointProductMatrix<Lhs,0,true,Rhs,RhsMode,false>, Lhs, Rhs >
{
  EIGEN_PRODUCT_PUBLIC_INTERFACE(SelfadjointProductMatrix)

  enum {
    RhsUpLo = RhsMode&(Upper|Lower)
  };

  SelfadjointProductMatrix(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs) {}

  template<typename Dest> void scaleAndAddTo(Dest& dest, const Scalar& alpha) const
  {
    // let's simply transpose the product
    Transpose<Dest> destT(dest);
    SelfadjointProductMatrix<Transpose<const Rhs>, int(RhsUpLo)==Upper ? Lower : Upper, false,
                             Transpose<const Lhs>, 0, true>(m_rhs.transpose(), m_lhs.transpose()).scaleAndAddTo(destT, alpha);
  }
};

} // end namespace Eigen

#endif // EIGEN_SELFADJOINT_MATRIX_VECTOR_H
