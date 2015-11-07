// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GENERAL_MATRIX_MATRIX_TRIANGULAR_H
#define EIGEN_GENERAL_MATRIX_MATRIX_TRIANGULAR_H

namespace Eigen { 

template<typename Scalar, typename Index, int StorageOrder, int UpLo, bool ConjLhs, bool ConjRhs>
struct selfadjoint_rank1_update;

namespace internal {

/**********************************************************************
* This file implements a general A * B product while
* evaluating only one triangular part of the product.
* This is more general version of self adjoint product (C += A A^T)
* as the level 3 SYRK Blas routine.
**********************************************************************/

// forward declarations (defined at the end of this file)
template<typename LhsScalar, typename RhsScalar, typename Index, int mr, int nr, bool ConjLhs, bool ConjRhs, int UpLo>
struct tribb_kernel;
  
/* Optimized matrix-matrix product evaluating only one triangular half */
template <typename Index,
          typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
          typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs,
                              int ResStorageOrder, int  UpLo, int Version = Specialized>
struct general_matrix_matrix_triangular_product;

// as usual if the result is row major => we transpose the product
template <typename Index, typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
                          typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs, int  UpLo, int Version>
struct general_matrix_matrix_triangular_product<Index,LhsScalar,LhsStorageOrder,ConjugateLhs,RhsScalar,RhsStorageOrder,ConjugateRhs,RowMajor,UpLo,Version>
{
  typedef typename scalar_product_traits<LhsScalar, RhsScalar>::ReturnType ResScalar;
  static EIGEN_STRONG_INLINE void run(Index size, Index depth,const LhsScalar* lhs, Index lhsStride,
                                      const RhsScalar* rhs, Index rhsStride, ResScalar* res, Index resStride, const ResScalar& alpha)
  {
    general_matrix_matrix_triangular_product<Index,
        RhsScalar, RhsStorageOrder==RowMajor ? ColMajor : RowMajor, ConjugateRhs,
        LhsScalar, LhsStorageOrder==RowMajor ? ColMajor : RowMajor, ConjugateLhs,
        ColMajor, UpLo==Lower?Upper:Lower>
      ::run(size,depth,rhs,rhsStride,lhs,lhsStride,res,resStride,alpha);
  }
};

template <typename Index, typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
                          typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs, int  UpLo, int Version>
struct general_matrix_matrix_triangular_product<Index,LhsScalar,LhsStorageOrder,ConjugateLhs,RhsScalar,RhsStorageOrder,ConjugateRhs,ColMajor,UpLo,Version>
{
  typedef typename scalar_product_traits<LhsScalar, RhsScalar>::ReturnType ResScalar;
  static EIGEN_STRONG_INLINE void run(Index size, Index depth,const LhsScalar* _lhs, Index lhsStride,
                                      const RhsScalar* _rhs, Index rhsStride, ResScalar* _res, Index resStride, const ResScalar& alpha)
  {
    typedef gebp_traits<LhsScalar,RhsScalar> Traits;

    typedef const_blas_data_mapper<LhsScalar, Index, LhsStorageOrder> LhsMapper;
    typedef const_blas_data_mapper<RhsScalar, Index, RhsStorageOrder> RhsMapper;
    typedef blas_data_mapper<typename Traits::ResScalar, Index, ColMajor> ResMapper;
    LhsMapper lhs(_lhs,lhsStride);
    RhsMapper rhs(_rhs,rhsStride);
    ResMapper res(_res, resStride);

    Index kc = depth; // cache block size along the K direction
    Index mc = size;  // cache block size along the M direction
    Index nc = size;  // cache block size along the N direction
    computeProductBlockingSizes<LhsScalar,RhsScalar>(kc, mc, nc, Index(1));
    // !!! mc must be a multiple of nr:
    if(mc > Traits::nr)
      mc = (mc/Traits::nr)*Traits::nr;

    ei_declare_aligned_stack_constructed_variable(LhsScalar, blockA, kc*mc, 0);
    ei_declare_aligned_stack_constructed_variable(RhsScalar, blockB, kc*size, 0);

    gemm_pack_lhs<LhsScalar, Index, LhsMapper, Traits::mr, Traits::LhsProgress, LhsStorageOrder> pack_lhs;
    gemm_pack_rhs<RhsScalar, Index, RhsMapper, Traits::nr, RhsStorageOrder> pack_rhs;
    gebp_kernel<LhsScalar, RhsScalar, Index, ResMapper, Traits::mr, Traits::nr, ConjugateLhs, ConjugateRhs> gebp;
    tribb_kernel<LhsScalar, RhsScalar, Index, Traits::mr, Traits::nr, ConjugateLhs, ConjugateRhs, UpLo> sybb;

    for(Index k2=0; k2<depth; k2+=kc)
    {
      const Index actual_kc = (std::min)(k2+kc,depth)-k2;

      // note that the actual rhs is the transpose/adjoint of mat
      pack_rhs(blockB, rhs.getSubMapper(k2,0), actual_kc, size);

      for(Index i2=0; i2<size; i2+=mc)
      {
        const Index actual_mc = (std::min)(i2+mc,size)-i2;

        pack_lhs(blockA, lhs.getSubMapper(i2, k2), actual_kc, actual_mc);

        // the selected actual_mc * size panel of res is split into three different part:
        //  1 - before the diagonal => processed with gebp or skipped
        //  2 - the actual_mc x actual_mc symmetric block => processed with a special kernel
        //  3 - after the diagonal => processed with gebp or skipped
        if (UpLo==Lower)
          gebp(res.getSubMapper(i2, 0), blockA, blockB, actual_mc, actual_kc,
               (std::min)(size,i2), alpha, -1, -1, 0, 0);


        sybb(_res+resStride*i2 + i2, resStride, blockA, blockB + actual_kc*i2, actual_mc, actual_kc, alpha);

        if (UpLo==Upper)
        {
          Index j2 = i2+actual_mc;
          gebp(res.getSubMapper(i2, j2), blockA, blockB+actual_kc*j2, actual_mc,
               actual_kc, (std::max)(Index(0), size-j2), alpha, -1, -1, 0, 0);
        }
      }
    }
  }
};

// Optimized packed Block * packed Block product kernel evaluating only one given triangular part
// This kernel is built on top of the gebp kernel:
// - the current destination block is processed per panel of actual_mc x BlockSize
//   where BlockSize is set to the minimal value allowing gebp to be as fast as possible
// - then, as usual, each panel is split into three parts along the diagonal,
//   the sub blocks above and below the diagonal are processed as usual,
//   while the triangular block overlapping the diagonal is evaluated into a
//   small temporary buffer which is then accumulated into the result using a
//   triangular traversal.
template<typename LhsScalar, typename RhsScalar, typename Index, int mr, int nr, bool ConjLhs, bool ConjRhs, int UpLo>
struct tribb_kernel
{
  typedef gebp_traits<LhsScalar,RhsScalar,ConjLhs,ConjRhs> Traits;
  typedef typename Traits::ResScalar ResScalar;

  enum {
    BlockSize  = EIGEN_PLAIN_ENUM_MAX(mr,nr)
  };
  void operator()(ResScalar* _res, Index resStride, const LhsScalar* blockA, const RhsScalar* blockB, Index size, Index depth, const ResScalar& alpha)
  {
    typedef blas_data_mapper<ResScalar, Index, ColMajor> ResMapper;
    ResMapper res(_res, resStride);
    gebp_kernel<LhsScalar, RhsScalar, Index, ResMapper, mr, nr, ConjLhs, ConjRhs> gebp_kernel;

    Matrix<ResScalar,BlockSize,BlockSize,ColMajor> buffer;

    // let's process the block per panel of actual_mc x BlockSize,
    // again, each is split into three parts, etc.
    for (Index j=0; j<size; j+=BlockSize)
    {
      Index actualBlockSize = std::min<Index>(BlockSize,size - j);
      const RhsScalar* actual_b = blockB+j*depth;

      if(UpLo==Upper)
        gebp_kernel(res.getSubMapper(0, j), blockA, actual_b, j, depth, actualBlockSize, alpha,
                    -1, -1, 0, 0);

      // selfadjoint micro block
      {
        Index i = j;
        buffer.setZero();
        // 1 - apply the kernel on the temporary buffer
        gebp_kernel(ResMapper(buffer.data(), BlockSize), blockA+depth*i, actual_b, actualBlockSize, depth, actualBlockSize, alpha,
                    -1, -1, 0, 0);
        // 2 - triangular accumulation
        for(Index j1=0; j1<actualBlockSize; ++j1)
        {
          ResScalar* r = &res(i, j + j1);
          for(Index i1=UpLo==Lower ? j1 : 0;
              UpLo==Lower ? i1<actualBlockSize : i1<=j1; ++i1)
            r[i1] += buffer(i1,j1);
        }
      }

      if(UpLo==Lower)
      {
        Index i = j+actualBlockSize;
        gebp_kernel(res.getSubMapper(i, j), blockA+depth*i, actual_b, size-i, 
                    depth, actualBlockSize, alpha, -1, -1, 0, 0);
      }
    }
  }
};

} // end namespace internal

// high level API

template<typename MatrixType, typename ProductType, int UpLo, bool IsOuterProduct>
struct general_product_to_triangular_selector;


template<typename MatrixType, typename ProductType, int UpLo>
struct general_product_to_triangular_selector<MatrixType,ProductType,UpLo,true>
{
  static void run(MatrixType& mat, const ProductType& prod, const typename MatrixType::Scalar& alpha)
  {
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Index Index;
    
    typedef typename internal::remove_all<typename ProductType::LhsNested>::type Lhs;
    typedef internal::blas_traits<Lhs> LhsBlasTraits;
    typedef typename LhsBlasTraits::DirectLinearAccessType ActualLhs;
    typedef typename internal::remove_all<ActualLhs>::type _ActualLhs;
    typename internal::add_const_on_value_type<ActualLhs>::type actualLhs = LhsBlasTraits::extract(prod.lhs());
    
    typedef typename internal::remove_all<typename ProductType::RhsNested>::type Rhs;
    typedef internal::blas_traits<Rhs> RhsBlasTraits;
    typedef typename RhsBlasTraits::DirectLinearAccessType ActualRhs;
    typedef typename internal::remove_all<ActualRhs>::type _ActualRhs;
    typename internal::add_const_on_value_type<ActualRhs>::type actualRhs = RhsBlasTraits::extract(prod.rhs());

    Scalar actualAlpha = alpha * LhsBlasTraits::extractScalarFactor(prod.lhs().derived()) * RhsBlasTraits::extractScalarFactor(prod.rhs().derived());

    enum {
      StorageOrder = (internal::traits<MatrixType>::Flags&RowMajorBit) ? RowMajor : ColMajor,
      UseLhsDirectly = _ActualLhs::InnerStrideAtCompileTime==1,
      UseRhsDirectly = _ActualRhs::InnerStrideAtCompileTime==1
    };
    
    internal::gemv_static_vector_if<Scalar,Lhs::SizeAtCompileTime,Lhs::MaxSizeAtCompileTime,!UseLhsDirectly> static_lhs;
    ei_declare_aligned_stack_constructed_variable(Scalar, actualLhsPtr, actualLhs.size(),
      (UseLhsDirectly ? const_cast<Scalar*>(actualLhs.data()) : static_lhs.data()));
    if(!UseLhsDirectly) Map<typename _ActualLhs::PlainObject>(actualLhsPtr, actualLhs.size()) = actualLhs;
    
    internal::gemv_static_vector_if<Scalar,Rhs::SizeAtCompileTime,Rhs::MaxSizeAtCompileTime,!UseRhsDirectly> static_rhs;
    ei_declare_aligned_stack_constructed_variable(Scalar, actualRhsPtr, actualRhs.size(),
      (UseRhsDirectly ? const_cast<Scalar*>(actualRhs.data()) : static_rhs.data()));
    if(!UseRhsDirectly) Map<typename _ActualRhs::PlainObject>(actualRhsPtr, actualRhs.size()) = actualRhs;
    
    
    selfadjoint_rank1_update<Scalar,Index,StorageOrder,UpLo,
                              LhsBlasTraits::NeedToConjugate && NumTraits<Scalar>::IsComplex,
                              RhsBlasTraits::NeedToConjugate && NumTraits<Scalar>::IsComplex>
          ::run(actualLhs.size(), mat.data(), mat.outerStride(), actualLhsPtr, actualRhsPtr, actualAlpha);
  }
};

template<typename MatrixType, typename ProductType, int UpLo>
struct general_product_to_triangular_selector<MatrixType,ProductType,UpLo,false>
{
  static void run(MatrixType& mat, const ProductType& prod, const typename MatrixType::Scalar& alpha)
  {
    typedef typename MatrixType::Index Index;
    
    typedef typename internal::remove_all<typename ProductType::LhsNested>::type Lhs;
    typedef internal::blas_traits<Lhs> LhsBlasTraits;
    typedef typename LhsBlasTraits::DirectLinearAccessType ActualLhs;
    typedef typename internal::remove_all<ActualLhs>::type _ActualLhs;
    typename internal::add_const_on_value_type<ActualLhs>::type actualLhs = LhsBlasTraits::extract(prod.lhs());
    
    typedef typename internal::remove_all<typename ProductType::RhsNested>::type Rhs;
    typedef internal::blas_traits<Rhs> RhsBlasTraits;
    typedef typename RhsBlasTraits::DirectLinearAccessType ActualRhs;
    typedef typename internal::remove_all<ActualRhs>::type _ActualRhs;
    typename internal::add_const_on_value_type<ActualRhs>::type actualRhs = RhsBlasTraits::extract(prod.rhs());

    typename ProductType::Scalar actualAlpha = alpha * LhsBlasTraits::extractScalarFactor(prod.lhs().derived()) * RhsBlasTraits::extractScalarFactor(prod.rhs().derived());

    internal::general_matrix_matrix_triangular_product<Index,
      typename Lhs::Scalar, _ActualLhs::Flags&RowMajorBit ? RowMajor : ColMajor, LhsBlasTraits::NeedToConjugate,
      typename Rhs::Scalar, _ActualRhs::Flags&RowMajorBit ? RowMajor : ColMajor, RhsBlasTraits::NeedToConjugate,
      MatrixType::Flags&RowMajorBit ? RowMajor : ColMajor, UpLo>
      ::run(mat.cols(), actualLhs.cols(),
            &actualLhs.coeffRef(0,0), actualLhs.outerStride(), &actualRhs.coeffRef(0,0), actualRhs.outerStride(),
            mat.data(), mat.outerStride(), actualAlpha);
  }
};

template<typename MatrixType, unsigned int UpLo>
template<typename ProductDerived, typename _Lhs, typename _Rhs>
TriangularView<MatrixType,UpLo>& TriangularView<MatrixType,UpLo>::assignProduct(const ProductBase<ProductDerived, _Lhs,_Rhs>& prod, const Scalar& alpha)
{
  eigen_assert(m_matrix.rows() == prod.rows() && m_matrix.cols() == prod.cols());

  general_product_to_triangular_selector<MatrixType, ProductDerived, UpLo, (_Lhs::ColsAtCompileTime==1) || (_Rhs::RowsAtCompileTime==1)>::run(m_matrix.const_cast_derived(), prod.derived(), alpha);
  
  return *this;
}

} // end namespace Eigen

#endif // EIGEN_GENERAL_MATRIX_MATRIX_TRIANGULAR_H
