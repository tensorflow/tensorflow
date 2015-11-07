// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GENERAL_PRODUCT_H
#define EIGEN_GENERAL_PRODUCT_H

namespace Eigen {

/** \class GeneralProduct
  * \ingroup Core_Module
  *
  * \brief Expression of the product of two general matrices or vectors
  *
  * \param LhsNested the type used to store the left-hand side
  * \param RhsNested the type used to store the right-hand side
  * \param ProductMode the type of the product
  *
  * This class represents an expression of the product of two general matrices.
  * We call a general matrix, a dense matrix with full storage. For instance,
  * This excludes triangular, selfadjoint, and sparse matrices.
  * It is the return type of the operator* between general matrices. Its template
  * arguments are determined automatically by ProductReturnType. Therefore,
  * GeneralProduct should never be used direclty. To determine the result type of a
  * function which involves a matrix product, use ProductReturnType::Type.
  *
  * \sa ProductReturnType, MatrixBase::operator*(const MatrixBase<OtherDerived>&)
  */
template<typename Lhs, typename Rhs, int ProductType = internal::product_type<Lhs,Rhs>::value>
class GeneralProduct;

enum {
  Large = 2,
  Small = 3
};

namespace internal {

template<int Rows, int Cols, int Depth> struct product_type_selector;

template<int Size, int MaxSize> struct product_size_category
{
  enum { is_large = MaxSize == Dynamic ||
                    Size >= EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD,
         value = is_large  ? Large
               : Size == 1 ? 1
                           : Small
  };
};

template<typename Lhs, typename Rhs> struct product_type
{
  typedef typename remove_all<Lhs>::type _Lhs;
  typedef typename remove_all<Rhs>::type _Rhs;
  enum {
    MaxRows  = _Lhs::MaxRowsAtCompileTime,
    Rows  = _Lhs::RowsAtCompileTime,
    MaxCols  = _Rhs::MaxColsAtCompileTime,
    Cols  = _Rhs::ColsAtCompileTime,
    MaxDepth = EIGEN_SIZE_MIN_PREFER_FIXED(_Lhs::MaxColsAtCompileTime,
                                           _Rhs::MaxRowsAtCompileTime),
    Depth = EIGEN_SIZE_MIN_PREFER_FIXED(_Lhs::ColsAtCompileTime,
                                        _Rhs::RowsAtCompileTime)
  };

  // the splitting into different lines of code here, introducing the _select enums and the typedef below,
  // is to work around an internal compiler error with gcc 4.1 and 4.2.
private:
  enum {
    rows_select = product_size_category<Rows,MaxRows>::value,
    cols_select = product_size_category<Cols,MaxCols>::value,
    depth_select = product_size_category<Depth,MaxDepth>::value
  };
  typedef product_type_selector<rows_select, cols_select, depth_select> selector;

public:
  enum {
    value = selector::ret
  };
#ifdef EIGEN_DEBUG_PRODUCT
  static void debug()
  {
      EIGEN_DEBUG_VAR(Rows);
      EIGEN_DEBUG_VAR(Cols);
      EIGEN_DEBUG_VAR(Depth);
      EIGEN_DEBUG_VAR(rows_select);
      EIGEN_DEBUG_VAR(cols_select);
      EIGEN_DEBUG_VAR(depth_select);
      EIGEN_DEBUG_VAR(value);
  }
#endif
};


/* The following allows to select the kind of product at compile time
 * based on the three dimensions of the product.
 * This is a compile time mapping from {1,Small,Large}^3 -> {product types} */
// FIXME I'm not sure the current mapping is the ideal one.
template<int M, int N>  struct product_type_selector<M,N,1>              { enum { ret = OuterProduct }; };
template<int Depth>     struct product_type_selector<1,    1,    Depth>  { enum { ret = InnerProduct }; };
template<>              struct product_type_selector<1,    1,    1>      { enum { ret = InnerProduct }; };
template<>              struct product_type_selector<Small,1,    Small>  { enum { ret = CoeffBasedProductMode }; };
template<>              struct product_type_selector<1,    Small,Small>  { enum { ret = CoeffBasedProductMode }; };
template<>              struct product_type_selector<Small,Small,Small>  { enum { ret = CoeffBasedProductMode }; };
template<>              struct product_type_selector<Small, Small, 1>    { enum { ret = LazyCoeffBasedProductMode }; };
template<>              struct product_type_selector<Small, Large, 1>    { enum { ret = LazyCoeffBasedProductMode }; };
template<>              struct product_type_selector<Large, Small, 1>    { enum { ret = LazyCoeffBasedProductMode }; };
template<>              struct product_type_selector<1,    Large,Small>  { enum { ret = CoeffBasedProductMode }; };
template<>              struct product_type_selector<1,    Large,Large>  { enum { ret = GemvProduct }; };
template<>              struct product_type_selector<1,    Small,Large>  { enum { ret = CoeffBasedProductMode }; };
template<>              struct product_type_selector<Large,1,    Small>  { enum { ret = CoeffBasedProductMode }; };
template<>              struct product_type_selector<Large,1,    Large>  { enum { ret = GemvProduct }; };
template<>              struct product_type_selector<Small,1,    Large>  { enum { ret = CoeffBasedProductMode }; };
template<>              struct product_type_selector<Small,Small,Large>  { enum { ret = GemmProduct }; };
template<>              struct product_type_selector<Large,Small,Large>  { enum { ret = GemmProduct }; };
template<>              struct product_type_selector<Small,Large,Large>  { enum { ret = GemmProduct }; };
template<>              struct product_type_selector<Large,Large,Large>  { enum { ret = GemmProduct }; };
template<>              struct product_type_selector<Large,Small,Small>  { enum { ret = GemmProduct }; };
template<>              struct product_type_selector<Small,Large,Small>  { enum { ret = GemmProduct }; };
template<>              struct product_type_selector<Large,Large,Small>  { enum { ret = GemmProduct }; };

} // end namespace internal

/** \class ProductReturnType
  * \ingroup Core_Module
  *
  * \brief Helper class to get the correct and optimized returned type of operator*
  *
  * \param Lhs the type of the left-hand side
  * \param Rhs the type of the right-hand side
  * \param ProductMode the type of the product (determined automatically by internal::product_mode)
  *
  * This class defines the typename Type representing the optimized product expression
  * between two matrix expressions. In practice, using ProductReturnType<Lhs,Rhs>::Type
  * is the recommended way to define the result type of a function returning an expression
  * which involve a matrix product. The class Product should never be
  * used directly.
  *
  * \sa class Product, MatrixBase::operator*(const MatrixBase<OtherDerived>&)
  */
template<typename Lhs, typename Rhs, int ProductType>
struct ProductReturnType
{
  // TODO use the nested type to reduce instanciations ????
//   typedef typename internal::nested<Lhs,Rhs::ColsAtCompileTime>::type LhsNested;
//   typedef typename internal::nested<Rhs,Lhs::RowsAtCompileTime>::type RhsNested;

  typedef GeneralProduct<Lhs/*Nested*/, Rhs/*Nested*/, ProductType> Type;
};

template<typename Lhs, typename Rhs>
struct ProductReturnType<Lhs,Rhs,CoeffBasedProductMode>
{
  typedef typename internal::nested<Lhs, Rhs::ColsAtCompileTime, typename internal::plain_matrix_type<Lhs>::type >::type LhsNested;
  typedef typename internal::nested<Rhs, Lhs::RowsAtCompileTime, typename internal::plain_matrix_type<Rhs>::type >::type RhsNested;
  typedef CoeffBasedProduct<LhsNested, RhsNested, EvalBeforeAssigningBit | EvalBeforeNestingBit> Type;
};

template<typename Lhs, typename Rhs>
struct ProductReturnType<Lhs,Rhs,LazyCoeffBasedProductMode>
{
  typedef typename internal::nested<Lhs, Rhs::ColsAtCompileTime, typename internal::plain_matrix_type<Lhs>::type >::type LhsNested;
  typedef typename internal::nested<Rhs, Lhs::RowsAtCompileTime, typename internal::plain_matrix_type<Rhs>::type >::type RhsNested;
  typedef CoeffBasedProduct<LhsNested, RhsNested, NestByRefBit> Type;
};

// this is a workaround for sun CC
template<typename Lhs, typename Rhs>
struct LazyProductReturnType : public ProductReturnType<Lhs,Rhs,LazyCoeffBasedProductMode>
{};

/***********************************************************************
*  Implementation of Inner Vector Vector Product
***********************************************************************/

// FIXME : maybe the "inner product" could return a Scalar
// instead of a 1x1 matrix ??
// Pro: more natural for the user
// Cons: this could be a problem if in a meta unrolled algorithm a matrix-matrix
// product ends up to a row-vector times col-vector product... To tackle this use
// case, we could have a specialization for Block<MatrixType,1,1> with: operator=(Scalar x);

namespace internal {

template<typename Lhs, typename Rhs>
struct traits<GeneralProduct<Lhs,Rhs,InnerProduct> >
 : traits<Matrix<typename scalar_product_traits<typename Lhs::Scalar, typename Rhs::Scalar>::ReturnType,1,1> >
{};

}

template<typename Lhs, typename Rhs>
class GeneralProduct<Lhs, Rhs, InnerProduct>
  : internal::no_assignment_operator,
    public Matrix<typename internal::scalar_product_traits<typename Lhs::Scalar, typename Rhs::Scalar>::ReturnType,1,1>
{
    typedef Matrix<typename internal::scalar_product_traits<typename Lhs::Scalar, typename Rhs::Scalar>::ReturnType,1,1> Base;
  public:
    GeneralProduct(const Lhs& lhs, const Rhs& rhs)
    {
      EIGEN_STATIC_ASSERT((internal::is_same<typename Lhs::RealScalar, typename Rhs::RealScalar>::value),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)

      Base::coeffRef(0,0) = (lhs.transpose().cwiseProduct(rhs)).sum();
    }

    /** Convertion to scalar */
    operator const typename Base::Scalar() const {
      return Base::coeff(0,0);
    }
};

/***********************************************************************
*  Implementation of Outer Vector Vector Product
***********************************************************************/

namespace internal {

// Column major
template<typename ProductType, typename Dest, typename Func>
EIGEN_DONT_INLINE void outer_product_selector_run(const ProductType& prod, Dest& dest, const Func& func, const false_type&)
{
  typedef typename Dest::Index Index;
  // FIXME make sure lhs is sequentially stored
  // FIXME not very good if rhs is real and lhs complex while alpha is real too
  const Index cols = dest.cols();
  for (Index j=0; j<cols; ++j)
    func(dest.col(j), prod.rhs().coeff(j) * prod.lhs());
}

// Row major
template<typename ProductType, typename Dest, typename Func>
EIGEN_DONT_INLINE void outer_product_selector_run(const ProductType& prod, Dest& dest, const Func& func, const true_type&) {
  typedef typename Dest::Index Index;
  // FIXME make sure rhs is sequentially stored
  // FIXME not very good if lhs is real and rhs complex while alpha is real too
  const Index rows = dest.rows();
  for (Index i=0; i<rows; ++i)
    func(dest.row(i), prod.lhs().coeff(i) * prod.rhs());
}

template<typename Lhs, typename Rhs>
struct traits<GeneralProduct<Lhs,Rhs,OuterProduct> >
 : traits<ProductBase<GeneralProduct<Lhs,Rhs,OuterProduct>, Lhs, Rhs> >
{};

}

template<typename Lhs, typename Rhs>
class GeneralProduct<Lhs, Rhs, OuterProduct>
  : public ProductBase<GeneralProduct<Lhs,Rhs,OuterProduct>, Lhs, Rhs>
{
    template<typename T> struct IsRowMajor : internal::conditional<(int(T::Flags)&RowMajorBit), internal::true_type, internal::false_type>::type {};

  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(GeneralProduct)

    GeneralProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs)
    {
      EIGEN_STATIC_ASSERT((internal::is_same<typename Lhs::RealScalar, typename Rhs::RealScalar>::value),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
    }

    struct set  { template<typename Dst, typename Src> void operator()(const Dst& dst, const Src& src) const { dst.const_cast_derived()  = src; } };
    struct add  { template<typename Dst, typename Src> void operator()(const Dst& dst, const Src& src) const { dst.const_cast_derived() += src; } };
    struct sub  { template<typename Dst, typename Src> void operator()(const Dst& dst, const Src& src) const { dst.const_cast_derived() -= src; } };
    struct adds {
      Scalar m_scale;
      adds(const Scalar& s) : m_scale(s) {}
      template<typename Dst, typename Src> void operator()(const Dst& dst, const Src& src) const {
        dst.const_cast_derived() += m_scale * src;
      }
    };

    template<typename Dest>
    inline void evalTo(Dest& dest) const {
      internal::outer_product_selector_run(*this, dest, set(), IsRowMajor<Dest>());
    }

    template<typename Dest>
    inline void addTo(Dest& dest) const {
      internal::outer_product_selector_run(*this, dest, add(), IsRowMajor<Dest>());
    }

    template<typename Dest>
    inline void subTo(Dest& dest) const {
      internal::outer_product_selector_run(*this, dest, sub(), IsRowMajor<Dest>());
    }

    template<typename Dest> void scaleAndAddTo(Dest& dest, const Scalar& alpha) const
    {
      internal::outer_product_selector_run(*this, dest, adds(alpha), IsRowMajor<Dest>());
    }
};

/***********************************************************************
*  Implementation of General Matrix Vector Product
***********************************************************************/

/*  According to the shape/flags of the matrix we have to distinghish 3 different cases:
 *   1 - the matrix is col-major, BLAS compatible and M is large => call fast BLAS-like colmajor routine
 *   2 - the matrix is row-major, BLAS compatible and N is large => call fast BLAS-like rowmajor routine
 *   3 - all other cases are handled using a simple loop along the outer-storage direction.
 *  Therefore we need a lower level meta selector.
 *  Furthermore, if the matrix is the rhs, then the product has to be transposed.
 */
namespace internal {

template<typename Lhs, typename Rhs>
struct traits<GeneralProduct<Lhs,Rhs,GemvProduct> >
 : traits<ProductBase<GeneralProduct<Lhs,Rhs,GemvProduct>, Lhs, Rhs> >
{};

template<int Side, int StorageOrder, bool BlasCompatible>
struct gemv_selector;

} // end namespace internal

template<typename Lhs, typename Rhs>
class GeneralProduct<Lhs, Rhs, GemvProduct>
  : public ProductBase<GeneralProduct<Lhs,Rhs,GemvProduct>, Lhs, Rhs>
{
  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(GeneralProduct)

    typedef typename Lhs::Scalar LhsScalar;
    typedef typename Rhs::Scalar RhsScalar;

    GeneralProduct(const Lhs& a_lhs, const Rhs& a_rhs) : Base(a_lhs,a_rhs)
    {
//       EIGEN_STATIC_ASSERT((internal::is_same<typename Lhs::Scalar, typename Rhs::Scalar>::value),
//         YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
    }

    enum { Side = Lhs::IsVectorAtCompileTime ? OnTheLeft : OnTheRight };
    typedef typename internal::conditional<int(Side)==OnTheRight,_LhsNested,_RhsNested>::type MatrixType;

    template<typename Dest> void scaleAndAddTo(Dest& dst, const Scalar& alpha) const
    {
      eigen_assert(m_lhs.rows() == dst.rows() && m_rhs.cols() == dst.cols());
      internal::gemv_selector<Side,(int(MatrixType::Flags)&RowMajorBit) ? RowMajor : ColMajor,
                       bool(internal::blas_traits<MatrixType>::HasUsableDirectAccess)>::run(*this, dst, alpha);
    }
};

namespace internal {

// The vector is on the left => transposition
template<int StorageOrder, bool BlasCompatible>
struct gemv_selector<OnTheLeft,StorageOrder,BlasCompatible>
{
  template<typename ProductType, typename Dest>
  static void run(const ProductType& prod, Dest& dest, const typename ProductType::Scalar& alpha)
  {
    Transpose<Dest> destT(dest);
    enum { OtherStorageOrder = StorageOrder == RowMajor ? ColMajor : RowMajor };
    gemv_selector<OnTheRight,OtherStorageOrder,BlasCompatible>
      ::run(GeneralProduct<Transpose<const typename ProductType::_RhsNested>,Transpose<const typename ProductType::_LhsNested>, GemvProduct>
        (prod.rhs().transpose(), prod.lhs().transpose()), destT, alpha);
  }
};

template<typename Scalar,int Size,int MaxSize,bool Cond> struct gemv_static_vector_if;

template<typename Scalar,int Size,int MaxSize>
struct gemv_static_vector_if<Scalar,Size,MaxSize,false>
{
  EIGEN_STRONG_INLINE  Scalar* data() { eigen_internal_assert(false && "should never be called"); return 0; }
};

template<typename Scalar,int Size>
struct gemv_static_vector_if<Scalar,Size,Dynamic,true>
{
  EIGEN_STRONG_INLINE Scalar* data() { return 0; }
};

template<typename Scalar,int Size,int MaxSize>
struct gemv_static_vector_if<Scalar,Size,MaxSize,true>
{
  #if EIGEN_ALIGN_STATICALLY
  internal::plain_array<Scalar,EIGEN_SIZE_MIN_PREFER_FIXED(Size,MaxSize),0> m_data;
  EIGEN_STRONG_INLINE Scalar* data() { return m_data.array; }
  #else
  // Some architectures cannot align on the stack,
  // => let's manually enforce alignment by allocating more data and return the address of the first aligned element.
  enum {
    ForceAlignment  = internal::packet_traits<Scalar>::Vectorizable,
    PacketSize      = internal::packet_traits<Scalar>::size
  };
  internal::plain_array<Scalar,EIGEN_SIZE_MIN_PREFER_FIXED(Size,MaxSize)+(ForceAlignment?PacketSize:0),0> m_data;
  EIGEN_STRONG_INLINE Scalar* data() {
    return ForceAlignment
            ? reinterpret_cast<Scalar*>((reinterpret_cast<size_t>(m_data.array) & ~(size_t(EIGEN_ALIGN_BYTES-1))) + EIGEN_ALIGN_BYTES)
            : m_data.array;
  }
  #endif
};

template<> struct gemv_selector<OnTheRight,ColMajor,true>
{
  template<typename ProductType, typename Dest>
  static inline void run(const ProductType& prod, Dest& dest, const typename ProductType::Scalar& alpha)
  {
    typedef typename ProductType::Index Index;
    typedef typename ProductType::LhsScalar   LhsScalar;
    typedef typename ProductType::RhsScalar   RhsScalar;
    typedef typename ProductType::Scalar      ResScalar;
    typedef typename ProductType::RealScalar  RealScalar;
    typedef typename ProductType::ActualLhsType ActualLhsType;
    typedef typename ProductType::ActualRhsType ActualRhsType;
    typedef typename ProductType::LhsBlasTraits LhsBlasTraits;
    typedef typename ProductType::RhsBlasTraits RhsBlasTraits;
    typedef Map<Matrix<ResScalar,Dynamic,1>, Aligned> MappedDest;

    ActualLhsType actualLhs = LhsBlasTraits::extract(prod.lhs());
    ActualRhsType actualRhs = RhsBlasTraits::extract(prod.rhs());

    ResScalar actualAlpha = alpha * LhsBlasTraits::extractScalarFactor(prod.lhs())
                                  * RhsBlasTraits::extractScalarFactor(prod.rhs());

    enum {
      // FIXME find a way to allow an inner stride on the result if packet_traits<Scalar>::size==1
      // on, the other hand it is good for the cache to pack the vector anyways...
      EvalToDestAtCompileTime = Dest::InnerStrideAtCompileTime==1,
      ComplexByReal = (NumTraits<LhsScalar>::IsComplex) && (!NumTraits<RhsScalar>::IsComplex),
      MightCannotUseDest = (Dest::InnerStrideAtCompileTime!=1) || ComplexByReal
    };

    gemv_static_vector_if<ResScalar,Dest::SizeAtCompileTime,Dest::MaxSizeAtCompileTime,MightCannotUseDest> static_dest;

    bool alphaIsCompatible = (!ComplexByReal) || (numext::imag(actualAlpha)==RealScalar(0));
    bool evalToDest = EvalToDestAtCompileTime && alphaIsCompatible;

    RhsScalar compatibleAlpha = get_factor<ResScalar,RhsScalar>::run(actualAlpha);

    ei_declare_aligned_stack_constructed_variable(ResScalar,actualDestPtr,dest.size(),
                                                  evalToDest ? dest.data() : static_dest.data());

    if(!evalToDest)
    {
      #ifdef EIGEN_DENSE_STORAGE_CTOR_PLUGIN
      int size = dest.size();
      EIGEN_DENSE_STORAGE_CTOR_PLUGIN
      #endif
      if(!alphaIsCompatible)
      {
        MappedDest(actualDestPtr, dest.size()).setZero();
        compatibleAlpha = RhsScalar(1);
      }
      else
        MappedDest(actualDestPtr, dest.size()) = dest;
    }

    typedef const_blas_data_mapper<LhsScalar,Index,ColMajor> LhsMapper;
    typedef const_blas_data_mapper<RhsScalar,Index,RowMajor> RhsMapper;
    general_matrix_vector_product
        <Index,LhsScalar,LhsMapper,ColMajor,LhsBlasTraits::NeedToConjugate,RhsScalar,RhsMapper,RhsBlasTraits::NeedToConjugate>::run(
        actualLhs.rows(), actualLhs.cols(),
        LhsMapper(actualLhs.data(), actualLhs.outerStride()),
        RhsMapper(actualRhs.data(), actualRhs.innerStride()),
        actualDestPtr, 1,
        compatibleAlpha);

    if (!evalToDest)
    {
      if(!alphaIsCompatible)
        dest += actualAlpha * MappedDest(actualDestPtr, dest.size());
      else
        dest = MappedDest(actualDestPtr, dest.size());
    }
  }
};

template<> struct gemv_selector<OnTheRight,RowMajor,true>
{
  template<typename ProductType, typename Dest>
  static void run(const ProductType& prod, Dest& dest, const typename ProductType::Scalar& alpha)
  {
    typedef typename ProductType::LhsScalar LhsScalar;
    typedef typename ProductType::RhsScalar RhsScalar;
    typedef typename ProductType::Scalar    ResScalar;
    typedef typename ProductType::Index Index;
    typedef typename ProductType::ActualLhsType ActualLhsType;
    typedef typename ProductType::ActualRhsType ActualRhsType;
    typedef typename ProductType::_ActualRhsType _ActualRhsType;
    typedef typename ProductType::LhsBlasTraits LhsBlasTraits;
    typedef typename ProductType::RhsBlasTraits RhsBlasTraits;

    typename add_const<ActualLhsType>::type actualLhs = LhsBlasTraits::extract(prod.lhs());
    typename add_const<ActualRhsType>::type actualRhs = RhsBlasTraits::extract(prod.rhs());

    ResScalar actualAlpha = alpha * LhsBlasTraits::extractScalarFactor(prod.lhs())
                                  * RhsBlasTraits::extractScalarFactor(prod.rhs());

    enum {
      // FIXME find a way to allow an inner stride on the result if packet_traits<Scalar>::size==1
      // on, the other hand it is good for the cache to pack the vector anyways...
      DirectlyUseRhs = _ActualRhsType::InnerStrideAtCompileTime==1
    };

    gemv_static_vector_if<RhsScalar,_ActualRhsType::SizeAtCompileTime,_ActualRhsType::MaxSizeAtCompileTime,!DirectlyUseRhs> static_rhs;

    ei_declare_aligned_stack_constructed_variable(RhsScalar,actualRhsPtr,actualRhs.size(),
        DirectlyUseRhs ? const_cast<RhsScalar*>(actualRhs.data()) : static_rhs.data());

    if(!DirectlyUseRhs)
    {
      #ifdef EIGEN_DENSE_STORAGE_CTOR_PLUGIN
      int size = actualRhs.size();
      EIGEN_DENSE_STORAGE_CTOR_PLUGIN
      #endif
      Map<typename _ActualRhsType::PlainObject>(actualRhsPtr, actualRhs.size()) = actualRhs;
    }

    typedef const_blas_data_mapper<LhsScalar,Index,RowMajor> LhsMapper;
    typedef const_blas_data_mapper<RhsScalar,Index,ColMajor> RhsMapper;
    general_matrix_vector_product
        <Index,LhsScalar,LhsMapper,RowMajor,LhsBlasTraits::NeedToConjugate,RhsScalar,RhsMapper,RhsBlasTraits::NeedToConjugate>::run(
        actualLhs.rows(), actualLhs.cols(),
        LhsMapper(actualLhs.data(), actualLhs.outerStride()),
        RhsMapper(actualRhsPtr, 1),
        dest.data(), dest.innerStride(),
        actualAlpha);
  }
};

template<> struct gemv_selector<OnTheRight,ColMajor,false>
{
  template<typename ProductType, typename Dest>
  static void run(const ProductType& prod, Dest& dest, const typename ProductType::Scalar& alpha)
  {
    typedef typename Dest::Index Index;
    // TODO makes sure dest is sequentially stored in memory, otherwise use a temp
    const Index size = prod.rhs().rows();
    for(Index k=0; k<size; ++k)
      dest += (alpha*prod.rhs().coeff(k)) * prod.lhs().col(k);
  }
};

template<> struct gemv_selector<OnTheRight,RowMajor,false>
{
  template<typename ProductType, typename Dest>
  static void run(const ProductType& prod, Dest& dest, const typename ProductType::Scalar& alpha)
  {
    typedef typename Dest::Index Index;
    // TODO makes sure rhs is sequentially stored in memory, otherwise use a temp
    const Index rows = prod.rows();
    for(Index i=0; i<rows; ++i)
      dest.coeffRef(i) += alpha * (prod.lhs().row(i).cwiseProduct(prod.rhs().transpose())).sum();
  }
};

} // end namespace internal

/***************************************************************************
* Implementation of matrix base methods
***************************************************************************/

/** \returns the matrix product of \c *this and \a other.
  *
  * \note If instead of the matrix product you want the coefficient-wise product, see Cwise::operator*().
  *
  * \sa lazyProduct(), operator*=(const MatrixBase&), Cwise::operator*()
  */
#ifndef __CUDACC__

#ifdef EIGEN_TEST_EVALUATORS
template<typename Derived>
template<typename OtherDerived>
inline const Product<Derived, OtherDerived>
MatrixBase<Derived>::operator*(const MatrixBase<OtherDerived> &other) const
{
  // A note regarding the function declaration: In MSVC, this function will sometimes
  // not be inlined since DenseStorage is an unwindable object for dynamic
  // matrices and product types are holding a member to store the result.
  // Thus it does not help tagging this function with EIGEN_STRONG_INLINE.
  enum {
    ProductIsValid =  Derived::ColsAtCompileTime==Dynamic
                   || OtherDerived::RowsAtCompileTime==Dynamic
                   || int(Derived::ColsAtCompileTime)==int(OtherDerived::RowsAtCompileTime),
    AreVectors = Derived::IsVectorAtCompileTime && OtherDerived::IsVectorAtCompileTime,
    SameSizes = EIGEN_PREDICATE_SAME_MATRIX_SIZE(Derived,OtherDerived)
  };
  // note to the lost user:
  //    * for a dot product use: v1.dot(v2)
  //    * for a coeff-wise product use: v1.cwiseProduct(v2)
  EIGEN_STATIC_ASSERT(ProductIsValid || !(AreVectors && SameSizes),
    INVALID_VECTOR_VECTOR_PRODUCT__IF_YOU_WANTED_A_DOT_OR_COEFF_WISE_PRODUCT_YOU_MUST_USE_THE_EXPLICIT_FUNCTIONS)
  EIGEN_STATIC_ASSERT(ProductIsValid || !(SameSizes && !AreVectors),
    INVALID_MATRIX_PRODUCT__IF_YOU_WANTED_A_COEFF_WISE_PRODUCT_YOU_MUST_USE_THE_EXPLICIT_FUNCTION)
  EIGEN_STATIC_ASSERT(ProductIsValid || SameSizes, INVALID_MATRIX_PRODUCT)
#ifdef EIGEN_DEBUG_PRODUCT
  internal::product_type<Derived,OtherDerived>::debug();
#endif

  return Product<Derived, OtherDerived>(derived(), other.derived());
}
#else
template<typename Derived>
template<typename OtherDerived>
inline const typename ProductReturnType<Derived, OtherDerived>::Type
MatrixBase<Derived>::operator*(const MatrixBase<OtherDerived> &other) const
{
  // A note regarding the function declaration: In MSVC, this function will sometimes
  // not be inlined since DenseStorage is an unwindable object for dynamic
  // matrices and product types are holding a member to store the result.
  // Thus it does not help tagging this function with EIGEN_STRONG_INLINE.
  enum {
    ProductIsValid =  Derived::ColsAtCompileTime==Dynamic
                   || OtherDerived::RowsAtCompileTime==Dynamic
                   || int(Derived::ColsAtCompileTime)==int(OtherDerived::RowsAtCompileTime),
    AreVectors = Derived::IsVectorAtCompileTime && OtherDerived::IsVectorAtCompileTime,
    SameSizes = EIGEN_PREDICATE_SAME_MATRIX_SIZE(Derived,OtherDerived)
  };
  // note to the lost user:
  //    * for a dot product use: v1.dot(v2)
  //    * for a coeff-wise product use: v1.cwiseProduct(v2)
  EIGEN_STATIC_ASSERT(ProductIsValid || !(AreVectors && SameSizes),
    INVALID_VECTOR_VECTOR_PRODUCT__IF_YOU_WANTED_A_DOT_OR_COEFF_WISE_PRODUCT_YOU_MUST_USE_THE_EXPLICIT_FUNCTIONS)
  EIGEN_STATIC_ASSERT(ProductIsValid || !(SameSizes && !AreVectors),
    INVALID_MATRIX_PRODUCT__IF_YOU_WANTED_A_COEFF_WISE_PRODUCT_YOU_MUST_USE_THE_EXPLICIT_FUNCTION)
  EIGEN_STATIC_ASSERT(ProductIsValid || SameSizes, INVALID_MATRIX_PRODUCT)
#ifdef EIGEN_DEBUG_PRODUCT
  internal::product_type<Derived,OtherDerived>::debug();
#endif
  return typename ProductReturnType<Derived,OtherDerived>::Type(derived(), other.derived());
}
#endif

#endif
/** \returns an expression of the matrix product of \c *this and \a other without implicit evaluation.
  *
  * The returned product will behave like any other expressions: the coefficients of the product will be
  * computed once at a time as requested. This might be useful in some extremely rare cases when only
  * a small and no coherent fraction of the result's coefficients have to be computed.
  *
  * \warning This version of the matrix product can be much much slower. So use it only if you know
  * what you are doing and that you measured a true speed improvement.
  *
  * \sa operator*(const MatrixBase&)
  */
template<typename Derived>
template<typename OtherDerived>
const typename LazyProductReturnType<Derived,OtherDerived>::Type
MatrixBase<Derived>::lazyProduct(const MatrixBase<OtherDerived> &other) const
{
  enum {
    ProductIsValid =  Derived::ColsAtCompileTime==Dynamic
                   || OtherDerived::RowsAtCompileTime==Dynamic
                   || int(Derived::ColsAtCompileTime)==int(OtherDerived::RowsAtCompileTime),
    AreVectors = Derived::IsVectorAtCompileTime && OtherDerived::IsVectorAtCompileTime,
    SameSizes = EIGEN_PREDICATE_SAME_MATRIX_SIZE(Derived,OtherDerived)
  };
  // note to the lost user:
  //    * for a dot product use: v1.dot(v2)
  //    * for a coeff-wise product use: v1.cwiseProduct(v2)
  EIGEN_STATIC_ASSERT(ProductIsValid || !(AreVectors && SameSizes),
    INVALID_VECTOR_VECTOR_PRODUCT__IF_YOU_WANTED_A_DOT_OR_COEFF_WISE_PRODUCT_YOU_MUST_USE_THE_EXPLICIT_FUNCTIONS)
  EIGEN_STATIC_ASSERT(ProductIsValid || !(SameSizes && !AreVectors),
    INVALID_MATRIX_PRODUCT__IF_YOU_WANTED_A_COEFF_WISE_PRODUCT_YOU_MUST_USE_THE_EXPLICIT_FUNCTION)
  EIGEN_STATIC_ASSERT(ProductIsValid || SameSizes, INVALID_MATRIX_PRODUCT)

  return typename LazyProductReturnType<Derived,OtherDerived>::Type(derived(), other.derived());
}

} // end namespace Eigen

#endif // EIGEN_PRODUCT_H
