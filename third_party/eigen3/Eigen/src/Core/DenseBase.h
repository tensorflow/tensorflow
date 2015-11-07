// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2007-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_DENSEBASE_H
#define EIGEN_DENSEBASE_H

namespace Eigen {

namespace internal {
  
// The index type defined by EIGEN_DEFAULT_DENSE_INDEX_TYPE must be a signed type.
// This dummy function simply aims at checking that at compile time.
static inline void check_DenseIndex_is_signed() {
  EIGEN_STATIC_ASSERT(NumTraits<DenseIndex>::IsSigned,THE_INDEX_TYPE_MUST_BE_A_SIGNED_TYPE); 
}

} // end namespace internal
  
/** \class DenseBase
  * \ingroup Core_Module
  *
  * \brief Base class for all dense matrices, vectors, and arrays
  *
  * This class is the base that is inherited by all dense objects (matrix, vector, arrays,
  * and related expression types). The common Eigen API for dense objects is contained in this class.
  *
  * \tparam Derived is the derived type, e.g., a matrix type or an expression.
  *
  * This class can be extended with the help of the plugin mechanism described on the page
  * \ref TopicCustomizingEigen by defining the preprocessor symbol \c EIGEN_DENSEBASE_PLUGIN.
  *
  * \sa \ref TopicClassHierarchy
  */
template<typename Derived> class DenseBase
#ifndef EIGEN_PARSED_BY_DOXYGEN
  : public internal::special_scalar_op_base<Derived,typename internal::traits<Derived>::Scalar,
                                     typename NumTraits<typename internal::traits<Derived>::Scalar>::Real>
#else
  : public DenseCoeffsBase<Derived>
#endif // not EIGEN_PARSED_BY_DOXYGEN
{
  public:
    using internal::special_scalar_op_base<Derived,typename internal::traits<Derived>::Scalar,
                typename NumTraits<typename internal::traits<Derived>::Scalar>::Real>::operator*;

    class InnerIterator;

    typedef typename internal::traits<Derived>::StorageKind StorageKind;

    /** \brief The type of indices 
      * \details To change this, \c \#define the preprocessor symbol \c EIGEN_DEFAULT_DENSE_INDEX_TYPE.
      * \sa \ref TopicPreprocessorDirectives.
      */
    typedef typename internal::traits<Derived>::Index Index; 

    typedef typename internal::traits<Derived>::Scalar Scalar;
    typedef typename internal::packet_traits<Scalar>::type PacketScalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;

    typedef DenseCoeffsBase<Derived> Base;
    using Base::derived;
    using Base::const_cast_derived;
    using Base::rows;
    using Base::cols;
    using Base::size;
    using Base::rowIndexByOuterInner;
    using Base::colIndexByOuterInner;
    using Base::coeff;
    using Base::coeffByOuterInner;
    using Base::packet;
    using Base::packetByOuterInner;
    using Base::writePacket;
    using Base::writePacketByOuterInner;
    using Base::coeffRef;
    using Base::coeffRefByOuterInner;
    using Base::copyCoeff;
    using Base::copyCoeffByOuterInner;
    using Base::copyPacket;
    using Base::copyPacketByOuterInner;
    using Base::operator();
    using Base::operator[];
    using Base::x;
    using Base::y;
    using Base::z;
    using Base::w;
    using Base::stride;
    using Base::innerStride;
    using Base::outerStride;
    using Base::rowStride;
    using Base::colStride;
    typedef typename Base::CoeffReturnType CoeffReturnType;

    enum {

      RowsAtCompileTime = internal::traits<Derived>::RowsAtCompileTime,
        /**< The number of rows at compile-time. This is just a copy of the value provided
          * by the \a Derived type. If a value is not known at compile-time,
          * it is set to the \a Dynamic constant.
          * \sa MatrixBase::rows(), MatrixBase::cols(), ColsAtCompileTime, SizeAtCompileTime */

      ColsAtCompileTime = internal::traits<Derived>::ColsAtCompileTime,
        /**< The number of columns at compile-time. This is just a copy of the value provided
          * by the \a Derived type. If a value is not known at compile-time,
          * it is set to the \a Dynamic constant.
          * \sa MatrixBase::rows(), MatrixBase::cols(), RowsAtCompileTime, SizeAtCompileTime */


      SizeAtCompileTime = (internal::size_at_compile_time<internal::traits<Derived>::RowsAtCompileTime,
                                                   internal::traits<Derived>::ColsAtCompileTime>::ret),
        /**< This is equal to the number of coefficients, i.e. the number of
          * rows times the number of columns, or to \a Dynamic if this is not
          * known at compile-time. \sa RowsAtCompileTime, ColsAtCompileTime */

      MaxRowsAtCompileTime = internal::traits<Derived>::MaxRowsAtCompileTime,
        /**< This value is equal to the maximum possible number of rows that this expression
          * might have. If this expression might have an arbitrarily high number of rows,
          * this value is set to \a Dynamic.
          *
          * This value is useful to know when evaluating an expression, in order to determine
          * whether it is possible to avoid doing a dynamic memory allocation.
          *
          * \sa RowsAtCompileTime, MaxColsAtCompileTime, MaxSizeAtCompileTime
          */

      MaxColsAtCompileTime = internal::traits<Derived>::MaxColsAtCompileTime,
        /**< This value is equal to the maximum possible number of columns that this expression
          * might have. If this expression might have an arbitrarily high number of columns,
          * this value is set to \a Dynamic.
          *
          * This value is useful to know when evaluating an expression, in order to determine
          * whether it is possible to avoid doing a dynamic memory allocation.
          *
          * \sa ColsAtCompileTime, MaxRowsAtCompileTime, MaxSizeAtCompileTime
          */

      MaxSizeAtCompileTime = (internal::size_at_compile_time<internal::traits<Derived>::MaxRowsAtCompileTime,
                                                      internal::traits<Derived>::MaxColsAtCompileTime>::ret),
        /**< This value is equal to the maximum possible number of coefficients that this expression
          * might have. If this expression might have an arbitrarily high number of coefficients,
          * this value is set to \a Dynamic.
          *
          * This value is useful to know when evaluating an expression, in order to determine
          * whether it is possible to avoid doing a dynamic memory allocation.
          *
          * \sa SizeAtCompileTime, MaxRowsAtCompileTime, MaxColsAtCompileTime
          */

      IsVectorAtCompileTime = internal::traits<Derived>::MaxRowsAtCompileTime == 1
                           || internal::traits<Derived>::MaxColsAtCompileTime == 1,
        /**< This is set to true if either the number of rows or the number of
          * columns is known at compile-time to be equal to 1. Indeed, in that case,
          * we are dealing with a column-vector (if there is only one column) or with
          * a row-vector (if there is only one row). */

      Flags = internal::traits<Derived>::Flags,
        /**< This stores expression \ref flags flags which may or may not be inherited by new expressions
          * constructed from this one. See the \ref flags "list of flags".
          */

      IsRowMajor = int(Flags) & RowMajorBit, /**< True if this expression has row-major storage order. */

      InnerSizeAtCompileTime = int(IsVectorAtCompileTime) ? int(SizeAtCompileTime)
                             : int(IsRowMajor) ? int(ColsAtCompileTime) : int(RowsAtCompileTime),

      CoeffReadCost = internal::traits<Derived>::CoeffReadCost,
        /**< This is a rough measure of how expensive it is to read one coefficient from
          * this expression.
          */

      InnerStrideAtCompileTime = internal::inner_stride_at_compile_time<Derived>::ret,
      OuterStrideAtCompileTime = internal::outer_stride_at_compile_time<Derived>::ret
    };

    enum { ThisConstantIsPrivateInPlainObjectBase };

    /** \returns the number of nonzero coefficients which is in practice the number
      * of stored coefficients. */
    EIGEN_DEVICE_FUNC
    inline Index nonZeros() const { return size(); }
    /** \returns true if either the number of rows or the number of columns is equal to 1.
      * In other words, this function returns
      * \code rows()==1 || cols()==1 \endcode
      * \sa rows(), cols(), IsVectorAtCompileTime. */

    /** \returns the outer size.
      *
      * \note For a vector, this returns just 1. For a matrix (non-vector), this is the major dimension
      * with respect to the \ref TopicStorageOrders "storage order", i.e., the number of columns for a
      * column-major matrix, and the number of rows for a row-major matrix. */
    EIGEN_DEVICE_FUNC
    Index outerSize() const
    {
      return IsVectorAtCompileTime ? 1
           : int(IsRowMajor) ? this->rows() : this->cols();
    }

    /** \returns the inner size.
      *
      * \note For a vector, this is just the size. For a matrix (non-vector), this is the minor dimension
      * with respect to the \ref TopicStorageOrders "storage order", i.e., the number of rows for a 
      * column-major matrix, and the number of columns for a row-major matrix. */
    EIGEN_DEVICE_FUNC
    Index innerSize() const
    {
      return IsVectorAtCompileTime ? this->size()
           : int(IsRowMajor) ? this->cols() : this->rows();
    }

    /** Only plain matrices/arrays, not expressions, may be resized; therefore the only useful resize methods are
      * Matrix::resize() and Array::resize(). The present method only asserts that the new size equals the old size, and does
      * nothing else.
      */
    EIGEN_DEVICE_FUNC
    void resize(Index newSize)
    {
      EIGEN_ONLY_USED_FOR_DEBUG(newSize);
      eigen_assert(newSize == this->size()
                && "DenseBase::resize() does not actually allow to resize.");
    }
    /** Only plain matrices/arrays, not expressions, may be resized; therefore the only useful resize methods are
      * Matrix::resize() and Array::resize(). The present method only asserts that the new size equals the old size, and does
      * nothing else.
      */
    EIGEN_DEVICE_FUNC
    void resize(Index nbRows, Index nbCols)
    {
      EIGEN_ONLY_USED_FOR_DEBUG(nbRows);
      EIGEN_ONLY_USED_FOR_DEBUG(nbCols);
      eigen_assert(nbRows == this->rows() && nbCols == this->cols()
                && "DenseBase::resize() does not actually allow to resize.");
    }

#ifndef EIGEN_PARSED_BY_DOXYGEN

    /** \internal Represents a matrix with all coefficients equal to one another*/
    typedef CwiseNullaryOp<internal::scalar_constant_op<Scalar>,Derived> ConstantReturnType;
    /** \internal Represents a vector with linearly spaced coefficients that allows sequential access only. */
    typedef CwiseNullaryOp<internal::linspaced_op<Scalar,false>,Derived> SequentialLinSpacedReturnType;
    /** \internal Represents a vector with linearly spaced coefficients that allows random access. */
    typedef CwiseNullaryOp<internal::linspaced_op<Scalar,true>,Derived> RandomAccessLinSpacedReturnType;
    /** \internal the return type of MatrixBase::eigenvalues() */
    typedef Matrix<typename NumTraits<typename internal::traits<Derived>::Scalar>::Real, internal::traits<Derived>::ColsAtCompileTime, 1> EigenvaluesReturnType;

#endif // not EIGEN_PARSED_BY_DOXYGEN

    /** Copies \a other into *this. \returns a reference to *this. */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    Derived& operator=(const DenseBase<OtherDerived>& other);

    /** Special case of the template operator=, in order to prevent the compiler
      * from generating a default operator= (issue hit with g++ 4.1)
      */
    EIGEN_DEVICE_FUNC
    Derived& operator=(const DenseBase& other);

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    Derived& operator=(const EigenBase<OtherDerived> &other);

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    Derived& operator+=(const EigenBase<OtherDerived> &other);

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    Derived& operator-=(const EigenBase<OtherDerived> &other);

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    Derived& operator=(const ReturnByValue<OtherDerived>& func);

#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** Copies \a other into *this without evaluating other. \returns a reference to *this. */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    Derived& lazyAssign(const DenseBase<OtherDerived>& other);
#endif // not EIGEN_PARSED_BY_DOXYGEN

    EIGEN_DEVICE_FUNC
    CommaInitializer<Derived> operator<< (const Scalar& s);

    template<unsigned int Added,unsigned int Removed>
    const Flagged<Derived, Added, Removed> flagged() const;

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    CommaInitializer<Derived> operator<< (const DenseBase<OtherDerived>& other);

    EIGEN_DEVICE_FUNC
    Eigen::Transpose<Derived> transpose();
    typedef typename internal::add_const<Transpose<const Derived> >::type ConstTransposeReturnType;
    EIGEN_DEVICE_FUNC
    ConstTransposeReturnType transpose() const;
    EIGEN_DEVICE_FUNC
    void transposeInPlace();
#ifndef EIGEN_NO_DEBUG
  protected:
    template<typename OtherDerived>
    void checkTransposeAliasing(const OtherDerived& other) const;
  public:
#endif


    EIGEN_DEVICE_FUNC static const ConstantReturnType
    Constant(Index rows, Index cols, const Scalar& value);
    EIGEN_DEVICE_FUNC static const ConstantReturnType
    Constant(Index size, const Scalar& value);
    EIGEN_DEVICE_FUNC static const ConstantReturnType
    Constant(const Scalar& value);

    EIGEN_DEVICE_FUNC static const SequentialLinSpacedReturnType
    LinSpaced(Sequential_t, Index size, const Scalar& low, const Scalar& high);
    EIGEN_DEVICE_FUNC static const RandomAccessLinSpacedReturnType
    LinSpaced(Index size, const Scalar& low, const Scalar& high);
    EIGEN_DEVICE_FUNC static const SequentialLinSpacedReturnType
    LinSpaced(Sequential_t, const Scalar& low, const Scalar& high);
    EIGEN_DEVICE_FUNC static const RandomAccessLinSpacedReturnType
    LinSpaced(const Scalar& low, const Scalar& high);

    template<typename CustomNullaryOp> EIGEN_DEVICE_FUNC
    static const CwiseNullaryOp<CustomNullaryOp, Derived>
    NullaryExpr(Index rows, Index cols, const CustomNullaryOp& func);
    template<typename CustomNullaryOp> EIGEN_DEVICE_FUNC
    static const CwiseNullaryOp<CustomNullaryOp, Derived>
    NullaryExpr(Index size, const CustomNullaryOp& func);
    template<typename CustomNullaryOp> EIGEN_DEVICE_FUNC
    static const CwiseNullaryOp<CustomNullaryOp, Derived>
    NullaryExpr(const CustomNullaryOp& func);

    EIGEN_DEVICE_FUNC static const ConstantReturnType Zero(Index rows, Index cols);
    EIGEN_DEVICE_FUNC static const ConstantReturnType Zero(Index size);
    EIGEN_DEVICE_FUNC static const ConstantReturnType Zero();
    EIGEN_DEVICE_FUNC static const ConstantReturnType Ones(Index rows, Index cols);
    EIGEN_DEVICE_FUNC static const ConstantReturnType Ones(Index size);
    EIGEN_DEVICE_FUNC static const ConstantReturnType Ones();

    EIGEN_DEVICE_FUNC void fill(const Scalar& value);
    EIGEN_DEVICE_FUNC Derived& setConstant(const Scalar& value);
    EIGEN_DEVICE_FUNC Derived& setLinSpaced(Index size, const Scalar& low, const Scalar& high);
    EIGEN_DEVICE_FUNC Derived& setLinSpaced(const Scalar& low, const Scalar& high);
    EIGEN_DEVICE_FUNC Derived& setZero();
    EIGEN_DEVICE_FUNC Derived& setOnes();
    EIGEN_DEVICE_FUNC Derived& setRandom();

    template<typename OtherDerived> EIGEN_DEVICE_FUNC
    bool isApprox(const DenseBase<OtherDerived>& other,
                  const RealScalar& prec = NumTraits<Scalar>::dummy_precision()) const;
    EIGEN_DEVICE_FUNC 
    bool isMuchSmallerThan(const RealScalar& other,
                           const RealScalar& prec = NumTraits<Scalar>::dummy_precision()) const;
    template<typename OtherDerived> EIGEN_DEVICE_FUNC
    bool isMuchSmallerThan(const DenseBase<OtherDerived>& other,
                           const RealScalar& prec = NumTraits<Scalar>::dummy_precision()) const;

    EIGEN_DEVICE_FUNC bool isApproxToConstant(const Scalar& value, const RealScalar& prec = NumTraits<Scalar>::dummy_precision()) const;
    EIGEN_DEVICE_FUNC bool isConstant(const Scalar& value, const RealScalar& prec = NumTraits<Scalar>::dummy_precision()) const;
    EIGEN_DEVICE_FUNC bool isZero(const RealScalar& prec = NumTraits<Scalar>::dummy_precision()) const;
    EIGEN_DEVICE_FUNC bool isOnes(const RealScalar& prec = NumTraits<Scalar>::dummy_precision()) const;
    
    inline bool hasNaN() const;
    inline bool allFinite() const;

    EIGEN_DEVICE_FUNC
    inline Derived& operator*=(const Scalar& other);
    EIGEN_DEVICE_FUNC
    inline Derived& operator/=(const Scalar& other);

    typedef typename internal::add_const_on_value_type<typename internal::eval<Derived>::type>::type EvalReturnType;
    /** \returns the matrix or vector obtained by evaluating this expression.
      *
      * Notice that in the case of a plain matrix or vector (not an expression) this function just returns
      * a const reference, in order to avoid a useless copy.
      */
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE EvalReturnType eval() const
    {
      // Even though MSVC does not honor strong inlining when the return type
      // is a dynamic matrix, we desperately need strong inlining for fixed
      // size types on MSVC.
      return typename internal::eval<Derived>::type(derived());
    }

    /** swaps *this with the expression \a other.
      *
      */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    void swap(const DenseBase<OtherDerived>& other,
              int = OtherDerived::ThisConstantIsPrivateInPlainObjectBase)
    {
      SwapWrapper<Derived>(derived()).lazyAssign(other.derived());
    }

    /** swaps *this with the matrix or array \a other.
      *
      */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    void swap(PlainObjectBase<OtherDerived>& other)
    {
      SwapWrapper<Derived>(derived()).lazyAssign(other.derived());
    }


    EIGEN_DEVICE_FUNC inline const NestByValue<Derived> nestByValue() const;
    EIGEN_DEVICE_FUNC inline const ForceAlignedAccess<Derived> forceAlignedAccess() const;
    EIGEN_DEVICE_FUNC inline ForceAlignedAccess<Derived> forceAlignedAccess();
    template<bool Enable> EIGEN_DEVICE_FUNC
    inline const typename internal::conditional<Enable,ForceAlignedAccess<Derived>,Derived&>::type forceAlignedAccessIf() const;
    template<bool Enable> EIGEN_DEVICE_FUNC
    inline typename internal::conditional<Enable,ForceAlignedAccess<Derived>,Derived&>::type forceAlignedAccessIf();

    EIGEN_DEVICE_FUNC Scalar sum() const;
    EIGEN_DEVICE_FUNC Scalar mean() const;
    EIGEN_DEVICE_FUNC Scalar trace() const;

    EIGEN_DEVICE_FUNC Scalar prod() const;

    EIGEN_DEVICE_FUNC typename internal::traits<Derived>::Scalar minCoeff() const;
    EIGEN_DEVICE_FUNC typename internal::traits<Derived>::Scalar maxCoeff() const;

    template<typename IndexType> EIGEN_DEVICE_FUNC
    typename internal::traits<Derived>::Scalar minCoeff(IndexType* row, IndexType* col) const;
    template<typename IndexType> EIGEN_DEVICE_FUNC
    typename internal::traits<Derived>::Scalar maxCoeff(IndexType* row, IndexType* col) const;
    template<typename IndexType> EIGEN_DEVICE_FUNC
    typename internal::traits<Derived>::Scalar minCoeff(IndexType* index) const;
    template<typename IndexType> EIGEN_DEVICE_FUNC
    typename internal::traits<Derived>::Scalar maxCoeff(IndexType* index) const;

    template<typename BinaryOp>
    EIGEN_DEVICE_FUNC
    typename internal::result_of<BinaryOp(typename internal::traits<Derived>::Scalar)>::type
    redux(const BinaryOp& func) const;

    template<typename Visitor>
    EIGEN_DEVICE_FUNC
    void visit(Visitor& func) const;

    inline const WithFormat<Derived> format(const IOFormat& fmt) const;

    /** \returns the unique coefficient of a 1x1 expression */
    EIGEN_DEVICE_FUNC
    CoeffReturnType value() const
    {
      EIGEN_STATIC_ASSERT_SIZE_1x1(Derived)
      eigen_assert(this->rows() == 1 && this->cols() == 1);
      return derived().coeff(0,0);
    }

    bool all() const;
    bool any() const;
    Index count() const;

    typedef VectorwiseOp<Derived, Horizontal> RowwiseReturnType;
    typedef const VectorwiseOp<const Derived, Horizontal> ConstRowwiseReturnType;
    typedef VectorwiseOp<Derived, Vertical> ColwiseReturnType;
    typedef const VectorwiseOp<const Derived, Vertical> ConstColwiseReturnType;

    ConstRowwiseReturnType rowwise() const;
    RowwiseReturnType rowwise();
    ConstColwiseReturnType colwise() const;
    ColwiseReturnType colwise();

    static const CwiseNullaryOp<internal::scalar_random_op<Scalar>,Derived> Random(Index rows, Index cols);
    static const CwiseNullaryOp<internal::scalar_random_op<Scalar>,Derived> Random(Index size);
    static const CwiseNullaryOp<internal::scalar_random_op<Scalar>,Derived> Random();

    template<typename ThenDerived,typename ElseDerived>
    const Select<Derived,ThenDerived,ElseDerived>
    select(const DenseBase<ThenDerived>& thenMatrix,
           const DenseBase<ElseDerived>& elseMatrix) const;

    template<typename ThenDerived>
    inline const Select<Derived,ThenDerived, typename ThenDerived::ConstantReturnType>
    select(const DenseBase<ThenDerived>& thenMatrix, const typename ThenDerived::Scalar& elseScalar) const;

    template<typename ElseDerived>
    inline const Select<Derived, typename ElseDerived::ConstantReturnType, ElseDerived >
    select(const typename ElseDerived::Scalar& thenScalar, const DenseBase<ElseDerived>& elseMatrix) const;

    template<int p> RealScalar lpNorm() const;

    template<int RowFactor, int ColFactor>
    const Replicate<Derived,RowFactor,ColFactor> replicate() const;
    const Replicate<Derived,Dynamic,Dynamic> replicate(Index rowFacor,Index colFactor) const;

    typedef Reverse<Derived, BothDirections> ReverseReturnType;
    typedef const Reverse<const Derived, BothDirections> ConstReverseReturnType;
    ReverseReturnType reverse();
    ConstReverseReturnType reverse() const;
    void reverseInPlace();

#define EIGEN_CURRENT_STORAGE_BASE_CLASS Eigen::DenseBase
#   include "../plugins/BlockMethods.h"
#   ifdef EIGEN_DENSEBASE_PLUGIN
#     include EIGEN_DENSEBASE_PLUGIN
#   endif
// Because of an intra-Google include scanner limitation,
// third_party/stan cannot define the EIGEN_DENSEBASE_PLUGIN
// macro
// as "stan/math/matrix/EigenDenseBaseAddons.hpp".  According to
// ambrose@google.com, this is a known limitation: the include
// scanner doesn't maintain any preprocessor state about macros,
// previously visited files, etc.  See also //base/stacktrace.cc.
#   ifdef STAN_MATH_MATRIX_EIGEN_DENSEBASE_PLUGIN
#     include "stan/math/matrix/EigenDenseBaseAddons.hpp"
#   endif
#undef EIGEN_CURRENT_STORAGE_BASE_CLASS

#ifdef EIGEN2_SUPPORT

    Block<Derived> corner(CornerType type, Index cRows, Index cCols);
    const Block<Derived> corner(CornerType type, Index cRows, Index cCols) const;
    template<int CRows, int CCols>
    Block<Derived, CRows, CCols> corner(CornerType type);
    template<int CRows, int CCols>
    const Block<Derived, CRows, CCols> corner(CornerType type) const;

#endif // EIGEN2_SUPPORT


    // disable the use of evalTo for dense objects with a nice compilation error
    template<typename Dest>
    EIGEN_DEVICE_FUNC
    inline void evalTo(Dest& ) const
    {
      EIGEN_STATIC_ASSERT((internal::is_same<Dest,void>::value),THE_EVAL_EVALTO_FUNCTION_SHOULD_NEVER_BE_CALLED_FOR_DENSE_OBJECTS);
    }

  protected:
    /** Default constructor. Do nothing. */
    EIGEN_DEVICE_FUNC DenseBase()
    {
      /* Just checks for self-consistency of the flags.
       * Only do it when debugging Eigen, as this borders on paranoiac and could slow compilation down
       */
#ifdef EIGEN_INTERNAL_DEBUGGING
      EIGEN_STATIC_ASSERT((EIGEN_IMPLIES(MaxRowsAtCompileTime==1 && MaxColsAtCompileTime!=1, int(IsRowMajor))
                        && EIGEN_IMPLIES(MaxColsAtCompileTime==1 && MaxRowsAtCompileTime!=1, int(!IsRowMajor))),
                          INVALID_STORAGE_ORDER_FOR_THIS_VECTOR_EXPRESSION)
#endif
    }

  private:
    EIGEN_DEVICE_FUNC explicit DenseBase(int);
    EIGEN_DEVICE_FUNC DenseBase(int,int);
    template<typename OtherDerived> EIGEN_DEVICE_FUNC explicit DenseBase(const DenseBase<OtherDerived>&);
};

} // end namespace Eigen

#endif // EIGEN_DENSEBASE_H
