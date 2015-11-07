// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PARTIAL_REDUX_H
#define EIGEN_PARTIAL_REDUX_H

namespace Eigen { 

/** \class PartialReduxExpr
  * \ingroup Core_Module
  *
  * \brief Generic expression of a partially reduxed matrix
  *
  * \tparam MatrixType the type of the matrix we are applying the redux operation
  * \tparam MemberOp type of the member functor
  * \tparam Direction indicates the direction of the redux (#Vertical or #Horizontal)
  *
  * This class represents an expression of a partial redux operator of a matrix.
  * It is the return type of some VectorwiseOp functions,
  * and most of the time this is the only way it is used.
  *
  * \sa class VectorwiseOp
  */

template< typename MatrixType, typename MemberOp, int Direction>
class PartialReduxExpr;

namespace internal {
template<typename MatrixType, typename MemberOp, int Direction>
struct traits<PartialReduxExpr<MatrixType, MemberOp, Direction> >
 : traits<MatrixType>
{
  typedef typename MemberOp::result_type Scalar;
  typedef typename traits<MatrixType>::StorageKind StorageKind;
  typedef typename traits<MatrixType>::XprKind XprKind;
  typedef typename MatrixType::Scalar InputScalar;
  typedef typename nested<MatrixType>::type MatrixTypeNested;
  typedef typename remove_all<MatrixTypeNested>::type _MatrixTypeNested;
  enum {
    RowsAtCompileTime = Direction==Vertical   ? 1 : MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = Direction==Horizontal ? 1 : MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = Direction==Vertical   ? 1 : MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = Direction==Horizontal ? 1 : MatrixType::MaxColsAtCompileTime,
    Flags0 = (unsigned int)_MatrixTypeNested::Flags & HereditaryBits,
    Flags = (Flags0 & ~RowMajorBit) | (RowsAtCompileTime == 1 ? RowMajorBit : 0),
    TraversalSize = Direction==Vertical ? MatrixType::RowsAtCompileTime :  MatrixType::ColsAtCompileTime
  };
  #if EIGEN_GNUC_AT_LEAST(3,4)
  typedef typename MemberOp::template Cost<InputScalar,int(TraversalSize)> CostOpType;
  #else
  typedef typename MemberOp::template Cost<InputScalar,TraversalSize> CostOpType;
  #endif
  enum {
    CoeffReadCost = TraversalSize==Dynamic ? Dynamic
                  : TraversalSize * traits<_MatrixTypeNested>::CoeffReadCost + int(CostOpType::value)
  };
};
}

template< typename MatrixType, typename MemberOp, int Direction>
class PartialReduxExpr : internal::no_assignment_operator,
  public internal::dense_xpr_base< PartialReduxExpr<MatrixType, MemberOp, Direction> >::type
{
  public:

    typedef typename internal::dense_xpr_base<PartialReduxExpr>::type Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(PartialReduxExpr)
    typedef typename internal::traits<PartialReduxExpr>::MatrixTypeNested MatrixTypeNested;
    typedef typename internal::traits<PartialReduxExpr>::_MatrixTypeNested _MatrixTypeNested;

    PartialReduxExpr(const MatrixType& mat, const MemberOp& func = MemberOp())
      : m_matrix(mat), m_functor(func) {}

    Index rows() const { return (Direction==Vertical   ? 1 : m_matrix.rows()); }
    Index cols() const { return (Direction==Horizontal ? 1 : m_matrix.cols()); }

    EIGEN_STRONG_INLINE const Scalar coeff(Index i, Index j) const
    {
      if (Direction==Vertical)
        return m_functor(m_matrix.col(j));
      else
        return m_functor(m_matrix.row(i));
    }

    const Scalar coeff(Index index) const
    {
      if (Direction==Vertical)
        return m_functor(m_matrix.col(index));
      else
        return m_functor(m_matrix.row(index));
    }

  protected:
    MatrixTypeNested m_matrix;
    const MemberOp m_functor;
};

#define EIGEN_MEMBER_FUNCTOR(MEMBER,COST)                               \
  template <typename ResultType>                                        \
  struct member_##MEMBER {                                              \
    EIGEN_EMPTY_STRUCT_CTOR(member_##MEMBER)                            \
    typedef ResultType result_type;                                     \
    template<typename Scalar, int Size> struct Cost                     \
    { enum { value = COST }; };                                         \
    template<typename XprType>                                          \
    EIGEN_STRONG_INLINE ResultType operator()(const XprType& mat) const \
    { return mat.MEMBER(); } \
  }

namespace internal {

EIGEN_MEMBER_FUNCTOR(squaredNorm, Size * NumTraits<Scalar>::MulCost + (Size-1)*NumTraits<Scalar>::AddCost);
EIGEN_MEMBER_FUNCTOR(norm, (Size+5) * NumTraits<Scalar>::MulCost + (Size-1)*NumTraits<Scalar>::AddCost);
EIGEN_MEMBER_FUNCTOR(stableNorm, (Size+5) * NumTraits<Scalar>::MulCost + (Size-1)*NumTraits<Scalar>::AddCost);
EIGEN_MEMBER_FUNCTOR(blueNorm, (Size+5) * NumTraits<Scalar>::MulCost + (Size-1)*NumTraits<Scalar>::AddCost);
EIGEN_MEMBER_FUNCTOR(hypotNorm, (Size-1) * functor_traits<scalar_hypot_op<Scalar> >::Cost );
EIGEN_MEMBER_FUNCTOR(sum, (Size-1)*NumTraits<Scalar>::AddCost);
EIGEN_MEMBER_FUNCTOR(mean, (Size-1)*NumTraits<Scalar>::AddCost + NumTraits<Scalar>::MulCost);
EIGEN_MEMBER_FUNCTOR(minCoeff, (Size-1)*NumTraits<Scalar>::AddCost);
EIGEN_MEMBER_FUNCTOR(maxCoeff, (Size-1)*NumTraits<Scalar>::AddCost);
EIGEN_MEMBER_FUNCTOR(all, (Size-1)*NumTraits<Scalar>::AddCost);
EIGEN_MEMBER_FUNCTOR(any, (Size-1)*NumTraits<Scalar>::AddCost);
EIGEN_MEMBER_FUNCTOR(count, (Size-1)*NumTraits<Scalar>::AddCost);
EIGEN_MEMBER_FUNCTOR(prod, (Size-1)*NumTraits<Scalar>::MulCost);


template <typename BinaryOp, typename Scalar>
struct member_redux {
  typedef typename result_of<
                     BinaryOp(Scalar)
                   >::type  result_type;
  template<typename _Scalar, int Size> struct Cost
  { enum { value = (Size-1) * functor_traits<BinaryOp>::Cost }; };
  member_redux(const BinaryOp func) : m_functor(func) {}
  template<typename Derived>
  inline result_type operator()(const DenseBase<Derived>& mat) const
  { return mat.redux(m_functor); }
  const BinaryOp m_functor;
};
}

/** \class VectorwiseOp
  * \ingroup Core_Module
  *
  * \brief Pseudo expression providing partial reduction operations
  *
  * \param ExpressionType the type of the object on which to do partial reductions
  * \param Direction indicates the direction of the redux (#Vertical or #Horizontal)
  *
  * This class represents a pseudo expression with partial reduction features.
  * It is the return type of DenseBase::colwise() and DenseBase::rowwise()
  * and most of the time this is the only way it is used.
  *
  * Example: \include MatrixBase_colwise.cpp
  * Output: \verbinclude MatrixBase_colwise.out
  *
  * \sa DenseBase::colwise(), DenseBase::rowwise(), class PartialReduxExpr
  */
template<typename ExpressionType, int Direction> class VectorwiseOp
{
  public:

    typedef typename ExpressionType::Scalar Scalar;
    typedef typename ExpressionType::RealScalar RealScalar;
    typedef typename ExpressionType::Index Index;
    typedef typename internal::conditional<internal::must_nest_by_value<ExpressionType>::ret,
        ExpressionType, ExpressionType&>::type ExpressionTypeNested;
    typedef typename internal::remove_all<ExpressionTypeNested>::type ExpressionTypeNestedCleaned;

    template<template<typename _Scalar> class Functor,
                      typename Scalar=typename internal::traits<ExpressionType>::Scalar> struct ReturnType
    {
      typedef PartialReduxExpr<ExpressionType,
                               Functor<Scalar>,
                               Direction
                              > Type;
    };

    template<typename BinaryOp> struct ReduxReturnType
    {
      typedef PartialReduxExpr<ExpressionType,
                               internal::member_redux<BinaryOp,typename internal::traits<ExpressionType>::Scalar>,
                               Direction
                              > Type;
    };

    enum {
      IsVertical   = (Direction==Vertical) ? 1 : 0,
      IsHorizontal = (Direction==Horizontal) ? 1 : 0
    };

  protected:

    /** \internal
      * \returns the i-th subvector according to the \c Direction */
    typedef typename internal::conditional<Direction==Vertical,
                               typename ExpressionType::ColXpr,
                               typename ExpressionType::RowXpr>::type SubVector;
    SubVector subVector(Index i)
    {
      return SubVector(m_matrix.derived(),i);
    }

    /** \internal
      * \returns the number of subvectors in the direction \c Direction */
    Index subVectors() const
    { return Direction==Vertical?m_matrix.cols():m_matrix.rows(); }

    template<typename OtherDerived> struct ExtendedType {
      typedef Replicate<OtherDerived,
                        Direction==Vertical   ? 1 : ExpressionType::RowsAtCompileTime,
                        Direction==Horizontal ? 1 : ExpressionType::ColsAtCompileTime> Type;
    };

    /** \internal
      * Replicates a vector to match the size of \c *this */
    template<typename OtherDerived>
    typename ExtendedType<OtherDerived>::Type
    extendedTo(const DenseBase<OtherDerived>& other) const
    {
      EIGEN_STATIC_ASSERT(EIGEN_IMPLIES(Direction==Vertical, OtherDerived::MaxColsAtCompileTime==1),
                          YOU_PASSED_A_ROW_VECTOR_BUT_A_COLUMN_VECTOR_WAS_EXPECTED)
      EIGEN_STATIC_ASSERT(EIGEN_IMPLIES(Direction==Horizontal, OtherDerived::MaxRowsAtCompileTime==1),
                          YOU_PASSED_A_COLUMN_VECTOR_BUT_A_ROW_VECTOR_WAS_EXPECTED)
      return typename ExtendedType<OtherDerived>::Type
                      (other.derived(),
                       Direction==Vertical   ? 1 : m_matrix.rows(),
                       Direction==Horizontal ? 1 : m_matrix.cols());
    }
    
    template<typename OtherDerived> struct OppositeExtendedType {
      typedef Replicate<OtherDerived,
                        Direction==Horizontal ? 1 : ExpressionType::RowsAtCompileTime,
                        Direction==Vertical   ? 1 : ExpressionType::ColsAtCompileTime> Type;
    };

    /** \internal
      * Replicates a vector in the opposite direction to match the size of \c *this */
    template<typename OtherDerived>
    typename OppositeExtendedType<OtherDerived>::Type
    extendedToOpposite(const DenseBase<OtherDerived>& other) const
    {
      EIGEN_STATIC_ASSERT(EIGEN_IMPLIES(Direction==Horizontal, OtherDerived::MaxColsAtCompileTime==1),
                          YOU_PASSED_A_ROW_VECTOR_BUT_A_COLUMN_VECTOR_WAS_EXPECTED)
      EIGEN_STATIC_ASSERT(EIGEN_IMPLIES(Direction==Vertical, OtherDerived::MaxRowsAtCompileTime==1),
                          YOU_PASSED_A_COLUMN_VECTOR_BUT_A_ROW_VECTOR_WAS_EXPECTED)
      return typename OppositeExtendedType<OtherDerived>::Type
                      (other.derived(),
                       Direction==Horizontal  ? 1 : m_matrix.rows(),
                       Direction==Vertical    ? 1 : m_matrix.cols());
    }

  public:

    inline VectorwiseOp(ExpressionType& matrix) : m_matrix(matrix) {}

    /** \internal */
    inline const ExpressionType& _expression() const { return m_matrix; }

    /** \returns a row or column vector expression of \c *this reduxed by \a func
      *
      * The template parameter \a BinaryOp is the type of the functor
      * of the custom redux operator. Note that func must be an associative operator.
      *
      * \sa class VectorwiseOp, DenseBase::colwise(), DenseBase::rowwise()
      */
    template<typename BinaryOp>
    const typename ReduxReturnType<BinaryOp>::Type
    redux(const BinaryOp& func = BinaryOp()) const
    { return typename ReduxReturnType<BinaryOp>::Type(_expression(), func); }

    /** \returns a row (or column) vector expression of the smallest coefficient
      * of each column (or row) of the referenced expression.
      * 
      * \warning the result is undefined if \c *this contains NaN.
      *
      * Example: \include PartialRedux_minCoeff.cpp
      * Output: \verbinclude PartialRedux_minCoeff.out
      *
      * \sa DenseBase::minCoeff() */
    const typename ReturnType<internal::member_minCoeff>::Type minCoeff() const
    { return _expression(); }

    /** \returns a row (or column) vector expression of the largest coefficient
      * of each column (or row) of the referenced expression.
      * 
      * \warning the result is undefined if \c *this contains NaN.
      *
      * Example: \include PartialRedux_maxCoeff.cpp
      * Output: \verbinclude PartialRedux_maxCoeff.out
      *
      * \sa DenseBase::maxCoeff() */
    const typename ReturnType<internal::member_maxCoeff>::Type maxCoeff() const
    { return _expression(); }

    /** \returns a row (or column) vector expression of the squared norm
      * of each column (or row) of the referenced expression.
      * This is a vector with real entries, even if the original matrix has complex entries.
      *
      * Example: \include PartialRedux_squaredNorm.cpp
      * Output: \verbinclude PartialRedux_squaredNorm.out
      *
      * \sa DenseBase::squaredNorm() */
    const typename ReturnType<internal::member_squaredNorm,RealScalar>::Type squaredNorm() const
    { return _expression(); }

    /** \returns a row (or column) vector expression of the norm
      * of each column (or row) of the referenced expression.
      * This is a vector with real entries, even if the original matrix has complex entries.
      *
      * Example: \include PartialRedux_norm.cpp
      * Output: \verbinclude PartialRedux_norm.out
      *
      * \sa DenseBase::norm() */
    const typename ReturnType<internal::member_norm,RealScalar>::Type norm() const
    { return _expression(); }


    /** \returns a row (or column) vector expression of the norm
      * of each column (or row) of the referenced expression, using
      * Blue's algorithm. 
      * This is a vector with real entries, even if the original matrix has complex entries.
      *
      * \sa DenseBase::blueNorm() */
    const typename ReturnType<internal::member_blueNorm,RealScalar>::Type blueNorm() const
    { return _expression(); }


    /** \returns a row (or column) vector expression of the norm
      * of each column (or row) of the referenced expression, avoiding
      * underflow and overflow.
      * This is a vector with real entries, even if the original matrix has complex entries.
      *
      * \sa DenseBase::stableNorm() */
    const typename ReturnType<internal::member_stableNorm,RealScalar>::Type stableNorm() const
    { return _expression(); }


    /** \returns a row (or column) vector expression of the norm
      * of each column (or row) of the referenced expression, avoiding
      * underflow and overflow using a concatenation of hypot() calls.
      * This is a vector with real entries, even if the original matrix has complex entries.
      *
      * \sa DenseBase::hypotNorm() */
    const typename ReturnType<internal::member_hypotNorm,RealScalar>::Type hypotNorm() const
    { return _expression(); }

    /** \returns a row (or column) vector expression of the sum
      * of each column (or row) of the referenced expression.
      *
      * Example: \include PartialRedux_sum.cpp
      * Output: \verbinclude PartialRedux_sum.out
      *
      * \sa DenseBase::sum() */
    const typename ReturnType<internal::member_sum>::Type sum() const
    { return _expression(); }

    /** \returns a row (or column) vector expression of the mean
    * of each column (or row) of the referenced expression.
    *
    * \sa DenseBase::mean() */
    const typename ReturnType<internal::member_mean>::Type mean() const
    { return _expression(); }

    /** \returns a row (or column) vector expression representing
      * whether \b all coefficients of each respective column (or row) are \c true.
      * This expression can be assigned to a vector with entries of type \c bool.
      *
      * \sa DenseBase::all() */
    const typename ReturnType<internal::member_all>::Type all() const
    { return _expression(); }

    /** \returns a row (or column) vector expression representing
      * whether \b at \b least one coefficient of each respective column (or row) is \c true.
      * This expression can be assigned to a vector with entries of type \c bool.
      *
      * \sa DenseBase::any() */
    const typename ReturnType<internal::member_any>::Type any() const
    { return _expression(); }

    /** \returns a row (or column) vector expression representing
      * the number of \c true coefficients of each respective column (or row).
      * This expression can be assigned to a vector whose entries have the same type as is used to
      * index entries of the original matrix; for dense matrices, this is \c std::ptrdiff_t .
      *
      * Example: \include PartialRedux_count.cpp
      * Output: \verbinclude PartialRedux_count.out
      *
      * \sa DenseBase::count() */
    const PartialReduxExpr<ExpressionType, internal::member_count<Index>, Direction> count() const
    { return _expression(); }

    /** \returns a row (or column) vector expression of the product
      * of each column (or row) of the referenced expression.
      *
      * Example: \include PartialRedux_prod.cpp
      * Output: \verbinclude PartialRedux_prod.out
      *
      * \sa DenseBase::prod() */
    const typename ReturnType<internal::member_prod>::Type prod() const
    { return _expression(); }


    /** \returns a matrix expression
      * where each column (or row) are reversed.
      *
      * Example: \include Vectorwise_reverse.cpp
      * Output: \verbinclude Vectorwise_reverse.out
      *
      * \sa DenseBase::reverse() */
    const Reverse<ExpressionType, Direction> reverse() const
    { return Reverse<ExpressionType, Direction>( _expression() ); }

    typedef Replicate<ExpressionType,Direction==Vertical?Dynamic:1,Direction==Horizontal?Dynamic:1> ReplicateReturnType;
    const ReplicateReturnType replicate(Index factor) const;

    /**
      * \return an expression of the replication of each column (or row) of \c *this
      *
      * Example: \include DirectionWise_replicate.cpp
      * Output: \verbinclude DirectionWise_replicate.out
      *
      * \sa VectorwiseOp::replicate(Index), DenseBase::replicate(), class Replicate
      */
    // NOTE implemented here because of sunstudio's compilation errors
    template<int Factor> const Replicate<ExpressionType,(IsVertical?Factor:1),(IsHorizontal?Factor:1)>
    replicate(Index factor = Factor) const
    {
      return Replicate<ExpressionType,Direction==Vertical?Factor:1,Direction==Horizontal?Factor:1>
          (_expression(),Direction==Vertical?factor:1,Direction==Horizontal?factor:1);
    }

/////////// Artithmetic operators ///////////

    /** Copies the vector \a other to each subvector of \c *this */
    template<typename OtherDerived>
    ExpressionType& operator=(const DenseBase<OtherDerived>& other)
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(OtherDerived)
      EIGEN_STATIC_ASSERT_SAME_XPR_KIND(ExpressionType, OtherDerived)
      //eigen_assert((m_matrix.isNull()) == (other.isNull())); FIXME
      return const_cast<ExpressionType&>(m_matrix = extendedTo(other.derived()));
    }

    /** Adds the vector \a other to each subvector of \c *this */
    template<typename OtherDerived>
    ExpressionType& operator+=(const DenseBase<OtherDerived>& other)
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(OtherDerived)
      EIGEN_STATIC_ASSERT_SAME_XPR_KIND(ExpressionType, OtherDerived)
      return const_cast<ExpressionType&>(m_matrix += extendedTo(other.derived()));
    }

    /** Substracts the vector \a other to each subvector of \c *this */
    template<typename OtherDerived>
    ExpressionType& operator-=(const DenseBase<OtherDerived>& other)
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(OtherDerived)
      EIGEN_STATIC_ASSERT_SAME_XPR_KIND(ExpressionType, OtherDerived)
      return const_cast<ExpressionType&>(m_matrix -= extendedTo(other.derived()));
    }

    /** Multiples each subvector of \c *this by the vector \a other */
    template<typename OtherDerived>
    ExpressionType& operator*=(const DenseBase<OtherDerived>& other)
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(OtherDerived)
      EIGEN_STATIC_ASSERT_ARRAYXPR(ExpressionType)
      EIGEN_STATIC_ASSERT_SAME_XPR_KIND(ExpressionType, OtherDerived)
      m_matrix *= extendedTo(other.derived());
      return const_cast<ExpressionType&>(m_matrix);
    }

    /** Divides each subvector of \c *this by the vector \a other */
    template<typename OtherDerived>
    ExpressionType& operator/=(const DenseBase<OtherDerived>& other)
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(OtherDerived)
      EIGEN_STATIC_ASSERT_ARRAYXPR(ExpressionType)
      EIGEN_STATIC_ASSERT_SAME_XPR_KIND(ExpressionType, OtherDerived)
      m_matrix /= extendedTo(other.derived());
      return const_cast<ExpressionType&>(m_matrix);
    }

    /** Returns the expression of the sum of the vector \a other to each subvector of \c *this */
    template<typename OtherDerived> EIGEN_STRONG_INLINE
    CwiseBinaryOp<internal::scalar_sum_op<Scalar>,
                  const ExpressionTypeNestedCleaned,
                  const typename ExtendedType<OtherDerived>::Type>
    operator+(const DenseBase<OtherDerived>& other) const
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(OtherDerived)
      EIGEN_STATIC_ASSERT_SAME_XPR_KIND(ExpressionType, OtherDerived)
      return m_matrix + extendedTo(other.derived());
    }

    /** Returns the expression of the difference between each subvector of \c *this and the vector \a other */
    template<typename OtherDerived>
    CwiseBinaryOp<internal::scalar_difference_op<Scalar>,
                  const ExpressionTypeNestedCleaned,
                  const typename ExtendedType<OtherDerived>::Type>
    operator-(const DenseBase<OtherDerived>& other) const
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(OtherDerived)
      EIGEN_STATIC_ASSERT_SAME_XPR_KIND(ExpressionType, OtherDerived)
      return m_matrix - extendedTo(other.derived());
    }

    /** Returns the expression where each subvector is the product of the vector \a other
      * by the corresponding subvector of \c *this */
    template<typename OtherDerived> EIGEN_STRONG_INLINE
    CwiseBinaryOp<internal::scalar_product_op<Scalar>,
                  const ExpressionTypeNestedCleaned,
                  const typename ExtendedType<OtherDerived>::Type>
    operator*(const DenseBase<OtherDerived>& other) const
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(OtherDerived)
      EIGEN_STATIC_ASSERT_ARRAYXPR(ExpressionType)
      EIGEN_STATIC_ASSERT_SAME_XPR_KIND(ExpressionType, OtherDerived)
      return m_matrix * extendedTo(other.derived());
    }

    /** Returns the expression where each subvector is the quotient of the corresponding
      * subvector of \c *this by the vector \a other */
    template<typename OtherDerived>
    CwiseBinaryOp<internal::scalar_quotient_op<Scalar>,
                  const ExpressionTypeNestedCleaned,
                  const typename ExtendedType<OtherDerived>::Type>
    operator/(const DenseBase<OtherDerived>& other) const
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(OtherDerived)
      EIGEN_STATIC_ASSERT_ARRAYXPR(ExpressionType)
      EIGEN_STATIC_ASSERT_SAME_XPR_KIND(ExpressionType, OtherDerived)
      return m_matrix / extendedTo(other.derived());
    }
    
    /** \returns an expression where each column of row of the referenced matrix are normalized.
      * The referenced matrix is \b not modified.
      * \sa MatrixBase::normalized(), normalize()
      */
    CwiseBinaryOp<internal::scalar_quotient_op<Scalar>,
                  const ExpressionTypeNestedCleaned,
                  const typename OppositeExtendedType<typename ReturnType<internal::member_norm,RealScalar>::Type>::Type>
    normalized() const { return m_matrix.cwiseQuotient(extendedToOpposite(this->norm())); }
    
    
    /** Normalize in-place each row or columns of the referenced matrix.
      * \sa MatrixBase::normalize(), normalized()
      */
    void normalize() {
      m_matrix = this->normalized();
    }

/////////// Geometry module ///////////

    #if EIGEN2_SUPPORT_STAGE > STAGE20_RESOLVE_API_CONFLICTS
    Homogeneous<ExpressionType,Direction> homogeneous() const;
    #endif

    typedef typename ExpressionType::PlainObject CrossReturnType;
    template<typename OtherDerived>
    const CrossReturnType cross(const MatrixBase<OtherDerived>& other) const;

    enum {
      HNormalized_Size = Direction==Vertical ? internal::traits<ExpressionType>::RowsAtCompileTime
                                             : internal::traits<ExpressionType>::ColsAtCompileTime,
      HNormalized_SizeMinusOne = HNormalized_Size==Dynamic ? Dynamic : HNormalized_Size-1
    };
    typedef Block<const ExpressionType,
                  Direction==Vertical   ? int(HNormalized_SizeMinusOne)
                                        : int(internal::traits<ExpressionType>::RowsAtCompileTime),
                  Direction==Horizontal ? int(HNormalized_SizeMinusOne)
                                        : int(internal::traits<ExpressionType>::ColsAtCompileTime)>
            HNormalized_Block;
    typedef Block<const ExpressionType,
                  Direction==Vertical   ? 1 : int(internal::traits<ExpressionType>::RowsAtCompileTime),
                  Direction==Horizontal ? 1 : int(internal::traits<ExpressionType>::ColsAtCompileTime)>
            HNormalized_Factors;
    typedef CwiseBinaryOp<internal::scalar_quotient_op<typename internal::traits<ExpressionType>::Scalar>,
                const HNormalized_Block,
                const Replicate<HNormalized_Factors,
                  Direction==Vertical   ? HNormalized_SizeMinusOne : 1,
                  Direction==Horizontal ? HNormalized_SizeMinusOne : 1> >
            HNormalizedReturnType;

    const HNormalizedReturnType hnormalized() const;

  protected:
    ExpressionTypeNested m_matrix;
};

/** \returns a VectorwiseOp wrapper of *this providing additional partial reduction operations
  *
  * Example: \include MatrixBase_colwise.cpp
  * Output: \verbinclude MatrixBase_colwise.out
  *
  * \sa rowwise(), class VectorwiseOp, \ref TutorialReductionsVisitorsBroadcasting
  */
template<typename Derived>
inline const typename DenseBase<Derived>::ConstColwiseReturnType
DenseBase<Derived>::colwise() const
{
  return derived();
}

/** \returns a writable VectorwiseOp wrapper of *this providing additional partial reduction operations
  *
  * \sa rowwise(), class VectorwiseOp, \ref TutorialReductionsVisitorsBroadcasting
  */
template<typename Derived>
inline typename DenseBase<Derived>::ColwiseReturnType
DenseBase<Derived>::colwise()
{
  return derived();
}

/** \returns a VectorwiseOp wrapper of *this providing additional partial reduction operations
  *
  * Example: \include MatrixBase_rowwise.cpp
  * Output: \verbinclude MatrixBase_rowwise.out
  *
  * \sa colwise(), class VectorwiseOp, \ref TutorialReductionsVisitorsBroadcasting
  */
template<typename Derived>
inline const typename DenseBase<Derived>::ConstRowwiseReturnType
DenseBase<Derived>::rowwise() const
{
  return derived();
}

/** \returns a writable VectorwiseOp wrapper of *this providing additional partial reduction operations
  *
  * \sa colwise(), class VectorwiseOp, \ref TutorialReductionsVisitorsBroadcasting
  */
template<typename Derived>
inline typename DenseBase<Derived>::RowwiseReturnType
DenseBase<Derived>::rowwise()
{
  return derived();
}

} // end namespace Eigen

#endif // EIGEN_PARTIAL_REDUX_H
