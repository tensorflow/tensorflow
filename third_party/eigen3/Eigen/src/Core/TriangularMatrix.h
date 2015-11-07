// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TRIANGULARMATRIX_H
#define EIGEN_TRIANGULARMATRIX_H

namespace Eigen { 

namespace internal {
  
template<int Side, typename TriangularType, typename Rhs> struct triangular_solve_retval;
  
}

/** \internal
  *
  * \class TriangularBase
  * \ingroup Core_Module
  *
  * \brief Base class for triangular part in a matrix
  */
template<typename Derived> class TriangularBase : public EigenBase<Derived>
{
  public:

    enum {
      Mode = internal::traits<Derived>::Mode,
      CoeffReadCost = internal::traits<Derived>::CoeffReadCost,
      RowsAtCompileTime = internal::traits<Derived>::RowsAtCompileTime,
      ColsAtCompileTime = internal::traits<Derived>::ColsAtCompileTime,
      MaxRowsAtCompileTime = internal::traits<Derived>::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = internal::traits<Derived>::MaxColsAtCompileTime
    };
    typedef typename internal::traits<Derived>::Scalar Scalar;
    typedef typename internal::traits<Derived>::StorageKind StorageKind;
    typedef typename internal::traits<Derived>::Index Index;
    typedef typename internal::traits<Derived>::DenseMatrixType DenseMatrixType;
    typedef DenseMatrixType DenseType;

    EIGEN_DEVICE_FUNC
    inline TriangularBase() { eigen_assert(!((Mode&UnitDiag) && (Mode&ZeroDiag))); }

    EIGEN_DEVICE_FUNC
    inline Index rows() const { return derived().rows(); }
    EIGEN_DEVICE_FUNC
    inline Index cols() const { return derived().cols(); }
    EIGEN_DEVICE_FUNC
    inline Index outerStride() const { return derived().outerStride(); }
    EIGEN_DEVICE_FUNC
    inline Index innerStride() const { return derived().innerStride(); }

    EIGEN_DEVICE_FUNC
    inline Scalar coeff(Index row, Index col) const  { return derived().coeff(row,col); }
    EIGEN_DEVICE_FUNC
    inline Scalar& coeffRef(Index row, Index col) { return derived().coeffRef(row,col); }

    /** \see MatrixBase::copyCoeff(row,col)
      */
    template<typename Other>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE void copyCoeff(Index row, Index col, Other& other)
    {
      derived().coeffRef(row, col) = other.coeff(row, col);
    }

    EIGEN_DEVICE_FUNC
    inline Scalar operator()(Index row, Index col) const
    {
      check_coordinates(row, col);
      return coeff(row,col);
    }
    EIGEN_DEVICE_FUNC
    inline Scalar& operator()(Index row, Index col)
    {
      check_coordinates(row, col);
      return coeffRef(row,col);
    }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    EIGEN_DEVICE_FUNC
    inline const Derived& derived() const { return *static_cast<const Derived*>(this); }
    EIGEN_DEVICE_FUNC
    inline Derived& derived() { return *static_cast<Derived*>(this); }
    #endif // not EIGEN_PARSED_BY_DOXYGEN

    template<typename DenseDerived>
    EIGEN_DEVICE_FUNC
    void evalTo(MatrixBase<DenseDerived> &other) const;
    template<typename DenseDerived>
    EIGEN_DEVICE_FUNC
    void evalToLazy(MatrixBase<DenseDerived> &other) const;

    EIGEN_DEVICE_FUNC
    DenseMatrixType toDenseMatrix() const
    {
      DenseMatrixType res(rows(), cols());
      evalToLazy(res);
      return res;
    }

  protected:

    void check_coordinates(Index row, Index col) const
    {
      EIGEN_ONLY_USED_FOR_DEBUG(row);
      EIGEN_ONLY_USED_FOR_DEBUG(col);
      eigen_assert(col>=0 && col<cols() && row>=0 && row<rows());
      const int mode = int(Mode) & ~SelfAdjoint;
      EIGEN_ONLY_USED_FOR_DEBUG(mode);
      eigen_assert((mode==Upper && col>=row)
                || (mode==Lower && col<=row)
                || ((mode==StrictlyUpper || mode==UnitUpper) && col>row)
                || ((mode==StrictlyLower || mode==UnitLower) && col<row));
    }

    #ifdef EIGEN_INTERNAL_DEBUGGING
    void check_coordinates_internal(Index row, Index col) const
    {
      check_coordinates(row, col);
    }
    #else
    void check_coordinates_internal(Index , Index ) const {}
    #endif

};

/** \class TriangularView
  * \ingroup Core_Module
  *
  * \brief Base class for triangular part in a matrix
  *
  * \param MatrixType the type of the object in which we are taking the triangular part
  * \param Mode the kind of triangular matrix expression to construct. Can be #Upper,
  *             #Lower, #UnitUpper, #UnitLower, #StrictlyUpper, or #StrictlyLower.
  *             This is in fact a bit field; it must have either #Upper or #Lower, 
  *             and additionnaly it may have #UnitDiag or #ZeroDiag or neither.
  *
  * This class represents a triangular part of a matrix, not necessarily square. Strictly speaking, for rectangular
  * matrices one should speak of "trapezoid" parts. This class is the return type
  * of MatrixBase::triangularView() and most of the time this is the only way it is used.
  *
  * \sa MatrixBase::triangularView()
  */
namespace internal {
template<typename MatrixType, unsigned int _Mode>
struct traits<TriangularView<MatrixType, _Mode> > : traits<MatrixType>
{
  typedef typename nested<MatrixType>::type MatrixTypeNested;
  typedef typename remove_reference<MatrixTypeNested>::type MatrixTypeNestedNonRef;
  typedef typename remove_all<MatrixTypeNested>::type MatrixTypeNestedCleaned;
  typedef MatrixType ExpressionType;
  typedef typename MatrixType::PlainObject DenseMatrixType;
  enum {
    Mode = _Mode,
    Flags = (MatrixTypeNestedCleaned::Flags & (HereditaryBits) & (~(PacketAccessBit | DirectAccessBit | LinearAccessBit))) | Mode,
    CoeffReadCost = MatrixTypeNestedCleaned::CoeffReadCost
  };
};
}

template<int Mode, bool LhsIsTriangular,
         typename Lhs, bool LhsIsVector,
         typename Rhs, bool RhsIsVector>
struct TriangularProduct;

template<typename _MatrixType, unsigned int _Mode> class TriangularView
  : public TriangularBase<TriangularView<_MatrixType, _Mode> >
{
  public:

    typedef TriangularBase<TriangularView> Base;
    typedef typename internal::traits<TriangularView>::Scalar Scalar;

    typedef _MatrixType MatrixType;
    typedef typename internal::traits<TriangularView>::DenseMatrixType DenseMatrixType;
    typedef DenseMatrixType PlainObject;

  protected:
    typedef typename internal::traits<TriangularView>::MatrixTypeNested MatrixTypeNested;
    typedef typename internal::traits<TriangularView>::MatrixTypeNestedNonRef MatrixTypeNestedNonRef;
    typedef typename internal::traits<TriangularView>::MatrixTypeNestedCleaned MatrixTypeNestedCleaned;

    typedef typename internal::remove_all<typename MatrixType::ConjugateReturnType>::type MatrixConjugateReturnType;
    
  public:
    using Base::evalToLazy;
  

    typedef typename internal::traits<TriangularView>::StorageKind StorageKind;
    typedef typename internal::traits<TriangularView>::Index Index;

    enum {
      Mode = _Mode,
      TransposeMode = (Mode & Upper ? Lower : 0)
                    | (Mode & Lower ? Upper : 0)
                    | (Mode & (UnitDiag))
                    | (Mode & (ZeroDiag))
    };

    EIGEN_DEVICE_FUNC
    inline TriangularView(const MatrixType& matrix) : m_matrix(matrix)
    {}

    EIGEN_DEVICE_FUNC
    inline Index rows() const { return m_matrix.rows(); }
    EIGEN_DEVICE_FUNC
    inline Index cols() const { return m_matrix.cols(); }
    EIGEN_DEVICE_FUNC
    inline Index outerStride() const { return m_matrix.outerStride(); }
    EIGEN_DEVICE_FUNC
    inline Index innerStride() const { return m_matrix.innerStride(); }

    /** \sa MatrixBase::operator+=() */    
    template<typename Other>
    EIGEN_DEVICE_FUNC
    TriangularView&  operator+=(const DenseBase<Other>& other) { return *this = m_matrix + other.derived(); }
    /** \sa MatrixBase::operator-=() */
    template<typename Other>
    EIGEN_DEVICE_FUNC
    TriangularView&  operator-=(const DenseBase<Other>& other) { return *this = m_matrix - other.derived(); }
    /** \sa MatrixBase::operator*=() */
    EIGEN_DEVICE_FUNC
    TriangularView&  operator*=(const typename internal::traits<MatrixType>::Scalar& other) { return *this = m_matrix * other; }
    /** \sa MatrixBase::operator/=() */
    EIGEN_DEVICE_FUNC
    TriangularView&  operator/=(const typename internal::traits<MatrixType>::Scalar& other) { return *this = m_matrix / other; }

    /** \sa MatrixBase::fill() */
    EIGEN_DEVICE_FUNC
    void fill(const Scalar& value) { setConstant(value); }
    /** \sa MatrixBase::setConstant() */
    EIGEN_DEVICE_FUNC
    TriangularView& setConstant(const Scalar& value)
    { return *this = MatrixType::Constant(rows(), cols(), value); }
    /** \sa MatrixBase::setZero() */
    EIGEN_DEVICE_FUNC
    TriangularView& setZero() { return setConstant(Scalar(0)); }
    /** \sa MatrixBase::setOnes() */
    EIGEN_DEVICE_FUNC
    TriangularView& setOnes() { return setConstant(Scalar(1)); }

    /** \sa MatrixBase::coeff()
      * \warning the coordinates must fit into the referenced triangular part
      */
    EIGEN_DEVICE_FUNC
    inline Scalar coeff(Index row, Index col) const
    {
      Base::check_coordinates_internal(row, col);
      return m_matrix.coeff(row, col);
    }

    /** \sa MatrixBase::coeffRef()
      * \warning the coordinates must fit into the referenced triangular part
      */
    EIGEN_DEVICE_FUNC
    inline Scalar& coeffRef(Index row, Index col)
    {
      Base::check_coordinates_internal(row, col);
      return m_matrix.const_cast_derived().coeffRef(row, col);
    }

    EIGEN_DEVICE_FUNC
    const MatrixTypeNestedCleaned& nestedExpression() const { return m_matrix; }
    EIGEN_DEVICE_FUNC
    MatrixTypeNestedCleaned& nestedExpression() { return *const_cast<MatrixTypeNestedCleaned*>(&m_matrix); }

    /** Assigns a triangular matrix to a triangular part of a dense matrix */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    TriangularView& operator=(const TriangularBase<OtherDerived>& other);

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    TriangularView& operator=(const MatrixBase<OtherDerived>& other);

    EIGEN_DEVICE_FUNC
    TriangularView& operator=(const TriangularView& other)
    { return *this = other.nestedExpression(); }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    void lazyAssign(const TriangularBase<OtherDerived>& other);

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    void lazyAssign(const MatrixBase<OtherDerived>& other);

    /** \sa MatrixBase::conjugate() */
    EIGEN_DEVICE_FUNC
    inline TriangularView<MatrixConjugateReturnType,Mode> conjugate()
    { return m_matrix.conjugate(); }
    /** \sa MatrixBase::conjugate() const */
    EIGEN_DEVICE_FUNC
    inline const TriangularView<MatrixConjugateReturnType,Mode> conjugate() const
    { return m_matrix.conjugate(); }

    /** \sa MatrixBase::adjoint() const */
    EIGEN_DEVICE_FUNC
    inline const TriangularView<const typename MatrixType::AdjointReturnType,TransposeMode> adjoint() const
    { return m_matrix.adjoint(); }

    /** \sa MatrixBase::transpose() */
    EIGEN_DEVICE_FUNC
    inline TriangularView<Transpose<MatrixType>,TransposeMode> transpose()
    {
      EIGEN_STATIC_ASSERT_LVALUE(MatrixType)
      return m_matrix.const_cast_derived().transpose();
    }
    /** \sa MatrixBase::transpose() const */
    EIGEN_DEVICE_FUNC
    inline const TriangularView<Transpose<MatrixType>,TransposeMode> transpose() const
    {
      return m_matrix.transpose();
    }

    /** Efficient triangular matrix times vector/matrix product */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    TriangularProduct<Mode,true,MatrixType,false,OtherDerived, OtherDerived::IsVectorAtCompileTime>
    operator*(const MatrixBase<OtherDerived>& rhs) const
    {
      return TriangularProduct
              <Mode,true,MatrixType,false,OtherDerived,OtherDerived::IsVectorAtCompileTime>
              (m_matrix, rhs.derived());
    }

    /** Efficient vector/matrix times triangular matrix product */
    template<typename OtherDerived> friend
    EIGEN_DEVICE_FUNC
    TriangularProduct<Mode,false,OtherDerived,OtherDerived::IsVectorAtCompileTime,MatrixType,false>
    operator*(const MatrixBase<OtherDerived>& lhs, const TriangularView& rhs)
    {
      return TriangularProduct
              <Mode,false,OtherDerived,OtherDerived::IsVectorAtCompileTime,MatrixType,false>
              (lhs.derived(),rhs.m_matrix);
    }

    #ifdef EIGEN2_SUPPORT
    template<typename OtherDerived>
    struct eigen2_product_return_type
    {
      typedef typename TriangularView<MatrixType,Mode>::DenseMatrixType DenseMatrixType;
      typedef typename OtherDerived::PlainObject::DenseType OtherPlainObject;
      typedef typename ProductReturnType<DenseMatrixType, OtherPlainObject>::Type ProdRetType;
      typedef typename ProdRetType::PlainObject type;
    };
    template<typename OtherDerived>
    const typename eigen2_product_return_type<OtherDerived>::type
    operator*(const EigenBase<OtherDerived>& rhs) const
    {
      typename OtherDerived::PlainObject::DenseType rhsPlainObject;
      rhs.evalTo(rhsPlainObject);
      return this->toDenseMatrix() * rhsPlainObject;
    }
    template<typename OtherMatrixType>
    bool isApprox(const TriangularView<OtherMatrixType, Mode>& other, typename NumTraits<Scalar>::Real precision = NumTraits<Scalar>::dummy_precision()) const
    {
      return this->toDenseMatrix().isApprox(other.toDenseMatrix(), precision);
    }
    template<typename OtherDerived>
    bool isApprox(const MatrixBase<OtherDerived>& other, typename NumTraits<Scalar>::Real precision = NumTraits<Scalar>::dummy_precision()) const
    {
      return this->toDenseMatrix().isApprox(other, precision);
    }
    #endif // EIGEN2_SUPPORT

    template<int Side, typename Other>
    EIGEN_DEVICE_FUNC
    inline const internal::triangular_solve_retval<Side,TriangularView, Other>
    solve(const MatrixBase<Other>& other) const;

    template<int Side, typename OtherDerived>
    EIGEN_DEVICE_FUNC
    void solveInPlace(const MatrixBase<OtherDerived>& other) const;

    template<typename Other>
    EIGEN_DEVICE_FUNC
    inline const internal::triangular_solve_retval<OnTheLeft,TriangularView, Other> 
    solve(const MatrixBase<Other>& other) const
    { return solve<OnTheLeft>(other); }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    void solveInPlace(const MatrixBase<OtherDerived>& other) const
    { return solveInPlace<OnTheLeft>(other); }

    EIGEN_DEVICE_FUNC
    const SelfAdjointView<MatrixTypeNestedNonRef,Mode> selfadjointView() const
    {
      EIGEN_STATIC_ASSERT((Mode&UnitDiag)==0,PROGRAMMING_ERROR);
      return SelfAdjointView<MatrixTypeNestedNonRef,Mode>(m_matrix);
    }
    EIGEN_DEVICE_FUNC
    SelfAdjointView<MatrixTypeNestedNonRef,Mode> selfadjointView()
    {
      EIGEN_STATIC_ASSERT((Mode&UnitDiag)==0,PROGRAMMING_ERROR);
      return SelfAdjointView<MatrixTypeNestedNonRef,Mode>(m_matrix);
    }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    void swap(TriangularBase<OtherDerived> const & other)
    {
      TriangularView<SwapWrapper<MatrixType>,Mode>(const_cast<MatrixType&>(m_matrix)).lazyAssign(other.derived());
    }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    void swap(MatrixBase<OtherDerived> const & other)
    {
      SwapWrapper<MatrixType> swaper(const_cast<MatrixType&>(m_matrix));
      TriangularView<SwapWrapper<MatrixType>,Mode>(swaper).lazyAssign(other.derived());
    }

    EIGEN_DEVICE_FUNC
    Scalar determinant() const
    {
      if (Mode & UnitDiag)
        return 1;
      else if (Mode & ZeroDiag)
        return 0;
      else
        return m_matrix.diagonal().prod();
    }
    
    // TODO simplify the following:
    template<typename ProductDerived, typename Lhs, typename Rhs>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TriangularView& operator=(const ProductBase<ProductDerived, Lhs,Rhs>& other)
    {
      setZero();
      return assignProduct(other,1);
    }
    
    template<typename ProductDerived, typename Lhs, typename Rhs>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TriangularView& operator+=(const ProductBase<ProductDerived, Lhs,Rhs>& other)
    {
      return assignProduct(other,1);
    }
    
    template<typename ProductDerived, typename Lhs, typename Rhs>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TriangularView& operator-=(const ProductBase<ProductDerived, Lhs,Rhs>& other)
    {
      return assignProduct(other,-1);
    }
    
    
    template<typename ProductDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TriangularView& operator=(const ScaledProduct<ProductDerived>& other)
    {
      setZero();
      return assignProduct(other,other.alpha());
    }
    
    template<typename ProductDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TriangularView& operator+=(const ScaledProduct<ProductDerived>& other)
    {
      return assignProduct(other,other.alpha());
    }
    
    template<typename ProductDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TriangularView& operator-=(const ScaledProduct<ProductDerived>& other)
    {
      return assignProduct(other,-other.alpha());
    }
    
  protected:
    
    template<typename ProductDerived, typename Lhs, typename Rhs>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TriangularView& assignProduct(const ProductBase<ProductDerived, Lhs,Rhs>& prod, const Scalar& alpha);

    MatrixTypeNested m_matrix;
};

/***************************************************************************
* Implementation of triangular evaluation/assignment
***************************************************************************/

namespace internal {

template<typename Derived1, typename Derived2, unsigned int Mode, int UnrollCount, bool ClearOpposite>
struct triangular_assignment_selector
{
  enum {
    col = (UnrollCount-1) / Derived1::RowsAtCompileTime,
    row = (UnrollCount-1) % Derived1::RowsAtCompileTime
  };
  
  typedef typename Derived1::Scalar Scalar;

  EIGEN_DEVICE_FUNC
  static inline void run(Derived1 &dst, const Derived2 &src)
  {
    triangular_assignment_selector<Derived1, Derived2, Mode, UnrollCount-1, ClearOpposite>::run(dst, src);

    eigen_assert( Mode == Upper || Mode == Lower
            || Mode == StrictlyUpper || Mode == StrictlyLower
            || Mode == UnitUpper || Mode == UnitLower);
    if((Mode == Upper && row <= col)
    || (Mode == Lower && row >= col)
    || (Mode == StrictlyUpper && row < col)
    || (Mode == StrictlyLower && row > col)
    || (Mode == UnitUpper && row < col)
    || (Mode == UnitLower && row > col))
      dst.copyCoeff(row, col, src);
    else if(ClearOpposite)
    {
      if (Mode&UnitDiag && row==col)
        dst.coeffRef(row, col) = Scalar(1);
      else
        dst.coeffRef(row, col) = Scalar(0);
    }
  }
};

// prevent buggy user code from causing an infinite recursion
template<typename Derived1, typename Derived2, unsigned int Mode, bool ClearOpposite>
struct triangular_assignment_selector<Derived1, Derived2, Mode, 0, ClearOpposite>
{
  EIGEN_DEVICE_FUNC
  static inline void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2, bool ClearOpposite>
struct triangular_assignment_selector<Derived1, Derived2, Upper, Dynamic, ClearOpposite>
{
  typedef typename Derived1::Index Index;
  typedef typename Derived1::Scalar Scalar;
  EIGEN_DEVICE_FUNC
  static inline void run(Derived1 &dst, const Derived2 &src)
  {
    for(Index j = 0; j < dst.cols(); ++j)
    {
      Index maxi = (std::min)(j, dst.rows()-1);
      for(Index i = 0; i <= maxi; ++i)
        dst.copyCoeff(i, j, src);
      if (ClearOpposite)
        for(Index i = maxi+1; i < dst.rows(); ++i)
          dst.coeffRef(i, j) = Scalar(0);
    }
  }
};

template<typename Derived1, typename Derived2, bool ClearOpposite>
struct triangular_assignment_selector<Derived1, Derived2, Lower, Dynamic, ClearOpposite>
{
  typedef typename Derived1::Index Index;
  EIGEN_DEVICE_FUNC
  static inline void run(Derived1 &dst, const Derived2 &src)
  {
    for(Index j = 0; j < dst.cols(); ++j)
    {
      for(Index i = j; i < dst.rows(); ++i)
        dst.copyCoeff(i, j, src);
      Index maxi = (std::min)(j, dst.rows());
      if (ClearOpposite)
        for(Index i = 0; i < maxi; ++i)
          dst.coeffRef(i, j) = static_cast<typename Derived1::Scalar>(0);
    }
  }
};

template<typename Derived1, typename Derived2, bool ClearOpposite>
struct triangular_assignment_selector<Derived1, Derived2, StrictlyUpper, Dynamic, ClearOpposite>
{
  typedef typename Derived1::Index Index;
  typedef typename Derived1::Scalar Scalar;
  EIGEN_DEVICE_FUNC
  static inline void run(Derived1 &dst, const Derived2 &src)
  {
    for(Index j = 0; j < dst.cols(); ++j)
    {
      Index maxi = (std::min)(j, dst.rows());
      for(Index i = 0; i < maxi; ++i)
        dst.copyCoeff(i, j, src);
      if (ClearOpposite)
        for(Index i = maxi; i < dst.rows(); ++i)
          dst.coeffRef(i, j) = Scalar(0);
    }
  }
};

template<typename Derived1, typename Derived2, bool ClearOpposite>
struct triangular_assignment_selector<Derived1, Derived2, StrictlyLower, Dynamic, ClearOpposite>
{
  typedef typename Derived1::Index Index;
  EIGEN_DEVICE_FUNC
  static inline void run(Derived1 &dst, const Derived2 &src)
  {
    for(Index j = 0; j < dst.cols(); ++j)
    {
      for(Index i = j+1; i < dst.rows(); ++i)
        dst.copyCoeff(i, j, src);
      Index maxi = (std::min)(j, dst.rows()-1);
      if (ClearOpposite)
        for(Index i = 0; i <= maxi; ++i)
          dst.coeffRef(i, j) = static_cast<typename Derived1::Scalar>(0);
    }
  }
};

template<typename Derived1, typename Derived2, bool ClearOpposite>
struct triangular_assignment_selector<Derived1, Derived2, UnitUpper, Dynamic, ClearOpposite>
{
  typedef typename Derived1::Index Index;
  EIGEN_DEVICE_FUNC
  static inline void run(Derived1 &dst, const Derived2 &src)
  {
    for(Index j = 0; j < dst.cols(); ++j)
    {
      Index maxi = (std::min)(j, dst.rows());
      for(Index i = 0; i < maxi; ++i)
        dst.copyCoeff(i, j, src);
      if (ClearOpposite)
      {
        for(Index i = maxi+1; i < dst.rows(); ++i)
          dst.coeffRef(i, j) = 0;
      }
    }
    dst.diagonal().setOnes();
  }
};
template<typename Derived1, typename Derived2, bool ClearOpposite>
struct triangular_assignment_selector<Derived1, Derived2, UnitLower, Dynamic, ClearOpposite>
{
  typedef typename Derived1::Index Index;
  EIGEN_DEVICE_FUNC
  static inline void run(Derived1 &dst, const Derived2 &src)
  {
    for(Index j = 0; j < dst.cols(); ++j)
    {
      Index maxi = (std::min)(j, dst.rows());
      for(Index i = maxi+1; i < dst.rows(); ++i)
        dst.copyCoeff(i, j, src);
      if (ClearOpposite)
      {
        for(Index i = 0; i < maxi; ++i)
          dst.coeffRef(i, j) = 0;
      }
    }
    dst.diagonal().setOnes();
  }
};

} // end namespace internal

// FIXME should we keep that possibility
template<typename MatrixType, unsigned int Mode>
template<typename OtherDerived>
inline TriangularView<MatrixType, Mode>&
TriangularView<MatrixType, Mode>::operator=(const MatrixBase<OtherDerived>& other)
{
  if(OtherDerived::Flags & EvalBeforeAssigningBit)
  {
    typename internal::plain_matrix_type<OtherDerived>::type other_evaluated(other.rows(), other.cols());
    other_evaluated.template triangularView<Mode>().lazyAssign(other.derived());
    lazyAssign(other_evaluated);
  }
  else
    lazyAssign(other.derived());
  return *this;
}

// FIXME should we keep that possibility
template<typename MatrixType, unsigned int Mode>
template<typename OtherDerived>
void TriangularView<MatrixType, Mode>::lazyAssign(const MatrixBase<OtherDerived>& other)
{
  enum {
    unroll = MatrixType::SizeAtCompileTime != Dynamic
          && internal::traits<OtherDerived>::CoeffReadCost != Dynamic
          && MatrixType::SizeAtCompileTime*internal::traits<OtherDerived>::CoeffReadCost/2 <= EIGEN_UNROLLING_LIMIT
  };
  eigen_assert(m_matrix.rows() == other.rows() && m_matrix.cols() == other.cols());

  internal::triangular_assignment_selector
    <MatrixType, OtherDerived, int(Mode),
    unroll ? int(MatrixType::SizeAtCompileTime) : Dynamic,
    false // do not change the opposite triangular part
    >::run(m_matrix.const_cast_derived(), other.derived());
}



template<typename MatrixType, unsigned int Mode>
template<typename OtherDerived>
inline TriangularView<MatrixType, Mode>&
TriangularView<MatrixType, Mode>::operator=(const TriangularBase<OtherDerived>& other)
{
  eigen_assert(Mode == int(OtherDerived::Mode));
  if(internal::traits<OtherDerived>::Flags & EvalBeforeAssigningBit)
  {
    typename OtherDerived::DenseMatrixType other_evaluated(other.rows(), other.cols());
    other_evaluated.template triangularView<Mode>().lazyAssign(other.derived().nestedExpression());
    lazyAssign(other_evaluated);
  }
  else
    lazyAssign(other.derived().nestedExpression());
  return *this;
}

template<typename MatrixType, unsigned int Mode>
template<typename OtherDerived>
void TriangularView<MatrixType, Mode>::lazyAssign(const TriangularBase<OtherDerived>& other)
{
  enum {
    unroll = MatrixType::SizeAtCompileTime != Dynamic
                   && internal::traits<OtherDerived>::CoeffReadCost != Dynamic
                   && MatrixType::SizeAtCompileTime * internal::traits<OtherDerived>::CoeffReadCost / 2
                        <= EIGEN_UNROLLING_LIMIT
  };
  eigen_assert(m_matrix.rows() == other.rows() && m_matrix.cols() == other.cols());

  internal::triangular_assignment_selector
    <MatrixType, OtherDerived, int(Mode),
    unroll ? int(MatrixType::SizeAtCompileTime) : Dynamic,
    false // preserve the opposite triangular part
    >::run(m_matrix.const_cast_derived(), other.derived().nestedExpression());
}

/***************************************************************************
* Implementation of TriangularBase methods
***************************************************************************/

/** Assigns a triangular or selfadjoint matrix to a dense matrix.
  * If the matrix is triangular, the opposite part is set to zero. */
template<typename Derived>
template<typename DenseDerived>
void TriangularBase<Derived>::evalTo(MatrixBase<DenseDerived> &other) const
{
  if(internal::traits<Derived>::Flags & EvalBeforeAssigningBit)
  {
    typename internal::plain_matrix_type<Derived>::type other_evaluated(rows(), cols());
    evalToLazy(other_evaluated);
    other.derived().swap(other_evaluated);
  }
  else
    evalToLazy(other.derived());
}

/** Assigns a triangular or selfadjoint matrix to a dense matrix.
  * If the matrix is triangular, the opposite part is set to zero. */
template<typename Derived>
template<typename DenseDerived>
void TriangularBase<Derived>::evalToLazy(MatrixBase<DenseDerived> &other) const
{
  enum {
    unroll = DenseDerived::SizeAtCompileTime != Dynamic
                   && internal::traits<Derived>::CoeffReadCost != Dynamic
                   && DenseDerived::SizeAtCompileTime * internal::traits<Derived>::CoeffReadCost / 2
                        <= EIGEN_UNROLLING_LIMIT
  };
  other.derived().resize(this->rows(), this->cols());

  internal::triangular_assignment_selector
    <DenseDerived, typename internal::traits<Derived>::MatrixTypeNestedCleaned, Derived::Mode,
    unroll ? int(DenseDerived::SizeAtCompileTime) : Dynamic,
    true // clear the opposite triangular part
    >::run(other.derived(), derived().nestedExpression());
}

/***************************************************************************
* Implementation of TriangularView methods
***************************************************************************/

/***************************************************************************
* Implementation of MatrixBase methods
***************************************************************************/

#ifdef EIGEN2_SUPPORT

// implementation of part<>(), including the SelfAdjoint case.

namespace internal {
template<typename MatrixType, unsigned int Mode>
struct eigen2_part_return_type
{
  typedef TriangularView<MatrixType, Mode> type;
};

template<typename MatrixType>
struct eigen2_part_return_type<MatrixType, SelfAdjoint>
{
  typedef SelfAdjointView<MatrixType, Upper> type;
};
}

/** \deprecated use MatrixBase::triangularView() */
template<typename Derived>
template<unsigned int Mode>
const typename internal::eigen2_part_return_type<Derived, Mode>::type MatrixBase<Derived>::part() const
{
  return derived();
}

/** \deprecated use MatrixBase::triangularView() */
template<typename Derived>
template<unsigned int Mode>
typename internal::eigen2_part_return_type<Derived, Mode>::type MatrixBase<Derived>::part()
{
  return derived();
}
#endif

/**
  * \returns an expression of a triangular view extracted from the current matrix
  *
  * The parameter \a Mode can have the following values: \c #Upper, \c #StrictlyUpper, \c #UnitUpper,
  * \c #Lower, \c #StrictlyLower, \c #UnitLower.
  *
  * Example: \include MatrixBase_extract.cpp
  * Output: \verbinclude MatrixBase_extract.out
  *
  * \sa class TriangularView
  */
template<typename Derived>
template<unsigned int Mode>
typename MatrixBase<Derived>::template TriangularViewReturnType<Mode>::Type
MatrixBase<Derived>::triangularView()
{
  return derived();
}

/** This is the const version of MatrixBase::triangularView() */
template<typename Derived>
template<unsigned int Mode>
typename MatrixBase<Derived>::template ConstTriangularViewReturnType<Mode>::Type
MatrixBase<Derived>::triangularView() const
{
  return derived();
}

/** \returns true if *this is approximately equal to an upper triangular matrix,
  *          within the precision given by \a prec.
  *
  * \sa isLowerTriangular()
  */
template<typename Derived>
bool MatrixBase<Derived>::isUpperTriangular(const RealScalar& prec) const
{
  using std::abs;
  RealScalar maxAbsOnUpperPart = static_cast<RealScalar>(-1);
  for(Index j = 0; j < cols(); ++j)
  {
    Index maxi = (std::min)(j, rows()-1);
    for(Index i = 0; i <= maxi; ++i)
    {
      RealScalar absValue = abs(coeff(i,j));
      if(absValue > maxAbsOnUpperPart) maxAbsOnUpperPart = absValue;
    }
  }
  RealScalar threshold = maxAbsOnUpperPart * prec;
  for(Index j = 0; j < cols(); ++j)
    for(Index i = j+1; i < rows(); ++i)
      if(abs(coeff(i, j)) > threshold) return false;
  return true;
}

/** \returns true if *this is approximately equal to a lower triangular matrix,
  *          within the precision given by \a prec.
  *
  * \sa isUpperTriangular()
  */
template<typename Derived>
bool MatrixBase<Derived>::isLowerTriangular(const RealScalar& prec) const
{
  using std::abs;
  RealScalar maxAbsOnLowerPart = static_cast<RealScalar>(-1);
  for(Index j = 0; j < cols(); ++j)
    for(Index i = j; i < rows(); ++i)
    {
      RealScalar absValue = abs(coeff(i,j));
      if(absValue > maxAbsOnLowerPart) maxAbsOnLowerPart = absValue;
    }
  RealScalar threshold = maxAbsOnLowerPart * prec;
  for(Index j = 1; j < cols(); ++j)
  {
    Index maxi = (std::min)(j, rows()-1);
    for(Index i = 0; i < maxi; ++i)
      if(abs(coeff(i, j)) > threshold) return false;
  }
  return true;
}

} // end namespace Eigen

#endif // EIGEN_TRIANGULARMATRIX_H
