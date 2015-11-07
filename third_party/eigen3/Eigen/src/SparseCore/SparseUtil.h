// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSEUTIL_H
#define EIGEN_SPARSEUTIL_H

namespace Eigen { 

#ifdef NDEBUG
#define EIGEN_DBG_SPARSE(X)
#else
#define EIGEN_DBG_SPARSE(X) X
#endif

#define EIGEN_SPARSE_INHERIT_ASSIGNMENT_OPERATOR(Derived, Op) \
template<typename OtherDerived> \
EIGEN_STRONG_INLINE Derived& operator Op(const Eigen::SparseMatrixBase<OtherDerived>& other) \
{ \
  return Base::operator Op(other.derived()); \
} \
EIGEN_STRONG_INLINE Derived& operator Op(const Derived& other) \
{ \
  return Base::operator Op(other); \
}

#define EIGEN_SPARSE_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Derived, Op) \
template<typename Other> \
EIGEN_STRONG_INLINE Derived& operator Op(const Other& scalar) \
{ \
  return Base::operator Op(scalar); \
}

#define EIGEN_SPARSE_INHERIT_ASSIGNMENT_OPERATORS(Derived) \
EIGEN_SPARSE_INHERIT_ASSIGNMENT_OPERATOR(Derived, =) \
EIGEN_SPARSE_INHERIT_ASSIGNMENT_OPERATOR(Derived, +=) \
EIGEN_SPARSE_INHERIT_ASSIGNMENT_OPERATOR(Derived, -=) \
EIGEN_SPARSE_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Derived, *=) \
EIGEN_SPARSE_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Derived, /=)

#define _EIGEN_SPARSE_PUBLIC_INTERFACE(Derived, BaseClass) \
  typedef BaseClass Base; \
  typedef typename Eigen::internal::traits<Derived >::Scalar Scalar; \
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar; \
  typedef typename Eigen::internal::nested<Derived >::type Nested; \
  typedef typename Eigen::internal::traits<Derived >::StorageKind StorageKind; \
  typedef typename Eigen::internal::traits<Derived >::Index Index; \
  enum { RowsAtCompileTime = Eigen::internal::traits<Derived >::RowsAtCompileTime, \
        ColsAtCompileTime = Eigen::internal::traits<Derived >::ColsAtCompileTime, \
        Flags = Eigen::internal::traits<Derived >::Flags, \
        CoeffReadCost = Eigen::internal::traits<Derived >::CoeffReadCost, \
        SizeAtCompileTime = Base::SizeAtCompileTime, \
        IsVectorAtCompileTime = Base::IsVectorAtCompileTime }; \
  using Base::derived; \
  using Base::const_cast_derived;

#define EIGEN_SPARSE_PUBLIC_INTERFACE(Derived) \
  _EIGEN_SPARSE_PUBLIC_INTERFACE(Derived, Eigen::SparseMatrixBase<Derived >)

const int CoherentAccessPattern     = 0x1;
const int InnerRandomAccessPattern  = 0x2 | CoherentAccessPattern;
const int OuterRandomAccessPattern  = 0x4 | CoherentAccessPattern;
const int RandomAccessPattern       = 0x8 | OuterRandomAccessPattern | InnerRandomAccessPattern;

template<typename Derived> class SparseMatrixBase;
template<typename _Scalar, int _Flags = 0, typename _Index = int>  class SparseMatrix;
template<typename _Scalar, int _Flags = 0, typename _Index = int>  class DynamicSparseMatrix;
template<typename _Scalar, int _Flags = 0, typename _Index = int>  class SparseVector;
template<typename _Scalar, int _Flags = 0, typename _Index = int>  class MappedSparseMatrix;

template<typename MatrixType, int Mode>           class SparseTriangularView;
template<typename MatrixType, unsigned int UpLo>  class SparseSelfAdjointView;
template<typename Lhs, typename Rhs>              class SparseDiagonalProduct;
template<typename MatrixType> class SparseView;

template<typename Lhs, typename Rhs>        class SparseSparseProduct;
template<typename Lhs, typename Rhs>        class SparseTimeDenseProduct;
template<typename Lhs, typename Rhs>        class DenseTimeSparseProduct;
template<typename Lhs, typename Rhs, bool Transpose> class SparseDenseOuterProduct;

template<typename Lhs, typename Rhs> struct SparseSparseProductReturnType;
template<typename Lhs, typename Rhs, int InnerSize = internal::traits<Lhs>::ColsAtCompileTime> struct DenseSparseProductReturnType;
template<typename Lhs, typename Rhs, int InnerSize = internal::traits<Lhs>::ColsAtCompileTime> struct SparseDenseProductReturnType;
template<typename MatrixType,int UpLo> class SparseSymmetricPermutationProduct;

namespace internal {

template<typename T,int Rows,int Cols> struct sparse_eval;

template<typename T> struct eval<T,Sparse>
  : public sparse_eval<T, traits<T>::RowsAtCompileTime,traits<T>::ColsAtCompileTime>
{};

template<typename T,int Cols> struct sparse_eval<T,1,Cols> {
    typedef typename traits<T>::Scalar _Scalar;
    typedef typename traits<T>::Index _Index;
  public:
    typedef SparseVector<_Scalar, RowMajor, _Index> type;
};

template<typename T,int Rows> struct sparse_eval<T,Rows,1> {
    typedef typename traits<T>::Scalar _Scalar;
    typedef typename traits<T>::Index _Index;
  public:
    typedef SparseVector<_Scalar, ColMajor, _Index> type;
};

template<typename T,int Rows,int Cols> struct sparse_eval {
    typedef typename traits<T>::Scalar _Scalar;
    typedef typename traits<T>::Index _Index;
    enum { _Options = ((traits<T>::Flags&RowMajorBit)==RowMajorBit) ? RowMajor : ColMajor };
  public:
    typedef SparseMatrix<_Scalar, _Options, _Index> type;
};

template<typename T> struct sparse_eval<T,1,1> {
    typedef typename traits<T>::Scalar _Scalar;
  public:
    typedef Matrix<_Scalar, 1, 1> type;
};

template<typename T> struct plain_matrix_type<T,Sparse>
{
  typedef typename traits<T>::Scalar _Scalar;
  typedef typename traits<T>::Index _Index;
  enum { _Options = ((traits<T>::Flags&RowMajorBit)==RowMajorBit) ? RowMajor : ColMajor };
  public:
    typedef SparseMatrix<_Scalar, _Options, _Index> type;
};

} // end namespace internal

/** \ingroup SparseCore_Module
  *
  * \class Triplet
  *
  * \brief A small structure to hold a non zero as a triplet (i,j,value).
  *
  * \sa SparseMatrix::setFromTriplets()
  */
template<typename Scalar, typename Index=typename SparseMatrix<Scalar>::Index >
class Triplet
{
public:
  Triplet() : m_row(0), m_col(0), m_value(0) {}

  Triplet(const Index& i, const Index& j, const Scalar& v = Scalar(0))
    : m_row(i), m_col(j), m_value(v)
  {}

  /** \returns the row index of the element */
  const Index& row() const { return m_row; }

  /** \returns the column index of the element */
  const Index& col() const { return m_col; }

  /** \returns the value of the element */
  const Scalar& value() const { return m_value; }
protected:
  Index m_row, m_col;
  Scalar m_value;
};

} // end namespace Eigen

#endif // EIGEN_SPARSEUTIL_H
