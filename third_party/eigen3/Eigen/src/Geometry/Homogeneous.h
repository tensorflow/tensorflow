// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_HOMOGENEOUS_H
#define EIGEN_HOMOGENEOUS_H

namespace Eigen { 

/** \geometry_module \ingroup Geometry_Module
  *
  * \class Homogeneous
  *
  * \brief Expression of one (or a set of) homogeneous vector(s)
  *
  * \param MatrixType the type of the object in which we are making homogeneous
  *
  * This class represents an expression of one (or a set of) homogeneous vector(s).
  * It is the return type of MatrixBase::homogeneous() and most of the time
  * this is the only way it is used.
  *
  * \sa MatrixBase::homogeneous()
  */

namespace internal {

template<typename MatrixType,int Direction>
struct traits<Homogeneous<MatrixType,Direction> >
 : traits<MatrixType>
{
  typedef typename traits<MatrixType>::StorageKind StorageKind;
  typedef typename nested<MatrixType>::type MatrixTypeNested;
  typedef typename remove_reference<MatrixTypeNested>::type _MatrixTypeNested;
  enum {
    RowsPlusOne = (MatrixType::RowsAtCompileTime != Dynamic) ?
                  int(MatrixType::RowsAtCompileTime) + 1 : Dynamic,
    ColsPlusOne = (MatrixType::ColsAtCompileTime != Dynamic) ?
                  int(MatrixType::ColsAtCompileTime) + 1 : Dynamic,
    RowsAtCompileTime = Direction==Vertical  ?  RowsPlusOne : MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = Direction==Horizontal ? ColsPlusOne : MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = RowsAtCompileTime,
    MaxColsAtCompileTime = ColsAtCompileTime,
    TmpFlags = _MatrixTypeNested::Flags & HereditaryBits,
    Flags = ColsAtCompileTime==1 ? (TmpFlags & ~RowMajorBit)
          : RowsAtCompileTime==1 ? (TmpFlags | RowMajorBit)
          : TmpFlags,
    CoeffReadCost = _MatrixTypeNested::CoeffReadCost
  };
};

template<typename MatrixType,typename Lhs> struct homogeneous_left_product_impl;
template<typename MatrixType,typename Rhs> struct homogeneous_right_product_impl;

} // end namespace internal

template<typename MatrixType,int _Direction> class Homogeneous
  : internal::no_assignment_operator, public MatrixBase<Homogeneous<MatrixType,_Direction> >
{
  public:

    enum { Direction = _Direction };

    typedef MatrixBase<Homogeneous> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(Homogeneous)

    inline Homogeneous(const MatrixType& matrix)
      : m_matrix(matrix)
    {}

    inline Index rows() const { return m_matrix.rows() + (int(Direction)==Vertical   ? 1 : 0); }
    inline Index cols() const { return m_matrix.cols() + (int(Direction)==Horizontal ? 1 : 0); }

    inline Scalar coeff(Index row, Index col) const
    {
      if(  (int(Direction)==Vertical   && row==m_matrix.rows())
        || (int(Direction)==Horizontal && col==m_matrix.cols()))
        return 1;
      return m_matrix.coeff(row, col);
    }

    template<typename Rhs>
    inline const internal::homogeneous_right_product_impl<Homogeneous,Rhs>
    operator* (const MatrixBase<Rhs>& rhs) const
    {
      eigen_assert(int(Direction)==Horizontal);
      return internal::homogeneous_right_product_impl<Homogeneous,Rhs>(m_matrix,rhs.derived());
    }

    template<typename Lhs> friend
    inline const internal::homogeneous_left_product_impl<Homogeneous,Lhs>
    operator* (const MatrixBase<Lhs>& lhs, const Homogeneous& rhs)
    {
      eigen_assert(int(Direction)==Vertical);
      return internal::homogeneous_left_product_impl<Homogeneous,Lhs>(lhs.derived(),rhs.m_matrix);
    }

    template<typename Scalar, int Dim, int Mode, int Options> friend
    inline const internal::homogeneous_left_product_impl<Homogeneous,Transform<Scalar,Dim,Mode,Options> >
    operator* (const Transform<Scalar,Dim,Mode,Options>& lhs, const Homogeneous& rhs)
    {
      eigen_assert(int(Direction)==Vertical);
      return internal::homogeneous_left_product_impl<Homogeneous,Transform<Scalar,Dim,Mode,Options> >(lhs,rhs.m_matrix);
    }

  protected:
    typename MatrixType::Nested m_matrix;
};

/** \geometry_module
  *
  * \return an expression of the equivalent homogeneous vector
  *
  * \only_for_vectors
  *
  * Example: \include MatrixBase_homogeneous.cpp
  * Output: \verbinclude MatrixBase_homogeneous.out
  *
  * \sa class Homogeneous
  */
template<typename Derived>
inline typename MatrixBase<Derived>::HomogeneousReturnType
MatrixBase<Derived>::homogeneous() const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return derived();
}

/** \geometry_module
  *
  * \returns a matrix expression of homogeneous column (or row) vectors
  *
  * Example: \include VectorwiseOp_homogeneous.cpp
  * Output: \verbinclude VectorwiseOp_homogeneous.out
  *
  * \sa MatrixBase::homogeneous() */
template<typename ExpressionType, int Direction>
inline Homogeneous<ExpressionType,Direction>
VectorwiseOp<ExpressionType,Direction>::homogeneous() const
{
  return _expression();
}

/** \geometry_module
  *
  * \returns an expression of the homogeneous normalized vector of \c *this
  *
  * Example: \include MatrixBase_hnormalized.cpp
  * Output: \verbinclude MatrixBase_hnormalized.out
  *
  * \sa VectorwiseOp::hnormalized() */
template<typename Derived>
inline const typename MatrixBase<Derived>::HNormalizedReturnType
MatrixBase<Derived>::hnormalized() const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return ConstStartMinusOne(derived(),0,0,
    ColsAtCompileTime==1?size()-1:1,
    ColsAtCompileTime==1?1:size()-1) / coeff(size()-1);
}

/** \geometry_module
  *
  * \returns an expression of the homogeneous normalized vector of \c *this
  *
  * Example: \include DirectionWise_hnormalized.cpp
  * Output: \verbinclude DirectionWise_hnormalized.out
  *
  * \sa MatrixBase::hnormalized() */
template<typename ExpressionType, int Direction>
inline const typename VectorwiseOp<ExpressionType,Direction>::HNormalizedReturnType
VectorwiseOp<ExpressionType,Direction>::hnormalized() const
{
  return HNormalized_Block(_expression(),0,0,
      Direction==Vertical   ? _expression().rows()-1 : _expression().rows(),
      Direction==Horizontal ? _expression().cols()-1 : _expression().cols()).cwiseQuotient(
      Replicate<HNormalized_Factors,
                Direction==Vertical   ? HNormalized_SizeMinusOne : 1,
                Direction==Horizontal ? HNormalized_SizeMinusOne : 1>
        (HNormalized_Factors(_expression(),
          Direction==Vertical    ? _expression().rows()-1:0,
          Direction==Horizontal  ? _expression().cols()-1:0,
          Direction==Vertical    ? 1 : _expression().rows(),
          Direction==Horizontal  ? 1 : _expression().cols()),
         Direction==Vertical   ? _expression().rows()-1 : 1,
         Direction==Horizontal ? _expression().cols()-1 : 1));
}

namespace internal {

template<typename MatrixOrTransformType>
struct take_matrix_for_product
{
  typedef MatrixOrTransformType type;
  static const type& run(const type &x) { return x; }
};

template<typename Scalar, int Dim, int Mode,int Options>
struct take_matrix_for_product<Transform<Scalar, Dim, Mode, Options> >
{
  typedef Transform<Scalar, Dim, Mode, Options> TransformType;
  typedef typename internal::add_const<typename TransformType::ConstAffinePart>::type type;
  static type run (const TransformType& x) { return x.affine(); }
};

template<typename Scalar, int Dim, int Options>
struct take_matrix_for_product<Transform<Scalar, Dim, Projective, Options> >
{
  typedef Transform<Scalar, Dim, Projective, Options> TransformType;
  typedef typename TransformType::MatrixType type;
  static const type& run (const TransformType& x) { return x.matrix(); }
};

template<typename MatrixType,typename Lhs>
struct traits<homogeneous_left_product_impl<Homogeneous<MatrixType,Vertical>,Lhs> >
{
  typedef typename take_matrix_for_product<Lhs>::type LhsMatrixType;
  typedef typename remove_all<MatrixType>::type MatrixTypeCleaned;
  typedef typename remove_all<LhsMatrixType>::type LhsMatrixTypeCleaned;
  typedef typename make_proper_matrix_type<
                 typename traits<MatrixTypeCleaned>::Scalar,
                 LhsMatrixTypeCleaned::RowsAtCompileTime,
                 MatrixTypeCleaned::ColsAtCompileTime,
                 MatrixTypeCleaned::PlainObject::Options,
                 LhsMatrixTypeCleaned::MaxRowsAtCompileTime,
                 MatrixTypeCleaned::MaxColsAtCompileTime>::type ReturnType;
};

template<typename MatrixType,typename Lhs>
struct homogeneous_left_product_impl<Homogeneous<MatrixType,Vertical>,Lhs>
  : public ReturnByValue<homogeneous_left_product_impl<Homogeneous<MatrixType,Vertical>,Lhs> >
{
  typedef typename traits<homogeneous_left_product_impl>::LhsMatrixType LhsMatrixType;
  typedef typename remove_all<LhsMatrixType>::type LhsMatrixTypeCleaned;
  typedef typename remove_all<typename LhsMatrixTypeCleaned::Nested>::type LhsMatrixTypeNested;
  typedef typename MatrixType::Index Index;
  homogeneous_left_product_impl(const Lhs& lhs, const MatrixType& rhs)
    : m_lhs(take_matrix_for_product<Lhs>::run(lhs)),
      m_rhs(rhs)
  {}

  inline Index rows() const { return m_lhs.rows(); }
  inline Index cols() const { return m_rhs.cols(); }

  template<typename Dest> void evalTo(Dest& dst) const
  {
    // FIXME investigate how to allow lazy evaluation of this product when possible
    dst = Block<const LhsMatrixTypeNested,
              LhsMatrixTypeNested::RowsAtCompileTime,
              LhsMatrixTypeNested::ColsAtCompileTime==Dynamic?Dynamic:LhsMatrixTypeNested::ColsAtCompileTime-1>
            (m_lhs,0,0,m_lhs.rows(),m_lhs.cols()-1) * m_rhs;
    dst += m_lhs.col(m_lhs.cols()-1).rowwise()
            .template replicate<MatrixType::ColsAtCompileTime>(m_rhs.cols());
  }

  typename LhsMatrixTypeCleaned::Nested m_lhs;
  typename MatrixType::Nested m_rhs;
};

template<typename MatrixType,typename Rhs>
struct traits<homogeneous_right_product_impl<Homogeneous<MatrixType,Horizontal>,Rhs> >
{
  typedef typename make_proper_matrix_type<typename traits<MatrixType>::Scalar,
                 MatrixType::RowsAtCompileTime,
                 Rhs::ColsAtCompileTime,
                 MatrixType::PlainObject::Options,
                 MatrixType::MaxRowsAtCompileTime,
                 Rhs::MaxColsAtCompileTime>::type ReturnType;
};

template<typename MatrixType,typename Rhs>
struct homogeneous_right_product_impl<Homogeneous<MatrixType,Horizontal>,Rhs>
  : public ReturnByValue<homogeneous_right_product_impl<Homogeneous<MatrixType,Horizontal>,Rhs> >
{
  typedef typename remove_all<typename Rhs::Nested>::type RhsNested;
  typedef typename MatrixType::Index Index;
  homogeneous_right_product_impl(const MatrixType& lhs, const Rhs& rhs)
    : m_lhs(lhs), m_rhs(rhs)
  {}

  inline Index rows() const { return m_lhs.rows(); }
  inline Index cols() const { return m_rhs.cols(); }

  template<typename Dest> void evalTo(Dest& dst) const
  {
    // FIXME investigate how to allow lazy evaluation of this product when possible
    dst = m_lhs * Block<const RhsNested,
                        RhsNested::RowsAtCompileTime==Dynamic?Dynamic:RhsNested::RowsAtCompileTime-1,
                        RhsNested::ColsAtCompileTime>
            (m_rhs,0,0,m_rhs.rows()-1,m_rhs.cols());
    dst += m_rhs.row(m_rhs.rows()-1).colwise()
            .template replicate<MatrixType::RowsAtCompileTime>(m_lhs.rows());
  }

  typename MatrixType::Nested m_lhs;
  typename Rhs::Nested m_rhs;
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_HOMOGENEOUS_H
