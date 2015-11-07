// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_TRIANGULARVIEW_H
#define EIGEN_SPARSE_TRIANGULARVIEW_H

namespace Eigen { 

namespace internal {
  
template<typename MatrixType, int Mode>
struct traits<SparseTriangularView<MatrixType,Mode> >
: public traits<MatrixType>
{};

} // namespace internal

template<typename MatrixType, int Mode> class SparseTriangularView
  : public SparseMatrixBase<SparseTriangularView<MatrixType,Mode> >
{
    enum { SkipFirst = ((Mode&Lower) && !(MatrixType::Flags&RowMajorBit))
                    || ((Mode&Upper) &&  (MatrixType::Flags&RowMajorBit)),
           SkipLast = !SkipFirst,
           SkipDiag = (Mode&ZeroDiag) ? 1 : 0,
           HasUnitDiag = (Mode&UnitDiag) ? 1 : 0
    };

  public:
    
    EIGEN_SPARSE_PUBLIC_INTERFACE(SparseTriangularView)

    class InnerIterator;
    class ReverseInnerIterator;

    inline Index rows() const { return m_matrix.rows(); }
    inline Index cols() const { return m_matrix.cols(); }

    typedef typename MatrixType::Nested MatrixTypeNested;
    typedef typename internal::remove_reference<MatrixTypeNested>::type MatrixTypeNestedNonRef;
    typedef typename internal::remove_all<MatrixTypeNested>::type MatrixTypeNestedCleaned;

    inline SparseTriangularView(const MatrixType& matrix) : m_matrix(matrix) {}

    /** \internal */
    inline const MatrixTypeNestedCleaned& nestedExpression() const { return m_matrix; }

    template<typename OtherDerived>
    typename internal::plain_matrix_type_column_major<OtherDerived>::type
    solve(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived> void solveInPlace(MatrixBase<OtherDerived>& other) const;
    template<typename OtherDerived> void solveInPlace(SparseMatrixBase<OtherDerived>& other) const;

  protected:
    MatrixTypeNested m_matrix;
};

template<typename MatrixType, int Mode>
class SparseTriangularView<MatrixType,Mode>::InnerIterator : public MatrixTypeNestedCleaned::InnerIterator
{
    typedef typename MatrixTypeNestedCleaned::InnerIterator Base;
    typedef typename SparseTriangularView::Index Index;
  public:

    EIGEN_STRONG_INLINE InnerIterator(const SparseTriangularView& view, Index outer)
      : Base(view.nestedExpression(), outer), m_returnOne(false)
    {
      if(SkipFirst)
      {
        while((*this) && ((HasUnitDiag||SkipDiag)  ? this->index()<=outer : this->index()<outer))
          Base::operator++();
        if(HasUnitDiag)
          m_returnOne = true;
      }
      else if(HasUnitDiag && ((!Base::operator bool()) || Base::index()>=Base::outer()))
      {
        if((!SkipFirst) && Base::operator bool())
          Base::operator++();
        m_returnOne = true;
      }
    }

    EIGEN_STRONG_INLINE InnerIterator& operator++()
    {
      if(HasUnitDiag && m_returnOne)
        m_returnOne = false;
      else
      {
        Base::operator++();
        if(HasUnitDiag && (!SkipFirst) && ((!Base::operator bool()) || Base::index()>=Base::outer()))
        {
          if((!SkipFirst) && Base::operator bool())
            Base::operator++();
          m_returnOne = true;
        }
      }
      return *this;
    }

    inline Index row() const { return (MatrixType::Flags&RowMajorBit ? Base::outer() : this->index()); }
    inline Index col() const { return (MatrixType::Flags&RowMajorBit ? this->index() : Base::outer()); }
    inline Index index() const
    {
      if(HasUnitDiag && m_returnOne)  return Base::outer();
      else                            return Base::index();
    }
    inline Scalar value() const
    {
      if(HasUnitDiag && m_returnOne)  return Scalar(1);
      else                            return Base::value();
    }

    EIGEN_STRONG_INLINE operator bool() const
    {
      if(HasUnitDiag && m_returnOne)
        return true;
      if(SkipFirst) return  Base::operator bool();
      else
      {
        if (SkipDiag) return (Base::operator bool() && this->index() < this->outer());
        else return (Base::operator bool() && this->index() <= this->outer());
      }
    }
  protected:
    bool m_returnOne;
};

template<typename MatrixType, int Mode>
class SparseTriangularView<MatrixType,Mode>::ReverseInnerIterator : public MatrixTypeNestedCleaned::ReverseInnerIterator
{
    typedef typename MatrixTypeNestedCleaned::ReverseInnerIterator Base;
    typedef typename SparseTriangularView::Index Index;
  public:

    EIGEN_STRONG_INLINE ReverseInnerIterator(const SparseTriangularView& view, Index outer)
      : Base(view.nestedExpression(), outer)
    {
      eigen_assert((!HasUnitDiag) && "ReverseInnerIterator does not support yet triangular views with a unit diagonal");
      if(SkipLast) {
        while((*this) && (SkipDiag ? this->index()>=outer : this->index()>outer))
          --(*this);
      }
    }

    EIGEN_STRONG_INLINE ReverseInnerIterator& operator--()
    { Base::operator--(); return *this; }

    inline Index row() const { return Base::row(); }
    inline Index col() const { return Base::col(); }

    EIGEN_STRONG_INLINE operator bool() const
    {
      if (SkipLast) return Base::operator bool() ;
      else
      {
        if(SkipDiag) return (Base::operator bool() && this->index() > this->outer());
        else return (Base::operator bool() && this->index() >= this->outer());
      }
    }
};

template<typename Derived>
template<int Mode>
inline const SparseTriangularView<Derived, Mode>
SparseMatrixBase<Derived>::triangularView() const
{
  return derived();
}

} // end namespace Eigen

#endif // EIGEN_SPARSE_TRIANGULARVIEW_H
