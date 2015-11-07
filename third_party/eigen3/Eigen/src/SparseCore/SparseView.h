// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Daniel Lowengrub <lowdanie@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSEVIEW_H
#define EIGEN_SPARSEVIEW_H

namespace Eigen { 

namespace internal {

template<typename MatrixType>
struct traits<SparseView<MatrixType> > : traits<MatrixType>
{
  typedef typename MatrixType::Index Index;
  typedef Sparse StorageKind;
  enum {
    Flags = int(traits<MatrixType>::Flags) & (RowMajorBit)
  };
};

} // end namespace internal

template<typename MatrixType>
class SparseView : public SparseMatrixBase<SparseView<MatrixType> >
{
  typedef typename MatrixType::Nested MatrixTypeNested;
  typedef typename internal::remove_all<MatrixTypeNested>::type _MatrixTypeNested;
public:
  EIGEN_SPARSE_PUBLIC_INTERFACE(SparseView)

  SparseView(const MatrixType& mat, const Scalar& m_reference = Scalar(0),
             typename NumTraits<Scalar>::Real m_epsilon = NumTraits<Scalar>::dummy_precision()) : 
    m_matrix(mat), m_reference(m_reference), m_epsilon(m_epsilon) {}

  class InnerIterator;

  inline Index rows() const { return m_matrix.rows(); }
  inline Index cols() const { return m_matrix.cols(); }

  inline Index innerSize() const { return m_matrix.innerSize(); }
  inline Index outerSize() const { return m_matrix.outerSize(); }

protected:
  MatrixTypeNested m_matrix;
  Scalar m_reference;
  typename NumTraits<Scalar>::Real m_epsilon;
};

template<typename MatrixType>
class SparseView<MatrixType>::InnerIterator : public _MatrixTypeNested::InnerIterator
{
  typedef typename SparseView::Index Index;
public:
  typedef typename _MatrixTypeNested::InnerIterator IterBase;
  InnerIterator(const SparseView& view, Index outer) :
  IterBase(view.m_matrix, outer), m_view(view)
  {
    incrementToNonZero();
  }

  EIGEN_STRONG_INLINE InnerIterator& operator++()
  {
    IterBase::operator++();
    incrementToNonZero();
    return *this;
  }

  using IterBase::value;

protected:
  const SparseView& m_view;

private:
  void incrementToNonZero()
  {
    while((bool(*this)) && internal::isMuchSmallerThan(value(), m_view.m_reference, m_view.m_epsilon))
    {
      IterBase::operator++();
    }
  }
};

template<typename Derived>
const SparseView<Derived> MatrixBase<Derived>::sparseView(const Scalar& m_reference,
                                                          const typename NumTraits<Scalar>::Real& m_epsilon) const
{
  return SparseView<Derived>(derived(), m_reference, m_epsilon);
}

} // end namespace Eigen

#endif
