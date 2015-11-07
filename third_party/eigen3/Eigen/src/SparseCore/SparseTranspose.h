// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSETRANSPOSE_H
#define EIGEN_SPARSETRANSPOSE_H

namespace Eigen { 

template<typename MatrixType> class TransposeImpl<MatrixType,Sparse>
  : public SparseMatrixBase<Transpose<MatrixType> >
{
    typedef typename internal::remove_all<typename MatrixType::Nested>::type _MatrixTypeNested;
  public:

    EIGEN_SPARSE_PUBLIC_INTERFACE(Transpose<MatrixType> )

    class InnerIterator;
    class ReverseInnerIterator;

    inline Index nonZeros() const { return derived().nestedExpression().nonZeros(); }
};

// NOTE: VC10 trigger an ICE if don't put typename TransposeImpl<MatrixType,Sparse>:: in front of Index,
// a typedef typename TransposeImpl<MatrixType,Sparse>::Index Index;
// does not fix the issue.
// An alternative is to define the nested class in the parent class itself.
template<typename MatrixType> class TransposeImpl<MatrixType,Sparse>::InnerIterator
  : public _MatrixTypeNested::InnerIterator
{
    typedef typename _MatrixTypeNested::InnerIterator Base;
    typedef typename TransposeImpl::Index Index;
  public:

    EIGEN_STRONG_INLINE InnerIterator(const TransposeImpl& trans, typename TransposeImpl<MatrixType,Sparse>::Index outer)
      : Base(trans.derived().nestedExpression(), outer)
    {}
    Index row() const { return Base::col(); }
    Index col() const { return Base::row(); }
};

template<typename MatrixType> class TransposeImpl<MatrixType,Sparse>::ReverseInnerIterator
  : public _MatrixTypeNested::ReverseInnerIterator
{
    typedef typename _MatrixTypeNested::ReverseInnerIterator Base;
    typedef typename TransposeImpl::Index Index;
  public:

    EIGEN_STRONG_INLINE ReverseInnerIterator(const TransposeImpl& xpr, typename TransposeImpl<MatrixType,Sparse>::Index outer)
      : Base(xpr.derived().nestedExpression(), outer)
    {}
    Index row() const { return Base::col(); }
    Index col() const { return Base::row(); }
};

} // end namespace Eigen

#endif // EIGEN_SPARSETRANSPOSE_H
