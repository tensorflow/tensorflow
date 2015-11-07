// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MAPPED_SPARSEMATRIX_H
#define EIGEN_MAPPED_SPARSEMATRIX_H

namespace Eigen { 

/** \class MappedSparseMatrix
  *
  * \brief Sparse matrix
  *
  * \param _Scalar the scalar type, i.e. the type of the coefficients
  *
  * See http://www.netlib.org/linalg/html_templates/node91.html for details on the storage scheme.
  *
  */
namespace internal {
template<typename _Scalar, int _Flags, typename _Index>
struct traits<MappedSparseMatrix<_Scalar, _Flags, _Index> > : traits<SparseMatrix<_Scalar, _Flags, _Index> >
{};
}

template<typename _Scalar, int _Flags, typename _Index>
class MappedSparseMatrix
  : public SparseMatrixBase<MappedSparseMatrix<_Scalar, _Flags, _Index> >
{
  public:
    EIGEN_SPARSE_PUBLIC_INTERFACE(MappedSparseMatrix)
    enum { IsRowMajor = Base::IsRowMajor };

  protected:

    Index   m_outerSize;
    Index   m_innerSize;
    Index   m_nnz;
    Index*  m_outerIndex;
    Index*  m_innerIndices;
    Scalar* m_values;

  public:

    inline Index rows() const { return IsRowMajor ? m_outerSize : m_innerSize; }
    inline Index cols() const { return IsRowMajor ? m_innerSize : m_outerSize; }
    inline Index innerSize() const { return m_innerSize; }
    inline Index outerSize() const { return m_outerSize; }
    
    bool isCompressed() const { return true; }

    //----------------------------------------
    // direct access interface
    inline const Scalar* valuePtr() const { return m_values; }
    inline Scalar* valuePtr() { return m_values; }

    inline const Index* innerIndexPtr() const { return m_innerIndices; }
    inline Index* innerIndexPtr() { return m_innerIndices; }

    inline const Index* outerIndexPtr() const { return m_outerIndex; }
    inline Index* outerIndexPtr() { return m_outerIndex; }
    //----------------------------------------

    inline Scalar coeff(Index row, Index col) const
    {
      const Index outer = IsRowMajor ? row : col;
      const Index inner = IsRowMajor ? col : row;

      Index start = m_outerIndex[outer];
      Index end = m_outerIndex[outer+1];
      if (start==end)
        return Scalar(0);
      else if (end>0 && inner==m_innerIndices[end-1])
        return m_values[end-1];
      // ^^  optimization: let's first check if it is the last coefficient
      // (very common in high level algorithms)

      const Index* r = std::lower_bound(&m_innerIndices[start],&m_innerIndices[end-1],inner);
      const Index id = r-&m_innerIndices[0];
      return ((*r==inner) && (id<end)) ? m_values[id] : Scalar(0);
    }

    inline Scalar& coeffRef(Index row, Index col)
    {
      const Index outer = IsRowMajor ? row : col;
      const Index inner = IsRowMajor ? col : row;

      Index start = m_outerIndex[outer];
      Index end = m_outerIndex[outer+1];
      eigen_assert(end>=start && "you probably called coeffRef on a non finalized matrix");
      eigen_assert(end>start && "coeffRef cannot be called on a zero coefficient");
      Index* r = std::lower_bound(&m_innerIndices[start],&m_innerIndices[end],inner);
      const Index id = r-&m_innerIndices[0];
      eigen_assert((*r==inner) && (id<end) && "coeffRef cannot be called on a zero coefficient");
      return m_values[id];
    }

    class InnerIterator;
    class ReverseInnerIterator;

    /** \returns the number of non zero coefficients */
    inline Index nonZeros() const  { return m_nnz; }

    inline MappedSparseMatrix(Index rows, Index cols, Index nnz, Index* outerIndexPtr, Index* innerIndexPtr, Scalar* valuePtr)
      : m_outerSize(IsRowMajor?rows:cols), m_innerSize(IsRowMajor?cols:rows), m_nnz(nnz), m_outerIndex(outerIndexPtr),
        m_innerIndices(innerIndexPtr), m_values(valuePtr)
    {}

    /** Empty destructor */
    inline ~MappedSparseMatrix() {}
};

template<typename Scalar, int _Flags, typename _Index>
class MappedSparseMatrix<Scalar,_Flags,_Index>::InnerIterator
{
  public:
    InnerIterator(const MappedSparseMatrix& mat, Index outer)
      : m_matrix(mat),
        m_outer(outer),
        m_id(mat.outerIndexPtr()[outer]),
        m_start(m_id),
        m_end(mat.outerIndexPtr()[outer+1])
    {}

    inline InnerIterator& operator++() { m_id++; return *this; }

    inline Scalar value() const { return m_matrix.valuePtr()[m_id]; }
    inline Scalar& valueRef() { return const_cast<Scalar&>(m_matrix.valuePtr()[m_id]); }

    inline Index index() const { return m_matrix.innerIndexPtr()[m_id]; }
    inline Index row() const { return IsRowMajor ? m_outer : index(); }
    inline Index col() const { return IsRowMajor ? index() : m_outer; }

    inline operator bool() const { return (m_id < m_end) && (m_id>=m_start); }

  protected:
    const MappedSparseMatrix& m_matrix;
    const Index m_outer;
    Index m_id;
    const Index m_start;
    const Index m_end;
};

template<typename Scalar, int _Flags, typename _Index>
class MappedSparseMatrix<Scalar,_Flags,_Index>::ReverseInnerIterator
{
  public:
    ReverseInnerIterator(const MappedSparseMatrix& mat, Index outer)
      : m_matrix(mat),
        m_outer(outer),
        m_id(mat.outerIndexPtr()[outer+1]),
        m_start(mat.outerIndexPtr()[outer]),
        m_end(m_id)
    {}

    inline ReverseInnerIterator& operator--() { m_id--; return *this; }

    inline Scalar value() const { return m_matrix.valuePtr()[m_id-1]; }
    inline Scalar& valueRef() { return const_cast<Scalar&>(m_matrix.valuePtr()[m_id-1]); }

    inline Index index() const { return m_matrix.innerIndexPtr()[m_id-1]; }
    inline Index row() const { return IsRowMajor ? m_outer : index(); }
    inline Index col() const { return IsRowMajor ? index() : m_outer; }

    inline operator bool() const { return (m_id <= m_end) && (m_id>m_start); }

  protected:
    const MappedSparseMatrix& m_matrix;
    const Index m_outer;
    Index m_id;
    const Index m_start;
    const Index m_end;
};

} // end namespace Eigen

#endif // EIGEN_MAPPED_SPARSEMATRIX_H
