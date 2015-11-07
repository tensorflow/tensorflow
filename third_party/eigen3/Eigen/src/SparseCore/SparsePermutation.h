// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_PERMUTATION_H
#define EIGEN_SPARSE_PERMUTATION_H

// This file implements sparse * permutation products

namespace Eigen { 

namespace internal {

template<typename PermutationType, typename MatrixType, int Side, bool Transposed>
struct traits<permut_sparsematrix_product_retval<PermutationType, MatrixType, Side, Transposed> >
{
  typedef typename remove_all<typename MatrixType::Nested>::type MatrixTypeNestedCleaned;
  typedef typename MatrixTypeNestedCleaned::Scalar Scalar;
  typedef typename MatrixTypeNestedCleaned::Index Index;
  enum {
    SrcStorageOrder = MatrixTypeNestedCleaned::Flags&RowMajorBit ? RowMajor : ColMajor,
    MoveOuter = SrcStorageOrder==RowMajor ? Side==OnTheLeft : Side==OnTheRight
  };

  typedef typename internal::conditional<MoveOuter,
        SparseMatrix<Scalar,SrcStorageOrder,Index>,
        SparseMatrix<Scalar,int(SrcStorageOrder)==RowMajor?ColMajor:RowMajor,Index> >::type ReturnType;
};

template<typename PermutationType, typename MatrixType, int Side, bool Transposed>
struct permut_sparsematrix_product_retval
 : public ReturnByValue<permut_sparsematrix_product_retval<PermutationType, MatrixType, Side, Transposed> >
{
    typedef typename remove_all<typename MatrixType::Nested>::type MatrixTypeNestedCleaned;
    typedef typename MatrixTypeNestedCleaned::Scalar Scalar;
    typedef typename MatrixTypeNestedCleaned::Index Index;

    enum {
      SrcStorageOrder = MatrixTypeNestedCleaned::Flags&RowMajorBit ? RowMajor : ColMajor,
      MoveOuter = SrcStorageOrder==RowMajor ? Side==OnTheLeft : Side==OnTheRight
    };

    permut_sparsematrix_product_retval(const PermutationType& perm, const MatrixType& matrix)
      : m_permutation(perm), m_matrix(matrix)
    {}

    inline int rows() const { return m_matrix.rows(); }
    inline int cols() const { return m_matrix.cols(); }

    template<typename Dest> inline void evalTo(Dest& dst) const
    {
      if(MoveOuter)
      {
        SparseMatrix<Scalar,SrcStorageOrder,Index> tmp(m_matrix.rows(), m_matrix.cols());
        Matrix<Index,Dynamic,1> sizes(m_matrix.outerSize());
        for(Index j=0; j<m_matrix.outerSize(); ++j)
        {
          Index jp = m_permutation.indices().coeff(j);
          sizes[((Side==OnTheLeft) ^ Transposed) ? jp : j] = m_matrix.innerVector(((Side==OnTheRight) ^ Transposed) ? jp : j).size();
        }
        tmp.reserve(sizes);
        for(Index j=0; j<m_matrix.outerSize(); ++j)
        {
          Index jp = m_permutation.indices().coeff(j);
          Index jsrc = ((Side==OnTheRight) ^ Transposed) ? jp : j;
          Index jdst = ((Side==OnTheLeft) ^ Transposed) ? jp : j;
          for(typename MatrixTypeNestedCleaned::InnerIterator it(m_matrix,jsrc); it; ++it)
            tmp.insertByOuterInner(jdst,it.index()) = it.value();
        }
        dst = tmp;
      }
      else
      {
        SparseMatrix<Scalar,int(SrcStorageOrder)==RowMajor?ColMajor:RowMajor,Index> tmp(m_matrix.rows(), m_matrix.cols());
        Matrix<Index,Dynamic,1> sizes(tmp.outerSize());
        sizes.setZero();
        PermutationMatrix<Dynamic,Dynamic,Index> perm;
        if((Side==OnTheLeft) ^ Transposed)
          perm = m_permutation;
        else
          perm = m_permutation.transpose();

        for(Index j=0; j<m_matrix.outerSize(); ++j)
          for(typename MatrixTypeNestedCleaned::InnerIterator it(m_matrix,j); it; ++it)
            sizes[perm.indices().coeff(it.index())]++;
        tmp.reserve(sizes);
        for(Index j=0; j<m_matrix.outerSize(); ++j)
          for(typename MatrixTypeNestedCleaned::InnerIterator it(m_matrix,j); it; ++it)
            tmp.insertByOuterInner(perm.indices().coeff(it.index()),j) = it.value();
        dst = tmp;
      }
    }

  protected:
    const PermutationType& m_permutation;
    typename MatrixType::Nested m_matrix;
};

}



/** \returns the matrix with the permutation applied to the columns
  */
template<typename SparseDerived, typename PermDerived>
inline const internal::permut_sparsematrix_product_retval<PermutationBase<PermDerived>, SparseDerived, OnTheRight, false>
operator*(const SparseMatrixBase<SparseDerived>& matrix, const PermutationBase<PermDerived>& perm)
{
  return internal::permut_sparsematrix_product_retval<PermutationBase<PermDerived>, SparseDerived, OnTheRight, false>(perm, matrix.derived());
}

/** \returns the matrix with the permutation applied to the rows
  */
template<typename SparseDerived, typename PermDerived>
inline const internal::permut_sparsematrix_product_retval<PermutationBase<PermDerived>, SparseDerived, OnTheLeft, false>
operator*( const PermutationBase<PermDerived>& perm, const SparseMatrixBase<SparseDerived>& matrix)
{
  return internal::permut_sparsematrix_product_retval<PermutationBase<PermDerived>, SparseDerived, OnTheLeft, false>(perm, matrix.derived());
}



/** \returns the matrix with the inverse permutation applied to the columns.
  */
template<typename SparseDerived, typename PermDerived>
inline const internal::permut_sparsematrix_product_retval<PermutationBase<PermDerived>, SparseDerived, OnTheRight, true>
operator*(const SparseMatrixBase<SparseDerived>& matrix, const Transpose<PermutationBase<PermDerived> >& tperm)
{
  return internal::permut_sparsematrix_product_retval<PermutationBase<PermDerived>, SparseDerived, OnTheRight, true>(tperm.nestedPermutation(), matrix.derived());
}

/** \returns the matrix with the inverse permutation applied to the rows.
  */
template<typename SparseDerived, typename PermDerived>
inline const internal::permut_sparsematrix_product_retval<PermutationBase<PermDerived>, SparseDerived, OnTheLeft, true>
operator*(const Transpose<PermutationBase<PermDerived> >& tperm, const SparseMatrixBase<SparseDerived>& matrix)
{
  return internal::permut_sparsematrix_product_retval<PermutationBase<PermDerived>, SparseDerived, OnTheLeft, true>(tperm.nestedPermutation(), matrix.derived());
}

} // end namespace Eigen

#endif // EIGEN_SPARSE_SELFADJOINTVIEW_H
