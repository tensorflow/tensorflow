// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Vincent Lejeune
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BLOCK_HOUSEHOLDER_H
#define EIGEN_BLOCK_HOUSEHOLDER_H

// This file contains some helper function to deal with block householder reflectors

namespace Eigen { 

namespace internal {

/** \internal */
template<typename TriangularFactorType,typename VectorsType,typename CoeffsType>
void make_block_householder_triangular_factor(TriangularFactorType& triFactor, const VectorsType& vectors, const CoeffsType& hCoeffs)
{
  typedef typename TriangularFactorType::Index Index;
  typedef typename VectorsType::Scalar Scalar;
  const Index nbVecs = vectors.cols();
  eigen_assert(triFactor.rows() == nbVecs && triFactor.cols() == nbVecs && vectors.rows()>=nbVecs);

  for(Index i = 0; i < nbVecs; i++)
  {
    Index rs = vectors.rows() - i;
    Scalar Vii = vectors(i,i);
    vectors.const_cast_derived().coeffRef(i,i) = Scalar(1);
    triFactor.col(i).head(i).noalias() = -hCoeffs(i) * vectors.block(i, 0, rs, i).adjoint()
                                       * vectors.col(i).tail(rs);
    vectors.const_cast_derived().coeffRef(i, i) = Vii;
    // FIXME add .noalias() once the triangular product can work inplace
    triFactor.col(i).head(i) = triFactor.block(0,0,i,i).template triangularView<Upper>()
                             * triFactor.col(i).head(i);
    triFactor(i,i) = hCoeffs(i);
  }
}

/** \internal */
template<typename MatrixType,typename VectorsType,typename CoeffsType>
void apply_block_householder_on_the_left(MatrixType& mat, const VectorsType& vectors, const CoeffsType& hCoeffs)
{
  typedef typename MatrixType::Index Index;
  enum { TFactorSize = MatrixType::ColsAtCompileTime };
  Index nbVecs = vectors.cols();
  Matrix<typename MatrixType::Scalar, TFactorSize, TFactorSize, ColMajor> T(nbVecs,nbVecs);
  make_block_householder_triangular_factor(T, vectors, hCoeffs);

  const TriangularView<const VectorsType, UnitLower>& V(vectors);

  // A -= V T V^* A
  Matrix<typename MatrixType::Scalar,VectorsType::ColsAtCompileTime,MatrixType::ColsAtCompileTime,0,
         VectorsType::MaxColsAtCompileTime,MatrixType::MaxColsAtCompileTime> tmp = V.adjoint() * mat;
  // FIXME add .noalias() once the triangular product can work inplace
  tmp = T.template triangularView<Upper>().adjoint() * tmp;
  mat.noalias() -= V * tmp;
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_BLOCK_HOUSEHOLDER_H
