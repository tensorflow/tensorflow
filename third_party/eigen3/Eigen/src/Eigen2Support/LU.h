// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN2_LU_H
#define EIGEN2_LU_H

namespace Eigen { 

template<typename MatrixType>
class LU : public FullPivLU<MatrixType>
{
  public:

    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
    typedef Matrix<int, 1, MatrixType::ColsAtCompileTime, MatrixType::Options, 1, MatrixType::MaxColsAtCompileTime> IntRowVectorType;
    typedef Matrix<int, MatrixType::RowsAtCompileTime, 1, MatrixType::Options, MatrixType::MaxRowsAtCompileTime, 1> IntColVectorType;
    typedef Matrix<Scalar, 1, MatrixType::ColsAtCompileTime, MatrixType::Options, 1, MatrixType::MaxColsAtCompileTime> RowVectorType;
    typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1, MatrixType::Options, MatrixType::MaxRowsAtCompileTime, 1> ColVectorType;

    typedef Matrix<typename MatrixType::Scalar,
                  MatrixType::ColsAtCompileTime, // the number of rows in the "kernel matrix" is the number of cols of the original matrix
                                                 // so that the product "matrix * kernel = zero" makes sense
                  Dynamic,                       // we don't know at compile-time the dimension of the kernel
                  MatrixType::Options,
                  MatrixType::MaxColsAtCompileTime, // see explanation for 2nd template parameter
                  MatrixType::MaxColsAtCompileTime // the kernel is a subspace of the domain space, whose dimension is the number
                                                   // of columns of the original matrix
    > KernelResultType;

    typedef Matrix<typename MatrixType::Scalar,
                   MatrixType::RowsAtCompileTime, // the image is a subspace of the destination space, whose dimension is the number
                                                  // of rows of the original matrix
                   Dynamic,                       // we don't know at compile time the dimension of the image (the rank)
                   MatrixType::Options,
                   MatrixType::MaxRowsAtCompileTime, // the image matrix will consist of columns from the original matrix,
                   MatrixType::MaxColsAtCompileTime  // so it has the same number of rows and at most as many columns.
    > ImageResultType;

    typedef FullPivLU<MatrixType> Base;

    template<typename T>
    explicit LU(const T& t) : Base(t), m_originalMatrix(t) {}

    template<typename OtherDerived, typename ResultType>
    bool solve(const MatrixBase<OtherDerived>& b, ResultType *result) const
    {
      *result = static_cast<const Base*>(this)->solve(b);
      return true;
    }

    template<typename ResultType>
    inline void computeInverse(ResultType *result) const
    {
      solve(MatrixType::Identity(this->rows(), this->cols()), result);
    }
    
    template<typename KernelMatrixType>
    void computeKernel(KernelMatrixType *result) const
    {
      *result = static_cast<const Base*>(this)->kernel();
    }
    
    template<typename ImageMatrixType>
    void computeImage(ImageMatrixType *result) const
    {
      *result = static_cast<const Base*>(this)->image(m_originalMatrix);
    }
    
    const ImageResultType image() const
    {
      return static_cast<const Base*>(this)->image(m_originalMatrix);
    }
    
    const MatrixType& m_originalMatrix;
};

#if EIGEN2_SUPPORT_STAGE < STAGE20_RESOLVE_API_CONFLICTS
/** \lu_module
  *
  * Synonym of partialPivLu().
  *
  * \return the partial-pivoting LU decomposition of \c *this.
  *
  * \sa class PartialPivLU
  */
template<typename Derived>
inline const LU<typename MatrixBase<Derived>::PlainObject>
MatrixBase<Derived>::lu() const
{
  return LU<PlainObject>(eval());
}
#endif

#ifdef EIGEN2_SUPPORT
/** \lu_module
  *
  * Synonym of partialPivLu().
  *
  * \return the partial-pivoting LU decomposition of \c *this.
  *
  * \sa class PartialPivLU
  */
template<typename Derived>
inline const LU<typename MatrixBase<Derived>::PlainObject>
MatrixBase<Derived>::eigen2_lu() const
{
  return LU<PlainObject>(eval());
}
#endif

} // end namespace Eigen

#endif // EIGEN2_LU_H
