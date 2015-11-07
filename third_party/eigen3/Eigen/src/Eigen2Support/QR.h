// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN2_QR_H
#define EIGEN2_QR_H

namespace Eigen { 

template<typename MatrixType>
class QR : public HouseholderQR<MatrixType>
{
  public:

    typedef HouseholderQR<MatrixType> Base;
    typedef Block<const MatrixType, MatrixType::ColsAtCompileTime, MatrixType::ColsAtCompileTime> MatrixRBlockType;

    QR() : Base() {}

    template<typename T>
    explicit QR(const T& t) : Base(t) {}

    template<typename OtherDerived, typename ResultType>
    bool solve(const MatrixBase<OtherDerived>& b, ResultType *result) const
    {
      *result = static_cast<const Base*>(this)->solve(b);
      return true;
    }

    MatrixType matrixQ(void) const {
      MatrixType ret = MatrixType::Identity(this->rows(), this->cols());
      ret = this->householderQ() * ret;
      return ret;
    }

    bool isFullRank() const {
      return true;
    }
    
    const TriangularView<MatrixRBlockType, UpperTriangular>
    matrixR(void) const
    {
      int cols = this->cols();
      return MatrixRBlockType(this->matrixQR(), 0, 0, cols, cols).template triangularView<UpperTriangular>();
    }
};

/** \return the QR decomposition of \c *this.
  *
  * \sa class QR
  */
template<typename Derived>
const QR<typename MatrixBase<Derived>::PlainObject>
MatrixBase<Derived>::qr() const
{
  return QR<PlainObject>(eval());
}

} // end namespace Eigen

#endif // EIGEN2_QR_H
