// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN2_SVD_H
#define EIGEN2_SVD_H

namespace Eigen {

/** \ingroup SVD_Module
  * \nonstableyet
  *
  * \class SVD
  *
  * \brief Standard SVD decomposition of a matrix and associated features
  *
  * \param MatrixType the type of the matrix of which we are computing the SVD decomposition
  *
  * This class performs a standard SVD decomposition of a real matrix A of size \c M x \c N
  * with \c M \>= \c N.
  *
  *
  * \sa MatrixBase::SVD()
  */
template<typename MatrixType> class SVD
{
  private:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;

    enum {
      PacketSize = internal::packet_traits<Scalar>::size,
      AlignmentMask = int(PacketSize)-1,
      MinSize = EIGEN_SIZE_MIN_PREFER_DYNAMIC(MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime)
    };

    typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> ColVector;
    typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> RowVector;

    typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MinSize> MatrixUType;
    typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, MatrixType::ColsAtCompileTime> MatrixVType;
    typedef Matrix<Scalar, MinSize, 1> SingularValuesType;

  public:

    SVD() {} // a user who relied on compiler-generated default compiler reported problems with MSVC in 2.0.7
    
    SVD(const MatrixType& matrix)
      : m_matU(matrix.rows(), (std::min)(matrix.rows(), matrix.cols())),
        m_matV(matrix.cols(),matrix.cols()),
        m_sigma((std::min)(matrix.rows(),matrix.cols()))
    {
      compute(matrix);
    }

    template<typename OtherDerived, typename ResultType>
    bool solve(const MatrixBase<OtherDerived> &b, ResultType* result) const;

    const MatrixUType& matrixU() const { return m_matU; }
    const SingularValuesType& singularValues() const { return m_sigma; }
    const MatrixVType& matrixV() const { return m_matV; }

    void compute(const MatrixType& matrix);
    SVD& sort();

    template<typename UnitaryType, typename PositiveType>
    void computeUnitaryPositive(UnitaryType *unitary, PositiveType *positive) const;
    template<typename PositiveType, typename UnitaryType>
    void computePositiveUnitary(PositiveType *positive, UnitaryType *unitary) const;
    template<typename RotationType, typename ScalingType>
    void computeRotationScaling(RotationType *unitary, ScalingType *positive) const;
    template<typename ScalingType, typename RotationType>
    void computeScalingRotation(ScalingType *positive, RotationType *unitary) const;

  protected:
    /** \internal */
    MatrixUType m_matU;
    /** \internal */
    MatrixVType m_matV;
    /** \internal */
    SingularValuesType m_sigma;
};

/** Computes / recomputes the SVD decomposition A = U S V^* of \a matrix
  *
  * \note this code has been adapted from JAMA (public domain)
  */
template<typename MatrixType>
void SVD<MatrixType>::compute(const MatrixType& matrix)
{
  const int m = matrix.rows();
  const int n = matrix.cols();
  const int nu = (std::min)(m,n);
  ei_assert(m>=n && "In Eigen 2.0, SVD only works for MxN matrices with M>=N. Sorry!");
  ei_assert(m>1 && "In Eigen 2.0, SVD doesn't work on 1x1 matrices");

  m_matU.resize(m, nu);
  m_matU.setZero();
  m_sigma.resize((std::min)(m,n));
  m_matV.resize(n,n);

  RowVector e(n);
  ColVector work(m);
  MatrixType matA(matrix);
  const bool wantu = true;
  const bool wantv = true;
  int i=0, j=0, k=0;

  // Reduce A to bidiagonal form, storing the diagonal elements
  // in s and the super-diagonal elements in e.
  int nct = (std::min)(m-1,n);
  int nrt = (std::max)(0,(std::min)(n-2,m));
  for (k = 0; k < (std::max)(nct,nrt); ++k)
  {
    if (k < nct)
    {
      // Compute the transformation for the k-th column and
      // place the k-th diagonal in m_sigma[k].
      m_sigma[k] = matA.col(k).end(m-k).norm();
      if (m_sigma[k] != 0.0) // FIXME
      {
        if (matA(k,k) < 0.0)
          m_sigma[k] = -m_sigma[k];
        matA.col(k).end(m-k) /= m_sigma[k];
        matA(k,k) += 1.0;
      }
      m_sigma[k] = -m_sigma[k];
    }

    for (j = k+1; j < n; ++j)
    {
      if ((k < nct) && (m_sigma[k] != 0.0))
      {
        // Apply the transformation.
        Scalar t = matA.col(k).end(m-k).eigen2_dot(matA.col(j).end(m-k)); // FIXME dot product or cwise prod + .sum() ??
        t = -t/matA(k,k);
        matA.col(j).end(m-k) += t * matA.col(k).end(m-k);
      }

      // Place the k-th row of A into e for the
      // subsequent calculation of the row transformation.
      e[j] = matA(k,j);
    }

    // Place the transformation in U for subsequent back multiplication.
    if (wantu & (k < nct))
      m_matU.col(k).end(m-k) = matA.col(k).end(m-k);

    if (k < nrt)
    {
      // Compute the k-th row transformation and place the
      // k-th super-diagonal in e[k].
      e[k] = e.end(n-k-1).norm();
      if (e[k] != 0.0)
      {
          if (e[k+1] < 0.0)
            e[k] = -e[k];
          e.end(n-k-1) /= e[k];
          e[k+1] += 1.0;
      }
      e[k] = -e[k];
      if ((k+1 < m) & (e[k] != 0.0))
      {
        // Apply the transformation.
        work.end(m-k-1) = matA.corner(BottomRight,m-k-1,n-k-1) * e.end(n-k-1);
        for (j = k+1; j < n; ++j)
          matA.col(j).end(m-k-1) += (-e[j]/e[k+1]) * work.end(m-k-1);
      }

      // Place the transformation in V for subsequent back multiplication.
      if (wantv)
        m_matV.col(k).end(n-k-1) = e.end(n-k-1);
    }
  }


  // Set up the final bidiagonal matrix or order p.
  int p = (std::min)(n,m+1);
  if (nct < n)
    m_sigma[nct] = matA(nct,nct);
  if (m < p)
    m_sigma[p-1] = 0.0;
  if (nrt+1 < p)
    e[nrt] = matA(nrt,p-1);
  e[p-1] = 0.0;

  // If required, generate U.
  if (wantu)
  {
    for (j = nct; j < nu; ++j)
    {
      m_matU.col(j).setZero();
      m_matU(j,j) = 1.0;
    }
    for (k = nct-1; k >= 0; k--)
    {
      if (m_sigma[k] != 0.0)
      {
        for (j = k+1; j < nu; ++j)
        {
          Scalar t = m_matU.col(k).end(m-k).eigen2_dot(m_matU.col(j).end(m-k)); // FIXME is it really a dot product we want ?
          t = -t/m_matU(k,k);
          m_matU.col(j).end(m-k) += t * m_matU.col(k).end(m-k);
        }
        m_matU.col(k).end(m-k) = - m_matU.col(k).end(m-k);
        m_matU(k,k) = Scalar(1) + m_matU(k,k);
        if (k-1>0)
          m_matU.col(k).start(k-1).setZero();
      }
      else
      {
        m_matU.col(k).setZero();
        m_matU(k,k) = 1.0;
      }
    }
  }

  // If required, generate V.
  if (wantv)
  {
    for (k = n-1; k >= 0; k--)
    {
      if ((k < nrt) & (e[k] != 0.0))
      {
        for (j = k+1; j < nu; ++j)
        {
          Scalar t = m_matV.col(k).end(n-k-1).eigen2_dot(m_matV.col(j).end(n-k-1)); // FIXME is it really a dot product we want ?
          t = -t/m_matV(k+1,k);
          m_matV.col(j).end(n-k-1) += t * m_matV.col(k).end(n-k-1);
        }
      }
      m_matV.col(k).setZero();
      m_matV(k,k) = 1.0;
    }
  }

  // Main iteration loop for the singular values.
  int pp = p-1;
  int iter = 0;
  Scalar eps = ei_pow(Scalar(2),ei_is_same_type<Scalar,float>::ret ? Scalar(-23) : Scalar(-52));
  while (p > 0)
  {
    int k=0;
    int kase=0;

    // Here is where a test for too many iterations would go.

    // This section of the program inspects for
    // negligible elements in the s and e arrays.  On
    // completion the variables kase and k are set as follows.

    // kase = 1     if s(p) and e[k-1] are negligible and k<p
    // kase = 2     if s(k) is negligible and k<p
    // kase = 3     if e[k-1] is negligible, k<p, and
    //              s(k), ..., s(p) are not negligible (qr step).
    // kase = 4     if e(p-1) is negligible (convergence).

    for (k = p-2; k >= -1; --k)
    {
      if (k == -1)
          break;
      if (ei_abs(e[k]) <= eps*(ei_abs(m_sigma[k]) + ei_abs(m_sigma[k+1])))
      {
          e[k] = 0.0;
          break;
      }
    }
    if (k == p-2)
    {
      kase = 4;
    }
    else
    {
      int ks;
      for (ks = p-1; ks >= k; --ks)
      {
        if (ks == k)
          break;
        Scalar t = (ks != p ? ei_abs(e[ks]) : Scalar(0)) + (ks != k+1 ? ei_abs(e[ks-1]) : Scalar(0));
        if (ei_abs(m_sigma[ks]) <= eps*t)
        {
          m_sigma[ks] = 0.0;
          break;
        }
      }
      if (ks == k)
      {
        kase = 3;
      }
      else if (ks == p-1)
      {
        kase = 1;
      }
      else
      {
        kase = 2;
        k = ks;
      }
    }
    ++k;

    // Perform the task indicated by kase.
    switch (kase)
    {

      // Deflate negligible s(p).
      case 1:
      {
        Scalar f(e[p-2]);
        e[p-2] = 0.0;
        for (j = p-2; j >= k; --j)
        {
          Scalar t(numext::hypot(m_sigma[j],f));
          Scalar cs(m_sigma[j]/t);
          Scalar sn(f/t);
          m_sigma[j] = t;
          if (j != k)
          {
            f = -sn*e[j-1];
            e[j-1] = cs*e[j-1];
          }
          if (wantv)
          {
            for (i = 0; i < n; ++i)
            {
              t = cs*m_matV(i,j) + sn*m_matV(i,p-1);
              m_matV(i,p-1) = -sn*m_matV(i,j) + cs*m_matV(i,p-1);
              m_matV(i,j) = t;
            }
          }
        }
      }
      break;

      // Split at negligible s(k).
      case 2:
      {
        Scalar f(e[k-1]);
        e[k-1] = 0.0;
        for (j = k; j < p; ++j)
        {
          Scalar t(numext::hypot(m_sigma[j],f));
          Scalar cs( m_sigma[j]/t);
          Scalar sn(f/t);
          m_sigma[j] = t;
          f = -sn*e[j];
          e[j] = cs*e[j];
          if (wantu)
          {
            for (i = 0; i < m; ++i)
            {
              t = cs*m_matU(i,j) + sn*m_matU(i,k-1);
              m_matU(i,k-1) = -sn*m_matU(i,j) + cs*m_matU(i,k-1);
              m_matU(i,j) = t;
            }
          }
        }
      }
      break;

      // Perform one qr step.
      case 3:
      {
        // Calculate the shift.
        Scalar scale = (std::max)((std::max)((std::max)((std::max)(
                        ei_abs(m_sigma[p-1]),ei_abs(m_sigma[p-2])),ei_abs(e[p-2])),
                        ei_abs(m_sigma[k])),ei_abs(e[k]));
        Scalar sp = m_sigma[p-1]/scale;
        Scalar spm1 = m_sigma[p-2]/scale;
        Scalar epm1 = e[p-2]/scale;
        Scalar sk = m_sigma[k]/scale;
        Scalar ek = e[k]/scale;
        Scalar b = ((spm1 + sp)*(spm1 - sp) + epm1*epm1)/Scalar(2);
        Scalar c = (sp*epm1)*(sp*epm1);
        Scalar shift(0);
        if ((b != 0.0) || (c != 0.0))
        {
          shift = ei_sqrt(b*b + c);
          if (b < 0.0)
            shift = -shift;
          shift = c/(b + shift);
        }
        Scalar f = (sk + sp)*(sk - sp) + shift;
        Scalar g = sk*ek;

        // Chase zeros.

        for (j = k; j < p-1; ++j)
        {
          Scalar t = numext::hypot(f,g);
          Scalar cs = f/t;
          Scalar sn = g/t;
          if (j != k)
            e[j-1] = t;
          f = cs*m_sigma[j] + sn*e[j];
          e[j] = cs*e[j] - sn*m_sigma[j];
          g = sn*m_sigma[j+1];
          m_sigma[j+1] = cs*m_sigma[j+1];
          if (wantv)
          {
            for (i = 0; i < n; ++i)
            {
              t = cs*m_matV(i,j) + sn*m_matV(i,j+1);
              m_matV(i,j+1) = -sn*m_matV(i,j) + cs*m_matV(i,j+1);
              m_matV(i,j) = t;
            }
          }
          t = numext::hypot(f,g);
          cs = f/t;
          sn = g/t;
          m_sigma[j] = t;
          f = cs*e[j] + sn*m_sigma[j+1];
          m_sigma[j+1] = -sn*e[j] + cs*m_sigma[j+1];
          g = sn*e[j+1];
          e[j+1] = cs*e[j+1];
          if (wantu && (j < m-1))
          {
            for (i = 0; i < m; ++i)
            {
              t = cs*m_matU(i,j) + sn*m_matU(i,j+1);
              m_matU(i,j+1) = -sn*m_matU(i,j) + cs*m_matU(i,j+1);
              m_matU(i,j) = t;
            }
          }
        }
        e[p-2] = f;
        iter = iter + 1;
      }
      break;

      // Convergence.
      case 4:
      {
        // Make the singular values positive.
        if (m_sigma[k] <= 0.0)
        {
          m_sigma[k] = m_sigma[k] < Scalar(0) ? -m_sigma[k] : Scalar(0);
          if (wantv)
            m_matV.col(k).start(pp+1) = -m_matV.col(k).start(pp+1);
        }

        // Order the singular values.
        while (k < pp)
        {
          if (m_sigma[k] >= m_sigma[k+1])
            break;
          Scalar t = m_sigma[k];
          m_sigma[k] = m_sigma[k+1];
          m_sigma[k+1] = t;
          if (wantv && (k < n-1))
            m_matV.col(k).swap(m_matV.col(k+1));
          if (wantu && (k < m-1))
            m_matU.col(k).swap(m_matU.col(k+1));
          ++k;
        }
        iter = 0;
        p--;
      }
      break;
    } // end big switch
  } // end iterations
}

template<typename MatrixType>
SVD<MatrixType>& SVD<MatrixType>::sort()
{
  int mu = m_matU.rows();
  int mv = m_matV.rows();
  int n  = m_matU.cols();

  for (int i=0; i<n; ++i)
  {
    int  k = i;
    Scalar p = m_sigma.coeff(i);

    for (int j=i+1; j<n; ++j)
    {
      if (m_sigma.coeff(j) > p)
      {
        k = j;
        p = m_sigma.coeff(j);
      }
    }
    if (k != i)
    {
      m_sigma.coeffRef(k) = m_sigma.coeff(i);  // i.e.
      m_sigma.coeffRef(i) = p;                 // swaps the i-th and the k-th elements

      int j = mu;
      for(int s=0; j!=0; ++s, --j)
        std::swap(m_matU.coeffRef(s,i), m_matU.coeffRef(s,k));

      j = mv;
      for (int s=0; j!=0; ++s, --j)
        std::swap(m_matV.coeffRef(s,i), m_matV.coeffRef(s,k));
    }
  }
  return *this;
}

/** \returns the solution of \f$ A x = b \f$ using the current SVD decomposition of A.
  * The parts of the solution corresponding to zero singular values are ignored.
  *
  * \sa MatrixBase::svd(), LU::solve(), LLT::solve()
  */
template<typename MatrixType>
template<typename OtherDerived, typename ResultType>
bool SVD<MatrixType>::solve(const MatrixBase<OtherDerived> &b, ResultType* result) const
{
  ei_assert(b.rows() == m_matU.rows());

  Scalar maxVal = m_sigma.cwise().abs().maxCoeff();
  for (int j=0; j<b.cols(); ++j)
  {
    Matrix<Scalar,MatrixUType::RowsAtCompileTime,1> aux = m_matU.transpose() * b.col(j);

    for (int i = 0; i <m_matU.cols(); ++i)
    {
      Scalar si = m_sigma.coeff(i);
      if (ei_isMuchSmallerThan(ei_abs(si),maxVal))
        aux.coeffRef(i) = 0;
      else
        aux.coeffRef(i) /= si;
    }

    result->col(j) = m_matV * aux;
  }
  return true;
}

/** Computes the polar decomposition of the matrix, as a product unitary x positive.
  *
  * If either pointer is zero, the corresponding computation is skipped.
  *
  * Only for square matrices.
  *
  * \sa computePositiveUnitary(), computeRotationScaling()
  */
template<typename MatrixType>
template<typename UnitaryType, typename PositiveType>
void SVD<MatrixType>::computeUnitaryPositive(UnitaryType *unitary,
                                             PositiveType *positive) const
{
  ei_assert(m_matU.cols() == m_matV.cols() && "Polar decomposition is only for square matrices");
  if(unitary) *unitary = m_matU * m_matV.adjoint();
  if(positive) *positive = m_matV * m_sigma.asDiagonal() * m_matV.adjoint();
}

/** Computes the polar decomposition of the matrix, as a product positive x unitary.
  *
  * If either pointer is zero, the corresponding computation is skipped.
  *
  * Only for square matrices.
  *
  * \sa computeUnitaryPositive(), computeRotationScaling()
  */
template<typename MatrixType>
template<typename UnitaryType, typename PositiveType>
void SVD<MatrixType>::computePositiveUnitary(UnitaryType *positive,
                                             PositiveType *unitary) const
{
  ei_assert(m_matU.rows() == m_matV.rows() && "Polar decomposition is only for square matrices");
  if(unitary) *unitary = m_matU * m_matV.adjoint();
  if(positive) *positive = m_matU * m_sigma.asDiagonal() * m_matU.adjoint();
}

/** decomposes the matrix as a product rotation x scaling, the scaling being
  * not necessarily positive.
  *
  * If either pointer is zero, the corresponding computation is skipped.
  *
  * This method requires the Geometry module.
  *
  * \sa computeScalingRotation(), computeUnitaryPositive()
  */
template<typename MatrixType>
template<typename RotationType, typename ScalingType>
void SVD<MatrixType>::computeRotationScaling(RotationType *rotation, ScalingType *scaling) const
{
  ei_assert(m_matU.rows() == m_matV.rows() && "Polar decomposition is only for square matrices");
  Scalar x = (m_matU * m_matV.adjoint()).determinant(); // so x has absolute value 1
  Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> sv(m_sigma);
  sv.coeffRef(0) *= x;
  if(scaling) scaling->lazyAssign(m_matV * sv.asDiagonal() * m_matV.adjoint());
  if(rotation)
  {
    MatrixType m(m_matU);
    m.col(0) /= x;
    rotation->lazyAssign(m * m_matV.adjoint());
  }
}

/** decomposes the matrix as a product scaling x rotation, the scaling being
  * not necessarily positive.
  *
  * If either pointer is zero, the corresponding computation is skipped.
  *
  * This method requires the Geometry module.
  *
  * \sa computeRotationScaling(), computeUnitaryPositive()
  */
template<typename MatrixType>
template<typename ScalingType, typename RotationType>
void SVD<MatrixType>::computeScalingRotation(ScalingType *scaling, RotationType *rotation) const
{
  ei_assert(m_matU.rows() == m_matV.rows() && "Polar decomposition is only for square matrices");
  Scalar x = (m_matU * m_matV.adjoint()).determinant(); // so x has absolute value 1
  Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> sv(m_sigma);
  sv.coeffRef(0) *= x;
  if(scaling) scaling->lazyAssign(m_matU * sv.asDiagonal() * m_matU.adjoint());
  if(rotation)
  {
    MatrixType m(m_matU);
    m.col(0) /= x;
    rotation->lazyAssign(m * m_matV.adjoint());
  }
}


/** \svd_module
  * \returns the SVD decomposition of \c *this
  */
template<typename Derived>
inline SVD<typename MatrixBase<Derived>::PlainObject>
MatrixBase<Derived>::svd() const
{
  return SVD<PlainObject>(derived());
}

} // end namespace Eigen

#endif // EIGEN2_SVD_H
