// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010,2012 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GENERALIZEDEIGENSOLVER_H
#define EIGEN_GENERALIZEDEIGENSOLVER_H

#include "./RealQZ.h"

namespace Eigen { 

/** \eigenvalues_module \ingroup Eigenvalues_Module
  *
  *
  * \class GeneralizedEigenSolver
  *
  * \brief Computes the generalized eigenvalues and eigenvectors of a pair of general matrices
  *
  * \tparam _MatrixType the type of the matrices of which we are computing the
  * eigen-decomposition; this is expected to be an instantiation of the Matrix
  * class template. Currently, only real matrices are supported.
  *
  * The generalized eigenvalues and eigenvectors of a matrix pair \f$ A \f$ and \f$ B \f$ are scalars
  * \f$ \lambda \f$ and vectors \f$ v \f$ such that \f$ Av = \lambda Bv \f$.  If
  * \f$ D \f$ is a diagonal matrix with the eigenvalues on the diagonal, and
  * \f$ V \f$ is a matrix with the eigenvectors as its columns, then \f$ A V =
  * B V D \f$. The matrix \f$ V \f$ is almost always invertible, in which case we
  * have \f$ A = B V D V^{-1} \f$. This is called the generalized eigen-decomposition.
  *
  * The generalized eigenvalues and eigenvectors of a matrix pair may be complex, even when the
  * matrices are real. Moreover, the generalized eigenvalue might be infinite if the matrix B is
  * singular. To workaround this difficulty, the eigenvalues are provided as a pair of complex \f$ \alpha \f$
  * and real \f$ \beta \f$ such that: \f$ \lambda_i = \alpha_i / \beta_i \f$. If \f$ \beta_i \f$ is (nearly) zero,
  * then one can consider the well defined left eigenvalue \f$ \mu = \beta_i / \alpha_i\f$ such that:
  * \f$ \mu_i A v_i = B v_i \f$, or even \f$ \mu_i u_i^T A  = u_i^T B \f$ where \f$ u_i \f$ is
  * called the left eigenvector.
  *
  * Call the function compute() to compute the generalized eigenvalues and eigenvectors of
  * a given matrix pair. Alternatively, you can use the
  * GeneralizedEigenSolver(const MatrixType&, const MatrixType&, bool) constructor which computes the
  * eigenvalues and eigenvectors at construction time. Once the eigenvalue and
  * eigenvectors are computed, they can be retrieved with the eigenvalues() and
  * eigenvectors() functions.
  *
  * Here is an usage example of this class:
  * Example: \include GeneralizedEigenSolver.cpp
  * Output: \verbinclude GeneralizedEigenSolver.out
  *
  * \sa MatrixBase::eigenvalues(), class ComplexEigenSolver, class SelfAdjointEigenSolver
  */
template<typename _MatrixType> class GeneralizedEigenSolver
{
  public:

    /** \brief Synonym for the template parameter \p _MatrixType. */
    typedef _MatrixType MatrixType;

    enum {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      Options = MatrixType::Options,
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };

    /** \brief Scalar type for matrices of type #MatrixType. */
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef typename MatrixType::Index Index;

    /** \brief Complex scalar type for #MatrixType. 
      *
      * This is \c std::complex<Scalar> if #Scalar is real (e.g.,
      * \c float or \c double) and just \c Scalar if #Scalar is
      * complex.
      */
    typedef std::complex<RealScalar> ComplexScalar;

    /** \brief Type for vector of real scalar values eigenvalues as returned by betas().
      *
      * This is a column vector with entries of type #Scalar.
      * The length of the vector is the size of #MatrixType.
      */
    typedef Matrix<Scalar, ColsAtCompileTime, 1, Options & ~RowMajor, MaxColsAtCompileTime, 1> VectorType;

    /** \brief Type for vector of complex scalar values eigenvalues as returned by betas().
      *
      * This is a column vector with entries of type #ComplexScalar.
      * The length of the vector is the size of #MatrixType.
      */
    typedef Matrix<ComplexScalar, ColsAtCompileTime, 1, Options & ~RowMajor, MaxColsAtCompileTime, 1> ComplexVectorType;

    /** \brief Expression type for the eigenvalues as returned by eigenvalues().
      */
    typedef CwiseBinaryOp<internal::scalar_quotient_op<ComplexScalar,Scalar>,ComplexVectorType,VectorType> EigenvalueType;

    /** \brief Type for matrix of eigenvectors as returned by eigenvectors(). 
      *
      * This is a square matrix with entries of type #ComplexScalar. 
      * The size is the same as the size of #MatrixType.
      */
    typedef Matrix<ComplexScalar, RowsAtCompileTime, ColsAtCompileTime, Options, MaxRowsAtCompileTime, MaxColsAtCompileTime> EigenvectorsType;

    /** \brief Default constructor.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via EigenSolver::compute(const MatrixType&, bool).
      *
      * \sa compute() for an example.
      */
    GeneralizedEigenSolver() : m_eivec(), m_alphas(), m_betas(), m_isInitialized(false), m_realQZ(), m_matS(), m_tmp() {}

    /** \brief Default constructor with memory preallocation
      *
      * Like the default constructor but with preallocation of the internal data
      * according to the specified problem \a size.
      * \sa GeneralizedEigenSolver()
      */
    GeneralizedEigenSolver(Index size)
      : m_eivec(size, size),
        m_alphas(size),
        m_betas(size),
        m_isInitialized(false),
        m_eigenvectorsOk(false),
        m_realQZ(size),
        m_matS(size, size),
        m_tmp(size)
    {}

    /** \brief Constructor; computes the generalized eigendecomposition of given matrix pair.
      * 
      * \param[in]  A  Square matrix whose eigendecomposition is to be computed.
      * \param[in]  B  Square matrix whose eigendecomposition is to be computed.
      * \param[in]  computeEigenvectors  If true, both the eigenvectors and the
      *    eigenvalues are computed; if false, only the eigenvalues are computed.
      *
      * This constructor calls compute() to compute the generalized eigenvalues
      * and eigenvectors.
      *
      * \sa compute()
      */
    GeneralizedEigenSolver(const MatrixType& A, const MatrixType& B, bool computeEigenvectors = true)
      : m_eivec(A.rows(), A.cols()),
        m_alphas(A.cols()),
        m_betas(A.cols()),
        m_isInitialized(false),
        m_eigenvectorsOk(false),
        m_realQZ(A.cols()),
        m_matS(A.rows(), A.cols()),
        m_tmp(A.cols())
    {
      compute(A, B, computeEigenvectors);
    }

    /* \brief Returns the computed generalized eigenvectors.
      *
      * \returns  %Matrix whose columns are the (possibly complex) eigenvectors.
      *
      * \pre Either the constructor 
      * GeneralizedEigenSolver(const MatrixType&,const MatrixType&, bool) or the member function
      * compute(const MatrixType&, const MatrixType& bool) has been called before, and
      * \p computeEigenvectors was set to true (the default).
      *
      * Column \f$ k \f$ of the returned matrix is an eigenvector corresponding
      * to eigenvalue number \f$ k \f$ as returned by eigenvalues().  The
      * eigenvectors are normalized to have (Euclidean) norm equal to one. The
      * matrix returned by this function is the matrix \f$ V \f$ in the
      * generalized eigendecomposition \f$ A = B V D V^{-1} \f$, if it exists.
      *
      * \sa eigenvalues()
      */
//    EigenvectorsType eigenvectors() const;

    /** \brief Returns an expression of the computed generalized eigenvalues.
      *
      * \returns An expression of the column vector containing the eigenvalues.
      *
      * It is a shortcut for \code this->alphas().cwiseQuotient(this->betas()); \endcode
      * Not that betas might contain zeros. It is therefore not recommended to use this function,
      * but rather directly deal with the alphas and betas vectors.
      *
      * \pre Either the constructor 
      * GeneralizedEigenSolver(const MatrixType&,const MatrixType&,bool) or the member function
      * compute(const MatrixType&,const MatrixType&,bool) has been called before.
      *
      * The eigenvalues are repeated according to their algebraic multiplicity,
      * so there are as many eigenvalues as rows in the matrix. The eigenvalues 
      * are not sorted in any particular order.
      *
      * \sa alphas(), betas(), eigenvectors()
      */
    EigenvalueType eigenvalues() const
    {
      eigen_assert(m_isInitialized && "GeneralizedEigenSolver is not initialized.");
      return EigenvalueType(m_alphas,m_betas);
    }

    /** \returns A const reference to the vectors containing the alpha values
      *
      * This vector permits to reconstruct the j-th eigenvalues as alphas(i)/betas(j).
      *
      * \sa betas(), eigenvalues() */
    ComplexVectorType alphas() const
    {
      eigen_assert(m_isInitialized && "GeneralizedEigenSolver is not initialized.");
      return m_alphas;
    }

    /** \returns A const reference to the vectors containing the beta values
      *
      * This vector permits to reconstruct the j-th eigenvalues as alphas(i)/betas(j).
      *
      * \sa alphas(), eigenvalues() */
    VectorType betas() const
    {
      eigen_assert(m_isInitialized && "GeneralizedEigenSolver is not initialized.");
      return m_betas;
    }

    /** \brief Computes generalized eigendecomposition of given matrix.
      * 
      * \param[in]  A  Square matrix whose eigendecomposition is to be computed.
      * \param[in]  B  Square matrix whose eigendecomposition is to be computed.
      * \param[in]  computeEigenvectors  If true, both the eigenvectors and the
      *    eigenvalues are computed; if false, only the eigenvalues are
      *    computed. 
      * \returns    Reference to \c *this
      *
      * This function computes the eigenvalues of the real matrix \p matrix.
      * The eigenvalues() function can be used to retrieve them.  If 
      * \p computeEigenvectors is true, then the eigenvectors are also computed
      * and can be retrieved by calling eigenvectors().
      *
      * The matrix is first reduced to real generalized Schur form using the RealQZ
      * class. The generalized Schur decomposition is then used to compute the eigenvalues
      * and eigenvectors.
      *
      * The cost of the computation is dominated by the cost of the
      * generalized Schur decomposition.
      *
      * This method reuses of the allocated data in the GeneralizedEigenSolver object.
      */
    GeneralizedEigenSolver& compute(const MatrixType& A, const MatrixType& B, bool computeEigenvectors = true);

    ComputationInfo info() const
    {
      eigen_assert(m_isInitialized && "EigenSolver is not initialized.");
      return m_realQZ.info();
    }

    /** Sets the maximal number of iterations allowed.
    */
    GeneralizedEigenSolver& setMaxIterations(Index maxIters)
    {
      m_realQZ.setMaxIterations(maxIters);
      return *this;
    }

  protected:
    MatrixType m_eivec;
    ComplexVectorType m_alphas;
    VectorType m_betas;
    bool m_isInitialized;
    bool m_eigenvectorsOk;
    RealQZ<MatrixType> m_realQZ;
    MatrixType m_matS;

    typedef Matrix<Scalar, ColsAtCompileTime, 1, Options & ~RowMajor, MaxColsAtCompileTime, 1> ColumnVectorType;
    ColumnVectorType m_tmp;
};

//template<typename MatrixType>
//typename GeneralizedEigenSolver<MatrixType>::EigenvectorsType GeneralizedEigenSolver<MatrixType>::eigenvectors() const
//{
//  eigen_assert(m_isInitialized && "EigenSolver is not initialized.");
//  eigen_assert(m_eigenvectorsOk && "The eigenvectors have not been computed together with the eigenvalues.");
//  Index n = m_eivec.cols();
//  EigenvectorsType matV(n,n);
//  // TODO
//  return matV;
//}

template<typename MatrixType>
GeneralizedEigenSolver<MatrixType>&
GeneralizedEigenSolver<MatrixType>::compute(const MatrixType& A, const MatrixType& B, bool computeEigenvectors)
{
  using std::sqrt;
  using std::abs;
  eigen_assert(A.cols() == A.rows() && B.cols() == A.rows() && B.cols() == B.rows());

  // Reduce to generalized real Schur form:
  // A = Q S Z and B = Q T Z
  m_realQZ.compute(A, B, computeEigenvectors);

  if (m_realQZ.info() == Success)
  {
    m_matS = m_realQZ.matrixS();
    if (computeEigenvectors)
      m_eivec = m_realQZ.matrixZ().transpose();
  
    // Compute eigenvalues from matS
    m_alphas.resize(A.cols());
    m_betas.resize(A.cols());
    Index i = 0;
    while (i < A.cols())
    {
      if (i == A.cols() - 1 || m_matS.coeff(i+1, i) == Scalar(0))
      {
        m_alphas.coeffRef(i) = m_matS.coeff(i, i);
        m_betas.coeffRef(i)  = m_realQZ.matrixT().coeff(i,i);
        ++i;
      }
      else
      {
        Scalar p = Scalar(0.5) * (m_matS.coeff(i, i) - m_matS.coeff(i+1, i+1));
        Scalar z = sqrt(abs(p * p + m_matS.coeff(i+1, i) * m_matS.coeff(i, i+1)));
        m_alphas.coeffRef(i)   = ComplexScalar(m_matS.coeff(i+1, i+1) + p, z);
        m_alphas.coeffRef(i+1) = ComplexScalar(m_matS.coeff(i+1, i+1) + p, -z);

        m_betas.coeffRef(i)   = m_realQZ.matrixT().coeff(i,i);
        m_betas.coeffRef(i+1) = m_realQZ.matrixT().coeff(i,i);
        i += 2;
      }
    }
  }

  m_isInitialized = true;
  m_eigenvectorsOk = false;//computeEigenvectors;

  return *this;
}

} // end namespace Eigen

#endif // EIGEN_GENERALIZEDEIGENSOLVER_H
