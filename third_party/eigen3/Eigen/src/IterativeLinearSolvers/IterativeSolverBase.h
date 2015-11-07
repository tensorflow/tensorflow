// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_ITERATIVE_SOLVER_BASE_H
#define EIGEN_ITERATIVE_SOLVER_BASE_H

namespace Eigen { 

/** \ingroup IterativeLinearSolvers_Module
  * \brief Base class for linear iterative solvers
  *
  * \sa class SimplicialCholesky, DiagonalPreconditioner, IdentityPreconditioner
  */
template< typename Derived>
class IterativeSolverBase : internal::noncopyable
{
public:
  typedef typename internal::traits<Derived>::MatrixType MatrixType;
  typedef typename internal::traits<Derived>::Preconditioner Preconditioner;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::Index Index;
  typedef typename MatrixType::RealScalar RealScalar;

public:

  Derived& derived() { return *static_cast<Derived*>(this); }
  const Derived& derived() const { return *static_cast<const Derived*>(this); }

  /** Default constructor. */
  IterativeSolverBase()
    : mp_matrix(0)
  {
    init();
  }

  /** Initialize the solver with matrix \a A for further \c Ax=b solving.
    * 
    * This constructor is a shortcut for the default constructor followed
    * by a call to compute().
    * 
    * \warning this class stores a reference to the matrix A as well as some
    * precomputed values that depend on it. Therefore, if \a A is changed
    * this class becomes invalid. Call compute() to update it with the new
    * matrix A, or modify a copy of A.
    */
  IterativeSolverBase(const MatrixType& A)
  {
    init();
    compute(A);
  }

  ~IterativeSolverBase() {}
  
  /** Initializes the iterative solver for the sparcity pattern of the matrix \a A for further solving \c Ax=b problems.
    *
    * Currently, this function mostly call analyzePattern on the preconditioner. In the future
    * we might, for instance, implement column reodering for faster matrix vector products.
    */
  Derived& analyzePattern(const MatrixType& A)
  {
    m_preconditioner.analyzePattern(A);
    m_isInitialized = true;
    m_analysisIsOk = true;
    m_info = Success;
    return derived();
  }
  
  /** Initializes the iterative solver with the numerical values of the matrix \a A for further solving \c Ax=b problems.
    *
    * Currently, this function mostly call factorize on the preconditioner.
    *
    * \warning this class stores a reference to the matrix A as well as some
    * precomputed values that depend on it. Therefore, if \a A is changed
    * this class becomes invalid. Call compute() to update it with the new
    * matrix A, or modify a copy of A.
    */
  Derived& factorize(const MatrixType& A)
  {
    eigen_assert(m_analysisIsOk && "You must first call analyzePattern()"); 
    mp_matrix = &A;
    m_preconditioner.factorize(A);
    m_factorizationIsOk = true;
    m_info = Success;
    return derived();
  }

  /** Initializes the iterative solver with the matrix \a A for further solving \c Ax=b problems.
    *
    * Currently, this function mostly initialized/compute the preconditioner. In the future
    * we might, for instance, implement column reodering for faster matrix vector products.
    *
    * \warning this class stores a reference to the matrix A as well as some
    * precomputed values that depend on it. Therefore, if \a A is changed
    * this class becomes invalid. Call compute() to update it with the new
    * matrix A, or modify a copy of A.
    */
  Derived& compute(const MatrixType& A)
  {
    mp_matrix = &A;
    m_preconditioner.compute(A);
    m_isInitialized = true;
    m_analysisIsOk = true;
    m_factorizationIsOk = true;
    m_info = Success;
    return derived();
  }

  /** \internal */
  Index rows() const { return mp_matrix ? mp_matrix->rows() : 0; }
  /** \internal */
  Index cols() const { return mp_matrix ? mp_matrix->cols() : 0; }

  /** \returns the tolerance threshold used by the stopping criteria */
  RealScalar tolerance() const { return m_tolerance; }
  
  /** Sets the tolerance threshold used by the stopping criteria */
  Derived& setTolerance(const RealScalar& tolerance)
  {
    m_tolerance = tolerance;
    return derived();
  }

  /** \returns a read-write reference to the preconditioner for custom configuration. */
  Preconditioner& preconditioner() { return m_preconditioner; }
  
  /** \returns a read-only reference to the preconditioner. */
  const Preconditioner& preconditioner() const { return m_preconditioner; }

  /** \returns the max number of iterations */
  int maxIterations() const
  {
    return (mp_matrix && m_maxIterations<0) ? mp_matrix->cols() : m_maxIterations;
  }
  
  /** Sets the max number of iterations */
  Derived& setMaxIterations(int maxIters)
  {
    m_maxIterations = maxIters;
    return derived();
  }

  /** \returns the number of iterations performed during the last solve */
  int iterations() const
  {
    eigen_assert(m_isInitialized && "ConjugateGradient is not initialized.");
    return m_iterations;
  }

  /** \returns the tolerance error reached during the last solve */
  RealScalar error() const
  {
    eigen_assert(m_isInitialized && "ConjugateGradient is not initialized.");
    return m_error;
  }

  /** \returns the solution x of \f$ A x = b \f$ using the current decomposition of A.
    *
    * \sa compute()
    */
  template<typename Rhs> inline const internal::solve_retval<Derived, Rhs>
  solve(const MatrixBase<Rhs>& b) const
  {
    eigen_assert(m_isInitialized && "IterativeSolverBase is not initialized.");
    eigen_assert(rows()==b.rows()
              && "IterativeSolverBase::solve(): invalid number of rows of the right hand side matrix b");
    return internal::solve_retval<Derived, Rhs>(derived(), b.derived());
  }
  
  /** \returns the solution x of \f$ A x = b \f$ using the current decomposition of A.
    *
    * \sa compute()
    */
  template<typename Rhs>
  inline const internal::sparse_solve_retval<IterativeSolverBase, Rhs>
  solve(const SparseMatrixBase<Rhs>& b) const
  {
    eigen_assert(m_isInitialized && "IterativeSolverBase is not initialized.");
    eigen_assert(rows()==b.rows()
              && "IterativeSolverBase::solve(): invalid number of rows of the right hand side matrix b");
    return internal::sparse_solve_retval<IterativeSolverBase, Rhs>(*this, b.derived());
  }

  /** \returns Success if the iterations converged, and NoConvergence otherwise. */
  ComputationInfo info() const
  {
    eigen_assert(m_isInitialized && "IterativeSolverBase is not initialized.");
    return m_info;
  }
  
  /** \internal */
  template<typename Rhs, typename DestScalar, int DestOptions, typename DestIndex>
  void _solve_sparse(const Rhs& b, SparseMatrix<DestScalar,DestOptions,DestIndex> &dest) const
  {
    eigen_assert(rows()==b.rows());
    
    int rhsCols = b.cols();
    int size = b.rows();
    Eigen::Matrix<DestScalar,Dynamic,1> tb(size);
    Eigen::Matrix<DestScalar,Dynamic,1> tx(size);
    for(int k=0; k<rhsCols; ++k)
    {
      tb = b.col(k);
      tx = derived().solve(tb);
      dest.col(k) = tx.sparseView(0);
    }
  }

protected:
  void init()
  {
    m_isInitialized = false;
    m_analysisIsOk = false;
    m_factorizationIsOk = false;
    m_maxIterations = -1;
    m_tolerance = NumTraits<Scalar>::epsilon();
  }
  const MatrixType* mp_matrix;
  Preconditioner m_preconditioner;

  int m_maxIterations;
  RealScalar m_tolerance;
  
  mutable RealScalar m_error;
  mutable int m_iterations;
  mutable ComputationInfo m_info;
  mutable bool m_isInitialized, m_analysisIsOk, m_factorizationIsOk;
};

namespace internal {
 
template<typename Derived, typename Rhs>
struct sparse_solve_retval<IterativeSolverBase<Derived>, Rhs>
  : sparse_solve_retval_base<IterativeSolverBase<Derived>, Rhs>
{
  typedef IterativeSolverBase<Derived> Dec;
  EIGEN_MAKE_SPARSE_SOLVE_HELPERS(Dec,Rhs)

  template<typename Dest> void evalTo(Dest& dst) const
  {
    dec().derived()._solve_sparse(rhs(),dst);
  }
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_ITERATIVE_SOLVER_BASE_H
