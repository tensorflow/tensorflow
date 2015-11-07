// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CHOLMODSUPPORT_H
#define EIGEN_CHOLMODSUPPORT_H

namespace Eigen { 

namespace internal {

template<typename Scalar, typename CholmodType>
void cholmod_configure_matrix(CholmodType& mat)
{
  if (internal::is_same<Scalar,float>::value)
  {
    mat.xtype = CHOLMOD_REAL;
    mat.dtype = CHOLMOD_SINGLE;
  }
  else if (internal::is_same<Scalar,double>::value)
  {
    mat.xtype = CHOLMOD_REAL;
    mat.dtype = CHOLMOD_DOUBLE;
  }
  else if (internal::is_same<Scalar,std::complex<float> >::value)
  {
    mat.xtype = CHOLMOD_COMPLEX;
    mat.dtype = CHOLMOD_SINGLE;
  }
  else if (internal::is_same<Scalar,std::complex<double> >::value)
  {
    mat.xtype = CHOLMOD_COMPLEX;
    mat.dtype = CHOLMOD_DOUBLE;
  }
  else
  {
    eigen_assert(false && "Scalar type not supported by CHOLMOD");
  }
}

} // namespace internal

/** Wraps the Eigen sparse matrix \a mat into a Cholmod sparse matrix object.
  * Note that the data are shared.
  */
template<typename _Scalar, int _Options, typename _Index>
cholmod_sparse viewAsCholmod(SparseMatrix<_Scalar,_Options,_Index>& mat)
{
  cholmod_sparse res;
  res.nzmax   = mat.nonZeros();
  res.nrow    = mat.rows();;
  res.ncol    = mat.cols();
  res.p       = mat.outerIndexPtr();
  res.i       = mat.innerIndexPtr();
  res.x       = mat.valuePtr();
  res.z       = 0;
  res.sorted  = 1;
  if(mat.isCompressed())
  {
    res.packed  = 1;
    res.nz = 0;
  }
  else
  {
    res.packed  = 0;
    res.nz = mat.innerNonZeroPtr();
  }

  res.dtype   = 0;
  res.stype   = -1;
  
  if (internal::is_same<_Index,int>::value)
  {
    res.itype = CHOLMOD_INT;
  }
  else if (internal::is_same<_Index,UF_long>::value)
  {
    res.itype = CHOLMOD_LONG;
  }
  else
  {
    eigen_assert(false && "Index type not supported yet");
  }

  // setup res.xtype
  internal::cholmod_configure_matrix<_Scalar>(res);
  
  res.stype = 0;
  
  return res;
}

template<typename _Scalar, int _Options, typename _Index>
const cholmod_sparse viewAsCholmod(const SparseMatrix<_Scalar,_Options,_Index>& mat)
{
  cholmod_sparse res = viewAsCholmod(mat.const_cast_derived());
  return res;
}

/** Returns a view of the Eigen sparse matrix \a mat as Cholmod sparse matrix.
  * The data are not copied but shared. */
template<typename _Scalar, int _Options, typename _Index, unsigned int UpLo>
cholmod_sparse viewAsCholmod(const SparseSelfAdjointView<SparseMatrix<_Scalar,_Options,_Index>, UpLo>& mat)
{
  cholmod_sparse res = viewAsCholmod(mat.matrix().const_cast_derived());
  
  if(UpLo==Upper) res.stype =  1;
  if(UpLo==Lower) res.stype = -1;

  return res;
}

/** Returns a view of the Eigen \b dense matrix \a mat as Cholmod dense matrix.
  * The data are not copied but shared. */
template<typename Derived>
cholmod_dense viewAsCholmod(MatrixBase<Derived>& mat)
{
  EIGEN_STATIC_ASSERT((internal::traits<Derived>::Flags&RowMajorBit)==0,THIS_METHOD_IS_ONLY_FOR_COLUMN_MAJOR_MATRICES);
  typedef typename Derived::Scalar Scalar;

  cholmod_dense res;
  res.nrow   = mat.rows();
  res.ncol   = mat.cols();
  res.nzmax  = res.nrow * res.ncol;
  res.d      = Derived::IsVectorAtCompileTime ? mat.derived().size() : mat.derived().outerStride();
  res.x      = (void*)(mat.derived().data());
  res.z      = 0;

  internal::cholmod_configure_matrix<Scalar>(res);

  return res;
}

/** Returns a view of the Cholmod sparse matrix \a cm as an Eigen sparse matrix.
  * The data are not copied but shared. */
template<typename Scalar, int Flags, typename Index>
MappedSparseMatrix<Scalar,Flags,Index> viewAsEigen(cholmod_sparse& cm)
{
  return MappedSparseMatrix<Scalar,Flags,Index>
         (cm.nrow, cm.ncol, static_cast<Index*>(cm.p)[cm.ncol],
          static_cast<Index*>(cm.p), static_cast<Index*>(cm.i),static_cast<Scalar*>(cm.x) );
}

enum CholmodMode {
  CholmodAuto, CholmodSimplicialLLt, CholmodSupernodalLLt, CholmodLDLt
};


/** \ingroup CholmodSupport_Module
  * \class CholmodBase
  * \brief The base class for the direct Cholesky factorization of Cholmod
  * \sa class CholmodSupernodalLLT, class CholmodSimplicialLDLT, class CholmodSimplicialLLT
  */
template<typename _MatrixType, int _UpLo, typename Derived>
class CholmodBase : internal::noncopyable
{
  public:
    typedef _MatrixType MatrixType;
    enum { UpLo = _UpLo };
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef MatrixType CholMatrixType;
    typedef typename MatrixType::Index Index;

  public:

    CholmodBase()
      : m_cholmodFactor(0), m_info(Success), m_isInitialized(false)
    {
      m_shiftOffset[0] = m_shiftOffset[1] = RealScalar(0.0);
      cholmod_start(&m_cholmod);
    }

    CholmodBase(const MatrixType& matrix)
      : m_cholmodFactor(0), m_info(Success), m_isInitialized(false)
    {
      m_shiftOffset[0] = m_shiftOffset[1] = RealScalar(0.0);
      cholmod_start(&m_cholmod);
      compute(matrix);
    }

    ~CholmodBase()
    {
      if(m_cholmodFactor)
        cholmod_free_factor(&m_cholmodFactor, &m_cholmod);
      cholmod_finish(&m_cholmod);
    }
    
    inline Index cols() const { return m_cholmodFactor->n; }
    inline Index rows() const { return m_cholmodFactor->n; }
    
    Derived& derived() { return *static_cast<Derived*>(this); }
    const Derived& derived() const { return *static_cast<const Derived*>(this); }
    
    /** \brief Reports whether previous computation was successful.
      *
      * \returns \c Success if computation was succesful,
      *          \c NumericalIssue if the matrix.appears to be negative.
      */
    ComputationInfo info() const
    {
      eigen_assert(m_isInitialized && "Decomposition is not initialized.");
      return m_info;
    }

    /** Computes the sparse Cholesky decomposition of \a matrix */
    Derived& compute(const MatrixType& matrix)
    {
      analyzePattern(matrix);
      factorize(matrix);
      return derived();
    }
    
    /** \returns the solution x of \f$ A x = b \f$ using the current decomposition of A.
      *
      * \sa compute()
      */
    template<typename Rhs>
    inline const internal::solve_retval<CholmodBase, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "LLT is not initialized.");
      eigen_assert(rows()==b.rows()
                && "CholmodDecomposition::solve(): invalid number of rows of the right hand side matrix b");
      return internal::solve_retval<CholmodBase, Rhs>(*this, b.derived());
    }
    
    /** \returns the solution x of \f$ A x = b \f$ using the current decomposition of A.
      *
      * \sa compute()
      */
    template<typename Rhs>
    inline const internal::sparse_solve_retval<CholmodBase, Rhs>
    solve(const SparseMatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "LLT is not initialized.");
      eigen_assert(rows()==b.rows()
                && "CholmodDecomposition::solve(): invalid number of rows of the right hand side matrix b");
      return internal::sparse_solve_retval<CholmodBase, Rhs>(*this, b.derived());
    }
    
    /** Performs a symbolic decomposition on the sparsity pattern of \a matrix.
      *
      * This function is particularly useful when solving for several problems having the same structure.
      * 
      * \sa factorize()
      */
    void analyzePattern(const MatrixType& matrix)
    {
      if(m_cholmodFactor)
      {
        cholmod_free_factor(&m_cholmodFactor, &m_cholmod);
        m_cholmodFactor = 0;
      }
      cholmod_sparse A = viewAsCholmod(matrix.template selfadjointView<UpLo>());
      m_cholmodFactor = cholmod_analyze(&A, &m_cholmod);
      
      this->m_isInitialized = true;
      this->m_info = Success;
      m_analysisIsOk = true;
      m_factorizationIsOk = false;
    }
    
    /** Performs a numeric decomposition of \a matrix
      *
      * The given matrix must have the same sparsity pattern as the matrix on which the symbolic decomposition has been performed.
      *
      * \sa analyzePattern()
      */
    void factorize(const MatrixType& matrix)
    {
      eigen_assert(m_analysisIsOk && "You must first call analyzePattern()");
      cholmod_sparse A = viewAsCholmod(matrix.template selfadjointView<UpLo>());
      cholmod_factorize_p(&A, m_shiftOffset, 0, 0, m_cholmodFactor, &m_cholmod);
      
      // If the factorization failed, minor is the column at which it did. On success minor == n.
      this->m_info = (m_cholmodFactor->minor == m_cholmodFactor->n ? Success : NumericalIssue);
      m_factorizationIsOk = true;
    }
    
    /** Returns a reference to the Cholmod's configuration structure to get a full control over the performed operations.
     *  See the Cholmod user guide for details. */
    cholmod_common& cholmod() { return m_cholmod; }
    
    #ifndef EIGEN_PARSED_BY_DOXYGEN
    /** \internal */
    template<typename Rhs,typename Dest>
    void _solve(const MatrixBase<Rhs> &b, MatrixBase<Dest> &dest) const
    {
      eigen_assert(m_factorizationIsOk && "The decomposition is not in a valid state for solving, you must first call either compute() or symbolic()/numeric()");
      const Index size = m_cholmodFactor->n;
      EIGEN_UNUSED_VARIABLE(size);
      eigen_assert(size==b.rows());

      // note: cd stands for Cholmod Dense
      Rhs& b_ref(b.const_cast_derived());
      cholmod_dense b_cd = viewAsCholmod(b_ref);
      cholmod_dense* x_cd = cholmod_solve(CHOLMOD_A, m_cholmodFactor, &b_cd, &m_cholmod);
      if(!x_cd)
      {
        this->m_info = NumericalIssue;
      }
      // TODO optimize this copy by swapping when possible (be careful with alignment, etc.)
      dest = Matrix<Scalar,Dest::RowsAtCompileTime,Dest::ColsAtCompileTime>::Map(reinterpret_cast<Scalar*>(x_cd->x),b.rows(),b.cols());
      cholmod_free_dense(&x_cd, &m_cholmod);
    }
    
    /** \internal */
    template<typename RhsScalar, int RhsOptions, typename RhsIndex, typename DestScalar, int DestOptions, typename DestIndex>
    void _solve(const SparseMatrix<RhsScalar,RhsOptions,RhsIndex> &b, SparseMatrix<DestScalar,DestOptions,DestIndex> &dest) const
    {
      eigen_assert(m_factorizationIsOk && "The decomposition is not in a valid state for solving, you must first call either compute() or symbolic()/numeric()");
      const Index size = m_cholmodFactor->n;
      EIGEN_UNUSED_VARIABLE(size);
      eigen_assert(size==b.rows());

      // note: cs stands for Cholmod Sparse
      cholmod_sparse b_cs = viewAsCholmod(b);
      cholmod_sparse* x_cs = cholmod_spsolve(CHOLMOD_A, m_cholmodFactor, &b_cs, &m_cholmod);
      if(!x_cs)
      {
        this->m_info = NumericalIssue;
      }
      // TODO optimize this copy by swapping when possible (be careful with alignment, etc.)
      dest = viewAsEigen<DestScalar,DestOptions,DestIndex>(*x_cs);
      cholmod_free_sparse(&x_cs, &m_cholmod);
    }
    #endif // EIGEN_PARSED_BY_DOXYGEN
    
    
    /** Sets the shift parameter that will be used to adjust the diagonal coefficients during the numerical factorization.
      *
      * During the numerical factorization, an offset term is added to the diagonal coefficients:\n
      * \c d_ii = \a offset + \c d_ii
      *
      * The default is \a offset=0.
      *
      * \returns a reference to \c *this.
      */
    Derived& setShift(const RealScalar& offset)
    {
      m_shiftOffset[0] = offset;
      return derived();
    }
    
    template<typename Stream>
    void dumpMemory(Stream& /*s*/)
    {}
    
  protected:
    mutable cholmod_common m_cholmod;
    cholmod_factor* m_cholmodFactor;
    RealScalar m_shiftOffset[2];
    mutable ComputationInfo m_info;
    bool m_isInitialized;
    int m_factorizationIsOk;
    int m_analysisIsOk;
};

/** \ingroup CholmodSupport_Module
  * \class CholmodSimplicialLLT
  * \brief A simplicial direct Cholesky (LLT) factorization and solver based on Cholmod
  *
  * This class allows to solve for A.X = B sparse linear problems via a simplicial LL^T Cholesky factorization
  * using the Cholmod library.
  * This simplicial variant is equivalent to Eigen's built-in SimplicialLLT class. Therefore, it has little practical interest.
  * The sparse matrix A must be selfadjoint and positive definite. The vectors or matrices
  * X and B can be either dense or sparse.
  *
  * \tparam _MatrixType the type of the sparse matrix A, it must be a SparseMatrix<>
  * \tparam _UpLo the triangular part that will be used for the computations. It can be Lower
  *               or Upper. Default is Lower.
  *
  * This class supports all kind of SparseMatrix<>: row or column major; upper, lower, or both; compressed or non compressed.
  *
  * \sa \ref TutorialSparseDirectSolvers, class CholmodSupernodalLLT, class SimplicialLLT
  */
template<typename _MatrixType, int _UpLo = Lower>
class CholmodSimplicialLLT : public CholmodBase<_MatrixType, _UpLo, CholmodSimplicialLLT<_MatrixType, _UpLo> >
{
    typedef CholmodBase<_MatrixType, _UpLo, CholmodSimplicialLLT> Base;
    using Base::m_cholmod;
    
  public:
    
    typedef _MatrixType MatrixType;
    
    CholmodSimplicialLLT() : Base() { init(); }

    CholmodSimplicialLLT(const MatrixType& matrix) : Base()
    {
      init();
      compute(matrix);
    }

    ~CholmodSimplicialLLT() {}
  protected:
    void init()
    {
      m_cholmod.final_asis = 0;
      m_cholmod.supernodal = CHOLMOD_SIMPLICIAL;
      m_cholmod.final_ll = 1;
    }
};


/** \ingroup CholmodSupport_Module
  * \class CholmodSimplicialLDLT
  * \brief A simplicial direct Cholesky (LDLT) factorization and solver based on Cholmod
  *
  * This class allows to solve for A.X = B sparse linear problems via a simplicial LDL^T Cholesky factorization
  * using the Cholmod library.
  * This simplicial variant is equivalent to Eigen's built-in SimplicialLDLT class. Therefore, it has little practical interest.
  * The sparse matrix A must be selfadjoint and positive definite. The vectors or matrices
  * X and B can be either dense or sparse.
  *
  * \tparam _MatrixType the type of the sparse matrix A, it must be a SparseMatrix<>
  * \tparam _UpLo the triangular part that will be used for the computations. It can be Lower
  *               or Upper. Default is Lower.
  *
  * This class supports all kind of SparseMatrix<>: row or column major; upper, lower, or both; compressed or non compressed.
  *
  * \sa \ref TutorialSparseDirectSolvers, class CholmodSupernodalLLT, class SimplicialLDLT
  */
template<typename _MatrixType, int _UpLo = Lower>
class CholmodSimplicialLDLT : public CholmodBase<_MatrixType, _UpLo, CholmodSimplicialLDLT<_MatrixType, _UpLo> >
{
    typedef CholmodBase<_MatrixType, _UpLo, CholmodSimplicialLDLT> Base;
    using Base::m_cholmod;
    
  public:
    
    typedef _MatrixType MatrixType;
    
    CholmodSimplicialLDLT() : Base() { init(); }

    CholmodSimplicialLDLT(const MatrixType& matrix) : Base()
    {
      init();
      compute(matrix);
    }

    ~CholmodSimplicialLDLT() {}
  protected:
    void init()
    {
      m_cholmod.final_asis = 1;
      m_cholmod.supernodal = CHOLMOD_SIMPLICIAL;
    }
};

/** \ingroup CholmodSupport_Module
  * \class CholmodSupernodalLLT
  * \brief A supernodal Cholesky (LLT) factorization and solver based on Cholmod
  *
  * This class allows to solve for A.X = B sparse linear problems via a supernodal LL^T Cholesky factorization
  * using the Cholmod library.
  * This supernodal variant performs best on dense enough problems, e.g., 3D FEM, or very high order 2D FEM.
  * The sparse matrix A must be selfadjoint and positive definite. The vectors or matrices
  * X and B can be either dense or sparse.
  *
  * \tparam _MatrixType the type of the sparse matrix A, it must be a SparseMatrix<>
  * \tparam _UpLo the triangular part that will be used for the computations. It can be Lower
  *               or Upper. Default is Lower.
  *
  * This class supports all kind of SparseMatrix<>: row or column major; upper, lower, or both; compressed or non compressed.
  *
  * \sa \ref TutorialSparseDirectSolvers
  */
template<typename _MatrixType, int _UpLo = Lower>
class CholmodSupernodalLLT : public CholmodBase<_MatrixType, _UpLo, CholmodSupernodalLLT<_MatrixType, _UpLo> >
{
    typedef CholmodBase<_MatrixType, _UpLo, CholmodSupernodalLLT> Base;
    using Base::m_cholmod;
    
  public:
    
    typedef _MatrixType MatrixType;
    
    CholmodSupernodalLLT() : Base() { init(); }

    CholmodSupernodalLLT(const MatrixType& matrix) : Base()
    {
      init();
      compute(matrix);
    }

    ~CholmodSupernodalLLT() {}
  protected:
    void init()
    {
      m_cholmod.final_asis = 1;
      m_cholmod.supernodal = CHOLMOD_SUPERNODAL;
    }
};

/** \ingroup CholmodSupport_Module
  * \class CholmodDecomposition
  * \brief A general Cholesky factorization and solver based on Cholmod
  *
  * This class allows to solve for A.X = B sparse linear problems via a LL^T or LDL^T Cholesky factorization
  * using the Cholmod library. The sparse matrix A must be selfadjoint and positive definite. The vectors or matrices
  * X and B can be either dense or sparse.
  *
  * This variant permits to change the underlying Cholesky method at runtime.
  * On the other hand, it does not provide access to the result of the factorization.
  * The default is to let Cholmod automatically choose between a simplicial and supernodal factorization.
  *
  * \tparam _MatrixType the type of the sparse matrix A, it must be a SparseMatrix<>
  * \tparam _UpLo the triangular part that will be used for the computations. It can be Lower
  *               or Upper. Default is Lower.
  *
  * This class supports all kind of SparseMatrix<>: row or column major; upper, lower, or both; compressed or non compressed.
  *
  * \sa \ref TutorialSparseDirectSolvers
  */
template<typename _MatrixType, int _UpLo = Lower>
class CholmodDecomposition : public CholmodBase<_MatrixType, _UpLo, CholmodDecomposition<_MatrixType, _UpLo> >
{
    typedef CholmodBase<_MatrixType, _UpLo, CholmodDecomposition> Base;
    using Base::m_cholmod;
    
  public:
    
    typedef _MatrixType MatrixType;
    
    CholmodDecomposition() : Base() { init(); }

    CholmodDecomposition(const MatrixType& matrix) : Base()
    {
      init();
      compute(matrix);
    }

    ~CholmodDecomposition() {}
    
    void setMode(CholmodMode mode)
    {
      switch(mode)
      {
        case CholmodAuto:
          m_cholmod.final_asis = 1;
          m_cholmod.supernodal = CHOLMOD_AUTO;
          break;
        case CholmodSimplicialLLt:
          m_cholmod.final_asis = 0;
          m_cholmod.supernodal = CHOLMOD_SIMPLICIAL;
          m_cholmod.final_ll = 1;
          break;
        case CholmodSupernodalLLt:
          m_cholmod.final_asis = 1;
          m_cholmod.supernodal = CHOLMOD_SUPERNODAL;
          break;
        case CholmodLDLt:
          m_cholmod.final_asis = 1;
          m_cholmod.supernodal = CHOLMOD_SIMPLICIAL;
          break;
        default:
          break;
      }
    }
  protected:
    void init()
    {
      m_cholmod.final_asis = 1;
      m_cholmod.supernodal = CHOLMOD_AUTO;
    }
};

namespace internal {
  
template<typename _MatrixType, int _UpLo, typename Derived, typename Rhs>
struct solve_retval<CholmodBase<_MatrixType,_UpLo,Derived>, Rhs>
  : solve_retval_base<CholmodBase<_MatrixType,_UpLo,Derived>, Rhs>
{
  typedef CholmodBase<_MatrixType,_UpLo,Derived> Dec;
  EIGEN_MAKE_SOLVE_HELPERS(Dec,Rhs)

  template<typename Dest> void evalTo(Dest& dst) const
  {
    dec()._solve(rhs(),dst);
  }
};

template<typename _MatrixType, int _UpLo, typename Derived, typename Rhs>
struct sparse_solve_retval<CholmodBase<_MatrixType,_UpLo,Derived>, Rhs>
  : sparse_solve_retval_base<CholmodBase<_MatrixType,_UpLo,Derived>, Rhs>
{
  typedef CholmodBase<_MatrixType,_UpLo,Derived> Dec;
  EIGEN_MAKE_SPARSE_SOLVE_HELPERS(Dec,Rhs)

  template<typename Dest> void evalTo(Dest& dst) const
  {
    dec()._solve(rhs(),dst);
  }
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_CHOLMODSUPPORT_H
