// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_UMFPACKSUPPORT_H
#define EIGEN_UMFPACKSUPPORT_H

namespace Eigen { 

/* TODO extract L, extract U, compute det, etc... */

// generic double/complex<double> wrapper functions:

inline void umfpack_free_numeric(void **Numeric, double)
{ umfpack_di_free_numeric(Numeric); *Numeric = 0; }

inline void umfpack_free_numeric(void **Numeric, std::complex<double>)
{ umfpack_zi_free_numeric(Numeric); *Numeric = 0; }

inline void umfpack_free_symbolic(void **Symbolic, double)
{ umfpack_di_free_symbolic(Symbolic); *Symbolic = 0; }

inline void umfpack_free_symbolic(void **Symbolic, std::complex<double>)
{ umfpack_zi_free_symbolic(Symbolic); *Symbolic = 0; }

inline int umfpack_symbolic(int n_row,int n_col,
                            const int Ap[], const int Ai[], const double Ax[], void **Symbolic,
                            const double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
  return umfpack_di_symbolic(n_row,n_col,Ap,Ai,Ax,Symbolic,Control,Info);
}

inline int umfpack_symbolic(int n_row,int n_col,
                            const int Ap[], const int Ai[], const std::complex<double> Ax[], void **Symbolic,
                            const double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
  return umfpack_zi_symbolic(n_row,n_col,Ap,Ai,&numext::real_ref(Ax[0]),0,Symbolic,Control,Info);
}

inline int umfpack_numeric( const int Ap[], const int Ai[], const double Ax[],
                            void *Symbolic, void **Numeric,
                            const double Control[UMFPACK_CONTROL],double Info [UMFPACK_INFO])
{
  return umfpack_di_numeric(Ap,Ai,Ax,Symbolic,Numeric,Control,Info);
}

inline int umfpack_numeric( const int Ap[], const int Ai[], const std::complex<double> Ax[],
                            void *Symbolic, void **Numeric,
                            const double Control[UMFPACK_CONTROL],double Info [UMFPACK_INFO])
{
  return umfpack_zi_numeric(Ap,Ai,&numext::real_ref(Ax[0]),0,Symbolic,Numeric,Control,Info);
}

inline int umfpack_solve( int sys, const int Ap[], const int Ai[], const double Ax[],
                          double X[], const double B[], void *Numeric,
                          const double Control[UMFPACK_CONTROL], double Info[UMFPACK_INFO])
{
  return umfpack_di_solve(sys,Ap,Ai,Ax,X,B,Numeric,Control,Info);
}

inline int umfpack_solve( int sys, const int Ap[], const int Ai[], const std::complex<double> Ax[],
                          std::complex<double> X[], const std::complex<double> B[], void *Numeric,
                          const double Control[UMFPACK_CONTROL], double Info[UMFPACK_INFO])
{
  return umfpack_zi_solve(sys,Ap,Ai,&numext::real_ref(Ax[0]),0,&numext::real_ref(X[0]),0,&numext::real_ref(B[0]),0,Numeric,Control,Info);
}

inline int umfpack_get_lunz(int *lnz, int *unz, int *n_row, int *n_col, int *nz_udiag, void *Numeric, double)
{
  return umfpack_di_get_lunz(lnz,unz,n_row,n_col,nz_udiag,Numeric);
}

inline int umfpack_get_lunz(int *lnz, int *unz, int *n_row, int *n_col, int *nz_udiag, void *Numeric, std::complex<double>)
{
  return umfpack_zi_get_lunz(lnz,unz,n_row,n_col,nz_udiag,Numeric);
}

inline int umfpack_get_numeric(int Lp[], int Lj[], double Lx[], int Up[], int Ui[], double Ux[],
                               int P[], int Q[], double Dx[], int *do_recip, double Rs[], void *Numeric)
{
  return umfpack_di_get_numeric(Lp,Lj,Lx,Up,Ui,Ux,P,Q,Dx,do_recip,Rs,Numeric);
}

inline int umfpack_get_numeric(int Lp[], int Lj[], std::complex<double> Lx[], int Up[], int Ui[], std::complex<double> Ux[],
                               int P[], int Q[], std::complex<double> Dx[], int *do_recip, double Rs[], void *Numeric)
{
  double& lx0_real = numext::real_ref(Lx[0]);
  double& ux0_real = numext::real_ref(Ux[0]);
  double& dx0_real = numext::real_ref(Dx[0]);
  return umfpack_zi_get_numeric(Lp,Lj,Lx?&lx0_real:0,0,Up,Ui,Ux?&ux0_real:0,0,P,Q,
                                Dx?&dx0_real:0,0,do_recip,Rs,Numeric);
}

inline int umfpack_get_determinant(double *Mx, double *Ex, void *NumericHandle, double User_Info [UMFPACK_INFO])
{
  return umfpack_di_get_determinant(Mx,Ex,NumericHandle,User_Info);
}

inline int umfpack_get_determinant(std::complex<double> *Mx, double *Ex, void *NumericHandle, double User_Info [UMFPACK_INFO])
{
  double& mx_real = numext::real_ref(*Mx);
  return umfpack_zi_get_determinant(&mx_real,0,Ex,NumericHandle,User_Info);
}

/** \ingroup UmfPackSupport_Module
  * \brief A sparse LU factorization and solver based on UmfPack
  *
  * This class allows to solve for A.X = B sparse linear problems via a LU factorization
  * using the UmfPack library. The sparse matrix A must be squared and full rank.
  * The vectors or matrices X and B can be either dense or sparse.
  *
  * \warning The input matrix A should be in a \b compressed and \b column-major form.
  * Otherwise an expensive copy will be made. You can call the inexpensive makeCompressed() to get a compressed matrix.
  * \tparam _MatrixType the type of the sparse matrix A, it must be a SparseMatrix<>
  *
  * \sa \ref TutorialSparseDirectSolvers
  */
template<typename _MatrixType>
class UmfPackLU : internal::noncopyable
{
  public:
    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::Index Index;
    typedef Matrix<Scalar,Dynamic,1> Vector;
    typedef Matrix<int, 1, MatrixType::ColsAtCompileTime> IntRowVectorType;
    typedef Matrix<int, MatrixType::RowsAtCompileTime, 1> IntColVectorType;
    typedef SparseMatrix<Scalar> LUMatrixType;
    typedef SparseMatrix<Scalar,ColMajor,int> UmfpackMatrixType;

  public:

    UmfPackLU() { init(); }

    UmfPackLU(const MatrixType& matrix)
    {
      init();
      compute(matrix);
    }

    ~UmfPackLU()
    {
      if(m_symbolic) umfpack_free_symbolic(&m_symbolic,Scalar());
      if(m_numeric)  umfpack_free_numeric(&m_numeric,Scalar());
    }

    inline Index rows() const { return m_copyMatrix.rows(); }
    inline Index cols() const { return m_copyMatrix.cols(); }

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

    inline const LUMatrixType& matrixL() const
    {
      if (m_extractedDataAreDirty) extractData();
      return m_l;
    }

    inline const LUMatrixType& matrixU() const
    {
      if (m_extractedDataAreDirty) extractData();
      return m_u;
    }

    inline const IntColVectorType& permutationP() const
    {
      if (m_extractedDataAreDirty) extractData();
      return m_p;
    }

    inline const IntRowVectorType& permutationQ() const
    {
      if (m_extractedDataAreDirty) extractData();
      return m_q;
    }

    /** Computes the sparse Cholesky decomposition of \a matrix 
     *  Note that the matrix should be column-major, and in compressed format for best performance.
     *  \sa SparseMatrix::makeCompressed().
     */
    void compute(const MatrixType& matrix)
    {
      analyzePattern(matrix);
      factorize(matrix);
    }

    /** \returns the solution x of \f$ A x = b \f$ using the current decomposition of A.
      *
      * \sa compute()
      */
    template<typename Rhs>
    inline const internal::solve_retval<UmfPackLU, Rhs> solve(const MatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "UmfPackLU is not initialized.");
      eigen_assert(rows()==b.rows()
                && "UmfPackLU::solve(): invalid number of rows of the right hand side matrix b");
      return internal::solve_retval<UmfPackLU, Rhs>(*this, b.derived());
    }

    /** \returns the solution x of \f$ A x = b \f$ using the current decomposition of A.
      *
      * \sa compute()
      */
    template<typename Rhs>
    inline const internal::sparse_solve_retval<UmfPackLU, Rhs> solve(const SparseMatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "UmfPackLU is not initialized.");
      eigen_assert(rows()==b.rows()
                && "UmfPackLU::solve(): invalid number of rows of the right hand side matrix b");
      return internal::sparse_solve_retval<UmfPackLU, Rhs>(*this, b.derived());
    }

    /** Performs a symbolic decomposition on the sparcity of \a matrix.
      *
      * This function is particularly useful when solving for several problems having the same structure.
      *
      * \sa factorize(), compute()
      */
    void analyzePattern(const MatrixType& matrix)
    {
      if(m_symbolic)
        umfpack_free_symbolic(&m_symbolic,Scalar());
      if(m_numeric)
        umfpack_free_numeric(&m_numeric,Scalar());
      
      grapInput(matrix);

      int errorCode = 0;
      errorCode = umfpack_symbolic(matrix.rows(), matrix.cols(), m_outerIndexPtr, m_innerIndexPtr, m_valuePtr,
                                   &m_symbolic, 0, 0);

      m_isInitialized = true;
      m_info = errorCode ? InvalidInput : Success;
      m_analysisIsOk = true;
      m_factorizationIsOk = false;
    }

    /** Performs a numeric decomposition of \a matrix
      *
      * The given matrix must has the same sparcity than the matrix on which the pattern anylysis has been performed.
      *
      * \sa analyzePattern(), compute()
      */
    void factorize(const MatrixType& matrix)
    {
      eigen_assert(m_analysisIsOk && "UmfPackLU: you must first call analyzePattern()");
      if(m_numeric)
        umfpack_free_numeric(&m_numeric,Scalar());

      grapInput(matrix);

      int errorCode;
      errorCode = umfpack_numeric(m_outerIndexPtr, m_innerIndexPtr, m_valuePtr,
                                  m_symbolic, &m_numeric, 0, 0);

      m_info = errorCode ? NumericalIssue : Success;
      m_factorizationIsOk = true;
    }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    /** \internal */
    template<typename BDerived,typename XDerived>
    bool _solve(const MatrixBase<BDerived> &b, MatrixBase<XDerived> &x) const;
    #endif

    Scalar determinant() const;

    void extractData() const;

  protected:


    void init()
    {
      m_info = InvalidInput;
      m_isInitialized = false;
      m_numeric = 0;
      m_symbolic = 0;
      m_outerIndexPtr = 0;
      m_innerIndexPtr = 0;
      m_valuePtr      = 0;
    }
    
    void grapInput(const MatrixType& mat)
    {
      m_copyMatrix.resize(mat.rows(), mat.cols());
      if( ((MatrixType::Flags&RowMajorBit)==RowMajorBit) || sizeof(typename MatrixType::Index)!=sizeof(int) || !mat.isCompressed() )
      {
        // non supported input -> copy
        m_copyMatrix = mat;
        m_outerIndexPtr = m_copyMatrix.outerIndexPtr();
        m_innerIndexPtr = m_copyMatrix.innerIndexPtr();
        m_valuePtr      = m_copyMatrix.valuePtr();
      }
      else
      {
        m_outerIndexPtr = mat.outerIndexPtr();
        m_innerIndexPtr = mat.innerIndexPtr();
        m_valuePtr      = mat.valuePtr();
      }
    }

    // cached data to reduce reallocation, etc.
    mutable LUMatrixType m_l;
    mutable LUMatrixType m_u;
    mutable IntColVectorType m_p;
    mutable IntRowVectorType m_q;

    UmfpackMatrixType m_copyMatrix;
    const Scalar* m_valuePtr;
    const int* m_outerIndexPtr;
    const int* m_innerIndexPtr;
    void* m_numeric;
    void* m_symbolic;

    mutable ComputationInfo m_info;
    bool m_isInitialized;
    int m_factorizationIsOk;
    int m_analysisIsOk;
    mutable bool m_extractedDataAreDirty;
    
  private:
    UmfPackLU(UmfPackLU& ) { }
};


template<typename MatrixType>
void UmfPackLU<MatrixType>::extractData() const
{
  if (m_extractedDataAreDirty)
  {
    // get size of the data
    int lnz, unz, rows, cols, nz_udiag;
    umfpack_get_lunz(&lnz, &unz, &rows, &cols, &nz_udiag, m_numeric, Scalar());

    // allocate data
    m_l.resize(rows,(std::min)(rows,cols));
    m_l.resizeNonZeros(lnz);

    m_u.resize((std::min)(rows,cols),cols);
    m_u.resizeNonZeros(unz);

    m_p.resize(rows);
    m_q.resize(cols);

    // extract
    umfpack_get_numeric(m_l.outerIndexPtr(), m_l.innerIndexPtr(), m_l.valuePtr(),
                        m_u.outerIndexPtr(), m_u.innerIndexPtr(), m_u.valuePtr(),
                        m_p.data(), m_q.data(), 0, 0, 0, m_numeric);

    m_extractedDataAreDirty = false;
  }
}

template<typename MatrixType>
typename UmfPackLU<MatrixType>::Scalar UmfPackLU<MatrixType>::determinant() const
{
  Scalar det;
  umfpack_get_determinant(&det, 0, m_numeric, 0);
  return det;
}

template<typename MatrixType>
template<typename BDerived,typename XDerived>
bool UmfPackLU<MatrixType>::_solve(const MatrixBase<BDerived> &b, MatrixBase<XDerived> &x) const
{
  const int rhsCols = b.cols();
  eigen_assert((BDerived::Flags&RowMajorBit)==0 && "UmfPackLU backend does not support non col-major rhs yet");
  eigen_assert((XDerived::Flags&RowMajorBit)==0 && "UmfPackLU backend does not support non col-major result yet");
  eigen_assert(b.derived().data() != x.derived().data() && " Umfpack does not support inplace solve");
  
  int errorCode;
  for (int j=0; j<rhsCols; ++j)
  {
    errorCode = umfpack_solve(UMFPACK_A,
        m_outerIndexPtr, m_innerIndexPtr, m_valuePtr,
        &x.col(j).coeffRef(0), &b.const_cast_derived().col(j).coeffRef(0), m_numeric, 0, 0);
    if (errorCode!=0)
      return false;
  }

  return true;
}


namespace internal {

template<typename _MatrixType, typename Rhs>
struct solve_retval<UmfPackLU<_MatrixType>, Rhs>
  : solve_retval_base<UmfPackLU<_MatrixType>, Rhs>
{
  typedef UmfPackLU<_MatrixType> Dec;
  EIGEN_MAKE_SOLVE_HELPERS(Dec,Rhs)

  template<typename Dest> void evalTo(Dest& dst) const
  {
    dec()._solve(rhs(),dst);
  }
};

template<typename _MatrixType, typename Rhs>
struct sparse_solve_retval<UmfPackLU<_MatrixType>, Rhs>
  : sparse_solve_retval_base<UmfPackLU<_MatrixType>, Rhs>
{
  typedef UmfPackLU<_MatrixType> Dec;
  EIGEN_MAKE_SPARSE_SOLVE_HELPERS(Dec,Rhs)

  template<typename Dest> void evalTo(Dest& dst) const
  {
    this->defaultEvalTo(dst);
  }
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_UMFPACKSUPPORT_H
