// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_SOLVE_H
#define EIGEN_SPARSE_SOLVE_H

namespace Eigen { 

namespace internal {

template<typename _DecompositionType, typename Rhs> struct sparse_solve_retval_base;
template<typename _DecompositionType, typename Rhs> struct sparse_solve_retval;
  
template<typename DecompositionType, typename Rhs>
struct traits<sparse_solve_retval_base<DecompositionType, Rhs> >
{
  typedef typename DecompositionType::MatrixType MatrixType;
  typedef SparseMatrix<typename Rhs::Scalar, Rhs::Options, typename Rhs::Index> ReturnType;
};

template<typename _DecompositionType, typename Rhs> struct sparse_solve_retval_base
 : public ReturnByValue<sparse_solve_retval_base<_DecompositionType, Rhs> >
{
  typedef typename remove_all<typename Rhs::Nested>::type RhsNestedCleaned;
  typedef _DecompositionType DecompositionType;
  typedef ReturnByValue<sparse_solve_retval_base> Base;
  typedef typename Base::Index Index;

  sparse_solve_retval_base(const DecompositionType& dec, const Rhs& rhs)
    : m_dec(dec), m_rhs(rhs)
  {}

  inline Index rows() const { return m_dec.cols(); }
  inline Index cols() const { return m_rhs.cols(); }
  inline const DecompositionType& dec() const { return m_dec; }
  inline const RhsNestedCleaned& rhs() const { return m_rhs; }

  template<typename Dest> inline void evalTo(Dest& dst) const
  {
    static_cast<const sparse_solve_retval<DecompositionType,Rhs>*>(this)->evalTo(dst);
  }

  protected:
    template<typename DestScalar, int DestOptions, typename DestIndex>
    inline void defaultEvalTo(SparseMatrix<DestScalar,DestOptions,DestIndex>& dst) const
    {
      // we process the sparse rhs per block of NbColsAtOnce columns temporarily stored into a dense matrix.
      static const int NbColsAtOnce = 4;
      int rhsCols = m_rhs.cols();
      int size = m_rhs.rows();
      // the temporary matrices do not need more columns than NbColsAtOnce:
      int tmpCols = (std::min)(rhsCols, NbColsAtOnce); 
      Eigen::Matrix<DestScalar,Dynamic,Dynamic> tmp(size,tmpCols);
      Eigen::Matrix<DestScalar,Dynamic,Dynamic> tmpX(size,tmpCols);
      for(int k=0; k<rhsCols; k+=NbColsAtOnce)
      {
        int actualCols = std::min<int>(rhsCols-k, NbColsAtOnce);
        tmp.leftCols(actualCols) = m_rhs.middleCols(k,actualCols);
        tmpX.leftCols(actualCols) = m_dec.solve(tmp.leftCols(actualCols));
        dst.middleCols(k,actualCols) = tmpX.leftCols(actualCols).sparseView();
      }
    }
    const DecompositionType& m_dec;
    typename Rhs::Nested m_rhs;
};

#define EIGEN_MAKE_SPARSE_SOLVE_HELPERS(DecompositionType,Rhs) \
  typedef typename DecompositionType::MatrixType MatrixType; \
  typedef typename MatrixType::Scalar Scalar; \
  typedef typename MatrixType::RealScalar RealScalar; \
  typedef typename MatrixType::Index Index; \
  typedef Eigen::internal::sparse_solve_retval_base<DecompositionType,Rhs> Base; \
  using Base::dec; \
  using Base::rhs; \
  using Base::rows; \
  using Base::cols; \
  sparse_solve_retval(const DecompositionType& dec, const Rhs& rhs) \
    : Base(dec, rhs) {}



template<typename DecompositionType, typename Rhs, typename Guess> struct solve_retval_with_guess;

template<typename DecompositionType, typename Rhs, typename Guess>
struct traits<solve_retval_with_guess<DecompositionType, Rhs, Guess> >
{
  typedef typename DecompositionType::MatrixType MatrixType;
  typedef Matrix<typename Rhs::Scalar,
                 MatrixType::ColsAtCompileTime,
                 Rhs::ColsAtCompileTime,
                 Rhs::PlainObject::Options,
                 MatrixType::MaxColsAtCompileTime,
                 Rhs::MaxColsAtCompileTime> ReturnType;
};

template<typename DecompositionType, typename Rhs, typename Guess> struct solve_retval_with_guess
 : public ReturnByValue<solve_retval_with_guess<DecompositionType, Rhs, Guess> >
{
  typedef typename DecompositionType::Index Index;

  solve_retval_with_guess(const DecompositionType& dec, const Rhs& rhs, const Guess& guess)
    : m_dec(dec), m_rhs(rhs), m_guess(guess)
  {}

  inline Index rows() const { return m_dec.cols(); }
  inline Index cols() const { return m_rhs.cols(); }

  template<typename Dest> inline void evalTo(Dest& dst) const
  {
    dst = m_guess;
    m_dec._solveWithGuess(m_rhs,dst);
  }

  protected:
    const DecompositionType& m_dec;
    const typename Rhs::Nested m_rhs;
    const typename Guess::Nested m_guess;
};

} // namepsace internal

} // end namespace Eigen

#endif // EIGEN_SPARSE_SOLVE_H
