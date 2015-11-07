// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MISC_SOLVE_H
#define EIGEN_MISC_SOLVE_H

namespace Eigen { 

namespace internal {

/** \class solve_retval_base
  *
  */
template<typename DecompositionType, typename Rhs>
struct traits<solve_retval_base<DecompositionType, Rhs> >
{
  typedef typename DecompositionType::MatrixType MatrixType;
  typedef Matrix<typename Rhs::Scalar,
                 MatrixType::ColsAtCompileTime,
                 Rhs::ColsAtCompileTime,
                 Rhs::PlainObject::Options,
                 MatrixType::MaxColsAtCompileTime,
                 Rhs::MaxColsAtCompileTime> ReturnType;
};

template<typename _DecompositionType, typename Rhs> struct solve_retval_base
 : public ReturnByValue<solve_retval_base<_DecompositionType, Rhs> >
{
  typedef typename remove_all<typename Rhs::Nested>::type RhsNestedCleaned;
  typedef _DecompositionType DecompositionType;
  typedef ReturnByValue<solve_retval_base> Base;
  typedef typename Base::Index Index;

  solve_retval_base(const DecompositionType& dec, const Rhs& rhs)
    : m_dec(dec), m_rhs(rhs)
  {}

  inline Index rows() const { return m_dec.cols(); }
  inline Index cols() const { return m_rhs.cols(); }
  inline const DecompositionType& dec() const { return m_dec; }
  inline const RhsNestedCleaned& rhs() const { return m_rhs; }

  template<typename Dest> inline void evalTo(Dest& dst) const
  {
    static_cast<const solve_retval<DecompositionType,Rhs>*>(this)->evalTo(dst);
  }

  protected:
    const DecompositionType& m_dec;
    typename Rhs::Nested m_rhs;
};

} // end namespace internal

#define EIGEN_MAKE_SOLVE_HELPERS(DecompositionType,Rhs) \
  typedef typename DecompositionType::MatrixType MatrixType; \
  typedef typename MatrixType::Scalar Scalar; \
  typedef typename MatrixType::RealScalar RealScalar; \
  typedef typename MatrixType::Index Index; \
  typedef Eigen::internal::solve_retval_base<DecompositionType,Rhs> Base; \
  using Base::dec; \
  using Base::rhs; \
  using Base::rows; \
  using Base::cols; \
  solve_retval(const DecompositionType& dec, const Rhs& rhs) \
    : Base(dec, rhs) {}

} // end namespace Eigen

#endif // EIGEN_MISC_SOLVE_H
