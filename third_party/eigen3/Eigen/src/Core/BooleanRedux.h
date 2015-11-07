// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_ALLANDANY_H
#define EIGEN_ALLANDANY_H

namespace Eigen { 

namespace internal {

template<typename Derived, int UnrollCount>
struct all_unroller
{
  enum {
    col = (UnrollCount-1) / Derived::RowsAtCompileTime,
    row = (UnrollCount-1) % Derived::RowsAtCompileTime
  };

  static inline bool run(const Derived &mat)
  {
    return all_unroller<Derived, UnrollCount-1>::run(mat) && mat.coeff(row, col);
  }
};

template<typename Derived>
struct all_unroller<Derived, 0>
{
  static inline bool run(const Derived &/*mat*/) { return true; }
};

template<typename Derived>
struct all_unroller<Derived, Dynamic>
{
  static inline bool run(const Derived &) { return false; }
};

template<typename Derived, int UnrollCount>
struct any_unroller
{
  enum {
    col = (UnrollCount-1) / Derived::RowsAtCompileTime,
    row = (UnrollCount-1) % Derived::RowsAtCompileTime
  };

  static inline bool run(const Derived &mat)
  {
    return any_unroller<Derived, UnrollCount-1>::run(mat) || mat.coeff(row, col);
  }
};

template<typename Derived>
struct any_unroller<Derived, 0>
{
  static inline bool run(const Derived & /*mat*/) { return false; }
};

template<typename Derived>
struct any_unroller<Derived, Dynamic>
{
  static inline bool run(const Derived &) { return false; }
};

} // end namespace internal

/** \returns true if all coefficients are true
  *
  * Example: \include MatrixBase_all.cpp
  * Output: \verbinclude MatrixBase_all.out
  *
  * \sa any(), Cwise::operator<()
  */
template<typename Derived>
inline bool DenseBase<Derived>::all() const
{
  enum {
    unroll = SizeAtCompileTime != Dynamic
          && CoeffReadCost != Dynamic
          && NumTraits<Scalar>::AddCost != Dynamic
          && SizeAtCompileTime * (CoeffReadCost + NumTraits<Scalar>::AddCost) <= EIGEN_UNROLLING_LIMIT
  };
  if(unroll)
    return internal::all_unroller<Derived, unroll ? int(SizeAtCompileTime) : Dynamic>::run(derived());
  else
  {
    for(Index j = 0; j < cols(); ++j)
      for(Index i = 0; i < rows(); ++i)
        if (!coeff(i, j)) return false;
    return true;
  }
}

/** \returns true if at least one coefficient is true
  *
  * \sa all()
  */
template<typename Derived>
inline bool DenseBase<Derived>::any() const
{
  enum {
    unroll = SizeAtCompileTime != Dynamic
          && CoeffReadCost != Dynamic
          && NumTraits<Scalar>::AddCost != Dynamic
          && SizeAtCompileTime * (CoeffReadCost + NumTraits<Scalar>::AddCost) <= EIGEN_UNROLLING_LIMIT
  };
  if(unroll)
    return internal::any_unroller<Derived, unroll ? int(SizeAtCompileTime) : Dynamic>::run(derived());
  else
  {
    for(Index j = 0; j < cols(); ++j)
      for(Index i = 0; i < rows(); ++i)
        if (coeff(i, j)) return true;
    return false;
  }
}

/** \returns the number of coefficients which evaluate to true
  *
  * \sa all(), any()
  */
template<typename Derived>
inline typename DenseBase<Derived>::Index DenseBase<Derived>::count() const
{
  return derived().template cast<bool>().template cast<Index>().sum();
}

/** \returns true is \c *this contains at least one Not A Number (NaN).
  *
  * \sa allFinite()
  */
template<typename Derived>
inline bool DenseBase<Derived>::hasNaN() const
{
  return !((derived().array()==derived().array()).all());
}

/** \returns true if \c *this contains only finite numbers, i.e., no NaN and no +/-INF values.
  *
  * \sa hasNaN()
  */
template<typename Derived>
inline bool DenseBase<Derived>::allFinite() const
{
  return !((derived()-derived()).hasNaN());
}
    
} // end namespace Eigen

#endif // EIGEN_ALLANDANY_H
