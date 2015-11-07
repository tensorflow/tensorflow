// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_LAZY_H
#define EIGEN_LAZY_H

namespace Eigen { 

/** \deprecated it is only used by lazy() which is deprecated
  *
  * \returns an expression of *this with added flags
  *
  * Example: \include MatrixBase_marked.cpp
  * Output: \verbinclude MatrixBase_marked.out
  *
  * \sa class Flagged, extract(), part()
  */
template<typename Derived>
template<unsigned int Added>
inline const Flagged<Derived, Added, 0>
MatrixBase<Derived>::marked() const
{
  return derived();
}

/** \deprecated use MatrixBase::noalias()
  *
  * \returns an expression of *this with the EvalBeforeAssigningBit flag removed.
  *
  * Example: \include MatrixBase_lazy.cpp
  * Output: \verbinclude MatrixBase_lazy.out
  *
  * \sa class Flagged, marked()
  */
template<typename Derived>
inline const Flagged<Derived, 0, EvalBeforeAssigningBit>
MatrixBase<Derived>::lazy() const
{
  return derived();
}


/** \internal
  * Overloaded to perform an efficient C += (A*B).lazy() */
template<typename Derived>
template<typename ProductDerived, typename Lhs, typename Rhs>
Derived& MatrixBase<Derived>::operator+=(const Flagged<ProductBase<ProductDerived, Lhs,Rhs>, 0,
                                                       EvalBeforeAssigningBit>& other)
{
  other._expression().derived().addTo(derived()); return derived();
}

/** \internal
  * Overloaded to perform an efficient C -= (A*B).lazy() */
template<typename Derived>
template<typename ProductDerived, typename Lhs, typename Rhs>
Derived& MatrixBase<Derived>::operator-=(const Flagged<ProductBase<ProductDerived, Lhs,Rhs>, 0,
                                                       EvalBeforeAssigningBit>& other)
{
  other._expression().derived().subTo(derived()); return derived();
}

} // end namespace Eigen

#endif // EIGEN_LAZY_H
