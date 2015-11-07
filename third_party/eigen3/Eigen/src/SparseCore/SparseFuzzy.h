// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_FUZZY_H
#define EIGEN_SPARSE_FUZZY_H

// template<typename Derived>
// template<typename OtherDerived>
// bool SparseMatrixBase<Derived>::isApprox(
//   const OtherDerived& other,
//   typename NumTraits<Scalar>::Real prec
// ) const
// {
//   const typename internal::nested<Derived,2>::type nested(derived());
//   const typename internal::nested<OtherDerived,2>::type otherNested(other.derived());
//   return    (nested - otherNested).cwise().abs2().sum()
//          <= prec * prec * (std::min)(nested.cwise().abs2().sum(), otherNested.cwise().abs2().sum());
// }

#endif // EIGEN_SPARSE_FUZZY_H
