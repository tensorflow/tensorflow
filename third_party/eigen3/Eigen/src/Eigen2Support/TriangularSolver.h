// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TRIANGULAR_SOLVER2_H
#define EIGEN_TRIANGULAR_SOLVER2_H

namespace Eigen { 

const unsigned int UnitDiagBit = UnitDiag;
const unsigned int SelfAdjointBit = SelfAdjoint;
const unsigned int UpperTriangularBit = Upper;
const unsigned int LowerTriangularBit = Lower;

const unsigned int UpperTriangular = Upper;
const unsigned int LowerTriangular = Lower;
const unsigned int UnitUpperTriangular = UnitUpper;
const unsigned int UnitLowerTriangular = UnitLower;

template<typename ExpressionType, unsigned int Added, unsigned int Removed>
template<typename OtherDerived>
typename ExpressionType::PlainObject
Flagged<ExpressionType,Added,Removed>::solveTriangular(const MatrixBase<OtherDerived>& other) const
{
  return m_matrix.template triangularView<Added>().solve(other.derived());
}

template<typename ExpressionType, unsigned int Added, unsigned int Removed>
template<typename OtherDerived>
void Flagged<ExpressionType,Added,Removed>::solveTriangularInPlace(const MatrixBase<OtherDerived>& other) const
{
  m_matrix.template triangularView<Added>().solveInPlace(other.derived());
}

} // end namespace Eigen
    
#endif // EIGEN_TRIANGULAR_SOLVER2_H
