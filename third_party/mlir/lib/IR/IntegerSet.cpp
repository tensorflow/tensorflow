//===- IntegerSet.cpp - MLIR Integer Set class ----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/IntegerSet.h"
#include "IntegerSetDetail.h"
#include "mlir/IR/AffineExpr.h"

using namespace mlir;
using namespace mlir::detail;

unsigned IntegerSet::getNumDims() const { return set->dimCount; }
unsigned IntegerSet::getNumSymbols() const { return set->symbolCount; }
unsigned IntegerSet::getNumInputs() const {
  return set->dimCount + set->symbolCount;
}

unsigned IntegerSet::getNumConstraints() const {
  return set->constraints.size();
}

unsigned IntegerSet::getNumEqualities() const {
  unsigned numEqualities = 0;
  for (unsigned i = 0, e = getNumConstraints(); i < e; i++)
    if (isEq(i))
      ++numEqualities;
  return numEqualities;
}

unsigned IntegerSet::getNumInequalities() const {
  return getNumConstraints() - getNumEqualities();
}

bool IntegerSet::isEmptyIntegerSet() const {
  // This will only work if uniquing is on.
  static_assert(kUniquingThreshold >= 1,
                "uniquing threshold should be at least one");
  return *this == getEmptySet(set->dimCount, set->symbolCount, getContext());
}

ArrayRef<AffineExpr> IntegerSet::getConstraints() const {
  return set->constraints;
}

AffineExpr IntegerSet::getConstraint(unsigned idx) const {
  return getConstraints()[idx];
}

/// Returns the equality bits, which specify whether each of the constraints
/// is an equality or inequality.
ArrayRef<bool> IntegerSet::getEqFlags() const { return set->eqFlags; }

/// Returns true if the idx^th constraint is an equality, false if it is an
/// inequality.
bool IntegerSet::isEq(unsigned idx) const { return getEqFlags()[idx]; }

MLIRContext *IntegerSet::getContext() const {
  return getConstraint(0).getContext();
}

/// Walk all of the AffineExpr's in this set. Each node in an expression
/// tree is visited in postorder.
void IntegerSet::walkExprs(function_ref<void(AffineExpr)> callback) const {
  for (auto expr : getConstraints())
    expr.walk(callback);
}

IntegerSet IntegerSet::replaceDimsAndSymbols(
    ArrayRef<AffineExpr> dimReplacements, ArrayRef<AffineExpr> symReplacements,
    unsigned numResultDims, unsigned numResultSyms) {
  SmallVector<AffineExpr, 8> constraints;
  constraints.reserve(getNumConstraints());
  for (auto cst : getConstraints())
    constraints.push_back(
        cst.replaceDimsAndSymbols(dimReplacements, symReplacements));

  return get(numResultDims, numResultSyms, constraints, getEqFlags());
}
