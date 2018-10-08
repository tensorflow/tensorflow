//===- IntegerSet.h - MLIR Integer Set Class --------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Integer sets are sets of points from the integer lattice constrained by
// affine equality/inequality constraints. This class is meant to represent
// affine equality/inequality conditions for MLFunctions' if statements. As
// such, it is only expected to contain a handful of affine constraints, and it
// is immutable like an Affine Map. Integer sets are however not unique'd -
// although affine expressions that make up the equalities and inequalites of an
// integer set are themselves unique.

// This class is not meant for affine analysis and operations like set
// operations, emptiness checks, or other math operations for analysis and
// transformation. Another data structure (TODO(bondhugula)) will be used to
// create and operate on such temporary constaint systems.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_INTEGER_SET_H
#define MLIR_IR_INTEGER_SET_H

#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {

class MLIRContext;

/// An integer set representing a conjunction of affine equalities and
/// inequalities. An integer set in the IR is immutable like the affine map, but
/// integer sets are not unique'd. The affine expressions that make up the
/// equalities and inequalities of an integer set are themselves unique.
class IntegerSet {
public:
  static IntegerSet *get(unsigned dimCount, unsigned symbolCount,
                         ArrayRef<AffineExpr> constraints,
                         ArrayRef<bool> eqFlags, MLIRContext *context);

  unsigned getNumDims() { return dimCount; }
  unsigned getNumSymbols() { return symbolCount; }
  unsigned getNumOperands() { return dimCount + symbolCount; }
  unsigned getNumConstraints() { return numConstraints; }

  ArrayRef<AffineExpr> getConstraints() { return constraints; }

  AffineExpr getConstraint(unsigned idx) { return getConstraints()[idx]; }

  /// Returns the equality bits, which specify whether each of the constraints
  /// is an equality or inequality.
  ArrayRef<bool> getEqFlags() { return eqFlags; }

  /// Returns true if the idx^th constraint is an equality, false if it is an
  /// inequality.
  bool isEq(unsigned idx) { return getEqFlags()[idx]; }

  void print(raw_ostream &os);
  void dump();

private:
  IntegerSet(unsigned dimCount, unsigned symbolCount, unsigned numConstraints,
             ArrayRef<AffineExpr> constraints, ArrayRef<bool> eqFlags);

  ~IntegerSet() = delete;

  unsigned dimCount;
  unsigned symbolCount;
  unsigned numConstraints;

  /// Array of affine constraints: a constaint is either an equality
  /// (affine_expr == 0) or an inequality (affine_expr >= 0).
  ArrayRef<AffineExpr> constraints;

  // Bits to check whether a constraint is an equality or an inequality.
  ArrayRef<bool> eqFlags;
};

} // end namespace mlir
#endif // MLIR_IR_INTEGER_SET_H
