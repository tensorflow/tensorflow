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
// integer sets in the IR - for 'affine.if' instructions and as attributes of
// other instructions. It is typically expected to contain only a handful of
// affine constraints, and is immutable like an affine map. Integer sets are not
// unique'd - although affine expressions that make up its equalities and
// inequalites are themselves unique.

// This class is not meant for affine analysis and operations like set
// operations, emptiness checks, or other math operations for analysis and
// transformation. For the latter, use FlatAffineConstraints.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_INTEGER_SET_H
#define MLIR_IR_INTEGER_SET_H

#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {

namespace detail {
struct IntegerSetStorage;
}

class MLIRContext;

/// An integer set representing a conjunction of one or more affine equalities
/// and inequalities. An integer set in the IR is immutable like the affine map,
/// but integer sets are not unique'd. The affine expressions that make up the
/// equalities and inequalities of an integer set are themselves unique and are
/// allocated by the bump pointer allocator.
class IntegerSet {
public:
  using ImplType = detail::IntegerSetStorage;

  IntegerSet() : set(nullptr) {}
  explicit IntegerSet(ImplType *set) : set(set) {}
  IntegerSet(const IntegerSet &other) : set(other.set) {}
  IntegerSet &operator=(const IntegerSet &other) = default;

  static IntegerSet get(unsigned dimCount, unsigned symbolCount,
                        ArrayRef<AffineExpr> constraints,
                        ArrayRef<bool> eqFlags);

  // Returns the canonical empty IntegerSet (i.e. a set with no integer points).
  static IntegerSet getEmptySet(unsigned numDims, unsigned numSymbols,
                                MLIRContext *context) {
    auto one = getAffineConstantExpr(1, context);
    /* 1 == 0 */
    return get(numDims, numSymbols, one, true);
  }

  /// Returns true if this is the canonical integer set.
  bool isEmptyIntegerSet() const;

  explicit operator bool() { return set; }
  bool operator==(IntegerSet other) const { return set == other.set; }

  unsigned getNumDims() const;
  unsigned getNumSymbols() const;
  unsigned getNumOperands() const;
  unsigned getNumConstraints() const;
  unsigned getNumEqualities() const;
  unsigned getNumInequalities() const;

  ArrayRef<AffineExpr> getConstraints() const;

  AffineExpr getConstraint(unsigned idx) const;

  /// Returns the equality bits, which specify whether each of the constraints
  /// is an equality or inequality.
  ArrayRef<bool> getEqFlags() const;

  /// Returns true if the idx^th constraint is an equality, false if it is an
  /// inequality.
  bool isEq(unsigned idx) const;

  MLIRContext *getContext() const;

  void print(raw_ostream &os) const;
  void dump() const;

  friend ::llvm::hash_code hash_value(IntegerSet arg);

private:
  ImplType *set;
  /// Sets with constraints fewer than kUniquingThreshold are uniqued.
  constexpr static unsigned kUniquingThreshold = 4;
};

// Make AffineExpr hashable.
inline ::llvm::hash_code hash_value(IntegerSet arg) {
  return ::llvm::hash_value(arg.set);
}

} // end namespace mlir
namespace llvm {

// IntegerSet hash just like pointers
template <> struct DenseMapInfo<mlir::IntegerSet> {
  static mlir::IntegerSet getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::IntegerSet(static_cast<mlir::IntegerSet::ImplType *>(pointer));
  }
  static mlir::IntegerSet getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::IntegerSet(static_cast<mlir::IntegerSet::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::IntegerSet val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::IntegerSet LHS, mlir::IntegerSet RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm
#endif // MLIR_IR_INTEGER_SET_H
