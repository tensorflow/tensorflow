//===- AffineStructures.h - MLIR Affine Structures Class --------*- C++ -*-===//
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
// Structures for affine/polyhedral analysis of ML functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_AFFINE_STRUCTURES_H
#define MLIR_ANALYSIS_AFFINE_STRUCTURES_H

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

class AffineApplyOp;
class AffineBound;
class AffineCondition;
class AffineMap;
class IntegerSet;
class MLIRContext;
class MLValue;
class HyperRectangularSet;

/// A mutable affine map. Its affine expressions are however unique.
struct MutableAffineMap {
public:
  MutableAffineMap(AffineMap map);

  AffineExpr getResult(unsigned idx) const { return results[idx]; }
  void setResult(unsigned idx, AffineExpr result) { results[idx] = result; }
  unsigned getNumResults() const { return results.size(); }
  unsigned getNumDims() const { return numDims; }
  void setNumDims(unsigned d) { numDims = d; }
  unsigned getNumSymbols() const { return numSymbols; }
  void setNumSymbols(unsigned d) { numSymbols = d; }
  MLIRContext *getContext() const { return context; }

  /// Returns true if the idx'th result expression is a multiple of factor.
  bool isMultipleOf(unsigned idx, int64_t factor) const;

  /// Simplify the (result) expressions in this map using analysis (used by
  //-simplify-affine-expr pass).
  void simplify();
  /// Get the AffineMap corresponding to this MutableAffineMap. Note that an
  /// AffineMap will be uniqued and stored in context, while a mutable one
  /// isn't.
  AffineMap getAffineMap();

private:
  // Same meaning as AffineMap's fields.
  SmallVector<AffineExpr, 8> results;
  SmallVector<AffineExpr, 8> rangeSizes;
  unsigned numDims;
  unsigned numSymbols;
  /// A pointer to the IR's context to store all newly created
  /// AffineExprStorage's.
  MLIRContext *context;
};

/// A mutable integer set. Its affine expressions are however unique.
struct MutableIntegerSet {
public:
  MutableIntegerSet(IntegerSet set, MLIRContext *context);

  /// Create a universal set (no constraints).
  MutableIntegerSet(unsigned numDims, unsigned numSymbols,
                    MLIRContext *context);

  unsigned getNumDims() const { return numDims; }
  unsigned getNumSymbols() const { return numSymbols; }
  unsigned getNumConstraints() const { return constraints.size(); }

  void clear() {
    constraints.clear();
    eqFlags.clear();
  }

private:
  unsigned numDims;
  unsigned numSymbols;

  SmallVector<AffineExpr, 8> constraints;
  SmallVector<bool, 8> eqFlags;
  /// A pointer to the IR's context to store all newly created
  /// AffineExprStorage's.
  MLIRContext *context;
};

/// An AffineValueMap is an affine map plus its ML value operands and
/// results for analysis purposes. The structure is still a tree form that is
/// same as that of an affine map or an AffineApplyOp. However, its operands,
/// results, and its map can themselves change  as a result of
/// substitutions, simplifications, and other analysis.
// An affine value map can readily be constructed from an AffineApplyOp, or an
// AffineBound of a ForStmt. It can be further transformed, substituted into,
// or simplified. Unlike AffineMap's, AffineValueMap's are created and destroyed
// during analysis. Only the AffineMap expressions that are pointed by them are
// unique'd.
// TODO(bondhugula): Some of these classes could go into separate files.
class AffineValueMap {
public:
  AffineValueMap(const AffineApplyOp &op);
  AffineValueMap(const AffineBound &bound);
  AffineValueMap(AffineMap map);
  AffineValueMap(AffineMap map, ArrayRef<MLValue *> operands);

  ~AffineValueMap();

  /// Substitute the results of inputMap into the operands of this map.
  // The new list of operands will be a union of this map's and that of the map
  // we are substituting from.
  // Example usage scenario: a subscript operand for a 'load' is forward
  // substituted into the memref's access map. The subscript operand itself is
  // then substituted by its defining affine_apply op instructions and
  // successively by a loop IV remap expression, eventually resulting in an
  // affine value map that has only the loop IVs and symbols as its operands.
  // Hence, the access pattern can then be analyzed for example.
  // TODO(bondhugula)
  void forwardSubstitute(const AffineValueMap &inputMap);
  void forwardSubstitute(const AffineApplyOp &inputOp);
  void forwardSubstituteSingle(const AffineApplyOp &inputOp,
                               unsigned inputResultIndex);
  // TODO(andydavis, bondhugula) Expose an affine map simplify function, which
  // can be used to amortize the cost of simplification over multiple fwd
  // substitutions).

  /// Return true if the idx^th result can be proved to be a multiple of
  /// 'factor', false otherwise.
  inline bool isMultipleOf(unsigned idx, int64_t factor) const;

  /// Return true if the result at 'idx' is a constant, false
  /// otherwise.
  bool isConstant(unsigned idx) const;

  /// Return true if this is an identity map.
  bool isIdentity() const;

  unsigned getNumOperands() const;
  SSAValue *getOperand(unsigned i) const;
  ArrayRef<MLValue *> getOperands() const;
  AffineMap getAffineMap();

private:
  void forwardSubstitute(const AffineApplyOp &inputOp,
                         ArrayRef<bool> inputResultsToSubstitute);
  // A mutable affine map.
  MutableAffineMap map;

  // TODO: make these trailing objects?
  /// The SSA operands binding to the dim's and symbols of 'map'.
  SmallVector<MLValue *, 4> operands;
  /// The SSA results binding to the results of 'map'.
  SmallVector<MLValue *, 4> results;
};

/// An IntegerValueSet is an integer set plus its operands.
// Both, the integer set being pointed to and the operands can change during
// analysis, simplification, and transformation.
class IntegerValueSet {
  // Constructs an integer value set map from an IntegerSet and operands.
  explicit IntegerValueSet(const AffineCondition &cond);

  /// Constructs an integer value set from an affine value map.
  // This will lead to a single equality in 'set'.
  explicit IntegerValueSet(const AffineValueMap &avm);

  /// Returns true if this integer set is empty.
  bool isEmpty() const;

  bool getNumDims() const { return set.getNumDims(); }
  bool getNumSymbols() const { return set.getNumSymbols(); }

private:
  // The set pointed to may itself change unlike in IR structures like
  // 'AffineCondition'.
  MutableIntegerSet set;
  /// The SSA operands binding to the dim's and symbols of 'set'.
  SmallVector<MLValue *, 4> operands;
};

/// A flat list of affine equalities and inequalities in the form.
/// Inequality: c_0*x_0 + c_1*x_1 + .... + c_{n-1}*x_{n-1} == 0
/// Equality: c_0*x_0 + c_1*x_1 + .... + c_{n-1}*x_{n-1} >= 0
///
/// The coefficients are stored. x_0, x_1, ... appear in the order: dimensional
/// identifiers, symbolic identifiers, and local identifiers.  / The local
/// identifiers correspond to local/internal variables created / temporarily and
/// are needed to increase representational power. Local identifiers / are
/// typically obtained when eliminating % and div constraints.
//  Usage scenario:
//  For a register tiling or unroll-jam, for example, if we need to check loop
//  bounds are a multiple of say the tile size say 4:
//
// %lb = affine_apply #map1 (%s, %i0)
// %ub = affine_apply #map2 (%N, %i0)
// for %i1 = %lb to %ub
//   ...
//
// Create AffineValueMap's that have result %lb, %ub (successively fwd
// substituting all affine_apply that lead to the %lb, %ub). Create another
// AffineValueMap: %trip_count = (%ub - %lb + 1). Create a
// FlatAffineConstraints set using all these. Add %trip_count % 4 = 0 to this,
// and check for feasibility.
class FlatAffineConstraints {
public:
  enum IdKind { Dimension, Symbol, Local };

  /// Construct a constraint system reserving memory for the specified number of
  /// constraints and identifiers..
  FlatAffineConstraints(unsigned numReservedInequalities,
                        unsigned numReservedEqualities, unsigned numReservedIds)
      : numReservedEqualities(numReservedEqualities),
        numReservedInequalities(numReservedInequalities),
        numReservedIds(numReservedIds) {
    equalities.reserve(numReservedIds * numReservedEqualities);
    inequalities.reserve(numReservedIds * numReservedInequalities);
  }

  explicit FlatAffineConstraints(const HyperRectangularSet &set);

  /// Create a flat affine constraint system from an AffineValueMap or a list of
  /// these. The constructed system will only include equalities.
  // TODO(bondhugula)
  explicit FlatAffineConstraints(const AffineValueMap &avm);
  explicit FlatAffineConstraints(ArrayRef<const AffineValueMap *> avmRef);

  /// Create an affine constraint system from an IntegerValueSet.
  // TODO(bondhugula)
  explicit FlatAffineConstraints(const IntegerValueSet &set);

  FlatAffineConstraints(ArrayRef<const AffineValueMap *> avmRef,
                        const IntegerSet &set);

  FlatAffineConstraints(const MutableAffineMap &map);

  ~FlatAffineConstraints() {}

  inline int64_t atEq(unsigned i, unsigned j) const {
    return equalities[i * (numIds + 1) + j];
  }

  inline int64_t &atEq(unsigned i, unsigned j) {
    return equalities[i * (numIds + 1) + j];
  }

  inline int64_t atIneq(unsigned i, unsigned j) const {
    return inequalities[i * (numIds + 1) + j];
  }

  inline int64_t &atIneq(unsigned i, unsigned j) {
    return inequalities[i * (numIds + 1) + j];
  }

  inline unsigned getNumCols() const { return numIds + 1; }

  inline unsigned getNumEqualities() const {
    return equalities.size() / getNumCols();
  }

  inline unsigned getNumInequalities() const {
    return inequalities.size() / getNumCols();
  }

  ArrayRef<int64_t> getEquality(unsigned idx) {
    return ArrayRef<int64_t>(&equalities[idx * getNumCols()], getNumCols());
  }

  ArrayRef<int64_t> getInequality(unsigned idx) {
    return ArrayRef<int64_t>(&inequalities[idx * getNumCols()], getNumCols());
  }

  AffineExpr toAffineExpr(unsigned idx, MLIRContext *context);

  void addInequality(ArrayRef<int64_t> inEq);
  void addEquality(ArrayRef<int64_t> eq);

  void addId(IdKind idKind, unsigned pos);
  void addDimId(unsigned pos);
  void addSymbolId(unsigned pos);
  void addLocalId(unsigned pos);

  void removeId(IdKind idKind, unsigned pos);

  void removeEquality(unsigned pos);
  void removeInequality(unsigned pos);

  unsigned getNumConstraints() const {
    return equalities.size() + inequalities.size();
  }
  inline unsigned getNumIds() const { return numIds; }
  inline unsigned getNumResultDimIds() const { return numResultDims; }
  inline unsigned getNumDimIds() const { return numDims; }
  inline unsigned getNumSymbolIds() const { return numSymbols; }
  inline unsigned getNumLocalIds() const {
    return numIds - numResultDims - numDims - numSymbols;
  }

  void print(raw_ostream &os) const;
  void dump() const;

private:
  /// Coefficients of affine equalities (in == 0 form).
  SmallVector<int64_t, 64> equalities;

  /// Coefficients of affine inequalities (in >= 0 form).
  SmallVector<int64_t, 64> inequalities;

  // Pre-allocated space.
  unsigned numReservedEqualities;
  unsigned numReservedInequalities;
  unsigned numReservedIds;

  /// Total number of identifiers.
  unsigned numIds;

  /// Number of identifiers corresponding to real dimensions.
  unsigned numResultDims;

  /// Number of identifiers corresponding to real dimensions.
  unsigned numDims;

  /// Number of identifiers corresponding to symbols (unknown but constant for
  /// analysis).
  unsigned numSymbols;
};

} // end namespace mlir.

#endif // MLIR_ANALYSIS_AFFINE_STRUCTURES_H
