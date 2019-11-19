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
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

class AffineApplyOp;
class AffineBound;
class AffineCondition;
class AffineMap;
class AffineForOp;
class IntegerSet;
class MLIRContext;
class Value;
class HyperRectangularSet;
class MemRefType;

/// A mutable affine map. Its affine expressions are however unique.
struct MutableAffineMap {
public:
  MutableAffineMap() {}
  MutableAffineMap(AffineMap map);

  ArrayRef<AffineExpr> getResults() const { return results; }
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

  /// Resets this MutableAffineMap with 'map'.
  void reset(AffineMap map);

  /// Simplify the (result) expressions in this map using analysis (used by
  //-simplify-affine-expr pass).
  void simplify();
  /// Get the AffineMap corresponding to this MutableAffineMap. Note that an
  /// AffineMap will be uniqued and stored in context, while a mutable one
  /// isn't.
  AffineMap getAffineMap() const;

private:
  // Same meaning as AffineMap's fields.
  SmallVector<AffineExpr, 8> results;
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
};

/// An AffineValueMap is an affine map plus its ML value operands and
/// results for analysis purposes. The structure is still a tree form that is
/// same as that of an affine map or an AffineApplyOp. However, its operands,
/// results, and its map can themselves change  as a result of
/// substitutions, simplifications, and other analysis.
// An affine value map can readily be constructed from an AffineApplyOp, or an
// AffineBound of a AffineForOp. It can be further transformed, substituted
// into, or simplified. Unlike AffineMap's, AffineValueMap's are created and
// destroyed during analysis. Only the AffineMap expressions that are pointed by
// them are unique'd. An affine value map, and the operations on it, maintain
// the invariant that operands are always positionally aligned with the
// AffineDimExpr and AffineSymbolExpr in the underlying AffineMap.
// TODO(bondhugula): Some of these classes could go into separate files.
class AffineValueMap {
public:
  // Creates an empty AffineValueMap (users should call 'reset' to reset map
  // and operands).
  AffineValueMap() {}
  AffineValueMap(AffineMap map, ArrayRef<Value *> operands,
                 ArrayRef<Value *> results = llvm::None);

  explicit AffineValueMap(AffineApplyOp applyOp);
  explicit AffineValueMap(AffineBound bound);

  ~AffineValueMap();

  // Resets this AffineValueMap with 'map', 'operands', and 'results'.
  void reset(AffineMap map, ArrayRef<Value *> operands,
             ArrayRef<Value *> results = llvm::None);

  /// Return the value map that is the difference of value maps 'a' and 'b',
  /// represented as an affine map and its operands. The output map + operands
  /// are canonicalized and simplified.
  static void difference(const AffineValueMap &a, const AffineValueMap &b,
                         AffineValueMap *res);

  /// Return true if the idx^th result can be proved to be a multiple of
  /// 'factor', false otherwise.
  inline bool isMultipleOf(unsigned idx, int64_t factor) const;

  /// Return true if the idx^th result depends on 'value', false otherwise.
  bool isFunctionOf(unsigned idx, Value *value) const;

  /// Return true if the result at 'idx' is a constant, false
  /// otherwise.
  bool isConstant(unsigned idx) const;

  /// Return true if this is an identity map.
  bool isIdentity() const;

  void setResult(unsigned i, AffineExpr e) { map.setResult(i, e); }
  AffineExpr getResult(unsigned i) { return map.getResult(i); }
  inline unsigned getNumOperands() const { return operands.size(); }
  inline unsigned getNumDims() const { return map.getNumDims(); }
  inline unsigned getNumSymbols() const { return map.getNumSymbols(); }
  inline unsigned getNumResults() const { return map.getNumResults(); }

  Value *getOperand(unsigned i) const;
  ArrayRef<Value *> getOperands() const;
  AffineMap getAffineMap() const;

private:
  // A mutable affine map.
  MutableAffineMap map;

  // TODO: make these trailing objects?
  /// The SSA operands binding to the dim's and symbols of 'map'.
  SmallVector<Value *, 4> operands;
  /// The SSA results binding to the results of 'map'.
  SmallVector<Value *, 4> results;
};

/// An IntegerValueSet is an integer set plus its operands.
// Both, the integer set being pointed to and the operands can change during
// analysis, simplification, and transformation.
class IntegerValueSet {
  /// Constructs an integer value set from an affine value map.
  // This will lead to a single equality in 'set'.
  explicit IntegerValueSet(const AffineValueMap &avm);

  /// Returns true if this integer set is determined to be empty. Emptiness is
  /// checked by by eliminating identifiers successively (through either
  /// Gaussian or Fourier-Motzkin) while using the GCD test and a trivial
  /// invalid constraint check. Returns 'true' if the constraint system is found
  /// to be empty; false otherwise. This method is exact for rational spaces but
  /// not integer spaces - thus, if it returns true, the set is provably integer
  /// empty as well, but if it returns false, it doesn't necessarily mean an
  /// integer point exists in it. This method also returns false where an
  /// explosion of constraints is detected - due to the super-exponential
  /// worse-case complexity of Fourier-Motzkin elimination (rare for realistic
  /// problem cases but possible for artificial adversarial or improperly
  // constructed ones), this method returns false conservatively.
  bool isEmpty() const;

  bool getNumDims() const { return set.getNumDims(); }
  bool getNumSymbols() const { return set.getNumSymbols(); }

private:
  // The set pointed to may itself change unlike in IR structures like
  // 'AffineCondition'.
  MutableIntegerSet set;
  /// The SSA operands binding to the dim's and symbols of 'set'.
  SmallVector<Value *, 4> operands;
};

/// A flat list of affine equalities and inequalities in the form.
/// Inequality: c_0*x_0 + c_1*x_1 + .... + c_{n-1}*x_{n-1} >= 0
/// Equality: c_0*x_0 + c_1*x_1 + .... + c_{n-1}*x_{n-1} == 0
///
/// FlatAffineConstraints stores coefficients in a contiguous buffer (one buffer
/// for equalities and one for inequalities). The size of each buffer is
/// numReservedCols * number of inequalities (or equalities). The reserved size
/// is numReservedCols * numReservedInequalities (or numReservedEqualities). A
/// coefficient (r, c) lives at the location numReservedCols * r + c in the
/// buffer. The extra space between getNumCols() and numReservedCols exists to
/// prevent frequent movement of data when adding columns, especially at the
/// end.
///
/// The identifiers x_0, x_1, ... appear in the order: dimensional identifiers,
/// symbolic identifiers, and local identifiers.  The local identifiers
/// correspond to local/internal variables created when converting from
/// AffineExpr's containing mod's and div's; they are thus needed to increase
/// representational power. Each local identifier is always (by construction) a
/// floordiv of a pure add/mul affine function of dimensional, symbolic, and
/// other local identifiers, in a non-mutually recursive way. Hence, every local
/// identifier can ultimately always be recovered as an affine function of
/// dimensional and symbolic identifiers (involving floordiv's); note however
/// that some floordiv combinations are converted to mod's by AffineExpr
/// construction.
///
class FlatAffineConstraints {
public:
  enum IdKind { Dimension, Symbol, Local };

  /// Constructs a constraint system reserving memory for the specified number
  /// of constraints and identifiers..
  FlatAffineConstraints(unsigned numReservedInequalities,
                        unsigned numReservedEqualities,
                        unsigned numReservedCols, unsigned numDims = 0,
                        unsigned numSymbols = 0, unsigned numLocals = 0,
                        ArrayRef<Optional<Value *>> idArgs = {})
      : numReservedCols(numReservedCols), numDims(numDims),
        numSymbols(numSymbols) {
    assert(numReservedCols >= numDims + numSymbols + 1);
    assert(idArgs.empty() || idArgs.size() == numDims + numSymbols + numLocals);
    equalities.reserve(numReservedCols * numReservedEqualities);
    inequalities.reserve(numReservedCols * numReservedInequalities);
    numIds = numDims + numSymbols + numLocals;
    ids.reserve(numReservedCols);
    if (idArgs.empty())
      ids.resize(numIds, None);
    else
      ids.append(idArgs.begin(), idArgs.end());
  }

  /// Constructs a constraint system with the specified number of
  /// dimensions and symbols.
  FlatAffineConstraints(unsigned numDims = 0, unsigned numSymbols = 0,
                        unsigned numLocals = 0,
                        ArrayRef<Optional<Value *>> idArgs = {})
      : numReservedCols(numDims + numSymbols + numLocals + 1), numDims(numDims),
        numSymbols(numSymbols) {
    assert(numReservedCols >= numDims + numSymbols + 1);
    assert(idArgs.empty() || idArgs.size() == numDims + numSymbols + numLocals);
    numIds = numDims + numSymbols + numLocals;
    ids.reserve(numIds);
    if (idArgs.empty())
      ids.resize(numIds, None);
    else
      ids.append(idArgs.begin(), idArgs.end());
  }

  explicit FlatAffineConstraints(const HyperRectangularSet &set);

  /// Create a flat affine constraint system from an AffineValueMap or a list of
  /// these. The constructed system will only include equalities.
  // TODO(bondhugula)
  explicit FlatAffineConstraints(const AffineValueMap &avm);
  explicit FlatAffineConstraints(ArrayRef<const AffineValueMap *> avmRef);

  /// Creates an affine constraint system from an IntegerSet.
  explicit FlatAffineConstraints(IntegerSet set);

  /// Create an affine constraint system from an IntegerValueSet.
  // TODO(bondhugula)
  explicit FlatAffineConstraints(const IntegerValueSet &set);

  FlatAffineConstraints(const FlatAffineConstraints &other);

  FlatAffineConstraints(ArrayRef<const AffineValueMap *> avmRef,
                        IntegerSet set);

  FlatAffineConstraints(const MutableAffineMap &map);

  ~FlatAffineConstraints() {}

  // Clears any existing data and reserves memory for the specified constraints.
  void reset(unsigned numReservedInequalities, unsigned numReservedEqualities,
             unsigned numReservedCols, unsigned numDims, unsigned numSymbols,
             unsigned numLocals = 0, ArrayRef<Value *> idArgs = {});

  void reset(unsigned numDims = 0, unsigned numSymbols = 0,
             unsigned numLocals = 0, ArrayRef<Value *> idArgs = {});

  /// Appends constraints from 'other' into this. This is equivalent to an
  /// intersection with no simplification of any sort attempted.
  void append(const FlatAffineConstraints &other);

  // Checks for emptiness by performing variable elimination on all identifiers,
  // running the GCD test on each equality constraint, and checking for invalid
  // constraints.
  // Returns true if the GCD test fails for any equality, or if any invalid
  // constraints are discovered on any row. Returns false otherwise.
  bool isEmpty() const;

  // Runs the GCD test on all equality constraints. Returns 'true' if this test
  // fails on any equality. Returns 'false' otherwise.
  // This test can be used to disprove the existence of a solution. If it
  // returns true, no integer solution to the equality constraints can exist.
  bool isEmptyByGCDTest() const;

  // Clones this object.
  std::unique_ptr<FlatAffineConstraints> clone() const;

  /// Returns the value at the specified equality row and column.
  inline int64_t atEq(unsigned i, unsigned j) const {
    return equalities[i * numReservedCols + j];
  }
  inline int64_t &atEq(unsigned i, unsigned j) {
    return equalities[i * numReservedCols + j];
  }

  inline int64_t atIneq(unsigned i, unsigned j) const {
    return inequalities[i * numReservedCols + j];
  }

  inline int64_t &atIneq(unsigned i, unsigned j) {
    return inequalities[i * numReservedCols + j];
  }

  /// Returns the number of columns in the constraint system.
  inline unsigned getNumCols() const { return numIds + 1; }

  inline unsigned getNumEqualities() const {
    assert(equalities.size() % numReservedCols == 0 &&
           "inconsistent equality buffer size");
    return equalities.size() / numReservedCols;
  }

  inline unsigned getNumInequalities() const {
    assert(inequalities.size() % numReservedCols == 0 &&
           "inconsistent inequality buffer size");
    return inequalities.size() / numReservedCols;
  }

  inline unsigned getNumReservedEqualities() const {
    return equalities.capacity() / numReservedCols;
  }

  inline unsigned getNumReservedInequalities() const {
    return inequalities.capacity() / numReservedCols;
  }

  inline ArrayRef<int64_t> getEquality(unsigned idx) const {
    return ArrayRef<int64_t>(&equalities[idx * numReservedCols], getNumCols());
  }

  inline ArrayRef<int64_t> getInequality(unsigned idx) const {
    return ArrayRef<int64_t>(&inequalities[idx * numReservedCols],
                             getNumCols());
  }

  AffineExpr toAffineExpr(unsigned idx, MLIRContext *context);

  /// Adds constraints (lower and upper bounds) for the specified 'affine.for'
  /// operation's Value using IR information stored in its bound maps. The
  /// right identifier is first looked up using forOp's Value. Asserts if the
  /// Value corresponding to the 'affine.for' operation isn't found in the
  /// constraint system. Returns failure for the yet unimplemented/unsupported
  /// cases.  Any new identifiers that are found in the bound operands of the
  /// 'affine.for' operation are added as trailing identifiers (either
  /// dimensional or symbolic depending on whether the operand is a valid
  /// symbol).
  //  TODO(bondhugula): add support for non-unit strides.
  LogicalResult addAffineForOpDomain(AffineForOp forOp);

  /// Adds a lower or an upper bound for the identifier at the specified
  /// position with constraints being drawn from the specified bound map and
  /// operands. If `eq` is true, add a single equality equal to the bound map's
  /// first result expr.
  LogicalResult addLowerOrUpperBound(unsigned pos, AffineMap boundMap,
                                     ArrayRef<Value *> operands, bool eq,
                                     bool lower = true);

  /// Computes the lower and upper bounds of the first 'num' dimensional
  /// identifiers (starting at 'offset') as an affine map of the remaining
  /// identifiers (dimensional and symbolic). This method is able to detect
  /// identifiers as floordiv's and mod's of affine expressions of other
  /// identifiers with respect to (positive) constants. Sets bound map to a
  /// null AffineMap if such a bound can't be found (or yet unimplemented).
  void getSliceBounds(unsigned offset, unsigned num, MLIRContext *context,
                      SmallVectorImpl<AffineMap> *lbMaps,
                      SmallVectorImpl<AffineMap> *ubMaps);

  /// Adds slice lower bounds represented by lower bounds in 'lbMaps' and upper
  /// bounds in 'ubMaps' to each identifier in the constraint system which has
  /// a value in 'values'. Note that both lower/upper bounds share the same
  /// operand list 'operands'.
  /// This function assumes 'values.size' == 'lbMaps.size' == 'ubMaps.size'.
  /// Note that both lower/upper bounds use operands from 'operands'.
  LogicalResult addSliceBounds(ArrayRef<Value *> values,
                               ArrayRef<AffineMap> lbMaps,
                               ArrayRef<AffineMap> ubMaps,
                               ArrayRef<Value *> operands);

  // Adds an inequality (>= 0) from the coefficients specified in inEq.
  void addInequality(ArrayRef<int64_t> inEq);
  // Adds an equality from the coefficients specified in eq.
  void addEquality(ArrayRef<int64_t> eq);

  /// Adds a constant lower bound constraint for the specified identifier.
  void addConstantLowerBound(unsigned pos, int64_t lb);
  /// Adds a constant upper bound constraint for the specified identifier.
  void addConstantUpperBound(unsigned pos, int64_t ub);

  /// Adds a new local identifier as the floordiv of an affine function of other
  /// identifiers, the coefficients of which are provided in 'dividend' and with
  /// respect to a positive constant 'divisor'. Two constraints are added to the
  /// system to capture equivalence with the floordiv:
  /// q = dividend floordiv c    <=>   c*q <= dividend <= c*q + c - 1.
  void addLocalFloorDiv(ArrayRef<int64_t> dividend, int64_t divisor);

  /// Adds a constant lower bound constraint for the specified expression.
  void addConstantLowerBound(ArrayRef<int64_t> expr, int64_t lb);
  /// Adds a constant upper bound constraint for the specified expression.
  void addConstantUpperBound(ArrayRef<int64_t> expr, int64_t ub);

  /// Sets the identifier at the specified position to a constant.
  void setIdToConstant(unsigned pos, int64_t val);

  /// Sets the identifier corresponding to the specified Value id to a
  /// constant. Asserts if the 'id' is not found.
  void setIdToConstant(Value &id, int64_t val);

  /// Looks up the position of the identifier with the specified Value. Returns
  /// true if found (false otherwise). `pos' is set to the (column) position of
  /// the identifier.
  bool findId(Value &id, unsigned *pos) const;

  /// Returns true if an identifier with the specified Value exists, false
  /// otherwise.
  bool containsId(Value &id) const;

  // Add identifiers of the specified kind - specified positions are relative to
  // the kind of identifier. The coefficient column corresponding to the added
  // identifier is initialized to zero. 'id' is the Value corresponding to the
  // identifier that can optionally be provided.
  void addDimId(unsigned pos, Value *id = nullptr);
  void addSymbolId(unsigned pos, Value *id = nullptr);
  void addLocalId(unsigned pos);
  void addId(IdKind kind, unsigned pos, Value *id = nullptr);

  /// Add the specified values as a dim or symbol id depending on its nature, if
  /// it already doesn't exist in the system. `id' has to be either a terminal
  /// symbol or a loop IV, i.e., it cannot be the result affine.apply of any
  /// symbols or loop IVs. The identifier is added to the end of the existing
  /// dims or symbols. Additional information on the identifier is extracted
  /// from the IR and added to the constraint system.
  void addInductionVarOrTerminalSymbol(Value *id);

  /// Composes the affine value map with this FlatAffineConstrains, adding the
  /// results of the map as dimensions at the front [0, vMap->getNumResults())
  /// and with the dimensions set to the equalities specified by the value map.
  /// Returns failure if the composition fails (when vMap is a semi-affine map).
  /// The vMap's operand Value's are used to look up the right positions in
  /// the FlatAffineConstraints with which to associate. The dimensional and
  /// symbolic operands of vMap should match 1:1 (in the same order) with those
  /// of this constraint system, but the latter could have additional trailing
  /// operands.
  LogicalResult composeMap(const AffineValueMap *vMap);

  /// Composes an affine map whose dimensions match one to one to the
  /// dimensions of this FlatAffineConstraints. The results of the map 'other'
  /// are added as the leading dimensions of this constraint system. Returns
  /// failure if 'other' is a semi-affine map.
  LogicalResult composeMatchingMap(AffineMap other);

  /// Projects out (aka eliminates) 'num' identifiers starting at position
  /// 'pos'. The resulting constraint system is the shadow along the dimensions
  /// that still exist. This method may not always be integer exact.
  // TODO(bondhugula): deal with integer exactness when necessary - can return a
  // value to mark exactness for example.
  void projectOut(unsigned pos, unsigned num);
  inline void projectOut(unsigned pos) { return projectOut(pos, 1); }

  /// Projects out the identifier that is associate with Value *.
  void projectOut(Value *id);

  void removeId(IdKind idKind, unsigned pos);
  void removeId(unsigned pos);

  void removeDim(unsigned pos);

  void removeEquality(unsigned pos);
  void removeInequality(unsigned pos);

  /// Changes the partition between dimensions and symbols. Depending on the new
  /// symbol count, either a chunk of trailing dimensional identifiers becomes
  /// symbols, or some of the leading symbols become dimensions.
  void setDimSymbolSeparation(unsigned newSymbolCount);

  /// Changes all symbol identifiers which are loop IVs to dim identifiers.
  void convertLoopIVSymbolsToDims();

  /// Sets the specified identifier to a constant and removes it.
  void setAndEliminate(unsigned pos, int64_t constVal);

  /// Tries to fold the specified identifier to a constant using a trivial
  /// equality detection; if successful, the constant is substituted for the
  /// identifier everywhere in the constraint system and then removed from the
  /// system.
  LogicalResult constantFoldId(unsigned pos);

  /// This method calls constantFoldId for the specified range of identifiers,
  /// 'num' identifiers starting at position 'pos'.
  void constantFoldIdRange(unsigned pos, unsigned num);

  /// Updates the constraints to be the smallest bounding (enclosing) box that
  /// contains the points of 'this' set and that of 'other', with the symbols
  /// being treated specially. For each of the dimensions, the min of the lower
  /// bounds (symbolic) and the max of the upper bounds (symbolic) is computed
  /// to determine such a bounding box. `other' is expected to have the same
  /// dimensional identifiers as this constraint system (in the same order).
  ///
  /// Eg: if 'this' is {0 <= d0 <= 127}, 'other' is {16 <= d0 <= 192}, the
  ///      output is {0 <= d0 <= 192}.
  /// 2) 'this' = {s0 + 5 <= d0 <= s0 + 20}, 'other' is {s0 + 1 <= d0 <= s0 +
  ///     9}, output = {s0 + 1 <= d0 <= s0 + 20}.
  /// 3) 'this' = {0 <= d0 <= 5, 1 <= d1 <= 9}, 'other' = {2 <= d0 <= 6, 5 <= d1
  ///     <= 15}, output = {0 <= d0 <= 6, 1 <= d1 <= 15}.
  LogicalResult unionBoundingBox(const FlatAffineConstraints &other);

  /// Returns 'true' if this constraint system and 'other' are in the same
  /// space, i.e., if they are associated with the same set of identifiers,
  /// appearing in the same order. Returns 'false' otherwise.
  bool areIdsAlignedWithOther(const FlatAffineConstraints &other);

  /// Merge and align the identifiers of 'this' and 'other' starting at
  /// 'offset', so that both constraint systems get the union of the contained
  /// identifiers that is dimension-wise and symbol-wise unique; both
  /// constraint systems are updated so that they have the union of all
  /// identifiers, with this's original identifiers appearing first followed by
  /// any of other's identifiers that didn't appear in 'this'. Local
  /// identifiers of each system are by design separate/local and are placed
  /// one after other (this's followed by other's).
  //  Eg: Input: 'this'  has ((%i %j) [%M %N])
  //             'other' has (%k, %j) [%P, %N, %M])
  //      Output: both 'this', 'other' have (%i, %j, %k) [%M, %N, %P]
  //
  void mergeAndAlignIdsWithOther(unsigned offset, FlatAffineConstraints *other);

  unsigned getNumConstraints() const {
    return getNumInequalities() + getNumEqualities();
  }
  inline unsigned getNumIds() const { return numIds; }
  inline unsigned getNumDimIds() const { return numDims; }
  inline unsigned getNumSymbolIds() const { return numSymbols; }
  inline unsigned getNumDimAndSymbolIds() const { return numDims + numSymbols; }
  inline unsigned getNumLocalIds() const {
    return numIds - numDims - numSymbols;
  }

  inline ArrayRef<Optional<Value *>> getIds() const {
    return {ids.data(), ids.size()};
  }
  inline MutableArrayRef<Optional<Value *>> getIds() {
    return {ids.data(), ids.size()};
  }

  /// Returns the optional Value corresponding to the pos^th identifier.
  inline Optional<Value *> getId(unsigned pos) const { return ids[pos]; }
  inline Optional<Value *> &getId(unsigned pos) { return ids[pos]; }

  /// Returns the Value associated with the pos^th identifier. Asserts if
  /// no Value identifier was associated.
  inline Value *getIdValue(unsigned pos) const {
    assert(ids[pos].hasValue() && "identifier's Value not set");
    return ids[pos].getValue();
  }

  /// Returns the Values associated with identifiers in range [start, end).
  /// Asserts if no Value was associated with one of these identifiers.
  void getIdValues(unsigned start, unsigned end,
                   SmallVectorImpl<Value *> *values) const {
    assert((start < numIds || start == end) && "invalid start position");
    assert(end <= numIds && "invalid end position");
    values->clear();
    values->reserve(end - start);
    for (unsigned i = start; i < end; i++) {
      values->push_back(getIdValue(i));
    }
  }
  inline void getAllIdValues(SmallVectorImpl<Value *> *values) const {
    getIdValues(0, numIds, values);
  }

  /// Sets Value associated with the pos^th identifier.
  inline void setIdValue(unsigned pos, Value *val) {
    assert(pos < numIds && "invalid id position");
    ids[pos] = val;
  }
  /// Sets Values associated with identifiers in the range [start, end).
  void setIdValues(unsigned start, unsigned end, ArrayRef<Value *> values) {
    assert((start < numIds || end == start) && "invalid start position");
    assert(end <= numIds && "invalid end position");
    assert(values.size() == end - start);
    for (unsigned i = start; i < end; ++i)
      ids[i] = values[i - start];
  }

  /// Clears this list of constraints and copies other into it.
  void clearAndCopyFrom(const FlatAffineConstraints &other);

  /// Returns the smallest known constant bound for the extent of the specified
  /// identifier (pos^th), i.e., the smallest known constant that is greater
  /// than or equal to 'exclusive upper bound' - 'lower bound' of the
  /// identifier. Returns None if it's not a constant. This method employs
  /// trivial (low complexity / cost) checks and detection. Symbolic identifiers
  /// are treated specially, i.e., it looks for constant differences between
  /// affine expressions involving only the symbolic identifiers. See comments
  /// at function definition for examples. 'lb' and 'lbDivisor', if provided,
  /// are used to express the lower bound associated with the constant
  /// difference: 'lb' has the coefficients and lbDivisor, the divisor. For eg.,
  /// if the lower bound is [(s0 + s2 - 1) floordiv 32] for a system with three
  /// symbolic identifiers, *lb = [1, 0, 1], lbDivisor = 32.
  Optional<int64_t>
  getConstantBoundOnDimSize(unsigned pos,
                            SmallVectorImpl<int64_t> *lb = nullptr,
                            int64_t *lbFloorDivisor = nullptr,
                            SmallVectorImpl<int64_t> *ub = nullptr) const;

  /// Returns the constant lower bound for the pos^th identifier if there is
  /// one; None otherwise.
  Optional<int64_t> getConstantLowerBound(unsigned pos) const;

  /// Returns the constant upper bound for the pos^th identifier if there is
  /// one; None otherwise.
  Optional<int64_t> getConstantUpperBound(unsigned pos) const;

  /// Gets the lower and upper bound of the pos^th identifier treating
  /// [0, offset) U [offset + num, symbStartPos) as dimensions and
  /// [symStartPos, getNumDimAndSymbolIds) as symbols. The returned
  /// multi-dimensional maps in the pair represent the max and min of
  /// potentially multiple affine expressions. The upper bound is exclusive.
  /// 'localExprs' holds pre-computed AffineExpr's for all local identifiers in
  /// the system.
  std::pair<AffineMap, AffineMap>
  getLowerAndUpperBound(unsigned pos, unsigned offset, unsigned num,
                        unsigned symStartPos, ArrayRef<AffineExpr> localExprs,
                        MLIRContext *context);

  /// Returns true if the set can be trivially detected as being
  /// hyper-rectangular on the specified contiguous set of identifiers.
  bool isHyperRectangular(unsigned pos, unsigned num) const;

  /// Removes duplicate constraints, trivially true constraints, and constraints
  /// that can be detected as redundant as a result of differing only in their
  /// constant term part. A constraint of the form <non-negative constant> >= 0
  /// is considered trivially true. This method is a linear time method on the
  /// constraints, does a single scan, and updates in place.
  void removeTrivialRedundancy();

  /// A more expensive check to detect redundant inequalities thatn
  /// removeTrivialRedundancy.
  void removeRedundantInequalities();

  // Removes all equalities and inequalities.
  void clearConstraints();

  void print(raw_ostream &os) const;
  void dump() const;

private:
  /// Returns false if the fields corresponding to various identifier counts, or
  /// equality/inequality buffer sizes aren't consistent; true otherwise. This
  /// is meant to be used within an assert internally.
  bool hasConsistentState() const;

  /// Checks all rows of equality/inequality constraints for trivial
  /// contradictions (for example: 1 == 0, 0 >= 1), which may have surfaced
  /// after elimination. Returns 'true' if an invalid constraint is found;
  /// 'false'otherwise.
  bool hasInvalidConstraint() const;

  /// Returns the constant lower bound bound if isLower is true, and the upper
  /// bound if isLower is false.
  template <bool isLower>
  Optional<int64_t> computeConstantLowerOrUpperBound(unsigned pos);

  // Eliminates a single identifier at 'position' from equality and inequality
  // constraints. Returns 'success' if the identifier was eliminated, and
  // 'failure' otherwise.
  inline LogicalResult gaussianEliminateId(unsigned position) {
    return success(gaussianEliminateIds(position, position + 1) == 1);
  }

  // Eliminates identifiers from equality and inequality constraints
  // in column range [posStart, posLimit).
  // Returns the number of variables eliminated.
  unsigned gaussianEliminateIds(unsigned posStart, unsigned posLimit);

  /// Eliminates identifier at the specified position using Fourier-Motzkin
  /// variable elimination, but uses Gaussian elimination if there is an
  /// equality involving that identifier. If the result of the elimination is
  /// integer exact, *isResultIntegerExact is set to true. If 'darkShadow' is
  /// set to true, a potential under approximation (subset) of the rational
  /// shadow / exact integer shadow is computed.
  // See implementation comments for more details.
  void FourierMotzkinEliminate(unsigned pos, bool darkShadow = false,
                               bool *isResultIntegerExact = nullptr);

  /// Tightens inequalities given that we are dealing with integer spaces. This
  /// is similar to the GCD test but applied to inequalities. The constant term
  /// can be reduced to the preceding multiple of the GCD of the coefficients,
  /// i.e.,
  ///  64*i - 100 >= 0  =>  64*i - 128 >= 0 (since 'i' is an integer). This is a
  /// fast method (linear in the number of coefficients).
  void GCDTightenInequalities();

  /// Normalized each constraints by the GCD of its coefficients.
  void normalizeConstraintsByGCD();

  /// Removes identifiers in column range [idStart, idLimit), and copies any
  /// remaining valid data into place, updates member variables, and resizes
  /// arrays as needed.
  void removeIdRange(unsigned idStart, unsigned idLimit);

  /// Coefficients of affine equalities (in == 0 form).
  SmallVector<int64_t, 64> equalities;

  /// Coefficients of affine inequalities (in >= 0 form).
  SmallVector<int64_t, 64> inequalities;

  /// Number of columns reserved. Actual ones in used are returned by
  /// getNumCols().
  unsigned numReservedCols;

  /// Total number of identifiers.
  unsigned numIds;

  /// Number of identifiers corresponding to real dimensions.
  unsigned numDims;

  /// Number of identifiers corresponding to symbols (unknown but constant for
  /// analysis).
  unsigned numSymbols;

  /// Values corresponding to the (column) identifiers of this constraint
  /// system appearing in the order the identifiers correspond to columns.
  /// Temporary ones or those that aren't associated to any Value are set to
  /// None.
  SmallVector<Optional<Value *>, 8> ids;

  /// A parameter that controls detection of an unrealistic number of
  /// constraints. If the number of constraints is this many times the number of
  /// variables, we consider such a system out of line with the intended use
  /// case of FlatAffineConstraints.
  // The rationale for 32 is that in the typical simplest of cases, an
  // identifier is expected to have one lower bound and one upper bound
  // constraint. With a level of tiling or a connection to another identifier
  // through a div or mod, an extra pair of bounds gets added. As a limit, we
  // don't expect an identifier to have more than 32 lower/upper/equality
  // constraints. This is conservatively set low and can be raised if needed.
  constexpr static unsigned kExplosionFactor = 32;
};

/// Simplify an affine expression by flattening and some amount of
/// simple analysis. This has complexity linear in the number of nodes in
/// 'expr'. Returns the simplified expression, which is the same as the input
///  expression if it can't be simplified.
AffineExpr simplifyAffineExpr(AffineExpr expr, unsigned numDims,
                              unsigned numSymbols);

/// Flattens 'expr' into 'flattenedExpr'. Returns failure if 'expr' could not be
/// flattened (i.e., semi-affine is not yet handled). 'cst' contains constraints
/// that connect newly introduced local identifiers to existing dimensional and
/// symbolic identifiers. See documentation for AffineExprFlattener on how
/// mod's and div's are flattened.
LogicalResult
getFlattenedAffineExpr(AffineExpr expr, unsigned numDims, unsigned numSymbols,
                       llvm::SmallVectorImpl<int64_t> *flattenedExpr,
                       FlatAffineConstraints *cst = nullptr);

/// Flattens the result expressions of the map to their corresponding flattened
/// forms and set in 'flattenedExprs'. Returns failure if any expression in the
/// map could not be flattened (i.e., semi-affine is not yet handled). 'cst'
/// contains constraints that connect newly introduced local identifiers to
/// existing dimensional and / symbolic identifiers. See documentation for
/// AffineExprFlattener on how mod's and div's are flattened. For all affine
/// expressions that share the same operands (like those of an affine map), this
/// method should be used instead of repeatedly calling getFlattenedAffineExpr
/// since local variables added to deal with div's and mod's will be reused
/// across expressions.
LogicalResult getFlattenedAffineExprs(
    AffineMap map, std::vector<llvm::SmallVector<int64_t, 8>> *flattenedExprs,
    FlatAffineConstraints *cst = nullptr);
LogicalResult getFlattenedAffineExprs(
    IntegerSet set, std::vector<llvm::SmallVector<int64_t, 8>> *flattenedExprs,
    FlatAffineConstraints *cst = nullptr);

} // end namespace mlir.

#endif // MLIR_ANALYSIS_AFFINE_STRUCTURES_H
