//===- AffineStructures.cpp - MLIR Affine Structures Class-----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Structures for affine/polyhedral analysis of MLIR functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "affine-structures"

using namespace mlir;
using llvm::SmallDenseMap;
using llvm::SmallDenseSet;

namespace {

// See comments for SimpleAffineExprFlattener.
// An AffineExprFlattener extends a SimpleAffineExprFlattener by recording
// constraint information associated with mod's, floordiv's, and ceildiv's
// in FlatAffineConstraints 'localVarCst'.
struct AffineExprFlattener : public SimpleAffineExprFlattener {
public:
  // Constraints connecting newly introduced local variables (for mod's and
  // div's) to existing (dimensional and symbolic) ones. These are always
  // inequalities.
  FlatAffineConstraints localVarCst;

  AffineExprFlattener(unsigned nDims, unsigned nSymbols, MLIRContext *ctx)
      : SimpleAffineExprFlattener(nDims, nSymbols) {
    localVarCst.reset(nDims, nSymbols, /*numLocals=*/0);
  }

private:
  // Add a local identifier (needed to flatten a mod, floordiv, ceildiv expr).
  // The local identifier added is always a floordiv of a pure add/mul affine
  // function of other identifiers, coefficients of which are specified in
  // `dividend' and with respect to the positive constant `divisor'. localExpr
  // is the simplified tree expression (AffineExpr) corresponding to the
  // quantifier.
  void addLocalFloorDivId(ArrayRef<int64_t> dividend, int64_t divisor,
                          AffineExpr localExpr) override {
    SimpleAffineExprFlattener::addLocalFloorDivId(dividend, divisor, localExpr);
    // Update localVarCst.
    localVarCst.addLocalFloorDiv(dividend, divisor);
  }
};

} // end anonymous namespace

// Flattens the expressions in map. Returns failure if 'expr' was unable to be
// flattened (i.e., semi-affine expressions not handled yet).
static LogicalResult
getFlattenedAffineExprs(ArrayRef<AffineExpr> exprs, unsigned numDims,
                        unsigned numSymbols,
                        std::vector<SmallVector<int64_t, 8>> *flattenedExprs,
                        FlatAffineConstraints *localVarCst) {
  if (exprs.empty()) {
    localVarCst->reset(numDims, numSymbols);
    return success();
  }

  AffineExprFlattener flattener(numDims, numSymbols, exprs[0].getContext());
  // Use the same flattener to simplify each expression successively. This way
  // local identifiers / expressions are shared.
  for (auto expr : exprs) {
    if (!expr.isPureAffine())
      return failure();

    flattener.walkPostOrder(expr);
  }

  assert(flattener.operandExprStack.size() == exprs.size());
  flattenedExprs->clear();
  flattenedExprs->assign(flattener.operandExprStack.begin(),
                         flattener.operandExprStack.end());

  if (localVarCst) {
    localVarCst->clearAndCopyFrom(flattener.localVarCst);
  }

  return success();
}

// Flattens 'expr' into 'flattenedExpr'. Returns failure if 'expr' was unable to
// be flattened (semi-affine expressions not handled yet).
LogicalResult
mlir::getFlattenedAffineExpr(AffineExpr expr, unsigned numDims,
                             unsigned numSymbols,
                             SmallVectorImpl<int64_t> *flattenedExpr,
                             FlatAffineConstraints *localVarCst) {
  std::vector<SmallVector<int64_t, 8>> flattenedExprs;
  LogicalResult ret = ::getFlattenedAffineExprs({expr}, numDims, numSymbols,
                                                &flattenedExprs, localVarCst);
  *flattenedExpr = flattenedExprs[0];
  return ret;
}

/// Flattens the expressions in map. Returns failure if 'expr' was unable to be
/// flattened (i.e., semi-affine expressions not handled yet).
LogicalResult mlir::getFlattenedAffineExprs(
    AffineMap map, std::vector<SmallVector<int64_t, 8>> *flattenedExprs,
    FlatAffineConstraints *localVarCst) {
  if (map.getNumResults() == 0) {
    localVarCst->reset(map.getNumDims(), map.getNumSymbols());
    return success();
  }
  return ::getFlattenedAffineExprs(map.getResults(), map.getNumDims(),
                                   map.getNumSymbols(), flattenedExprs,
                                   localVarCst);
}

LogicalResult mlir::getFlattenedAffineExprs(
    IntegerSet set, std::vector<SmallVector<int64_t, 8>> *flattenedExprs,
    FlatAffineConstraints *localVarCst) {
  if (set.getNumConstraints() == 0) {
    localVarCst->reset(set.getNumDims(), set.getNumSymbols());
    return success();
  }
  return ::getFlattenedAffineExprs(set.getConstraints(), set.getNumDims(),
                                   set.getNumSymbols(), flattenedExprs,
                                   localVarCst);
}

//===----------------------------------------------------------------------===//
// MutableAffineMap.
//===----------------------------------------------------------------------===//

MutableAffineMap::MutableAffineMap(AffineMap map)
    : numDims(map.getNumDims()), numSymbols(map.getNumSymbols()),
      // A map always has at least 1 result by construction
      context(map.getResult(0).getContext()) {
  for (auto result : map.getResults())
    results.push_back(result);
}

void MutableAffineMap::reset(AffineMap map) {
  results.clear();
  numDims = map.getNumDims();
  numSymbols = map.getNumSymbols();
  // A map always has at least 1 result by construction
  context = map.getResult(0).getContext();
  for (auto result : map.getResults())
    results.push_back(result);
}

bool MutableAffineMap::isMultipleOf(unsigned idx, int64_t factor) const {
  if (results[idx].isMultipleOf(factor))
    return true;

  // TODO(bondhugula): use simplifyAffineExpr and FlatAffineConstraints to
  // complete this (for a more powerful analysis).
  return false;
}

// Simplifies the result affine expressions of this map. The expressions have to
// be pure for the simplification implemented.
void MutableAffineMap::simplify() {
  // Simplify each of the results if possible.
  // TODO(ntv): functional-style map
  for (unsigned i = 0, e = getNumResults(); i < e; i++) {
    results[i] = simplifyAffineExpr(getResult(i), numDims, numSymbols);
  }
}

AffineMap MutableAffineMap::getAffineMap() const {
  return AffineMap::get(numDims, numSymbols, results);
}

MutableIntegerSet::MutableIntegerSet(IntegerSet set, MLIRContext *context)
    : numDims(set.getNumDims()), numSymbols(set.getNumSymbols()) {
  // TODO(bondhugula)
}

// Universal set.
MutableIntegerSet::MutableIntegerSet(unsigned numDims, unsigned numSymbols,
                                     MLIRContext *context)
    : numDims(numDims), numSymbols(numSymbols) {}

//===----------------------------------------------------------------------===//
// AffineValueMap.
//===----------------------------------------------------------------------===//

AffineValueMap::AffineValueMap(AffineMap map, ArrayRef<Value> operands,
                               ArrayRef<Value> results)
    : map(map), operands(operands.begin(), operands.end()),
      results(results.begin(), results.end()) {}

AffineValueMap::AffineValueMap(AffineApplyOp applyOp)
    : map(applyOp.getAffineMap()),
      operands(applyOp.operand_begin(), applyOp.operand_end()) {
  results.push_back(applyOp.getResult());
}

AffineValueMap::AffineValueMap(AffineBound bound)
    : map(bound.getMap()),
      operands(bound.operand_begin(), bound.operand_end()) {}

void AffineValueMap::reset(AffineMap map, ArrayRef<Value> operands,
                           ArrayRef<Value> results) {
  this->map.reset(map);
  this->operands.assign(operands.begin(), operands.end());
  this->results.assign(results.begin(), results.end());
}

void AffineValueMap::difference(const AffineValueMap &a,
                                const AffineValueMap &b, AffineValueMap *res) {
  assert(a.getNumResults() == b.getNumResults() && "invalid inputs");

  // Fully compose A's map + operands.
  auto aMap = a.getAffineMap();
  SmallVector<Value, 4> aOperands(a.getOperands().begin(),
                                  a.getOperands().end());
  fullyComposeAffineMapAndOperands(&aMap, &aOperands);

  // Use the affine apply normalizer to get B's map into A's coordinate space.
  AffineApplyNormalizer normalizer(aMap, aOperands);
  SmallVector<Value, 4> bOperands(b.getOperands().begin(),
                                  b.getOperands().end());
  auto bMap = b.getAffineMap();
  normalizer.normalize(&bMap, &bOperands);

  assert(std::equal(bOperands.begin(), bOperands.end(),
                    normalizer.getOperands().begin()) &&
         "operands are expected to be the same after normalization");

  // Construct the difference expressions.
  SmallVector<AffineExpr, 4> diffExprs;
  diffExprs.reserve(a.getNumResults());
  for (unsigned i = 0, e = bMap.getNumResults(); i < e; ++i)
    diffExprs.push_back(normalizer.getAffineMap().getResult(i) -
                        bMap.getResult(i));

  auto diffMap = AffineMap::get(normalizer.getNumDims(),
                                normalizer.getNumSymbols(), diffExprs);
  canonicalizeMapAndOperands(&diffMap, &bOperands);
  diffMap = simplifyAffineMap(diffMap);
  res->reset(diffMap, bOperands);
}

// Returns true and sets 'indexOfMatch' if 'valueToMatch' is found in
// 'valuesToSearch' beginning at 'indexStart'. Returns false otherwise.
static bool findIndex(Value valueToMatch, ArrayRef<Value> valuesToSearch,
                      unsigned indexStart, unsigned *indexOfMatch) {
  unsigned size = valuesToSearch.size();
  for (unsigned i = indexStart; i < size; ++i) {
    if (valueToMatch == valuesToSearch[i]) {
      *indexOfMatch = i;
      return true;
    }
  }
  return false;
}

inline bool AffineValueMap::isMultipleOf(unsigned idx, int64_t factor) const {
  return map.isMultipleOf(idx, factor);
}

/// This method uses the invariant that operands are always positionally aligned
/// with the AffineDimExpr in the underlying AffineMap.
bool AffineValueMap::isFunctionOf(unsigned idx, Value value) const {
  unsigned index;
  if (!findIndex(value, operands, /*indexStart=*/0, &index)) {
    return false;
  }
  auto expr = const_cast<AffineValueMap *>(this)->getAffineMap().getResult(idx);
  // TODO(ntv): this is better implemented on a flattened representation.
  // At least for now it is conservative.
  return expr.isFunctionOfDim(index);
}

Value AffineValueMap::getOperand(unsigned i) const {
  return static_cast<Value>(operands[i]);
}

ArrayRef<Value> AffineValueMap::getOperands() const {
  return ArrayRef<Value>(operands);
}

AffineMap AffineValueMap::getAffineMap() const { return map.getAffineMap(); }

AffineValueMap::~AffineValueMap() {}

//===----------------------------------------------------------------------===//
// FlatAffineConstraints.
//===----------------------------------------------------------------------===//

// Copy constructor.
FlatAffineConstraints::FlatAffineConstraints(
    const FlatAffineConstraints &other) {
  numReservedCols = other.numReservedCols;
  numDims = other.getNumDimIds();
  numSymbols = other.getNumSymbolIds();
  numIds = other.getNumIds();

  auto otherIds = other.getIds();
  ids.reserve(numReservedCols);
  ids.append(otherIds.begin(), otherIds.end());

  unsigned numReservedEqualities = other.getNumReservedEqualities();
  unsigned numReservedInequalities = other.getNumReservedInequalities();

  equalities.reserve(numReservedEqualities * numReservedCols);
  inequalities.reserve(numReservedInequalities * numReservedCols);

  for (unsigned r = 0, e = other.getNumInequalities(); r < e; r++) {
    addInequality(other.getInequality(r));
  }
  for (unsigned r = 0, e = other.getNumEqualities(); r < e; r++) {
    addEquality(other.getEquality(r));
  }
}

// Clones this object.
std::unique_ptr<FlatAffineConstraints> FlatAffineConstraints::clone() const {
  return std::make_unique<FlatAffineConstraints>(*this);
}

// Construct from an IntegerSet.
FlatAffineConstraints::FlatAffineConstraints(IntegerSet set)
    : numReservedCols(set.getNumInputs() + 1),
      numIds(set.getNumDims() + set.getNumSymbols()), numDims(set.getNumDims()),
      numSymbols(set.getNumSymbols()) {
  equalities.reserve(set.getNumEqualities() * numReservedCols);
  inequalities.reserve(set.getNumInequalities() * numReservedCols);
  ids.resize(numIds, None);

  // Flatten expressions and add them to the constraint system.
  std::vector<SmallVector<int64_t, 8>> flatExprs;
  FlatAffineConstraints localVarCst;
  if (failed(getFlattenedAffineExprs(set, &flatExprs, &localVarCst))) {
    assert(false && "flattening unimplemented for semi-affine integer sets");
    return;
  }
  assert(flatExprs.size() == set.getNumConstraints());
  for (unsigned l = 0, e = localVarCst.getNumLocalIds(); l < e; l++) {
    addLocalId(getNumLocalIds());
  }

  for (unsigned i = 0, e = flatExprs.size(); i < e; ++i) {
    const auto &flatExpr = flatExprs[i];
    assert(flatExpr.size() == getNumCols());
    if (set.getEqFlags()[i]) {
      addEquality(flatExpr);
    } else {
      addInequality(flatExpr);
    }
  }
  // Add the other constraints involving local id's from flattening.
  append(localVarCst);
}

void FlatAffineConstraints::reset(unsigned numReservedInequalities,
                                  unsigned numReservedEqualities,
                                  unsigned newNumReservedCols,
                                  unsigned newNumDims, unsigned newNumSymbols,
                                  unsigned newNumLocals,
                                  ArrayRef<Value> idArgs) {
  assert(newNumReservedCols >= newNumDims + newNumSymbols + newNumLocals + 1 &&
         "minimum 1 column");
  numReservedCols = newNumReservedCols;
  numDims = newNumDims;
  numSymbols = newNumSymbols;
  numIds = numDims + numSymbols + newNumLocals;
  assert(idArgs.empty() || idArgs.size() == numIds);

  clearConstraints();
  if (numReservedEqualities >= 1)
    equalities.reserve(newNumReservedCols * numReservedEqualities);
  if (numReservedInequalities >= 1)
    inequalities.reserve(newNumReservedCols * numReservedInequalities);
  if (idArgs.empty()) {
    ids.resize(numIds, None);
  } else {
    ids.assign(idArgs.begin(), idArgs.end());
  }
}

void FlatAffineConstraints::reset(unsigned newNumDims, unsigned newNumSymbols,
                                  unsigned newNumLocals,
                                  ArrayRef<Value> idArgs) {
  reset(0, 0, newNumDims + newNumSymbols + newNumLocals + 1, newNumDims,
        newNumSymbols, newNumLocals, idArgs);
}

void FlatAffineConstraints::append(const FlatAffineConstraints &other) {
  assert(other.getNumCols() == getNumCols());
  assert(other.getNumDimIds() == getNumDimIds());
  assert(other.getNumSymbolIds() == getNumSymbolIds());

  inequalities.reserve(inequalities.size() +
                       other.getNumInequalities() * numReservedCols);
  equalities.reserve(equalities.size() +
                     other.getNumEqualities() * numReservedCols);

  for (unsigned r = 0, e = other.getNumInequalities(); r < e; r++) {
    addInequality(other.getInequality(r));
  }
  for (unsigned r = 0, e = other.getNumEqualities(); r < e; r++) {
    addEquality(other.getEquality(r));
  }
}

void FlatAffineConstraints::addLocalId(unsigned pos) {
  addId(IdKind::Local, pos);
}

void FlatAffineConstraints::addDimId(unsigned pos, Value id) {
  addId(IdKind::Dimension, pos, id);
}

void FlatAffineConstraints::addSymbolId(unsigned pos, Value id) {
  addId(IdKind::Symbol, pos, id);
}

/// Adds a dimensional identifier. The added column is initialized to
/// zero.
void FlatAffineConstraints::addId(IdKind kind, unsigned pos, Value id) {
  if (kind == IdKind::Dimension) {
    assert(pos <= getNumDimIds());
  } else if (kind == IdKind::Symbol) {
    assert(pos <= getNumSymbolIds());
  } else {
    assert(pos <= getNumLocalIds());
  }

  unsigned oldNumReservedCols = numReservedCols;

  // Check if a resize is necessary.
  if (getNumCols() + 1 > numReservedCols) {
    equalities.resize(getNumEqualities() * (getNumCols() + 1));
    inequalities.resize(getNumInequalities() * (getNumCols() + 1));
    numReservedCols++;
  }

  int absolutePos;

  if (kind == IdKind::Dimension) {
    absolutePos = pos;
    numDims++;
  } else if (kind == IdKind::Symbol) {
    absolutePos = pos + getNumDimIds();
    numSymbols++;
  } else {
    absolutePos = pos + getNumDimIds() + getNumSymbolIds();
  }
  numIds++;

  // Note that getNumCols() now will already return the new size, which will be
  // at least one.
  int numInequalities = static_cast<int>(getNumInequalities());
  int numEqualities = static_cast<int>(getNumEqualities());
  int numCols = static_cast<int>(getNumCols());
  for (int r = numInequalities - 1; r >= 0; r--) {
    for (int c = numCols - 2; c >= 0; c--) {
      if (c < absolutePos)
        atIneq(r, c) = inequalities[r * oldNumReservedCols + c];
      else
        atIneq(r, c + 1) = inequalities[r * oldNumReservedCols + c];
    }
    atIneq(r, absolutePos) = 0;
  }

  for (int r = numEqualities - 1; r >= 0; r--) {
    for (int c = numCols - 2; c >= 0; c--) {
      // All values in column absolutePositions < absolutePos have the same
      // coordinates in the 2-d view of the coefficient buffer.
      if (c < absolutePos)
        atEq(r, c) = equalities[r * oldNumReservedCols + c];
      else
        // Those at absolutePosition >= absolutePos, get a shifted
        // absolutePosition.
        atEq(r, c + 1) = equalities[r * oldNumReservedCols + c];
    }
    // Initialize added dimension to zero.
    atEq(r, absolutePos) = 0;
  }

  // If an 'id' is provided, insert it; otherwise use None.
  if (id) {
    ids.insert(ids.begin() + absolutePos, id);
  } else {
    ids.insert(ids.begin() + absolutePos, None);
  }
  assert(ids.size() == getNumIds());
}

/// Checks if two constraint systems are in the same space, i.e., if they are
/// associated with the same set of identifiers, appearing in the same order.
static bool areIdsAligned(const FlatAffineConstraints &A,
                          const FlatAffineConstraints &B) {
  return A.getNumDimIds() == B.getNumDimIds() &&
         A.getNumSymbolIds() == B.getNumSymbolIds() &&
         A.getNumIds() == B.getNumIds() && A.getIds().equals(B.getIds());
}

/// Calls areIdsAligned to check if two constraint systems have the same set
/// of identifiers in the same order.
bool FlatAffineConstraints::areIdsAlignedWithOther(
    const FlatAffineConstraints &other) {
  return areIdsAligned(*this, other);
}

/// Checks if the SSA values associated with `cst''s identifiers are unique.
static bool LLVM_ATTRIBUTE_UNUSED
areIdsUnique(const FlatAffineConstraints &cst) {
  SmallPtrSet<Value, 8> uniqueIds;
  for (auto id : cst.getIds()) {
    if (id.hasValue() && !uniqueIds.insert(id.getValue()).second)
      return false;
  }
  return true;
}

// Swap the posA^th identifier with the posB^th identifier.
static void swapId(FlatAffineConstraints *A, unsigned posA, unsigned posB) {
  assert(posA < A->getNumIds() && "invalid position A");
  assert(posB < A->getNumIds() && "invalid position B");

  if (posA == posB)
    return;

  for (unsigned r = 0, e = A->getNumInequalities(); r < e; r++) {
    std::swap(A->atIneq(r, posA), A->atIneq(r, posB));
  }
  for (unsigned r = 0, e = A->getNumEqualities(); r < e; r++) {
    std::swap(A->atEq(r, posA), A->atEq(r, posB));
  }
  std::swap(A->getId(posA), A->getId(posB));
}

/// Merge and align the identifiers of A and B starting at 'offset', so that
/// both constraint systems get the union of the contained identifiers that is
/// dimension-wise and symbol-wise unique; both constraint systems are updated
/// so that they have the union of all identifiers, with A's original
/// identifiers appearing first followed by any of B's identifiers that didn't
/// appear in A. Local identifiers of each system are by design separate/local
/// and are placed one after other (A's followed by B's).
//  Eg: Input: A has ((%i %j) [%M %N]) and B has (%k, %j) [%P, %N, %M])
//      Output: both A, B have (%i, %j, %k) [%M, %N, %P]
//
static void mergeAndAlignIds(unsigned offset, FlatAffineConstraints *A,
                             FlatAffineConstraints *B) {
  assert(offset <= A->getNumDimIds() && offset <= B->getNumDimIds());
  // A merge/align isn't meaningful if a cst's ids aren't distinct.
  assert(areIdsUnique(*A) && "A's id values aren't unique");
  assert(areIdsUnique(*B) && "B's id values aren't unique");

  assert(std::all_of(A->getIds().begin() + offset,
                     A->getIds().begin() + A->getNumDimAndSymbolIds(),
                     [](Optional<Value> id) { return id.hasValue(); }));

  assert(std::all_of(B->getIds().begin() + offset,
                     B->getIds().begin() + B->getNumDimAndSymbolIds(),
                     [](Optional<Value> id) { return id.hasValue(); }));

  // Place local id's of A after local id's of B.
  for (unsigned l = 0, e = A->getNumLocalIds(); l < e; l++) {
    B->addLocalId(0);
  }
  for (unsigned t = 0, e = B->getNumLocalIds() - A->getNumLocalIds(); t < e;
       t++) {
    A->addLocalId(A->getNumLocalIds());
  }

  SmallVector<Value, 4> aDimValues, aSymValues;
  A->getIdValues(offset, A->getNumDimIds(), &aDimValues);
  A->getIdValues(A->getNumDimIds(), A->getNumDimAndSymbolIds(), &aSymValues);
  {
    // Merge dims from A into B.
    unsigned d = offset;
    for (auto aDimValue : aDimValues) {
      unsigned loc;
      if (B->findId(*aDimValue, &loc)) {
        assert(loc >= offset && "A's dim appears in B's aligned range");
        assert(loc < B->getNumDimIds() &&
               "A's dim appears in B's non-dim position");
        swapId(B, d, loc);
      } else {
        B->addDimId(d);
        B->setIdValue(d, aDimValue);
      }
      d++;
    }

    // Dimensions that are in B, but not in A, are added at the end.
    for (unsigned t = A->getNumDimIds(), e = B->getNumDimIds(); t < e; t++) {
      A->addDimId(A->getNumDimIds());
      A->setIdValue(A->getNumDimIds() - 1, B->getIdValue(t));
    }
  }
  {
    // Merge symbols: merge A's symbols into B first.
    unsigned s = B->getNumDimIds();
    for (auto aSymValue : aSymValues) {
      unsigned loc;
      if (B->findId(*aSymValue, &loc)) {
        assert(loc >= B->getNumDimIds() && loc < B->getNumDimAndSymbolIds() &&
               "A's symbol appears in B's non-symbol position");
        swapId(B, s, loc);
      } else {
        B->addSymbolId(s - B->getNumDimIds());
        B->setIdValue(s, aSymValue);
      }
      s++;
    }
    // Symbols that are in B, but not in A, are added at the end.
    for (unsigned t = A->getNumDimAndSymbolIds(),
                  e = B->getNumDimAndSymbolIds();
         t < e; t++) {
      A->addSymbolId(A->getNumSymbolIds());
      A->setIdValue(A->getNumDimAndSymbolIds() - 1, B->getIdValue(t));
    }
  }
  assert(areIdsAligned(*A, *B) && "IDs expected to be aligned");
}

// Call 'mergeAndAlignIds' to align constraint systems of 'this' and 'other'.
void FlatAffineConstraints::mergeAndAlignIdsWithOther(
    unsigned offset, FlatAffineConstraints *other) {
  mergeAndAlignIds(offset, this, other);
}

// This routine may add additional local variables if the flattened expression
// corresponding to the map has such variables due to mod's, ceildiv's, and
// floordiv's in it.
LogicalResult FlatAffineConstraints::composeMap(const AffineValueMap *vMap) {
  std::vector<SmallVector<int64_t, 8>> flatExprs;
  FlatAffineConstraints localCst;
  if (failed(getFlattenedAffineExprs(vMap->getAffineMap(), &flatExprs,
                                     &localCst))) {
    LLVM_DEBUG(llvm::dbgs()
               << "composition unimplemented for semi-affine maps\n");
    return failure();
  }
  assert(flatExprs.size() == vMap->getNumResults());

  // Add localCst information.
  if (localCst.getNumLocalIds() > 0) {
    localCst.setIdValues(0, /*end=*/localCst.getNumDimAndSymbolIds(),
                         /*values=*/vMap->getOperands());
    // Align localCst and this.
    mergeAndAlignIds(/*offset=*/0, &localCst, this);
    // Finally, append localCst to this constraint set.
    append(localCst);
  }

  // Add dimensions corresponding to the map's results.
  for (unsigned t = 0, e = vMap->getNumResults(); t < e; t++) {
    // TODO: Consider using a batched version to add a range of IDs.
    addDimId(0);
  }

  // We add one equality for each result connecting the result dim of the map to
  // the other identifiers.
  // For eg: if the expression is 16*i0 + i1, and this is the r^th
  // iteration/result of the value map, we are adding the equality:
  //  d_r - 16*i0 - i1 = 0. Hence, when flattening say (i0 + 1, i0 + 8*i2), we
  //  add two equalities overall: d_0 - i0 - 1 == 0, d1 - i0 - 8*i2 == 0.
  for (unsigned r = 0, e = flatExprs.size(); r < e; r++) {
    const auto &flatExpr = flatExprs[r];
    assert(flatExpr.size() >= vMap->getNumOperands() + 1);

    // eqToAdd is the equality corresponding to the flattened affine expression.
    SmallVector<int64_t, 8> eqToAdd(getNumCols(), 0);
    // Set the coefficient for this result to one.
    eqToAdd[r] = 1;

    // Dims and symbols.
    for (unsigned i = 0, e = vMap->getNumOperands(); i < e; i++) {
      unsigned loc;
      bool ret = findId(*vMap->getOperand(i), &loc);
      assert(ret && "value map's id can't be found");
      (void)ret;
      // Negate 'eq[r]' since the newly added dimension will be set to this one.
      eqToAdd[loc] = -flatExpr[i];
    }
    // Local vars common to eq and localCst are at the beginning.
    unsigned j = getNumDimIds() + getNumSymbolIds();
    unsigned end = flatExpr.size() - 1;
    for (unsigned i = vMap->getNumOperands(); i < end; i++, j++) {
      eqToAdd[j] = -flatExpr[i];
    }

    // Constant term.
    eqToAdd[getNumCols() - 1] = -flatExpr[flatExpr.size() - 1];

    // Add the equality connecting the result of the map to this constraint set.
    addEquality(eqToAdd);
  }

  return success();
}

// Similar to composeMap except that no Value's need be associated with the
// constraint system nor are they looked at -- since the dimensions and
// symbols of 'other' are expected to correspond 1:1 to 'this' system. It
// is thus not convenient to share code with composeMap.
LogicalResult FlatAffineConstraints::composeMatchingMap(AffineMap other) {
  assert(other.getNumDims() == getNumDimIds() && "dim mismatch");
  assert(other.getNumSymbols() == getNumSymbolIds() && "symbol mismatch");

  std::vector<SmallVector<int64_t, 8>> flatExprs;
  FlatAffineConstraints localCst;
  if (failed(getFlattenedAffineExprs(other, &flatExprs, &localCst))) {
    LLVM_DEBUG(llvm::dbgs()
               << "composition unimplemented for semi-affine maps\n");
    return failure();
  }
  assert(flatExprs.size() == other.getNumResults());

  // Add localCst information.
  if (localCst.getNumLocalIds() > 0) {
    // Place local id's of A after local id's of B.
    for (unsigned l = 0, e = localCst.getNumLocalIds(); l < e; l++) {
      addLocalId(0);
    }
    // Finally, append localCst to this constraint set.
    append(localCst);
  }

  // Add dimensions corresponding to the map's results.
  for (unsigned t = 0, e = other.getNumResults(); t < e; t++) {
    addDimId(0);
  }

  // We add one equality for each result connecting the result dim of the map to
  // the other identifiers.
  // For eg: if the expression is 16*i0 + i1, and this is the r^th
  // iteration/result of the value map, we are adding the equality:
  //  d_r - 16*i0 - i1 = 0. Hence, when flattening say (i0 + 1, i0 + 8*i2), we
  //  add two equalities overall: d_0 - i0 - 1 == 0, d1 - i0 - 8*i2 == 0.
  for (unsigned r = 0, e = flatExprs.size(); r < e; r++) {
    const auto &flatExpr = flatExprs[r];
    assert(flatExpr.size() >= other.getNumInputs() + 1);

    // eqToAdd is the equality corresponding to the flattened affine expression.
    SmallVector<int64_t, 8> eqToAdd(getNumCols(), 0);
    // Set the coefficient for this result to one.
    eqToAdd[r] = 1;

    // Dims and symbols.
    for (unsigned i = 0, f = other.getNumInputs(); i < f; i++) {
      // Negate 'eq[r]' since the newly added dimension will be set to this one.
      eqToAdd[e + i] = -flatExpr[i];
    }
    // Local vars common to eq and localCst are at the beginning.
    unsigned j = getNumDimIds() + getNumSymbolIds();
    unsigned end = flatExpr.size() - 1;
    for (unsigned i = other.getNumInputs(); i < end; i++, j++) {
      eqToAdd[j] = -flatExpr[i];
    }

    // Constant term.
    eqToAdd[getNumCols() - 1] = -flatExpr[flatExpr.size() - 1];

    // Add the equality connecting the result of the map to this constraint set.
    addEquality(eqToAdd);
  }

  return success();
}

// Turn a dimension into a symbol.
static void turnDimIntoSymbol(FlatAffineConstraints *cst, Value id) {
  unsigned pos;
  if (cst->findId(id, &pos) && pos < cst->getNumDimIds()) {
    swapId(cst, pos, cst->getNumDimIds() - 1);
    cst->setDimSymbolSeparation(cst->getNumSymbolIds() + 1);
  }
}

// Turn a symbol into a dimension.
static void turnSymbolIntoDim(FlatAffineConstraints *cst, Value id) {
  unsigned pos;
  if (cst->findId(id, &pos) && pos >= cst->getNumDimIds() &&
      pos < cst->getNumDimAndSymbolIds()) {
    swapId(cst, pos, cst->getNumDimIds());
    cst->setDimSymbolSeparation(cst->getNumSymbolIds() - 1);
  }
}

// Changes all symbol identifiers which are loop IVs to dim identifiers.
void FlatAffineConstraints::convertLoopIVSymbolsToDims() {
  // Gather all symbols which are loop IVs.
  SmallVector<Value, 4> loopIVs;
  for (unsigned i = getNumDimIds(), e = getNumDimAndSymbolIds(); i < e; i++) {
    if (ids[i].hasValue() && getForInductionVarOwner(ids[i].getValue()))
      loopIVs.push_back(ids[i].getValue());
  }
  // Turn each symbol in 'loopIVs' into a dim identifier.
  for (auto iv : loopIVs) {
    turnSymbolIntoDim(this, *iv);
  }
}

void FlatAffineConstraints::addInductionVarOrTerminalSymbol(Value id) {
  if (containsId(*id))
    return;

  // Caller is expected to fully compose map/operands if necessary.
  assert((isTopLevelValue(id) || isForInductionVar(id)) &&
         "non-terminal symbol / loop IV expected");
  // Outer loop IVs could be used in forOp's bounds.
  if (auto loop = getForInductionVarOwner(id)) {
    addDimId(getNumDimIds(), id);
    if (failed(this->addAffineForOpDomain(loop)))
      LLVM_DEBUG(
          loop.emitWarning("failed to add domain info to constraint system"));
    return;
  }
  // Add top level symbol.
  addSymbolId(getNumSymbolIds(), id);
  // Check if the symbol is a constant.
  if (auto constOp = dyn_cast_or_null<ConstantIndexOp>(id->getDefiningOp()))
    setIdToConstant(*id, constOp.getValue());
}

LogicalResult FlatAffineConstraints::addAffineForOpDomain(AffineForOp forOp) {
  unsigned pos;
  // Pre-condition for this method.
  if (!findId(*forOp.getInductionVar(), &pos)) {
    assert(false && "Value not found");
    return failure();
  }

  int64_t step = forOp.getStep();
  if (step != 1) {
    if (!forOp.hasConstantLowerBound())
      forOp.emitWarning("domain conservatively approximated");
    else {
      // Add constraints for the stride.
      // (iv - lb) % step = 0 can be written as:
      // (iv - lb) - step * q = 0 where q = (iv - lb) / step.
      // Add local variable 'q' and add the above equality.
      // The first constraint is q = (iv - lb) floordiv step
      SmallVector<int64_t, 8> dividend(getNumCols(), 0);
      int64_t lb = forOp.getConstantLowerBound();
      dividend[pos] = 1;
      dividend.back() -= lb;
      addLocalFloorDiv(dividend, step);
      // Second constraint: (iv - lb) - step * q = 0.
      SmallVector<int64_t, 8> eq(getNumCols(), 0);
      eq[pos] = 1;
      eq.back() -= lb;
      // For the local var just added above.
      eq[getNumCols() - 2] = -step;
      addEquality(eq);
    }
  }

  if (forOp.hasConstantLowerBound()) {
    addConstantLowerBound(pos, forOp.getConstantLowerBound());
  } else {
    // Non-constant lower bound case.
    SmallVector<Value, 4> lbOperands(forOp.getLowerBoundOperands().begin(),
                                     forOp.getLowerBoundOperands().end());
    if (failed(addLowerOrUpperBound(pos, forOp.getLowerBoundMap(), lbOperands,
                                    /*eq=*/false, /*lower=*/true)))
      return failure();
  }

  if (forOp.hasConstantUpperBound()) {
    addConstantUpperBound(pos, forOp.getConstantUpperBound() - 1);
    return success();
  }
  // Non-constant upper bound case.
  SmallVector<Value, 4> ubOperands(forOp.getUpperBoundOperands().begin(),
                                   forOp.getUpperBoundOperands().end());
  return addLowerOrUpperBound(pos, forOp.getUpperBoundMap(), ubOperands,
                              /*eq=*/false, /*lower=*/false);
}

// Searches for a constraint with a non-zero coefficient at 'colIdx' in
// equality (isEq=true) or inequality (isEq=false) constraints.
// Returns true and sets row found in search in 'rowIdx'.
// Returns false otherwise.
static bool
findConstraintWithNonZeroAt(const FlatAffineConstraints &constraints,
                            unsigned colIdx, bool isEq, unsigned *rowIdx) {
  auto at = [&](unsigned rowIdx) -> int64_t {
    return isEq ? constraints.atEq(rowIdx, colIdx)
                : constraints.atIneq(rowIdx, colIdx);
  };
  unsigned e =
      isEq ? constraints.getNumEqualities() : constraints.getNumInequalities();
  for (*rowIdx = 0; *rowIdx < e; ++(*rowIdx)) {
    if (at(*rowIdx) != 0) {
      return true;
    }
  }
  return false;
}

// Normalizes the coefficient values across all columns in 'rowIDx' by their
// GCD in equality or inequality constraints as specified by 'isEq'.
template <bool isEq>
static void normalizeConstraintByGCD(FlatAffineConstraints *constraints,
                                     unsigned rowIdx) {
  auto at = [&](unsigned colIdx) -> int64_t {
    return isEq ? constraints->atEq(rowIdx, colIdx)
                : constraints->atIneq(rowIdx, colIdx);
  };
  uint64_t gcd = std::abs(at(0));
  for (unsigned j = 1, e = constraints->getNumCols(); j < e; ++j) {
    gcd = llvm::GreatestCommonDivisor64(gcd, std::abs(at(j)));
  }
  if (gcd > 0 && gcd != 1) {
    for (unsigned j = 0, e = constraints->getNumCols(); j < e; ++j) {
      int64_t v = at(j) / static_cast<int64_t>(gcd);
      isEq ? constraints->atEq(rowIdx, j) = v
           : constraints->atIneq(rowIdx, j) = v;
    }
  }
}

void FlatAffineConstraints::normalizeConstraintsByGCD() {
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
    normalizeConstraintByGCD</*isEq=*/true>(this, i);
  }
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    normalizeConstraintByGCD</*isEq=*/false>(this, i);
  }
}

bool FlatAffineConstraints::hasConsistentState() const {
  if (inequalities.size() != getNumInequalities() * numReservedCols)
    return false;
  if (equalities.size() != getNumEqualities() * numReservedCols)
    return false;
  if (ids.size() != getNumIds())
    return false;

  // Catches errors where numDims, numSymbols, numIds aren't consistent.
  if (numDims > numIds || numSymbols > numIds || numDims + numSymbols > numIds)
    return false;

  return true;
}

/// Checks all rows of equality/inequality constraints for trivial
/// contradictions (for example: 1 == 0, 0 >= 1), which may have surfaced
/// after elimination. Returns 'true' if an invalid constraint is found;
/// 'false' otherwise.
bool FlatAffineConstraints::hasInvalidConstraint() const {
  assert(hasConsistentState());
  auto check = [&](bool isEq) -> bool {
    unsigned numCols = getNumCols();
    unsigned numRows = isEq ? getNumEqualities() : getNumInequalities();
    for (unsigned i = 0, e = numRows; i < e; ++i) {
      unsigned j;
      for (j = 0; j < numCols - 1; ++j) {
        int64_t v = isEq ? atEq(i, j) : atIneq(i, j);
        // Skip rows with non-zero variable coefficients.
        if (v != 0)
          break;
      }
      if (j < numCols - 1) {
        continue;
      }
      // Check validity of constant term at 'numCols - 1' w.r.t 'isEq'.
      // Example invalid constraints include: '1 == 0' or '-1 >= 0'
      int64_t v = isEq ? atEq(i, numCols - 1) : atIneq(i, numCols - 1);
      if ((isEq && v != 0) || (!isEq && v < 0)) {
        return true;
      }
    }
    return false;
  };
  if (check(/*isEq=*/true))
    return true;
  return check(/*isEq=*/false);
}

// Eliminate identifier from constraint at 'rowIdx' based on coefficient at
// pivotRow, pivotCol. Columns in range [elimColStart, pivotCol) will not be
// updated as they have already been eliminated.
static void eliminateFromConstraint(FlatAffineConstraints *constraints,
                                    unsigned rowIdx, unsigned pivotRow,
                                    unsigned pivotCol, unsigned elimColStart,
                                    bool isEq) {
  // Skip if equality 'rowIdx' if same as 'pivotRow'.
  if (isEq && rowIdx == pivotRow)
    return;
  auto at = [&](unsigned i, unsigned j) -> int64_t {
    return isEq ? constraints->atEq(i, j) : constraints->atIneq(i, j);
  };
  int64_t leadCoeff = at(rowIdx, pivotCol);
  // Skip if leading coefficient at 'rowIdx' is already zero.
  if (leadCoeff == 0)
    return;
  int64_t pivotCoeff = constraints->atEq(pivotRow, pivotCol);
  int64_t sign = (leadCoeff * pivotCoeff > 0) ? -1 : 1;
  int64_t lcm = mlir::lcm(pivotCoeff, leadCoeff);
  int64_t pivotMultiplier = sign * (lcm / std::abs(pivotCoeff));
  int64_t rowMultiplier = lcm / std::abs(leadCoeff);

  unsigned numCols = constraints->getNumCols();
  for (unsigned j = 0; j < numCols; ++j) {
    // Skip updating column 'j' if it was just eliminated.
    if (j >= elimColStart && j < pivotCol)
      continue;
    int64_t v = pivotMultiplier * constraints->atEq(pivotRow, j) +
                rowMultiplier * at(rowIdx, j);
    isEq ? constraints->atEq(rowIdx, j) = v
         : constraints->atIneq(rowIdx, j) = v;
  }
}

// Remove coefficients in column range [colStart, colLimit) in place.
// This removes in data in the specified column range, and copies any
// remaining valid data into place.
static void shiftColumnsToLeft(FlatAffineConstraints *constraints,
                               unsigned colStart, unsigned colLimit,
                               bool isEq) {
  assert(colLimit <= constraints->getNumIds());
  if (colLimit <= colStart)
    return;

  unsigned numCols = constraints->getNumCols();
  unsigned numRows = isEq ? constraints->getNumEqualities()
                          : constraints->getNumInequalities();
  unsigned numToEliminate = colLimit - colStart;
  for (unsigned r = 0, e = numRows; r < e; ++r) {
    for (unsigned c = colLimit; c < numCols; ++c) {
      if (isEq) {
        constraints->atEq(r, c - numToEliminate) = constraints->atEq(r, c);
      } else {
        constraints->atIneq(r, c - numToEliminate) = constraints->atIneq(r, c);
      }
    }
  }
}

// Removes identifiers in column range [idStart, idLimit), and copies any
// remaining valid data into place, and updates member variables.
void FlatAffineConstraints::removeIdRange(unsigned idStart, unsigned idLimit) {
  assert(idLimit < getNumCols() && "invalid id limit");

  if (idStart >= idLimit)
    return;

  // We are going to be removing one or more identifiers from the range.
  assert(idStart < numIds && "invalid idStart position");

  // TODO(andydavis) Make 'removeIdRange' a lambda called from here.
  // Remove eliminated identifiers from equalities.
  shiftColumnsToLeft(this, idStart, idLimit, /*isEq=*/true);

  // Remove eliminated identifiers from inequalities.
  shiftColumnsToLeft(this, idStart, idLimit, /*isEq=*/false);

  // Update members numDims, numSymbols and numIds.
  unsigned numDimsEliminated = 0;
  unsigned numLocalsEliminated = 0;
  unsigned numColsEliminated = idLimit - idStart;
  if (idStart < numDims) {
    numDimsEliminated = std::min(numDims, idLimit) - idStart;
  }
  // Check how many local id's were removed. Note that our identifier order is
  // [dims, symbols, locals]. Local id start at position numDims + numSymbols.
  if (idLimit > numDims + numSymbols) {
    numLocalsEliminated = std::min(
        idLimit - std::max(idStart, numDims + numSymbols), getNumLocalIds());
  }
  unsigned numSymbolsEliminated =
      numColsEliminated - numDimsEliminated - numLocalsEliminated;

  numDims -= numDimsEliminated;
  numSymbols -= numSymbolsEliminated;
  numIds = numIds - numColsEliminated;

  ids.erase(ids.begin() + idStart, ids.begin() + idLimit);

  // No resize necessary. numReservedCols remains the same.
}

/// Returns the position of the identifier that has the minimum <number of lower
/// bounds> times <number of upper bounds> from the specified range of
/// identifiers [start, end). It is often best to eliminate in the increasing
/// order of these counts when doing Fourier-Motzkin elimination since FM adds
/// that many new constraints.
static unsigned getBestIdToEliminate(const FlatAffineConstraints &cst,
                                     unsigned start, unsigned end) {
  assert(start < cst.getNumIds() && end < cst.getNumIds() + 1);

  auto getProductOfNumLowerUpperBounds = [&](unsigned pos) {
    unsigned numLb = 0;
    unsigned numUb = 0;
    for (unsigned r = 0, e = cst.getNumInequalities(); r < e; r++) {
      if (cst.atIneq(r, pos) > 0) {
        ++numLb;
      } else if (cst.atIneq(r, pos) < 0) {
        ++numUb;
      }
    }
    return numLb * numUb;
  };

  unsigned minLoc = start;
  unsigned min = getProductOfNumLowerUpperBounds(start);
  for (unsigned c = start + 1; c < end; c++) {
    unsigned numLbUbProduct = getProductOfNumLowerUpperBounds(c);
    if (numLbUbProduct < min) {
      min = numLbUbProduct;
      minLoc = c;
    }
  }
  return minLoc;
}

// Checks for emptiness of the set by eliminating identifiers successively and
// using the GCD test (on all equality constraints) and checking for trivially
// invalid constraints. Returns 'true' if the constraint system is found to be
// empty; false otherwise.
bool FlatAffineConstraints::isEmpty() const {
  if (isEmptyByGCDTest() || hasInvalidConstraint())
    return true;

  // First, eliminate as many identifiers as possible using Gaussian
  // elimination.
  FlatAffineConstraints tmpCst(*this);
  unsigned currentPos = 0;
  while (currentPos < tmpCst.getNumIds()) {
    tmpCst.gaussianEliminateIds(currentPos, tmpCst.getNumIds());
    ++currentPos;
    // We check emptiness through trivial checks after eliminating each ID to
    // detect emptiness early. Since the checks isEmptyByGCDTest() and
    // hasInvalidConstraint() are linear time and single sweep on the constraint
    // buffer, this appears reasonable - but can optimize in the future.
    if (tmpCst.hasInvalidConstraint() || tmpCst.isEmptyByGCDTest())
      return true;
  }

  // Eliminate the remaining using FM.
  for (unsigned i = 0, e = tmpCst.getNumIds(); i < e; i++) {
    tmpCst.FourierMotzkinEliminate(
        getBestIdToEliminate(tmpCst, 0, tmpCst.getNumIds()));
    // Check for a constraint explosion. This rarely happens in practice, but
    // this check exists as a safeguard against improperly constructed
    // constraint systems or artificially created arbitrarily complex systems
    // that aren't the intended use case for FlatAffineConstraints. This is
    // needed since FM has a worst case exponential complexity in theory.
    if (tmpCst.getNumConstraints() >= kExplosionFactor * getNumIds()) {
      LLVM_DEBUG(llvm::dbgs() << "FM constraint explosion detected\n");
      return false;
    }

    // FM wouldn't have modified the equalities in any way. So no need to again
    // run GCD test. Check for trivial invalid constraints.
    if (tmpCst.hasInvalidConstraint())
      return true;
  }
  return false;
}

// Runs the GCD test on all equality constraints. Returns 'true' if this test
// fails on any equality. Returns 'false' otherwise.
// This test can be used to disprove the existence of a solution. If it returns
// true, no integer solution to the equality constraints can exist.
//
// GCD test definition:
//
// The equality constraint:
//
//  c_1*x_1 + c_2*x_2 + ... + c_n*x_n = c_0
//
// has an integer solution iff:
//
//  GCD of c_1, c_2, ..., c_n divides c_0.
//
bool FlatAffineConstraints::isEmptyByGCDTest() const {
  assert(hasConsistentState());
  unsigned numCols = getNumCols();
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
    uint64_t gcd = std::abs(atEq(i, 0));
    for (unsigned j = 1; j < numCols - 1; ++j) {
      gcd = llvm::GreatestCommonDivisor64(gcd, std::abs(atEq(i, j)));
    }
    int64_t v = std::abs(atEq(i, numCols - 1));
    if (gcd > 0 && (v % gcd != 0)) {
      return true;
    }
  }
  return false;
}

/// Tightens inequalities given that we are dealing with integer spaces. This is
/// analogous to the GCD test but applied to inequalities. The constant term can
/// be reduced to the preceding multiple of the GCD of the coefficients, i.e.,
///  64*i - 100 >= 0  =>  64*i - 128 >= 0 (since 'i' is an integer). This is a
/// fast method - linear in the number of coefficients.
// Example on how this affects practical cases: consider the scenario:
// 64*i >= 100, j = 64*i; without a tightening, elimination of i would yield
// j >= 100 instead of the tighter (exact) j >= 128.
void FlatAffineConstraints::GCDTightenInequalities() {
  unsigned numCols = getNumCols();
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    uint64_t gcd = std::abs(atIneq(i, 0));
    for (unsigned j = 1; j < numCols - 1; ++j) {
      gcd = llvm::GreatestCommonDivisor64(gcd, std::abs(atIneq(i, j)));
    }
    if (gcd > 0 && gcd != 1) {
      int64_t gcdI = static_cast<int64_t>(gcd);
      // Tighten the constant term and normalize the constraint by the GCD.
      atIneq(i, numCols - 1) = mlir::floorDiv(atIneq(i, numCols - 1), gcdI);
      for (unsigned j = 0, e = numCols - 1; j < e; ++j)
        atIneq(i, j) /= gcdI;
    }
  }
}

// Eliminates all identifier variables in column range [posStart, posLimit).
// Returns the number of variables eliminated.
unsigned FlatAffineConstraints::gaussianEliminateIds(unsigned posStart,
                                                     unsigned posLimit) {
  // Return if identifier positions to eliminate are out of range.
  assert(posLimit <= numIds);
  assert(hasConsistentState());

  if (posStart >= posLimit)
    return 0;

  GCDTightenInequalities();

  unsigned pivotCol = 0;
  for (pivotCol = posStart; pivotCol < posLimit; ++pivotCol) {
    // Find a row which has a non-zero coefficient in column 'j'.
    unsigned pivotRow;
    if (!findConstraintWithNonZeroAt(*this, pivotCol, /*isEq=*/true,
                                     &pivotRow)) {
      // No pivot row in equalities with non-zero at 'pivotCol'.
      if (!findConstraintWithNonZeroAt(*this, pivotCol, /*isEq=*/false,
                                       &pivotRow)) {
        // If inequalities are also non-zero in 'pivotCol', it can be
        // eliminated.
        continue;
      }
      break;
    }

    // Eliminate identifier at 'pivotCol' from each equality row.
    for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
      eliminateFromConstraint(this, i, pivotRow, pivotCol, posStart,
                              /*isEq=*/true);
      normalizeConstraintByGCD</*isEq=*/true>(this, i);
    }

    // Eliminate identifier at 'pivotCol' from each inequality row.
    for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
      eliminateFromConstraint(this, i, pivotRow, pivotCol, posStart,
                              /*isEq=*/false);
      normalizeConstraintByGCD</*isEq=*/false>(this, i);
    }
    removeEquality(pivotRow);
    GCDTightenInequalities();
  }
  // Update position limit based on number eliminated.
  posLimit = pivotCol;
  // Remove eliminated columns from all constraints.
  removeIdRange(posStart, posLimit);
  return posLimit - posStart;
}

// Detect the identifier at 'pos' (say id_r) as modulo of another identifier
// (say id_n) w.r.t a constant. When this happens, another identifier (say id_q)
// could be detected as the floordiv of n. For eg:
// id_n - 4*id_q - id_r = 0, 0 <= id_r <= 3    <=>
//                          id_r = id_n mod 4, id_q = id_n floordiv 4.
// lbConst and ubConst are the constant lower and upper bounds for 'pos' -
// pre-detected at the caller.
static bool detectAsMod(const FlatAffineConstraints &cst, unsigned pos,
                        int64_t lbConst, int64_t ubConst,
                        SmallVectorImpl<AffineExpr> *memo) {
  assert(pos < cst.getNumIds() && "invalid position");

  // Check if 0 <= id_r <= divisor - 1 and if id_r is equal to
  // id_n - divisor * id_q. If these are true, then id_n becomes the dividend
  // and id_q the quotient when dividing id_n by the divisor.

  if (lbConst != 0 || ubConst < 1)
    return false;

  int64_t divisor = ubConst + 1;

  // Now check for: id_r =  id_n - divisor * id_q. As an example, we
  // are looking r = d - 4q, i.e., either r - d + 4q = 0 or -r + d - 4q = 0.
  unsigned seenQuotient = 0, seenDividend = 0;
  int quotientPos = -1, dividendPos = -1;
  for (unsigned r = 0, e = cst.getNumEqualities(); r < e; r++) {
    // id_n should have coeff 1 or -1.
    if (std::abs(cst.atEq(r, pos)) != 1)
      continue;
    // constant term should be 0.
    if (cst.atEq(r, cst.getNumCols() - 1) != 0)
      continue;
    unsigned c, f;
    int quotientSign = 1, dividendSign = 1;
    for (c = 0, f = cst.getNumDimAndSymbolIds(); c < f; c++) {
      if (c == pos)
        continue;
      // The coefficient of the quotient should be +/-divisor.
      // TODO(bondhugula): could be extended to detect an affine function for
      // the quotient (i.e., the coeff could be a non-zero multiple of divisor).
      int64_t v = cst.atEq(r, c) * cst.atEq(r, pos);
      if (v == divisor || v == -divisor) {
        seenQuotient++;
        quotientPos = c;
        quotientSign = v > 0 ? 1 : -1;
      }
      // The coefficient of the dividend should be +/-1.
      // TODO(bondhugula): could be extended to detect an affine function of
      // the other identifiers as the dividend.
      else if (v == -1 || v == 1) {
        seenDividend++;
        dividendPos = c;
        dividendSign = v < 0 ? 1 : -1;
      } else if (cst.atEq(r, c) != 0) {
        // Cannot be inferred as a mod since the constraint has a coefficient
        // for an identifier that's neither a unit nor the divisor (see TODOs
        // above).
        break;
      }
    }
    if (c < f)
      // Cannot be inferred as a mod since the constraint has a coefficient for
      // an identifier that's neither a unit nor the divisor (see TODOs above).
      continue;

    // We are looking for exactly one identifier as the dividend.
    if (seenDividend == 1 && seenQuotient >= 1) {
      if (!(*memo)[dividendPos])
        return false;
      // Successfully detected a mod.
      (*memo)[pos] = (*memo)[dividendPos] % divisor * dividendSign;
      auto ub = cst.getConstantUpperBound(dividendPos);
      if (ub.hasValue() && ub.getValue() < divisor)
        // The mod can be optimized away.
        (*memo)[pos] = (*memo)[dividendPos] * dividendSign;
      else
        (*memo)[pos] = (*memo)[dividendPos] % divisor * dividendSign;

      if (seenQuotient == 1 && !(*memo)[quotientPos])
        // Successfully detected a floordiv as well.
        (*memo)[quotientPos] =
            (*memo)[dividendPos].floorDiv(divisor) * quotientSign;
      return true;
    }
  }
  return false;
}

// Gather lower and upper bounds for the pos^th identifier.
static void getLowerAndUpperBoundIndices(const FlatAffineConstraints &cst,
                                         unsigned pos,
                                         SmallVectorImpl<unsigned> *lbIndices,
                                         SmallVectorImpl<unsigned> *ubIndices) {
  assert(pos < cst.getNumIds() && "invalid position");

  // Gather all lower bounds and upper bounds of the variable. Since the
  // canonical form c_1*x_1 + c_2*x_2 + ... + c_0 >= 0, a constraint is a lower
  // bound for x_i if c_i >= 1, and an upper bound if c_i <= -1.
  for (unsigned r = 0, e = cst.getNumInequalities(); r < e; r++) {
    if (cst.atIneq(r, pos) >= 1) {
      // Lower bound.
      lbIndices->push_back(r);
    } else if (cst.atIneq(r, pos) <= -1) {
      // Upper bound.
      ubIndices->push_back(r);
    }
  }
}

// Check if the pos^th identifier can be expressed as a floordiv of an affine
// function of other identifiers (where the divisor is a positive constant).
// For eg: 4q <= i + j <= 4q + 3   <=>   q = (i + j) floordiv 4.
bool detectAsFloorDiv(const FlatAffineConstraints &cst, unsigned pos,
                      SmallVectorImpl<AffineExpr> *memo, MLIRContext *context) {
  assert(pos < cst.getNumIds() && "invalid position");

  SmallVector<unsigned, 4> lbIndices, ubIndices;
  getLowerAndUpperBoundIndices(cst, pos, &lbIndices, &ubIndices);

  // Check if any lower bound, upper bound pair is of the form:
  // divisor * id >=  expr - (divisor - 1)    <-- Lower bound for 'id'
  // divisor * id <=  expr                    <-- Upper bound for 'id'
  // Then, 'id' is equivalent to 'expr floordiv divisor'.  (where divisor > 1).
  //
  // For example, if -32*k + 16*i + j >= 0
  //                  32*k - 16*i - j + 31 >= 0   <=>
  //             k = ( 16*i + j ) floordiv 32
  unsigned seenDividends = 0;
  for (auto ubPos : ubIndices) {
    for (auto lbPos : lbIndices) {
      // Check if lower bound's constant term is 'divisor - 1'. The 'divisor'
      // here is cst.atIneq(lbPos, pos) and we already know that it's positive
      // (since cst.Ineq(lbPos, ...) is a lower bound expression for 'pos'.
      if (cst.atIneq(lbPos, cst.getNumCols() - 1) != cst.atIneq(lbPos, pos) - 1)
        continue;
      // Check if upper bound's constant term is 0.
      if (cst.atIneq(ubPos, cst.getNumCols() - 1) != 0)
        continue;
      // For the remaining part, check if the lower bound expr's coeff's are
      // negations of corresponding upper bound ones'.
      unsigned c, f;
      for (c = 0, f = cst.getNumCols() - 1; c < f; c++) {
        if (cst.atIneq(lbPos, c) != -cst.atIneq(ubPos, c))
          break;
        if (c != pos && cst.atIneq(lbPos, c) != 0)
          seenDividends++;
      }
      // Lb coeff's aren't negative of ub coeff's (for the non constant term
      // part).
      if (c < f)
        continue;
      if (seenDividends >= 1) {
        // The divisor is the constant term of the lower bound expression.
        // We already know that cst.atIneq(lbPos, pos) > 0.
        int64_t divisor = cst.atIneq(lbPos, pos);
        // Construct the dividend expression.
        auto dividendExpr = getAffineConstantExpr(0, context);
        unsigned c, f;
        for (c = 0, f = cst.getNumCols() - 1; c < f; c++) {
          if (c == pos)
            continue;
          int64_t ubVal = cst.atIneq(ubPos, c);
          if (ubVal == 0)
            continue;
          if (!(*memo)[c])
            break;
          dividendExpr = dividendExpr + ubVal * (*memo)[c];
        }
        // Expression can't be constructed as it depends on a yet unknown
        // identifier.
        // TODO(mlir-team): Visit/compute the identifiers in an order so that
        // this doesn't happen. More complex but much more efficient.
        if (c < f)
          continue;
        // Successfully detected the floordiv.
        (*memo)[pos] = dividendExpr.floorDiv(divisor);
        return true;
      }
    }
  }
  return false;
}

// Fills an inequality row with the value 'val'.
static inline void fillInequality(FlatAffineConstraints *cst, unsigned r,
                                  int64_t val) {
  for (unsigned c = 0, f = cst->getNumCols(); c < f; c++) {
    cst->atIneq(r, c) = val;
  }
}

// Negates an inequality.
static inline void negateInequality(FlatAffineConstraints *cst, unsigned r) {
  for (unsigned c = 0, f = cst->getNumCols(); c < f; c++) {
    cst->atIneq(r, c) = -cst->atIneq(r, c);
  }
}

// A more complex check to eliminate redundant inequalities. Uses FourierMotzkin
// to check if a constraint is redundant.
void FlatAffineConstraints::removeRedundantInequalities() {
  SmallVector<bool, 32> redun(getNumInequalities(), false);
  // To check if an inequality is redundant, we replace the inequality by its
  // complement (for eg., i - 1 >= 0 by i <= 0), and check if the resulting
  // system is empty. If it is, the inequality is redundant.
  FlatAffineConstraints tmpCst(*this);
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    // Change the inequality to its complement.
    negateInequality(&tmpCst, r);
    tmpCst.atIneq(r, tmpCst.getNumCols() - 1)--;
    if (tmpCst.isEmpty()) {
      redun[r] = true;
      // Zero fill the redundant inequality.
      fillInequality(this, r, /*val=*/0);
      fillInequality(&tmpCst, r, /*val=*/0);
    } else {
      // Reverse the change (to avoid recreating tmpCst each time).
      tmpCst.atIneq(r, tmpCst.getNumCols() - 1)++;
      negateInequality(&tmpCst, r);
    }
  }

  // Scan to get rid of all rows marked redundant, in-place.
  auto copyRow = [&](unsigned src, unsigned dest) {
    if (src == dest)
      return;
    for (unsigned c = 0, e = getNumCols(); c < e; c++) {
      atIneq(dest, c) = atIneq(src, c);
    }
  };
  unsigned pos = 0;
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    if (!redun[r])
      copyRow(r, pos++);
  }
  inequalities.resize(numReservedCols * pos);
}

std::pair<AffineMap, AffineMap> FlatAffineConstraints::getLowerAndUpperBound(
    unsigned pos, unsigned offset, unsigned num, unsigned symStartPos,
    ArrayRef<AffineExpr> localExprs, MLIRContext *context) const {
  assert(pos + offset < getNumDimIds() && "invalid dim start pos");
  assert(symStartPos >= (pos + offset) && "invalid sym start pos");
  assert(getNumLocalIds() == localExprs.size() &&
         "incorrect local exprs count");

  SmallVector<unsigned, 4> lbIndices, ubIndices;
  getLowerAndUpperBoundIndices(*this, pos + offset, &lbIndices, &ubIndices);

  /// Add to 'b' from 'a' in set [0, offset) U [offset + num, symbStartPos).
  auto addCoeffs = [&](ArrayRef<int64_t> a, SmallVectorImpl<int64_t> &b) {
    b.clear();
    for (unsigned i = 0, e = a.size(); i < e; ++i) {
      if (i < offset || i >= offset + num)
        b.push_back(a[i]);
    }
  };

  SmallVector<int64_t, 8> lb, ub;
  SmallVector<AffineExpr, 4> exprs;
  unsigned dimCount = symStartPos - num;
  unsigned symCount = getNumDimAndSymbolIds() - symStartPos;
  exprs.reserve(lbIndices.size());
  // Lower bound expressions.
  for (auto idx : lbIndices) {
    auto ineq = getInequality(idx);
    // Extract the lower bound (in terms of other coeff's + const), i.e., if
    // i - j + 1 >= 0 is the constraint, 'pos' is for i the lower bound is j
    // - 1.
    addCoeffs(ineq, lb);
    std::transform(lb.begin(), lb.end(), lb.begin(), std::negate<int64_t>());
    auto expr = mlir::toAffineExpr(lb, dimCount, symCount, localExprs, context);
    exprs.push_back(expr);
  }
  auto lbMap =
      exprs.empty() ? AffineMap() : AffineMap::get(dimCount, symCount, exprs);

  exprs.clear();
  exprs.reserve(ubIndices.size());
  // Upper bound expressions.
  for (auto idx : ubIndices) {
    auto ineq = getInequality(idx);
    // Extract the upper bound (in terms of other coeff's + const).
    addCoeffs(ineq, ub);
    auto expr = mlir::toAffineExpr(ub, dimCount, symCount, localExprs, context);
    // Upper bound is exclusive.
    exprs.push_back(expr + 1);
  }
  auto ubMap =
      exprs.empty() ? AffineMap() : AffineMap::get(dimCount, symCount, exprs);

  return {lbMap, ubMap};
}

/// Computes the lower and upper bounds of the first 'num' dimensional
/// identifiers (starting at 'offset') as affine maps of the remaining
/// identifiers (dimensional and symbolic identifiers). Local identifiers are
/// themselves explicitly computed as affine functions of other identifiers in
/// this process if needed.
void FlatAffineConstraints::getSliceBounds(unsigned offset, unsigned num,
                                           MLIRContext *context,
                                           SmallVectorImpl<AffineMap> *lbMaps,
                                           SmallVectorImpl<AffineMap> *ubMaps) {
  assert(num < getNumDimIds() && "invalid range");

  // Basic simplification.
  normalizeConstraintsByGCD();

  LLVM_DEBUG(llvm::dbgs() << "getSliceBounds for first " << num
                          << " identifiers\n");
  LLVM_DEBUG(dump());

  // Record computed/detected identifiers.
  SmallVector<AffineExpr, 8> memo(getNumIds());
  // Initialize dimensional and symbolic identifiers.
  for (unsigned i = 0, e = getNumDimIds(); i < e; i++) {
    if (i < offset)
      memo[i] = getAffineDimExpr(i, context);
    else if (i >= offset + num)
      memo[i] = getAffineDimExpr(i - num, context);
  }
  for (unsigned i = getNumDimIds(), e = getNumDimAndSymbolIds(); i < e; i++)
    memo[i] = getAffineSymbolExpr(i - getNumDimIds(), context);

  bool changed;
  do {
    changed = false;
    // Identify yet unknown identifiers as constants or mod's / floordiv's of
    // other identifiers if possible.
    for (unsigned pos = 0; pos < getNumIds(); pos++) {
      if (memo[pos])
        continue;

      auto lbConst = getConstantLowerBound(pos);
      auto ubConst = getConstantUpperBound(pos);
      if (lbConst.hasValue() && ubConst.hasValue()) {
        // Detect equality to a constant.
        if (lbConst.getValue() == ubConst.getValue()) {
          memo[pos] = getAffineConstantExpr(lbConst.getValue(), context);
          changed = true;
          continue;
        }

        // Detect an identifier as modulo of another identifier w.r.t a
        // constant.
        if (detectAsMod(*this, pos, lbConst.getValue(), ubConst.getValue(),
                        &memo)) {
          changed = true;
          continue;
        }
      }

      // Detect an identifier as floordiv of another identifier w.r.t a
      // constant.
      if (detectAsFloorDiv(*this, pos, &memo, context)) {
        changed = true;
        continue;
      }

      // Detect an identifier as an expression of other identifiers.
      unsigned idx;
      if (!findConstraintWithNonZeroAt(*this, pos, /*isEq=*/true, &idx)) {
        continue;
      }

      // Build AffineExpr solving for identifier 'pos' in terms of all others.
      auto expr = getAffineConstantExpr(0, context);
      unsigned j, e;
      for (j = 0, e = getNumIds(); j < e; ++j) {
        if (j == pos)
          continue;
        int64_t c = atEq(idx, j);
        if (c == 0)
          continue;
        // If any of the involved IDs hasn't been found yet, we can't proceed.
        if (!memo[j])
          break;
        expr = expr + memo[j] * c;
      }
      if (j < e)
        // Can't construct expression as it depends on a yet uncomputed
        // identifier.
        continue;

      // Add constant term to AffineExpr.
      expr = expr + atEq(idx, getNumIds());
      int64_t vPos = atEq(idx, pos);
      assert(vPos != 0 && "expected non-zero here");
      if (vPos > 0)
        expr = (-expr).floorDiv(vPos);
      else
        // vPos < 0.
        expr = expr.floorDiv(-vPos);
      // Successfully constructed expression.
      memo[pos] = expr;
      changed = true;
    }
    // This loop is guaranteed to reach a fixed point - since once an
    // identifier's explicit form is computed (in memo[pos]), it's not updated
    // again.
  } while (changed);

  // Set the lower and upper bound maps for all the identifiers that were
  // computed as affine expressions of the rest as the "detected expr" and
  // "detected expr + 1" respectively; set the undetected ones to null.
  Optional<FlatAffineConstraints> tmpClone;
  for (unsigned pos = 0; pos < num; pos++) {
    unsigned numMapDims = getNumDimIds() - num;
    unsigned numMapSymbols = getNumSymbolIds();
    AffineExpr expr = memo[pos + offset];
    if (expr)
      expr = simplifyAffineExpr(expr, numMapDims, numMapSymbols);

    AffineMap &lbMap = (*lbMaps)[pos];
    AffineMap &ubMap = (*ubMaps)[pos];

    if (expr) {
      lbMap = AffineMap::get(numMapDims, numMapSymbols, expr);
      ubMap = AffineMap::get(numMapDims, numMapSymbols, expr + 1);
    } else {
      // TODO(bondhugula): Whenever there are local identifiers in the
      // dependence constraints, we'll conservatively over-approximate, since we
      // don't always explicitly compute them above (in the while loop).
      if (getNumLocalIds() == 0) {
        // Work on a copy so that we don't update this constraint system.
        if (!tmpClone) {
          tmpClone.emplace(FlatAffineConstraints(*this));
          // Removing redundant inequalities is necessary so that we don't get
          // redundant loop bounds.
          tmpClone->removeRedundantInequalities();
        }
        std::tie(lbMap, ubMap) = tmpClone->getLowerAndUpperBound(
            pos, offset, num, getNumDimIds(), {}, context);
      }

      // If the above fails, we'll just use the constant lower bound and the
      // constant upper bound (if they exist) as the slice bounds.
      // TODO(b/126426796): being conservative for the moment in cases that
      // lead to multiple bounds - until getConstDifference in LoopFusion.cpp is
      // fixed (b/126426796).
      if (!lbMap || lbMap.getNumResults() > 1) {
        LLVM_DEBUG(llvm::dbgs()
                   << "WARNING: Potentially over-approximating slice lb\n");
        auto lbConst = getConstantLowerBound(pos + offset);
        if (lbConst.hasValue()) {
          lbMap = AffineMap::get(
              numMapDims, numMapSymbols,
              getAffineConstantExpr(lbConst.getValue(), context));
        }
      }
      if (!ubMap || ubMap.getNumResults() > 1) {
        LLVM_DEBUG(llvm::dbgs()
                   << "WARNING: Potentially over-approximating slice ub\n");
        auto ubConst = getConstantUpperBound(pos + offset);
        if (ubConst.hasValue()) {
          (ubMap) = AffineMap::get(
              numMapDims, numMapSymbols,
              getAffineConstantExpr(ubConst.getValue() + 1, context));
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs()
               << "lb map for pos = " << Twine(pos + offset) << ", expr: ");
    LLVM_DEBUG(lbMap.dump(););
    LLVM_DEBUG(llvm::dbgs()
               << "ub map for pos = " << Twine(pos + offset) << ", expr: ");
    LLVM_DEBUG(ubMap.dump(););
  }
}

LogicalResult
FlatAffineConstraints::addLowerOrUpperBound(unsigned pos, AffineMap boundMap,
                                            ArrayRef<Value> boundOperands,
                                            bool eq, bool lower) {
  assert(pos < getNumDimAndSymbolIds() && "invalid position");
  // Equality follows the logic of lower bound except that we add an equality
  // instead of an inequality.
  assert((!eq || boundMap.getNumResults() == 1) && "single result expected");
  if (eq)
    lower = true;

  // Fully compose map and operands; canonicalize and simplify so that we
  // transitively get to terminal symbols or loop IVs.
  auto map = boundMap;
  SmallVector<Value, 4> operands(boundOperands.begin(), boundOperands.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);
  for (auto operand : operands)
    addInductionVarOrTerminalSymbol(operand);

  FlatAffineConstraints localVarCst;
  std::vector<SmallVector<int64_t, 8>> flatExprs;
  if (failed(getFlattenedAffineExprs(map, &flatExprs, &localVarCst))) {
    LLVM_DEBUG(llvm::dbgs() << "semi-affine expressions not yet supported\n");
    return failure();
  }

  // Merge and align with localVarCst.
  if (localVarCst.getNumLocalIds() > 0) {
    // Set values for localVarCst.
    localVarCst.setIdValues(0, localVarCst.getNumDimAndSymbolIds(), operands);
    for (auto operand : operands) {
      unsigned pos;
      if (findId(*operand, &pos)) {
        if (pos >= getNumDimIds() && pos < getNumDimAndSymbolIds()) {
          // If the local var cst has this as a dim, turn it into its symbol.
          turnDimIntoSymbol(&localVarCst, *operand);
        } else if (pos < getNumDimIds()) {
          // Or vice versa.
          turnSymbolIntoDim(&localVarCst, *operand);
        }
      }
    }
    mergeAndAlignIds(/*offset=*/0, this, &localVarCst);
    append(localVarCst);
  }

  // Record positions of the operands in the constraint system. Need to do
  // this here since the constraint system changes after a bound is added.
  SmallVector<unsigned, 8> positions;
  unsigned numOperands = operands.size();
  for (auto operand : operands) {
    unsigned pos;
    if (!findId(*operand, &pos))
      assert(0 && "expected to be found");
    positions.push_back(pos);
  }

  for (const auto &flatExpr : flatExprs) {
    SmallVector<int64_t, 4> ineq(getNumCols(), 0);
    ineq[pos] = lower ? 1 : -1;
    // Dims and symbols.
    for (unsigned j = 0, e = map.getNumInputs(); j < e; j++) {
      ineq[positions[j]] = lower ? -flatExpr[j] : flatExpr[j];
    }
    // Copy over the local id coefficients.
    unsigned numLocalIds = flatExpr.size() - 1 - numOperands;
    for (unsigned jj = 0, j = getNumIds() - numLocalIds; jj < numLocalIds;
         jj++, j++) {
      ineq[j] =
          lower ? -flatExpr[numOperands + jj] : flatExpr[numOperands + jj];
    }
    // Constant term.
    ineq[getNumCols() - 1] =
        lower ? -flatExpr[flatExpr.size() - 1]
              // Upper bound in flattenedExpr is an exclusive one.
              : flatExpr[flatExpr.size() - 1] - 1;
    eq ? addEquality(ineq) : addInequality(ineq);
  }
  return success();
}

// Adds slice lower bounds represented by lower bounds in 'lbMaps' and upper
// bounds in 'ubMaps' to each value in `values' that appears in the constraint
// system. Note that both lower/upper bounds share the same operand list
// 'operands'.
// This function assumes 'values.size' == 'lbMaps.size' == 'ubMaps.size', and
// skips any null AffineMaps in 'lbMaps' or 'ubMaps'.
// Note that both lower/upper bounds use operands from 'operands'.
// Returns failure for unimplemented cases such as semi-affine expressions or
// expressions with mod/floordiv.
LogicalResult FlatAffineConstraints::addSliceBounds(ArrayRef<Value> values,
                                                    ArrayRef<AffineMap> lbMaps,
                                                    ArrayRef<AffineMap> ubMaps,
                                                    ArrayRef<Value> operands) {
  assert(values.size() == lbMaps.size());
  assert(lbMaps.size() == ubMaps.size());

  for (unsigned i = 0, e = lbMaps.size(); i < e; ++i) {
    unsigned pos;
    if (!findId(*values[i], &pos))
      continue;

    AffineMap lbMap = lbMaps[i];
    AffineMap ubMap = ubMaps[i];
    assert(!lbMap || lbMap.getNumInputs() == operands.size());
    assert(!ubMap || ubMap.getNumInputs() == operands.size());

    // Check if this slice is just an equality along this dimension.
    if (lbMap && ubMap && lbMap.getNumResults() == 1 &&
        ubMap.getNumResults() == 1 &&
        lbMap.getResult(0) + 1 == ubMap.getResult(0)) {
      if (failed(addLowerOrUpperBound(pos, lbMap, operands, /*eq=*/true,
                                      /*lower=*/true)))
        return failure();
      continue;
    }

    if (lbMap && failed(addLowerOrUpperBound(pos, lbMap, operands, /*eq=*/false,
                                             /*lower=*/true)))
      return failure();

    if (ubMap && failed(addLowerOrUpperBound(pos, ubMap, operands, /*eq=*/false,
                                             /*lower=*/false)))
      return failure();
  }
  return success();
}

void FlatAffineConstraints::addEquality(ArrayRef<int64_t> eq) {
  assert(eq.size() == getNumCols());
  unsigned offset = equalities.size();
  equalities.resize(equalities.size() + numReservedCols);
  std::copy(eq.begin(), eq.end(), equalities.begin() + offset);
}

void FlatAffineConstraints::addInequality(ArrayRef<int64_t> inEq) {
  assert(inEq.size() == getNumCols());
  unsigned offset = inequalities.size();
  inequalities.resize(inequalities.size() + numReservedCols);
  std::copy(inEq.begin(), inEq.end(), inequalities.begin() + offset);
}

void FlatAffineConstraints::addConstantLowerBound(unsigned pos, int64_t lb) {
  assert(pos < getNumCols());
  unsigned offset = inequalities.size();
  inequalities.resize(inequalities.size() + numReservedCols);
  std::fill(inequalities.begin() + offset,
            inequalities.begin() + offset + getNumCols(), 0);
  inequalities[offset + pos] = 1;
  inequalities[offset + getNumCols() - 1] = -lb;
}

void FlatAffineConstraints::addConstantUpperBound(unsigned pos, int64_t ub) {
  assert(pos < getNumCols());
  unsigned offset = inequalities.size();
  inequalities.resize(inequalities.size() + numReservedCols);
  std::fill(inequalities.begin() + offset,
            inequalities.begin() + offset + getNumCols(), 0);
  inequalities[offset + pos] = -1;
  inequalities[offset + getNumCols() - 1] = ub;
}

void FlatAffineConstraints::addConstantLowerBound(ArrayRef<int64_t> expr,
                                                  int64_t lb) {
  assert(expr.size() == getNumCols());
  unsigned offset = inequalities.size();
  inequalities.resize(inequalities.size() + numReservedCols);
  std::fill(inequalities.begin() + offset,
            inequalities.begin() + offset + getNumCols(), 0);
  std::copy(expr.begin(), expr.end(), inequalities.begin() + offset);
  inequalities[offset + getNumCols() - 1] += -lb;
}

void FlatAffineConstraints::addConstantUpperBound(ArrayRef<int64_t> expr,
                                                  int64_t ub) {
  assert(expr.size() == getNumCols());
  unsigned offset = inequalities.size();
  inequalities.resize(inequalities.size() + numReservedCols);
  std::fill(inequalities.begin() + offset,
            inequalities.begin() + offset + getNumCols(), 0);
  for (unsigned i = 0, e = getNumCols(); i < e; i++) {
    inequalities[offset + i] = -expr[i];
  }
  inequalities[offset + getNumCols() - 1] += ub;
}

/// Adds a new local identifier as the floordiv of an affine function of other
/// identifiers, the coefficients of which are provided in 'dividend' and with
/// respect to a positive constant 'divisor'. Two constraints are added to the
/// system to capture equivalence with the floordiv.
///      q = expr floordiv c    <=>   c*q <= expr <= c*q + c - 1.
void FlatAffineConstraints::addLocalFloorDiv(ArrayRef<int64_t> dividend,
                                             int64_t divisor) {
  assert(dividend.size() == getNumCols() && "incorrect dividend size");
  assert(divisor > 0 && "positive divisor expected");

  addLocalId(getNumLocalIds());

  // Add two constraints for this new identifier 'q'.
  SmallVector<int64_t, 8> bound(dividend.size() + 1);

  // dividend - q * divisor >= 0
  std::copy(dividend.begin(), dividend.begin() + dividend.size() - 1,
            bound.begin());
  bound.back() = dividend.back();
  bound[getNumIds() - 1] = -divisor;
  addInequality(bound);

  // -dividend +qdivisor * q + divisor - 1 >= 0
  std::transform(bound.begin(), bound.end(), bound.begin(),
                 std::negate<int64_t>());
  bound[bound.size() - 1] += divisor - 1;
  addInequality(bound);
}

bool FlatAffineConstraints::findId(Value id, unsigned *pos) const {
  unsigned i = 0;
  for (const auto &mayBeId : ids) {
    if (mayBeId.hasValue() && mayBeId.getValue() == id) {
      *pos = i;
      return true;
    }
    i++;
  }
  return false;
}

bool FlatAffineConstraints::containsId(Value id) const {
  return llvm::any_of(ids, [&](const Optional<Value> &mayBeId) {
    return mayBeId.hasValue() && mayBeId.getValue() == id;
  });
}

void FlatAffineConstraints::setDimSymbolSeparation(unsigned newSymbolCount) {
  assert(newSymbolCount <= numDims + numSymbols &&
         "invalid separation position");
  numDims = numDims + numSymbols - newSymbolCount;
  numSymbols = newSymbolCount;
}

/// Sets the specified identifier to a constant value.
void FlatAffineConstraints::setIdToConstant(unsigned pos, int64_t val) {
  unsigned offset = equalities.size();
  equalities.resize(equalities.size() + numReservedCols);
  std::fill(equalities.begin() + offset,
            equalities.begin() + offset + getNumCols(), 0);
  equalities[offset + pos] = 1;
  equalities[offset + getNumCols() - 1] = -val;
}

/// Sets the specified identifier to a constant value; asserts if the id is not
/// found.
void FlatAffineConstraints::setIdToConstant(Value id, int64_t val) {
  unsigned pos;
  if (!findId(id, &pos))
    // This is a pre-condition for this method.
    assert(0 && "id not found");
  setIdToConstant(pos, val);
}

void FlatAffineConstraints::removeEquality(unsigned pos) {
  unsigned numEqualities = getNumEqualities();
  assert(pos < numEqualities);
  unsigned outputIndex = pos * numReservedCols;
  unsigned inputIndex = (pos + 1) * numReservedCols;
  unsigned numElemsToCopy = (numEqualities - pos - 1) * numReservedCols;
  std::copy(equalities.begin() + inputIndex,
            equalities.begin() + inputIndex + numElemsToCopy,
            equalities.begin() + outputIndex);
  equalities.resize(equalities.size() - numReservedCols);
}

/// Finds an equality that equates the specified identifier to a constant.
/// Returns the position of the equality row. If 'symbolic' is set to true,
/// symbols are also treated like a constant, i.e., an affine function of the
/// symbols is also treated like a constant.
static int findEqualityToConstant(const FlatAffineConstraints &cst,
                                  unsigned pos, bool symbolic = false) {
  assert(pos < cst.getNumIds() && "invalid position");
  for (unsigned r = 0, e = cst.getNumEqualities(); r < e; r++) {
    int64_t v = cst.atEq(r, pos);
    if (v * v != 1)
      continue;
    unsigned c;
    unsigned f = symbolic ? cst.getNumDimIds() : cst.getNumIds();
    // This checks for zeros in all positions other than 'pos' in [0, f)
    for (c = 0; c < f; c++) {
      if (c == pos)
        continue;
      if (cst.atEq(r, c) != 0) {
        // Dependent on another identifier.
        break;
      }
    }
    if (c == f)
      // Equality is free of other identifiers.
      return r;
  }
  return -1;
}

void FlatAffineConstraints::setAndEliminate(unsigned pos, int64_t constVal) {
  assert(pos < getNumIds() && "invalid position");
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    atIneq(r, getNumCols() - 1) += atIneq(r, pos) * constVal;
  }
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    atEq(r, getNumCols() - 1) += atEq(r, pos) * constVal;
  }
  removeId(pos);
}

LogicalResult FlatAffineConstraints::constantFoldId(unsigned pos) {
  assert(pos < getNumIds() && "invalid position");
  int rowIdx;
  if ((rowIdx = findEqualityToConstant(*this, pos)) == -1)
    return failure();

  // atEq(rowIdx, pos) is either -1 or 1.
  assert(atEq(rowIdx, pos) * atEq(rowIdx, pos) == 1);
  int64_t constVal = -atEq(rowIdx, getNumCols() - 1) / atEq(rowIdx, pos);
  setAndEliminate(pos, constVal);
  return success();
}

void FlatAffineConstraints::constantFoldIdRange(unsigned pos, unsigned num) {
  for (unsigned s = pos, t = pos, e = pos + num; s < e; s++) {
    if (failed(constantFoldId(t)))
      t++;
  }
}

/// Returns the extent (upper bound - lower bound) of the specified
/// identifier if it is found to be a constant; returns None if it's not a
/// constant. This methods treats symbolic identifiers specially, i.e.,
/// it looks for constant differences between affine expressions involving
/// only the symbolic identifiers. See comments at function definition for
/// example. 'lb', if provided, is set to the lower bound associated with the
/// constant difference. Note that 'lb' is purely symbolic and thus will contain
/// the coefficients of the symbolic identifiers and the constant coefficient.
//  Egs: 0 <= i <= 15, return 16.
//       s0 + 2 <= i <= s0 + 17, returns 16. (s0 has to be a symbol)
//       s0 + s1 + 16 <= d0 <= s0 + s1 + 31, returns 16.
//       s0 - 7 <= 8*j <= s0 returns 1 with lb = s0, lbDivisor = 8 (since lb =
//       ceil(s0 - 7 / 8) = floor(s0 / 8)).
Optional<int64_t> FlatAffineConstraints::getConstantBoundOnDimSize(
    unsigned pos, SmallVectorImpl<int64_t> *lb, int64_t *lbFloorDivisor,
    SmallVectorImpl<int64_t> *ub) const {
  assert(pos < getNumDimIds() && "Invalid identifier position");
  assert(getNumLocalIds() == 0);

  // TODO(bondhugula): eliminate all remaining dimensional identifiers (other
  // than the one at 'pos' to make this more powerful. Not needed for
  // hyper-rectangular spaces.

  // Find an equality for 'pos'^th identifier that equates it to some function
  // of the symbolic identifiers (+ constant).
  int eqRow = findEqualityToConstant(*this, pos, /*symbolic=*/true);
  if (eqRow != -1) {
    // This identifier can only take a single value.
    if (lb) {
      // Set lb to the symbolic value.
      lb->resize(getNumSymbolIds() + 1);
      if (ub)
        ub->resize(getNumSymbolIds() + 1);
      for (unsigned c = 0, f = getNumSymbolIds() + 1; c < f; c++) {
        int64_t v = atEq(eqRow, pos);
        // atEq(eqRow, pos) is either -1 or 1.
        assert(v * v == 1);
        (*lb)[c] = v < 0 ? atEq(eqRow, getNumDimIds() + c) / -v
                         : -atEq(eqRow, getNumDimIds() + c) / v;
        // Since this is an equality, ub = lb.
        if (ub)
          (*ub)[c] = (*lb)[c];
      }
      assert(lbFloorDivisor &&
             "both lb and divisor or none should be provided");
      *lbFloorDivisor = 1;
    }
    return 1;
  }

  // Check if the identifier appears at all in any of the inequalities.
  unsigned r, e;
  for (r = 0, e = getNumInequalities(); r < e; r++) {
    if (atIneq(r, pos) != 0)
      break;
  }
  if (r == e)
    // If it doesn't, there isn't a bound on it.
    return None;

  // Positions of constraints that are lower/upper bounds on the variable.
  SmallVector<unsigned, 4> lbIndices, ubIndices;

  // Gather all symbolic lower bounds and upper bounds of the variable. Since
  // the canonical form c_1*x_1 + c_2*x_2 + ... + c_0 >= 0, a constraint is a
  // lower bound for x_i if c_i >= 1, and an upper bound if c_i <= -1.
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    unsigned c, f;
    for (c = 0, f = getNumDimIds(); c < f; c++) {
      if (c != pos && atIneq(r, c) != 0)
        break;
    }
    if (c < getNumDimIds())
      // Not a pure symbolic bound.
      continue;
    if (atIneq(r, pos) >= 1)
      // Lower bound.
      lbIndices.push_back(r);
    else if (atIneq(r, pos) <= -1)
      // Upper bound.
      ubIndices.push_back(r);
  }

  // TODO(bondhugula): eliminate other dimensional identifiers to make this more
  // powerful. Not needed for hyper-rectangular iteration spaces.

  Optional<int64_t> minDiff = None;
  unsigned minLbPosition, minUbPosition;
  for (auto ubPos : ubIndices) {
    for (auto lbPos : lbIndices) {
      // Look for a lower bound and an upper bound that only differ by a
      // constant, i.e., pairs of the form  0 <= c_pos - f(c_i's) <= diffConst.
      // For example, if ii is the pos^th variable, we are looking for
      // constraints like ii >= i, ii <= ii + 50, 50 being the difference. The
      // minimum among all such constant differences is kept since that's the
      // constant bounding the extent of the pos^th variable.
      unsigned j, e;
      for (j = 0, e = getNumCols() - 1; j < e; j++)
        if (atIneq(ubPos, j) != -atIneq(lbPos, j)) {
          break;
        }
      if (j < getNumCols() - 1)
        continue;
      int64_t diff = ceilDiv(atIneq(ubPos, getNumCols() - 1) +
                                 atIneq(lbPos, getNumCols() - 1) + 1,
                             atIneq(lbPos, pos));
      if (minDiff == None || diff < minDiff) {
        minDiff = diff;
        minLbPosition = lbPos;
        minUbPosition = ubPos;
      }
    }
  }
  if (lb && minDiff.hasValue()) {
    // Set lb to the symbolic lower bound.
    lb->resize(getNumSymbolIds() + 1);
    if (ub)
      ub->resize(getNumSymbolIds() + 1);
    // The lower bound is the ceildiv of the lb constraint over the coefficient
    // of the variable at 'pos'. We express the ceildiv equivalently as a floor
    // for uniformity. For eg., if the lower bound constraint was: 32*d0 - N +
    // 31 >= 0, the lower bound for d0 is ceil(N - 31, 32), i.e., floor(N, 32).
    *lbFloorDivisor = atIneq(minLbPosition, pos);
    assert(*lbFloorDivisor == -atIneq(minUbPosition, pos));
    for (unsigned c = 0, e = getNumSymbolIds() + 1; c < e; c++) {
      (*lb)[c] = -atIneq(minLbPosition, getNumDimIds() + c);
    }
    if (ub) {
      for (unsigned c = 0, e = getNumSymbolIds() + 1; c < e; c++)
        (*ub)[c] = atIneq(minUbPosition, getNumDimIds() + c);
    }
    // The lower bound leads to a ceildiv while the upper bound is a floordiv
    // whenever the coefficient at pos != 1. ceildiv (val / d) = floordiv (val +
    // d - 1 / d); hence, the addition of 'atIneq(minLbPosition, pos) - 1' to
    // the constant term for the lower bound.
    (*lb)[getNumSymbolIds()] += atIneq(minLbPosition, pos) - 1;
  }
  return minDiff;
}

template <bool isLower>
Optional<int64_t>
FlatAffineConstraints::computeConstantLowerOrUpperBound(unsigned pos) {
  assert(pos < getNumIds() && "invalid position");
  // Project to 'pos'.
  projectOut(0, pos);
  projectOut(1, getNumIds() - 1);
  // Check if there's an equality equating the '0'^th identifier to a constant.
  int eqRowIdx = findEqualityToConstant(*this, 0, /*symbolic=*/false);
  if (eqRowIdx != -1)
    // atEq(rowIdx, 0) is either -1 or 1.
    return -atEq(eqRowIdx, getNumCols() - 1) / atEq(eqRowIdx, 0);

  // Check if the identifier appears at all in any of the inequalities.
  unsigned r, e;
  for (r = 0, e = getNumInequalities(); r < e; r++) {
    if (atIneq(r, 0) != 0)
      break;
  }
  if (r == e)
    // If it doesn't, there isn't a bound on it.
    return None;

  Optional<int64_t> minOrMaxConst = None;

  // Take the max across all const lower bounds (or min across all constant
  // upper bounds).
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    if (isLower) {
      if (atIneq(r, 0) <= 0)
        // Not a lower bound.
        continue;
    } else if (atIneq(r, 0) >= 0) {
      // Not an upper bound.
      continue;
    }
    unsigned c, f;
    for (c = 0, f = getNumCols() - 1; c < f; c++)
      if (c != 0 && atIneq(r, c) != 0)
        break;
    if (c < getNumCols() - 1)
      // Not a constant bound.
      continue;

    int64_t boundConst =
        isLower ? mlir::ceilDiv(-atIneq(r, getNumCols() - 1), atIneq(r, 0))
                : mlir::floorDiv(atIneq(r, getNumCols() - 1), -atIneq(r, 0));
    if (isLower) {
      if (minOrMaxConst == None || boundConst > minOrMaxConst)
        minOrMaxConst = boundConst;
    } else {
      if (minOrMaxConst == None || boundConst < minOrMaxConst)
        minOrMaxConst = boundConst;
    }
  }
  return minOrMaxConst;
}

Optional<int64_t>
FlatAffineConstraints::getConstantLowerBound(unsigned pos) const {
  FlatAffineConstraints tmpCst(*this);
  return tmpCst.computeConstantLowerOrUpperBound</*isLower=*/true>(pos);
}

Optional<int64_t>
FlatAffineConstraints::getConstantUpperBound(unsigned pos) const {
  FlatAffineConstraints tmpCst(*this);
  return tmpCst.computeConstantLowerOrUpperBound</*isLower=*/false>(pos);
}

// A simple (naive and conservative) check for hyper-rectangularity.
bool FlatAffineConstraints::isHyperRectangular(unsigned pos,
                                               unsigned num) const {
  assert(pos < getNumCols() - 1);
  // Check for two non-zero coefficients in the range [pos, pos + sum).
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    unsigned sum = 0;
    for (unsigned c = pos; c < pos + num; c++) {
      if (atIneq(r, c) != 0)
        sum++;
    }
    if (sum > 1)
      return false;
  }
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    unsigned sum = 0;
    for (unsigned c = pos; c < pos + num; c++) {
      if (atEq(r, c) != 0)
        sum++;
    }
    if (sum > 1)
      return false;
  }
  return true;
}

void FlatAffineConstraints::print(raw_ostream &os) const {
  assert(hasConsistentState());
  os << "\nConstraints (" << getNumDimIds() << " dims, " << getNumSymbolIds()
     << " symbols, " << getNumLocalIds() << " locals), (" << getNumConstraints()
     << " constraints)\n";
  os << "(";
  for (unsigned i = 0, e = getNumIds(); i < e; i++) {
    if (ids[i] == None)
      os << "None ";
    else
      os << "Value ";
  }
  os << " const)\n";
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
    for (unsigned j = 0, f = getNumCols(); j < f; ++j) {
      os << atEq(i, j) << " ";
    }
    os << "= 0\n";
  }
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    for (unsigned j = 0, f = getNumCols(); j < f; ++j) {
      os << atIneq(i, j) << " ";
    }
    os << ">= 0\n";
  }
  os << '\n';
}

void FlatAffineConstraints::dump() const { print(llvm::errs()); }

/// Removes duplicate constraints, trivially true constraints, and constraints
/// that can be detected as redundant as a result of differing only in their
/// constant term part. A constraint of the form <non-negative constant> >= 0 is
/// considered trivially true.
//  Uses a DenseSet to hash and detect duplicates followed by a linear scan to
//  remove duplicates in place.
void FlatAffineConstraints::removeTrivialRedundancy() {
  SmallDenseSet<ArrayRef<int64_t>, 8> rowSet;

  // A map used to detect redundancy stemming from constraints that only differ
  // in their constant term. The value stored is <row position, const term>
  // for a given row.
  SmallDenseMap<ArrayRef<int64_t>, std::pair<unsigned, int64_t>>
      rowsWithoutConstTerm;

  // Check if constraint is of the form <non-negative-constant> >= 0.
  auto isTriviallyValid = [&](unsigned r) -> bool {
    for (unsigned c = 0, e = getNumCols() - 1; c < e; c++) {
      if (atIneq(r, c) != 0)
        return false;
    }
    return atIneq(r, getNumCols() - 1) >= 0;
  };

  // Detect and mark redundant constraints.
  SmallVector<bool, 256> redunIneq(getNumInequalities(), false);
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    int64_t *rowStart = inequalities.data() + numReservedCols * r;
    auto row = ArrayRef<int64_t>(rowStart, getNumCols());
    if (isTriviallyValid(r) || !rowSet.insert(row).second) {
      redunIneq[r] = true;
      continue;
    }

    // Among constraints that only differ in the constant term part, mark
    // everything other than the one with the smallest constant term redundant.
    // (eg: among i - 16j - 5 >= 0, i - 16j - 1 >=0, i - 16j - 7 >= 0, the
    // former two are redundant).
    int64_t constTerm = atIneq(r, getNumCols() - 1);
    auto rowWithoutConstTerm = ArrayRef<int64_t>(rowStart, getNumCols() - 1);
    const auto &ret =
        rowsWithoutConstTerm.insert({rowWithoutConstTerm, {r, constTerm}});
    if (!ret.second) {
      // Check if the other constraint has a higher constant term.
      auto &val = ret.first->second;
      if (val.second > constTerm) {
        // The stored row is redundant. Mark it so, and update with this one.
        redunIneq[val.first] = true;
        val = {r, constTerm};
      } else {
        // The one stored makes this one redundant.
        redunIneq[r] = true;
      }
    }
  }

  auto copyRow = [&](unsigned src, unsigned dest) {
    if (src == dest)
      return;
    for (unsigned c = 0, e = getNumCols(); c < e; c++) {
      atIneq(dest, c) = atIneq(src, c);
    }
  };

  // Scan to get rid of all rows marked redundant, in-place.
  unsigned pos = 0;
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    if (!redunIneq[r])
      copyRow(r, pos++);
  }
  inequalities.resize(numReservedCols * pos);

  // TODO(bondhugula): consider doing this for equalities as well, but probably
  // not worth the savings.
}

void FlatAffineConstraints::clearAndCopyFrom(
    const FlatAffineConstraints &other) {
  FlatAffineConstraints copy(other);
  std::swap(*this, copy);
  assert(copy.getNumIds() == copy.getIds().size());
}

void FlatAffineConstraints::removeId(unsigned pos) {
  removeIdRange(pos, pos + 1);
}

static std::pair<unsigned, unsigned>
getNewNumDimsSymbols(unsigned pos, const FlatAffineConstraints &cst) {
  unsigned numDims = cst.getNumDimIds();
  unsigned numSymbols = cst.getNumSymbolIds();
  unsigned newNumDims, newNumSymbols;
  if (pos < numDims) {
    newNumDims = numDims - 1;
    newNumSymbols = numSymbols;
  } else if (pos < numDims + numSymbols) {
    assert(numSymbols >= 1);
    newNumDims = numDims;
    newNumSymbols = numSymbols - 1;
  } else {
    newNumDims = numDims;
    newNumSymbols = numSymbols;
  }
  return {newNumDims, newNumSymbols};
}

#undef DEBUG_TYPE
#define DEBUG_TYPE "fm"

/// Eliminates identifier at the specified position using Fourier-Motzkin
/// variable elimination. This technique is exact for rational spaces but
/// conservative (in "rare" cases) for integer spaces. The operation corresponds
/// to a projection operation yielding the (convex) set of integer points
/// contained in the rational shadow of the set. An emptiness test that relies
/// on this method will guarantee emptiness, i.e., it disproves the existence of
/// a solution if it says it's empty.
/// If a non-null isResultIntegerExact is passed, it is set to true if the
/// result is also integer exact. If it's set to false, the obtained solution
/// *may* not be exact, i.e., it may contain integer points that do not have an
/// integer pre-image in the original set.
///
/// Eg:
/// j >= 0, j <= i + 1
/// i >= 0, i <= N + 1
/// Eliminating i yields,
///   j >= 0, 0 <= N + 1, j - 1 <= N + 1
///
/// If darkShadow = true, this method computes the dark shadow on elimination;
/// the dark shadow is a convex integer subset of the exact integer shadow. A
/// non-empty dark shadow proves the existence of an integer solution. The
/// elimination in such a case could however be an under-approximation, and thus
/// should not be used for scanning sets or used by itself for dependence
/// checking.
///
/// Eg: 2-d set, * represents grid points, 'o' represents a point in the set.
///            ^
///            |
///            | * * * * o o
///         i  | * * o o o o
///            | o * * * * *
///            --------------->
///                 j ->
///
/// Eliminating i from this system (projecting on the j dimension):
/// rational shadow / integer light shadow:  1 <= j <= 6
/// dark shadow:                             3 <= j <= 6
/// exact integer shadow:                    j = 1 \union  3 <= j <= 6
/// holes/splinters:                         j = 2
///
/// darkShadow = false, isResultIntegerExact = nullptr are default values.
// TODO(bondhugula): a slight modification to yield dark shadow version of FM
// (tightened), which can prove the existence of a solution if there is one.
void FlatAffineConstraints::FourierMotzkinEliminate(
    unsigned pos, bool darkShadow, bool *isResultIntegerExact) {
  LLVM_DEBUG(llvm::dbgs() << "FM input (eliminate pos " << pos << "):\n");
  LLVM_DEBUG(dump());
  assert(pos < getNumIds() && "invalid position");
  assert(hasConsistentState());

  // Check if this identifier can be eliminated through a substitution.
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    if (atEq(r, pos) != 0) {
      // Use Gaussian elimination here (since we have an equality).
      LogicalResult ret = gaussianEliminateId(pos);
      (void)ret;
      assert(succeeded(ret) && "Gaussian elimination guaranteed to succeed");
      LLVM_DEBUG(llvm::dbgs() << "FM output (through Gaussian elimination):\n");
      LLVM_DEBUG(dump());
      return;
    }
  }

  // A fast linear time tightening.
  GCDTightenInequalities();

  // Check if the identifier appears at all in any of the inequalities.
  unsigned r, e;
  for (r = 0, e = getNumInequalities(); r < e; r++) {
    if (atIneq(r, pos) != 0)
      break;
  }
  if (r == getNumInequalities()) {
    // If it doesn't appear, just remove the column and return.
    // TODO(andydavis,bondhugula): refactor removeColumns to use it from here.
    removeId(pos);
    LLVM_DEBUG(llvm::dbgs() << "FM output:\n");
    LLVM_DEBUG(dump());
    return;
  }

  // Positions of constraints that are lower bounds on the variable.
  SmallVector<unsigned, 4> lbIndices;
  // Positions of constraints that are lower bounds on the variable.
  SmallVector<unsigned, 4> ubIndices;
  // Positions of constraints that do not involve the variable.
  std::vector<unsigned> nbIndices;
  nbIndices.reserve(getNumInequalities());

  // Gather all lower bounds and upper bounds of the variable. Since the
  // canonical form c_1*x_1 + c_2*x_2 + ... + c_0 >= 0, a constraint is a lower
  // bound for x_i if c_i >= 1, and an upper bound if c_i <= -1.
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    if (atIneq(r, pos) == 0) {
      // Id does not appear in bound.
      nbIndices.push_back(r);
    } else if (atIneq(r, pos) >= 1) {
      // Lower bound.
      lbIndices.push_back(r);
    } else {
      // Upper bound.
      ubIndices.push_back(r);
    }
  }

  // Set the number of dimensions, symbols in the resulting system.
  const auto &dimsSymbols = getNewNumDimsSymbols(pos, *this);
  unsigned newNumDims = dimsSymbols.first;
  unsigned newNumSymbols = dimsSymbols.second;

  SmallVector<Optional<Value>, 8> newIds;
  newIds.reserve(numIds - 1);
  newIds.append(ids.begin(), ids.begin() + pos);
  newIds.append(ids.begin() + pos + 1, ids.end());

  /// Create the new system which has one identifier less.
  FlatAffineConstraints newFac(
      lbIndices.size() * ubIndices.size() + nbIndices.size(),
      getNumEqualities(), getNumCols() - 1, newNumDims, newNumSymbols,
      /*numLocals=*/getNumIds() - 1 - newNumDims - newNumSymbols, newIds);

  assert(newFac.getIds().size() == newFac.getNumIds());

  // This will be used to check if the elimination was integer exact.
  unsigned lcmProducts = 1;

  // Let x be the variable we are eliminating.
  // For each lower bound, lb <= c_l*x, and each upper bound c_u*x <= ub, (note
  // that c_l, c_u >= 1) we have:
  // lb*lcm(c_l, c_u)/c_l <= lcm(c_l, c_u)*x <= ub*lcm(c_l, c_u)/c_u
  // We thus generate a constraint:
  // lcm(c_l, c_u)/c_l*lb <= lcm(c_l, c_u)/c_u*ub.
  // Note if c_l = c_u = 1, all integer points captured by the resulting
  // constraint correspond to integer points in the original system (i.e., they
  // have integer pre-images). Hence, if the lcm's are all 1, the elimination is
  // integer exact.
  for (auto ubPos : ubIndices) {
    for (auto lbPos : lbIndices) {
      SmallVector<int64_t, 4> ineq;
      ineq.reserve(newFac.getNumCols());
      int64_t lbCoeff = atIneq(lbPos, pos);
      // Note that in the comments above, ubCoeff is the negation of the
      // coefficient in the canonical form as the view taken here is that of the
      // term being moved to the other size of '>='.
      int64_t ubCoeff = -atIneq(ubPos, pos);
      // TODO(bondhugula): refactor this loop to avoid all branches inside.
      for (unsigned l = 0, e = getNumCols(); l < e; l++) {
        if (l == pos)
          continue;
        assert(lbCoeff >= 1 && ubCoeff >= 1 && "bounds wrongly identified");
        int64_t lcm = mlir::lcm(lbCoeff, ubCoeff);
        ineq.push_back(atIneq(ubPos, l) * (lcm / ubCoeff) +
                       atIneq(lbPos, l) * (lcm / lbCoeff));
        lcmProducts *= lcm;
      }
      if (darkShadow) {
        // The dark shadow is a convex subset of the exact integer shadow. If
        // there is a point here, it proves the existence of a solution.
        ineq[ineq.size() - 1] += lbCoeff * ubCoeff - lbCoeff - ubCoeff + 1;
      }
      // TODO: we need to have a way to add inequalities in-place in
      // FlatAffineConstraints instead of creating and copying over.
      newFac.addInequality(ineq);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "FM isResultIntegerExact: " << (lcmProducts == 1)
                          << "\n");
  if (lcmProducts == 1 && isResultIntegerExact)
    *isResultIntegerExact = true;

  // Copy over the constraints not involving this variable.
  for (auto nbPos : nbIndices) {
    SmallVector<int64_t, 4> ineq;
    ineq.reserve(getNumCols() - 1);
    for (unsigned l = 0, e = getNumCols(); l < e; l++) {
      if (l == pos)
        continue;
      ineq.push_back(atIneq(nbPos, l));
    }
    newFac.addInequality(ineq);
  }

  assert(newFac.getNumConstraints() ==
         lbIndices.size() * ubIndices.size() + nbIndices.size());

  // Copy over the equalities.
  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    SmallVector<int64_t, 4> eq;
    eq.reserve(newFac.getNumCols());
    for (unsigned l = 0, e = getNumCols(); l < e; l++) {
      if (l == pos)
        continue;
      eq.push_back(atEq(r, l));
    }
    newFac.addEquality(eq);
  }

  // GCD tightening and normalization allows detection of more trivially
  // redundant constraints.
  newFac.GCDTightenInequalities();
  newFac.normalizeConstraintsByGCD();
  newFac.removeTrivialRedundancy();
  clearAndCopyFrom(newFac);
  LLVM_DEBUG(llvm::dbgs() << "FM output:\n");
  LLVM_DEBUG(dump());
}

#undef DEBUG_TYPE
#define DEBUG_TYPE "affine-structures"

void FlatAffineConstraints::projectOut(unsigned pos, unsigned num) {
  if (num == 0)
    return;

  // 'pos' can be at most getNumCols() - 2 if num > 0.
  assert((getNumCols() < 2 || pos <= getNumCols() - 2) && "invalid position");
  assert(pos + num < getNumCols() && "invalid range");

  // Eliminate as many identifiers as possible using Gaussian elimination.
  unsigned currentPos = pos;
  unsigned numToEliminate = num;
  unsigned numGaussianEliminated = 0;

  while (currentPos < getNumIds()) {
    unsigned curNumEliminated =
        gaussianEliminateIds(currentPos, currentPos + numToEliminate);
    ++currentPos;
    numToEliminate -= curNumEliminated + 1;
    numGaussianEliminated += curNumEliminated;
  }

  // Eliminate the remaining using Fourier-Motzkin.
  for (unsigned i = 0; i < num - numGaussianEliminated; i++) {
    unsigned numToEliminate = num - numGaussianEliminated - i;
    FourierMotzkinEliminate(
        getBestIdToEliminate(*this, pos, pos + numToEliminate));
  }

  // Fast/trivial simplifications.
  GCDTightenInequalities();
  // Normalize constraints after tightening since the latter impacts this, but
  // not the other way round.
  normalizeConstraintsByGCD();
}

void FlatAffineConstraints::projectOut(Value id) {
  unsigned pos;
  bool ret = findId(*id, &pos);
  assert(ret);
  (void)ret;
  FourierMotzkinEliminate(pos);
}

void FlatAffineConstraints::clearConstraints() {
  equalities.clear();
  inequalities.clear();
}

namespace {

enum BoundCmpResult { Greater, Less, Equal, Unknown };

/// Compares two affine bounds whose coefficients are provided in 'first' and
/// 'second'. The last coefficient is the constant term.
static BoundCmpResult compareBounds(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  assert(a.size() == b.size());

  // For the bounds to be comparable, their corresponding identifier
  // coefficients should be equal; the constant terms are then compared to
  // determine less/greater/equal.

  if (!std::equal(a.begin(), a.end() - 1, b.begin()))
    return Unknown;

  if (a.back() == b.back())
    return Equal;

  return a.back() < b.back() ? Less : Greater;
}
} // namespace

// Computes the bounding box with respect to 'other' by finding the min of the
// lower bounds and the max of the upper bounds along each of the dimensions.
LogicalResult
FlatAffineConstraints::unionBoundingBox(const FlatAffineConstraints &otherCst) {
  assert(otherCst.getNumDimIds() == numDims && "dims mismatch");
  assert(otherCst.getIds()
             .slice(0, getNumDimIds())
             .equals(getIds().slice(0, getNumDimIds())) &&
         "dim values mismatch");
  assert(otherCst.getNumLocalIds() == 0 && "local ids not supported here");
  assert(getNumLocalIds() == 0 && "local ids not supported yet here");

  Optional<FlatAffineConstraints> otherCopy;
  if (!areIdsAligned(*this, otherCst)) {
    otherCopy.emplace(FlatAffineConstraints(otherCst));
    mergeAndAlignIds(/*offset=*/numDims, this, &otherCopy.getValue());
  }

  const auto &other = otherCopy ? *otherCopy : otherCst;

  std::vector<SmallVector<int64_t, 8>> boundingLbs;
  std::vector<SmallVector<int64_t, 8>> boundingUbs;
  boundingLbs.reserve(2 * getNumDimIds());
  boundingUbs.reserve(2 * getNumDimIds());

  // To hold lower and upper bounds for each dimension.
  SmallVector<int64_t, 4> lb, otherLb, ub, otherUb;
  // To compute min of lower bounds and max of upper bounds for each dimension.
  SmallVector<int64_t, 4> minLb(getNumSymbolIds() + 1);
  SmallVector<int64_t, 4> maxUb(getNumSymbolIds() + 1);
  // To compute final new lower and upper bounds for the union.
  SmallVector<int64_t, 8> newLb(getNumCols()), newUb(getNumCols());

  int64_t lbFloorDivisor, otherLbFloorDivisor;
  for (unsigned d = 0, e = getNumDimIds(); d < e; ++d) {
    auto extent = getConstantBoundOnDimSize(d, &lb, &lbFloorDivisor, &ub);
    if (!extent.hasValue())
      // TODO(bondhugula): symbolic extents when necessary.
      // TODO(bondhugula): handle union if a dimension is unbounded.
      return failure();

    auto otherExtent = other.getConstantBoundOnDimSize(
        d, &otherLb, &otherLbFloorDivisor, &otherUb);
    if (!otherExtent.hasValue() || lbFloorDivisor != otherLbFloorDivisor)
      // TODO(bondhugula): symbolic extents when necessary.
      return failure();

    assert(lbFloorDivisor > 0 && "divisor always expected to be positive");

    auto res = compareBounds(lb, otherLb);
    // Identify min.
    if (res == BoundCmpResult::Less || res == BoundCmpResult::Equal) {
      minLb = lb;
      // Since the divisor is for a floordiv, we need to convert to ceildiv,
      // i.e., i >= expr floordiv div <=> i >= (expr - div + 1) ceildiv div <=>
      // div * i >= expr - div + 1.
      minLb.back() -= lbFloorDivisor - 1;
    } else if (res == BoundCmpResult::Greater) {
      minLb = otherLb;
      minLb.back() -= otherLbFloorDivisor - 1;
    } else {
      // Uncomparable - check for constant lower/upper bounds.
      auto constLb = getConstantLowerBound(d);
      auto constOtherLb = other.getConstantLowerBound(d);
      if (!constLb.hasValue() || !constOtherLb.hasValue())
        return failure();
      std::fill(minLb.begin(), minLb.end(), 0);
      minLb.back() = std::min(constLb.getValue(), constOtherLb.getValue());
    }

    // Do the same for ub's but max of upper bounds. Identify max.
    auto uRes = compareBounds(ub, otherUb);
    if (uRes == BoundCmpResult::Greater || uRes == BoundCmpResult::Equal) {
      maxUb = ub;
    } else if (uRes == BoundCmpResult::Less) {
      maxUb = otherUb;
    } else {
      // Uncomparable - check for constant lower/upper bounds.
      auto constUb = getConstantUpperBound(d);
      auto constOtherUb = other.getConstantUpperBound(d);
      if (!constUb.hasValue() || !constOtherUb.hasValue())
        return failure();
      std::fill(maxUb.begin(), maxUb.end(), 0);
      maxUb.back() = std::max(constUb.getValue(), constOtherUb.getValue());
    }

    std::fill(newLb.begin(), newLb.end(), 0);
    std::fill(newUb.begin(), newUb.end(), 0);

    // The divisor for lb, ub, otherLb, otherUb at this point is lbDivisor,
    // and so it's the divisor for newLb and newUb as well.
    newLb[d] = lbFloorDivisor;
    newUb[d] = -lbFloorDivisor;
    // Copy over the symbolic part + constant term.
    std::copy(minLb.begin(), minLb.end(), newLb.begin() + getNumDimIds());
    std::transform(newLb.begin() + getNumDimIds(), newLb.end(),
                   newLb.begin() + getNumDimIds(), std::negate<int64_t>());
    std::copy(maxUb.begin(), maxUb.end(), newUb.begin() + getNumDimIds());

    boundingLbs.push_back(newLb);
    boundingUbs.push_back(newUb);
  }

  // Clear all constraints and add the lower/upper bounds for the bounding box.
  clearConstraints();
  for (unsigned d = 0, e = getNumDimIds(); d < e; ++d) {
    addInequality(boundingLbs[d]);
    addInequality(boundingUbs[d]);
  }
  // TODO(mlir-team): copy over pure symbolic constraints from this and 'other'
  // over to the union (since the above are just the union along dimensions); we
  // shouldn't be discarding any other constraints on the symbols.

  return success();
}
