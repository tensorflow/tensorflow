//===- AffineStructures.cpp - MLIR Affine Structures Class-------*- C++ -*-===//
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
// Structures for affine/polyhedral analysis of MLIR functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLValue.h"
#include "mlir/IR/Statements.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "affine-structures"

using namespace mlir;
using namespace llvm;

namespace {

// Affine map composition terminology:
// *) current: refers to the target map of the composition operation. It is the
//    map into which results from the 'input' map are forward substituted.
// *) input: refers to the map which is being forward substituted into the
//    'current' map.
// *) output: refers to the resulting affine map after composition.

// AffineMapCompositionUpdate encapsulates the state necessary to compose
// AffineExprs for two affine maps using AffineExprComposer (see below).
struct AffineMapCompositionUpdate {
  using PositionMap = DenseMap<unsigned, unsigned>;

  explicit AffineMapCompositionUpdate(ArrayRef<AffineExpr> inputResults)
      : inputResults(inputResults), outputNumDims(0), outputNumSymbols(0) {}

  // Map from 'curr' affine map dim position to 'output' affine map
  // dim position.
  PositionMap currDimMap;
  // Map from dim position of 'curr' affine map to index into 'inputResults'.
  PositionMap currDimToInputResultMap;
  // Map from 'curr' affine map symbol position to 'output' affine map
  // symbol position.
  PositionMap currSymbolMap;
  // Map from 'input' affine map dim position to 'output' affine map
  // dim position.
  PositionMap inputDimMap;
  // Map from 'input' affine map symbol position to 'output' affine map
  // symbol position.
  PositionMap inputSymbolMap;
  // Results of 'input' affine map.
  ArrayRef<AffineExpr> inputResults;
  // Number of dimension operands for 'output' affine map.
  unsigned outputNumDims;
  // Number of symbol operands for 'output' affine map.
  unsigned outputNumSymbols;
};

// AffineExprComposer composes two AffineExprs as specified by 'mapUpdate'.
class AffineExprComposer {
public:
  // Compose two AffineExprs using dimension and symbol position update maps,
  // as well as input map result AffineExprs specified in 'mapUpdate'.
  AffineExprComposer(const AffineMapCompositionUpdate &mapUpdate)
      : mapUpdate(mapUpdate), walkingInputMap(false) {}

  AffineExpr walk(AffineExpr expr) {
    switch (expr.getKind()) {
    case AffineExprKind::Add:
      return walkBinExpr(
          expr, [](AffineExpr lhs, AffineExpr rhs) { return lhs + rhs; });
    case AffineExprKind::Mul:
      return walkBinExpr(
          expr, [](AffineExpr lhs, AffineExpr rhs) { return lhs * rhs; });
    case AffineExprKind::Mod:
      return walkBinExpr(
          expr, [](AffineExpr lhs, AffineExpr rhs) { return lhs % rhs; });
    case AffineExprKind::FloorDiv:
      return walkBinExpr(expr, [](AffineExpr lhs, AffineExpr rhs) {
        return lhs.floorDiv(rhs);
      });
    case AffineExprKind::CeilDiv:
      return walkBinExpr(expr, [](AffineExpr lhs, AffineExpr rhs) {
        return lhs.ceilDiv(rhs);
      });
    case AffineExprKind::Constant:
      return expr;
    case AffineExprKind::DimId: {
      unsigned dimPosition = expr.cast<AffineDimExpr>().getPosition();
      if (walkingInputMap) {
        return getAffineDimExpr(mapUpdate.inputDimMap.lookup(dimPosition),
                                expr.getContext());
      }
      // Check if we are just mapping this dim to another position.
      if (mapUpdate.currDimMap.count(dimPosition) > 0) {
        assert(mapUpdate.currDimToInputResultMap.count(dimPosition) == 0);
        return getAffineDimExpr(mapUpdate.currDimMap.lookup(dimPosition),
                                expr.getContext());
      }
      // We are substituting an input map result at 'dimPositon'
      // Forward substitute currDimToInputResultMap[dimPosition] into this
      // map.
      AffineExprComposer composer(mapUpdate, /*walkingInputMap=*/true);
      unsigned inputResultIndex =
          mapUpdate.currDimToInputResultMap.lookup(dimPosition);
      assert(inputResultIndex < mapUpdate.inputResults.size());
      return composer.walk(mapUpdate.inputResults[inputResultIndex]);
    }
    case AffineExprKind::SymbolId:
      unsigned symbolPosition = expr.cast<AffineSymbolExpr>().getPosition();
      if (walkingInputMap) {
        return getAffineSymbolExpr(
            mapUpdate.inputSymbolMap.lookup(symbolPosition), expr.getContext());
      }
      return getAffineSymbolExpr(mapUpdate.currSymbolMap.lookup(symbolPosition),
                                 expr.getContext());
    }
  }

private:
  AffineExprComposer(const AffineMapCompositionUpdate &mapUpdate,
                     bool walkingInputMap)
      : mapUpdate(mapUpdate), walkingInputMap(walkingInputMap) {}

  AffineExpr walkBinExpr(AffineExpr expr,
                         std::function<AffineExpr(AffineExpr, AffineExpr)> op) {
    auto binOpExpr = expr.cast<AffineBinaryOpExpr>();
    return op(walk(binOpExpr.getLHS()), walk(binOpExpr.getRHS()));
  }

  // Map update specifies to dim and symbol postion maps, as well as the input
  // result AffineExprs to forward subustitute into the input map.
  const AffineMapCompositionUpdate &mapUpdate;
  // True if we are walking an AffineExpr in the 'input' map, false if
  // we are walking the 'input' map.
  bool walkingInputMap;
};

} // end anonymous namespace

static void
forwardSubstituteMutableAffineMap(const AffineMapCompositionUpdate &mapUpdate,
                                  MutableAffineMap *map) {
  for (unsigned i = 0, e = map->getNumResults(); i < e; i++) {
    AffineExprComposer composer(mapUpdate);
    map->setResult(i, composer.walk(map->getResult(i)));
  }
  // TODO(andydavis) Evaluate if we need to update range sizes here.
  map->setNumDims(mapUpdate.outputNumDims);
  map->setNumSymbols(mapUpdate.outputNumSymbols);
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
  for (auto rangeSize : map.getRangeSizes())
    results.push_back(rangeSize);
}

void MutableAffineMap::reset(AffineMap map) {
  results.clear();
  rangeSizes.clear();
  numDims = map.getNumDims();
  numSymbols = map.getNumSymbols();
  // A map always has at least 1 result by construction
  context = map.getResult(0).getContext();
  for (auto result : map.getResults())
    results.push_back(result);
  for (auto rangeSize : map.getRangeSizes())
    results.push_back(rangeSize);
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
  return AffineMap::get(numDims, numSymbols, results, rangeSizes);
}

MutableIntegerSet::MutableIntegerSet(IntegerSet set, MLIRContext *context)
    : numDims(set.getNumDims()), numSymbols(set.getNumSymbols()),
      context(context) {
  // TODO(bondhugula)
}

// Universal set.
MutableIntegerSet::MutableIntegerSet(unsigned numDims, unsigned numSymbols,
                                     MLIRContext *context)
    : numDims(numDims), numSymbols(numSymbols), context(context) {}

//===----------------------------------------------------------------------===//
// AffineValueMap.
//===----------------------------------------------------------------------===//

AffineValueMap::AffineValueMap(const AffineApplyOp &op)
    : map(op.getAffineMap()) {
  for (auto *operand : op.getOperands())
    operands.push_back(cast<MLValue>(const_cast<SSAValue *>(operand)));
  for (unsigned i = 0, e = op.getNumResults(); i < e; i++)
    results.push_back(cast<MLValue>(const_cast<SSAValue *>(op.getResult(i))));
}

AffineValueMap::AffineValueMap(AffineMap map, ArrayRef<MLValue *> operands)
    : map(map) {
  for (MLValue *operand : operands) {
    this->operands.push_back(operand);
  }
}

void AffineValueMap::reset(AffineMap map, ArrayRef<MLValue *> operands) {
  this->operands.clear();
  this->results.clear();
  this->map.reset(map);
  for (MLValue *operand : operands) {
    this->operands.push_back(operand);
  }
}

void AffineValueMap::forwardSubstitute(const AffineApplyOp &inputOp) {
  SmallVector<bool, 4> inputResultsToSubstitute(inputOp.getNumResults());
  for (unsigned i = 0, e = inputOp.getNumResults(); i < e; i++)
    inputResultsToSubstitute[i] = true;
  forwardSubstitute(inputOp, inputResultsToSubstitute);
}

void AffineValueMap::forwardSubstituteSingle(const AffineApplyOp &inputOp,
                                             unsigned inputResultIndex) {
  SmallVector<bool, 4> inputResultsToSubstitute(inputOp.getNumResults(), false);
  inputResultsToSubstitute[inputResultIndex] = true;
  forwardSubstitute(inputOp, inputResultsToSubstitute);
}

// Returns true and sets 'indexOfMatch' if 'valueToMatch' is found in
// 'valuesToSearch' beginning at 'indexStart'. Returns false otherwise.
static bool findIndex(MLValue *valueToMatch, ArrayRef<MLValue *> valuesToSearch,
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

// AffineValueMap forward substitution composes results from the affine map
// associated with 'inputOp', with the map it currently represents. This is
// accomplished by updating its MutableAffineMap and operand list to represent
// a new 'output' map which is the composition of the 'current' and 'input'
// maps (see "Affine map composition terminology" above for details).
//
// Affine map forward substitution is comprised of the following steps:
// *) Compute input affine map result indices used by the current map.
// *) Gather all dim and symbol positions from all AffineExpr input results
//    computed in previous step.
// *) Build output operand list:
//  *) Add curr map dim operands:
//    *) If curr dim operand is being forward substituted by result of input
//       map, store mapping from curr postion to input result index.
//    *) Else add curr dim operand to output operand list.
//  *) Add input map dim operands:
//    *) If input map dim operand is used (step 2), add to output operand
//       list (scanning current list for dups before updating mapping).
//  *) Add curr map dim symbols.
//  *) Add input map dim symbols (if used from step 2), dedup if needed.
// *) Update operands and forward substitute new dim and symbol mappings
//    into MutableAffineMap 'map'.
//
// TODO(andydavis) Move this to a function which can be shared with
// forwardSubstitute(const AffineValueMap &inputMap).
void AffineValueMap::forwardSubstitute(
    const AffineApplyOp &inputOp, ArrayRef<bool> inputResultsToSubstitute) {
  unsigned currNumDims = map.getNumDims();
  unsigned inputNumResults = inputOp.getNumResults();

  // Gather result indices from 'inputOp' used by current map.
  DenseSet<unsigned> inputResultsUsed;
  DenseMap<unsigned, unsigned> currOperandToInputResult;
  for (unsigned i = 0; i < currNumDims; ++i) {
    for (unsigned j = 0; j < inputNumResults; ++j) {
      if (!inputResultsToSubstitute[j])
        continue;
      if (operands[i] ==
          cast<MLValue>(const_cast<SSAValue *>(inputOp.getResult(j)))) {
        currOperandToInputResult[i] = j;
        inputResultsUsed.insert(j);
      }
    }
  }

  // Return if there were no uses of 'inputOp' results in 'operands'.
  if (inputResultsUsed.empty()) {
    return;
  }

  class AffineExprPositionGatherer
      : public AffineExprVisitor<AffineExprPositionGatherer> {
  public:
    unsigned numDims;
    DenseSet<unsigned> *positions;
    AffineExprPositionGatherer(unsigned numDims, DenseSet<unsigned> *positions)
        : numDims(numDims), positions(positions) {}
    void visitDimExpr(AffineDimExpr expr) {
      positions->insert(expr.getPosition());
    }
    void visitSymbolExpr(AffineSymbolExpr expr) {
      positions->insert(numDims + expr.getPosition());
    }
  };

  // Gather dim and symbol positions from 'inputOp' on which
  // 'inputResultsUsed' depend.
  AffineMap inputMap = inputOp.getAffineMap();
  unsigned inputNumDims = inputMap.getNumDims();
  DenseSet<unsigned> inputPositionsUsed;
  AffineExprPositionGatherer gatherer(inputNumDims, &inputPositionsUsed);
  for (unsigned i = 0; i < inputNumResults; ++i) {
    if (inputResultsUsed.count(i) == 0)
      continue;
    gatherer.walkPostOrder(inputMap.getResult(i));
  }

  // Build new output operands list and map update.
  SmallVector<MLValue *, 4> outputOperands;
  unsigned outputOperandPosition = 0;
  AffineMapCompositionUpdate mapUpdate(inputOp.getAffineMap().getResults());

  // Add dim operands from current map.
  for (unsigned i = 0; i < currNumDims; ++i) {
    if (currOperandToInputResult.count(i) > 0) {
      mapUpdate.currDimToInputResultMap[i] = currOperandToInputResult[i];
    } else {
      mapUpdate.currDimMap[i] = outputOperandPosition++;
      outputOperands.push_back(operands[i]);
    }
  }

  // Add dim operands from input map.
  for (unsigned i = 0; i < inputNumDims; ++i) {
    // Skip input dim operands that we won't use.
    if (inputPositionsUsed.count(i) == 0)
      continue;
    // Check if input operand has a dup in current operand list.
    auto *inputOperand =
        cast<MLValue>(const_cast<SSAValue *>(inputOp.getOperand(i)));
    unsigned outputIndex;
    if (findIndex(inputOperand, outputOperands, /*indexStart=*/0,
                  &outputIndex)) {
      mapUpdate.inputDimMap[i] = outputIndex;
    } else {
      mapUpdate.inputDimMap[i] = outputOperandPosition++;
      outputOperands.push_back(inputOperand);
    }
  }

  // Done adding dimension operands, so store new output num dims.
  unsigned outputNumDims = outputOperandPosition;

  // Add symbol operands from current map.
  unsigned currNumOperands = operands.size();
  for (unsigned i = currNumDims; i < currNumOperands; ++i) {
    unsigned currSymbolPosition = i - currNumDims;
    unsigned outputSymbolPosition = outputOperandPosition - outputNumDims;
    mapUpdate.currSymbolMap[currSymbolPosition] = outputSymbolPosition;
    outputOperands.push_back(operands[i]);
    ++outputOperandPosition;
  }

  // Add symbol operands from input map.
  unsigned inputNumOperands = inputOp.getNumOperands();
  for (unsigned i = inputNumDims; i < inputNumOperands; ++i) {
    // Skip input symbol operands that we won't use.
    if (inputPositionsUsed.count(i) == 0)
      continue;
    unsigned inputSymbolPosition = i - inputNumDims;
    // Check if input operand has a dup in current operand list.
    auto *inputOperand =
        cast<MLValue>(const_cast<SSAValue *>(inputOp.getOperand(i)));
    // Find output operand index of 'inputOperand' dup.
    unsigned outputIndex;
    // Start at index 'outputNumDims' so that only symbol operands are searched.
    if (findIndex(inputOperand, outputOperands, /*indexStart=*/outputNumDims,
                  &outputIndex)) {
      unsigned outputSymbolPosition = outputIndex - outputNumDims;
      mapUpdate.inputSymbolMap[inputSymbolPosition] = outputSymbolPosition;
    } else {
      unsigned outputSymbolPosition = outputOperandPosition - outputNumDims;
      mapUpdate.inputSymbolMap[inputSymbolPosition] = outputSymbolPosition;
      outputOperands.push_back(inputOperand);
      ++outputOperandPosition;
    }
  }

  // Set output number of dimension and symbol operands.
  mapUpdate.outputNumDims = outputNumDims;
  mapUpdate.outputNumSymbols = outputOperands.size() - outputNumDims;

  // Update 'operands' with new 'outputOperands'.
  operands.swap(outputOperands);
  // Forward substitute 'mapUpdate' into 'map'.
  forwardSubstituteMutableAffineMap(mapUpdate, &map);
}

inline bool AffineValueMap::isMultipleOf(unsigned idx, int64_t factor) const {
  return map.isMultipleOf(idx, factor);
}

/// This method uses the invariant that operands are always positionally aligned
/// with the AffineDimExpr in the underlying AffineMap.
bool AffineValueMap::isFunctionOf(unsigned idx, MLValue *value) const {
  unsigned index;
  findIndex(value, operands, /*indexStart=*/0, &index);
  auto expr = const_cast<AffineValueMap *>(this)->getAffineMap().getResult(idx);
  // TODO(ntv): this is better implemented on a flattened representation.
  // At least for now it is conservative.
  return expr.isFunctionOfDim(index);
}

SSAValue *AffineValueMap::getOperand(unsigned i) const {
  return static_cast<SSAValue *>(operands[i]);
}

ArrayRef<MLValue *> AffineValueMap::getOperands() const {
  return ArrayRef<MLValue *>(operands);
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
  ids.insert(ids.end(), otherIds.begin(), otherIds.end());

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
    : numReservedCols(set.getNumOperands() + 1),
      numIds(set.getNumDims() + set.getNumSymbols()), numDims(set.getNumDims()),
      numSymbols(set.getNumSymbols()) {
  equalities.reserve(set.getNumEqualities() * numReservedCols);
  inequalities.reserve(set.getNumInequalities() * numReservedCols);
  ids.resize(numIds, None);

  for (unsigned i = 0, e = set.getNumConstraints(); i < e; ++i) {
    AffineExpr expr = set.getConstraint(i);
    SmallVector<int64_t, 4> flattenedExpr;
    getFlattenedAffineExpr(expr, set.getNumDims(), set.getNumSymbols(),
                           &flattenedExpr);
    assert(flattenedExpr.size() == getNumCols());
    if (set.getEqFlags()[i]) {
      addEquality(flattenedExpr);
    } else {
      addInequality(flattenedExpr);
    }
  }
}

void FlatAffineConstraints::reset(unsigned numReservedInequalities,
                                  unsigned numReservedEqualities,
                                  unsigned newNumReservedCols,
                                  unsigned newNumDims, unsigned newNumSymbols,
                                  unsigned newNumLocals,
                                  ArrayRef<MLValue *> idArgs) {
  assert(newNumReservedCols >= newNumDims + newNumSymbols + newNumLocals + 1 &&
         "minimum 1 column");
  numReservedCols = newNumReservedCols;
  numDims = newNumDims;
  numSymbols = newNumSymbols;
  numIds = numDims + numSymbols + newNumLocals;
  equalities.clear();
  inequalities.clear();
  if (numReservedEqualities >= 1)
    equalities.reserve(newNumReservedCols * numReservedEqualities);
  if (numReservedInequalities >= 1)
    inequalities.reserve(newNumReservedCols * numReservedInequalities);
  ids.clear();
  if (idArgs.empty()) {
    ids.resize(numIds, None);
  } else {
    ids.reserve(idArgs.size());
    ids.insert(ids.end(), idArgs.begin(), idArgs.end());
  }
}

void FlatAffineConstraints::reset(unsigned newNumDims, unsigned newNumSymbols,
                                  unsigned newNumLocals,
                                  ArrayRef<MLValue *> idArgs) {
  reset(0, 0, newNumDims + newNumSymbols + newNumLocals + 1, newNumDims,
        newNumSymbols, newNumLocals, idArgs);
}

void FlatAffineConstraints::append(const FlatAffineConstraints &other) {
  assert(other.getNumCols() == getNumCols());
  assert(other.getNumDimIds() == getNumDimIds());

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

void FlatAffineConstraints::addDimId(unsigned pos, MLValue *id) {
  addId(IdKind::Dimension, pos, id);
}

void FlatAffineConstraints::addSymbolId(unsigned pos) {
  addId(IdKind::Symbol, pos);
}

/// Adds a dimensional identifier. The added column is initialized to
/// zero.
void FlatAffineConstraints::addId(IdKind kind, unsigned pos, MLValue *id) {
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

  unsigned absolutePos;

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

// This routine may add additional local variables if the flattened
// expression corresponding to the map has such variables due to the presence of
// mod's, ceildiv's, and floordiv's.
bool FlatAffineConstraints::composeMap(AffineValueMap *vMap, unsigned pos) {
  assert(pos <= getNumIds() && "invalid position");
  assert(vMap->getNumSymbols() == getNumSymbolIds());

  AffineMap map = vMap->getAffineMap();

  // We add one equality for each result connecting the result dim of the map to
  // the other identifiers.
  // For eg: if the expression is 16*i0 + i1, and this is the r^th
  // iteration/result of the value map, we are adding the equality:
  //  d_r - 16*i0 - i1 = 0. Hence, when flattening say (i0 + 1, i0 + 8*i2), we
  //  add two equalities overall: d_0 - i0 - 1 == 0, d1 - i0 - 8*i2 == 0.
  for (unsigned r = 0, e = map.getNumResults(); r < e; r++) {
    // Add dimension.
    addDimId(pos + r);
    SmallVector<int64_t, 4> eq;
    eq.reserve(getNumCols());
    FlatAffineConstraints cst;
    bool ret = getFlattenedAffineExpr(map.getResult(r), map.getNumDims(),
                                      map.getNumSymbols(), &eq, &cst);
    if (!ret) {
      LLVM_DEBUG(llvm::dbgs()
                 << "composition unimplemented for semi-affine maps");
      return false;
    }
    // Make the value map and the flat affine cst dimensions compatible.
    // A lot of this code will be refactored/cleaned up.
    for (unsigned l = 0, e = cst.getNumLocalIds(); l < e; l++) {
      addLocalId(0);
    }
    // TODO(bondhugula): the next ~20 lines of code is pretty UGLY. This needs
    // to be factored out into an FlatAffineConstraints::alignAndMerge().
    for (unsigned t = 0, e = r + 1; t < e; t++) {
      // TODO: Consider using a batched version to add a range of IDs.
      cst.addDimId(0);
    }

    assert(cst.getNumDimIds() <= getNumDimIds());
    for (unsigned t = 0, e = getNumDimIds() - cst.getNumDimIds(); t < e; t++) {
      // Dimensions that are in 'this' but not in vMap/cst are added at the end.
      cst.addDimId(cst.getNumDimIds());
    }
    assert(cst.getNumLocalIds() <= getNumLocalIds());
    for (unsigned t = 0, e = getNumLocalIds() - cst.getNumLocalIds(); t < e;
         t++) {
      cst.addLocalId(cst.getNumLocalIds());
    }
    /// Finally, append cst to this constraint set.
    append(cst);

    // eqToAdd is the equality corresponding to the flattened affine expression.
    SmallVector<int64_t, 8> eqToAdd(getNumCols(), 0);
    // Set the coefficient for this result to one.
    eqToAdd[r] = 1;

    // Dims and symbols.
    for (unsigned i = 0, e = vMap->getNumOperands(); i < e; i++) {
      unsigned loc;
      bool ret = findId(*cast<MLValue>(vMap->getOperand(i)), &loc);
      assert(ret && "id expected, but not found");
      (void)ret;
      // We need to negate 'eq' since the newly added dimension is going to be
      // set to this one.
      eqToAdd[loc] = -eq[i];
    }
    // Local vars common to eq and cst are at the beginning.
    int j = getNumDimIds() + getNumSymbolIds();
    int end = eq.size() - 1;
    for (int i = vMap->getNumOperands(); i < end; i++, j++) {
      eqToAdd[j] = -eq[i];
    }

    // Constant term.
    eqToAdd[getNumCols() - 1] = -eq[eq.size() - 1];

    // Add the equality connecting the result of the map to this constraint set.
    addEquality(eqToAdd);
  }
  return true;
}

// Searches for a constraint with a non-zero coefficient at 'colIdx' in
// equality (isEq=true) or inequality (isEq=false) constraints.
// Returns true and sets row found in search in 'rowIdx'.
// Returns false otherwise.
static bool
findConstraintWithNonZeroAt(const FlatAffineConstraints &constraints,
                            unsigned colIdx, bool isEq, unsigned &rowIdx) {
  auto at = [&](unsigned rowIdx) -> int64_t {
    return isEq ? constraints.atEq(rowIdx, colIdx)
                : constraints.atIneq(rowIdx, colIdx);
  };
  unsigned e =
      isEq ? constraints.getNumEqualities() : constraints.getNumInequalities();
  for (rowIdx = 0; rowIdx < e; ++rowIdx) {
    if (at(rowIdx) != 0) {
      return true;
    }
  }
  return false;
}

// Normalizes the coefficient values across all columns in 'rowIDx' by their
// GCD in equality or inequality contraints as specified by 'isEq'.
static void normalizeConstraintByGCD(FlatAffineConstraints *constraints,
                                     unsigned rowIdx, bool isEq) {
  auto at = [&](unsigned colIdx) -> int64_t {
    return isEq ? constraints->atEq(rowIdx, colIdx)
                : constraints->atIneq(rowIdx, colIdx);
  };
  uint64_t gcd = std::abs(at(0));
  for (unsigned j = 1; j < constraints->getNumCols(); ++j) {
    gcd = llvm::GreatestCommonDivisor64(gcd, std::abs(at(j)));
  }
  if (gcd > 0 && gcd != 1) {
    for (unsigned j = 0; j < constraints->getNumCols(); ++j) {
      int64_t v = at(j) / static_cast<int64_t>(gcd);
      isEq ? constraints->atEq(rowIdx, j) = v
           : constraints->atIneq(rowIdx, j) = v;
    }
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
/// 'false'otherwise.
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
  assert(colStart >= 0 && colLimit <= constraints->getNumIds());
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

// Removes coefficients in column range [colStart, colLimit), and copies any
// remaining valid data into place, and updates member variables.
void FlatAffineConstraints::removeColumnRange(unsigned colStart,
                                              unsigned colLimit) {
  assert(colStart >= 0 && colLimit <= getNumCols());
  // TODO(andydavis) Make 'removeColumns' a lambda called from here.
  // Remove eliminated columns from equalities.
  shiftColumnsToLeft(this, colStart, colLimit, /*isEq=*/true);
  // Remove eliminated columns from inequalities.
  shiftColumnsToLeft(this, colStart, colLimit, /*isEq=*/false);
  // Update members numDims, numSymbols and numIds.
  unsigned numDimsEliminated = 0;
  if (colStart < numDims) {
    numDimsEliminated = std::min(numDims, colLimit) - colStart;
  }
  unsigned numColsEliminated = colLimit - colStart;
  unsigned numSymbolsEliminated =
      std::min(numSymbols, numColsEliminated - numDimsEliminated);
  numDims -= numDimsEliminated;
  numSymbols -= numSymbolsEliminated;
  numIds = numIds - numColsEliminated;
  ids.erase(ids.begin() + colStart, ids.begin() + colLimit);

  // No resize necessary. numReservedCols remains the same.
}

// Performs variable elimination on all identifiers, runs the GCD test on
// all equality constraint rows, and checks the constraint validity.
// Returns 'true' if the GCD test fails on any row, or if any invalid
// constraint is detected. Returns 'false' otherwise.
bool FlatAffineConstraints::isEmpty() const {
  if (isEmptyByGCDTest())
    return true;
  auto tmpCst = clone();
  if (tmpCst->gaussianEliminateIds(0, numIds) < numIds) {
    for (unsigned i = 0, e = tmpCst->getNumIds(); i < e; i++)
      tmpCst->FourierMotzkinEliminate(0);
  }
  if (tmpCst->hasInvalidConstraint())
    return true;
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
/// similar to the GCD test but applied to inequalities. The constant term can
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
    if (gcd > 0) {
      atIneq(i, numCols - 1) =
          gcd * mlir::floorDiv(atIneq(i, numCols - 1), gcd);
    }
  }
}

// Eliminates a single identifier at 'position' from equality and inequality
// constraints. Returns 'true' if the identifier was eliminated.
// Returns 'false' otherwise.
bool FlatAffineConstraints::gaussianEliminateId(unsigned position) {
  return gaussianEliminateIds(position, position + 1) == 1;
}

// Eliminates all identifer variables in column range [posStart, posLimit).
// Returns the number of variables eliminated.
unsigned FlatAffineConstraints::gaussianEliminateIds(unsigned posStart,
                                                     unsigned posLimit) {
  // Return if identifier positions to eliminate are out of range.
  assert(posStart >= 0 && posLimit <= numIds);
  assert(hasConsistentState());

  if (posStart >= posLimit)
    return 0;

  GCDTightenInequalities();

  unsigned pivotCol = 0;
  for (pivotCol = posStart; pivotCol < posLimit; ++pivotCol) {
    // Find a row which has a non-zero coefficient in column 'j'.
    unsigned pivotRow;
    if (!findConstraintWithNonZeroAt(*this, pivotCol, /*isEq=*/true,
                                     pivotRow)) {
      // No pivot row in equalities with non-zero at 'pivotCol'.
      if (!findConstraintWithNonZeroAt(*this, pivotCol, /*isEq=*/false,
                                       pivotRow)) {
        // If inequalities are also non-zero in 'pivotCol' it can be eliminated.
        continue;
      }
      break;
    }

    // Eliminate identifier at 'pivotCol' from each equality row.
    for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
      eliminateFromConstraint(this, i, pivotRow, pivotCol, posStart,
                              /*isEq=*/true);
      normalizeConstraintByGCD(this, i, /*isEq=*/true);
    }

    // Eliminate identifier at 'pivotCol' from each inequality row.
    for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
      eliminateFromConstraint(this, i, pivotRow, pivotCol, posStart,
                              /*isEq=*/false);
      normalizeConstraintByGCD(this, i, /*isEq=*/false);
    }
    removeEquality(pivotRow);
  }
  // Update position limit based on number eliminated.
  posLimit = pivotCol;
  // Remove eliminated columns from all constraints.
  removeColumnRange(posStart, posLimit);
  return posLimit - posStart;
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

void FlatAffineConstraints::addLowerBound(ArrayRef<int64_t> expr,
                                          ArrayRef<int64_t> lb) {
  assert(expr.size() == getNumCols());
  assert(lb.size() == getNumCols());
  unsigned offset = inequalities.size();
  inequalities.resize(inequalities.size() + numReservedCols);
  std::fill(inequalities.begin() + offset,
            inequalities.begin() + offset + getNumCols(), 0);
  for (unsigned i = 0, e = getNumCols(); i < e; i++) {
    inequalities[offset + i] = expr[i] - lb[i];
  }
}

void FlatAffineConstraints::addUpperBound(ArrayRef<int64_t> expr,
                                          ArrayRef<int64_t> ub) {
  assert(expr.size() == getNumCols());
  assert(ub.size() == getNumCols());
  unsigned offset = inequalities.size();
  inequalities.resize(inequalities.size() + numReservedCols);
  std::fill(inequalities.begin() + offset,
            inequalities.begin() + offset + getNumCols(), 0);
  for (unsigned i = 0, e = getNumCols(); i < e; i++) {
    inequalities[offset + i] = ub[i] - expr[i];
  }
}

bool FlatAffineConstraints::findId(const MLValue &operand, unsigned *pos) {
  unsigned i = 0;
  for (const auto &mayBeId : ids) {
    if (mayBeId.hasValue() && mayBeId.getValue() == &operand) {
      *pos = i;
      return true;
    }
    i++;
  }
  return false;
}

// TODO(andydavis, bondhugula) AFFINE REFACTOR: merge with loop bounds
// code in dependence analysis.
bool FlatAffineConstraints::addBoundsFromForStmt(unsigned pos,
                                                 ForStmt *forStmt) {
  // Adds a lower or upper bound when the bounds aren't constant.
  auto addLowerOrUpperBound = [&](bool lower) -> bool {
    const auto &operands = lower ? forStmt->getLowerBoundOperands()
                                 : forStmt->getUpperBoundOperands();
    SmallVector<unsigned, 8> positions;

    for (const auto &operand : operands) {
      unsigned loc;
      // TODO(andydavis, bondhugula) AFFINE REFACTOR: merge with loop bounds
      // code in dependence analysis.
      if (!findId(*operand, &loc)) {
        addDimId(getNumDimIds(), operand);
        loc = getNumDimIds() - 1;
      }
      positions.push_back(loc);
    }

    auto boundMap =
        lower ? forStmt->getLowerBoundMap() : forStmt->getUpperBoundMap();

    for (auto result : boundMap.getResults()) {
      SmallVector<int64_t, 4> flattenedExpr;
      SmallVector<int64_t, 4> ineq(getNumCols(), 0);
      // TODO(andydavis, bondhugula) AFFINE REFACTOR: merge with loop bounds in
      // dependence analysis.
      FlatAffineConstraints cst;
      if (!getFlattenedAffineExpr(result, boundMap.getNumDims(),
                                  boundMap.getNumSymbols(), &flattenedExpr,
                                  &cst)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "semi-affine expressions not yet supported\n");
        return false;
      }
      if (cst.getNumLocalIds() > 0) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "loop bounds with mod/floordiv expr's not yet supported\n");
        return false;
      }

      ineq[pos] = lower ? 1 : -1;
      for (unsigned j = 0, e = boundMap.getNumInputs(); j < e; j++) {
        ineq[positions[j]] = lower ? -flattenedExpr[j] : flattenedExpr[j];
      }
      // Constant term.
      ineq[getNumCols() - 1] =
          lower ? -flattenedExpr[flattenedExpr.size() - 1]
                // Upper bound in flattenedExpr is an exclusive one.
                : flattenedExpr[flattenedExpr.size() - 1] - 1;
      addInequality(ineq);
    }
    return true;
  };

  if (forStmt->hasConstantLowerBound()) {
    addConstantLowerBound(pos, forStmt->getConstantLowerBound());
  } else {
    // Non-constant lower bound case.
    if (!addLowerOrUpperBound(/*lower=*/true))
      return false;
  }

  if (forStmt->hasConstantUpperBound()) {
    addConstantUpperBound(pos, forStmt->getConstantUpperBound() - 1);
    return true;
  }
  // Non-constant upper bound case.
  return addLowerOrUpperBound(/*lower=*/false);
}

/// Sets the specified identifer to a constant value.
void FlatAffineConstraints::setIdToConstant(unsigned pos, int64_t val) {
  unsigned offset = equalities.size();
  equalities.resize(equalities.size() + numReservedCols);
  std::fill(equalities.begin() + offset,
            equalities.begin() + offset + getNumCols(), 0);
  equalities[offset + pos] = 1;
  equalities[offset + getNumCols() - 1] = -val;
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

bool FlatAffineConstraints::getDimensionBounds(unsigned pos, unsigned num,
                                               SmallVectorImpl<AffineMap> *lbs,
                                               SmallVectorImpl<AffineMap> *ubs,
                                               MLIRContext *context) {
  assert(pos + num < getNumCols());

  projectOut(0, pos);
  projectOut(pos + num, getNumIds() - num);

  lbs->resize(num, AffineMap::Null());
  ubs->resize(num, AffineMap::Null());

  for (int i = static_cast<int>(num) - 1; i >= 0; i--) {
    // Only constant dim bounds for now.
    auto lb = getConstantLowerBound(i);
    auto ub = getConstantUpperBound(i);
    // TODO(mlir-team): handle arbitrary bounds.
    if (!lb.hasValue() || !ub.hasValue())
      return false;
    (*lbs)[i] = AffineMap::getConstantMap(lb.getValue(), context);
    (*ubs)[i] = AffineMap::getConstantMap(ub.getValue(), context);
    projectOut(i, 1);
  }
  return true;
}

Optional<int64_t>
FlatAffineConstraints::getConstantLowerBound(unsigned pos) const {
  assert(pos < getNumCols() - 1);
  Optional<int64_t> lb = None;
  for (unsigned r = 0; r < getNumInequalities(); r++) {
    if (atIneq(r, pos) <= 0)
      // Not a lower bound.
      continue;
    unsigned c;
    for (c = 0; c < getNumCols() - 1; c++) {
      if (c != pos && atIneq(r, c) != 0)
        break;
    }
    // Not a constant lower bound.
    if (c < getNumCols() - 1)
      return None;
    auto mayLb = mlir::ceilDiv(-atIneq(r, getNumCols() - 1), atIneq(r, pos));
    if (!lb.hasValue() || mayLb > lb.getValue())
      lb = mayLb;
  }
  // TODO(andydavis,bondhugula): consider equalities (and an equality
  // contradicting an inequality, i.e, an empty set).
  return lb;
}

/// Returns the extent of the specified identifier (upper bound - lower bound)
/// if it found to be a constant; returns None if it's not a constant.
/// 'lbPosition' is set to the row position of the corresponding lower bound.
Optional<int64_t>
FlatAffineConstraints::getConstantBoundDifference(unsigned pos,
                                                  unsigned *lbPosition) const {
  // Check if the identifier appears at all in any of the inequalities.
  unsigned r, e;
  for (r = 0, e = getNumInequalities(); r < e; r++) {
    if (atIneq(r, pos) != 0)
      break;
  }
  if (r == e) {
    // If it doesn't appear, just remove the column and return.
    // TODO(andydavis,bondhugula): refactor removeColumns to use it from here.
    return None;
  }

  // Positions of constraints that are lower/upper bounds on the variable.
  SmallVector<unsigned, 4> lbIndices, ubIndices;

  // Gather all lower bounds and upper bounds of the variable. Since the
  // canonical form c_1*x_1 + c_2*x_2 + ... + c_0 >= 0, a constraint is a lower
  // bound for x_i if c_i >= 1, and an upper bound if c_i <= -1.
  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    if (atIneq(r, pos) >= 1)
      // Lower bound.
      lbIndices.push_back(r);
    else if (atIneq(r, pos) <= -1)
      // Upper bound.
      ubIndices.push_back(r);
  }

  // TODO(bondhugula): eliminate all variables that aren't part of any of the
  // lower/upper bounds - to make this more powerful.

  Optional<int64_t> minDiff = None;
  for (auto ubPos : ubIndices) {
    for (auto lbPos : lbIndices) {
      // Look for a lower bound and an upper bound that only differ by a
      // constant, i.e., pairs of the form  0 <= c_pos - f(c_i's) <= diffConst.
      // For example, if ii is the pos^th variable, we are looking for
      // constraints like ii >= i, ii <= ii + 50, 50 being the difference. The
      // minimum among all such constant differences is kept since that's the
      // constant bounding the extent of the pos^th variable.
      unsigned j;
      for (j = 0; j < getNumCols() - 1; j++)
        if (atIneq(ubPos, j) != -atIneq(lbPos, j)) {
          break;
        }
      if (j < getNumCols() - 1)
        continue;
      int64_t mayDiff =
          atIneq(ubPos, getNumCols() - 1) + atIneq(lbPos, getNumCols() - 1) + 1;
      if (minDiff == None || mayDiff < minDiff) {
        minDiff = mayDiff;
        *lbPosition = lbPos;
      }
    }
  }
  return minDiff;
}

Optional<int64_t>
FlatAffineConstraints::getConstantUpperBound(unsigned pos) const {
  assert(pos < getNumCols() - 1);
  Optional<int64_t> ub = None;
  for (unsigned r = 0; r < getNumInequalities(); r++) {
    // Not a upper bound.
    if (atIneq(r, pos) >= 0)
      continue;
    unsigned c;
    for (c = 0; c < getNumCols() - 1; c++) {
      if (c != pos && atIneq(r, c) != 0)
        break;
    }
    // Not a constant upper bound.
    if (c < getNumCols() - 1)
      return None;
    auto mayUb = mlir::floorDiv(atIneq(r, getNumCols() - 1), -atIneq(r, pos));
    if (!ub.hasValue() || mayUb < ub.getValue())
      ub = mayUb;
  }
  // TODO(andydavis,bondhugula): consider equalities (and an equality
  // contradicting an inequality, i.e, an empty set).
  return ub;
}

// A simple (naive and conservative) check for hyper-rectangularlity.
bool FlatAffineConstraints::isHyperRectangular(unsigned pos,
                                               unsigned num) const {
  assert(pos < getNumCols() - 1);
  // Check for two non-zero coefficients in the range [pos, pos + sum).
  for (unsigned r = 0; r < getNumInequalities(); r++) {
    unsigned sum = 0;
    for (unsigned c = pos; c < pos + num; c++) {
      if (atIneq(r, c) != 0)
        sum++;
    }
    if (sum > 1)
      return false;
  }
  for (unsigned r = 0; r < getNumEqualities(); r++) {
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
     << " symbols, " << getNumLocalIds() << " locals): \n";
  os << "(";
  for (unsigned i = 0, e = getNumIds(); i < e; i++) {
    if (ids[i] == None)
      os << "None ";
    else
      os << "MLValue ";
  }
  os << ")\n";
  for (unsigned i = 0, e = getNumEqualities(); i < e; ++i) {
    for (unsigned j = 0; j < getNumCols(); ++j) {
      os << atEq(i, j) << " ";
    }
    os << "= 0\n";
  }
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    for (unsigned j = 0; j < getNumCols(); ++j) {
      os << atIneq(i, j) << " ";
    }
    os << ">= 0\n";
  }
  os << '\n';
}

void FlatAffineConstraints::dump() const { print(llvm::errs()); }

void FlatAffineConstraints::removeDuplicates() {
  // TODO: remove redundant constraints.
}

void FlatAffineConstraints::clearAndCopyFrom(
    const FlatAffineConstraints &other) {
  FlatAffineConstraints copy(other);
  std::swap(*this, copy);
  assert(copy.getNumIds() == copy.getIds().size());
}

void FlatAffineConstraints::removeId(unsigned pos) {
  assert(pos < getNumIds());

  if (pos < numDims)
    numDims--;
  else if (pos < numDims + numSymbols)
    numSymbols--;
  numIds--;

  for (unsigned r = 0, e = getNumInequalities(); r < e; r++) {
    for (unsigned c = pos; c < getNumCols(); c++) {
      atIneq(r, c) = atIneq(r, c + 1);
    }
  }

  for (unsigned r = 0, e = getNumEqualities(); r < e; r++) {
    for (unsigned c = pos; c < getNumCols(); c++) {
      atEq(r, c) = atEq(r, c + 1);
    }
  }
  ids.erase(ids.begin() + pos);
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

  // A fast linear time tightening.
  GCDTightenInequalities();

  // Check if this identifier can be eliminated through a substitution.
  for (unsigned r = 0; r < getNumEqualities(); r++) {
    if (atEq(r, pos) != 0) {
      // Use Gaussian elimination here (since we have an equality).
      bool ret = gaussianEliminateId(pos);
      (void)ret;
      assert(ret && "Gaussian elimination guaranteed to succeed");
      LLVM_DEBUG(llvm::dbgs() << "FM output:\n");
      LLVM_DEBUG(dump());
      return;
    }
  }

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

  SmallVector<Optional<MLValue *>, 8> newIds;
  newIds.reserve(numIds - 1);
  newIds.insert(newIds.end(), ids.begin(), ids.begin() + pos);
  newIds.insert(newIds.end(), ids.begin() + pos + 1, ids.end());

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

  if (lcmProducts == 1 && isResultIntegerExact)
    *isResultIntegerExact = 1;

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

  newFac.removeDuplicates();
  clearAndCopyFrom(newFac);
  LLVM_DEBUG(llvm::dbgs() << "FM output:\n");
  LLVM_DEBUG(dump());
}

void FlatAffineConstraints::projectOut(unsigned pos, unsigned num) {
  // 'pos' can be at most getNumCols() - 2.
  if (num == 0)
    return;
  assert(pos <= getNumCols() - 2 && "invalid position");
  assert(pos + num < getNumCols() && "invalid range");
  for (unsigned i = 0; i < num; i++) {
    FourierMotzkinEliminate(pos);
  }
}

void FlatAffineConstraints::projectOut(MLValue *id) {
  unsigned pos;
  bool ret = findId(*id, &pos);
  assert(ret);
  (void)ret;
  FourierMotzkinEliminate(pos);
}
