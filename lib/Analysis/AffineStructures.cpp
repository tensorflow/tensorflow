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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLValue.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

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

MutableAffineMap::MutableAffineMap(AffineMap map)
    : numDims(map.getNumDims()), numSymbols(map.getNumSymbols()),
      // A map always has at leat 1 result by construction
      context(map.getResult(0).getContext()) {
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

AffineMap MutableAffineMap::getAffineMap() {
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
// 'valuesToSearch'. Returns false otherwise.
static bool findIndex(MLValue *valueToMatch, ArrayRef<MLValue *> valuesToSearch,
                      unsigned *indexOfMatch) {
  unsigned size = valuesToSearch.size();
  for (unsigned i = 0; i < size; ++i) {
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
    if (findIndex(inputOperand, outputOperands, &outputIndex)) {
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
    if (findIndex(inputOperand, outputOperands, &outputIndex)) {
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
  findIndex(value, operands, &index);
  auto expr = const_cast<AffineValueMap *>(this)->getAffineMap().getResult(idx);
  // TODO(ntv): this is better implemented on a flattened representation.
  // At least for now it is conservative.
  return expr.isFunctionOfDim(index);
}

unsigned AffineValueMap::getNumOperands() const { return operands.size(); }

SSAValue *AffineValueMap::getOperand(unsigned i) const {
  return static_cast<SSAValue *>(operands[i]);
}

ArrayRef<MLValue *> AffineValueMap::getOperands() const {
  return ArrayRef<MLValue *>(operands);
}

AffineMap AffineValueMap::getAffineMap() { return map.getAffineMap(); }

AffineValueMap::~AffineValueMap() {}

void FlatAffineConstraints::addEquality(ArrayRef<int64_t> eq) {
  assert(eq.size() == getNumCols());
  unsigned offset = equalities.size();
  equalities.resize(equalities.size() + eq.size());
  for (unsigned i = 0, e = eq.size(); i < e; i++) {
    equalities[offset + i] = eq[i];
  }
}
