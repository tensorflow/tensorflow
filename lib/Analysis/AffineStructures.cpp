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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLValue.h"
#include "mlir/Support/MathExtras.h"
#include "third_party/llvm/llvm/projects/google-mlir/include/mlir/Analysis/AffineStructures.h"
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

//===----------------------------------------------------------------------===//
// FlatAffineConstraints.
//===----------------------------------------------------------------------===//

// Copy constructor.
FlatAffineConstraints::FlatAffineConstraints(
    const FlatAffineConstraints &other) {
  numReservedEqualities = other.numReservedEqualities;
  numReservedInequalities = other.numReservedInequalities;
  numIds = other.getNumIds();
  numDims = other.getNumDimIds();
  numSymbols = other.getNumSymbolIds();

  equalities.reserve(numReservedEqualities * getNumCols());
  inequalities.reserve(numReservedEqualities * getNumCols());

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
    : numReservedEqualities(set.getNumEqualities()),
      numReservedInequalities(set.getNumInequalities()), numReservedIds(0),
      numIds(set.getNumDims() + set.getNumSymbols()), numDims(set.getNumDims()),
      numSymbols(set.getNumSymbols()) {
  equalities.reserve(set.getNumEqualities() * getNumCols());
  inequalities.reserve(set.getNumInequalities() * getNumCols());

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
static bool isEmptyByGCDTest(const FlatAffineConstraints &constraints) {
  unsigned numCols = constraints.getNumCols();
  for (unsigned i = 0, e = constraints.getNumEqualities(); i < e; ++i) {
    uint64_t gcd = std::abs(constraints.atEq(i, 0));
    for (unsigned j = 1; j < numCols - 1; ++j) {
      gcd =
          llvm::GreatestCommonDivisor64(gcd, std::abs(constraints.atEq(i, j)));
    }
    int64_t v = std::abs(constraints.atEq(i, numCols - 1));
    if (gcd > 0 && (v % gcd != 0)) {
      return true;
    }
  }
  return false;
}

// Checks all rows of equality/inequality constraints for contradictions
// (i.e. 1 == 0), which may have surfaced after elimination.
// Returns 'true' if a valid constraint is detected. Returns 'false' otherwise.
static bool hasInvalidConstraint(const FlatAffineConstraints &constraints) {
  auto check = [constraints](bool isEq) -> bool {
    unsigned numCols = constraints.getNumCols();
    unsigned numRows = isEq ? constraints.getNumEqualities()
                            : constraints.getNumInequalities();
    for (unsigned i = 0, e = numRows; i < e; ++i) {
      unsigned j;
      for (j = 0; j < numCols - 1; ++j) {
        int64_t v = isEq ? constraints.atEq(i, j) : constraints.atIneq(i, j);
        // Skip rows with non-zero variable coefficients.
        if (v != 0)
          break;
      }
      if (j < numCols - 1) {
        continue;
      }
      // Check validity of constant term at 'numCols - 1' w.r.t 'isEq'.
      // Example invalid constraints include: '1 == 0' or '-1 >= 0'
      int64_t v = isEq ? constraints.atEq(i, numCols - 1)
                       : constraints.atIneq(i, numCols - 1);
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
static void removeColumns(FlatAffineConstraints *constraints, unsigned colStart,
                          unsigned colLimit, bool isEq) {
  unsigned numCols = constraints->getNumCols();
  unsigned newNumCols = numCols - (colLimit - colStart);
  unsigned numRows = isEq ? constraints->getNumEqualities()
                          : constraints->getNumInequalities();
  for (unsigned i = 0, e = numRows; i < e; ++i) {
    for (unsigned j = 0; j < numCols; ++j) {
      if (j >= colStart && j < colLimit)
        continue;
      unsigned inputIndex = i * numCols + j;
      unsigned outputOffset = j >= colLimit ? j - (colLimit - colStart) : j;
      unsigned outputIndex = i * newNumCols + outputOffset;
      assert(outputIndex <= inputIndex);
      if (isEq) {
        constraints->atEqIdx(outputIndex) = constraints->atEqIdx(inputIndex);
      } else {
        constraints->atIneqIdx(outputIndex) =
            constraints->atIneqIdx(inputIndex);
      }
    }
  }
}

// Removes coefficients in column range [colStart, colLimit),and copies any
// remaining valid data into place, updates member variables, and resizes
// arrays as needed.
void FlatAffineConstraints::removeColumnRange(unsigned colStart,
                                              unsigned colLimit) {
  // TODO(andydavis) Make 'removeColumns' a lambda called from here.
  // Remove eliminated columns from equalities.
  removeColumns(this, colStart, colLimit, /*isEq=*/true);
  // Remove eliminated columns from inequalities.
  removeColumns(this, colStart, colLimit, /*isEq=*/false);
  // Update members numDims, numSymbols and numIds.
  unsigned numDimsEliminated = 0;
  if (colStart < numDims) {
    numDimsEliminated = std::min(numDims, colLimit) - colStart;
  }
  unsigned numEqualities = getNumEqualities();
  unsigned numInequalities = getNumInequalities();
  unsigned numColsEliminated = colLimit - colStart;
  unsigned numSymbolsEliminated =
      std::min(numSymbols, numColsEliminated - numDimsEliminated);
  numDims -= numDimsEliminated;
  numSymbols -= numSymbolsEliminated;
  numIds = numIds - numColsEliminated;
  equalities.resize(numEqualities * getNumCols());
  inequalities.resize(numInequalities * getNumCols());
}

// Performs variable elimination on all identifiers, runs the GCD test on
// all equality constraint rows, and checks the constraint validity.
// Returns 'true' if the GCD test fails on any row, or if any invalid
// constraint is detected. Returns 'false' otherwise.
bool FlatAffineConstraints::isEmpty() const {
  auto tmpCst = clone();
  if (tmpCst->gaussianEliminateIds(0, numIds) < numIds) {
    for (unsigned i = 0, e = tmpCst->getNumIds(); i < e; i++)
      if (!tmpCst->FourierMotzkinEliminate(0))
        return false;
  }
  if (isEmptyByGCDTest(*tmpCst))
    return true;
  if (hasInvalidConstraint(*tmpCst))
    return true;
  return false;
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
  if (posStart >= posLimit || posLimit > numIds)
    return 0;
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
  equalities.resize(equalities.size() + eq.size());
  for (unsigned i = 0, e = eq.size(); i < e; i++) {
    equalities[offset + i] = eq[i];
  }
}

void FlatAffineConstraints::removeEquality(unsigned pos) {
  unsigned numEqualities = getNumEqualities();
  assert(pos < numEqualities);
  unsigned numCols = getNumCols();
  unsigned outputIndex = pos * numCols;
  unsigned inputIndex = (pos + 1) * numCols;
  unsigned numElemsToCopy = (numEqualities - pos - 1) * numCols;
  for (unsigned i = 0; i < numElemsToCopy; ++i) {
    equalities[outputIndex + i] = equalities[inputIndex + i];
  }
  equalities.resize(equalities.size() - numCols);
}

void FlatAffineConstraints::addInequality(ArrayRef<int64_t> inEq) {
  assert(inEq.size() == getNumCols());
  unsigned offset = inequalities.size();
  inequalities.resize(inequalities.size() + inEq.size());
  for (unsigned i = 0, e = inEq.size(); i < e; i++) {
    inequalities[offset + i] = inEq[i];
  }
}

void FlatAffineConstraints::print(raw_ostream &os) const {
  os << "\nConstraints:\n";
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
}

void FlatAffineConstraints::removeId(unsigned pos) {
  assert(pos >= 0 && pos < getNumIds());

  for (unsigned r = 0; r < getNumInequalities(); r++) {
    for (unsigned c = pos; c < getNumCols() - 1; c++) {
      atIneq(r, c) = atIneq(r, c + 1);
    }
  }

  for (unsigned r = 0; r < getNumEqualities(); r++) {
    for (unsigned c = pos; c < getNumCols() - 1; c++) {
      atEq(r, c) = atEq(r, c + 1);
    }
  }

  if (pos < numDims)
    numDims--;
  else if (pos < numSymbols)
    numSymbols--;
  numIds--;
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
bool FlatAffineConstraints::FourierMotzkinEliminate(
    unsigned pos, bool darkShadow, bool *isResultIntegerExact) {
  assert(pos < getNumIds() && "invalid position");
  LLVM_DEBUG(llvm::dbgs() << "FM Input:\n");
  LLVM_DEBUG(dump());

  // Check if this identifier can be eliminated through a substitution.
  for (unsigned r = 0; r < getNumEqualities(); r++) {
    if (atIneq(r, pos) != 0) {
      // Use Gaussian elimination here (since we have an equality).
      bool ret = gaussianEliminateId(pos);
      assert(ret && "Gaussian elimination guaranteed to succeed");
      return ret;
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
    return true;
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

  /// Create the new system which has one identifier less.
  FlatAffineConstraints newFac(
      lbIndices.size() * ubIndices.size() + nbIndices.size(),
      getNumEqualities(), getNumCols() - 1, newNumDims, newNumSymbols,
      /*numLocals=*/getNumIds() - 1 - newNumDims - newNumSymbols);

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
  return true;
}
