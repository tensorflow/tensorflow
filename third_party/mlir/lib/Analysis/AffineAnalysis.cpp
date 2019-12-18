//===- AffineAnalysis.cpp - Affine structures analysis routines -----------===//
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
// This file implements miscellaneous analysis routines for affine structures
// (expressions, maps, sets), and other utilities relying on such analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "affine-analysis"

using namespace mlir;

using llvm::dbgs;

/// Returns the sequence of AffineApplyOp Operations operation in
/// 'affineApplyOps', which are reachable via a search starting from 'operands',
/// and ending at operands which are not defined by AffineApplyOps.
// TODO(andydavis) Add a method to AffineApplyOp which forward substitutes
// the AffineApplyOp into any user AffineApplyOps.
void mlir::getReachableAffineApplyOps(
    ArrayRef<Value *> operands, SmallVectorImpl<Operation *> &affineApplyOps) {
  struct State {
    // The ssa value for this node in the DFS traversal.
    Value *value;
    // The operand index of 'value' to explore next during DFS traversal.
    unsigned operandIndex;
  };
  SmallVector<State, 4> worklist;
  for (auto *operand : operands) {
    worklist.push_back({operand, 0});
  }

  while (!worklist.empty()) {
    State &state = worklist.back();
    auto *opInst = state.value->getDefiningOp();
    // Note: getDefiningOp will return nullptr if the operand is not an
    // Operation (i.e. block argument), which is a terminator for the search.
    if (!isa_and_nonnull<AffineApplyOp>(opInst)) {
      worklist.pop_back();
      continue;
    }

    if (state.operandIndex == 0) {
      // Pre-Visit: Add 'opInst' to reachable sequence.
      affineApplyOps.push_back(opInst);
    }
    if (state.operandIndex < opInst->getNumOperands()) {
      // Visit: Add next 'affineApplyOp' operand to worklist.
      // Get next operand to visit at 'operandIndex'.
      auto *nextOperand = opInst->getOperand(state.operandIndex);
      // Increment 'operandIndex' in 'state'.
      ++state.operandIndex;
      // Add 'nextOperand' to worklist.
      worklist.push_back({nextOperand, 0});
    } else {
      // Post-visit: done visiting operands AffineApplyOp, pop off stack.
      worklist.pop_back();
    }
  }
}

// Builds a system of constraints with dimensional identifiers corresponding to
// the loop IVs of the forOps appearing in that order. Any symbols founds in
// the bound operands are added as symbols in the system. Returns failure for
// the yet unimplemented cases.
// TODO(andydavis,bondhugula) Handle non-unit steps through local variables or
// stride information in FlatAffineConstraints. (For eg., by using iv - lb %
// step = 0 and/or by introducing a method in FlatAffineConstraints
// setExprStride(ArrayRef<int64_t> expr, int64_t stride)
LogicalResult mlir::getIndexSet(MutableArrayRef<AffineForOp> forOps,
                                FlatAffineConstraints *domain) {
  SmallVector<Value *, 4> indices;
  extractForInductionVars(forOps, &indices);
  // Reset while associated Values in 'indices' to the domain.
  domain->reset(forOps.size(), /*numSymbols=*/0, /*numLocals=*/0, indices);
  for (auto forOp : forOps) {
    // Add constraints from forOp's bounds.
    if (failed(domain->addAffineForOpDomain(forOp)))
      return failure();
  }
  return success();
}

// Computes the iteration domain for 'opInst' and populates 'indexSet', which
// encapsulates the constraints involving loops surrounding 'opInst' and
// potentially involving any Function symbols. The dimensional identifiers in
// 'indexSet' correspond to the loops surrounding 'op' from outermost to
// innermost.
// TODO(andydavis) Add support to handle IfInsts surrounding 'op'.
static LogicalResult getInstIndexSet(Operation *op,
                                     FlatAffineConstraints *indexSet) {
  // TODO(andydavis) Extend this to gather enclosing IfInsts and consider
  // factoring it out into a utility function.
  SmallVector<AffineForOp, 4> loops;
  getLoopIVs(*op, &loops);
  return getIndexSet(loops, indexSet);
}

// ValuePositionMap manages the mapping from Values which represent dimension
// and symbol identifiers from 'src' and 'dst' access functions to positions
// in new space where some Values are kept separate (using addSrc/DstValue)
// and some Values are merged (addSymbolValue).
// Position lookups return the absolute position in the new space which
// has the following format:
//
//   [src-dim-identifiers] [dst-dim-identifiers] [symbol-identifiers]
//
// Note: access function non-IV dimension identifiers (that have 'dimension'
// positions in the access function position space) are assigned as symbols
// in the output position space. Convenience access functions which lookup
// an Value in multiple maps are provided (i.e. getSrcDimOrSymPos) to handle
// the common case of resolving positions for all access function operands.
//
// TODO(andydavis) Generalize this: could take a template parameter for
// the number of maps (3 in the current case), and lookups could take indices
// of maps to check. So getSrcDimOrSymPos would be "getPos(value, {0, 2})".
class ValuePositionMap {
public:
  void addSrcValue(Value *value) {
    if (addValueAt(value, &srcDimPosMap, numSrcDims))
      ++numSrcDims;
  }
  void addDstValue(Value *value) {
    if (addValueAt(value, &dstDimPosMap, numDstDims))
      ++numDstDims;
  }
  void addSymbolValue(Value *value) {
    if (addValueAt(value, &symbolPosMap, numSymbols))
      ++numSymbols;
  }
  unsigned getSrcDimOrSymPos(Value *value) const {
    return getDimOrSymPos(value, srcDimPosMap, 0);
  }
  unsigned getDstDimOrSymPos(Value *value) const {
    return getDimOrSymPos(value, dstDimPosMap, numSrcDims);
  }
  unsigned getSymPos(Value *value) const {
    auto it = symbolPosMap.find(value);
    assert(it != symbolPosMap.end());
    return numSrcDims + numDstDims + it->second;
  }

  unsigned getNumSrcDims() const { return numSrcDims; }
  unsigned getNumDstDims() const { return numDstDims; }
  unsigned getNumDims() const { return numSrcDims + numDstDims; }
  unsigned getNumSymbols() const { return numSymbols; }

private:
  bool addValueAt(Value *value, DenseMap<Value *, unsigned> *posMap,
                  unsigned position) {
    auto it = posMap->find(value);
    if (it == posMap->end()) {
      (*posMap)[value] = position;
      return true;
    }
    return false;
  }
  unsigned getDimOrSymPos(Value *value,
                          const DenseMap<Value *, unsigned> &dimPosMap,
                          unsigned dimPosOffset) const {
    auto it = dimPosMap.find(value);
    if (it != dimPosMap.end()) {
      return dimPosOffset + it->second;
    }
    it = symbolPosMap.find(value);
    assert(it != symbolPosMap.end());
    return numSrcDims + numDstDims + it->second;
  }

  unsigned numSrcDims = 0;
  unsigned numDstDims = 0;
  unsigned numSymbols = 0;
  DenseMap<Value *, unsigned> srcDimPosMap;
  DenseMap<Value *, unsigned> dstDimPosMap;
  DenseMap<Value *, unsigned> symbolPosMap;
};

// Builds a map from Value to identifier position in a new merged identifier
// list, which is the result of merging dim/symbol lists from src/dst
// iteration domains, the format of which is as follows:
//
//   [src-dim-identifiers, dst-dim-identifiers, symbol-identifiers, const_term]
//
// This method populates 'valuePosMap' with mappings from operand Values in
// 'srcAccessMap'/'dstAccessMap' (as well as those in 'srcDomain'/'dstDomain')
// to the position of these values in the merged list.
static void buildDimAndSymbolPositionMaps(
    const FlatAffineConstraints &srcDomain,
    const FlatAffineConstraints &dstDomain, const AffineValueMap &srcAccessMap,
    const AffineValueMap &dstAccessMap, ValuePositionMap *valuePosMap,
    FlatAffineConstraints *dependenceConstraints) {
  auto updateValuePosMap = [&](ArrayRef<Value *> values, bool isSrc) {
    for (unsigned i = 0, e = values.size(); i < e; ++i) {
      auto *value = values[i];
      if (!isForInductionVar(values[i])) {
        assert(isValidSymbol(values[i]) &&
               "access operand has to be either a loop IV or a symbol");
        valuePosMap->addSymbolValue(value);
      } else if (isSrc) {
        valuePosMap->addSrcValue(value);
      } else {
        valuePosMap->addDstValue(value);
      }
    }
  };

  SmallVector<Value *, 4> srcValues, destValues;
  srcDomain.getIdValues(0, srcDomain.getNumDimAndSymbolIds(), &srcValues);
  dstDomain.getIdValues(0, dstDomain.getNumDimAndSymbolIds(), &destValues);
  // Update value position map with identifiers from src iteration domain.
  updateValuePosMap(srcValues, /*isSrc=*/true);
  // Update value position map with identifiers from dst iteration domain.
  updateValuePosMap(destValues, /*isSrc=*/false);
  // Update value position map with identifiers from src access function.
  updateValuePosMap(srcAccessMap.getOperands(), /*isSrc=*/true);
  // Update value position map with identifiers from dst access function.
  updateValuePosMap(dstAccessMap.getOperands(), /*isSrc=*/false);
}

// Sets up dependence constraints columns appropriately, in the format:
// [src-dim-ids, dst-dim-ids, symbol-ids, local-ids, const_term]
void initDependenceConstraints(const FlatAffineConstraints &srcDomain,
                               const FlatAffineConstraints &dstDomain,
                               const AffineValueMap &srcAccessMap,
                               const AffineValueMap &dstAccessMap,
                               const ValuePositionMap &valuePosMap,
                               FlatAffineConstraints *dependenceConstraints) {
  // Calculate number of equalities/inequalities and columns required to
  // initialize FlatAffineConstraints for 'dependenceDomain'.
  unsigned numIneq =
      srcDomain.getNumInequalities() + dstDomain.getNumInequalities();
  AffineMap srcMap = srcAccessMap.getAffineMap();
  assert(srcMap.getNumResults() == dstAccessMap.getAffineMap().getNumResults());
  unsigned numEq = srcMap.getNumResults();
  unsigned numDims = srcDomain.getNumDimIds() + dstDomain.getNumDimIds();
  unsigned numSymbols = valuePosMap.getNumSymbols();
  unsigned numLocals = srcDomain.getNumLocalIds() + dstDomain.getNumLocalIds();
  unsigned numIds = numDims + numSymbols + numLocals;
  unsigned numCols = numIds + 1;

  // Set flat affine constraints sizes and reserving space for constraints.
  dependenceConstraints->reset(numIneq, numEq, numCols, numDims, numSymbols,
                               numLocals);

  // Set values corresponding to dependence constraint identifiers.
  SmallVector<Value *, 4> srcLoopIVs, dstLoopIVs;
  srcDomain.getIdValues(0, srcDomain.getNumDimIds(), &srcLoopIVs);
  dstDomain.getIdValues(0, dstDomain.getNumDimIds(), &dstLoopIVs);

  dependenceConstraints->setIdValues(0, srcLoopIVs.size(), srcLoopIVs);
  dependenceConstraints->setIdValues(
      srcLoopIVs.size(), srcLoopIVs.size() + dstLoopIVs.size(), dstLoopIVs);

  // Set values for the symbolic identifier dimensions.
  auto setSymbolIds = [&](ArrayRef<Value *> values) {
    for (auto *value : values) {
      if (!isForInductionVar(value)) {
        assert(isValidSymbol(value) && "expected symbol");
        dependenceConstraints->setIdValue(valuePosMap.getSymPos(value), value);
      }
    }
  };

  setSymbolIds(srcAccessMap.getOperands());
  setSymbolIds(dstAccessMap.getOperands());

  SmallVector<Value *, 8> srcSymbolValues, dstSymbolValues;
  srcDomain.getIdValues(srcDomain.getNumDimIds(),
                        srcDomain.getNumDimAndSymbolIds(), &srcSymbolValues);
  dstDomain.getIdValues(dstDomain.getNumDimIds(),
                        dstDomain.getNumDimAndSymbolIds(), &dstSymbolValues);
  setSymbolIds(srcSymbolValues);
  setSymbolIds(dstSymbolValues);

  for (unsigned i = 0, e = dependenceConstraints->getNumDimAndSymbolIds();
       i < e; i++)
    assert(dependenceConstraints->getIds()[i].hasValue());
}

// Adds iteration domain constraints from 'srcDomain' and 'dstDomain' into
// 'dependenceDomain'.
// Uses 'valuePosMap' to determine the position in 'dependenceDomain' to which a
// srcDomain/dstDomain Value maps.
static void addDomainConstraints(const FlatAffineConstraints &srcDomain,
                                 const FlatAffineConstraints &dstDomain,
                                 const ValuePositionMap &valuePosMap,
                                 FlatAffineConstraints *dependenceDomain) {
  unsigned depNumDimsAndSymbolIds = dependenceDomain->getNumDimAndSymbolIds();

  SmallVector<int64_t, 4> cst(dependenceDomain->getNumCols());

  auto addDomain = [&](bool isSrc, bool isEq, unsigned localOffset) {
    const FlatAffineConstraints &domain = isSrc ? srcDomain : dstDomain;
    unsigned numCsts =
        isEq ? domain.getNumEqualities() : domain.getNumInequalities();
    unsigned numDimAndSymbolIds = domain.getNumDimAndSymbolIds();
    auto at = [&](unsigned i, unsigned j) -> int64_t {
      return isEq ? domain.atEq(i, j) : domain.atIneq(i, j);
    };
    auto map = [&](unsigned i) -> int64_t {
      return isSrc ? valuePosMap.getSrcDimOrSymPos(domain.getIdValue(i))
                   : valuePosMap.getDstDimOrSymPos(domain.getIdValue(i));
    };

    for (unsigned i = 0; i < numCsts; ++i) {
      // Zero fill.
      std::fill(cst.begin(), cst.end(), 0);
      // Set coefficients for identifiers corresponding to domain.
      for (unsigned j = 0; j < numDimAndSymbolIds; ++j)
        cst[map(j)] = at(i, j);
      // Local terms.
      for (unsigned j = 0, e = domain.getNumLocalIds(); j < e; j++)
        cst[depNumDimsAndSymbolIds + localOffset + j] =
            at(i, numDimAndSymbolIds + j);
      // Set constant term.
      cst[cst.size() - 1] = at(i, domain.getNumCols() - 1);
      // Add constraint.
      if (isEq)
        dependenceDomain->addEquality(cst);
      else
        dependenceDomain->addInequality(cst);
    }
  };

  // Add equalities from src domain.
  addDomain(/*isSrc=*/true, /*isEq=*/true, /*localOffset=*/0);
  // Add inequalities from src domain.
  addDomain(/*isSrc=*/true, /*isEq=*/false, /*localOffset=*/0);
  // Add equalities from dst domain.
  addDomain(/*isSrc=*/false, /*isEq=*/true,
            /*localOffset=*/srcDomain.getNumLocalIds());
  // Add inequalities from dst domain.
  addDomain(/*isSrc=*/false, /*isEq=*/false,
            /*localOffset=*/srcDomain.getNumLocalIds());
}

// Adds equality constraints that equate src and dst access functions
// represented by 'srcAccessMap' and 'dstAccessMap' for each result.
// Requires that 'srcAccessMap' and 'dstAccessMap' have the same results count.
// For example, given the following two accesses functions to a 2D memref:
//
//   Source access function:
//     (a0 * d0 + a1 * s0 + a2, b0 * d0 + b1 * s0 + b2)
//
//   Destination access function:
//     (c0 * d0 + c1 * s0 + c2, f0 * d0 + f1 * s0 + f2)
//
// This method constructs the following equality constraints in
// 'dependenceDomain', by equating the access functions for each result
// (i.e. each memref dim). Notice that 'd0' for the destination access function
// is mapped into 'd0' in the equality constraint:
//
//   d0      d1      s0         c
//   --      --      --         --
//   a0     -c0      (a1 - c1)  (a1 - c2) = 0
//   b0     -f0      (b1 - f1)  (b1 - f2) = 0
//
// Returns failure if any AffineExpr cannot be flattened (due to it being
// semi-affine). Returns success otherwise.
static LogicalResult
addMemRefAccessConstraints(const AffineValueMap &srcAccessMap,
                           const AffineValueMap &dstAccessMap,
                           const ValuePositionMap &valuePosMap,
                           FlatAffineConstraints *dependenceDomain) {
  AffineMap srcMap = srcAccessMap.getAffineMap();
  AffineMap dstMap = dstAccessMap.getAffineMap();
  assert(srcMap.getNumResults() == dstMap.getNumResults());
  unsigned numResults = srcMap.getNumResults();

  unsigned srcNumIds = srcMap.getNumDims() + srcMap.getNumSymbols();
  ArrayRef<Value *> srcOperands = srcAccessMap.getOperands();

  unsigned dstNumIds = dstMap.getNumDims() + dstMap.getNumSymbols();
  ArrayRef<Value *> dstOperands = dstAccessMap.getOperands();

  std::vector<SmallVector<int64_t, 8>> srcFlatExprs;
  std::vector<SmallVector<int64_t, 8>> destFlatExprs;
  FlatAffineConstraints srcLocalVarCst, destLocalVarCst;
  // Get flattened expressions for the source destination maps.
  if (failed(getFlattenedAffineExprs(srcMap, &srcFlatExprs, &srcLocalVarCst)) ||
      failed(getFlattenedAffineExprs(dstMap, &destFlatExprs, &destLocalVarCst)))
    return failure();

  unsigned domNumLocalIds = dependenceDomain->getNumLocalIds();
  unsigned srcNumLocalIds = srcLocalVarCst.getNumLocalIds();
  unsigned dstNumLocalIds = destLocalVarCst.getNumLocalIds();
  unsigned numLocalIdsToAdd = srcNumLocalIds + dstNumLocalIds;
  for (unsigned i = 0; i < numLocalIdsToAdd; i++) {
    dependenceDomain->addLocalId(dependenceDomain->getNumLocalIds());
  }

  unsigned numDims = dependenceDomain->getNumDimIds();
  unsigned numSymbols = dependenceDomain->getNumSymbolIds();
  unsigned numSrcLocalIds = srcLocalVarCst.getNumLocalIds();
  unsigned newLocalIdOffset = numDims + numSymbols + domNumLocalIds;

  // Equality to add.
  SmallVector<int64_t, 8> eq(dependenceDomain->getNumCols());
  for (unsigned i = 0; i < numResults; ++i) {
    // Zero fill.
    std::fill(eq.begin(), eq.end(), 0);

    // Flattened AffineExpr for src result 'i'.
    const auto &srcFlatExpr = srcFlatExprs[i];
    // Set identifier coefficients from src access function.
    for (unsigned j = 0, e = srcOperands.size(); j < e; ++j)
      eq[valuePosMap.getSrcDimOrSymPos(srcOperands[j])] = srcFlatExpr[j];
    // Local terms.
    for (unsigned j = 0, e = srcNumLocalIds; j < e; j++)
      eq[newLocalIdOffset + j] = srcFlatExpr[srcNumIds + j];
    // Set constant term.
    eq[eq.size() - 1] = srcFlatExpr[srcFlatExpr.size() - 1];

    // Flattened AffineExpr for dest result 'i'.
    const auto &destFlatExpr = destFlatExprs[i];
    // Set identifier coefficients from dst access function.
    for (unsigned j = 0, e = dstOperands.size(); j < e; ++j)
      eq[valuePosMap.getDstDimOrSymPos(dstOperands[j])] -= destFlatExpr[j];
    // Local terms.
    for (unsigned j = 0, e = dstNumLocalIds; j < e; j++)
      eq[newLocalIdOffset + numSrcLocalIds + j] = -destFlatExpr[dstNumIds + j];
    // Set constant term.
    eq[eq.size() - 1] -= destFlatExpr[destFlatExpr.size() - 1];

    // Add equality constraint.
    dependenceDomain->addEquality(eq);
  }

  // Add equality constraints for any operands that are defined by constant ops.
  auto addEqForConstOperands = [&](ArrayRef<Value *> operands) {
    for (unsigned i = 0, e = operands.size(); i < e; ++i) {
      if (isForInductionVar(operands[i]))
        continue;
      auto *symbol = operands[i];
      assert(isValidSymbol(symbol));
      // Check if the symbol is a constant.
      if (auto cOp = dyn_cast_or_null<ConstantIndexOp>(symbol->getDefiningOp()))
        dependenceDomain->setIdToConstant(valuePosMap.getSymPos(symbol),
                                          cOp.getValue());
    }
  };

  // Add equality constraints for any src symbols defined by constant ops.
  addEqForConstOperands(srcOperands);
  // Add equality constraints for any dst symbols defined by constant ops.
  addEqForConstOperands(dstOperands);

  // By construction (see flattener), local var constraints will not have any
  // equalities.
  assert(srcLocalVarCst.getNumEqualities() == 0 &&
         destLocalVarCst.getNumEqualities() == 0);
  // Add inequalities from srcLocalVarCst and destLocalVarCst into the
  // dependence domain.
  SmallVector<int64_t, 8> ineq(dependenceDomain->getNumCols());
  for (unsigned r = 0, e = srcLocalVarCst.getNumInequalities(); r < e; r++) {
    std::fill(ineq.begin(), ineq.end(), 0);

    // Set identifier coefficients from src local var constraints.
    for (unsigned j = 0, e = srcOperands.size(); j < e; ++j)
      ineq[valuePosMap.getSrcDimOrSymPos(srcOperands[j])] =
          srcLocalVarCst.atIneq(r, j);
    // Local terms.
    for (unsigned j = 0, e = srcNumLocalIds; j < e; j++)
      ineq[newLocalIdOffset + j] = srcLocalVarCst.atIneq(r, srcNumIds + j);
    // Set constant term.
    ineq[ineq.size() - 1] =
        srcLocalVarCst.atIneq(r, srcLocalVarCst.getNumCols() - 1);
    dependenceDomain->addInequality(ineq);
  }

  for (unsigned r = 0, e = destLocalVarCst.getNumInequalities(); r < e; r++) {
    std::fill(ineq.begin(), ineq.end(), 0);
    // Set identifier coefficients from dest local var constraints.
    for (unsigned j = 0, e = dstOperands.size(); j < e; ++j)
      ineq[valuePosMap.getDstDimOrSymPos(dstOperands[j])] =
          destLocalVarCst.atIneq(r, j);
    // Local terms.
    for (unsigned j = 0, e = dstNumLocalIds; j < e; j++)
      ineq[newLocalIdOffset + numSrcLocalIds + j] =
          destLocalVarCst.atIneq(r, dstNumIds + j);
    // Set constant term.
    ineq[ineq.size() - 1] =
        destLocalVarCst.atIneq(r, destLocalVarCst.getNumCols() - 1);

    dependenceDomain->addInequality(ineq);
  }
  return success();
}

// Returns the number of outer loop common to 'src/dstDomain'.
// Loops common to 'src/dst' domains are added to 'commonLoops' if non-null.
static unsigned
getNumCommonLoops(const FlatAffineConstraints &srcDomain,
                  const FlatAffineConstraints &dstDomain,
                  SmallVectorImpl<AffineForOp> *commonLoops = nullptr) {
  // Find the number of common loops shared by src and dst accesses.
  unsigned minNumLoops =
      std::min(srcDomain.getNumDimIds(), dstDomain.getNumDimIds());
  unsigned numCommonLoops = 0;
  for (unsigned i = 0; i < minNumLoops; ++i) {
    if (!isForInductionVar(srcDomain.getIdValue(i)) ||
        !isForInductionVar(dstDomain.getIdValue(i)) ||
        srcDomain.getIdValue(i) != dstDomain.getIdValue(i))
      break;
    if (commonLoops != nullptr)
      commonLoops->push_back(getForInductionVarOwner(srcDomain.getIdValue(i)));
    ++numCommonLoops;
  }
  if (commonLoops != nullptr)
    assert(commonLoops->size() == numCommonLoops);
  return numCommonLoops;
}

// Returns Block common to 'srcAccess.opInst' and 'dstAccess.opInst'.
static Block *getCommonBlock(const MemRefAccess &srcAccess,
                             const MemRefAccess &dstAccess,
                             const FlatAffineConstraints &srcDomain,
                             unsigned numCommonLoops) {
  if (numCommonLoops == 0) {
    auto *block = srcAccess.opInst->getBlock();
    while (!llvm::isa<FuncOp>(block->getParentOp())) {
      block = block->getParentOp()->getBlock();
    }
    return block;
  }
  auto *commonForValue = srcDomain.getIdValue(numCommonLoops - 1);
  auto forOp = getForInductionVarOwner(commonForValue);
  assert(forOp && "commonForValue was not an induction variable");
  return forOp.getBody();
}

// Returns true if the ancestor operation of 'srcAccess' appears before the
// ancestor operation of 'dstAccess' in the common ancestral block. Returns
// false otherwise.
// Note that because 'srcAccess' or 'dstAccess' may be nested in conditionals,
// the function is named 'srcAppearsBeforeDstInCommonBlock'. Note that
// 'numCommonLoops' is the number of contiguous surrounding outer loops.
static bool srcAppearsBeforeDstInAncestralBlock(
    const MemRefAccess &srcAccess, const MemRefAccess &dstAccess,
    const FlatAffineConstraints &srcDomain, unsigned numCommonLoops) {
  // Get Block common to 'srcAccess.opInst' and 'dstAccess.opInst'.
  auto *commonBlock =
      getCommonBlock(srcAccess, dstAccess, srcDomain, numCommonLoops);
  // Check the dominance relationship between the respective ancestors of the
  // src and dst in the Block of the innermost among the common loops.
  auto *srcInst = commonBlock->findAncestorOpInBlock(*srcAccess.opInst);
  assert(srcInst != nullptr);
  auto *dstInst = commonBlock->findAncestorOpInBlock(*dstAccess.opInst);
  assert(dstInst != nullptr);

  // Determine whether dstInst comes after srcInst.
  return srcInst->isBeforeInBlock(dstInst);
}

// Adds ordering constraints to 'dependenceDomain' based on number of loops
// common to 'src/dstDomain' and requested 'loopDepth'.
// Note that 'loopDepth' cannot exceed the number of common loops plus one.
// EX: Given a loop nest of depth 2 with IVs 'i' and 'j':
// *) If 'loopDepth == 1' then one constraint is added: i' >= i + 1
// *) If 'loopDepth == 2' then two constraints are added: i == i' and j' > j + 1
// *) If 'loopDepth == 3' then two constraints are added: i == i' and j == j'
static void addOrderingConstraints(const FlatAffineConstraints &srcDomain,
                                   const FlatAffineConstraints &dstDomain,
                                   unsigned loopDepth,
                                   FlatAffineConstraints *dependenceDomain) {
  unsigned numCols = dependenceDomain->getNumCols();
  SmallVector<int64_t, 4> eq(numCols);
  unsigned numSrcDims = srcDomain.getNumDimIds();
  unsigned numCommonLoops = getNumCommonLoops(srcDomain, dstDomain);
  unsigned numCommonLoopConstraints = std::min(numCommonLoops, loopDepth);
  for (unsigned i = 0; i < numCommonLoopConstraints; ++i) {
    std::fill(eq.begin(), eq.end(), 0);
    eq[i] = -1;
    eq[i + numSrcDims] = 1;
    if (i == loopDepth - 1) {
      eq[numCols - 1] = -1;
      dependenceDomain->addInequality(eq);
    } else {
      dependenceDomain->addEquality(eq);
    }
  }
}

// Computes distance and direction vectors in 'dependences', by adding
// variables to 'dependenceDomain' which represent the difference of the IVs,
// eliminating all other variables, and reading off distance vectors from
// equality constraints (if possible), and direction vectors from inequalities.
static void computeDirectionVector(
    const FlatAffineConstraints &srcDomain,
    const FlatAffineConstraints &dstDomain, unsigned loopDepth,
    FlatAffineConstraints *dependenceDomain,
    SmallVector<DependenceComponent, 2> *dependenceComponents) {
  // Find the number of common loops shared by src and dst accesses.
  SmallVector<AffineForOp, 4> commonLoops;
  unsigned numCommonLoops =
      getNumCommonLoops(srcDomain, dstDomain, &commonLoops);
  if (numCommonLoops == 0)
    return;
  // Compute direction vectors for requested loop depth.
  unsigned numIdsToEliminate = dependenceDomain->getNumIds();
  // Add new variables to 'dependenceDomain' to represent the direction
  // constraints for each shared loop.
  for (unsigned j = 0; j < numCommonLoops; ++j) {
    dependenceDomain->addDimId(j);
  }

  // Add equality constraints for each common loop, setting newly introduced
  // variable at column 'j' to the 'dst' IV minus the 'src IV.
  SmallVector<int64_t, 4> eq;
  eq.resize(dependenceDomain->getNumCols());
  unsigned numSrcDims = srcDomain.getNumDimIds();
  // Constraint variables format:
  // [num-common-loops][num-src-dim-ids][num-dst-dim-ids][num-symbols][constant]
  for (unsigned j = 0; j < numCommonLoops; ++j) {
    std::fill(eq.begin(), eq.end(), 0);
    eq[j] = 1;
    eq[j + numCommonLoops] = 1;
    eq[j + numCommonLoops + numSrcDims] = -1;
    dependenceDomain->addEquality(eq);
  }

  // Eliminate all variables other than the direction variables just added.
  dependenceDomain->projectOut(numCommonLoops, numIdsToEliminate);

  // Scan each common loop variable column and set direction vectors based
  // on eliminated constraint system.
  dependenceComponents->resize(numCommonLoops);
  for (unsigned j = 0; j < numCommonLoops; ++j) {
    (*dependenceComponents)[j].op = commonLoops[j].getOperation();
    auto lbConst = dependenceDomain->getConstantLowerBound(j);
    (*dependenceComponents)[j].lb =
        lbConst.getValueOr(std::numeric_limits<int64_t>::min());
    auto ubConst = dependenceDomain->getConstantUpperBound(j);
    (*dependenceComponents)[j].ub =
        ubConst.getValueOr(std::numeric_limits<int64_t>::max());
  }
}

// Populates 'accessMap' with composition of AffineApplyOps reachable from
// indices of MemRefAccess.
void MemRefAccess::getAccessMap(AffineValueMap *accessMap) const {
  // Get affine map from AffineLoad/Store.
  AffineMap map;
  if (auto loadOp = dyn_cast<AffineLoadOp>(opInst))
    map = loadOp.getAffineMap();
  else if (auto storeOp = dyn_cast<AffineStoreOp>(opInst))
    map = storeOp.getAffineMap();
  SmallVector<Value *, 8> operands(indices.begin(), indices.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);
  accessMap->reset(map, operands);
}

// Builds a flat affine constraint system to check if there exists a dependence
// between memref accesses 'srcAccess' and 'dstAccess'.
// Returns 'NoDependence' if the accesses can be definitively shown not to
// access the same element.
// Returns 'HasDependence' if the accesses do access the same element.
// Returns 'Failure' if an error or unsupported case was encountered.
// If a dependence exists, returns in 'dependenceComponents' a direction
// vector for the dependence, with a component for each loop IV in loops
// common to both accesses (see Dependence in AffineAnalysis.h for details).
//
// The memref access dependence check is comprised of the following steps:
// *) Compute access functions for each access. Access functions are computed
//    using AffineValueMaps initialized with the indices from an access, then
//    composed with AffineApplyOps reachable from operands of that access,
//    until operands of the AffineValueMap are loop IVs or symbols.
// *) Build iteration domain constraints for each access. Iteration domain
//    constraints are pairs of inequality constraints representing the
//    upper/lower loop bounds for each AffineForOp in the loop nest associated
//    with each access.
// *) Build dimension and symbol position maps for each access, which map
//    Values from access functions and iteration domains to their position
//    in the merged constraint system built by this method.
//
// This method builds a constraint system with the following column format:
//
//  [src-dim-identifiers, dst-dim-identifiers, symbols, constant]
//
// For example, given the following MLIR code with "source" and "destination"
// accesses to the same memref label, and symbols %M, %N, %K:
//
//   affine.for %i0 = 0 to 100 {
//     affine.for %i1 = 0 to 50 {
//       %a0 = affine.apply
//         (d0, d1) -> (d0 * 2 - d1 * 4 + s1, d1 * 3 - s0) (%i0, %i1)[%M, %N]
//       // Source memref access.
//       store %v0, %m[%a0#0, %a0#1] : memref<4x4xf32>
//     }
//   }
//
//   affine.for %i2 = 0 to 100 {
//     affine.for %i3 = 0 to 50 {
//       %a1 = affine.apply
//         (d0, d1) -> (d0 * 7 + d1 * 9 - s1, d1 * 11 + s0) (%i2, %i3)[%K, %M]
//       // Destination memref access.
//       %v1 = load %m[%a1#0, %a1#1] : memref<4x4xf32>
//     }
//   }
//
// The access functions would be the following:
//
//   src: (%i0 * 2 - %i1 * 4 + %N, %i1 * 3 - %M)
//   dst: (%i2 * 7 + %i3 * 9 - %M, %i3 * 11 - %K)
//
// The iteration domains for the src/dst accesses would be the following:
//
//   src: 0 <= %i0 <= 100, 0 <= %i1 <= 50
//   dst: 0 <= %i2 <= 100, 0 <= %i3 <= 50
//
// The symbols by both accesses would be assigned to a canonical position order
// which will be used in the dependence constraint system:
//
//   symbol name: %M  %N  %K
//   symbol  pos:  0   1   2
//
// Equality constraints are built by equating each result of src/destination
// access functions. For this example, the following two equality constraints
// will be added to the dependence constraint system:
//
//   [src_dim0, src_dim1, dst_dim0, dst_dim1, sym0, sym1, sym2, const]
//      2         -4        -7        -9       1      1     0     0    = 0
//      0          3         0        -11     -1      0     1     0    = 0
//
// Inequality constraints from the iteration domain will be meged into
// the dependence constraint system
//
//   [src_dim0, src_dim1, dst_dim0, dst_dim1, sym0, sym1, sym2, const]
//       1         0         0         0        0     0     0     0    >= 0
//      -1         0         0         0        0     0     0     100  >= 0
//       0         1         0         0        0     0     0     0    >= 0
//       0        -1         0         0        0     0     0     50   >= 0
//       0         0         1         0        0     0     0     0    >= 0
//       0         0        -1         0        0     0     0     100  >= 0
//       0         0         0         1        0     0     0     0    >= 0
//       0         0         0        -1        0     0     0     50   >= 0
//
//
// TODO(andydavis) Support AffineExprs mod/floordiv/ceildiv.
DependenceResult mlir::checkMemrefAccessDependence(
    const MemRefAccess &srcAccess, const MemRefAccess &dstAccess,
    unsigned loopDepth, FlatAffineConstraints *dependenceConstraints,
    SmallVector<DependenceComponent, 2> *dependenceComponents, bool allowRAR) {
  LLVM_DEBUG(llvm::dbgs() << "Checking for dependence at depth: "
                          << Twine(loopDepth) << " between:\n";);
  LLVM_DEBUG(srcAccess.opInst->dump(););
  LLVM_DEBUG(dstAccess.opInst->dump(););

  // Return 'NoDependence' if these accesses do not access the same memref.
  if (srcAccess.memref != dstAccess.memref)
    return DependenceResult::NoDependence;

  // Return 'NoDependence' if one of these accesses is not an AffineStoreOp.
  if (!allowRAR && !isa<AffineStoreOp>(srcAccess.opInst) &&
      !isa<AffineStoreOp>(dstAccess.opInst))
    return DependenceResult::NoDependence;

  // Get composed access function for 'srcAccess'.
  AffineValueMap srcAccessMap;
  srcAccess.getAccessMap(&srcAccessMap);

  // Get composed access function for 'dstAccess'.
  AffineValueMap dstAccessMap;
  dstAccess.getAccessMap(&dstAccessMap);

  // Get iteration domain for the 'srcAccess' operation.
  FlatAffineConstraints srcDomain;
  if (failed(getInstIndexSet(srcAccess.opInst, &srcDomain)))
    return DependenceResult::Failure;

  // Get iteration domain for 'dstAccess' operation.
  FlatAffineConstraints dstDomain;
  if (failed(getInstIndexSet(dstAccess.opInst, &dstDomain)))
    return DependenceResult::Failure;

  // Return 'NoDependence' if loopDepth > numCommonLoops and if the ancestor
  // operation of 'srcAccess' does not properly dominate the ancestor
  // operation of 'dstAccess' in the same common operation block.
  // Note: this check is skipped if 'allowRAR' is true, because because RAR
  // deps can exist irrespective of lexicographic ordering b/w src and dst.
  unsigned numCommonLoops = getNumCommonLoops(srcDomain, dstDomain);
  assert(loopDepth <= numCommonLoops + 1);
  if (!allowRAR && loopDepth > numCommonLoops &&
      !srcAppearsBeforeDstInAncestralBlock(srcAccess, dstAccess, srcDomain,
                                           numCommonLoops)) {
    return DependenceResult::NoDependence;
  }
  // Build dim and symbol position maps for each access from access operand
  // Value to position in merged constraint system.
  ValuePositionMap valuePosMap;
  buildDimAndSymbolPositionMaps(srcDomain, dstDomain, srcAccessMap,
                                dstAccessMap, &valuePosMap,
                                dependenceConstraints);

  initDependenceConstraints(srcDomain, dstDomain, srcAccessMap, dstAccessMap,
                            valuePosMap, dependenceConstraints);

  assert(valuePosMap.getNumDims() ==
         srcDomain.getNumDimIds() + dstDomain.getNumDimIds());

  // Create memref access constraint by equating src/dst access functions.
  // Note that this check is conservative, and will fail in the future when
  // local variables for mod/div exprs are supported.
  if (failed(addMemRefAccessConstraints(srcAccessMap, dstAccessMap, valuePosMap,
                                        dependenceConstraints)))
    return DependenceResult::Failure;

  // Add 'src' happens before 'dst' ordering constraints.
  addOrderingConstraints(srcDomain, dstDomain, loopDepth,
                         dependenceConstraints);
  // Add src and dst domain constraints.
  addDomainConstraints(srcDomain, dstDomain, valuePosMap,
                       dependenceConstraints);

  // Return 'NoDependence' if the solution space is empty: no dependence.
  if (dependenceConstraints->isEmpty()) {
    return DependenceResult::NoDependence;
  }

  // Compute dependence direction vector and return true.
  if (dependenceComponents != nullptr) {
    computeDirectionVector(srcDomain, dstDomain, loopDepth,
                           dependenceConstraints, dependenceComponents);
  }

  LLVM_DEBUG(llvm::dbgs() << "Dependence polyhedron:\n");
  LLVM_DEBUG(dependenceConstraints->dump());
  return DependenceResult::HasDependence;
}

/// Gathers dependence components for dependences between all ops in loop nest
/// rooted at 'forOp' at loop depths in range [1, maxLoopDepth].
void mlir::getDependenceComponents(
    AffineForOp forOp, unsigned maxLoopDepth,
    std::vector<SmallVector<DependenceComponent, 2>> *depCompsVec) {
  // Collect all load and store ops in loop nest rooted at 'forOp'.
  SmallVector<Operation *, 8> loadAndStoreOpInsts;
  forOp.getOperation()->walk([&](Operation *opInst) {
    if (isa<AffineLoadOp>(opInst) || isa<AffineStoreOp>(opInst))
      loadAndStoreOpInsts.push_back(opInst);
  });

  unsigned numOps = loadAndStoreOpInsts.size();
  for (unsigned d = 1; d <= maxLoopDepth; ++d) {
    for (unsigned i = 0; i < numOps; ++i) {
      auto *srcOpInst = loadAndStoreOpInsts[i];
      MemRefAccess srcAccess(srcOpInst);
      for (unsigned j = 0; j < numOps; ++j) {
        auto *dstOpInst = loadAndStoreOpInsts[j];
        MemRefAccess dstAccess(dstOpInst);

        FlatAffineConstraints dependenceConstraints;
        SmallVector<DependenceComponent, 2> depComps;
        // TODO(andydavis,bondhugula) Explore whether it would be profitable
        // to pre-compute and store deps instead of repeatedly checking.
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, d, &dependenceConstraints, &depComps);
        if (hasDependence(result))
          depCompsVec->push_back(depComps);
      }
    }
  }
}
