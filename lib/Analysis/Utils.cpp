//===- Utils.cpp ---- Misc utilities for analysis -------------------------===//
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
// This file implements miscellaneous analysis routines for non-loop IR
// structures.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Utils.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/StandardOps/StandardOps.h"

using namespace mlir;

/// Returns true if statement 'a' properly dominates statement b.
bool mlir::properlyDominates(const Statement &a, const Statement &b) {
  if (&a == &b)
    return false;

  if (a.findFunction() != b.findFunction())
    return false;

  if (a.getBlock() == b.getBlock()) {
    // Do a linear scan to determine whether b comes after a.
    auto aIter = StmtBlock::const_iterator(a);
    auto bIter = StmtBlock::const_iterator(b);
    auto aBlockStart = a.getBlock()->begin();
    while (bIter != aBlockStart) {
      --bIter;
      if (aIter == bIter)
        return true;
    }
    return false;
  }

  // Traverse up b's hierarchy to check if b's block is contained in a's.
  if (const auto *bAncestor = a.getBlock()->findAncestorStmtInBlock(b))
    // a and bAncestor are in the same block; check if the former dominates it.
    return dominates(a, *bAncestor);

  // b's block is not contained in A.
  return false;
}

/// Returns true if statement A dominates statement B.
bool mlir::dominates(const Statement &a, const Statement &b) {
  return &a == &b || properlyDominates(a, b);
}

/// Returns the memory region accessed by this memref.
// TODO(bondhugula): extend this to store's and other memref dereferencing ops.
bool mlir::getMemoryRegion(OperationStmt *opStmt,
                           FlatAffineConstraints *region) {
  OpPointer<LoadOp> loadOp;
  if (!(loadOp = opStmt->dyn_cast<LoadOp>()))
    return false;

  unsigned rank = loadOp->getMemRefType().getRank();
  MLFuncBuilder b(opStmt);
  auto idMap = b.getMultiDimIdentityMap(rank);

  SmallVector<MLValue *, 4> indices;
  for (auto *index : loadOp->getIndices()) {
    indices.push_back(cast<MLValue>(index));
  }

  // Initialize 'accessMap' and compose with reachable AffineApplyOps.
  AffineValueMap accessMap(idMap, indices);
  forwardSubstituteReachableOps(&accessMap);
  AffineMap srcMap = accessMap.getAffineMap();

  region->reset(srcMap.getNumDims(), srcMap.getNumSymbols());

  // Add equality constraints.
  AffineMap map = accessMap.getAffineMap();
  unsigned numDims = map.getNumDims();
  unsigned numSymbols = map.getNumSymbols();
  // Add inEqualties for loop lower/upper bounds.
  for (unsigned i = 0; i < numDims + numSymbols; ++i) {
    if (auto *loop = dyn_cast<ForStmt>(accessMap.getOperand(i))) {
      if (!loop->hasConstantBounds())
        return false;
      // Add lower bound and upper bounds.
      region->addConstantLowerBound(i, loop->getConstantLowerBound());
      region->addConstantUpperBound(i, loop->getConstantUpperBound() - 1);
    } else {
      // Has to be a valid symbol.
      auto *symbol = cast<MLValue>(accessMap.getOperand(i));
      assert(symbol->isValidSymbol());
      // Check if the symbols is a constant.
      if (auto *opStmt = symbol->getDefiningStmt()) {
        if (auto constOp = opStmt->dyn_cast<ConstantIndexOp>()) {
          region->setIdToConstant(i, constOp->getValue());
        }
      }
    }
  }

  // Add access function equalities to connect loop IVs to data dimensions.
  region->composeMap(&accessMap);

  // Eliminate the loop IVs and any local variables to yield the memory region
  // involving just the memref dimensions.
  region->projectOut(srcMap.getNumResults(),
                     accessMap.getNumOperands() + region->getNumLocalIds());
  assert(region->getNumDimIds() == rank);
  return true;
}
