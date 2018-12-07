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
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "analysis-utils"

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

/// Populates 'loops' with IVs of the loops surrounding 'stmt' ordered from
/// the outermost 'for' statement to the innermost one.
void mlir::getLoopIVs(const Statement &stmt,
                      SmallVector<const ForStmt *, 4> *loops) {
  const auto *currStmt = stmt.getParentStmt();
  while (currStmt != nullptr && isa<ForStmt>(currStmt)) {
    loops->push_back(dyn_cast<ForStmt>(currStmt));
    currStmt = currStmt->getParentStmt();
  }
  std::reverse(loops->begin(), loops->end());
}

unsigned MemRefRegion::getRank() const {
  return memref->getType().cast<MemRefType>().getRank();
}

Optional<int64_t> MemRefRegion::getBoundingConstantSizeAndShape(
    SmallVectorImpl<int> *shape,
    std::vector<SmallVector<int64_t, 4>> *lbs) const {
  auto memRefType = memref->getType().cast<MemRefType>();
  unsigned rank = memRefType.getRank();
  shape->reserve(rank);

  // Find a constant upper bound on the extent of this memref region along each
  // dimension.
  int64_t numElements = 1;
  int64_t diffConstant;
  for (unsigned d = 0; d < rank; d++) {
    SmallVector<int64_t, 4> lb;
    Optional<int64_t> diff = cst.getConstantBoundOnDimSize(d, &lb);
    if (diff.hasValue()) {
      diffConstant = diff.getValue();
    } else {
      // If no constant bound is found, then it can always be bound by the
      // memref's dim size if the latter has a constant size along this dim.
      auto dimSize = memRefType.getDimSize(d);
      if (dimSize == -1)
        return None;
      diffConstant = dimSize;
      // Lower bound becomes 0.
      lb.resize(cst.getNumSymbolIds() + 1, 0);
    }
    numElements *= diffConstant;
    if (lbs) {
      lbs->push_back(lb);
    }
    if (shape) {
      shape->push_back(diffConstant);
    }
  }
  return numElements;
}

/// Computes the memory region accessed by this memref with the region
/// represented as constraints symbolic/parameteric in 'loopDepth' loops
/// surrounding opStmt. Returns false if this fails due to yet unimplemented
/// cases.
//  For example, the memref region for this load operation at loopDepth = 1 will
//  be as below:
//
//    for %i = 0 to 32 {
//      for %ii = %i to (d0) -> (d0 + 8) (%i) {
//        load %A[%ii]
//      }
//    }
//
// region:  {memref = %A, write = false, {%i <= m0 <= %i + 7} }
// The last field is a 2-d FlatAffineConstraints symbolic in %i.
//
// TODO(bondhugula): extend this to any other memref dereferencing ops
// (dma_start, dma_wait).
bool mlir::getMemRefRegion(OperationStmt *opStmt, unsigned loopDepth,
                           MemRefRegion *region) {
  OpPointer<LoadOp> loadOp;
  OpPointer<StoreOp> storeOp;
  unsigned rank;
  SmallVector<MLValue *, 4> indices;

  if ((loadOp = opStmt->dyn_cast<LoadOp>())) {
    rank = loadOp->getMemRefType().getRank();
    for (auto *index : loadOp->getIndices()) {
      indices.push_back(cast<MLValue>(index));
    }
    region->memref = cast<MLValue>(loadOp->getMemRef());
    region->setWrite(false);
  } else if ((storeOp = opStmt->dyn_cast<StoreOp>())) {
    rank = storeOp->getMemRefType().getRank();
    for (auto *index : storeOp->getIndices()) {
      indices.push_back(cast<MLValue>(index));
    }
    region->memref = cast<MLValue>(storeOp->getMemRef());
    region->setWrite(true);
  } else {
    return false;
  }

  // Build the constraints for this region.
  FlatAffineConstraints *regionCst = region->getConstraints();

  MLFuncBuilder b(opStmt);
  auto idMap = b.getMultiDimIdentityMap(rank);

  // Initialize 'accessValueMap' and compose with reachable AffineApplyOps.
  AffineValueMap accessValueMap(idMap, indices);
  forwardSubstituteReachableOps(&accessValueMap);
  AffineMap accessMap = accessValueMap.getAffineMap();

  regionCst->reset(accessMap.getNumDims(), accessMap.getNumSymbols(), 0,
                   accessValueMap.getOperands());

  // Add equality constraints.
  unsigned numDims = accessMap.getNumDims();
  unsigned numSymbols = accessMap.getNumSymbols();
  // Add inequalties for loop lower/upper bounds.
  for (unsigned i = 0; i < numDims + numSymbols; ++i) {
    if (auto *loop = dyn_cast<ForStmt>(accessValueMap.getOperand(i))) {
      // Note that regionCst can now have more dimensions than accessMap if the
      // bounds expressions involve outer loops or other symbols.
      if (!regionCst->addBoundsFromForStmt(*loop))
        return false;
    } else {
      // Has to be a valid symbol.
      auto *symbol = cast<MLValue>(accessValueMap.getOperand(i));
      assert(symbol->isValidSymbol());
      // Check if the symbol is a constant.
      if (auto *opStmt = symbol->getDefiningStmt()) {
        if (auto constOp = opStmt->dyn_cast<ConstantIndexOp>()) {
          regionCst->setIdToConstant(*symbol, constOp->getValue());
        }
      }
    }
  }

  // Add access function equalities to connect loop IVs to data dimensions.
  if (!regionCst->composeMap(&accessValueMap)) {
    LLVM_DEBUG(llvm::dbgs() << "getMemRefRegion: compose affine map failed\n");
    return false;
  }

  // Eliminate any loop IVs other than the outermost 'loopDepth' IVs, on which
  // this memref region is symbolic.
  SmallVector<const ForStmt *, 4> outerIVs;
  getLoopIVs(*opStmt, &outerIVs);
  outerIVs.resize(loopDepth);
  for (auto *operand : accessValueMap.getOperands()) {
    ForStmt *iv;
    if ((iv = dyn_cast<ForStmt>(operand)) &&
        std::find(outerIVs.begin(), outerIVs.end(), iv) == outerIVs.end()) {
      regionCst->projectOut(operand);
    }
  }
  // Project out any local variables (these would have been added for any
  // mod/divs).
  regionCst->projectOut(regionCst->getNumDimIds() +
                            regionCst->getNumSymbolIds(),
                        regionCst->getNumLocalIds());

  // Tighten the set.
  regionCst->GCDTightenInequalities();

  // Set all identifiers appearing after the first 'rank' identifiers as
  // symbolic identifiers - so that the ones correspoding to the memref
  // dimensions are the dimensional identifiers for the memref region.
  regionCst->setDimSymbolSeparation(regionCst->getNumIds() - rank);

  // Constant fold any symbolic identifiers.
  regionCst->constantFoldIdRange(/*pos=*/regionCst->getNumDimIds(),
                                 /*num=*/regionCst->getNumSymbolIds());

  assert(regionCst->getNumDimIds() == rank && "unexpected MemRefRegion format");

  return true;
}

/// Returns the size of memref data in bytes if it's statically shaped, None
/// otherwise.
Optional<uint64_t> mlir::getMemRefSizeInBytes(MemRefType memRefType) {
  if (memRefType.getNumDynamicDims() > 0)
    return None;
  uint64_t sizeInBits = memRefType.getElementType().getBitWidth();
  for (unsigned i = 0, e = memRefType.getRank(); i < e; i++) {
    sizeInBits = sizeInBits * memRefType.getDimSize(i);
  }
  return llvm::divideCeil(sizeInBits, 8);
}
