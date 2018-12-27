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
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "analysis-utils"

using namespace mlir;

/// Returns true if statement 'a' properly dominates statement b.
bool mlir::properlyDominates(const Statement &a, const Statement &b) {
  if (&a == &b)
    return false;

  if (a.getFunction() != b.getFunction())
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
                      SmallVectorImpl<ForStmt *> *loops) {
  auto *currStmt = stmt.getParentStmt();
  ForStmt *currForStmt;
  // Traverse up the hierarchy collecing all 'for' statement while skipping over
  // 'if' statements.
  while (currStmt && ((currForStmt = dyn_cast<ForStmt>(currStmt)) ||
                      isa<IfStmt>(currStmt))) {
    if (currForStmt)
      loops->push_back(currForStmt);
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
/// surrounding opStmt and any additional MLFunction symbols. Returns false if
/// this fails due to yet unimplemented cases.
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
  SmallVector<Value *, 4> indices;

  if ((loadOp = opStmt->dyn_cast<LoadOp>())) {
    rank = loadOp->getMemRefType().getRank();
    for (auto *index : loadOp->getIndices()) {
      indices.push_back(index);
    }
    region->memref = loadOp->getMemRef();
    region->setWrite(false);
  } else if ((storeOp = opStmt->dyn_cast<StoreOp>())) {
    rank = storeOp->getMemRefType().getRank();
    for (auto *index : storeOp->getIndices()) {
      indices.push_back(index);
    }
    region->memref = storeOp->getMemRef();
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

  // We'll first associate the dims and symbols of the access map to the dims
  // and symbols resp. of regionCst. This will change below once regionCst is
  // fully constructed out.
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
      // TODO(bondhugula): rewrite this to use getStmtIndexSet; this way
      // conditionals will be handled when the latter supports it.
      if (!regionCst->addForStmtDomain(*loop))
        return false;
    } else {
      // Has to be a valid symbol.
      auto *symbol = accessValueMap.getOperand(i);
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
  SmallVector<ForStmt *, 4> outerIVs;
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
/// otherwise.  If the element of the memref has vector type, takes into account
/// size of the vector as well.
Optional<uint64_t> mlir::getMemRefSizeInBytes(MemRefType memRefType) {
  if (memRefType.getNumDynamicDims() > 0)
    return None;
  auto elementType = memRefType.getElementType();
  if (!elementType.isIntOrFloat() && !elementType.isa<VectorType>())
    return None;

  uint64_t sizeInBits;
  if (elementType.isIntOrFloat()) {
    sizeInBits = elementType.getIntOrFloatBitWidth();
  } else {
    auto vectorType = elementType.cast<VectorType>();
    sizeInBits =
        vectorType.getElementTypeBitWidth() * vectorType.getNumElements();
  }
  for (unsigned i = 0, e = memRefType.getRank(); i < e; i++) {
    sizeInBits = sizeInBits * memRefType.getDimSize(i);
  }
  return llvm::divideCeil(sizeInBits, 8);
}

template <typename LoadOrStoreOpPointer>
bool mlir::boundCheckLoadOrStoreOp(LoadOrStoreOpPointer loadOrStoreOp,
                                   bool emitError) {
  static_assert(
      std::is_same<LoadOrStoreOpPointer, OpPointer<LoadOp>>::value ||
          std::is_same<LoadOrStoreOpPointer, OpPointer<StoreOp>>::value,
      "function argument should be either a LoadOp or a StoreOp");

  OperationStmt *opStmt = cast<OperationStmt>(loadOrStoreOp->getOperation());
  MemRefRegion region;
  if (!getMemRefRegion(opStmt, /*loopDepth=*/0, &region))
    return false;
  LLVM_DEBUG(llvm::dbgs() << "Memory region");
  LLVM_DEBUG(region.getConstraints()->dump());

  bool outOfBounds = false;
  unsigned rank = loadOrStoreOp->getMemRefType().getRank();

  // For each dimension, check for out of bounds.
  for (unsigned r = 0; r < rank; r++) {
    FlatAffineConstraints ucst(*region.getConstraints());

    // Intersect memory region with constraint capturing out of bounds (both out
    // of upper and out of lower), and check if the constraint system is
    // feasible. If it is, there is at least one point out of bounds.
    SmallVector<int64_t, 4> ineq(rank + 1, 0);
    int dimSize = loadOrStoreOp->getMemRefType().getDimSize(r);
    // TODO(bondhugula): handle dynamic dim sizes.
    if (dimSize == -1)
      continue;

    // Check for overflow: d_i >= memref dim size.
    ucst.addConstantLowerBound(r, dimSize);
    outOfBounds = !ucst.isEmpty();
    if (outOfBounds && emitError) {
      loadOrStoreOp->emitOpError(
          "memref out of upper bound access along dimension #" + Twine(r + 1));
    }

    // Check for a negative index.
    FlatAffineConstraints lcst(*region.getConstraints());
    std::fill(ineq.begin(), ineq.end(), 0);
    // d_i <= -1;
    lcst.addConstantUpperBound(r, -1);
    outOfBounds = !lcst.isEmpty();
    if (outOfBounds && emitError) {
      loadOrStoreOp->emitOpError(
          "memref out of lower bound access along dimension #" + Twine(r + 1));
    }
  }
  return outOfBounds;
}

// Explicitly instantiate the template so that the compiler knows we need them!
template bool mlir::boundCheckLoadOrStoreOp(OpPointer<LoadOp> loadOp,
                                            bool emitError);
template bool mlir::boundCheckLoadOrStoreOp(OpPointer<StoreOp> storeOp,
                                            bool emitError);

// Returns in 'positions' the StmtBlock positions of 'stmt' in each ancestor
// StmtBlock from the StmtBlock containing statement, stopping at 'limitBlock'.
static void findStmtPosition(const Statement *stmt, StmtBlock *limitBlock,
                             SmallVectorImpl<unsigned> *positions) {
  StmtBlock *block = stmt->getBlock();
  while (block != limitBlock) {
    int stmtPosInBlock = block->findStmtPosInBlock(*stmt);
    assert(stmtPosInBlock >= 0);
    positions->push_back(stmtPosInBlock);
    stmt = block->getContainingStmt();
    block = stmt->getBlock();
  }
  std::reverse(positions->begin(), positions->end());
}

// Returns the Statement in a possibly nested set of StmtBlocks, where the
// position of the statement is represented by 'positions', which has a
// StmtBlock position for each level of nesting.
static Statement *getStmtAtPosition(ArrayRef<unsigned> positions,
                                    unsigned level, StmtBlock *block) {
  unsigned i = 0;
  for (auto &stmt : *block) {
    if (i != positions[level]) {
      ++i;
      continue;
    }
    if (level == positions.size() - 1)
      return &stmt;
    if (auto *childForStmt = dyn_cast<ForStmt>(&stmt))
      return getStmtAtPosition(positions, level + 1, childForStmt->getBody());

    if (auto *ifStmt = dyn_cast<IfStmt>(&stmt)) {
      auto *ret = getStmtAtPosition(positions, level + 1, ifStmt->getThen());
      if (ret != nullptr)
        return ret;
      if (auto *elseClause = ifStmt->getElse())
        return getStmtAtPosition(positions, level + 1, elseClause);
    }
  }
  return nullptr;
}

// Computes memref dependence between 'srcAccess' and 'dstAccess' and uses the
// dependence constraint system to create AffineMaps with which to adjust the
// loop bounds of the inserted compution slice so that they are functions of the
// loop IVs and symbols of the loops surrounding 'dstAccess'.
ForStmt *mlir::insertBackwardComputationSlice(MemRefAccess *srcAccess,
                                              MemRefAccess *dstAccess,
                                              unsigned srcLoopDepth,
                                              unsigned dstLoopDepth) {
  FlatAffineConstraints dependenceConstraints;
  if (!checkMemrefAccessDependence(*srcAccess, *dstAccess, /*loopDepth=*/1,
                                   &dependenceConstraints,
                                   /*dependenceComponents=*/nullptr)) {
    return nullptr;
  }
  // Get loop nest surrounding src operation.
  SmallVector<ForStmt *, 4> srcLoopNest;
  getLoopIVs(*srcAccess->opStmt, &srcLoopNest);
  unsigned srcLoopNestSize = srcLoopNest.size();
  assert(srcLoopDepth <= srcLoopNestSize);

  // Get loop nest surrounding dst operation.
  SmallVector<ForStmt *, 4> dstLoopNest;
  getLoopIVs(*dstAccess->opStmt, &dstLoopNest);
  unsigned dstLoopNestSize = dstLoopNest.size();
  (void)dstLoopNestSize;
  assert(dstLoopDepth > 0);
  assert(dstLoopDepth <= dstLoopNestSize);

  // Solve for src IVs in terms of dst IVs, symbols and constants.
  SmallVector<AffineMap, 4> srcIvMaps(srcLoopNestSize, AffineMap::Null());
  std::vector<SmallVector<Value *, 2>> srcIvOperands(srcLoopNestSize);
  for (unsigned i = 0; i < srcLoopNestSize; ++i) {
    // Skip IVs which are greater than requested loop depth.
    if (i >= srcLoopDepth) {
      srcIvMaps[i] = AffineMap::Null();
      continue;
    }
    auto cst = dependenceConstraints.clone();
    for (int j = srcLoopNestSize - 1; j >= 0; --j) {
      if (i != j)
        cst->projectOut(j);
    }
    // TODO(andydavis) Check for case with two equalities where we have
    // set on IV to a constant. Set a constant IV map for these cases.
    if (cst->getNumEqualities() != 1) {
      srcIvMaps[i] = AffineMap::Null();
      continue;
    }
    SmallVector<unsigned, 2> nonZeroDimIds;
    SmallVector<unsigned, 2> nonZeroSymbolIds;
    srcIvMaps[i] = cst->toAffineMapFromEq(0, 0, srcAccess->opStmt->getContext(),
                                          &nonZeroDimIds, &nonZeroSymbolIds);
    if (srcIvMaps[i] == AffineMap::Null()) {
      continue;
    }
    // Add operands for all non-zero dst dims and symbols.
    // TODO(andydavis) Add local variable support.
    for (auto dimId : nonZeroDimIds) {
      if (dimId - 1 >= dstLoopDepth) {
        // This src IV has a dependence on dst IV dstLoopDepth where it will
        // be inserted. So we cannot slice the iteration space at srcLoopDepth,
        // and also insert it into the dst loop nest at 'dstLoopDepth'.
        return nullptr;
      }
      srcIvOperands[i].push_back(dstLoopNest[dimId - 1]);
    }
    // TODO(andydavis) Add symbols from the access function. Ideally, we
    // should be able to query the constaint system for the Value associated
    // with a symbol identifiers in 'nonZeroSymbolIds'.
  }

  // Find the stmt block positions of 'srcAccess->opStmt' within 'srcLoopNest'.
  SmallVector<unsigned, 4> positions;
  findStmtPosition(srcAccess->opStmt, srcLoopNest[0]->getBlock(), &positions);

  // Clone src loop nest and insert it a the beginning of the statement block
  // of the loop at 'dstLoopDepth' in 'dstLoopNest'.
  auto *dstForStmt = dstLoopNest[dstLoopDepth - 1];
  MLFuncBuilder b(dstForStmt->getBody(), dstForStmt->getBody()->begin());
  DenseMap<const Value *, Value *> operandMap;
  auto *sliceLoopNest = cast<ForStmt>(b.clone(*srcLoopNest[0], operandMap));

  // Lookup stmt in cloned 'sliceLoopNest' at 'positions'.
  Statement *sliceStmt =
      getStmtAtPosition(positions, /*level=*/0, sliceLoopNest->getBody());
  // Get loop nest surrounding 'sliceStmt'.
  SmallVector<ForStmt *, 4> sliceSurroundingLoops;
  getLoopIVs(*sliceStmt, &sliceSurroundingLoops);
  unsigned sliceSurroundingLoopsSize = sliceSurroundingLoops.size();
  (void)sliceSurroundingLoopsSize;

  // Update loop bounds for loops in 'sliceLoopNest'.
  unsigned sliceLoopLimit = dstLoopDepth + srcLoopNestSize;
  assert(sliceLoopLimit <= sliceSurroundingLoopsSize);
  for (unsigned i = dstLoopDepth; i < sliceLoopLimit; ++i) {
    auto *forStmt = sliceSurroundingLoops[i];
    unsigned index = i - dstLoopDepth;
    AffineMap lbMap = srcIvMaps[index];
    if (lbMap == AffineMap::Null())
      continue;
    forStmt->setLowerBound(srcIvOperands[index], lbMap);
    // Create upper bound map with is lower bound map + 1;
    assert(lbMap.getNumResults() == 1);
    AffineExpr ubResultExpr = lbMap.getResult(0) + 1;
    AffineMap ubMap = AffineMap::get(lbMap.getNumDims(), lbMap.getNumSymbols(),
                                     {ubResultExpr}, {});
    forStmt->setUpperBound(srcIvOperands[index], ubMap);
  }
  return sliceLoopNest;
}
