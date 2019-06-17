//===- LoopFusionUtils.cpp ---- Utilities for loop fusion ----------===//
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
// This file implements loop fusion transformation utility functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/LoopFusionUtils.h"

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/StandardOps/Ops.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "loop-fusion-utils"

using namespace mlir;

// Gathers all load and store memref accesses in 'opA' into 'values', where
// 'values[memref] == true' for each store operation.
static void getLoadAndStoreMemRefAccesses(Operation *opA,
                                          DenseMap<Value *, bool> &values) {
  opA->walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<LoadOp>(op)) {
      if (values.count(loadOp.getMemRef()) == 0)
        values[loadOp.getMemRef()] = false;
    } else if (auto storeOp = dyn_cast<StoreOp>(op)) {
      values[storeOp.getMemRef()] = true;
    }
  });
}

// Returns true if 'op' is a load or store operation which access an memref
// accessed 'values' and at least one of the access is a store operation.
// Returns false otherwise.
static bool isDependentLoadOrStoreOp(Operation *op,
                                     DenseMap<Value *, bool> &values) {
  if (auto loadOp = dyn_cast<LoadOp>(op)) {
    return values.count(loadOp.getMemRef()) > 0 &&
           values[loadOp.getMemRef()] == true;
  } else if (auto storeOp = dyn_cast<StoreOp>(op)) {
    return values.count(storeOp.getMemRef()) > 0;
  }
  return false;
}

// Returns the first operation in range ('opA', 'opB') which has a data
// dependence on 'opA'. Returns 'nullptr' of no dependence exists.
static Operation *getFirstDependentOpInRange(Operation *opA, Operation *opB) {
  // Record memref values from all loads/store in loop nest rooted at 'opA'.
  // Map from memref value to bool which is true if store, false otherwise.
  DenseMap<Value *, bool> values;
  getLoadAndStoreMemRefAccesses(opA, values);

  // For each 'opX' in block in range ('opA', 'opB'), check if there is a data
  // dependence from 'opA' to 'opX' ('opA' and 'opX' access the same memref
  // and at least one of the accesses is a store).
  Operation *firstDepOp = nullptr;
  for (Block::iterator it = std::next(Block::iterator(opA));
       it != Block::iterator(opB); ++it) {
    Operation *opX = &(*it);
    opX->walk([&](Operation *op) {
      if (!firstDepOp && isDependentLoadOrStoreOp(op, values))
        firstDepOp = opX;
    });
    if (firstDepOp)
      break;
  }
  return firstDepOp;
}

// Returns the last operation 'opX' in range ('opA', 'opB'), for which there
// exists a data dependence from 'opX' to 'opB'.
// Returns 'nullptr' of no dependence exists.
static Operation *getLastDependentOpInRange(Operation *opA, Operation *opB) {
  // Record memref values from all loads/store in loop nest rooted at 'opB'.
  // Map from memref value to bool which is true if store, false otherwise.
  DenseMap<Value *, bool> values;
  getLoadAndStoreMemRefAccesses(opB, values);

  // For each 'opX' in block in range ('opA', 'opB') in reverse order,
  // check if there is a data dependence from 'opX' to 'opB':
  // *) 'opX' and 'opB' access the same memref and at least one of the accesses
  //    is a store.
  // *) 'opX' produces an SSA Value which is used by 'opB'.
  Operation *lastDepOp = nullptr;
  for (Block::reverse_iterator it = std::next(Block::reverse_iterator(opB));
       it != Block::reverse_iterator(opA); ++it) {
    Operation *opX = &(*it);
    opX->walk([&](Operation *op) {
      if (lastDepOp)
        return;
      if (isa<LoadOp>(op) || isa<StoreOp>(op)) {
        if (isDependentLoadOrStoreOp(op, values))
          lastDepOp = opX;
        return;
      }
      for (auto *value : op->getResults()) {
        for (auto *user : value->getUsers()) {
          SmallVector<AffineForOp, 4> loops;
          // Check if any loop in loop nest surrounding 'user' is 'opB'.
          getLoopIVs(*user, &loops);
          if (llvm::is_contained(loops, cast<AffineForOp>(opB))) {
            lastDepOp = opX;
          }
        }
      }
    });
    if (lastDepOp)
      break;
  }
  return lastDepOp;
}

// Computes and returns an insertion point operation, before which the
// the fused <srcForOp, dstForOp> loop nest can be inserted while preserving
// dependences. Returns nullptr if no such insertion point is found.
static Operation *getFusedLoopNestInsertionPoint(AffineForOp srcForOp,
                                                 AffineForOp dstForOp) {
  bool isSrcForOpBeforeDstForOp =
      srcForOp.getOperation()->isBeforeInBlock(dstForOp.getOperation());
  auto forOpA = isSrcForOpBeforeDstForOp ? srcForOp : dstForOp;
  auto forOpB = isSrcForOpBeforeDstForOp ? dstForOp : srcForOp;

  auto *firstDepOpA =
      getFirstDependentOpInRange(forOpA.getOperation(), forOpB.getOperation());
  auto *lastDepOpB =
      getLastDependentOpInRange(forOpA.getOperation(), forOpB.getOperation());
  // Block:
  //      ...
  //  |-- opA
  //  |   ...
  //  |   lastDepOpB --|
  //  |   ...          |
  //  |-> firstDepOpA  |
  //      ...          |
  //      opB <---------
  //
  // Valid insertion point range: (lastDepOpB, firstDepOpA)
  //
  if (firstDepOpA != nullptr) {
    if (lastDepOpB != nullptr) {
      if (firstDepOpA->isBeforeInBlock(lastDepOpB) || firstDepOpA == lastDepOpB)
        // No valid insertion point exists which preserves dependences.
        return nullptr;
    }
    // Return insertion point in valid range closest to 'opB'.
    // TODO(andydavis) Consider other insertion points in valid range.
    return firstDepOpA;
  }
  // No dependences from 'opA' to operation in range ('opA', 'opB'), return
  // 'opB' insertion point.
  return forOpB.getOperation();
}

// Gathers all load and store ops in loop nest rooted at 'forOp' into
// 'loadAndStoreOps'.
static bool
gatherLoadsAndStores(AffineForOp forOp,
                     SmallVectorImpl<Operation *> &loadAndStoreOps) {
  bool hasIfOp = false;
  forOp.getOperation()->walk([&](Operation *op) {
    if (isa<LoadOp>(op) || isa<StoreOp>(op))
      loadAndStoreOps.push_back(op);
    else if (isa<AffineIfOp>(op))
      hasIfOp = true;
  });
  return !hasIfOp;
}

// TODO(andydavis) Prevent fusion of loop nests with side-effecting operations.
FusionResult mlir::canFuseLoops(AffineForOp srcForOp, AffineForOp dstForOp,
                                unsigned dstLoopDepth,
                                ComputationSliceState *srcSlice) {
  // Return 'failure' if 'dstLoopDepth == 0'.
  if (dstLoopDepth == 0) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot fuse loop nests at depth 0\n.");
    return FusionResult::FailPrecondition;
  }
  // Return 'failure' if 'srcForOp' and 'dstForOp' are not in the same block.
  auto *block = srcForOp.getOperation()->getBlock();
  if (block != dstForOp.getOperation()->getBlock()) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot fuse loop nests in different blocks\n.");
    return FusionResult::FailPrecondition;
  }

  // Return 'failure' if no valid insertion point for fused loop nest in 'block'
  // exists which would preserve dependences.
  if (!getFusedLoopNestInsertionPoint(srcForOp, dstForOp)) {
    LLVM_DEBUG(llvm::dbgs() << "Fusion would violate dependences in block\n.");
    return FusionResult::FailBlockDependence;
  }

  // Check if 'srcForOp' precedeces 'dstForOp' in 'block'.
  bool isSrcForOpBeforeDstForOp =
      srcForOp.getOperation()->isBeforeInBlock(dstForOp.getOperation());
  // 'forOpA' executes before 'forOpB' in 'block'.
  auto forOpA = isSrcForOpBeforeDstForOp ? srcForOp : dstForOp;
  auto forOpB = isSrcForOpBeforeDstForOp ? dstForOp : srcForOp;

  // Gather all load and store from 'forOpA' which precedes 'forOpB' in 'block'.
  SmallVector<Operation *, 4> opsA;
  if (!gatherLoadsAndStores(forOpA, opsA)) {
    LLVM_DEBUG(llvm::dbgs() << "Fusing loops with affine.if unsupported.\n.");
    return FusionResult::FailPrecondition;
  }

  // Gather all load and store from 'forOpB' which succeeds 'forOpA' in 'block'.
  SmallVector<Operation *, 4> opsB;
  if (!gatherLoadsAndStores(forOpB, opsB)) {
    LLVM_DEBUG(llvm::dbgs() << "Fusing loops with affine.if unsupported.\n.");
    return FusionResult::FailPrecondition;
  }

  // Calculate the number of common loops surrounding 'srcForOp' and 'dstForOp'.
  unsigned numCommonLoops = mlir::getNumCommonSurroundingLoops(
      *srcForOp.getOperation(), *dstForOp.getOperation());

  // Compute union of computation slices computed between all pairs of ops
  // from 'forOpA' and 'forOpB'.
  if (failed(mlir::computeSliceUnion(opsA, opsB, dstLoopDepth, numCommonLoops,
                                     isSrcForOpBeforeDstForOp, srcSlice))) {
    LLVM_DEBUG(llvm::dbgs() << "computeSliceUnion failed\n");
    return FusionResult::FailPrecondition;
  }

  return FusionResult::Success;
}
