//===- ComposeAffineMaps.cpp - MLIR Affine Transform Class-----*- C++ -*-===//
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
// This file implements a pass to compose affine maps for all loads and stores.
// This transformation enables other transformations which require private
// affine apply operations for each load and store operation.
//
// For example. If you wanted to shift the compute and store operations in
// the following mlir code:
//
//  for %i = 0 to 255 {
//    %idx = affine_apply d0 -> d0 mod 2 (%i)
//    %v = load %A [%idx]
//    %x = compute (%v)
//    store %x, %A [%idx]
//  }
//
// First, you would apply the compose affine maps transformation to get the
// following mlir code where each load and store has its own private affine
// apply operation:
//
//  for %i = 0 to 255 {
//    %idx0 = affine_apply d0 -> d0 mod 2 (%i)
//    %v = load %A [%idx0]
//    %idx1 = affine_apply d0 -> d0 mod 2 (%i)
//    %x = compute (%v)
//    store %x, %A [%idx1]
//  }
//
// Next, you would apply your transformation to shift the compute and store
// operations, by applying the shift directly to store operations affine map,
// which is now private to the store operation after the compose affine maps
// transformation.
//
//  for %i = 0 to 255 {
//    %idx0 = affine_apply d0 -> d0 mod 2 (%i)
//    %v = load %A [%idx0]
//    %idx1 = affine_apply d0 -> d0 mod 2 (%i - 1)  // Shift transformation
//    %x = compute (%v)
//    store %x, %A [%idx1]
//  }
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardOps.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/Transforms/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace {

// ComposeAffineMaps composes affine maps, creating new single-use
// AffineApplyOp ops for each load and store op in an MLFunction.
// TODO(andydavis) Support composition with load/store layout affine maps
// (requires re-writing memref types and may not be possible if the memrefs
// are passsed in as MLFunction args).
// TODO(andydavis) Extend support to AffineBounds in for loops.
struct ComposeAffineMaps : public MLFunctionPass {
  explicit ComposeAffineMaps() {}

  PassResult runOnMLFunction(MLFunction *f);
};

} // end anonymous namespace

MLFunctionPass *mlir::createComposeAffineMapsPass() {
  return new ComposeAffineMaps();
}

// Creates and inserts into 'builder' a new AffineApplyOp with the number of
// results equal to the rank of 'memrefType'. The AffineApplyOp is composed
// with all other AffineApplyOps reachable from input paramter 'operands'.
// The final results of the composed AffineApplyOp are returned in output
// paramter 'results'.
static void createComposedAffineApplyOp(
    MLFuncBuilder *builder, Location *loc, MemRefType *memrefType,
    const SmallVector<MLValue *, 4> &indices,
    const SmallVector<OperationStmt *, 4> &affineApplyOps,
    SmallVector<SSAValue *, 4> *results) {
  // Get rank of memref type.
  unsigned rank = memrefType->getRank();
  assert(indices.size() == rank);
  // Create identity map with same number of dimensions as 'memrefType'.
  auto *map = builder->getMultiDimIdentityMap(rank);
  // Initialize AffineValueMap with identity map.
  AffineValueMap valueMap(map, indices, builder->getContext());

  for (auto *opStmt : affineApplyOps) {
    assert(opStmt->is<AffineApplyOp>());
    auto affineApplyOp = opStmt->getAs<AffineApplyOp>();
    // Forward substitute 'affineApplyOp' into 'valueMap'.
    valueMap.forwardSubstitute(*affineApplyOp);
  }
  // Compose affine maps from all ancestor AffineApplyOps.
  // Create new AffineApplyOp from 'valueMap'.
  unsigned numOperands = valueMap.getNumOperands();
  SmallVector<SSAValue *, 4> operands(numOperands);
  for (unsigned i = 0; i < numOperands; ++i) {
    operands[i] = valueMap.getOperand(i);
  }
  // Create new AffineApplyOp based on 'valueMap'.
  auto affineApplyOp =
      builder->create<AffineApplyOp>(loc, valueMap.getAffineMap(), operands);
  results->resize(rank);
  for (unsigned i = 0; i < rank; ++i) {
    (*results)[i] = affineApplyOp->getResult(i);
  }
}

PassResult ComposeAffineMaps::runOnMLFunction(MLFunction *f) {
  // Gather all loads, stores and affine apply ops.
  struct OpGatherer : public StmtWalker<OpGatherer> {
    std::vector<OpPointer<AffineApplyOp>> affineApplyOps;
    std::vector<OpPointer<LoadOp>> loadOps;
    std::vector<OpPointer<StoreOp>> storeOps;

    void visitOperationStmt(OperationStmt *opStmt) {
      if (auto affineApplyOp = opStmt->getAs<AffineApplyOp>()) {
        affineApplyOps.push_back(affineApplyOp);
      }
      if (auto loadOp = opStmt->getAs<LoadOp>()) {
        loadOps.push_back(loadOp);
      }
      if (auto storeOp = opStmt->getAs<StoreOp>()) {
        storeOps.push_back(storeOp);
      }
    }
  };

  OpGatherer og;
  og.walk(f);

  // Replace each LoadOp (and update its uses) with a new LoadOp which takes a
  // single-use composed affine map.
  std::vector<OpPointer<LoadOp>> loadOpsToDelete;
  loadOpsToDelete.reserve(og.loadOps.size());
  for (auto loadOp : og.loadOps) {
    auto *opStmt = cast<OperationStmt>(loadOp->getOperation());
    MLFuncBuilder builder(opStmt);
    auto *memrefType = cast<MemRefType>(loadOp->getMemRef()->getType());

    SmallVector<MLValue *, 4> indices;
    indices.reserve(memrefType->getRank());
    for (auto *index : loadOp->getIndices()) {
      indices.push_back(cast<MLValue>(index));
    }

    // Gather sequnce of AffineApplyOps reachable from 'indices'.
    SmallVector<OperationStmt *, 4> affineApplyOps;
    getReachableAffineApplyOps(indices, &affineApplyOps);
    // Skip transforming 'loadOp' if there are no affine maps to compose.
    if (affineApplyOps.size() <= 1)
      continue;

    SmallVector<SSAValue *, 4> results;
    createComposedAffineApplyOp(&builder, opStmt->getLoc(), memrefType, indices,
                                affineApplyOps, &results);
    // Create new LoadOp with new affine apply op.
    auto *newLoadResult =
        builder.create<LoadOp>(opStmt->getLoc(), loadOp->getMemRef(), results)
            ->getResult();
    // Update all uses of old LoadOp to take new LoadOp.
    loadOp->getResult()->replaceAllUsesWith(newLoadResult);
    loadOpsToDelete.push_back(loadOp);
  }

  // Replace each StoreOp (and update its uses) with a new StoreOp which takes a
  // single-use composed affine map.
  std::vector<OpPointer<StoreOp>> storeOpsToDelete;
  storeOpsToDelete.reserve(og.storeOps.size());
  for (auto storeOp : og.storeOps) {
    auto *opStmt = cast<OperationStmt>(storeOp->getOperation());
    MLFuncBuilder builder(opStmt);
    auto *memrefType = cast<MemRefType>(storeOp->getMemRef()->getType());

    SmallVector<MLValue *, 4> indices;
    indices.reserve(memrefType->getRank());
    for (auto *index : storeOp->getIndices()) {
      indices.push_back(cast<MLValue>(index));
    }
    // Gather sequnce of AffineApplyOps reachable from 'indices'.
    SmallVector<OperationStmt *, 4> affineApplyOps;
    getReachableAffineApplyOps(indices, &affineApplyOps);
    // Skip transforming 'storeOp' if there are no affine maps to compose.
    if (affineApplyOps.size() <= 1)
      continue;

    SmallVector<SSAValue *, 4> results;
    createComposedAffineApplyOp(&builder, opStmt->getLoc(), memrefType, indices,
                                affineApplyOps, &results);
    // Create new StoreOp with new affine apply op.
    builder.create<StoreOp>(opStmt->getLoc(), storeOp->getValueToStore(),
                            storeOp->getMemRef(), results);
    storeOpsToDelete.push_back(storeOp);
  }

  // Erase all unused StoreOps.
  for (auto storeOp : storeOpsToDelete) {
    cast<OperationStmt>(storeOp->getOperation())->eraseFromBlock();
  }

  // Erase all unused LoadOps.
  for (auto loadOp : loadOpsToDelete) {
    assert(loadOp->getResult()->use_empty());
    cast<OperationStmt>(loadOp->getOperation())->eraseFromBlock();
  }

  // Erase all unused AffineApplyOps in reverse order, as uses of
  // nested AffineApplyOps where not updated earlier.
  auto it_end = og.affineApplyOps.rend();
  for (auto it = og.affineApplyOps.rbegin(); it != it_end; ++it) {
    auto affineApplyOp = *it;
    bool allUsesEmpty = true;
    for (auto *result : affineApplyOp->getOperation()->getResults()) {
      if (!result->use_empty()) {
        allUsesEmpty = false;
        break;
      }
    }
    if (allUsesEmpty)
      cast<OperationStmt>(affineApplyOp->getOperation())->eraseFromBlock();
  }

  return success();
}
