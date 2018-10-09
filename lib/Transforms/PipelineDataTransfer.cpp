//===- PipelineDataTransfer.cpp --- Pass for pipelining data movement ---*-===//
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
// This file implements a pass to pipeline data transfers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardOps.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Pass.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace {

struct PipelineDataTransfer : public MLFunctionPass {
  explicit PipelineDataTransfer() {}
  PassResult runOnMLFunction(MLFunction *f) override;
};

} // end anonymous namespace

/// Creates a pass to pipeline explicit movement of data across levels of the
/// memory hierarchy.
MLFunctionPass *mlir::createPipelineDataTransferPass() {
  return new PipelineDataTransfer();
}

/// Given a DMA start operation, returns the operand position of either the
/// source or destination memref depending on the one that is at the higher
/// level of the memory hierarchy.
// Temporary utility: will be replaced when DmaStart/DmaFinish abstract op's are
// added.  TODO(b/117228571)
static unsigned getHigherMemRefPos(OpPointer<DmaStartOp> dmaStartOp) {
  unsigned srcDmaPos = 0;
  unsigned destDmaPos = dmaStartOp->getSrcMemRefRank() + 1;

  if (dmaStartOp->getSrcMemorySpace() > dmaStartOp->getDstMemorySpace())
    return srcDmaPos;
  return destDmaPos;
}

// Returns the position of the tag memref operand given a DMA statement.
// Temporary utility: will be replaced when DmaStart/DmaFinish abstract op's are
// added.  TODO(b/117228571)
static unsigned getTagMemRefPos(const OperationStmt &dmaStmt) {
  assert(dmaStmt.is<DmaStartOp>() || dmaStmt.is<DmaWaitOp>());
  if (dmaStmt.is<DmaStartOp>()) {
    // Second to last operand.
    return dmaStmt.getNumOperands() - 2;
  }
  // First operand for a dma finish statement.
  return 0;
}

/// Doubles the buffer of the supplied memref while replacing all uses of the
/// old memref. Returns false if such a replacement cannot be performed.
static bool doubleBuffer(MLValue *oldMemRef, ForStmt *forStmt) {
  MLFuncBuilder bInner(forStmt, forStmt->begin());
  bInner.setInsertionPoint(forStmt, forStmt->begin());

  // Doubles the shape with a leading dimension extent of 2.
  auto doubleShape = [&](MemRefType *origMemRefType) -> MemRefType * {
    // Add the leading dimension in the shape for the double buffer.
    ArrayRef<int> shape = origMemRefType->getShape();
    SmallVector<int, 4> shapeSizes(shape.begin(), shape.end());
    shapeSizes.insert(shapeSizes.begin(), 2);

    auto *newMemRefType = bInner.getMemRefType(shapeSizes, bInner.getF32Type());
    return newMemRefType;
  };

  auto *newMemRefType = doubleShape(cast<MemRefType>(oldMemRef->getType()));

  // Create and place the alloc at the top level.
  auto *func = forStmt->getFunction();
  MLFuncBuilder topBuilder(func, func->begin());
  auto *newMemRef = cast<MLValue>(
      topBuilder.create<AllocOp>(forStmt->getLoc(), newMemRefType)
          ->getResult());

  auto d0 = bInner.getAffineDimExpr(0);
  auto *modTwoMap = bInner.getAffineMap(1, 0, {d0 % 2}, {});
  auto ivModTwoOp =
      bInner.create<AffineApplyOp>(forStmt->getLoc(), modTwoMap, forStmt);
  if (!replaceAllMemRefUsesWith(oldMemRef, newMemRef, ivModTwoOp->getResult(0)))
    return false;
  // We don't need ivMod2Op any more - this is cloned by
  // replaceAllMemRefUsesWith wherever the memref replacement happens. Once
  // b/117159533 is addressed, we'll eventually only need to pass
  // ivModTwoOp->getResult(0) to replaceAllMemRefUsesWith.
  cast<OperationStmt>(ivModTwoOp->getOperation())->eraseFromBlock();
  return true;
}

// For testing purposes, this just runs on the first 'for' statement of an
// MLFunction at the top level.
// TODO(bondhugula): upgrade this to scan all the relevant 'for' statements when
// the other TODOs listed inside are dealt with.
PassResult PipelineDataTransfer::runOnMLFunction(MLFunction *f) {
  if (f->empty())
    return PassResult::Success;

  ForStmt *forStmt = nullptr;
  for (auto &stmt : *f) {
    if ((forStmt = dyn_cast<ForStmt>(&stmt))) {
      break;
    }
  }
  if (!forStmt)
    return PassResult::Success;

  unsigned numStmts = forStmt->getStatements().size();

  if (numStmts == 0)
    return PassResult::Success;

  SmallVector<OperationStmt *, 4> dmaStartStmts;
  SmallVector<OperationStmt *, 4> dmaFinishStmts;
  for (auto &stmt : *forStmt) {
    auto *opStmt = dyn_cast<OperationStmt>(&stmt);
    if (!opStmt)
      continue;
    if (opStmt->is<DmaStartOp>()) {
      dmaStartStmts.push_back(opStmt);
    } else if (opStmt->is<DmaWaitOp>()) {
      dmaFinishStmts.push_back(opStmt);
    }
  }

  // TODO(bondhugula,andydavis): match tag memref's (requires memory-based
  // subscript check utilities). Assume for now that start/finish are matched in
  // the order they appear.
  if (dmaStartStmts.size() != dmaFinishStmts.size())
    return PassResult::Failure;

  // Double the buffers for the higher memory space memref's.
  // TODO(bondhugula): assuming we don't have multiple DMA starts for the same
  // memref.
  // TODO(bondhugula): check whether double-buffering is even necessary.
  // TODO(bondhugula): make this work with different layouts: assuming here that
  // the dimension we are adding here for the double buffering is the outermost
  // dimension.
  // Identify memref's to replace by scanning through all DMA start statements.
  // A DMA start statement has two memref's - the one from the higher level of
  // memory hierarchy is the one to double buffer.
  for (auto *dmaStartStmt : dmaStartStmts) {
    MLValue *oldMemRef = cast<MLValue>(dmaStartStmt->getOperand(
        getHigherMemRefPos(dmaStartStmt->getAs<DmaStartOp>())));
    if (!doubleBuffer(oldMemRef, forStmt))
      return PassResult::Failure;
  }

  // Double the buffers for tag memref's.
  for (auto *dmaFinishStmt : dmaFinishStmts) {
    MLValue *oldTagMemRef = cast<MLValue>(
        dmaFinishStmt->getOperand(getTagMemRefPos(*dmaFinishStmt)));
    if (!doubleBuffer(oldTagMemRef, forStmt))
      return PassResult::Failure;
  }

  // Collect all compute ops.
  std::vector<const Statement *> computeOps;
  computeOps.reserve(forStmt->getStatements().size());
  // Store delay for statement for later lookup for AffineApplyOp's.
  DenseMap<const Statement *, unsigned> opDelayMap;
  for (const auto &stmt : *forStmt) {
    auto *opStmt = dyn_cast<OperationStmt>(&stmt);
    if (!opStmt) {
      // All for and if stmt's are treated as pure compute operations.
      // TODO(bondhugula): check whether such statements do not have any DMAs
      // nested within.
      opDelayMap[&stmt] = 1;
    } else if (opStmt->is<DmaStartOp>()) {
      // DMA starts are not shifted.
      opDelayMap[&stmt] = 0;
    } else if (opStmt->is<DmaWaitOp>()) {
      // DMA finish op shifted by one.
      opDelayMap[&stmt] = 1;
    } else if (!opStmt->is<AffineApplyOp>()) {
      // Compute op shifted by one.
      opDelayMap[&stmt] = 1;
      computeOps.push_back(&stmt);
    }
    // Shifts for affine apply op's determined later.
  }

  // Get the ancestor of a 'stmt' that lies in forStmt's block.
  auto getAncestorInForBlock =
      [&](const Statement *stmt, const StmtBlock &block) -> const Statement * {
    // Traverse up the statement hierarchy starting from the owner of operand to
    // find the ancestor statement that resides in the block of 'forStmt'.
    while (stmt != nullptr && stmt->getBlock() != &block) {
      stmt = stmt->getParentStmt();
    }
    return stmt;
  };

  // Determine delays for affine apply op's: look up delay from its consumer op.
  // This code will be thrown away once we have a way to obtain indices through
  // a composed affine_apply op. See TODO(b/117159533). Such a composed
  // affine_apply will be used exclusively by a given memref deferencing op.
  for (const auto &stmt : *forStmt) {
    auto *opStmt = dyn_cast<OperationStmt>(&stmt);
    // Skip statements that aren't affine apply ops.
    if (!opStmt || !opStmt->is<AffineApplyOp>())
      continue;
    // Traverse uses of each result of the affine apply op.
    for (auto *res : opStmt->getResults()) {
      for (auto &use : res->getUses()) {
        auto *ancestorInForBlock =
            getAncestorInForBlock(use.getOwner(), *forStmt);
        assert(ancestorInForBlock &&
               "traversing parent should reach forStmt block");
        auto *opCheck = dyn_cast<OperationStmt>(ancestorInForBlock);
        if (!opCheck || opCheck->is<AffineApplyOp>())
          continue;
        assert(opDelayMap.find(ancestorInForBlock) != opDelayMap.end());
        if (opDelayMap.find(&stmt) != opDelayMap.end()) {
          // This is where we enforce all uses of this affine_apply to have
          // the same shifts - so that we know what shift to use for the
          // affine_apply to preserve semantics.
          assert(opDelayMap[&stmt] == opDelayMap[ancestorInForBlock]);
        } else {
          // Obtain delay from its consumer.
          opDelayMap[&stmt] = opDelayMap[ancestorInForBlock];
        }
      }
    }
  }

  // Get delays stored in map.
  std::vector<uint64_t> delays(forStmt->getStatements().size());
  unsigned s = 0;
  for (const auto &stmt : *forStmt) {
    delays[s++] = opDelayMap[&stmt];
  }

  if (!checkDominancePreservationOnShift(*forStmt, delays)) {
    // Violates SSA dominance.
    return PassResult::Failure;
  }

  if (stmtBodySkew(forStmt, delays))
    return PassResult::Failure;

  return PassResult::Success;
}
