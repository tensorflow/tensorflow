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

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pipeline-data-transfer"

using namespace mlir;

namespace {

struct PipelineDataTransfer : public FunctionPass,
                              StmtWalker<PipelineDataTransfer> {
  PipelineDataTransfer() : FunctionPass(&PipelineDataTransfer::passID) {}
  PassResult runOnMLFunction(MLFunction *f) override;
  PassResult runOnForStmt(ForStmt *forStmt);

  // Collect all 'for' statements.
  void visitForStmt(ForStmt *forStmt) { forStmts.push_back(forStmt); }
  std::vector<ForStmt *> forStmts;

  static char passID;
};

} // end anonymous namespace

char PipelineDataTransfer::passID = 0;

/// Creates a pass to pipeline explicit movement of data across levels of the
/// memory hierarchy.
FunctionPass *mlir::createPipelineDataTransferPass() {
  return new PipelineDataTransfer();
}

// Returns the position of the tag memref operand given a DMA statement.
// Temporary utility: will be replaced when DmaStart/DmaFinish abstract op's are
// added.  TODO(b/117228571)
static unsigned getTagMemRefPos(const OperationStmt &dmaStmt) {
  assert(dmaStmt.isa<DmaStartOp>() || dmaStmt.isa<DmaWaitOp>());
  if (dmaStmt.isa<DmaStartOp>()) {
    // Second to last operand.
    return dmaStmt.getNumOperands() - 2;
  }
  // First operand for a dma finish statement.
  return 0;
}

/// Doubles the buffer of the supplied memref on the specified 'for' statement
/// by adding a leading dimension of size two to the memref. Replaces all uses
/// of the old memref by the new one while indexing the newly added dimension by
/// the loop IV of the specified 'for' statement modulo 2. Returns false if such
/// a replacement cannot be performed.
static bool doubleBuffer(MLValue *oldMemRef, ForStmt *forStmt) {
  auto *forBody = forStmt->getBody();
  MLFuncBuilder bInner(forBody, forBody->begin());
  bInner.setInsertionPoint(forBody, forBody->begin());

  // Doubles the shape with a leading dimension extent of 2.
  auto doubleShape = [&](MemRefType oldMemRefType) -> MemRefType {
    // Add the leading dimension in the shape for the double buffer.
    ArrayRef<int> shape = oldMemRefType.getShape();
    SmallVector<int, 4> shapeSizes(shape.begin(), shape.end());
    shapeSizes.insert(shapeSizes.begin(), 2);

    auto newMemRefType =
        bInner.getMemRefType(shapeSizes, oldMemRefType.getElementType(), {},
                             oldMemRefType.getMemorySpace());
    return newMemRefType;
  };

  auto oldMemRefType = oldMemRef->getType().cast<MemRefType>();
  auto newMemRefType = doubleShape(oldMemRefType);

  // Put together alloc operands for the dynamic dimensions of the memref.
  MLFuncBuilder bOuter(forStmt);
  SmallVector<SSAValue *, 4> allocOperands;
  unsigned dynamicDimCount = 0;
  for (auto dimSize : oldMemRefType.getShape()) {
    if (dimSize == -1)
      allocOperands.push_back(bOuter.create<DimOp>(forStmt->getLoc(), oldMemRef,
                                                   dynamicDimCount++));
  }

  // Create and place the alloc right before the 'for' statement.
  // TODO(mlir-team): we are assuming scoped allocation here, and aren't
  // inserting a dealloc -- this isn't the right thing.
  SSAValue *newMemRef =
      bOuter.create<AllocOp>(forStmt->getLoc(), newMemRefType, allocOperands);

  // Create 'iv mod 2' value to index the leading dimension.
  auto d0 = bInner.getAffineDimExpr(0);
  auto modTwoMap =
      bInner.getAffineMap(/*dimCount=*/1, /*symbolCount=*/0, {d0 % 2}, {});
  auto ivModTwoOp =
      bInner.create<AffineApplyOp>(forStmt->getLoc(), modTwoMap, forStmt);

  // replaceAllMemRefUsesWith will always succeed unless the forStmt body has
  // non-deferencing uses of the memref.
  if (!replaceAllMemRefUsesWith(oldMemRef, cast<MLValue>(newMemRef),
                                ivModTwoOp->getResult(0), AffineMap::Null(), {},
                                &*forStmt->getBody()->begin())) {
    LLVM_DEBUG(llvm::dbgs()
                   << "memref replacement for double buffering failed\n";);
    ivModTwoOp->getOperation()->erase();
    return false;
  }
  return true;
}

/// Returns success if the IR is in a valid state.
PassResult PipelineDataTransfer::runOnMLFunction(MLFunction *f) {
  // Do a post order walk so that inner loop DMAs are processed first. This is
  // necessary since 'for' statements nested within would otherwise become
  // invalid (erased) when the outer loop is pipelined (the pipelined one gets
  // deleted and replaced by a prologue, a new steady-state loop and an
  // epilogue).
  forStmts.clear();
  walkPostOrder(f);
  bool ret = false;
  for (auto *forStmt : forStmts) {
    ret = ret | runOnForStmt(forStmt);
  }
  return ret ? failure() : success();
}

// Check if tags of the dma start op and dma wait op match.
static bool checkTagMatch(OpPointer<DmaStartOp> startOp,
                          OpPointer<DmaWaitOp> waitOp) {
  if (startOp->getTagMemRef() != waitOp->getTagMemRef())
    return false;
  auto startIndices = startOp->getTagIndices();
  auto waitIndices = waitOp->getTagIndices();
  // Both of these have the same number of indices since they correspond to the
  // same tag memref.
  for (auto it = startIndices.begin(), wIt = waitIndices.begin(),
            e = startIndices.end();
       it != e; ++it, ++wIt) {
    // Keep it simple for now, just checking if indices match.
    // TODO(mlir-team): this would in general need to check if there is no
    // intervening write writing to the same tag location, i.e., memory last
    // write/data flow analysis. This is however sufficient/powerful enough for
    // now since the DMA generation pass or the input for it will always have
    // start/wait with matching tags (same SSA operand indices).
    if (*it != *wIt)
      return false;
  }
  return true;
}

// Identify matching DMA start/finish statements to overlap computation with.
static void findMatchingStartFinishStmts(
    ForStmt *forStmt,
    SmallVectorImpl<std::pair<OperationStmt *, OperationStmt *>>
        &startWaitPairs) {

  // Collect outgoing DMA statements - needed to check for dependences below.
  SmallVector<OpPointer<DmaStartOp>, 4> outgoingDmaOps;
  for (auto &stmt : *forStmt->getBody()) {
    auto *opStmt = dyn_cast<OperationStmt>(&stmt);
    if (!opStmt)
      continue;
    OpPointer<DmaStartOp> dmaStartOp;
    if ((dmaStartOp = opStmt->dyn_cast<DmaStartOp>()) &&
        dmaStartOp->isSrcMemorySpaceFaster())
      outgoingDmaOps.push_back(dmaStartOp);
  }

  SmallVector<OperationStmt *, 4> dmaStartStmts, dmaFinishStmts;
  for (auto &stmt : *forStmt->getBody()) {
    auto *opStmt = dyn_cast<OperationStmt>(&stmt);
    if (!opStmt)
      continue;
    // Collect DMA finish statements.
    if (opStmt->isa<DmaWaitOp>()) {
      dmaFinishStmts.push_back(opStmt);
      continue;
    }
    OpPointer<DmaStartOp> dmaStartOp;
    if (!(dmaStartOp = opStmt->dyn_cast<DmaStartOp>()))
      continue;
    // Only DMAs incoming into higher memory spaces are pipelined for now.
    // TODO(bondhugula): handle outgoing DMA pipelining.
    if (!dmaStartOp->isDestMemorySpaceFaster())
      continue;

    // Check for dependence with outgoing DMAs. Doing this conservatively.
    // TODO(andydavis,bondhugula): use the dependence analysis to check for
    // dependences between an incoming and outgoing DMA in the same iteration.
    auto it = outgoingDmaOps.begin();
    for (; it != outgoingDmaOps.end(); ++it) {
      if ((*it)->getDstMemRef() == dmaStartOp->getSrcMemRef())
        break;
    }
    if (it != outgoingDmaOps.end())
      continue;

    // We only double buffer if the buffer is not live out of loop.
    const MLValue *memref =
        cast<MLValue>(dmaStartOp->getOperand(dmaStartOp->getFasterMemPos()));
    bool escapingUses = false;
    for (const auto &use : memref->getUses()) {
      if (!dominates(*forStmt->getBody()->begin(), *use.getOwner())) {
        LLVM_DEBUG(llvm::dbgs()
                       << "can't pipeline: buffer is live out of loop\n";);
        escapingUses = true;
        break;
      }
    }
    if (!escapingUses)
      dmaStartStmts.push_back(opStmt);
  }

  // For each start statement, we look for a matching finish statement.
  for (auto *dmaStartStmt : dmaStartStmts) {
    for (auto *dmaFinishStmt : dmaFinishStmts) {
      if (checkTagMatch(dmaStartStmt->cast<DmaStartOp>(),
                        dmaFinishStmt->cast<DmaWaitOp>())) {
        startWaitPairs.push_back({dmaStartStmt, dmaFinishStmt});
        break;
      }
    }
  }
}

/// Overlap DMA transfers with computation in this loop. If successful,
/// 'forStmt' is deleted, and a prologue, a new pipelined loop, and epilogue are
/// inserted right before where it was.
PassResult PipelineDataTransfer::runOnForStmt(ForStmt *forStmt) {
  auto mayBeConstTripCount = getConstantTripCount(*forStmt);
  if (!mayBeConstTripCount.hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "unknown trip count loop\n");
    return success();
  }

  SmallVector<std::pair<OperationStmt *, OperationStmt *>, 4> startWaitPairs;
  findMatchingStartFinishStmts(forStmt, startWaitPairs);

  if (startWaitPairs.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No dma start/finish pairs\n";);
    return success();
  }

  // Double the buffers for the higher memory space memref's.
  // Identify memref's to replace by scanning through all DMA start statements.
  // A DMA start statement has two memref's - the one from the higher level of
  // memory hierarchy is the one to double buffer.
  // TODO(bondhugula): check whether double-buffering is even necessary.
  // TODO(bondhugula): make this work with different layouts: assuming here that
  // the dimension we are adding here for the double buffering is the outermost
  // dimension.
  for (auto &pair : startWaitPairs) {
    auto *dmaStartStmt = pair.first;
    MLValue *oldMemRef = cast<MLValue>(dmaStartStmt->getOperand(
        dmaStartStmt->cast<DmaStartOp>()->getFasterMemPos()));
    if (!doubleBuffer(oldMemRef, forStmt)) {
      // Normally, double buffering should not fail because we already checked
      // that there are no uses outside.
      LLVM_DEBUG(llvm::dbgs() << "double buffering failed for: \n";);
      LLVM_DEBUG(dmaStartStmt->dump());
      // IR still in a valid state.
      return success();
    }
    // If the old memref has no more uses, remove its 'dead' alloc if it was
    // alloc'ed. (note: DMA buffers are rarely function live-in; but a 'dim'
    // operation could have been used on it if it was dynamically shaped in
    // order to create the double buffer above)
    if (oldMemRef->use_empty())
      if (auto *allocStmt = oldMemRef->getDefiningStmt())
        allocStmt->erase();
  }

  // Double the buffers for tag memrefs.
  for (auto &pair : startWaitPairs) {
    auto *dmaFinishStmt = pair.second;
    MLValue *oldTagMemRef = cast<MLValue>(
        dmaFinishStmt->getOperand(getTagMemRefPos(*dmaFinishStmt)));
    if (!doubleBuffer(oldTagMemRef, forStmt)) {
      LLVM_DEBUG(llvm::dbgs() << "tag double buffering failed\n";);
      return success();
    }
    // If the old tag has no more uses, remove its 'dead' alloc if it was
    // alloc'ed.
    if (oldTagMemRef->use_empty())
      if (auto *allocStmt = oldTagMemRef->getDefiningStmt())
        allocStmt->erase();
  }

  // Double buffering would have invalidated all the old DMA start/wait stmts.
  startWaitPairs.clear();
  findMatchingStartFinishStmts(forStmt, startWaitPairs);

  // Store shift for statement for later lookup for AffineApplyOp's.
  DenseMap<const Statement *, unsigned> stmtShiftMap;
  for (auto &pair : startWaitPairs) {
    auto *dmaStartStmt = pair.first;
    assert(dmaStartStmt->isa<DmaStartOp>());
    stmtShiftMap[dmaStartStmt] = 0;
    // Set shifts for DMA start stmt's affine operand computation slices to 0.
    if (auto *slice = mlir::createAffineComputationSlice(dmaStartStmt)) {
      stmtShiftMap[slice] = 0;
    } else {
      // If a slice wasn't created, the reachable affine_apply op's from its
      // operands are the ones that go with it.
      SmallVector<OperationStmt *, 4> affineApplyStmts;
      SmallVector<MLValue *, 4> operands(dmaStartStmt->getOperands());
      getReachableAffineApplyOps(operands, affineApplyStmts);
      for (const auto *stmt : affineApplyStmts) {
        stmtShiftMap[stmt] = 0;
      }
    }
  }
  // Everything else (including compute ops and dma finish) are shifted by one.
  for (const auto &stmt : *forStmt->getBody()) {
    if (stmtShiftMap.find(&stmt) == stmtShiftMap.end()) {
      stmtShiftMap[&stmt] = 1;
    }
  }

  // Get shifts stored in map.
  std::vector<uint64_t> shifts(forStmt->getBody()->getStatements().size());
  unsigned s = 0;
  for (auto &stmt : *forStmt->getBody()) {
    assert(stmtShiftMap.find(&stmt) != stmtShiftMap.end());
    shifts[s++] = stmtShiftMap[&stmt];
    LLVM_DEBUG(
        // Tagging statements with shifts for debugging purposes.
        if (auto *opStmt = dyn_cast<OperationStmt>(&stmt)) {
          MLFuncBuilder b(opStmt);
          opStmt->setAttr(b.getIdentifier("shift"),
                          b.getIntegerAttr(shifts[s - 1]));
        });
  }

  if (!isStmtwiseShiftValid(*forStmt, shifts)) {
    // Violates dependences.
    LLVM_DEBUG(llvm::dbgs() << "Shifts invalid - unexpected\n";);
    return success();
  }

  if (stmtBodySkew(forStmt, shifts)) {
    LLVM_DEBUG(llvm::dbgs() << "stmt body skewing failed - unexpected\n";);
    return success();
  }

  return success();
}

static PassRegistration<PipelineDataTransfer> pass(
    "pipeline-data-transfer",
    "Pipeline non-blocking data transfers between explicitly managed levels of "
    "the memory hierarchy");
