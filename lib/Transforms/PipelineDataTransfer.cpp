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
#include "mlir/IR/InstVisitor.h"
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
                              InstWalker<PipelineDataTransfer> {
  PipelineDataTransfer() : FunctionPass(&PipelineDataTransfer::passID) {}
  PassResult runOnFunction(Function *f) override;
  PassResult runOnForInst(ForInst *forInst);

  // Collect all 'for' instructions.
  void visitForInst(ForInst *forInst) { forInsts.push_back(forInst); }
  std::vector<ForInst *> forInsts;

  static char passID;
};

} // end anonymous namespace

char PipelineDataTransfer::passID = 0;

/// Creates a pass to pipeline explicit movement of data across levels of the
/// memory hierarchy.
FunctionPass *mlir::createPipelineDataTransferPass() {
  return new PipelineDataTransfer();
}

// Returns the position of the tag memref operand given a DMA instruction.
// Temporary utility: will be replaced when DmaStart/DmaFinish abstract op's are
// added.  TODO(b/117228571)
static unsigned getTagMemRefPos(const OperationInst &dmaInst) {
  assert(dmaInst.isa<DmaStartOp>() || dmaInst.isa<DmaWaitOp>());
  if (dmaInst.isa<DmaStartOp>()) {
    // Second to last operand.
    return dmaInst.getNumOperands() - 2;
  }
  // First operand for a dma finish instruction.
  return 0;
}

/// Doubles the buffer of the supplied memref on the specified 'for' instruction
/// by adding a leading dimension of size two to the memref. Replaces all uses
/// of the old memref by the new one while indexing the newly added dimension by
/// the loop IV of the specified 'for' instruction modulo 2. Returns false if
/// such a replacement cannot be performed.
static bool doubleBuffer(Value *oldMemRef, ForInst *forInst) {
  auto *forBody = forInst->getBody();
  FuncBuilder bInner(forBody, forBody->begin());
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
  FuncBuilder bOuter(forInst);
  SmallVector<Value *, 4> allocOperands;
  unsigned dynamicDimCount = 0;
  for (auto dimSize : oldMemRefType.getShape()) {
    if (dimSize == -1)
      allocOperands.push_back(bOuter.create<DimOp>(forInst->getLoc(), oldMemRef,
                                                   dynamicDimCount++));
  }

  // Create and place the alloc right before the 'for' instruction.
  // TODO(mlir-team): we are assuming scoped allocation here, and aren't
  // inserting a dealloc -- this isn't the right thing.
  Value *newMemRef =
      bOuter.create<AllocOp>(forInst->getLoc(), newMemRefType, allocOperands);

  // Create 'iv mod 2' value to index the leading dimension.
  auto d0 = bInner.getAffineDimExpr(0);
  auto modTwoMap =
      bInner.getAffineMap(/*dimCount=*/1, /*symbolCount=*/0, {d0 % 2}, {});
  auto ivModTwoOp =
      bInner.create<AffineApplyOp>(forInst->getLoc(), modTwoMap, forInst);

  // replaceAllMemRefUsesWith will always succeed unless the forInst body has
  // non-deferencing uses of the memref.
  if (!replaceAllMemRefUsesWith(oldMemRef, newMemRef, ivModTwoOp->getResult(0),
                                AffineMap::Null(), {},
                                &*forInst->getBody()->begin())) {
    LLVM_DEBUG(llvm::dbgs()
                   << "memref replacement for double buffering failed\n";);
    ivModTwoOp->getInstruction()->erase();
    return false;
  }
  return true;
}

/// Returns success if the IR is in a valid state.
PassResult PipelineDataTransfer::runOnFunction(Function *f) {
  // Do a post order walk so that inner loop DMAs are processed first. This is
  // necessary since 'for' instructions nested within would otherwise become
  // invalid (erased) when the outer loop is pipelined (the pipelined one gets
  // deleted and replaced by a prologue, a new steady-state loop and an
  // epilogue).
  forInsts.clear();
  walkPostOrder(f);
  bool ret = false;
  for (auto *forInst : forInsts) {
    ret = ret | runOnForInst(forInst);
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

// Identify matching DMA start/finish instructions to overlap computation with.
static void findMatchingStartFinishInsts(
    ForInst *forInst,
    SmallVectorImpl<std::pair<OperationInst *, OperationInst *>>
        &startWaitPairs) {

  // Collect outgoing DMA instructions - needed to check for dependences below.
  SmallVector<OpPointer<DmaStartOp>, 4> outgoingDmaOps;
  for (auto &inst : *forInst->getBody()) {
    auto *opInst = dyn_cast<OperationInst>(&inst);
    if (!opInst)
      continue;
    OpPointer<DmaStartOp> dmaStartOp;
    if ((dmaStartOp = opInst->dyn_cast<DmaStartOp>()) &&
        dmaStartOp->isSrcMemorySpaceFaster())
      outgoingDmaOps.push_back(dmaStartOp);
  }

  SmallVector<OperationInst *, 4> dmaStartInsts, dmaFinishInsts;
  for (auto &inst : *forInst->getBody()) {
    auto *opInst = dyn_cast<OperationInst>(&inst);
    if (!opInst)
      continue;
    // Collect DMA finish instructions.
    if (opInst->isa<DmaWaitOp>()) {
      dmaFinishInsts.push_back(opInst);
      continue;
    }
    OpPointer<DmaStartOp> dmaStartOp;
    if (!(dmaStartOp = opInst->dyn_cast<DmaStartOp>()))
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
    auto *memref = dmaStartOp->getOperand(dmaStartOp->getFasterMemPos());
    bool escapingUses = false;
    for (const auto &use : memref->getUses()) {
      if (!forInst->getBody()->findAncestorInstInBlock(*use.getOwner())) {
        LLVM_DEBUG(llvm::dbgs()
                       << "can't pipeline: buffer is live out of loop\n";);
        escapingUses = true;
        break;
      }
    }
    if (!escapingUses)
      dmaStartInsts.push_back(opInst);
  }

  // For each start instruction, we look for a matching finish instruction.
  for (auto *dmaStartInst : dmaStartInsts) {
    for (auto *dmaFinishInst : dmaFinishInsts) {
      if (checkTagMatch(dmaStartInst->cast<DmaStartOp>(),
                        dmaFinishInst->cast<DmaWaitOp>())) {
        startWaitPairs.push_back({dmaStartInst, dmaFinishInst});
        break;
      }
    }
  }
}

/// Overlap DMA transfers with computation in this loop. If successful,
/// 'forInst' is deleted, and a prologue, a new pipelined loop, and epilogue are
/// inserted right before where it was.
PassResult PipelineDataTransfer::runOnForInst(ForInst *forInst) {
  auto mayBeConstTripCount = getConstantTripCount(*forInst);
  if (!mayBeConstTripCount.hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "unknown trip count loop\n");
    return success();
  }

  SmallVector<std::pair<OperationInst *, OperationInst *>, 4> startWaitPairs;
  findMatchingStartFinishInsts(forInst, startWaitPairs);

  if (startWaitPairs.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No dma start/finish pairs\n";);
    return success();
  }

  // Double the buffers for the higher memory space memref's.
  // Identify memref's to replace by scanning through all DMA start
  // instructions. A DMA start instruction has two memref's - the one from the
  // higher level of memory hierarchy is the one to double buffer.
  // TODO(bondhugula): check whether double-buffering is even necessary.
  // TODO(bondhugula): make this work with different layouts: assuming here that
  // the dimension we are adding here for the double buffering is the outermost
  // dimension.
  for (auto &pair : startWaitPairs) {
    auto *dmaStartInst = pair.first;
    Value *oldMemRef = dmaStartInst->getOperand(
        dmaStartInst->cast<DmaStartOp>()->getFasterMemPos());
    if (!doubleBuffer(oldMemRef, forInst)) {
      // Normally, double buffering should not fail because we already checked
      // that there are no uses outside.
      LLVM_DEBUG(llvm::dbgs() << "double buffering failed for: \n";);
      LLVM_DEBUG(dmaStartInst->dump());
      // IR still in a valid state.
      return success();
    }
    // If the old memref has no more uses, remove its 'dead' alloc if it was
    // alloc'ed. (note: DMA buffers are rarely function live-in; but a 'dim'
    // operation could have been used on it if it was dynamically shaped in
    // order to create the double buffer above)
    if (oldMemRef->use_empty())
      if (auto *allocInst = oldMemRef->getDefiningInst())
        allocInst->erase();
  }

  // Double the buffers for tag memrefs.
  for (auto &pair : startWaitPairs) {
    auto *dmaFinishInst = pair.second;
    Value *oldTagMemRef =
        dmaFinishInst->getOperand(getTagMemRefPos(*dmaFinishInst));
    if (!doubleBuffer(oldTagMemRef, forInst)) {
      LLVM_DEBUG(llvm::dbgs() << "tag double buffering failed\n";);
      return success();
    }
    // If the old tag has no more uses, remove its 'dead' alloc if it was
    // alloc'ed.
    if (oldTagMemRef->use_empty())
      if (auto *allocInst = oldTagMemRef->getDefiningInst())
        allocInst->erase();
  }

  // Double buffering would have invalidated all the old DMA start/wait insts.
  startWaitPairs.clear();
  findMatchingStartFinishInsts(forInst, startWaitPairs);

  // Store shift for instruction for later lookup for AffineApplyOp's.
  DenseMap<const Instruction *, unsigned> instShiftMap;
  for (auto &pair : startWaitPairs) {
    auto *dmaStartInst = pair.first;
    assert(dmaStartInst->isa<DmaStartOp>());
    instShiftMap[dmaStartInst] = 0;
    // Set shifts for DMA start inst's affine operand computation slices to 0.
    if (auto *slice = mlir::createAffineComputationSlice(dmaStartInst)) {
      instShiftMap[slice] = 0;
    } else {
      // If a slice wasn't created, the reachable affine_apply op's from its
      // operands are the ones that go with it.
      SmallVector<OperationInst *, 4> affineApplyInsts;
      SmallVector<Value *, 4> operands(dmaStartInst->getOperands());
      getReachableAffineApplyOps(operands, affineApplyInsts);
      for (const auto *inst : affineApplyInsts) {
        instShiftMap[inst] = 0;
      }
    }
  }
  // Everything else (including compute ops and dma finish) are shifted by one.
  for (const auto &inst : *forInst->getBody()) {
    if (instShiftMap.find(&inst) == instShiftMap.end()) {
      instShiftMap[&inst] = 1;
    }
  }

  // Get shifts stored in map.
  std::vector<uint64_t> shifts(forInst->getBody()->getInstructions().size());
  unsigned s = 0;
  for (auto &inst : *forInst->getBody()) {
    assert(instShiftMap.find(&inst) != instShiftMap.end());
    shifts[s++] = instShiftMap[&inst];
    LLVM_DEBUG(
        // Tagging instructions with shifts for debugging purposes.
        if (auto *opInst = dyn_cast<OperationInst>(&inst)) {
          FuncBuilder b(opInst);
          opInst->setAttr(b.getIdentifier("shift"),
                          b.getI64IntegerAttr(shifts[s - 1]));
        });
  }

  if (!isInstwiseShiftValid(*forInst, shifts)) {
    // Violates dependences.
    LLVM_DEBUG(llvm::dbgs() << "Shifts invalid - unexpected\n";);
    return success();
  }

  if (instBodySkew(forInst, shifts)) {
    LLVM_DEBUG(llvm::dbgs() << "inst body skewing failed - unexpected\n";);
    return success();
  }

  return success();
}

static PassRegistration<PipelineDataTransfer> pass(
    "pipeline-data-transfer",
    "Pipeline non-blocking data transfers between explicitly managed levels of "
    "the memory hierarchy");
