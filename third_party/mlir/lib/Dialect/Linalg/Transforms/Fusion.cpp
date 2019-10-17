//===- Fusion.cpp - Implementation of linalg Fusion -----------------------===//
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
// This file implements the linalg dialect Fusion pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Dominance.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Intrinsics.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-fusion"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace mlir::linalg::intrinsics;

using llvm::dbgs;

/// Implements a simple high-level fusion pass of linalg library operations.
///
/// In each block, linalg ops are processed in reverse textual order.
/// Given a linalg op `O`, fusion occurs by:
///   1. inspecting the linalg ops that write into the views read by `O`. This
///      uses the SSA value of the views and a simple subview/slice analysis to
///      determine producer-consumer dependences;
///   2. greedily fuse the linalg ops that produce subview
///   3. inspect the fused ops and determine whether they have other remaining
///      LinalgOp uses. If not, then erase the original producing linalg op.
///
/// More advanced use cases, analyses as well as profitability heuristics are
/// left for future work.

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");
static llvm::cl::list<unsigned> clTileSizes(
    "linalg-fusion-tile-sizes",
    llvm::cl::desc(
        "Tile sizes by which to tile linalg operations during linalg fusion"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated,
    llvm::cl::cat(clOptionsCategory));

// Return a cloned version of `op` that operates on `loopRanges`, assumed to be
// a subset of the original loop ranges of `op`.
// This is achieved by applying the `loopToOperandRangesMaps` permutation maps
// to the `loopRanges` in order to obtain view ranges.
static LinalgOp cloneWithLoopRanges(OpBuilder &b, Location loc, LinalgOp op,
                                    ArrayRef<SubViewOp::Range> loopRanges,
                                    OperationFolder &state) {
  ScopedContext scope(b, loc);

  auto maps = loopToOperandRangesMaps(op);
  SmallVector<Value *, 8> clonedViews;
  clonedViews.reserve(op.getNumInputsAndOutputs());
  // Iterate over the inputs and outputs in order.
  // Extract the subranges from the linearized ranges.
  SmallVector<Value *, 8> ios(op.getInputsAndOutputs());
  for (auto en : llvm::enumerate(ios)) {
    unsigned idx = en.index();
    auto map = maps[idx];
    LLVM_DEBUG(dbgs() << "map: " << map << "\n");
    Value *view = en.value();
    SmallVector<SubViewOp::Range, 8> viewRanges(map.getNumResults());
    for (auto en2 : llvm::enumerate(map.getResults())) {
      unsigned d = en2.index();
      // loopToOperandRangesMaps are permutations-only.
      unsigned loopPos = en2.value().cast<AffineDimExpr>().getPosition();
      viewRanges[d] = loopRanges[loopPos];
      LLVM_DEBUG(dbgs() << "\ni,j: " << en.index() << ", " << en2.index()
                        << "\t"
                        << "loopPos: " << loopPos << "\t" << viewRanges[d]);
    }
    // TODO(ntv): opportunities for folding/CSE here rather than build new IR.
    clonedViews.push_back(b.create<SubViewOp>(loc, view, viewRanges));
  }
  auto operands = getAssumedNonViewOperands(op);
  clonedViews.append(operands.begin(), operands.end());
  return op.clone(b, loc, clonedViews);
}

struct ViewDimension {
  Value *view;
  unsigned dimension;
};

// Given an `op`, returns the first (`view`, `dimension`) pair that identifies
// the loop range at `loopDepth`. The semantics of the loopToOperandRangesMaps
// guarantees at least one such dimension is found. If multiple candidates exist
// they must agree by construction (i.e. have the same size) and we just return
// the first one.
static ViewDimension getViewDefiningLoopRange(LinalgOp op, unsigned loopDepth) {
  auto maps = loopToOperandRangesMaps(op);
  // Iterate over the inputs and outputs in order.
  // Extract the subranges from the linearized ranges.
  SmallVector<Value *, 8> ios(op.getInputsAndOutputs());
  for (auto en : llvm::enumerate(ios)) {
    unsigned idx = en.index();
    auto map = maps[idx];
    LLVM_DEBUG(dbgs() << "getViewDefiningLoopRange I/O idx: " << idx << "\n");
    LLVM_DEBUG(dbgs() << "getViewDefiningLoopRange map: " << map << "\n");
    Value *view = en.value();
    SmallVector<Value *, 8> viewRanges(map.getNumResults(), nullptr);
    for (auto en2 : llvm::enumerate(map.getResults())) {
      if (loopDepth == en2.value().cast<AffineDimExpr>().getPosition()) {
        LLVM_DEBUG(dbgs() << "getViewDefiningLoopRange loopDepth: " << loopDepth
                          << "\n");
        LLVM_DEBUG(dbgs() << "getViewDefiningLoopRange view: " << *view
                          << "\n");
        return ViewDimension{view, static_cast<unsigned>(en2.index())};
      }
    }
  }
  llvm_unreachable("Expect to be able to extract a view defining loop range");
}

static LinalgOp fuse(Value *producedView, LinalgOp producer, LinalgOp consumer,
                     unsigned consumerIdx, unsigned producerIdx,
                     OperationFolder &state) {
  auto subView = dyn_cast_or_null<SubViewOp>(
      consumer.getInput(consumerIdx)->getDefiningOp());
  auto slice = dyn_cast_or_null<SliceOp>(
      consumer.getInput(consumerIdx)->getDefiningOp());
  assert(subView || slice);
  (void)subView;
  (void)slice;

  // loopToOperandRangesMaps are permutations-only by construction:
  //   we can always identify a data dimension with a (at least one) loop
  //   dimension.
  AffineMap producerMap =
      loopToOperandRangesMaps(producer)[producer.getNumInputs() + producerIdx];
  LLVM_DEBUG(dbgs() << "Producer Idx: " << producerIdx
                    << ", producer map: " << producerMap << "\n");

  unsigned nPar = producer.getNumParallelLoops();
  unsigned nRed = producer.getNumReductionLoops();
  unsigned nWin = producer.getNumWindowLoops();
  SmallVector<SubViewOp::Range, 8> loopRanges(nPar + nRed + nWin);

  // Iterate over dimensions identified by the producer map for `producerIdx`.
  // This defines a subset of the loop ranges that we need to complete later.
  for (auto en : llvm::enumerate(producerMap.getResults())) {
    unsigned posInProducerLoop = en.value().cast<AffineDimExpr>().getPosition();
    loopRanges[posInProducerLoop] = subView.getRange(en.index());
  }

  OpBuilder b(consumer.getOperation());
  auto loc = consumer.getLoc();
  // Iterate over all dimensions. For the dimensions not identified by the
  // producer map for `producerIdx`, we need to explicitly compute the view that
  // defines the loop ranges using the `producer`.
  for (unsigned i = 0, nLoops = loopRanges.size(); i < nLoops; ++i) {
    if (loopRanges[i].min)
      LLVM_DEBUG(llvm::dbgs()
                 << "existing LoopRange: " << loopRanges[i] << "\n");
    else {
      auto viewDim = getViewDefiningLoopRange(producer, i);
      loopRanges[i] =
          SubViewOp::Range{state.create<ConstantIndexOp>(b, loc, 0),
                           dim(viewDim.view, viewDim.dimension),
                           state.create<ConstantIndexOp>(b, loc, 1)};
      LLVM_DEBUG(llvm::dbgs() << "new LoopRange: " << loopRanges[i] << "\n");
    }
  }

  return cloneWithLoopRanges(b, loc, producer, loopRanges, state);
}

// Encode structural fusion safety preconditions.
// Some of these will be lifted in the future with better analysis.
static bool isStructurallyFusableProducer(LinalgOp producer, Value *readView,
                                          LinalgOp consumer) {
  if (producer.getNumOutputs() != 1) {
    LLVM_DEBUG(dbgs() << "\nNot structurally fusable (multi-output)");
    return false;
  }
  // Must be a subview or a slice to guarantee there are loops we can fuse into.
  auto subView = dyn_cast_or_null<SubViewOp>(readView->getDefiningOp());
  auto slice = dyn_cast_or_null<SliceOp>(readView->getDefiningOp());
  if (!subView && !slice) {
    LLVM_DEBUG(dbgs() << "\nNot structurally fusable (not a subview or slice)");
    return false;
  }
  // Only fuse when the producer block dominates.
  DominanceInfo dom(producer.getOperation());
  if (!dom.dominates(producer.getOperation()->getBlock(),
                     consumer.getOperation()->getBlock())) {
    LLVM_DEBUG(
        dbgs()
        << "\nNot structurally fusable (producer block does not dominate)");
    return false;
  }
  return true;
}

// Only consider RAW atm.
Optional<FusionInfo> mlir::linalg::fuseProducerOf(LinalgOp consumer,
                                                  unsigned consumerIdx,
                                                  LinalgDependenceGraph &graph,
                                                  OperationFolder &state) {
  LLVM_DEBUG(dbgs() << "\nStart examining consumer: "
                    << *consumer.getOperation());
  for (auto dependence : graph.getDependencesInto(
           consumer, LinalgDependenceGraph::DependenceType::RAW)) {
    LLVM_DEBUG(dbgs() << "\n***Consider producer:\t"
                      << *dependence.dependentOpView.op << "\n");
    auto producer = cast<LinalgOp>(dependence.dependentOpView.op);

    // Check that the dependence is indeed on the input `consumerIdx` view.
    auto *readView = dependence.indexingView;
    if (consumer.getInput(consumerIdx) != readView)
      continue;

    // Consumer consumes this view, `isStructurallyFusableProducer` also checks
    // whether it is a strict subview of the producer view.
    auto *producedView = dependence.dependentOpView.view;
    auto producerIdx = producer.getIndexOfOutput(producedView).getValue();
    // `consumerIdx` and `producerIdx` exist by construction.
    LLVM_DEBUG(dbgs() << "\nRAW producer: " << *producer.getOperation()
                      << " view: " << *producedView
                      << " output index: " << producerIdx);

    // Make some simple structural checks that alleviate the need for more
    // complex analyses.
    if (!isStructurallyFusableProducer(producer, readView, consumer)) {
      LLVM_DEBUG(dbgs() << "\n***Not fusable:\t" << *producer.getOperation());
      continue;
    }

    // Check for fusion-preventing write that would violate dependences.
    // `view` is a producer write that cannot bypass any other write or read.
    if (!graph.findCoveringDependences(producer, consumer).empty())
      continue;

    // Fuse `producer` just before `consumer`.
    OpBuilder builder(consumer.getOperation());
    ScopedContext scope(builder, consumer.getLoc());
    LLVM_DEBUG(dbgs() << "Fuse into consumer: " << *consumer << "\n");
    auto fusedProducer =
        fuse(producedView, producer, consumer, consumerIdx, producerIdx, state);

    return FusionInfo{producer, fusedProducer};
  }
  return llvm::None;
}

static void fuseLinalgOpsGreedily(FuncOp f) {
  LLVM_DEBUG(f.print(dbgs() << "\nBefore linalg-fusion: \n"));

  OperationFolder state(f.getContext());
  DenseSet<Operation *> eraseSet;

  // Save original Linalg ops, we only want to make a pass over those.
  SmallVector<Operation *, 8> linalgOps;
  f.walk([&](LinalgOp op) { linalgOps.push_back(op); });

  Aliases aliases;
  LinalgDependenceGraph G(aliases, linalgOps);
  for (auto *op : llvm::reverse(linalgOps)) {
    for (unsigned consumerIdx = 0, e = LinalgOp(op).getNumInputs();
         consumerIdx < e; ++consumerIdx) {
      if (auto fusionInfo = fuseProducerOf(op, consumerIdx, G, state))
        eraseSet.insert(fusionInfo->originalProducer.getOperation());
    }
  }

  // The `fuseProducerOf` function performs structural checks and in particular
  // that no covering read or write exist between the consumer and the producer.
  // As a consequence, the only fusions that may occur preserve subsequent
  // dependences and are guaranteed by construction to produce the whole view.
  // We may thus erase the producer once it is fused.
  for (auto *e : eraseSet)
    e->erase();
  LLVM_DEBUG(f.print(dbgs() << "\nAfter linalg-fusion: \n"));
}

namespace {
struct LinalgFusionPass : public FunctionPass<LinalgFusionPass> {
  void runOnFunction() override { fuseLinalgOpsGreedily(getFunction()); }
};
} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::linalg::createLinalgFusionPass() {
  return std::make_unique<LinalgFusionPass>();
}

static PassRegistration<LinalgFusionPass>
    pass("linalg-fusion", "Fuse operations in the linalg dialect");
