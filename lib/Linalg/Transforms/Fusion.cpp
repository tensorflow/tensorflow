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

#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Linalg/IR/LinalgOps.h"
#include "mlir/Linalg/IR/LinalgTypes.h"
#include "mlir/Linalg/Passes.h"
#include "mlir/Linalg/Utils/Intrinsics.h"
#include "mlir/Linalg/Utils/Utils.h"
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
/// Given a linalg op, fusion occurs by:
///   1. tiling the op by a given multi-dimensional tile size;
///   2. inspecting the linalg ops that write into the views read by the op in
///      step 1. This uses the SSA value of the views to determine producer-
///      consumer dependences: only identical SSA views are considered for
///      fusion at this point;
///   3. greedily fuse the producing linalg ops into the consuming loop tiles;
///   4. inspect the fused ops and determine whether they have other remaining
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
      LLVM_DEBUG(dbgs() << "i,j: " << en.index() << ", " << en2.index() << "\t"
                        << "loopPos: " << loopPos << "\t" << viewRanges[d]);
    }
    // TODO(ntv) opportunities for folding/CSE here rather than build new IR.
    clonedViews.push_back(b.create<SubViewOp>(loc, view, viewRanges));
  }
  return op.create(b, loc, clonedViews);
}

struct ViewDimension {
  Value *view;
  unsigned dimension;
};

static ViewDimension getViewDefiningLoopRange(LinalgOp op, unsigned loopDepth) {
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
    SmallVector<Value *, 8> viewRanges(map.getNumResults(), nullptr);
    for (auto en2 : llvm::enumerate(map.getResults())) {
      if (loopDepth == en2.value().cast<AffineDimExpr>().getPosition())
        return ViewDimension{view, static_cast<unsigned>(en2.index())};
    }
  }
  llvm_unreachable("Expect to be able to extract a view defining loop range");
}

static Optional<LinalgOp> fuse(Value *producedView, LinalgOp producer,
                               LinalgOp consumer, LinalgOp tiledConsumer,
                               OperationFolder &state) {
  auto maybeConsumerIdx = consumer.getIndexOfInput(producedView);
  if (!maybeConsumerIdx.hasValue())
    return llvm::None;
  unsigned consumerIdx = maybeConsumerIdx.getValue();

  auto maybeProducerIdx = producer.getIndexOfOutput(producedView);
  if (!maybeProducerIdx.hasValue())
    return llvm::None;
  unsigned producerIdx = maybeProducerIdx.getValue();

  auto sliceOp = dyn_cast_or_null<SubViewOp>(
      tiledConsumer.getInput(consumerIdx)->getDefiningOp());
  // If we don't have a slice, this also means we don't have loops and the
  // producer cannot be fused at this level.
  if (!sliceOp)
    return llvm::None;

  AffineMap producerMap =
      loopToOperandRangesMaps(producer)[producer.getNumInputs() + producerIdx];
  LLVM_DEBUG(dbgs() << "Consumer Idx: " << consumerIdx << "\tmap: "
                    << loopToOperandRangesMaps(consumer)[consumerIdx] << "\n");
  LLVM_DEBUG(dbgs() << "Producer Idx: " << producerIdx
                    << "\tmap: " << producerMap << "\n");

  unsigned nPar = producer.getNumParallelLoops();
  unsigned nRed = producer.getNumReductionLoops();
  unsigned nWin = producer.getNumWindowLoops();
  SmallVector<SubViewOp::Range, 8> loopRanges(nPar + nRed + nWin);
  DenseSet<unsigned> fromSlice;
  for (auto en : llvm::enumerate(producerMap.getResults())) {
    // loopToOperandRangesMaps are permutations-only.
    unsigned posInProducerLoop = en.value().cast<AffineDimExpr>().getPosition();
    loopRanges[posInProducerLoop] = sliceOp.getRange(en.index());
    fromSlice.insert(posInProducerLoop);
  }

  OpBuilder b(tiledConsumer.getOperation());
  auto loc = tiledConsumer.getLoc();
  for (unsigned i = 0; i < loopRanges.size(); ++i) {
    if (fromSlice.count(i))
      LLVM_DEBUG(llvm::dbgs() << "LR: " << loopRanges[i] << "\n");
    else {
      auto viewDim = getViewDefiningLoopRange(producer, i);
      loopRanges[i] = SubViewOp::Range{
          state.create<ConstantIndexOp>(b, loc, 0),
          linalg::intrinsics::dim(viewDim.view, viewDim.dimension),
          state.create<ConstantIndexOp>(b, loc, 1)};
      LLVM_DEBUG(llvm::dbgs() << "new LR: " << loopRanges[i] << "\n");
    }
  }

  return cloneWithLoopRanges(b, loc, producer, loopRanges, state);
}

// Encode structural fusion safety preconditions.
// Some of these will be lifted in the future with better analysis.
static bool isStructurallyFusableProducer(LinalgOp producer, Value *readView,
                                          LinalgOp consumer) {
  // If a producer has multiple outputs, the analysis needs to take the tiling
  // of other outputs into account.
  if (producer.getNumOutputs() != 1)
    return false;
  // Until subview analysis is available, same SSA value is required for fusion.
  if (producer.getOutput(0) != readView)
    return false;
  // No control-flow divergence supported. Only straightline op fusion allowed.
  // TODO(ntv) allow fusion when a dominance relation exists.
  if (producer.getOperation()->getBlock() !=
      consumer.getOperation()->getBlock())
    return false;
  return true;
}

static void fuseLinalgOps(Function &f, ArrayRef<int64_t> tileSizes) {
  OperationFolder state(&f);
  DenseSet<Operation *> eraseSet;

  // 1. Record the linalg ops so we can traverse them in reverse order.
  SmallVector<Operation *, 8> linalgOps;
  f.walk<LinalgOp>(
      [&](LinalgOp op) { linalgOps.push_back(op.getOperation()); });

  // 2. Setup the dependences graph, aliases are populated lazily.
  Aliases aliases;
  LinalgDependenceGraph G(aliases, linalgOps);

  // 2. For each original linalg op (in reverse order to allow chained
  // fusions).
  for (auto *op : llvm::reverse(linalgOps)) {
    auto consumer = cast<LinalgOp>(op);
    LLVM_DEBUG(dbgs() << "\n******\nStart processing:\t" << *op);
    // 3. If marked for erasure, it has already been fused. Skip fusing op.
    if (eraseSet.count(op) > 0) {
      LLVM_DEBUG(dbgs() << "\nAlready fused and marked for erasure, skip.");
      continue;
    }

    // 4. Apply loop tiling to enable fusion. If unsuccessful, skip fusing op.
    auto tiledOp = tileLinalgOp(op, tileSizes, state);
    if (!tiledOp) {
      LLVM_DEBUG(dbgs() << "\nTile sizes did not produce loops, skip.");
      continue;
    }

    // 5. For now, we only fuse RAW dependences.
    SmallVector<Operation *, 8> fusedProducers;
    SmallVector<Value *, 8> fusedViews;
    for (auto dependence : G.getDependencesInto(
             consumer, LinalgDependenceGraph::DependenceType::RAW)) {
      auto producer = cast<LinalgOp>(dependence.dependentOpView.op);
      LLVM_DEBUG(dbgs() << "\n***Consider producer:\t"
                        << *producer.getOperation());

      // a. For now we require fusion on identical SSA values, this allows us to
      // not worry about partial writes etc.
      // TODO(ntv) support more elaborate fusion with non identical SSA values.
      auto *view = dependence.indexingView;
      if (view != dependence.dependentOpView.view) {
        LLVM_DEBUG(dbgs() << "\nviews are different SSA values, skip.");
        continue;
      }
      // b. Make some simple structural checks that alleviate the need for more
      // complex analyses.
      if (!isStructurallyFusableProducer(producer, view, op)) {
        LLVM_DEBUG(dbgs() << "\n***Not fusable:\t" << *producer.getOperation());
        continue;
      }
      // c. Check for fusion-preventing write that would violate dependences.
      // `view` is a producer write that cannot bypass any other write or read.
      bool preventFusion = false;
      for (auto *op : G.findCoveringDependences(producer, consumer))
        if (eraseSet.count(op) == 0) {
          preventFusion = true;
          LLVM_DEBUG(dbgs() << "\n***Found fusion preventing dep via: " << *op);
          break;
        }
      if (preventFusion)
        continue;

      // 6. Try to fuse `producer` just before `tiledOp`.
      auto tOp = tiledOp->op;
      OpBuilder builder(tOp.getOperation());
      ScopedContext scope(builder, tOp.getLoc());
      auto maybeFusedProducer = fuse(view, producer, op, tOp, state);
      if (!maybeFusedProducer) {
        LLVM_DEBUG(dbgs() << "\nFusion did not do anything, skip.");
        continue;
      }

      fusedProducers.push_back(producer.getOperation());
      fusedViews.push_back(view);
    }

    // 7. If no fusion occurred, or a drop the outer tiled loop which undoes
    // everything we did.
    if (fusedProducers.empty()) {
      tiledOp->loops[0].erase();
      continue;
    }

    eraseSet.insert(op);
    eraseSet.insert(fusedProducers.begin(), fusedProducers.end());
  }

  for (auto *op : eraseSet)
    op->erase();

  LLVM_DEBUG(f.print(dbgs() << "\nAfter linalg-fusion: \n"));
}

namespace {
struct LinalgFusionPass : public FunctionPass<LinalgFusionPass> {
  LinalgFusionPass();
  LinalgFusionPass(ArrayRef<int64_t> sizes);

  void runOnFunction() { fuseLinalgOps(getFunction(), tileSizes); }

  SmallVector<int64_t, 8> tileSizes;
};
} // namespace

LinalgFusionPass::LinalgFusionPass()
    : tileSizes(clTileSizes.begin(), clTileSizes.end()) {}

LinalgFusionPass::LinalgFusionPass(ArrayRef<int64_t> sizes)
    : LinalgFusionPass() {
  if (!sizes.empty())
    this->tileSizes.assign(sizes.begin(), sizes.end());
}

FunctionPassBase *
mlir::linalg::createLinalgFusionPass(ArrayRef<int64_t> tileSizes) {
  return new LinalgFusionPass(tileSizes);
}

static PassRegistration<LinalgFusionPass>
    pass("linalg-fusion", "Fuse operations in the linalg dialect");
