//===- LoopUnrollAndJam.cpp - Code to perform loop unroll and jam ---------===//
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
// This file implements loop unroll and jam. Unroll and jam is a transformation
// that improves locality, in particular, register reuse, while also improving
// operation level parallelism. The example below shows what it does in nearly
// the general case. Loop unroll and jam currently works if the bounds of the
// loops inner to the loop being unroll-jammed do not depend on the latter.
//
// Before      After unroll and jam of i by factor 2:
//
//             for i, step = 2
// for i         S1(i);
//   S1;         S2(i);
//   S2;         S1(i+1);
//   for j       S2(i+1);
//     S3;       for j
//     S4;         S3(i, j);
//   S5;           S4(i, j);
//   S6;           S3(i+1, j)
//                 S4(i+1, j)
//               S5(i);
//               S6(i);
//               S5(i+1);
//               S6(i+1);
//
// Note: 'if/else' blocks are not jammed. So, if there are loops inside if
// op's, bodies of those loops will not be jammed.
//===----------------------------------------------------------------------===//
#include "mlir/Transforms/Passes.h"

#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

#define DEBUG_TYPE "affine-loop-unroll-jam"

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

// Loop unroll and jam factor.
static llvm::cl::opt<unsigned>
    clUnrollJamFactor("unroll-jam-factor", llvm::cl::Hidden,
                      llvm::cl::desc("Use this unroll jam factor for all loops"
                                     " (default 4)"),
                      llvm::cl::cat(clOptionsCategory));

namespace {
/// Loop unroll jam pass. Currently, this just unroll jams the first
/// outer loop in a Function.
struct LoopUnrollAndJam : public FunctionPass<LoopUnrollAndJam> {
  Optional<unsigned> unrollJamFactor;
  static const unsigned kDefaultUnrollJamFactor = 4;

  explicit LoopUnrollAndJam(Optional<unsigned> unrollJamFactor = None)
      : unrollJamFactor(unrollJamFactor) {}

  void runOnFunction() override;
  LogicalResult runOnAffineForOp(AffineForOp forOp);
};
} // end anonymous namespace

std::unique_ptr<OpPassBase<FuncOp>>
mlir::createLoopUnrollAndJamPass(int unrollJamFactor) {
  return std::make_unique<LoopUnrollAndJam>(
      unrollJamFactor == -1 ? None : Optional<unsigned>(unrollJamFactor));
}

void LoopUnrollAndJam::runOnFunction() {
  // Currently, just the outermost loop from the first loop nest is
  // unroll-and-jammed by this pass. However, runOnAffineForOp can be called on
  // any for operation.
  auto &entryBlock = getFunction().front();
  if (auto forOp = dyn_cast<AffineForOp>(entryBlock.front()))
    runOnAffineForOp(forOp);
}

/// Unroll and jam a 'affine.for' op. Default unroll jam factor is
/// kDefaultUnrollJamFactor. Return failure if nothing was done.
LogicalResult LoopUnrollAndJam::runOnAffineForOp(AffineForOp forOp) {
  // Unroll and jam by the factor that was passed if any.
  if (unrollJamFactor.hasValue())
    return loopUnrollJamByFactor(forOp, unrollJamFactor.getValue());
  // Otherwise, unroll jam by the command-line factor if one was specified.
  if (clUnrollJamFactor.getNumOccurrences() > 0)
    return loopUnrollJamByFactor(forOp, clUnrollJamFactor);

  // Unroll and jam by four otherwise.
  return loopUnrollJamByFactor(forOp, kDefaultUnrollJamFactor);
}

LogicalResult mlir::loopUnrollJamUpToFactor(AffineForOp forOp,
                                            uint64_t unrollJamFactor) {
  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(forOp);

  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() < unrollJamFactor)
    return loopUnrollJamByFactor(forOp, mayBeConstantTripCount.getValue());
  return loopUnrollJamByFactor(forOp, unrollJamFactor);
}

/// Unrolls and jams this loop by the specified factor.
LogicalResult mlir::loopUnrollJamByFactor(AffineForOp forOp,
                                          uint64_t unrollJamFactor) {
  // Gathers all maximal sub-blocks of operations that do not themselves
  // include a for op (a operation could have a descendant for op though
  // in its tree).  Ignore the block terminators.
  struct JamBlockGatherer {
    // Store iterators to the first and last op of each sub-block found.
    std::vector<std::pair<Block::iterator, Block::iterator>> subBlocks;

    // This is a linear time walk.
    void walk(Operation *op) {
      for (auto &region : op->getRegions())
        for (auto &block : region)
          walk(block);
    }
    void walk(Block &block) {
      for (auto it = block.begin(), e = std::prev(block.end()); it != e;) {
        auto subBlockStart = it;
        while (it != e && !isa<AffineForOp>(&*it))
          ++it;
        if (it != subBlockStart)
          subBlocks.push_back({subBlockStart, std::prev(it)});
        // Process all for insts that appear next.
        while (it != e && isa<AffineForOp>(&*it))
          walk(&*it++);
      }
    }
  };

  assert(unrollJamFactor >= 1 && "unroll jam factor should be >= 1");

  if (unrollJamFactor == 1)
    return promoteIfSingleIteration(forOp);

  if (forOp.getBody()->empty() ||
      forOp.getBody()->begin() == std::prev(forOp.getBody()->end()))
    return failure();

  // Loops where both lower and upper bounds are multi-result maps won't be
  // unrolled (since the trip can't be expressed as an affine function in
  // general).
  // TODO(mlir-team): this may not be common, but we could support the case
  // where the lower bound is a multi-result map and the ub is a single result
  // one.
  if (forOp.getLowerBoundMap().getNumResults() != 1)
    return failure();

  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(forOp);
  // If the trip count is lower than the unroll jam factor, no unroll jam.
  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() < unrollJamFactor)
    return failure();

  auto *forInst = forOp.getOperation();

  // Gather all sub-blocks to jam upon the loop being unrolled.
  JamBlockGatherer jbg;
  jbg.walk(forInst);
  auto &subBlocks = jbg.subBlocks;

  // Generate the cleanup loop if trip count isn't a multiple of
  // unrollJamFactor.
  if (getLargestDivisorOfTripCount(forOp) % unrollJamFactor != 0) {
    // Insert the cleanup loop right after 'forOp'.
    OpBuilder builder(forInst->getBlock(), std::next(Block::iterator(forInst)));
    auto cleanupAffineForOp = cast<AffineForOp>(builder.clone(*forInst));
    // Adjust the lower bound of the cleanup loop; its upper bound is the same
    // as the original loop's upper bound.
    AffineMap cleanupMap;
    SmallVector<Value *, 4> cleanupOperands;
    getCleanupLoopLowerBound(forOp, unrollJamFactor, &cleanupMap,
                             &cleanupOperands, builder);
    cleanupAffineForOp.setLowerBound(cleanupOperands, cleanupMap);

    // Promote the cleanup loop if it has turned into a single iteration loop.
    promoteIfSingleIteration(cleanupAffineForOp);

    // Adjust the upper bound of the original loop - it will be the same as the
    // cleanup loop's lower bound. Its lower bound remains unchanged.
    forOp.setUpperBound(cleanupOperands, cleanupMap);
  }

  // Scale the step of loop being unroll-jammed by the unroll-jam factor.
  int64_t step = forOp.getStep();
  forOp.setStep(step * unrollJamFactor);

  auto *forOpIV = forOp.getInductionVar();
  // Unroll and jam (appends unrollJamFactor - 1 additional copies).
  for (unsigned i = unrollJamFactor - 1; i >= 1; --i) {
    // Operand map persists across all sub-blocks.
    BlockAndValueMapping operandMapping;
    for (auto &subBlock : subBlocks) {
      // Builder to insert unroll-jammed bodies. Insert right at the end of
      // sub-block.
      OpBuilder builder(subBlock.first->getBlock(), std::next(subBlock.second));

      // If the induction variable is used, create a remapping to the value for
      // this unrolled instance.
      if (!forOpIV->use_empty()) {
        // iv' = iv + i, i = 1 to unrollJamFactor-1.
        auto d0 = builder.getAffineDimExpr(0);
        auto bumpMap = builder.getAffineMap(1, 0, {d0 + i * step});
        auto ivUnroll =
            builder.create<AffineApplyOp>(forInst->getLoc(), bumpMap, forOpIV);
        operandMapping.map(forOpIV, ivUnroll);
      }
      // Clone the sub-block being unroll-jammed.
      for (auto it = subBlock.first; it != std::next(subBlock.second); ++it) {
        builder.clone(*it, operandMapping);
      }
    }
  }

  // Promote the loop body up if this has turned into a single iteration loop.
  promoteIfSingleIteration(forOp);
  return success();
}

static PassRegistration<LoopUnrollAndJam> pass("affine-loop-unroll-jam",
                                               "Unroll and jam loops");
