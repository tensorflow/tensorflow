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
// instruction level parallelism. The example below shows what it does in nearly
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
// inst's, bodies of those loops will not be jammed.
//===----------------------------------------------------------------------===//
#include "mlir/Transforms/Passes.h"

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/InstVisitor.h"
#include "mlir/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

#define DEBUG_TYPE "loop-unroll-jam"

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
struct LoopUnrollAndJam : public FunctionPass {
  Optional<unsigned> unrollJamFactor;
  static const unsigned kDefaultUnrollJamFactor = 4;

  explicit LoopUnrollAndJam(Optional<unsigned> unrollJamFactor = None)
      : FunctionPass(&LoopUnrollAndJam::passID),
        unrollJamFactor(unrollJamFactor) {}

  PassResult runOnFunction(Function *f) override;
  bool runOnAffineForOp(OpPointer<AffineForOp> forOp);

  static char passID;
};
} // end anonymous namespace

char LoopUnrollAndJam::passID = 0;

FunctionPass *mlir::createLoopUnrollAndJamPass(int unrollJamFactor) {
  return new LoopUnrollAndJam(
      unrollJamFactor == -1 ? None : Optional<unsigned>(unrollJamFactor));
}

PassResult LoopUnrollAndJam::runOnFunction(Function *f) {
  // Currently, just the outermost loop from the first loop nest is
  // unroll-and-jammed by this pass. However, runOnAffineForOp can be called on
  // any for Inst.
  auto &entryBlock = f->front();
  if (!entryBlock.empty())
    if (auto forOp =
            cast<OperationInst>(entryBlock.front()).dyn_cast<AffineForOp>())
      runOnAffineForOp(forOp);

  return success();
}

/// Unroll and jam a 'for' inst. Default unroll jam factor is
/// kDefaultUnrollJamFactor. Return false if nothing was done.
bool LoopUnrollAndJam::runOnAffineForOp(OpPointer<AffineForOp> forOp) {
  // Unroll and jam by the factor that was passed if any.
  if (unrollJamFactor.hasValue())
    return loopUnrollJamByFactor(forOp, unrollJamFactor.getValue());
  // Otherwise, unroll jam by the command-line factor if one was specified.
  if (clUnrollJamFactor.getNumOccurrences() > 0)
    return loopUnrollJamByFactor(forOp, clUnrollJamFactor);

  // Unroll and jam by four otherwise.
  return loopUnrollJamByFactor(forOp, kDefaultUnrollJamFactor);
}

bool mlir::loopUnrollJamUpToFactor(OpPointer<AffineForOp> forOp,
                                   uint64_t unrollJamFactor) {
  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(forOp);

  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() < unrollJamFactor)
    return loopUnrollJamByFactor(forOp, mayBeConstantTripCount.getValue());
  return loopUnrollJamByFactor(forOp, unrollJamFactor);
}

/// Unrolls and jams this loop by the specified factor.
bool mlir::loopUnrollJamByFactor(OpPointer<AffineForOp> forOp,
                                 uint64_t unrollJamFactor) {
  // Gathers all maximal sub-blocks of instructions that do not themselves
  // include a for inst (a instruction could have a descendant for inst though
  // in its tree).
  class JamBlockGatherer : public InstWalker<JamBlockGatherer> {
  public:
    using InstListType = llvm::iplist<Instruction>;
    using InstWalker<JamBlockGatherer>::walk;

    // Store iterators to the first and last inst of each sub-block found.
    std::vector<std::pair<Block::iterator, Block::iterator>> subBlocks;

    // This is a linear time walk.
    void walk(InstListType::iterator Start, InstListType::iterator End) {
      for (auto it = Start; it != End;) {
        auto subBlockStart = it;
        while (it != End && !cast<OperationInst>(it)->isa<AffineForOp>())
          ++it;
        if (it != subBlockStart)
          subBlocks.push_back({subBlockStart, std::prev(it)});
        // Process all for insts that appear next.
        while (it != End && cast<OperationInst>(it)->isa<AffineForOp>())
          walk(&*it++);
      }
    }
  };

  assert(unrollJamFactor >= 1 && "unroll jam factor should be >= 1");

  if (unrollJamFactor == 1 || forOp->getBody()->empty())
    return false;

  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(forOp);

  if (!mayBeConstantTripCount.hasValue() &&
      getLargestDivisorOfTripCount(forOp) % unrollJamFactor != 0)
    return false;

  auto lbMap = forOp->getLowerBoundMap();
  auto ubMap = forOp->getUpperBoundMap();

  // Loops with max/min expressions won't be unrolled here (the output can't be
  // expressed as a Function in the general case). However, the right way to
  // do such unrolling for a Function would be to specialize the loop for the
  // 'hotspot' case and unroll that hotspot.
  if (lbMap.getNumResults() != 1 || ubMap.getNumResults() != 1)
    return false;

  // Same operand list for lower and upper bound for now.
  // TODO(bondhugula): handle bounds with different sets of operands.
  if (!forOp->matchingBoundOperandList())
    return false;

  // If the trip count is lower than the unroll jam factor, no unroll jam.
  // TODO(bondhugula): option to specify cleanup loop unrolling.
  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() < unrollJamFactor)
    return false;

  auto *forInst = forOp->getInstruction();

  // Gather all sub-blocks to jam upon the loop being unrolled.
  JamBlockGatherer jbg;
  jbg.walkOpInst(forInst);
  auto &subBlocks = jbg.subBlocks;

  // Generate the cleanup loop if trip count isn't a multiple of
  // unrollJamFactor.
  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() % unrollJamFactor != 0) {
    // Insert the cleanup loop right after 'forOp'.
    FuncBuilder builder(forInst->getBlock(),
                        std::next(Block::iterator(forInst)));
    auto cleanupAffineForOp =
        cast<OperationInst>(builder.clone(*forInst))->cast<AffineForOp>();
    cleanupAffineForOp->setLowerBoundMap(
        getCleanupLoopLowerBound(forOp, unrollJamFactor, &builder));

    // The upper bound needs to be adjusted.
    forOp->setUpperBoundMap(
        getUnrolledLoopUpperBound(forOp, unrollJamFactor, &builder));

    // Promote the loop body up if this has turned into a single iteration loop.
    promoteIfSingleIteration(cleanupAffineForOp);
  }

  // Scale the step of loop being unroll-jammed by the unroll-jam factor.
  int64_t step = forOp->getStep();
  forOp->setStep(step * unrollJamFactor);

  auto *forOpIV = forOp->getInductionVar();
  for (auto &subBlock : subBlocks) {
    // Builder to insert unroll-jammed bodies. Insert right at the end of
    // sub-block.
    FuncBuilder builder(subBlock.first->getBlock(), std::next(subBlock.second));

    // Unroll and jam (appends unrollJamFactor-1 additional copies).
    for (unsigned i = 1; i < unrollJamFactor; i++) {
      BlockAndValueMapping operandMapping;

      // If the induction variable is used, create a remapping to the value for
      // this unrolled instance.
      if (!forOpIV->use_empty()) {
        // iv' = iv + i, i = 1 to unrollJamFactor-1.
        auto d0 = builder.getAffineDimExpr(0);
        auto bumpMap = builder.getAffineMap(1, 0, {d0 + i * step}, {});
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

  return true;
}

static PassRegistration<LoopUnrollAndJam> pass("loop-unroll-jam",
                                               "Unroll and jam loops");
