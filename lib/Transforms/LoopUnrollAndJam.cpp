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
// This file implements loop unroll and jam for MLFunctions. Unroll and jam is a
// transformation that improves locality, in particular, register reuse, while
// also improving instruction level parallelism. The example below shows what it
// does in nearly the general case. Loop unroll and jam currently works if the
// bounds of the loops inner to the loop being unroll-jammed do not depend on
// the latter.
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
// stmt's, bodies of those loops will not be jammed.
//===----------------------------------------------------------------------===//
#include "mlir/Transforms/Passes.h"

#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

// Loop unroll and jam factor.
static llvm::cl::opt<unsigned>
    clUnrollJamFactor("unroll-jam-factor", llvm::cl::Hidden,
                      llvm::cl::desc("Use this unroll jam factor for all loops"
                                     " (default 4)"));

namespace {
/// Loop unroll jam pass. Currently, this just unroll jams the first
/// outer loop in an MLFunction.
struct LoopUnrollAndJam : public FunctionPass {
  Optional<unsigned> unrollJamFactor;
  static const unsigned kDefaultUnrollJamFactor = 4;

  explicit LoopUnrollAndJam(Optional<unsigned> unrollJamFactor = None)
      : FunctionPass(&LoopUnrollAndJam::passID),
        unrollJamFactor(unrollJamFactor) {}

  PassResult runOnMLFunction(MLFunction *f) override;
  bool runOnForStmt(ForStmt *forStmt);

  static char passID;
};
} // end anonymous namespace

char LoopUnrollAndJam::passID = 0;

FunctionPass *mlir::createLoopUnrollAndJamPass(int unrollJamFactor) {
  return new LoopUnrollAndJam(
      unrollJamFactor == -1 ? None : Optional<unsigned>(unrollJamFactor));
}

PassResult LoopUnrollAndJam::runOnMLFunction(MLFunction *f) {
  // Currently, just the outermost loop from the first loop nest is
  // unroll-and-jammed by this pass. However, runOnForStmt can be called on any
  // for Stmt.
  auto *forStmt = dyn_cast<ForStmt>(f->getBody()->begin());
  if (!forStmt)
    return success();

  runOnForStmt(forStmt);
  return success();
}

/// Unroll and jam a 'for' stmt. Default unroll jam factor is
/// kDefaultUnrollJamFactor. Return false if nothing was done.
bool LoopUnrollAndJam::runOnForStmt(ForStmt *forStmt) {
  // Unroll and jam by the factor that was passed if any.
  if (unrollJamFactor.hasValue())
    return loopUnrollJamByFactor(forStmt, unrollJamFactor.getValue());
  // Otherwise, unroll jam by the command-line factor if one was specified.
  if (clUnrollJamFactor.getNumOccurrences() > 0)
    return loopUnrollJamByFactor(forStmt, clUnrollJamFactor);

  // Unroll and jam by four otherwise.
  return loopUnrollJamByFactor(forStmt, kDefaultUnrollJamFactor);
}

bool mlir::loopUnrollJamUpToFactor(ForStmt *forStmt, uint64_t unrollJamFactor) {
  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(*forStmt);

  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() < unrollJamFactor)
    return loopUnrollJamByFactor(forStmt, mayBeConstantTripCount.getValue());
  return loopUnrollJamByFactor(forStmt, unrollJamFactor);
}

/// Unrolls and jams this loop by the specified factor.
bool mlir::loopUnrollJamByFactor(ForStmt *forStmt, uint64_t unrollJamFactor) {
  // Gathers all maximal sub-blocks of statements that do not themselves include
  // a for stmt (a statement could have a descendant for stmt though in its
  // tree).
  class JamBlockGatherer : public StmtWalker<JamBlockGatherer> {
  public:
    using StmtListType = llvm::iplist<Statement>;

    // Store iterators to the first and last stmt of each sub-block found.
    std::vector<std::pair<StmtBlock::iterator, StmtBlock::iterator>> subBlocks;

    // This is a linear time walk.
    void walk(StmtListType::iterator Start, StmtListType::iterator End) {
      for (auto it = Start; it != End;) {
        auto subBlockStart = it;
        while (it != End && !isa<ForStmt>(it))
          ++it;
        if (it != subBlockStart)
          subBlocks.push_back({subBlockStart, std::prev(it)});
        // Process all for stmts that appear next.
        while (it != End && isa<ForStmt>(it))
          walkForStmt(cast<ForStmt>(it++));
      }
    }
  };

  assert(unrollJamFactor >= 1 && "unroll jam factor should be >= 1");

  if (unrollJamFactor == 1 || forStmt->getBody()->empty())
    return false;

  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(*forStmt);

  if (!mayBeConstantTripCount.hasValue() &&
      getLargestDivisorOfTripCount(*forStmt) % unrollJamFactor != 0)
    return false;

  auto lbMap = forStmt->getLowerBoundMap();
  auto ubMap = forStmt->getUpperBoundMap();

  // Loops with max/min expressions won't be unrolled here (the output can't be
  // expressed as an MLFunction in the general case). However, the right way to
  // do such unrolling for an MLFunction would be to specialize the loop for the
  // 'hotspot' case and unroll that hotspot.
  if (lbMap.getNumResults() != 1 || ubMap.getNumResults() != 1)
    return false;

  // Same operand list for lower and upper bound for now.
  // TODO(bondhugula): handle bounds with different sets of operands.
  if (!forStmt->matchingBoundOperandList())
    return false;

  // If the trip count is lower than the unroll jam factor, no unroll jam.
  // TODO(bondhugula): option to specify cleanup loop unrolling.
  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() < unrollJamFactor)
    return false;

  // Gather all sub-blocks to jam upon the loop being unrolled.
  JamBlockGatherer jbg;
  jbg.walkForStmt(forStmt);
  auto &subBlocks = jbg.subBlocks;

  // Generate the cleanup loop if trip count isn't a multiple of
  // unrollJamFactor.
  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() % unrollJamFactor != 0) {
    DenseMap<const MLValue *, MLValue *> operandMap;
    // Insert the cleanup loop right after 'forStmt'.
    MLFuncBuilder builder(forStmt->getBlock(),
                          std::next(StmtBlock::iterator(forStmt)));
    auto *cleanupForStmt = cast<ForStmt>(builder.clone(*forStmt, operandMap));
    cleanupForStmt->setLowerBoundMap(
        getCleanupLoopLowerBound(*forStmt, unrollJamFactor, &builder));

    // The upper bound needs to be adjusted.
    forStmt->setUpperBoundMap(
        getUnrolledLoopUpperBound(*forStmt, unrollJamFactor, &builder));

    // Promote the loop body up if this has turned into a single iteration loop.
    promoteIfSingleIteration(cleanupForStmt);
  }

  // Scale the step of loop being unroll-jammed by the unroll-jam factor.
  int64_t step = forStmt->getStep();
  forStmt->setStep(step * unrollJamFactor);

  for (auto &subBlock : subBlocks) {
    // Builder to insert unroll-jammed bodies. Insert right at the end of
    // sub-block.
    MLFuncBuilder builder(subBlock.first->getBlock(),
                          std::next(subBlock.second));

    // Unroll and jam (appends unrollJamFactor-1 additional copies).
    for (unsigned i = 1; i < unrollJamFactor; i++) {
      DenseMap<const MLValue *, MLValue *> operandMapping;

      // If the induction variable is used, create a remapping to the value for
      // this unrolled instance.
      if (!forStmt->use_empty()) {
        // iv' = iv + i, i = 1 to unrollJamFactor-1.
        auto d0 = builder.getAffineDimExpr(0);
        auto bumpMap = builder.getAffineMap(1, 0, {d0 + i * step}, {});
        auto *ivUnroll =
            builder.create<AffineApplyOp>(forStmt->getLoc(), bumpMap, forStmt)
                ->getResult(0);
        operandMapping[forStmt] = cast<MLValue>(ivUnroll);
      }
      // Clone the sub-block being unroll-jammed.
      for (auto it = subBlock.first; it != std::next(subBlock.second); ++it) {
        builder.clone(*it, operandMapping);
      }
    }
  }

  // Promote the loop body up if this has turned into a single iteration loop.
  promoteIfSingleIteration(forStmt);

  return true;
}

static PassRegistration<LoopUnrollAndJam> pass("loop-unroll-jam",
                                               "Unroll and jam loops");
