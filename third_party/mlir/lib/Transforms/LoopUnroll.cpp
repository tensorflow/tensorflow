//===- LoopUnroll.cpp - Code to perform loop unrolling --------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop unrolling.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "affine-loop-unroll"

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

// Loop unrolling factor.
static llvm::cl::opt<unsigned> clUnrollFactor(
    "unroll-factor",
    llvm::cl::desc("Use this unroll factor for all loops being unrolled"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clUnrollFull("unroll-full",
                                        llvm::cl::desc("Fully unroll loops"),
                                        llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<unsigned> clUnrollNumRepetitions(
    "unroll-num-reps",
    llvm::cl::desc("Unroll innermost loops repeatedly this many times"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<unsigned> clUnrollFullThreshold(
    "unroll-full-threshold", llvm::cl::Hidden,
    llvm::cl::desc(
        "Unroll all loops with trip count less than or equal to this"),
    llvm::cl::cat(clOptionsCategory));

namespace {
/// Loop unrolling pass. Unrolls all innermost loops unless full unrolling and a
/// full unroll threshold was specified, in which case, fully unrolls all loops
/// with trip count less than the specified threshold. The latter is for testing
/// purposes, especially for testing outer loop unrolling.
struct LoopUnroll : public FunctionPass<LoopUnroll> {
  const Optional<unsigned> unrollFactor;
  const Optional<bool> unrollFull;
  // Callback to obtain unroll factors; if this has a callable target, takes
  // precedence over command-line argument or passed argument.
  const std::function<unsigned(AffineForOp)> getUnrollFactor;

  explicit LoopUnroll(
      Optional<unsigned> unrollFactor = None, Optional<bool> unrollFull = None,
      const std::function<unsigned(AffineForOp)> &getUnrollFactor = nullptr)
      : unrollFactor(unrollFactor), unrollFull(unrollFull),
        getUnrollFactor(getUnrollFactor) {}

  void runOnFunction() override;

  /// Unroll this for op. Returns failure if nothing was done.
  LogicalResult runOnAffineForOp(AffineForOp forOp);

  static const unsigned kDefaultUnrollFactor = 4;
};
} // end anonymous namespace

void LoopUnroll::runOnFunction() {
  // Gathers all innermost loops through a post order pruned walk.
  struct InnermostLoopGatherer {
    // Store innermost loops as we walk.
    std::vector<AffineForOp> loops;

    void walkPostOrder(FuncOp f) {
      for (auto &b : f)
        walkPostOrder(b.begin(), b.end());
    }

    bool walkPostOrder(Block::iterator Start, Block::iterator End) {
      bool hasInnerLoops = false;
      // We need to walk all elements since all innermost loops need to be
      // gathered as opposed to determining whether this list has any inner
      // loops or not.
      while (Start != End)
        hasInnerLoops |= walkPostOrder(&(*Start++));
      return hasInnerLoops;
    }
    bool walkPostOrder(Operation *opInst) {
      bool hasInnerLoops = false;
      for (auto &region : opInst->getRegions())
        for (auto &block : region)
          hasInnerLoops |= walkPostOrder(block.begin(), block.end());
      if (isa<AffineForOp>(opInst)) {
        if (!hasInnerLoops)
          loops.push_back(cast<AffineForOp>(opInst));
        return true;
      }
      return hasInnerLoops;
    }
  };

  if (clUnrollFull.getNumOccurrences() > 0 &&
      clUnrollFullThreshold.getNumOccurrences() > 0) {
    // Store short loops as we walk.
    std::vector<AffineForOp> loops;

    // Gathers all loops with trip count <= minTripCount. Do a post order walk
    // so that loops are gathered from innermost to outermost (or else unrolling
    // an outer one may delete gathered inner ones).
    getFunction().walk([&](AffineForOp forOp) {
      Optional<uint64_t> tripCount = getConstantTripCount(forOp);
      if (tripCount.hasValue() && tripCount.getValue() <= clUnrollFullThreshold)
        loops.push_back(forOp);
    });
    for (auto forOp : loops)
      loopUnrollFull(forOp);
    return;
  }

  unsigned numRepetitions = clUnrollNumRepetitions.getNumOccurrences() > 0
                                ? clUnrollNumRepetitions
                                : 1;
  // If the call back is provided, we will recurse until no loops are found.
  FuncOp func = getFunction();
  for (unsigned i = 0; i < numRepetitions || getUnrollFactor; i++) {
    InnermostLoopGatherer ilg;
    ilg.walkPostOrder(func);
    auto &loops = ilg.loops;
    if (loops.empty())
      break;
    bool unrolled = false;
    for (auto forOp : loops)
      unrolled |= succeeded(runOnAffineForOp(forOp));
    if (!unrolled)
      // Break out if nothing was unrolled.
      break;
  }
}

/// Unrolls a 'affine.for' op. Returns success if the loop was unrolled,
/// failure otherwise. The default unroll factor is 4.
LogicalResult LoopUnroll::runOnAffineForOp(AffineForOp forOp) {
  // Use the function callback if one was provided.
  if (getUnrollFactor) {
    return loopUnrollByFactor(forOp, getUnrollFactor(forOp));
  }
  // Unroll by the factor passed, if any.
  if (unrollFactor.hasValue())
    return loopUnrollByFactor(forOp, unrollFactor.getValue());
  // Unroll by the command line factor if one was specified.
  if (clUnrollFactor.getNumOccurrences() > 0)
    return loopUnrollByFactor(forOp, clUnrollFactor);
  // Unroll completely if full loop unroll was specified.
  if (clUnrollFull.getNumOccurrences() > 0 ||
      (unrollFull.hasValue() && unrollFull.getValue()))
    return loopUnrollFull(forOp);

  // Unroll by four otherwise.
  return loopUnrollByFactor(forOp, kDefaultUnrollFactor);
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createLoopUnrollPass(
    int unrollFactor, int unrollFull,
    const std::function<unsigned(AffineForOp)> &getUnrollFactor) {
  return std::make_unique<LoopUnroll>(
      unrollFactor == -1 ? None : Optional<unsigned>(unrollFactor),
      unrollFull == -1 ? None : Optional<bool>(unrollFull), getUnrollFactor);
}

static PassRegistration<LoopUnroll> pass("affine-loop-unroll", "Unroll loops");
