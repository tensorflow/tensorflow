//===- LoopUnroll.cpp - Code to perform loop unrolling --------------------===//
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
// This file implements loop unrolling.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "loop-unroll"

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

// Loop unrolling factor.
static llvm::cl::opt<unsigned> clUnrollFactor(
    "unroll-factor", llvm::cl::Hidden,
    llvm::cl::desc("Use this unroll factor for all loops being unrolled"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clUnrollFull("unroll-full", llvm::cl::Hidden,
                                        llvm::cl::desc("Fully unroll loops"),
                                        llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<unsigned> clUnrollNumRepetitions(
    "unroll-num-reps", llvm::cl::Hidden,
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
struct LoopUnroll : public FunctionPass {
  const Optional<unsigned> unrollFactor;
  const Optional<bool> unrollFull;
  // Callback to obtain unroll factors; if this has a callable target, takes
  // precedence over command-line argument or passed argument.
  const std::function<unsigned(ConstOpPointer<AffineForOp>)> getUnrollFactor;

  explicit LoopUnroll(Optional<unsigned> unrollFactor = None,
                      Optional<bool> unrollFull = None,
                      const std::function<unsigned(ConstOpPointer<AffineForOp>)>
                          &getUnrollFactor = nullptr)
      : FunctionPass(&LoopUnroll::passID), unrollFactor(unrollFactor),
        unrollFull(unrollFull), getUnrollFactor(getUnrollFactor) {}

  PassResult runOnFunction(Function *f) override;

  /// Unroll this for inst. Returns false if nothing was done.
  bool runOnAffineForOp(OpPointer<AffineForOp> forOp);

  static const unsigned kDefaultUnrollFactor = 4;

  static char passID;
};
} // end anonymous namespace

char LoopUnroll::passID = 0;

PassResult LoopUnroll::runOnFunction(Function *f) {
  // Gathers all innermost loops through a post order pruned walk.
  struct InnermostLoopGatherer {
    // Store innermost loops as we walk.
    std::vector<OpPointer<AffineForOp>> loops;

    void walkPostOrder(Function *f) {
      for (auto &b : *f)
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
    bool walkPostOrder(Instruction *opInst) {
      bool hasInnerLoops = false;
      for (auto &blockList : opInst->getBlockLists())
        for (auto &block : blockList)
          hasInnerLoops |= walkPostOrder(block.begin(), block.end());
      if (opInst->isa<AffineForOp>()) {
        if (!hasInnerLoops)
          loops.push_back(opInst->cast<AffineForOp>());
        return true;
      }
      return hasInnerLoops;
    }
  };

  if (clUnrollFull.getNumOccurrences() > 0 &&
      clUnrollFullThreshold.getNumOccurrences() > 0) {
    // Store short loops as we walk.
    std::vector<OpPointer<AffineForOp>> loops;

    // Gathers all loops with trip count <= minTripCount. Do a post order walk
    // so that loops are gathered from innermost to outermost (or else unrolling
    // an outer one may delete gathered inner ones).
    f->walkPostOrder<AffineForOp>([&](OpPointer<AffineForOp> forOp) {
      Optional<uint64_t> tripCount = getConstantTripCount(forOp);
      if (tripCount.hasValue() && tripCount.getValue() <= clUnrollFullThreshold)
        loops.push_back(forOp);
    });
    for (auto forOp : loops)
      loopUnrollFull(forOp);
    return success();
  }

  unsigned numRepetitions = clUnrollNumRepetitions.getNumOccurrences() > 0
                                ? clUnrollNumRepetitions
                                : 1;
  // If the call back is provided, we will recurse until no loops are found.
  for (unsigned i = 0; i < numRepetitions || getUnrollFactor; i++) {
    InnermostLoopGatherer ilg;
    ilg.walkPostOrder(f);
    auto &loops = ilg.loops;
    if (loops.empty())
      break;
    bool unrolled = false;
    for (auto forOp : loops)
      unrolled |= runOnAffineForOp(forOp);
    if (!unrolled)
      // Break out if nothing was unrolled.
      break;
  }
  return success();
}

/// Unrolls a 'for' inst. Returns true if the loop was unrolled, false
/// otherwise. The default unroll factor is 4.
bool LoopUnroll::runOnAffineForOp(OpPointer<AffineForOp> forOp) {
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

FunctionPass *mlir::createLoopUnrollPass(
    int unrollFactor, int unrollFull,
    const std::function<unsigned(ConstOpPointer<AffineForOp>)>
        &getUnrollFactor) {
  return new LoopUnroll(
      unrollFactor == -1 ? None : Optional<unsigned>(unrollFactor),
      unrollFull == -1 ? None : Optional<bool>(unrollFull), getUnrollFactor);
}

static PassRegistration<LoopUnroll> pass("loop-unroll", "Unroll loops");
