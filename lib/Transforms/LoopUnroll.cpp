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

// Loop unrolling factor.
static llvm::cl::opt<unsigned>
    clUnrollFactor("unroll-factor", llvm::cl::Hidden,
                   llvm::cl::desc("Use this unroll factor for all loops"));

static llvm::cl::opt<bool> clUnrollFull("unroll-full", llvm::cl::Hidden,
                                        llvm::cl::desc("Fully unroll loops"));

static llvm::cl::opt<unsigned> clUnrollFullThreshold(
    "unroll-full-threshold", llvm::cl::Hidden,
    llvm::cl::desc(
        "Unroll all loops with trip count less than or equal to this"));

namespace {
/// Loop unrolling pass. Unrolls all innermost loops unless full unrolling and a
/// full unroll threshold was specified, in which case, fully unrolls all loops
/// with trip count less than the specified threshold. The latter is for testing
/// purposes, especially for testing outer loop unrolling.
struct LoopUnroll : public FunctionPass {
  Optional<unsigned> unrollFactor;
  Optional<bool> unrollFull;

  explicit LoopUnroll(Optional<unsigned> unrollFactor = None,
                      Optional<bool> unrollFull = None)
      : FunctionPass(&LoopUnroll::passID), unrollFactor(unrollFactor),
        unrollFull(unrollFull) {}

  PassResult runOnMLFunction(MLFunction *f) override;

  /// Unroll this for stmt. Returns false if nothing was done.
  bool runOnForStmt(ForStmt *forStmt);

  static char passID;
};
} // end anonymous namespace

char LoopUnroll::passID = 0;

PassResult LoopUnroll::runOnMLFunction(MLFunction *f) {
  // Gathers all innermost loops through a post order pruned walk.
  class InnermostLoopGatherer : public StmtWalker<InnermostLoopGatherer, bool> {
  public:
    // Store innermost loops as we walk.
    std::vector<ForStmt *> loops;

    // This method specialized to encode custom return logic.
    using StmtListType = llvm::iplist<Statement>;
    bool walkPostOrder(StmtListType::iterator Start,
                       StmtListType::iterator End) {
      bool hasInnerLoops = false;
      // We need to walk all elements since all innermost loops need to be
      // gathered as opposed to determining whether this list has any inner
      // loops or not.
      while (Start != End)
        hasInnerLoops |= walkPostOrder(&(*Start++));
      return hasInnerLoops;
    }

    bool walkForStmtPostOrder(ForStmt *forStmt) {
      bool hasInnerLoops = walkPostOrder(forStmt->begin(), forStmt->end());
      if (!hasInnerLoops)
        loops.push_back(forStmt);
      return true;
    }

    bool walkIfStmtPostOrder(IfStmt *ifStmt) {
      bool hasInnerLoops =
          walkPostOrder(ifStmt->getThen()->begin(), ifStmt->getThen()->end());
      if (ifStmt->hasElse())
        hasInnerLoops |=
            walkPostOrder(ifStmt->getElse()->begin(), ifStmt->getElse()->end());
      return hasInnerLoops;
    }

    bool visitOperationStmt(OperationStmt *opStmt) { return false; }

    // FIXME: can't use base class method for this because that in turn would
    // need to use the derived class method above. CRTP doesn't allow it, and
    // the compiler error resulting from it is also misleading.
    using StmtWalker<InnermostLoopGatherer, bool>::walkPostOrder;
  };

  // Gathers all loops with trip count <= minTripCount.
  class ShortLoopGatherer : public StmtWalker<ShortLoopGatherer> {
  public:
    // Store short loops as we walk.
    std::vector<ForStmt *> loops;
    const unsigned minTripCount;
    ShortLoopGatherer(unsigned minTripCount) : minTripCount(minTripCount) {}

    void visitForStmt(ForStmt *forStmt) {
      Optional<uint64_t> tripCount = getConstantTripCount(*forStmt);
      if (tripCount.hasValue() && tripCount.getValue() <= minTripCount)
        loops.push_back(forStmt);
    }
  };

  if (clUnrollFull.getNumOccurrences() > 0 &&
      clUnrollFullThreshold.getNumOccurrences() > 0) {
    ShortLoopGatherer slg(clUnrollFullThreshold);
    // Do a post order walk so that loops are gathered from innermost to
    // outermost (or else unrolling an outer one may delete gathered inner
    // ones).
    slg.walkPostOrder(f);
    auto &loops = slg.loops;
    for (auto *forStmt : loops)
      loopUnrollFull(forStmt);
    return success();
  }

  InnermostLoopGatherer ilg;
  ilg.walkPostOrder(f);
  auto &loops = ilg.loops;
  for (auto *forStmt : loops)
    runOnForStmt(forStmt);
  return success();
}

/// Unroll a 'for' stmt. Default unroll factor is 4.
bool LoopUnroll::runOnForStmt(ForStmt *forStmt) {
  // Unroll by the factor passed, if any.
  if (unrollFactor.hasValue())
    return loopUnrollByFactor(forStmt, unrollFactor.getValue());
  // Unroll by the command line factor if one was specified.
  if (clUnrollFactor.getNumOccurrences() > 0)
    return loopUnrollByFactor(forStmt, clUnrollFactor);
  // Unroll completely if full loop unroll was specified.
  if (clUnrollFull.getNumOccurrences() > 0 ||
      (unrollFull.hasValue() && unrollFull.getValue()))
    return loopUnrollFull(forStmt);

  // Unroll by four otherwise.
  return loopUnrollByFactor(forStmt, 4);
}

FunctionPass *mlir::createLoopUnrollPass(int unrollFactor, int unrollFull) {
  return new LoopUnroll(unrollFactor == -1 ? None
                                           : Optional<unsigned>(unrollFactor),
                        unrollFull == -1 ? None : Optional<bool>(unrollFull));
}

static PassRegistration<LoopUnroll> pass("loop-unroll", "Unroll loops");
