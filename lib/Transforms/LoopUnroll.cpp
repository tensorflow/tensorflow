//===- Unroll.cpp - Code to perform loop unrolling ------------------------===//
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
#include "mlir/IR/StandardOps.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/Transforms/Pass.h"
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
struct LoopUnroll : public MLFunctionPass {
  Optional<unsigned> unrollFactor;
  Optional<bool> unrollFull;

  explicit LoopUnroll(Optional<unsigned> unrollFactor,
                      Optional<bool> unrollFull)
      : unrollFactor(unrollFactor), unrollFull(unrollFull) {}

  PassResult runOnMLFunction(MLFunction *f) override;
  /// Unroll this for stmt. Returns false if nothing was done.
  bool runOnForStmt(ForStmt *forStmt);
};
} // end anonymous namespace

MLFunctionPass *mlir::createLoopUnrollPass(int unrollFactor, int unrollFull) {
  return new LoopUnroll(unrollFactor == -1 ? None
                                           : Optional<unsigned>(unrollFactor),
                        unrollFull == -1 ? None : Optional<bool>(unrollFull));
}

PassResult LoopUnroll::runOnMLFunction(MLFunction *f) {
  // Gathers all innermost loops through a post order pruned walk.
  class InnermostLoopGatherer : public StmtWalker<InnermostLoopGatherer, bool> {
  public:
    // Store innermost loops as we walk.
    std::vector<ForStmt *> loops;

    // This method specialized to encode custom return logic.
    typedef llvm::iplist<Statement> StmtListType;
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

/// Unroll a for stmt. Default unroll factor is 4.
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

/// Unrolls this loop completely.
bool mlir::loopUnrollFull(ForStmt *forStmt) {
  Optional<uint64_t> tripCount = getConstantTripCount(*forStmt);
  if (tripCount.hasValue())
    return loopUnrollByFactor(forStmt, tripCount.getValue());
  return false;
}

/// Returns the upper bound of an unrolled loop with lower bound 'lb' and with
/// the specified trip count, stride, and unroll factor.
static AffineMap *getUnrolledLoopUpperBound(AffineMap *lbMap,
                                            uint64_t tripCount,
                                            unsigned unrollFactor, int64_t step,
                                            MLFuncBuilder *builder) {
  assert(lbMap->getNumResults() == 1);
  auto *lbExpr = lbMap->getResult(0);
  // lbExpr + (count - count % unrollFactor - 1) * step).
  auto *expr = builder->getAddExpr(
      lbExpr, builder->getConstantExpr(
                  (tripCount - tripCount % unrollFactor - 1) * step));
  return builder->getAffineMap(lbMap->getNumDims(), lbMap->getNumSymbols(),
                               {expr}, {});
}

/// Returns the lower bound of the cleanup loop when unrolling a loop with lower
/// bound 'lb' and with the specified trip count, stride, and unroll factor.
static AffineMap *getCleanupLoopLowerBound(AffineMap *lbMap, uint64_t tripCount,
                                           unsigned unrollFactor, int64_t step,
                                           MLFuncBuilder *builder) {
  assert(lbMap->getNumResults() == 1);
  auto *lbExpr = lbMap->getResult(0);
  // lbExpr + (count - count % unrollFactor) * step);
  auto *expr = builder->getAddExpr(
      lbExpr,
      builder->getConstantExpr((tripCount - tripCount % unrollFactor) * step));
  return builder->getAffineMap(lbMap->getNumDims(), lbMap->getNumSymbols(),
                               {expr}, {});
}

/// Unrolls this loop by the specified unroll factor.
bool mlir::loopUnrollByFactor(ForStmt *forStmt, uint64_t unrollFactor) {
  assert(unrollFactor >= 1 && "unroll factor shoud be >= 1");

  if (unrollFactor == 1 || forStmt->getStatements().empty())
    return false;

  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(*forStmt);

  if (!mayBeConstantTripCount.hasValue() &&
      getLargestDivisorOfTripCount(*forStmt) % unrollFactor != 0)
    return false;

  const AffineBound &lb = forStmt->getLowerBound();
  const AffineBound &ub = forStmt->getLowerBound();
  auto lbMap = lb.getMap();
  auto ubMap = lb.getMap();

  // Loops with max/min expressions won't be unrolled here (the output can't be
  // expressed as an MLFunction in the general case). However, the right way to
  // do such unrolling for an MLFunction would be to specialize the loop for the
  // 'hotspot' case and unroll that hotspot case.
  if (lbMap->getNumResults() != 1 || ubMap->getNumResults() != 1)
    return false;

  // TODO(bondhugula): handle bounds with different sets of operands.
  // Same operand list for now.
  if (lbMap->getNumDims() != ubMap->getNumDims() ||
      lbMap->getNumSymbols() != ubMap->getNumSymbols())
    return false;
  unsigned i, e = lb.getNumOperands();
  for (i = 0; i < e; i++) {
    if (lb.getStmtOperand(i).get() != ub.getStmtOperand(i).get())
      break;
  }
  if (i != e)
    return false;

  int64_t step = forStmt->getStep();

  // If the trip count is lower than the unroll factor, no unrolled body.
  // TODO(bondhugula): option to specify cleanup loop unrolling.
  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() < unrollFactor)
    return false;

  // Generate the cleanup loop if trip count isn't a multiple of unrollFactor.
  // If the trip count is unknown, we currently unroll only when the unknown
  // trip count is known to be a multiple of unroll factor - hence, no cleanup
  // loop will be necessary in those cases.
  // TODO(bondhugula): handle generation of cleanup loop for unknown trip count
  // when it's not known to be a multiple of unroll factor (still for single
  // result / same operands case).
  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() % unrollFactor != 0) {
    uint64_t tripCount = mayBeConstantTripCount.getValue();
    DenseMap<const MLValue *, MLValue *> operandMap;
    MLFuncBuilder builder(forStmt->getBlock(), ++StmtBlock::iterator(forStmt));
    auto *cleanupForStmt = cast<ForStmt>(builder.clone(*forStmt, operandMap));
    if (forStmt->hasConstantLowerBound()) {
      cleanupForStmt->setConstantLowerBound(
          forStmt->getConstantLowerBound() +
          (tripCount - tripCount % unrollFactor) * step);
    } else {
      cleanupForStmt->setLowerBoundMap(
          getCleanupLoopLowerBound(forStmt->getLowerBoundMap(), tripCount,
                                   unrollFactor, step, &builder));
    }
    // Promote the loop body up if this has turned into a single iteration loop.
    promoteIfSingleIteration(cleanupForStmt);

    // The upper bound needs to be adjusted.
    if (forStmt->hasConstantUpperBound()) {
      forStmt->setConstantUpperBound(
          forStmt->getConstantLowerBound() +
          (tripCount - tripCount % unrollFactor - 1) * step);
    } else {
      forStmt->setUpperBoundMap(
          getUnrolledLoopUpperBound(forStmt->getLowerBoundMap(), tripCount,
                                    unrollFactor, step, &builder));
    }
  }

  // Scale the step of loop being unrolled by unroll factor.
  forStmt->setStep(step * unrollFactor);

  // Builder to insert unrolled bodies right after the last statement in the
  // body of 'forStmt'.
  MLFuncBuilder builder(forStmt, StmtBlock::iterator(forStmt->end()));

  // Keep a pointer to the last statement in the original block so that we know
  // what to clone (since we are doing this in-place).
  StmtBlock::iterator srcBlockEnd = std::prev(forStmt->end());

  // Unroll the contents of 'forStmt' (append unrollFactor-1 additional copies).
  for (unsigned i = 1; i < unrollFactor; i++) {
    DenseMap<const MLValue *, MLValue *> operandMap;

    // If the induction variable is used, create a remapping to the value for
    // this unrolled instance.
    if (!forStmt->use_empty()) {
      // iv' = iv + 1/2/3...unrollFactor-1;
      auto *bumpExpr = builder.getAddExpr(builder.getDimExpr(0),
                                          builder.getConstantExpr(i * step));
      auto *bumpMap = builder.getAffineMap(1, 0, {bumpExpr}, {});
      auto *ivUnroll =
          builder.create<AffineApplyOp>(forStmt->getLoc(), bumpMap, forStmt)
              ->getResult(0);
      operandMap[forStmt] = cast<MLValue>(ivUnroll);
    }

    // Clone the original body of 'forStmt'.
    for (auto it = forStmt->begin(); it != std::next(srcBlockEnd); it++) {
      builder.clone(*it, operandMap);
    }
  }

  // Promote the loop body up if this has turned into a single iteration loop.
  promoteIfSingleIteration(forStmt);

  return true;
}
