//===- LoopUtils.cpp ---- Misc utilities for loop transformation ----------===//
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
// This file implements miscellaneous loop transformation routines.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/LoopUtils.h"

#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Statements.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/StandardOps/StandardOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "LoopUtils"

using namespace mlir;

/// Returns the upper bound of an unrolled loop with lower bound 'lb' and with
/// the specified trip count, stride, and unroll factor. Returns nullptr when
/// the trip count can't be expressed as an affine expression.
AffineMap mlir::getUnrolledLoopUpperBound(const ForStmt &forStmt,
                                          unsigned unrollFactor,
                                          MLFuncBuilder *builder) {
  auto lbMap = forStmt.getLowerBoundMap();

  // Single result lower bound map only.
  if (lbMap.getNumResults() != 1)
    return AffineMap::Null();

  // Sometimes, the trip count cannot be expressed as an affine expression.
  auto tripCount = getTripCountExpr(forStmt);
  if (!tripCount)
    return AffineMap::Null();

  AffineExpr lb(lbMap.getResult(0));
  unsigned step = forStmt.getStep();
  auto newUb = lb + (tripCount - tripCount % unrollFactor - 1) * step;

  return builder->getAffineMap(lbMap.getNumDims(), lbMap.getNumSymbols(),
                               {newUb}, {});
}

/// Returns the lower bound of the cleanup loop when unrolling a loop with lower
/// bound 'lb' and with the specified trip count, stride, and unroll factor.
/// Returns an AffinMap with nullptr storage (that evaluates to false)
/// when the trip count can't be expressed as an affine expression.
AffineMap mlir::getCleanupLoopLowerBound(const ForStmt &forStmt,
                                         unsigned unrollFactor,
                                         MLFuncBuilder *builder) {
  auto lbMap = forStmt.getLowerBoundMap();

  // Single result lower bound map only.
  if (lbMap.getNumResults() != 1)
    return AffineMap::Null();

  // Sometimes the trip count cannot be expressed as an affine expression.
  AffineExpr tripCount(getTripCountExpr(forStmt));
  if (!tripCount)
    return AffineMap::Null();

  AffineExpr lb(lbMap.getResult(0));
  unsigned step = forStmt.getStep();
  auto newLb = lb + (tripCount - tripCount % unrollFactor) * step;
  return builder->getAffineMap(lbMap.getNumDims(), lbMap.getNumSymbols(),
                               {newLb}, {});
}

/// Promotes the loop body of a forStmt to its containing block if the forStmt
/// was known to have a single iteration. Returns false otherwise.
// TODO(bondhugula): extend this for arbitrary affine bounds.
bool mlir::promoteIfSingleIteration(ForStmt *forStmt) {
  Optional<uint64_t> tripCount = getConstantTripCount(*forStmt);
  if (!tripCount.hasValue() || tripCount.getValue() != 1)
    return false;

  // TODO(mlir-team): there is no builder for a max.
  if (forStmt->getLowerBoundMap().getNumResults() != 1)
    return false;

  // Replaces all IV uses to its single iteration value.
  if (!forStmt->use_empty()) {
    if (forStmt->hasConstantLowerBound()) {
      auto *mlFunc = forStmt->findFunction();
      MLFuncBuilder topBuilder(&mlFunc->front());
      auto constOp = topBuilder.create<ConstantIndexOp>(
          forStmt->getLoc(), forStmt->getConstantLowerBound());
      forStmt->replaceAllUsesWith(constOp);
    } else {
      const AffineBound lb = forStmt->getLowerBound();
      SmallVector<SSAValue *, 4> lbOperands(lb.operand_begin(),
                                            lb.operand_end());
      MLFuncBuilder builder(forStmt->getBlock(), StmtBlock::iterator(forStmt));
      auto affineApplyOp = builder.create<AffineApplyOp>(
          forStmt->getLoc(), lb.getMap(), lbOperands);
      forStmt->replaceAllUsesWith(affineApplyOp->getResult(0));
    }
  }
  // Move the loop body statements to the loop's containing block.
  auto *block = forStmt->getBlock();
  block->getStatements().splice(StmtBlock::iterator(forStmt),
                                forStmt->getStatements());
  forStmt->erase();
  return true;
}

/// Promotes all single iteration for stmt's in the MLFunction, i.e., moves
/// their body into the containing StmtBlock.
void mlir::promoteSingleIterationLoops(MLFunction *f) {
  // Gathers all innermost loops through a post order pruned walk.
  class LoopBodyPromoter : public StmtWalker<LoopBodyPromoter> {
  public:
    void visitForStmt(ForStmt *forStmt) { promoteIfSingleIteration(forStmt); }
  };

  LoopBodyPromoter fsw;
  fsw.walkPostOrder(f);
}

/// Generates a for 'stmt' with the specified lower and upper bounds while
/// generating the right IV remappings for the delayed statements. The
/// statement blocks that go into the loop are specified in stmtGroupQueue
/// starting from the specified offset, and in that order; the first element of
/// the pair specifies the delay applied to that group of statements. Returns
/// nullptr if the generated loop simplifies to a single iteration one.
static ForStmt *
generateLoop(AffineMap lb, AffineMap ub,
             const std::vector<std::pair<uint64_t, ArrayRef<Statement *>>>
                 &stmtGroupQueue,
             unsigned offset, ForStmt *srcForStmt, MLFuncBuilder *b) {
  SmallVector<MLValue *, 4> lbOperands(srcForStmt->getLowerBoundOperands());
  SmallVector<MLValue *, 4> ubOperands(srcForStmt->getUpperBoundOperands());

  auto *loopChunk =
      b->createFor(srcForStmt->getLoc(), lbOperands, lb, ubOperands, ub);
  OperationStmt::OperandMapTy operandMap;

  for (auto it = stmtGroupQueue.begin() + offset, e = stmtGroupQueue.end();
       it != e; ++it) {
    auto elt = *it;
    // All 'same delay' statements get added with the operands being remapped
    // (to results of cloned statements).
    // Generate the remapping if the delay is not zero: oldIV = newIV - delay.
    // TODO(bondhugula): check if srcForStmt is actually used in elt.second
    // instead of just checking if it's used at all.
    if (!srcForStmt->use_empty() && elt.first != 0) {
      auto b = MLFuncBuilder::getForStmtBodyBuilder(loopChunk);
      auto *oldIV =
          b.create<AffineApplyOp>(
               srcForStmt->getLoc(),
               b.getSingleDimShiftAffineMap(-static_cast<int64_t>(elt.first)),
               loopChunk)
              ->getResult(0);
      operandMap[srcForStmt] = cast<MLValue>(oldIV);
    } else {
      operandMap[srcForStmt] = static_cast<MLValue *>(loopChunk);
    }
    for (auto *stmt : elt.second) {
      loopChunk->push_back(stmt->clone(operandMap, b->getContext()));
    }
  }
  if (promoteIfSingleIteration(loopChunk))
    return nullptr;
  return loopChunk;
}

/// Skew the statements in the body of a 'for' statement with the specified
/// statement-wise delays. The delays are with respect to the original execution
/// order. A delay of zero for each statement will lead to no change.
// The skewing of statements with respect to one another can be used for example
// to allow overlap of asynchronous operations (such as DMA communication) with
// computation, or just relative shifting of statements for better register
// reuse, locality or parallelism. As such, the delays are typically expected to
// be at most of the order of the number of statements. This method should not
// be used as a substitute for loop distribution/fission.
// This method uses an algorithm// in time linear in the number of statements in
// the body of the for loop - (using the 'sweep line' paradigm). This method
// asserts preservation of SSA dominance. A check for that as well as that for
// memory-based depedence preservation check rests with the users of this
// method.
UtilResult mlir::stmtBodySkew(ForStmt *forStmt, ArrayRef<uint64_t> delays,
                              bool unrollPrologueEpilogue) {
  if (forStmt->getStatements().empty())
    return UtilResult::Success;

  // If the trip counts aren't constant, we would need versioning and
  // conditional guards (or context information to prevent such versioning). The
  // better way to pipeline for such loops is to first tile them and extract
  // constant trip count "full tiles" before applying this.
  auto mayBeConstTripCount = getConstantTripCount(*forStmt);
  if (!mayBeConstTripCount.hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "non-constant trip count loop\n";);
    return UtilResult::Success;
  }
  uint64_t tripCount = mayBeConstTripCount.getValue();

  assert(isStmtwiseShiftValid(*forStmt, delays) &&
         "shifts will lead to an invalid transformation\n");

  unsigned numChildStmts = forStmt->getStatements().size();

  // Do a linear time (counting) sort for the delays.
  uint64_t maxDelay = 0;
  for (unsigned i = 0; i < numChildStmts; i++) {
    maxDelay = std::max(maxDelay, delays[i]);
  }
  // Such large delays are not the typical use case.
  if (maxDelay >= numChildStmts) {
    LLVM_DEBUG(llvm::dbgs() << "stmt delays too large - unexpected\n";);
    return UtilResult::Success;
  }

  // An array of statement groups sorted by delay amount; each group has all
  // statements with the same delay in the order in which they appear in the
  // body of the 'for' stmt.
  std::vector<std::vector<Statement *>> sortedStmtGroups(maxDelay + 1);
  unsigned pos = 0;
  for (auto &stmt : *forStmt) {
    auto delay = delays[pos++];
    sortedStmtGroups[delay].push_back(&stmt);
  }

  // Unless the shifts have a specific pattern (which actually would be the
  // common use case), prologue and epilogue are not meaningfully defined.
  // Nevertheless, if 'unrollPrologueEpilogue' is set, we will treat the first
  // loop generated as the prologue and the last as epilogue and unroll these
  // fully.
  ForStmt *prologue = nullptr;
  ForStmt *epilogue = nullptr;

  // Do a sweep over the sorted delays while storing open groups in a
  // vector, and generating loop portions as necessary during the sweep. A block
  // of statements is paired with its delay.
  std::vector<std::pair<uint64_t, ArrayRef<Statement *>>> stmtGroupQueue;

  auto origLbMap = forStmt->getLowerBoundMap();
  uint64_t lbDelay = 0;
  MLFuncBuilder b(forStmt);
  for (uint64_t d = 0, e = sortedStmtGroups.size(); d < e; ++d) {
    // If nothing is delayed by d, continue.
    if (sortedStmtGroups[d].empty())
      continue;
    if (!stmtGroupQueue.empty()) {
      assert(d >= 1 &&
             "Queue expected to be empty when the first block is found");
      // The interval for which the loop needs to be generated here is:
      // ( lbDelay, min(lbDelay + tripCount, d)) and the body of the
      // loop needs to have all statements in stmtQueue in that order.
      ForStmt *res;
      if (lbDelay + tripCount < d) {
        res =
            generateLoop(b.getShiftedAffineMap(origLbMap, lbDelay),
                         b.getShiftedAffineMap(origLbMap, lbDelay + tripCount),
                         stmtGroupQueue, 0, forStmt, &b);
        // Entire loop for the queued stmt groups generated, empty it.
        stmtGroupQueue.clear();
        lbDelay += tripCount;
      } else {
        res = generateLoop(b.getShiftedAffineMap(origLbMap, lbDelay),
                           b.getShiftedAffineMap(origLbMap, d), stmtGroupQueue,
                           0, forStmt, &b);
        lbDelay = d;
      }
      if (!prologue && res)
        prologue = res;
      epilogue = res;
    } else {
      // Start of first interval.
      lbDelay = d;
    }
    // Augment the list of statements that get into the current open interval.
    stmtGroupQueue.push_back({d, sortedStmtGroups[d]});
  }

  // Those statements groups left in the queue now need to be processed (FIFO)
  // and their loops completed.
  for (unsigned i = 0, e = stmtGroupQueue.size(); i < e; ++i) {
    uint64_t ubDelay = stmtGroupQueue[i].first + tripCount;
    epilogue = generateLoop(b.getShiftedAffineMap(origLbMap, lbDelay),
                            b.getShiftedAffineMap(origLbMap, ubDelay),
                            stmtGroupQueue, i, forStmt, &b);
    lbDelay = ubDelay;
    if (!prologue)
      prologue = epilogue;
  }

  // Erase the original for stmt.
  forStmt->erase();

  if (unrollPrologueEpilogue && prologue)
    loopUnrollFull(prologue);
  if (unrollPrologueEpilogue && !epilogue && epilogue != prologue)
    loopUnrollFull(epilogue);

  return UtilResult::Success;
}

/// Unrolls this loop completely.
bool mlir::loopUnrollFull(ForStmt *forStmt) {
  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(*forStmt);
  if (mayBeConstantTripCount.hasValue()) {
    uint64_t tripCount = mayBeConstantTripCount.getValue();
    if (tripCount == 1) {
      return promoteIfSingleIteration(forStmt);
    }
    return loopUnrollByFactor(forStmt, tripCount);
  }
  return false;
}

/// Unrolls and jams this loop by the specified factor or by the trip count (if
/// constant) whichever is lower.
bool mlir::loopUnrollUpToFactor(ForStmt *forStmt, uint64_t unrollFactor) {
  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(*forStmt);

  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() < unrollFactor)
    return loopUnrollByFactor(forStmt, mayBeConstantTripCount.getValue());
  return loopUnrollByFactor(forStmt, unrollFactor);
}

/// Unrolls this loop by the specified factor. Returns true if the loop
/// is successfully unrolled.
bool mlir::loopUnrollByFactor(ForStmt *forStmt, uint64_t unrollFactor) {
  assert(unrollFactor >= 1 && "unroll factor should be >= 1");

  if (unrollFactor == 1 || forStmt->getStatements().empty())
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
  // TODO(bondhugula): handle bounds with different operand lists.
  if (!forStmt->matchingBoundOperandList())
    return false;

  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(*forStmt);

  // If the trip count is lower than the unroll factor, no unrolled body.
  // TODO(bondhugula): option to specify cleanup loop unrolling.
  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() < unrollFactor)
    return false;

  // Generate the cleanup loop if trip count isn't a multiple of unrollFactor.
  if (getLargestDivisorOfTripCount(*forStmt) % unrollFactor != 0) {
    DenseMap<const MLValue *, MLValue *> operandMap;
    MLFuncBuilder builder(forStmt->getBlock(), ++StmtBlock::iterator(forStmt));
    auto *cleanupForStmt = cast<ForStmt>(builder.clone(*forStmt, operandMap));
    auto clLbMap = getCleanupLoopLowerBound(*forStmt, unrollFactor, &builder);
    assert(clLbMap &&
           "cleanup loop lower bound map for single result bound maps can "
           "always be determined");
    cleanupForStmt->setLowerBoundMap(clLbMap);
    // Promote the loop body up if this has turned into a single iteration loop.
    promoteIfSingleIteration(cleanupForStmt);

    // Adjust upper bound.
    auto unrolledUbMap =
        getUnrolledLoopUpperBound(*forStmt, unrollFactor, &builder);
    assert(unrolledUbMap &&
           "upper bound map can alwayys be determined for an unrolled loop "
           "with single result bounds");
    forStmt->setUpperBoundMap(unrolledUbMap);
  }

  // Scale the step of loop being unrolled by unroll factor.
  int64_t step = forStmt->getStep();
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
      auto d0 = builder.getAffineDimExpr(0);
      auto bumpMap = builder.getAffineMap(1, 0, {d0 + i * step}, {});
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
