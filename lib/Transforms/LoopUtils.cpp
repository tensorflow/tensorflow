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
#include "mlir/IR/StandardOps.h"
#include "mlir/IR/Statements.h"
#include "mlir/IR/StmtVisitor.h"
#include "llvm/ADT/DenseMap.h"

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
    return AffineMap::Invalid();

  // Sometimes, the trip count cannot be expressed as an affine expression.
  auto tripCount = getTripCountExpr(forStmt);
  if (!tripCount)
    return AffineMap::Invalid();

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
    return AffineMap::Invalid();

  // Sometimes the trip count cannot be expressed as an affine expression.
  AffineExpr tripCount(getTripCountExpr(forStmt));
  if (!tripCount)
    return AffineMap::Invalid();

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
      forStmt->replaceAllUsesWith(constOp->getResult());
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
  forStmt->eraseFromBlock();
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

// Returns delay of that child statement of 'forStmt' which either has 'operand'
// as one of its operands or has a descendant statement with operand 'operand'.
// This is a naive implementation. If performance becomes an issue, a map can
// be used to store 'delays' - to look up the delay for a statement in constant
// time.
static uint64_t getContainingStmtDelay(const StmtOperand &operand,
                                       const ForStmt &forStmt,
                                       ArrayRef<uint64_t> delays) {
  // Traverse up the statement hierarchy starting from the owner of operand to
  // find the ancestor statement that resides in the block of 'forStmt'.
  const Statement *stmt = operand.getOwner();
  assert(stmt != nullptr);
  while (stmt->getParentStmt() != &forStmt) {
    stmt = stmt->getParentStmt();
    assert(stmt && "traversing parent's should reach forStmt block");
  }
  // Look up the delay of 'stmt'.
  unsigned j = 0;
  for (const auto &s : forStmt) {
    if (&s == stmt)
      break;
    j++;
  }
  assert(j < forStmt.getStatements().size() && "child stmt should be found");
  return delays[j];
}

/// Checks if SSA dominance would be violated if a for stmt's body statements
/// are shifted by the specified delays. This method checks if a 'def' and all
/// its uses have the same delay factor.
bool mlir::checkDominancePreservationOnShift(const ForStmt &forStmt,
                                             ArrayRef<uint64_t> delays) {
  assert(delays.size() == forStmt.getStatements().size());
  unsigned s = 0;
  for (const auto &stmt : forStmt) {
    // A for or if stmt does not produce any def/results (that are used
    // outside).
    if (auto *opStmt = dyn_cast<OperationStmt>(&stmt)) {
      for (unsigned i = 0, e = opStmt->getNumResults(); i < e; ++i) {
        const MLValue *result = opStmt->getResult(i);
        for (const StmtOperand &use : result->getUses()) {
          if (delays[s] != getContainingStmtDelay(use, forStmt, delays))
            return false;
        }
      }
    }
    s++;
  }
  return true;
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
  if (!mayBeConstTripCount.hasValue())
    return UtilResult::Failure;
  uint64_t tripCount = mayBeConstTripCount.getValue();

  assert(checkDominancePreservationOnShift(*forStmt, delays) &&
         "dominance preservation failed\n");

  unsigned numChildStmts = forStmt->getStatements().size();

  // Do a linear time (counting) sort for the delays.
  uint64_t maxDelay = 0;
  for (unsigned i = 0; i < numChildStmts; i++) {
    maxDelay = std::max(maxDelay, delays[i]);
  }
  // Such large delays are not the typical use case.
  if (maxDelay >= numChildStmts)
    return UtilResult::Failure;

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
      // ( lbDelay, min(lbDelay + tripCount - 1, d - 1) ] and the body of the
      // loop needs to have all statements in stmtQueue in that order.
      ForStmt *res;
      if (lbDelay + tripCount - 1 < d - 1) {
        res = generateLoop(
            b.getShiftedAffineMap(origLbMap, lbDelay),
            b.getShiftedAffineMap(origLbMap, lbDelay + tripCount - 1),
            stmtGroupQueue, 0, forStmt, &b);
        // Entire loop for the queued stmt groups generated, empty it.
        stmtGroupQueue.clear();
        lbDelay += tripCount;
      } else {
        res = generateLoop(b.getShiftedAffineMap(origLbMap, lbDelay),
                           b.getShiftedAffineMap(origLbMap, d - 1),
                           stmtGroupQueue, 0, forStmt, &b);
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
    uint64_t ubDelay = stmtGroupQueue[i].first + tripCount - 1;
    epilogue = generateLoop(b.getShiftedAffineMap(origLbMap, lbDelay),
                            b.getShiftedAffineMap(origLbMap, ubDelay),
                            stmtGroupQueue, i, forStmt, &b);
    lbDelay = ubDelay + 1;
    if (!prologue)
      prologue = epilogue;
  }

  // Erase the original for stmt.
  forStmt->eraseFromBlock();

  if (unrollPrologueEpilogue && prologue)
    loopUnrollFull(prologue);
  if (unrollPrologueEpilogue && !epilogue && epilogue != prologue)
    loopUnrollFull(epilogue);

  return UtilResult::Success;
}
