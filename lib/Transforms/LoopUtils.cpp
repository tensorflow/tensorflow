//===- LoopUtils.cpp - Misc loop utilities for simplification //-----------===//
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
// This file implements miscellaneous loop simplification routines.
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

using namespace mlir;

/// Returns the upper bound of an unrolled loop with lower bound 'lb' and with
/// the specified trip count, stride, and unroll factor. Returns nullptr when
/// the trip count can't be expressed as an affine expression.
AffineMap *mlir::getUnrolledLoopUpperBound(const ForStmt &forStmt,
                                           unsigned unrollFactor,
                                           MLFuncBuilder *builder) {
  auto *lbMap = forStmt.getLowerBoundMap();

  // Single result lower bound map only.
  if (lbMap->getNumResults() != 1)
    return nullptr;

  // Sometimes, the trip count cannot be expressed as an affine expression.
  auto *tripCountExpr = getTripCountExpr(forStmt);
  if (!tripCountExpr)
    return nullptr;

  AffineExpr *newUbExpr;
  auto *lbExpr = lbMap->getResult(0);
  int64_t step = forStmt.getStep();
  // lbExpr + (count - count % unrollFactor - 1) * step).
  if (auto *cTripCountExpr = dyn_cast<AffineConstantExpr>(tripCountExpr)) {
    uint64_t tripCount = static_cast<uint64_t>(cTripCountExpr->getValue());
    newUbExpr = builder->getAddExpr(
        lbExpr, builder->getConstantExpr(
                    (tripCount - tripCount % unrollFactor - 1) * step));
  } else {
    newUbExpr = builder->getAddExpr(
        lbExpr, builder->getMulExpr(
                    builder->getSubExpr(
                        builder->getSubExpr(
                            tripCountExpr,
                            builder->getModExpr(tripCountExpr, unrollFactor)),
                        1),
                    step));
  }
  return builder->getAffineMap(lbMap->getNumDims(), lbMap->getNumSymbols(),
                               {newUbExpr}, {});
}

/// Returns the lower bound of the cleanup loop when unrolling a loop with lower
/// bound 'lb' and with the specified trip count, stride, and unroll factor.
/// Returns nullptr when the trip count can't be expressed as an affine
/// expression.
AffineMap *mlir::getCleanupLoopLowerBound(const ForStmt &forStmt,
                                          unsigned unrollFactor,
                                          MLFuncBuilder *builder) {
  auto *lbMap = forStmt.getLowerBoundMap();

  // Single result lower bound map only.
  if (lbMap->getNumResults() != 1)
    return nullptr;

  // Sometimes the trip count cannot be expressed as an affine expression.
  auto *tripCountExpr = getTripCountExpr(forStmt);
  if (!tripCountExpr)
    return nullptr;

  AffineExpr *newLbExpr;
  auto *lbExpr = lbMap->getResult(0);
  int64_t step = forStmt.getStep();

  // lbExpr + (count - count % unrollFactor) * step);
  if (auto *cTripCountExpr = dyn_cast<AffineConstantExpr>(tripCountExpr)) {
    uint64_t tripCount = static_cast<uint64_t>(cTripCountExpr->getValue());
    newLbExpr = builder->getAddExpr(
        lbExpr, builder->getConstantExpr(
                    (tripCount - tripCount % unrollFactor) * step));
  } else {
    newLbExpr = builder->getAddExpr(
        lbExpr, builder->getMulExpr(
                    builder->getSubExpr(
                        tripCountExpr,
                        builder->getModExpr(tripCountExpr, unrollFactor)),
                    step));
  }
  return builder->getAffineMap(lbMap->getNumDims(), lbMap->getNumSymbols(),
                               {newLbExpr}, {});
}

/// Promotes the loop body of a forStmt to its containing block if the forStmt
/// was known to have a single iteration. Returns false otherwise.
// TODO(bondhugula): extend this for arbitrary affine bounds.
bool mlir::promoteIfSingleIteration(ForStmt *forStmt) {
  Optional<uint64_t> tripCount = getConstantTripCount(*forStmt);
  if (!tripCount.hasValue() || tripCount.getValue() != 1)
    return false;

  // TODO(mlir-team): there is no builder for a max.
  if (forStmt->getLowerBoundMap()->getNumResults() != 1)
    return false;

  // Replaces all IV uses to its single iteration value.
  if (!forStmt->use_empty()) {
    if (forStmt->hasConstantLowerBound()) {
      auto *mlFunc = forStmt->findFunction();
      MLFuncBuilder topBuilder(&mlFunc->front());
      auto constOp = topBuilder.create<ConstantAffineIntOp>(
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
