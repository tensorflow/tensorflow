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

#include "mlir/Transforms/Passes.h"

#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardOps.h"
#include "mlir/IR/Statements.h"
#include "mlir/IR/StmtVisitor.h"

/// Promotes the loop body of a forStmt to its containing block if the forStmt
/// was known to have a single iteration. Returns false otherwise.
// TODO(bondhugula): extend this for arbitrary affine bounds.
bool mlir::promoteIfSingleIteration(ForStmt *forStmt) {
  Optional<uint64_t> tripCount = getConstantTripCount(*forStmt);
  if (!tripCount.hasValue() || !forStmt->hasConstantLowerBound())
    return false;

  if (tripCount.getValue() != 1)
    return false;

  // Replaces all IV uses to its single iteration value.
  auto *mlFunc = forStmt->findFunction();
  MLFuncBuilder topBuilder(&mlFunc->front());
  auto constOp = topBuilder.create<ConstantAffineIntOp>(
      forStmt->getLoc(), forStmt->getConstantLowerBound());
  forStmt->replaceAllUsesWith(constOp->getResult());
  // Move the statements to the containing block.
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
