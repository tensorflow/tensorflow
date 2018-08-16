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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OperationSet.h"
#include "mlir/IR/StandardOps.h"
#include "mlir/IR/Statements.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/Transforms/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
/// Loop unrolling pass. For now, this unrolls all the innermost loops of this
/// MLFunction.
struct LoopUnroll : public MLFunctionPass {
  void runOnMLFunction(MLFunction *f) override;
  void runOnForStmt(ForStmt *forStmt);
};

/// Unrolls all loops with trip count <= minTripCount.
struct ShortLoopUnroll : public LoopUnroll {
  const unsigned minTripCount;
  void runOnMLFunction(MLFunction *f) override;
  ShortLoopUnroll(unsigned minTripCount) : minTripCount(minTripCount) {}
};
} // end anonymous namespace

MLFunctionPass *mlir::createLoopUnrollPass() { return new LoopUnroll(); }

MLFunctionPass *mlir::createLoopUnrollPass(unsigned minTripCount) {
  return new ShortLoopUnroll(minTripCount);
}

void LoopUnroll::runOnMLFunction(MLFunction *f) {
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

  InnermostLoopGatherer ilg;
  ilg.walkPostOrder(f);
  auto &loops = ilg.loops;
  for (auto *forStmt : loops)
    runOnForStmt(forStmt);
}

void ShortLoopUnroll::runOnMLFunction(MLFunction *f) {
  // Gathers all loops with trip count <= minTripCount.
  class ShortLoopGatherer : public StmtWalker<ShortLoopGatherer> {
  public:
    // Store short loops as we walk.
    std::vector<ForStmt *> loops;
    const unsigned minTripCount;
    ShortLoopGatherer(unsigned minTripCount) : minTripCount(minTripCount) {}

    void visitForStmt(ForStmt *forStmt) {
      auto lb = forStmt->getLowerBound()->getValue();
      auto ub = forStmt->getUpperBound()->getValue();
      auto step = forStmt->getStep()->getValue();

      if ((ub - lb) / step + 1 <= minTripCount)
        loops.push_back(forStmt);
    }
  };

  ShortLoopGatherer slg(minTripCount);
  // Do a post order walk so that loops are gathered from innermost to
  // outermost (or else unrolling an outer one may delete gathered inner ones).
  slg.walkPostOrder(f);
  auto &loops = slg.loops;
  for (auto *forStmt : loops)
    runOnForStmt(forStmt);
}

/// Unroll this For loop completely.
void LoopUnroll::runOnForStmt(ForStmt *forStmt) {
  auto lb = forStmt->getLowerBound()->getValue();
  auto ub = forStmt->getUpperBound()->getValue();
  auto step = forStmt->getStep()->getValue();

  // Builder to add constants need for the unrolled iterator.
  auto *mlFunc = forStmt->findFunction();
  MLFuncBuilder funcTopBuilder(&mlFunc->front());

  // Builder to insert the unrolled bodies.  We insert right after the
  /// ForStmt we're unrolling.
  MLFuncBuilder builder(forStmt->getBlock(), ++StmtBlock::iterator(forStmt));

  // Unroll the contents of 'forStmt'.
  for (int64_t i = lb; i <= ub; i += step) {
    DenseMap<const MLValue *, MLValue *> operandMapping;

    // If the induction variable is used, create a constant for this unrolled
    // value and add an operand mapping for it.
    if (!forStmt->use_empty()) {
      auto *ivConst =
          funcTopBuilder.create<ConstantAffineIntOp>(i)->getResult();
      operandMapping[forStmt] = cast<MLValue>(ivConst);
    }

    // Clone the body of the loop.
    for (auto &childStmt : *forStmt) {
      (void)builder.clone(childStmt, operandMapping);
    }
  }
  // Erase the original 'for' stmt from the block.
  forStmt->eraseFromBlock();
}
