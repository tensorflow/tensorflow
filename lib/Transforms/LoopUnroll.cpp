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
#include "mlir/IR/Pass.h"
#include "mlir/IR/StandardOps.h"
#include "mlir/IR/Statements.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
struct LoopUnroll : public MLFunctionPass {
  void runOnMLFunction(MLFunction *f) override;
  void runOnForStmt(ForStmt *forStmt);
};
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

/// Unrolls all the innermost loops of this MLFunction.
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
      bool hasInnerLoops = walkPostOrder(ifStmt->getThenClause()->begin(),
                                         ifStmt->getThenClause()->end());
      hasInnerLoops |= walkPostOrder(ifStmt->getElseClause()->begin(),
                                     ifStmt->getElseClause()->end());
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

/// Unrolls all loops with trip count <= minTripCount.
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
  slg.walk(f);
  auto &loops = slg.loops;
  for (auto *forStmt : loops)
    runOnForStmt(forStmt);
}

/// Replace all uses of oldVal with newVal from begin to end.
static void replaceUses(StmtBlock::iterator begin, StmtBlock::iterator end,
                        MLValue *oldVal, MLValue *newVal) {
  // TODO(bondhugula,clattner): do this more efficiently by walking those uses
  // of oldVal that fall within this list of statements (instead of iterating
  // through all statements / through all operands of operations found).
  for (auto it = begin; it != end; it++) {
    it->replaceUses(oldVal, newVal);
  }
}

/// Replace all uses of oldVal with newVal.
void replaceUses(StmtBlock *block, MLValue *oldVal, MLValue *newVal) {
  // TODO(bondhugula,clattner): do this more efficiently by walking those uses
  // of oldVal that fall within this StmtBlock (instead of iterating through
  // all statements / through all operands of operations found).
  for (auto it = block->begin(); it != block->end(); it++) {
    it->replaceUses(oldVal, newVal);
  }
}

/// Clone the list of stmt's from 'block' and insert into the current
/// position of the builder.
// TODO(bondhugula,clattner): replace this with a parameterizable clone.
void cloneStmtListFromBlock(MLFuncBuilder *builder, const StmtBlock &block) {
  // Pairs of <old op stmt result whose uses need to be replaced,
  // new result generated by the corresponding cloned op stmt>.
  SmallVector<std::pair<MLValue *, MLValue *>, 8> oldNewResultPairs;

  // Iterator pointing to just before 'this' (i^th) unrolled iteration.
  StmtBlock::iterator beforeUnrolledBody = --builder->getInsertionPoint();

  for (auto &stmt : block.getStatements()) {
    auto *cloneStmt = builder->clone(stmt);
    // Whenever we have an op stmt, we'll have a new ML Value defined: replace
    // uses of the old result with this one.
    if (auto *opStmt = dyn_cast<OperationStmt>(&stmt)) {
      if (opStmt->getNumResults()) {
        auto *cloneOpStmt = cast<OperationStmt>(cloneStmt);
        for (unsigned i = 0, e = opStmt->getNumResults(); i < e; i++) {
          // Store old/new result pairs.
          // TODO(bondhugula) *only* if needed later: storing of old/new
          // results can be avoided by cloning the statement list in the
          // reverse direction (and running the IR builder in the reverse
          // (iplist.insertAfter()). That way, a newly created result can be
          // immediately propagated to all its uses.
          oldNewResultPairs.push_back(std::make_pair(
              const_cast<StmtResult *>(&opStmt->getStmtResult(i)),
              &cloneOpStmt->getStmtResult(i)));
        }
      }
    }
  }

  // Replace uses of old op results' with the new results.
  StmtBlock::iterator startOfUnrolledBody = ++beforeUnrolledBody;
  StmtBlock::iterator endOfUnrolledBody = builder->getInsertionPoint();

  // Replace uses of old op results' with the newly created ones.
  for (unsigned i = 0; i < oldNewResultPairs.size(); i++) {
    replaceUses(startOfUnrolledBody, endOfUnrolledBody,
                oldNewResultPairs[i].first, oldNewResultPairs[i].second);
  }
}

/// Unroll this 'for stmt' / loop completely.
void LoopUnroll::runOnForStmt(ForStmt *forStmt) {
  auto lb = forStmt->getLowerBound()->getValue();
  auto ub = forStmt->getUpperBound()->getValue();
  auto step = forStmt->getStep()->getValue();

  // Builder to add constants need for the unrolled iterator.
  auto *mlFunc = forStmt->Statement::findFunction();
  MLFuncBuilder funcTopBuilder(mlFunc);
  funcTopBuilder.setInsertionPointAtStart(mlFunc);

  // Builder to insert the unrolled bodies.
  MLFuncBuilder builder(forStmt->getBlock());
  // Set insertion point to right after where the for stmt ends.
  builder.setInsertionPoint(forStmt->getBlock(),
                            ++StmtBlock::iterator(forStmt));

  // Unroll the contents of 'forStmt'.
  for (int64_t i = lb; i <= ub; i += step) {
    MLValue *ivConst = nullptr;
    if (!forStmt->use_empty()) {
      auto constOp = funcTopBuilder.create<ConstantAffineIntOp>(i);
      ivConst = cast<OperationStmt>(constOp->getOperation())->getResult(0);
    }
    StmtBlock::iterator beforeUnrolledBody = --builder.getInsertionPoint();

    // Clone the loop body and insert it right after the loop - the latter will
    // be erased after all unrolling has been done.
    cloneStmtListFromBlock(&builder, *forStmt);

    // Replace unrolled loop IV with the unrolled constant.
    if (ivConst) {
      StmtBlock::iterator startOfUnrolledBody = ++beforeUnrolledBody;
      StmtBlock::iterator endOfUnrolledBody = builder.getInsertionPoint();
      replaceUses(startOfUnrolledBody, endOfUnrolledBody, forStmt, ivConst);
    }
  }
  // Erase the original 'for' stmt from the block.
  forStmt->eraseFromBlock();
}
