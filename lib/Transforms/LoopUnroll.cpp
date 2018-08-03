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
} // end anonymous namespace

MLFunctionPass *mlir::createLoopUnrollPass() { return new LoopUnroll(); }

/// Unrolls all the innermost loops of this MLFunction.
void LoopUnroll::runOnMLFunction(MLFunction *f) {
  // Gathers all innermost loops through a post order pruned walk.
  // TODO: figure out the right reusable template here to better refactor code.
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

    // FIXME: can't use base class method for this because that in turn would
    // need to use the derived class method above. CRTP doesn't allow it, and
    // the compiler error resulting from it is also very misleading!
    void walkPostOrder(MLFunction *f) { walkPostOrder(f->begin(), f->end()); }

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

    bool walkOpStmt(OperationStmt *opStmt) { return false; }

    using StmtWalker<InnermostLoopGatherer, bool>::walkPostOrder;
  };

  InnermostLoopGatherer ilg;
  ilg.walkPostOrder(f);
  auto &loops = ilg.loops;
  for (auto *forStmt : loops)
    runOnForStmt(forStmt);
}

/// Replace all uses of 'oldVal' with 'newVal' in 'stmt'
static void replaceAllStmtUses(Statement *stmt, MLValue *oldVal,
                               MLValue *newVal) {
  struct ReplaceUseWalker : public StmtWalker<ReplaceUseWalker> {
    // Value to be replaced.
    MLValue *oldVal;
    // Value to be replaced with.
    MLValue *newVal;

    ReplaceUseWalker(MLValue *oldVal, MLValue *newVal)
        : oldVal(oldVal), newVal(newVal){};

    void visitOperationStmt(OperationStmt *os) {
      for (auto &operand : os->getStmtOperands()) {
        if (operand.get() == oldVal)
          operand.set(newVal);
      }
    }
  };

  ReplaceUseWalker ri(oldVal, newVal);
  ri.walk(stmt);
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
  for (int i = lb; i <= ub; i += step) {
    // TODO(bondhugula): generate constants only when IV actually appears.
    auto constOp = funcTopBuilder.create<ConstantIntOp>(i, 32);
    auto *ivConst = cast<OperationStmt>(constOp->getOperation())->getResult(0);

    // Iterator pointing to just before 'this' (i^th) unrolled iteration.
    StmtBlock::iterator beforeUnrolledBody = --builder.getInsertionPoint();

    // Pairs of <old op stmt result whose uses need to be replaced,
    // new result generated by the corresponding cloned op stmt>.
    SmallVector<std::pair<MLValue *, MLValue *>, 8> oldNewResultPairs;

    for (auto &loopBodyStmt : forStmt->getStatements()) {
      auto *cloneStmt = builder.clone(loopBodyStmt);
      // Replace all uses of the IV in the clone with constant iteration value.
      replaceAllStmtUses(cloneStmt, forStmt, ivConst);

      // Whenever we have an op stmt, we'll have a new ML Value defined: replace
      // uses of the old result with this one.
      if (auto *opStmt = dyn_cast<OperationStmt>(&loopBodyStmt)) {
        if (opStmt->getNumResults()) {
          auto *cloneOpStmt = cast<OperationStmt>(cloneStmt);
          for (unsigned i = 0, e = opStmt->getNumResults(); i < e; i++) {
            // Store old/new result pairs.
            // TODO *only* if needed later: storing of old/new results can be
            // avoided, by cloning the statement list in the reverse direction
            // (and running the IR builder in the reverse
            // (iplist.insertAfter()). That way, a newly created result can be
            // immediately propagated to all its uses, which would already  been
            // cloned/inserted.
            oldNewResultPairs.push_back(std::make_pair(
                &opStmt->getStmtResult(i), &cloneOpStmt->getStmtResult(i)));
          }
        }
      }
    }
    // Replace uses of old op results' with the results in the just
    // unrolled body.
    StmtBlock::iterator endOfUnrolledBody = builder.getInsertionPoint();
    for (auto it = ++beforeUnrolledBody; it != endOfUnrolledBody; it++) {
      for (unsigned i = 0; i < oldNewResultPairs.size(); i++) {
        replaceAllStmtUses(&(*it), oldNewResultPairs[i].first,
                           oldNewResultPairs[i].second);
      }
    }
  }
  // Erase the original for stmt from the block.
  forStmt->eraseFromBlock();
}
