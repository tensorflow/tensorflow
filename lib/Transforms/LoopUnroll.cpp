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

/// Replace an IV with a constant value.
static void replaceIterator(Statement *stmt, const ForStmt &iv,
                            MLValue *constVal) {
  struct ReplaceIterator : public StmtWalker<ReplaceIterator> {
    // IV to be replaced.
    const ForStmt *iv;
    // Constant to be replaced with.
    MLValue *constVal;

    ReplaceIterator(const ForStmt &iv, MLValue *constVal)
        : iv(&iv), constVal(constVal){};

    void visitOperationStmt(OperationStmt *os) {
      for (auto &operand : os->getStmtOperands()) {
        if (operand.get() == static_cast<const MLValue *>(iv)) {
          operand.set(constVal);
        }
      }
    }
  };

  ReplaceIterator ri(iv, constVal);
  ri.walk(stmt);
}

/// Unrolls this loop completely.
void LoopUnroll::runOnForStmt(ForStmt *forStmt) {
  auto lb = forStmt->getLowerBound()->getValue();
  auto ub = forStmt->getUpperBound()->getValue();
  auto step = forStmt->getStep()->getValue();
  auto trip_count = (ub - lb + 1) / step;

  auto *mlFunc = forStmt->Statement::findFunction();
  MLFuncBuilder funcTopBuilder(mlFunc);
  funcTopBuilder.setInsertionPointAtStart(mlFunc);

  MLFuncBuilder builder(forStmt->getBlock());
  for (int i = 0; i < trip_count; i++) {
    auto *ivUnrolledVal = funcTopBuilder.createConstInt32Op(i)->getResult(0);
    for (auto &stmt : forStmt->getStatements()) {
      switch (stmt.getKind()) {
      case Statement::Kind::For:
        llvm_unreachable("unrolling loops that have only operations");
        break;
      case Statement::Kind::If:
        llvm_unreachable("unrolling loops that have only operations");
        break;
      case Statement::Kind::Operation:
        auto *cloneOp = builder.cloneOperation(*cast<OperationStmt>(&stmt));
        // TODO(bondhugula): only generate constants when the IV actually
        // appears in the body.
        replaceIterator(cloneOp, *forStmt, ivUnrolledVal);
        break;
      }
    }
  }
  forStmt->eraseFromBlock();
}
