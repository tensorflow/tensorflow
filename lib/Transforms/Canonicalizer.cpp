//===- Canonicalizer.cpp - Canonicalize MLIR operations -------------------===//
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
// This transformation pass converts operations into their canonical forms by
// folding constants, applying operation identity transformations etc.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Transforms/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// Definition of a few patterns for canonicalizing operations.
//===----------------------------------------------------------------------===//

namespace {
/// subi(x,x) -> 0
///
struct SimplifyXMinusX : public Pattern {
  SimplifyXMinusX(MLIRContext *context)
      // FIXME: rename getOperationName and add a proper one.
      : Pattern(OperationName(SubIOp::getOperationName(), context), 1) {}

  std::pair<PatternBenefit, std::unique_ptr<PatternState>>
  match(Operation *op) const override {
    auto subi = op->dyn_cast<SubIOp>();
    assert(subi && "Matcher should have produced this");

    if (subi->getOperand(0) == subi->getOperand(1))
      return matchSuccess();

    return matchFailure();
  }

  // Rewrite the IR rooted at the specified operation with the result of
  // this pattern, generating any new operations with the specified
  // builder.  If an unexpected error is encountered (an internal
  // compiler error), it is emitted through the normal MLIR diagnostic
  // hooks and the IR is left in a valid state.
  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto subi = op->dyn_cast<SubIOp>();
    assert(subi && "Matcher should have produced this");

    auto result =
        rewriter.create<ConstantIntOp>(op->getLoc(), 0, subi->getType());

    rewriter.replaceSingleResultOp(op, result);
  }
};
} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// The actual Canonicalizer Pass.
//===----------------------------------------------------------------------===//

namespace {
class CanonicalizerRewriter;

/// Canonicalize operations in functions.
struct Canonicalizer : public FunctionPass {
  PassResult runOnCFGFunction(CFGFunction *f) override;
  PassResult runOnMLFunction(MLFunction *f) override;

  void simplifyFunction(std::vector<Operation *> &worklist,
                        CanonicalizerRewriter &rewriter);

  void addToWorklist(Operation *op) {
    worklistMap[op] = worklist.size();
    worklist.push_back(op);
  }

  Operation *popFromWorklist() {
    auto *op = worklist.back();
    worklist.pop_back();

    // This operation is no longer in the worklist, keep worklistMap up to date.
    if (op)
      worklistMap.erase(op);
    return op;
  }

  /// If the specified operation is in the worklist, remove it.  If not, this is
  /// a no-op.
  void removeFromWorklist(Operation *op) {
    auto it = worklistMap.find(op);
    if (it != worklistMap.end()) {
      assert(worklist[it->second] == op && "malformed worklist data structure");
      worklist[it->second] = nullptr;
    }
  }

private:
  /// The worklist for this transformation keeps track of the operations that
  /// need to be revisited, plus their index in the worklist.  This allows us to
  /// efficiently remove operations from the worklist when they are removed even
  /// if they aren't the root of a pattern.
  std::vector<Operation *> worklist;
  DenseMap<Operation *, unsigned> worklistMap;
};
} // end anonymous namespace

namespace {
class CanonicalizerRewriter : public PatternRewriter {
public:
  CanonicalizerRewriter(Canonicalizer &thePass, MLIRContext *context)
      : PatternRewriter(context), thePass(thePass) {}

  virtual void setInsertionPoint(Operation *op) = 0;

  // If an operation is about to be removed, make sure it is not in our
  // worklist anymore because we'd get dangling references to it.
  void notifyOperationRemoved(Operation *op) override {
    thePass.removeFromWorklist(op);
  }

  // When the root of a pattern is about to be replaced, it can trigger
  // simplifications to its users - make sure to add them to the worklist
  // before the root is changed.
  void notifyRootReplaced(Operation *op) override {
    auto *opStmt = cast<OperationStmt>(op);
    for (auto *result : opStmt->getResults())
      // TODO: Add a result->getUsers() iterator.
      for (auto &user : result->getUses()) {
        if (auto *op = dyn_cast<OperationStmt>(user.getOwner()))
          thePass.addToWorklist(op);
      }

    // TODO: Walk the operand list dropping them as we go.  If any of them
    // drop to zero uses, then add them to the worklist to allow them to be
    // deleted as dead.
  }

  Canonicalizer &thePass;
};

} // end anonymous namespace

PassResult Canonicalizer::runOnCFGFunction(CFGFunction *f) {
  // TODO: Add this.
  return success();
}

PassResult Canonicalizer::runOnMLFunction(MLFunction *f) {
  worklist.reserve(64);

  f->walk([&](OperationStmt *stmt) { addToWorklist(stmt); });

  class MLFuncRewriter : public CanonicalizerRewriter {
  public:
    MLFuncRewriter(Canonicalizer &thePass, MLFuncBuilder &builder)
        : CanonicalizerRewriter(thePass, builder.getContext()),
          builder(builder) {}

    // Implement the hook for creating operations, and make sure that newly
    // created ops are added to the worklist for processing.
    Operation *createOperation(const OperationState &state) override {
      auto *result = builder.createOperation(state);
      thePass.addToWorklist(result);
      return result;
    }

    void setInsertionPoint(Operation *op) override {
      // Any new operations should be added before this statement.
      builder.setInsertionPoint(cast<OperationStmt>(op));
    }

  private:
    MLFuncBuilder &builder;
  };

  MLFuncBuilder mlBuilder(f);
  MLFuncRewriter rewriter(*this, mlBuilder);
  simplifyFunction(worklist, rewriter);
  return success();
}

// TODO: This should work on both ML and CFG functions.
void Canonicalizer::simplifyFunction(std::vector<Operation *> &worklist,
                                     CanonicalizerRewriter &rewriter) {
  // TODO: Instead of a hard coded list of patterns, ask the registered dialects
  // for their canonicalization patterns.

  PatternMatcher matcher({new SimplifyXMinusX(rewriter.getContext())});

  // These are scratch vectors used in the constant folding loop below.
  SmallVector<Attribute *, 8> operandConstants, resultConstants;

  while (!worklist.empty()) {
    auto *op = popFromWorklist();

    // Nulls get added to the worklist when operations are removed, ignore them.
    if (op == nullptr)
      continue;

    // If the operation has no side effects, and no users, then it is trivially
    // dead - remove it.
    if (op->hasNoSideEffect() && op->use_empty()) {
      op->erase();
      continue;
    }

    // TODO: If this is a constant op and if it hasn't already been added to the
    // canonical constants set, move it and remember it.

    // Check to see if any operands to the instruction is constant and whether
    // the operation knows how to constant fold itself.
    operandConstants.clear();
    for (auto *operand : op->getOperands()) {
      Attribute *operandCst = nullptr;
      if (auto *operandOp = operand->getDefiningOperation()) {
        if (auto operandConstantOp = operandOp->dyn_cast<ConstantOp>())
          operandCst = operandConstantOp->getValue();
      }
      operandConstants.push_back(operandCst);
    }

    // If constant folding was successful, create the result constants, RAUW the
    // operation and remove it.
    resultConstants.clear();
    if (!op->isa<ConstantOp>() &&
        !op->constantFold(operandConstants, resultConstants)) {
      rewriter.setInsertionPoint(op);

      // TODO: Put these in the entry block and unique them.
      for (unsigned i = 0, e = op->getNumResults(); i != e; ++i) {
        auto *res = op->getResult(i);
        if (res->use_empty()) // ignore dead uses.
          continue;

        auto cst = rewriter.create<ConstantOp>(op->getLoc(), resultConstants[i],
                                               res->getType());
        res->replaceAllUsesWith(cst);
      }

      assert(op->hasNoSideEffect() && "Constant folded op with side effects?");
      op->erase();
      continue;
    }

    // If this is an associative binary operation with a constant on the LHS,
    // move it to the right side.
    if (operandConstants.size() == 2 && operandConstants[0] &&
        !operandConstants[1]) {
      auto *newLHS = op->getOperand(1);
      op->setOperand(1, op->getOperand(0));
      op->setOperand(0, newLHS);
    }

    // Check to see if we have any patterns that match this node.
    auto match = matcher.findMatch(op);
    if (!match.first)
      continue;

    // Make sure that any new operations are inserted at this point.
    rewriter.setInsertionPoint(op);

    // TODO: Need to be a bit trickier to make sure new instructions get into
    // the worklist.
    // TODO: Need to be careful to remove instructions from the worklist when
    // they are eliminated by the replace method.
    match.first->rewrite(op, std::move(match.second), rewriter);
  }
}

/// Create a Canonicalizer pass.
FunctionPass *mlir::createCanonicalizerPass() { return new Canonicalizer(); }
