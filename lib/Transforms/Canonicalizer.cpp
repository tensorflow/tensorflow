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

#include <memory>
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
    // TODO: Rename getAs -> dyn_cast, and add a cast<> method.
    auto subi = op->getAs<SubIOp>();
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
  virtual void rewrite(Operation *op, FuncBuilder &builder) const override {
    // TODO: Rename getAs -> dyn_cast, and add a cast<> method.
    auto subi = op->getAs<SubIOp>();
    assert(subi && "Matcher should have produced this");

    auto result =
        builder.create<ConstantIntOp>(op->getLoc(), 0, subi->getType());

    replaceSingleResultOp(op, result);
  }
};
} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// The actual Canonicalizer Pass.
//===----------------------------------------------------------------------===//

// TODO: Canonicalize and unique all constant operations into the entry of the
// function.

namespace {
/// Canonicalize operations in functions.
struct Canonicalizer : public FunctionPass {
  PassResult runOnCFGFunction(CFGFunction *f) override;
  PassResult runOnMLFunction(MLFunction *f) override;

  void simplifyFunction(std::vector<Operation *> &worklist,
                        FuncBuilder &builder);

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

PassResult Canonicalizer::runOnCFGFunction(CFGFunction *f) {
  // TODO: Add this.
  return success();
}

PassResult Canonicalizer::runOnMLFunction(MLFunction *f) {
  worklist.reserve(64);

  f->walk([&](OperationStmt *stmt) { addToWorklist(stmt); });

  MLFuncBuilder mlBuilder(f);
  FuncBuilder builder(mlBuilder);
  simplifyFunction(worklist, builder);
  return success();
}

// TODO: This should work on both ML and CFG functions.
void Canonicalizer::simplifyFunction(std::vector<Operation *> &worklist,
                                     FuncBuilder &builder) {
  // TODO: Instead of a hard coded list of patterns, ask the registered dialects
  // for their canonicalization patterns.

  PatternMatcher matcher({new SimplifyXMinusX(builder.getContext())});

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
      // FIXME: Generalize to support CFG statements as well.
      cast<OperationStmt>(op)->eraseFromBlock();
      continue;
    }

    // Check to see if any operands to the instruction is constant and whether
    // the operation knows how to constant fold itself.
    operandConstants.clear();
    for (auto *operand : op->getOperands()) {
      Attribute *operandCst = nullptr;
      if (auto *operandOp = operand->getDefiningOperation()) {
        if (auto operandConstantOp = operandOp->getAs<ConstantOp>())
          operandCst = operandConstantOp->getValue();
      }
      operandConstants.push_back(operandCst);
    }

    // If constant folding was successful, create the result constants, RAUW the
    // operation and remove it.
    resultConstants.clear();
    if (!op->constantFold(operandConstants, resultConstants)) {
      // TODO: Put these in the entry block and unique them.
      FuncBuilder cstBuilder(builder);
      cstBuilder.setInsertionPoint(op);

      for (unsigned i = 0, e = op->getNumResults(); i != e; ++i) {
        auto *res = op->getResult(i);
        if (res->use_empty()) // ignore dead uses.
          continue;

        auto cst = cstBuilder.create<ConstantOp>(
            op->getLoc(), resultConstants[i], res->getType());
        res->replaceAllUsesWith(cst);
      }

      assert(op->hasNoSideEffect() && "Constant folded op with side effects?");

      // FIXME: Generalize to support CFG statements as well.
      cast<OperationStmt>(op)->eraseFromBlock();
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

    // TODO: Need to be a bit trickier to make sure new instructions get into
    // the worklist.
    // TODO: Need to be careful to remove instructions from the worklist when
    // they are eliminated by the replace method.
    match.first->rewrite(op, std::move(match.second), builder);
  }
}

/// Create a Canonicalizer pass.
FunctionPass *mlir::createCanonicalizerPass() { return new Canonicalizer(); }
