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
  virtual void rewrite(Operation *op, MLFuncBuilder &builder) const override {
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
                        MLFuncBuilder &builder);
};
} // end anonymous namespace

PassResult Canonicalizer::runOnCFGFunction(CFGFunction *f) {
  // TODO: Add this.
  return success();
}

PassResult Canonicalizer::runOnMLFunction(MLFunction *f) {
  std::vector<Operation *> worklist;
  worklist.reserve(64);

  f->walk([&](OperationStmt *stmt) { worklist.push_back(stmt); });

  MLFuncBuilder builder(f);
  simplifyFunction(worklist, builder);
  return success();
}

// TODO: This should work on both ML and CFG functions.
void Canonicalizer::simplifyFunction(std::vector<Operation *> &worklist,
                                     MLFuncBuilder &builder) {
  // TODO: Instead of a hard coded list of patterns, ask the registered dialects
  // for their canonicalization patterns.

  PatternMatcher matcher({new SimplifyXMinusX(builder.getContext())});

  while (!worklist.empty()) {
    auto *op = worklist.back();
    worklist.pop_back();

    // TODO: If no side effects, and operation has no users, then it is
    // trivially dead - remove it.

    // TODO: Call the constant folding hook on this operation, and canonicalize
    // constants into the entry node.

    // Check to see if we have any patterns that match this node.
    auto match = matcher.findMatch(op);
    if (!match.first)
      continue;

    // TODO: Need to be a bit trickier to make sure new instructions get into
    // the worklist.
    match.first->rewrite(op, std::move(match.second), builder);
  }
}

/// Create a Canonicalizer pass.
FunctionPass *mlir::createCanonicalizerPass() { return new Canonicalizer(); }
