//===- TestConstantFold.cpp - Pass to test constant folding ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;

namespace {
/// Simple constant folding pass.
struct TestConstantFold : public FunctionPass<TestConstantFold> {
  // All constants in the function post folding.
  SmallVector<Operation *, 8> existingConstants;

  void foldOperation(Operation *op, OperationFolder &helper);
  void runOnFunction() override;
};
} // end anonymous namespace

void TestConstantFold::foldOperation(Operation *op, OperationFolder &helper) {
  auto processGeneratedConstants = [this](Operation *op) {
    existingConstants.push_back(op);
  };

  // Attempt to fold the specified operation, including handling unused or
  // duplicated constants.
  (void)helper.tryToFold(op, processGeneratedConstants);
}

// For now, we do a simple top-down pass over a function folding constants.  We
// don't handle conditional control flow, block arguments, folding conditional
// branches, or anything else fancy.
void TestConstantFold::runOnFunction() {
  existingConstants.clear();

  // Collect and fold the operations within the function.
  SmallVector<Operation *, 8> ops;
  getFunction().walk([&](Operation *op) { ops.push_back(op); });

  // Fold the constants in reverse so that the last generated constants from
  // folding are at the beginning. This creates somewhat of a linear ordering to
  // the newly generated constants that matches the operation order and improves
  // the readability of test cases.
  OperationFolder helper(&getContext());
  for (Operation *op : llvm::reverse(ops))
    foldOperation(op, helper);

  // By the time we are done, we may have simplified a bunch of code, leaving
  // around dead constants.  Check for them now and remove them.
  for (auto *cst : existingConstants) {
    if (cst->use_empty())
      cst->erase();
  }
}

static PassRegistration<TestConstantFold>
    pass("test-constant-fold", "Test operation constant folding");
