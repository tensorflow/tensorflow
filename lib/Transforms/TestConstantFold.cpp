//===- TestConstantFold.cpp - Pass to test constant folding ---------------===//
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

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;

namespace {
/// Simple constant folding pass.
struct TestConstantFold : public FunctionPass<TestConstantFold> {
  // All constants in the function post folding.
  SmallVector<Operation *, 8> existingConstants;

  void foldOperation(Operation *op, FoldHelper &helper);
  void runOnFunction() override;
};
} // end anonymous namespace

void TestConstantFold::foldOperation(Operation *op, FoldHelper &helper) {
  // Attempt to fold the specified operation, including handling unused or
  // duplicated constants.
  if (succeeded(helper.tryToFold(op)))
    return;

  // If this op is a constant that are used and cannot be de-duplicated,
  // remember it for cleanup later.
  if (auto constant = dyn_cast<ConstantOp>(op))
    existingConstants.push_back(op);
}

// For now, we do a simple top-down pass over a function folding constants.  We
// don't handle conditional control flow, block arguments, folding conditional
// branches, or anything else fancy.
void TestConstantFold::runOnFunction() {
  existingConstants.clear();

  auto &f = getFunction();
  FoldHelper helper(&f);

  // Collect and fold the operations within the function.
  SmallVector<Operation *, 8> ops;
  f.walk([&](Operation *op) { ops.push_back(op); });

  // Fold the constants in reverse so that the last generated constants from
  // folding are at the beginning. This creates somewhat of a linear ordering to
  // the newly generated constants that matches the operation order and improves
  // the readability of test cases.
  for (Operation *op : llvm::reverse(ops))
    foldOperation(op, helper);

  // By the time we are done, we may have simplified a bunch of code, leaving
  // around dead constants.  Check for them now and remove them.
  for (auto *cst : existingConstants) {
    if (cst->use_empty())
      cst->erase();
  }
}

/// Creates a constant folding pass.
FunctionPassBase *mlir::createTestConstantFoldPass() {
  return new TestConstantFold();
}

static PassRegistration<TestConstantFold>
    pass("test-constant-fold", "Test operation constant folding");
