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
#include "mlir/Transforms/ConstantFoldUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;

namespace {
/// Simple constant folding pass.
struct TestConstantFold : public FunctionPass<TestConstantFold> {
  // All constants in the function post folding.
  SmallVector<Operation *, 8> existingConstants;
  // Operations that were folded and that need to be erased.
  std::vector<Operation *> opsToErase;

  void foldOperation(Operation *op, ConstantFoldHelper &helper);
  void runOnFunction() override;
};
} // end anonymous namespace

void TestConstantFold::foldOperation(Operation *op,
                                     ConstantFoldHelper &helper) {
  // Attempt to fold the specified operation, including handling unused or
  // duplicated constants.
  if (helper.tryToConstantFold(op)) {
    opsToErase.push_back(op);
  }
  // If this op is a constant that are used and cannot be de-duplicated,
  // remember it for cleanup later.
  else if (auto constant = op->dyn_cast<ConstantOp>()) {
    existingConstants.push_back(op);
  }
}

// For now, we do a simple top-down pass over a function folding constants.  We
// don't handle conditional control flow, block arguments, folding conditional
// branches, or anything else fancy.
void TestConstantFold::runOnFunction() {
  existingConstants.clear();
  opsToErase.clear();

  auto &f = getFunction();

  ConstantFoldHelper helper(&f, /*insertAtHead=*/false);

  f.walk([&](Operation *op) { foldOperation(op, helper); });

  // At this point, these operations are dead, remove them.
  for (auto *op : opsToErase) {
    assert(op->hasNoSideEffect() && "Constant folded op with side effects?");
    op->erase();
  }

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
