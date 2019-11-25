//===- TestInlining.cpp - Pass to inline calls in the test dialect --------===//
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
// TODO(riverriddle) This pass is only necessary because the main inlining pass
// has no abstracted away the call+callee relationship. When the inlining
// interface has this support, this pass should be removed.
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace {
struct Inliner : public FunctionPass<Inliner> {
  void runOnFunction() override {
    auto function = getFunction();

    // Collect each of the direct function calls within the module.
    SmallVector<CallIndirectOp, 16> callers;
    function.walk([&](CallIndirectOp caller) { callers.push_back(caller); });

    // Build the inliner interface.
    InlinerInterface interface(&getContext());

    // Try to inline each of the call operations.
    for (auto caller : callers) {
      auto callee = dyn_cast_or_null<FunctionalRegionOp>(
          caller.getCallee()->getDefiningOp());
      if (!callee)
        continue;

      // Inline the functional region operation, but only clone the internal
      // region if there is more than one use.
      if (failed(inlineRegion(
              interface, &callee.body(), caller,
              llvm::to_vector<8>(caller.getArgOperands()),
              llvm::to_vector<8>(caller.getResults()), caller.getLoc(),
              /*shouldCloneInlinedRegion=*/!callee.getResult()->hasOneUse())))
        continue;

      // If the inlining was successful then erase the call and callee if
      // possible.
      caller.erase();
      if (callee.use_empty())
        callee.erase();
    }
  }
};
} // end anonymous namespace

static PassRegistration<Inliner> pass("test-inline",
                                      "Test inlining region calls");
