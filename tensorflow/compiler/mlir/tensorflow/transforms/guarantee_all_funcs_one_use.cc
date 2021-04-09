/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Utils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {

namespace {

// Clones FuncOp's until they have a single use only (or no users).
//
// The tf-shape-inference pass doesn't support functions that have more than
// a single use. But some real code from frontends does end up creating code
// like that. For example, the same LSTM cell function or loop body function
// will be reused.
//
// This pass clones functions as needed to establish the invariant that all
// functions have a single use. This can in principle cause exponential code
// size bloat, and should in general be guided by a proper cost model.
//
// There are two factors which should be considered by a principled replacement
// to this pass:
//
// 1. TF currently relies on "sufficiently good shape inference" for
// correctness so for now the cost of doing this seems acceptable since
// pathological cases haven't hit us yet.
//
// 2. Cloning functions can help by allowing code to be specialized (much as
// inlining does). In fact, tf-shape-inference attempts to do specialization
// of callees which is difficult if callees have multiple uses.
class GuaranteeAllFuncsOneUse
    : public PassWrapper<GuaranteeAllFuncsOneUse, OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
    if (failed(Run())) {
      signalPassFailure();
    }
  }

  LogicalResult Run() {
    auto module = getOperation();

    // Overall strategy:
    // Fixed point iteration, iteratively applying a rule that clones
    // any FuncOp with more than one use to eliminate its uses.

    SymbolTable symbol_table(module);
    bool made_changes = false;
    // This value needs to be low enough to actually stop compilation in a
    // reasonable time, but not too low that it blocks real programs.
    // This number was chosen semi-randomly.
    const int k_max_clones = 1000;
    int num_clones = 0;
    do {
      made_changes = false;
      for (auto func : llvm::make_early_inc_range(module.getOps<FuncOp>())) {
        auto uses_optional = symbol_table.getSymbolUses(func, module);
        if (!uses_optional.hasValue()) {
          return func.emitError() << "could not walk uses of func";
        }
        auto &uses = *uses_optional;
        if (llvm::size(uses) <= 1) {
          continue;
        }
        // At this point, we know we are going to change the module.
        made_changes = true;
        for (const SymbolTable::SymbolUse &use : llvm::drop_begin(uses, 1)) {
          if (num_clones++ > k_max_clones) {
            return func.emitError()
                   << "reached cloning limit (likely recursive call graph or "
                      "repeated diamond-like call structure "
                      "or just very large program)";
          }
          auto new_func = func.clone();
          symbol_table.insert(new_func);
          new_func.setPrivate();
          if (failed(symbol_table.replaceAllSymbolUses(func, new_func.getName(),
                                                       use.getUser()))) {
            return func.emitError() << "could not replace symbol use";
          }
        }
      }
    } while (made_changes);

    return success();
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateGuaranteeAllFuncsOneUsePass() {
  return std::make_unique<GuaranteeAllFuncsOneUse>();
}

static PassRegistration<GuaranteeAllFuncsOneUse> pass(
    "tf-guarantee-all-funcs-one-use",
    "Guarantee all FuncOp's have only a single use.");

}  // namespace TF

}  // namespace mlir
