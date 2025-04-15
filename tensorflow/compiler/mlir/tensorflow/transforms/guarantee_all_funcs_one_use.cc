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

#include <memory>

#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Analysis/CallGraph.h"  // from @llvm-project
#include "mlir/Dialect/Affine/Utils.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {

namespace {

// Check that there is no recursion in the module's call graph.
LogicalResult CheckNoRecursion(ModuleOp module, CallGraph &call_graph) {
  for (llvm::scc_iterator<const CallGraph *> scci =
           llvm::scc_begin<const CallGraph *>(&call_graph);
       !scci.isAtEnd(); ++scci) {
    if (scci.hasCycle()) {
      auto err = module.emitError()
                 << "A recursive call graph cannot be transformed to "
                    "one use for all functions. Functions in the "
                    "recursive cycle are: ";
      llvm::interleaveComma(*scci, err, [&](CallGraphNode *node) {
        err << node->getCallableRegion()->getLoc();
      });
      return err;
    }
  }
  return success();
}

#define GEN_PASS_DEF_GUARANTEEALLFUNCSONEUSEPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

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
    : public impl::GuaranteeAllFuncsOneUsePassBase<GuaranteeAllFuncsOneUse> {
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
    SymbolTableCollection symbol_table_collection;
    SymbolTable &symbol_table = symbol_table_collection.getSymbolTable(module);
    bool made_changes = false;

    if (failed(CheckNoRecursion(module, getAnalysis<CallGraph>())))
      return failure();

    do {
      SymbolUserMap symbol_users(symbol_table_collection, module);

      made_changes = false;
      for (auto func :
           llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
        ArrayRef<Operation *> users = symbol_users.getUsers(func);
        if (users.size() <= 1) {
          continue;
        }

        // At this point, we know we are going to change the module.
        made_changes = true;
        for (Operation *user : users.drop_front()) {
          func::FuncOp new_func = func.clone();
          symbol_table.insert(new_func);
          new_func.setPrivate();
          if (failed(SymbolTable::replaceAllSymbolUses(
                  func, new_func.getSymNameAttr(), user))) {
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

}  // namespace TF

}  // namespace mlir
