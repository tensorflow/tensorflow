/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This pass uses tf_saved_model dialect linkage information to delete
// unused func's.

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace tf_saved_model {

namespace {
struct DeleteUnusedFuncsPass : public ModulePass<DeleteUnusedFuncsPass> {
  void runOnModule() override;
};
}  // namespace

void DeleteUnusedFuncsPass::runOnModule() {
  // If the model doesn't have tf_saved_model semantics, we can't do anything.
  if (!HasTfSavedModelSemantics(getModule())) {
    return;
  }

  // TODO(silvasean): Use more generic MLIR functionality when available.
  // This is just a basic call graph reachability pass (which in the case of TF
  // functional control flow also implies handling tf.If/tf.While).
  // The only thing specific to tf_saved_model is the set of roots.

  auto module = getModule();
  SymbolTable symbol_table(module);

  // Calculate func reachability with a DFS on the symbol reference graph.
  SmallPtrSet<FuncOp, 8> dfs_visited_set;
  SmallVector<FuncOp, 16> dfs_stack;

  // Initialize the roots of the DFS search.
  for (auto func : module.getOps<FuncOp>()) {
    if (IsExported(func)) {
      dfs_stack.push_back(func);
    }
  }

  // Do the DFS.
  while (!dfs_stack.empty()) {
    FuncOp func = dfs_stack.pop_back_val();
    if (!dfs_visited_set.insert(func).second) {
      // If we already visited this node, skip it.
      continue;
    }

    SmallPtrSet<FuncOp, 8> callees;
    auto uses = SymbolTable::getSymbolUses(func);
    for (auto use : *uses) {
      auto func = symbol_table.lookup<FuncOp>(
          use.getSymbolRef().cast<FlatSymbolRefAttr>().getValue());
      if (func) {
        callees.insert(func);
      }
    }

    for (auto callee : callees) {
      dfs_stack.push_back(callee);
    }
  }

  // Erase all unreachable func's.
  for (auto func : llvm::make_early_inc_range(module.getOps<FuncOp>())) {
    if (dfs_visited_set.find(func) == dfs_visited_set.end()) {
      func.erase();
    }
  }
}

std::unique_ptr<OpPassBase<ModuleOp>> CreateDeleteUnusedFuncsPass() {
  return std::make_unique<DeleteUnusedFuncsPass>();
}

static PassRegistration<DeleteUnusedFuncsPass> pass(
    "tf-saved-model-delete-unused-funcs",
    "Use tf_saved_model linkage information to delete unused func's.");

}  // namespace tf_saved_model
}  // namespace mlir
