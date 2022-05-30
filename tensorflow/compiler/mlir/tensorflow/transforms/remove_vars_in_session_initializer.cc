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

#include <algorithm>
#include <vector>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/savedmodel_passes_detail.h"

namespace mlir {
namespace tf_saved_model {
namespace {
using mlir::Operation;
using mlir::TF::VarHandleOp;

class RemoveVariablesInSessionInitializerPass
    : public RemoveVariablesInSessionInitializerPassBase<
          RemoveVariablesInSessionInitializerPass> {
 public:
  void runOnOperation() override;
};

void RecursiveRemove(Operation* op,
                     llvm::SmallVectorImpl<Operation*>& erase_list,
                     llvm::SmallPtrSetImpl<Operation*>& dead_ops) {
  for (mlir::Value res : op->getResults()) {
    for (Operation* user : res.getUsers()) {
      if (!dead_ops.insert(user).second) continue;
      RecursiveRemove(user, erase_list, dead_ops);
    }
  }

  erase_list.push_back(op);

  for (auto& use : op->getOpOperands()) {
    if (auto op_result = use.get().dyn_cast<mlir::OpResult>()) {
      Operation* def = op_result.getDefiningOp();
      if (!dead_ops.insert(def).second) continue;
      RecursiveRemove(def, erase_list, dead_ops);
    }
  }
}

void RemoveVariables(llvm::ArrayRef<VarHandleOp> vars) {
  // TODO(b/160906885): Repalce the following code with an non-recursive one.
  llvm::SmallVector<Operation*, 4> erase_list;
  llvm::SmallPtrSet<Operation*, 4> dead_ops;

  // Marks all the variables dead.
  dead_ops.insert(vars.begin(), vars.end());

  // Removes relevant ops in topological order.
  for (auto& op : vars) RecursiveRemove(op, erase_list, dead_ops);

  // Erases the ops.
  for (auto op : erase_list) op->erase();
}

void RemoveVariablesInSessionInitializerPass::runOnOperation() {
  ModuleOp module = getOperation();
  SessionInitializerOp session_init_op = GetSessionInitializerOp(module);

  if (!session_init_op) return;

  SymbolTable symbol_table(module);

  for (auto sym_ref : session_init_op.initializers()) {
    func::FuncOp init_func_op = symbol_table.lookup<mlir::func::FuncOp>(
        sym_ref.cast<FlatSymbolRefAttr>().getValue());

    if (!init_func_op) {
      module.emitError("no session initializer function found");
      return signalPassFailure();
    }

    if (init_func_op.getBlocks().size() != 1) {
      init_func_op.emitError("expects exactly one block in the MLIR function");
      return signalPassFailure();
    }

    auto var_handle_ops =
        init_func_op.getBlocks().front().getOps<VarHandleOp>();
    llvm::SmallVector<VarHandleOp, 4> init_vars(var_handle_ops.begin(),
                                                var_handle_ops.end());
    RemoveVariables(init_vars);
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateRemoveVariablesInSessionInitializerPass() {
  return std::make_unique<RemoveVariablesInSessionInitializerPass>();
}

}  // namespace tf_saved_model
}  // namespace mlir
