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

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace tf_saved_model {
namespace {

// This pass will replace a func's bound inputs which are bound to
// tf.ReadVariable ops global tensors with tf.Const ops inside the func's body.
// If this pass runs successfully, the resultant IR will be guaranteed to:
//
// 1. Not contain any tf_saved_model.global_tensor ops
// 2. Not contain any  tf_saved_model.bound_input arg attrs on tf_saved_model
// exported functions
// Else, the pass fails.
//
// The reason this pass has this contract is so that once this succeeds, we know
// the IR is in correct form for inference backends (like lite) that do not
// support resources/variables . Further, this contract also ensures that this
// pass lowers from saved model to pure TF. Hence it fails, if it cannot lower.
struct FreezeGlobalTensorsPass
    : public PassWrapper<FreezeGlobalTensorsPass, OperationPass<ModuleOp>> {
  FreezeGlobalTensorsPass() = default;

  explicit FreezeGlobalTensorsPass(bool allow_mutable_tensors) {
    this->allow_mutable_tensors = allow_mutable_tensors;
  }
  FreezeGlobalTensorsPass(const FreezeGlobalTensorsPass& pass) {}

  void runOnOperation() override;

 private:
  // Force a specified data format for all layout sensitive operations.
  Option<bool> allow_mutable_tensors{
      *this, "allow-mutable-tensors",
      llvm::cl::desc("Allows mutable tensors to be in the graph. Default is "
                     "false which means only immutable are allowed.")};
};

void FreezeGlobalTensorsPass::runOnOperation() {
  auto module = getOperation();
  if (!tf_saved_model::HasTfSavedModelSemantics(module)) {
    return;
  }
  SymbolTable symbol_table(module);
  DenseSet<Operation*> frozen_global_tensors;

  for (auto func : module.getOps<FuncOp>()) {
    SmallVector<unsigned, 4> args_to_erase;
    OpBuilder builder(func.getBody());

    for (int i = 0, e = func.getNumArguments(); i < e; ++i) {
      SmallVector<TF::ReadVariableOp, 4> read_variable_ops_to_erase;
      auto global_tensor =
          LookupBoundInputOfType<GlobalTensorOp>(func, i, symbol_table);

      if (!global_tensor) continue;

      // This pass assumes that all global tensors as immutable (e.g. by a
      // previous optimize global tensors pass). If not, this pass has to fail
      // since it cannot perform one of its goals.
      if (global_tensor.is_mutable()) {
        if (allow_mutable_tensors) continue;
        global_tensor.emitError()
            << "is not immutable, try removing mutable variables in your model "
               "since mutable variables are currently not supported through "
               "this converter";
        return signalPassFailure();
      }
      frozen_global_tensors.insert(global_tensor);

      auto arg = func.getArgument(i);
      for (auto user : arg.getUsers()) {
        if (auto read_op = llvm::dyn_cast<TF::ReadVariableOp>(user)) {
          // Collect all read variable ops so that all its uses can be replaced
          // with the tf.constant corresponding to the global tensor op.
          read_variable_ops_to_erase.push_back(read_op);
        } else {
          // Current assumption is all users are tf.ReadVariableOp. Need to
          // expand this to handle control flow and call ops.
          user->emitError() << "could not rewrite use of immutable bound input";
          return signalPassFailure();
        }
      }

      // Replace the arg with a tf.Const op in the function body.
      builder.setInsertionPointToStart(&func.getBody().front());
      auto const_op = builder.create<TF::ConstOp>(global_tensor.getLoc(),
                                                  global_tensor.value());
      args_to_erase.push_back(i);
      for (auto read_op : read_variable_ops_to_erase) {
        read_op.getResult().replaceAllUsesWith(const_op.getResult());
        read_op.erase();
      }
    }
    func.eraseArguments(args_to_erase);
  }
  // Erase all global tensors that were frozen.
  for (auto global_tensor : frozen_global_tensors) {
    global_tensor->erase();
  }

  if (!allow_mutable_tensors && !module.getOps<GlobalTensorOp>().empty()) {
    module.emitError() << "could not freeze all global tensors in the module";
    return signalPassFailure();
  }
}

}  // namespace

// For "opt" to pick up this pass.
static PassRegistration<FreezeGlobalTensorsPass> pass(
    "tf-saved-model-freeze-global-tensors",
    "Freeze tf_saved_model.global_tensor's in func bodies.");

std::unique_ptr<OperationPass<ModuleOp>> CreateFreezeGlobalTensorsPass(
    bool allow_mutable_tensors) {
  return std::make_unique<FreezeGlobalTensorsPass>(allow_mutable_tensors);
}

}  // namespace tf_saved_model
}  // namespace mlir
