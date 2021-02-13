/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/None.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace TFL {
namespace {
// Pass which removes any unused bounded function arguments which maps to
// variables, also removes the GlobalTensor which is the variable.
class RemoveArgsAndGlobalTensors
    : public PassWrapper<RemoveArgsAndGlobalTensors, OperationPass<ModuleOp>> {
 public:
  RemoveArgsAndGlobalTensors() = default;
  RemoveArgsAndGlobalTensors(const RemoveArgsAndGlobalTensors&) {}

  void runOnOperation() override {
    auto module = getOperation();
    SymbolTable symbol_table(module);

    // Remove unused arguments in the functions which are bounded input
    // for a global tensor. Also, removes the now unused global tensors.
    std::set<mlir::tf_saved_model::GlobalTensorOp> global_tensors_to_remove;
    for (auto func : module.getOps<FuncOp>()) {
      llvm::SmallVector<unsigned int> index_to_remove;
      for (int i = 0; i < func.getNumArguments(); ++i) {
        if (auto sym = func.template getArgAttrOfType<FlatSymbolRefAttr>(
                i, "tf_saved_model.bound_input")) {
          auto global_tensor =
              symbol_table.lookup<tf_saved_model::GlobalTensorOp>(
                  sym.getValue());
          if (global_tensor && func.getArgument(i).getUsers().empty()) {
            index_to_remove.push_back(i);
            global_tensors_to_remove.insert(global_tensor);
          }
        }
      }
      func.eraseArguments(index_to_remove);
    }
    for (auto global_tensor : global_tensors_to_remove) {
      global_tensor->erase();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateRemoveArgsAndGlobalTensors() {
  return std::make_unique<RemoveArgsAndGlobalTensors>();
}

static PassRegistration<RemoveArgsAndGlobalTensors> pass(
    "tfl-remove-unused-function-args",
    "Removes unused bounded input arguments to function which are unused and "
    "maps to GlobalTensor.");

}  // namespace TFL
}  // namespace mlir
