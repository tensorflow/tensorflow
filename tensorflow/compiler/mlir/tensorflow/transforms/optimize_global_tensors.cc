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

// This pass optimizes tf_saved_model.global_tensor ops.

#include <map>
#include <set>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace tf_saved_model {
namespace {
struct OptimizeGlobalTensorsPass
    : public ModulePass<OptimizeGlobalTensorsPass> {
  void runOnModule() override;
};

// A global tensor is bound to arguments of multiple funcs.
// This struct tracks which funcs (and which argument to that func) the global
// tensor is bound to.
struct GlobalTensorUse {
  mutable FuncOp func;
  size_t arg_index;
};

using GlobalTensorUsesMap =
    std::map<GlobalTensorOp, std::vector<GlobalTensorUse>>;

// TODO(silvasean): Are there other read-only variable ops?
// It would be nice if we eventually had an interface that we could use
// to determine if an op is read-only and how to rewrite it.
// For now, IsReadOnlyVariableOp and RewriteReadOnlyVariableOpToTensorOp need to
// be keep in sync.
bool IsReadOnlyVariableOp(Operation* op) { return isa<TF::ReadVariableOp>(op); }

void RewriteReadOnlyVariableOpToTensorOp(Operation* op, Value tensor_value) {
  auto read_variable = cast<TF::ReadVariableOp>(op);
  read_variable.value()->replaceAllUsesWith(tensor_value);
}

bool IsFreezable(GlobalTensorOp global_tensor,
                 ArrayRef<GlobalTensorUse> global_tensor_uses) {
  // If this tensor is already immutable, don't freeze it.
  if (!global_tensor.is_mutable()) {
    return false;
  }
  // Can't freeze if exported.
  if (IsExported(global_tensor)) {
    return false;
  }

  // Can't freeze if it is used by anything that we aren't sure is read-only.
  // Right now, this uses a very simple algorithm that only checks the top-level
  // func for tf.ReadVariableOp. If the resource is passed into other functions
  // or control flow, we fail to prove it is freezable even though we could.
  for (auto& global_tensor_use : global_tensor_uses) {
    auto arg = global_tensor_use.func.getArgument(global_tensor_use.arg_index);
    for (auto user : arg->getUsers()) {
      if (!IsReadOnlyVariableOp(user)) {
        return false;
      }
    }
  }
  return true;
}

static GlobalTensorUsesMap CreateGlobalTensorUsesMap(ModuleOp module) {
  GlobalTensorUsesMap global_tensor_uses;

  SymbolTable symbol_table(module);
  for (auto func : module.getOps<FuncOp>()) {
    for (size_t i = 0, e = func.getNumArguments(); i < e; i++) {
      auto sym =
          func.getArgAttrOfType<SymbolRefAttr>(i, "tf_saved_model.bound_input");
      if (!sym) {
        continue;
      }
      auto global_tensor = symbol_table.lookup<GlobalTensorOp>(
          sym.cast<FlatSymbolRefAttr>().getValue());
      global_tensor_uses[global_tensor].push_back({func, i});
    }
  }

  return global_tensor_uses;
}

void FreezeGlobalTensors(ModuleOp module,
                         const GlobalTensorUsesMap& global_tensor_uses_map) {
  SmallVector<GlobalTensorOp, 4> freezable_global_tensors;
  for (auto& kv : global_tensor_uses_map) {
    auto global_tensor = kv.first;
    const auto& global_tensor_uses = kv.second;
    if (IsFreezable(global_tensor, global_tensor_uses)) {
      freezable_global_tensors.push_back(global_tensor);
    }
  }

  // Remove `is_mutable` attribute from tf_saved_model.global_tensor
  // and update func arguments to match.
  //
  // This amounts to changing the type of the argument to a tensor type, and
  // replacing all the tf.ReadVariableOp's with the new tensor argument value.
  OpBuilder builder(module.getBodyRegion());
  for (const auto& kv : global_tensor_uses_map) {
    auto global_tensor = kv.first;
    const auto& global_tensor_uses = kv.second;
    if (!IsFreezable(global_tensor, global_tensor_uses)) {
      continue;
    }
    for (auto global_tensor_use : global_tensor_uses) {
      auto func = global_tensor_use.func;
      auto arg_index = global_tensor_use.arg_index;
      Value arg = func.getArgument(arg_index);
      for (Operation* user : llvm::make_early_inc_range(arg->getUsers())) {
        RewriteReadOnlyVariableOpToTensorOp(user, arg);
        user->erase();
      }
      Type new_type = global_tensor.value().Attribute::getType();
      arg->setType(new_type);
      auto old_ftype = func.getType();
      auto input_types = old_ftype.getInputs().vec();
      input_types[arg_index] = new_type;
      func.setType(
          builder.getFunctionType(input_types, old_ftype.getResults()));
    }
    global_tensor.removeAttr("is_mutable");
  }
}

void EraseUnusedGlobalTensors(ModuleOp module,
                              const GlobalTensorUsesMap& global_tensor_uses) {
  for (auto global_tensor :
       llvm::make_early_inc_range(module.getOps<GlobalTensorOp>())) {
    // If the tensor is exported, then it is used.
    if (IsExported(global_tensor)) {
      continue;
    }
    // If the tensor is bound to an argument, then it is used.
    if (global_tensor_uses.find(global_tensor) != global_tensor_uses.end()) {
      continue;
    }
    // Erase it.
    global_tensor.erase();
  }
}

void EraseUnusedBoundInputs(ModuleOp module) {
  for (auto func : module.getOps<FuncOp>()) {
    SmallVector<unsigned, 4> args_to_erase;
    for (int i = 0, e = func.getNumArguments(); i < e; i++) {
      if (func.getArgAttr(i, "tf_saved_model.bound_input") &&
          func.getArgument(i)->use_empty()) {
        args_to_erase.push_back(i);
      }
    }
    func.eraseArguments(args_to_erase);
  }
}

void OptimizeGlobalTensorsPass::runOnModule() {
  // This analysis could be much more elaborate, including tracking global
  // tensors interprocedurally and uses in a wide variety of ops. But I don't
  // know if we need that complexity.
  auto module = getModule();

  EraseUnusedBoundInputs(module);

  // Figure out which func's use each tf_saved_model.global_tensor.
  GlobalTensorUsesMap global_tensor_uses = CreateGlobalTensorUsesMap(module);

  FreezeGlobalTensors(module, global_tensor_uses);
  EraseUnusedGlobalTensors(module, global_tensor_uses);
}

}  // namespace

// For "opt" to pick up this pass.
static PassRegistration<OptimizeGlobalTensorsPass> pass(
    "tf-saved-model-optimize-global-tensors",
    "Optimize tf_saved_model.global_tensor's.");

std::unique_ptr<OpPassBase<ModuleOp>> CreateOptimizeGlobalTensorsPass() {
  return std::make_unique<OptimizeGlobalTensorsPass>();
}

}  // namespace tf_saved_model
}  // namespace mlir
