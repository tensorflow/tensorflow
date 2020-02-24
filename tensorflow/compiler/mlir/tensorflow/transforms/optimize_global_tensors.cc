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
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
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

bool IsImmutable(GlobalTensorOp global_tensor,
                 ArrayRef<GlobalTensorUse> global_tensor_uses) {
  // Global tensor is already known to be immutable.
  if (!global_tensor.is_mutable()) {
    return false;
  }
  // An exported global tensor that is not already known to be immutable might
  // be externally mutated.
  if (IsExported(global_tensor)) {
    return false;
  }

  // Check the uses to see if this global tensor is only used in a way that
  // is compatible with being immutable.
  // Right now, this uses a very simple algorithm that only checks the top-level
  // func for tf.ReadVariableOp. If the resource is passed into other functions
  // or control flow, we fail to prove it is freezable even though we could.
  for (auto& global_tensor_use : global_tensor_uses) {
    auto arg = global_tensor_use.func.getArgument(global_tensor_use.arg_index);
    for (auto user : arg.getUsers()) {
      if (!isa<TF::ReadVariableOp>(user)) {
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

// Removes `is_mutable` attribute from tf_saved_model.global_tensor ops where we
// can prove it is safe to do so.
void MarkGlobalTensorsImmutable(
    ModuleOp module, const GlobalTensorUsesMap& global_tensor_uses_map) {
  for (const auto& kv : global_tensor_uses_map) {
    auto global_tensor = kv.first;
    const auto& global_tensor_uses = kv.second;
    if (IsImmutable(global_tensor, global_tensor_uses)) {
      global_tensor.removeAttr("is_mutable");
    }
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
          func.getArgument(i).use_empty()) {
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

  MarkGlobalTensorsImmutable(module, global_tensor_uses);
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
