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

#include <cstddef>
#include <map>
#include <set>

#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_value_typed_analyzer.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace tf_saved_model {
namespace {

#define GEN_PASS_DEF_OPTIMIZEGLOBALTENSORSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_savedmodel_passes.h.inc"
struct OptimizeGlobalTensorsPass
    : public impl::OptimizeGlobalTensorsPassBase<OptimizeGlobalTensorsPass> {
  void runOnOperation() override;
};

// A global tensor is bound to arguments of multiple funcs.
// This struct tracks which funcs (and which argument to that func) the global
// tensor is bound to.
struct GlobalTensorUse {
  mutable func::FuncOp func;
  size_t arg_index;
};

using GlobalTensorUsesMap =
    std::map<GlobalTensorOp, std::vector<GlobalTensorUse>>;

bool IsImmutable(GlobalTensorOp global_tensor,
                 ArrayRef<GlobalTensorUse> global_tensor_uses,
                 const TF::ResourceAnalyzer& resource_analyzer) {
  // Global tensor is already known to be immutable.
  if (!global_tensor.getIsMutable()) {
    return false;
  }
  // An exported global tensor that is not already known to be immutable might
  // be externally mutated.
  if (IsExported(global_tensor)) {
    return false;
  }

  // A global tensor is immutable if the resource analyzer deems it so.
  for (auto& global_tensor_use : global_tensor_uses) {
    auto arg = global_tensor_use.func.getArgument(global_tensor_use.arg_index);
    if (resource_analyzer.IsPotentiallyWritten(arg)) {
      return false;
    }
  }
  return true;
}

GlobalTensorUsesMap CreateGlobalTensorUsesMap(ModuleOp module) {
  GlobalTensorUsesMap global_tensor_uses;

  SymbolTable symbol_table(module);
  for (auto func : module.getOps<func::FuncOp>()) {
    for (size_t i = 0, e = func.getNumArguments(); i < e; i++) {
      auto sym =
          func.getArgAttrOfType<SymbolRefAttr>(i, "tf_saved_model.bound_input");
      if (!sym) {
        continue;
      }
      auto global_tensor = symbol_table.lookup<GlobalTensorOp>(
          mlir::cast<FlatSymbolRefAttr>(sym).getValue());
      if (!global_tensor) {
        continue;
      }
      global_tensor_uses[global_tensor].push_back({func, i});
    }
  }

  return global_tensor_uses;
}

// Removes `is_mutable` attribute from tf_saved_model.global_tensor ops where we
// can prove it is safe to do so.
void MarkGlobalTensorsImmutable(
    ModuleOp module, const GlobalTensorUsesMap& global_tensor_uses_map,
    const TF::ResourceAnalyzer& resource_analyzer) {
  for (const auto& kv : global_tensor_uses_map) {
    auto global_tensor = kv.first;
    const auto& global_tensor_uses = kv.second;
    if (IsImmutable(global_tensor, global_tensor_uses, resource_analyzer)) {
      global_tensor->removeAttr("is_mutable");
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
  for (auto func : module.getOps<func::FuncOp>()) {
    llvm::BitVector args_to_erase(func.getNumArguments());
    for (int i = 0, e = func.getNumArguments(); i < e; i++) {
      if (func.getArgAttr(i, "tf_saved_model.bound_input") &&
          func.getArgument(i).use_empty()) {
        args_to_erase.set(i);
      }
    }
    func.eraseArguments(args_to_erase);
  }
}

void OptimizeGlobalTensorsPass::runOnOperation() {
  auto module = getOperation();
  if (!tf_saved_model::HasTfSavedModelSemantics(module)) {
    return;
  }

  EraseUnusedBoundInputs(module);

  TF::ResourceAnalyzer resource_analyzer(module);

  GlobalTensorUsesMap global_tensor_uses = CreateGlobalTensorUsesMap(module);

  MarkGlobalTensorsImmutable(module, global_tensor_uses, resource_analyzer);

  EraseUnusedGlobalTensors(module, global_tensor_uses);
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateOptimizeGlobalTensorsPass() {
  return std::make_unique<OptimizeGlobalTensorsPass>();
}

}  // namespace tf_saved_model
}  // namespace mlir
