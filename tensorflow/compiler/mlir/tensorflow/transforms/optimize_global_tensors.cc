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
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace tf_saved_model {
namespace {
struct OptimizeGlobalTensorsPass
    : public PassWrapper<OptimizeGlobalTensorsPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
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

static bool IsResourceType(Type type) {
  if (auto tensor_type = type.dyn_cast<TensorType>()) {
    return tensor_type.getElementType().isa<TF::ResourceType>();
  }
  return false;
}

static bool IsResource(Value value) { return IsResourceType(value.getType()); }

class ResourceAnalyzer {
 public:
  explicit ResourceAnalyzer(ModuleOp module) {
    SymbolTable symbol_table(module);
    for (auto func : module.getOps<FuncOp>()) {
      AnalyzeFunc(func, symbol_table);
    }
  }

  bool IsPotentiallyWritten(Value resource) const {
    assert(IsResource(resource));
    auto it = resource_infos_.find(resource);
    if (it == resource_infos_.end()) {
      return false;
    }
    return it->second.potentially_written;
  }

 private:
  // Analyze the specified func for resource mutating operations, namely
  // TF::AssignVariableOp, if so, set the resource associated as "potentially
  // written". Do this recursively across the chain of funcs via call or control
  // flow ops.
  // TODO(ashwinm): Move to iterative traversal.
  LogicalResult AnalyzeFunc(FuncOp func, const SymbolTable& symbol_table) {
    // Avoid infinite recursion.
    if (!discovered_.insert(func).second) {
      return success();
    }

    func.walk([&](Operation* op) {
      if (isa<TF::ReadVariableOp>(op) || isa<ReturnOp>(op)) {
        return;
      }
      if (auto assign_variable = dyn_cast<TF::AssignVariableOp>(op)) {
        SetPotentiallyWritten(assign_variable.resource());
        return;
      }
      if (auto call = dyn_cast<CallOpInterface>(op)) {
        if (auto sym = op->getAttrOfType<SymbolRefAttr>("f")) {
          PropagatePotentiallyWrittenUpFromCallee(
              sym.cast<FlatSymbolRefAttr>().getValue(), call.getArgOperands(),
              symbol_table);
        }
        return;
      }
      if (auto if_op = dyn_cast<TF::IfOp>(op)) {
        for (auto callee : {if_op.then_branch(), if_op.else_branch()}) {
          PropagatePotentiallyWrittenUpFromCallee(callee, if_op.input(),
                                                  symbol_table);
        }
        return;
      }
      if (auto while_op = dyn_cast<TF::WhileOp>(op)) {
        for (auto callee : {while_op.cond(), while_op.body()}) {
          PropagatePotentiallyWrittenUpFromCallee(callee, while_op.input(),
                                                  symbol_table);
        }
        return;
      }
      // For all other ops, we assume it mutates all resources it uses, so
      // this errs on the side of being conservative. We should improve
      // this by using either a property or a trait that clearly
      // identifies ops with resource mutating behavior.
      if (PropagatePotentiallyWrittenWithinUnhandledOp(op)) {
        return;
      }
    });
    return success();
  }

  // If an op is not one of the handled ones, we assume all resource usages
  // within its purview are mutating in nature.
  bool PropagatePotentiallyWrittenWithinUnhandledOp(Operation* op) {
    for (auto operand : op->getOperands()) {
      if (IsResource(operand)) {
        SetPotentiallyWritten(operand);
        return true;
      }
    }
    bool uses_resources = false;
    visitUsedValuesDefinedAbove(op->getRegions(), [&](OpOperand* operand) {
      if (IsResource(operand->get())) {
        SetPotentiallyWritten(operand->get());
        uses_resources = true;
      }
    });
    return uses_resources;
  }

  // Given a funcOp associated with the callee and operands from the
  // corresponding callOp, propagate the potentially written decision to the
  // callOp's operands, if the corresponding func's arguments are potentially
  // written resources.
  void PropagatePotentiallyWrittenUpFromCallee(
      StringRef callee, Operation::operand_range propagate_to,
      const SymbolTable& symbol_table) {
    auto func = symbol_table.lookup<FuncOp>(callee);
    AnalyzeFunc(func, symbol_table);
    for (auto t : llvm::zip(func.getArguments(), propagate_to)) {
      if (!IsResource(std::get<0>(t))) {
        continue;
      }
      if (IsPotentiallyWritten(std::get<0>(t))) {
        SetPotentiallyWritten(std::get<1>(t));
      }
    }
  }

  void SetPotentiallyWritten(Value resource) {
    assert(IsResource(resource));
    resource_infos_[resource].potentially_written = true;
  }
  struct ResourceInfo {
    bool potentially_written = false;
  };
  // Key: Resource Value's
  // Value: Information we know about that Value.
  // Note that these Value's are in general in different functions.
  DenseMap<Value, ResourceInfo> resource_infos_;
  // The set of func's we already discovered.
  DenseSet<FuncOp> discovered_;
};

bool IsImmutable(GlobalTensorOp global_tensor,
                 ArrayRef<GlobalTensorUse> global_tensor_uses,
                 const ResourceAnalyzer& resource_analyzer) {
  // Global tensor is already known to be immutable.
  if (!global_tensor.is_mutable()) {
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
    ModuleOp module, const GlobalTensorUsesMap& global_tensor_uses_map,
    const ResourceAnalyzer& resource_analyzer) {
  for (const auto& kv : global_tensor_uses_map) {
    auto global_tensor = kv.first;
    const auto& global_tensor_uses = kv.second;
    if (IsImmutable(global_tensor, global_tensor_uses, resource_analyzer)) {
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

void OptimizeGlobalTensorsPass::runOnOperation() {
  auto module = getOperation();
  EraseUnusedBoundInputs(module);

  ResourceAnalyzer resource_analyzer(module);

  GlobalTensorUsesMap global_tensor_uses = CreateGlobalTensorUsesMap(module);

  MarkGlobalTensorsImmutable(module, global_tensor_uses, resource_analyzer);

  EraseUnusedGlobalTensors(module, global_tensor_uses);
}

}  // namespace

// For "opt" to pick up this pass.
static PassRegistration<OptimizeGlobalTensorsPass> pass(
    "tf-saved-model-optimize-global-tensors",
    "Optimize tf_saved_model.global_tensor's.");

std::unique_ptr<OperationPass<ModuleOp>> CreateOptimizeGlobalTensorsPass() {
  return std::make_unique<OptimizeGlobalTensorsPass>();
}

}  // namespace tf_saved_model
}  // namespace mlir
