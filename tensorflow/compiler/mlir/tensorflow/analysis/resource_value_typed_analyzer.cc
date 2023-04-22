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

#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_value_typed_analyzer.h"

#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {
namespace {
bool IsResourceType(Type type) {
  if (auto tensor_type = type.dyn_cast<TensorType>()) {
    return tensor_type.getElementType().isa<TF::ResourceType>();
  }
  return false;
}

bool IsResource(Value value) { return IsResourceType(value.getType()); }

// Helper that returns the FuncOp that is the SessionInit function which
// will be called to initialize all resources.
// Returns nullptr if no function is found.
FuncOp GetSessionInitializerFunc(ModuleOp module) {
  auto session_init_op = tf_saved_model::GetSessionInitializerOp(module);
  if (session_init_op && !session_init_op.initializers().empty()) {
    SymbolTable symbol_table(module);
    FuncOp init_func_op = symbol_table.lookup<mlir::FuncOp>(
        session_init_op.initializers()[0].cast<FlatSymbolRefAttr>().getValue());
    return init_func_op;
  }
  return nullptr;
}

// Returns ID for identifying a resource.
std::tuple<llvm::StringRef, llvm::StringRef, llvm::StringRef> GetResourceKey(
    Operation* op) {
  llvm::StringRef device;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("device")) {
    device = attr.getValue();
  }

  llvm::StringRef container;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("container")) {
    container = attr.getValue();
  }

  llvm::StringRef shared_name;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("shared_name")) {
    shared_name = attr.getValue();
  }

  return std::tuple<llvm::StringRef, llvm::StringRef, llvm::StringRef>{
      device, container, shared_name};
}
}  // namespace
ResourceAnalyzer::ResourceAnalyzer(ModuleOp module, bool skip_session_init) {
  auto session_init_func = GetSessionInitializerFunc(module);
  for (auto func : module.getOps<FuncOp>()) {
    if (skip_session_init && func == session_init_func) continue;
    (void)AnalyzeRegion(func.getRegion());
  }
}

void ResourceAnalyzer::SetPotentiallyWritten(Value resource) {
  assert(IsResource(resource));
  resource_infos_[resource].potentially_written = true;
  auto* operation = resource.getDefiningOp();
  if (operation && llvm::isa<TF::VarHandleOp>(operation)) {
    mutable_variables_.insert(GetResourceKey(operation));
  }
}

bool ResourceAnalyzer::IsPotentiallyWritten(Value resource) const {
  assert(IsResource(resource));
  auto* operation = resource.getDefiningOp();
  if (operation && llvm::isa<TF::VarHandleOp>(operation))
    return mutable_variables_.contains(GetResourceKey(operation));
  auto it = resource_infos_.find(resource);
  if (it == resource_infos_.end()) {
    return false;
  }
  return it->second.potentially_written;
}

// Analyze the specified region for resource mutating operations, namely
// TF::AssignVariableOp, if so, set the resource associated as "potentially
// written". Do this recursively across the chain of regions via call or
// control flow ops.
// TODO(ashwinm): Move to iterative traversal.
LogicalResult ResourceAnalyzer::AnalyzeRegion(Region& region) {
  // Avoid infinite recursion.
  if (!discovered_.insert(&region).second) {
    return success();
  }

  region.walk([&](Operation* op) {
    if (isa<TF::ReadVariableOp, ReturnOp, YieldOp>(op)) {
      return;
    }
    if (auto assign_variable = dyn_cast<TF::AssignVariableOp>(op)) {
      SetPotentiallyWritten(assign_variable.resource());
      return;
    }
    if (auto call = dyn_cast<CallOpInterface>(op)) {
      if (auto func = dyn_cast<FuncOp>(call.resolveCallable())) {
        PropagatePotentiallyWrittenUpFromCallee(func.getRegion(),
                                                call.getArgOperands());
      }
      return;
    }
    if (auto if_op = dyn_cast<TF::IfOp>(op)) {
      for (auto callee : {if_op.then_function(), if_op.else_function()}) {
        PropagatePotentiallyWrittenUpFromCallee(callee.getRegion(),
                                                if_op.input());
      }
      return;
    }
    if (auto if_op = dyn_cast<TF::IfRegionOp>(op)) {
      PropagatePotentiallyWrittenUpFromCallee(if_op.then_branch(),
                                              if_op.getODSOperands(1));
      PropagatePotentiallyWrittenUpFromCallee(if_op.else_branch(),
                                              if_op.getODSOperands(1));
      return;
    }
    if (auto while_op = dyn_cast<TF::WhileOp>(op)) {
      for (auto callee : {while_op.cond_function(), while_op.body_function()}) {
        PropagatePotentiallyWrittenUpFromCallee(callee.getRegion(),
                                                while_op.input());
      }
      return;
    }
    if (auto while_op = dyn_cast<TF::WhileRegionOp>(op)) {
      PropagatePotentiallyWrittenUpFromCallee(while_op.cond(),
                                              while_op.input());
      PropagatePotentiallyWrittenUpFromCallee(while_op.body(),
                                              while_op.input());
      return;
    }
    // For all other ops, we assume it mutates all resources it uses, so
    // this errs on the side of being conservative. We should improve
    // this by using either a property or a trait that clearly
    // identifies ops with resource mutating behavior.
    PropagatePotentiallyWrittenWithinUnhandledOp(op);
  });
  return success();
}

void ResourceAnalyzer::PropagatePotentiallyWrittenWithinUnhandledOp(
    Operation* op) {
  for (auto operand : op->getOperands()) {
    if (IsResource(operand)) {
      SetPotentiallyWritten(operand);
    }
  }
  visitUsedValuesDefinedAbove(op->getRegions(), [&](OpOperand* operand) {
    if (IsResource(operand->get())) {
      SetPotentiallyWritten(operand->get());
    }
  });
}

void ResourceAnalyzer::PropagatePotentiallyWrittenUpFromCallee(
    Region& region, Operation::operand_range propagate_to) {
  (void)AnalyzeRegion(region);
  for (auto t : llvm::zip(region.getArguments(), propagate_to)) {
    if (!IsResource(std::get<0>(t))) {
      continue;
    }
    if (IsPotentiallyWritten(std::get<0>(t))) {
      SetPotentiallyWritten(std::get<1>(t));
    }
  }
}
}  // namespace TF
}  // namespace mlir
