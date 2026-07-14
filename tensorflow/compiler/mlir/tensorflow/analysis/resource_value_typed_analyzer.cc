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

#include <cassert>
#include <tuple>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {
namespace {

bool IsResourceType(Type type) {
  if (auto tensor_type = mlir::dyn_cast<TensorType>(type)) {
    return mlir::isa<TF::ResourceType>(tensor_type.getElementType());
  }
  return false;
}

bool IsResource(Value value) { return IsResourceType(value.getType()); }

// Helper that returns the FuncOp that is the SessionInit function which
// will be called to initialize all resources.
// Returns nullptr if no function is found.
func::FuncOp GetSessionInitializerFunc(ModuleOp module) {
  auto session_init_op = tf_saved_model::GetSessionInitializerOp(module);
  if (session_init_op && !session_init_op.getInitializers().empty()) {
    SymbolTable symbol_table(module);
    func::FuncOp init_func_op = symbol_table.lookup<func::FuncOp>(
        mlir::cast<FlatSymbolRefAttr>(session_init_op.getInitializers()[0])
            .getValue());
    return init_func_op;
  }
  return nullptr;
}

// Returns ID for identifying a resource.
std::tuple<StringRef, StringRef, StringRef> GetResourceKey(Operation* op) {
  StringRef device;
  if (auto attr = op->getAttrOfType<StringAttr>("device")) {
    device = attr.getValue();
  }

  StringRef container;
  if (auto attr = op->getAttrOfType<StringAttr>("container")) {
    container = attr.getValue();
  }

  StringRef shared_name;
  if (auto attr = op->getAttrOfType<StringAttr>("shared_name")) {
    shared_name = attr.getValue();
  }

  return std::tuple<StringRef, StringRef, StringRef>{device, container,
                                                     shared_name};
}

}  // namespace

ResourceAnalyzer::ResourceAnalyzer(ModuleOp module, bool skip_session_init) {
  auto session_init_func = GetSessionInitializerFunc(module);
  for (auto func : module.getOps<func::FuncOp>()) {
    if (skip_session_init && func == session_init_func) continue;
    (void)AnalyzeRegion(func.getRegion());
  }
}

void ResourceAnalyzer::SetPotentiallyWritten(Value resource) {
  assert(IsResource(resource));
  resource_infos_[resource].potentially_written = true;
  auto* operation = resource.getDefiningOp();
  if (operation && isa<TF::VarHandleOp>(operation)) {
    mutable_variables_.insert(GetResourceKey(operation));
  }
}

bool ResourceAnalyzer::IsPotentiallyWritten(Value resource) const {
  assert(IsResource(resource));
  auto* operation = resource.getDefiningOp();
  if (operation && isa<TF::VarHandleOp>(operation))
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
    if (isa<TF::ReadVariableOp, func::ReturnOp, YieldOp>(op)) {
      return;
    }
    if (auto assign_variable = dyn_cast<TF::AssignVariableOp>(op)) {
      SetPotentiallyWritten(assign_variable.getResource());
      return;
    }
    if (auto call = dyn_cast<CallOpInterface>(op)) {
      if (auto func = dyn_cast<func::FuncOp>(call.resolveCallable())) {
        PropagatePotentiallyWrittenUpFromCallee(func.getRegion(),
                                                call.getArgOperands());
      }
      return;
    }
    if (auto if_op = dyn_cast<TF::IfOp>(op)) {
      for (auto callee : {if_op.then_function(), if_op.else_function()}) {
        PropagatePotentiallyWrittenUpFromCallee(callee.getRegion(),
                                                if_op.getInput());
      }
      return;
    }
    if (auto if_op = dyn_cast<TF::IfRegionOp>(op)) {
      PropagatePotentiallyWrittenUpFromCallee(if_op.getThenBranch(),
                                              if_op.getODSOperands(1));
      PropagatePotentiallyWrittenUpFromCallee(if_op.getElseBranch(),
                                              if_op.getODSOperands(1));
      return;
    }
    if (auto while_op = dyn_cast<TF::WhileOp>(op)) {
      for (auto callee : {while_op.cond_function(), while_op.body_function()}) {
        PropagatePotentiallyWrittenUpFromCallee(callee.getRegion(),
                                                while_op.getInput());
      }
      return;
    }
    if (auto while_op = dyn_cast<TF::WhileRegionOp>(op)) {
      PropagatePotentiallyWrittenUpFromCallee(while_op.getCond(),
                                              while_op.getInput());
      PropagatePotentiallyWrittenUpFromCallee(while_op.getBody(),
                                              while_op.getInput());
      return;
    }
    // `TF::BatchFunctionOp`, although it looks like a function call, does not
    // interface the `CallOpInterface` so it should be handled separately.
    if (auto batch_function = dyn_cast<TF::BatchFunctionOp>(op)) {
      // Propagate the analysis results from within the callee's body.
      PropagatePotentiallyWrittenUpFromCallee(batch_function.func().getRegion(),
                                              batch_function.getOperands());
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
