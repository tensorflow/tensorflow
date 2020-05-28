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

#include <iterator>
#include <memory>
#include <tuple>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {

namespace {
constexpr char kDeviceAttr[] = "device";
constexpr char kFuncDeviceAttr[] = "tf.device";

// A pass that propagates device assignment of resources on a module. It
// performs in-function propagation, as well as cross-function propagation from
// callers to callees.
//
// This pass changes the module by adding "tf.device" attribute to function
// arguments and adding "device" attribute to TF ops.
struct ResourceDeviceInference
    : public PassWrapper<ResourceDeviceInference, OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

// A class that records each resource's device assignment in a function.
class PerFunctionResult {
 public:
  explicit PerFunctionResult(FuncOp func_op) : alias_analysis_(func_op) {}

  // Returns the recorded device assignment for a resource, if any.
  llvm::Optional<llvm::StringRef> DeviceForResource(
      const Value resource) const {
    llvm::Optional<llvm::StringRef> result;
    if (alias_analysis_.IsUnknownResource(resource)) return result;
    for (int64_t id : alias_analysis_.GetResourceUniqueIds(resource)) {
      auto it = resource_id_to_device_.find(id);
      if (it == resource_id_to_device_.end()) continue;
      if (!result) {
        result = it->getSecond();
        continue;
      }
      if (result != it->getSecond()) {
        // Got conflicting assignments, clear the result.
        result.reset();
        return result;
      }
    }
    return result;
  }

  // Records the device assignment for a resource. If the new assignment
  // conflicts with an existing one, returns an error.
  //
  // If `changed` is provided, assign *changed to true if anything is modified.
  LogicalResult AddResourceDevice(const Value resource, llvm::StringRef device,
                                  bool* changed = nullptr) {
    if (alias_analysis_.IsUnknownResource(resource)) return success();
    for (int64_t id : alias_analysis_.GetResourceUniqueIds(resource)) {
      auto emplace_res = resource_id_to_device_.try_emplace(id, device);
      if (emplace_res.second) {
        if (changed) *changed = true;
      } else if (emplace_res.first->getSecond() != device) {
        // Existing assignment does not equal the new assignment.
        return failure();
      }
    }
    return success();
  }

 private:
  llvm::SmallDenseMap<int64_t, llvm::StringRef, 8> resource_id_to_device_;
  TF::ResourceAliasAnalysis alias_analysis_;
};

// Tries to record device assignment for a resource.
LogicalResult AddResourceDeviceAndEmitError(const Value resource,
                                            llvm::StringRef device,
                                            Operation* error_reporting_op,
                                            PerFunctionResult* result,
                                            bool* changed = nullptr) {
  auto res = result->AddResourceDevice(resource, device, changed);
  if (failed(res)) {
    error_reporting_op->emitError()
        << "Conflicting device assignment for resource";
  }
  return res;
}

// Propagates device assignment inside a function.
LogicalResult ComputeResourceDevicesInComputation(FuncOp func_op,
                                                  PerFunctionResult* result) {
  OpBuilder builder(func_op);
  // Function arguments.
  for (auto arg : func_op.getArguments()) {
    if (!mlir::getElementTypeOrSelf(arg.getType()).isa<TF::ResourceType>()) {
      continue;
    }
    auto device_attr = func_op.getArgAttrOfType<mlir::StringAttr>(
        arg.getArgNumber(), kFuncDeviceAttr);
    if (!device_attr || device_attr.getValue() == "") {
      // If device_attr does not exist, try to construct it from any recorded
      // assignment.
      if (auto device = result->DeviceForResource(arg)) {
        func_op.setArgAttr(arg.getArgNumber(), kFuncDeviceAttr,
                           builder.getStringAttr(*device));
      }
      continue;
    }
    // Record the attribute.
    auto res = AddResourceDeviceAndEmitError(arg, device_attr.getValue(),
                                             func_op, result);
    if (failed(res)) return res;
  }
  auto walk_res = func_op.walk([&](Operation* op) {
    if (auto var_handle = llvm::dyn_cast<TF::VarHandleOp>(op)) {
      // Record VarHandleOp's device attribute.
      auto device_attr =
          var_handle.getAttrOfType<mlir::StringAttr>(kDeviceAttr);
      if (!device_attr || device_attr.getValue().empty()) {
        return WalkResult::advance();
      }
      auto res = AddResourceDeviceAndEmitError(
          var_handle.resource(), device_attr.getValue(), op, result);
      if (failed(res)) return WalkResult::interrupt();
    }
    if (auto identity = llvm::dyn_cast<TF::IdentityOp>(op)) {
      // Try to construct IdentityOp's attribute from recorded assignment.
      if (!mlir::getElementTypeOrSelf(identity.output().getType())
               .isa<TF::ResourceType>()) {
        return WalkResult::advance();
      }
      if (auto device = result->DeviceForResource(identity.output())) {
        auto device_attr =
            identity.getAttrOfType<mlir::StringAttr>(kDeviceAttr);
        if (!device_attr || device_attr.getValue().empty()) {
          identity.setAttr(kDeviceAttr, builder.getStringAttr(*device));
        }
      }
      return WalkResult::advance();
    }
    // Propagate and record output device assignment for other ops based on
    // existing recording. E.g., IdentityN.
    for (auto output : op->getResults()) {
      if (!mlir::getElementTypeOrSelf(output.getType())
               .isa<TF::ResourceType>()) {
        continue;
      }
      if (auto device = result->DeviceForResource(output)) {
        auto res = AddResourceDeviceAndEmitError(output, *device, op, result);
        if (failed(res)) return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(walk_res.wasInterrupted());
}

void ResourceDeviceInference::runOnOperation() {
  auto module = getOperation();
  llvm::SmallDenseMap<Operation*, PerFunctionResult, 4> per_function_results;
  llvm::SetVector<FuncOp> worklist;
  module.walk([&](FuncOp func_op) {
    worklist.insert(func_op);
    per_function_results.try_emplace(func_op, func_op);
  });
  // Helper that propagates an op's recorded operand device assignments to its
  // called function's arguments.
  auto propagate_operands_to_callee_arguments =
      [&](Operation* caller, Operation::operand_range caller_operands,
          llvm::StringRef called_func_name,
          const PerFunctionResult& caller_res) {
        auto callee =
            llvm::dyn_cast<FuncOp>(module.lookupSymbol(called_func_name));
        assert(callee);
        auto& callee_res = per_function_results.find(callee)->getSecond();
        bool callee_needs_recompute = false;
        for (auto operand_and_argument :
             llvm::zip(caller_operands, callee.getArguments())) {
          if (!mlir::getElementTypeOrSelf(
                   std::get<0>(operand_and_argument).getType())
                   .isa<TF::ResourceType>()) {
            continue;
          }
          auto device =
              caller_res.DeviceForResource(std::get<0>(operand_and_argument));
          if (!device) continue;
          if (failed(AddResourceDeviceAndEmitError(
                  std::get<1>(operand_and_argument), *device, caller,
                  &callee_res, &callee_needs_recompute))) {
            return failure();
          }
        }
        // If the callee recording is modified, make sure that it will be
        // reprocessed.
        if (callee_needs_recompute) {
          worklist.insert(callee);
        }
        return success();
      };
  while (!worklist.empty()) {
    auto func_op = worklist.back();
    worklist.pop_back();
    auto& func_res = per_function_results.find(func_op)->getSecond();
    // In-function propagation.
    if (failed(ComputeResourceDevicesInComputation(func_op, &func_res))) {
      return signalPassFailure();
    }
    // Propagation to callees.
    auto walk_res = func_op.walk([&](Operation* op) {
      if (auto while_op = llvm::dyn_cast<TF::WhileOp>(op)) {
        if (failed(propagate_operands_to_callee_arguments(
                while_op, while_op.getOperands(), while_op.body(), func_res)) ||
            failed(propagate_operands_to_callee_arguments(
                while_op, while_op.getOperands(), while_op.cond(), func_res))) {
          return WalkResult::interrupt();
        }
      } else if (auto if_op = llvm::dyn_cast<TF::IfOp>(op)) {
        if (failed(propagate_operands_to_callee_arguments(
                if_op, if_op.input(), if_op.then_branch(), func_res)) ||
            failed(propagate_operands_to_callee_arguments(
                if_op, if_op.input(), if_op.else_branch(), func_res))) {
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (walk_res.wasInterrupted()) return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateResourceDeviceInferencePass() {
  return std::make_unique<ResourceDeviceInference>();
}

static PassRegistration<ResourceDeviceInference> pass(
    "tf-resource-device-inference",
    "Propagates the device attribute on resources from callers to callees.");

}  // namespace TF
}  // namespace mlir
