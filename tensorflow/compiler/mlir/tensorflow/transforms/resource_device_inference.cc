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
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_alias_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/visitor_util.h"

#define DEBUG_TYPE "tf-resource-device-inference"

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
    : public ResourceDeviceInferencePassBase<ResourceDeviceInference> {
  void runOnOperation() override;
};

// A class that records each resource's device assignment in a function.
class PerFunctionResult {
 public:
  explicit PerFunctionResult(
      FuncOp func_op, const TF::ResourceAliasAnalysis::Info& alias_analysis)
      : alias_analysis_(alias_analysis) {}

  // Returns the recorded device assignment for a resource, if any.
  Optional<StringRef> DeviceForResource(Value resource) const {
    Optional<StringRef> result;
    if (alias_analysis_.IsUnknownResource(resource)) return llvm::None;
    for (int64_t id : alias_analysis_.GetResourceUniqueIds(resource)) {
      auto it = resource_id_to_device_.find(id);
      if (it == resource_id_to_device_.end()) continue;
      if (!result || result == it->second) {
        result = it->getSecond();
        continue;
      }
      // Got conflicting assignments
      return llvm::None;
    }
    return result;
  }

  // Records the device assignment for a resource. If the new assignment
  // conflicts with an existing one, returns an error.
  //
  // If `changed` is provided, assign *changed to true if anything is modified.
  LogicalResult AddResourceDevice(Value resource, StringRef device,
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
  llvm::SmallDenseMap<int64_t, StringRef, 8> resource_id_to_device_;
  const TF::ResourceAliasAnalysis::Info& alias_analysis_;
};

// Tries to record device assignment for a resource.
LogicalResult AddResourceDeviceAndEmitError(Value resource, StringRef device,
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

// Extracts and canonicalizes the device attribute.
inline StringRef GetDeviceAttr(FuncOp func, int arg_no) {
  auto device_attr =
      func.getArgAttrOfType<mlir::StringAttr>(arg_no, kFuncDeviceAttr);
  return device_attr ? device_attr.getValue() : "";
}

// Extracts and canonicalizes the device attribute.
inline StringRef GetDeviceAttr(Operation* op) {
  auto device_attr = op->getAttrOfType<mlir::StringAttr>(kDeviceAttr);
  return device_attr ? device_attr.getValue() : "";
}

// Print operation with debug info (to get line number info for debugging)
void dump(StringRef message, Operation* op) {
  llvm::dbgs() << message;
  op->print(llvm::dbgs(), OpPrintingFlags().enableDebugInfo(true));
  llvm::dbgs() << "\n";
}

// Propagates device assignment inside a function.
LogicalResult ComputeResourceDevicesInComputation(FuncOp func_op,
                                                  PerFunctionResult* result) {
  OpBuilder builder(func_op);
  // Function arguments.
  for (auto arg : filter_resources(func_op.getArguments())) {
    StringRef device_attr = GetDeviceAttr(func_op, arg.getArgNumber());
    if (device_attr.empty()) {
      // If device_attr does not exist, try to construct it from any recorded
      // assignment.
      if (auto device = result->DeviceForResource(arg)) {
        func_op.setArgAttr(arg.getArgNumber(), kFuncDeviceAttr,
                           builder.getStringAttr(*device));
      }
      continue;
    }
    // Record the attribute.
    auto res = AddResourceDeviceAndEmitError(arg, device_attr, func_op, result);
    if (failed(res)) return res;
  }

  // To support WhileRegion, we need to propagate device attributes from
  // WhileRegion operands to body/cond region arguments *prior* to visiting
  // these regions. Use tensorflow::walk() instead of MLIR core walker to
  // implement such a pre-order walk.
  auto walk_res = tensorflow::GenericWalk(
      func_op, [&](Operation* op, const tensorflow::WalkStage& stage) {
        // We just need to visit operations in pre-order mode.
        if (!stage.IsBeforeAllRegions()) return WalkResult::advance();

        if (auto var_handle = dyn_cast<VarHandleOp>(op)) {
          // Record VarHandleOp's device attribute.
          StringRef device_attr = GetDeviceAttr(op);
          if (device_attr.empty()) return WalkResult::advance();
          auto res = AddResourceDeviceAndEmitError(var_handle.resource(),
                                                   device_attr, op, result);
          if (failed(res)) return WalkResult::interrupt();
        } else if (auto identity = dyn_cast<IdentityOp>(op)) {
          LLVM_DEBUG(dump("Visiting ", identity));
          // Try to construct IdentityOp's attribute from recorded assignment.
          if (!GetDeviceAttr(op).empty()) return WalkResult::advance();
          for (auto output : filter_resources(op->getResults())) {
            LLVM_DEBUG(llvm::dbgs() << "  Processing output #"
                                    << output.getResultNumber() << "\n");
            if (auto device = result->DeviceForResource(output)) {
              LLVM_DEBUG(llvm::dbgs()
                         << " Setting device = " << *device << "\n");
              identity->setAttr(kDeviceAttr, builder.getStringAttr(*device));
            }
          }
        } else if (auto while_region = dyn_cast<WhileRegionOp>(op)) {
          // For WhileRegion, do local analysis prior to visiting the attached
          // regions and propagate device annotations to the cond and body
          // region arguments. The annotations are the union of annotations
          // on the input and result. Resource alias analysis already propagates
          // resource ID from the inputs to the results for a while, so just
          // need to consider the results.
          LLVM_DEBUG(llvm::dbgs() << "Visiting WhileRegion\n");

          for (auto output : filter_resources(while_region.getResults())) {
            auto device = result->DeviceForResource(output);
            int output_index = output.getResultNumber();
            if (!device) {
              LLVM_DEBUG(llvm::dbgs()
                         << "  No device for output #" << output_index << "\n");
              continue;
            }
            // Transfer the annotation to both region arguments
            for (Region* region : while_region.getRegions()) {
              BlockArgument arg = region->getArgument(output_index);
              LLVM_DEBUG(llvm::dbgs()
                         << "  Propagating device = '" << *device
                         << "' to arg #" << output_index << " of region #"
                         << region->getRegionNumber() << "\n");
              if (failed(AddResourceDeviceAndEmitError(arg, *device,
                                                       while_region, result)))
                return WalkResult::interrupt();
            }
          }
        }
        return WalkResult::advance();
      });
  return failure(walk_res.wasInterrupted());
}

void ResourceDeviceInference::runOnOperation() {
  auto module = getOperation();
  const auto& resource_alias_analysis =
      getAnalysis<TF::ResourceAliasAnalysis>();

  llvm::SmallDenseMap<FuncOp, PerFunctionResult, 4> per_function_results;
  llvm::SetVector<FuncOp> worklist;
  for (auto func_op : module.getOps<FuncOp>()) {
    worklist.insert(func_op);
    per_function_results.try_emplace(
        func_op, func_op, resource_alias_analysis.GetAnalysisForFunc(func_op));
  }
  // Helper that propagates an op's recorded operand device assignments to its
  // called function's arguments.
  auto propagate_operands_to_callee_arguments =
      [&](Operation* caller, Operation::operand_range caller_operands,
          ArrayRef<FuncOp> callees, const PerFunctionResult& caller_res) {
        for (FuncOp callee : callees) {
          assert(callee);
          auto& callee_res = per_function_results.find(callee)->getSecond();
          bool callee_needs_recompute = false;
          for (BlockArgument arg : filter_resources(callee.getArguments())) {
            Value arg_operand = caller_operands[arg.getArgNumber()];
            auto device = caller_res.DeviceForResource(arg_operand);
            if (!device) continue;
            LLVM_DEBUG(llvm::dbgs()
                       << "Propagating '" << *device << "' to arg #"
                       << arg.getArgNumber() << " of function @"
                       << callee.getName() << "\n");
            if (failed(AddResourceDeviceAndEmitError(arg, *device, caller,
                                                     &callee_res,
                                                     &callee_needs_recompute)))
              return failure();
          }
          // If the callee recording is modified, make sure that it will be
          // reprocessed.
          if (callee_needs_recompute) worklist.insert(callee);
        }
        return success();
      };

  while (!worklist.empty()) {
    auto func_op = worklist.pop_back_val();
    auto& func_res = per_function_results.find(func_op)->getSecond();
    // In-function propagation.
    if (failed(ComputeResourceDevicesInComputation(func_op, &func_res)))
      return signalPassFailure();

    // Propagation to callees.
    auto walk_res = func_op.walk([&](Operation* op) {
      if (auto while_op = dyn_cast<WhileOp>(op)) {
        if (failed(propagate_operands_to_callee_arguments(
                while_op, while_op.getOperands(),
                {while_op.body_function(), while_op.cond_function()},
                func_res)))
          return WalkResult::interrupt();
      } else if (auto if_op = dyn_cast<IfOp>(op)) {
        if (failed(propagate_operands_to_callee_arguments(
                if_op, if_op.input(),
                {if_op.then_function(), if_op.else_function()}, func_res)))
          return WalkResult::interrupt();
      } else if (auto call = dyn_cast<CallOpInterface>(op)) {
        auto func = dyn_cast<FuncOp>(call.resolveCallable());
        if (!func) {
          op->emitError(
              "Cannot propagate device attribute to callee: Unable to resolve "
              "call");
          return WalkResult::interrupt();
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "Visiting call to function @" << func.getName() << "\n");
        if (failed(propagate_operands_to_callee_arguments(
                call, call.getArgOperands(), {func}, func_res)))
          return WalkResult::interrupt();
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

}  // namespace TF
}  // namespace mlir
