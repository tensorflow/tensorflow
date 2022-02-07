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

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

#define DEBUG_TYPE "tf-device-mark-input-output-aliases"

namespace mlir {
namespace TFDevice {

namespace {
struct MarkInputOutputAliasesPass
    : public TF::MarkInputOutputAliasesPassBase<MarkInputOutputAliasesPass> {
  void runOnOperation() override;
};

constexpr char kAliasingAttr[] = "tf.aliasing_output";
constexpr int kUnassigned = -1;

struct AliasInfo {
  AliasInfo() : input_index(kUnassigned), output_index(kUnassigned) {}
  int input_index;
  int output_index;
};

// Returns the consuming `AssignVariableOp` user of `result`, if there is one
// and if it is unique.
// TODO(b/184420848) Generalize this to work in more situations, e.g.:
// - consider more pass-through ops (like `tf_device::ReturnOp`)
// - consider multiple uses
absl::optional<TF::AssignVariableOp> FindConsumingAssignVariableOp(
    OpResult& result) {
  // We currently don't handle multiple uses.
  if (!result.hasOneUse()) return absl::nullopt;

  OpOperand& use = *(result.use_begin());
  Operation* user = use.getOwner();
  if (!user) return absl::nullopt;

  // Follow the value through `tf_device::ReturnOp` inside of a parallel execute
  // region.
  auto parent_op = user->getParentOp();
  if (llvm::isa<tf_device::ReturnOp>(user) && parent_op &&
      llvm::isa<tf_device::ParallelExecuteOp>(parent_op)) {
    Region* parent_region = user->getParentRegion();
    auto parallel_execute = llvm::cast<tf_device::ParallelExecuteOp>(parent_op);

    // Get the parallel execute result that corresponds to `use`.
    int operand_num = use.getOperandNumber();
    auto region_outputs =
        parallel_execute.GetRegionOutputs(parent_region->getRegionNumber());
    if (operand_num >= region_outputs.size()) return absl::nullopt;
    OpResult parallel_execute_result = region_outputs[operand_num];

    // We currently don't handle multiple uses.
    if (!parallel_execute_result.hasOneUse()) return absl::nullopt;

    user = parallel_execute_result.use_begin()->getOwner();
    if (!user) return absl::nullopt;
  }
  auto assign_op = llvm::dyn_cast_or_null<TF::AssignVariableOp>(user);
  if (!assign_op) return absl::nullopt;
  return assign_op;
}

// Identify tf_device.cluster_func input-output alias pairs.
// This is currently conservative, and handles the following simple case:
// ```
// %value = tf.ReadVariableOp(%resource_var)
// %output:N = tf_device.cluster_func(..., /*input index = a*/ %value, ...)
// tf.AssignVariableOp(%resource_var, %output#b) // write output #b to resource
// ```
// where `%value` and `%output#b` have only one use. (a, b) would be added as
// input-output alias pair for `%resource_var`.
//
// TODO(b/184420848): Explore relaxing these constraints.
LogicalResult BuildAliasingInfo(
    tf_device::ClusterFuncOp cluster_func,
    llvm::DenseMap<Value, AliasInfo>& resource_alias_info_map) {
  for (OpResult result : cluster_func.getResults()) {
    auto assign_op_optional = FindConsumingAssignVariableOp(result);
    if (!assign_op_optional.has_value()) continue;
    TF::AssignVariableOp assign_op = assign_op_optional.value();
    AliasInfo& alias_info = resource_alias_info_map[assign_op.resource()];
    // TODO(b/184420848): We may not need to skip aliasing for entire function
    // in case of multiple assigns.
    if (alias_info.output_index != kUnassigned) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Skip adding aliasing information because of multiple assigns to "
             "the same resource from tf_device.cluster_func outputs. This can "
             "lead to poor memory management on device.\n");

      return failure();
    }
    alias_info.output_index = result.getResultNumber();
  }

  for (auto& operand : cluster_func->getOpOperands()) {
    auto read_op = llvm::dyn_cast_or_null<TF::ReadVariableOp>(
        operand.get().getDefiningOp());
    if (!read_op) continue;
    if (!read_op->hasOneUse()) continue;
    auto it = resource_alias_info_map.find(read_op.resource());
    if (it == resource_alias_info_map.end()) continue;
    AliasInfo& alias_info = it->getSecond();
    // TODO(b/184420848): We may not need to skip aliasing for entire function
    // in case of multiple reads from same resource variable.
    if (alias_info.input_index != kUnassigned) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Skip adding aliasing information because of multiple reads of "
             "the same resource in tf_device.cluster_func inputs. This can "
             "lead to poor memory management on device.\n");
      return failure();
    }

    alias_info.input_index = operand.getOperandNumber();
  }
  return success();
}

void AddAliasingAttributeToDeviceFunc(
    FuncOp device_func,
    llvm::DenseMap<Value, AliasInfo>& resource_alias_info_map) {
  OpBuilder builder(device_func.getContext());
  for (const auto& resource_alias_entry : resource_alias_info_map) {
    const AliasInfo& alias_info = resource_alias_entry.second;
    if (alias_info.input_index == kUnassigned ||
        alias_info.output_index == kUnassigned)
      continue;
    auto aliasing_attr = device_func.getArgAttrOfType<mlir::IntegerAttr>(
        alias_info.input_index, kAliasingAttr);

    // Set only if aliasing attribute does not exist.
    if (!aliasing_attr) {
      device_func.setArgAttr(
          alias_info.input_index, kAliasingAttr,
          builder.getI64IntegerAttr(alias_info.output_index));
      continue;
    }
    // If aliasing attribute already exists, it must match the new value.
    assert(aliasing_attr.getInt() == alias_info.output_index);
  }
}

void MarkInputOutputAliasesPass::runOnOperation() {
  SmallVector<tf_device::ClusterFuncOp, 4> cluster_funcs;
  ModuleOp module = getOperation();
  module.walk([&](tf_device::ClusterFuncOp cluster_func) {
    // Map resource values to pair of input-output indices.
    llvm::DenseMap<Value, AliasInfo> resource_alias_info_map;
    if (failed(BuildAliasingInfo(cluster_func, resource_alias_info_map)) ||
        resource_alias_info_map.empty()) {
      return;
    }

    FlatSymbolRefAttr func_attr = cluster_func.funcAttr();
    FuncOp device_func = module.lookupSymbol<FuncOp>(func_attr.getValue());
    AddAliasingAttributeToDeviceFunc(device_func, resource_alias_info_map);
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateMarkInputOutputAliasesPass() {
  return std::make_unique<MarkInputOutputAliasesPass>();
}

}  // namespace TFDevice
}  // namespace mlir
