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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Identifier.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "mlir/Transforms/RegionUtils.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

#define DEBUG_TYPE "tf-tpu-merge-variables-with-execute"

namespace mlir {
namespace TFTPU {

namespace {
constexpr char kDeviceAttr[] = "device";
constexpr char kFuncDeviceAttr[] = "tf.device";

// A pass that finds on-device resource variable reads/assigns surrounding a
// tf.TPUExecute op, and merges them into a tf.TPUExecuteAndUpdateVariables.
// This allows the TPU execution to perform in-place variable updates.
//
// For example,
//
//   %0 = "tf.ReadVariableOp"(%arg0)
//   %1 = "tf.ReadVariableOp"(%arg1)
//   %2 = "tf.TPUExecute"(%0, %1, %compile)
//   %3 = "tf.AssignVariableOp"(%arg0, %2)
//
// will be transformed into
//
//   %2 = "tf.TPUExecuteAndUpdateVariables"(%arg0, %arg1, %compile)
//     { device_var_reads_indices = [0, 1],
//       device_var_updates_indices = [0, -1] }
//
// The transformation happens only for on-device variables. The above
// transformation requires %arg0, %arg1 to have the same device assignment as
// the TPUExecute op.

struct TPUMergeVariablesWithExecutePass
    : public FunctionPass<TPUMergeVariablesWithExecutePass> {
  void runOnFunction() override;
};

// Information for a pair of input/output of the TPUExecute op and the
// surrounding read/assign ops.
struct VariableAccessInfo {
  int execute_input_index = -1;
  int execute_output_index = -1;
  Operation* read = nullptr;
  Operation* assign = nullptr;
};

// Information about all resource accesses to be fused into a TPUExecute op.
struct VariableAccessesForTPUExecute {
  // Maps each resource detected to VariableAccessInfo.
  llvm::SmallDenseMap<Value*, VariableAccessInfo, 8> per_resource_info;
  // The corresponding new output index in TPUExecuteAndUpdateVariables for
  // each old output index in TPUExecute.
  llvm::SmallVector<int, 8> old_to_new_output_mapping;
  // The resources read by ReadVariableOps that are inputs to TPUExecute.
  // Ordered by the input indices to TPUExecute
  llvm::SmallVector<Value*, 8> resources_read;
  // Operands for the new TPUExecuteAndUpdateVariables.
  llvm::SmallVector<Value*, 8> new_operand_values;
};

// Returns if an op accesses a resource.
//
// TODO(yuanzx): Decide how to make this fine-grained. Right now we do not know
// if the resources alias.
bool OpAccessesResource(Operation* op) {
  return llvm::any_of(op->getOperandTypes(), [](const Type& type) {
    return type.isa<TF::ResourceType>() ||
           (type.isa<TensorType>() &&
            type.cast<TensorType>().getElementType().isa<TF::ResourceType>());
  });
}

// Finds the variable access info for a TPUExecute op. `check_device` specifies
// whether it checks the device assignment of the variables to match the
// TPUExecute op. This is optional in some context, e.g., guaranteed by
// replication.
VariableAccessesForTPUExecute BuildVariableAccessInfo(Operation* execute,
                                                      bool check_device) {
  VariableAccessesForTPUExecute infos;
  auto device_attr = execute->getAttr(kDeviceAttr);
  if (check_device && !device_attr) return infos;
  auto func = execute->getParentOfType<mlir::FuncOp>();

  // Track the first read op found, which is used later to check if there are
  // assign ops between it and the TPUExecute op. We will exclude reads before
  // interferencing accesses in a conservative way (see below). We do not
  // consider resource accesses in other islands since they ordering is enforced
  // by inter-island dependencies.
  Operation* first_read = nullptr;
  // Find inputs that are variable reads.
  for (auto operand : llvm::enumerate(execute->getOpOperands())) {
    infos.new_operand_values.push_back(operand.value().get());
    if (!operand.value().get()->getDefiningOp()) continue;
    auto read_op = llvm::dyn_cast<TF::ReadVariableOp>(
        operand.value().get()->getDefiningOp());
    if (!read_op) continue;
    auto resource = read_op.resource();

    if (check_device) {
      if (auto resource_op = resource->getDefiningOp()) {
        auto resource_attr = resource_op->getAttr(kDeviceAttr);
        // Check device matching for the node defining the resource.
        if (!resource_attr || resource_attr != device_attr) continue;
      } else {
        auto resource_arg = llvm::dyn_cast<BlockArgument>(resource);
        assert(resource_arg);
        // Check device matching for the argument defining the resource.
        auto resource_attr = func.getArgAttrOfType<mlir::StringAttr>(
            resource_arg->getArgNumber(), kFuncDeviceAttr);
        if (!resource_attr || resource_attr != device_attr) continue;
      }
    }

    auto emplace_res =
        infos.per_resource_info.try_emplace(resource, VariableAccessInfo());
    if (!emplace_res.second) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping execute that has multiple reads of a variable: "
                 << *execute << "\n");
      infos.per_resource_info.shrink_and_clear();
      return infos;
    }

    auto& info = emplace_res.first->getSecond();
    info.execute_input_index = operand.index();
    info.read = read_op;
    infos.new_operand_values[operand.index()] = resource;
    infos.resources_read.push_back(resource);
    if (!first_read || info.read->isBeforeInBlock(first_read)) {
      first_read = info.read;
    }
  }

  if (!first_read) return infos;

  // Conservatively find the last resource-accessing op between first_read and
  // execute, excluding ReadVariableOps since they are read-only. This should
  // work fine for the reads/assigns created by resource lifting, since they are
  // placed close to the TPUExecute.
  Operation* last_may_modify_resource_access_before_execute = nullptr;
  for (Operation& op : llvm::reverse(llvm::make_range(
           std::next(first_read->getIterator()), execute->getIterator()))) {
    if (llvm::dyn_cast<TF::ReadVariableOp>(&op)) continue;
    if (!OpAccessesResource(&op)) continue;
    last_may_modify_resource_access_before_execute = &op;
    break;
  }

  if (last_may_modify_resource_access_before_execute) {
    // Remove the reads before last_unknown_resource_access_before_execute.
    for (auto& op : llvm::make_range(
             first_read->getIterator(),
             last_may_modify_resource_access_before_execute->getIterator())) {
      auto read = llvm::dyn_cast<TF::ReadVariableOp>(&op);
      if (!read) continue;
      auto info_it = infos.per_resource_info.find(read.resource());
      if (info_it == infos.per_resource_info.end()) continue;
      int input_index = info_it->getSecond().execute_input_index;
      infos.new_operand_values[input_index] = execute->getOperand(input_index);
      infos.per_resource_info.erase(info_it);
    }
    infos.resources_read.erase(
        llvm::remove_if(infos.resources_read,
                        [&](const Value* resource) {
                          return infos.per_resource_info.count(resource) == 0;
                        }),
        infos.resources_read.end());
  }

  if (infos.per_resource_info.empty()) {
    return infos;
  }

  // Find outputs that are variable assigns.
  Operation* last_assign = nullptr;
  llvm::SmallPtrSet<Operation*, 8> all_assigns;
  llvm::SmallVector<bool, 8> output_fused(execute->getNumResults(), false);
  for (int i = 0; i < execute->getNumResults(); ++i) {
    auto result = execute->getResult(i);
    if (!result->hasOneUse()) continue;
    auto assign_op =
        llvm::dyn_cast<TF::AssignVariableOp>(*result->user_begin());
    if (!assign_op) continue;
    auto resource = assign_op.resource();
    auto it = infos.per_resource_info.find(resource);
    if (it == infos.per_resource_info.end()) continue;
    auto& info = it->getSecond();
    if (info.assign) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping execute that has multiple assigns of a variable: "
                 << *execute << "\n");
      infos.per_resource_info.shrink_and_clear();
      return infos;
    }
    info.execute_output_index = i;
    info.assign = assign_op;
    if (!last_assign || last_assign->isBeforeInBlock(assign_op)) {
      last_assign = assign_op;
    }
    all_assigns.insert(assign_op);
    output_fused[i] = true;
  }

  // Check if there are other resource accesses after execute.
  Operation* first_unknown_resource_access_after_execute = nullptr;
  if (last_assign) {
    for (auto& op : llvm::make_range(std::next(execute->getIterator()),
                                     last_assign->getIterator())) {
      if (all_assigns.count(&op) > 0) continue;
      if (!OpAccessesResource(&op)) continue;
      first_unknown_resource_access_after_execute = &op;
      break;
    }
  }
  if (first_unknown_resource_access_after_execute) {
    // Remove the assigns after first_unknown_resource_access_after_execute.
    for (auto& op : llvm::make_range(
             first_unknown_resource_access_after_execute->getIterator(),
             std::next(last_assign->getIterator()))) {
      if (auto assign = llvm::dyn_cast<TF::AssignVariableOp>(&op)) {
        if (all_assigns.count(assign) == 0) continue;
        auto info_it = infos.per_resource_info.find(assign.resource());
        if (info_it == infos.per_resource_info.end()) continue;
        output_fused[info_it->second.execute_output_index] = false;
        info_it->second.execute_output_index = -1;
        info_it->second.assign = nullptr;
      }
    }
  }

  // Populate infos.old_to_new_output_mapping.
  int new_output_index = 0;
  infos.old_to_new_output_mapping.resize(execute->getNumResults());
  for (int i = 0; i < execute->getNumResults(); ++i) {
    if (output_fused[i]) {
      infos.old_to_new_output_mapping[i] = -1;
    } else {
      infos.old_to_new_output_mapping[i] = new_output_index;
      ++new_output_index;
    }
  }
  return infos;
}

// Merges the variable accesses into one TPUExecute op.
void MergeForOneTPUExecute(Operation* execute, bool check_device,
                           OpBuilder* builder) {
  auto infos = BuildVariableAccessInfo(execute, check_device);
  if (infos.per_resource_info.empty()) {
    return;
  }
  // Start creating the new TPUExecuteAndUpdateVariables op.
  builder->setInsertionPoint(execute);
  // Output types. Skip the original outputs for fused assigns.
  llvm::SmallVector<Type, 8> new_output_types;
  int old_output_index = 0;
  for (const auto& type : execute->getResultTypes()) {
    if (infos.old_to_new_output_mapping[old_output_index] >= 0) {
      new_output_types.push_back(type);
    }
    ++old_output_index;
  }
  // The attributes for fused variable reads and updates.
  llvm::SmallVector<int64_t, 8> device_var_reads_indices;
  llvm::SmallVector<int64_t, 8> device_var_updates_indices;
  for (auto resource : infos.resources_read) {
    const auto& info = infos.per_resource_info[resource];
    device_var_reads_indices.push_back(info.execute_input_index);
    device_var_updates_indices.push_back(info.execute_output_index);
  }
  auto merged_execute = builder->create<TF::TPUExecuteAndUpdateVariablesOp>(
      execute->getLoc(), new_output_types, infos.new_operand_values,
      llvm::ArrayRef<NamedAttribute>{
          builder->getNamedAttr(
              "device_var_reads_indices",
              builder->getI64ArrayAttr(device_var_reads_indices)),
          builder->getNamedAttr(
              "device_var_updates_indices",
              builder->getI64ArrayAttr(device_var_updates_indices))});

  if (auto device = execute->getAttr(kDeviceAttr)) {
    merged_execute.setAttr(kDeviceAttr, device);
  }

  // Replace the uses.
  for (int i = 0; i < infos.old_to_new_output_mapping.size(); ++i) {
    if (infos.old_to_new_output_mapping[i] < 0) continue;
    execute->getResult(i)->replaceAllUsesWith(
        merged_execute.getResult(infos.old_to_new_output_mapping[i]));
  }
  // Remove the assign ops.
  for (const auto& entry : infos.per_resource_info) {
    const auto& info = entry.getSecond();
    if (info.assign) info.assign->erase();
  }
  // Remove the original TPUExecute op.
  execute->erase();
  // Remove the read ops if they have no more uses.
  for (const auto& entry : infos.per_resource_info) {
    const auto& info = entry.getSecond();
    if (info.read->use_empty()) info.read->erase();
  }
}

void TPUMergeVariablesWithExecutePass::runOnFunction() {
  // Find all the executes first, since we will mutate the nodes around each
  // execute.
  llvm::SmallVector<Operation*, 8> executes;
  getFunction().walk([&](TF::TPUExecuteOp op) { executes.push_back(op); });

  for (auto execute : executes) {
    OpBuilder builder(&getContext());
    const bool parent_is_replicate =
        llvm::isa<tf_device::ReplicateOp>(execute->getParentOp());
    // If this is inside a tf_device::ReplicateOp, the variables are guaranteed
    // to be on the same device as the TPUExecute op. Skip device checking in
    // that case.
    MergeForOneTPUExecute(execute, !parent_is_replicate, &builder);
  }
}

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> CreateTPUMergeVariablesWithExecutePass() {
  return std::make_unique<TPUMergeVariablesWithExecutePass>();
}

static PassRegistration<TPUMergeVariablesWithExecutePass> pass(
    "tf-tpu-merge-variables-with-execute",
    "Merges device variable reads/updates into tpu execute nodes");

}  // namespace TFTPU
}  // namespace mlir
