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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

#define DEBUG_TYPE "tf-hoist-replicate-invariant-resource-writes"

namespace mlir {
namespace TF {

namespace {

struct HoistReplicateInvariantResourceWritesPass
    : public TF::HoistReplicateInvariantResourceWritesPassBase<
          HoistReplicateInvariantResourceWritesPass> {
  void runOnOperation() override;
};

// TODO(prakalps): This is a common utility and other passes use something
// similar. Move to common utils.
bool IsResourceType(Type type) {
  return type.isa<TF::ResourceType>() ||
         (type.isa<TensorType>() &&
          type.cast<TensorType>().getElementType().isa<TF::ResourceType>());
}

SmallVector<Value> GetAccessedResources(Operation& op) {
  SmallVector<Value, 4> accessed_resources;
  for (auto operand : op.getOperands()) {
    if (!IsResourceType(operand.getType())) continue;
    accessed_resources.push_back(operand);
  }
  return std::move(accessed_resources);
}

// Lifts the tail writes outside of tf_device.replicate. The written value is
// added to the values returned by tf_device.replicate op. Modify the assign
// variable ops to use the value from first replica.
void MoveTailWritesAfterReplicate(
    tf_device::ReplicateOp replicate_op,
    llvm::ArrayRef<TF::AssignVariableOp> tail_assign_variable_ops) {
  const auto num_replicas = replicate_op.n();
  auto return_op = llvm::dyn_cast<tf_device::ReturnOp>(
      replicate_op.getRegion().front().getTerminator());

  // Get the new result types.
  // TODO(prakalps): Do not add a value to returned values if it is already
  // returned.
  auto new_result_types = llvm::to_vector<4>(replicate_op->getResultTypes());
  for (auto assign : tail_assign_variable_ops) {
    return_op->insertOperands(return_op->getNumOperands(), assign.value());
    new_result_types.insert(new_result_types.end(), num_replicas,
                            assign.value().getType());
  }

  OpBuilder builder(replicate_op);
  // Clone this old replicate op but with new result types.
  auto new_replicate_op = builder.create<tf_device::ReplicateOp>(
      replicate_op->getLoc(), new_result_types, replicate_op->getOperands(),
      replicate_op->getAttrs());

  // Move region to the new op.
  new_replicate_op.getRegion().takeBody(replicate_op.getRegion());

  // Replace all old uses with new op results.
  int old_num_results = replicate_op->getNumResults();
  replicate_op->replaceAllUsesWith(
      new_replicate_op->getResults().take_front(old_num_results));

  // Move assign ops after replicate and use the output of first replica.
  for (auto indexed_assign : llvm::enumerate(tail_assign_variable_ops)) {
    auto assign_op = indexed_assign.value();
    auto index = indexed_assign.index();
    assign_op->moveAfter(new_replicate_op);
    assign_op->setOperand(
        1, new_replicate_op->getResult(old_num_results + num_replicas * index));
  }
  replicate_op->erase();
}

// Looks for AssignVariable ops from the end of the tf_device.replicate op. It
// returns all the last writes to replicate invariant resource variables
// (resource handles defined outside the tf_device.replicate op).
SmallVector<TF::AssignVariableOp> GetTailWritesToReplicateInvariantResourceVars(
    tf_device::ReplicateOp replicate_op) {
  SmallVector<TF::AssignVariableOp, 16> tail_assign_variable_ops;
  llvm::SmallDenseSet<Value, 16> visited_resources;
  for (auto& op :
       llvm::reverse(replicate_op.getRegion().front().getOperations())) {
    SmallVector<Value> op_accessed_resources = GetAccessedResources(op);
    if (op_accessed_resources.empty()) continue;

    if (auto assign = llvm::dyn_cast<TF::AssignVariableOp>(op)) {
      Value resource_var = assign.resource();
      if (visited_resources.contains(resource_var) ||
          !resource_var.getParentRegion()->isProperAncestor(
              &replicate_op.getRegion()))
        continue;
      tail_assign_variable_ops.push_back(assign);
    }

    for (Value resource : op_accessed_resources)
      visited_resources.insert(resource);
  }
  return std::move(tail_assign_variable_ops);
}

void HoistReplicateInvariantResourceWritesPass::runOnOperation() {
  SmallVector<tf_device::ReplicateOp, 2> replicate_ops;
  getOperation().walk([&](tf_device::ReplicateOp replicate_op) {
    replicate_ops.push_back(replicate_op);
  });
  for (auto replicate_op : replicate_ops) {
    SmallVector<TF::AssignVariableOp> tail_writes =
        GetTailWritesToReplicateInvariantResourceVars(replicate_op);

    if (tail_writes.empty()) continue;
    MoveTailWritesAfterReplicate(replicate_op, tail_writes);
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateHoistReplicateInvariantResourceWritesPass() {
  return std::make_unique<HoistReplicateInvariantResourceWritesPass>();
}

}  // namespace TF
}  // namespace mlir
