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

// This transformation forms clusters from instructions in same island and
// assigned to save devices. Clusters are represented as regions.
// Note that side-effecting ops are not correctly handled yet.

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Block.h"  // TF:local_config_mlir
#include "mlir/IR/BlockAndValueMapping.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/core/platform/logging.h"

namespace mlir {
namespace TFDevice {

namespace {

struct ClusterFormationPass : public FunctionPass<ClusterFormationPass> {
  void runOnFunction() override;
};

// Cluster structure captures all the operations that are assigned to same
// device and can form a legal strict cluster.
// Ops must follow same ordering in their parent block. We rely on this
// assumption to perform analysis.
struct Cluster {
  llvm::SmallVector<Operation*, 4> ops;
  StringRef device;
};

StringRef GetDevice(Operation* op) {
  auto device_attr = op->getAttrOfType<StringAttr>("device");
  return device_attr ? device_attr.getValue() : "";
}

// An op can be merged into cluster if all of its operands are one of the
// following:
//  1) A block argument
//  2) A value produced by other islands
//  1) Defined before the cluster
//  2) Defined by an operation in the cluster
// TODO(ycao): This is not optimal as it doesn't consider the situation of
// defining_op's operands all meet the requirements above. In that case, the
// defining_op can be moved and to_merge op would be legal to absorb.
// TODO(ycao): Take op side-effects into consideration since they can not be
// re-ordered but forming clusters of non-continuous ops is effectively
// re-ordering them..
bool CanMergeIntoCluster(const Cluster& c, Operation* to_merge) {
  return llvm::all_of(to_merge->getOperands(), [&](Value* operand) {
    // Block arguments.
    if (isa<BlockArgument>(operand)) return true;

    Operation* defining_op = operand->getDefiningOp();

    // Operand produced by other islands.
    if (defining_op->getBlock() != c.ops.front()->getBlock()) return true;

    // Defining op is before the cluster.
    if (defining_op->isBeforeInBlock(c.ops.front())) return true;

    // Defining op is between first and last operation in cluster. Note that
    // cluster may contain operations that are non-continuous in their original
    // block, thus we also need to check defining_op is also assigned to
    // cluster's device to be sure. This is a faster check than linearly
    // searching through all ops in cluster.
    if (defining_op->isBeforeInBlock(c.ops.back()->getNextNode()) &&
        GetDevice(defining_op) == c.device)
      return true;

    // Other cases, operand is generated after or outside the cluster, this
    // means it is illegal to merge operation.
    return false;
  });
}

void ReplaceLiveOutExternalUses(llvm::ArrayRef<Value*> live_outs,
                                Operation* launch_op) {
  Region* launch_op_region = &launch_op->getRegion(0);
  for (const auto& p : llvm::zip(live_outs, launch_op->getResults())) {
    Value* from = std::get<0>(p);
    for (auto& use : from->getUses()) {
      if (launch_op_region->isAncestor(use.getOwner()->getParentRegion()))
        continue;
      use.set(std::get<1>(p));
    }
  }
}

// Get all escaped live-out values of a region.
void GetLiveOuts(Region* region, llvm::SmallVectorImpl<Value*>* live_outs) {
  live_outs->clear();

  for (Operation& op : region->front()) {
    for (Value* v : op.getResults()) {
      // A value is live-out if any of its users are not inside value producer's
      // region.
      bool is_live_out = llvm::any_of(v->getUsers(), [&](Operation* user) {
        return !region->isAncestor(user->getParentRegion());
      });

      if (is_live_out) live_outs->emplace_back(v);
    }
  }
}

// TODO(b/138909768): Define `tf_device.return` op and use its build method
// instead.
void BuildReturn(llvm::ArrayRef<Value*> live_outs, OpBuilder* builder) {
  OperationState return_op_state(builder->getUnknownLoc(), "tf_device.return");
  return_op_state.addOperands(live_outs);
  builder->createOperation(return_op_state);
}

// Build a `tf_device.launch` op with a region that contains all the operations
// in given cluster. Then all ops in cluster are replaced by `tf_device.launch`.
// TODO(b/138909768): Define `tf_device.launch` op and use its build method
// instead.
void BuildLaunchForCluster(const Cluster& c, OpBuilder* builder) {
  // Set insertion point to right after all operations in cluster.
  builder->setInsertionPoint(c.ops.back()->getNextNode());

  // Create an empty `tf_device.launch` op with a device attribute matching
  // given cluster.
  OperationState launch_op_state(builder->getUnknownLoc(), "tf_device.launch");
  launch_op_state.addAttribute("device", builder->getStringAttr(c.device));
  Region* region = launch_op_state.addRegion();
  region->push_back(new Block);

  // Move all operations in cluster to newly created region, stripping their
  // "device" attribute since launch op already carries device information.
  Block* block = &region->front();
  for (Operation* op : c.ops) {
    op->moveBefore(block, block->end());
    op->removeAttr(builder->getIdentifier("device"));
  }

  // Get all escaped live-out values of region, they are used later to determine
  // return values and types of launch op.
  llvm::SmallVector<Value*, 4> live_outs;
  GetLiveOuts(region, &live_outs);

  // Build a `tf_device.return` op at end of region, with all live-out values
  // as operand.
  OpBuilder return_builder(builder->getContext());
  return_builder.setInsertionPointToEnd(block);
  BuildReturn(live_outs, &return_builder);

  for (Value* v : live_outs) launch_op_state.types.emplace_back(v->getType());

  Operation* launch_op = builder->createOperation(launch_op_state);

  // Replace any external uses of live-out values with return values of launch
  // op. So live-out values no longer escape the region.
  ReplaceLiveOutExternalUses(live_outs, launch_op);
}

void ClusterFormationPass::runOnFunction() {
  OpBuilder builder(getFunction().getContext());
  getFunction().walk<tf_executor::IslandOp>([&](tf_executor::IslandOp island) {
    // Iteratively find clusters of different devices within an island.
    // Whenever we see an operation that is assigned to an accelerator device
    // (ie. device != ""), we try to merge it into the last cluster of same
    // device. If that is infeasible (say because of violating def-before-use),
    // create a new cluster with that operation and move on.
    llvm::MapVector<StringRef, Cluster> nearest_clusters;
    for (Operation& op : llvm::make_early_inc_range(island.GetBody())) {
      auto device = GetDevice(&op);
      if (device == "") continue;

      // If no cluster of same device has been formed yet, create a new cluster
      // with op alone.
      auto it = nearest_clusters.find(device);
      if (it == nearest_clusters.end()) {
        nearest_clusters[device] = Cluster{{&op}, device};
        continue;
      }

      // Check if it is legal to merge op into nearest cluster of same device.
      // If positive, update cluster and move on to next operation.
      Cluster& nearest_cluster = it->second;
      if (CanMergeIntoCluster(nearest_cluster, &op)) {
        nearest_cluster.ops.emplace_back(&op);
        continue;
      }

      // If nearest cluster of same device can not absorb `op`, then that
      // cluster needs to be finalized by building a `tf_device.launch` op with
      // a region that contains all operations in clusters.
      BuildLaunchForCluster(nearest_cluster, &builder);

      // Create a new cluster to hold op alone and update nearest_clusters.
      nearest_clusters[device] = Cluster{{&op}, device};
    }

    // At the end, there might be left-over found clusters that need to be
    // built.
    for (auto& device_cluster : nearest_clusters)
      BuildLaunchForCluster(device_cluster.second, &builder);
  });
}

}  // namespace

FunctionPassBase* CreateClusterFormationPass() {
  return new ClusterFormationPass();
}

static PassRegistration<ClusterFormationPass> pass(
    "tf-device-cluster-formation",
    "Form clusters from instructions assigned to same device");

}  // namespace TFDevice
}  // namespace mlir
