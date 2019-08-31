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

// This transformation pass takes ops with the same `_tpu_replicate` attribute
// in a block and clusters them together under a `tf_device::LaunchOp`.
// Associated TPUReplicateMetadata ops are removed and its attributes are copied
// over to the associated `tf_device::LaunchOp`. This pass also assumes ops of
// the same cluster do not have ops outside of the cluster that are both
// operands and results of the cluster. Note, this currently side effecting ops
// yet.

#include <memory>
#include <tuple>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Identifier.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kTPUReplicateAttr[] = "_tpu_replicate";
constexpr char kDeviceAttr[] = "device";
constexpr char kNameAttr[] = "name";

constexpr char kBadTPUReplicateAttrMsg[] =
    "requires '_tpu_replicate' string attribute";

// Mapping for `_tpu_replicate` attribute to TPUReplicateMetadata attributes.
using MetadataMap = llvm::SmallDenseMap<llvm::StringRef, NamedAttributeList, 8>;

// Mapping for `_tpu_replicate` attribute to ops of a cluster.
using ClusterMap = llvm::SmallDenseMap<llvm::StringRef,
                                       llvm::SmallSetVector<Operation*, 8>, 8>;

struct TPUClusterFormation : public FunctionPass<TPUClusterFormation> {
  void runOnFunction() override;
};

// Creates a mapping from the TPUReplicateMetadata ops `_tpu_replicate`
// attribute to its attributes and removes the ops. If multiple
// TPUReplicateMetadata ops have the same `_tpu_replicate` attribute, an error
// will be returned.
LogicalResult CollectMetadata(Operation* op, MetadataMap* metadata_map) {
  auto result =
      op->walk([&](TF::TPUReplicateMetadataOp metadata_op) -> WalkResult {
        NamedAttributeList attrs = metadata_op.getAttrs();

        // Missing or bad `_tpu_replicate` attribute.
        auto tpu_replicate_attr = attrs.get(kTPUReplicateAttr);
        if (!tpu_replicate_attr)
          return metadata_op.emitError() << kBadTPUReplicateAttrMsg;

        auto tpu_replicate_attr_str = tpu_replicate_attr.dyn_cast<StringAttr>();
        if (!tpu_replicate_attr_str ||
            tpu_replicate_attr_str.getValue().empty())
          return metadata_op.emitError() << kBadTPUReplicateAttrMsg;

        // Remove `name` attribute.
        attrs.remove(Identifier::get(kNameAttr, metadata_op.getContext()));

        auto it = metadata_map->try_emplace(tpu_replicate_attr_str.getValue(),
                                            std::move(attrs));

        // There are multiple TPUReplicateMetadata ops with the same
        // `_tpu_replicate` attribute.
        if (!it.second) {
          return metadata_op.emitError()
                 << "multiple TPUReplicateMetadata ops with the same '"
                 << kTPUReplicateAttr << "' attribute '"
                 << tpu_replicate_attr_str.getValue() << "' found";
        }

        metadata_op.erase();
        return WalkResult::advance();
      });

  // Return failure if the walk was interrupted.
  return failure(result.wasInterrupted());
}

// Collects and clusters ops with the same `_tpu_replicate` attribute. This will
// return an error if a `_tpu_replicate` attribute of an op is empty.
LogicalResult CollectAndGroupClusterOps(Block* block, ClusterMap* clusters) {
  for (Operation& op : *block) {
    if (auto attr = op.getAttrOfType<StringAttr>(kTPUReplicateAttr)) {
      if (attr.getValue().empty())
        return op.emitError()
               << "attribute '" << kTPUReplicateAttr << "' is empty";

      auto it = clusters->try_emplace(attr.getValue());
      it.first->getSecond().insert(&op);
    }
  }

  return success();
}

// Checks if an op should be moved after a cluster. There may be users of a
// cluster interleaved among the cluster ops.
bool ShouldMoveOpAfterCluster(
    Block* block, Operation* op,
    const llvm::SmallSetVector<Operation*, 8>& cluster_ops,
    const llvm::SmallSetVector<Operation*, 8>& preceding_users) {
  auto result = op->walk([&](Operation* op) {
    for (Value* operand : op->getOperands()) {
      Operation* def = operand->getDefiningOp();
      // Operands may not have a defining op (BlockArgument) or is from a
      // different block.
      if (!def || def->getBlock() != block) continue;

      if (cluster_ops.count(def) != 0 || preceding_users.count(def) != 0) {
        // Op is a user of a cluster or another op that is a user of the
        // cluster (transitively), but is before the cluster.
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  return result.wasInterrupted();
}

// Collects ops that are before ops in the cluster but are users of other ops
// in the cluster. This may happen because users of individual ops in the
// cluster may be interleaved with other ops in the cluster.
llvm::SmallSetVector<Operation*, 8> CollectClusterPrecedingUsers(
    Block* block, const llvm::SmallSetVector<Operation*, 8>& cluster_ops) {
  llvm::SmallSetVector<Operation*, 8> preceding_users;

  for (Operation& op : llvm::make_range(Block::iterator(cluster_ops.front()),
                                        Block::iterator(cluster_ops.back()))) {
    if (cluster_ops.count(&op) == 0 &&
        ShouldMoveOpAfterCluster(block, &op, cluster_ops, preceding_users))
      preceding_users.insert(&op);
  }

  return preceding_users;
}

// Collects results and associated types of the cluster that are used outside of
// the cluster. These results and types are used to create the clusters
// `tf_device::LaunchOp` and associated terminator. Results that have no uses
// outside of the cluster (i.e. results of ops in the cluster are only consumed
// by other ops in the cluster) are pruned.
llvm::SmallVector<Value*, 8> CollectClusterResults(
    Block* block, const llvm::SmallSetVector<Operation*, 8>& cluster_ops) {
  llvm::SmallVector<Value*, 8> results;

  for (Operation* op : cluster_ops) {
    for (Value* result : op->getResults()) {
      for (Operation* user : result->getUsers()) {
        // Check if user is not an op in the cluster.
        if (cluster_ops.count(block->findAncestorInstInBlock(*user)) == 0) {
          results.push_back(result);
          break;
        }
      }
    }
  }

  return results;
}

// Creates a `tf_device::LaunchOp` to wrap cluster ops.
tf_device::LaunchOp CreateLaunchOpForCluster(Operation* last_cluster_op,
                                             llvm::ArrayRef<Value*> results) {
  // `tf_device::LaunchOp` will be placed at where the last op of the cluster
  // is.
  OpBuilder builder(last_cluster_op);

  llvm::SmallVector<Type, 8> result_types;
  for (Value* result : results) result_types.push_back(result->getType());

  // An empty string placeholder is used for the device as that will be later
  // populated with the device of the associated TPUReplicateMetadata op.
  auto launch_op = builder.create<tf_device::LaunchOp>(
      last_cluster_op->getLoc(), builder.getStringAttr(""), result_types);

  launch_op.body().push_back(new Block);

  // Add terminator.
  builder.setInsertionPointToEnd(&launch_op.GetBody());
  builder.create<tf_device::ReturnOp>(last_cluster_op->getLoc(), results);

  return launch_op;
}

// Moves cluster ops to associated `tf_device.LaunchOp` body.
void MoveClusterOpsToLaunchOp(
    tf_device::LaunchOp launch_op,
    const llvm::SmallSetVector<Operation*, 8>& cluster_ops) {
  MLIRContext* context = launch_op.getContext();
  Operation* terminator = &launch_op.GetBody().back();

  for (Operation* cluster_op : cluster_ops) {
    // Remove `_tpu_replicate` and `device` attribute from ops in the cluster
    // as that information will be present in the `tf_device.LaunchOp`.
    cluster_op->removeAttr(Identifier::get(kTPUReplicateAttr, context));
    cluster_op->removeAttr(Identifier::get(kDeviceAttr, context));
    cluster_op->moveBefore(terminator);
  }
}

// Replaces uses of cluster ops results outside of cluster with the associated
// `tf_device::LaunchOp` results.
void UpdateLaunchOpResultExternalUses(tf_device::LaunchOp launch_op,
                                      llvm::ArrayRef<Value*> results) {
  Block& launch_op_block = launch_op.GetBody();
  for (auto ret_vals : llvm::zip(results, launch_op.getResults())) {
    Value* old_ret = std::get<0>(ret_vals);
    Value* new_ret = std::get<1>(ret_vals);
    for (auto& use : llvm::make_early_inc_range(old_ret->getUses())) {
      if (!launch_op_block.findAncestorInstInBlock(*use.getOwner())) {
        use.getOwner()->setOperand(use.getOperandNumber(), new_ret);
      }
    }
  }
}

// Moves users of cluster that are before the cluster to after the cluster.
void MovePrecedingClusterUsers(tf_device::LaunchOp launch_op,
                               llvm::ArrayRef<Operation*> preceding_users) {
  Operation* op_after_launch_op = launch_op.getOperation()->getNextNode();
  for (Operation* user : preceding_users) user->moveBefore(op_after_launch_op);
}

// Forms clusters with ops of the same `_tpu_replicate` attribute under a block.
//
// For a given block, clusters are formed via grouping ops by `_tpu_replicate`
// attributes.
// For every cluster formed:
//   1. Find associated TPUReplicateMetadata attributes with the same
//      `_tpu_replicate` attribute.
//   2. Find users not in cluster that are interleaved between cluster ops.
//   3. Find external uses of cluster ops.
//   4. Create `tf_device::LaunchOp` with results consisting of the external
//      uses of cluster ops determined at 3.
//   5. Move cluster ops to `tf_device::LaunchOp` body.
//   6. Replace external uses of cluster ops uses with `tf_device::LaunchOp`
//      results.
//   7. Move users from 2 to after the `tf_device::LaunchOp`.
//   8. Copy over TPUReplicateMetadata attributes to `tf_device::LaunchOp`.
LogicalResult FormClustersInBlock(Block* block,
                                  const MetadataMap& metadata_map) {
  ClusterMap clusters;
  LogicalResult result = CollectAndGroupClusterOps(block, &clusters);
  if (failed(result)) return result;

  for (const auto& cluster : clusters) {
    const auto& cluster_ops = cluster.getSecond();

    auto cluster_metadata = metadata_map.find(cluster.getFirst());

    // No TPUReplicateMetadata for a `_tpu_replicate` attribute.
    if (cluster_metadata == metadata_map.end())
      return cluster_ops.front()->emitError()
             << "TPUReplicateMetadata for associated '" << kTPUReplicateAttr
             << "' attribute '" << cluster.getFirst() << "' is missing";

    llvm::SmallSetVector<Operation*, 8> preceding_users =
        CollectClusterPrecedingUsers(block, cluster_ops);

    llvm::SmallVector<Value*, 8> results =
        CollectClusterResults(block, cluster_ops);

    tf_device::LaunchOp launch_op =
        CreateLaunchOpForCluster(cluster_ops.back(), results);

    MoveClusterOpsToLaunchOp(launch_op, cluster_ops);

    UpdateLaunchOpResultExternalUses(launch_op, results);

    MovePrecedingClusterUsers(launch_op, preceding_users.getArrayRef());

    // Copy TPUReplicateMetadata attributes to launch.
    launch_op.setAttrs(cluster_metadata->second);
  }

  return success();
}

void TPUClusterFormation::runOnFunction() {
  MetadataMap metadata_map;
  if (failed(CollectMetadata(getFunction(), &metadata_map)))
    return signalPassFailure();

  for (Block& block : getFunction())
    if (failed(FormClustersInBlock(&block, metadata_map)))
      return signalPassFailure();

  getFunction().walk([&](tf_executor::IslandOp island) {
    FormClustersInBlock(&island.GetBody(), metadata_map);
  });
}
}  // anonymous namespace

std::unique_ptr<FunctionPassBase> CreateTPUClusterFormationPass() {
  return std::make_unique<TPUClusterFormation>();
}

static PassRegistration<TPUClusterFormation> pass(
    "tf-tpu-cluster-formation",
    "Form clusters from operations assigned to the same TPU cluster");

}  // namespace TFTPU
}  // namespace mlir
