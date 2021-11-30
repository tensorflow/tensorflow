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

#include <algorithm>
#include <iterator>
#include <memory>
#include <tuple>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Identifier.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_alias_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kTPUReplicateAttr[] = "_tpu_replicate";
constexpr char kDeviceAttr[] = "device";
constexpr char kNameAttr[] = "name";
constexpr char kNumCoresPerReplicaAttr[] = "num_cores_per_replica";
constexpr char kNumReplicasAttr[] = "num_replicas";
constexpr char kReplicatedInputIndicesAttr[] = "_replicated_input_indices";
constexpr char kMirroredVariableIndicesAttr[] = "_mirrored_variable_indices";

constexpr char kBadTPUReplicateAttrMsg[] =
    "requires '_tpu_replicate' string attribute";

// Mapping for `_tpu_replicate` attribute to TPUReplicateMetadata attributes.
using MetadataMap = llvm::SmallDenseMap<llvm::StringRef, NamedAttrList, 8>;

// A set of operations in a cluster.
using ClusterOps = llvm::SmallSetVector<Operation*, 8>;

// Mapping for `_tpu_replicate` attribute to ops of a cluster.
using ClusterMap = llvm::SmallDenseMap<llvm::StringRef, ClusterOps, 8>;

struct TPUClusterFormationPass
    : public TF::TPUClusterFormationPassBase<TPUClusterFormationPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<tf_device::TensorFlowDeviceDialect>();
  }

  void runOnOperation() override;
};

// Creates a mapping from the TPUReplicateMetadata ops `_tpu_replicate`
// attribute to its attributes and removes the ops. If multiple
// TPUReplicateMetadata ops have the same `_tpu_replicate` attribute, an error
// will be returned.
LogicalResult CollectMetadata(Block* block, MetadataMap* metadata_map) {
  // Just look at top-level operations in the block (not nested ones)
  for (Operation& op : llvm::make_early_inc_range(*block)) {
    auto metadata_op = dyn_cast<TF::TPUReplicateMetadataOp>(op);
    if (!metadata_op) continue;

    NamedAttrList attrs(metadata_op->getAttrDictionary());

    // Missing or bad `_tpu_replicate` attribute.
    auto tpu_replicate_attr = attrs.get(kTPUReplicateAttr);
    if (!tpu_replicate_attr)
      return metadata_op.emitError() << kBadTPUReplicateAttrMsg;

    auto tpu_replicate_attr_str = tpu_replicate_attr.dyn_cast<StringAttr>();
    if (!tpu_replicate_attr_str || tpu_replicate_attr_str.getValue().empty())
      return metadata_op.emitError() << kBadTPUReplicateAttrMsg;

    // Remove `name` attribute.
    attrs.erase(Identifier::get(kNameAttr, metadata_op.getContext()));

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
  }
  return success();
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

// Collects all resource ids from an op.
void CollectResourceIdsFromOp(
    Operation& op,
    const TF::ResourceAliasAnalysis::Info& resource_alias_analysis,
    llvm::SmallDenseSet<int64_t>& observed_resource_ids) {
  op.walk([&](Operation* inner_op) {
    for (Value operand : TF::filter_resources(inner_op->getOperands())) {
      if (resource_alias_analysis.IsUnknownResource(operand)) continue;
      const auto& ids = resource_alias_analysis.GetResourceUniqueIds(operand);
      observed_resource_ids.insert(ids.begin(), ids.end());
    }
    for (Value result : TF::filter_resources(inner_op->getResults())) {
      if (resource_alias_analysis.IsUnknownResource(result)) continue;
      const auto& ids = resource_alias_analysis.GetResourceUniqueIds(result);
      observed_resource_ids.insert(ids.begin(), ids.end());
    }
  });
}

// Checks if an op should be moved after a cluster. There may be users of a
// cluster interleaved among the cluster ops.
bool ShouldMoveOpAfterCluster(
    Block* block, Operation* op, const ClusterOps& cluster_ops,
    const llvm::SmallSetVector<Operation*, 8>& preceding_users,
    const TF::ResourceAliasAnalysis::Info& resource_alias_analysis,
    const llvm::SmallDenseSet<int64_t>& observed_resource_ids) {
  const bool is_replicate = llvm::isa<tf_device::ReplicateOp>(op);
  auto result = op->walk([&](Operation* inner_op) {
    for (Value operand : inner_op->getOperands()) {
      Operation* def = operand.getDefiningOp();
      // Operands may not have a defining op (BlockArgument) or is from a
      // different block.
      if (!def || def->getBlock() != block) continue;

      if (cluster_ops.count(def) != 0 || preceding_users.count(def) != 0) {
        // Op is a user of a cluster or another op that is a user of the
        // cluster (transitively), but is before the cluster.
        return WalkResult::interrupt();
      }
    }

    // Don't visit replicate op inner op operands as new resource
    // values/arguments may have been created but are not known in
    // `resource_alias_analysis`.
    if (is_replicate && inner_op != op) return WalkResult::advance();

    // Check for uses of any resource in or after cluster.
    for (Value operand : TF::filter_resources(inner_op->getOperands())) {
      if (resource_alias_analysis.IsUnknownResource(operand)) continue;
      auto ids = resource_alias_analysis.GetResourceUniqueIds(operand);
      for (const auto& id : ids)
        if (observed_resource_ids.contains(id)) return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return result.wasInterrupted();
}

// Collects ops that are before ops in the cluster but are users of other ops
// in the cluster. This may happen because users of individual ops in the
// cluster may be interleaved with other ops in the cluster. Resource IDs are
// also captured, to keep track of resource usage before, in, or after the
// cluster.
// TODO(b/175701589): Extend this to handle all side effecting ops while
// handling transitive data dependencies.
llvm::SmallSetVector<Operation*, 8> CollectClusterPrecedingUsers(
    Block* block, const ClusterOps& cluster_ops,
    const TF::ResourceAliasAnalysis::Info& resource_alias_analysis) {
  llvm::SmallSetVector<Operation*, 8> preceding_users;
  llvm::SmallDenseSet<int64_t> observed_resource_ids;

  auto front = Block::iterator(cluster_ops.front());
  auto back = Block::iterator(cluster_ops.back());
  for (Operation& op : llvm::make_range(front, back)) {
    if (cluster_ops.contains(&op)) {
      CollectResourceIdsFromOp(op, resource_alias_analysis,
                               observed_resource_ids);
    } else if (ShouldMoveOpAfterCluster(
                   block, &op, cluster_ops, preceding_users,
                   resource_alias_analysis, observed_resource_ids)) {
      preceding_users.insert(&op);
      CollectResourceIdsFromOp(op, resource_alias_analysis,
                               observed_resource_ids);
    }
  }

  return preceding_users;
}

// Collects results and associated types of the cluster that are used outside of
// the cluster. These results and types are used to create the clusters
// `tf_device.cluster` and associated terminator. Results that have no uses
// outside of the cluster (i.e. results of ops in the cluster are only consumed
// by other ops in the cluster) are pruned.
llvm::SmallVector<Value, 8> CollectClusterResults(
    Block* block, const ClusterOps& cluster_ops) {
  llvm::SmallVector<Value, 8> results;

  for (Operation* op : cluster_ops) {
    for (Value result : op->getResults()) {
      for (Operation* user : result.getUsers()) {
        // Check if user is not an op in the cluster.
        if (cluster_ops.count(block->findAncestorOpInBlock(*user)) == 0) {
          results.push_back(result);
          break;
        }
      }
    }
  }

  return results;
}

// Creates a `tf_device.cluster` to wrap cluster ops.
tf_device::ClusterOp CreateClusterOp(
    Block* block, const ClusterOps& cluster_ops, llvm::ArrayRef<Value> results,
    llvm::ArrayRef<Operation*> preceding_users) {
  // `tf_device.cluster` will be placed at where the last op of the cluster is.
  Operation* last_cluster_op = cluster_ops.back();
  OpBuilder builder(last_cluster_op);

  llvm::SmallVector<Type, 8> result_types;
  for (Value result : results) result_types.push_back(result.getType());
  auto cluster = builder.create<tf_device::ClusterOp>(last_cluster_op->getLoc(),
                                                      result_types);

  Block* body = new Block;
  cluster.body().push_back(body);

  // Move cluster ops to the cluster body. Also remove `_tpu_replicate` and
  // `device` attribute from ops in the cluster when that information is
  // redundant will the `tf_device.cluster`. Do this for all ops including
  // nested ops.
  for (Operation* cluster_op : cluster_ops) {
    cluster_op->moveBefore(body, body->end());
    cluster_op->walk([&](Operation* inner_op) {
      inner_op->removeAttr(kTPUReplicateAttr);

      if (auto attr = inner_op->getAttrOfType<StringAttr>(kDeviceAttr)) {
        // Preserve device attribute if the op is placed on a replicated core
        // device. Device attribute is used to infer the appropriate sharding
        // within TPUs for this op.
        // TODO(b/183598857): Use explicit sharding ops from the front-end.
        // For example, dequeue ops generated by
        // tensorflow/python/tpu/tpu_feed.py
        if (!tensorflow::IsTPUReplicatedCore(attr.getValue())) {
          inner_op->removeAttr(kDeviceAttr);
        }
      }
    });
  }

  // Add terminator.
  builder.setInsertionPointToEnd(body);
  builder.create<tf_device::ReturnOp>(last_cluster_op->getLoc(), results);

  // Replaces uses of cluster ops results outside of cluster with the associated
  // `tf_device.cluster` results.
  for (auto ret_vals : llvm::zip(results, cluster.getResults())) {
    Value old_ret = std::get<0>(ret_vals);
    Value new_ret = std::get<1>(ret_vals);
    for (auto& use : llvm::make_early_inc_range(old_ret.getUses())) {
      Operation* user = use.getOwner();
      if (!body->findAncestorOpInBlock(*user)) use.set(new_ret);
    }
  }

  // Move users of cluster that are before the cluster to after the cluster.
  Operation* op_after_cluster = cluster.getOperation()->getNextNode();
  for (Operation* user : preceding_users) user->moveBefore(op_after_cluster);
  return cluster;
}

// Sorts `tf.TPUReplicatedInput` ops by `index` attribute. Ops with an `index`
// of -1 are always after ops with a non negative `index`, and an arbitrary
// ordering is used as there are no dependencies on their relative ordering. If
// there are multiple `tf.TPUReplicatedInput` ops with the same non negative
// index or if indices are less than -1, an error will be returned.
LogicalResult SortTPUReplicatedInputsByIndex(
    llvm::ArrayRef<Operation*> inputs,
    llvm::SmallVectorImpl<Operation*>* sorted_inputs) {
  llvm::SmallDenseSet<int64_t, 8> unique_indices;
  for (Operation* input : inputs) {
    int64_t index = llvm::cast<TF::TPUReplicatedInputOp>(input).index();
    if (index < -1)
      return input->emitOpError()
             << "requires index to be at least -1, but got " << index;
    if (index == -1) continue;
    if (!unique_indices.insert(index).second)
      return input->emitOpError()
             << "requires indices to be unique, but found multiple '"
             << input->getName() << "' ops with index " << index;
  }

  // Sort all TPUReplicatedInputs by `index` attribute to have
  // TPUReplicatedInputs with indices be added to the `tf_device.replicate` op
  // deterministically. If `index` attribute is -1, instead move them to the
  // end.
  sorted_inputs->assign(inputs.begin(), inputs.end());
  std::stable_sort(
      sorted_inputs->begin(), sorted_inputs->end(),
      [](Operation* l, Operation* r) {
        int64_t l_index = llvm::cast<TF::TPUReplicatedInputOp>(l).index();
        int64_t r_index = llvm::cast<TF::TPUReplicatedInputOp>(r).index();
        if (l_index == -1 && r_index != -1) return false;
        if (r_index == -1 && l_index != -1) return true;
        return l_index < r_index;
      });

  return success();
}

// Creates a `tf_device.replicate` to represent replication for the cluster, if
// necessary.
LogicalResult ReplicateCluster(tf_device::ClusterOp cluster, int num_replicas,
                               int num_cores_per_replica) {
  // No need to replicate.
  if (num_replicas == 1) return success();

  if (num_replicas < 1)
    return cluster.emitError() << "requires '" << kNumReplicasAttr
                               << "' int attribute to be at least 1";

  LogicalResult status = success();
  // Collect all used TPUReplicatedInput ops and sort by `index`.
  llvm::SmallSetVector<Operation*, 8> unique_replicated_input_ops;
  mlir::visitUsedValuesDefinedAbove(
      cluster.body(), cluster.body(), [&](mlir::OpOperand* operand) {
        Operation* def = operand->get().getDefiningOp();
        if (llvm::isa_and_nonnull<TF::TPUReplicatedInputOp>(def))
          unique_replicated_input_ops.insert(def);
        // When model parallelism is used in conjunction with data parallelism
        // for resource inputs, we need to collect the per replica resource
        // inputs from input to `tf.TPUPartitionedInput` ops.
        if (auto pi = llvm::dyn_cast_or_null<TF::TPUPartitionedInputOp>(def)) {
          if (pi->getNumOperands() != num_cores_per_replica)
            status = pi.emitOpError()
                     << "requires " << num_cores_per_replica
                     << " operands but found " << pi->getNumOperands();
          for (auto operand : pi.inputs()) {
            if (llvm::isa_and_nonnull<TF::TPUReplicatedInputOp>(
                    operand.getDefiningOp()))
              unique_replicated_input_ops.insert(operand.getDefiningOp());
          }
        }
      });

  if (failed(status)) return failure();
  llvm::SmallVector<Operation*, 8> replicated_input_ops;
  if (failed(SortTPUReplicatedInputsByIndex(
          unique_replicated_input_ops.getArrayRef(), &replicated_input_ops)))
    return failure();

  // Index attribute value stored on TPUReplicatedInput op. These will be used
  // later for dynamic padder.
  llvm::SmallVector<int64_t, 8> replicated_input_indices;
  llvm::SmallVector<int64_t, 8> packed_input_indices;
  bool has_replicated_input_index = false;

  // Indices of the replicate op's arguments that are mirrored variables.
  llvm::SmallVector<int64_t, 8> mirrored_variable_indices;

  // Check if number of operands of each used TPUReplicatedInput op matches
  // `num_replicas` or 1. Collect all their operands and associated type for
  // creating the replicate op.
  llvm::SmallVector<std::pair<ValueRange, Type>, 8> replicated_inputs;
  llvm::SmallVector<Value, 8> packed_inputs;
  llvm::SmallVector<Operation*, 8> replicated_ops;
  llvm::SmallVector<Operation*, 8> packed_ops;
  for (auto& pos_and_input : llvm::enumerate(replicated_input_ops)) {
    auto input = pos_and_input.value();
    bool is_packed = llvm::cast<TF::TPUReplicatedInputOp>(input).is_packed();
    const int num_operands = input->getNumOperands();
    int num_inputs = is_packed ? 1 : num_replicas;
    if (num_operands != num_inputs)
      return input->emitOpError() << "requires " << num_inputs << " operands";

    auto tpu_replicated_input = llvm::cast<TF::TPUReplicatedInputOp>(input);
    int64_t tpu_replicated_input_index = tpu_replicated_input.index();
    if (is_packed) {
      packed_inputs.push_back(input->getOperand(0));
      packed_input_indices.push_back(tpu_replicated_input_index);
      packed_ops.push_back(input);
    } else {
      replicated_inputs.push_back(
          {input->getOperands(), input->getOperand(0).getType()});
      replicated_input_indices.push_back(tpu_replicated_input_index);
      replicated_ops.push_back(input);
    }
    if (tpu_replicated_input_index != -1) has_replicated_input_index = true;

    if (tpu_replicated_input.is_mirrored_variable())
      mirrored_variable_indices.push_back(pos_and_input.index());
  }

  replicated_input_indices.append(packed_input_indices.begin(),
                                  packed_input_indices.end());

  // Create replicate op.
  OpBuilder builder(cluster);
  auto replicate_op = builder.create<tf_device::ReplicateOp>(
      cluster.getLoc(), num_replicas,
      llvm::SmallDenseMap<llvm::StringRef, llvm::SmallVector<StringRef, 4>>(),
      replicated_inputs, packed_inputs, cluster.getResultTypes());
  if (has_replicated_input_index)
    replicate_op->setAttr(kReplicatedInputIndicesAttr,
                          builder.getI64ArrayAttr(replicated_input_indices));

  if (!mirrored_variable_indices.empty())
    replicate_op->setAttr(kMirroredVariableIndicesAttr,
                          builder.getI64ArrayAttr(mirrored_variable_indices));

  // Replace replicated cluster results with replicate op results.
  for (auto result_and_idx : llvm::enumerate(cluster.getResults())) {
    Value result = result_and_idx.value();
    int idx = result_and_idx.index();
    auto replicate_outputs = llvm::make_range(
        std::next(replicate_op.result_begin(), idx * num_replicas),
        std::next(replicate_op.result_begin(), (idx + 1) * num_replicas));

    for (auto& use : llvm::make_early_inc_range(result.getUses())) {
      Operation* def = use.getOwner();
      if (!llvm::isa<TF::TPUReplicatedOutputOp>(def)) {
        // If user is not a `tf.TPUReplicatedOutput`, simply forward the first
        // replica output. Certain Graphs under V1 create `tf.Identity` users of
        // replicated ops to pin the TPU computation for execution.
        use.set(*replicate_outputs.begin());
        continue;
      }

      const int def_num_results = def->getNumResults();
      if (def_num_results != num_replicas)
        return def->emitOpError() << "requires " << num_replicas << " results";

      def->replaceAllUsesWith(replicate_outputs);
    }
  }

  // Collect all `tf.TPUPartitionedInput` ops to be moved inside the
  // `tf_device.replicate` later.
  llvm::SmallSet<Operation*, 4> partitioned_inputs;
  // Update replicated inputs with replicate op block arguments.
  auto ordered_tpu_replicate_inputs =
      llvm::concat<Operation*>(replicated_ops, packed_ops);
  for (auto input_and_block_arg :
       llvm::zip(ordered_tpu_replicate_inputs,
                 replicate_op.GetBody().getArguments())) {
    Operation* input = std::get<0>(input_and_block_arg);
    Value block_arg = std::get<1>(input_and_block_arg);
    mlir::replaceAllUsesInRegionWith(input->getResult(0), block_arg,
                                     cluster.body());
    // Update replicated input use in tf.TPUPartitionedInput op.
    for (auto& use : input->getUses()) {
      auto pi = llvm::dyn_cast<TF::TPUPartitionedInputOp>(use.getOwner());
      if (pi) {
        pi.setOperand(use.getOperandNumber(), block_arg);
        partitioned_inputs.insert(pi.getOperation());
      }
    }
  }

  // Create terminator for replicate op and move `tf_device.cluster` and
  // `tf.TPUPartitionedInput`(s) into replicate body.
  builder.setInsertionPointToEnd(&replicate_op.GetBody());
  auto return_op = builder.create<tf_device::ReturnOp>(replicate_op.getLoc(),
                                                       cluster.getResults());
  for (auto pi : partitioned_inputs) pi->moveBefore(return_op);

  cluster.getOperation()->moveBefore(return_op);

  return success();
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
//   4. Create `tf_device.cluster` with results consisting of the external uses
//      of cluster ops determined at 3.
//   5. Move cluster ops to `tf_device.cluster` body.
//   6. Replace external uses of cluster ops uses with `tf_device.cluster`
//      results.
//   7. Move users from 2 to after the `tf_device.cluster`.
//   8. Wrap cluster (`tf_device.cluster`) in a `tf_device.replicate` if
//      attribute `num_replicas` is greater than 1.
//   9. Copy over TPUReplicateMetadata attributes to `tf_device.cluster`.
LogicalResult FormClustersInBlock(
    Block* block,
    const TF::ResourceAliasAnalysis::Info& resource_alias_analysis) {
  MetadataMap metadata_map;
  LogicalResult result = CollectMetadata(block, &metadata_map);
  if (failed(result)) return result;

  // If there is no TPUReplicateMetadata op in this block, process blocks in
  // regions attached to the op's in the block.
  if (metadata_map.empty()) {
    for (Operation& op : *block) {
      for (Region& region : op.getRegions()) {
        if (!llvm::hasSingleElement(region))
          return op.emitOpError("Expected single block region");
        if (failed(
                FormClustersInBlock(&region.front(), resource_alias_analysis)))
          return failure();
      }
    }
    return success();
  }

  ClusterMap clusters;
  result = CollectAndGroupClusterOps(block, &clusters);
  if (failed(result)) return result;

  for (const auto& cluster_metadata_and_ops : clusters) {
    const auto& cluster_ops = cluster_metadata_and_ops.getSecond();

    auto cluster_metadata =
        metadata_map.find(cluster_metadata_and_ops.getFirst());

    // No TPUReplicateMetadata for a `_tpu_replicate` attribute.
    if (cluster_metadata == metadata_map.end()) {
      cluster_ops.front()->emitWarning()
          << "TPUReplicateMetadata for associated '" << kTPUReplicateAttr
          << "' attribute '" << cluster_metadata_and_ops.getFirst()
          << "' is missing";
      continue;
    }

    llvm::SmallSetVector<Operation*, 8> preceding_users =
        CollectClusterPrecedingUsers(block, cluster_ops,
                                     resource_alias_analysis);

    llvm::SmallVector<Value, 8> results =
        CollectClusterResults(block, cluster_ops);

    tf_device::ClusterOp cluster = CreateClusterOp(
        block, cluster_ops, results, preceding_users.getArrayRef());

    auto num_replicas = cluster_metadata->getSecond().get(kNumReplicasAttr);
    if (!num_replicas || !num_replicas.isa<mlir::IntegerAttr>())
      return cluster.emitError()
             << "requires '" << kNumReplicasAttr << "' int attribute";

    int num_cores_per_replica = 1;
    auto num_cores_per_replica_attr =
        cluster_metadata->getSecond()
            .get(kNumCoresPerReplicaAttr)
            .dyn_cast_or_null<mlir::IntegerAttr>();
    if (num_cores_per_replica_attr)
      num_cores_per_replica = num_cores_per_replica_attr.getInt();

    if (failed(ReplicateCluster(cluster,
                                num_replicas.cast<mlir::IntegerAttr>().getInt(),
                                num_cores_per_replica)))
      return failure();

    // Copy TPUReplicateMetadata attributes to `tf_device.cluster`.
    cluster->setAttrs(
        cluster_metadata->second.getDictionary(cluster.getContext()));
    // Exclude `num_replicas` as cluster should be replicated if necessary.
    cluster->removeAttr(kNumReplicasAttr);
  }

  return success();
}

LogicalResult FormClustersInFunction(
    FuncOp func,
    const TF::ResourceAliasAnalysis::Info& resource_alias_analysis) {
  if (!llvm::hasSingleElement(func))
    return func.emitOpError("Expecting a single block function");

  if (failed(FormClustersInBlock(&func.front(), resource_alias_analysis)))
    return failure();

  // Remove TPUReplicatedInput and TPUReplicatedOutput nodes.
  auto remove_result = func.walk([&](Operation* op) {
    if (!llvm::isa<TF::TPUReplicatedInputOp, TF::TPUReplicatedOutputOp>(op))
      return WalkResult::advance();

    // Forward operand to result. When `num_replicas` attribute is 1, no
    // `tf_device.replicate` is created and replicated (1) operands/results are
    // untouched.
    if (op->getNumOperands() == 1 && op->getNumResults() == 1)
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));

    // Leftover TPUReplicatedInput/TPUReplicatedOutput that are not of
    // `num_replicas` to 1.
    if (!op->use_empty()) {
      op->emitOpError() << "is expected to have no uses, but it is operand#"
                        << op->use_begin()->getOperandNumber() << " of "
                        << *op->use_begin()->getOwner();
      return WalkResult::interrupt();
    }

    op->erase();

    return WalkResult::advance();
  });

  return failure(remove_result.wasInterrupted());
}

void TPUClusterFormationPass::runOnOperation() {
  auto& resource_alias_analysis = getAnalysis<TF::ResourceAliasAnalysis>();
  for (auto func : getOperation().getOps<FuncOp>())
    if (!func.isExternal() &&
        failed(FormClustersInFunction(
            func, resource_alias_analysis.GetAnalysisForFunc(func))))
      return signalPassFailure();
}
}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPUClusterFormationPass() {
  return std::make_unique<TPUClusterFormationPass>();
}

}  // namespace TFTPU
}  // namespace mlir
