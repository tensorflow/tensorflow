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
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr llvm::StringRef kDeviceAttr = "device";
constexpr llvm::StringRef kNameAttr = "name";
constexpr llvm::StringRef kNumCoresPerReplicaAttr = "num_cores_per_replica";
constexpr llvm::StringRef kNumReplicasAttr = "num_replicas";
constexpr llvm::StringRef kReplicatedInputIndicesAttr =
    "_replicated_input_indices";
constexpr llvm::StringRef kMirroredVariableIndicesAttr =
    "_mirrored_variable_indices";
constexpr llvm::StringRef kNoReplicationCluster = "__no_replication_cluster";

constexpr llvm::StringRef kBadReplicateInfoAttrMsg =
    "requires '_replication_info' string attribute";

// Mapping for `_replication_info` attribute to TPUReplicateMetadata attributes.
using MetadataMap = llvm::SmallDenseMap<llvm::StringRef, NamedAttrList, 8>;

// A set of operations. We use a `SmallSetVector` in order to have deterministic
// traversal order (= insertion order), independent of the pointer keys.
using OpSetVector = llvm::SmallSetVector<Operation*, 8>;

// Mapping for `_replication_info` attribute to ops of a cluster.
using ClusterMap = llvm::SmallDenseMap<llvm::StringRef, OpSetVector, 8>;

struct TPUClusterFormationPass
    : public TF::TPUClusterFormationPassBase<TPUClusterFormationPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<tf_device::TensorFlowDeviceDialect>();
  }

  void runOnOperation() override;
};

// Creates a mapping from the TPUReplicateMetadata ops `_replication_info`
// attribute to its attributes and removes the ops. If multiple
// TPUReplicateMetadata ops have the same `_replication_info` attribute, an
// error will be returned.
LogicalResult CollectMetadata(Block* block, MetadataMap* metadata_map) {
  // Just look at top-level operations in the block (not nested ones)
  for (Operation& op : llvm::make_early_inc_range(*block)) {
    auto metadata_op = dyn_cast<TF::TPUReplicateMetadataOp>(op);
    if (!metadata_op) continue;

    NamedAttrList attrs(metadata_op->getAttrDictionary());

    // Missing or bad `_replication_info` attribute.
    auto replication_info_attr = attrs.get(TF::kReplicationInfoAttr);
    if (!replication_info_attr)
      return metadata_op.emitError() << kBadReplicateInfoAttrMsg;

    auto replication_info_attr_str =
        replication_info_attr.dyn_cast<StringAttr>();
    if (!replication_info_attr_str ||
        replication_info_attr_str.getValue().empty())
      return metadata_op.emitError() << kBadReplicateInfoAttrMsg;

    // Remove `name` attribute.
    attrs.erase(StringAttr::get(metadata_op.getContext(), kNameAttr));

    auto it = metadata_map->try_emplace(replication_info_attr_str.getValue(),
                                        std::move(attrs));

    // There are multiple TPUReplicateMetadata ops with the same
    // `_replication_info` attribute.
    if (!it.second) {
      return metadata_op.emitError()
             << "multiple TPUReplicateMetadata ops with the same '"
             << TF::kReplicationInfoAttr << "' attribute '"
             << replication_info_attr_str.getValue() << "' found";
    }
    metadata_op.erase();
  }
  return success();
}

// Collects and clusters ops either based on `_replication_info` attribute
// (replicated case) or using one single cluster (non-replicated case). Also
// sets `device_type` if there is any cluster (note that the device type must be
// unique, otherwise we emit an error).
// Returns an error in case of invalid compilation or replication attribute(s).
LogicalResult CollectAndGroupClusterOps(Block* block, ClusterMap* clusters,
                                        std::string& device_type) {
  bool has_replicated_compiled_op = false;
  bool has_non_replicated_compiled_op = false;
  // Use ordered set here to make error message below deterministic.
  std::set<llvm::StringRef> device_types;
  for (Operation& op : *block) {
    LogicalResult result = TF::HasValidCompilationAndReplicationAttributes(op);
    if (failed(result)) return result;

    // Collect device types which currently must be consistent per block
    // (checked later).
    auto device_type_attr =
        op.getAttrOfType<StringAttr>(TF::kCompileDeviceTypeAttr);
    if (device_type_attr) device_types.insert(device_type_attr);

    if (op.hasAttr(TF::kReplicationInfoAttr)) {
      // For replicated case, borrow cluster structure from replication info.
      // Following condition is already checked in
      // `HasValidCompilationAndReplicationAttributes` above, assert here for
      // documentation and to avoid breakage when that function is changed.
      assert(op.hasAttr(TF::kCompileDeviceTypeAttr));
      has_replicated_compiled_op = true;
      auto attr = op.getAttrOfType<StringAttr>(TF::kReplicationInfoAttr);
      auto it = clusters->try_emplace(attr.getValue());
      it.first->getSecond().insert(&op);
    } else if (op.hasAttr(TF::kCompileDeviceTypeAttr)) {
      // For non-replicated case, assume one cluster per block (in line with
      // Framework behavior).
      has_non_replicated_compiled_op = true;
      auto it = clusters->try_emplace(kNoReplicationCluster);
      it.first->getSecond().insert(&op);
    }
  }
  // Do some checks for unsupported cases.
  if (has_replicated_compiled_op && has_non_replicated_compiled_op) {
    return block->getParentOp()->emitError()
           << "found mixed replicated and non-replicated compiled ops in same "
              "block which is not supported";
  }
  if (device_types.size() > 1) {
    return block->getParentOp()->emitError()
           << "found different '" << TF::kCompileDeviceTypeAttr
           << "' attribute values (" << llvm::join(device_types, ",")
           << ") in same block which is not supported";
  }
  if (!clusters->empty()) {
    // Note that for size < 1 we shouldn't have any cluster while for size > 1
    // we should have returned with an error above.
    assert(device_types.size() == 1);
    device_type = device_types.begin()->str();
  }
  return success();
}

// Returns true iff `op` has a direct control dependency from (`incoming` ==
// true) or to (`incoming` == false) any op in `cluster_ops` or
// `cluster_dependent_ops`.
bool hasOpClusterControlDependency(
    Operation* op, bool incoming, const OpSetVector& cluster_ops,
    const OpSetVector& cluster_dependent_ops,
    const TF::SideEffectAnalysis::Info& side_effect_analysis) {
  auto filter = [&](Operation* other_op) {
    return cluster_ops.contains(other_op) ||
           cluster_dependent_ops.contains(other_op);
  };
  return incoming ? !side_effect_analysis.DirectControlPredecessors(op, filter)
                         .empty()
                  : !side_effect_analysis.DirectControlSuccessors(op, filter)
                         .empty();
}

// Returns true iff `op` has a direct data dependency from (`incoming` == true
// or to (`incoming` == false) any op in `cluster_ops` or
// `cluster_dependent_ops`.
bool hasOpClusterDataDependency(Operation* op, bool incoming,
                                const OpSetVector& cluster_ops,
                                const OpSetVector& cluster_dependent_ops) {
  auto result = op->walk([&](Operation* inner_op) {
    ValueRange values = incoming ? ValueRange(inner_op->getOperands())
                                 : ValueRange(inner_op->getResults());
    llvm::SmallVector<Operation*, 4> candidates;
    for (Value value : values) {
      if (incoming) {
        candidates = {value.getDefiningOp()};
      } else {
        candidates.assign(value.getUsers().begin(), value.getUsers().end());
      }
      for (Operation* candidate_op : candidates) {
        if (cluster_ops.contains(candidate_op) ||
            cluster_dependent_ops.contains(candidate_op)) {
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

// Collects ops that need to be moved behind the cluster due to data or control
// dependencies.
llvm::SmallSetVector<Operation*, 8> CollectClusterSuccessorOps(
    Block* block, const OpSetVector& cluster_ops,
    const TF::SideEffectAnalysis::Info& side_effect_analysis) {
  OpSetVector cluster_predecessor_ops;
  OpSetVector cluster_successor_ops;

  // Collect non-cluster ops that have a dependency to the cluster. For this
  // traverse all ops from last to first cluster op and keep track of in-between
  // non-cluster ops that have some outgoing (transitive) dependency to some
  // cluster op (`cluster_predecessor_ops`).
  auto rfront = Block::reverse_iterator(cluster_ops.front());
  auto rback = Block::reverse_iterator(cluster_ops.back());
  for (Operation& op : llvm::make_range(rback, rfront)) {
    if (cluster_ops.contains(&op)) continue;
    bool has_dependency_to_cluster =
        hasOpClusterDataDependency(&op, /*incoming=*/false, cluster_ops,
                                   cluster_predecessor_ops) ||
        hasOpClusterControlDependency(&op, /*incoming=*/false, cluster_ops,
                                      cluster_predecessor_ops,
                                      side_effect_analysis);
    if (has_dependency_to_cluster) cluster_predecessor_ops.insert(&op);
  }
  // Collect non-cluster ops that have a dependency from the cluster. For this
  // traverse all ops from first to last cluster op and keep track of in-between
  // non-cluster ops that have some incoming (transitive) dependency from some
  // cluster op (`cluster_successor_ops`).
  auto front = Block::iterator(cluster_ops.front());
  auto back = Block::iterator(cluster_ops.back());
  for (Operation& op : llvm::make_range(front, back)) {
    if (cluster_ops.contains(&op)) continue;
    bool has_dependency_from_cluster =
        hasOpClusterDataDependency(&op, /*incoming=*/true, cluster_ops,
                                   cluster_successor_ops) ||
        hasOpClusterControlDependency(&op, /*incoming=*/true, cluster_ops,
                                      cluster_successor_ops,
                                      side_effect_analysis);
    if (has_dependency_from_cluster) {
      if (cluster_predecessor_ops.contains(&op)) {
        // Op has a dependency from and to the cluster which is invalid. Instead
        // of erroring out we don't add the op to `cluster_successor_ops` which
        // is in line with previous behavior when certain control dependencies
        // were not considered.
        // TODO(b/216706460) Establish some contract here: Should we expect only
        // valid clusters, or should we split clusters accordingly? The latter
        // might have runtime impact for existing models.
        // We should make this message an error once there is such a contract
        // and once existing cases have been fixed.
        op.emitWarning()
            << "op has cyclic dependency with a compilation cluster";
      } else {
        cluster_successor_ops.insert(&op);
      }
    }
  }
  return cluster_successor_ops;
}

// Collects results and associated types of the cluster that are used outside of
// the cluster. These results and types are used to create the clusters
// `tf_device.cluster` and associated terminator. Results that have no uses
// outside of the cluster (i.e. results of ops in the cluster are only consumed
// by other ops in the cluster) are pruned.
llvm::SmallVector<Value, 8> CollectClusterResults(
    Block* block, const OpSetVector& cluster_ops) {
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
    Block* block, const OpSetVector& cluster_ops, llvm::ArrayRef<Value> results,
    llvm::ArrayRef<Operation*> cluster_successor_ops) {
  // `tf_device.cluster` will be placed at where the last op of the cluster is.
  Operation* last_cluster_op = cluster_ops.back();
  OpBuilder builder(last_cluster_op);

  llvm::SmallVector<Type, 8> result_types;
  for (Value result : results) result_types.push_back(result.getType());
  auto cluster = builder.create<tf_device::ClusterOp>(last_cluster_op->getLoc(),
                                                      result_types);

  Block* body = new Block;
  cluster.body().push_back(body);

  // Move cluster ops to the cluster body. Also remove `_replication_info` and
  // `device` attribute from ops in the cluster when that information is
  // redundant will the `tf_device.cluster`. Do this for all ops including
  // nested ops.
  for (Operation* cluster_op : cluster_ops) {
    cluster_op->moveBefore(body, body->end());
    cluster_op->walk([&](Operation* inner_op) {
      inner_op->removeAttr(TF::kReplicationInfoAttr);
      inner_op->removeAttr(TF::kCompileDeviceTypeAttr);

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

  // Move ops that depend on something in the cluster behind the cluster.
  Operation* op_after_cluster = cluster.getOperation()->getNextNode();
  for (Operation* op : cluster_successor_ops) op->moveBefore(op_after_cluster);
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
  OpSetVector unique_replicated_input_ops;
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

void SetNoReplicationClusterAttrs(tf_device::ClusterOp cluster,
                                  llvm::StringRef device_type) {
  OpBuilder builder(cluster);
  cluster->setAttr(TF::kReplicationInfoAttr,
                   builder.getStringAttr(kNoReplicationCluster));
  cluster->setAttr(TF::kCompileDeviceTypeAttr,
                   builder.getStringAttr(device_type));

  // TODO(b/229992058) Propagate `allow_soft_placement` (and other attributes?)
  // instead of hard-coding.
  cluster->setAttr("allow_soft_placement", builder.getBoolAttr(true));
  cluster->setAttr("topology", builder.getStringAttr(""));
  cluster->setAttr("num_cores_per_replica",
                   builder.getIntegerAttr(builder.getI32Type(), 1));
  cluster->setAttr("device_assignment", builder.getArrayAttr({}));
  cluster->setAttr("use_spmd_for_xla_partitioning", builder.getBoolAttr(false));
  cluster->setAttr("step_marker_location", builder.getStringAttr(""));
}

// Forms compilation clusters in `block`. If the block contains a
// `TPUReplicateMetadata` op, then we form clusters according to
// `_replication_info` values (ops with same value go to same cluster).
// Otherwise, in the non-replicated case, we build one compilation cluster per
// block.
//
// We do this in following steps:
//   1. Find `TPUReplicateMetadata` op in `block` (might not exist).
//   2. Collect and group cluster ops (either based on `_replication_info`
//      attributes or forming one single cluster).
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
    Block* block, const TF::SideEffectAnalysis::Info& side_effect_analysis) {
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
        if (failed(FormClustersInBlock(&region.front(), side_effect_analysis)))
          return failure();
      }
    }
  }

  ClusterMap clusters;
  std::string device_type;
  result = CollectAndGroupClusterOps(block, &clusters, device_type);
  if (failed(result)) return result;

  for (const auto& cluster_metadata_and_ops : clusters) {
    const auto& cluster_ops = cluster_metadata_and_ops.getSecond();

    bool has_replication =
        cluster_metadata_and_ops.getFirst() != kNoReplicationCluster;
    auto cluster_metadata =
        metadata_map.find(cluster_metadata_and_ops.getFirst());

    // No TPUReplicateMetadata for a `_replication_info` attribute.
    if (has_replication && cluster_metadata == metadata_map.end()) {
      block->getParentOp()->emitWarning()
          << "TPUReplicateMetadata for associated '" << TF::kReplicationInfoAttr
          << "' attribute '" << cluster_metadata_and_ops.getFirst()
          << "' is missing";
      continue;
    }

    OpSetVector cluster_successor_ops =
        CollectClusterSuccessorOps(block, cluster_ops, side_effect_analysis);

    llvm::SmallVector<Value, 8> results =
        CollectClusterResults(block, cluster_ops);

    tf_device::ClusterOp cluster = CreateClusterOp(
        block, cluster_ops, results, cluster_successor_ops.getArrayRef());

    if (!has_replication) {
      SetNoReplicationClusterAttrs(cluster, device_type);
      continue;
    }
    // Determine `num_replicas`.
    auto num_replicas_attr =
        cluster_metadata->getSecond().get(kNumReplicasAttr);
    if (!num_replicas_attr || !num_replicas_attr.isa<mlir::IntegerAttr>())
      return cluster.emitError()
             << "requires '" << kNumReplicasAttr << "' int attribute";
    int num_replicas = num_replicas_attr.cast<mlir::IntegerAttr>().getInt();

    // Determine `num_cores_per_replica`.
    int num_cores_per_replica = 1;
    auto num_cores_per_replica_attr =
        cluster_metadata->getSecond()
            .get(kNumCoresPerReplicaAttr)
            .dyn_cast_or_null<mlir::IntegerAttr>();
    if (num_cores_per_replica_attr)
      num_cores_per_replica = num_cores_per_replica_attr.getInt();
    if (failed(ReplicateCluster(cluster, num_replicas, num_cores_per_replica)))
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
    func::FuncOp func,
    const TF::SideEffectAnalysis::Info& side_effect_analysis) {
  if (!llvm::hasSingleElement(func))
    return func.emitOpError("Expecting a single block function");

  if (failed(FormClustersInBlock(&func.front(), side_effect_analysis)))
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
  // Attributes on tf.Constant aren't reliable: CSE will merge ConstantLike ops
  // with the same value (but different attributes!) into the same tf.Const
  // definition, potentially leading to bogus _replication_info attributes. So
  // we just scrub all tf.Constants of all extra attributes.
  // TODO(kramm): Remove this once tf.Const's folder is aware of extra
  // attributes.
  auto value_str_attr = StringAttr::get(&getContext(), "value");
  getOperation().walk([&](TF::ConstOp cst) {
    auto dict = cst->getAttrDictionary();
    if (dict.size() == 1) {
      return;  // Optimization. Assume the one attribute is "value".
    }
    // Recreate the attributes dictionary to only contain "value".
    NamedAttrList attributes;
    attributes.append(NamedAttribute(value_str_attr, cst->getAttr("value")));
    cst->setAttrs(attributes.getDictionary(&getContext()));
  });

  auto& side_effect_analysis = getAnalysis<TF::SideEffectAnalysis>();
  for (auto func : getOperation().getOps<func::FuncOp>())
    if (!func.isExternal() &&
        failed(FormClustersInFunction(
            func, side_effect_analysis.GetAnalysisForFunc(func))))
      return signalPassFailure();
}
}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPUClusterFormationPass() {
  return std::make_unique<TPUClusterFormationPass>();
}

}  // namespace TFTPU
}  // namespace mlir
