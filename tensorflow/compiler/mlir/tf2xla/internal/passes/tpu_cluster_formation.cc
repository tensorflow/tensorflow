/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/string_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

namespace {

using mlir::Block;
using mlir::DialectRegistry;
using mlir::LogicalResult;
using mlir::ModuleOp;
using mlir::NamedAttribute;
using mlir::NamedAttrList;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OpResult;
using mlir::Region;
using mlir::StringAttr;
using mlir::success;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;
using mlir::WalkResult;

constexpr llvm::StringRef kDeviceAttr = "device";
constexpr llvm::StringRef kNameAttr = "name";
constexpr llvm::StringRef kNumCoresPerReplicaAttr = "num_cores_per_replica";
constexpr llvm::StringRef kNumReplicasAttr = "num_replicas";
constexpr llvm::StringRef kMirroredVariableIndicesAttr =
    "_mirrored_variable_indices";

constexpr llvm::StringRef kBadReplicateInfoAttrMsg =
    "requires '_replication_info' string attribute";

// Mapping for `_replication_info` attribute to TPUReplicateMetadata attributes.
using MetadataMap = llvm::SmallDenseMap<llvm::StringRef, NamedAttrList, 8>;

// A set of operations. We use a `SmallSetVector` in order to have deterministic
// traversal order (= insertion order), independent of the pointer keys.
using OpSetVector = llvm::SmallSetVector<Operation*, 8>;

// Mapping for `_replication_info` attribute to ops of a cluster.
using ClusterMap = llvm::SmallDenseMap<llvm::StringRef, OpSetVector, 8>;

#define GEN_PASS_DEF_TPUCLUSTERFORMATIONPASS
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h.inc"

class TPUClusterFormationPass
    : public impl::TPUClusterFormationPassBase<TPUClusterFormationPass> {
 public:
  explicit TPUClusterFormationPass(bool strict_clusters)
      : strict_clusters_(strict_clusters) {}

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::tf_device::TensorFlowDeviceDialect>();
  }

  void runOnOperation() override;

 private:
  bool strict_clusters_;
};

// Creates a mapping from the TPUReplicateMetadata ops `_replication_info`
// attribute to its attributes and removes the ops. If multiple
// TPUReplicateMetadata ops have the same `_replication_info` attribute, an
// error will be returned.
LogicalResult CollectMetadata(Block* block, MetadataMap* metadata_map) {
  // Just look at top-level operations in the block (not nested ones)
  for (Operation& op : llvm::make_early_inc_range(*block)) {
    auto metadata_op = llvm::dyn_cast<mlir::TF::TPUReplicateMetadataOp>(op);
    if (!metadata_op) continue;

    NamedAttrList attrs(metadata_op->getAttrDictionary());

    // Missing or bad `_replication_info` attribute.
    auto replication_info_attr = attrs.get(mlir::TF::kReplicationInfoAttr);
    if (!replication_info_attr)
      return metadata_op.emitError() << kBadReplicateInfoAttrMsg;

    auto replication_info_attr_str =
        mlir::dyn_cast<StringAttr>(replication_info_attr);
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
             << mlir::TF::kReplicationInfoAttr << "' attribute '"
             << replication_info_attr_str.getValue() << "' found";
    }
    metadata_op.erase();
  }
  return success();
}

struct OpDevice {
  Operation* op;
  std::string device;
};

LogicalResult HasValidDeviceTypeAttribute(Block* block) {
  // Use ordered set here to make error message below deterministic.
  std::set<llvm::StringRef> device_types;
  for (Operation& op : *block) {
    // Collect device types which currently must be consistent per block
    // (checked later).
    if (auto device_type_attr =
            op.getAttrOfType<StringAttr>(mlir::TF::kCompileDeviceTypeAttr)) {
      // tf.StatefulPartitionedCall ops with and without
      // _tpu_replicate attributes may exist in the same graph. Ops without
      // the attribute but with _XlaMustCompile=true would have
      // _xla_compile_device_type="" after
      // CanonicalizeCompileAndReplicateAttributesPass. Skip empty value here.
      if (!device_type_attr.getValue().empty()) {
        device_types.insert(device_type_attr);
      }
    }
  }

  if (device_types.size() > 1) {
    return block->getParentOp()->emitError()
           << "found different '" << mlir::TF::kCompileDeviceTypeAttr
           << "' attribute values (" << llvm::join(device_types, ",")
           << ") in same block which is not supported";
  }
  return success();
}

// Collects and clusters ops based on `_replication_info` attribute. Returns
// an error in case of invalid compilation or replication attribute(s).
LogicalResult CollectAndGroupClusterOps(Block* block, ClusterMap* clusters) {
  LogicalResult result = HasValidDeviceTypeAttribute(block);
  if (failed(result)) return result;

  for (Operation& op : *block) {
    LogicalResult result =
        mlir::TF::HasValidCompilationAndReplicationAttributes(op);
    if (failed(result)) return result;

    // Skip ops with non-TPU device type, they are handled elsewhere.
    auto device_type_attr =
        op.getAttrOfType<StringAttr>(mlir::TF::kCompileDeviceTypeAttr);
    if (device_type_attr) {
      if (device_type_attr.getValue().empty()) continue;
      if (device_type_attr.getValue() != mlir::TF::kTpuDevice) continue;
    }

    if (op.hasAttr(mlir::TF::kReplicationInfoAttr)) {
      // For replicated case, borrow cluster structure from replication info.
      // Following condition is already checked in
      // `HasValidCompilationAndReplicationAttributes` above, assert here for
      // documentation and to avoid breakage when that function is changed.
      assert(op.hasAttr(mlir::TF::kCompileDeviceTypeAttr));
      auto attr = op.getAttrOfType<StringAttr>(mlir::TF::kReplicationInfoAttr);
      auto it = clusters->try_emplace(attr.getValue());
      it.first->getSecond().insert(&op);
    }
  }
  return success();
}

// Returns true iff `op` has a direct control dependency from (`incoming` ==
// true) or to (`incoming` == false) any op in `cluster_ops` or
// `cluster_dependent_ops`.
Operation* getOpClusterControlDependency(
    Operation* op, bool incoming, const OpSetVector& cluster_ops,
    const OpSetVector& cluster_dependent_ops,
    const mlir::TF::SideEffectAnalysis::Info& side_effect_analysis) {
  auto filter = [&](Operation* other_op) {
    return cluster_ops.contains(other_op) ||
           cluster_dependent_ops.contains(other_op);
  };
  auto other_ops =
      incoming ? side_effect_analysis.DirectControlPredecessors(op, filter)
               : side_effect_analysis.DirectControlSuccessors(op, filter);
  if (other_ops.empty())
    return nullptr;
  else
    return *other_ops.begin();
}

// Returns true iff `op` has a direct data dependency from (`incoming` == true
// or to (`incoming` == false) any op in `cluster_ops` or
// `cluster_dependent_ops`.
Operation* getOpClusterDataDependency(
    Operation* op, bool incoming, const OpSetVector& cluster_ops,
    const OpSetVector& cluster_dependent_ops) {
  Operation* other_op = nullptr;
  op->walk([&](Operation* inner_op) {
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
          other_op = candidate_op;
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  return other_op;
}

// Collects ops that need to be moved behind the cluster due to data or control
// dependencies.
mlir::FailureOr<llvm::SmallSetVector<Operation*, 8>> CollectClusterSuccessorOps(
    Block* block, const OpSetVector& cluster_ops,
    const mlir::TF::SideEffectAnalysis::Info& side_effect_analysis,
    bool strict_clusters) {
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
        getOpClusterDataDependency(&op, /*incoming=*/false, cluster_ops,
                                   cluster_predecessor_ops) != nullptr ||
        getOpClusterControlDependency(&op, /*incoming=*/false, cluster_ops,
                                      cluster_predecessor_ops,
                                      side_effect_analysis) != nullptr;
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
    Operation* data_predecessor = getOpClusterDataDependency(
        &op, /*incoming=*/true, cluster_ops, cluster_successor_ops);
    Operation* control_predecessor = getOpClusterControlDependency(
        &op, /*incoming=*/true, cluster_ops, cluster_successor_ops,
        side_effect_analysis);
    if (data_predecessor || control_predecessor) {
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
        mlir::InFlightDiagnostic error =
            strict_clusters ? mlir::emitError(op.getLoc(), "")
                            : mlir::emitWarning(op.getLoc(), "");
        error << "Op has cyclic dependency with a compilation cluster:\n";
        error << "The cluster depends on\n";
        error << op.getName() << "\n"
              << "which is outside of cluster, but itself depends ";
        if (data_predecessor) {
          error << "on\n" << data_predecessor->getName() << "\n";
        } else if (control_predecessor) {
          error << "(via control) on\n"
                << control_predecessor->getName() << "\n";
        }
        if (cluster_ops.contains(data_predecessor) ||
            cluster_ops.contains(control_predecessor))
          error << "which is inside the cluster.\n";
        else
          error << "which the cluster depends on.\n";
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
mlir::tf_device::ClusterOp CreateClusterOp(
    Block* block, const OpSetVector& cluster_ops, llvm::ArrayRef<Value> results,
    llvm::ArrayRef<Operation*> cluster_successor_ops) {
  // `tf_device.cluster` will be placed at where the last op of the cluster is.
  Operation* last_cluster_op = cluster_ops.back();
  OpBuilder builder(last_cluster_op);

  llvm::SmallVector<Type, 8> result_types;
  for (Value result : results) result_types.push_back(result.getType());
  auto cluster = builder.create<mlir::tf_device::ClusterOp>(
      last_cluster_op->getLoc(), result_types);

  Block* body = new Block;
  cluster.getBody().push_back(body);

  // Move cluster ops to the cluster body. Also remove `_replication_info` and
  // `device` attribute from ops in the cluster when that information is
  // redundant will the `tf_device.cluster`. Do this for all ops including
  // nested ops.
  for (Operation* cluster_op : cluster_ops) {
    cluster_op->moveBefore(body, body->end());
    cluster_op->walk([&](Operation* inner_op) {
      inner_op->removeAttr(mlir::TF::kReplicationInfoAttr);
      inner_op->removeAttr(mlir::TF::kCompileDeviceTypeAttr);

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
  builder.create<mlir::tf_device::ReturnOp>(last_cluster_op->getLoc(), results);

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

// Returns an op of the given type that uses the result, along with
// a list of identity ops along the way.
template <typename T>
std::tuple<T, llvm::SmallVector<mlir::TF::IdentityOp, 4>> GetSingleUserOfType(
    OpResult result) {
  llvm::SmallVector<mlir::TF::IdentityOp, 4> identity_ops;

  do {
    Operation* user = result.hasOneUse() ? *result.getUsers().begin() : nullptr;
    if (auto t = llvm::dyn_cast_or_null<T>(user)) {
      return std::make_tuple(t, identity_ops);
    } else if (auto identity =
                   llvm::dyn_cast_or_null<mlir::TF::IdentityOp>(user)) {
      identity_ops.emplace_back(identity);
      result = identity->getResult(0);
    } else {
      result = OpResult();  // reset to stop iterating
    }
  } while (result);

  return std::make_tuple(T(), identity_ops);
}

using PartitionedClusterOutputMap = absl::flat_hash_map<
    uint64_t, llvm::SmallVector<mlir::TF::TPUPartitionedOutputV2Op, 8>>;

// Returns the partitioned output ops from the cluster if there are any,
// along with any single user identity ops between them. Not all outputs
// of a cluster must be partitioned, so the output is a map from cluster
// output ids to ops.
std::tuple<PartitionedClusterOutputMap,
           llvm::SmallVector<mlir::TF::IdentityOp, 8>>
GetPartitionedOutputsAndIdentityOps(mlir::tf_device::ClusterOp cluster) {
  PartitionedClusterOutputMap partitioned_outputs;
  llvm::SmallVector<mlir::TF::IdentityOp, 8> erase_list;

  for (auto [cluster_result_id, cluster_result] :
       llvm::enumerate(cluster.getResults())) {
    auto [replicated_output, _] =
        GetSingleUserOfType<mlir::TF::TPUReplicatedOutputOp>(cluster_result);
    if (replicated_output) {
      for (OpResult per_replica_result : replicated_output->getResults()) {
        auto [partitioned_output, id_ops] =
            GetSingleUserOfType<mlir::TF::TPUPartitionedOutputV2Op>(
                per_replica_result);
        if (partitioned_output) {
          erase_list.insert(erase_list.end(), id_ops.begin(), id_ops.end());
          partitioned_outputs[cluster_result_id].emplace_back(
              partitioned_output);
        }
      }
    }
  }

  return std::forward_as_tuple(partitioned_outputs, erase_list);
}

// Inlines the partitioned output ops into the cluster, and updates
// their users to point to the replicate op instead.
Operation* BuildPartitionedOutputs(
    OpBuilder& builder, mlir::tf_device::ClusterOp cluster,
    mlir::tf_device::ReplicateOp replicate_op,
    PartitionedClusterOutputMap& partitioned_outputs,
    llvm::SmallVector<mlir::TF::IdentityOp, 8>& erase_list,
    llvm::SmallVector<Type, 8>& result_types, int num_replicas) {
  Operation* result_op;
  llvm::SmallVector<Value, 8> results;
  uint64_t num_results = cluster.getNumResults();
  for (uint64_t result_id = 0; result_id < num_results; ++result_id) {
    auto search = partitioned_outputs.find(result_id);
    if (search == partitioned_outputs.end()) {
      // If the output is not partitioned, directly pass it through.
      results.emplace_back(cluster.getResult(result_id));

      continue;
    }

    // Otherwise, "inline" the partitioned output ops by:
    // - Building a new op within the cluster.
    // - Replacing all the uses of the original ops with the cluster's outputs.
    llvm::SmallVector<mlir::TF::TPUPartitionedOutputV2Op, 8>& ops =
        search->second;
    for (auto [replica_id, partitioned_output] : llvm::enumerate(ops)) {
      for (auto [core_id, result] :
           llvm::enumerate(partitioned_output->getResults())) {
        // outputs from replicate op are interleaved:
        // [(replica:0,core:0), (replica:1,core:0), ...,
        //  (replica:0,core:1), (replica:1,core:1), ...]
        uint64_t output_id =
            core_id * num_replicas + replica_id + results.size();
        result.replaceAllUsesWith(replicate_op.getResult(output_id));
      }
    }

    // Assume all the replicas have the same structure.
    mlir::TF::TPUPartitionedOutputV2Op first_op = *(ops.begin());
    mlir::ArrayAttr dims = first_op.getPartitionDimsAttr();
    StringAttr sharding = first_op.get_XlaShardingAttr();
    Operation::result_type_range output_types = first_op.getResultTypes();
    result_op = builder.create<mlir::TF::TPUPartitionedOutputV2Op>(
        replicate_op.getLoc(), output_types, cluster.getResult(result_id), dims,
        sharding);

    results.insert(results.end(), result_op->getResults().begin(),
                   result_op->getResults().end());
  }

  // Once we've accumulated all the cluster's results, build a return op.
  builder.create<mlir::tf_device::ReturnOp>(result_op->getLoc(), results);

  // Then erase all the identity and partitioned output ops.
  for (const auto& [_, ops] : partitioned_outputs) {
    for (mlir::TF::TPUPartitionedOutputV2Op op : ops) {
      op->erase();
    }
  }

  for (mlir::TF::IdentityOp to_erase : erase_list) {
    to_erase->erase();
  }

  return result_op;
}

// Return the cluster's per-replica result type, converting any full-shaped
// tensor types into sharded-shaped ones if they're partitioned.
llvm::SmallVector<Type, 8> GetClusterResultTypes(
    mlir::tf_device::ClusterOp cluster,
    const PartitionedClusterOutputMap& partitioned_outputs) {
  llvm::SmallVector<Type, 8> result_types;
  Operation::result_type_range cluster_result_types = cluster.getResultTypes();
  if (partitioned_outputs.empty()) {
    // Directly pass through the cluster's outputs if none are partitioned.
    result_types.insert(result_types.end(), cluster_result_types.begin(),
                        cluster_result_types.end());
  } else {
    // For each output of the cluster...
    for (auto [output_id, result_type] :
         llvm::enumerate(cluster_result_types)) {
      auto search = partitioned_outputs.find(output_id);
      if (search == std::end(partitioned_outputs)) {
        // If it's not partitioned, directly pass it through.
        result_types.emplace_back(result_type);
      } else {
        // Otherwise, pass through the result shard types.
        Operation::result_type_range partitioned_result_types =
            (*search->second.begin())->getResultTypes();
        result_types.insert(result_types.end(),
                            partitioned_result_types.begin(),
                            partitioned_result_types.end());
      }
    }
  }
  return result_types;
}

// Creates a `tf_device.replicate` to represent replication for the cluster, if
// necessary. Erases Identity ops between partitioned and replicated output ops.
LogicalResult ReplicateCluster(mlir::tf_device::ClusterOp cluster,
                               int num_replicas, int num_cores_per_replica) {
  OpBuilder builder(cluster);
  auto [partitioned_outputs, erase_list] =
      GetPartitionedOutputsAndIdentityOps(cluster);

  for (auto [_, ops] : partitioned_outputs) {
    if (!(ops.empty() || ops.size() == num_replicas)) {
      return (ops.begin())->emitOpError()
             << "expected zero or " << num_replicas
             << " 'TPUPartitionedOutput' op(s), instead got "
             << partitioned_outputs.size();
    }
  }

  // No need to replicate.
  if (num_replicas == 1) {
    // Collapse all the Identity ops between the TRO and TPO ops.
    if (!partitioned_outputs.empty()) {
      for (mlir::TF::IdentityOp to_erase : erase_list) {
        Value in = to_erase->getOperand(0);
        OpResult out = to_erase->getResult(0);
        out.replaceAllUsesWith(in);
        to_erase->erase();
      }
    }

    return success();
  }

  if (num_replicas < 1)
    return cluster.emitError() << "requires '" << kNumReplicasAttr
                               << "' int attribute to be at least 1";

  LogicalResult status = success();
  // Collect all used TPUReplicatedInput ops.
  llvm::SmallVector<mlir::TF::TPUReplicatedInputOp, 8> replicated_input_ops;
  llvm::SmallSet<mlir::TF::TPUReplicatedInputOp, 8> seen_ops;
  mlir::visitUsedValuesDefinedAbove(
      cluster.getBody(), cluster.getBody(), [&](mlir::OpOperand* operand) {
        Operation* def = operand->get().getDefiningOp();
        if (auto ri =
                llvm::dyn_cast_or_null<mlir::TF::TPUReplicatedInputOp>(def)) {
          if (!seen_ops.contains(ri)) {
            seen_ops.insert(ri);
            replicated_input_ops.push_back(ri);
          }
        }
        // When model parallelism is used in conjunction with data parallelism
        // for resource inputs, we need to collect the per replica resource
        // inputs from input to `tf.TPUPartitionedInputV2` ops.
        if (auto pi = llvm::dyn_cast_or_null<mlir::TF::TPUPartitionedInputV2Op>(
                def)) {
          if (pi->getNumOperands() != num_cores_per_replica)
            status = pi.emitOpError()
                     << "requires " << num_cores_per_replica
                     << " operands but found " << pi->getNumOperands();
          for (auto operand : pi.getInputs()) {
            if (auto ri =
                    llvm::dyn_cast_or_null<mlir::TF::TPUReplicatedInputOp>(
                        operand.getDefiningOp())) {
              if (!seen_ops.contains(ri)) {
                seen_ops.insert(ri);
                replicated_input_ops.push_back(ri);
              }
            }
          }
        }
      });

  if (failed(status)) return mlir::failure();

  // Indices of the replicate op's arguments that are mirrored variables.
  llvm::SmallVector<int64_t, 8> mirrored_variable_indices;

  // Check if number of operands of each used TPUReplicatedInput op matches
  // `num_replicas` or 1. Collect all their operands and associated type for
  // creating the replicate op.
  llvm::SmallVector<std::pair<ValueRange, Type>, 8> replicated_inputs;
  llvm::SmallVector<Value, 8> packed_inputs;
  llvm::SmallVector<mlir::TF::TPUReplicatedInputOp, 8> replicated_ops;
  llvm::SmallVector<mlir::TF::TPUReplicatedInputOp, 8> packed_ops;
  for (const auto& pos_and_input : llvm::enumerate(replicated_input_ops)) {
    auto input = pos_and_input.value();
    bool is_packed = input.getIsPacked();
    const int num_operands = input->getNumOperands();
    int num_inputs = is_packed ? 1 : num_replicas;
    if (num_operands != num_inputs)
      return input->emitOpError() << "requires " << num_inputs << " operands";
    if (is_packed) {
      packed_inputs.push_back(input->getOperand(0));
      packed_ops.push_back(input);
    } else {
      replicated_inputs.push_back(
          {input->getOperands(), input->getOperand(0).getType()});
      replicated_ops.push_back(input);
    }
  }

  // Create `ordered_tpu_replicate_inputs` which contains the final ordered
  // replicate inputs. All packed arguments are moved to the end of the arg
  // list.
  llvm::SmallVector<mlir::TF::TPUReplicatedInputOp, 8>
      ordered_tpu_replicate_inputs = replicated_ops;
  ordered_tpu_replicate_inputs.append(packed_ops.begin(), packed_ops.end());

  // Assign `mirrored_variable_indices` based on the ordered replicated inputs.
  for (const auto& pos_and_input :
       llvm::enumerate(ordered_tpu_replicate_inputs)) {
    auto tpu_replicated_input = pos_and_input.value();
    if (tpu_replicated_input.getIsMirroredVariable()) {
      mirrored_variable_indices.push_back(pos_and_input.index());
    }
  }

  // Create replicate op.
  auto result_types = GetClusterResultTypes(cluster, partitioned_outputs);
  auto replicate_op = builder.create<mlir::tf_device::ReplicateOp>(
      cluster.getLoc(), num_replicas,
      llvm::SmallDenseMap<llvm::StringRef,
                          llvm::SmallVector<llvm::StringRef, 4>>(),
      replicated_inputs, packed_inputs, result_types);

  if (!mirrored_variable_indices.empty())
    replicate_op->setAttr(kMirroredVariableIndicesAttr,
                          builder.getI64ArrayAttr(mirrored_variable_indices));

  // Replace replicated cluster results with replicate op results.
  uint64_t offset = 0;
  for (auto [idx, result] : llvm::enumerate(cluster.getResults())) {
    if (partitioned_outputs.contains(idx)) {
      // Partitioned output propagation happens in BuildPartitionedOutputs.
      offset += num_replicas * num_cores_per_replica;
      continue;
    }

    auto replicate_outputs = llvm::make_range(
        std::next(replicate_op.result_begin(), offset),
        std::next(replicate_op.result_begin(), offset + num_replicas));
    for (auto& use : llvm::make_early_inc_range(result.getUses())) {
      Operation* def = use.getOwner();
      if (!llvm::isa<mlir::TF::TPUReplicatedOutputOp>(def)) {
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

    offset += num_replicas;
  }

  // Collect all `tf.TPUPartitionedInputV2` ops to be moved inside the
  // `tf_device.replicate` later.
  llvm::SmallSet<Operation*, 4> partitioned_inputs;
  for (auto input_and_block_arg :
       llvm::zip(ordered_tpu_replicate_inputs,
                 replicate_op.GetBody().getArguments())) {
    mlir::TF::TPUReplicatedInputOp input = std::get<0>(input_and_block_arg);
    Value block_arg = std::get<1>(input_and_block_arg);
    mlir::replaceAllUsesInRegionWith(input->getResult(0), block_arg,
                                     cluster.getBody());
    // Update replicated input use in tf.TPUPartitionedInputV2 op.
    for (auto& use : input->getUses()) {
      auto pi =
          llvm::dyn_cast<mlir::TF::TPUPartitionedInputV2Op>(use.getOwner());
      if (pi) {
        pi.setOperand(use.getOperandNumber(), block_arg);
        partitioned_inputs.insert(pi.getOperation());
      }
    }
  }

  // Create terminator for replicate op and move `tf_device.cluster` and
  // `tf.TPUPartitionedInputV2`(s) into replicate body.
  builder.setInsertionPointToEnd(&replicate_op.GetBody());

  Operation* result_op;
  if (!partitioned_outputs.empty()) {
    result_op = BuildPartitionedOutputs(builder, cluster, replicate_op,
                                        partitioned_outputs, erase_list,
                                        result_types, num_replicas);
  } else {
    result_op = builder.create<mlir::tf_device::ReturnOp>(replicate_op.getLoc(),
                                                          cluster.getResults());
  }

  for (auto pi : partitioned_inputs) pi->moveBefore(result_op);

  cluster.getOperation()->moveBefore(result_op);

  return success();
}

// Forms clusters with ops of the same `_replication_info` attribute under a
// block.
//
// For a given block, clusters are formed via grouping ops by
// `_replication_info` attributes. For every cluster formed:
//   1. Find associated TPUReplicateMetadata attributes with the same
//      `_replication_info` attribute.
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
    const mlir::TF::SideEffectAnalysis::Info& side_effect_analysis,
    bool strict_clusters) {
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
        if (failed(FormClustersInBlock(&region.front(), side_effect_analysis,
                                       strict_clusters)))
          return mlir::failure();
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

    // llvm::errs() << __func__ << "\n";
    //  No TPUReplicateMetadata for a `_replication_info` attribute.
    if (cluster_metadata == metadata_map.end()) {
      block->getParentOp()->emitWarning()
          << "TPUReplicateMetadata for associated '"
          << mlir::TF::kReplicationInfoAttr << "' attribute '"
          << cluster_metadata_and_ops.getFirst() << "' is missing";
      continue;
    }

    auto status = CollectClusterSuccessorOps(
        block, cluster_ops, side_effect_analysis, strict_clusters);
    if (failed(status)) return status;
    OpSetVector cluster_successor_ops = *status;

    llvm::SmallVector<Value, 8> results =
        CollectClusterResults(block, cluster_ops);

    mlir::tf_device::ClusterOp cluster = CreateClusterOp(
        block, cluster_ops, results, cluster_successor_ops.getArrayRef());

    auto num_replicas = cluster_metadata->getSecond().get(kNumReplicasAttr);
    if (!num_replicas || !num_replicas.isa<mlir::IntegerAttr>())
      return cluster.emitError()
             << "requires '" << kNumReplicasAttr << "' int attribute";

    int num_cores_per_replica = 1;
    auto num_cores_per_replica_attr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
        cluster_metadata->getSecond().get(kNumCoresPerReplicaAttr));
    if (num_cores_per_replica_attr)
      num_cores_per_replica = num_cores_per_replica_attr.getInt();
    if (failed(ReplicateCluster(cluster,
                                num_replicas.cast<mlir::IntegerAttr>().getInt(),
                                num_cores_per_replica)))
      return mlir::failure();

    // Copy TPUReplicateMetadata attributes to `tf_device.cluster`.
    cluster->setAttrs(
        cluster_metadata->second.getDictionary(cluster.getContext()));
    // Exclude `num_replicas` as cluster should be replicated if necessary.
    cluster->removeAttr(kNumReplicasAttr);
  }

  return success();
}

LogicalResult FormClustersInFunction(
    mlir::func::FuncOp func,
    const mlir::TF::SideEffectAnalysis::Info& side_effect_analysis,
    bool strict_clusters) {
  if (!llvm::hasSingleElement(func))
    return func.emitOpError("Expecting a single block function");

  if (failed(FormClustersInBlock(&func.front(), side_effect_analysis,
                                 strict_clusters)))
    return mlir::failure();

  // Remove TPUReplicatedInput and TPUReplicatedOutput nodes.
  auto remove_result = func.walk([&](Operation* op) {
    if (!llvm::isa<mlir::TF::TPUReplicatedInputOp,
                   mlir::TF::TPUReplicatedOutputOp>(op))
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

  return mlir::failure(remove_result.wasInterrupted());
}

void TPUClusterFormationPass::runOnOperation() {
  // Attributes on tf.Constant aren't reliable: CSE will merge ConstantLike ops
  // with the same value (but different attributes!) into the same tf.Const
  // definition, potentially leading to bogus _replication_info attributes. So
  // we just scrub all tf.Constants of all extra attributes.
  // TODO(kramm): Remove this once tf.Const's folder is aware of extra
  // attributes.
  auto value_str_attr = StringAttr::get(&getContext(), "value");
  getOperation().walk([&](mlir::TF::ConstOp cst) {
    auto dict = cst->getAttrDictionary();
    if (dict.size() == 1) {
      return;  // Optimization. Assume the one attribute is "value".
    }
    // Recreate the attributes dictionary to only contain "value".
    NamedAttrList attributes;
    attributes.append(NamedAttribute(value_str_attr, cst->getAttr("value")));
    cst->setAttrs(attributes.getDictionary(&getContext()));
  });

  auto& side_effect_analysis = getAnalysis<mlir::TF::SideEffectAnalysis>();
  for (auto func : getOperation().getOps<mlir::func::FuncOp>())
    if (!func.isExternal() &&
        failed(FormClustersInFunction(
            func, side_effect_analysis.GetAnalysisForFunc(func),
            strict_clusters_)))
      return signalPassFailure();
}
}  // anonymous namespace

std::unique_ptr<mlir::OperationPass<ModuleOp>> CreateTPUClusterFormationPass(
    bool strict_clusters) {
  return std::make_unique<TPUClusterFormationPass>(strict_clusters);
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
