/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFTPU {

// This pass extracts a CPU computation cluster with `_xla_outside_compilation`
// annotation from the head or tail of a TPU cluster.

namespace {

constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";

bool HasOutsideCompilationAttribute(Operation* op) {
  return op->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr) != nullptr;
}

// Finds op that created a given value. If the value is a BlockArgument, this
// returns the owner of the Block.
Operation* GetOpOfValue(Value value) {
  if (auto block_arg = value.dyn_cast<BlockArgument>())
    return block_arg.getOwner()->getParentOp();

  return value.getDefiningOp();
}

// Checks if `op` is nested in `block`.
bool OpInBlock(Operation* op, Block* block) {
  Block* op_block = op->getBlock();
  while (op_block) {
    if (op_block == block) return true;
    if (auto* parent_op = op_block->getParentOp()) {
      op_block = parent_op->getBlock();
    } else {
      break;
    }
  }
  return false;
}

// Wraps block in a Launch. External uses of ops in the block will be return
// values of the Launch and remapped to the Launch results. If `before` is set
// to true, the Launch is created before `op`. Otherwise the Launch is created
// after `op`.
tf_device::LaunchOp CreateLaunchForBlock(OpBuilder* builder, Operation* op,
                                         bool before, Block* launch_block,
                                         llvm::StringRef host_device) {
  // Find results and result types of ops in block that needs to returned.
  llvm::SmallVector<Value, 4> launch_results;
  llvm::SmallVector<Type, 4> launch_result_types;
  for (Operation& head_outside_compiled_op : *launch_block) {
    for (Value result : head_outside_compiled_op.getResults()) {
      bool has_external_uses = false;
      for (Operation* user : result.getUsers()) {
        if (OpInBlock(user, launch_block)) continue;
        has_external_uses = true;
        break;
      }
      if (has_external_uses) {
        launch_results.push_back(result);
        launch_result_types.push_back(result.getType());
      }
    }
  }

  before ? builder->setInsertionPoint(op) : builder->setInsertionPointAfter(op);
  auto launch = builder->create<tf_device::LaunchOp>(
      op->getLoc(), builder->getStringAttr(host_device), launch_result_types);
  launch.body().push_back(launch_block);

  builder->setInsertionPointToEnd(&launch.GetBody());
  builder->create<tf_device::ReturnOp>(op->getLoc(), launch_results);

  return launch;
}

// Parses TPU compilation and execution devices from a TPU cluster and returns
// the host device for the head and tail computations. If the TPU computation is
// replicated, kTPUReplicatedHost is returned instead.
LogicalResult GetHostDeviceForHeadTailComputation(
    mlir::TF::RuntimeDevices devices, tf_device::ClusterOp cluster,
    std::string* host_device) {
  auto replicate = cluster.getParentOfType<tf_device::ReplicateOp>();
  if (replicate) {
    *host_device = tensorflow::kTPUReplicatedHost;
    return success();
  }

  auto num_cores_per_replica_attr =
      cluster.getAttrOfType<IntegerAttr>(tensorflow::kNumCoresPerReplicaAttr);
  if (!num_cores_per_replica_attr)
    return cluster.emitOpError(
        "cluster op missing `num_cores_per_replica` attribute");

  if (num_cores_per_replica_attr.getInt() != 1)
    return cluster.emitOpError(
        "outside compilation is not supported with model parallelism.");

  auto topology_attr =
      cluster.getAttrOfType<StringAttr>(tensorflow::kTopologyAttr);
  if (!topology_attr)
    return cluster.emitOpError("cluster op missing `topology` attribute");

  auto device_assignment_attr =
      cluster.getAttrOfType<mlir::ArrayAttr>(tensorflow::kDeviceAssignmentAttr);
  if (!device_assignment_attr)
    return cluster.emitOpError(llvm::formatv("requires attribute '{0}'",
                                             tensorflow::kDeviceAssignmentAttr)
                                   .str());

  auto status_or_device_coodinates =
      tensorflow::GetDeviceCoordinates(device_assignment_attr);

  if (!status_or_device_coodinates.ok())
    return cluster.emitError()
           << "error in fetching tpu device coordinates: "
           << status_or_device_coodinates.status().error_message();

  // Determine compilation and execution devices.
  auto status_or_tpu_device_assignment =
      tensorflow::GetTPUCompilationAndExecutionDevices(
          devices.device_names(), /*num_replicas=*/1,
          /*num_cores_per_replica=*/1, topology_attr.getValue(),
          status_or_device_coodinates.ConsumeValueOrDie());
  if (!status_or_tpu_device_assignment.ok())
    return cluster.emitError()
           << "error in fetching TPU compilation/execution devices: "
           << status_or_tpu_device_assignment.status().error_message();
  auto& tpu_device_assignment = status_or_tpu_device_assignment.ValueOrDie();

  *host_device = tpu_device_assignment.tpu_devices[0][0].host;
  return success();
}

// Returns a set of ops that are outside compiled and can be extracted to before
// the TPU computation. These ops are either connected to the inputs of the TPU
// computation or other ops that can be extracted, and have no operands from
// other ops in the TPU computation that cannot be extracted.
llvm::SmallVector<Operation*, 4> FindOutsideCompiledOpsAtHead(
    tf_device::ClusterOp cluster) {
  Region* cluster_region = &cluster.body();
  llvm::SmallSetVector<Operation*, 4> head_outside_compiled_ops;

  auto cluster_ops = cluster.GetBody().without_terminator();
  for (Operation& cluster_op : cluster_ops) {
    if (!HasOutsideCompilationAttribute(&cluster_op)) continue;
    // An outside compiled op can be extracted if its operands are not from
    // other ops in the cluster that cannot be extracted.
    auto walk_result = cluster_op.walk([&](Operation* op) {
      for (Value operand : op->getOperands()) {
        Operation* operand_op = GetOpOfValue(operand);
        if (head_outside_compiled_ops.count(operand_op)) continue;

        if (operand_op->getParentRegion() == cluster_region)
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!walk_result.wasInterrupted())
      head_outside_compiled_ops.insert(&cluster_op);
  }

  return head_outside_compiled_ops.takeVector();
}

// Moves head outside compiled ops into its own `tf_device.LaunchOp`
// computation before the cluster.
void CreateHeadComputation(OpBuilder* builder, tf_device::ClusterOp cluster,
                           llvm::ArrayRef<Operation*> head_outside_compiled_ops,
                           llvm::StringRef host_device) {
  Block* launch_block = new Block;
  for (Operation* head_outside_compiled_op : head_outside_compiled_ops)
    head_outside_compiled_op->moveBefore(launch_block, launch_block->end());

  tf_device::LaunchOp launch = CreateLaunchForBlock(
      builder, cluster, /*before=*/true, launch_block, host_device);

  for (auto result : llvm::zip(launch.GetBody().getTerminator()->getOperands(),
                               launch.getResults()))
    replaceAllUsesInRegionWith(std::get<0>(result), std::get<1>(result),
                               cluster.body());
}

// Extracts and move outside compiled ops that have no dependencies in the
// cluster to before the cluster.
mlir::LogicalResult LiftHeadOutsideCompiledOps(
    OpBuilder* builder, const mlir::TF::RuntimeDevices& devices,
    tf_device::ClusterOp cluster, std::string* host_device,
    bool* cluster_updated) {
  llvm::SmallVector<Operation*, 4> head_outside_compiled_ops =
      FindOutsideCompiledOpsAtHead(cluster);
  if (head_outside_compiled_ops.empty()) return success();
  if (failed(
          GetHostDeviceForHeadTailComputation(devices, cluster, host_device)))
    return failure();

  CreateHeadComputation(builder, cluster, head_outside_compiled_ops,
                        *host_device);

  *cluster_updated = true;
  return success();
}

// Fills `tail_outside_compiled_ops` with ops that are outside compiled and
// can be extracted to after the TPU computation, and `cluster_results` with new
// results of the cluster. These ops are either connected to the output of the
// TPU computation or other ops that can be extracted, and have no results used
// by other ops in the TPU computation that cannot be extracted.
void FindOutsideCompiledOpsAtTailAndClusterResults(
    tf_device::ClusterOp cluster,
    llvm::SmallVectorImpl<Operation*>* tail_outside_compiled_ops,
    llvm::SmallVectorImpl<Value>* cluster_results) {
  Region* cluster_region = &cluster.body();
  llvm::SmallSetVector<Operation*, 4> tail_outside_compiled_ops_set;
  Operation* terminator = cluster.GetBody().getTerminator();
  llvm::SmallSetVector<Value, 4> cluster_results_set;
  cluster_results_set.insert(terminator->getOperands().begin(),
                             terminator->getOperands().end());

  auto cluster_ops = llvm::reverse(cluster.GetBody().without_terminator());
  for (Operation& cluster_op : cluster_ops) {
    if (!HasOutsideCompilationAttribute(&cluster_op)) continue;

    llvm::SmallVector<int, 4> results_to_forward;
    bool can_be_extracted =
        llvm::all_of(cluster_op.getUsers(), [&](Operation* op) {
          return op == terminator || tail_outside_compiled_ops_set.count(op);
        });
    if (!can_be_extracted) continue;

    // Collect operands of cluster op that are generated within the cluster.
    // These values should be returned by the cluster.
    cluster_op.walk([&](Operation* op) {
      for (Value operand : op->getOperands()) {
        Operation* operand_op = GetOpOfValue(operand);
        if (operand_op->getParentRegion() == cluster_region)
          cluster_results_set.insert(operand);
      }
    });

    // Remove results of op to be extracted as there are no uses in the cluster.
    for (Value result : cluster_op.getResults())
      cluster_results_set.remove(result);
    tail_outside_compiled_ops_set.insert(&cluster_op);
  }

  *tail_outside_compiled_ops = tail_outside_compiled_ops_set.takeVector();
  *cluster_results = cluster_results_set.takeVector();
}

// Moves tail outside compiled ops into its own `tf_device.LaunchOp`
// computation after the cluster.
void CreateTailComputation(OpBuilder* builder, tf_device::ClusterOp cluster,
                           llvm::ArrayRef<Operation*> tail_outside_compiled_ops,
                           llvm::StringRef host_device) {
  Block* launch_block = new Block;
  for (Operation* tail_outside_compiled_op : tail_outside_compiled_ops)
    tail_outside_compiled_op->moveBefore(launch_block, launch_block->begin());

  tf_device::LaunchOp launch = CreateLaunchForBlock(
      builder, cluster, /*before=*/false, launch_block, host_device);

  auto operand_not_in_launch = [&](OpOperand& operand) {
    return !launch.getOperation()->isProperAncestor(operand.getOwner());
  };
  for (auto result : llvm::zip(launch.GetBody().getTerminator()->getOperands(),
                               launch.getResults()))
    std::get<0>(result).replaceUsesWithIf(std::get<1>(result),
                                          operand_not_in_launch);
}

// Updates cluster with updated cluster results after extracting tail outside
// compiled ops.
tf_device::ClusterOp UpdateClusterResults(
    OpBuilder* builder, tf_device::ClusterOp cluster,
    llvm::ArrayRef<Value> new_cluster_results) {
  Operation* old_terminator = cluster.GetBody().getTerminator();
  builder->setInsertionPoint(old_terminator);
  builder->create<tf_device::ReturnOp>(old_terminator->getLoc(),
                                       new_cluster_results);
  old_terminator->erase();

  builder->setInsertionPoint(cluster);
  llvm::SmallVector<Type, 4> new_cluster_result_types;
  new_cluster_result_types.reserve(new_cluster_results.size());
  for (const auto& new_cluster_result : new_cluster_results)
    new_cluster_result_types.push_back(new_cluster_result.getType());

  auto new_cluster = builder->create<tf_device::ClusterOp>(
      cluster.getLoc(), new_cluster_result_types,
      /*operands=*/llvm::ArrayRef<Value>{}, cluster.getAttrs());
  new_cluster.body().takeBody(cluster.body());

  auto operand_not_in_cluster = [&](OpOperand& operand) {
    return !new_cluster.getOperation()->isProperAncestor(operand.getOwner());
  };
  for (auto result :
       llvm::zip(new_cluster.GetBody().getTerminator()->getOperands(),
                 new_cluster.getResults()))
    std::get<0>(result).replaceUsesWithIf(std::get<1>(result),
                                          operand_not_in_cluster);

  cluster.erase();
  return new_cluster;
}

// Extracts and move outside compiled ops that do not create dependencies in the
// cluster to after the cluster.
mlir::LogicalResult LiftTailOutsideCompiledOps(
    OpBuilder* builder, const mlir::TF::RuntimeDevices& devices,
    std::string host_device, tf_device::ClusterOp* cluster,
    bool* cluster_updated) {
  llvm::SmallVector<Operation*, 4> tail_outside_compiled_ops;
  llvm::SmallVector<Value, 4> cluster_results;
  FindOutsideCompiledOpsAtTailAndClusterResults(
      *cluster, &tail_outside_compiled_ops, &cluster_results);
  if (tail_outside_compiled_ops.empty()) return success();

  if (host_device.empty())
    if (failed(GetHostDeviceForHeadTailComputation(devices, *cluster,
                                                   &host_device)))
      return failure();

  // Forward all results of cluster first. These results will be remapped once
  // a new cluster is formed.
  cluster->replaceAllUsesWith(
      cluster->GetBody().getTerminator()->getOperands());

  CreateTailComputation(builder, *cluster, tail_outside_compiled_ops,
                        host_device);

  *cluster = UpdateClusterResults(builder, *cluster, cluster_results);

  *cluster_updated = true;
  return success();
}

// Removes aliased outputs in cluster from ops outside of cluster.
void RemoveClusterAliasedOutputs(OpBuilder* builder,
                                 tf_device::ClusterOp cluster) {
  llvm::SmallVector<Value, 4> used_old_cluster_results;
  llvm::SmallVector<Value, 4> new_cluster_results;
  llvm::SmallVector<Type, 4> new_cluster_result_types;
  Operation* cluster_terminator = cluster.GetBody().getTerminator();
  for (auto result :
       llvm::zip(cluster_terminator->getOperands(), cluster.getResults())) {
    Value cluster_terminator_operand = std::get<0>(result);
    if (cluster.getOperation()->isProperAncestor(
            cluster_terminator_operand.getDefiningOp())) {
      new_cluster_results.push_back(cluster_terminator_operand);
      new_cluster_result_types.push_back(cluster_terminator_operand.getType());
      used_old_cluster_results.push_back(std::get<1>(result));
    } else {
      std::get<1>(result).replaceAllUsesWith(cluster_terminator_operand);
    }
  }

  if (new_cluster_results.size() == cluster.getNumResults()) return;

  builder->setInsertionPoint(cluster);
  auto new_cluster = builder->create<tf_device::ClusterOp>(
      cluster.getLoc(), new_cluster_result_types,
      /*operands=*/llvm::ArrayRef<Value>{}, cluster.getAttrs());
  new_cluster.body().takeBody(cluster.body());
  new_cluster.GetBody().getTerminator()->setOperands(new_cluster_results);

  for (auto result :
       llvm::zip(used_old_cluster_results, new_cluster.getResults()))
    std::get<0>(result).replaceAllUsesWith(std::get<1>(result));

  cluster.erase();
}

struct TPUExtractHeadTailOutsideCompilation
    : public PassWrapper<TPUExtractHeadTailOutsideCompilation,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

void TPUExtractHeadTailOutsideCompilation::runOnOperation() {
  // Get runtime devices information from the closest parent module.
  auto module = getOperation();
  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(module, &devices)))
    return signalPassFailure();

  OpBuilder builder(&getContext());
  llvm::SmallVector<tf_device::ClusterOp, 4> clusters;
  module.walk(
      [&](tf_device::ClusterOp cluster) { clusters.push_back(cluster); });

  for (tf_device::ClusterOp cluster : clusters) {
    std::string host_device;
    bool cluster_updated = false;
    if (failed(LiftHeadOutsideCompiledOps(&builder, devices, cluster,
                                          &host_device, &cluster_updated)) ||
        failed(LiftTailOutsideCompiledOps(&builder, devices, host_device,
                                          &cluster, &cluster_updated)))
      return signalPassFailure();
    if (cluster_updated) RemoveClusterAliasedOutputs(&builder, cluster);
  }
}

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUExtractHeadTailOutsideCompilationPass() {
  return std::make_unique<TPUExtractHeadTailOutsideCompilation>();
}

static PassRegistration<TPUExtractHeadTailOutsideCompilation> pass(
    "tf-tpu-extract-head-tail-outside-compilation",
    "Extracts TPU head or tail outside compilation to separate "
    "parallel_execute.");

}  // namespace TFTPU
}  // namespace mlir
