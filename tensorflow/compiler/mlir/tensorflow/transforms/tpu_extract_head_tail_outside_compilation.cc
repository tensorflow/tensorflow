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
// values of the Launch and remapped to the Launch results.
tf_device::LaunchOp CreateLaunchForBlock(OpBuilder* builder, Operation* op,
                                         Block* launch_block,
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

  builder->setInsertionPoint(op);
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

  auto walk_operands = [&](Operation* op) {
    for (Value operand : op->getOperands()) {
      Operation* operand_op = GetOpOfValue(operand);
      if (head_outside_compiled_ops.count(operand_op)) continue;

      if (operand_op->getParentRegion() == cluster_region)
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  auto cluster_ops = cluster.GetBody().without_terminator();
  for (Operation& cluster_op : cluster_ops) {
    if (!HasOutsideCompilationAttribute(&cluster_op)) continue;
    // An outside compiled op can be extracted if its operands are not from
    // other ops in the cluster that cannot be extracted.
    auto walk_result = cluster_op.walk(walk_operands);

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

  tf_device::LaunchOp launch =
      CreateLaunchForBlock(builder, cluster, launch_block, host_device);

  for (auto result : llvm::zip(launch.GetBody().getTerminator()->getOperands(),
                               launch.getResults()))
    replaceAllUsesInRegionWith(std::get<0>(result), std::get<1>(result),
                               cluster.body());
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

// Extracts and move outside compiled ops that have no dependencies in the
// cluster to before the cluster.
mlir::LogicalResult LiftHeadOutsideCompiledOps(
    OpBuilder* builder, const mlir::TF::RuntimeDevices& devices,
    tf_device::ClusterOp cluster) {
  llvm::SmallVector<Operation*, 4> head_outside_compiled_ops =
      FindOutsideCompiledOpsAtHead(cluster);
  if (head_outside_compiled_ops.empty()) return success();
  std::string host_device;
  if (failed(
          GetHostDeviceForHeadTailComputation(devices, cluster, &host_device)))
    return failure();

  CreateHeadComputation(builder, cluster, head_outside_compiled_ops,
                        host_device);

  RemoveClusterAliasedOutputs(builder, cluster);
  return success();
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
    if (failed(LiftHeadOutsideCompiledOps(&builder, devices, cluster)))
      return signalPassFailure();

    // TODO(b/157160906): Implement tail outside compiled op extraction.
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
