/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_rewrite_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_sharding_util.h"
#include "tensorflow/core/ir/types/dialect.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

namespace {

using llvm::dyn_cast;
using mlir::Block;
using mlir::BlockArgument;
using mlir::DenseIntElementsAttr;
using mlir::failure;
using mlir::Location;
using mlir::LogicalResult;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationPass;
using mlir::OpOperand;
using mlir::RankedTensorType;
using mlir::StringAttr;
using mlir::success;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;
using mlir::WalkResult;
using mlir::func::FuncOp;
using mlir::TF::ConstOp;
using mlir::TF::IdentityOp;
using mlir::TF::ShapeAttr;
using mlir::TF::TPUDummyInputOp;
using mlir::TF::XlaAllReduceOp;
using mlir::tf_device::ClusterOp;
using mlir::tf_device::LaunchOp;
using mlir::tf_device::ParallelExecuteOp;
using mlir::tf_device::ReplicateOp;

#define GEN_PASS_DEF_XLABROADCASTPASS
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h.inc"

struct XlaBroadcast : public impl::XlaBroadcastPassBase<XlaBroadcast> {
  void runOnOperation() override;
};

// Returns true iff the broadcast val can be substituted with an XlaAllReduce.
// Sets `zero_type` and `shape` as params for the dummy zeros that will be
// created.
bool GetDummyParams(OpBuilder& builder, Value val_bcast, Type& zero_type,
                    ShapeAttr& shape) {
  Type type = val_bcast.getType();
  Type elem_type = getElementTypeOrSelf(type);
  // Xla's all_reduce legalizer bitcasts to 32 bits, so only
  // element types size <= 4 bytes are supported.
  if (elem_type.isBF16() || elem_type.isF16() || elem_type.isTF32() ||
      elem_type.isF32()) {
    zero_type = type;
  } else {
    return false;
  }
  if (auto ranked_type = dyn_cast<RankedTensorType>(type)) {
    llvm::ArrayRef<int64_t> type_shape = ranked_type.getShape();
    for (int64_t i : type_shape) {
      if (i < 0) return false;
    }
    shape = ShapeAttr::get(builder.getContext(), type_shape);
  } else {
    return false;
  }
  return true;
}

// Create a dummy zero to be fed locally from the host to the TPUExecute.
Value CreateZeroInput(Location loc, OpBuilder& builder, Type zero_type,
                      ShapeAttr shape_attr) {
  TPUDummyInputOp tpu_dummy_input =
      builder.create<TPUDummyInputOp>(loc, zero_type, shape_attr);
  tpu_dummy_input->setAttr(kICIWeightDistributionMlirBridgeMarker,
                           builder.getBoolAttr(true));
  return tpu_dummy_input;
}

// Add parallel collection of inputs to the replicated op.
LogicalResult AppendReplicatedInput(OpBuilder& builder, ReplicateOp replicate,
                                    ValueRange inputs, Type type,
                                    BlockArgument& block_arg) {
  // check that inputs length is same as num_replicas.
  if (inputs.size() != replicate.getN()) {
    return replicate.emitError()
           << "Expected numper of inputs (" << inputs.size()
           << ") to append to replicate to be num_replicas ("
           << replicate.getN() << ")";
  }

  // add block arg to region. This is in $body.
  unsigned block_arg_idx = replicate.GetNumReplicatedBlockArguments();
  Block& block = replicate.GetBody();
  block_arg = block.insertArgument(block_arg_idx, type, replicate.getLoc());

  // add to $replicated_inputs. This also updates OperandSegmentSizeAttr.
  replicate.getReplicatedInputsMutable().append(inputs);

  return success();
}

// Insert an XlaAllReduce.
Value CreateAllReduce(ReplicateOp replicate, OpBuilder& builder,
                      Value block_arg) {
  // This group_assignment is a list of all replicas. This says that the
  // reduction sent to each replica is over all replicas.
  uint32_t num_replicas = replicate.getN();
  llvm::SmallVector<int32_t, 4> group_assignment_val;
  for (int i = 0; i < num_replicas; ++i) group_assignment_val.push_back(i);
  Value group_assignment = builder.create<ConstOp>(
      block_arg.getLoc(),
      DenseIntElementsAttr::get(
          RankedTensorType::get({1, num_replicas}, builder.getIntegerType(32)),
          group_assignment_val));

  StringAttr reduce_op = builder.getStringAttr("Add");
  StringAttr mode = builder.getStringAttr("CrossReplica");
  return builder.create<XlaAllReduceOp>(block_arg.getLoc(), block_arg.getType(),
                                        block_arg, group_assignment, reduce_op,
                                        mode);
}

// Creates a missing attribute error message.
std::string CreateMissingAttributeMsg(llvm::StringRef attribute) {
  return llvm::formatv("requires attribute '{0}'", attribute).str();
}

LogicalResult GetTpuDeviceAssignment(
    ClusterOp cluster, ReplicateOp replicate, mlir::ModuleOp module,
    absl::StatusOr<TPUDeviceAssignment>& status_or_tpu_device_assignment) {
  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(module, &devices))) return failure();

  uint32_t num_replicas = replicate.getN();

  auto num_cores_per_replica_attr = cluster->getAttrOfType<mlir::IntegerAttr>(
      tensorflow::kNumCoresPerReplicaAttr);
  if (!num_cores_per_replica_attr)
    return cluster.emitOpError(
        CreateMissingAttributeMsg(tensorflow::kNumCoresPerReplicaAttr));
  int num_cores_per_replica = num_cores_per_replica_attr.getInt();

  auto topology_attr = cluster->getAttrOfType<StringAttr>("topology");
  if (!topology_attr)
    return cluster.emitOpError(
        CreateMissingAttributeMsg(tensorflow::kTopologyAttr));

  auto device_assignment_attr = cluster->getAttrOfType<mlir::ArrayAttr>(
      tensorflow::kDeviceAssignmentAttr);
  if (!device_assignment_attr)
    return cluster.emitOpError(llvm::formatv("requires attribute '{0}'",
                                             tensorflow::kDeviceAssignmentAttr)
                                   .str());

  auto status_or_device_coodinates =
      tensorflow::GetDeviceCoordinates(device_assignment_attr);
  if (!status_or_device_coodinates.ok())
    return cluster.emitError()
           << "error in fetching tpu device coordinates: "
           << status_or_device_coodinates.status().message();

  status_or_tpu_device_assignment =
      tensorflow::GetTPUCompilationAndExecutionDevices(
          devices.device_names(), num_replicas, num_cores_per_replica,
          topology_attr.getValue(), status_or_device_coodinates.value());
  return success();
}

// Move a broadcast into the XLA cluster, converting it to an XlaAllReduce. This
// skips if the element type is not known to be valid for XlaAllReduce.
LogicalResult MoveBroadcastToCluster(OpBuilder& builder,
                                     OpBuilder& inner_builder,
                                     ClusterOp cluster, ReplicateOp replicate,
                                     llvm::DenseMap<Value, Value>& orig_to_new,
                                     Value val_bcast, mlir::ModuleOp module) {
  Type zero_type;
  ShapeAttr shape_attr;
  if (!GetDummyParams(builder, val_bcast, zero_type, shape_attr))
    return success();
  llvm::SmallVector<Value, 4> inputs;
  inputs.push_back(val_bcast);
  uint32_t num_replicas = replicate.getN();

  absl::StatusOr<TPUDeviceAssignment> status_or_tpu_device_assignment;
  if (failed(GetTpuDeviceAssignment(cluster, replicate, module,
                                    status_or_tpu_device_assignment))) {
    return failure();
  }
  if (!status_or_tpu_device_assignment.ok())
    return cluster.emitError()
           << "error in fetching TPU compilation/execution devices: "
           << status_or_tpu_device_assignment.status().message();

  llvm::ArrayRef<llvm::SmallVector<tensorflow::TPUDeviceAndHost, 8>>
      tpu_devices = status_or_tpu_device_assignment.value().tpu_devices;

  std::unordered_map<std::string, Value> host_to_fill;

  for (uint32_t replica = 1; replica < num_replicas; ++replica) {
    std::string host = tpu_devices[replica][0].host;
    if (host_to_fill.find(host) == host_to_fill.end()) {
      host_to_fill[host] =
          CreateZeroInput(val_bcast.getLoc(), builder, zero_type, shape_attr);
    }
    inputs.push_back(host_to_fill[host]);
  }

  BlockArgument block_arg;
  if (failed(AppendReplicatedInput(builder, replicate, inputs,
                                   val_bcast.getType(), block_arg))) {
    return failure();
  }

  OpBuilder before_cluster_builder(cluster);
  IdentityOp assigned_id = before_cluster_builder.create<IdentityOp>(
      val_bcast.getLoc(), block_arg.getType(), block_arg);
  assigned_id->setAttr(kICIWeightDistributionMlirBridgeMarker,
                       before_cluster_builder.getBoolAttr(true));
  std::string device = tensorflow::GetDeviceAliasForHostOfLogicalCore(0);
  LaunchOp launch = tensorflow::WrapOpInLaunch(
      &before_cluster_builder, val_bcast.getLoc(), assigned_id, device);

  launch->setAttr(kICIWeightDistributionMlirBridgeMarker,
                  before_cluster_builder.getBoolAttr(true));

  Value all_reduce =
      CreateAllReduce(replicate, inner_builder, launch.getResult(0));

  orig_to_new[val_bcast] = all_reduce;
  return success();
}

// Move all suitable broadcasts across replicas to the `cluster` into the
// `cluster`.
LogicalResult MoveAllBroadcastsToCluster(ClusterOp cluster,
                                         ReplicateOp replicate,
                                         mlir::ModuleOp module) {
  // TODO(b/325153657): Fix tpu_rewrite_pass so the parallel_execute case does
  //                    not need to be skipped.
  if (cluster->getParentOfType<ParallelExecuteOp>()) return success();

  auto num_cores_per_replica_attr = cluster->getAttrOfType<mlir::IntegerAttr>(
      tensorflow::kNumCoresPerReplicaAttr);
  if (!num_cores_per_replica_attr)
    return cluster.emitOpError(
        CreateMissingAttributeMsg(tensorflow::kNumCoresPerReplicaAttr));

  llvm::SetVector<Value> bcasts;
  cluster->walk([&](Operation* op) {
    if (op == cluster) return WalkResult::advance();
    for (auto operand : op->getOperands()) {
      Operation* scope = operand.getParentBlock()->getParentOp();
      if (scope->isProperAncestor(replicate)) {
        bcasts.insert(operand);
      }
    }
    return WalkResult::advance();
  });
  OpBuilder builder(replicate);
  OpBuilder inner_builder = OpBuilder::atBlockBegin(&cluster.getBody().front());
  // mapping from original operand to new XlaAllReduce op.
  llvm::DenseMap<Value, Value> orig_to_new;

  for (Value bcast : bcasts) {
    if (failed(MoveBroadcastToCluster(builder, inner_builder, cluster,
                                      replicate, orig_to_new, bcast, module))) {
      return failure();
    }
  }

  cluster->walk([&](Operation* op) {
    if (op == cluster) return WalkResult::advance();
    for (OpOperand& operand : op->getOpOperands()) {
      if (orig_to_new.count(operand.get())) {
        operand.assign(orig_to_new[operand.get()]);
      }
    }
    return WalkResult::advance();
  });

  return success();
}

void XlaBroadcast::runOnOperation() {
  FuncOp func = getOperation();
  mlir::ModuleOp module = func->getParentOfType<mlir::ModuleOp>();
  if (!module) return signalPassFailure();
  func.walk([&](ClusterOp cluster) {
    if (auto replicate = cluster->getParentOfType<ReplicateOp>()) {
      if (failed(MoveAllBroadcastsToCluster(cluster, replicate, module))) {
        return signalPassFailure();
      }
    }
  });
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateXlaBroadcastPass() {
  return std::make_unique<XlaBroadcast>();
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
