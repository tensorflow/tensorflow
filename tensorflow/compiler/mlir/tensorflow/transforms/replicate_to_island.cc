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

// This pass forms `tf_executor.island` per replica from a single
// `tf_device.replicate` island.

#include <memory>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/core/platform/logging.h"

namespace mlir {
namespace TFDevice {
namespace {
constexpr char kDeviceAttr[] = "device";

struct ReplicateToIslandPass
    : public PassWrapper<ReplicateToIslandPass, FunctionPass> {
  void runOnFunction() override;
};

// Creates islands per replica from `tf_device.replicate` region. If for a
// `tf_device.launch` op the device is an aliased device of the
// `tf_device.replicate`, the device will be remapped to an explicit device
// for the associated replica island.
llvm::SmallVector<tf_executor::IslandOp, 8> ExpandReplicateIntoReplicas(
    const Dialect* tf_dialect, OpBuilder* builder,
    tf_executor::IslandOp island_op, tf_device::ReplicateOp replicate_op,
    int num_replicas) {
  auto devices = replicate_op.devices();
  const bool has_devices = devices.hasValue();
  llvm::SmallVector<tf_executor::IslandOp, 8> replicas;
  replicas.reserve(num_replicas);

  // Collect result types and operands.
  Operation& terminator = replicate_op.GetBody().back();
  llvm::SmallVector<Type, 8> output_types(terminator.getOperandTypes());
  auto control_type = tf_executor::ControlType::get(island_op.getContext());
  llvm::SmallVector<Value, 8> replica_inputs(island_op.controlInputs());

  // Replace replicate terminator with YieldOp.
  builder->setInsertionPoint(&terminator);
  builder->create<tf_executor::YieldOp>(terminator.getLoc(),
                                        terminator.getOperands());
  terminator.erase();

  builder->setInsertionPoint(island_op);
  BlockAndValueMapping mapping;
  for (int i : llvm::seq<int>(0, num_replicas)) {
    // Create new island for replica.
    auto replica = builder->create<tf_executor::IslandOp>(
        island_op.getLoc(), output_types, control_type, replica_inputs);

    // Map block arg to replica arg.
    mapping.clear();
    for (auto& block_arg : replicate_op.GetBody().getArguments())
      mapping.map(block_arg, replicate_op.getOperand(
                                 block_arg.getArgNumber() * num_replicas + i));

    // Copy over replicate region into replica island.
    replicate_op.body().cloneInto(&replica.body(), mapping);

    // Map aliased devices to explicit devices based on replica.
    if (has_devices) {
      replica.walk([&](tf_device::LaunchOp launch) {
        if (auto device_by_replica = devices.getValue().get(launch.device()))
          launch.setAttr(
              kDeviceAttr,
              device_by_replica.cast<ArrayAttr>()[i].cast<StringAttr>());
      });
    }

    replicas.push_back(replica);
  }

  return replicas;
}

// Creates islands per replica from `tf_device.replicate` region and remap
// replicate results with new island outputs. A single island is created to
// forward control dependencies if there is a control dependency output from the
// replicate island. Devices are remapped from aliased devices to explicit
// devices, for `tf_device.launch` ops.
//
// For example, the following:
//
// %0:2 = tf_executor.island(%control) {
//   %1:4 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<i1>)
//              {n = 2 : i32,
//               devices = {DEVICE_ALIAS_0 = ["/DEVICE:0", "/DEVICE:1"],
//                          DEVICE_ALIAS_1 = ["/DEVICE:2", "/DEVICE:3"]}} {
//     %a = "tf_device.launch"() ( {
//       %2 = "tf.opA"(%ri) : (tensor<i1>) -> tensor<i1>
//       tf_device.return %2 : tensor<i1>
//     }) {device = "DEVICE_ALIAS_0"} : () -> tensor<i1>
//     %b = "tf_device.launch"() ( {
//       %3 = "tf.opB"(%a) : (tensor<i1>) -> tensor<i1>
//       tf_device.return %3 : tensor<i1>
//     }) {device = "DEVICE_ALIAS_1"} : () -> tensor<i1>
//     tf_device.return %a, %b : tensor<i1>, tensor<i1>
//   }
//   tf_executor.yield %1#0 : tensor<i1>
// }
//
// gets lowered to:
//
// %0:3 = tf_executor.island(%control) {
//   %a0 = "tf_device.launch"() ( {
//     %1 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
//     tf_device.return %1 : tensor<i1>
//   }) {device = "/DEVICE:0"} : () -> tensor<i1>
//   %b0 = "tf_device.launch"() ( {
//     %2 = "tf.opB"(%a0) : (tensor<i1>) -> tensor<i1>
//     tf_device.return %2 : tensor<i1>
//   }) {device = "/DEVICE:2"} : () -> tensor<i1>
//   tf_executor.yield %a0, %b0 : tensor<i1>, tensor<i1>
// }
// %3:3 = tf_executor.island(%control) {
//   %a1 = "tf_device.launch"() ( {
//     %4 = "tf.opA"(%arg1) : (tensor<i1>) -> tensor<i1>
//     tf_device.return %4 : tensor<i1>
//   }) {device = "/DEVICE:1"} : () -> tensor<i1>
//   %b1 = "tf_device.launch"() ( {
//     %5 = "tf.opB"(%a1) : (tensor<i1>) -> tensor<i1>
//     tf_device.return %5 : tensor<i1>
//   }) {device = "/DEVICE:3"} : () -> tensor<i1>
//   tf_executor.yield %a1, %b1 : tensor<i1>, tensor<i1>
// }
LogicalResult CreateIslandsFromReplicate(const Dialect* tf_dialect,
                                         tf_executor::IslandOp island_op,
                                         tf_device::ReplicateOp replicate_op) {
  OpBuilder builder(island_op);
  const int num_replicas = replicate_op.n().getLimitedValue();

  // Create islands per replica.
  llvm::SmallVector<tf_executor::IslandOp, 8> replicas =
      ExpandReplicateIntoReplicas(tf_dialect, &builder, island_op, replicate_op,
                                  num_replicas);

  // Collect all replica results.
  llvm::SmallVector<Value, 8> replicas_outputs(replicate_op.getNumResults(),
                                               nullptr);
  for (auto replica_and_idx : llvm::enumerate(replicas))
    for (auto replica_result_and_idx :
         llvm::enumerate(replica_and_idx.value().outputs()))
      replicas_outputs[num_replicas * replica_result_and_idx.index() +
                       replica_and_idx.index()] =
          replica_result_and_idx.value();

  // Remap replicate results to per replica result.
  for (auto result : llvm::zip(island_op.outputs(), replicas_outputs))
    std::get<0>(result).replaceAllUsesWith(std::get<1>(result));

  // Add sink island to pin all replicas as a control dependency if there is a
  // control dependency leading from the replicate originally.
  if (!island_op.control().use_empty()) {
    llvm::SmallVector<Value, 8> island_operands;
    for (auto& replica : replicas) island_operands.push_back(replica.control());

    builder.setInsertionPoint(island_op);
    auto island_sink = builder.create<tf_executor::IslandOp>(
        island_op.getLoc(), llvm::ArrayRef<Type>{},
        tf_executor::ControlType::get(island_op.getContext()), island_operands);
    island_sink.body().push_back(new Block);
    builder.setInsertionPointToEnd(&island_sink.GetBody());
    builder.create<tf_executor::YieldOp>(island_op.getLoc(),
                                         llvm::ArrayRef<Value>{});
    island_op.control().replaceAllUsesWith(island_sink.control());
  }

  island_op.erase();
  return success();
}

// Finds islands with a single `tf_device.replicate` and create individual
// islands per replica of the replicate.
LogicalResult LowerSingleIslandReplicateToIslands(
    const Dialect* tf_dialect, tf_executor::IslandOp island_op) {
  if (!hasSingleElement(island_op.GetBody().without_terminator()))
    return success();

  if (auto replicate_op =
          llvm::dyn_cast<tf_device::ReplicateOp>(&island_op.GetBody().front()))
    return CreateIslandsFromReplicate(tf_dialect, island_op, replicate_op);

  return success();
}

void ReplicateToIslandPass::runOnFunction() {
  const Dialect* tf_dialect = getContext().getRegisteredDialect("tf");
  if (!tf_dialect) {
    signalPassFailure();
    getFunction().emitError() << "'tf' dialect is not registered";
  }

  auto result = getFunction().walk([&](tf_executor::IslandOp island_op) {
    if (failed(LowerSingleIslandReplicateToIslands(tf_dialect, island_op)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) return signalPassFailure();
}
}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> CreateReplicateToIslandPass() {
  return std::make_unique<ReplicateToIslandPass>();
}

static PassRegistration<ReplicateToIslandPass> pass(
    "tf-replicate-to-island", "Lowers device replicate to executor islands");

}  // namespace TFDevice
}  // namespace mlir
