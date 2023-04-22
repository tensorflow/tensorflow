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

#include <tuple>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kFuncDeviceAttr[] = "tf.device";

// Checks if a function only contains a tf_executor.graph.
bool IsSupportedGraph(FuncOp func) {
  if (!llvm::hasSingleElement(func)) return false;

  Block& block = func.front();
  if (!llvm::hasSingleElement(block.without_terminator())) return false;

  auto graph = llvm::dyn_cast<tf_executor::GraphOp>(block.front());
  if (!graph) return false;

  Operation* terminator = block.getTerminator();
  if (graph.getNumResults() != terminator->getNumOperands()) return false;
  for (auto result : llvm::zip(graph.results(), terminator->getOperands()))
    if (std::get<0>(result) != std::get<1>(result)) return false;

  return true;
}

// Checks if an operation of the tf_executor dialect can have TPU devices
// propagated through.
bool IsSupportedExecutorOp(Operation& op) {
  auto ops_have_same_device = [](Operation* lhs, Operation* rhs) {
    auto lhs_device_attr = lhs->getAttrOfType<StringAttr>(kDeviceAttr);
    auto rhs_device_attr = rhs->getAttrOfType<StringAttr>(kDeviceAttr);
    return (!lhs_device_attr && !rhs_device_attr) ||
           (lhs_device_attr && rhs_device_attr &&
            lhs_device_attr.getValue() == rhs_device_attr.getValue());
  };

  // Check if tf_executor.NextIteration.Source/tf_executor.NextIteration.Sink
  // pair has matching devices or no devices.
  if (auto source = llvm::dyn_cast<tf_executor::NextIterationSourceOp>(op)) {
    return ops_have_same_device(source, source.GetSink());
  } else if (auto sink = llvm::dyn_cast<tf_executor::NextIterationSinkOp>(op)) {
    return ops_have_same_device(sink.GetSource(), sink);
  }

  return llvm::isa<tf_executor::EnterOp, tf_executor::ExitOp,
                   tf_executor::IslandOp, tf_executor::MergeOp,
                   tf_executor::SwitchOp>(op);
}

// Assigns all data results to a specified device.
void PopulateDeviceForOpResults(
    Operation& op, llvm::StringRef device,
    llvm::DenseMap<Value, llvm::StringRef>& value_to_device) {
  Operation* op_to_update = &op;
  // Use tf_executor.island op if present as non v1 control flow op results are
  // forwarded by a parent tf_executor.island op.
  if (llvm::isa<tf_executor::IslandOp>(op_to_update->getParentOp()))
    op_to_update = op_to_update->getParentOp();

  for (Value result : op_to_update->getResults()) {
    if (result.getType().isa<tf_executor::TokenType>()) continue;
    if (result.getType().isa<tf_executor::ControlType>()) break;

    value_to_device.insert({result, device});
  }
}

// Checks if an operation can have TPU devices propagated through.
bool IsSupportedOpToSetDevice(Operation& op) {
  return IsSupportedExecutorOp(op) ||
         isa<TF::IdentityOp, TF::IdentityNOp, TF::ShapeOp>(op);
}

// Finds nonconflicting TPU device for an operation from its operands. If an
// operand has no device or a non TPU device, or if there are conflicting
// devices, and empty StringRef will be returned. Control dependencies,
// NextIteration.Source -> NextIteration.Sink token dependencies, and
// LoopCond -> Switch data dependencies are ignored.
llvm::StringRef FindDeviceFromOperands(
    Operation& op,
    const llvm::DenseMap<Value, llvm::StringRef>& value_to_device) {
  llvm::StringRef new_device;
  const bool is_switch = llvm::isa<tf_executor::SwitchOp>(op);
  for (Value operand : op.getOperands()) {
    if (operand.getType().isa<tf_executor::TokenType>()) continue;
    if (operand.getType().isa<tf_executor::ControlType>()) break;

    if (is_switch &&
        llvm::isa_and_nonnull<tf_executor::LoopCondOp>(operand.getDefiningOp()))
      continue;

    auto it = value_to_device.find(operand);
    if (it == value_to_device.end()) return llvm::StringRef();

    if (new_device.empty()) {
      new_device = it->getSecond();
      continue;
    }

    if (new_device != it->getSecond()) return llvm::StringRef();
  }

  return new_device;
}

// Propagates devices from function arguments.
void PropagateDevicesFromArguments(
    FuncOp func, llvm::DenseMap<Value, llvm::StringRef>& value_to_device) {
  for (BlockArgument& arg : func.getArguments()) {
    auto arg_device_attr =
        func.getArgAttrOfType<StringAttr>(arg.getArgNumber(), kFuncDeviceAttr);
    if (!arg_device_attr || arg_device_attr.getValue().empty() ||
        !tensorflow::IsTPUDevice(arg_device_attr.getValue()))
      continue;
    value_to_device.insert({arg, arg_device_attr.getValue()});
  }
}

// Propagates devices from operation operands to results. Updating the device of
// a tf_executor.NextIteration.Source/tf_executor.NextIteration.Sink will result
// in multiple passes over the tf_executor.graph to propagate devices in loops.
void PropagateDevicesInGraph(
    tf_executor::GraphOp graph,
    llvm::DenseMap<Value, llvm::StringRef>& value_to_device) {
  auto ops = graph.GetBody().without_terminator();

  bool updated_next_iteration = false;
  do {
    updated_next_iteration = false;
    for (Operation& op : ops) {
      if (!IsSupportedExecutorOp(op)) continue;

      Operation* op_to_update = &op;
      // Unpack inner op of tf_executor.island.
      if (auto island_op =
              llvm::dyn_cast<tf_executor::IslandOp>(op_to_update)) {
        if (!island_op.WrapsSingleOp()) continue;
        op_to_update = &island_op.GetBody().front();
      }

      // If op already has a TPU device set, simply propagate its device.
      auto device_attr = op_to_update->getAttrOfType<StringAttr>(kDeviceAttr);
      const bool has_device = device_attr && !device_attr.getValue().empty();
      if (has_device && tensorflow::IsTPUDevice(device_attr.getValue())) {
        PopulateDeviceForOpResults(*op_to_update, device_attr.getValue(),
                                   value_to_device);
        continue;
      }

      // Op has an unsupported device.
      if (has_device) continue;

      if (!IsSupportedOpToSetDevice(*op_to_update)) continue;

      llvm::StringRef new_device =
          FindDeviceFromOperands(*op_to_update, value_to_device);
      if (new_device.empty()) continue;

      auto new_device_attr =
          mlir::StringAttr::get(op_to_update->getContext(), new_device);
      op_to_update->setAttr(kDeviceAttr, new_device_attr);
      PopulateDeviceForOpResults(*op_to_update, new_device_attr.getValue(),
                                 value_to_device);

      if (auto sink =
              llvm::dyn_cast<tf_executor::NextIterationSinkOp>(op_to_update)) {
        auto source = sink.GetSource();
        source->setAttr(kDeviceAttr, new_device_attr);
        PopulateDeviceForOpResults(*source, new_device_attr.getValue(),
                                   value_to_device);
        updated_next_iteration = true;
      }
    }
  } while (updated_next_iteration);
}

// Propagates devices to function results.
void PropagateDevicesToResults(
    FuncOp func, tf_executor::FetchOp fetch,
    const llvm::DenseMap<Value, llvm::StringRef>& value_to_device) {
  for (OpOperand& operand : fetch.getOperation()->getOpOperands()) {
    if (operand.get().getType().isa<tf_executor::ControlType>()) break;
    auto it = value_to_device.find(operand.get());
    if (it != value_to_device.end()) {
      auto device_attr = func.getResultAttrOfType<StringAttr>(
          operand.getOperandNumber(), kFuncDeviceAttr);
      if (device_attr && !device_attr.getValue().empty()) continue;
      func.setResultAttr(operand.getOperandNumber(), kFuncDeviceAttr,
                         StringAttr::get(func.getContext(), it->getSecond()));
    }
  }
}

struct TPUDevicePropagation
    : public PassWrapper<TPUDevicePropagation, FunctionPass> {
  void runOnFunction() override;
  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tf-tpu-device-propagation";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Propagates TPU devices from ops to users";
  }
};

void TPUDevicePropagation::runOnFunction() {
  FuncOp func = getFunction();
  if (!IsSupportedGraph(func)) return;

  llvm::DenseMap<Value, llvm::StringRef> value_to_device;
  PropagateDevicesFromArguments(func, value_to_device);
  auto graph = llvm::cast<tf_executor::GraphOp>(func.front().front());
  PropagateDevicesInGraph(graph, value_to_device);
  PropagateDevicesToResults(func, graph.GetFetch(), value_to_device);
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateTPUDevicePropagationPass() {
  return std::make_unique<TPUDevicePropagation>();
}

static PassRegistration<TPUDevicePropagation> pass;

}  // namespace TFTPU
}  // namespace mlir
