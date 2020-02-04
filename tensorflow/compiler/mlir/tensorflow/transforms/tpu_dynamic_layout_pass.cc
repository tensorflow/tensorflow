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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/OperationSupport.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/TypeUtilities.h"  // TF:llvm-project
#include "mlir/IR/Types.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kDeviceAttr[] = "device";

// A pass that allows TPU input layout to be determined after JIT compilation.
// This is done by adding run-time ops that interpret compilation result and
// copy the input to device with that layout.
//
// Example: original program:
//
//   %input = "tf.IteratorGetNext"(...) {device = "/CPU:0"}
//   %compile:2 = "tf._TPUCompileMlir"(...)
//   %execute = "tf.TPUExecute"(%input, ..., %compile#1) {device = "/TPU:0"}
//
// Without this pass, later TF graph partitioning passes will insert send/recv
// between %input and %execute and data will be copied to device in a fixed
// layout. With this pass, the program will be transformed into:
//
//   %input = "tf.IteratorGetNext"(...) {device = "/CPU:0"}
//   %compile:2 = "tf._TPUCompileMlir"(...)
//   %get_layout = "tf.TPUGetLayoutOp"(%compile#1) {...}
//   %copy_to_device = "tf.TPUCopyWithLayout"(%input, %get_layout)
//       {device = "/TPU:0"}
//   %execute = "tf.TPUExecute"(%copy_to_device, ..., %compile#1)
//       {device = "/TPU:0"}
//
// This way, %compile will determine the layout, which will be respected by
// %copy_to_device. There will not be send/recv ops added by later passes,
// because tf.TPUCopyWithLayout accepts a host input and produces a device
// output.
struct TPUDynamicLayoutPass : public FunctionPass<TPUDynamicLayoutPass> {
  void runOnFunction() override;
};

// Checks if the input producer op is supported in this transform. Right now, we
// only check if it is a host tf.IteratorGetNext.
bool IsSupportedInputOp(Operation* op) {
  if (!llvm::isa<TF::IteratorGetNextOp>(op)) return false;
  auto device = op->getAttrOfType<StringAttr>(kDeviceAttr);
  if (!device) return false;
  tensorflow::DeviceNameUtils::ParsedName parsed_device;
  if (!tensorflow::DeviceNameUtils::ParseFullName(device.getValue().str(),
                                                  &parsed_device)) {
    return false;
  }
  return parsed_device.type == "CPU";
}

// Builds a TPUGetLayoutOp with the given compile op and input index.
TF::TPUGetLayoutOp BuildGetLayout(Operation* compile, int64_t index,
                                  OpBuilder* builder) {
  builder->setInsertionPointAfter(compile);
  return builder->create<TF::TPUGetLayoutOp>(
      compile->getLoc(),
      llvm::ArrayRef<Type>{
          RankedTensorType::get({-1}, builder->getIntegerType(64))},
      llvm::ArrayRef<Value>{compile->getResult(1)},
      llvm::ArrayRef<NamedAttribute>{
          builder->getNamedAttr("index", builder->getI64IntegerAttr(index)),
          builder->getNamedAttr("is_output", builder->getBoolAttr(false))});
}

// Builds a TPUCopyWithLayoutOp with the given get_layout op and input.
// walk_order for ops in the original IR is needed because we need to insert the
// ops after both get_layout and input, so we use the walk order to find which
// one comes later.
TF::TPUCopyWithLayoutOp BuildCopyWithLayout(
    TF::TPUExecuteOp execute, Operation* compile, TF::TPUGetLayoutOp get_layout,
    Value input, const llvm::SmallDenseMap<Operation*, int64_t>& walk_order,
    OpBuilder* builder) {
  auto input_op = input.getDefiningOp();
  int64_t compile_walk_order = walk_order.find(compile)->getSecond();
  int64_t input_walk_order = walk_order.find(input_op)->getSecond();
  if (compile_walk_order > input_walk_order) {
    builder->setInsertionPointAfter(get_layout);
  } else {
    builder->setInsertionPointAfter(input_op);
  }
  return builder->create<TF::TPUCopyWithLayoutOp>(
      execute.getLoc(), llvm::ArrayRef<Type>{input.getType()},
      llvm::ArrayRef<Value>{input, get_layout.layout()},
      llvm::ArrayRef<NamedAttribute>{});
}

// Performs transformation for a non-replicated input.
void HandleInput(Value input, int64_t index, TF::TPUExecuteOp execute,
                 Operation* compile,
                 const llvm::SmallDenseMap<Operation*, int64_t>& walk_order) {
  OpBuilder builder(compile->getContext());
  auto get_layout = BuildGetLayout(compile, index, &builder);
  auto copy_with_layout = BuildCopyWithLayout(execute, compile, get_layout,
                                              input, walk_order, &builder);
  if (auto device = execute.getAttrOfType<StringAttr>(kDeviceAttr)) {
    copy_with_layout.setAttr(kDeviceAttr, device);
  }
  execute.setOperand(index, copy_with_layout);
}

// Performs transformation for replicated inputs. Returns true if this is a
// supported case (thus transform happened).
bool HandleReplicatedInputs(
    int64_t index, TF::TPUExecuteOp execute, Operation* compile,
    int64_t replicate_arg_index, tf_device::ReplicateOp replicate,
    const llvm::SmallDenseMap<Operation*, int64_t>& walk_order) {
  // We need to know the devices to copy to.
  if (!replicate.devices()) return false;
  int64_t num_replicas = replicate.n().getZExtValue();
  auto inputs = replicate.getOperands()
                    .drop_front(replicate_arg_index * num_replicas)
                    .take_front(num_replicas);
  for (auto entry : llvm::enumerate(inputs)) {
    auto input_op = entry.value().getDefiningOp();
    if (!input_op || !IsSupportedInputOp(input_op)) return false;
  }
  OpBuilder builder(execute.getContext());
  auto get_layout = BuildGetLayout(compile, index, &builder);
  for (auto entry : llvm::enumerate(inputs)) {
    auto copy_with_layout = BuildCopyWithLayout(
        execute, compile, get_layout, entry.value(), walk_order, &builder);
    copy_with_layout.setAttr(kDeviceAttr,
                             replicate.devices()->getValue()[entry.index()]);
    replicate.setOperand(num_replicas * replicate_arg_index + entry.index(),
                         copy_with_layout);
  }
  return true;
}

// Performs transformation on a pair of execute and compile ops. The compile
// should not have other uses.
void HandleExecute(TF::TPUExecuteOp execute, Operation* compile,
                   const llvm::SmallDenseMap<Operation*, int64_t>& walk_order) {
  auto maybe_replicate = execute.getParentOfType<tf_device::ReplicateOp>();
  llvm::SmallVector<int64_t, 8> unrestricted_input_indices;
  for (auto input : llvm::enumerate(execute.args())) {
    if (auto block_arg = input.value().dyn_cast<BlockArgument>()) {
      // For a block argument, consider transforms only when it is a replicated
      // input (defining ops will be outside the replicate node).
      if (maybe_replicate != block_arg.getParentRegion()->getParentOp() ||
          !HandleReplicatedInputs(input.index(), execute, compile,
                                  block_arg.getArgNumber(), maybe_replicate,
                                  walk_order)) {
        continue;
      }
    } else {
      // For an op output, consider transforms only when 1) there is no
      // replicateion or 2) it is outside the replicate node that encloses the
      // execute node. (Because if the op is inside replicate, it is probably
      // not on the host.)
      auto input_op = input.value().getDefiningOp();
      if (maybe_replicate &&
          maybe_replicate.body().isAncestor(input_op->getParentRegion())) {
        continue;
      }
      if (!IsSupportedInputOp(input_op)) continue;
      HandleInput(input.value(), input.index(), execute, compile, walk_order);
    }
    unrestricted_input_indices.push_back(input.index());
  }
  if (unrestricted_input_indices.empty()) return;

  // Update the compilation metadata if we changed anything.
  auto metadata_attr = compile->getAttrOfType<StringAttr>("metadata");
  assert(metadata_attr && "Missing compilation metadata");
  tensorflow::tpu::TPUCompileMetadataProto metadata;
  metadata.ParseFromString(std::string(metadata_attr.getValue()));
  for (int64_t input_index : unrestricted_input_indices) {
    metadata.mutable_args(input_index)->set_unrestricted_layout(true);
  }
  compile->setAttr("metadata", OpBuilder(compile).getStringAttr(
                                   metadata.SerializeAsString()));
}

void TPUDynamicLayoutPass::runOnFunction() {
  llvm::SmallVector<std::pair<TF::TPUExecuteOp, Operation*>, 4>
      executes_and_compiles;
  llvm::SmallDenseMap<Operation*, int64_t> walk_order;
  int64_t next_walk_order = 0;
  getFunction().walk([&](Operation* op) {
    walk_order[op] = next_walk_order++;
    // Detect tf._TPUCompileMlir -> tf.TPUExecute
    auto execute = llvm::dyn_cast<TF::TPUExecuteOp>(op);
    if (!execute) return;
    auto compile = execute.key().getDefiningOp();
    if (!compile || compile->getName().getStringRef() != "tf._TPUCompileMlir" ||
        !compile->getResult(1).hasOneUse()) {
      return;
    }
    executes_and_compiles.emplace_back(execute, compile);
  });
  for (auto execute_and_compile : executes_and_compiles) {
    HandleExecute(execute_and_compile.first, execute_and_compile.second,
                  walk_order);
  }
}

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> CreateTPUDynamicLayoutPass() {
  return std::make_unique<TPUDynamicLayoutPass>();
}

static PassRegistration<TPUDynamicLayoutPass> pass(
    "tf-tpu-dynamic-layout-pass",
    "Adds ops that allow TPU program inputs to have layouts determined at JIT "
    "compile time.");

}  // namespace TFTPU
}  // namespace mlir
