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

// This pass hoists replicate invariant ops, or ops that yield the same
// result(s) regardless of replication, out of their respective replicate.

#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/IR/Visitors.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "mlir/Transforms/RegionUtils.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFDevice {

namespace {
struct ReplicateInvariantOpHoistingPass
    : public FunctionPass<ReplicateInvariantOpHoistingPass> {
  void runOnFunction() override;
};

// Make ShapeOp replicate invariant if it is possible. This currently updates or
// replace ShapeOps of replicated arguments, either tensors or resources.
//
// For example, the following:
//
// tf_device.replicate([%0, %1] as %ri: tensor<*xi32>) {n = 2 : i32} {
//   %2 = "tf.Shape"(%ri) : (tensor<*xi32>) -> tensor<?xi32>
//   tf_device.return
// }
//
// gets converted to:
//
// tf_device.replicate([%0, %1] as %ri: tensor<*xi32>) {n = 2 : i32} {
//   %2 = "tf.Shape"(%0) : (tensor<*xi32>) -> tensor<?xi32>
//   tf_device.return
// }
//
// and for resource variables:
//
// tf_device.replicate([%0, %1] as %ri: tensor<*x!tf.resource>) {n = 2 : i32} {
//   %2 = "tf.ReadVariableOp"(%ri) : tensor<*x!tf.resource> -> tensor<*xi32>
//   %3 = "tf.Shape"(%2) : (tensor<*xi32>) -> tensor<?xi32>
//   tf_device.return
// }
//
// gets converted to:
//
// tf_device.replicate([%0, %1] as %ri: tensor<*x!tf.resource>) {n = 2 : i32} {
//   %2 = "tf.ReadVariableOp"(%ri) : tensor<*x!tf.resource> -> tensor<*xi32>
//   %3 = "tf.VariableShape"(%0) : (tensor<*x!tf.resource>) -> tensor<?xi32>
//   tf_device.return
// }
void MakeShapeOpInvariant(tf_device::ReplicateOp replicate_op, int num_replicas,
                          Block* replicate_block, TF::ShapeOp shape_op) {
  Value input = shape_op.input();
  // If ShapeOp operand is replicate tensor block argument, replace with the
  // associated first replica operand.
  if (auto block_arg = input.dyn_cast<BlockArgument>()) {
    if (block_arg.getOwner() != replicate_block) return;

    shape_op.setOperand(
        replicate_op.getOperand(num_replicas * block_arg.getArgNumber()));

    return;
  }

  Operation* input_def = input.getDefiningOp();

  // If ShapeOp operand is a ReadVariableOp result where the ReadVariableOp
  // operand is a replicate resource block argument, replace ShapeOp with
  // VariableShapeOp and use the associated first replica operand as its
  // operand.
  auto read_var_op = llvm::dyn_cast<TF::ReadVariableOp>(input_def);
  if (!read_var_op) return;

  // TODO(lyandy): Check if resource (first replica or replicate block arg)
  // shape has not changed in replicate prior to read. Currently after both
  // ResourceOpLiftingPass and TPURewritePass, there should not be any updates
  // to resources prior to their respective ReadVariableOp.
  if (auto block_arg = read_var_op.resource().dyn_cast<BlockArgument>()) {
    if (block_arg.getOwner() != replicate_block) return;

    OpBuilder builder(shape_op);
    auto new_shape_op = builder.create<TF::VariableShapeOp>(
        shape_op.getLoc(), shape_op.getType(),
        replicate_op.getOperand(num_replicas * block_arg.getArgNumber()));
    shape_op.replaceAllUsesWith(new_shape_op.getOperation());
    shape_op.erase();
  }
}

// Checks if op and inner op operands are all replicate invariant.
bool IsOpReplicateInvariant(Region* replicate_region, Operation* op) {
  auto ancestor_of_replicate = [&](Region* region) {
    return region && region->isProperAncestor(replicate_region);
  };

  for (Value operand : op->getOperands())
    if (!ancestor_of_replicate(operand.getParentRegion())) return false;

  bool has_replicate_operands = false;
  visitUsedValuesDefinedAbove(op->getRegions(), [&](OpOperand* operand) {
    if (!ancestor_of_replicate(operand->get().getParentRegion()))
      has_replicate_operands = true;
  });

  return !has_replicate_operands;
}

// Hoists replicate invariant ops out of associated `tf_device.replicate` op.
// Ops to be hoisted are determined by if all of their operands are replicate
// invariant. Shape ops are rewritten to be invariant when possible, prior to
// hoisting ops.
void HoistReplicateInvariantOps(tf_device::ReplicateOp replicate_op) {
  const int num_replicas = replicate_op.n().getLimitedValue();
  Block* replicate_block = &replicate_op.GetBody();

  replicate_op.walk([&](TF::ShapeOp shape_op) {
    MakeShapeOpInvariant(replicate_op, num_replicas, replicate_block, shape_op);
  });

  Region* replicate_region = &replicate_op.body();
  for (Operation& inner_op :
       llvm::make_early_inc_range(replicate_op.GetBody())) {
    if (llvm::isa<tf_device::ReturnOp>(inner_op)) continue;

    if (IsOpReplicateInvariant(replicate_region, &inner_op))
      inner_op.moveBefore(replicate_op);
  }
}

void ReplicateInvariantOpHoistingPass::runOnFunction() {
  getFunction().walk(
      [](tf_device::ReplicateOp op) { HoistReplicateInvariantOps(op); });
}
}  // anonymous namespace

std::unique_ptr<OpPassBase<FuncOp>> CreateReplicateInvariantOpHoistingPass() {
  return std::make_unique<ReplicateInvariantOpHoistingPass>();
}

static PassRegistration<ReplicateInvariantOpHoistingPass> pass(
    "tf-replicate-invariant-op-hoisting",
    "Hoists replicate invariant operations out of replicate");

}  // namespace TFDevice
}  // namespace mlir
