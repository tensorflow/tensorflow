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

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#define DEBUG_TYPE "tf-extract-tpu-copy-with-dynamic-shape-op"

namespace mlir {
namespace TFTPU {

namespace {

#define GEN_PASS_DEF_EXTRACTTPUCOPYWITHDYNAMICSHAPEOPPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

class ExtractTPUCopyWithDynamicShapeOpPass
    : public impl::ExtractTPUCopyWithDynamicShapeOpPassBase<
          ExtractTPUCopyWithDynamicShapeOpPass> {
  void runOnOperation() override;
};

// Finds op that created a given value. If the value is a BlockArgument, this
// returns the owner of the Block.
Operation* GetOpOfValue(Value value) {
  if (auto block_arg = value.dyn_cast<BlockArgument>())
    return block_arg.getOwner()->getParentOp();

  return value.getDefiningOp();
}

// Check if the TPUCopyWithDynamicShapeOp is valid.
// 1. The op should be wrapped inside a launch op.
// 2. The wrapped launch op should be placed on CPU.
LogicalResult CheckOpIsValid(Operation* op) {
  auto launch_op = llvm::dyn_cast<tf_device::LaunchOp>(op->getParentOp());
  if (!launch_op) {
    op->emitError() << "TPUCopyWithDynamicShapeOp is not in a launch";
  }
  std::string device_str = launch_op.getDeviceAttr().getValue().str();
  std::string cpu0_device;
  if (failed(tensorflow::GetNonReplicatedCPU0(op, &cpu0_device)))
    return failure();
  if (device_str != tensorflow::GetDeviceAliasForHostOfLogicalCore(0) &&
      device_str != cpu0_device) {
    op->emitError()
        << "TPUCopyWithDynamicShapeOp's device is not a recognized host 0: "
        << device_str;
    return failure();
  }
  return success();
}

// Check if we can move TPUCopyWithDynamicShapeOp out of a launch. This is the
// case if its results aren't used by other ops except for the return op.
bool CanMove(Operation* op) {
  auto launch_op = llvm::dyn_cast<tf_device::LaunchOp>(op->getParentOp());
  if (!launch_op) return false;
  for (Value result : op->getResults()) {
    for (Operation* user : result.getUsers()) {
      if (user != launch_op.GetBody().getTerminator()) return false;
    }
  }
  return true;
}

// Get the new launch op results. This is the results if the copy op is removed
// from the old launch op.
llvm::SmallVector<Value, 4> CreateNewLaunchOpResults(
    tf_device::LaunchOp* old_launch_op,
    Operation* tpu_copy_with_dynamic_shape_op) {
  llvm::SmallSetVector<Value, 4> new_launch_op_results;

  new_launch_op_results.insert(
      old_launch_op->GetBody().getTerminator()->getOperands().begin(),
      old_launch_op->GetBody().getTerminator()->getOperands().end());

  for (Value operand : tpu_copy_with_dynamic_shape_op->getOperands()) {
    if (GetOpOfValue(operand)->getParentRegion() ==
        tpu_copy_with_dynamic_shape_op->getParentRegion()) {
      new_launch_op_results.insert(operand);
    }
  }

  for (Value result : tpu_copy_with_dynamic_shape_op->getResults()) {
    new_launch_op_results.remove(result);
  }

  return new_launch_op_results.takeVector();
}

// Create a new host launch op which contains all the old launch op body
// except the dynamic shape copy op.
tf_device::LaunchOp CreateNewHostLaunchOpWithNewResult(
    tf_device::LaunchOp* old_launch_op,
    llvm::SmallVector<Value, 4>& new_launch_op_results) {
  OpBuilder builder(*old_launch_op);

  builder.setInsertionPointAfter(*old_launch_op);

  llvm::SmallVector<Type, 4> new_launch_op_results_types;
  for (Value result : new_launch_op_results)
    new_launch_op_results_types.push_back(result.getType());

  auto new_launch_op = builder.create<tf_device::LaunchOp>(
      old_launch_op->getLoc(), old_launch_op->getDeviceAttr(),
      /*result_types=*/new_launch_op_results_types);

  new_launch_op.getBody().takeBody(old_launch_op->getBody());
  new_launch_op.GetBody().getTerminator()->setOperands(new_launch_op_results);

  return new_launch_op;
}

// Create the new device launch op which wraps the copy op.
LogicalResult CreateNewDeviceLaunchOp(
    Operation* tpu_copy_with_dynamic_shape_op, bool replicated,
    tf_device::LaunchOp& new_device_launch_op) {
  OpBuilder builder(tpu_copy_with_dynamic_shape_op);

  builder.setInsertionPointAfter(tpu_copy_with_dynamic_shape_op);

  // Set the copy op's device to the first TPU.
  std::string device_str;
  if (replicated) {
    device_str = tensorflow::GetDeviceAliasForLogicalCore(0);
  } else if (failed(tensorflow::GetNonReplicatedTPU0(
                 tpu_copy_with_dynamic_shape_op, &device_str))) {
    return failure();
  }

  new_device_launch_op = builder.create<tf_device::LaunchOp>(
      tpu_copy_with_dynamic_shape_op->getLoc(),
      builder.getStringAttr(device_str),
      /*result_types=*/tpu_copy_with_dynamic_shape_op->getResultTypes());

  new_device_launch_op.getBody().push_back(new Block);
  builder.setInsertionPointToEnd(&new_device_launch_op.GetBody());
  auto* return_op = builder
                        .create<tf_device::ReturnOp>(
                            tpu_copy_with_dynamic_shape_op->getLoc(),
                            tpu_copy_with_dynamic_shape_op->getResults())
                        .getOperation();
  tpu_copy_with_dynamic_shape_op->moveBefore(return_op);
  return success();
}

// Update all the usage of tf_device.return op with launch op result.
void UpdateReturnOpResultWithLaunchOpResult(tf_device::LaunchOp* launch_op) {
  auto operand_not_in_launch = [&](OpOperand& operand) {
    return !launch_op->getOperation()->isProperAncestor(operand.getOwner());
  };

  for (auto result :
       llvm::zip(launch_op->getResults(),
                 launch_op->GetBody().getTerminator()->getOperands()))
    std::get<1>(result).replaceUsesWithIf(std::get<0>(result),
                                          operand_not_in_launch);
}

void ExtractTPUCopyWithDynamicShapeOpPass::runOnOperation() {
  llvm::SmallVector<Operation*, 4> tpu_copy_with_dynamic_shape_ops;
  auto walk_result = getOperation().walk([&](Operation* op) {
    if (isa<TF::TPUCopyWithDynamicShapeOp>(op)) {
      if (failed(CheckOpIsValid(op))) return WalkResult::interrupt();
      if (CanMove(op)) {
        tpu_copy_with_dynamic_shape_ops.push_back(op);
      }
    }
    return WalkResult::advance();
  });
  if (walk_result.wasInterrupted()) return signalPassFailure();

  for (Operation* op : tpu_copy_with_dynamic_shape_ops) {
    OpBuilder builder(op);

    auto old_launch_op = llvm::dyn_cast<tf_device::LaunchOp>(op->getParentOp());

    bool replicated = old_launch_op.getDeviceAttr().getValue().str() ==
                      tensorflow::GetDeviceAliasForHostOfLogicalCore(0);

    for (auto result :
         llvm::zip(old_launch_op->getResults(),
                   old_launch_op.GetBody().getTerminator()->getOperands()))
      std::get<0>(result).replaceAllUsesWith(std::get<1>(result));

    llvm::SmallVector<Value, 4> new_launch_op_results =
        CreateNewLaunchOpResults(&old_launch_op, op);

    op->moveAfter(old_launch_op);

    auto new_host_launch_op = CreateNewHostLaunchOpWithNewResult(
        &old_launch_op, new_launch_op_results);
    UpdateReturnOpResultWithLaunchOpResult(&new_host_launch_op);

    old_launch_op->erase();

    tf_device::LaunchOp new_device_launch_op;
    if (failed(CreateNewDeviceLaunchOp(op, replicated, new_device_launch_op)))
      return signalPassFailure();
    UpdateReturnOpResultWithLaunchOpResult(&new_device_launch_op);
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateExtractTPUCopyWithDynamicShapeOpPass() {
  return std::make_unique<ExtractTPUCopyWithDynamicShapeOpPass>();
}
}  // namespace TFTPU
}  // namespace mlir
