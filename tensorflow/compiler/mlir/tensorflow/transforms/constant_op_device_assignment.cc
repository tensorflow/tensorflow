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

// This pass adds the device attribute to every tf.Const op based on the device
// attribute of the operations that read its result. If the result of a tf.Const
// op is read by operations placed on multiple devices, then the pass will
// replicate the tf.Const op once for each device.

#include <memory>

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {
namespace {

constexpr const char *kDeviceAttr = "device";

#define GEN_PASS_DEF_CONSTANTOPDEVICEASSIGNMENTPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct ConstantOpDeviceAssignmentPass
    : public impl::ConstantOpDeviceAssignmentPassBase<
          ConstantOpDeviceAssignmentPass> {
  void runOnOperation() override;
};

void ConstantOpDeviceAssignmentPass::runOnOperation() {
  ModuleOp module = getOperation();

  module.walk([&](TF::ConstOp op) {
    // Keep the ConstOp if the op already have the device attribute.
    if (StringAttr device_attr = op->getAttrOfType<StringAttr>(kDeviceAttr)) {
      return WalkResult::advance();
    }
    OpBuilder builder(op);
    llvm::StringMap<mlir::Operation *> cloned_op_by_device;
    bool all_uses_replaced = true;

    for (mlir::OpOperand &use :
         llvm::make_early_inc_range(op.getResult().getUses())) {
      mlir::Operation *user_op = use.getOwner();
      StringAttr device_attr = user_op->getAttrOfType<StringAttr>(kDeviceAttr);
      if (!device_attr) {
        all_uses_replaced = false;
        continue;
      }
      // Cloned the ConstOp and set its device attribute to be the same as the
      // device of the user operation.
      if (cloned_op_by_device.find(device_attr.getValue()) ==
          cloned_op_by_device.end()) {
        mlir::Operation *new_op = builder.clone(*op.getOperation());
        new_op->setAttr(kDeviceAttr, device_attr);
        cloned_op_by_device[device_attr.getValue()] = new_op;
      }
      // Update the user operation to use the result of the cloned ConstOp.
      mlir::Operation *new_op = cloned_op_by_device[device_attr.getValue()];
      user_op->setOperand(use.getOperandNumber(), new_op->getResult(0));
    }
    // Erase the original ConstOp if all its uses have been replaced.
    if (all_uses_replaced) {
      op.erase();
    }
    return WalkResult::advance();
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateConstantOpDeviceAssignmentPass() {
  return std::make_unique<ConstantOpDeviceAssignmentPass>();
}

}  // namespace TF
}  // namespace mlir
