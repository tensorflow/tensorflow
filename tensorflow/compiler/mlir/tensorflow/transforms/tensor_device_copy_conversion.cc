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

// This pass folds the tf.Identity op if the operation has the same device as
// its operand.

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/PassOptions.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"

namespace mlir {
namespace TF {
namespace {

constexpr const char *kDeviceAttr = "device";
constexpr const char *kTFDeviceAttr = "tf.device";

class TensorDeviceCopyConversionPass
    : public PassWrapper<TensorDeviceCopyConversionPass, FunctionPass> {
 public:
  void runOnFunction() override {
    FuncOp func_op = getFunction();
    StringAttr empty_string = StringAttr::get(func_op.getContext(), "");
    func_op.walk([&](TF::IdentityOp op) {
      StringAttr arg_device = empty_string;
      mlir::Value arg = op.getOperand();
      if (BlockArgument block_arg = arg.dyn_cast<BlockArgument>()) {
        // Skip the folding logic if the block argument is not from the function
        // arguments. This can happen when the argument is from a while loop.
        if (block_arg.getParentRegion() != &func_op.getRegion()) {
          return WalkResult::advance();
        }
        if (StringAttr attr = func_op.getArgAttrOfType<StringAttr>(
                block_arg.getArgNumber(), kTFDeviceAttr)) {
          arg_device = attr;
        }
      } else if (StringAttr attr =
                     arg.getDefiningOp()->getAttrOfType<StringAttr>(
                         kDeviceAttr)) {
        arg_device = attr;
      }

      StringAttr op_device = op->getAttrOfType<StringAttr>(kDeviceAttr);
      if (!op_device) op_device = empty_string;
      // Skip the folding logic if the argument's device is different from the
      // operation's device.
      if (op_device != arg_device) return WalkResult::advance();

      op.replaceAllUsesWith(op.getOperand());
      op.erase();
      return WalkResult::advance();
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::FuncOp>>
CreateTensorDeviceCopyConversionPass() {
  return std::make_unique<TensorDeviceCopyConversionPass>();
}

static mlir::PassRegistration<TensorDeviceCopyConversionPass>
    tensor_device_copy_pass(
        "tf-tensor-device-copy",
        "Fold the tf.Identity op if the op has the same device as its operand");

}  // namespace TF
}  // namespace mlir
