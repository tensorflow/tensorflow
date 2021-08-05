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
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/PassOptions.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"

namespace mlir {
namespace TF {

namespace {

constexpr const char *kDeviceAttr = "device";
constexpr const char *kTFDeviceAttr = "tf.device";

struct TensorDeviceCopyConversionPass
    : public TensorDeviceCopyConversionPassBase<
          TensorDeviceCopyConversionPass> {
  void runOnFunction() override;
};

// Folds tf.IdentityOp and tf.IdentityNOp if op device and the argument devices
// from the defining ops match.
void TensorDeviceCopyConversionPass::runOnFunction() {
  FuncOp func_op = getFunction();

  auto should_fold_op_func = [&func_op](const Value &arg,
                                        const StringAttr &op_device) {
    // In TFRT TPU, tensor transfer is handled specifically by D2H and
    // H2D transfer kernels. So fold the tf.Identity op if:
    // * the identity op is placed on TPU, and
    // * the arg to the identity op is produced by a TPUExecuteOp.
    if (op_device && op_device.getValue().contains("TPU")) {
      return true;
    }

    Operation *def_op = arg.getDefiningOp();
    // If the arg to this identity op is the arg of a function, there's no
    // defining op.
    if (def_op != nullptr &&
        (isa<TF::TPUExecuteOp, TF::TPUExecuteAndUpdateVariablesOp>(def_op))) {
      return true;
    }
    if (BlockArgument block_arg = arg.dyn_cast<BlockArgument>()) {
      // Skip the folding logic if the block argument is not from the function
      // arguments. This can happen when the argument is from a while loop.
      if (block_arg.getParentRegion() != &func_op.getRegion()) {
        return false;
      }
      if (StringAttr attr = func_op.getArgAttrOfType<StringAttr>(
              block_arg.getArgNumber(), kTFDeviceAttr)) {
        return op_device == attr;
      }
    } else if (StringAttr attr = arg.getDefiningOp()->getAttrOfType<StringAttr>(
                   kDeviceAttr)) {
      return op_device == attr;
    }
    // Fold tf.Identity when arg device is not defined.
    return true;
  };

  func_op.walk([&should_fold_op_func](TF::IdentityOp op) {
    StringAttr op_device = op->getAttrOfType<StringAttr>(kDeviceAttr);
    if (should_fold_op_func(op.getOperand(), op_device)) {
      op.replaceAllUsesWith(op.getOperand());
      op.erase();
    }
    return WalkResult::advance();
  });

  func_op.walk([&should_fold_op_func](TF::IdentityNOp op) {
    StringAttr op_device = op->getAttrOfType<StringAttr>(kDeviceAttr);
    bool should_fold = llvm::all_of(
        op.getOperands(), [&op_device, &should_fold_op_func](const Value &arg) {
          return should_fold_op_func(arg, op_device);
        });
    if (should_fold) {
      op.replaceAllUsesWith(op.getOperands());
      op.erase();
    }
    return WalkResult::advance();
  });
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateTensorDeviceCopyConversionPass() {
  return std::make_unique<TensorDeviceCopyConversionPass>();
}

}  // namespace TF
}  // namespace mlir
