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

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/PassOptions.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"

namespace mlir {
namespace TF {
namespace {

// Deletes the op and forwards the arguments.
template <typename TF_Op>
class PassThroughConversion : public mlir::OpConversionPattern<TF_Op> {
 public:
  explicit PassThroughConversion(MLIRContext *context)
      : mlir::OpConversionPattern<TF_Op>(context) {}

  LogicalResult matchAndRewrite(
      TF_Op op, ArrayRef<mlir::Value> operands,
      ConversionPatternRewriter &rewriter) const override {  // NOLINT
    // Just forward the arguments to results.
    rewriter.replaceOp(op, operands);
    return success();
  }
};

class TensorDeviceCopyConversionPass
    : public PassWrapper<TensorDeviceCopyConversionPass, FunctionPass> {
 public:
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    mlir::ConversionTarget target(getContext());

    // TODO(tfrt-devs): when device placer is introduced in the lowering pass,
    // we need to check if Identity op and it's previous op are placed on the
    // same device. If not, we don't fold Identity op since it's used for tensor
    // copying between devices.
    patterns.insert<PassThroughConversion<TF::IdentityOp>,
                    PassThroughConversion<TF::IdentityNOp>>(&getContext());

    if (failed(applyPartialConversion(getFunction(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
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
        "Handle ops that copy tensors between devices. E.g., tf.Identity.");

}  // namespace TF
}  // namespace mlir
