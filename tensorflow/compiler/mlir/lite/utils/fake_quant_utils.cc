/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/utils/fake_quant_utils.h"

#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"

namespace mlir {
namespace TFL {

// Moves the TF operations out from the tfl.TFCustomOps wrappers inside the
// function. This is a no-op for the ops which are not wrapped.
LogicalResult UnwrapTFCustomOps(FuncOp fn, OpBuilder& builder) {
  llvm::SmallVector<Operation*, 4> wrapped_ops;
  fn.walk([&](TFL::CustomTfOp custom_op) {
    auto* real_op = &custom_op.body().front().front();
    wrapped_ops.push_back(real_op);
  });

  for (auto* op : wrapped_ops) {
    auto parent_op = op->getParentOfType<TFL::CustomTfOp>();
    if (!parent_op) continue;
    builder.setInsertionPoint(parent_op);

    // Recreate the operation by using the wrapper's operands and return types.
    // TODO(fengliuai): copy the regions.
    OperationState state(op->getLoc(), op->getName().getStringRef(),
                         parent_op->getOperands(), parent_op->getResultTypes(),
                         op->getAttrs(), op->getSuccessors());
    Operation* inlined = builder.createOperation(state);

    parent_op->replaceAllUsesWith(inlined);
    parent_op->erase();
  }
  return success();
}

// Three instances of the rule to cover the three different types of
// TF::FakeQuant operators
using PreparePerTensorFakeQuant = InsertTFLQuantOpsAfterTFFakeQuantOp<
    TF::FakeQuantWithMinMaxVarsOp, /*PerAxis=*/false,
    FetchConstantMinMaxInputs<TF::FakeQuantWithMinMaxVarsOp>>;

using PreparePerChannelFakeQuant = InsertTFLQuantOpsAfterTFFakeQuantOp<
    TF::FakeQuantWithMinMaxVarsPerChannelOp, /*PerAxis=*/true,
    FetchConstantMinMaxInputs<TF::FakeQuantWithMinMaxVarsPerChannelOp>>;

using PreparePerTensorFakeQuantWithMinMaxArgs =
    InsertTFLQuantOpsAfterTFFakeQuantOp<
        TF::FakeQuantWithMinMaxArgsOp, /*PerAxis=*/false,
        FetchMinMaxAttrs<TF::FakeQuantWithMinMaxArgsOp>>;

// Removes the wrapper of the tf.FakeQuant* ops and creates the tfl.quantize
// and tfl.dequantize pairs before tf.FakeQuant* being foled.
LogicalResult ConvertFakeQuantOps(FuncOp func, MLIRContext* ctx) {
  OpBuilder builder(func);
  if (failed(UnwrapTFCustomOps(func, builder))) {
    return failure();
  }

  // Insert the tfl.quantize/tfl.dequantize ops after the tf.FakeQuant* ops to
  // preserve the quantization parameters.
  func.walk([&](Operation* op) {
    if (auto fake_quant = llvm::dyn_cast<TF::FakeQuantWithMinMaxArgsOp>(op)) {
      (void)PreparePerTensorFakeQuantWithMinMaxArgs().matchAndRewrite(
          fake_quant, builder);
    } else if (auto fake_quant =
                   llvm::dyn_cast<TF::FakeQuantWithMinMaxVarsOp>(op)) {
      (void)PreparePerTensorFakeQuant().matchAndRewrite(fake_quant, builder);
    } else if (auto fake_quant =
                   llvm::dyn_cast<TF::FakeQuantWithMinMaxVarsPerChannelOp>(
                       op)) {
      (void)PreparePerChannelFakeQuant().matchAndRewrite(fake_quant, builder);
    }
  });

  return success();
}

std::vector<std::string> AllTfFakeQuantOps() {
  return {
      mlir::TF::FakeQuantWithMinMaxVarsOp::getOperationName().str(),
      mlir::TF::FakeQuantWithMinMaxVarsPerChannelOp::getOperationName().str(),
      mlir::TF::FakeQuantWithMinMaxArgsOp::getOperationName().str()};
}

}  // end namespace TFL
}  // end namespace mlir
