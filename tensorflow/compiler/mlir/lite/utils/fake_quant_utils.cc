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

#include <string>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"

namespace mlir {
namespace TFL {

// Moves the TF operations out from the tfl.TFCustomOps wrappers inside the
// function. This is a no-op for the ops which are not wrapped.
LogicalResult UnwrapTFCustomOps(func::FuncOp fn, OpBuilder& builder) {
  llvm::SmallVector<Operation*, 4> wrapped_ops;
  fn.walk([&](TFL::CustomTfOp custom_op) {
    auto* real_op = &custom_op.getBody().front().front();
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
    Operation* inlined = builder.create(state);

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
LogicalResult ConvertFakeQuantOps(func::FuncOp func, MLIRContext* ctx,
                                  bool use_fake_quant_num_bits) {
  OpBuilder builder(func);
  if (failed(UnwrapTFCustomOps(func, builder))) {
    return failure();
  }

  // Insert the tfl.quantize/tfl.dequantize ops after the tf.FakeQuant* ops to
  // preserve the quantization parameters.
  func.walk([&](Operation* op) {
    if (auto fake_quant = llvm::dyn_cast<TF::FakeQuantWithMinMaxArgsOp>(op)) {
      (void)PreparePerTensorFakeQuantWithMinMaxArgs(use_fake_quant_num_bits)
          .matchAndRewrite(fake_quant, builder);
    } else if (auto fake_quant =
                   llvm::dyn_cast<TF::FakeQuantWithMinMaxVarsOp>(op)) {
      (void)PreparePerTensorFakeQuant(use_fake_quant_num_bits)
          .matchAndRewrite(fake_quant, builder);
    } else if (auto fake_quant =
                   llvm::dyn_cast<TF::FakeQuantWithMinMaxVarsPerChannelOp>(
                       op)) {
      (void)PreparePerChannelFakeQuant(use_fake_quant_num_bits)
          .matchAndRewrite(fake_quant, builder);
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
