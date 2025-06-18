/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// Copied and modified from
// //third_party/tensorflow/compiler/mlir/lite/utils/fake_quant_utils.cc
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/temp_fake_quant_utils.h"

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"

namespace mlir {
namespace tf_quant {

// Three instances of the rule to cover the three different types of
// TF::FakeQuant operators
using PreparePerTensorFakeQuant = ConvertFakeQuantOpToQuantOps<
    TF::FakeQuantWithMinMaxVarsOp, /*PerAxis=*/false,
    FetchConstantMinMaxInputs<TF::FakeQuantWithMinMaxVarsOp>>;

using PreparePerChannelFakeQuant = ConvertFakeQuantOpToQuantOps<
    TF::FakeQuantWithMinMaxVarsPerChannelOp, /*PerAxis=*/true,
    FetchConstantMinMaxInputs<TF::FakeQuantWithMinMaxVarsPerChannelOp>>;

using PreparePerTensorFakeQuantWithMinMaxArgs = ConvertFakeQuantOpToQuantOps<
    TF::FakeQuantWithMinMaxArgsOp, /*PerAxis=*/false,
    FetchMinMaxAttrs<TF::FakeQuantWithMinMaxArgsOp>>;

// Removes the wrapper of the tf.FakeQuant* ops and creates the quant.qcast
// and quant.dcast pairs before tf.FakeQuant* ops are being foled.
LogicalResult ConvertFakeQuantOps(func::FuncOp func, MLIRContext* ctx,
                                  bool use_fake_quant_num_bits) {
  OpBuilder builder(func);

  // Insert the quant.qcast/quant.dcast ops in place of the tf.FakeQuant* ops to
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

}  // namespace tf_quant
}  // namespace mlir
