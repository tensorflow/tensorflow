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
#include <memory>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/ir/importexport/mangling.h"

namespace mlir {
namespace quant {
namespace {

class ConvertTFQuantOpsToMHLOPass
    : public PassWrapper<ConvertTFQuantOpsToMHLOPass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTFQuantOpsToMHLOPass)

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-convert-tf-quant-ops-to-mhlo";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Convert TF Quant ops to MHLO quantization";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TF::TensorFlowDialect>();
    registry.insert<mhlo::MhloDialect>();
    registry.insert<tf_type::TFTypeDialect>();
    registry.insert<quant::QuantizationDialect>();
    registry.insert<quantfork::QuantizationForkDialect>();
  }

  void runOnOperation() override;
};

static PassRegistration<ConvertTFQuantOpsToMHLOPass> pass;

void ConvertTFQuantOpsToMHLOPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  func::FuncOp func = getOperation();

  RewritePatternSet patterns(ctx);
  mhlo::PopulateLegalizeTfQuantizationPatterns(ctx, &patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertTFQuantOpsToMHLOPass() {
  return std::make_unique<ConvertTFQuantOpsToMHLOPass>();
}

}  // namespace quant
}  // namespace mlir
