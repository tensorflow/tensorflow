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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project  // IWYU pragma: keep, for applyPatternsGreedily
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/temp_fake_quant_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

namespace mlir {
namespace tf_quant {
namespace {

class TFConvertFakeQuantToQdqPass
    : public PassWrapper<TFConvertFakeQuantToQdqPass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TFConvertFakeQuantToQdqPass)

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tf-quant-convert-fake-quant-to-qdq";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Convert Fake Quant op to quant.qcast and quant.dcast pairs";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect>();
    registry.insert<quant::QuantDialect>();
    registry.insert<mlir::quant::ir::TFQuantDialect>();
  }

  void runOnOperation() override;
};

static PassRegistration<TFConvertFakeQuantToQdqPass> pass;

void TFConvertFakeQuantToQdqPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  func::FuncOp func = getOperation();

  if (failed(tf_quant::ConvertFakeQuantOps(
          func, ctx, /*use_fake_quant_num_bits=*/false))) {
    func.emitError() << "quant-convert-fake-quant-to-qdq pass failed.";
    signalPassFailure();
  }

  // For removing dead FakeQuant* ops
  RewritePatternSet patterns(ctx);
  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateTFConvertFakeQuantToQdqPass() {
  return std::make_unique<TFConvertFakeQuantToQdqPass>();
}

}  // namespace tf_quant
}  // namespace mlir
