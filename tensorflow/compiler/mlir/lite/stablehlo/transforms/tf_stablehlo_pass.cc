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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/tf_stablehlo_pass.h"

#include <memory>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_util.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/register.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/rewriters.h"

namespace mlir {
namespace odml {

class TFToStablehloPass
    : public mlir::PassWrapper<TFToStablehloPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  explicit TFToStablehloPass(bool skip_quantization_ops = false,
                             bool skip_resize = false)
      : PassWrapper() {
    skip_quantization_ops_ = skip_quantization_ops;
    skip_resize_ = skip_resize;
  }

  TFToStablehloPass(const TFToStablehloPass &pass) {
    skip_quantization_ops_ = pass.skip_quantization_ops_;
    skip_resize_ = pass.skip_resize_;
  }

 private:
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::mhlo::registerAllMhloDialects(registry);
    mlir::stablehlo::registerAllDialects(registry);
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect>();
    registry.insert<shape::ShapeDialect>();
  }

 public:
  StringRef getArgument() const final { return "tf-stablehlo"; }
  StringRef getDescription() const final {
    return "This pass will legalize TF Ops to StableHLO Ops..";
  }

 protected:
  Option<bool> skip_quantization_ops_{
      *this, "skip-quantization-ops",
      ::llvm::cl::desc("Skip quantization ops")};

  Option<bool> skip_resize_{
      *this, "skip-resize",
      ::llvm::cl::desc("Skip tf.ResizeBilinear and tf.ResizeNearestNeighbor")};
};

void TFToStablehloPass::runOnOperation() {
  auto func = getOperation();
  MLIRContext *context = func->getContext();

  RewritePatternSet patterns(context);
  mhlo::PopulateLegalizeTfPatterns(context, &patterns);
  TF::PopulateTFLoweringBeforeHLOPatterns(context, &patterns);
  mhlo::PopulateLegalizeTfWithTf2XlaPatterns("XLA_CPU_JIT", patterns, context,
                                             /*prefer_tf2xla=*/false);
  chlo::populateDecomposeChloPatterns(context, &patterns);
  chlo::populateChloBroadcastingPatterns(context, &patterns);
  chlo::ConstantLikeOp::getCanonicalizationPatterns(patterns, context);

  ConversionTarget target(*context);
  target.addIllegalDialect<chlo::ChloDialect>();
  target.addLegalDialect<mlir::mhlo::MhloDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<shape::ShapeDialect>();
  target.addLegalOp<func::CallOp>();

  if (skip_quantization_ops_) {
    target.addLegalOp<TF::FakeQuantWithMinMaxVarsOp>();
    target.addLegalOp<TF::FakeQuantWithMinMaxVarsPerChannelOp>();
    target.addLegalOp<TF::FakeQuantWithMinMaxArgsOp>();
    target.addLegalOp<TF::QuantizeAndDequantizeV2Op>();
    target.addLegalOp<TF::QuantizeAndDequantizeV3Op>();
  }
  if (skip_resize_) {
    target.addLegalOp<TF::ResizeBilinearOp>();
    target.addLegalOp<TF::ResizeNearestNeighborOp>();
  }

  FrozenRewritePatternSet frozen_patterns(std::move(patterns));
  if (failed(applyPartialConversion(func, target, frozen_patterns))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateTFToStablehloPass(
    bool skip_quantization_ops, bool skip_resize) {
  return std::make_unique<TFToStablehloPass>(skip_quantization_ops,
                                             skip_resize);
}

static PassRegistration<TFToStablehloPass> pass;

}  // namespace odml
}  // namespace mlir
