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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/tf_mhlo_pass.h"

#include <utility>

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/mhlo_util.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace mhlo {

class TFToMhloPass
    : public mlir::PassWrapper<TFToMhloPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  explicit TFToMhloPass(bool skip_quantization_ops = false,
                        bool skip_resize = false)
      : PassWrapper() {
    skip_quantization_ops_ = skip_quantization_ops;
    skip_resize_ = skip_resize;
  }

  TFToMhloPass(const TFToMhloPass &pass) {
    skip_quantization_ops_ = pass.skip_quantization_ops_;
    skip_resize_ = pass.skip_resize_;
  }

 private:
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::mhlo::registerAllMhloDialects(registry);
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithmeticDialect>();
    registry.insert<::mlir::mhlo::MhloDialect>();
    registry.insert<shape::ShapeDialect>();
  }

 public:
  StringRef getArgument() const final { return "tf-mhlo"; }
  StringRef getDescription() const final {
    return "This pass will legalize TF Ops to MHLO Ops..";
  }

 protected:
  Option<bool> skip_quantization_ops_{
      *this, "skip-quantization-ops",
      ::llvm::cl::desc("Skip quantization ops")};

  Option<bool> skip_resize_{
      *this, "skip-resize",
      ::llvm::cl::desc("Skip tf.ResizeBilinear and tf.ResizeNearestNeighbor")};
};

void TFToMhloPass::runOnOperation() {
  auto func = getOperation();
  MLIRContext *context = func->getContext();

  RewritePatternSet patterns(context);
  // Add TF to MHLO patterns.
  PopulateTFToMhloPatterns(
      context, /*legalize_chlo=*/true,
      /*tf2xla_fallback_device_type=*/llvm::StringRef("XLA_CPU_JIT"),
      /*prefer_tf2xla=*/false, &patterns);

  ConversionTarget target(*context);
  target.addIllegalDialect<chlo::ChloDialect>();
  target.addLegalDialect<mlir::mhlo::MhloDialect>();
  target.addLegalDialect<arith::ArithmeticDialect>();
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

std::unique_ptr<OperationPass<func::FuncOp>> CreateTFToMhloPass(
    bool skip_quantization_ops, bool skip_resize) {
  return std::make_unique<TFToMhloPass>(skip_quantization_ops, skip_resize);
}

static PassRegistration<TFToMhloPass> pass;

}  // namespace mhlo
}  // namespace TFL
}  // namespace mlir
