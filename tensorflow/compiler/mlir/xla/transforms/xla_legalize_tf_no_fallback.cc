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

#include <memory>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf_passes_detail.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/stablehlo/stablehlo/dialect/ChloOps.h"

namespace mlir {
namespace mhlo {
namespace {

class LegalizeTFNoFallback
    : public LegalizeTFNoFallbackBase<LegalizeTFNoFallback> {
 public:
  explicit LegalizeTFNoFallback(bool allow_partial_conversion) {
    allow_partial_conversion_ = allow_partial_conversion;
  }
  /// Performs the lowering to HLO dialect.
  void runOnOperation() override;
};

void LegalizeTFNoFallback::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *context = op->getContext();
  RewritePatternSet patterns(context);

  // Add TF->HLO legalization patterns.
  PopulateLegalizeTfPatterns(context, &patterns);

  // ConstantLike op is convenient to create splat constants, but is
  // canonicalized to plain HLO constant if statically shaped. Add the
  // canonicalization pattern to pattern list to enable multi-hop lowering.
  chlo::ConstantLikeOp::getCanonicalizationPatterns(patterns, context);

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithmeticDialect>();
  target.addLegalDialect<chlo::ChloDialect>();
  target.addLegalDialect<MhloDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<shape::ShapeDialect>();
  target.addLegalOp<func::CallOp>();

  // Add TF->TF lowering patterns.
  TF::PopulateTFLoweringBeforeHLOPatterns(context, &patterns);
  if (!allow_partial_conversion_) {
    // Fully qualify ReturnOp here as mhlo dialect also defines a ReturnOp.
    target.addLegalOp<ModuleOp, func::FuncOp, ::mlir::func::ReturnOp>();
    llvm::DenseSet<Operation *> nonlegalized_ops;
    LogicalResult result = applyPartialConversion(
        op, target, std::move(patterns), &nonlegalized_ops);
    // In order to enforce that the conversion result is fully converted,
    // fail if there are any nonlegalized ops in the set.
    if (failed(result) || !nonlegalized_ops.empty()) {
      return signalPassFailure();
    }
  } else if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // end namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeTFNoFallbackPass(
    bool allow_partial_conversion) {
  return std::make_unique<LegalizeTFNoFallback>(allow_partial_conversion);
}

}  // end namespace mhlo
}  // end namespace mlir
