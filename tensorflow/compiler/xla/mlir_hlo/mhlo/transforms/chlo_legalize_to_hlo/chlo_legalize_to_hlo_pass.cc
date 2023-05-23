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

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/ChloOps.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_CHLOLEGALIZETOHLOPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

struct ChloLegalizeToHloPass
    : public impl::ChloLegalizeToHloPassBase<ChloLegalizeToHloPass> {
  explicit ChloLegalizeToHloPass(bool legalizeBroadcasts,
                                 bool expandCompositions)
      : ChloLegalizeToHloPassBase<
            ChloLegalizeToHloPass>::ChloLegalizeToHloPassBase() {
    this->legalize_broadcasts_ = legalizeBroadcasts;
    this->expand_compositions_ = expandCompositions;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mhlo::MhloDialect, shape::ShapeDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ConversionTarget conversionTarget(getContext());
    RewritePatternSet conversionPatterns(&getContext());
    conversionTarget.addIllegalDialect<chlo::ChloDialect>();

    // Consider the mhlo dialect legal for tests. Also add helper dialects
    // that are needed by the patterns.
    conversionTarget
        .addLegalDialect<MhloDialect, mlir::arith::ArithDialect,
                         mlir::func::FuncDialect, mlir::tensor::TensorDialect,
                         mlir::shape::ShapeDialect, mlir::scf::SCFDialect>();
    conversionTarget.addLegalOp<chlo::MinimumBroadcastShapesOp>();

    if (legalize_broadcasts_) {
      chlo::populateChloBroadcastingPatterns(&getContext(),
                                             &conversionPatterns);
    }

    if (expand_compositions_) {
      chlo::populateDecomposeChloPatterns(&getContext(), &conversionPatterns);
    } else {
      conversionTarget
          .addLegalOp<chlo::NextAfterOp, chlo::PolygammaOp, chlo::ZetaOp>();
    }

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(conversionPatterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createChloLegalizeToHloPass(
    bool legalizeBroadcasts, bool expandCompositions) {
  return std::make_unique<ChloLegalizeToHloPass>(legalizeBroadcasts,
                                                 expandCompositions);
}

}  // namespace mhlo
}  // namespace mlir
