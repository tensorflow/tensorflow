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

#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace mhlo {

namespace {

struct ChloLegalizeToHloPass
    : public ChloLegalizeToHloPassBase<ChloLegalizeToHloPass> {
  explicit ChloLegalizeToHloPass(bool legalize_broadcasts,
                                 bool expand_compositions)
      : ChloLegalizeToHloPassBase<
            ChloLegalizeToHloPass>::ChloLegalizeToHloPassBase() {
    this->legalize_broadcasts_ = legalize_broadcasts;
    this->expand_compositions_ = expand_compositions;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mhlo::MhloDialect, shape::ShapeDialect, scf::SCFDialect>();
  }

  void runOnFunction() override {
    ConversionTarget conversionTarget(getContext());
    OwningRewritePatternList conversionPatterns(&getContext());
    conversionTarget.addIllegalDialect<chlo::HloClientDialect>();

    // Consider the mhlo dialect legal for tests. Also add helper dialects
    // that are needed by the patterns.
    conversionTarget.addLegalDialect<
        MhloDialect, mlir::StandardOpsDialect, mlir::tensor::TensorDialect,
        mlir::shape::ShapeDialect, mlir::scf::SCFDialect>();
    conversionTarget.addLegalOp<chlo::MinimumBroadcastShapesOp>();

    if (legalize_broadcasts_) {
      chlo::PopulateChloBroadcastingPatterns(&getContext(),
                                             &conversionPatterns);
    }

    if (expand_compositions_) {
      chlo::PopulateDecomposeChloPatterns(&getContext(), &conversionPatterns);
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

std::unique_ptr<FunctionPass> createChloLegalizeToHloPass(
    bool legalize_broadcasts, bool expand_compositions) {
  return std::make_unique<ChloLegalizeToHloPass>(legalize_broadcasts,
                                                 expand_compositions);
}

}  // namespace mhlo
}  // namespace mlir
