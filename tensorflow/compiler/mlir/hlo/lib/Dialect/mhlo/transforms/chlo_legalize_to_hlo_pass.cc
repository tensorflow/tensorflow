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
  explicit ChloLegalizeToHloPass(bool broadcast_only)
      : ChloLegalizeToHloPassBase<
            ChloLegalizeToHloPass>::ChloLegalizeToHloPassBase() {
    this->broadcast_only_ = broadcast_only;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mhlo::MhloDialect, shape::ShapeDialect, scf::SCFDialect>();
  }

  void runOnFunction() override {
    ConversionTarget conversionTarget(getContext());
    OwningRewritePatternList conversionPatterns;
    conversionTarget.addIllegalDialect<chlo::HloClientDialect>();

    // Consider the mhlo dialect legal for tests. Also add helper dialects
    // that are needed by the patterns.
    conversionTarget.addLegalDialect<
        MhloDialect, mlir::StandardOpsDialect, mlir::tensor::TensorDialect,
        mlir::shape::ShapeDialect, mlir::scf::SCFDialect>();
    conversionTarget.addLegalOp<chlo::MinimumBroadcastShapesOp>();

    if (broadcast_only_) {
      chlo::PopulateChloBroadcastingPatterns(&getContext(),
                                             &conversionPatterns);
      conversionTarget.addLegalOp<chlo::ZetaOp, chlo::PolygammaOp>();
    } else {
      chlo::PopulateLegalizeChloToHloPatterns(&getContext(),
                                              &conversionPatterns);
    }

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(conversionPatterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<FunctionPass> createChloLegalizeToHloPass(bool broadcast_only) {
  return std::make_unique<ChloLegalizeToHloPass>(broadcast_only);
}

}  // namespace mhlo
}  // namespace mlir
