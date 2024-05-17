/* Copyright 2020 The OpenXLA Authors.

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

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mhlo/utils/type_conversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_CHLOLEGALIZETOHLOPASS
#define GEN_PASS_DEF_CHLOLEGALIZETOHIGHLEVELMHLOPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

struct ChloLegalizeToHighLevelMhloPass
    : public impl::ChloLegalizeToHighLevelMhloPassBase<
          ChloLegalizeToHighLevelMhloPass> {
  using ChloLegalizeToHighLevelMhloPassBase::
      ChloLegalizeToHighLevelMhloPassBase;

  void runOnOperation() override {
    MLIRContext &context = getContext();
    ConversionTarget conversionTarget(context);
    RewritePatternSet conversionPatterns(&context);

    chlo::populateChloToHighLevelMhloOpPatterns(&context, &conversionPatterns);

    // Consider the mhlo dialect legal for tests. Also add helper dialects
    // that are needed by the patterns.
    conversionTarget.addLegalDialect<chlo::ChloDialect, mhlo::MhloDialect>();
    conversionTarget.addIllegalOp<chlo::TopKOp, chlo::ErfOp, chlo::TanOp>();

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(conversionPatterns)))) {
      return signalPassFailure();
    }
  }
};

struct ChloLegalizeToHloPass
    : public impl::ChloLegalizeToHloPassBase<ChloLegalizeToHloPass> {
  using ChloLegalizeToHloPassBase::ChloLegalizeToHloPassBase;

  void runOnOperation() override {
    MLIRContext &context = getContext();
    ConversionTarget conversionTarget(context);
    RewritePatternSet conversionPatterns(&context);

    stablehlo::StablehloToHloTypeConverter typeConverter;
    chlo::populateChloToHloPatterns(&context, &typeConverter,
                                    &conversionPatterns);

    // Consider the mhlo dialect legal for tests. Also add helper dialects
    // that are needed by the patterns.
    conversionTarget
        .addIllegalDialect<chlo::ChloDialect, stablehlo::StablehloDialect>();
    conversionTarget.addLegalDialect<
        MhloDialect, mlir::arith::ArithDialect, mlir::func::FuncDialect,
        mlir::tensor::TensorDialect, mlir::shape::ShapeDialect>();

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(conversionPatterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mhlo

namespace chlo {
namespace {
#include "chlo_legalize_to_hlo/generated_chlo_legalize_to_hlo.inc"

}  // namespace

void populateChloToHighLevelMhloOpPatterns(MLIRContext *,
                                           RewritePatternSet *patterns) {
  populateWithGenerated(*patterns);
}

void populateChloToHloPatterns(MLIRContext *context,
                               TypeConverter *typeConverter,
                               RewritePatternSet *patterns) {
  chlo::populateChloToHighLevelMhloOpPatterns(context, patterns);
  stablehlo::populateChloToStablehloPatterns(context, patterns);
  stablehlo::populateStablehloToHloPatterns(patterns, typeConverter, context);
}

}  // namespace chlo
}  // namespace mlir
