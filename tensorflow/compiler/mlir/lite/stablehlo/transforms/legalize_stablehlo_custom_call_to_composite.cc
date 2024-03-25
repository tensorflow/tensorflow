/* Copyright 2022 The StableHLO Authors.
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

#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"

namespace mlir {
namespace odml {

#define GEN_PASS_DEF_LEGALIZESTABLEHLOCUSTOMCALLTOCOMPOSITEPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h.inc"

struct ReplaceCustomCallWithComposite final
    : OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;

  explicit ReplaceCustomCallWithComposite(MLIRContext *context)
      : OpRewritePattern<mlir::stablehlo::CustomCallOp>(context) {}

  LogicalResult matchAndRewrite(mlir::stablehlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    auto backendConfig =
        op->getAttr("composite.backend_config").dyn_cast<DictionaryAttr>();
    if (!backendConfig)
      return op->emitError(
          "custom_call has no 'composite.backend_config' attribute or the "
          "attribute is not a dictionary");

    auto name = backendConfig.get("name").dyn_cast<StringAttr>();
    if (!name)
      return op->emitError(
          "backend_config has no 'name' key or the name value is not a string");

    auto attrs = backendConfig.get("attributes").dyn_cast<DictionaryAttr>();
    if (!attrs)
      return op->emitError(
          "backend_config has no 'attributes' key or the attributes value is "
          "not a dictionary");

    auto calledComputations = op.getCalledComputations();
    if (!calledComputations || calledComputations.size() != 1)
      return op->emitError("expected exactly one called_computation");

    auto decomposition = calledComputations[0].cast<FlatSymbolRefAttr>();

    auto composite = rewriter.create<mlir::stablehlo::CompositeOp>(
        op.getLoc(), op.getResultTypes(), op.getOperands(), name.str(), attrs,
        decomposition.getValue());
    rewriter.replaceOp(op, composite.getResults());
    return success();
  }
};

struct LegalizeStablehloCustomCallToCompositePass
    : public impl::LegalizeStablehloCustomCallToCompositePassBase<
          LegalizeStablehloCustomCallToCompositePass> {
  using LegalizeStablehloCustomCallToCompositePassBase::
      LegalizeStablehloCustomCallToCompositePassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<mlir::stablehlo::StablehloDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addDynamicallyLegalOp<mlir::stablehlo::CustomCallOp>(
        [&](mlir::stablehlo::CustomCallOp op) {
          return op.getCallTargetName() != "stablehlo.composite";
        });

    RewritePatternSet patterns(context);
    patterns.add<ReplaceCustomCallWithComposite>(context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

static PassRegistration<LegalizeStablehloCustomCallToCompositePass>
    pass_shlo_sc2c;

}  // namespace odml
}  // namespace mlir
