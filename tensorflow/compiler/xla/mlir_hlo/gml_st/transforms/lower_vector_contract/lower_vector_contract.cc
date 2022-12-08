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

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/transforms.h"
#include "gml_st/utils/vector_utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

using vector::OuterProductOp;

#define GEN_PASS_DEF_LOWERVECTORCONTRACTPASS
#include "gml_st/transforms/passes.h.inc"

struct OuterProductOpCanonicalizationPattern
    : public OpRewritePattern<OuterProductOp> {
  OuterProductOpCanonicalizationPattern(
      MLIRContext *context, llvm::function_ref<bool(OuterProductOp)> filterFn,
      PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), filterFn(filterFn) {}

  LogicalResult matchAndRewrite(OuterProductOp op,
                                PatternRewriter &rewriter) const override {
    if (!filterFn(op))
      return rewriter.notifyMatchFailure(op, "did not match filter");

    bool changed = false;
    SmallVector<Value> newAccs{op.getAcc()};
    for (auto &acc : newAccs) {
      auto materializeOp = acc.getDefiningOp<MaterializeOp>();
      auto src = materializeOp.getSource();
      auto srcType = src.getType().cast<ShapedType>();
      if (auto resType = op.getResult().getType().dyn_cast<ShapedType>()) {
        if (resType.hasStaticShape() && srcType == resType) {
          acc = src;
          changed = true;
        }
      }
    }
    if (!changed) return failure();
    rewriter.updateRootInPlace(op,
                               [&]() { op.getAccMutable().assign(newAccs); });
    return success();
  }

 private:
  llvm::function_ref<bool(OuterProductOp)> filterFn;
};

struct LowerVectorContractPass
    : public impl::LowerVectorContractPassBase<LowerVectorContractPass> {
  LowerVectorContractPass() = default;

  void runOnOperation() override {
    auto func = getOperation();
    auto *ctx = func.getContext();

    // Reduce vector.contract dimensions to fit one of the lowering patterns to
    // vector.outerproduct.
    {
      RewritePatternSet castAwayUnitDimPatterns(ctx);
      vector::populateCastAwayVectorLeadingOneDimPatterns(
          castAwayUnitDimPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              func, std::move(castAwayUnitDimPatterns)))) {
        return signalPassFailure();
      }

      RewritePatternSet reductionToContractPatterns(ctx);
      vector::populateVectorReductionToContractPatterns(
          reductionToContractPatterns);
      vector::ExtractOp::getCanonicalizationPatterns(
          reductionToContractPatterns, ctx);
      if (failed(applyPatternsAndFoldGreedily(
              func, std::move(reductionToContractPatterns)))) {
        return signalPassFailure();
      }
    }

    RewritePatternSet patterns(ctx);

    auto outerProductOpFilter = [&](OuterProductOp op) {
      return (llvm::any_of(op.getAcc(), [](auto acc) {
        return acc.template getDefiningOp<MaterializeOp>() != nullptr;
      }));
    };

    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    // Currently we always lower vector.contract into vector.outerproduct.
    patterns.add<vector::ContractionOpToOuterProductOpLowering>(
        vector::VectorTransformsOptions().setVectorTransformsOptions(
            vector::VectorContractLowering::OuterProduct),
        ctx, 2);
    patterns.add<OuterProductOpCanonicalizationPattern>(ctx,
                                                        outerProductOpFilter);
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLowerVectorContractPass() {
  return std::make_unique<LowerVectorContractPass>();
}
}  // namespace gml_st
}  // namespace mlir
