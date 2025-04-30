/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2023 The StableHLO Authors.
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

#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo_ext/IR/base.h"
#include "stablehlo_ext/IR/stablehlo_ops.h"
#include "stablehlo_ext/transforms/passes.h"  // NOLINT: Used in passes.h.inc

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_STABLEHLOCANONICALIZEDYNAMISMPASS
#include "stablehlo_ext/transforms/passes.h.inc"

namespace {

struct CanonicalizeDynamicReduceWindowOpPattern
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CustomCallOp impl,
                                PatternRewriter& rewriter) const override {
    auto maybeOp = getDynamicReduceWindowOp(impl);
    if (!maybeOp || failed(maybeOp->verify())) return failure();
    DynamicReduceWindowOpAdaptor op = *maybeOp;

    // ReduceWindowOp supports dynamic shapes for operands and results, so we
    // don't check for that here unlike in some other patterns in this pass.
    SmallVector<int64_t> windowDimensions, windowStrides, baseDilations,
        windowDilations, padding;
    if (failed(hlo::matchInts(op.getWindowDimensions(), windowDimensions)))
      return rewriter.notifyMatchFailure(op,
                                         "expected static window_dimensions");
    if (failed(hlo::matchInts(op.getWindowStrides(), windowStrides)))
      return rewriter.notifyMatchFailure(op, "expected static window_strides");
    if (failed(hlo::matchInts(op.getBaseDilations(), baseDilations)))
      return rewriter.notifyMatchFailure(op, "expected static base_dilations");
    if (failed(hlo::matchInts(op.getWindowDilations(), windowDilations)))
      return rewriter.notifyMatchFailure(op,
                                         "expected static window_dilations");
    if (failed(hlo::matchInts(op.getPadding(), padding)))
      return rewriter.notifyMatchFailure(op, "expected static padding");
    auto newOp = rewriter.create<stablehlo::ReduceWindowOp>(
        op->getLoc(), op->getResultTypes(), op.getInputs(), op.getInitValues(),
        rewriter.getDenseI64ArrayAttr(windowDimensions),
        rewriter.getDenseI64ArrayAttr(windowStrides),
        rewriter.getDenseI64ArrayAttr(baseDilations),
        rewriter.getDenseI64ArrayAttr(windowDilations),
        hlo::getPaddingAttr(&rewriter, padding));

    // Inline the called computation into newOp.
    // This is somewhat annoying because we also have to rewrite the original
    // func::ReturnOp into stablehlo::ReturnOp.
    rewriter.cloneRegionBefore(op.getBody(), newOp.getBody(),
                               newOp.getBody().end());
    auto funcReturnOp =
        cast<func::ReturnOp>(newOp.getBody().front().getTerminator());
    rewriter.setInsertionPointToEnd(&newOp.getBody().front());
    rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(
        funcReturnOp, funcReturnOp.getOperands());
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct CanonicalizeDynamicRngBitGeneratorOpPattern
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CustomCallOp impl,
                                PatternRewriter& rewriter) const override {
    auto maybeOp = getDynamicRngBitGeneratorOp(impl);
    if (!maybeOp || failed(maybeOp->verify())) return failure();
    DynamicRngBitGeneratorOpAdaptor op = *maybeOp;

    // This pattern ignores and discards the output_shape operand. We rely on
    // the verifier to make sure that its value is consistent with result type.
    if (!succeeded(hlo::matchInts(op.getOutputShape())))
      return rewriter.notifyMatchFailure(op, "expected static output_shape");
    if (!cast<ShapedType>(op.getOutput().getType()).hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static output type");
    rewriter.replaceOpWithNewOp<stablehlo::RngBitGeneratorOp>(
        op, op->getResultTypes(), op.getRngAlgorithm(), op.getInitialState());
    return success();
  }
};

struct CanonicalizeDynamicTopKOpPattern
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CustomCallOp impl,
                                PatternRewriter& rewriter) const override {
    auto maybeOp = getDynamicTopKOp(impl);
    if (!maybeOp || failed(maybeOp->verify())) return failure();
    DynamicTopKOpAdaptor op = *maybeOp;

    SmallVector<int64_t> k;
    if (failed(hlo::matchInts(op.getK(), k)))
      return rewriter.notifyMatchFailure(impl, "expected constant k");

    // We rely on many of the properties checked by verification.
    auto valuesType = cast<ShapedType>(op.getValues().getType());
    auto valuesLastDimSize = valuesType.getShape()[valuesType.getRank() - 1];
    if (hlo::isDynamicDimSize(valuesLastDimSize) || valuesLastDimSize != k[0])
      return rewriter.notifyMatchFailure(
          op,
          "expected value of k to match the values last dimension size of "
          "static values type (result #0)");

    rewriter.replaceOpWithNewOp<chlo::TopKOp>(op, op->getResultTypes(),
                                              op.getOperand(), k[0]);
    return success();
  }
};

struct CanonicalizeApproxDynamicTopKOpPattern
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CustomCallOp impl,
                                PatternRewriter& rewriter) const override {
    auto maybeOp = getDynamicApproxTopKOp(impl);
    if (!maybeOp || failed(maybeOp->verify())) return failure();
    DynamicApproxTopKOpAdaptor op = *maybeOp;

    SmallVector<int64_t> k;
    if (failed(hlo::matchInts(op.getK(), k))) {
      return rewriter.notifyMatchFailure(impl, "expected constant k");
    }

    SmallVector<Value> newOperands;
    for (size_t i = 0; i < op.getNumInputs(); ++i) {
      newOperands.push_back(op.getInput(i));
    }
    for (size_t i = 0; i < op.getNumInputs(); ++i) {
      newOperands.push_back(op.getInitialValue(i));
    }

    auto stablehloBackendConfig = "mhlo.backend_config";
    auto backend_config = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
        impl->getAttr(stablehloBackendConfig));
    if (!backend_config)
      return rewriter.notifyMatchFailure(op,
                                         "Missing backend_config attribute");
    SmallVector<NamedAttribute> backend_config_attrs{backend_config.begin(),
                                                     backend_config.end()};
    backend_config_attrs.push_back(
        rewriter.getNamedAttr("top_k", rewriter.getI64IntegerAttr(k[0])));

    auto newOp = rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, op->getResultTypes(), newOperands, op->getAttrs());
    newOp.setCallTargetName("ApproxTopK");
    newOp->setAttr(stablehloBackendConfig,
                   rewriter.getDictionaryAttr(backend_config_attrs));
    return success();
  }
};

struct StablehloCanonicalizeDynamismPass
    : public impl::StablehloCanonicalizeDynamismPassBase<
          StablehloCanonicalizeDynamismPass> {
  using StablehloCanonicalizeDynamismPassBase::
      StablehloCanonicalizeDynamismPassBase;

  void runOnOperation() override {
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true)
        .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Aggressive)
        .setMaxIterations(2)
        .setMaxNumRewrites(GreedyRewriteConfig::kNoLimit)
        .setStrictness(GreedyRewriteStrictness::AnyOp);

    RewritePatternSet patterns(&getContext());
    stablehlo::populateStablehloCanonicalizeDynamismPatterns(&patterns,
                                                             &getContext());
    patterns.add<CanonicalizeDynamicReduceWindowOpPattern>(&getContext());
    patterns.add<CanonicalizeDynamicRngBitGeneratorOpPattern>(&getContext());
    patterns.add<CanonicalizeDynamicTopKOpPattern>(&getContext());
    patterns.add<CanonicalizeApproxDynamicTopKOpPattern>(&getContext());

    auto funcOp = getOperation();
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      funcOp.emitError("Failed to converge StablehloCanonicalizeDynamism in ")
          << config.getMaxIterations() << " iterations";
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace stablehlo_ext
}  // namespace mlir
