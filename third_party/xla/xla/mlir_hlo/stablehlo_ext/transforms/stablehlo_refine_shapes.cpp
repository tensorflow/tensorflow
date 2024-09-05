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

#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/TypeInference.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/StablehloRefineShapes.h"
#include "stablehlo_ext/IR/base.h"
#include "stablehlo_ext/IR/stablehlo_ops.h"
#include "stablehlo_ext/transforms/passes.h"  // NOLINT: Used in passes.h.inc

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_STABLEHLOREFINESHAPESPASS
#include "stablehlo_ext/transforms/passes.h.inc"

namespace {

struct RefineDynamicReduceWindowOpPattern
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CustomCallOp impl,
                                PatternRewriter& rewriter) const override {
    auto maybeOp = getDynamicReduceWindowOp(impl);
    if (!maybeOp || failed(maybeOp->verify())) return failure();
    DynamicReduceWindowOpAdaptor op = *maybeOp;

    // At the moment, we only support refining return types using fully static
    // shape values which serves the current use cases well.
    // Support for partially static shape values is left for future work.
    SmallVector<int64_t> windowDimensions, windowStrides, baseDilations,
        windowDilations, padding;
    if (failed(hlo::matchInts(op.getWindowDimensions(), windowDimensions)))
      return rewriter.notifyMatchFailure(op,
                                         "expected constant window_dimensions");
    if (failed(hlo::matchInts(op.getWindowStrides(), windowStrides)))
      return rewriter.notifyMatchFailure(op,
                                         "expected constant window_strides");
    if (failed(hlo::matchInts(op.getBaseDilations(), baseDilations)))
      return rewriter.notifyMatchFailure(op,
                                         "expected constant base_dilations");
    if (failed(hlo::matchInts(op.getWindowDilations(), windowDilations)))
      return rewriter.notifyMatchFailure(op,
                                         "expected constant window_dilations");
    if (failed(hlo::matchInts(op.getPadding(), padding)))
      return rewriter.notifyMatchFailure(op, "expected constant padding");

    SmallVector<ShapedTypeComponents> inferredReturnTypes;
    if (failed(hlo::inferReduceWindowOp(
            /*location=*/{}, op.getInputs(), op.getInitValues(),
            llvm::to_vector(rewriter.getI64TensorAttr(windowDimensions)
                                .getValues<int64_t>()),
            llvm::to_vector(
                rewriter.getI64TensorAttr(windowStrides).getValues<int64_t>()),
            llvm::to_vector(
                rewriter.getI64TensorAttr(baseDilations).getValues<int64_t>()),
            llvm::to_vector(rewriter.getI64TensorAttr(windowDilations)
                                .getValues<int64_t>()),
            hlo::getPaddingAttr(&rewriter, padding), op.getBody(),
            inferredReturnTypes)))
      return rewriter.notifyMatchFailure(op, "inferReduceWindowOp failed");
    return stablehlo::refineReturnTypes(rewriter, op, inferredReturnTypes);
  }
};

struct RefineDynamicRngBitGeneratorOpPattern
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CustomCallOp impl,
                                PatternRewriter& rewriter) const override {
    auto maybeOp = getDynamicRngBitGeneratorOp(impl);
    if (!maybeOp || failed(maybeOp->verify())) return failure();
    DynamicRngBitGeneratorOpAdaptor op = *maybeOp;

    // At the moment, we only support refining return types using fully static
    // shape values which serves the current use cases well.
    // Support for partially static shape values is left for future work.
    auto initialStateType = cast<ShapedType>(op.getInitialState().getType());
    SmallVector<int64_t> outputShape;
    if (failed(hlo::matchInts(op.getOutputShape(), outputShape)))
      return rewriter.notifyMatchFailure(op, "expected constant output_shape");

    // We only need to refine the shape of `output` (the second result).
    // The shape of `output_state` (the first result) is determined by the shape
    // of `initial_state`, so we ignore it and provide an empty refinement.
    return stablehlo::refineReturnTypes(rewriter, op,
                                        {{initialStateType}, {outputShape}});
  }
};

struct RefineDynamicTopKOpPattern
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CustomCallOp impl,
                                PatternRewriter& rewriter) const override {
    auto maybeOp = getDynamicTopKOp(impl);
    if (!maybeOp || failed(maybeOp->verify())) return failure();
    DynamicTopKOpAdaptor op = *maybeOp;

    auto operandType = cast<ShapedType>(op.getOperand().getType());
    SmallVector<int64_t> outputShape(operandType.getShape());
    SmallVector<int64_t> k;
    if (failed(hlo::matchInts(op.getK(), k)))
      return rewriter.notifyMatchFailure(op, "expected constant k");

    outputShape[operandType.getRank() - 1] = k[0];
    return stablehlo::refineReturnTypes(rewriter, op,
                                        {{outputShape}, {outputShape}});
  }
};

struct StablehloRefineShapesPass
    : public impl::StablehloRefineShapesPassBase<StablehloRefineShapesPass> {
  using StablehloRefineShapesPassBase::StablehloRefineShapesPassBase;

  void runOnOperation() override {
    auto func = stablehlo::getStablehloRefineShapesTarget(getOperation());
    if (!func) return signalPassFailure();

    // The algorithm behind this pass consists of a single traversal of the
    // function. This is sufficient because we only support one function per
    // program at the moment.
    // TODO(#1048): Find out why .maxIterations = 1 no longer works.
    // There have been recent refactors to applyPatternsAndFoldGreedily
    // upstream, and that might be the reason.
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Aggressive;
    config.maxIterations = 3;
    config.maxNumRewrites = GreedyRewriteConfig::kNoLimit;
    config.strictMode = GreedyRewriteStrictness::AnyOp;

    RewritePatternSet patterns(&getContext());
    stablehlo::populateStablehloRefineShapesPatterns(&patterns, &getContext());
    stablehlo::populateStablehloShapeFolderPatterns(&patterns, &getContext());
    patterns.add<RefineDynamicReduceWindowOpPattern>(&getContext());
    patterns.add<RefineDynamicRngBitGeneratorOpPattern>(&getContext());
    patterns.add<RefineDynamicTopKOpPattern>(&getContext());
    if (failed(
            applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
      func.emitError()
          << "Greedy rewriter in StablehloRefineShapes does not converge after "
          << config.maxIterations << " iterations.";
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace stablehlo_ext
}  // namespace mlir
