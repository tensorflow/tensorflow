/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/peeling/peeling.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMDOTFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

constexpr llvm::StringRef kDotTransformedLabel = "__dot_transformed_label__";

FailureOr<scf::SCFTilingResult> tileReductionDim(PatternRewriter &rewriter,
                                                 Operation *op,
                                                 ArrayRef<int64_t> tileSizes) {
  scf::SCFTilingOptions opts;
  opts.setTileSizes(tileSizes);

  auto tilingResult = scf::tileUsingSCFForOp(rewriter, op, opts);
  if (failed(tilingResult)) return failure();

  // Update the results if tiling occurred.
  if (!tilingResult->loops.empty()) {
    rewriter.replaceOp(op, tilingResult->replacements);
    op = tilingResult->tiledOps.front();
  }

  setLabel(op, kDotTransformedLabel);
  return tilingResult;
}

FailureOr<GMLSTTilingResult> tileParallelDims(PatternRewriter &rewriter,
                                              Operation *op,
                                              ArrayRef<int64_t> tileSizes) {
  auto tilingResult = tileUsingSCFForallOp(rewriter, cast<TilingInterface>(op),
                                           getSCFTilingOptions(tileSizes));
  if (failed(tilingResult)) return failure();

  // Update the results if tiling occurred.
  if (tilingResult->loop != nullptr) {
    rewriter.replaceOp(op, tilingResult->loop->getResults());
  }

  return tilingResult;
}

MatmulSizes getMatmulSizes(linalg::VecmatOp op) {
  // [1, k] x [k, n]
  ShapedType ty = op->getOperand(1).getType().cast<ShapedType>();
  MatmulSizes sizes;
  sizes.m = 1;
  sizes.k = ty.getDimSize(0);
  sizes.n = ty.getDimSize(1);
  return sizes;
}

MatmulSizes getMatmulSizes(linalg::MatvecOp op) {
  // [m, k] x [k, 1]
  ShapedType ty = op->getOperand(0).getType().cast<ShapedType>();
  MatmulSizes sizes;
  sizes.m = ty.getDimSize(0);
  sizes.k = ty.getDimSize(1);
  sizes.n = 1;
  return sizes;
}

MatmulSizes getMatmulSizes(linalg::DotOp op) {
  // [1, k] x [k, 1]
  ShapedType ty = op->getOperand(0).getType().cast<ShapedType>();
  MatmulSizes sizes;
  sizes.m = 1;
  sizes.k = ty.getDimSize(0);
  sizes.n = 1;
  return sizes;
}

/// Pattern to tile dot operations (linalg.matvec, linalg.vecmat, linalg.dot)
/// and peel the generated loops.
template <typename DotTy>
struct DotTransformPattern : public OpRewritePattern<DotTy> {
  using OpRewritePattern<DotTy>::OpRewritePattern;

  explicit DotTransformPattern(
      MLIRContext *context, MatmulTileSizeComputationFn tileSizeFn,
      std::function<SmallVector<int64_t>(MatmulSizes)> parallelDimTileSizeFn,
      std::function<SmallVector<int64_t>(MatmulSizes)> reductionDimTileSizeFn,
      PatternBenefit benefit = 1)
      : OpRewritePattern<DotTy>(context, benefit),
        tileSizeFn(std::move(tileSizeFn)),
        parallelDimTileSizeFn(std::move(parallelDimTileSizeFn)),
        reductionDimTileSizeFn(std::move(reductionDimTileSizeFn)) {}

  LogicalResult matchAndRewrite(DotTy dotOp,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(dotOp, kDotTransformedLabel)) {
      return rewriter.notifyMatchFailure(dotOp,
                                         "has already been transformed.");
    }
    if (isa<scf::ForallOp, scf::ForOp>(dotOp->getParentOp())) {
      return rewriter.notifyMatchFailure(
          dotOp, "has already been tiled by another pass.");
    }

    auto tileSizes = tileSizeFn(getMatmulSizes(dotOp));
    auto tilingParallelDimsResult = tileParallelDims(
        rewriter, dotOp.getOperation(), parallelDimTileSizeFn(tileSizes));
    if (failed(tilingParallelDimsResult)) return failure();

    scf::ForallOp forallOp = tilingParallelDimsResult->loop;
    if (forallOp != nullptr) {
      dotOp = cast<DotTy>(tilingParallelDimsResult->tiledOps.back());
    }

    // Second level tiling: reduction dimension.
    auto tilingReductionDimResult = tileReductionDim(
        rewriter, dotOp.getOperation(), reductionDimTileSizeFn(tileSizes));
    if (failed(tilingReductionDimResult)) return failure();

    if (!tilingReductionDimResult->loops.empty()) {
      dotOp = cast<DotTy>(tilingReductionDimResult->tiledOps.back());
    }
    // Peel parallel loops.
    if (forallOp != nullptr) {
      (void)peelAllLoops(forallOp, rewriter);
    }

    // Peel reduction loop inside the main parallel loop, label the main loop as
    // "perfectly tiled" one, to enable vectorization after canonicalization.
    auto peelingResult =
        peelSCFForOp(rewriter, tilingReductionDimResult->loops.front());
    setLabel(peelingResult.mainLoop, kPerfectlyTiledLoopLabel);

    return success();
  }

 private:
  MatmulTileSizeComputationFn tileSizeFn;
  std::function<SmallVector<int64_t>(MatmulSizes)> parallelDimTileSizeFn;
  std::function<SmallVector<int64_t>(MatmulSizes)> reductionDimTileSizeFn;
};

Value transposeMatrixConstant(ImplicitLocOpBuilder &builder, Value input) {
  ElementsAttr inputValues;
  matchPattern(input, m_Constant(&inputValues));

  auto inputType = input.getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  assert(inputShape.size() == 2);

  auto outputType = RankedTensorType::get({inputShape[1], inputShape[0]},
                                          inputType.getElementType());

  SmallVector<Attribute, 4> outputValues(inputType.getNumElements());
  for (const auto &it : llvm::enumerate(inputValues.getValues<Attribute>())) {
    auto row = it.index() / inputShape[1];
    auto col = it.index() % inputShape[1];
    outputValues[col * inputShape[0] + row] = it.value();
  }
  return builder.create<arith::ConstantOp>(
      outputType, DenseElementsAttr::get(outputType, outputValues));
}

// If we have a matvec with a constant matrix it's profitable to transpose the
// matrix at compile time and use vecmat instead. This has a friendlier memory
// access pattern.
struct MatVecToVecMatPattern : public OpRewritePattern<linalg::MatvecOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatvecOp matvecOp,
                                PatternRewriter &rewriter) const override {
    auto constantMatrix =
        matvecOp.getOperand(0).getDefiningOp<arith::ConstantOp>();
    if (!constantMatrix) return failure();

    ImplicitLocOpBuilder builder(constantMatrix.getLoc(), rewriter);
    Value transposed = transposeMatrixConstant(builder, constantMatrix);
    rewriter.replaceOpWithNewOp<linalg::VecmatOp>(
        matvecOp, ValueRange{matvecOp.getOperand(1), transposed},
        matvecOp.getOutputs());
    return success();
  }
};

struct TransformDotForCpuPass
    : public impl::TransformDotForCpuPassBase<TransformDotForCpuPass> {
  TransformDotForCpuPass() = default;

  explicit TransformDotForCpuPass(MatmulTileSizeComputationFn tileSizeFn)
      : tileSizeFn(std::move(tileSizeFn)) {}

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<mlir::gml_st::GmlStDialect, arith::ArithDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
    tensor::registerTilingInterfaceExternalModels(registry);
    tensor::registerInferTypeOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    // Dot operations can have at most 3 dimensions ((upto) 2 parallel + 1
    // reduction), so the first two tileSizes' elements are for parallel
    // dimensions tiling, and the last element is for reduction dimension
    // tiling.
    // - for linalg.matmul: the whole tileSizes vector will be used.
    // - for linalg.matvec: only the first and last elements of tileSizes are
    // used.
    // - for linalg.vecmat: only the second and last elements of tileSizes are
    // used.
    // - for linalg.dot: only the last element of tileSizes is used.

    RewritePatternSet patterns(ctx);
    patterns.add<MatVecToVecMatPattern>(ctx, 2);
    patterns.add<DotTransformPattern<linalg::MatvecOp>>(
        ctx, tileSizeFn,
        [&](MatmulSizes sizes) -> SmallVector<int64_t> {
          return {sizes.m, 0};
        },
        [&](MatmulSizes sizes) -> SmallVector<int64_t> {
          return {0, sizes.k};
        });
    patterns.add<DotTransformPattern<linalg::VecmatOp>>(
        ctx, tileSizeFn,
        [&](MatmulSizes sizes) -> SmallVector<int64_t> {
          return {sizes.n, 0};
        },
        [&](MatmulSizes sizes) -> SmallVector<int64_t> {
          return {0, sizes.k};
        });
    patterns.add<DotTransformPattern<linalg::DotOp>>(
        ctx, tileSizeFn,
        [&](MatmulSizes) -> SmallVector<int64_t> { return {}; },
        [&](MatmulSizes sizes) -> SmallVector<int64_t> { return {sizes.k}; });

    populateCollapseForallOpDimensionsPattern(patterns);
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Ensure we drop the marker in the end.
    f.walk([](Operation *op) {
      if (isa<linalg::MatvecOp, linalg::VecmatOp, linalg::DotOp>(op))
        removeLabel(op, kDotTransformedLabel);
    });
  }

  MatmulTileSizeComputationFn tileSizeFn;
};
}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformDotForCpuPass(MatmulTileSizeComputationFn tileSizeFn) {
  return std::make_unique<mlir::gml_st::TransformDotForCpuPass>(
      std::move(tileSizeFn));
}

}  // namespace mlir::gml_st
