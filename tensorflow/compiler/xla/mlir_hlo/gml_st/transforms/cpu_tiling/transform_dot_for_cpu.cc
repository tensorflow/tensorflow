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
#include "gml_st/transforms/fusion/fusion.h"
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
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMDOTFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

struct MatmulSizes {
  // [m, k] x [k, n]
  int64_t m;
  int64_t n;
  int64_t k;
};

using MatmulTileSizeComputationFn = std::function<MatmulSizes(MatmulSizes)>;

int64_t roundDownToPowerOfTwo(int64_t n) {
  if ((n & (n - 1)) == 0) return n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  return (n + 1) >> 1;
}

// Tiling heuristic that was tuned for static power-of-two sized shapes on
// Skylake.
MatmulSizes skylakeTilingHeuristic(MatmulSizes sizes) {
  if (sizes.m == 1) {
    // Limit the maximum tiling to an arbitrary 32 to limit code growth. This
    // needs re-tuning.
    return {1, std::min<int64_t>(sizes.n, 32), 1};
  }

  if (sizes.n == 1) {
    if (sizes.k <= 8) {
      return {1, 1, 1};
    }
    return {std::min<int64_t>(8, sizes.m), 1, 4};
  }

  MatmulSizes result;
  result.k = sizes.k <= 8 ? 1 : 4;
  result.n = std::min<int64_t>(8, sizes.n) << (sizes.m <= 16 ? 1 : 0);
  result.m = std::min<int64_t>(32, sizes.m) << (sizes.n <= 4 ? 1 : 0);
  return result;
}

// Tiling heuristic that was tuned for static power-of-two sized shapes on Zen
// v2 ("Rome").
MatmulSizes znver2TilingHeuristic(MatmulSizes sizes) {
  MatmulSizes result;
  result.k = sizes.n == 1 ? 8 : 1;
  if (sizes.n == 1) {
    result.m = sizes.k >= 32 ? 16 : 8;
  } else {
    result.m = sizes.n <= 8 ? 8 : 4;
  }
  if (sizes.m == 1) {
    result.n = std::min<int64_t>(64, sizes.n) * (sizes.k <= 64 ? 1 : 2);
  } else {
    result.n = std::min<int64_t>(16, sizes.n);
  }
  return result;
}

std::function<MatmulSizes(MatmulSizes)> wrapHeuristic(
    const std::function<MatmulSizes(MatmulSizes)> &heuristic,
    MatmulSizes dynamicDefault) {
  return [=](MatmulSizes sizes) {
    if (sizes.n < 0 || sizes.m < 0 || sizes.k < 0) {
      return dynamicDefault;
    }

    sizes.m = roundDownToPowerOfTwo(sizes.m);
    sizes.n = roundDownToPowerOfTwo(sizes.n);
    sizes.k = roundDownToPowerOfTwo(sizes.k);

    return heuristic(sizes);
  };
}

MatmulSizes getMatmulSizes(linalg::MatmulOp op) {
  // [m, k] x [k, n]
  ShapedType lhsTy = op->getOperand(0).getType().cast<ShapedType>();
  ShapedType rhsTy = op->getOperand(1).getType().cast<ShapedType>();
  MatmulSizes sizes;
  sizes.m = lhsTy.getDimSize(0);
  sizes.k = rhsTy.getDimSize(0);
  sizes.n = rhsTy.getDimSize(1);
  return sizes;
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

SmallVector<int64_t> dropZeros(ArrayRef<int64_t> tileSizes) {
  return to_vector(llvm::make_filter_range(
      tileSizes, [](int64_t size) { return size != 0; }));
}

struct DotAddTransformPattern : public OpRewritePattern<linalg::MapOp> {
  using OpRewritePattern<linalg::MapOp>::OpRewritePattern;

  explicit DotAddTransformPattern(MLIRContext *context,
                                  PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MapOp>(context, benefit) {}

  LogicalResult matchAndRewrite(linalg::MapOp mapOp,
                                PatternRewriter &rewriter) const override {
    auto &region = mapOp.getMapper();
    if (!region.hasOneBlock()) return failure();

    auto &body = region.front();
    // The body region should only have one add operation and a linalg.yield.
    if (body.getOperations().size() != 2) return failure();

    auto &mapperOp = body.front();
    if (!isa<arith::AddIOp, arith::AddFOp>(mapperOp)) return failure();

    // Map of add should always be binary.
    if (mapOp.getInputs().size() != 2) return failure();
    if (ValueRange{body.getArguments()} != ValueRange{mapperOp.getOperands()})
      return failure();

    if (!llvm::any_of(mapOp.getInputs(), [](Value operand) {
          auto linalgOp = operand.getDefiningOp<linalg::LinalgOp>();
          return linalg::isaContractionOpInterface(linalgOp);
        }))
      return failure();

    auto foldAddIntoDotOperand = [&](unsigned opIdx) {
      auto dotOp = mapOp.getInputs()[opIdx].getDefiningOp<linalg::LinalgOp>();
      auto otherOp = mapOp.getInputs()[1 - opIdx];
      if (!linalg::isaContractionOpInterface(dotOp)) return false;
      if (!dotOp.getDpsInitOperand(0)->get().getDefiningOp<linalg::FillOp>())
        return false;
      if (!dotOp->hasOneUse()) return false;
      // TODO(vuson): handle the case where we need to move dotOp up or otherOp
      // down.
      mlir::DominanceInfo domInfo(mapOp->getParentOp());
      if (!domInfo.properlyDominates(otherOp, dotOp)) return false;
      rewriter.updateRootInPlace(
          dotOp, [&]() { dotOp.setDpsInitOperand(0, otherOp); });
      rewriter.replaceOp(mapOp, dotOp->getResults());
      return true;
    };

    return success(foldAddIntoDotOperand(0) || foldAddIntoDotOperand(1));
  }
};

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
    if (hasLabel(dotOp, kTransformedLabel)) {
      return rewriter.notifyMatchFailure(dotOp,
                                         "has already been transformed.");
    }
    auto producerFilterFn = [](Operation *op) {
      return isa<linalg::FillOp, thlo::ReverseOp, tensor::CastOp>(op);
    };
    auto consumerFilterFn = [](Operation *op) {
      if (auto mapOp = dyn_cast<linalg::MapOp>(op))
        return mapOp.getNumDpsInputs() == 1;
      return isa<thlo::ReverseOp>(op);
    };

    auto cluster = getFusionCluster(dotOp, producerFilterFn, consumerFilterFn);
    auto fusionCluster = cluster.operations;
    auto *tilingRoot = cluster.root;

    // First level tiling: parallel dimension.
    auto tileSizes = tileSizeFn(getMatmulSizes(dotOp));
    auto parallelDimsTileSizes = parallelDimTileSizeFn(tileSizes);
    if (!isa<DotTy>(tilingRoot))
      parallelDimsTileSizes = dropZeros(parallelDimsTileSizes);

    auto tilingParallelDimsResult = tileUsingSCFForallOpAndFuseGreedily(
        rewriter, tilingRoot, getSCFTilingOptions(parallelDimsTileSizes),
        [&](Operation *op) { return fusionCluster.contains(op); });
    if (failed(tilingParallelDimsResult)) return failure();

    if (!tilingParallelDimsResult->loop) {
      return tileAndPeelReductionDim(
          rewriter, dotOp, reductionDimTileSizeFn(tileSizes), producerFilterFn);
    }
    auto peeledParallelLoop =
        peelAllLoops(tilingParallelDimsResult->loop, rewriter);

    // Process main parallel loop.
    scf::ForallOp mainParallelLoop = peeledParallelLoop.mainLoop;
    if (mainParallelLoop) {
      auto tiledDotOp = *mainParallelLoop.getBody()->getOps<DotTy>().begin();
      if (failed(tileAndPeelReductionDim(rewriter, tiledDotOp,
                                         reductionDimTileSizeFn(tileSizes),
                                         producerFilterFn))) {
        return failure();
      }
    }

    // Process tail parallel loop.
    for (scf::ForallOp tailParallelLoop : peeledParallelLoop.tailLoops) {
      for (auto tiledDotOp : llvm::to_vector(
               tailParallelLoop.getBody()->template getOps<DotTy>())) {
        auto reductionDimTilingResult = tileUsingSCFForOpAndFuseGreedily(
            rewriter, tiledDotOp,
            getSCFTilingOptions(reductionDimTileSizeFn(tileSizes)),
            producerFilterFn);
        if (failed(reductionDimTilingResult)) return failure();
      }
    }
    return success();
  }

 private:
  LogicalResult tileAndPeelReductionDim(
      PatternRewriter &rewriter, DotTy reduceOp,
      ArrayRef<int64_t> reductionDimTileSizes,
      llvm::function_ref<bool(Operation *)> producerFilterFn) const {
    FailureOr<scf::SCFTilingResult> reductionDimTilingResult =
        tileUsingSCFForOpAndFuseGreedily(
            rewriter, reduceOp, getSCFTilingOptions(reductionDimTileSizes),
            producerFilterFn);
    if (failed(reductionDimTilingResult)) return failure();

    SCFForPeelingResult reductionDimPeelingResult =
        peelSCFForOp(rewriter, reductionDimTilingResult->loops.front());
    if (reductionDimPeelingResult.mainLoop) {
      setLabel(reductionDimPeelingResult.mainLoop, kPerfectlyTiledLoopLabel);
    }
    return success();
  }

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

    // Peephole optimization of dot followed by add.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<DotAddTransformPattern>(ctx);

      if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

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
    patterns.add<DotTransformPattern<linalg::MatmulOp>>(
        ctx, tileSizeFn,
        [&](MatmulSizes sizes) -> SmallVector<int64_t> {
          return {sizes.m, sizes.n, 0};
        },
        [&](MatmulSizes sizes) -> SmallVector<int64_t> {
          return {0, 0, sizes.k};
        });
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

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns))))
      return signalPassFailure();
  }

  MatmulTileSizeComputationFn tileSizeFn;
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformDotForCpuPass(ArrayRef<int64_t> tileSizes, StringRef cpuName) {
  std::function<MatmulSizes(MatmulSizes)> tilingHeuristic;
  if (!tileSizes.empty()) {
    assert(tileSizes.size() == 3 && "Expected exactly 3 tile sizes for matmul");
    MatmulSizes fixedSizes{tileSizes[0], tileSizes[1], tileSizes[2]};
    tilingHeuristic = [=](MatmulSizes) { return fixedSizes; };
  } else {
    tilingHeuristic = cpuName.starts_with("znver")
                          ? wrapHeuristic(znver2TilingHeuristic, {16, 8, 8})
                          : wrapHeuristic(skylakeTilingHeuristic, {16, 16, 4});
  }
  return std::make_unique<mlir::gml_st::TransformDotForCpuPass>(
      std::move(tilingHeuristic));
}

}  // namespace mlir::gml_st
