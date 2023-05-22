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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/peeling/peeling.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMREDUCEFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

struct Reduce1DTileSizes {
  int64_t tileSize;
  int64_t splitRatio;
};
using Reduce1DTileSizeComputationFn = std::function<Reduce1DTileSizes(int64_t)>;

SmallVector<int64_t> getParallelDimTileSizes(int64_t reductionDim,
                                             int64_t parallelDimTileSize) {
  return reductionDim ? SmallVector<int64_t>{parallelDimTileSize, 0}
                      : SmallVector<int64_t>{0, parallelDimTileSize};
}

SmallVector<int64_t> getReductionDimTileSizes(int64_t reductionDim,
                                              int64_t reductionDimTileSize) {
  return reductionDim ? SmallVector<int64_t>{0, reductionDimTileSize}
                      : SmallVector<int64_t>{reductionDimTileSize, 0};
}

LogicalResult validateOp(linalg::ReduceOp reduceOp, PatternRewriter &rewriter,
                         int64_t expectedRank) {
  ArrayRef<int64_t> reduceDimensions = reduceOp.getDimensions();
  if (reduceDimensions.size() != 1) {
    return rewriter.notifyMatchFailure(
        reduceOp, "expects 1 reduction dimension element. 0 or > 1 received.");
  }
  OpOperandVector operands = reduceOp.getDpsInputOperands();
  if (operands.size() != 1) {
    return rewriter.notifyMatchFailure(reduceOp,
                                       "expects 1 operand. 0 or > 1 received.");
  }
  const int64_t operandRank =
      operands[0]->get().getType().cast<RankedTensorType>().getRank();
  if (operandRank != expectedRank) {
    return rewriter.notifyMatchFailure(reduceOp, [&](Diagnostic &diag) {
      diag << "expects rank " << expectedRank << ". " << operandRank
           << "received.";
    });
  }
  return success();
}

struct Reduce1DTransformPattern : public OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  explicit Reduce1DTransformPattern(MLIRContext *context,
                                    Reduce1DTileSizeComputationFn tileSizeFn,
                                    PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::ReduceOp>(context, benefit),
        tileSizeFn(std::move(tileSizeFn)) {}

  LogicalResult matchAndRewrite(linalg::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(reduceOp, kTransformedLabel)) {
      return rewriter.notifyMatchFailure(reduceOp,
                                         "has already been transformed.");
    }
    if (failed(validateOp(reduceOp, rewriter, /*expectedRank=*/1)))
      return failure();

    int64_t inputSize =
        reduceOp.getOperand(0).getType().cast<RankedTensorType>().getDimSize(0);
    Reduce1DTileSizes tileSizes = tileSizeFn(inputSize);

    // Rewrite as a tree reduction.
    FailureOr<SplitReduce1DResult> splitReduce = rewriteReduce1D(
        rewriter, reduceOp, tileSizes.tileSize, tileSizes.splitRatio);
    if (failed(splitReduce)) {
      return rewriter.notifyMatchFailure(reduceOp,
                                         "failed to split reduction dimension");
    }
    scf::ForOp mainLoop = splitReduce->mainLoop;
    scf::ForOp tailLoop = splitReduce->tailLoop;

    // Fusion.
    auto fusionFilterFn = [](Operation *op) {
      return isa<linalg::FillOp, linalg::MapOp, thlo::ReverseOp>(op);
    };
    SmallVector<Block *> blocks;
    if (mainLoop) blocks.push_back(mainLoop.getBody());
    if (tailLoop) blocks.push_back(tailLoop.getBody());
    fuseGreedily(rewriter, blocks, fusionFilterFn);

    // Tiling to 1 and fusion in the tail loop.
    if (tailLoop) {
      for (auto reduOp :
           llvm::to_vector(tailLoop.getBody()->getOps<linalg::ReduceOp>())) {
        if (failed(tileUsingSCFForOpAndFuseGreedily(
                rewriter, reduOp, getSCFTilingOptions({1}), fusionFilterFn))) {
          return failure();
        }
      }
    }
    return success();
  }

 private:
  struct SplitReduce1DResult {
    scf::ForOp mainLoop;
    scf::ForOp tailLoop;
    linalg::ReduceOp horizontalReduce;
    Value result;
  };
  // Split reduction tensor<N*tile_size+M x elem_type> -> tensor<elem_type> into
  //  * scf.for that reduces
  //    tensor<N*tile_size> -> tensor<split_ratio x elem_type>
  //  * horizontal reduce tensor<split_ratio x elem_type> -> tensor<elem_type>
  //  * scf.for that reduces the remaining M elements.
  FailureOr<SplitReduce1DResult> rewriteReduce1D(PatternRewriter &rewriter,
                                                 linalg::ReduceOp reduceOp,
                                                 int64_t tileSize,
                                                 int64_t splitRatio) const {
    // 0-d tensor with the neutral elements.
    auto fillOp = reduceOp.getInits().front().getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return rewriter.notifyMatchFailure(reduceOp,
                                         "init not defined by fill op");
    auto neutralValue = fillOp.value();

    // Constants.
    Location loc = reduceOp.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value tileSizeValue =
        rewriter.create<arith::ConstantIndexOp>(loc, tileSize);

    // Input.
    Value input = reduceOp.getInputs().front();
    FailureOr<OpFoldResult> inputSizeOfr =
        tensor::createDimValue(rewriter, loc, input, 0);
    if (failed(inputSizeOfr))
      return rewriter.notifyMatchFailure(reduceOp, "cannot get input size");

    // Loop boundaries.
    //   tileableBound = inputSize - inputSize % tileSize
    //   remainderSize = inputSize - tileableBound
    OpFoldResult tileableBoundOfr =
        getTileableBound(rewriter, loc, *inputSizeOfr, tileSize);
    Value tileableBoundValue =
        getValueOrCreateConstantIndexOp(rewriter, loc, tileableBoundOfr);

    OpFoldResult remainderSize =
        getRemainderSize(rewriter, loc, tileableBoundOfr, *inputSizeOfr);

    // Create tensor<SPLIT_RATIOxELEM_TYPE> with neutral elements for tile loop
    // init.
    Type elementType = neutralValue.getType();
    Value emptyVector = rewriter.create<tensor::EmptyOp>(
        loc, llvm::ArrayRef({splitRatio}), elementType);
    Value filledVector =
        rewriter.create<linalg::FillOp>(loc, neutralValue, emptyVector)
            .getResult(0);

    // Create a tiled loop
    SplitReduce1DResult splitResult;
    splitResult.result = fillOp.getResult(0);

    std::optional<int64_t> tileableBoundConstant =
        getConstantIntValue(tileableBoundOfr);
    if (!tileableBoundConstant || tileableBoundConstant != 0) {
      auto tiledLoopBodyBuilder = [&](OpBuilder &b, Location loc, Value iv,
                                      ValueRange inits) {
        // Tile input as tensor<TILE_SIZExELEM_TYPE> and reshape into
        // tensor<(TILE_SIZE/SPLIT_RATIO)xSPLIT_RATIOxELEM_TYPE>.
        Value inputSlice = tileAndReshapeInput(b, loc, iv, input, elementType,
                                               tileSize, splitRatio);

        tensor::ExtractSliceOp initSlice =
            create1DSlice(b, loc, inits.front(), b.getIndexAttr(0),
                          b.getIndexAttr(splitRatio));

        // Create `linalg.reduce` to combine
        // `tensor<(TILE_SIZE/SPLIT_RATIO)xSPLIT_RATIOxELEM_TYPE> input with the
        // `tensor<SPLIT_RATIOxELEM_TYPE>` accumulator.
        auto tiledReduceOp = b.create<linalg::ReduceOp>(
            loc, ValueRange{inputSlice}, ValueRange{initSlice},
            /*dimensions=*/SmallVector<int64_t>{0},
            /*bodyBuilder=*/nullptr, linalg::getPrunedAttributeList(reduceOp));
        OpBuilder::InsertionGuard g(rewriter);
        Region &region = tiledReduceOp.getRegion();
        rewriter.cloneRegionBefore(reduceOp.getRegion(), region, region.end());
        setLabel(tiledReduceOp, kTransformedLabel);

        b.create<scf::YieldOp>(loc, tiledReduceOp.getResults());
      };

      splitResult.mainLoop = rewriter.create<scf::ForOp>(
          loc, zero, tileableBoundValue, tileSizeValue, filledVector,
          tiledLoopBodyBuilder);
      setLabel(splitResult.mainLoop, kPerfectlyTiledLoopLabel);

      // Create `linalg.reduce` from tensor<SPLIT_RATIOxELEM_TYPE> to
      // tensor<ELEM_TYPE>.
      splitResult.horizontalReduce =
          cloneReduceOp(rewriter, reduceOp, splitResult.mainLoop.getResult(0),
                        reduceOp.getInits().front());
      splitResult.result = splitResult.horizontalReduce.getResult(0);
    }

    // Combine `horizontal reduce` with the tail of the input. The tail is
    // always smaller than TILE_SIZE.
    std::optional<int64_t> tripCount = constantTripCount(
        tileableBoundOfr, *inputSizeOfr, rewriter.getIndexAttr(tileSize));
    scf::ForOp remainderLoop;
    if (!tripCount || *tripCount > 0) {
      auto remainderLoopBodyBuilder = [&](OpBuilder &b, Location loc, Value iv,
                                          ValueRange inits) {
        Value inputSlice = create1DSlice(b, loc, input, iv, remainderSize);

        Value initSlice = b.create<tensor::ExtractSliceOp>(
            loc, inits.front(), /*offsets=*/SmallVector<OpFoldResult>{},
            /*sizes=*/SmallVector<OpFoldResult>{},
            /*strides=*/SmallVector<OpFoldResult>{});

        linalg::ReduceOp newReduce =
            cloneReduceOp(b, reduceOp, inputSlice, initSlice);
        b.create<scf::YieldOp>(loc, newReduce->getResults());
      };
      splitResult.tailLoop = rewriter.create<scf::ForOp>(
          loc, tileableBoundValue,
          getValueOrCreateConstantIndexOp(rewriter, loc, *inputSizeOfr),
          tileSizeValue, splitResult.result, remainderLoopBodyBuilder);
      splitResult.result = splitResult.tailLoop.getResult(0);
    }
    rewriter.replaceOp(reduceOp, splitResult.result);
    return splitResult;
  }

  OpFoldResult getTileableBound(OpBuilder &b, Location loc,
                                OpFoldResult inputSizeOfr,
                                int64_t tileSize) const {
    if (tileSize == 1) return inputSizeOfr;

    auto inputSizeInt = getConstantIntValue(inputSizeOfr);
    if (inputSizeInt && *inputSizeInt < tileSize) return b.getIndexAttr(0);

    AffineExpr sym0;
    bindSymbols(b.getContext(), sym0);

    auto modMap = AffineMap::get(0, 1, {sym0 - sym0 % tileSize});
    return makeComposedFoldedAffineApply(b, loc, modMap, inputSizeOfr);
  }

  OpFoldResult getRemainderSize(OpBuilder &b, Location loc,
                                OpFoldResult tileableBoundOfr,
                                OpFoldResult inputSize) const {
    AffineExpr sym0, sym1;
    bindSymbols(b.getContext(), sym0, sym1);
    auto diffMap = AffineMap::get(0, 2, {sym1 - sym0});
    return makeComposedFoldedAffineApply(b, loc, diffMap,
                                         {tileableBoundOfr, inputSize});
  }

  tensor::ExtractSliceOp create1DSlice(OpBuilder &b, Location loc, Value source,
                                       OpFoldResult offset,
                                       OpFoldResult size) const {
    SmallVector<OpFoldResult> offsets{offset};
    SmallVector<OpFoldResult> sizes{size};
    SmallVector<OpFoldResult> strides{b.getIndexAttr(1)};

    return b.create<tensor::ExtractSliceOp>(loc, source, offsets, sizes,
                                            strides);
  }

  linalg::ReduceOp cloneReduceOp(OpBuilder &b, linalg::ReduceOp reduceOp,
                                 ValueRange newInputs, Value newInit) const {
    IRMapping bvm;
    bvm.map(reduceOp.getInputs(), newInputs);
    bvm.map(reduceOp.getInits(), ValueRange{newInit});

    auto *newReduceOp = b.clone(*reduceOp.getOperation(), bvm);
    setLabel(newReduceOp, kTransformedLabel);
    return cast<linalg::ReduceOp>(newReduceOp);
  }

  Value tileAndReshapeInput(OpBuilder &b, Location loc, Value iv, Value input,
                            Type elementType, int64_t tileSize,
                            int64_t splitRatio) const {
    Value inputSlice =
        create1DSlice(b, loc, input, iv, b.getIndexAttr(tileSize));

    auto reshapeType =
        RankedTensorType::get({tileSize / splitRatio, splitRatio}, elementType);
    SmallVector<ReassociationIndices> ri = {{0, 1}};
    return b.create<tensor::ExpandShapeOp>(loc, reshapeType, inputSlice, ri);
  }

  Reduce1DTileSizeComputationFn tileSizeFn;
};

/// Pattern to tile `linalg.reduce` and fuse `linalg.fill` into generated
/// `scf.forall`.
struct Reduce2DTransformPattern : public OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  explicit Reduce2DTransformPattern(MLIRContext *context,
                                    int64_t parallelDimTileSize = 4,
                                    int64_t reductionDimTileSize = 2,
                                    PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::ReduceOp>(context, benefit),
        parallelDimTileSize(parallelDimTileSize),
        reductionDimTileSize(reductionDimTileSize) {}

  LogicalResult matchAndRewrite(linalg::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (reduceOp.getDimensions().size() != 1) return failure();
    int64_t reductionDim = reduceOp.getDimensions()[0];

    if (hasLabel(reduceOp, kTransformedLabel)) {
      return rewriter.notifyMatchFailure(reduceOp,
                                         "has already been transformed.");
    }
    if (isa<scf::ForallOp, scf::ForOp>(reduceOp->getParentOp())) {
      return rewriter.notifyMatchFailure(
          reduceOp, "has already been tiled by another pass.");
    }
    if (failed(validateOp(reduceOp, rewriter, /*expectedRank=*/2)))
      return failure();

    auto producerFilterFn = [](Operation *op) {
      return isa<linalg::BroadcastOp, linalg::FillOp, linalg::MapOp,
                 linalg::TransposeOp, tensor::CastOp>(op);
    };
    auto consumerFilterFn = [](Operation *op) {
      return isa<linalg::MapOp, thlo::ReverseOp>(op);
    };
    auto fusionClusterFn = [&](Operation *op) {
      return producerFilterFn(op) || isa<linalg::ReduceOp>(op);
    };
    auto cluster =
        getFusionCluster(reduceOp, producerFilterFn, consumerFilterFn);
    auto fusionCluster = cluster.operations;
    auto *tilingRoot = cluster.root;

    // First level tiling: parallel dimension.
    auto parallelDimsTileSizes =
        isa<linalg::ReduceOp>(tilingRoot)
            ? getParallelDimTileSizes(reduceOp.getDimensions()[0],
                                      parallelDimTileSize)
            : SmallVector<int64_t>{parallelDimTileSize};
    auto tilingParallelDimsResult = tileUsingSCFForallOpAndFuseGreedily(
        rewriter, tilingRoot, getSCFTilingOptions(parallelDimsTileSizes),
        [&](Operation *op) { return fusionCluster.contains(op); });
    if (failed(tilingParallelDimsResult)) return failure();

    auto peeledParallelLoop =
        peelAllLoops(tilingParallelDimsResult->loop, rewriter);

    // Process main parallel loop.
    scf::ForallOp mainParallelLoop = peeledParallelLoop.mainLoop;
    if (mainParallelLoop) {
      auto tiledReduceOp =
          *mainParallelLoop.getBody()->getOps<linalg::ReduceOp>().begin();
      if (failed(tileAndPeelReductionDim(rewriter, tiledReduceOp, reductionDim,
                                         producerFilterFn))) {
        return failure();
      }
    }

    // Process tail parallel loop.
    scf::ForallOp tailParallelLoop = peeledParallelLoop.tailLoops.size() == 1
                                         ? peeledParallelLoop.tailLoops.front()
                                         : nullptr;
    if (tailParallelLoop) {
      Value yieldedTensor =
          getYieldedValues(tailParallelLoop.getTerminator()).front();
      auto *definingOp = yieldedTensor.getDefiningOp();
      if (!definingOp) return failure();

      auto opts = getSCFTilingOptions(SmallVector<int64_t>(
          definingOp->getResult(0).getType().cast<RankedTensorType>().getRank(),
          1));
      auto parallelDimTilingOpts =
          isa<linalg::ReduceOp>(definingOp)
              ? getSCFTilingOptions(getParallelDimTileSizes(reductionDim, 1))
              : getSCFTilingOptions({1});
      auto parallelDimTilingResult = tileUsingSCFForallOpAndFuseGreedily(
          rewriter, definingOp, parallelDimTilingOpts, fusionClusterFn);
      if (failed(parallelDimTilingResult)) return failure();

      for (auto tiledReduceOp :
           llvm::to_vector(parallelDimTilingResult->loop.getBody()
                               ->getOps<linalg::ReduceOp>())) {
        auto reductionDimTilingResult = tileUsingSCFForOpAndFuseGreedily(
            rewriter, tiledReduceOp,
            getSCFTilingOptions(getReductionDimTileSizes(reductionDim, 1)),
            producerFilterFn);
        if (failed(reductionDimTilingResult)) return failure();
      }
    }

    return success();
  }

 private:
  LogicalResult tileAndPeelReductionDim(
      PatternRewriter &rewriter, linalg::ReduceOp reduceOp,
      int64_t reductionDim,
      llvm::function_ref<bool(Operation *)> producerFilterFn) const {
    FailureOr<scf::SCFTilingResult> reductionDimTilingResult =
        tileUsingSCFForOpAndFuseGreedily(
            rewriter, reduceOp,
            getSCFTilingOptions(
                getReductionDimTileSizes(reductionDim, reductionDimTileSize)),
            producerFilterFn);
    if (failed(reductionDimTilingResult)) return failure();

    SCFForPeelingResult reductionDimPeelingResult =
        peelSCFForOp(rewriter, reductionDimTilingResult->loops.front());
    if (reductionDimPeelingResult.mainLoop) {
      setLabel(reductionDimPeelingResult.mainLoop, kPerfectlyTiledLoopLabel);
    }
    if (reductionDimPeelingResult.tailLoop) {
      for (auto reduOp :
           llvm::to_vector(reductionDimPeelingResult.tailLoop.getBody()
                               ->getOps<linalg::ReduceOp>())) {
        // Column reductions have to be tiled even further, otherwise we
        // would get vector.multi_reduction 4x1 -> 1, which is expensive.
        // Potentially, we could lower it to a horizontal add.
        if (reductionDim == 0) {
          auto parallelDimSizeOneTilingResult =
              tileUsingSCFForOpAndFuseGreedily(
                  rewriter, reduOp,
                  getSCFTilingOptions(getParallelDimTileSizes(reductionDim, 1)),
                  producerFilterFn);
          if (failed(parallelDimSizeOneTilingResult)) return failure();

          reduOp = cast<linalg::ReduceOp>(
              parallelDimSizeOneTilingResult->tiledOps.front());
        }
        if (failed(tileUsingSCFForOpAndFuseGreedily(
                rewriter, reduOp,
                getSCFTilingOptions(getReductionDimTileSizes(reductionDim, 1)),
                producerFilterFn))) {
          return failure();
        }
      }
    }
    return success();
  }

  int64_t parallelDimTileSize;
  int64_t reductionDimTileSize;
};

struct TransformReduceForCpuPass
    : public impl::TransformReduceForCpuPassBase<TransformReduceForCpuPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    Reduce1DTileSizeComputationFn tilingHeuristic;
    if (enableHeuristic) {
      tilingHeuristic = [](int64_t size) {
        if (!ShapedType::isDynamic(size) && size > 96)
          return Reduce1DTileSizes{32, 8};
        return Reduce1DTileSizes{8, 8};
      };
    } else {
      tilingHeuristic = [=](int64_t) {
        return Reduce1DTileSizes{tileSize1D, splitRatio1D};
      };
    }
    patterns.add<Reduce1DTransformPattern>(ctx, std::move(tilingHeuristic));
    patterns.add<Reduce2DTransformPattern>(ctx, parallelDimTileSize2D,
                                           reductionDimTileSize2D);
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createTransformReduceForCpuPass(
    const TransformReduceForCpuPassOptions &opts) {
  return std::make_unique<TransformReduceForCpuPass>(opts);
}

}  // namespace mlir::gml_st
