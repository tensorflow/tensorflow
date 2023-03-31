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
#include <memory>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/peeling/peeling.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMREDUCEFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

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
    return rewriter.notifyMatchFailure(reduceOp, [&](::mlir::Diagnostic &diag) {
      diag << "expects rank " << expectedRank << ". " << operandRank
           << "received.";
    });
  }
  return success();
}

struct Reduce1DTransformPattern : public OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  explicit Reduce1DTransformPattern(MLIRContext *context, int64_t vectorSize,
                                    int64_t tileSize,
                                    PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::ReduceOp>(context, benefit),
        vectorSize(vectorSize),
        tileSize(tileSize) {}

  LogicalResult matchAndRewrite(linalg::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(reduceOp, kTransformedLabel)) {
      return rewriter.notifyMatchFailure(reduceOp,
                                         "has already been transformed.");
    }

    if (failed(validateOp(reduceOp, rewriter, /*expectedRank=*/1)))
      return failure();

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
    Value inputSize = rewriter.create<tensor::DimOp>(loc, input, 0);

    // Loop boundaries.
    //   tileableBound = inputSize - inputSize % tileSize
    //   remainderSize = inputSize - tileableBound
    Value tileableBound = getTileableBound(rewriter, loc, inputSize);
    Value remainderSize =
        getRemainderSize(rewriter, loc, tileableBound, inputSize);

    // Create tensor<VECTOR_SIZExELEM_TYPE> with neutral elements for tile loop
    // init.
    Type elementType = neutralValue.getType();
    Value emptyVector = rewriter.create<tensor::EmptyOp>(
        loc, llvm::ArrayRef({vectorSize}), elementType);
    Value filledVector =
        rewriter.create<linalg::FillOp>(loc, neutralValue, emptyVector)
            .getResult(0);

    auto tiledLoopBodyBuilder = [&](OpBuilder &b, Location loc, Value iv,
                                    ValueRange inits) {
      // Tile input as tensor<TILE_SIZExELEM_TYPE> and reshape into
      // tensor<(TILE_SIZE/VECTOR_SIZE)xVECTOR_SIZExELEM_TYPE>.
      Value inputSlice = tileAndReshapeInput(b, loc, iv, input, elementType);

      tensor::ExtractSliceOp initSlice = create1DSlice(
          b, loc, inits.front(), b.getIndexAttr(0), b.getIndexAttr(vectorSize));

      // Create `linalg.reduce` to combine
      // `tensor<(TILE_SIZE/VECTOR_SIZE)xVECTOR_SIZExELEM_TYPE> input with the
      // `tensor<VECTOR_SIZExELEM_TYPE>` accumulator.
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

    // Create a tiled loop
    auto tiledLoop =
        rewriter.create<scf::ForOp>(loc, zero, tileableBound, tileSizeValue,
                                    filledVector, tiledLoopBodyBuilder);
    setLabel(tiledLoop, kPerfectlyTiledLoopLabel);

    // Create `linalg.reduce` from tensor<VECTOR_SIZExELEM_TYPE> to
    // tensor<ELEM_TYPE>.
    auto horizontalReduce =
        cloneReduceOp(rewriter, reduceOp, tiledLoop.getResult(0),
                      reduceOp.getInits().front());

    auto remainderLoopBodyBuilder = [&](OpBuilder &b, Location loc, Value iv,
                                        ValueRange inits) {
      Value inputSlice = create1DSlice(b, loc, input, iv, remainderSize);

      Value initSlice = b.create<tensor::ExtractSliceOp>(
          loc, inits.front(), /*offsets=*/SmallVector<OpFoldResult>{},
          /*sizes=*/SmallVector<OpFoldResult>{},
          /*strides=*/SmallVector<OpFoldResult>{});

      auto newReduce = cloneReduceOp(b, reduceOp, inputSlice, initSlice);
      b.create<scf::YieldOp>(loc, newReduce);
    };

    // Combine `horizontal reduce` with the tail of the input. The tail is
    // always smaller than TILE_SIZE.
    auto remainderLoop =
        rewriter
            .create<scf::ForOp>(loc, tileableBound, inputSize, tileSizeValue,
                                horizontalReduce, remainderLoopBodyBuilder)
            .getResult(0);

    rewriter.replaceOp(reduceOp, remainderLoop);

    return success();
  }

 private:
  Value getTileableBound(OpBuilder &b, Location loc, Value inputSize) const {
    if (tileSize == 1) return inputSize;

    auto inputSizeInt = getConstantIntValue(inputSize);
    if (inputSizeInt && *inputSizeInt % tileSize == 0) return inputSize;

    AffineExpr sym0;
    bindSymbols(b.getContext(), sym0);

    auto modMap = AffineMap::get(0, 1, {sym0 - sym0 % tileSize});
    return b.createOrFold<AffineApplyOp>(loc, modMap, ValueRange{inputSize});
  }

  Value getRemainderSize(OpBuilder &b, Location loc, Value tileableBound,
                         Value inputSize) const {
    AffineExpr sym0, sym1;
    bindSymbols(b.getContext(), sym0, sym1);
    auto diffMap = AffineMap::get(0, 2, {sym1 - sym0});
    return b.create<AffineApplyOp>(loc, diffMap,
                                   ValueRange{tileableBound, inputSize});
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

  Value cloneReduceOp(OpBuilder &b, linalg::ReduceOp reduceOp,
                      ValueRange newInputs, Value newInit) const {
    IRMapping bvm;
    bvm.map(reduceOp.getInputs(), newInputs);
    bvm.map(reduceOp.getInits(), ValueRange{newInit});

    auto *newReduceOp = b.clone(*reduceOp.getOperation(), bvm);
    setLabel(newReduceOp, kTransformedLabel);
    return newReduceOp->getResult(0);
  }

  Value tileAndReshapeInput(OpBuilder &b, Location loc, Value iv, Value input,
                            Type elementType) const {
    Value inputSlice =
        create1DSlice(b, loc, input, iv, b.getIndexAttr(tileSize));

    auto reshapeType =
        RankedTensorType::get({tileSize / vectorSize, vectorSize}, elementType);
    SmallVector<ReassociationIndices> ri = {{0, 1}};
    return b.create<tensor::ExpandShapeOp>(loc, reshapeType, inputSlice, ri);
  }

  int64_t vectorSize;
  int64_t tileSize;
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
  TransformReduceForCpuPass() = default;

  explicit TransformReduceForCpuPass(int64_t reduceVectorSize = 8,
                                     int64_t reduceTileSize1D = 32,
                                     ArrayRef<int64_t> reduceTileSizes2D = {}) {
    vectorSize = reduceVectorSize;
    tileSize1D = reduceTileSize1D;
    tileSizes2D = reduceTileSizes2D;
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<mlir::gml_st::GmlStDialect, arith::ArithDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    if (tileSizes2D.empty()) {
      tileSizes2D = {4, 2};
    }

    assert(tileSizes2D.size() == 2 &&
           "Tiling sizes for Reduce should have 2 element.");

    RewritePatternSet patterns(ctx);
    patterns.add<Reduce1DTransformPattern>(ctx, vectorSize, tileSize1D);
    patterns.add<Reduce2DTransformPattern>(ctx, tileSizes2D[0], tileSizes2D[1]);
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformReduceForCpuPass(int64_t vectorSize, int64_t tileSize1D,
                                ArrayRef<int64_t> tileSizes2D) {
  return std::make_unique<mlir::gml_st::TransformReduceForCpuPass>(
      vectorSize, tileSize1D, tileSizes2D);
}

}  // namespace mlir::gml_st
