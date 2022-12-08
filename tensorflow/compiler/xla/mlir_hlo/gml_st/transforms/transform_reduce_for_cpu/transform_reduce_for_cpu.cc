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
#include "gml_st/interfaces/tiling_interface_impl.h"
#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/peeling/peeling.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMREDUCEFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

static constexpr llvm::StringRef kReduceTransformedLabel =
    "__reduce_transformed_label__";

FailureOr<TilingResult> tileReduce(PatternRewriter &rewriter,
                                   linalg::ReduceOp reduceOp,
                                   ArrayRef<int64_t> tileSizes,
                                   bool distribute) {
  TilingOptions opts;
  opts.setTileSizeComputationFn(tileSizes);
  opts.distribute = distribute;
  return tile(opts, rewriter, cast<TilingInterface>(reduceOp.getOperation()));
}

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
  if (reduceDimensions.size() != 1)
    return rewriter.notifyMatchFailure(
        reduceOp, "expects 1 reduction dimension element. 0 or > 1 received.");
  OpOperandVector operands = reduceOp.getDpsInputOperands();
  if (operands.size() != 1)
    return rewriter.notifyMatchFailure(reduceOp,
                                       "expects 1 operand. 0 or > 1 received.");
  const int64_t operandRank =
      operands[0]->get().getType().cast<RankedTensorType>().getRank();
  if (operandRank != expectedRank)
    return rewriter.notifyMatchFailure(reduceOp, [&](::mlir::Diagnostic &diag) {
      diag << "expects rank " << expectedRank << ". " << operandRank
           << "received.";
    });
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
    if (hasLabel(reduceOp, kReduceTransformedLabel))
      return rewriter.notifyMatchFailure(reduceOp,
                                         "has already been transformed.");

    if (failed(validateOp(reduceOp, rewriter, /*expectedRank=*/1)))
      return failure();

    Location loc = reduceOp.getLoc();

    // Constants.
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

    // 0-d tensor with the neutral elements.
    auto fillOp = reduceOp.getInits().front().getDefiningOp<linalg::FillOp>();
    if (!fillOp) return failure();
    auto neutralValue = fillOp.value();
    // .get

    // fillOp.getValue();
    Type elementType = neutralValue.getType();

    // Create tensor<VECTOR_SIZExELEM_TYPE> with neutral elements for tile loop
    // init.
    Value emptyVector = rewriter.create<tensor::EmptyOp>(
        loc, llvm::makeArrayRef({vectorSize}), elementType);
    Value filledVector =
        rewriter.create<linalg::FillOp>(loc, neutralValue, emptyVector)
            .getResult(0);

    auto tiledLoopBodyBuilder = [&](OpBuilder &b, Location loc, ValueRange ivs,
                                    ValueRange inits) {
      // Tile input as tensor<TILE_SIZExELEM_TYPE> and reshape into
      // tensor<(TILE_SIZE/VECTOR_SIZE)xVECTOR_SIZExELEM_TYPE>.
      Value inputSlice =
          tileAndReshapeInput(b, loc, ivs.front(), input, elementType);

      Value initTile =
          create1DTile(b, loc, b.getIndexAttr(0), b.getIndexAttr(vectorSize));
      Value initSlice =
          b.create<gml_st::MaterializeOp>(loc, inits.front(), initTile);

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
      setLabel(tiledReduceOp, kReduceTransformedLabel);

      b.create<gml_st::SetYieldOp>(loc, tiledReduceOp.getResults(), inits,
                                   initTile);
    };

    // Create a tiled loop
    auto tiledLoop = rewriter.create<ForOp>(loc, filledVector.getType(), zero,
                                            tileableBound, tileSizeValue,
                                            filledVector, tiledLoopBodyBuilder);
    setLabel(tiledLoop, kPerfectlyTiledLoopLabel);

    // Create `linalg.reduce` from tensor<VECTOR_SIZExELEM_TYPE> to
    // tensor<ELEM_TYPE>.
    auto horizontalReduce =
        cloneReduceOp(rewriter, reduceOp, tiledLoop.getResult(0),
                      reduceOp.getInits().front());

    auto remainderLoopBodyBuilder = [&](OpBuilder &b, Location loc,
                                        ValueRange ivs, ValueRange inits) {
      Value inputTile = create1DTile(b, loc, ivs.front(), remainderSize);
      Value inputSlice = b.create<gml_st::MaterializeOp>(loc, input, inputTile);

      Value initTile = b.create<gml_st::TileOp>(
          loc, /*offsets=*/SmallVector<OpFoldResult>{});
      Value initSlice =
          b.create<gml_st::MaterializeOp>(loc, inits.front(), initTile);

      auto newReduceOp = cloneReduceOp(b, reduceOp, inputSlice, initSlice);

      b.create<gml_st::SetYieldOp>(loc, newReduceOp, inits, initTile);
    };

    // Combine `horizontal reduce` with the tail of the input. The tail is
    // always smaller than TILE_SIZE.
    auto remainderLoop =
        rewriter
            .create<ForOp>(loc, reduceOp.getResultTypes(), tileableBound,
                           inputSize, tileSizeValue, horizontalReduce,
                           remainderLoopBodyBuilder)
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

  Value create1DTile(OpBuilder &b, Location loc, OpFoldResult offset,
                     OpFoldResult size) const {
    SmallVector<OpFoldResult> offsets{offset};
    SmallVector<OpFoldResult> sizes{size};
    SmallVector<OpFoldResult> strides{b.getIndexAttr(1)};

    return b.create<gml_st::TileOp>(loc, offsets, sizes, strides);
  }

  Value cloneReduceOp(OpBuilder &b, linalg::ReduceOp reduceOp,
                      ValueRange newInputs, Value newInit) const {
    BlockAndValueMapping bvm;
    bvm.map(reduceOp.getInputs(), newInputs);
    bvm.map(reduceOp.getInits(), ValueRange{newInit});

    auto *newReduceOp = b.clone(*reduceOp.getOperation(), bvm);
    setLabel(newReduceOp, kReduceTransformedLabel);
    return newReduceOp->getResult(0);
  }

  Value tileAndReshapeInput(OpBuilder &b, Location loc, Value iv, Value input,
                            Type elementType) const {
    Value inputTile = create1DTile(b, loc, iv, b.getIndexAttr(tileSize));
    Value inputSlice = b.create<gml_st::MaterializeOp>(loc, input, inputTile);

    auto reshapeType =
        RankedTensorType::get({tileSize / vectorSize, vectorSize}, elementType);
    SmallVector<ReassociationIndices> ri = {{0, 1}};
    return b.create<tensor::ExpandShapeOp>(loc, reshapeType, inputSlice, ri);
  }

  int64_t vectorSize;
  int64_t tileSize;
};

/// Pattern to tile `linalg.reduce` and fuse `linalg.fill` into generated
/// `gml_st.parallel`.
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
    if (hasLabel(reduceOp, kReduceTransformedLabel))
      return rewriter.notifyMatchFailure(reduceOp,
                                         "has already been transformed.");

    if (failed(validateOp(reduceOp, rewriter, /*expectedRank=*/2)))
      return failure();

    // First level tiling: parallel dimension.
    auto tilingParallelDimsResult =
        tileReduce(rewriter, reduceOp,
                   getParallelDimTileSizes(reduceOp.getDimensions()[0],
                                           parallelDimTileSize),
                   /*distribute=*/true);
    if (failed(tilingParallelDimsResult)) return failure();

    // Update the results if tiling occurred.
    if (tilingParallelDimsResult->loop != nullptr) {
      rewriter.replaceOp(reduceOp,
                         tilingParallelDimsResult->loop->getResults());
      reduceOp = cast<linalg::ReduceOp>(tilingParallelDimsResult->tiledOp);
      // Fuse linalg.map ops into the loop.
      fuseGreedily(rewriter, *reduceOp->getBlock(),
                   [](Operation *op) { return isa<linalg::MapOp>(op); });
    }

    // Fusion into the output.
    OpOperand *reduceOutput = reduceOp.getDpsInitOperand(0);
    auto materialize = reduceOutput->get().getDefiningOp<MaterializeOp>();
    if (!materialize) {
      return rewriter.notifyMatchFailure(
          reduceOp,
          "has failed to 'materialize' output during 'linalg.fill' fusion.");
    }
    if (materialize.getSource().getDefiningOp<linalg::FillOp>()) {
      if (failed(fuse(rewriter, materialize))) return failure();
    }

    // Second level tiling: reduction dimension.
    auto tilingReductionDimsResult =
        tileReduce(rewriter, reduceOp,
                   getReductionDimTileSizes(reduceOp.getDimensions()[0],
                                            reductionDimTileSize),
                   /*distribute=*/false);
    if (failed(tilingReductionDimsResult)) return failure();

    // Update the results if tiling occurred.
    if (tilingReductionDimsResult->loop != nullptr) {
      rewriter.replaceOp(reduceOp,
                         tilingReductionDimsResult->loop->getResults());
      reduceOp = cast<linalg::ReduceOp>(tilingReductionDimsResult->tiledOp);
      // Fuse linalg.map ops into the loop.
      fuseGreedily(rewriter, *reduceOp->getBlock(),
                   [](Operation *op) { return isa<linalg::MapOp>(op); });
    }

    setLabel(reduceOp, kReduceTransformedLabel);

    // Peel parallel loops.
    if (auto loop =
            dyn_cast_or_null<ParallelOp>(tilingParallelDimsResult->loop)) {
      auto peelingResult = peelAllLoops(loop, rewriter);
    }

    // Peel reduction loop inside the main parallel loop, label the main loop as
    // "perfectly tiled" one, to enable vectorization after canonicalization.
    if (auto loop = dyn_cast_or_null<ForOp>(tilingReductionDimsResult->loop)) {
      auto peelingResult = peelAllLoops(loop, rewriter);
      setLabel(loop, kPerfectlyTiledLoopLabel);
    }

    return success();
  }

 private:
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
                    linalg::LinalgDialect, tensor::TensorDialect>();
    mlir::gml_st::registerGmlStTilingInterfaceExternalModels(registry);
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

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Ensure we drop the marker in the end.
    f.walk([](linalg::ReduceOp reduceOp) {
      removeLabel(reduceOp, kReduceTransformedLabel);
    });
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
