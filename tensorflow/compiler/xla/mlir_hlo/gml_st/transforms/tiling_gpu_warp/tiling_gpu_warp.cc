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
#include "gml_st/interfaces/tiling_interface.h"
#include "gml_st/interfaces/tiling_interface_impl.h"
#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_TILINGGPUWARPPASS
#include "gml_st/transforms/passes.h.inc"

namespace mlir {
namespace gml_st {

namespace {

static constexpr llvm::StringRef kTileGpuWarpAppliedLabel =
    "__tile_gpu_warp_applied_label__";

constexpr const char* kWarpDistributionLabel = "warp";
constexpr const char* kThreadDistributionLabel = "thread";

using OpFoldResults = SmallVector<OpFoldResult>;

// Returns 'count' rounded up to power of two, up to warp size (32).
static int64_t getGroupSize(int64_t count) {
  constexpr int64_t kWarpSize = 32;
  if (count < 0) return kWarpSize;
  for (int64_t i = 1; i < kWarpSize; i *= 2)
    if (i >= count) return i;
  return kWarpSize;
}

bool isWarpLevelOp(Operation* op) {
  if (!op) return false;
  auto parentPloop = op->getParentOfType<ParallelOp>();
  return parentPloop && parentPloop.getDistributionType() &&
         *parentPloop.getDistributionType() == kWarpDistributionLabel;
}

struct TilingCwisePattern : OpRewritePattern<linalg::MapOp> {
  using OpRewritePattern<linalg::MapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MapOp mapOp,
                                PatternRewriter& rewriter) const override {
    if (hasLabel(mapOp, kTileGpuWarpAppliedLabel)) {
      return rewriter.notifyMatchFailure(mapOp, "already transformed");
    }

    // Match only `linalg.map` ops on the shape 1x?.
    if (mapOp.getNumDpsInits() != 1) {
      return rewriter.notifyMatchFailure(mapOp, "not element-wise");
    }
    Value mapOpResult = mapOp.getResult().front();
    auto ploopTy = mapOpResult.getType().dyn_cast<RankedTensorType>();
    if (!ploopTy || ploopTy.getRank() != 2 || ploopTy.getDimSize(0) != 1) {
      return rewriter.notifyMatchFailure(mapOp, "result no tensor<1x?>");
    }

    // Only tile root ops on the warp level.
    if (!isWarpLevelOp(mapOp) || !mapOp->hasOneUse() ||
        !llvm::isa<SetYieldOp>(*mapOp->getUsers().begin())) {
      return rewriter.notifyMatchFailure(mapOp, "not a warp level root op");
    }

    // The number of threads per row (power of two, <= kWarpSize).
    int64_t groupSize = getGroupSize(ploopTy.getDimSize(1));

    // Constants and attributes.
    Location loc = mapOp.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cGroupSize = rewriter.create<arith::ConstantIndexOp>(loc, groupSize);
    Value cGroupSizeMinusOne =
        rewriter.create<arith::ConstantIndexOp>(loc, groupSize - 1);
    Attribute zeroAttr = rewriter.getIndexAttr(0);
    Attribute oneAttr = rewriter.getIndexAttr(1);
    Attribute groupSizeAttr = rewriter.getIndexAttr(groupSize);
    StringAttr threadDistrLabel =
        rewriter.getStringAttr(kThreadDistributionLabel);

    // Create `gml_st.parallel` loop to distribute among lanes.
    Value init = mapOp.getInit();
    Value dimSize = rewriter.createOrFold<tensor::DimOp>(loc, mapOpResult, c1);
    Value dimSizePlusWarpSizeMinusOne =
        rewriter.createOrFold<arith::AddIOp>(loc, dimSize, cGroupSizeMinusOne);
    auto ploop = rewriter.create<gml_st::ParallelOp>(
        loc, ploopTy, c0, cGroupSize, c1, threadDistrLabel,
        [&](OpBuilder& b, Location loc, ValueRange ivs) {
          // Compute the lane tile with a stride of `warpSize`. This tile
          // defines the subset of the result that is produced by the lane.
          // The `laneId` defines the initial offset into the tensor. The
          // remaining length to be addressed by the lane is
          //     `dimSize` - `laneId`.
          // With a stride of `warpSize`, every lane addresses a total of
          //     ceil((`dimSize` - `laneId`) / `cWarpSize`)
          //     = (`dimSize` + `cWarpSize` - 1 - `laneId`) / `cWarpSize`
          // elements.
          Value laneId = ivs.front();
          Value laneTileSize = b.create<arith::DivUIOp>(
              loc,
              b.create<arith::SubIOp>(loc, dimSizePlusWarpSizeMinusOne, laneId),
              cGroupSize);
          Value laneInit = materializeSlice(
              b, loc, init, OpFoldResults{zeroAttr, laneId},
              OpFoldResults{oneAttr, laneTileSize},
              OpFoldResults{oneAttr, groupSizeAttr}, /*useExtractSlice=*/false);

          // Create `gml_st.for` loop to iterate over the lane's tile.
          auto sloopTy = ploopTy.clone({1, ShapedType::kDynamic});
          auto sloop = b.create<gml_st::ForOp>(
              loc, sloopTy, c0, laneTileSize, c1, laneInit,
              [&](OpBuilder& b, Location loc, ValueRange ivs, ValueRange aggr) {
                // Create the iteration tile. This specifies the scalar subset
                // in the warp-level operands.
                Value i = ivs.front();
                Value iterTileOffset = b.create<arith::AddIOp>(
                    loc, laneId, b.create<arith::MulIOp>(loc, i, cGroupSize));

                // Materialize scalar subsets per operand.
                SmallVector<Value> iterOperands = llvm::to_vector(
                    llvm::map_range(mapOp.getInputs(), [&](Value arg) -> Value {
                      return materializePoint(
                          b, loc, arg, OpFoldResults{zeroAttr, iterTileOffset},
                          /*useExtractSlice=*/false);
                    }));

                // Create scalar computation from `linalg.map` body by (i)
                // mapping its block arguments to the newly materialized
                // scalar operands, and (ii) cloning the body.
                BlockAndValueMapping bvm;
                bvm.map(mapOp.getBlock()->getArguments(), iterOperands);
                for (auto& innerOp : mapOp.getBody()->without_terminator()) {
                  rewriter.clone(innerOp, bvm);
                }

                // Yield iteration result.
                Value iterResult =
                    bvm.lookup(mapOp.getBody()->getTerminator()->getOperand(0));
                Value iterTileInLaneTile =
                    b.create<gml_st::TileOp>(loc, OpFoldResults{zeroAttr, i},
                                             OpFoldResults{oneAttr, oneAttr},
                                             OpFoldResults{oneAttr, oneAttr});
                b.create<gml_st::SetYieldOp>(loc, iterResult, aggr,
                                             iterTileInLaneTile);
              });
          Value laneTile = b.createOrFold<gml_st::TileOp>(
              loc, OpFoldResults{zeroAttr, laneId},
              OpFoldResults{oneAttr, laneTileSize},
              OpFoldResults{oneAttr, groupSizeAttr});
          b.create<gml_st::SetYieldOp>(loc, sloop.getResult(0), init, laneTile);
        });

    rewriter.replaceOp(mapOp, ploop.getResults());
    return success();
  }
};

struct TilingReductionPattern : OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp reduceOp,
                                PatternRewriter& rewriter) const override {
    if (hasLabel(reduceOp, kTileGpuWarpAppliedLabel)) {
      return rewriter.notifyMatchFailure(reduceOp, "already transformed");
    }

    // Only tile ops on the warp level.
    if (!isWarpLevelOp(reduceOp)) {
      return rewriter.notifyMatchFailure(reduceOp, "not a warp level op");
    }

    // Match only if it's a linalg.reduce tensor<1x?xf32> -> tensor<1xf32>
    if (reduceOp.getNumDpsInputs() != 1 || reduceOp.getNumDpsInits() != 1) {
      return rewriter.notifyMatchFailure(reduceOp,
                                         "Expected single input and output");
    }

    auto inputTy =
        reduceOp.getInputs().front().getType().dyn_cast<RankedTensorType>();

    // The number of threads per row (power of two, <= kWarpSize).
    int64_t groupSize = getGroupSize(inputTy.getDimSize(1));

    // Attributes and constants.
    Location loc = reduceOp->getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cGroupSize = rewriter.create<arith::ConstantIndexOp>(loc, groupSize);
    IntegerAttr zeroAttr = rewriter.getIndexAttr(0);
    IntegerAttr oneAttr = rewriter.getIndexAttr(1);
    IntegerAttr groupSizeAttr = rewriter.getIndexAttr(groupSize);
    StringAttr threadDistrLabel =
        rewriter.getStringAttr(kThreadDistributionLabel);

    Value operand = reduceOp.getInputs().front();
    Value init = reduceOp.getInits().front();

    Type scalarTy = inputTy.getElementType();

    Value reductionDimSize = rewriter.create<tensor::DimOp>(loc, operand, c1);

    // Create warp-sized partial reduction result tensor.
    Value warpResult = rewriter.create<tensor::EmptyOp>(
        loc, OpFoldResults{oneAttr, groupSizeAttr}, scalarTy);
    Value initMaterialized =
        materializePoint(rewriter, loc, init, OpFoldResults{zeroAttr},
                         /*useExtractSlice=*/false);
    warpResult =
        rewriter.create<linalg::FillOp>(loc, initMaterialized, warpResult)
            .getResult(0);

    // Create gml_st.parallel finalizing the partial result.
    auto parallelOpBodyBuilderFn = [&](OpBuilder& b, Location loc,
                                       ValueRange ivs) {
      Value laneId = ivs.front();
      Value laneResult = materializeSlice(
          b, loc, warpResult, OpFoldResults{zeroAttr, laneId},
          OpFoldResults{oneAttr, oneAttr}, OpFoldResults{oneAttr, oneAttr},
          /*useExtractSlice=*/false);

      // Create gml_st.for sequentially reducing parts of the row.
      auto forOpBodyBuilderFn = [&](OpBuilder& b, Location loc, ValueRange ivs,
                                    ValueRange outputs) {
        Value iterationId = ivs.front();
        Value laneAcc = outputs.front();

        // Materialize operand subset.
        Value operandMaterialized = materializePoint(
            b, loc, operand, ArrayRef<OpFoldResult>{zeroAttr, iterationId},
            /*useExtractSlice=*/false);

        // Materialize intermediate result.
        Value iterationResult = materializePoint(
            rewriter, loc, laneAcc, OpFoldResults{zeroAttr, zeroAttr},
            /*useExtractSlice=*/false);

        // Create scalar computation based on `linalg.reduce` body.
        BlockAndValueMapping bvm;
        bvm.map(reduceOp.getBlock()->getArguments()[0], operandMaterialized);
        bvm.map(reduceOp.getBlock()->getArguments()[1], iterationResult);
        for (Operation& inner : reduceOp.getBody()->without_terminator()) {
          rewriter.clone(inner, bvm);
        }
        iterationResult =
            bvm.lookup(reduceOp.getBody()->getTerminator()->getOperand(0));

        Value iterationTile =
            rewriter.create<TileOp>(loc, OpFoldResults{zeroAttr, zeroAttr});
        b.create<gml_st::SetYieldOp>(loc, iterationResult, laneAcc,
                                     iterationTile);
      };
      laneResult = b.create<gml_st::ForOp>(loc, laneResult.getType(), laneId,
                                           reductionDimSize, cGroupSize,
                                           laneResult, forOpBodyBuilderFn)
                       .getResult(0);

      Value laneTile = b.create<TileOp>(loc, OpFoldResults{zeroAttr, laneId});
      b.create<gml_st::SetYieldOp>(loc, laneResult, warpResult, laneTile);
    };
    warpResult = rewriter
                     .create<gml_st::ParallelOp>(
                         loc, warpResult.getType(), c0, cGroupSize, c1,
                         threadDistrLabel, parallelOpBodyBuilderFn)
                     .getResult(0);

    // Change existing linalg.generic to warp-reduce the partial results.
    rewriter.updateRootInPlace(reduceOp, [&] {
      reduceOp->setOperand(0, warpResult);
      setLabel(reduceOp, kTileGpuWarpAppliedLabel);
    });

    return success();
  }
};

struct TilingGPUWarpPass
    : public ::impl::TilingGPUWarpPassBase<TilingGPUWarpPass> {
  void getDependentDialects(DialectRegistry& registry) const final {
    ::impl::TilingGPUWarpPassBase<TilingGPUWarpPass>::getDependentDialects(
        registry);
    registerGmlStTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    MLIRContext* ctx = &getContext();

    // Populate tiling patterns
    RewritePatternSet patterns(ctx);
    patterns.add<TilingCwisePattern, TilingReductionPattern>(ctx);

    // Populate fusion patterns.
    auto fuseGreedilyFilterFn = [](Operation* op) {
      auto materializeOp = llvm::dyn_cast<MaterializeOp>(op);
      Operation* source = materializeOp.getSource().getDefiningOp();

      // Do not fuse warp-level reductions.
      auto reductionOp = llvm::dyn_cast_or_null<linalg::ReduceOp>(source);
      if (reductionOp && reductionOp.getNumDpsInits() == 1 &&
          isWarpLevelOp(source))
        return failure();

      return success();
    };
    populateFusionPatterns(ctx, fuseGreedilyFilterFn, &patterns);

    func::FuncOp func = getOperation();
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Clean up by removing temporary attributes.
    func.walk([](Operation* op) { removeLabel(op, kTileGpuWarpAppliedLabel); });
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createTilingGpuWarpPass() {
  return std::make_unique<TilingGPUWarpPass>();
}

}  // namespace gml_st
}  // namespace mlir
