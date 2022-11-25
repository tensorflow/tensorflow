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
#include "gml_st/interfaces/tiling_interface_impl.h"
#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/transforms.h"
#include "gml_st/utils/linalg_utils.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_TILINGGPUWARPPASS
#include "gml_st/transforms/passes.h.inc"

namespace mlir {
namespace gml_st {

namespace {

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

struct TilingCwisePattern : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter& rewriter) const override {
    if (hasTransformationAttr(genericOp)) {
      return rewriter.notifyMatchFailure(genericOp, "already transformed");
    }

    // Match only cwise `linalg.generic` ops on the shape 1x?.
    if (!isCwiseGenericOp(genericOp)) {
      return rewriter.notifyMatchFailure(genericOp, "not element-wise");
    }
    Value genericOpResult = genericOp.getResult(0);
    auto ploopTy = genericOpResult.getType().dyn_cast<RankedTensorType>();
    if (!ploopTy || ploopTy.getRank() != 2 || ploopTy.getDimSize(0) != 1) {
      return rewriter.notifyMatchFailure(genericOp, "result no tensor<1x?>");
    }

    // Only tile root ops on the warp level.
    if (!isWarpLevelOp(genericOp) || !genericOp->hasOneUse() ||
        !llvm::isa<SetYieldOp>(*genericOp->getUsers().begin())) {
      return rewriter.notifyMatchFailure(genericOp, "not a warp level root op");
    }

    // The number of threads per row (power of two, <= kWarpSize).
    int64_t groupSize = getGroupSize(ploopTy.getDimSize(1));

    // Constants and attributes.
    Location loc = genericOp.getLoc();
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
    Value init = genericOp.getOutputs().front();
    Value dimSize =
        rewriter.createOrFold<tensor::DimOp>(loc, genericOpResult, c1);
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
          Value laneTile = b.createOrFold<gml_st::TileOp>(
              loc, OpFoldResults{zeroAttr, laneId},
              OpFoldResults{oneAttr, laneTileSize},
              OpFoldResults{oneAttr, groupSizeAttr});
          Value laneInit = b.create<gml_st::MaterializeOp>(loc, init, laneTile);

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
                Value iterTile = b.create<gml_st::TileOp>(
                    loc, OpFoldResults{zeroAttr, iterTileOffset},
                    OpFoldResults{oneAttr, oneAttr},
                    OpFoldResults{oneAttr, oneAttr});

                // Materialize scalar subsets per operand.
                SmallVector<Value> iterOperands =
                    llvm::to_vector(llvm::map_range(
                        genericOp.getInputs(), [&](Value arg) -> Value {
                          return b.create<gml_st::MaterializeOp>(
                              loc, ploopTy.getElementType(), arg, iterTile);
                        }));

                // Create scalar computation from `linalg.generic` body by (i)
                // mapping its block arguments to the newly materialized
                // scalar operands, and (ii) cloning the body.
                BlockAndValueMapping bvm;
                bvm.map(genericOp.getBlock()->getArguments(), iterOperands);
                for (auto& innerOp :
                     genericOp.getBody()->without_terminator()) {
                  rewriter.clone(innerOp, bvm);
                }

                // Yield iteration result.
                Value iterResult = bvm.lookup(
                    genericOp.getBody()->getTerminator()->getOperand(0));
                Value iterTileInLaneTile =
                    b.create<gml_st::TileOp>(loc, OpFoldResults{zeroAttr, i},
                                             OpFoldResults{oneAttr, oneAttr},
                                             OpFoldResults{oneAttr, oneAttr});
                b.create<gml_st::SetYieldOp>(loc, iterResult, aggr,
                                             iterTileInLaneTile);
              });
          b.create<gml_st::SetYieldOp>(loc, sloop.getResult(0), init, laneTile);
        });

    rewriter.replaceOp(genericOp, ploop.getResults());
    return success();
  }
};

struct TilingReductionPattern : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter& rewriter) const override {
    if (hasTransformationAttr(genericOp)) {
      return rewriter.notifyMatchFailure(genericOp, "already transformed");
    }

    // Only tile ops on the warp level.
    if (!isWarpLevelOp(genericOp)) {
      return rewriter.notifyMatchFailure(genericOp, "not a warp level op");
    }

    // Match only if it's a linalg.generic tensor<1x?xf32> -> tensor<1xf32> with
    // iterator_types = ["parallel", "reduction"].
    auto itTypes = genericOp.getIteratorTypesArray();
    if (itTypes.size() != 2 || !linalg::isParallelIterator(itTypes[0]) ||
        !linalg::isReductionIterator(itTypes[1])) {
      return rewriter.notifyMatchFailure(genericOp,
                                         "Expected ['parallel', 'reduction']");
    }
    if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
      return rewriter.notifyMatchFailure(genericOp,
                                         "Expected single input and output");
    }

    auto inputTy =
        genericOp.getInputs().front().getType().dyn_cast<RankedTensorType>();

    // The number of threads per row (power of two, <= kWarpSize).
    int64_t groupSize = getGroupSize(inputTy.getDimSize(1));

    // Attributes and constants.
    Location loc = genericOp->getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cGroupSize = rewriter.create<arith::ConstantIndexOp>(loc, groupSize);
    IntegerAttr zeroAttr = rewriter.getIndexAttr(0);
    IntegerAttr oneAttr = rewriter.getIndexAttr(1);
    IntegerAttr groupSizeAttr = rewriter.getIndexAttr(groupSize);
    StringAttr threadDistrLabel =
        rewriter.getStringAttr(kThreadDistributionLabel);

    Value operand = genericOp.getInputs().front();
    Value init = genericOp.getOutputs().front();

    Type scalarTy = inputTy.getElementType();

    Value reductionDimSize = rewriter.create<tensor::DimOp>(loc, operand, c1);

    // Create warp-sized partial reduction result tensor.
    Value warpResult = rewriter.create<tensor::EmptyOp>(
        loc, OpFoldResults{oneAttr, groupSizeAttr}, scalarTy);
    Value initTile = rewriter.create<TileOp>(loc, OpFoldResults{zeroAttr});
    Value initMaterialized =
        rewriter.create<MaterializeOp>(loc, scalarTy, init, initTile);
    warpResult =
        rewriter.create<linalg::FillOp>(loc, initMaterialized, warpResult)
            .getResult(0);

    // Create gml_st.parallel finalizing the partial result.
    auto parallelOpBodyBuilderFn = [&](OpBuilder& b, Location loc,
                                       ValueRange ivs) {
      Value laneId = ivs.front();
      Value laneTile = b.create<TileOp>(loc, OpFoldResults{zeroAttr, laneId});
      Value laneResult = b.create<MaterializeOp>(loc, warpResult, laneTile);

      // Create gml_st.for sequentially reducing parts of the row.
      auto forOpBodyBuilderFn = [&](OpBuilder& b, Location loc, ValueRange ivs,
                                    ValueRange outputs) {
        Value iterationId = ivs.front();
        Value laneAcc = outputs.front();

        // Materialize operand subset.
        Value operandTile = b.create<TileOp>(
            loc, ArrayRef<OpFoldResult>{zeroAttr, iterationId});
        Value operandMaterialized =
            b.create<MaterializeOp>(loc, scalarTy, operand, operandTile);

        // Materialize intermediate result.
        Value iterationTile =
            rewriter.create<TileOp>(loc, OpFoldResults{zeroAttr, zeroAttr});
        Value iterationResult = rewriter.create<MaterializeOp>(
            loc, scalarTy, laneAcc, iterationTile);

        // Create scalar computation based on `linalg.generic` body.
        BlockAndValueMapping bvm;
        bvm.map(genericOp.getBlock()->getArguments()[0], operandMaterialized);
        bvm.map(genericOp.getBlock()->getArguments()[1], iterationResult);
        for (Operation& inner : genericOp.getBody()->without_terminator()) {
          rewriter.clone(inner, bvm);
        }
        iterationResult =
            bvm.lookup(genericOp.getBody()->getTerminator()->getOperand(0));

        b.create<gml_st::SetYieldOp>(loc, iterationResult, laneAcc,
                                     iterationTile);
      };
      laneResult = b.create<gml_st::ForOp>(loc, laneResult.getType(), laneId,
                                           reductionDimSize, cGroupSize,
                                           laneResult, forOpBodyBuilderFn)
                       .getResult(0);

      b.create<gml_st::SetYieldOp>(loc, laneResult, warpResult, laneTile);
    };
    warpResult = rewriter
                     .create<gml_st::ParallelOp>(
                         loc, warpResult.getType(), c0, cGroupSize, c1,
                         threadDistrLabel, parallelOpBodyBuilderFn)
                     .getResult(0);

    // Change existing linalg.generic to warp-reduce the partial results.
    rewriter.updateRootInPlace(genericOp, [&] {
      genericOp->setOperand(0, warpResult);
      gml_st::setTransformationAttr(rewriter, genericOp);
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

      // Do not fuse wap-level reductions.
      if (isSimpleReduction(source) && isWarpLevelOp(source)) return failure();

      return success();
    };
    populateFusionPatterns(ctx, fuseGreedilyFilterFn, &patterns);

    func::FuncOp func = getOperation();
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Clean up by removing temporary attributes.
    func.walk([](Operation* op) { removeTransformationAttr(op); });
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createTilingGpuWarpPass() {
  return std::make_unique<TilingGPUWarpPass>();
}

}  // namespace gml_st
}  // namespace mlir
