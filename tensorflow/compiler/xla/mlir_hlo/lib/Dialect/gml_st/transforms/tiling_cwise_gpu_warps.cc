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

#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/linalg_utils.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_TILINGCWISEGPUWARPSPASS
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h.inc"

namespace mlir {
namespace gml_st {

namespace {

struct TilingCwiseGPUWarpsPattern : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter& rewriter) const override {
    // Only match on cwise ops.
    int64_t arity;
    if (!isCwiseGenericOp(genericOp, arity)) return failure();

    // Only match ops of rank.
    auto genericOpTy =
        genericOp.getResultTypes().front().cast<RankedTensorType>();
    if (genericOpTy.getRank() != 1) return failure();

    // Constants and attributes.
    Location loc = genericOp.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cWarpSize = rewriter.create<arith::ConstantIndexOp>(loc, 32);
    Attribute oneAttr = rewriter.getIndexAttr(1);
    Attribute warpSizeAttr = rewriter.getIndexAttr(32);
    StringAttr warpDist = rewriter.getStringAttr("warp");

    // Create `gml_st.parallel` loop to distribute among lanes.
    Value init = genericOp.getOutputs().front();
    auto ploop = rewriter.create<gml_st::ParallelOp>(
        loc, genericOpTy, c0, cWarpSize, c1, warpDist,
        [&](OpBuilder& b, Location loc, ValueRange ivs) {
          // Compute the lane tile with stride 32. This tile defines the subset
          // of the result that is produced by the lane.
          Value laneId = ivs.front();
          Value genericOpResult = genericOp.getResults().front();
          Value laneTileSize = b.create<arith::DivUIOp>(
              loc,
              b.create<arith::SubIOp>(
                  loc, b.createOrFold<tensor::DimOp>(loc, genericOpResult, c0),
                  laneId),
              cWarpSize);
          Value laneTile = b.createOrFold<gml_st::TileOp>(
              loc, OpFoldResult(laneId), OpFoldResult(laneTileSize),
              OpFoldResult(warpSizeAttr));

          // Create `gml_st.for` loop to iterate over the lane's tile.
          Type elemTy = genericOpTy.getElementType();
          auto sloopTy =
              RankedTensorType::get({ShapedType::kDynamicSize}, elemTy);
          Value laneInit = b.create<gml_st::MaterializeOp>(loc, init, laneTile);
          auto sloop = b.create<gml_st::ForOp>(
              loc, sloopTy, c0, laneTileSize, c1, laneInit,
              [&](OpBuilder& b, Location loc, ValueRange ivs, ValueRange aggr) {
                // Create the iteration tile. This specifies the scalar subset
                // in the warp-level operands.
                Value i = ivs.front();
                Value iterTileOffset =
                    b.create<arith::MulIOp>(loc, i, cWarpSize);
                Value iterTile = b.create<gml_st::TileOp>(
                    loc, OpFoldResult(iterTileOffset), OpFoldResult(oneAttr),
                    OpFoldResult(oneAttr));

                // Materialize scalar subsets per operand.
                SmallVector<Value> iterOperands =
                    llvm::to_vector(llvm::map_range(
                        genericOp.getInputs(), [&](Value arg) -> Value {
                          return b.create<gml_st::MaterializeOp>(loc, elemTy,
                                                                 arg, iterTile);
                        }));

                // Create scalar computation from `linalg.generic` body by (i)
                // mapping its block arguments to the newly materialized scalar
                // operands, and (ii) cloning the body.
                BlockAndValueMapping bvm;
                for (const auto& [blockArg, iterOperand] : llvm::zip(
                         genericOp.getBlock()->getArguments(), iterOperands)) {
                  bvm.map(blockArg, iterOperand);
                }
                for (auto& innerop :
                     genericOp.getBody()->without_terminator()) {
                  rewriter.clone(innerop, bvm);
                }

                // Yield iteration result.
                Value iterResult = bvm.lookup(genericOp.getBody()
                                                  ->getTerminator()
                                                  ->getOperands()
                                                  .front());
                Value iterTileInLaneTile = b.create<gml_st::TileOp>(
                    loc, OpFoldResult(i), OpFoldResult(oneAttr),
                    OpFoldResult(oneAttr));
                b.create<gml_st::SetYieldOp>(loc, iterResult, aggr,
                                             iterTileInLaneTile);
              });
          b.create<gml_st::SetYieldOp>(loc, sloop.getResults().front(),
                                       laneInit, laneTile);
        });

    rewriter.replaceOp(genericOp, ploop.getResults());
    return success();
  }
};

struct TilingCwiseGPUWarpsPass
    : public ::impl::TilingCwiseGPUWarpsPassBase<TilingCwiseGPUWarpsPass> {
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();

    // Populate patterns
    RewritePatternSet patterns(ctx);
    patterns.add<TilingCwiseGPUWarpsPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createTilingCwiseGPUWarpsPass() {
  return std::make_unique<TilingCwiseGPUWarpsPass>();
}

}  // namespace gml_st
}  // namespace mlir