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
#include "mlir-hlo/Dialect/gml_st/transforms/fusion.h"
#include "mlir-hlo/Dialect/gml_st/transforms/linalg_utils.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir-hlo/Dialect/gml_st/transforms/tiling_interface_impl.h"
#include "mlir-hlo/Dialect/gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_TILINGGPUWARPPASS
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h.inc"

namespace mlir {
namespace gml_st {

namespace {

bool isFusible(Operation* op) {
  if (!op) return false;

  // Do not fuse reductions on the warp level.
  if (isSimpleReduction(op)) return false;

  // Fuse into materializes, cwise, reductions, and bcasts.
  if (llvm::any_of(op->getUsers(), [](Operation* user) {
        return llvm::isa<MaterializeOp>(user) || isCwiseGenericOp(user) ||
               isSimpleReduction(user) || isSimpleBcastReduction(user);
      })) {
    return true;
  }

  return false;
}

constexpr const char* kThreadDistributionLabel = "thread";

struct TilingCwisePattern : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter& rewriter) const override {
    // Only match on cwise ops.
    if (!isCwiseGenericOp(genericOp)) return failure();

    // Expect ops on rank 1.
    auto genericOpTy =
        genericOp.getResultTypes().front().cast<RankedTensorType>();
    if (genericOpTy.getRank() != 1) return failure();

    // Tile only on thread level.
    auto parentPloop = genericOp->getParentOfType<ParallelOp>();
    if (parentPloop && parentPloop.getDistributionType() &&
        *parentPloop.getDistributionType() == kThreadDistributionLabel) {
      return failure();
    }

    // Tile only the roots and fuse if possible.
    if (isFusible(genericOp)) return failure();

    // Constants and attributes.
    Location loc = genericOp.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cWarpSize = rewriter.create<arith::ConstantIndexOp>(loc, 32);
    Value cWarpSizeMinusOne = rewriter.create<arith::ConstantIndexOp>(loc, 31);
    Attribute oneAttr = rewriter.getIndexAttr(1);
    Attribute warpSizeAttr = rewriter.getIndexAttr(32);
    StringAttr threadDist = rewriter.getStringAttr(kThreadDistributionLabel);

    // Create `gml_st.parallel` loop to distribute among lanes.
    Value init = genericOp.getOutputs().front();
    Value genericOpResult = genericOp.getResults().front();
    Value dimSize =
        rewriter.createOrFold<tensor::DimOp>(loc, genericOpResult, c0);
    Value dimSizePlusWarpSizeMinusOne =
        rewriter.create<arith::AddIOp>(loc, dimSize, cWarpSizeMinusOne);
    auto ploop = rewriter.create<gml_st::ParallelOp>(
        loc, genericOpTy, c0, cWarpSize, c1, threadDist,
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
                // mapping its block arguments to the newly materialized
                // scalar operands, and (ii) cloning the body.
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
          b.create<gml_st::SetYieldOp>(loc, sloop.getResults().front(), init,
                                       laneTile);
        });

    rewriter.replaceOp(genericOp, ploop.getResults());
    return success();
  }
};

struct TilingReductionPattern : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter& rewriter) const override {
    if (gml_st::hasTransformationAttr(genericOp)) return failure();

    // Match only if it's a linalg.generic tensor<1x?xf32> -> tensor<1xf32> with
    // iterator_types = ["parallel", "reduction"].
    auto itTypes = llvm::to_vector(
        genericOp.getIteratorTypes().getAsValueRange<StringAttr>());
    if (itTypes.size() != 2 || itTypes[0] != getParallelIteratorTypeName() ||
        itTypes[1] != getReductionIteratorTypeName()) {
      return rewriter.notifyMatchFailure(genericOp,
                                         "Expected ['parallel', 'reduction']");
    }
    if (genericOp.getNumInputs() != 1 || genericOp.getNumOutputs() != 1) {
      return rewriter.notifyMatchFailure(genericOp,
                                         "Expected single input and output");
    }
    auto input = genericOp.getInputs().front();
    auto output = genericOp.getOutputs().front();
    auto outType = output.getType().dyn_cast<TensorType>();
    constexpr int kWarpSize = 32;

    Location loc = genericOp->getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cWarpSize = rewriter.create<arith::ConstantIndexOp>(loc, kWarpSize);
    Value reductionDim = rewriter.create<tensor::DimOp>(loc, input, c1);
    OpFoldResult zeroAttr = rewriter.getIndexAttr(0);
    OpFoldResult oneAttr = rewriter.getIndexAttr(1);
    auto threadDist = rewriter.getStringAttr("thread");

    auto getResult = [](Operation* op) { return op->getResult(0); };

    // Create warp-sized partial reduction result tensor.
    Type elType = outType.getElementType();
    Value partial = rewriter.create<tensor::EmptyOp>(loc, kWarpSize, elType);
    Value outPoint =
        rewriter.create<gml_st::TileOp>(loc, zeroAttr, oneAttr, oneAttr);
    Value outElement =
        rewriter.create<gml_st::MaterializeOp>(loc, elType, output, outPoint);
    partial =
        getResult(rewriter.create<linalg::FillOp>(loc, outElement, partial));

    // Create gml_st.parallel finalizing the partial result.
    partial = getResult(rewriter.create<gml_st::ParallelOp>(
        loc, partial.getType(), c0, cWarpSize, c1, threadDist,
        [&](OpBuilder& builder, Location loc, ValueRange ivs) {
          Value laneIdx = ivs.front();
          Value partPoint = builder.create<gml_st::TileOp>(
              loc, OpFoldResult(laneIdx), oneAttr, oneAttr);
          Value initVal =
              builder.create<gml_st::MaterializeOp>(loc, partial, partPoint);
          // Create gml_st.for sequentially reducing parts of the row.
          auto forOp = builder.create<gml_st::ForOp>(
              loc, outType, laneIdx, reductionDim, cWarpSize, initVal,
              [&](OpBuilder& builder, Location loc, ValueRange ivs,
                  ValueRange outputs) {
                Value colIdx = ivs.front();
                Value partElement = outputs.front();
                using OFRs = ArrayRef<OpFoldResult>;
                Value inPoint = builder.create<gml_st::TileOp>(
                    loc, OFRs{rewriter.getIndexAttr(0), colIdx},
                    OFRs{oneAttr, oneAttr}, OFRs{oneAttr, oneAttr});
                Value inElement =
                    builder.create<gml_st::MaterializeOp>(loc, input, inPoint);
                // Clone linalg.generic op reducing two 1-element tensors.
                Operation* clonedOp = builder.clone(*genericOp.getOperation());
                clonedOp->setOperands({inElement, partElement});
                gml_st::setTransformationAttr(builder, clonedOp);
                builder.create<gml_st::SetYieldOp>(loc, clonedOp->getResult(0),
                                                   partElement, outPoint);
              });
          builder.create<gml_st::SetYieldOp>(loc, forOp.getResults(), partial,
                                             partPoint);
        }));

    // Change existing linalg.generic to warp-reduce the partial result.
    partial = rewriter.create<tensor::ExpandShapeOp>(
        loc, outType.clone({1, 32}), partial, ReassociationIndices{0, 1});
    rewriter.updateRootInPlace(genericOp, [&] {
      genericOp->setOperand(0, partial);
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
      if (isFusible(source)) return success();
      return failure();
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

std::unique_ptr<OperationPass<func::FuncOp>> createTilingGPUWarpPass() {
  return std::make_unique<TilingGPUWarpPass>();
}

}  // namespace gml_st
}  // namespace mlir
