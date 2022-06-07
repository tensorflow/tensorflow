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

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/pass_detail.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

Value materializeSpaceFromTensor(Value val, PatternRewriter& rewriter) {
  auto loc = val.getLoc();
  auto ty = val.getType().cast<RankedTensorType>();

  // Collect dimension info and materialize `dim` ops for dynamic dimensions.
  SmallVector<int64_t> staticDims;
  SmallVector<Value> dynamicDims;
  for (const auto& it : llvm::enumerate(ty.getShape())) {
    int64_t d = it.value();
    if (d != ShapedType::kDynamicSize) {
      staticDims.push_back(d);
    } else {
      auto dynDim = rewriter.create<tensor::DimOp>(loc, val, it.index());
      staticDims.push_back(ShapedType::kDynamicSize);
      dynamicDims.push_back(dynDim);
    }
  }

  // Materialize `space` op.
  auto spaceTy = rewriter.getType<TileType>(ty.getShape());
  auto staticDimsAttr = rewriter.getI64ArrayAttr(staticDims);
  return rewriter.create<SpaceOp>(loc, spaceTy, dynamicDims, staticDimsAttr);
}

// TODO(frgossen): This should become a tiling interface.
Value whatWillBeTheTilingIface(gml_st::DynamicBroadcastInDimOp op, Value tile,
                               PatternRewriter& rewriter) {
  auto loc = op.getLoc();
  DenseMap<int64_t, Value> localCsts;
  auto getCst = [&](int64_t c) -> Value {
    auto it = localCsts.find(c);
    if (it != localCsts.end()) return it->second;
    auto cst = rewriter.create<arith::ConstantIndexOp>(loc, c);
    localCsts[c] = cst;
    return cst;
  };

  Value operand = op.operand();
  auto operandTy = operand.getType().cast<RankedTensorType>();
  auto tileTy = tile.getType().cast<TileType>();
  auto resultTy = op.getType().cast<RankedTensorType>();

  // Materialize operand and result space.
  Value operandSpace = materializeSpaceFromTensor(operand, rewriter);

  // Materialize offsets and sizes for operand tile.
  auto collapsedTile =
      rewriter.create<CollapseTileOp>(loc, tile, op.broadcast_dimensions());
  SmallVector<Value> argTileOffsets;
  SmallVector<Value> argTileSizes;
  for (const auto& it : llvm::enumerate(op.broadcast_dimensions())) {
    Value argIdx = getCst(it.index());
    Value resultIdx = getCst(it.value().getLimitedValue());

    // If corresponding operand and result dimensions are different, the
    // dimension is expanding.
    // TODO(frgossen): Share these dim ops with those created for the operand
    // space above.
    auto argDim = rewriter.create<tensor::DimOp>(loc, op.operand(), argIdx);
    auto resultDim = rewriter.create<tensor::DimOp>(loc, op.init(), resultIdx);
    auto isExpanding = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, argDim, resultDim);

    // Copy offset for non-expanding dimensions and index at 0, otrherwise.
    auto tileOffset = rewriter.create<OffsetOp>(loc, collapsedTile, argIdx);
    auto tileSize = rewriter.create<SizeOp>(loc, collapsedTile, argIdx);
    argTileOffsets.push_back(rewriter.create<arith::SelectOp>(
        loc, isExpanding, getCst(0), tileOffset));
    argTileSizes.push_back(rewriter.create<arith::SelectOp>(
        loc, isExpanding, getCst(1), tileSize));
  }

  // Materialize operand tile.
  int64_t rank = operandTy.getRank();
  auto staticOffsets = rewriter.getI64ArrayAttr(
      SmallVector<int64_t>(rank, ShapedType::kDynamicStrideOrOffset));
  auto staticSizes = rewriter.getI64ArrayAttr(
      SmallVector<int64_t>(rank, ShapedType::kDynamicSize));
  auto staticStrides = rewriter.getI64ArrayAttr(SmallVector<int64_t>(rank, 1));
  auto operandTileTy = rewriter.getType<TileType>(
      SmallVector<int64_t>(rank, ShapedType::kDynamicSize));
  auto operandTile = rewriter.create<TileOp>(
      loc, operandTileTy, operandSpace, argTileOffsets, argTileSizes,
      ValueRange{}, staticOffsets, staticSizes, staticStrides);

  // Materialize operands' subsets.
  Value tiledInit = rewriter.create<MaterializeOp>(loc, op.init(), tile);
  Value tiledOperand =
      rewriter.create<MaterializeOp>(loc, operand, operandTile);

  // Finally, materialize tiled broadcast.
  auto tiledResultTy =
      RankedTensorType::get(tileTy.getShape(), resultTy.getElementType());
  return rewriter.create<DynamicBroadcastInDimOp>(
      loc, tiledResultTy, tiledInit, tiledOperand, op.broadcast_dimensions(),
      op.known_expanding_dimensionsAttr(),
      op.known_nonexpanding_dimensionsAttr());
}

// TODO(frgossen): This should become a tiling interface.
Value whatWillBeTheTilingIface(mhlo::AddOp op, Value tile,
                               PatternRewriter& rewriter) {
  auto loc = op.getLoc();
  auto lhsSub = rewriter.create<MaterializeOp>(loc, op.lhs(), tile);
  auto rhsSub = rewriter.create<MaterializeOp>(loc, op.rhs(), tile);
  return rewriter.create<mhlo::AddOp>(loc, lhsSub, rhsSub);
}

struct TilingPattern : public OpRewritePattern<gml_st::MaterializeOp> {
  using OpRewritePattern<gml_st::MaterializeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gml_st::MaterializeOp op,
                                PatternRewriter& rewriter) const override {
    Operation* def = op.source().getDefiningOp();

    // TODO(frgossen): The below cases should eventually be replaced by the use
    // of a common tiling interface.

    // Case `dynamic_broadcast_in_dim`.
    if (auto bcast =
            llvm::dyn_cast_or_null<gml_st::DynamicBroadcastInDimOp>(def)) {
      Value result = whatWillBeTheTilingIface(bcast, op.subset(), rewriter);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Case `add`.
    if (auto add = llvm::dyn_cast_or_null<mhlo::AddOp>(def)) {
      Value result = whatWillBeTheTilingIface(add, op.subset(), rewriter);
      rewriter.replaceOp(op, result);
      return success();
    }

    return failure();
  }
};

class FusionPass : public FusionPassBase<FusionPass> {
  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<GmlStDialect>();
  }

  void runOnOperation() final {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);

    // List of patterns.
    patterns.insert<TilingPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createFusionPass() {
  return std::make_unique<FusionPass>();
}

}  // namespace gml_st
}  // namespace mlir
