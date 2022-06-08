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
#include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface.h"
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

Value materializeSpaceFromTensor(Value operand, const SmallVector<Value>& dims,
                                 PatternRewriter& rewriter) {
  auto loc = operand.getLoc();
  auto ty = operand.getType().cast<RankedTensorType>();

  // Collect dimension info and materialize `dim` ops for dynamic dimensions.
  SmallVector<int64_t> staticDims;
  SmallVector<Value> dynamicDims;
  for (const auto& it : llvm::enumerate(ty.getShape())) {
    int64_t d = it.value();
    if (d != ShapedType::kDynamicSize) {
      staticDims.push_back(d);
    } else {
      staticDims.push_back(ShapedType::kDynamicSize);
      dynamicDims.push_back(dims[it.index()]);
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
  DenseMap<uint64_t, Value> localIndexCsts;
  auto getIndexCst = [&](uint64_t c) -> Value {
    auto it = localIndexCsts.find(c);
    if (it != localIndexCsts.end()) return it->second;
    auto cst = rewriter.create<arith::ConstantIndexOp>(loc, c);
    localIndexCsts[c] = cst;
    return cst;
  };

  Value operand = op.operand();
  auto operandTy = operand.getType().cast<RankedTensorType>();
  auto tileTy = tile.getType().cast<TileType>();
  auto resultTy = op.getType().cast<RankedTensorType>();

  // Materiaize operand dimensions.
  SmallVector<Value> operandDims;
  operandDims.reserve(operandTy.getRank());
  for (const auto& it : llvm::enumerate(operandTy.getShape())) {
    int64_t d = it.value();
    Value dim =
        d == ShapedType::kDynamicSize
            ? rewriter.create<tensor::DimOp>(loc, operand, it.index())
                  .getResult()
            : rewriter.create<arith::ConstantIndexOp>(loc, d).getResult();
    operandDims.push_back(dim);
  }

  // Materialize operand and result space.
  Value operandSpace =
      materializeSpaceFromTensor(operand, operandDims, rewriter);

  // Materialize offsets and sizes for operand tile.
  auto collapsedTile =
      rewriter.create<CollapseTileOp>(loc, tile, op.broadcast_dimensions());
  SmallVector<Value> argTileOffsets;
  SmallVector<Value> argTileSizes;
  for (const auto& it : llvm::enumerate(op.broadcast_dimensions())) {
    Value argIdx = getIndexCst(it.index());
    Value resultIdx = getIndexCst(it.value().getLimitedValue());

    // If corresponding operand and result dimensions are different, the
    // dimension is expanding.
    auto argDim = operandDims[it.index()];
    auto resultDim = rewriter.create<tensor::DimOp>(loc, op.init(), resultIdx);
    auto isExpanding = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, argDim, resultDim);

    // Copy offset for non-expanding dimensions and index at 0, otrherwise.
    auto tileOffset = rewriter.create<OffsetOp>(loc, collapsedTile, argIdx);
    auto tileSize = rewriter.create<SizeOp>(loc, collapsedTile, argIdx);
    argTileOffsets.push_back(rewriter.create<arith::SelectOp>(
        loc, isExpanding, getIndexCst(0), tileOffset));
    argTileSizes.push_back(rewriter.create<arith::SelectOp>(
        loc, isExpanding, getIndexCst(1), tileSize));
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
template <class OpTy>
Value whatWillBeTheTilingIfaceUnaryOp(OpTy op, Value tile,
                                      PatternRewriter& rewriter) {
  auto loc = op.getLoc();
  auto operandSub = rewriter.create<MaterializeOp>(loc, op.operand(), tile);
  return rewriter.create<OpTy>(loc, operandSub);
}

struct TilingPattern : public OpRewritePattern<gml_st::MaterializeOp> {
  using OpRewritePattern<gml_st::MaterializeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gml_st::MaterializeOp op,
                                PatternRewriter& rewriter) const override {
    Operation* def = op.source().getDefiningOp();

    if (auto iface = llvm::dyn_cast_or_null<FusionIterface>(def)) {
      rewriter.replaceOp(op, iface.fuse(op, rewriter));
      return success();
    }

    // TODO(frgossen): The below cases should eventually be replaced by the use
    // of a common tiling interface.

    // Case `dynamic_broadcast_in_dim`.
    if (auto bcast =
            llvm::dyn_cast_or_null<gml_st::DynamicBroadcastInDimOp>(def)) {
      Value result = whatWillBeTheTilingIface(bcast, op.subset(), rewriter);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Case `cos`.
    if (auto cos = llvm::dyn_cast_or_null<mhlo::CosOp>(def)) {
      rewriter.replaceOp(
          op, whatWillBeTheTilingIfaceUnaryOp(cos, op.subset(), rewriter));
      return success();
    }

    // Case `tanh`.
    if (auto tanh = llvm::dyn_cast_or_null<mhlo::TanhOp>(def)) {
      rewriter.replaceOp(
          op, whatWillBeTheTilingIfaceUnaryOp(tanh, op.subset(), rewriter));
      return success();
    }

    return failure();
  }
};

class FusionPass : public FusionPassBase<FusionPass> {
  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<GmlStDialect>();
    registerFusionInterfaceExternalModels(registry);
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
