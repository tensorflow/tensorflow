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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/pass_detail.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

// TODO(frgossen): Move this upstream as `tensor::createDimValues`.
llvm::SmallVector<Value> createDimValues(OpBuilder &b, Location loc,
                                         Value tensorValue) {
  auto ty = tensorValue.getType().cast<RankedTensorType>();
  llvm::SmallVector<Value> dims;
  for (const auto &en : llvm::enumerate(ty.getShape())) {
    int64_t d = en.value();
    if (ShapedType::isDynamic(d)) {
      dims.push_back(b.create<tensor::DimOp>(loc, tensorValue, en.index()));
    } else {
      dims.push_back(b.create<arith::ConstantIndexOp>(loc, d));
    }
  }
  return dims;
}

Value createPointSet(OpBuilder &b, Location loc, Value space, ValueRange ivs) {
  size_t rank = ivs.size();
  SmallVector<int64_t> allDynamicOffsets(rank,
                                         ShapedType::kDynamicStrideOrOffset);
  ArrayAttr allDynamicOffsetsAttr = b.getI64ArrayAttr(allDynamicOffsets);
  return b.create<PointOp>(loc, space, ivs, allDynamicOffsetsAttr);
}

Value createTileSet(OpBuilder &b, Location loc, Value space, ValueRange ivs,
                    ValueRange upperBounds, ValueRange steps,
                    ArrayRef<int64_t> tileSizes) {
  // Compute the actual sizes of the tile.
  ArrayRef<int64_t> spaceShape = space.getType().cast<TileType>().getShape();
  size_t rank = ivs.size();
  SmallVector<int64_t> staticSizes;
  SmallVector<Value> dynamicSizes;
  staticSizes.reserve(rank);
  for (int64_t i = 0; i < static_cast<int64_t>(rank); ++i) {
    // Check if this dimension can be perfectly tiled.
    if (tileSizes[i] == 1 || (spaceShape[i] != ShapedType::kDynamicSize &&
                              spaceShape[i] % tileSizes[i] == 0)) {
      staticSizes.push_back(tileSizes[i]);
      continue;
    }

    auto nextIv = b.create<arith::AddIOp>(loc, ivs[i], steps[i]);
    auto isPartialTile = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                                 nextIv, upperBounds[i]);
    auto remainder = b.create<arith::SubIOp>(loc, upperBounds[i], ivs[i]);
    auto size =
        b.create<arith::SelectOp>(loc, isPartialTile, remainder, steps[i]);
    staticSizes.push_back(ShapedType::kDynamicSize);
    dynamicSizes.push_back(size);
  }

  SmallVector<int64_t> allDynamicOffsets(rank,
                                         ShapedType::kDynamicStrideOrOffset);
  SmallVector<int64_t> allUnitStrides(rank, 1);
  auto staticSizesAttr = b.getI64ArrayAttr(staticSizes);
  auto staticOffsetsAttr = b.getI64ArrayAttr(allDynamicOffsets);
  auto staticStridesAttr = b.getI64ArrayAttr(allUnitStrides);
  auto tileTy = b.getType<TileType>(staticSizes);
  return b.create<TileOp>(loc, tileTy, space, ivs, dynamicSizes, ValueRange{},
                          staticOffsetsAttr, staticSizesAttr,
                          staticStridesAttr);
}

Value createSet(OpBuilder &b, Location loc, Value space, ValueRange ivs,
                ValueRange upperBounds, ValueRange steps,
                ArrayRef<int64_t> tileSizes) {
  if (llvm::all_of(tileSizes, [](int64_t s) { return s == 1; })) {
    return createPointSet(b, loc, space, ivs);
  }
  return createTileSet(b, loc, space, ivs, upperBounds, steps, tileSizes);
}

Value createParallelLoopTiling(OpBuilder &b, Location loc, Value target,
                               ArrayRef<int64_t> tileSizes) {
  auto ty = target.getType().cast<RankedTensorType>();
  assert(ty.getRank() == static_cast<int64_t>(tileSizes.size()) &&
         "expect tile sizes to match rank of target value");

  // Create space.
  SmallVector<Value> dynamicDims =
      tensor::createDynamicDimValues(b, loc, target);
  auto spaceTy = b.getType<TileType>(ty.getShape());
  Value space = b.create<SpaceOp>(loc, spaceTy, dynamicDims,
                                  b.getI64ArrayAttr(ty.getShape()));

  // Create init tensor.
  auto init = b.create<linalg::InitTensorOp>(loc, dynamicDims, ty.getShape(),
                                             ty.getElementType());

  // Create loop bounds.
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> lowerBounds(ty.getRank(), zero);
  SmallVector<Value> upperBounds = createDimValues(b, loc, target);
  auto steps =
      llvm::to_vector(llvm::map_range(tileSizes, [&](int64_t s) -> Value {
        return b.create<arith::ConstantIndexOp>(loc, s).getResult();
      }));

  // Create ploop.
  auto ploop = b.create<ParallelOp>(
      loc, ty, lowerBounds, upperBounds, steps,
      [&](OpBuilder &b, Location loc, ValueRange ivs) {
        Value set =
            createSet(b, loc, space, ivs, upperBounds, steps, tileSizes);
        auto materialized = b.create<MaterializeOp>(loc, target, set);
        b.create<SetYieldOp>(loc, ValueRange{materialized}, ValueRange{init},
                             ValueRange{set});
      });
  return ploop.getResults().front();
}

const llvm::StringLiteral kHasTiledOperandsAttrName = "__has_tiled_operands";

template <typename OpTy>
struct AllOperandsTilingPattern : public OpRewritePattern<OpTy> {
  AllOperandsTilingPattern(ArrayRef<int64_t> tileSizes, MLIRContext *context,
                           PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>(context, benefit),
        tileSizes(tileSizes.begin(), tileSizes.end()) {}

  LogicalResult matchAndRewrite(OpTy rootOp,
                                PatternRewriter &rewriter) const override {
    // Avoid infinite rewrites.
    if (rootOp->hasAttr(kHasTiledOperandsAttrName)) return failure();

    // Fail if any of the operands is not a tensor of the expected rank.
    int64_t rank = tileSizes.size();
    if (!llvm::all_of(rootOp->getOperandTypes(), [&](Type ty) {
          auto rankedTy = ty.dyn_cast<RankedTensorType>();
          return rankedTy && rankedTy.getRank() == rank;
        })) {
      return failure();
    }

    // Tile all operands.
    auto tiledOperands = llvm::to_vector(
        llvm::map_range(rootOp->getOperands(), [&](auto operand) {
          return createParallelLoopTiling(rewriter, operand.getLoc(), operand,
                                          tileSizes);
        }));

    // Replace root op and mark it as tiled.
    auto res = rewriter.replaceOpWithNewOp<func::ReturnOp>(
        rootOp, rootOp->getResultTypes(), tiledOperands, rootOp->getAttrs());
    res->setAttr(kHasTiledOperandsAttrName, rewriter.getUnitAttr());

    return success();
  }

 private:
  SmallVector<int64_t> tileSizes;
};

struct TilingPass : public TilingPassBase<TilingPass> {
  TilingPass() = default;
  explicit TilingPass(llvm::ArrayRef<int64_t> sizes) { tileSizes = sizes; }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<GmlStDialect>();
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<AllOperandsTilingPattern<func::ReturnOp>>(tileSizes, ctx);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Clean up by removing temporary attributes.
    f.walk([](Operation *op) { op->removeAttr(kHasTiledOperandsAttrName); });
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createTilingPass(
    ArrayRef<int64_t> tileSizes) {
  return std::make_unique<TilingPass>(tileSizes);
}

}  // namespace gml_st
}  // namespace mlir
