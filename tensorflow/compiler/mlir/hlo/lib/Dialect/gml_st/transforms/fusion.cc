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

Value MaterializeSpaceFromTensor(Value val, PatternRewriter& rewriter) {
  auto loc = val.getLoc();
  auto ty = val.getType().cast<RankedTensorType>();

  // Collect dimension info and materialize `dim` ops for dynamic dimensions.
  SmallVector<int64_t> static_dims;
  SmallVector<Value> dynamic_dims;
  for (const auto& it : llvm::enumerate(ty.getShape())) {
    int64_t d = it.value();
    if (d != ShapedType::kDynamicSize) {
      static_dims.push_back(d);
    } else {
      auto dyn_dim = rewriter.create<tensor::DimOp>(loc, val, it.index());
      static_dims.push_back(ShapedType::kDynamicSize);
      dynamic_dims.push_back(dyn_dim);
    }
  }

  // Materialize `space` op.
  auto space_ty = rewriter.getType<TileType>(ty.getShape());
  auto static_dims_attr = rewriter.getI64ArrayAttr(static_dims);
  return rewriter.create<SpaceOp>(loc, space_ty, dynamic_dims,
                                  static_dims_attr);
}

// TODO(frgossen): This should be an interface common to all subset ops.
Value GetSpace(Value subset) {
  Operation* subset_op = subset.getDefiningOp();

  // Case: `gml.tile`:
  if (auto tile_op = llvm::dyn_cast_or_null<TileOp>(subset_op))
    return GetSpace(tile_op.subset());

  // Otherwise, `subset` must be the space itself.
  return subset;
}

// TODO(frgossen): This should become a tiling interface.
Value WhatWillBeTheTilingIface(gml_st::DynamicBroadcastInDimOp op, Value tile,
                               PatternRewriter& rewriter) {
  auto loc = op.getLoc();
  Value operand = op.operand();
  auto operand_ty = operand.getType().cast<RankedTensorType>();
  auto tile_ty = tile.getType().cast<TileType>();
  auto result_ty = op.getType().cast<RankedTensorType>();

  // // Materialize tiled output dimensions.
  // SmallVector<Value> tiled_output_dimensions_vec;
  // for (int i = 0; i < result_ty.getRank(); i++) {
  //   auto cst = rewriter.create<arith::ConstantIndexOp>(loc, i);
  //   // TODO(frgossen): Use `tensor.dim` op and dimension reification and
  //   remove
  //   // `gml_st.dim` from the dialect.
  //   auto dim = rewriter.create<DimOp>(loc, tile, cst);
  //   tiled_output_dimensions_vec.push_back(dim);
  // }
  // auto tiled_output_dimensions =
  //     rewriter.create<tensor::FromElementsOp>(loc,
  //     tiled_output_dimensions_vec);

  // Materialize operand and result space.
  Value result_space = GetSpace(tile);
  Value operand_space = MaterializeSpaceFromTensor(operand, rewriter);

  // Materialize operand tile.
  SmallVector<int64_t> operand_tile_shape(operand_ty.getRank(),
                                          ShapedType::kDynamicSize);
  auto operand_tile_ty = rewriter.getType<TileType>(operand_tile_shape);
  // TODO(frgossen): Split this into a new `gml_st.extend_tile` op and express
  // the remainder through a `gml_st.tile` op.
  Value operand_tile = rewriter.create<OperandTileForDynamicBroadcastInDimOp>(
      loc, operand_tile_ty, tile, operand_space, result_space,
      op.broadcast_dimensions(), op.known_expanding_dimensionsAttr(),
      op.known_nonexpanding_dimensionsAttr());

  // Materialize operands' subsets.
  Value tiled_init = rewriter.create<MaterializeOp>(loc, op.init(), tile);
  Value tiled_operand =
      rewriter.create<MaterializeOp>(loc, operand, operand_tile);

  // Finally, materialize tiled broadcast.
  auto tiled_result_ty =
      RankedTensorType::get(tile_ty.getShape(), result_ty.getElementType());
  return rewriter.create<DynamicBroadcastInDimOp>(
      loc, tiled_result_ty, tiled_init, tiled_operand,
      op.broadcast_dimensions(), op.known_expanding_dimensionsAttr(),
      op.known_nonexpanding_dimensionsAttr());
}

// TODO(frgossen): This should become a tiling interface.
Value WhatWillBeTheTilingIface(mhlo::AddOp op, Value tile,
                               PatternRewriter& rewriter) {
  auto loc = op.getLoc();
  auto lhs_sub = rewriter.create<MaterializeOp>(loc, op.lhs(), tile);
  auto rhs_sub = rewriter.create<MaterializeOp>(loc, op.rhs(), tile);
  return rewriter.create<mhlo::AddOp>(loc, lhs_sub, rhs_sub);
}

struct TilingPattern : public OpRewritePattern<gml_st::MaterializeOp> {
  using OpRewritePattern<gml_st::MaterializeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gml_st::MaterializeOp op,
                                PatternRewriter& rewriter) const override {
    auto def = op.source().getDefiningOp();

    // TODO(frgossen): The below cases should eventually be replaced by the use
    // of a common tiling interface.

    // Case `dynamic_broadcast_in_dim`.
    if (auto bcast =
            llvm::dyn_cast_or_null<gml_st::DynamicBroadcastInDimOp>(def)) {
      Value result = WhatWillBeTheTilingIface(bcast, op.subset(), rewriter);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Case `add`.
    if (auto add = llvm::dyn_cast_or_null<mhlo::AddOp>(def)) {
      Value result = WhatWillBeTheTilingIface(add, op.subset(), rewriter);
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
