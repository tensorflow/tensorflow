/* Copyright 2025 The OpenXLA Authors.

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

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/cpu/codegen/tiled/transforms/lowering_utils.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"

namespace xla::cpu {

#define GEN_PASS_DECL_XTILETOVECTORPASS
#define GEN_PASS_DEF_XTILETOVECTORPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

// Dims are dropped in the subview so we use the identity map.
mlir::AffineMapAttr GetIdentityMap(xtile::TiledBufferInterface op) {
  int64_t rank = op.getTile().getType().getRank();
  return mlir::AffineMapAttr::get(
      mlir::AffineMap::getMultiDimIdentityMap(rank, op.getContext()));
}

mlir::TypedValue<mlir::MemRefType> GetSubView(
    mlir::ImplicitLocOpBuilder& builder, xtile::TiledBufferInterface op) {
  auto get_static_fold_result = [&](llvm::ArrayRef<int64_t> input) {
    return llvm::map_to_vector(input, [&builder](int64_t value) {
      return mlir::OpFoldResult(builder.getIndexAttr(value));
    });
  };

  auto offsets = llvm::SmallVector<mlir::OpFoldResult>(op.getOffsets());
  auto full_tile_shape = get_static_fold_result(op.getFullTileShape());
  auto strides = get_static_fold_result(op.getStrides());

  mlir::MemRefType subview_type =
      mlir::memref::SubViewOp::inferRankReducedResultType(
          op.getTile().getType().getShape(), op.getBuffer().getType(), offsets,
          full_tile_shape, get_static_fold_result(op.getStrides()));

  return builder.create<mlir::memref::SubViewOp>(
      subview_type, op.getBuffer(), offsets, full_tile_shape, strides);
}

llvm::SmallVector<mlir::Value> GetZeroIndexVector(
    mlir::ImplicitLocOpBuilder& builder, int64_t rank) {
  return llvm::SmallVector<mlir::Value>(
      rank, builder.create<mlir::arith::ConstantIndexOp>(0));
}

mlir::ArrayAttr GetInBoundsAttr(mlir::ImplicitLocOpBuilder& builder,
                                int64_t rank) {
  // TODO(willfroom): Add proper support for inBounds attr.
  llvm::SmallVector<mlir::Attribute> in_bounds(rank,
                                               builder.getBoolAttr(false));
  return builder.getArrayAttr(in_bounds);
}

// Get the mask for the given transfer_<read/write> op on a subview of the
// original memeref.
// The inequality we need to satisfy in 1D is:
//  1. offset + subview_idx * stride <= size - 1
//  2. subview_idx * stride <= size - 1 - offset
//  3. subview_idx <= (size - 1 - offset) / stride
//  4. subview_idx < ((size - 1 - offset) / stride) + 1
//  5. subview_idx < (size + stride - 1 - offset) / stride
mlir::Value GetMask(mlir::ImplicitLocOpBuilder& builder,
                    xtile::TiledBufferInterface op) {
  mlir::RankedTensorType tile_tensor_type = op.getTile().getType();

  auto get_const_index_op = [&](int64_t value) {
    return builder.create<mlir::arith::ConstantIndexOp>(value);
  };

  if (tile_tensor_type.getRank() == 0) {
    // Vector transfer read/write currently don't support 0D masks.
    auto mask_0D_type = mlir::VectorType::get({1}, builder.getI1Type());
    return builder.create<mlir::vector::CreateMaskOp>(
        mask_0D_type, mlir::OpFoldResult(builder.getIndexAttr(1)));
  }

  llvm::SmallDenseSet<unsigned> reduced_dims = op.getReducedDimensions();
  llvm::SmallVector<mlir::Value> upper_bounds;
  int64_t idx = 0;
  for (auto [offset, size, stride] :
       llvm::zip(op.getOffsets(), op.getBuffer().getType().getShape(),
                 op.getStrides())) {
    if (reduced_dims.contains(idx++)) {
      continue;
    }
    upper_bounds.push_back(builder.create<mlir::arith::DivSIOp>(
        builder.create<mlir::arith::SubIOp>(
            get_const_index_op(size + stride - 1), offset),
        get_const_index_op(stride)));
  }

  auto mask_type = mlir::VectorType::get(op.getTile().getType().getShape(),
                                         builder.getI1Type());
  return builder.create<mlir::vector::CreateMaskOp>(mask_type, upper_bounds);
}

struct LowerExtractTile : mlir::OpRewritePattern<xtile::ExtractTileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      xtile::ExtractTileOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    auto vector_type = GetVectorType(op.getResult().getType());

    mlir::TypedValue<mlir::MemRefType> buffer_subview = GetSubView(builder, op);

    int64_t reduced_rank = vector_type.getRank();

    // The subview is already offset so the read has zero offsets.
    auto zero_index = GetZeroIndexVector(builder, reduced_rank);
    mlir::Value padding =
        builder.create<mlir::ub::PoisonOp>(vector_type.getElementType());
    mlir::Value mask = GetMask(builder, op);
    auto in_bounds = GetInBoundsAttr(builder, reduced_rank);

    mlir::Value vector_value = rewriter.create<mlir::vector::TransferReadOp>(
        op->getLoc(), vector_type, buffer_subview, zero_index,
        GetIdentityMap(op), padding, mask, in_bounds);

    rewriter.replaceOp(op, WriteVectorToTensor(builder, vector_value));
    return mlir::success();
  }
};

struct LowerInsertTile : mlir::OpRewritePattern<xtile::InsertTileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      xtile::InsertTileOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    mlir::TypedValue<mlir::VectorType> vector_tile =
        ReadTensorToVector(builder, op.getSource());

    mlir::TypedValue<mlir::MemRefType> buffer_subview = GetSubView(builder, op);

    int64_t reduced_rank = vector_tile.getType().getRank();

    // The subview is already offset so the write has zero offsets.
    auto zero_index = GetZeroIndexVector(builder, reduced_rank);
    mlir::Value mask = GetMask(builder, op);
    auto in_bounds = GetInBoundsAttr(builder, reduced_rank);

    mlir::vector::TransferWriteOp transfer_write =
        builder.create<mlir::vector::TransferWriteOp>(
            vector_tile, buffer_subview, zero_index, GetIdentityMap(op), mask,
            in_bounds);

    rewriter.replaceOp(op, transfer_write);
    return mlir::success();
  }
};

class XTileToVectorPass
    : public impl::XTileToVectorPassBase<XTileToVectorPass> {
 public:
  using XTileToVectorPassBase::XTileToVectorPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerExtractTile, LowerInsertTile>(context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateXTileToVectorPass() {
  return std::make_unique<XTileToVectorPass>();
}

}  // namespace xla::cpu
