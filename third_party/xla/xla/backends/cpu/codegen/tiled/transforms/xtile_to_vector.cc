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
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"

namespace xla::cpu {

#define GEN_PASS_DECL_XTILETOVECTORPASS
#define GEN_PASS_DEF_XTILETOVECTORPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

mlir::AffineMap GetFilteredDims(mlir::MLIRContext* context, unsigned rank,
                                llvm::SmallDenseSet<unsigned> reduced_dims) {
  return mlir::AffineMap::getFilteredIdentityMap(
      context, rank, [&reduced_dims](mlir::AffineDimExpr dim) {
        return !reduced_dims.contains(dim.getPosition());
      });
}

struct LowerExtractTile : mlir::OpRewritePattern<xtile::ExtractTileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      xtile::ExtractTileOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::RankedTensorType dest_tensor_type = op.getResult().getType();
    auto vector_type = mlir::VectorType::get(dest_tensor_type.getShape(),
                                             dest_tensor_type.getElementType());

    // TODO(willfroom): Add support for inBounds attr.
    mlir::Value vector_value = rewriter.create<mlir::vector::TransferReadOp>(
        op->getLoc(), vector_type, op.getSource(), op.getOffsets(),
        /*padding=*/std::nullopt,
        GetFilteredDims(rewriter.getContext(),
                        op.getSource().getType().getRank(),
                        op.getReducedDimensions()));
    mlir::UnrealizedConversionCastOp cast =
        rewriter.create<mlir::UnrealizedConversionCastOp>(
            op->getLoc(), op.getResult().getType(), vector_value);
    rewriter.replaceOp(op, cast);
    return mlir::success();
  }
};

struct LowerInsertTile : mlir::OpRewritePattern<xtile::InsertTileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      xtile::InsertTileOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::RankedTensorType source_tensor_type = op.getSource().getType();
    auto vector_type = mlir::VectorType::get(
        source_tensor_type.getShape(), source_tensor_type.getElementType());
    mlir::Value cast = rewriter
                           .create<mlir::UnrealizedConversionCastOp>(
                               op->getLoc(), vector_type, op.getSource())
                           .getResult(0);
    // TODO(willfroom): Add support for inBounds attr.
    mlir::vector::TransferWriteOp transfer_write =
        rewriter.create<mlir::vector::TransferWriteOp>(
            op->getLoc(), cast, op.getDestination(), op.getOffsets(),
            GetFilteredDims(rewriter.getContext(),
                            op.getDestination().getType().getRank(),
                            op.getReducedDimensions()));

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
