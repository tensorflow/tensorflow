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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"

namespace xla::cpu {

#define GEN_PASS_DECL_REWRITEDYNAMICVECTOREXTRACTPASS
#define GEN_PASS_DEF_REWRITEDYNAMICVECTOREXTRACTPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

// Check if the given extract operands depends on the given value.
bool ExtractDependsOnValue(mlir::vector::ExtractOp extract_op,
                           mlir::Value value) {
  mlir::BackwardSliceOptions backward_slice_options;
  backward_slice_options.omitUsesFromAbove = false;
  backward_slice_options.inclusive = true;

  for (mlir::Value dynamic_index : extract_op.getDynamicPosition()) {
    // We have to explicitly check the index itself as getBackwardSlice only
    // starts from the defining operation.
    if (dynamic_index == value) {
      return true;
    }

    llvm::SetVector<mlir::Operation*> backwardSlice;
    if (mlir::failed(mlir::getBackwardSlice(dynamic_index, &backwardSlice,
                                            backward_slice_options))) {
      continue;
    }

    for (mlir::Operation* op : backwardSlice) {
      for (mlir::Value operand : op->getOperands()) {
        if (operand == value) {
          return true;
        }
      }
    }
  }

  return false;
}

struct FoldExtractIntoTransferRead
    : mlir::OpRewritePattern<mlir::vector::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::vector::ExtractOp op,
      mlir::PatternRewriter& rewriter) const override {
    if (!op.hasDynamicPosition()) {
      return rewriter.notifyMatchFailure(
          op, "extract does not have dynamic position");
    }

    auto transfer_read_op =
        op.getSource().getDefiningOp<mlir::vector::TransferReadOp>();

    if (!transfer_read_op) {
      return rewriter.notifyMatchFailure(op,
                                         "source is not a transfer_read op");
    }

    mlir::ValueRange transfer_read_indices = transfer_read_op.getIndices();

    llvm::SmallVector<mlir::OpFoldResult> extended_positions(
        op.getMixedPosition());
    for (int64_t idx = extended_positions.size();
         idx < transfer_read_indices.size(); ++idx) {
      extended_positions.push_back(rewriter.getIndexAttr(0));
    }

    llvm::SmallVector<mlir::Value> new_offsets;
    new_offsets.reserve(transfer_read_indices.size());
    for (auto [tile_offset, extract_offset] :
         llvm::zip(transfer_read_indices, extended_positions)) {
      if (auto static_position =
              mlir::dyn_cast<mlir::Attribute>(extract_offset)) {
        new_offsets.push_back(mlir::arith::AddIOp::create(
            rewriter, op.getLoc(), rewriter.getIndexType(), tile_offset,
            mlir::arith::ConstantIndexOp::create(
                rewriter, op.getLoc(),
                mlir::cast<mlir::IntegerAttr>(static_position).getInt())));
      } else {
        auto dynamic_position = mlir::dyn_cast<mlir::Value>(extract_offset);
        new_offsets.push_back(mlir::arith::AddIOp::create(
            rewriter, op.getLoc(), rewriter.getIndexType(), tile_offset,
            dynamic_position));
      }
    }

    // Output of extract can be either a vector or a scalar.
    auto maybe_vector_type = mlir::dyn_cast<mlir::VectorType>(op.getType());

    mlir::Value submask;
    if (auto mask = transfer_read_op.getMask()) {
      // Transfer read result and mask must be non-0D vectors.
      auto sub_mask_type = mlir::VectorType::get(
          maybe_vector_type ? maybe_vector_type.getShape() : 1,
          rewriter.getI1Type());
      submask = mlir::vector::ExtractOp::create(
          rewriter, op.getLoc(), sub_mask_type, mask, op.getDynamicPosition(),
          op.getStaticPosition());
    }

    int64_t input_rank = transfer_read_op.getBase().getType().getRank();
    int64_t output_rank = maybe_vector_type ? maybe_vector_type.getRank() : 1;

    // Drop major dimensions which reflects the behaviour of vector::ExtractOp.
    int64_t num_dropped_dims = input_rank - output_rank;
    mlir::AffineMap new_permutation_map =
        mlir::AffineMap::getFilteredIdentityMap(
            rewriter.getContext(), input_rank, [&](mlir::AffineDimExpr expr) {
              return expr.getPosition() >= num_dropped_dims;
            });

    llvm::SmallVector<mlir::Attribute> in_bounds(
        transfer_read_op.getInBounds().begin() + num_dropped_dims,
        transfer_read_op.getInBounds().end());

    auto output_type = mlir::VectorType::get(
        maybe_vector_type ? maybe_vector_type.getShape() : 1,
        mlir::getElementTypeOrSelf(op.getType()));
    auto new_transfer_read = mlir::vector::TransferReadOp::create(
        rewriter, op.getLoc(), output_type, transfer_read_op.getBase(),
        new_offsets, new_permutation_map, transfer_read_op.getPadding(),
        submask, rewriter.getArrayAttr(in_bounds));

    if (maybe_vector_type) {
      rewriter.replaceOp(op, new_transfer_read);
    } else {
      rewriter.replaceOpWithNewOp<mlir::vector::ExtractOp>(
          op, new_transfer_read, 0);
    }

    return mlir::success();
  }
};

// FoldExtractIntoTransferRead creates its own dynamic extracts if a mask is
// present, so we need to fold these.
// We do this by shifting the offset and then extracting with static indices.
struct FoldExtractIntoCreateMask
    : mlir::OpRewritePattern<mlir::vector::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::vector::ExtractOp op,
      mlir::PatternRewriter& rewriter) const override {
    if (!op.hasDynamicPosition()) {
      return rewriter.notifyMatchFailure(
          op, "extract does not have dynamic position");
    }
    auto mask_op = op.getSource().getDefiningOp<mlir::vector::CreateMaskOp>();
    if (!mask_op) {
      return rewriter.notifyMatchFailure(op, "source is not a create_mask op");
    }

    mlir::ValueRange mask_operands = mask_op.getOperands();

    llvm::SmallVector<mlir::OpFoldResult> extended_positions(
        op.getMixedPosition());
    for (int64_t idx = extended_positions.size(); idx < mask_operands.size();
         ++idx) {
      extended_positions.push_back(rewriter.getIndexAttr(0));
    }

    llvm::SmallVector<mlir::Value> new_bounds;
    new_bounds.reserve(mask_operands.size());
    for (auto [mask_bound, extract_offset] :
         llvm::zip(mask_operands, extended_positions)) {
      if (auto static_position =
              mlir::dyn_cast<mlir::Attribute>(extract_offset)) {
        new_bounds.push_back(mlir::arith::SubIOp::create(
            rewriter, op.getLoc(), rewriter.getIndexType(), mask_bound,
            mlir::arith::ConstantIndexOp::create(
                rewriter, op.getLoc(),
                mlir::cast<mlir::IntegerAttr>(static_position).getInt())));
      } else {
        auto dynamic_position = mlir::dyn_cast<mlir::Value>(extract_offset);
        new_bounds.push_back(mlir::arith::SubIOp::create(
            rewriter, op.getLoc(), rewriter.getIndexType(), mask_bound,
            dynamic_position));
      }
    }

    // As we are going to extract only the output shape vector from the minor
    // dimensions we only need to create a mask of size 1 for the remaining
    // dimensions.
    llvm::SmallVector<int64_t> new_mask_shape;
    if (auto output_shape = mlir::dyn_cast<mlir::ShapedType>(op.getType())) {
      new_mask_shape.assign(
          mask_op.getType().getRank() - output_shape.getRank(), 1);
      new_mask_shape.append(output_shape.getShape().begin(),
                            output_shape.getShape().end());
    } else {
      new_mask_shape.assign(mask_op.getType().getRank(), 1);
    }

    auto shifted_mask = mlir::vector::CreateMaskOp::create(
        rewriter, op.getLoc(),
        mlir::VectorType::get(new_mask_shape, rewriter.getI1Type()),
        new_bounds);

    llvm::SmallVector<int64_t> zero_index(op.getMixedPosition().size(), 0);
    rewriter.replaceOpWithNewOp<mlir::vector::ExtractOp>(
        op, op.getType(), shifted_mask, mlir::ValueRange{}, zero_index);

    return mlir::success();
  }
};

// Unroll loops that have a vector.extract that depend on the loop induction
// variable.
struct UnrollExtractLoops : mlir::OpRewritePattern<mlir::scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::scf::ForOp for_op, mlir::PatternRewriter& rewriter) const override {
    std::optional<mlir::LogicalResult> unroll_result;
    // Walk the body of the loop and unroll if we have a dependent extract.
    for_op.getBody()->walk([&](mlir::vector::ExtractOp extract) {
      if (!ExtractDependsOnValue(extract, for_op.getInductionVar())) {
        return mlir::WalkResult::advance();
      }
      unroll_result = mlir::loopUnrollFull(for_op);
      return mlir::WalkResult::interrupt();
    });

    if (!unroll_result.has_value()) {
      return rewriter.notifyMatchFailure(
          for_op, "loop does not contain a dependent extract");
    }

    if (mlir::failed(*unroll_result)) {
      return rewriter.notifyMatchFailure(for_op, "failed to unroll loop");
    }

    return mlir::success();
  }
};

class RewriteDynamicVectorExtractPass
    : public impl::RewriteDynamicVectorExtractPassBase<
          RewriteDynamicVectorExtractPass> {
 public:
  using RewriteDynamicVectorExtractPassBase::
      RewriteDynamicVectorExtractPassBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext* context = &getContext();

    {
      mlir::RewritePatternSet patterns(context);
      patterns.add<FoldExtractIntoTransferRead, FoldExtractIntoCreateMask>(
          context);
      if (mlir::failed(
              mlir::applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    // As a final sledge hammer, we can unroll the loops if we have any
    // dependent extracts.
    {
      mlir::RewritePatternSet patterns(context);
      patterns.add<UnrollExtractLoops>(context);
      if (mlir::failed(
              mlir::applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateRewriteDynamicVectorExtractPass() {
  return std::make_unique<RewriteDynamicVectorExtractPass>();
}

}  // namespace xla::cpu
