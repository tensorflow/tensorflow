/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/chlo_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace xla_chlo {

namespace {

// Converts binary ops that statically are determined to not broadcast directly
// to the corresponding xla_hlo non-broadcasting op.
template <typename ChloOpTy, typename HloOpTy>
struct ConvertTrivialNonBroadcastBinaryOp
    : public OpConversionPattern<ChloOpTy> {
  using OpConversionPattern<ChloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ChloOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Only rewrite for statically determinable non-broadcasting cases.
    auto lhs = operands[0].getType().dyn_cast<RankedTensorType>();
    auto rhs = operands[1].getType().dyn_cast<RankedTensorType>();
    if (!lhs || !rhs) return failure();

    // Requires rank broadcast.
    if (lhs.getRank() != rhs.getRank()) return failure();
    // Any dynamic dimension may require broadcasting and requires more
    // analysis.
    if (!lhs.hasStaticShape() || !rhs.hasStaticShape()) return failure();

    for (auto extents : llvm::zip(lhs.getShape(), rhs.getShape())) {
      auto lhs_extent = std::get<0>(extents);
      auto rhs_extent = std::get<1>(extents);
      if (lhs_extent != rhs_extent) {
        return failure();
      }
    }

    rewriter.replaceOp(op, rewriter.createOrFold<HloOpTy>(
                               op.getLoc(), operands[0], operands[1],
                               /*broadcast_dimensions=*/nullptr));
    return success();
  }
};

// Checks whether the given operand types and broadcast_dims attr represent a
// legal combination for "numpy" style broadcasting (where 1-dims are prepended
// to the smaller ranked operand until it is of the same rank as the larger).
bool IsLegalNumpyRankedBroadcast(RankedTensorType lhs_type,
                                 RankedTensorType rhs_type,
                                 DenseIntElementsAttr broadcast_dims) {
  if (lhs_type.getRank() == rhs_type.getRank()) return true;

  // Otherwise, verify that broadcast_dims strictly performs left-padding.
  auto smaller_rank = std::min(lhs_type.getRank(), rhs_type.getRank());
  auto larger_rank = std::max(lhs_type.getRank(), rhs_type.getRank());

  auto expected_extents = llvm::to_vector<4>(
      llvm::seq<int64_t>(larger_rank - smaller_rank, larger_rank));
  if (expected_extents.size() != broadcast_dims.getNumElements()) {
    return false;
  }
  return std::equal(expected_extents.begin(), expected_extents.end(),
                    broadcast_dims.getIntValues().begin());
}

// Converts a binary op with ranked broadcasting operands to explicitly
// broadcast and invoke the corresponding xla_hlo non-broadcasting op.
// Note that dynamic broadcasting supported by this pattern is only valid for
// "numpy" broadcasting semantics as defined here:
//   https://docs.scipy.org/doc/numpy/reference/ufuncs.html
// Specifically, this includes the following cases:
//   - Same rank broadcast (operands have the same static rank).
//   - Different-rank broadcast, either without a broadcast_dims attribte or
//     with the broadcast_dims attribute set to map to a prefix padding.
//   - Legal combinations of degenerate (1-dim) implicit broadcasting.
// The restriction on broadcast_dims derives from the definition of the
// `shape.broadcast` op, which only supports prefix-padding.
//
// It may be possible to expand this pattern to operate on unranked tensors in
// the future by emitting more code to dynamically differentiate based on rank.
// Whether that is of any practical benefit remains to be seen.
template <typename ChloOpTy, typename HloOpTy>
struct ConvertRankedDynamicBroadcastBinaryOp
    : public OpConversionPattern<ChloOpTy> {
  using OpConversionPattern<ChloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ChloOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Only support ranked operands.
    Value lhs = operands[0];
    Value rhs = operands[1];
    auto lhs_type = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhs_type = rhs.getType().dyn_cast<RankedTensorType>();
    auto result_type = op.getResult().getType();
    if (!lhs_type || !rhs_type) return failure();

    // Check for "numpy"-style rank broadcast.
    auto broadcast_dimensions = op.broadcast_dimensions();
    if (broadcast_dimensions &&
        !IsLegalNumpyRankedBroadcast(lhs_type, rhs_type,
                                     *op.broadcast_dimensions())) {
      // Note: It is unclear whether the general specification of explicit
      // broadcast_dimensions on binary ops is a feature we want to carry
      // forward. While it can technically be implemented for ranked-dynamic,
      // it is incompatible with unranked inputs. If this warning is emitted
      // in real programs, it is an indication that the feature should be
      // implemented versus just falling back on the more standard definition
      // of numpy-like prefix-padding.
      op.emitWarning() << "unsupported non prefix-padded dynamic rank "
                       << "broadcast_dimensions = " << *broadcast_dimensions;
      return failure();
    }

    // Compute result shape.
    auto loc = op.getLoc();
    int64_t result_rank = std::max(lhs_type.getRank(), rhs_type.getRank());
    auto shape_type = shape::ShapeType::get(rewriter.getContext());
    Value lhs_shape_v =
        rewriter.createOrFold<shape::ShapeOfOp>(loc, shape_type, lhs);
    Value rhs_shape_v =
        rewriter.createOrFold<shape::ShapeOfOp>(loc, shape_type, rhs);
    Value result_shape_v = rewriter.createOrFold<shape::BroadcastOp>(
        loc, shape_type, lhs_shape_v, rhs_shape_v, nullptr /* error */);
    Value result_extents = rewriter.createOrFold<shape::ToExtentTensorOp>(
        loc, RankedTensorType::get({result_rank}, rewriter.getIndexType()),
        result_shape_v);

    // Note that we unconditionally emit DynamicBroadcastInDim ops and let
    // downstream canonicalizations fold them away if possible. This is
    // because, in the dynamic case, there are many corner cases regarding
    // when it is safe to omit, and some of them require analysis to prove
    // properly.
    auto lhs_broadcast_dimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(result_rank - lhs_type.getRank(), result_rank));
    Value broadcasted_lhs = rewriter.create<xla_hlo::DynamicBroadcastInDimOp>(
        loc, result_type, lhs, result_extents,
        rewriter.getI64TensorAttr(lhs_broadcast_dimensions));
    auto rhs_broadcast_dimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(result_rank - rhs_type.getRank(), result_rank));
    Value broadcasted_rhs = rewriter.create<xla_hlo::DynamicBroadcastInDimOp>(
        loc, result_type, rhs, result_extents,
        rewriter.getI64TensorAttr(rhs_broadcast_dimensions));

    // And generate the final non-broadcasted binary op.
    rewriter.replaceOpWithNewOp<HloOpTy>(op, result_type, broadcasted_lhs,
                                         broadcasted_rhs,
                                         /*broadcast_dimensions=*/nullptr);
    return success();
  }
};

template <typename ChloOpTy, typename HloOpTy>
void PopulateForBinaryOp(MLIRContext *context,
                         OwningRewritePatternList *patterns) {
  patterns->insert<ConvertTrivialNonBroadcastBinaryOp<ChloOpTy, HloOpTy>>(
      context, 10);
  patterns->insert<ConvertRankedDynamicBroadcastBinaryOp<ChloOpTy, HloOpTy>>(
      context, 5);
}

}  // namespace

void PopulateLegalizeChloToHloPatterns(MLIRContext *context,
                                       OwningRewritePatternList *patterns) {
#define POPULATE_BCAST(ChloOp, HloOp) \
  PopulateForBinaryOp<ChloOp, xla_hlo::HloOp>(context, patterns);

  POPULATE_BCAST(BroadcastAddOp, AddOp);
  POPULATE_BCAST(BroadcastAndOp, AndOp);
  POPULATE_BCAST(BroadcastAtan2Op, Atan2Op);
  POPULATE_BCAST(BroadcastDivOp, DivOp);
  POPULATE_BCAST(BroadcastMaxOp, MaxOp);
  POPULATE_BCAST(BroadcastMinOp, MinOp);
  POPULATE_BCAST(BroadcastMulOp, MulOp);
  POPULATE_BCAST(BroadcastOrOp, OrOp);
  POPULATE_BCAST(BroadcastPowOp, PowOp);
  POPULATE_BCAST(BroadcastRemOp, RemOp);
  POPULATE_BCAST(BroadcastShiftLeftOp, ShiftLeftOp);
  POPULATE_BCAST(BroadcastShiftRightArithmeticOp, ShiftRightArithmeticOp);
  POPULATE_BCAST(BroadcastShiftRightLogicalOp, ShiftRightLogicalOp);
  POPULATE_BCAST(BroadcastSubOp, SubOp);
  POPULATE_BCAST(BroadcastXorOp, XorOp);
}

}  // namespace xla_chlo
}  // namespace mlir
