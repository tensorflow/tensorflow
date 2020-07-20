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
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/broadcast_utils.h"

namespace mlir {
namespace chlo {

namespace {

// Converts binary ops that statically are determined to not broadcast directly
// to the corresponding mhlo non-broadcasting op.
template <typename ChloOpTy, typename HloOpTy, typename Adaptor>
struct ConvertTrivialNonBroadcastBinaryOp : public OpRewritePattern<ChloOpTy> {
  using OpRewritePattern<ChloOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(ChloOpTy op,
                                PatternRewriter &rewriter) const override {
    // Only rewrite for statically determinable non-broadcasting cases.
    auto lhs_type = op.lhs().getType().template dyn_cast<RankedTensorType>();
    auto rhs_type = op.rhs().getType().template dyn_cast<RankedTensorType>();
    if (!lhs_type || !rhs_type) return failure();

    // Requires rank broadcast.
    if (lhs_type.getRank() != rhs_type.getRank()) return failure();
    // Any dynamic dimension may require broadcasting and requires more
    // analysis.
    if (!lhs_type.hasStaticShape() || !rhs_type.hasStaticShape())
      return failure();

    for (auto extents : llvm::zip(lhs_type.getShape(), rhs_type.getShape())) {
      auto lhs_extent = std::get<0>(extents);
      auto rhs_extent = std::get<1>(extents);
      if (lhs_extent != rhs_extent) {
        return failure();
      }
    }

    rewriter.replaceOp(op, {Adaptor::CreateOp(op, op.getResult().getType(),
                                              op.lhs(), op.rhs(), rewriter)});
    return success();
  }
};

// Converts a binary op with ranked broadcasting operands to explicitly
// broadcast and invoke the corresponding mhlo non-broadcasting op.
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
template <typename ChloOpTy, typename HloOpTy, typename Adaptor>
struct ConvertRankedDynamicBroadcastBinaryOp
    : public OpRewritePattern<ChloOpTy> {
  using OpRewritePattern<ChloOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(ChloOpTy op,
                                PatternRewriter &rewriter) const override {
    // Only support ranked operands.
    Value lhs = op.lhs();
    Value rhs = op.rhs();
    auto lhs_type = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhs_type = rhs.getType().dyn_cast<RankedTensorType>();
    auto result_type =
        op.getResult().getType().template dyn_cast<RankedTensorType>();
    if (!lhs_type || !rhs_type || !result_type) return failure();

    // Check for "numpy"-style rank broadcast.
    auto broadcast_dimensions = op.broadcast_dimensions();
    if (broadcast_dimensions &&
        !hlo::IsLegalNumpyRankedBroadcast(lhs, rhs, *broadcast_dimensions)) {
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

    // Insert a constraint on the shapes being broadcastable and insert all
    // future code into an assuming block reliant on the constraint.
    Value lhs_shape = rewriter.create<shape::ShapeOfOp>(loc, lhs);
    Value rhs_shape = rewriter.create<shape::ShapeOfOp>(loc, rhs);
    auto broadcastable_cstr =
        rewriter.create<shape::CstrBroadcastableOp>(loc, lhs_shape, rhs_shape);
    auto assuming_op = rewriter.create<shape::AssumingOp>(
        loc, ArrayRef<Type>{result_type}, broadcastable_cstr.result());

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&assuming_op.doRegion());

    int64_t result_rank = std::max(lhs_type.getRank(), rhs_type.getRank());
    Value result_extents =
        hlo::ComputeBinaryElementwiseBroadcastingResultExtents(loc, lhs, rhs,
                                                               rewriter);

    // Note that we unconditionally emit DynamicBroadcastInDim ops and let
    // downstream canonicalizations fold them away if possible. This is
    // because, in the dynamic case, there are many corner cases regarding
    // when it is safe to omit, and some of them require analysis to prove
    // properly.
    auto lhs_broadcast_dimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(result_rank - lhs_type.getRank(), result_rank));
    Value broadcasted_lhs = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc,
        RankedTensorType::get(result_type.getShape(),
                              lhs_type.getElementType()),
        lhs, result_extents,
        rewriter.getI64TensorAttr(lhs_broadcast_dimensions));
    auto rhs_broadcast_dimensions = llvm::to_vector<4>(
        llvm::seq<int64_t>(result_rank - rhs_type.getRank(), result_rank));
    Value broadcasted_rhs = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc,
        RankedTensorType::get(result_type.getShape(),
                              rhs_type.getElementType()),
        rhs, result_extents,
        rewriter.getI64TensorAttr(rhs_broadcast_dimensions));

    // And generate the final non-broadcasted binary op.
    Value final_result = Adaptor::CreateOp(op, result_type, broadcasted_lhs,
                                           broadcasted_rhs, rewriter);
    rewriter.create<shape::AssumingYieldOp>(loc, final_result);
    rewriter.replaceOp(op, {assuming_op.getResult(0)});
    return success();
  }
};

template <typename ChloOpTy, typename HloOpTy, typename Adaptor>
void PopulateForBinaryOp(MLIRContext *context,
                         OwningRewritePatternList *patterns) {
  patterns
      ->insert<ConvertTrivialNonBroadcastBinaryOp<ChloOpTy, HloOpTy, Adaptor>>(
          context, 10);
  patterns->insert<
      ConvertRankedDynamicBroadcastBinaryOp<ChloOpTy, HloOpTy, Adaptor>>(
      context, 5);
}

template <typename FromOpTy, typename ToOpTy>
struct HloBinaryElementwiseAdaptor {
  static ToOpTy CreateOp(FromOpTy from_op, Type result_type,
                         Value broadcasted_lhs, Value broadcasted_rhs,
                         OpBuilder &builder) {
    return builder.create<ToOpTy>(from_op.getLoc(), result_type,
                                  broadcasted_lhs, broadcasted_rhs);
  }
};

struct HloComplexAdaptor {
  static mhlo::ComplexOp CreateOp(BroadcastComplexOp from_op, Type result_type,
                                  Value broadcasted_lhs, Value broadcasted_rhs,
                                  OpBuilder &builder) {
    return builder.create<mhlo::ComplexOp>(from_op.getLoc(), result_type,
                                           broadcasted_lhs, broadcasted_rhs);
  }
};

struct HloCompareAdaptor {
  static mhlo::CompareOp CreateOp(BroadcastCompareOp from_op, Type result_type,
                                  Value broadcasted_lhs, Value broadcasted_rhs,
                                  OpBuilder &builder) {
    return builder.create<mhlo::CompareOp>(from_op.getLoc(), result_type,
                                           broadcasted_lhs, broadcasted_rhs,
                                           from_op.comparison_direction());
  }
};

}  // namespace

void PopulateLegalizeChloToHloPatterns(MLIRContext *context,
                                       OwningRewritePatternList *patterns) {
  // Instantiate conversion templates for conforming binary elementwise ops
  // that do not have different dtypes between operands and results and do
  // not have special attributes that need to be preserved.
#define POPULATE_BCAST(ChloOp, HloOp)                                      \
  PopulateForBinaryOp<ChloOp, HloOp,                                       \
                      HloBinaryElementwiseAdaptor<ChloOp, HloOp>>(context, \
                                                                  patterns);

  POPULATE_BCAST(BroadcastAddOp, mhlo::AddOp);
  POPULATE_BCAST(BroadcastAndOp, mhlo::AndOp);
  POPULATE_BCAST(BroadcastAtan2Op, mhlo::Atan2Op);
  POPULATE_BCAST(BroadcastDivOp, mhlo::DivOp);
  POPULATE_BCAST(BroadcastMaxOp, mhlo::MaxOp);
  POPULATE_BCAST(BroadcastMinOp, mhlo::MinOp);
  POPULATE_BCAST(BroadcastMulOp, mhlo::MulOp);
  POPULATE_BCAST(BroadcastOrOp, mhlo::OrOp);
  POPULATE_BCAST(BroadcastPowOp, mhlo::PowOp);
  POPULATE_BCAST(BroadcastRemOp, mhlo::RemOp);
  POPULATE_BCAST(BroadcastShiftLeftOp, mhlo::ShiftLeftOp);
  POPULATE_BCAST(BroadcastShiftRightArithmeticOp, mhlo::ShiftRightArithmeticOp);
  POPULATE_BCAST(BroadcastShiftRightLogicalOp, mhlo::ShiftRightLogicalOp);
  POPULATE_BCAST(BroadcastSubOp, mhlo::SubOp);
  POPULATE_BCAST(BroadcastXorOp, mhlo::XorOp);

  // Broadcasting ops requiring special construction.
  PopulateForBinaryOp<BroadcastComplexOp, mhlo::ComplexOp, HloComplexAdaptor>(
      context, patterns);
  PopulateForBinaryOp<BroadcastCompareOp, mhlo::CompareOp, HloCompareAdaptor>(
      context, patterns);
}

}  // namespace chlo
}  // namespace mlir
