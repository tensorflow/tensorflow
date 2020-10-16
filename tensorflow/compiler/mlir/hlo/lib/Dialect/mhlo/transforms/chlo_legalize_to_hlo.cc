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

#include <numeric>

#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir-hlo/utils/broadcast_utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace chlo {
namespace {

struct ConvertConstantLikeOp : public OpConversionPattern<ConstantLikeOp> {
  using OpConversionPattern<ConstantLikeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ConstantLikeOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto result_ty = op.getType().cast<ShapedType>();

    // Unranked uses are not supported.  Consider `transform-unranked-hlo`.
    if (!result_ty.hasRank()) return failure();

    // Lower to MHLO constant if statically shaped.
    if (result_ty.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<mhlo::ConstOp>(
          op, DenseElementsAttr::get(result_ty, op.value()));
      return success();
    }

    // Lower to broadcasted constant.
    ConstantLikeOp::Adaptor transformed(operands);
    auto loc = op.getLoc();
    Type extent_tensor_type = shape::getExtentTensorType(op.getContext());
    Value constant = rewriter.create<mhlo::ConstOp>(loc, op.value());
    Value uncasted_shape = rewriter.create<shape::ShapeOfOp>(
        loc, extent_tensor_type, transformed.operand());
    Type shape_ty =
        RankedTensorType::get({result_ty.getRank()}, rewriter.getIndexType());
    Value shape = rewriter.create<TensorCastOp>(loc, shape_ty, uncasted_shape);
    rewriter.replaceOpWithNewOp<mhlo::DynamicBroadcastInDimOp>(
        op, result_ty, constant, shape, rewriter.getI64TensorAttr({}));
    return success();
  }
};

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
        hlo::ComputeBinaryElementwiseBroadcastingResultExtents(
            loc, lhs, rhs, rewriter, /*unsafe_as_extent_tensor=*/true);

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

// Converts a broadcasting binary operation with a scalar operand and an
// unranked operand to a ranked broadcasting operation by dynamically reshaping
// the unranked operand to a 1D tensor. This will always be safe because
// broadcasting from a scalar to another shape always works.
template <typename ChloOpTy, typename HloOpTy>
struct ConvertUnrankedScalarDynamicBroadcastBinaryOp
    : public OpRewritePattern<ChloOpTy> {
  using OpRewritePattern<ChloOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(ChloOpTy op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value lhs = op.lhs();
    Value rhs = op.rhs();

    auto lhs_ranked_type = lhs.getType().dyn_cast<RankedTensorType>();
    auto lhs_unranked_type = lhs.getType().dyn_cast<UnrankedTensorType>();

    auto rhs_ranked_type = rhs.getType().dyn_cast<RankedTensorType>();
    auto rhs_unranked_type = rhs.getType().dyn_cast<UnrankedTensorType>();

    bool lhs_is_scalar = lhs_ranked_type &&
                         lhs_ranked_type.getShape().empty() &&
                         rhs_unranked_type;
    bool rhs_is_scalar = rhs_ranked_type &&
                         rhs_ranked_type.getShape().empty() &&
                         lhs_unranked_type;

    // Only support the case where exactly one operand is scalar and the other
    // is unranked. Other patterns in this file will create more efficient
    // lowerings for cases where both ranks are known or will handle the more
    // generic case of both inputs being unranked.
    if (!(lhs_is_scalar ^ rhs_is_scalar)) return failure();

    auto result_type = op.getResult().getType().template dyn_cast<TensorType>();

    // Reshape the non-scalar value into a dynamically sized, rank-1 tensor
    Value shape =
        rewriter.create<shape::ShapeOfOp>(loc, lhs_is_scalar ? rhs : lhs);
    Value num_elements = rewriter.create<shape::NumElementsOp>(loc, shape);
    Value size_tensor =
        rewriter.create<TensorFromElementsOp>(loc, num_elements);
    Value reshaped = rewriter.create<mhlo::DynamicReshapeOp>(
        loc, RankedTensorType::get({-1}, result_type.getElementType()),
        lhs_is_scalar ? rhs : lhs, size_tensor);

    // Create a new ranked Chlo op that will be further lowered by other
    // patterns into Mhlo.
    SmallVector<Value, 2> operands{lhs_is_scalar ? lhs : reshaped,
                                   rhs_is_scalar ? rhs : reshaped};
    Value computed = rewriter.create<ChloOpTy>(
        loc, SmallVector<Type, 1>{reshaped.getType()}, operands, op.getAttrs());

    // Reshape the result back into an unranked tensor.
    rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(op, result_type,
                                                        computed, shape);

    return success();
  }
};

// Handles lowering of the following pattern to patterns that will be further
// matched by other patterns until they result in LHLO:
//   %result = "chlo.op"(%lhs, %rhs) : (<*xTy>, <*xTy>) -> <*xTy>
//
// The sequence of specializations this handles is:
//   - Either operand being scalar
//   - Operands having equal shapes
//   - The resulting value being any of ranks [2,6]
template <typename ChloOpTy, typename HloOpTy, typename Adaptor>
struct ConvertUnrankedDynamicBroadcastBinaryOp
    : public OpRewritePattern<ChloOpTy> {
  using OpRewritePattern<ChloOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(ChloOpTy op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value lhs = op.lhs();
    Value rhs = op.rhs();
    auto lhs_type = lhs.getType().dyn_cast<UnrankedTensorType>();
    auto rhs_type = rhs.getType().dyn_cast<UnrankedTensorType>();
    auto result_type = op.getResult().getType().template dyn_cast<TensorType>();

    // Only support unranked operands. If either operand is ranked, another
    // pattern will handle the lowering.
    if (!lhs_type || !rhs_type) return failure();

    // If lhs is scalar
    auto if_op = rewriter.create<scf::IfOp>(
        loc, result_type, IsScalarTensor(rewriter, op, lhs), true);
    OpBuilder if_lhs_scalar_builder = if_op.getThenBodyBuilder();
    Value reshaped_lhs = if_lhs_scalar_builder.create<TensorCastOp>(
        loc, RankedTensorType::get({}, lhs_type.getElementType()), lhs);
    Value if_lhs_scalar_result = if_lhs_scalar_builder.create<ChloOpTy>(
        loc, ArrayRef<Type>{result_type}, ArrayRef<Value>{reshaped_lhs, rhs},
        op.getAttrs());
    if_lhs_scalar_builder.create<scf::YieldOp>(loc, if_lhs_scalar_result);

    // If lhs is NOT scalar
    //
    // See if rhs is scalar
    OpBuilder else_lhs_scalar_builder = if_op.getElseBodyBuilder();
    auto if_rhs_scalar_op = else_lhs_scalar_builder.create<scf::IfOp>(
        loc, result_type, IsScalarTensor(else_lhs_scalar_builder, op, rhs),
        true);
    else_lhs_scalar_builder.create<scf::YieldOp>(loc,
                                                 if_rhs_scalar_op.getResult(0));
    OpBuilder if_rhs_scalar_builder = if_rhs_scalar_op.getThenBodyBuilder();
    Value reshaped_rhs = if_rhs_scalar_builder.create<TensorCastOp>(
        loc, RankedTensorType::get({}, lhs_type.getElementType()), rhs);
    Value if_rhs_scalar_result = if_rhs_scalar_builder.create<ChloOpTy>(
        loc, ArrayRef<Type>{result_type}, ArrayRef<Value>{lhs, reshaped_rhs},
        op.getAttrs());
    if_rhs_scalar_builder.create<scf::YieldOp>(loc, if_rhs_scalar_result);

    // If NEITHER shape is scalar
    //
    // See if shapes are equal.
    OpBuilder else_no_scalars_builder = if_rhs_scalar_op.getElseBodyBuilder();
    Value shape_of_lhs =
        else_no_scalars_builder.create<shape::ShapeOfOp>(loc, lhs);
    Value shape_of_rhs =
        else_no_scalars_builder.create<shape::ShapeOfOp>(loc, rhs);
    Value equal_shapes = else_no_scalars_builder.create<shape::ShapeEqOp>(
        loc, shape_of_lhs, shape_of_rhs);

    auto if_eq_shapes_op = else_no_scalars_builder.create<scf::IfOp>(
        loc, result_type, equal_shapes, true);
    else_no_scalars_builder.create<scf::YieldOp>(loc,
                                                 if_eq_shapes_op.getResult(0));

    OpBuilder if_eq_shapes_builder = if_eq_shapes_op.getThenBodyBuilder();
    Value non_broadcast_op =
        Adaptor::CreateOp(op, result_type, lhs, rhs, if_eq_shapes_builder);
    if_eq_shapes_builder.create<scf::YieldOp>(loc, non_broadcast_op);

    // If shapes are not scalar, nor equal
    //
    // See if values are of a rank that we support.
    OpBuilder if_neq_shapes_builder = if_eq_shapes_op.getElseBodyBuilder();
    if_neq_shapes_builder.create<scf::YieldOp>(
        loc, HandleBroadcastAndOp(if_neq_shapes_builder, op, lhs, rhs));

    rewriter.replaceOp(op, {if_op.getResult(0)});
    return success();
  }

 private:
  // Returns the dyanamic result of checking the given value is a scalar
  // tensor.
  Value IsScalarTensor(OpBuilder &rewriter, ChloOpTy op, Value tensor) const {
    auto loc = op.getLoc();

    Value shape_of_tensor = rewriter.create<shape::ShapeOfOp>(loc, tensor);
    Value rank_tensor = rewriter.create<shape::RankOp>(
        loc, rewriter.getIndexType(), shape_of_tensor);
    return rewriter.create<CmpIOp>(loc, rewriter.getI1Type(), CmpIPredicate::eq,
                                   rank_tensor,
                                   rewriter.create<ConstantIndexOp>(loc, 0));
  }

  // Create the if statement and code for a broadcasting op with a result of a
  // given rank.
  scf::IfOp createRankSpecializedBroadcastAndOp(OpBuilder &builder, ChloOpTy op,
                                                Value lhs, Value rhs,
                                                Value actual_rank,
                                                int targeted_rank) const {
    auto loc = op.getLoc();

    // Create the if block to place the current specialized logic in.
    Value greater_rank_is_n = builder.create<CmpIOp>(
        loc, CmpIPredicate::eq, actual_rank,
        builder.create<ConstantIndexOp>(loc, targeted_rank));
    auto if_op =
        builder.create<scf::IfOp>(loc, lhs.getType(), greater_rank_is_n, true);
    OpBuilder if_builder = if_op.getThenBodyBuilder();

    // Handle shape broadcasting and inferrence.
    Value lhs_shape = if_builder.create<shape::ShapeOfOp>(loc, lhs);
    Value rhs_shape = if_builder.create<shape::ShapeOfOp>(loc, rhs);
    SmallVector<int64_t, 6> ranked_shape(targeted_rank, 1);
    auto unknown_rank_extent_tensor_type = RankedTensorType::get(
        {RankedTensorType::kDynamicSize}, builder.getIndexType());
    auto known_rank_extent_tensor_type =
        RankedTensorType::get({targeted_rank}, builder.getIndexType());
    auto reshaped_type = RankedTensorType::get(
        llvm::SmallVector<int64_t, 6>(targeted_rank,
                                      RankedTensorType::kDynamicSize),
        lhs.getType().template dyn_cast<TensorType>().getElementType());
    Value ranked_shape_val = if_builder.create<shape::ConstShapeOp>(
        loc, known_rank_extent_tensor_type,
        mlir::DenseIntElementsAttr::get(known_rank_extent_tensor_type,
                                        ranked_shape));
    Value extended_lhs = if_builder.create<shape::BroadcastOp>(
        loc, unknown_rank_extent_tensor_type, lhs_shape, ranked_shape_val,
        nullptr);
    Value extended_lhs_casted = if_builder.create<TensorCastOp>(
        loc, known_rank_extent_tensor_type, extended_lhs);
    Value extended_rhs = if_builder.create<shape::BroadcastOp>(
        loc, unknown_rank_extent_tensor_type, rhs_shape, ranked_shape_val,
        nullptr);
    Value extended_rhs_casted = if_builder.create<TensorCastOp>(
        loc, known_rank_extent_tensor_type, extended_rhs);

    // 1. Reshape operands to the given rank (with the same number of elements)
    // 2. Compute the ranked-broadcasted ChloOp (which will assert that the ops
    //    can be broadcasted and do the actual broadcasting)
    // 3. Type erase the output back to unranked
    Value reshaped_lhs = if_builder.create<mhlo::DynamicReshapeOp>(
        loc, reshaped_type, lhs, extended_lhs_casted);
    Value reshaped_rhs = if_builder.create<mhlo::DynamicReshapeOp>(
        loc, reshaped_type, rhs, extended_rhs_casted);
    Value result = if_builder.create<ChloOpTy>(
        loc, ArrayRef<Type>{reshaped_type},
        ArrayRef<Value>{reshaped_lhs, reshaped_rhs}, op.getAttrs());
    Value reshaped_result = if_builder.create<TensorCastOp>(
        loc, UnrankedTensorType::get(reshaped_type.getElementType()), result);
    if_builder.create<scf::YieldOp>(loc, reshaped_result);

    // Return the if_op, so the result can be used and the else block can be
    // used for the next rank specialized step.
    return if_op;
  }

  // Iterates over the desired ranks to be specialized and generates the code
  // snippet for each case.
  Value HandleBroadcastAndOp(OpBuilder &rewriter, ChloOpTy op, Value lhs,
                             Value rhs) const {
    constexpr int max_rank_specialization = 7;
    auto loc = op.getLoc();

    // Find the larger rank of the 2 operands.
    auto extent_tensor_type = RankedTensorType::get({ShapedType::kDynamicSize},
                                                    rewriter.getIndexType());
    Value lhs_shape =
        rewriter.create<shape::ShapeOfOp>(loc, extent_tensor_type, lhs);
    Value rhs_shape =
        rewriter.create<shape::ShapeOfOp>(loc, extent_tensor_type, rhs);
    Value lhs_rank =
        rewriter.create<RankOp>(loc, rewriter.getIndexType(), lhs_shape);
    Value rhs_rank =
        rewriter.create<RankOp>(loc, rewriter.getIndexType(), rhs_shape);
    Value greater_rank_lhs =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::sgt, lhs_rank, rhs_rank);
    Value greater_rank =
        rewriter.create<SelectOp>(loc, greater_rank_lhs, lhs_rank, rhs_rank);

    // Generate a list of nested if/else statements to handle rank
    // specializations from 2-6.
    scf::IfOp if_op = createRankSpecializedBroadcastAndOp(rewriter, op, lhs,
                                                          rhs, greater_rank, 2);

    // Put each subsequent rank specialization inside the else statement of the
    // previous one.
    OpBuilder else_builder = if_op.getElseBodyBuilder();
    for (int i = 3; i < max_rank_specialization; i++) {
      auto inner_if = createRankSpecializedBroadcastAndOp(else_builder, op, lhs,
                                                          rhs, greater_rank, i);

      else_builder.create<scf::YieldOp>(loc, inner_if.getResult(0));
      else_builder = inner_if.getElseBodyBuilder();
    }

    // Fire an assertion if none of the rank specializations applied (one of the
    // ranks was greater than 6).
    else_builder.create<AssertOp>(
        loc, else_builder.create<ConstantIntOp>(loc, 0, 1),
        "Input for dynamic binary op lowering was of a rank greater than 6");
    else_builder.create<scf::YieldOp>(loc, lhs);

    // Return the result of the outermost if statement.
    return if_op.getResult(0);
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
  patterns->insert<
      ConvertUnrankedScalarDynamicBroadcastBinaryOp<ChloOpTy, HloOpTy>,
      ConvertUnrankedDynamicBroadcastBinaryOp<ChloOpTy, HloOpTy, Adaptor>>(
      context);
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

#include "generated_chlo_legalize_to_hlo.inc"
}  // namespace

void PopulateLegalizeChloToHloPatterns(MLIRContext *context,
                                       OwningRewritePatternList *patterns) {
  populateWithGenerated(context, *patterns);

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

  // Other patterns.
  patterns->insert<ConvertConstantLikeOp>(context);
}

}  // namespace chlo
}  // namespace mlir
