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

// Enable the use of M_* math constants.
// NOTE: this must be first in the file to ensure that if cmath is transitively
// included by any other header it has the define set on first processing.
// https://docs.microsoft.com/en-us/cpp/c-runtime-library/math-constants
#define _USE_MATH_DEFINES
#include <cmath>
#include <numeric>
#include <vector>

#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_chlo_to_hlo_op.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir-hlo/utils/broadcast_utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
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
    Value shape =
        rewriter.create<tensor::CastOp>(loc, shape_ty, uncasted_shape);
    rewriter.replaceOpWithNewOp<mhlo::DynamicBroadcastInDimOp>(
        op, result_ty, constant, shape, rewriter.getI64TensorAttr({}));
    return success();
  }
};

Value MaterializePolynomialApproximation(
    ConversionPatternRewriter &rewriter, Location loc, Value x,
    const std::vector<float> &coefficients) {
  Value poly = chlo::getConstantLike(rewriter, loc, 0.0, x);
  for (float c : coefficients) {
    poly = rewriter.create<mhlo::MulOp>(loc, x.getType(), poly, x);
    poly = rewriter.create<mhlo::AddOp>(
        loc, x.getType(), poly, chlo::getConstantLike(rewriter, loc, c, x));
  }
  return poly;
}

Value MaterializeErfApproximationF32(ConversionPatternRewriter &rewriter,
                                     Location loc, Value operand) {
  const std::vector<float> kAlpha{
      -2.72614225801306e-10f, 2.77068142495902e-08f,  -2.10102402082508e-06f,
      -5.69250639462346e-05f, -7.34990630326855e-04f, -2.95459980854025e-03f,
      -1.60960333262415e-02f,
  };
  const std::vector<float> kBeta{
      -1.45660718464996e-05f, -2.13374055278905e-04f, -1.68282697438203e-03f,
      -7.37332916720468e-03f, -1.42647390514189e-02f,
  };

  // Clamp argument between -4 and 4.
  Value lb = chlo::getConstantLike(rewriter, loc, -4.0, operand);
  Value ub = chlo::getConstantLike(rewriter, loc, 4.0, operand);
  Value x =
      rewriter.create<mhlo::ClampOp>(loc, operand.getType(), lb, operand, ub);
  Value x_sq = rewriter.create<mhlo::MulOp>(loc, x, x);

  // Materialize polynomial approximation for x in [-4, 4].
  Value alpha_poly =
      MaterializePolynomialApproximation(rewriter, loc, x_sq, kAlpha);
  Value beta_poly =
      MaterializePolynomialApproximation(rewriter, loc, x_sq, kBeta);
  Value mul_x_alpha_poly = rewriter.create<mhlo::MulOp>(loc, x, alpha_poly);
  return rewriter.create<mhlo::DivOp>(loc, mul_x_alpha_poly, beta_poly);
}

struct ConvertErfOp : public OpConversionPattern<ErfOp> {
  using OpConversionPattern<ErfOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ErfOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Type ty = getElementTypeOrSelf(op.getType());

    // For now, we support only f32.
    if (!ty.isF32()) return failure();

    ErfOp::Adaptor transformed(operands);
    rewriter.replaceOp(op, MaterializeErfApproximationF32(
                               rewriter, op.getLoc(), transformed.operand()));
    return success();
  }
};

// Converts binary ops that statically are determined to not broadcast directly
// to the corresponding mhlo non-broadcasting op.
template <typename ChloOpTy, typename HloOpTy, typename Adaptor>
struct ConvertTrivialNonBroadcastBinaryOp
    : public OpConversionPattern<ChloOpTy> {
  using OpConversionPattern<ChloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ChloOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Only rewrite for statically determinable non-broadcasting cases.
    typename ChloOpTy::Adaptor transformed(operands);
    auto lhs_type =
        transformed.lhs().getType().template dyn_cast<RankedTensorType>();
    auto rhs_type =
        transformed.rhs().getType().template dyn_cast<RankedTensorType>();
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

    rewriter.replaceOp(
        op, {Adaptor::CreateOp(op, op.getResult().getType(), operands[0],
                               operands[1], rewriter)});
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
    : public OpConversionPattern<ChloOpTy> {
  using OpConversionPattern<ChloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ChloOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Only support ranked operands.
    typename ChloOpTy::Adaptor transformed(operands);
    Value lhs = transformed.lhs();
    Value rhs = transformed.rhs();
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

#include "generated_chlo_legalize_to_hlo.inc"
}  // namespace

void PopulateLegalizeChloToHloPatterns(MLIRContext *context,
                                       OwningRewritePatternList *patterns) {
  populateWithGenerated(context, *patterns);

  // Instantiate conversion templates for conforming binary elementwise ops
  // that do not have different dtypes between operands and results and do
  // not have special attributes that need to be preserved.
  PopulateForBroadcastingBinaryOp<ConvertTrivialNonBroadcastBinaryOp>(
      context, patterns, 10);
  PopulateForBroadcastingBinaryOp<ConvertRankedDynamicBroadcastBinaryOp>(
      context, patterns, 5);

  // Other patterns.
  patterns->insert<ConvertConstantLikeOp, ConvertErfOp>(context);
}

}  // namespace chlo
}  // namespace mlir
