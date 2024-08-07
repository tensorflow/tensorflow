/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/slice.h"

#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/op_util_common.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {
namespace {

//===----------------------------------------------------------------------===//
// mhlo.slice
//===----------------------------------------------------------------------===//

// Cast the value to i32.
Value BuildTFLCastOp(OpBuilder& b, Value value) {
  return b.create<TFL::CastOp>(
      value.getLoc(),
      RankedTensorType::get(llvm::cast<ShapedType>(value.getType()).getShape(),
                            b.getI32Type()),
      value);
}

class LegalizeSliceOp : public OpConversionPattern<mhlo::SliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::SliceOp slice_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto begin = rewriter.create<arith::ConstantOp>(slice_op.getLoc(),
                                                    slice_op.getStartIndices());
    auto end = rewriter.create<arith::ConstantOp>(slice_op.getLoc(),
                                                  slice_op.getLimitIndices());
    auto strides = rewriter.create<arith::ConstantOp>(slice_op.getLoc(),
                                                      slice_op.getStrides());
    auto zero = rewriter.getIntegerAttr(rewriter.getI32Type(), 0);
    auto no_offset = rewriter.getBoolAttr(false);

    rewriter.replaceOpWithNewOp<TFL::StridedSliceOp>(
        slice_op, slice_op.getType(), slice_op.getOperand(),
        BuildTFLCastOp(rewriter, begin), BuildTFLCastOp(rewriter, end),
        BuildTFLCastOp(rewriter, strides), zero, zero, zero, zero, zero,
        no_offset);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// mhlo.dynamic_slice
//===----------------------------------------------------------------------===//

class CastSliceIndicesToSignless
    : public OpRewritePattern<mhlo::DynamicSliceOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DynamicSliceOp op,
                                PatternRewriter& rewriter) const final;
};

LogicalResult CastSliceIndicesToSignless::matchAndRewrite(
    mhlo::DynamicSliceOp op, PatternRewriter& rewriter) const {
  // All start inds have the same element type.
  auto start_type =
      llvm::cast<ShapedType>(op.getStartIndices().front().getType());
  auto start_e_type = start_type.getElementType();

  if (start_e_type.isSignlessIntOrFloat()) {
    return rewriter.notifyMatchFailure(op, "Already signless.");
  }
  auto new_start_e_type =
      rewriter.getIntegerType(start_e_type.getIntOrFloatBitWidth());

  llvm::SmallVector<Value> casted_start_inds;
  for (auto start_ind_opr : op.getStartIndices()) {
    auto casted_start_ind_opr = rewriter.create<mhlo::ConvertOp>(
        start_ind_opr.getLoc(), start_ind_opr, new_start_e_type);
    casted_start_inds.push_back(casted_start_ind_opr.getResult());
  }

  rewriter.replaceOpWithNewOp<mhlo::DynamicSliceOp>(
      op, op.getOperand(), casted_start_inds, op.getSliceSizes());

  return success();
}

bool IsDynamicSliceLegal(mhlo::DynamicSliceOp op) {
  return !llvm::cast<ShapedType>(op.getOperand().getType()).hasStaticShape();
}

class LegalizeDynamicSliceOp
    : public OpConversionPattern<mhlo::DynamicSliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DynamicSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};

LogicalResult LegalizeDynamicSliceOp::matchAndRewrite(
    mhlo::DynamicSliceOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto start_type =
      llvm::cast<ShapedType>(op.getStartIndices().front().getType());
  auto start_e_type = start_type.getElementType();
  if (!start_e_type.isSignlessIntOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "Must be signless integer for start indices.");
  }

  auto input_type = llvm::cast<ShapedType>(op.getOperand().getType());
  if (!input_type.hasStaticShape()) {
    return rewriter.notifyMatchFailure(op, "Input must be statically shaped.");
  }

  //
  // clamp start indices between zero and shape(operand) - slice_sizes
  //=-----

  Value clamp_left_cst = rewriter.create<arith::ConstantOp>(
      op->getLoc(), rewriter.getZeroAttr(start_type));

  llvm::SmallVector<Value> new_start_indices;
  const auto stride_sizes = UnrollI64Splat(op.getSliceSizes());

  for (auto [dim_size, start_ind_opr, stride_size] :
       llvm::zip(input_type.getShape(), op.getStartIndices(), stride_sizes)) {
    const int64_t clamp_right_val = dim_size - stride_size;
    auto clamp_right_cst = rewriter.create<arith::ConstantOp>(
        op->getLoc(),
        DenseElementsAttr::get(start_type, rewriter.getIntegerAttr(
                                               start_e_type, clamp_right_val)));

    Value new_start_ind = rewriter.create<TFL::MaximumOp>(
        op->getLoc(), start_type, clamp_left_cst, start_ind_opr);
    new_start_ind = rewriter.create<TFL::MinimumOp>(
        op->getLoc(), start_type, clamp_right_cst, new_start_ind);

    new_start_indices.push_back(new_start_ind);
  }

  //
  // pack variadic scalar start indices into one tensor
  //=-----

  const int64_t packed_start_indices_size = new_start_indices.size();
  auto packed_start_indices_type =
      RankedTensorType::get({packed_start_indices_size}, start_e_type);

  auto values_count_attr =
      rewriter.getI32IntegerAttr(packed_start_indices_size);
  auto pack_axis_attr = rewriter.getI32IntegerAttr(0);

  auto packed_start_inds = rewriter.create<TFL::PackOp>(
      op->getLoc(), packed_start_indices_type, new_start_indices,
      values_count_attr, pack_axis_attr);

  //
  // build tfl
  //=-----

  auto slice_sizes_cst =
      rewriter.create<arith::ConstantOp>(op->getLoc(), op.getSliceSizes());

  rewriter.replaceOpWithNewOp<TFL::SliceOp>(op, op.getType(), op.getOperand(),
                                            packed_start_inds, slice_sizes_cst);

  return success();
}

}  // namespace

void PopulateLegalizeSlicePatterns(MLIRContext* ctx,
                                   RewritePatternSet& patterns,
                                   ConversionTarget& target) {
  patterns.add<LegalizeSliceOp, LegalizeDynamicSliceOp>(ctx);

  target.addIllegalOp<mhlo::SliceOp>();
  target.addDynamicallyLegalOp<mhlo::DynamicSliceOp>(IsDynamicSliceLegal);
}

void PopulatePrepareSlicePatterns(MLIRContext* ctx,
                                  RewritePatternSet& patterns) {
  patterns.add<CastSliceIndicesToSignless>(ctx);
}

}  // namespace mlir::odml
