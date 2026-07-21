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
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {
namespace {

// mhlo encodes ND indice arguments as a variadiac of scalars. Pack them
// into a single tensor for use in TFL.
Value PackScalarIndices(mlir::ValueRange indices, OpBuilder& b) {
  auto e_type =
      llvm::cast<ShapedType>(indices.front().getType()).getElementType();
  const int64_t num_indices = indices.size();
  auto packed_indices_type = RankedTensorType::get({num_indices}, e_type);

  auto values_count_attr = b.getI32IntegerAttr(num_indices);
  auto pack_axis_attr = b.getI32IntegerAttr(0);

  return TFL::PackOp::create(b, indices.back().getLoc(), packed_indices_type,
                             indices, values_count_attr, pack_axis_attr);
}

//===----------------------------------------------------------------------===//
// mhlo.slice
//===----------------------------------------------------------------------===//

// Cast the value to i32.
Value BuildTFLCastOp(OpBuilder& b, Value value) {
  auto type = mlir::cast<ShapedType>(value.getType());
  if (type.getElementType().isInteger(32)) {
    return value;
  }
  return TFL::CastOp::create(
      b, value.getLoc(), RankedTensorType::get(type.getShape(), b.getI32Type()),
      value);
}

class LegalizeSliceOp : public OpConversionPattern<mhlo::SliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::SliceOp slice_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto input = adaptor.getOperand();

    auto convert_to_i32 = [&](DenseIntElementsAttr attr) {
      return attr.mapValues(rewriter.getI32Type(), [](const APInt& val) {
        return val.sextOrTrunc(32);
      });
    };

    auto begin_const =
        arith::ConstantOp::create(rewriter, slice_op.getLoc(),
                                  convert_to_i32(slice_op.getStartIndices()));
    auto limit_const =
        arith::ConstantOp::create(rewriter, slice_op.getLoc(),
                                  convert_to_i32(slice_op.getLimitIndices()));
    auto strides_const = arith::ConstantOp::create(
        rewriter, slice_op.getLoc(), convert_to_i32(slice_op.getStrides()));

    auto zero = rewriter.getIntegerAttr(rewriter.getI32Type(), 0);
    auto no_offset = rewriter.getBoolAttr(false);

    rewriter.replaceOpWithNewOp<TFL::StridedSliceOp>(
        slice_op, slice_op.getType(), input, begin_const, limit_const,
        strides_const, zero, zero, zero, zero, zero, no_offset);

    return success();
  }
};

class CollapseStridedSliceRank : public OpRewritePattern<TFL::StridedSliceOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::StridedSliceOp op,
                                PatternRewriter& rewriter) const final {
    auto to_i32 = [](ArrayRef<int64_t> values) {
      return llvm::map_to_vector(
          values, [](int64_t v) { return static_cast<int32_t>(v); });
    };

    auto input = op.getInput();
    auto input_type = mlir::cast<ShapedType>(input.getType());
    if (!input_type.hasStaticShape() || input_type.getRank() <= 5) {
      return rewriter.notifyMatchFailure(op, "Rank <= 5 or dynamic shape.");
    }

    // Only handle simple strided slices (no masks).
    if (op.getBeginMask() != 0 || op.getEndMask() != 0 ||
        op.getEllipsisMask() != 0 || op.getNewAxisMask() != 0 ||
        op.getShrinkAxisMask() != 0) {
      return rewriter.notifyMatchFailure(op, "Has masks.");
    }

    DenseIntElementsAttr start_indices_attr, limit_indices_attr, strides_attr;
    if (!matchPattern(op.getBegin(), m_Constant(&start_indices_attr)) ||
        !matchPattern(op.getEnd(), m_Constant(&limit_indices_attr)) ||
        !matchPattern(op.getStrides(), m_Constant(&strides_attr))) {
      return rewriter.notifyMatchFailure(op, "Indices are not constant.");
    }

    SmallVector<int64_t> start_indices(
        llvm::map_range(start_indices_attr.getValues<APInt>(),
                        [](const APInt& val) { return val.getSExtValue(); }));
    SmallVector<int64_t> limit_indices(
        llvm::map_range(limit_indices_attr.getValues<APInt>(),
                        [](const APInt& val) { return val.getSExtValue(); }));
    SmallVector<int64_t> strides(
        llvm::map_range(strides_attr.getValues<APInt>(),
                        [](const APInt& val) { return val.getSExtValue(); }));
    SmallVector<int64_t> input_shape(input_type.getShape().begin(),
                                     input_type.getShape().end());

    // Collapse dimensions if rank > 5.
    while (input_shape.size() > 5) {
      bool merged = false;
      for (int i = 0; i < static_cast<int>(input_shape.size()) - 1; ++i) {
        // This merging condition ensures the inner dimension is fully covered
        // by the slice with stride 1, making it safe to collapse.
        if (strides[i] == 1 && strides[i + 1] == 1 &&
            start_indices[i + 1] == 0 &&
            limit_indices[i + 1] == input_shape[i + 1]) {
          // Merge i and i+1
          start_indices[i] = start_indices[i] * input_shape[i + 1];
          limit_indices[i] = limit_indices[i] * input_shape[i + 1];
          input_shape[i] = input_shape[i] * input_shape[i + 1];

          start_indices.erase(start_indices.begin() + i + 1);
          limit_indices.erase(limit_indices.begin() + i + 1);
          input_shape.erase(input_shape.begin() + i + 1);
          strides.erase(strides.begin() + i + 1);
          merged = true;
          break;
        }
      }
      if (!merged) break;
    }

    if (input_shape.size() > 5) {
      return rewriter.notifyMatchFailure(
          op, "Input rank > 5 and could not be collapsed.");
    }

    auto collapsed_type =
        RankedTensorType::get(input_shape, input_type.getElementType());
    Value collapsed_input = TFL::ReshapeOp::create(
        rewriter, op.getLoc(), collapsed_type, input,
        arith::ConstantOp::create(
            rewriter, op.getLoc(),
            rewriter.getI32TensorAttr(to_i32(input_shape))));

    auto begin_const = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getI32TensorAttr(to_i32(start_indices)));
    auto limit_const = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getI32TensorAttr(to_i32(limit_indices)));
    auto strides_const = arith::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI32TensorAttr(to_i32(strides)));

    auto zero = rewriter.getIntegerAttr(rewriter.getI32Type(), 0);
    auto no_offset = rewriter.getBoolAttr(false);

    SmallVector<int64_t> intermediate_result_shape;
    for (int i = 0; i < static_cast<int>(input_shape.size()); ++i) {
      intermediate_result_shape.push_back(
          (limit_indices[i] - start_indices[i] + strides[i] - 1) / strides[i]);
    }
    auto intermediate_type = RankedTensorType::get(intermediate_result_shape,
                                                   input_type.getElementType());

    Value slice_res = TFL::StridedSliceOp::create(
        rewriter, op.getLoc(), intermediate_type, collapsed_input, begin_const,
        limit_const, strides_const, zero, zero, zero, zero, zero, no_offset);

    auto result_type = mlir::cast<ShapedType>(op.getType());
    auto reshape_op = TFL::ReshapeOp::create(
        rewriter, op.getLoc(), op.getType(), slice_res,
        arith::ConstantOp::create(
            rewriter, op.getLoc(),
            rewriter.getI32TensorAttr(to_i32(result_type.getShape()))));
    rewriter.replaceOp(op, reshape_op);

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
    auto casted_start_ind_opr = mhlo::ConvertOp::create(
        rewriter, start_ind_opr.getLoc(), start_ind_opr, new_start_e_type);
    casted_start_inds.push_back(casted_start_ind_opr.getResult());
  }

  auto new_op = mhlo::DynamicSliceOp::create(
      rewriter, op.getLoc(), op.getType(), op.getOperand(), casted_start_inds,
      op.getSliceSizes());
  rewriter.replaceOp(op, new_op);

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

  Value clamp_left_cst = arith::ConstantOp::create(
      rewriter, op->getLoc(), rewriter.getZeroAttr(start_type));

  llvm::SmallVector<Value> new_start_indices;

  for (auto [dim_size, start_ind_opr, stride_size] :
       llvm::zip(input_type.getShape(), op.getStartIndices(),
                 mlir::cast<DenseIntElementsAttr>(op.getSliceSizes())
                     .getValues<int64_t>())) {
    const int64_t clamp_right_val = dim_size - stride_size;
    auto clamp_right_cst = arith::ConstantOp::create(
        rewriter, op->getLoc(),
        DenseElementsAttr::get(start_type, rewriter.getIntegerAttr(
                                               start_e_type, clamp_right_val)));

    Value new_start_ind = TFL::MaximumOp::create(
        rewriter, op->getLoc(), start_type, clamp_left_cst, start_ind_opr);
    new_start_ind = TFL::MinimumOp::create(rewriter, op->getLoc(), start_type,
                                           clamp_right_cst, new_start_ind);

    new_start_indices.push_back(new_start_ind);
  }

  //
  // build tfl
  //=-----

  auto packed_indices = PackScalarIndices(new_start_indices, rewriter);

  auto slice_sizes_cst =
      arith::ConstantOp::create(rewriter, op->getLoc(), op.getSliceSizes());

  auto slice_op =
      TFL::SliceOp::create(rewriter, op.getLoc(), op.getType(), op.getOperand(),
                           packed_indices, slice_sizes_cst);
  rewriter.replaceOp(op, slice_op);

  return success();
}

//===----------------------------------------------------------------------===//
// mhlo.real_dynamic_slice
//===----------------------------------------------------------------------===//

class LegalizeRealDynamicSliceOp
    : public OpConversionPattern<mhlo::RealDynamicSliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::RealDynamicSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};

LogicalResult LegalizeRealDynamicSliceOp::matchAndRewrite(
    mhlo::RealDynamicSliceOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto start_indices_type =
      mlir::cast<RankedTensorType>(op.getStartIndices().getType());
  auto end_indices_type =
      mlir::cast<RankedTensorType>(op.getLimitIndices().getType());

  if (start_indices_type.getNumDynamicDims() != 0 ||
      end_indices_type.getNumDynamicDims() != 0) {
    return rewriter.notifyMatchFailure(
        op,
        "Start indices and limit indices must not have dynamic dimensions.");
  }

  auto zero = rewriter.getIntegerAttr(rewriter.getI32Type(), 0);
  auto no_offset = rewriter.getBoolAttr(false);

  auto ss_op = TFL::StridedSliceOp::create(
      rewriter, op.getLoc(), op.getType(), op.getOperand(),
      BuildTFLCastOp(rewriter, op.getStartIndices()),
      BuildTFLCastOp(rewriter, op.getLimitIndices()),
      BuildTFLCastOp(rewriter, op.getStrides()), zero, zero, zero, zero, zero,
      no_offset);
  rewriter.replaceOp(op, ss_op);
  return success();
};

//===----------------------------------------------------------------------===//
// mhlo.dynamic_update_slice
//===----------------------------------------------------------------------===//

class LegalizeDynamicUpdateSliceOp
    : public OpConversionPattern<mhlo::DynamicUpdateSliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DynamicUpdateSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};

LogicalResult LegalizeDynamicUpdateSliceOp::matchAndRewrite(
    mhlo::DynamicUpdateSliceOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto packed_indices = PackScalarIndices(op.getStartIndices(), rewriter);
  auto dus_op = TFL::DynamicUpdateSliceOp::create(
      rewriter, op.getLoc(), op.getType(), op.getOperand(), op.getUpdate(),
      packed_indices);
  rewriter.replaceOp(op, dus_op);
  return success();
};

}  // namespace

void PopulateLegalizeSlicePatterns(MLIRContext* ctx,
                                   RewritePatternSet& patterns,
                                   ConversionTarget& target) {
  patterns.add<LegalizeSliceOp, LegalizeDynamicSliceOp,
               LegalizeDynamicUpdateSliceOp, LegalizeRealDynamicSliceOp,
               CollapseStridedSliceRank>(ctx);

  target.addIllegalOp<mhlo::SliceOp, mhlo::DynamicUpdateSliceOp,
                      mhlo::RealDynamicSliceOp>();
  target.addDynamicallyLegalOp<mhlo::DynamicSliceOp>(IsDynamicSliceLegal);
  // This dynamic legality check is crucial. It correctly marks
  // `tfl.strided_slice` operations with rank > 5 as illegal, allowing the
  // `CollapseStridedSliceRank` pattern to kick in and perform the necessary
  // reshaping.
  target.addDynamicallyLegalOp<TFL::StridedSliceOp>([](TFL::StridedSliceOp op) {
    auto input_type = mlir::cast<ShapedType>(op.getInput().getType());
    return input_type.getRank() <= 5;
  });
}

void PopulatePrepareSlicePatterns(MLIRContext* ctx,
                                  RewritePatternSet& patterns) {
  patterns.add<CastSliceIndicesToSignless>(ctx);
}

}  // namespace mlir::odml
