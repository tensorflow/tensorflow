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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/conv.h"

#include <cstdint>
#include <tuple>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/conv_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {

namespace {

using ::llvm::ArrayRef;

//===----------------------------------------------------------------------===//
// support/legality checking
//===----------------------------------------------------------------------===//

bool IsShapeFullyStatic(ArrayRef<int64_t> shape) {
  return llvm::all_of(shape, [](int64_t d) { return d >= 0; });
}

bool AreShapesSupported(const ConvView& data) {
  return IsShapeFullyStatic(data.InputShape()) &&
         IsShapeFullyStatic(data.KernelShape()) &&
         IsShapeFullyStatic(data.OutputShape());
}

bool IsPaddingSupported(const ConvView& data) {
  return llvm::all_of(data.Padding(), [](const DimPadding& p) {
    return p.Hi() == 0 && p.Lo() == 0;
  });
}

bool IsInputDilationSupported(const ConvView& data) {
  return llvm::all_of(data.InputDilations(), [](int64_t v) { return v == 1; });
}

bool IsBatchGroupSupported(const ConvView& data) {
  return data.BatchGroupCount() == 1;
}

bool IsWindowReversalSupported(const ConvView& data) {
  return llvm::all_of(data.WindowReversal(), [](bool b) { return !b; });
}

// Determines if it is OK to leave given mhlo.convolution in the mhlo dialect.
// Used externally to setup a ConversionTarget with dynamically legal
// mhlo.convolution. Doubles as matching predicate during legalization.
bool IsConvLegal(mhlo::ConvolutionOp op) {
  const ConvView data(op);

  const bool supported_conv_type =
      IsStandardConv(data) || IsDepthwiseConv(data);

  return !supported_conv_type || !IsBatchGroupSupported(data) ||
         !IsInputDilationSupported(data) || !AreShapesSupported(data) ||
         !IsTFLNativeLayout(data) || !IsPaddingSupported(data) ||
         !IsWindowReversalSupported(data);
}

//===----------------------------------------------------------------------===//
// mhlo.convolution -> tfl
//===----------------------------------------------------------------------===//

// Bias is a zero tensor of shape [output_channels].
arith::ConstantOp BuildEmptyBias(OpBuilder& b, Location loc,
                                 const ConvView& data) {
  auto bias_type = RankedTensorType::get(
      {data.OutputLayout().SpecialDim2(data.OutputShape())},
      data.ElementType());
  auto bias_const_data = b.getZeroAttr(bias_type);
  return b.create<arith::ConstantOp>(loc, bias_const_data);
}

class LegalizeConv2D : public OpConversionPattern<mhlo::ConvolutionOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ConvolutionOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};

LogicalResult LegalizeConv2D::matchAndRewrite(
    mhlo::ConvolutionOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // Parse mhlo.convolution attrs into cc types.
  const ConvView data(op);

  if (IsConvLegal(op) || !IsStandardConv(data) ||
      data.InputLayout().Rank() != 4) {
    return failure();
  }

  //
  // dilations
  //===-------

  const auto& kernel_dilations = data.KernelDilations();
  auto tfl_h_dilation = rewriter.getI32IntegerAttr(kernel_dilations[0]);
  auto tfl_w_dilation = rewriter.getI32IntegerAttr(kernel_dilations[1]);

  //
  // strides
  //===-----

  const auto& window_strides = data.Strides();
  auto tfl_h_stride = rewriter.getI32IntegerAttr(window_strides[0]);
  auto tfl_w_stride = rewriter.getI32IntegerAttr(window_strides[1]);

  //
  // padding
  //===-----

  // Explicit and same padding should be handeled in upstream "prepare" phase.
  // Same padding will be fused in downstream "optimize" phase on tfl dialect.
  auto tfl_padding = rewriter.getStringAttr("VALID");

  //
  // build tfl
  //===-------

  auto bias = BuildEmptyBias(rewriter, op->getLoc(), data);

  auto tfl_faf_none = rewriter.getStringAttr("NONE");

  rewriter.replaceOpWithNewOp<TFL::Conv2DOp>(
      op, op.getResult().getType(), op.getLhs(), op.getRhs(), bias,
      tfl_h_dilation, tfl_w_dilation, tfl_faf_none, tfl_padding, tfl_h_stride,
      tfl_w_stride);

  return success();
}

class LegalizeConvDepthwise : public OpConversionPattern<mhlo::ConvolutionOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ConvolutionOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};

LogicalResult LegalizeConvDepthwise::matchAndRewrite(
    mhlo::ConvolutionOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // Parse mhlo.convolution attrs into cc types.
  const ConvView data(op);

  if (IsConvLegal(op) || !IsDepthwiseConv(data)) {
    return failure();
  }

  //
  // dilations
  //===-------

  const auto& kernel_dilations = data.KernelDilations();
  auto tfl_h_dilation = rewriter.getI32IntegerAttr(kernel_dilations[0]);
  auto tfl_w_dilation = rewriter.getI32IntegerAttr(kernel_dilations[1]);

  //
  // strides
  //===-----

  const auto& window_strides = data.Strides();
  auto tfl_h_stride = rewriter.getI32IntegerAttr(window_strides[0]);
  auto tfl_w_stride = rewriter.getI32IntegerAttr(window_strides[1]);

  //
  // padding
  //===-----

  // Explicit and same padding should be handeled in upstream "prepare" phase.
  // Same padding will be fused in downstream "optimize" phase on tfl dialect.
  auto tfl_padding = rewriter.getStringAttr("VALID");

  //
  // depth multiplier
  //===-----

  const int64_t out_channels =
      data.OutputLayout().SpecialDim2(data.OutputShape());
  const int64_t in_channels = data.InputLayout().SpecialDim2(data.InputShape());
  const int32_t depth_multiplier = out_channels / in_channels;
  auto depth_multipler_attr = rewriter.getI32IntegerAttr(depth_multiplier);

  //
  // build tfl
  //===-------

  auto bias = BuildEmptyBias(rewriter, op->getLoc(), data);

  auto tfl_faf_none = rewriter.getStringAttr("NONE");

  rewriter.replaceOpWithNewOp<TFL::DepthwiseConv2DOp>(
      op, op.getResult().getType(), op.getLhs(), op.getRhs(), bias,
      tfl_h_dilation, tfl_w_dilation, tfl_faf_none, tfl_padding, tfl_h_stride,
      tfl_w_stride, depth_multipler_attr);

  return success();
}

class LegalizeConv3D : public OpConversionPattern<mhlo::ConvolutionOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ConvolutionOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};

LogicalResult LegalizeConv3D::matchAndRewrite(
    mhlo::ConvolutionOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // Parse mhlo.convolution attrs into cc types.
  const ConvView data(op);

  if (IsConvLegal(op) || !IsStandardConv(data) ||
      data.InputLayout().Rank() != 5) {
    return failure();
  }

  //
  // dilations
  //===-------

  const auto& kernel_dilations = data.KernelDilations();
  auto tfl_d_dilation = rewriter.getI32IntegerAttr(kernel_dilations[0]);
  auto tfl_h_dilation = rewriter.getI32IntegerAttr(kernel_dilations[1]);
  auto tfl_w_dilation = rewriter.getI32IntegerAttr(kernel_dilations[2]);

  //
  // strides
  //===-----

  const auto& window_strides = data.Strides();
  auto tfl_d_stride = rewriter.getI32IntegerAttr(window_strides[0]);
  auto tfl_h_stride = rewriter.getI32IntegerAttr(window_strides[1]);
  auto tfl_w_stride = rewriter.getI32IntegerAttr(window_strides[2]);

  //
  // padding
  //===-----

  // Explicit and same padding should be handeled in upstream "prepare" phase.
  // Same padding will be fused in downstream "optimize" phase on tfl dialect.
  auto tfl_padding = rewriter.getStringAttr("VALID");

  //
  // build tfl
  //===-------

  auto bias = BuildEmptyBias(rewriter, op->getLoc(), data);

  auto tfl_faf_none = rewriter.getStringAttr("NONE");

  rewriter.replaceOpWithNewOp<TFL::Conv3DOp>(
      op, op.getResult().getType(), op.getLhs(), op.getRhs(), bias,
      tfl_d_dilation, tfl_h_dilation, tfl_w_dilation, tfl_faf_none, tfl_padding,
      tfl_d_stride, tfl_h_stride, tfl_w_stride);

  return success();
}

// Convert a 1-D convolution into a 2-D convolution (which TF supports) so that
// it can be rewritten by the pattern `Convert2DConvOp`.
class Conv1DToConv2D : public OpRewritePattern<mhlo::ConvolutionOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConvolutionOp op,
                                PatternRewriter& rewriter) const final;
};

std::tuple<llvm::SmallVector<int64_t>, Layout> InsertTrivialSpatialDim(
    const Layout& layout, ArrayRef<int64_t> shape) {
  // Make new Layout with extra spatial dimension.
  const int64_t last_spatial = layout.Spatials()[layout.Rank() - 3];
  const int64_t new_dim1 = (layout.SpecialDim1() > last_spatial)
                               ? layout.SpecialDim1() + 1
                               : layout.SpecialDim1();
  const int64_t new_dim2 = (layout.SpecialDim2() > last_spatial)
                               ? layout.SpecialDim2() + 1
                               : layout.SpecialDim2();

  llvm::SmallVector<int64_t> new_spatials(layout.Spatials());
  const int64_t new_last_spatial = new_spatials.back() + 1;
  new_spatials.push_back(new_last_spatial);

  // Get new shape.
  llvm::SmallVector<int64_t, 4> new_shape(shape.size() + 1, 1);
  new_shape[new_dim1] = layout.SpecialDim1(shape);
  new_shape[new_dim2] = layout.SpecialDim2(shape);
  for (auto new_spatial : new_spatials) {
    if (new_spatial == new_last_spatial) {
      continue;
    }
    new_shape[new_spatial] = shape[new_spatial];
  }
  return std::tuple(new_shape, Layout(new_dim1, new_dim2, new_spatials));
}

LogicalResult Conv1DToConv2D::matchAndRewrite(mhlo::ConvolutionOp op,
                                              PatternRewriter& rewriter) const {
  const ConvView view(op);

  if (view.InputLayout().Rank() != 3) {
    return rewriter.notifyMatchFailure(op, "Not 1D conv.");
  }

  if (!IsInputDilationSupported(view)) {
    return rewriter.notifyMatchFailure(op, "Expects trivial lhs dims.");
  }

  if (!AreShapesSupported(view)) {
    return rewriter.notifyMatchFailure(op, "Expects static dims.");
  }

  if (!IsWindowReversalSupported(view)) {
    return rewriter.notifyMatchFailure(op, "Expects window reversal trivial.");
  }

  if (!view.InputLayout().AreSpatialsIota() ||
      !view.KernelLayout().AreSpatialsIota() ||
      !view.OutputLayout().AreSpatialsIota()) {
    return rewriter.notifyMatchFailure(op,
                                       "Expects well formed spatials dims.");
  }

  //
  // Transpose and reshape the input and kernel
  //=-----

  // Add new trivial spatial dimension to input (LHS).
  auto [lhs_new_shape, lhs_new_layout] =
      InsertTrivialSpatialDim(view.InputLayout(), view.InputShape());
  auto lhs_new_type = op.getLhs().getType().clone(lhs_new_shape);
  auto new_lhs =
      rewriter.create<mhlo::ReshapeOp>(op.getLoc(), lhs_new_type, op.getLhs());

  // Add new trivial spatial dimension to kernel.
  auto [rhs_new_shape, rhs_new_layout] =
      InsertTrivialSpatialDim(view.KernelLayout(), view.KernelShape());
  auto rhs_new_type = op.getRhs().getType().clone(rhs_new_shape);
  auto new_rhs =
      rewriter.create<mhlo::ReshapeOp>(op.getLoc(), rhs_new_type, op.getRhs());

  // Add new trivial spatial dimension to output (insert reshape later).
  auto [out_new_shape, out_new_layout] =
      InsertTrivialSpatialDim(view.OutputLayout(), view.OutputShape());
  auto out_new_type = op.getResult().getType().clone(out_new_shape);

  //
  // Create 2d equivalents for 1d convolution attributes.
  //=-----

  // Window Strides
  llvm::SmallVector<int64_t, 2> strides_2d;
  strides_2d.push_back(view.Strides()[0]);
  strides_2d.push_back(1);
  auto strides_2d_attr = DenseIntElementsAttr::get(
      RankedTensorType::get({2}, rewriter.getI64Type()), strides_2d);

  // Padding
  SmallVector<int64_t, 4> padding_2d;
  const auto& dim_pad = view.Padding()[0];
  padding_2d.push_back(dim_pad.Lo());
  padding_2d.push_back(dim_pad.Hi());
  padding_2d.push_back(0);
  padding_2d.push_back(0);
  auto padding_2d_attr = DenseIntElementsAttr::get(
      RankedTensorType::get({2, 2}, rewriter.getI64Type()), padding_2d);

  // LHS dilation
  SmallVector<int64_t, 2> lhs_dilation_2d(2, 1);
  auto lhs_dilation_2d_attr = DenseIntElementsAttr::get(
      RankedTensorType::get({2}, rewriter.getI64Type()), lhs_dilation_2d);

  // RHS dilation
  SmallVector<int64_t, 2> rhs_dilation_2d;
  rhs_dilation_2d.push_back(view.KernelDilations()[0]);
  rhs_dilation_2d.push_back(1);
  auto rhs_dilation_2d_attr = DenseIntElementsAttr::get(
      RankedTensorType::get({2}, rewriter.getI64Type()), rhs_dilation_2d);

  auto window_reversal_2d_attr = DenseIntElementsAttr::get(
      RankedTensorType::get({2}, rewriter.getIntegerType(1)),
      SmallVector<bool>({false, false}));

  // New dnums.
  auto dnums_2d = mhlo::ConvDimensionNumbersAttr::get(
      rewriter.getContext(), lhs_new_layout.SpecialDim1(),
      lhs_new_layout.SpecialDim2(), lhs_new_layout.Spatials(),
      rhs_new_layout.SpecialDim1(), rhs_new_layout.SpecialDim2(),
      rhs_new_layout.Spatials(), out_new_layout.SpecialDim1(),
      out_new_layout.SpecialDim2(), out_new_layout.Spatials());

  //
  // Build 2-D convolution with reshaped output.
  //=-----

  auto conv2d_op = rewriter.create<mhlo::ConvolutionOp>(
      op.getLoc(), out_new_type, new_lhs, new_rhs, strides_2d_attr,
      padding_2d_attr, lhs_dilation_2d_attr, rhs_dilation_2d_attr,
      window_reversal_2d_attr, dnums_2d, op.getFeatureGroupCount(),
      op.getBatchGroupCount(), op.getPrecisionConfigAttr());

  rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(op, op.getResult().getType(),
                                               conv2d_op.getResult());
  return success();
}

}  // namespace

void PopulateLegalizeConvPatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                                  ConversionTarget& target) {
  patterns.add<LegalizeConv2D, LegalizeConv3D, LegalizeConvDepthwise>(ctx);
  target.addDynamicallyLegalOp<mhlo::ConvolutionOp>(IsConvLegal);
}

void PopulatePrepareConvPatterns(MLIRContext* ctx,
                                 RewritePatternSet& patterns) {
  patterns.add<Conv1DToConv2D>(ctx);
}
}  // namespace mlir::odml
