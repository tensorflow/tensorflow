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

bool AreShapesSupported(const ConvData& data) {
  return IsShapeFullyStatic(data.InputShape()) &&
         IsShapeFullyStatic(data.KernelShape()) &&
         IsShapeFullyStatic(data.OutputShape());
}

bool IsPaddingSupported(const ConvData& data) {
  return llvm::all_of(data.Padding(), [](const DimPadding& p) {
    return p.Hi() == 0 && p.Lo() == 0;
  });
}

bool IsInputDilationSupported(const ConvData& data) {
  return llvm::all_of(data.InputDilations(), [](int64_t v) { return v == 1; });
}

bool IsBatchGroupSupported(const ConvData& data) {
  return data.BatchGroupCount() == 1;
}

bool IsWindowReversalSupported(const ConvData& data) {
  return llvm::all_of(data.WindowReversal(), [](bool b) { return !b; });
}

// Determines if it is OK to leave given mhlo.convolution in the mhlo dialect.
// Used externally to setup a ConversionTarget with dynamically legal
// mhlo.convolution. Doubles as matching predicate during legalization.
bool IsConvLegal(mhlo::ConvolutionOp op) {
  const ConvData data(op);

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
                                 const ConvData& data) {
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
  const ConvData data(op);

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
  const ConvData data(op);

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
  const ConvData data(op);

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

}  // namespace

void PopulateConvPatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                          ConversionTarget& target) {
  patterns.add<LegalizeConv2D, LegalizeConv3D, LegalizeConvDepthwise>(ctx);
  target.addDynamicallyLegalOp<mhlo::ConvolutionOp>(IsConvLegal);
}
}  // namespace mlir::odml
