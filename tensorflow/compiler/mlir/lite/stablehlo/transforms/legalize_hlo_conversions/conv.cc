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
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/conv_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {

using ::llvm::ArrayRef;

//===----------------------------------------------------------------------===//
// support/legality checking
//===----------------------------------------------------------------------===//

bool IsRankSupported(const ConvData& data) {
  return data.InputShape().size() == 4;
}

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

  const bool are_groups_supported =
      IsStandardFeatureGroup(data) && IsBatchGroupSupported(data);

  return !are_groups_supported || !IsRankSupported(data) ||
         !IsInputDilationSupported(data) || !AreShapesSupported(data) ||
         !IsTFLNativeLayout(data) || !IsPaddingSupported(data) ||
         !IsWindowReversalSupported(data) || !IsStandardConv(op);
}

//===----------------------------------------------------------------------===//
// mhlo.convolution -> tfl
//===----------------------------------------------------------------------===//

LogicalResult LegalizeConv::matchAndRewrite(
    mhlo::ConvolutionOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // Parse mhlo.convolution attrs into cc types.
  const ConvData data(op);

  if (IsConvLegal(op)) {
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
  // empty bias
  //===--------

  // Bias is a zero tensor of shape [output_channels].
  auto bias_type = RankedTensorType::get(
      {data.OutputLayout().SpecialDim2(data.OutputShape())},
      data.ElementType());
  auto bias_const_data = rewriter.getZeroAttr(bias_type);
  auto bias = rewriter.create<arith::ConstantOp>(op->getLoc(), bias_const_data);

  //
  // build tfl
  //===-------

  auto tfl_faf_none = rewriter.getStringAttr("NONE");

  rewriter.replaceOpWithNewOp<TFL::Conv2DOp>(
      op, op.getResult().getType(), op.getLhs(), op.getRhs(), bias,
      tfl_h_dilation, tfl_w_dilation, tfl_faf_none, tfl_padding, tfl_h_stride,
      tfl_w_stride);

  return success();
}

}  // namespace mlir::odml
