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
#include <string>
#include <tuple>
#include <variant>

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

bool FullyStaticShape(ArrayRef<int64_t> shape) {
  return llvm::all_of(shape, [](int64_t d) { return d >= 0; });
}

bool IsInputShapeSupported(const ConvData& data) {
  return FullyStaticShape(data.InputShape());
}

bool IsKernelShapeSupported(const ConvData& data) {
  return FullyStaticShape(data.KernelShape());
}

bool IsOutputShapeSupported(const ConvData& data) {
  return FullyStaticShape(data.OutputShape());
}

bool IsRankSupported(const ConvData& data) {
  return data.InputShape().size() == 4;
}

bool IsInputLayoutSupported(const ConvData& data) {
  return data.Layouts().Input().IsNHWC();
}

bool IsKernelLayoutSupported(const ConvData& data) {
  return data.Layouts().Kernel().IsOHWI();
}

bool IsOutputLayoutSupported(const ConvData& data) {
  return data.Layouts().Output().IsNHWC();
}

bool IsStrideSupported(const ConvData& data) {
  return llvm::all_of(data.Strides(), [](int64_t v) { return v == 1; });
}

bool IsPaddingSupported(const ConvData& data) {
  ArrayRef<DimPadding> padding = data.Padding();
  for (const auto& d : padding) {
    const bool hi_trivial = d.Hi() == 0;
    const bool lo_trivial = d.Lo() == 0;
    if (!hi_trivial || !lo_trivial) {
      return false;
    }
  }
  return true;
}

bool IsInputDilationSupported(const ConvData& data) {
  return llvm::all_of(data.InputDilations(), [](int64_t v) { return v == 1; });
}

bool IsKernelDilationSupported(const ConvData& data) {
  return llvm::all_of(data.KernelDilations(), [](int64_t v) { return v == 1; });
}

bool IsFeatureGroupSupported(const ConvData& data) {
  return data.FeatureGroupCount() == 1;
}

bool IsBatchGroupSupported(const ConvData& data) {
  return data.BatchGroupCount() == 1;
}

bool IsWindowReversalSupported(const ConvData& data) {
  return llvm::all_of(data.WindowReversal(), [](bool b) { return !b; });
}

// Determines if it is OK to leave given mhlo.convolution in the mhlo dialect.
// Used externally to setup a ConversionTarget with dynamically legal
// mhlo.convolution.
bool IsConvLegal(mhlo::ConvolutionOp op) {
  const ConvData data(op);

  const bool are_shapes_supported = IsInputShapeSupported(data) &&
                                    IsKernelShapeSupported(data) &&
                                    IsOutputShapeSupported(data);

  const bool are_layouts_supported = IsInputLayoutSupported(data) &&
                                     IsKernelLayoutSupported(data) &&
                                     IsOutputLayoutSupported(data);

  const bool are_groups_supported =
      IsFeatureGroupSupported(data) && IsBatchGroupSupported(data);

  const bool are_dilations_supported =
      IsInputDilationSupported(data) && IsKernelDilationSupported(data);

  const bool is_legal = !are_shapes_supported || !are_layouts_supported ||
                        !are_groups_supported || !are_dilations_supported ||
                        !IsRankSupported(data) || !IsStrideSupported(data) ||
                        !IsPaddingSupported(data) ||
                        !IsWindowReversalSupported(data);

  return is_legal;
}

//===----------------------------------------------------------------------===//
// mhlo.convolution -> tfl
//===----------------------------------------------------------------------===//

// Gets pair of dilations attrs on spatial dims for tfl.conv_2d if
// corresponding mhlo.convolution dilations attrs are supported. Returns failure
// message if not.
std::variant<std::tuple<mlir::IntegerAttr, mlir::IntegerAttr>, std::string>
GetTflDilationAttrs(const ConvData& data, OpBuilder& b) {
  if (!IsInputDilationSupported(data)) {
    return "Input (lhs) dilations are not supported.";
  }
  // TODO: b/351437662 - Handle non-trivial kernel dilations.
  if (!IsKernelDilationSupported(data)) {
    return "Kernel (rhs) dilations are not supported.";
  }

  const auto& kernel_dilations = data.KernelDilations();
  auto tfl_h_dilation = b.getI32IntegerAttr(kernel_dilations[0]);
  auto tfl_w_dilation = b.getI32IntegerAttr(kernel_dilations[1]);

  return std::tuple{tfl_h_dilation, tfl_w_dilation};
}

// Gets the pair of stride attrs on spatial dims for tfl.conv_2d if
// corresponding mhlo.convolution stride attrs are supported. Returns failure
// message if not.
std::variant<std::tuple<IntegerAttr, IntegerAttr>, std::string>
GetTflStrideAttr(const ConvData& data, OpBuilder& b) {
  if (!IsStrideSupported(data)) {
    // TODO: b/351437662 - Handle non-trivial strides.
    return "Strides are not supported";
  }

  const auto& window_strides = data.Strides();
  auto tfl_h_stride = b.getI32IntegerAttr(window_strides[0]);
  auto tfl_w_stride = b.getI32IntegerAttr(window_strides[1]);

  return std::tuple{tfl_h_stride, tfl_w_stride};
}

LogicalResult LegalizeConv::matchAndRewrite(
    mhlo::ConvolutionOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // Parse mhlo.convolution attrs into cc types.
  const ConvData data(op);

  //
  // pre-requisite support checks
  //===--------------------------

  if (!IsRankSupported(data)) {
    return rewriter.notifyMatchFailure(op->getLoc(), "Rank is not supported.");
  }

  if (!IsOutputLayoutSupported(data)) {
    return rewriter.notifyMatchFailure(op->getLoc(),
                                       "Output layout is not supported.");
  }

  if (!IsOutputShapeSupported(data)) {
    return rewriter.notifyMatchFailure(op->getLoc(),
                                       "Output shape is not supported.");
  }

  if (!IsInputLayoutSupported(data)) {
    return rewriter.notifyMatchFailure(op->getLoc(),
                                       "Input layout is not supported.");
  }

  if (!IsInputShapeSupported(data)) {
    return rewriter.notifyMatchFailure(op->getLoc(),
                                       "Input shape is not supported.");
  }

  if (!IsKernelLayoutSupported(data)) {
    return rewriter.notifyMatchFailure(op->getLoc(),
                                       "Kernel layout is not supported.");
  }

  if (!IsKernelShapeSupported(data)) {
    return rewriter.notifyMatchFailure(op->getLoc(),
                                       "Input shape is not supported.");
  }

  if (!IsBatchGroupSupported(data)) {
    return rewriter.notifyMatchFailure(op->getLoc(),
                                       "Batch group is not supported.");
  }

  if (!IsFeatureGroupSupported(data)) {
    return rewriter.notifyMatchFailure(op->getLoc(),
                                       "Feature group is not supported.");
  }

  if (!IsWindowReversalSupported(data)) {
    return rewriter.notifyMatchFailure(op->getLoc(),
                                       "Window reversal is not supported.");
  }

  if (!IsPaddingSupported(data)) {
    return rewriter.notifyMatchFailure(op->getLoc(),
                                       "Padding is not supported.");
  }

  //
  // parse/validate dilations
  //===----------------------

  auto tfl_dilations_or_msg = GetTflDilationAttrs(data, rewriter);
  if (std::holds_alternative<std::string>(tfl_dilations_or_msg)) {
    return rewriter.notifyMatchFailure(
        op->getLoc(), std::get<std::string>(tfl_dilations_or_msg));
  }
  auto [tfl_h_dilation, tfl_w_dilation] =
      std::get<std::tuple<IntegerAttr, IntegerAttr>>(tfl_dilations_or_msg);

  //
  // parse/validate strides
  //===--------------------

  auto tfl_strides_or_msg = GetTflStrideAttr(data, rewriter);
  if (std::holds_alternative<std::string>(tfl_strides_or_msg)) {
    return rewriter.notifyMatchFailure(
        op->getLoc(), std::get<std::string>(tfl_strides_or_msg));
  }
  auto [tfl_h_stride, tfl_w_stride] =
      std::get<std::tuple<IntegerAttr, IntegerAttr>>(tfl_strides_or_msg);

  //
  // build tfl.conv_2d
  //===---------------

  auto tfl_faf_none = rewriter.getStringAttr("NONE");

  // Explicit and same padding will be handeled in upstream "prepare" phase.
  // Same padding will be fused in downstream "optimize" phase on tfl dialect.
  auto tfl_padding = rewriter.getStringAttr("VALID");

  // Bias is a zero tensor of shape [output_channels].
  auto bias_type = RankedTensorType::get(
      {data.Layouts().Output().Feature(data.OutputShape())},
      data.ElementType());
  auto bias_const_data = rewriter.getZeroAttr(bias_type);
  auto bias = rewriter.create<arith::ConstantOp>(op->getLoc(), bias_const_data);

  auto new_tfl_conv = rewriter.create<TFL::Conv2DOp>(
      op->getLoc(), op.getResult().getType(), op.getLhs(), op.getRhs(), bias,
      tfl_h_dilation, tfl_w_dilation, tfl_faf_none, tfl_padding, tfl_h_stride,
      tfl_w_stride);

  //
  // final IR edits
  //==-------------

  rewriter.replaceOp(op, new_tfl_conv);

  return mlir::success();
}

}  // namespace mlir::odml
