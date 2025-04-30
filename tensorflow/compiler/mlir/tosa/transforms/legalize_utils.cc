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

#include "tensorflow/compiler/mlir/tosa/transforms/legalize_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"                 // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"                // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"               // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"                // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"     // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"          // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"                   // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"               // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                        // from @llvm-project
#include "mlir/Support/LLVM.h"                           // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/kernels/internal/common.h"
#include "tensorflow/compiler/mlir/lite/kernels/internal/quantization_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/hard_swish.h"
#include "xla/tsl/framework/fixedpoint/FixedPoint.h"

// Implements legalization and post-legalization optimization helper functions

namespace mlir {
namespace tosa {

mlir::TypeAttr getConvAccTypeAttr(PatternRewriter& rewriter,
                                  mlir::Type input_etype,
                                  mlir::Type output_etype) {
  // in case of quantized types: get base element types
  if (auto qtype =
          llvm::dyn_cast<mlir::quant::UniformQuantizedType>(input_etype))
    input_etype = qtype.getStorageType();

  if (auto qtype =
          llvm::dyn_cast<mlir::quant::UniformQuantizedType>(output_etype))
    output_etype = qtype.getStorageType();

  // special cases: input_etype and output_etype are both f16 or bf16: use
  // acc_type=f32
  if ((input_etype.isF16() && output_etype.isF16()) ||
      (input_etype.isBF16() && output_etype.isBF16())) {
    return mlir::TypeAttr::get(rewriter.getF32Type());
  }

  // otherwise, use output_etype as acc_type
  return mlir::TypeAttr::get(output_etype);
}

std::optional<Value> convertTFConv2DCommon(
    PatternRewriter& rewriter, Operation* op, RankedTensorType output_type,
    Value input, Value filter, Value bias, ArrayAttr strides_attr,
    ArrayAttr dilations_attr, ArrayAttr explicit_padding_attr,
    StringRef padding_ref, StringRef data_format_ref) {
  RankedTensorType input_type = dyn_cast<RankedTensorType>(input.getType());
  RankedTensorType filter_type = dyn_cast<RankedTensorType>(filter.getType());
  // Not a ranked tensor output
  if (!input_type) return std::nullopt;
  if (!filter_type) return std::nullopt;

  // Transpose [H, W, I, O] to [O, H, W, I]
  auto filter_shape = filter_type.getShape();
  SmallVector<int64_t, 4> a1_transpose_dims;
  a1_transpose_dims.push_back(filter_shape[3]);
  a1_transpose_dims.push_back(filter_shape[0]);
  a1_transpose_dims.push_back(filter_shape[1]);
  a1_transpose_dims.push_back(filter_shape[2]);

  auto a1_filter_transpose_perm_attr =
      rewriter.getDenseI32ArrayAttr({3, 0, 1, 2});

  auto a1_filter_transpose_op = CreateOpAndInfer<tosa::TransposeOp>(
      rewriter, op->getLoc(),
      tensorflow::GetTypeFromTFTensorShape(a1_transpose_dims,
                                           filter_type.getElementType()),
      filter, a1_filter_transpose_perm_attr);

  // Only support NHWC now.
  if (data_format_ref.str() != "NHWC") {
    op->emitWarning("convertTDConv2DCommon only supports NHWC!");
    return std::nullopt;
  }

  DenseI64ArrayAttr stride;
  DenseI64ArrayAttr dilation;
  DenseI64ArrayAttr pad;
  {
    if (!strides_attr) {
      stride = rewriter.getDenseI64ArrayAttr({1, 1});
    } else {
      // Note: hardcoded to NHWC for now
      int64_t stride_h = mlir::cast<IntegerAttr>(strides_attr[1]).getInt();
      int64_t stride_w = mlir::cast<IntegerAttr>(strides_attr[2]).getInt();
      stride = rewriter.getDenseI64ArrayAttr({stride_h, stride_w});
    }
  }
  {
    if (!dilations_attr) {
      dilation = rewriter.getDenseI64ArrayAttr({1, 1});
    } else {
      // Note: hardcoded to NHWC for now
      int64_t dilation_h = mlir::cast<IntegerAttr>(dilations_attr[1]).getInt();
      int64_t dilation_w = mlir::cast<IntegerAttr>(dilations_attr[2]).getInt();
      dilation = rewriter.getDenseI64ArrayAttr({dilation_h, dilation_w});
    }
  }
  {
    tensorflow::Padding tf_pad;
    if (!GetPaddingFromString(padding_ref.str(), &tf_pad).ok()) {
      op->emitWarning("Could not get padding data from padding string term!");
      return std::nullopt;
    }

    tensorflow::TensorFormat data_format_tf;
    if (!FormatFromString(data_format_ref.str(), &data_format_tf))
      return std::nullopt;

    if (tf_pad == tensorflow::Padding::EXPLICIT) {
      pad = getPaddingValuesFromExplicitPadAttr(explicit_padding_attr,
                                                data_format_tf, rewriter);
    } else {
      if (!getPaddingValuesFromPadType(tf_pad, data_format_tf,
                                       0,  // tensorflow::FORMAT_HWIO
                                       input_type, filter_type, stride,
                                       dilation, rewriter, pad))
        return std::nullopt;
    }
  }

  auto acc_type =
      getConvAccTypeAttr(rewriter,
                         /* input_etype = */ input_type.getElementType(),
                         /* output_etype = */ output_type.getElementType());
  return CreateOpAndInfer<tosa::Conv2DOp>(rewriter, op->getLoc(), output_type,
                                          input,
                                          a1_filter_transpose_op.getResult(),
                                          bias, pad, stride, dilation, acc_type)
      .getResult();
}

std::optional<Value> convertTFConv3DCommon(
    PatternRewriter& rewriter, Operation* op, ShapedType output_type,
    Value input, Value filter, Value bias, ArrayAttr strides_attr,
    ArrayAttr dilations_attr, StringRef padding_ref,
    StringRef data_format_ref) {
  DenseI64ArrayAttr strides;
  if (!strides_attr) {
    // Defaults to [1, 1, 1].
    strides = rewriter.getDenseI64ArrayAttr({1, 1, 1});
  } else {
    int64_t stride_d = mlir::cast<IntegerAttr>(strides_attr[1]).getInt();
    int64_t stride_h = mlir::cast<IntegerAttr>(strides_attr[2]).getInt();
    int64_t stride_w = mlir::cast<IntegerAttr>(strides_attr[3]).getInt();
    strides = rewriter.getDenseI64ArrayAttr({stride_d, stride_h, stride_w});
  }

  DenseI64ArrayAttr dilations;
  if (!dilations_attr) {
    // Defaults to [1, 1, 1].
    dilations = rewriter.getDenseI64ArrayAttr({1, 1, 1});
  } else {
    int64_t dilation_d = mlir::cast<IntegerAttr>(dilations_attr[1]).getInt();
    int64_t dilation_h = mlir::cast<IntegerAttr>(dilations_attr[2]).getInt();
    int64_t dilation_w = mlir::cast<IntegerAttr>(dilations_attr[3]).getInt();
    dilations =
        rewriter.getDenseI64ArrayAttr({dilation_d, dilation_h, dilation_w});
  }

  RankedTensorType input_type = mlir::cast<RankedTensorType>(input.getType());
  DenseI64ArrayAttr pads;
  {
    RankedTensorType filter_type =
        mlir::cast<RankedTensorType>(filter.getType());

    tensorflow::TensorFormat data_format_tf;
    if (!FormatFromString(data_format_ref, &data_format_tf)) {
      return std::nullopt;
    }

    tensorflow::Padding tf_pad;
    if (!GetPaddingFromString(padding_ref.str(), &tf_pad).ok()) {
      (void)rewriter.notifyMatchFailure(
          op, "could not get padding data from padding string term");
      return std::nullopt;
    }

    if (tf_pad == tensorflow::Padding::EXPLICIT) {
      (void)rewriter.notifyMatchFailure(op, "don't have explicit padding");
      return std::nullopt;
    }

    if (!getPaddingValuesFromPadType(tf_pad, data_format_tf, 0, input_type,
                                     filter_type, strides, dilations, rewriter,
                                     pads)) {
      return std::nullopt;
    }
  }

  auto acc_type =
      getConvAccTypeAttr(rewriter,
                         /* input_etype = */ input_type.getElementType(),
                         /* output_etype = */ output_type.getElementType());

  return convertConv3DCommon(rewriter, op, output_type, input, filter, bias,
                             pads, strides, dilations, acc_type,
                             data_format_ref);
}

LogicalResult getDynamicDims(PatternRewriter& rewriter, Operation* op,
                             Value value, llvm::SmallVector<Value>& dims) {
  auto value_ty = dyn_cast<ShapedType>(value.getType());
  if (!value_ty || !value_ty.hasRank()) return failure();

  dims.resize(value_ty.getRank());
  RankedTensorType dim_ty =
      tensorflow::GetTypeFromTFTensorShape({}, rewriter.getI32Type());

  for (int i = 0, s = value_ty.getRank(); i < s; ++i) {
    if (!value_ty.isDynamicDim(i)) {
      dims[i] = rewriter.create<tosa::ConstOp>(
          op->getLoc(), dim_ty,
          SplatElementsAttr::get(dim_ty, value_ty.getDimSize(i)));
      continue;
    }

    // TODO(suderman): This should be changed to TOSA operations when TOSA has
    // a TOSA dimension op.
    Value dim = rewriter.create<tensor::DimOp>(op->getLoc(), value, i);
    dim = rewriter.create<arith::IndexCastOp>(op->getLoc(),
                                              rewriter.getI32Type(), dim);
    dim = rewriter.create<tensor::FromElementsOp>(op->getLoc(), dim_ty,
                                                  ValueRange{dim});
    dims[i] = dim;
  }

  return success();
}

std::optional<Value> buildReshapeWithDynamicDims(PatternRewriter& rewriter,
                                                 Operation* op,
                                                 Value input_value,
                                                 ShapedType output_type,
                                                 llvm::ArrayRef<Value> dims) {
  const ShapedType input_ty = dyn_cast<ShapedType>(input_value.getType());
  if (!input_ty) {
    (void)rewriter.notifyMatchFailure(op, "input is not a shaped type");
    return std::nullopt;
  }

  const auto e_ty = input_ty.getElementType();
  llvm::SmallVector<int64_t> static_dims;

  if (output_type.hasRank()) {
    static_dims = tensorflow::ConvertMlirShapeToTF(output_type.getShape());
  } else {
    static_dims.resize(dims.size(), tensorflow::kTFDynamicSize);
  }

  int64_t dyn_count = 0;
  for (int i = 0, s = dims.size(); i < s; ++i) {
    auto dim = dims[i];
    SplatElementsAttr dim_attr;
    if (matchPattern(dim, m_Constant(&dim_attr))) {
      if (mlir::cast<ShapedType>(dim_attr.getType()).getRank() != 0) {
        (void)rewriter.notifyMatchFailure(
            op, "dim for building tosa::ReshapeOp should be rank-0");
        return std::nullopt;
      }
      int64_t size = dim_attr.getSplatValue<APInt>().getSExtValue();

      // Check that static shapes agree.
      if (size != tensorflow::kTFDynamicSize &&
          static_dims[i] != tensorflow::kTFDynamicSize &&
          size != static_dims[i]) {
        (void)rewriter.notifyMatchFailure(
            op, "mismatch reshape static dim when creating tosa::ReshapeOp");
        return std::nullopt;
      }

      static_dims[i] =
          size == tensorflow::kTFDynamicSize ? static_dims[i] : size;
    }

    if (static_dims[i] == tensorflow::kTFDynamicSize) dyn_count++;
  }

  if (dyn_count > 1) {
    (void)rewriter.notifyMatchFailure(
        op, "multiple dynamic shapes when creating tosa::ReshapeOp");
    return std::nullopt;
  }

  // If the input shape is static and only one dynamic dim is detected, we
  // can easily resolve the dim to be static
  if (input_ty.hasStaticShape() && dyn_count == 1) {
    const int64_t total_elements = input_ty.getNumElements();
    const int64_t shape_elements = std::accumulate(
        static_dims.begin(), static_dims.end(), 1, [](int64_t a, int64_t b) {
          return b == tensorflow::kTFDynamicSize ? a : a * b;
        });
    const int64_t dynamic_dim_value = total_elements / shape_elements;
    std::replace(static_dims.begin(), static_dims.end(),
                 tensorflow::kTFDynamicSize, dynamic_dim_value);
  }

  DenseI64ArrayAttr shape_attr = rewriter.getDenseI64ArrayAttr(static_dims);
  auto output_ty = tensorflow::GetTypeFromTFTensorShape(static_dims, e_ty);

  auto shape_value = getTosaConstShape(rewriter, op->getLoc(), static_dims);

  return rewriter
      .create<tosa::ReshapeOp>(op->getLoc(), output_ty, input_value,
                               shape_value)
      .getResult();
}

Value buildRescaleMultiplier(bool scale32, OpBuilder& builder, Location loc,
                             ArrayRef<int32_t> multipliers) {
  if (scale32) {
    return tosa::getConstTensorInt<int32_t>(builder, loc, multipliers);
  } else {
    SmallVector<int16_t> vec(multipliers.begin(), multipliers.end());
    return tosa::getConstTensorInt<int16_t>(builder, loc, vec);
  }
}

// Create a TOSA rescale op from TFLite scaling multiplier, scaling shift, zero
// points and rounding mode
Value buildRescale(PatternRewriter& rewriter, Operation* op,
                   ShapedType output_type, Value input_val,
                   int32_t scale_multiplier, int32_t scale_shift,
                   int64_t input_zp, int64_t output_zp, StringRef rounding_mode,
                   bool scale32) {
  bool input_unsigned = input_val.getType().isUnsignedInteger();
  bool output_unsigned = output_type.isUnsignedInteger();
  auto loc = op->getLoc();
  Value multiplier_val =
      buildRescaleMultiplier(scale32, rewriter, loc, {scale_multiplier});
  auto shift_val = tosa::getConstTensorInt<int8_t>(rewriter, loc,
                                            {static_cast<int8_t>(scale_shift)});

  // Create input_zp matches the input type and output_zp matches the output
  // type of RescaleOp
  const Value empty_output_val = rewriter.create<tensor::EmptyOp>(
      loc, output_type.getShape(), output_type.getElementType());
  const auto input_zp_val =
      tosa::createZeroPointTensor(rewriter, loc, input_val.getType(), input_zp);
  if (!input_zp_val.has_value())
    op->emitError("Failed to create input zero-point tensor for RescaleOp.");

  const auto output_zp_val =
      tosa::createZeroPointTensor(rewriter, loc, empty_output_val.getType(), output_zp);
  if (!output_zp_val.has_value())
    op->emitError("Failed to create output zero-point tensor for RescaleOp.");

  auto rescale_op = CreateOpAndInfer<tosa::RescaleOp>(
      rewriter, loc, output_type, input_val, multiplier_val, shift_val,
      input_zp_val.value(), output_zp_val.value(),
      rewriter.getBoolAttr(scale32), rewriter.getStringAttr(rounding_mode),
      rewriter.getBoolAttr(false), rewriter.getBoolAttr(input_unsigned),
      rewriter.getBoolAttr(output_unsigned));

  return rescale_op.getResult();
}

// Create a TOSA rescale op from TFLite scaling, zero points and rounding mode
Value buildRescale(PatternRewriter& rewriter, Operation* op,
                   ShapedType output_type, Value input_val, double scale,
                   int64_t input_zp, int64_t output_zp, StringRef rounding_mode,
                   bool scale32) {
  int32_t multiplier;
  int32_t shift;

  int32_t scale_width = scale32 ? 32 : 16;

  if (!computeMultiplierAndShift(scale, multiplier, shift, scale_width)) {
    op->emitError("buildRescale: shift must be in the range 2 <= shift <= 62");
  }

  return buildRescale(rewriter, op, output_type, input_val, multiplier, shift,
                      input_zp, output_zp, rounding_mode, scale32);
}

// Removes the zero point and cast to int32, no need to handle roundings modes
Value removeZeroPointAndCastToInt32(PatternRewriter& rewriter, Operation* op,
                                    Value input_val, int64_t input_zp) {
  return buildRescaleToInt32(rewriter, op, input_val, 1.0f, input_zp);
}

// Creates TOSA rescale op with int32 output
Value buildRescaleToInt32(PatternRewriter& rewriter, Operation* op,
                          Value input_val, int32_t input_scale_multiplier,
                          int32_t input_scale_shift, int64_t input_zp) {
  // Output is always int32 type
  auto input_type = dyn_cast<mlir::ShapedType>(input_val.getType());
  assert(input_type);
  auto output_type = input_type.clone(rewriter.getI32Type());

  std::string rounding_mode =
      IsTFLDoubleRoundingMode() ? "DOUBLE_ROUND" : "SINGLE_ROUND";

  return buildRescale(rewriter, op, output_type, input_val,
                      input_scale_multiplier, input_scale_shift, input_zp,
                      /*output_zp=*/0, rounding_mode,
                      /*scale32=*/true);
}

// Creates TOSA rescale op with int32 output
Value buildRescaleToInt32(PatternRewriter& rewriter, Operation* op,
                          Value input_val, double input_scale,
                          int64_t input_zp) {
  int32_t multiplier;
  int32_t shift;

  const int32_t scale_width = 32;
  computeMultiplierAndShift(input_scale, multiplier, shift, scale_width);

  return buildRescaleToInt32(rewriter, op, input_val, multiplier, shift,
                             input_zp);
}

// Creates TOSA rescale op with int32 input
Value buildRescaleFromInt32(PatternRewriter& rewriter, Operation* op,
                            ShapedType output_type, Value input_val,
                            double output_scale, int64_t output_zp) {
  // Input should be int32 type
  auto input_type = dyn_cast<mlir::ShapedType>(input_val.getType());
  (void)input_type;
  assert(input_type && input_type.getElementType().isInteger(32) &&
         "expected rescale input element type to be i32");

  std::string rounding_mode =
      IsTFLDoubleRoundingMode() ? "DOUBLE_ROUND" : "SINGLE_ROUND";

  // Potentially check input_shape == output_shape here
  return buildRescale(rewriter, op, output_type, input_val, output_scale,
                      /*input_zp=*/0, output_zp, rounding_mode,
                      /*scale32=*/true);
}

// Creates a TOSA rescale op based on conv2d parameters.
Value buildRescaleOpConvOutput(PatternRewriter& rewriter, Operation* op,
                               Value conv_val, ShapedType input_type,
                               ShapedType weight_type, ShapedType output_type) {
  auto input_qtype =
      dyn_cast<mlir::quant::UniformQuantizedType>(input_type.getElementType());
  auto output_qtype =
      dyn_cast<mlir::quant::UniformQuantizedType>(output_type.getElementType());

  double input_scale = input_qtype.getScale();

  int64_t output_zp = output_qtype.getZeroPoint();
  double output_scale = output_qtype.getScale();

  bool scale32 = isScale32(output_qtype);
  int32_t scale_width = scale32 ? 32 : 16;
  // Only use double round if we are doing 32 bit scaling
  std::string rounding_mode = scale32 ? "DOUBLE_ROUND" : "SINGLE_ROUND";

  bool input_unsigned = input_qtype.isUnsignedInteger();
  bool output_unsigned = output_qtype.isUnsignedInteger();

  auto loc = op->getLoc();
  const Value empty_output_val = rewriter.create<tensor::EmptyOp>(
      loc, output_type.getShape(), output_type.getElementType());

  const auto input_zp_val = tosa::createZeroPointTensor(
      rewriter, loc, conv_val.getType(), static_cast<int64_t>(0));
  if (!input_zp_val.has_value())
    op->emitError("Failed to create input zero-point tensor for RescaleOp.");

  const auto output_zp_val =
      tosa::createZeroPointTensor(rewriter, loc, empty_output_val.getType(), output_zp);
  if (!output_zp_val.has_value())
    op->emitError("Failed to create output zero-point tensor for RescaleOp.");

  if (auto weight_per_tensor_qtype =
          dyn_cast<mlir::quant::UniformQuantizedType>(
              weight_type.getElementType())) {
    // Per-tensor quantization
    double weight_scale = weight_per_tensor_qtype.getScale();

    int32_t multiplier;
    int32_t shift;

    double op_tensor_scale = (input_scale * weight_scale) / output_scale;

    computeMultiplierAndShift(op_tensor_scale, multiplier, shift, scale_width);

    Value multiplier_val =
        buildRescaleMultiplier(scale32, rewriter, loc, {multiplier});
    auto shift_val =
        tosa::getConstTensorInt<int8_t>(rewriter, loc, {static_cast<int8_t>(shift)});

    auto rescale_op = CreateOpAndInfer<tosa::RescaleOp>(
        rewriter, loc, output_type, conv_val, multiplier_val, shift_val,
        input_zp_val.value(), output_zp_val.value(),
        rewriter.getBoolAttr(scale32), rewriter.getStringAttr(rounding_mode),
        rewriter.getBoolAttr(false), rewriter.getBoolAttr(input_unsigned),
        rewriter.getBoolAttr(output_unsigned));

    return rescale_op.getResult();

  } else if (auto weight_per_channel_qtype =
                 dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(
                     weight_type.getElementType())) {
    // Per-channel quantization
    SmallVector<int32_t> multiplier_arr;
    SmallVector<int8_t> shift_arr;

    SmallVector<double> weight_scale_arr(
        weight_per_channel_qtype.getScales().begin(),
        weight_per_channel_qtype.getScales().end());

    int64_t output_zp = output_qtype.getZeroPoint();
    double output_scale = output_qtype.getScale();

    for (double weight_scale : weight_scale_arr) {
      int32_t multiplier;
      int32_t shift;

      double op_channel_scale = (input_scale * weight_scale) / output_scale;

      if (!computeMultiplierAndShift(op_channel_scale, multiplier, shift, 32)) {
        op->emitError(
            "buildRescaleOpConvOutput: shift must be in the range 2 <= shift "
            "<= 62");
      }
      // We are matching the tflite behaviour here by scaling by 32-bit
      // then down-scaling to 16-bit for int16x8
      // Reference: tensorflow/lite/kernels/internal/common.cc
      if (!scale32) {
        multiplier = (multiplier < 0x7FFF0000)
                         ? ((multiplier + (1 << 15)) >> 16)
                         : 0x7FFF;
        shift = shift - 16;
      }

      multiplier_arr.push_back(multiplier);
      shift_arr.push_back(static_cast<int8_t>(shift));
    }

    Value multiplier_val =
        buildRescaleMultiplier(scale32, rewriter, loc, multiplier_arr);
    auto shift_val = tosa::getConstTensorInt<int8_t>(rewriter, loc, shift_arr);

    auto rescale_op = CreateOpAndInfer<tosa::RescaleOp>(
        rewriter, loc, output_type, conv_val, multiplier_val, shift_val,
        input_zp_val.value(), output_zp_val.value(),
        rewriter.getBoolAttr(scale32), rewriter.getStringAttr(rounding_mode),
        rewriter.getBoolAttr(true), rewriter.getBoolAttr(input_unsigned),
        rewriter.getBoolAttr(output_unsigned));

    return rescale_op.getResult();

  } else {
    op->emitOpError("buildConvRescaleOp: unknown weight quantized type");
    return nullptr;
  }
}

Value getTosaConstHardSwish8bitTable(PatternRewriter& rewriter, Operation* op,
                                     float input_scale, int32_t input_zp,
                                     float output_scale, int32_t output_zp) {
  // Define tflite params:
  // See: HardSwishPrepare / HardSwishParams
  const float hires_input_scale = (1.0f / 128.0f) * input_scale;
  const float reluish_scale = 3.0f / 32768.0f;
  const float output_multiplier = hires_input_scale / output_scale;

  int16_t output_multiplier_fixedpoint_int16;
  int output_multiplier_exponent;

  int16_t reluish_multiplier_fixedpoint_int16;
  int reluish_multiplier_exponent;

  int32_t output_multiplier_fixedpoint_int32;
  tflite::QuantizeMultiplier(output_multiplier,
                             &output_multiplier_fixedpoint_int32,
                             &output_multiplier_exponent);
  tflite::DownScaleInt32ToInt16Multiplier(output_multiplier_fixedpoint_int32,
                                          &output_multiplier_fixedpoint_int16);
  assert(output_multiplier_exponent <= 0);

  const float reluish_multiplier = hires_input_scale / reluish_scale;
  int32_t reluish_multiplier_fixedpoint_int32;

  tflite::QuantizeMultiplier(reluish_multiplier,
                             &reluish_multiplier_fixedpoint_int32,
                             &reluish_multiplier_exponent);
  tflite::DownScaleInt32ToInt16Multiplier(reluish_multiplier_fixedpoint_int32,
                                          &reluish_multiplier_fixedpoint_int16);

  // See HardSwish function in
  // tensorflow/lite/kernels/internal/reference/hardswish.h
  SmallVector<int8_t, 256> table;
  for (int32_t i = -128; i < 128; i++) {
    const int16_t input_value = i - input_zp;
    const int16_t input_value_on_hires_input_scale = input_value * (1 << 7);
    const int16_t input_value_on_preshift_output_scale =
        gemmlowp::SaturatingRoundingDoublingHighMul(
            input_value_on_hires_input_scale,
            output_multiplier_fixedpoint_int16);
    int16_t reluish_value = input_value_on_hires_input_scale;
    if (reluish_multiplier_exponent > 0) {
      reluish_value = tflite::reference_ops::SaturatingLeftShift(
          reluish_value, reluish_multiplier_exponent - 1);
    }
    reluish_value = gemmlowp::SaturatingRoundingDoublingHighMul(
        reluish_value, reluish_multiplier_fixedpoint_int16);
    if (reluish_multiplier_exponent > 0) {
      reluish_value =
          tflite::reference_ops::SaturatingLeftShift(reluish_value, 1);
    }
    if (reluish_multiplier_exponent < 0) {
      reluish_value = gemmlowp::RoundingDivideByPOT(
          reluish_value, -reluish_multiplier_exponent);
    }
    reluish_value = (reluish_value + (1 << 15)) >> 1;
    const int16_t preshift_output_value =
        tflite::reference_ops::SaturatingDoublingHighMul(
            reluish_value, input_value_on_preshift_output_scale);
    int16_t output_value = gemmlowp::RoundingDivideByPOT(
        preshift_output_value, -output_multiplier_exponent);
    output_value += output_zp;
    output_value =
        std::min<int16_t>(output_value, std::numeric_limits<int8_t>::max());
    output_value =
        std::max<int16_t>(output_value, std::numeric_limits<int8_t>::min());
    table.push_back(output_value);
  }

  auto element_qtype =
      UniformQuantizedType::get(true, rewriter.getIntegerType(8),
                                rewriter.getF32Type(), 1.0f, 0, -128, 127);
  auto const_type = tensorflow::GetTypeFromTFTensorShape({256}, element_qtype);
  auto storage_type = tensorflow::GetTypeFromTFTensorShape(
      {256}, element_qtype.getStorageType());
  auto const_attr = DenseElementsAttr::get(storage_type, llvm::ArrayRef(table));

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

Value getTosaConstRsqrt8bitTable(PatternRewriter& rewriter, Operation* op,
                                 float input_scale, int32_t input_zp,
                                 float output_scale, int32_t output_zp) {
  // See RsqrtEvalQuantized (elementwise.cc)
  const int kMin = std::numeric_limits<int8_t>::min();
  const int kMax = std::numeric_limits<int8_t>::max();
  SmallVector<int8_t, 256> table;

  int32_t output_scale_multiplier;
  int32_t output_scale_shift;

  const double scale = 1. / (std::sqrt(input_scale) * output_scale);
  tflite_migration::QuantizeMultiplier(scale, &output_scale_multiplier,
                                       &output_scale_shift);

  std::function<int8_t(int8_t)> quantRsqrtFunc = [&](int8_t i) {
    const int32_t value = (i - input_zp);
    const int32_t kShift = 20;  // Shift to keep value integer.
    if (value <= 0) {
      // Assume that any value close to 0 (or negative values) represents the max
      // output value.
      return static_cast<int8_t>(kMax);
    }
    int32_t inv_sqrt_multiplier;
    int inv_sqrt_shift;
    tflite_migration::GetInvSqrtQuantizedMultiplierExp(
        value, tflite_migration::kReverseShift, &inv_sqrt_multiplier,
        &inv_sqrt_shift);
    const int32_t data = tflite_migration::MultiplyByQuantizedMultiplier(
        1, inv_sqrt_multiplier, inv_sqrt_shift + kShift);
    const int32_t output =
        tflite_migration::MultiplyByQuantizedMultiplier(
            data, output_scale_multiplier, output_scale_shift - kShift) +
        output_zp;
    return static_cast<int8_t>(std::min(std::max(output, kMin), kMax));
  };

  for (int32_t i = -128; i < 128; i++) {
    table.push_back(quantRsqrtFunc(i));
  }

  auto element_qtype =
      UniformQuantizedType::get(true, rewriter.getIntegerType(8),
                                rewriter.getF32Type(), 1.0f, 0, -128, 127);
  auto const_type = tensorflow::GetTypeFromTFTensorShape({256}, element_qtype);
  auto storage_type = tensorflow::GetTypeFromTFTensorShape(
      {256}, element_qtype.getStorageType());
  auto const_attr = DenseElementsAttr::get(storage_type, llvm::ArrayRef(table));

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Create a 8-bit TOSA TABLE constant tensor with int8[256] array.
// Follow LUTPopulateInt8() tensorflow/lite/kernels/internal/common.h
Value getTosaConst8bitTable(PatternRewriter& rewriter, Operation* op,
                            float input_scale, int32_t input_zp,
                            float output_scale, int32_t output_zp,
                            std::function<float(float)> func) {
  SmallVector<int8_t, 256> table;

  float inverse_scale = 1.0f / output_scale;
  for (int32_t i = -128; i < 128; i++) {
    float dequantized = input_scale * (i - input_zp);
    float transformed = func(dequantized);

    float max = (output_scale > 1.0) ? FLT_MAX : (FLT_MAX * output_scale);
    if (transformed >= max) {
      table.push_back(INT8_MAX);
      continue;
    }

    int32_t rescaled = std::round(transformed * inverse_scale);
    int32_t quantized = static_cast<int32_t>(rescaled + output_zp);
    table.push_back(
        static_cast<int8_t>(std::min(std::max(quantized, -128), 127)));
  }

  auto element_qtype =
      UniformQuantizedType::get(true, rewriter.getIntegerType(8),
                                rewriter.getF32Type(), 1.0f, 0, -128, 127);
  auto const_type = tensorflow::GetTypeFromTFTensorShape({256}, element_qtype);
  auto storage_type = tensorflow::GetTypeFromTFTensorShape(
      {256}, element_qtype.getStorageType());
  auto const_attr = DenseElementsAttr::get(storage_type, llvm::ArrayRef(table));

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Create a 16-bit TOSA TABLE constant tensor.
// A float should be used by default for FloatT except if a double is required
// for backward compatibility.
// Follow LUTPopulateInt16() tensorflow/lite/kernels/internal/common.h
template <typename FloatT>
Value getTosaConst16bitTable(PatternRewriter& rewriter, Operation* op,
                             FloatT input_scale, int32_t input_zp,
                             FloatT output_scale, int32_t output_zp,
                             std::function<FloatT(FloatT)> func) {
  static_assert(std::is_floating_point<FloatT>::value,
                "FloatT must be a floating-point type.");

  SmallVector<int16_t, 513> table;

  FloatT input_min =
      input_scale * (std::numeric_limits<int16_t>::min() - input_zp);
  FloatT input_max =
      input_scale * (std::numeric_limits<int16_t>::max() - input_zp);
  FloatT output_min =
      output_scale * (std::numeric_limits<int16_t>::min() - output_zp);
  FloatT output_max =
      output_scale * (std::numeric_limits<int16_t>::max() - output_zp);

  FloatT step = (input_max - input_min) / 512;
  FloatT half_step = step / 2;
  FloatT output_scaling_inv = 65536 / (output_max - output_min);

  for (int32_t i = 0; i < 512; i++) {
    FloatT sample_val =
        std::round(func(input_min + (i * step)) * output_scaling_inv);
    FloatT midpoint_interp_val = std::round(
        ((func(input_min + (i + 1) * step) * output_scaling_inv) +
         std::round(func(input_min + (i * step)) * output_scaling_inv)) /
        2);
    FloatT midpoint_val = std::round(func(input_min + (i * step) + half_step) *
                                     output_scaling_inv);
    FloatT midpoint_err = midpoint_interp_val - midpoint_val;
    FloatT bias = std::round(midpoint_err / 2);

    table.push_back(static_cast<int16_t>(
        std::min<FloatT>(std::max<FloatT>(sample_val - bias, -32768), 32767)));
  }

  FloatT max_val = std::round(func(input_max) * output_scaling_inv);
  table.push_back(static_cast<int16_t>(
      std::min<FloatT>(std::max<FloatT>(max_val, -32768), 32767)));

  auto const_type =
      tensorflow::GetTypeFromTFTensorShape({513}, rewriter.getIntegerType(16));
  auto const_attr = DenseElementsAttr::get(const_type, llvm::ArrayRef(table));

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

template Value getTosaConst16bitTable<float>(PatternRewriter& rewriter,
                                             Operation* op, float input_scale,
                                             int32_t input_zp,
                                             float output_scale,
                                             int32_t output_zp,
                                             std::function<float(float)> func);

template Value getTosaConst16bitTable<double>(
    PatternRewriter& rewriter, Operation* op, double input_scale,
    int32_t input_zp, double output_scale, int32_t output_zp,
    std::function<double(double)> func);

// Create a 32-bit TOSA TABLE for Softmax Exp
void getTosaConst32bitSoftmaxExpTable(PatternRewriter& rewriter, Operation* op,
                                      double beta, double input_scale,
                                      Value& first_const, Value& second_const,
                                      Value& third_const, Value& fourth_const) {
  const int kScaledDiffIntegerBits = 5;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32_t, kScaledDiffIntegerBits>;

  int32_t input_beta_multiplier;
  int input_beta_left_shift;
  tflite_migration::PreprocessSoftmaxScaling(
      beta, input_scale, kScaledDiffIntegerBits, &input_beta_multiplier,
      &input_beta_left_shift);

  int diff_min = -tflite_migration::CalculateInputRadius(kScaledDiffIntegerBits,
                                                         input_beta_left_shift);

  SmallVector<int16_t, 513> first_table, second_table, third_table,
      fourth_table;
  for (int32_t input_diff = -256; input_diff <= 256; input_diff++) {
    int32_t output = 0;
    if (input_diff >= diff_min) {
      const int32_t input_diff_rescaled =
          tflite_migration::MultiplyByQuantizedMultiplierGreaterThanOne(
              input_diff, input_beta_multiplier, input_beta_left_shift);
      const FixedPointScaledDiff input_diff_fixed_point =
          FixedPointScaledDiff::FromRaw(input_diff_rescaled);
      output = gemmlowp::exp_on_negative_values(input_diff_fixed_point).raw();
    }

    // Only copy the 8-bit groups
    int32_t first = (output >> 24) & 0xFF;
    int32_t second = (output >> 16) & 0xFF;
    int32_t third = (output >> 8) & 0xFF;
    int32_t fourth = (output) & 0xFF;

    first_table.push_back(first);
    second_table.push_back(second);
    third_table.push_back(third);
    fourth_table.push_back(fourth);
  }

  auto element_qtype =
      UniformQuantizedType::get(true, rewriter.getIntegerType(16),
                                rewriter.getF32Type(), 1.0f, 0, -32768, 32767);
  auto const_type = tensorflow::GetTypeFromTFTensorShape({513}, element_qtype);
  auto storage_type = tensorflow::GetTypeFromTFTensorShape(
      {513}, element_qtype.getStorageType());

  auto first_const_attr =
      DenseElementsAttr::get(storage_type, llvm::ArrayRef(first_table));
  auto second_const_attr =
      DenseElementsAttr::get(storage_type, llvm::ArrayRef(second_table));
  auto third_const_attr =
      DenseElementsAttr::get(storage_type, llvm::ArrayRef(third_table));
  auto fourth_const_attr =
      DenseElementsAttr::get(storage_type, llvm::ArrayRef(fourth_table));

  first_const =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, first_const_attr)
          .getResult();
  second_const =
      rewriter
          .create<tosa::ConstOp>(op->getLoc(), const_type, second_const_attr)
          .getResult();
  third_const =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, third_const_attr)
          .getResult();
  fourth_const =
      rewriter
          .create<tosa::ConstOp>(op->getLoc(), const_type, fourth_const_attr)
          .getResult();
}

// Create a 32-bit float constant operator from a float
Value getTosaConstTensorSingleF32(PatternRewriter& rewriter, Operation* op,
                                  float val, int rank) {
  assert(rank >= 0);
  mlir::RankedTensorType const_type;
  mlir::DenseElementsAttr const_attr;
  auto element_type = rewriter.getF32Type();
  if (rank == 0) {
    const_type = tensorflow::GetTypeFromTFTensorShape({}, element_type);
    const_attr = DenseElementsAttr::get(const_type, val);
  } else {
    std::vector<int64_t> shape(rank, 1);
    const_type = tensorflow::GetTypeFromTFTensorShape(llvm::ArrayRef(shape),
                                                      element_type);
    std::vector<float> values(1, val);
    const_attr = DenseElementsAttr::get(const_type, llvm::ArrayRef(values));
  }

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Create a 32-bit integer constant operator from an int
Value getTosaConstTensorSingleI32(PatternRewriter& rewriter, Operation* op,
                                  int32_t val, int rank) {
  assert(rank >= 0);
  mlir::RankedTensorType const_type;
  mlir::DenseElementsAttr const_attr;
  auto element_type = rewriter.getIntegerType(32);
  if (rank == 0) {
    const_type = tensorflow::GetTypeFromTFTensorShape({}, element_type);
    const_attr = DenseElementsAttr::get(const_type, val);
  } else {
    std::vector<int64_t> shape(rank, 1);
    const_type = tensorflow::GetTypeFromTFTensorShape(llvm::ArrayRef(shape),
                                                      element_type);
    std::vector<int32_t> values(1, val);
    const_attr = DenseElementsAttr::get(const_type, llvm::ArrayRef(values));
  }

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Create an expected bitwidth integer constant operator based on the type
// parameter.
Value getTosaConstTensorScalarInt(ImplicitLocOpBuilder& builder, Type type,
                                  int64_t val, int rank) {
  assert(rank >= 0);
  assert(type.isa<IntegerType>());
  mlir::RankedTensorType const_type;
  mlir::DenseElementsAttr const_attr;
  auto bit_width = type.getIntOrFloatBitWidth();
  auto element_type = builder.getIntegerType(bit_width);
  if (rank == 0) {
    const_type = tensorflow::GetTypeFromTFTensorShape({}, element_type);
    const_attr = DenseElementsAttr::get(const_type, val);
  } else {
    std::vector<int64_t> shape(rank, 1);
    const_type = tensorflow::GetTypeFromTFTensorShape(llvm::ArrayRef(shape),
                                                      element_type);
    std::vector<int32_t> values(1, val);
    const_attr = DenseElementsAttr::get(const_type, llvm::ArrayRef(values));
  }

  auto const_op =
      builder.create<tosa::ConstOp>(builder.getLoc(), const_type, const_attr);
  return const_op.getResult();
}

Value getTosaConstShape(PatternRewriter& rewriter, Operation* op,
                        llvm::ArrayRef<int64_t> values) {
  auto attr = rewriter.getIndexTensorAttr(values);
  auto type =
      tosa::shapeType::get(rewriter.getContext(), /* rank = */ values.size());
  return CreateOpAndInfer<tosa::ConstShapeOp>(rewriter, op->getLoc(), type,
                                              attr);
}

// Create a vector from a 32-bit value tensor.  Returns the size of
// the new vector or -1 on error.
// Populate a int32_t vector from a val tensor
// return failure if val is not a constant value
// return success otherwise
LogicalResult getVectorFromValue32(Value val, SmallVectorImpl<int32_t>& vec) {
  int i = 0;

  ElementsAttr elems;

  vec.clear();

  if (!matchPattern(val, m_Constant(&elems))) return failure();

  for (auto idx : elems.getValues<IntegerAttr>()) {
    vec.push_back(idx.getInt());
    i++;
  }

  return success();
}

// Populate a int64_t vector from a val tensor
// return failure if val is not a constant value
// return success otherwise
LogicalResult getVectorFromValue64(Value val, SmallVectorImpl<int64_t>& vec) {
  int i = 0;

  ElementsAttr elems;

  vec.clear();

  if (!matchPattern(val, m_Constant(&elems))) return failure();

  for (auto idx : elems.getValues<IntegerAttr>()) {
    vec.push_back(static_cast<int64_t>(idx.getInt()));
    i++;
  }

  return success();
}

// Calculates the TOSA padding values based on TF operators padded with
// SAME/VALID.
//
// This could pass tensorflow::FilterTensorFormat and do
// GetFilterTensorSpatialDimIndex but the current TF core libs do not support
// FORMAT_OHWI parsing by that function in core/util/tensor_format.h
bool getPaddingValuesFromPadType(tensorflow::Padding tf_pad,
                                 tensorflow::TensorFormat data_format_tf,
                                 uint32_t first_filter_spatial_dim,
                                 ShapedType input_type, ShapedType filter_type,
                                 DenseI64ArrayAttr strides,
                                 DenseI64ArrayAttr dilations,
                                 PatternRewriter& rewriter,
                                 DenseI64ArrayAttr& explicit_padding) {
  assert(tf_pad != tensorflow::Padding::EXPLICIT);
  if (!input_type.hasRank() || !filter_type.getRank()) return false;
  // Only support NHWC for now.
  if (data_format_tf != tensorflow::FORMAT_NHWC) return false;

  // Storing the numeric padding values is useful for TOSA codegen, as opposed
  // to holding the padding regime mnemonic, i.e. SAME, VALID, FULL, ...
  SmallVector<int64_t> computed_paddings;

  int64_t dim_index_shift, dim_count;
  if (input_type.getRank() == 4) {
    // 4D tensor, NHWC/NCHW format.
    dim_index_shift = GetTensorSpatialDimIndex(4, data_format_tf, 0);
    dim_count = 2;
  } else {
    if (input_type.getRank() != 5) return false;
    // 5D tensor, NDHWC/NCDHW format.
    dim_index_shift = 1;
    dim_count = 3;
  }

  int64_t pad_before, pad_after;
  // Iterating the given spatial dimensions.
  for (int i = 0; i < dim_count; i++) {
    int64_t ifm_dim = i + dim_index_shift;
    int64_t filter_dim = first_filter_spatial_dim + i;

    int64_t dim_dilation = dilations[i];
    int64_t dim_stride = strides[i];

    int64_t ip_size = input_type.getDimSize(ifm_dim);
    int64_t f_size = filter_type.getDimSize(filter_dim);
    // If we have a dynamic shape we should assume it is wide enough.
    ip_size = ip_size < 0 ? f_size * dim_dilation : ip_size;
    int64_t op_size, pad_before_tf,
        pad_after_tf;  // Complains if using int64_T
    absl::Status status = tensorflow::GetWindowedOutputSizeVerbose(
        ip_size, f_size, dim_dilation, dim_stride, tf_pad, &op_size,
        &pad_before_tf, &pad_after_tf);
    if (!status.ok()) return false;

    pad_before = pad_before_tf;
    pad_after = pad_after_tf;
    computed_paddings.push_back(pad_before);
    computed_paddings.push_back(pad_after);
  }

  explicit_padding = rewriter.getDenseI64ArrayAttr(computed_paddings);
  return true;
}

// Calculates the TOSA padding values for explicit-padded TF operators.
//
// This function only handles the TF padding array explicit_padding, which is
// only present in certain TF ops. All others encode padding using the string
// SAME/VALID, which is interpreted using the getPaddingValuesFromPadString
// function below.

// The explicit padding array in TF holds 2 pad values for every
// dimension, even those that are not the 2 spatial ones. Just extract the
// 2x pad values for the XY dims.
DenseI64ArrayAttr getPaddingValuesFromExplicitPadAttr(
    ArrayAttr explicit_pad, tensorflow::TensorFormat data_format_tf,
    PatternRewriter& rewriter) {
  SmallVector<int64_t> computed_paddings;

  int64_t pad_before, pad_after;
  for (int i = 0; i < 2; i++) {  // Two spatial dimensions X&Y
    int64_t dim = GetTensorSpatialDimIndex(4, data_format_tf,
                                           i);  // 4D tensor, NHWC/NCHW format
    pad_before = mlir::cast<IntegerAttr>(explicit_pad[dim * 2]).getInt();
    pad_after = mlir::cast<IntegerAttr>(explicit_pad[dim * 2 + 1]).getInt();
    computed_paddings.push_back(pad_before);
    computed_paddings.push_back(pad_after);
  }

  return rewriter.getDenseI64ArrayAttr(computed_paddings);
}

// Calculates the TOSA padding values for transposeConv2d
bool getTransposeConv2dPaddingValues(
    tensorflow::Padding tf_pad, tensorflow::TensorFormat data_format_tf,
    uint32_t first_filter_spatial_dim, ShapedType input_type,
    ShapedType filter_type, ShapedType output_type, DenseI64ArrayAttr strides,
    PatternRewriter& rewriter, DenseI64ArrayAttr& explicit_padding) {
  assert(tf_pad != tensorflow::Padding::EXPLICIT);
  if (!input_type.hasRank() || !filter_type.hasRank() || !output_type.hasRank())
    return false;

  // Storing the numeric padding values is useful for TOSA codegen, as opposed
  // to holding the padding regime mnemonic, i.e. SAME, VALID, FULL, ...

  SmallVector<int64_t> computed_paddings;

  int64_t pad_before, pad_after;
  for (int i = 0; i < 2; i++) {  // Two spatial dimensions X&Y
    int64_t ifm_dim = GetTensorSpatialDimIndex(
        4, data_format_tf, i);  // 4D tensor, NHWC/NCHW format
    int64_t ofm_dim = GetTensorSpatialDimIndex(
        4, data_format_tf, i);  // 4D tensor, NHWC/NCHW format
    int64_t filter_dim = first_filter_spatial_dim + i;

    int64_t ifm_size = input_type.getDimSize(ifm_dim);
    int64_t filter_size = filter_type.getDimSize(filter_dim);
    int64_t ofm_size = output_type.getDimSize(ofm_dim);
    int64_t dim_stride = strides[i];

    // These dimensions need to be static to legalize.
    if (ShapedType::isDynamic(filter_size) || ShapedType::isDynamic(ifm_size) ||
        ShapedType::isDynamic(ofm_size)) {
      return false;
    }

    int total_padding =
        ((ifm_size - 1) * dim_stride + filter_size - ofm_size);

    pad_before = total_padding / 2;
    pad_after = total_padding - pad_before;

    computed_paddings.push_back(-pad_before);
    computed_paddings.push_back(-pad_after);
  }

  explicit_padding = rewriter.getDenseI64ArrayAttr(computed_paddings);
  return true;
}

// Templated function to create a constant op for given type and shape.
// T: storage C type.
// Default template creates a constant tensor in T.
template <typename T>
std::optional<Value> getConstTensor(PatternRewriter& rewriter, Operation* op,
                                    ArrayRef<T> vec, ArrayRef<int64_t> shape) {
  int64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto const_type = tensorflow::GetTypeFromTFTensorShape(
      shape, rewriter.getIntegerType(sizeof(T) * 8));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template specialization for APInt
template <>
std::optional<Value> getConstTensor<APInt>(PatternRewriter& rewriter,
                                           Operation* op, ArrayRef<APInt> vec,
                                           ArrayRef<int64_t> shape) {
  int64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto const_type = tensorflow::GetTypeFromTFTensorShape(
      shape, rewriter.getIntegerType(vec[0].getBitWidth()));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template specialization for float
template <>
std::optional<Value> getConstTensor<float>(PatternRewriter& rewriter,
                                           Operation* op, ArrayRef<float> vec,
                                           ArrayRef<int64_t> shape) {
  int64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto const_type =
      tensorflow::GetTypeFromTFTensorShape(shape, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template instantiation
template std::optional<Value> getConstTensor<int32_t>(PatternRewriter&,
                                                      Operation*,
                                                      ArrayRef<int32_t> vec,
                                                      ArrayRef<int64_t> shape);

llvm::SmallVector<int64_t> getOutputSpatialSizeRemainder(
    tensorflow::TensorFormat data_format_tf, ShapedType input_type,
    DenseI64ArrayAttr kernel_size, DenseI64ArrayAttr pads,
    DenseI64ArrayAttr strides, DenseI64ArrayAttr dilations) {
  llvm::SmallVector<int64_t> output_size_remainder;

  const int nb_spatial_dims =
      GetTensorSpatialDims(input_type.getRank(), data_format_tf);
  for (int spatial_dim = 0; spatial_dim < nb_spatial_dims; spatial_dim++) {
    const int64_t in_size = input_type.getDimSize(GetTensorSpatialDimIndex(
        input_type.getRank(), data_format_tf, spatial_dim));
    const int64_t full_pad =
        pads[2 * spatial_dim + 0] + pads[2 * spatial_dim + 1];

    const int64_t full_size =
        in_size - 1 + full_pad -
        (kernel_size[spatial_dim] - 1) * dilations[spatial_dim];
    output_size_remainder.push_back(full_size % strides[spatial_dim]);
  }

  return output_size_remainder;
}

Value getInputSlicedToItsUsedSize(PatternRewriter& rewriter, Operation* op,
                                  tensorflow::TensorFormat data_format_tf,
                                  ShapedType input_type, Value input_val,
                                  DenseI64ArrayAttr kernel_size,
                                  DenseI64ArrayAttr pads,
                                  DenseI64ArrayAttr strides,
                                  DenseI64ArrayAttr dilations) {
  const int nb_spatial_dims =
      GetTensorSpatialDims(input_type.getRank(), data_format_tf);
  // Don't slice the input if any spatial dimension is dynamic
  for (int spatial_dim = 0; spatial_dim < nb_spatial_dims; spatial_dim++) {
    if (input_type.isDynamicDim(GetTensorSpatialDimIndex(
            input_type.getRank(), data_format_tf, spatial_dim))) {
      return input_val;
    }
  }

  const llvm::SmallVector<int64_t> output_size_remainder =
      getOutputSpatialSizeRemainder(data_format_tf, input_type, kernel_size,
                                    pads, strides, dilations);

  const bool need_slicing =
      llvm::any_of(output_size_remainder, [](int64_t v) { return v > 0; });
  const bool zero_pads =
      llvm::all_of(pads.asArrayRef(), [](int64_t v) { return v == 0; });
  if (need_slicing && zero_pads) {
    llvm::SmallVector<int64_t> start(input_type.getRank(), 0);
    llvm::SmallVector<int64_t> size =
        tensorflow::ConvertMlirShapeToTF(input_type.getShape());
    for (int spatial_dim = 0; spatial_dim < nb_spatial_dims; spatial_dim++) {
      const int index = GetTensorSpatialDimIndex(input_type.getRank(),
                                                 data_format_tf, spatial_dim);
      size[index] -= output_size_remainder[spatial_dim];
    }

    auto slice_op = CreateOpAndInfer<tosa::SliceOp>(
        rewriter, op->getLoc(),
        UnrankedTensorType::get(input_type.getElementType()), input_val,
        getTosaConstShape(rewriter, op->getLoc(), start),
        getTosaConstShape(rewriter, op->getLoc(), size));
    return slice_op.getResult();
  }

  return input_val;
}

// Check if scale32 mode is used for given output_element_type
bool isScale32(mlir::quant::UniformQuantizedType output_element_type) {
  return (output_element_type.getStorageTypeIntegralWidth() == 8);
}

LogicalResult ApplyPatternsWithShapeResolution(
    func::FuncOp func, const FrozenRewritePatternSet& patterns) {
  // We use top-down traversal so that shape inference can fully infer types
  // during pattern rewrite.
  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  if (failed(applyPatternsGreedily(func, patterns, config))) {
    return failure();
  }

  // Check that constant attributes types and op types match up. If the lowering
  // needs to change a type (e.g. fp16 -> fp32) its possible the return type
  // could be incorrect.
  //
  // This should be investigate for whether it is still necessary due to quant
  // type stripping changing.
  func.walk([&](tosa::ConstOp op) {
    if (mlir::isa<QuantizedType>(op.getType().getElementType())) {
      return;
    }
    auto ety = op.getValues().getShapedType().getElementType();
    auto new_ty = mlir::cast<TensorType>(op.getType()).clone(ety);
    op.getResult().setType(new_ty);
  });

  auto returnOp = cast<func::ReturnOp>(func.getBody().front().getTerminator());
  llvm::SmallVector<Type> result_tys(returnOp.getOperandTypes());

  func.setType(FunctionType::get(
      func.getContext(), func.getFunctionType().getInputs(), result_tys));

  return success();
}

void TrimQuantizedIntegerRangeMin(UniformQuantizedType dtype,
                                  int64_t& val_min) {
  val_min =
      val_min < dtype.getStorageTypeMin() ? dtype.getStorageTypeMin() : val_min;
}

void TrimQuantizedIntegerRangeMax(UniformQuantizedType dtype,
                                  int64_t& val_max) {
  val_max =
      val_max > dtype.getStorageTypeMax() ? dtype.getStorageTypeMax() : val_max;
}

void TrimQuantizedIntegerRange(UniformQuantizedType dtype, int64_t& val_min,
                               int64_t& val_max) {
  TrimQuantizedIntegerRangeMin(dtype, val_min);
  TrimQuantizedIntegerRangeMax(dtype, val_max);
}

tosa::MulOp CreateMulOpAndInfer(PatternRewriter& rewriter, Operation* op,
                                Type result_ty, Value input1, Value input2,
                                int8_t shift) {
  if (EqualizeRanks(rewriter, op->getLoc(), input1, input2).failed()) {
    // uncompatible broadcast shapes, no reshape is inserted
    // ResultsBroadcastableShape verify will handle this
  }
  return CreateOpAndInfer<tosa::MulOp>(rewriter, op->getLoc(), result_ty, input1,
      input2, getConstTensor<int8_t>(rewriter, op, shift, {1}).value());
}

Value reshapeScalarTo1D(PatternRewriter& rewriter, Location loc, Value value) {
  ShapedType type = dyn_cast<ShapedType>(value.getType());
  if (!type || !type.hasRank()) {
    return nullptr;
  }
  if (type.getRank() == 1) {
    return value;
  }
  if (type.getRank() != 0) {
    return nullptr;
  }

  auto element_type = type.getElementType();
  auto output_type = tensorflow::GetTypeFromTFTensorShape({1}, element_type);

  // for rank-0 constant value, return a rank-1 constant value
  mlir::DenseElementsAttr attr;
  if (matchPattern(value, m_Constant(&attr))) {
    auto storage_type = tensorflow::GetTypeFromTFTensorShape({1}, element_type);
    auto element_qtype = dyn_cast<quant::QuantizedType>(element_type);
    if (element_qtype) {
      storage_type = tensorflow::GetTypeFromTFTensorShape(
          {1}, element_qtype.getStorageType());
    }

    DenseElementsAttr const_attr;
    if (mlir::isa<mlir::IntegerType>(attr.getElementType())) {
      const_attr = DenseElementsAttr::get(storage_type,
                                          {attr.getValues<mlir::APInt>()[0]});
    } else if (mlir::isa<mlir::FloatType>(attr.getElementType())) {
      const_attr = DenseElementsAttr::get(storage_type,
                                          {attr.getValues<mlir::APFloat>()[0]});
    } else {
      // unexpected attribute element type
      return nullptr;
    }

    auto const_op =
        rewriter.create<tosa::ConstOp>(loc, output_type, const_attr);
    return const_op.getResult();
  }

  // reshape rank0 value to rank1
  auto rank1_scalar_shape = getTosaConstShape(rewriter, loc, {1});
  Value rank1_value = CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, loc, output_type, value, rank1_scalar_shape);
  return rank1_value;
}

LogicalResult broadcastLowRankTensor(PatternRewriter& rewriter, Operation* op,
                                     Value& input1, Value& input2) {
  auto input1_ty = llvm::dyn_cast<RankedTensorType>(input1.getType());
  auto input2_ty = llvm::dyn_cast<RankedTensorType>(input2.getType());

  if (!input1_ty || !input2_ty) {
    return rewriter.notifyMatchFailure(op,
                                       "input tensors should be all ranked");
  }

  int64_t input1_rank = input1_ty.getRank();
  int64_t input2_rank = input2_ty.getRank();

  if (input1_rank == input2_rank) return success();

  Value high_rank_tensor, low_rank_tensor;
  int64_t high_rank, low_rank;
  if (input1_rank > input2_rank) {
    high_rank_tensor = input1;
    low_rank_tensor = input2;
    high_rank = input1_rank;
    low_rank = input2_rank;
  } else {
    high_rank_tensor = input2;
    low_rank_tensor = input1;
    high_rank = input2_rank;
    low_rank = input1_rank;
  }

  ArrayRef<int64_t> high_rank_shape =
      llvm::cast<RankedTensorType>(high_rank_tensor.getType()).getShape();
  ArrayRef<int64_t> low_rank_shape =
      llvm::cast<RankedTensorType>(low_rank_tensor.getType()).getShape();

  // broadcasting if the first dimension of the low-rank tensor is '1'
  // example hight_rank: [1,a,b,c]; low_rank: [1,d,e]; low_rank should broadcast
  // to [1, a, d, e]
  if (low_rank_shape[0] == 1) {
    low_rank -= 1;
  }
  int64_t broadcast_rank = high_rank - low_rank;
  SmallVector<int64_t> broadcast_shape(high_rank, 1);

  for (int i = 0; i < broadcast_rank; i++) {
    broadcast_shape[i] = high_rank_shape[i];
  }

  auto broadcast_shape_value =
      getTosaConstShape(rewriter, op->getLoc(),
                        tensorflow::ConvertMlirShapeToTF(broadcast_shape));

  std::optional<Value> result = convertBroadcastToOp(
      rewriter, op, low_rank_tensor, broadcast_shape_value);
  if (!result) return failure();

  low_rank_tensor = result.value();

  if (input1_rank < input2_rank) {
    input1 = low_rank_tensor;
    input2 = high_rank_tensor;
  } else {
    input1 = high_rank_tensor;
    input2 = low_rank_tensor;
  }
  return success();
}

bool checkUniqueConstantScatterIndices(ShapedType indices_type,
                                       ShapedType result_type,
                                       ElementsAttr const_data) {
  llvm::ArrayRef<int64_t> const indices_shape = indices_type.getShape();
  const unsigned int indices_rank = indices_shape.size();
  const unsigned int result_rank = result_type.getRank();
  const unsigned int last_dim_size = indices_shape[indices_rank - 1];

  // Reconstruct each index from the unshaped constant data array and
  // calculate the corresponding flattened index
  auto const const_data_range = const_data.getValues<int32_t>();
  assert((const_data_range.size() % last_dim_size == 0) &&
         "Constant data length should be a multiple of indices_shape[-1]");

  std::vector<int64_t> flattened_indices;
  flattened_indices.reserve(const_data_range.size() / last_dim_size);
  for (auto beg = const_data_range.begin(); beg < const_data_range.end();
       beg += last_dim_size) {
    std::vector<uint64_t> current_single_index(result_rank);
    std::copy(beg, beg + last_dim_size, current_single_index.begin());
    const uint64_t f_index{
        ElementsAttr::getFlattenedIndex(result_type, current_single_index)};
    flattened_indices.push_back(f_index);
  }

  // If adjacent flattened values are found, there are non-unique indices
  std::sort(flattened_indices.begin(), flattened_indices.end());
  return std::adjacent_find(flattened_indices.begin(),
                            flattened_indices.end()) == flattened_indices.end();
}

}  // namespace tosa
}  // namespace mlir
