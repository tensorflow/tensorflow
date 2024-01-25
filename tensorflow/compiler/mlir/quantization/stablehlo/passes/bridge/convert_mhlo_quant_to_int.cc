/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include "absl/algorithm/container.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/xla_legalize_targets.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/rewriters.h"

namespace mlir::quant::stablehlo {
namespace {

// TODO: b/311218165 - consider extract this to common utils and better ways to
// handle polymorphism.
using QuantType =
    std::variant<UniformQuantizedType, UniformQuantizedPerAxisType>;
FailureOr<QuantType> GetQuantType(Type type) {
  if (auto quant_type =
          getElementTypeOrSelf(type).dyn_cast<UniformQuantizedType>()) {
    return QuantType(quant_type);
  } else if (auto quant_type = getElementTypeOrSelf(type)
                                   .dyn_cast<UniformQuantizedPerAxisType>()) {
    return QuantType(quant_type);
  } else {
    return failure();
  }
}

bool IsPerTensorType(QuantType quant_type) {
  return std::holds_alternative<UniformQuantizedType>(quant_type);
}

bool IsPerChannelType(QuantType quant_type) {
  return std::holds_alternative<UniformQuantizedPerAxisType>(quant_type);
}

UniformQuantizedType GetPerTensorType(QuantType quant_type) {
  return std::get<UniformQuantizedType>(quant_type);
}

UniformQuantizedPerAxisType GetPerChannelType(QuantType quant_type) {
  return std::get<UniformQuantizedPerAxisType>(quant_type);
}

// Extract scale and zero point info from input quant type info.
void GetQuantizationParams(OpBuilder &builder, Location loc,
                           QuantType quant_type, Value &scales,
                           Value &zero_points, bool output_zero_point_in_fp,
                           DenseIntElementsAttr &broadcast_dims) {
  // Get scales/zero points for per-tensor and per-axis quantization cases.
  if (auto *quant_per_tensor_type =
          std::get_if<UniformQuantizedType>(&quant_type)) {
    scales = builder.create<mhlo::ConstantOp>(
        loc, builder.getF32FloatAttr(quant_per_tensor_type->getScale()));
    if (output_zero_point_in_fp) {
      zero_points = builder.create<mhlo::ConstantOp>(
          loc, builder.getF32FloatAttr(
                   static_cast<float>(quant_per_tensor_type->getZeroPoint())));
    } else {
      zero_points = builder.create<mhlo::ConstantOp>(
          loc, builder.getI32IntegerAttr(static_cast<int32_t>(
                   quant_per_tensor_type->getZeroPoint())));
    }
  } else {
    auto &quant_per_channel_type =
        std::get<UniformQuantizedPerAxisType>(quant_type);
    SmallVector<float> scales_vec;
    for (auto scale : quant_per_channel_type.getScales())
      scales_vec.push_back(scale);
    scales = builder.create<mhlo::ConstantOp>(
        loc, DenseFPElementsAttr::get(
                 RankedTensorType::get(
                     {static_cast<int64_t>(
                         quant_per_channel_type.getScales().size())},
                     builder.getF32Type()),
                 scales_vec));
    if (output_zero_point_in_fp) {
      SmallVector<float> zero_points_vec;
      for (auto zero_point : quant_per_channel_type.getZeroPoints())
        zero_points_vec.push_back(zero_point);
      zero_points = builder.create<mhlo::ConstantOp>(
          loc, DenseFPElementsAttr::get(
                   RankedTensorType::get(
                       {static_cast<int64_t>(
                           quant_per_channel_type.getZeroPoints().size())},
                       builder.getF32Type()),
                   zero_points_vec));
    } else {
      SmallVector<int32_t> zero_points_vec;
      for (auto zero_point : quant_per_channel_type.getZeroPoints())
        zero_points_vec.push_back(zero_point);
      zero_points = builder.create<mhlo::ConstantOp>(
          loc, DenseIntElementsAttr::get(
                   RankedTensorType::get(
                       {static_cast<int64_t>(
                           quant_per_channel_type.getZeroPoints().size())},
                       builder.getI32Type()),
                   zero_points_vec));
    }
    broadcast_dims = DenseIntElementsAttr::get(
        RankedTensorType::get({1}, builder.getI64Type()),
        {static_cast<int64_t>(quant_per_channel_type.getQuantizedDimension())});
  }
}

// Extract storage min/max from input quant type info.
void GetQuantizationStorageInfo(OpBuilder &builder, Location loc,
                                QuantType quant_type, Value &storage_min,
                                Value &storage_max) {
  if (auto *quant_per_tensor_type =
          std::get_if<UniformQuantizedType>(&quant_type)) {
    storage_min = builder.create<mhlo::ConstantOp>(
        loc, builder.getF32FloatAttr(static_cast<float>(
                 quant_per_tensor_type->getStorageTypeMin())));
    storage_max = builder.create<mhlo::ConstantOp>(
        loc, builder.getF32FloatAttr(static_cast<float>(
                 quant_per_tensor_type->getStorageTypeMax())));
  } else {
    auto &quant_per_channel_type =
        std::get<UniformQuantizedPerAxisType>(quant_type);
    storage_min = builder.create<mhlo::ConstantOp>(
        loc, builder.getF32FloatAttr(static_cast<float>(
                 quant_per_channel_type.getStorageTypeMin())));
    storage_max = builder.create<mhlo::ConstantOp>(
        loc, builder.getF32FloatAttr(static_cast<float>(
                 quant_per_channel_type.getStorageTypeMax())));
  }
}

// Get storage type of a UQ type. Return original type if it is no UQ type.
Type GetQuantStorageType(Type type) {
  if (auto shaped = type.dyn_cast<ShapedType>()) {
    return shaped.clone(GetQuantStorageType(shaped.getElementType()));
  }

  if (auto element_type =
          getElementTypeOrSelf(type).dyn_cast<UniformQuantizedType>()) {
    return element_type.getStorageType();
  } else if (auto element_type = getElementTypeOrSelf(type)
                                     .dyn_cast<UniformQuantizedPerAxisType>()) {
    return element_type.getStorageType();
  } else {
    return type;
  }
}

Type GetQuantStorageType(QuantType type) {
  if (IsPerTensorType(type)) {
    return GetPerTensorType(type).getStorageType();
  } else {
    return GetPerChannelType(type).getStorageType();
  }
}

Value ApplyMergedScalesAndZps(OpBuilder &builder, Location loc,
                              QuantType input_quant_type,
                              QuantType output_quant_type,
                              Value input_float_tensor) {
  // Use single merged scale and merged zp if both input and output are
  // per-tensor quantized. Otherwise use a vector.
  if (IsPerTensorType(input_quant_type) && IsPerTensorType(output_quant_type)) {
    UniformQuantizedType input_per_tensor_tyep =
        GetPerTensorType(input_quant_type);
    UniformQuantizedType output_per_tensor_tyep =
        GetPerTensorType(output_quant_type);
    double merged_scale_fp =
        input_per_tensor_tyep.getScale() / output_per_tensor_tyep.getScale();
    auto merged_scale = builder.create<mhlo::ConstantOp>(
        loc, builder.getF32FloatAttr(static_cast<float>(merged_scale_fp)));
    input_float_tensor = builder.create<chlo::BroadcastMulOp>(
        loc, input_float_tensor, merged_scale,
        /*broadcast_dimensions=*/nullptr);
    // Add merged_zp only when it is non-zero.
    double merged_zp_fp =
        output_per_tensor_tyep.getZeroPoint() -
        input_per_tensor_tyep.getZeroPoint() * merged_scale_fp;
    if (merged_zp_fp != 0) {
      Value merged_zp = builder.create<mhlo::ConstantOp>(
          loc, builder.getF32FloatAttr(static_cast<float>(merged_zp_fp)));
      input_float_tensor = builder.create<chlo::BroadcastAddOp>(
          loc, input_float_tensor, merged_zp, /*broadcast_dimensions=*/nullptr);
    }
  } else {
    int64_t channel_size =
        IsPerChannelType(output_quant_type)
            ? GetPerChannelType(output_quant_type).getScales().size()
            : GetPerChannelType(input_quant_type).getScales().size();
    int64_t quantized_dimension =
        IsPerChannelType(output_quant_type)
            ? GetPerChannelType(output_quant_type).getQuantizedDimension()
            : GetPerChannelType(input_quant_type).getQuantizedDimension();
    SmallVector<double> merged_scale_double, merged_zp_double;
    merged_scale_double.resize(channel_size);
    merged_zp_double.resize(channel_size);
    for (int i = 0; i < channel_size; ++i) {
      merged_scale_double[i] =
          (IsPerChannelType(input_quant_type)
               ? GetPerChannelType(input_quant_type).getScales()[i]
               : GetPerTensorType(input_quant_type).getScale()) /
          (IsPerChannelType(output_quant_type)
               ? GetPerChannelType(output_quant_type).getScales()[i]
               : GetPerTensorType(output_quant_type).getScale());
      merged_zp_double[i] =
          (IsPerChannelType(output_quant_type)
               ? GetPerChannelType(output_quant_type).getZeroPoints()[i]
               : GetPerTensorType(output_quant_type).getZeroPoint()) -
          (IsPerChannelType(input_quant_type)
               ? GetPerChannelType(input_quant_type).getZeroPoints()[i]
               : GetPerTensorType(input_quant_type).getZeroPoint()) *
              merged_scale_double[i];
    }
    SmallVector<float> merged_scale_float(merged_scale_double.begin(),
                                          merged_scale_double.end()),
        merged_zp_float(merged_zp_double.begin(), merged_zp_double.end());

    auto broadcast_dims = DenseIntElementsAttr::get(
        RankedTensorType::get({1}, builder.getI64Type()),
        {quantized_dimension});
    Value merged_scale = builder.create<mhlo::ConstantOp>(
        loc, DenseFPElementsAttr::get(
                 RankedTensorType::get({channel_size}, builder.getF32Type()),
                 merged_scale_float));
    input_float_tensor = builder.create<chlo::BroadcastMulOp>(
        loc, input_float_tensor, merged_scale, broadcast_dims);
    if (llvm::any_of(merged_zp_float, [](double zp) { return zp != 0; })) {
      Value merged_zp = builder.create<mhlo::ConstantOp>(
          loc, DenseFPElementsAttr::get(
                   RankedTensorType::get({channel_size}, builder.getF32Type()),
                   merged_zp_float));
      input_float_tensor = builder.create<chlo::BroadcastAddOp>(
          loc, input_float_tensor, merged_zp, broadcast_dims);
    }
  }
  return input_float_tensor;
}

// This helper function create ops to requantize `input` tensor and returns the
// output tensor. Clamping is done if output integer bit-width < i32. It assumes
// that if both input and output tensor are per-channel quantized, they have the
// same quantization axis.
//
// Requantization is essentially dequantize --> quantize.
//
// Dequantize: (input - zp) * scale
// Quantize: input / scale + zp
//
// Hence,
//   output = (input - input_zp) * input_scale / output_scale + output_zp
//
// This is simplified as:
//   output = input * merged_scale + merged_zp
// where:
//   merged_zp = output_zp - input_zp * merged_scale.
//   merged_scale = input_scale / output_scale.
Value Requantize(mlir::OpState op, Value input, QuantType input_quant_type,
                 QuantType output_quant_type, TensorType output_tensor_type,
                 ConversionPatternRewriter &rewriter) {
  // Skip requantization when input and result have the same type.
  if (input_quant_type == output_quant_type) {
    return rewriter.create<mhlo::ConvertOp>(op->getLoc(), output_tensor_type,
                                            input);
  }

  auto float_tensor_type = output_tensor_type.clone(rewriter.getF32Type());
  Value output_float =
      rewriter.create<mhlo::ConvertOp>(op->getLoc(), float_tensor_type, input);

  output_float =
      ApplyMergedScalesAndZps(rewriter, op->getLoc(), input_quant_type,
                              output_quant_type, output_float);

  // Clamp output if the output integer bit-width <32.
  if (output_tensor_type.getElementType().cast<IntegerType>().getWidth() < 32) {
    Value quantization_min, quantization_max;
    GetQuantizationStorageInfo(rewriter, op->getLoc(), output_quant_type,
                               quantization_min, quantization_max);
    // Clamp results by [quantization_min, quantization_max].
    output_float = rewriter.create<mhlo::ClampOp>(
        op->getLoc(), quantization_min, output_float, quantization_max);
  }

  output_float = rewriter.create<mhlo::RoundNearestEvenOp>(
      op->getLoc(), float_tensor_type, output_float);
  return rewriter.create<mhlo::ConvertOp>(op->getLoc(), output_tensor_type,
                                          output_float);
}

class ConvertUniformQuantizeOp
    : public OpConversionPattern<mhlo::UniformQuantizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::UniformQuantizeOp op, mhlo::UniformQuantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto input_element_type = getElementTypeOrSelf(op.getOperand().getType());
    if (input_element_type.isF32()) {
      auto quant_type = GetQuantType(op.getResult().getType());
      if (succeeded(quant_type)) {
        return matchAndRewriteQuantize(op, adaptor, rewriter, *quant_type);
      }
    } else if (input_element_type.isa<quant::UniformQuantizedType,
                                      quant::UniformQuantizedPerAxisType>()) {
      auto input_quant_type = GetQuantType(input_element_type);
      auto output_quant_type = GetQuantType(op.getResult().getType());
      if (succeeded(input_quant_type) && succeeded(output_quant_type)) {
        if (IsPerChannelType(*input_quant_type) &&
            IsPerChannelType(*output_quant_type) &&
            GetPerChannelType(*input_quant_type).getQuantizedDimension() !=
                GetPerChannelType(*output_quant_type).getQuantizedDimension()) {
          op->emitError("Cannot requantize while changing quantization_axis");
          return failure();
        }
        return matchAndRewriteRequantize(op, adaptor, rewriter,
                                         *input_quant_type, *output_quant_type);
      }
    }
    op->emitError("Unsupported input element type.");
    return failure();
  }

  LogicalResult matchAndRewriteQuantize(mhlo::UniformQuantizeOp op,
                                        mhlo::UniformQuantizeOpAdaptor adaptor,
                                        ConversionPatternRewriter &rewriter,
                                        QuantType quant_type) const {
    Value scales, zero_points;
    DenseIntElementsAttr broadcast_dims;
    GetQuantizationParams(rewriter, op->getLoc(), quant_type, scales,
                          zero_points, /*output_zero_point_in_fp=*/true,
                          broadcast_dims);

    Value quantization_min, quantization_max;
    GetQuantizationStorageInfo(rewriter, op->getLoc(), quant_type,
                               quantization_min, quantization_max);

    auto res_float_tensor_type =
        op.getOperand().getType().clone(rewriter.getF32Type());
    Value res_float = rewriter.create<chlo::BroadcastDivOp>(
        op->getLoc(), res_float_tensor_type, adaptor.getOperand(), scales,
        broadcast_dims);
    res_float = rewriter.create<chlo::BroadcastAddOp>(
        op->getLoc(), res_float_tensor_type, res_float, zero_points,
        broadcast_dims);

    res_float = rewriter.create<mhlo::ClampOp>(
        op->getLoc(), res_float_tensor_type, quantization_min, res_float,
        quantization_max);
    res_float = rewriter.create<mhlo::RoundNearestEvenOp>(
        op->getLoc(), res_float_tensor_type, res_float);
    auto res_final_tensor_type = res_float_tensor_type.clone(
        GetQuantStorageType(op.getResult().getType().getElementType()));
    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(op, res_final_tensor_type,
                                                 res_float);
    return success();
  }

  LogicalResult matchAndRewriteRequantize(
      mhlo::UniformQuantizeOp op, mhlo::UniformQuantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter, QuantType input_quant_type,
      QuantType output_quant_type) const {
    rewriter.replaceOp(
        op, Requantize(op, adaptor.getOperand(), input_quant_type,
                       output_quant_type,
                       /*output_tensor_type=*/
                       op.getResult().getType().cast<TensorType>().clone(
                           GetQuantStorageType(output_quant_type)),
                       rewriter));
    return success();
  }
};

class ConvertUniformDequantizeOp
    : public OpConversionPattern<mhlo::UniformDequantizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::UniformDequantizeOp op, mhlo::UniformDequantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto quant_type = GetQuantType(op.getOperand().getType());
    if (failed(quant_type)) {
      return failure();
    }
    Value scales, zero_points;
    DenseIntElementsAttr broadcast_dims;
    GetQuantizationParams(rewriter, op->getLoc(), *quant_type, scales,
                          zero_points,
                          /*output_zero_point_in_fp=*/false, broadcast_dims);

    Value input = adaptor.getOperand();
    // TODO: b/260280919 - Consider avoiding conversion to int32.
    auto res_int32_tensor_type =
        input.getType().cast<TensorType>().clone(rewriter.getI32Type());
    Value res_int32 = rewriter.create<mhlo::ConvertOp>(
        op->getLoc(), res_int32_tensor_type, input);
    res_int32 = rewriter.create<chlo::BroadcastSubOp>(
        op->getLoc(), res_int32_tensor_type, res_int32, zero_points,
        broadcast_dims);
    auto res_float_tensor_type =
        res_int32.getType().cast<TensorType>().clone(rewriter.getF32Type());
    Value res_float = rewriter.create<mhlo::ConvertOp>(
        op->getLoc(), res_float_tensor_type, res_int32);
    res_float = rewriter.replaceOpWithNewOp<chlo::BroadcastMulOp>(
        op, res_float_tensor_type, res_float, scales, broadcast_dims);
    return success();
  }
};

class ConvertUniformQuantizedAddOp : public OpConversionPattern<mhlo::AddOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::AddOp op, mhlo::AddOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto lhs_quant_type =
        GetQuantType(getElementTypeOrSelf(op.getLhs().getType()));
    auto rhs_quant_type =
        GetQuantType(getElementTypeOrSelf(op.getRhs().getType()));
    auto res_quant_type =
        GetQuantType(getElementTypeOrSelf(op.getResult().getType()));

    // We only handle cases where lhs, rhs and results all have quantized
    // element type.
    if (failed(lhs_quant_type) || IsPerChannelType(*lhs_quant_type) ||
        failed(rhs_quant_type) || IsPerChannelType(*rhs_quant_type) ||
        failed(res_quant_type) || IsPerChannelType(*res_quant_type)) {
      op->emitError(
          "AddOp requires the same quantized element type for all operands and "
          "results");
      return failure();
    }

    // TODO: b/260280919 - Consider avoiding conversion to int32.
    auto res_int32_tensor_type =
        op.getResult().getType().clone(rewriter.getI32Type());

    // When lhs, rhs and result have different scale and zps, requantize them to
    // be the same as the result.
    // TODO: b/260280919 - Consider avoiding conversion to int32.
    Value lhs = adaptor.getLhs();
    Value lhs_int32_tensor =
        Requantize(op, lhs, *lhs_quant_type, *res_quant_type,
                   res_int32_tensor_type, rewriter);

    Value rhs = adaptor.getRhs();
    Value rhs_int32_tensor =
        Requantize(op, rhs, *rhs_quant_type, *res_quant_type,
                   res_int32_tensor_type, rewriter);

    Value zero_point = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                          GetPerTensorType(*res_quant_type).getZeroPoint())));

    // Now the lhs and rhs have been coverted to the same scale and zps.
    // Given:
    // lhs_fp = (lhs_quant - zp) * scale
    // rhs_fp = (rhs_quant - zp) * scale
    // res_fp = lhs_fp + rhs_fp
    //        = ((lhs_quant + rhs_quant - zp) - zp) * scale
    // res_quant = res_fp / scale + zp
    //           = lhs_quant + rhs_quant - zp
    // The following add the inputs and then substract by zero point.
    Value add_result = rewriter.create<chlo::BroadcastAddOp>(
        op->getLoc(), res_int32_tensor_type, lhs_int32_tensor, rhs_int32_tensor,
        nullptr);
    Value res_int32 = rewriter.create<chlo::BroadcastSubOp>(
        op->getLoc(), res_int32_tensor_type, add_result, zero_point, nullptr);

    if (GetQuantStorageType(*res_quant_type).isInteger(32)) {
      // For i32, clamping is not needed.
      rewriter.replaceOp(op, res_int32);
    } else {
      // Clamp results by [quantization_min, quantization_max] when storage type
      // is not i32.
      Value result_quantization_min = rewriter.create<mhlo::ConstantOp>(
          op->getLoc(),
          rewriter.getI32IntegerAttr(static_cast<int32_t>(
              GetPerTensorType(*res_quant_type).getStorageTypeMin())));
      Value result_quantization_max = rewriter.create<mhlo::ConstantOp>(
          op->getLoc(),
          rewriter.getI32IntegerAttr(static_cast<int32_t>(
              GetPerTensorType(*res_quant_type).getStorageTypeMax())));
      res_int32 = rewriter.create<mhlo::ClampOp>(
          op->getLoc(), res_int32_tensor_type, result_quantization_min,
          res_int32, result_quantization_max);
      // Convert results back to result storage type.
      auto res_final_tensor_type =
          res_int32_tensor_type.clone(GetQuantStorageType(*res_quant_type));
      rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(op, res_final_tensor_type,
                                                   res_int32);
    }

    return success();
  }
};

// This is a convenient struct for holding dimension numbers for dot-like ops
// including DotGeneral and Convolution. So that we can share code for all
// dot-like ops.
// For Convolution, only NHWC format is supported.
// For DotGeneral, there is no contracting dims. The batching and contracting
// dimensions are defined in
// https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dot_general.
struct DotLikeDimensionNumbers {
  SmallVector<int64_t> lhs_batching_dims;
  SmallVector<int64_t> lhs_spatial_dims;
  SmallVector<int64_t> lhs_contracting_dims;
  SmallVector<int64_t> rhs_batching_dims;
  SmallVector<int64_t> rhs_spatial_dims;
  SmallVector<int64_t> rhs_contracting_dims;
};

// A shared matchAndRewrite implementation for dot-like hybrid quantized
// operators. Hybrid ops are currently only interpreted as weight-only
// quantization ops, this might change in the future.
//
// All attrs of the original op are preserved after the conversion.
template <typename OpType, typename OpAdaptorType>
LogicalResult matchAndRewriteDotLikeHybridOp(
    OpType &op, OpAdaptorType &adaptor, ConversionPatternRewriter &rewriter) {
  // For dot like hybrid ops, lhs is float type, rhs is uniform
  // quantized type and result is float type.
  // For weight-only quantization:
  // result = hybridOp(lhs, dequant(rhs))
  Value lhs_float32_tensor = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  UniformQuantizedType rhs_element_type =
      getElementTypeOrSelf(op.getRhs().getType())
          .template cast<UniformQuantizedType>();
  auto res_float32_tensor_type =
      op.getResult().getType().template cast<TensorType>();
  auto rhs_float32_tensor_type =
      op.getRhs().getType().template cast<TensorType>().clone(
          rewriter.getF32Type());

  // Get scales and zero points for rhs.
  Value rhs_zero_point = rewriter.create<mhlo::ConstantOp>(
      op->getLoc(),
      rewriter.getF32FloatAttr((rhs_element_type.getZeroPoint())));
  Value rhs_scale_constant = rewriter.create<mhlo::ConstantOp>(
      op->getLoc(), rewriter.getF32FloatAttr(
                        static_cast<float>(rhs_element_type.getScale())));

  // Dequantize rhs_float32_tensor.
  Value rhs_float32_tensor = rewriter.create<mhlo::ConvertOp>(
      op->getLoc(), rhs_float32_tensor_type, rhs);
  rhs_float32_tensor = rewriter.create<chlo::BroadcastSubOp>(
      op->getLoc(), rhs_float32_tensor_type, rhs_float32_tensor, rhs_zero_point,
      nullptr);
  rhs_float32_tensor = rewriter.create<chlo::BroadcastMulOp>(
      op->getLoc(), rhs_float32_tensor_type, rhs_float32_tensor,
      rhs_scale_constant, nullptr);

  // Execute conversion target op.
  SmallVector<Value, 2> operands{lhs_float32_tensor, rhs_float32_tensor};
  rewriter.replaceOpWithNewOp<OpType>(op, res_float32_tensor_type, operands,
                                      op->getAttrs());
  return success();
}

Value CreateZeroPointPartialOffset(OpBuilder &builder, Location loc,
                                   Value tensor, const int64_t other_tensor_zp,
                                   SmallVector<int64_t> reduction_dims) {
  // This function calculates part of the zero-point-offset by using
  // mhlo::Reduce to sum over the contracting dims of the tensor, and then
  // multiply by zp of the other tensor.
  auto output_element_type = builder.getI32Type();

  // Calculate the output tensor shape. This is input tensor dims minus
  // contracting dims.
  auto ranked_tensor = tensor.getType().cast<RankedTensorType>();
  SmallVector<int64_t> output_dims;
  for (int64_t i = 0; i < ranked_tensor.getRank(); ++i) {
    if (absl::c_count(reduction_dims, i) == 0) {
      output_dims.push_back(ranked_tensor.getDimSize(i));
    }
  }

  // Convert input tensor to output type since mhlo::Reduce only supports same
  // element type for input/output.
  tensor = builder.create<mhlo::ConvertOp>(
      loc, tensor.getType().cast<TensorType>().clone(output_element_type),
      tensor);
  auto reducer_tensor_type = RankedTensorType::get({}, output_element_type);

  // Initial value for reduced tensor. This is set 0.
  Value init_values = builder.create<mhlo::ConstantOp>(
      loc, DenseIntElementsAttr::get(reducer_tensor_type, {0}));
  mhlo::ReduceOp reduce = builder.create<mhlo::ReduceOp>(
      loc, RankedTensorType::get(output_dims, output_element_type), tensor,
      init_values, builder.getI64TensorAttr(reduction_dims));
  // Define reducer function to compute sum.
  Region &region = reduce.getBody();
  Block &block = region.emplaceBlock();
  block.addArgument(reducer_tensor_type, loc);
  block.addArgument(reducer_tensor_type, loc);
  auto *firstArgument = block.args_begin();
  auto secondArgument = block.args_rbegin();
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&block);
    Value sum =
        builder.create<mhlo::AddOp>(loc, *firstArgument, *secondArgument);
    builder.create<mhlo::ReturnOp>(loc, sum);
  }
  Value zp = builder.create<mhlo::ConstantOp>(
      loc, builder.getI32IntegerAttr(other_tensor_zp));
  Value mul_op = builder.create<chlo::BroadcastMulOp>(loc, reduce.getResult(0),
                                                      zp, nullptr);
  return mul_op;
}

Value GetDimValue(OpBuilder &builder, Location loc, Value tensor,
                  mlir::ShapedType tensor_shape, int64_t idx) {
  if (tensor_shape.isDynamicDim(idx)) {
    // Get dynamic dim using GetDimensionSizeOp and convert result from <i32> to
    // <1xi64>.
    Value dynamic_dim = builder.create<mhlo::GetDimensionSizeOp>(
        loc, tensor, builder.getI64IntegerAttr(idx));
    dynamic_dim = builder.create<mhlo::ConvertOp>(
        loc, RankedTensorType::get(ArrayRef<int64_t>{}, builder.getI64Type()),
        dynamic_dim);
    return builder.create<mhlo::ReshapeOp>(
        loc, RankedTensorType::get({1}, builder.getI64Type()), dynamic_dim);
  } else {
    return builder.create<mhlo::ConstantOp>(
        loc, DenseIntElementsAttr::get(
                 RankedTensorType::get({1}, builder.getI64Type()),
                 {tensor_shape.getDimSize(idx)}));
  }
}

Value CalculateDynamicOutputDims(OpBuilder &builder, Location loc, Value output,
                                 ShapedType output_tensor_type,
                                 const DotLikeDimensionNumbers &dims) {
  // Calculate each output tensor dim and concatenate into a 1D tensor.
  SmallVector<Value> output_dims;
  for (int64_t i = 0; i < output_tensor_type.getRank(); ++i) {
    output_dims.push_back(
        GetDimValue(builder, loc, output, output_tensor_type, i));
  }
  return builder.create<mhlo::ConcatenateOp>(loc, output_dims,
                                             builder.getI64IntegerAttr(0));
}

Value BroadcastZpContribution(OpBuilder &builder, Location loc,
                              Value zp_contribution,
                              ArrayRef<int64_t> reduction_dims,
                              ArrayRef<int64_t> batching_dims,
                              int64_t non_batching_starting_idx, Value output,
                              TensorType output_tensor_type,
                              Value &output_dims_value,
                              const DotLikeDimensionNumbers &dims) {
  // This function calculates the dims for broadcasting from the
  // zero-point-offset tensor to the final output tensor, and then do the
  // broadcast.
  auto zp_contribution_rank =
      zp_contribution.getType().cast<ShapedType>().getRank();
  SmallVector<int64_t> broadcast_dims;
  broadcast_dims.resize(zp_contribution_rank, 0);
  // Result tensor will have batching dims first, then LHS result dims, then
  // RHS result dims. So non-batching result dims index doesn't start from 0.
  // The arg non_batching_starting_idx is used to distinguish LHS and RHS.
  int64_t result_batching_idx = 0;
  int64_t result_non_batching_idx = non_batching_starting_idx;
  for (int64_t idx = 0, original_idx = 0; idx < zp_contribution_rank;
       ++idx, ++original_idx) {
    // zp_contribution has removed contracting/spatial dims from the tensor
    // after reduction. The following recovers the index in the original tensor.
    while (absl::c_count(reduction_dims, original_idx) != 0) {
      original_idx++;
    }
    if (absl::c_count(batching_dims, original_idx) == 0) {
      broadcast_dims[idx] = result_non_batching_idx++;
    } else {
      broadcast_dims[idx] = result_batching_idx++;
    }
  }
  // Use broadcast_in_dim or dyanmic_broadcast_in_dim based on output shape
  // dynamism.
  if (output_tensor_type.cast<ShapedType>().hasStaticShape()) {
    zp_contribution = builder.create<mhlo::BroadcastInDimOp>(
        loc, output_tensor_type, zp_contribution,
        DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(broadcast_dims.size())},
                                  builder.getI64Type()),
            broadcast_dims));
  } else {
    if (!output_dims_value) {
      output_dims_value = CalculateDynamicOutputDims(builder, loc, output,
                                                     output_tensor_type, dims);
    }
    zp_contribution = builder.create<mhlo::DynamicBroadcastInDimOp>(
        loc, output_tensor_type, zp_contribution, output_dims_value,
        DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(broadcast_dims.size())},
                                  builder.getI64Type()),
            broadcast_dims));
  }
  return zp_contribution;
}

Value CalculateZeroPointOffset(OpBuilder &builder, Location loc, Value lhs,
                               Value rhs, Value output, int64_t lhs_zp,
                               int64_t rhs_zp, TensorType output_tensor_type,
                               const DotLikeDimensionNumbers &dims) {
  mlir::ShapedType lhs_shape = lhs.getType().cast<mlir::ShapedType>();
  mlir::ShapedType rhs_shape = rhs.getType().cast<mlir::ShapedType>();
  Value result = nullptr;
  Value output_dims_value = nullptr;
  // Calculate LHS contribution when RHS zp is non-zero.
  if (rhs_zp != 0) {
    SmallVector<int64_t> reduction_dims = to_vector(llvm::concat<const int64_t>(
        dims.lhs_spatial_dims, dims.lhs_contracting_dims));
    Value lhs_zp_contribution =
        CreateZeroPointPartialOffset(builder, loc, lhs, rhs_zp, reduction_dims);
    // Broadcast lhs ZP contribution to result tensor shape.
    lhs_zp_contribution = BroadcastZpContribution(
        builder, loc, lhs_zp_contribution, reduction_dims,
        dims.lhs_batching_dims, dims.lhs_batching_dims.size(), output,
        output_tensor_type, output_dims_value, dims);
    result = lhs_zp_contribution;
  }
  // Calculate RHS contribution when LHS zp is non-zero.
  if (lhs_zp != 0) {
    SmallVector<int64_t> reduction_dims = to_vector(llvm::concat<const int64_t>(
        dims.rhs_spatial_dims, dims.rhs_contracting_dims));
    Value rhs_zp_contribution =
        CreateZeroPointPartialOffset(builder, loc, rhs, lhs_zp, reduction_dims);
    // Broadcast rhs ZP contribution to result tensor shape.
    rhs_zp_contribution = BroadcastZpContribution(
        builder, loc, rhs_zp_contribution, reduction_dims,
        dims.rhs_batching_dims,
        lhs_shape.getRank() - dims.lhs_contracting_dims.size(), output,
        output_tensor_type, output_dims_value, dims);
    if (result) {
      result = builder.create<mhlo::AddOp>(loc, result, rhs_zp_contribution);
    } else {
      result = rhs_zp_contribution;
    }
  }

  if (lhs_zp != 0 && rhs_zp != 0) {
    // Contributions from LHS_ZP * RHS_ZP.
    // This is multiplied by the product of all contracting dimensions.
    int32_t contracting_dim_total_int = 1;
    bool has_dynamic_contracting_dim = false;
    Value dynamic_contracting_dim_total = builder.create<mhlo::ConstantOp>(
        loc, builder.getI32IntegerAttr(static_cast<int32_t>(1)));
    // Calculate the product for static/dynamic dims separately.
    for (int64_t rhs_idx : llvm::concat<const int64_t>(
             dims.rhs_spatial_dims, dims.rhs_contracting_dims)) {
      if (rhs_shape.isDynamicDim(rhs_idx)) {
        has_dynamic_contracting_dim = true;
        auto dim = builder.create<mhlo::GetDimensionSizeOp>(
            loc, rhs, builder.getI64IntegerAttr(rhs_idx));
        dynamic_contracting_dim_total = builder.create<mhlo::MulOp>(
            loc, dynamic_contracting_dim_total, dim);
      } else {
        contracting_dim_total_int *= rhs_shape.getDimSize(rhs_idx);
      }
    }
    Value zp_offset_value = builder.create<mhlo::ConstantOp>(
        loc, builder.getI32IntegerAttr(static_cast<int32_t>(lhs_zp) *
                                       static_cast<int32_t>(rhs_zp) *
                                       contracting_dim_total_int));
    // Multiply the static dims contribution by the dynamic one if needed.
    if (has_dynamic_contracting_dim) {
      zp_offset_value = builder.create<mhlo::MulOp>(
          loc, zp_offset_value, dynamic_contracting_dim_total);
    }
    result = builder.create<chlo::BroadcastSubOp>(loc, result, zp_offset_value,
                                                  nullptr);
  }
  return result;
}

// Generic function to create DotGeneral kernel for Dot/DotGeneral ops.
template <typename DotLikeOp>
Value CreateDotLikeKernel(OpBuilder &builder, Location loc, DotLikeOp op,
                          Type result_type, Value &lhs, Value &rhs,
                          ArrayRef<NamedAttribute> attrs) {
  return builder.create<mhlo::DotGeneralOp>(loc, result_type,
                                            ArrayRef<Value>{lhs, rhs}, attrs);
}

// Template specialization for Convolution op.
// This function may pad LHS if needed. If so, lhs is updated in place.
template <>
Value CreateDotLikeKernel<mhlo::ConvolutionOp>(OpBuilder &builder, Location loc,
                                               mhlo::ConvolutionOp op,
                                               Type result_type, Value &lhs,
                                               Value &rhs,
                                               ArrayRef<NamedAttribute> attrs) {
  // We only handle the case where RHS zp is zero.
  // Explicitly pad LHS with zp and update LHS value.
  SmallVector<NamedAttribute> new_attrs(attrs);
  if (op.getPadding().has_value() &&
      llvm::any_of(op.getPaddingAttr().getValues<int64_t>(),
                   [](int64_t x) { return x != 0; })) {
    auto original_padding = op.getPaddingAttr().getValues<int64_t>();

    Value zp = builder.create<mhlo::ConstantOp>(
        loc,
        DenseIntElementsAttr::get(
            RankedTensorType::get({}, builder.getI8Type()),
            {static_cast<int8_t>(getElementTypeOrSelf(op.getLhs().getType())
                                     .cast<UniformQuantizedType>()
                                     .getZeroPoint())}));
    // Convert Padding attributes from mhlo::Convolution to mhlo::Pad. Note that
    // Padding is applied for spatial dimensions [1...rank-1) only for
    // mhlo::Convolution. But mhlo::Pad require those for all dimensions. Hence
    // we add 0 to the beginning and end of the padding vectors.
    int64_t rank = lhs.getType().cast<TensorType>().getRank();
    SmallVector<int64_t> padding_low(rank, 0), padding_high(rank, 0),
        padding_interior(rank, 0);
    for (int64_t i = 1; i < rank - 1; ++i) {
      padding_low[i] = original_padding[i * 2 - 2];
      padding_high[i] = original_padding[i * 2 - 1];
    }
    lhs = builder.create<mhlo::PadOp>(
        loc, lhs, zp,
        DenseIntElementsAttr::get(
            RankedTensorType::get({rank}, builder.getI64Type()), padding_low),
        DenseIntElementsAttr::get(
            RankedTensorType::get({rank}, builder.getI64Type()), padding_high),
        DenseIntElementsAttr::get(
            RankedTensorType::get({rank}, builder.getI64Type()),
            padding_interior));

    // After explicitly padding/dilating LHS, update attributes so that LHS is
    // not padded/dilated again during Convolution.
    for (auto &attr : new_attrs) {
      if (attr.getName().getValue() == "padding") {
        attr.setValue(SplatElementsAttr::get(
            RankedTensorType::get({rank - 2, 2}, builder.getI64Type()),
            builder.getI64IntegerAttr(0)));
      }
    }
  }
  return builder.create<mhlo::ConvolutionOp>(
      loc, result_type, ArrayRef<Value>{lhs, rhs}, new_attrs);
}

template <typename DotLikeOp, typename DotLikeOpAdaptor>
LogicalResult matchAndRewriteDotLikeOp(DotLikeOp op, DotLikeOpAdaptor adaptor,
                                       ArrayRef<NamedAttribute> attrs,
                                       const DotLikeDimensionNumbers &dims,
                                       ConversionPatternRewriter &rewriter) {
  // Lower Dot/DotGeneral UQ ops to DotGeneral int.
  // Assumes that operands and results are uq types.
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  auto res_int32_tensor_type =
      op.getResult().getType().clone(rewriter.getI32Type());

  // Dot result
  //   = dot((lhs - zp_l) * scale_l, (rhs - zp_r) * scale_r) / scale_res
  //       + zp_res
  //   = dot(lhs - zp_l, rhs - zp_r) * scale_l * scale_r / scale_res + zp_res
  //   = dot(lhs, rhs) * combined_scale + combined_zp
  // where:
  //   combined_scale = scale_l * scale_r / scale_res
  //   combined_zp = res_zp - zp_offset * combined_scale
  //   zp_offset = zp_l*rhs + zp_r*lhs - zp_l*zp_r
  Value res_i32 = CreateDotLikeKernel(rewriter, op->getLoc(), op,
                                      res_int32_tensor_type, lhs, rhs, attrs);

  auto lhs_element_quant_type = getElementTypeOrSelf(op.getLhs().getType())
                                    .template cast<UniformQuantizedType>();
  auto rhs_element_quant_type = getElementTypeOrSelf(op.getRhs().getType())
                                    .template dyn_cast<UniformQuantizedType>();
  auto rhs_element_quant_per_channel_type =
      getElementTypeOrSelf(op.getRhs().getType())
          .template dyn_cast<UniformQuantizedPerAxisType>();
  auto res_element_quant_type = getElementTypeOrSelf(op.getResult())
                                    .template dyn_cast<UniformQuantizedType>();
  auto res_element_quant_per_channel_type =
      getElementTypeOrSelf(op.getResult())
          .template dyn_cast<UniformQuantizedPerAxisType>();

  // Here we assume LHS must be per-tensor quantized.
  // If RHS is per-channel quantized, it must has 0 zp.
  Value zp_offset = CalculateZeroPointOffset(
      rewriter, op->getLoc(), lhs, rhs, res_i32,
      lhs_element_quant_type.getZeroPoint(),
      (rhs_element_quant_type ? rhs_element_quant_type.getZeroPoint() : 0),
      res_int32_tensor_type, dims);

  // For per-channel quantization, we assume that result scales are proportional
  // to rhs scales for each channels.
  double combined_scale_fp =
      rhs_element_quant_type
          ? lhs_element_quant_type.getScale() *
                rhs_element_quant_type.getScale() /
                res_element_quant_type.getScale()
          : lhs_element_quant_type.getScale() *
                rhs_element_quant_per_channel_type.getScales()[0] /
                res_element_quant_per_channel_type.getScales()[0];

  // Multiply dot result and zp_offset by combined_scale only if it is not 1.0.
  if (std::abs(combined_scale_fp - 1.0) > 0.001) {
    Value combined_scale = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getF32FloatAttr(combined_scale_fp));

    auto res_float32_tensor_type =
        op.getResult().getType().clone(rewriter.getF32Type());
    Value res_f32 = rewriter.create<mhlo::ConvertOp>(
        op->getLoc(), res_float32_tensor_type, res_i32);
    res_f32 = rewriter.create<chlo::BroadcastMulOp>(
        op->getLoc(), res_float32_tensor_type, res_f32, combined_scale,
        nullptr);
    res_i32 = rewriter.create<mhlo::ConvertOp>(op->getLoc(),
                                               res_int32_tensor_type, res_f32);

    // Skip zp_offset if it is 0.
    if (zp_offset) {
      auto zp_offset_float32_tensor_type =
          zp_offset.getType().cast<TensorType>().clone(rewriter.getF32Type());
      zp_offset = rewriter.create<mhlo::ConvertOp>(
          op->getLoc(), zp_offset_float32_tensor_type, zp_offset);
      zp_offset = rewriter.create<chlo::BroadcastMulOp>(
          op->getLoc(), zp_offset_float32_tensor_type, zp_offset,
          combined_scale, nullptr);
      zp_offset = rewriter.create<mhlo::ConvertOp>(
          op->getLoc(),
          zp_offset_float32_tensor_type.clone(rewriter.getI32Type()),
          zp_offset);
    }
  }

  // If result is per-channel quantized, it must has 0 zp.
  Value combined_zp = rewriter.create<mhlo::ConstantOp>(
      op->getLoc(),
      rewriter.getI32IntegerAttr(
          res_element_quant_type ? res_element_quant_type.getZeroPoint() : 0));
  if (zp_offset) {
    combined_zp = rewriter.create<chlo::BroadcastSubOp>(
        op->getLoc(), res_int32_tensor_type, combined_zp, zp_offset, nullptr);
  }
  rewriter.replaceOpWithNewOp<chlo::BroadcastAddOp>(
      op, res_int32_tensor_type, res_i32, combined_zp, nullptr);
  return success();
}

template <typename DotLikeOp>
FailureOr<bool> IsDotLikeOpHybrid(DotLikeOp op) {
  // Checks whether a dot-like op is hybrid by looking at input/output types.
  // Returns failure() when the type is not supported.
  bool is_lhs_quant =
      isa<UniformQuantizedType>(getElementTypeOrSelf(op.getLhs().getType()));
  bool is_lhs_quant_per_channel = isa<UniformQuantizedPerAxisType>(
      getElementTypeOrSelf(op.getLhs().getType()));
  bool is_rhs_quant =
      isa<UniformQuantizedType>(getElementTypeOrSelf(op.getRhs().getType()));
  bool is_rhs_quant_per_channel = isa<UniformQuantizedPerAxisType>(
      getElementTypeOrSelf(op.getRhs().getType()));
  bool is_res_quant =
      isa<UniformQuantizedType>(getElementTypeOrSelf(op.getResult()));
  bool is_res_quant_per_channel =
      isa<UniformQuantizedPerAxisType>(getElementTypeOrSelf(op.getResult()));

  if (is_lhs_quant &&
      ((is_rhs_quant && is_res_quant) ||
       (isa<mhlo::ConvolutionOp>(op) && is_rhs_quant_per_channel &&
        is_res_quant_per_channel))) {
    // For quantized ops, RHS and result must be both per-channel quantized.
    // For Convolution, we also support per-channel quantized RHS/result.
    return false;
  } else if (!is_lhs_quant && !is_lhs_quant_per_channel && is_rhs_quant &&
             !is_res_quant && !is_res_quant_per_channel) {
    return true;
  } else {
    op->emitError("Invalid input/output type for Dot/Convolution op");
    return failure();
  }
}

class ConvertUniformQuantizedDotOp : public OpConversionPattern<mhlo::DotOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DotOp op, mhlo::DotOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto is_hybrid = IsDotLikeOpHybrid(op);
    if (failed(is_hybrid)) {
      return failure();
    }
    if (*is_hybrid) {
      return matchAndRewriteDotLikeHybridOp(op, adaptor, rewriter);
    } else {
      // DotOp is a special case of DotGeneralOp, where LHS and RHS are both
      // rank-2 tensors and have contracting dims of 1 and 0 respectively.
      auto dims = mhlo::DotDimensionNumbersAttr::get(
          rewriter.getContext(), /*lhsBatchingDimensions=*/{},
          /*rhsBatchingDimensions=*/{}, /*lhsContractingDimensions=*/{1},
          /*rhsContractingDimensions=*/{0});
      SmallVector<mlir::NamedAttribute> attrs(op->getAttrs());
      attrs.push_back(
          {StringAttr::get(rewriter.getContext(), "dot_dimension_numbers"),
           dims});
      return matchAndRewriteDotLikeOp(
          op, adaptor, attrs,
          DotLikeDimensionNumbers{/*lhs_batching_dims=*/{},
                                  /*lhs_spatial_dims=*/{},
                                  /*lhs_contracting_dims=*/{1},
                                  /*rhs_batching_dims=*/{},
                                  /*rhs_spatial_dims=*/{},
                                  /*rhs_contracting_dims=*/{0}},
          rewriter);
    }
  }
};

class ConvertUniformQuantizedDotGeneralOp
    : public OpConversionPattern<mhlo::DotGeneralOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DotGeneralOp op, mhlo::DotGeneralOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto is_hybrid = IsDotLikeOpHybrid(op);
    if (failed(is_hybrid)) {
      return failure();
    }
    if (*is_hybrid) {
      return matchAndRewriteDotLikeHybridOp(op, adaptor, rewriter);
    } else {
      return matchAndRewriteDotLikeOp(
          op, adaptor, op->getAttrs(),
          DotLikeDimensionNumbers{
              to_vector(op.getDotDimensionNumbers().getLhsBatchingDimensions()),
              /*lhs_spatial_dims=*/{},
              to_vector(
                  op.getDotDimensionNumbers().getLhsContractingDimensions()),
              to_vector(op.getDotDimensionNumbers().getRhsBatchingDimensions()),
              /*rhs_spatial_dims=*/{},
              to_vector(
                  op.getDotDimensionNumbers().getRhsContractingDimensions())},
          rewriter);
    }
  }
};

bool IsConvNHWC(const mhlo::ConvDimensionNumbersAttr &dims) {
  return dims.getInputBatchDimension() == 0 &&
         dims.getInputFeatureDimension() == 3 &&
         dims.getInputSpatialDimensions().size() == 2 &&
         dims.getInputSpatialDimensions()[0] == 1 &&
         dims.getInputSpatialDimensions()[1] == 2 &&
         dims.getKernelInputFeatureDimension() == 2 &&
         dims.getKernelOutputFeatureDimension() == 3 &&
         dims.getKernelSpatialDimensions().size() == 2 &&
         dims.getKernelSpatialDimensions()[0] == 0 &&
         dims.getKernelSpatialDimensions()[1] == 1 &&
         dims.getOutputBatchDimension() == 0 &&
         dims.getOutputFeatureDimension() == 3 &&
         dims.getOutputSpatialDimensions().size() == 2 &&
         dims.getOutputSpatialDimensions()[0] == 1 &&
         dims.getOutputSpatialDimensions()[1] == 2;
}

bool IsConvNDHWC(const mhlo::ConvDimensionNumbersAttr &dims) {
  return dims.getInputBatchDimension() == 0 &&
         dims.getInputFeatureDimension() == 4 &&
         dims.getInputSpatialDimensions().size() == 3 &&
         dims.getInputSpatialDimensions()[0] == 1 &&
         dims.getInputSpatialDimensions()[1] == 2 &&
         dims.getInputSpatialDimensions()[2] == 3 &&
         dims.getKernelInputFeatureDimension() == 3 &&
         dims.getKernelOutputFeatureDimension() == 4 &&
         dims.getKernelSpatialDimensions().size() == 3 &&
         dims.getKernelSpatialDimensions()[0] == 0 &&
         dims.getKernelSpatialDimensions()[1] == 1 &&
         dims.getKernelSpatialDimensions()[2] == 2 &&
         dims.getOutputBatchDimension() == 0 &&
         dims.getOutputFeatureDimension() == 4 &&
         dims.getOutputSpatialDimensions().size() == 3 &&
         dims.getOutputSpatialDimensions()[0] == 1 &&
         dims.getOutputSpatialDimensions()[1] == 2 &&
         dims.getOutputSpatialDimensions()[2] == 3;
}

FailureOr<DotLikeDimensionNumbers> VerifyAndConstructDims(
    mhlo::ConvolutionOp op) {
  // RHS (weight) must have zero zp.
  // Here assumes RHS/result must be both per-tensor or both per-channel
  // quantized.
  auto failed_or = GetQuantType(op.getRhs().getType());
  if (failed(failed_or)) {
    return failure();
  }
  QuantType rhs_element_quant_type = *failed_or;
  bool is_rhs_quant_per_tensor =
      std::get_if<UniformQuantizedType>(&rhs_element_quant_type);

  if (is_rhs_quant_per_tensor
          ? (std::get<UniformQuantizedType>(rhs_element_quant_type)
                 .getZeroPoint() != 0)
          : llvm::any_of(llvm::concat<const int64_t>(
                             std::get<UniformQuantizedPerAxisType>(
                                 rhs_element_quant_type)
                                 .getZeroPoints(),
                             getElementTypeOrSelf(op.getResult())
                                 .cast<UniformQuantizedPerAxisType>()
                                 .getZeroPoints()),
                         [](int64_t zp) { return zp != 0; })) {
    op->emitError("RHS/result UQ type must have zero zp.");
    return failure();
  }
  // For per-channel quantization, RHS quantized axis must be out channel axis.
  if (!is_rhs_quant_per_tensor &&
      (std::get<UniformQuantizedPerAxisType>(rhs_element_quant_type)
           .getQuantizedDimension() !=
       op.getRhs().getType().cast<TensorType>().getRank() - 1)) {
    op->emitError("Conv quantized axis must be out channel axis");
    return failure();
  }
  // For per-channel quantization, ratio between RHS and Result scales must be
  // the same for each channel.
  if (!is_rhs_quant_per_tensor) {
    auto res_element_quant_per_channel_type =
        getElementTypeOrSelf(op.getResult())
            .cast<UniformQuantizedPerAxisType>();
    SmallVector<double> scale_ratios(
        res_element_quant_per_channel_type.getScales().size());
    for (int i = 0; i < scale_ratios.size(); ++i) {
      scale_ratios[i] =
          res_element_quant_per_channel_type.getScales()[i] /
          std::get<UniformQuantizedPerAxisType>(rhs_element_quant_type)
              .getScales()[i];
      auto diff = (scale_ratios[i] - scale_ratios[0]) / scale_ratios[0];
      // Check all ratios within a threshold.
      if (std::abs(diff) > 0.001) {
        op->emitError(
            "Per-channel quantizated Conv must have same RHS/Result scale "
            "ratio for each channel");
        return failure();
      }
    }
  }
  // lhs_dilation must not exist.
  if (op.getLhsDilation().has_value() &&
      llvm::any_of(op.getLhsDilationAttr().getValues<int64_t>(),
                   [](int64_t dilate) { return dilate != 1; })) {
    op->emitError("lhs_dilation must be 1.");
    return failure();
  }

  // We only support NHWC Conv2D and NDHWC Conv3D.
  auto dims = op.getDimensionNumbers();
  if (IsConvNHWC(dims)) {
    // 2D Convolution.
    return DotLikeDimensionNumbers{/*lhs_batching_dims=*/{0},
                                   /*lhs_spatial_dims=*/{1, 2},
                                   /*lhs_contracting_dims=*/{3},
                                   /*rhs_batching_dims=*/{},
                                   /*rhs_spatial_dims=*/{0, 1},
                                   /*rhs_contracting_dims=*/{2}};
  } else if (IsConvNDHWC(dims)) {
    // 3D Convolution.
    return DotLikeDimensionNumbers{/*lhs_batching_dims=*/{0},
                                   /*lhs_spatial_dims=*/{1, 2, 3},
                                   /*lhs_contracting_dims=*/{4},
                                   /*rhs_batching_dims=*/{},
                                   /*rhs_spatial_dims=*/{0, 1, 2},
                                   /*rhs_contracting_dims=*/{3}};
  }
  op->emitError("Convolution data format must be NHWC.");
  return failure();
}

class ConvertUniformQuantizedConvolutionOp
    : public OpConversionPattern<mhlo::ConvolutionOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConvolutionOp op, mhlo::ConvolutionOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto is_hybrid = IsDotLikeOpHybrid(op);
    if (failed(is_hybrid)) {
      return failure();
    }
    if (*is_hybrid) {
      return matchAndRewriteDotLikeHybridOp(op, adaptor, rewriter);
    } else {
      auto dims = VerifyAndConstructDims(op);
      if (failed(dims)) return failure();
      return matchAndRewriteDotLikeOp(op, adaptor, op->getAttrs(), *dims,
                                      rewriter);
    }
  }
};

// This pattern lowers a generic MHLO op for uq->int.
// This pattern essentially just performs type change, with no algorithm change.
// TODO: b/310685906 - Add operand/result type validations.
class ConvertGenericOp : public ConversionPattern {
 public:
  explicit ConvertGenericOp(MLIRContext *ctx, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // This pattern only handle selected ops.
    if (!isa<mhlo::BroadcastInDimOp, mhlo::ConcatenateOp, mhlo::ConstantOp,
             mhlo::ConvertOp, mhlo::GatherOp, mhlo::MaxOp, mhlo::MinOp,
             mhlo::PadOp, mhlo::ReduceWindowOp, mhlo::ReshapeOp, mhlo::ReturnOp,
             mhlo::SelectOp, mhlo::SliceOp, mhlo::TransposeOp,
             mhlo::GetDimensionSizeOp, mhlo::DynamicBroadcastInDimOp>(op)) {
      return failure();
    }

    // Determine new result type: use storage type for uq types; use original
    // type otherwise.
    SmallVector<Type, 4> new_result_types;
    for (auto result_type : op->getResultTypes()) {
      new_result_types.push_back(GetQuantStorageType(result_type));
    }

    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         new_result_types, op->getAttrs(), op->getSuccessors());
    for (Region &region : op->getRegions()) {
      Region &new_region = *state.addRegion();
      rewriter.inlineRegionBefore(region, new_region, new_region.begin());
      if (failed(
              rewriter.convertRegionTypes(&new_region, *getTypeConverter()))) {
        return failure();
      }
    }
    Operation *new_op = rewriter.create(state);
    rewriter.replaceOp(op, new_op);
    return success();
  }
};

// TypeConverter for converting UQ type to int type.
class UQTypeConverter : public TypeConverter {
 public:
  UQTypeConverter() {
    addConversion([](Type type) -> Type { return GetQuantStorageType(type); });
  }
};

#define GEN_PASS_DEF_CONVERTMHLOQUANTTOINT
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h.inc"

class ConvertMHLOQuantToInt
    : public impl::ConvertMHLOQuantToIntBase<ConvertMHLOQuantToInt> {
 public:
  ConvertMHLOQuantToInt() = default;
  ConvertMHLOQuantToInt(const ConvertMHLOQuantToInt &) {}

  explicit ConvertMHLOQuantToInt(bool legalize_chlo) {
    legalize_chlo_ = legalize_chlo;
  }

  // Performs conversion of MHLO quant ops to primitive ops.
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);

    // Populate MHLO quant ops conversion patterns.
    patterns.add<ConvertUniformQuantizeOp, ConvertUniformDequantizeOp,
                 ConvertUniformQuantizedAddOp, ConvertUniformQuantizedDotOp,
                 ConvertUniformQuantizedDotGeneralOp,
                 ConvertUniformQuantizedConvolutionOp>(context);

    // uq->int convert patterns for func.func, func.return and generic ops.
    UQTypeConverter converter;
    patterns.add<ConvertGenericOp>(context, converter);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateReturnOpTypeConversionPattern(patterns, converter);

    ConversionTarget target(*op->getContext());
    auto is_legal = [&converter](Operation *op) {
      return converter.isLegal(op);
    };
    target.addDynamicallyLegalDialect<mhlo::MhloDialect>(is_legal);
    target.addDynamicallyLegalDialect<chlo::ChloDialect>(is_legal);
    target.addDynamicallyLegalDialect<func::FuncDialect>(
        [&converter](Operation *op) {
          if (auto func = dyn_cast<func::FuncOp>(op)) {
            return converter.isSignatureLegal(func.getFunctionType());
          }
          return converter.isLegal(op);
        });

    LogicalResult result =
        applyPartialConversion(op, target, std::move(patterns));
    if (failed(result)) {
      signalPassFailure();
    }

    // Legalize CHLO if needed.
    if (!legalize_chlo_) return;
    RewritePatternSet patterns_2(context);

    chlo::populateDecomposeChloPatterns(context, &patterns_2);
    chlo::populateChloBroadcastingPatterns(context, &patterns_2);

    ConversionTarget target_2 = mhlo::GetDefaultLegalConversionTargets(
        *op->getContext(), legalize_chlo_);

    result = applyPartialConversion(op, target_2, std::move(patterns_2));
    if (failed(result)) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createConvertMHLOQuantToIntPass(
    bool legalize_chlo) {
  return std::make_unique<ConvertMHLOQuantToInt>(legalize_chlo);
}

}  // namespace mlir::quant::stablehlo
