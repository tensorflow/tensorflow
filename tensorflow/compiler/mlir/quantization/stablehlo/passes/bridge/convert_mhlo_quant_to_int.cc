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

#define GEN_PASS_DEF_CONVERTMHLOQUANTTOINT
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h.inc"

// This helper function create ops to requantize `input` tensor and returns the
// output tensor. Clamping is done if output integer bit-width < 32.
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
Value Requantize(mlir::OpState op, Value input,
                 UniformQuantizedType input_quantized_type,
                 UniformQuantizedType output_quantized_type,
                 TensorType output_tensor_type,
                 ConversionPatternRewriter &rewriter) {
  // Skip requantization when input and result have the same type.
  if (input_quantized_type == output_quantized_type) {
    return rewriter.create<mhlo::ConvertOp>(op->getLoc(), output_tensor_type,
                                            input);
  }

  double merged_scale_fp =
      input_quantized_type.getScale() / output_quantized_type.getScale();
  Value merged_scale = rewriter.create<mhlo::ConstantOp>(
      op->getLoc(),
      rewriter.getF32FloatAttr(static_cast<float>(merged_scale_fp)));

  auto float_tensor_type =
      input.getType().cast<TensorType>().clone(rewriter.getF32Type());
  Value output_float =
      rewriter.create<mhlo::ConvertOp>(op->getLoc(), float_tensor_type, input);

  output_float = rewriter.create<chlo::BroadcastMulOp>(
      op->getLoc(), float_tensor_type, output_float, merged_scale, nullptr);

  // Add merged_zp only when it is non-zero.
  double merged_zp_fp = output_quantized_type.getZeroPoint() -
                        input_quantized_type.getZeroPoint() * merged_scale_fp;
  if (merged_zp_fp != 0) {
    Value merged_zp = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(),
        rewriter.getF32FloatAttr(static_cast<float>(merged_zp_fp)));
    output_float = rewriter.create<chlo::BroadcastAddOp>(
        op->getLoc(), float_tensor_type, output_float, merged_zp, nullptr);
  }

  // Clamp output if the output integer bit-width <32.
  if (output_tensor_type.getElementType().cast<IntegerType>().getWidth() < 32) {
    Value quantization_min = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getF32FloatAttr(static_cast<float>(
                          output_quantized_type.getStorageTypeMin())));
    Value quantization_max = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getF32FloatAttr(static_cast<float>(
                          output_quantized_type.getStorageTypeMax())));
    // Clamp results by [quantization_min, quantization_max].
    output_float = rewriter.create<mhlo::ClampOp>(
        op->getLoc(), float_tensor_type, quantization_min, output_float,
        quantization_max);
  }

  output_float = rewriter.create<mhlo::RoundNearestEvenOp>(
      op->getLoc(), float_tensor_type, output_float);
  return rewriter.create<mhlo::ConvertOp>(op->getLoc(), output_tensor_type,
                                          output_float);
}

class ConvertMHLOQuantToInt
    : public impl::ConvertMHLOQuantToIntBase<ConvertMHLOQuantToInt> {
 public:
  ConvertMHLOQuantToInt() = default;
  ConvertMHLOQuantToInt(const ConvertMHLOQuantToInt &) {}

  explicit ConvertMHLOQuantToInt(bool legalize_chlo) {
    legalize_chlo_ = legalize_chlo;
  }

  // Performs conversion of MHLO quant ops to primitive ops.
  void runOnOperation() override;
};

class ConvertUniformQuantizeOp
    : public OpConversionPattern<mhlo::UniformQuantizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::UniformQuantizeOp op, mhlo::UniformQuantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto quantized_type = getElementTypeOrSelf(op.getResult().getType())
                              .dyn_cast<UniformQuantizedType>();
    // Currently for activation, PTQ supports per-tensor quantization only, and
    // UniformQuantize op is only for activation.
    if (!quantized_type) {
      return rewriter.notifyMatchFailure(
          op, "Legalization supports only per-tensor quantization.");
    }
    auto input_element_type = getElementTypeOrSelf(op.getOperand().getType());
    if (input_element_type.isF32()) {
      return matchAndRewriteQuantize(op, adaptor, rewriter, quantized_type);
    } else if (input_element_type.isa<UniformQuantizedType>()) {
      return matchAndRewriteRequantize(op, adaptor, rewriter, quantized_type);
    }
    return rewriter.notifyMatchFailure(op, "Unsupported input element type.");
  }

  LogicalResult matchAndRewriteQuantize(
      mhlo::UniformQuantizeOp op, mhlo::UniformQuantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter,
      const UniformQuantizedType &quantized_type) const {
    Value scale = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getF32FloatAttr(quantized_type.getScale()));
    Value zero_point = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getF32FloatAttr(
                          static_cast<float>(quantized_type.getZeroPoint())));
    Value quantization_min = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getF32FloatAttr(static_cast<float>(
                          quantized_type.getStorageTypeMin())));
    Value quantization_max = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getF32FloatAttr(static_cast<float>(
                          quantized_type.getStorageTypeMax())));

    auto res_float_tensor_type =
        op.getOperand().getType().clone(rewriter.getF32Type());
    Value res_float = rewriter.create<chlo::BroadcastDivOp>(
        op->getLoc(), res_float_tensor_type, adaptor.getOperand(), scale,
        nullptr);
    res_float = rewriter.create<chlo::BroadcastAddOp>(
        op->getLoc(), res_float_tensor_type, res_float, zero_point, nullptr);

    res_float = rewriter.create<mhlo::ClampOp>(
        op->getLoc(), res_float_tensor_type, quantization_min, res_float,
        quantization_max);
    res_float = rewriter.create<mhlo::RoundNearestEvenOp>(
        op->getLoc(), res_float_tensor_type, res_float);
    auto res_final_tensor_type =
        res_float_tensor_type.clone(quantized_type.getStorageType());
    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(op, res_final_tensor_type,
                                                 res_float);
    return success();
  }

  LogicalResult matchAndRewriteRequantize(
      mhlo::UniformQuantizeOp op, mhlo::UniformQuantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter,
      const UniformQuantizedType &output_quantized_type) const {
    auto input_quantized_type = getElementTypeOrSelf(op.getOperand().getType())
                                    .cast<UniformQuantizedType>();
    rewriter.replaceOp(
        op, Requantize(op, adaptor.getOperand(), input_quantized_type,
                       output_quantized_type,
                       op.getResult().getType().cast<TensorType>().clone(
                           output_quantized_type.getStorageType()),
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
    auto element_type = getElementTypeOrSelf(op.getOperand().getType())
                            .dyn_cast<UniformQuantizedType>();
    // Currently for activation, PTQ supports per-tensor quantization only, and
    // UniformQuantize op is only for activation.
    if (!element_type) {
      return rewriter.notifyMatchFailure(
          op, "Legalization supports only per-tensor quantization.");
    }
    Value scale = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getF32FloatAttr(element_type.getScale()));
    Value zero_point = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(
                          static_cast<int32_t>(element_type.getZeroPoint())));

    Value input = adaptor.getOperand();
    // TODO: b/260280919 - Consider avoiding conversion to int32.
    auto res_int32_tensor_type =
        input.getType().cast<TensorType>().clone(rewriter.getI32Type());
    Value res_int32 = rewriter.create<mhlo::ConvertOp>(
        op->getLoc(), res_int32_tensor_type, input);
    res_int32 = rewriter.create<chlo::BroadcastSubOp>(
        op->getLoc(), res_int32_tensor_type, res_int32, zero_point, nullptr);
    auto res_float_tensor_type =
        res_int32.getType().cast<TensorType>().clone(rewriter.getF32Type());
    Value res_float = rewriter.create<mhlo::ConvertOp>(
        op->getLoc(), res_float_tensor_type, res_int32);
    res_float = rewriter.replaceOpWithNewOp<chlo::BroadcastMulOp>(
        op, res_float_tensor_type, res_float, scale, nullptr);
    return success();
  }
};

class ConvertUniformQuantizedAddOp : public OpConversionPattern<mhlo::AddOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::AddOp op, mhlo::AddOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto lhs_element_type =
        op.getLhs().getType().getElementType().dyn_cast<UniformQuantizedType>();
    auto rhs_element_type =
        op.getRhs().getType().getElementType().dyn_cast<UniformQuantizedType>();
    auto result_element_type = op.getResult()
                                   .getType()
                                   .getElementType()
                                   .dyn_cast<UniformQuantizedType>();

    // We only handle cases where lhs, rhs and results all have quantized
    // element type.
    if (!lhs_element_type || !rhs_element_type || !result_element_type) {
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
        Requantize(op, lhs, lhs_element_type, result_element_type,
                   res_int32_tensor_type, rewriter);

    Value rhs = adaptor.getRhs();
    Value rhs_int32_tensor =
        Requantize(op, rhs, rhs_element_type, result_element_type,
                   res_int32_tensor_type, rewriter);

    Value zero_point = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                          result_element_type.getZeroPoint())));

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

    if (result_element_type.getStorageType().isInteger(32)) {
      // For i32, clamping is not needed.
      rewriter.replaceOp(op, res_int32);
    } else {
      // Clamp results by [quantization_min, quantization_max] when storage type
      // is not i32.
      Value result_quantization_min = rewriter.create<mhlo::ConstantOp>(
          op->getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                            result_element_type.getStorageTypeMin())));
      Value result_quantization_max = rewriter.create<mhlo::ConstantOp>(
          op->getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                            result_element_type.getStorageTypeMax())));
      res_int32 = rewriter.create<mhlo::ClampOp>(
          op->getLoc(), res_int32_tensor_type, result_quantization_min,
          res_int32, result_quantization_max);
      // Convert results back to result storage type.
      auto res_final_tensor_type =
          res_int32_tensor_type.clone(result_element_type.getStorageType());
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
  ArrayRef<int64_t> lhs_batching_dims;
  ArrayRef<int64_t> lhs_spatial_dims;
  ArrayRef<int64_t> lhs_contracting_dims;
  ArrayRef<int64_t> rhs_batching_dims;
  ArrayRef<int64_t> rhs_spatial_dims;
  ArrayRef<int64_t> rhs_contracting_dims;
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
                                   ArrayRef<int64_t> reduction_dims) {
  // This function calculates part of the zero-point-offset by using
  // mhlo::Reduce to sum over the contracting dims of the tensor, and then
  // multiply by zp of the other tensor.
  auto output_element_type = builder.getI32Type();

  // Calculate the output tensor shape. This is input tensor dims minus
  // contracting dims.
  auto ranked_tensor = tensor.getType().cast<RankedTensorType>();
  llvm::SmallVector<int64_t> output_dims;
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

Value CalculateDynamicOutputDims(OpBuilder &builder, Location loc, Value lhs,
                                 Value rhs,
                                 const DotLikeDimensionNumbers &dims) {
  mlir::ShapedType lhs_shape = lhs.getType().cast<mlir::ShapedType>();
  mlir::ShapedType rhs_shape = rhs.getType().cast<mlir::ShapedType>();
  // Calculate each output dim and concatenate into a 1D tensor.
  // Output dims are batching dims, spatial dims, LHS result dims, RHS result
  // dims.
  llvm::SmallVector<Value> output_dims;
  for (int64_t i = 0; i < lhs_shape.getRank(); ++i) {
    if (absl::c_count(dims.lhs_batching_dims, i) != 0) {
      output_dims.push_back(GetDimValue(builder, loc, lhs, lhs_shape, i));
    }
  }
  for (int64_t i = 0; i < lhs_shape.getRank(); ++i) {
    if (absl::c_count(dims.lhs_spatial_dims, i) != 0) {
      output_dims.push_back(GetDimValue(builder, loc, lhs, lhs_shape, i));
    }
  }
  for (int64_t i = 0; i < lhs_shape.getRank(); ++i) {
    if (absl::c_count(dims.lhs_batching_dims, i) == 0 &&
        absl::c_count(dims.lhs_spatial_dims, i) == 0 &&
        absl::c_count(dims.lhs_contracting_dims, i) == 0) {
      output_dims.push_back(GetDimValue(builder, loc, lhs, lhs_shape, i));
    }
  }
  for (int64_t i = 0; i < rhs_shape.getRank(); ++i) {
    if (absl::c_count(dims.rhs_batching_dims, i) == 0 &&
        absl::c_count(dims.rhs_spatial_dims, i) == 0 &&
        absl::c_count(dims.rhs_contracting_dims, i) == 0) {
      output_dims.push_back(GetDimValue(builder, loc, rhs, rhs_shape, i));
    }
  }
  return builder.create<mhlo::ConcatenateOp>(loc, output_dims,
                                             builder.getI64IntegerAttr(0));
}

Value BroadcastZpContribution(OpBuilder &builder, Location loc,
                              Value zp_contribution,
                              llvm::ArrayRef<int64_t> reduction_dims,
                              llvm::ArrayRef<int64_t> batching_dims,
                              int64_t non_batching_starting_idx,
                              TensorType output_tensor_type,
                              Value &output_dims_value, Value lhs, Value rhs,
                              const DotLikeDimensionNumbers &dims) {
  // This function calculates the dims for broadcasting from the
  // zero-point-offset tensor to the final output tensor, and then do the
  // broadcast.
  auto zp_contribution_rank =
      zp_contribution.getType().cast<ShapedType>().getRank();
  llvm::SmallVector<int64_t> broadcast_dims;
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
  // Use broadcast_in_dim or dyanmic_broadcast_in_dim based on input shape
  // dynamism.
  if (zp_contribution.getType().cast<ShapedType>().hasStaticShape()) {
    zp_contribution = builder.create<mhlo::BroadcastInDimOp>(
        loc, output_tensor_type, zp_contribution,
        DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(broadcast_dims.size())},
                                  builder.getI64Type()),
            broadcast_dims));
  } else {
    if (!output_dims_value) {
      output_dims_value =
          CalculateDynamicOutputDims(builder, loc, lhs, rhs, dims);
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
                               Value rhs, int64_t lhs_zp, int64_t rhs_zp,
                               TensorType output_tensor_type,
                               const DotLikeDimensionNumbers &dims) {
  mlir::ShapedType lhs_shape = lhs.getType().cast<mlir::ShapedType>();
  mlir::ShapedType rhs_shape = rhs.getType().cast<mlir::ShapedType>();
  Value result = nullptr;
  Value output_dims_value = nullptr;
  // Calculate LHS contribution when RHS zp is non-zero.
  if (rhs_zp != 0) {
    llvm::SmallVector<int64_t> reduction_dims =
        llvm::to_vector(llvm::concat<const int64_t>(dims.lhs_spatial_dims,
                                                    dims.lhs_contracting_dims));
    Value lhs_zp_contribution =
        CreateZeroPointPartialOffset(builder, loc, lhs, rhs_zp, reduction_dims);
    // Broadcast lhs ZP contribution to result tensor shape.
    lhs_zp_contribution = BroadcastZpContribution(
        builder, loc, lhs_zp_contribution, reduction_dims,
        dims.lhs_batching_dims, dims.lhs_batching_dims.size(),
        output_tensor_type, output_dims_value, lhs, rhs, dims);
    result = lhs_zp_contribution;
  }
  // Calculate RHS contribution when LHS zp is non-zero.
  if (lhs_zp != 0) {
    llvm::SmallVector<int64_t> reduction_dims =
        llvm::to_vector(llvm::concat<const int64_t>(dims.rhs_spatial_dims,
                                                    dims.rhs_contracting_dims));
    Value rhs_zp_contribution =
        CreateZeroPointPartialOffset(builder, loc, rhs, lhs_zp, reduction_dims);
    // Broadcast rhs ZP contribution to result tensor shape.
    rhs_zp_contribution = BroadcastZpContribution(
        builder, loc, rhs_zp_contribution, reduction_dims,
        dims.rhs_batching_dims,
        lhs_shape.getRank() - dims.lhs_contracting_dims.size(),
        output_tensor_type, output_dims_value, lhs, rhs, dims);
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
  auto original_padding = op.getPaddingAttr().getValues<int64_t>();

  // Explicitly pad LHS with zp and update LHS value.
  llvm::SmallVector<NamedAttribute> new_attrs(attrs);
  if (llvm::any_of(original_padding, [](int64_t x) { return x != 0; })) {
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
    llvm::SmallVector<int64_t> padding_low(rank, 0), padding_high(rank, 0),
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
  auto lhs_element_quant_type = getElementTypeOrSelf(op.getLhs().getType())
                                    .template dyn_cast<UniformQuantizedType>();
  auto rhs_element_quant_type = getElementTypeOrSelf(op.getRhs().getType())
                                    .template dyn_cast<UniformQuantizedType>();
  auto res_element_quant_type = getElementTypeOrSelf(op.getResult())
                                    .template dyn_cast<UniformQuantizedType>();
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

  Value zp_offset = CalculateZeroPointOffset(
      rewriter, op->getLoc(), lhs, rhs, lhs_element_quant_type.getZeroPoint(),
      rhs_element_quant_type.getZeroPoint(), res_int32_tensor_type, dims);

  // Multiply dot result and zp_offset by combined_scale only if it is not 1.0.
  double combined_scale_fp = lhs_element_quant_type.getScale() *
                             rhs_element_quant_type.getScale() /
                             res_element_quant_type.getScale();
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

  Value combined_zp = rewriter.create<mhlo::ConstantOp>(
      op->getLoc(),
      rewriter.getI32IntegerAttr(res_element_quant_type.getZeroPoint()));
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
  auto lhs_element_quant_type = getElementTypeOrSelf(op.getLhs().getType())
                                    .template dyn_cast<UniformQuantizedType>();
  auto rhs_element_quant_type = getElementTypeOrSelf(op.getRhs().getType())
                                    .template dyn_cast<UniformQuantizedType>();
  auto res_element_quant_type = getElementTypeOrSelf(op.getResult())
                                    .template dyn_cast<UniformQuantizedType>();
  if (lhs_element_quant_type && rhs_element_quant_type &&
      res_element_quant_type) {
    return false;
  } else if (!lhs_element_quant_type && rhs_element_quant_type &&
             !res_element_quant_type) {
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
      llvm::SmallVector<mlir::NamedAttribute> attrs(op->getAttrs());
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
              op.getDotDimensionNumbers().getLhsBatchingDimensions(),
              /*lhs_spatial_dims=*/{},
              op.getDotDimensionNumbers().getLhsContractingDimensions(),
              op.getDotDimensionNumbers().getRhsBatchingDimensions(),
              /*rhs_spatial_dims=*/{},
              op.getDotDimensionNumbers().getRhsContractingDimensions()},
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

FailureOr<DotLikeDimensionNumbers> VerifyConvolutionOp(mhlo::ConvolutionOp op) {
  // RHS (weight) must have zero zp.
  auto rhs_element_quant_type =
      getElementTypeOrSelf(op.getRhs().getType()).cast<UniformQuantizedType>();
  if (rhs_element_quant_type.getZeroPoint() != 0) {
    op->emitError("RHS UQ type must have zero zp.");
    return failure();
  }
  // lhs_dilation must not exist.
  if (llvm::any_of(op.getLhsDilationAttr().getValues<int64_t>(),
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
      auto dims = VerifyConvolutionOp(op);
      if (failed(dims)) return failure();
      return matchAndRewriteDotLikeOp(op, adaptor, op->getAttrs(), *dims,
                                      rewriter);
    }
  }
};

// This pattern lowers a generic MHLO op for uq->int.
// This pattern essentially just performs type change, with no algorithm change.
class ConvertGenericOp : public ConversionPattern {
 public:
  explicit ConvertGenericOp(MLIRContext *ctx)
      : ConversionPattern(MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // This pattern only handle selected ops.
    if (!llvm::isa<mhlo::ConstantOp, mhlo::ConvertOp, mhlo::BroadcastInDimOp,
                   mhlo::MaxOp, mhlo::MinOp>(op)) {
      return failure();
    }

    // Check that all operands and result uq types are the same.
    llvm::SmallVector<Type> uq_types;
    for (auto result_type : op->getResultTypes()) {
      auto type =
          getElementTypeOrSelf(result_type).dyn_cast<UniformQuantizedType>();
      if (type) {
        uq_types.push_back(type);
      }
    }
    for (auto operand : op->getOperands()) {
      auto type = getElementTypeOrSelf(operand.getType())
                      .dyn_cast<UniformQuantizedType>();
      if (type) {
        uq_types.push_back(type);
      }
    }
    for (auto type : uq_types) {
      if (type != uq_types.front()) {
        return failure();
      }
    }

    // Determine new result type: use storage type for uq types; use original
    // type otherwise.
    llvm::SmallVector<Type, 4> new_result_types;
    for (auto result_type : op->getResultTypes()) {
      if (getElementTypeOrSelf(result_type).isa<UniformQuantizedType>()) {
        new_result_types.push_back(result_type.cast<TensorType>().clone(
            getElementTypeOrSelf(result_type)
                .cast<UniformQuantizedType>()
                .getStorageType()));
      } else {
        new_result_types.push_back(result_type);
      }
    }

    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         new_result_types, op->getAttrs(), op->getSuccessors());
    Operation *new_op = rewriter.create(state);
    rewriter.replaceOp(op, new_op);
    return success();
  }
};

// TypeConverter for converting UQ type to int type.
class UQTypeConverter : public TypeConverter {
 public:
  UQTypeConverter() {
    addConversion([](Type type) -> Type {
      auto to_legal_type = [](Type type) {
        if (auto uq_type = dyn_cast<UniformQuantizedType>(type)) {
          return uq_type.getStorageType();
        }
        return type;
      };
      if (auto shaped = type.dyn_cast<ShapedType>()) {
        return shaped.clone(to_legal_type(shaped.getElementType()));
      } else {
        return to_legal_type(type);
      }
    });
  }
};

// Performs conversion of MHLO quant ops to primitive ops.
void ConvertMHLOQuantToInt::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *context = op->getContext();
  RewritePatternSet patterns(context);

  // Populate MHLO quant ops conversion patterns.
  patterns.add<ConvertUniformQuantizeOp, ConvertUniformDequantizeOp,
               ConvertUniformQuantizedAddOp, ConvertUniformQuantizedDotOp,
               ConvertUniformQuantizedDotGeneralOp,
               ConvertUniformQuantizedConvolutionOp, ConvertGenericOp>(context);

  // uq->int convert patterns for func.func and func.return.
  UQTypeConverter converter;
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);
  populateReturnOpTypeConversionPattern(patterns, converter);

  ConversionTarget target(*op->getContext());
  auto is_legal = [&converter](Operation *op) { return converter.isLegal(op); };
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

  ConversionTarget target_2 =
      mhlo::GetDefaultLegalConversionTargets(*op->getContext(), legalize_chlo_);

  result = applyPartialConversion(op, target_2, std::move(patterns_2));
  if (failed(result)) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createConvertMHLOQuantToIntPass(
    bool legalize_chlo) {
  return std::make_unique<ConvertMHLOQuantToInt>(legalize_chlo);
}

}  // namespace mlir::quant::stablehlo
