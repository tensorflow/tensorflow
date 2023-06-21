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
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/math_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/utils.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace mhlo {
namespace {

#define GEN_PASS_DEF_CONVERTMHLOQUANTTOINT
#include "tensorflow/compiler/mlir/tf2xla/transforms/xla_legalize_tf_passes.h.inc"

FailureOr<TensorType> GetSameShapeTensorType(Operation *op,
                                             TensorType tensor_type,
                                             Type element_type,
                                             PatternRewriter &rewriter) {
  if (auto ranked_ty = tensor_type.dyn_cast_or_null<RankedTensorType>()) {
    Attribute encoding = ranked_ty.getEncoding();
    if (!(!encoding || encoding.isa<TypeExtensionsAttr>() ||
          encoding.isa<sparse_tensor::SparseTensorEncodingAttr>())) {
      return rewriter.notifyMatchFailure(
          op,
          "Ranked tensor encoding must be either null, TypeExtensionsAttr, or "
          "SparseTensorEncodingAttr.");
    }
    return RankedTensorType::get(ranked_ty.getShape(), element_type, encoding);
  }
  if (auto unranked_ty = tensor_type.dyn_cast_or_null<UnrankedTensorType>()) {
    return UnrankedTensorType::get(element_type);
  }
  llvm_unreachable("unhandled type");
}

// This helper function create ops to requantize `input` tensor and output to
// `res_int32` tensor. Clamping is omitted because for some ops clamping can be
// done later to avoid duplicate.
LogicalResult RequantizeWithoutClamping(
    mlir::OpState op, Value input, TensorType input_int32_tensor_type,
    quant::UniformQuantizedType input_quantized_type,
    quant::UniformQuantizedType result_quantized_type, Value &res_int32,
    ConversionPatternRewriter &rewriter) {
  // Convert input to int32 tensor.
  res_int32 = rewriter.create<mhlo::ConvertOp>(op->getLoc(),
                                               input_int32_tensor_type, input);
  // Undo the input zero point.
  Value input_zero_point = rewriter.create<mhlo::ConstantOp>(
      op->getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                        input_quantized_type.getZeroPoint())));
  res_int32 = rewriter.create<chlo::BroadcastSubOp>(
      op->getLoc(), input_int32_tensor_type, res_int32, input_zero_point,
      nullptr);

  // Adjust the scale.
  const double effective_scale =
      input_quantized_type.getScale() / result_quantized_type.getScale();
  int32_t effective_quantized_fraction;
  int32_t effective_shift;
  if (failed(stablehlo::QuantizeMultiplier(
          effective_scale, effective_quantized_fraction, effective_shift))) {
    op->emitError("Invalid effective quantization scale.");
    return failure();
  }
  Value multiplier = rewriter.create<mhlo::ConstantOp>(
      op->getLoc(), rewriter.getI32IntegerAttr(
                        static_cast<int32_t>(effective_quantized_fraction)));
  // The effective_quantized_fraction value has been quantized by multiplying
  // (1 << 15).  So, we have to shift it back by (15 - effective_shift) to get
  // the desired outcome.
  Value total_shift = rewriter.create<mhlo::ConstantOp>(
      op->getLoc(),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(15 - effective_shift)));

  // Apply the effective scale with rounding.
  Value half = rewriter.create<mhlo::ConstantOp>(
      op->getLoc(), rewriter.getI32IntegerAttr(
                        static_cast<int32_t>(1 << (14 - effective_shift))));
  res_int32 = rewriter.create<chlo::BroadcastMulOp>(
      op->getLoc(), input_int32_tensor_type, res_int32, multiplier, nullptr);
  res_int32 = rewriter.create<chlo::BroadcastAddOp>(
      op->getLoc(), input_int32_tensor_type, res_int32, half, nullptr);
  res_int32 = rewriter.create<chlo::BroadcastShiftRightArithmeticOp>(
      op->getLoc(), input_int32_tensor_type, res_int32, total_shift, nullptr);

  // Apply the output zero point.
  Value output_zero_point = rewriter.create<mhlo::ConstantOp>(
      op->getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                        result_quantized_type.getZeroPoint())));
  res_int32 = rewriter.create<chlo::BroadcastAddOp>(
      op->getLoc(), input_int32_tensor_type, res_int32, output_zero_point,
      nullptr);

  return success();
}

class ConvertMHLOQuantToInt
    : public impl::ConvertMHLOQuantToIntBase<ConvertMHLOQuantToInt> {
 public:
  // Performs conversion of MHLO quant ops to primitive ops.
  void runOnOperation() override;
};

class ConvertUniformQuantizeOp
    : public OpConversionPattern<mhlo::UniformQuantizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::UniformQuantizeOp op, UniformQuantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto quantized_type = getElementTypeOrSelf(op.getResult().getType())
                              .dyn_cast<quant::UniformQuantizedType>();
    // Currently for activation, PTQ supports per-tensor quantization only, and
    // UniformQuantize op is only for activation.
    if (!quantized_type) {
      return rewriter.notifyMatchFailure(
          op, "Legalization supports only per-tensor quantization.");
    }
    auto input_element_type = getElementTypeOrSelf(op.getOperand().getType());
    if (input_element_type.isF32()) {
      return matchAndRewriteQuantize(op, adaptor, rewriter, quantized_type);
    } else if (input_element_type.isa<quant::UniformQuantizedType>()) {
      return matchAndRewriteRequantize(op, adaptor, rewriter, quantized_type);
    }
    return rewriter.notifyMatchFailure(op, "Unsupported input element type.");
  }

  LogicalResult matchAndRewriteQuantize(
      mhlo::UniformQuantizeOp op, UniformQuantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter,
      const quant::UniformQuantizedType &quantized_type) const {
    Value scale = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getF32FloatAttr(quantized_type.getScale()));
    Value zero_point = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(
                          static_cast<int32_t>(quantized_type.getZeroPoint())));
    Value half = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getF32FloatAttr(0.5f));
    Value quantization_min = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                          quantized_type.getStorageTypeMin())));
    Value quantization_max = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                          quantized_type.getStorageTypeMax())));

    auto res_float_tensor_type_or =
        GetSameShapeTensorType(op, op.getOperand().getType().cast<TensorType>(),
                               rewriter.getF32Type(), rewriter);
    if (failed(res_float_tensor_type_or)) {
      return failure();
    }
    Value res_float = rewriter.create<chlo::BroadcastDivOp>(
        op->getLoc(), *res_float_tensor_type_or, adaptor.getOperand(), scale,
        nullptr);
    // TODO: b/260280919 - Consider using round_nearest_even.
    res_float = rewriter.create<chlo::BroadcastAddOp>(
        op->getLoc(), *res_float_tensor_type_or, res_float, half, nullptr);
    res_float = rewriter.create<mhlo::FloorOp>(op->getLoc(), res_float);
    // TODO: b/260280919 - Consider avoiding conversion to int32.
    auto res_int32_tensor_type_or =
        GetSameShapeTensorType(op, res_float.getType().cast<TensorType>(),
                               rewriter.getI32Type(), rewriter);
    if (failed(res_int32_tensor_type_or)) {
      return failure();
    }
    Value res_int32 = rewriter.create<mhlo::ConvertOp>(
        op->getLoc(), *res_int32_tensor_type_or, res_float);
    // TODO: b/260280919 - Use mhlo::Clamp instead.
    res_int32 = rewriter.create<chlo::BroadcastAddOp>(
        op->getLoc(), *res_int32_tensor_type_or, res_int32, zero_point,
        nullptr);
    res_int32 = rewriter.create<chlo::BroadcastMaxOp>(
        op->getLoc(), *res_int32_tensor_type_or, res_int32, quantization_min,
        nullptr);
    res_int32 = rewriter.create<chlo::BroadcastMinOp>(
        op->getLoc(), *res_int32_tensor_type_or, res_int32, quantization_max,
        nullptr);
    auto res_final_tensor_type_or =
        GetSameShapeTensorType(op, res_int32.getType().cast<TensorType>(),
                               quantized_type.getStorageType(), rewriter);
    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(op, *res_final_tensor_type_or,
                                                 res_int32);
    return success();
  }

  // Requantization is essentially dequantize --> quantize.
  //
  // Dequantize: (input - zp) * scale
  // Quantize: input / scale + zp
  //
  // Hence,
  // Requantize: (input - input_zp) * input_scale / output_scale + output_zp
  //
  // This function makes the above formula slightly more efficient by
  // precalculating and bundling the "* input_scale / output_scale" part into
  // a single multiplication.
  LogicalResult matchAndRewriteRequantize(
      mhlo::UniformQuantizeOp op, UniformQuantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter,
      const quant::UniformQuantizedType &output_quantized_type) const {
    auto input_quantized_type = getElementTypeOrSelf(op.getOperand().getType())
                                    .cast<quant::UniformQuantizedType>();
    auto result_quantized_type = getElementTypeOrSelf(op.getResult().getType())
                                     .cast<quant::UniformQuantizedType>();

    Value input = adaptor.getOperand();
    Value res_int32;
    auto res_int32_tensor_type_or =
        GetSameShapeTensorType(op, input.getType().cast<TensorType>(),
                               rewriter.getI32Type(), rewriter);
    if (failed(res_int32_tensor_type_or)) {
      return failure();
    }

    // Requantize input tensor to have be the same scale/zp as the result.
    auto res = RequantizeWithoutClamping(
        op, input, *res_int32_tensor_type_or, input_quantized_type,
        result_quantized_type, res_int32, rewriter);
    if (failed(res)) {
      return failure();
    }

    Value quantization_min = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                          output_quantized_type.getStorageTypeMin())));
    Value quantization_max = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                          output_quantized_type.getStorageTypeMax())));

    // Clamp results by [quantization_min, quantization_max].
    res_int32 = rewriter.create<mhlo::ClampOp>(
        op->getLoc(), *res_int32_tensor_type_or, quantization_min, res_int32,
        quantization_max);

    auto res_final_tensor_type_or = GetSameShapeTensorType(
        op, res_int32.getType().cast<TensorType>(),
        output_quantized_type.getStorageType(), rewriter);
    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(op, *res_final_tensor_type_or,
                                                 res_int32);
    return success();
  }
};

class ConvertUniformDequantizeOp
    : public OpConversionPattern<mhlo::UniformDequantizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::UniformDequantizeOp op, UniformDequantizeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto element_type = getElementTypeOrSelf(op.getOperand().getType())
                            .dyn_cast<quant::UniformQuantizedType>();
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
    auto res_int32_tensor_type_or =
        GetSameShapeTensorType(op, input.getType().cast<TensorType>(),
                               rewriter.getI32Type(), rewriter);
    if (failed(res_int32_tensor_type_or)) {
      return failure();
    }
    Value res_int32 = rewriter.create<mhlo::ConvertOp>(
        op->getLoc(), *res_int32_tensor_type_or, input);
    res_int32 = rewriter.create<chlo::BroadcastSubOp>(
        op->getLoc(), *res_int32_tensor_type_or, res_int32, zero_point,
        nullptr);
    auto res_float_tensor_type_or =
        GetSameShapeTensorType(op, res_int32.getType().cast<TensorType>(),
                               rewriter.getF32Type(), rewriter);
    if (failed(res_float_tensor_type_or)) {
      return failure();
    }
    Value res_float = rewriter.create<mhlo::ConvertOp>(
        op->getLoc(), *res_float_tensor_type_or, res_int32);
    res_float = rewriter.replaceOpWithNewOp<chlo::BroadcastMulOp>(
        op, *res_float_tensor_type_or, res_float, scale, nullptr);
    return success();
  }
};

class ConvertUniformQuantizedAddOp : public OpConversionPattern<mhlo::AddOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::AddOp op, AddOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto lhs_element_type = op.getLhs()
                                .getType()
                                .getElementType()
                                .dyn_cast<quant::UniformQuantizedType>();
    auto rhs_element_type = op.getRhs()
                                .getType()
                                .getElementType()
                                .dyn_cast<quant::UniformQuantizedType>();
    auto result_element_type = op.getResult()
                                   .getType()
                                   .getElementType()
                                   .dyn_cast<quant::UniformQuantizedType>();

    // We only handle cases where lhs, rhs and results all have the same
    // quantized element type.
    if (!lhs_element_type || !rhs_element_type || !result_element_type ||
        lhs_element_type != rhs_element_type ||
        lhs_element_type != result_element_type) {
      op->emitError(
          "AddOp requires the same quantized element type for all operands and "
          "results");
      return failure();
    }

    // TODO: b/260280919 - Consider avoiding conversion to int32.
    auto res_int32_tensor_type_or =
        GetSameShapeTensorType(op, op.getResult().getType().cast<TensorType>(),
                               rewriter.getI32Type(), rewriter);
    if (failed(res_int32_tensor_type_or)) {
      return failure();
    }

    Value result_quantization_min = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                          result_element_type.getStorageTypeMin())));
    Value result_quantization_max = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                          result_element_type.getStorageTypeMax())));
    Value zero_point = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(
                          result_element_type.getZeroPoint())));

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Given:
    // lhs_fp = (lhs_quant - zp) * scale
    // rhs_fp = (rhs_quant - zp) * scale
    // res_fp = lhs_fp + rhs_fp
    //        = ((lhs_quant + rhs_quant - zp) - zp) * scale
    // res_quant = res_fp / scale + zp
    //           = lhs_quant + rhs_quant - zp
    // The following converts inputs quantized operands to int32 type and add,
    // and then substract by zero point.
    Value lhs_int32_tensor = rewriter.create<mhlo::ConvertOp>(
        op->getLoc(), *res_int32_tensor_type_or, lhs);
    Value rhs_int32_tensor = rewriter.create<mhlo::ConvertOp>(
        op->getLoc(), *res_int32_tensor_type_or, rhs);
    Value add_result = rewriter.create<chlo::BroadcastAddOp>(
        op->getLoc(), *res_int32_tensor_type_or, lhs_int32_tensor,
        rhs_int32_tensor, nullptr);
    Value res_int32 = rewriter.create<chlo::BroadcastSubOp>(
        op->getLoc(), *res_int32_tensor_type_or, add_result, zero_point,
        nullptr);

    // Clamp results by [quantization_min, quantization_max].
    res_int32 = rewriter.create<mhlo::ClampOp>(
        op->getLoc(), *res_int32_tensor_type_or, result_quantization_min,
        res_int32, result_quantization_max);

    // Convert results back to result storage type.
    auto res_final_tensor_type_or =
        GetSameShapeTensorType(op, res_int32_tensor_type_or->cast<TensorType>(),
                               result_element_type.getStorageType(), rewriter);
    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(op, *res_final_tensor_type_or,
                                                 res_int32);
    return success();
  }
};

// Performs conversion of MHLO quant ops to primitive ops.
void ConvertMHLOQuantToInt::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *context = op->getContext();
  RewritePatternSet patterns(context);

  // Populate MHLO quant ops conversion patterns.
  patterns.add<ConvertUniformQuantizeOp, ConvertUniformDequantizeOp,
               ConvertUniformQuantizedAddOp>(context);

  ConversionTarget target(*op->getContext());
  // An addDynamicallyLegalDialect callback that declares a given operation as
  // legal only if its all operands and results are non-quantized types.
  auto is_legal = [](Operation *op) {
    auto is_not_quant = [](Type type) {
      return !getElementTypeOrSelf(type).isa<quant::UniformQuantizedType>();
    };
    return llvm::all_of(op->getOperandTypes(), is_not_quant) &&
           llvm::all_of(op->getResultTypes(), is_not_quant);
  };
  target.addDynamicallyLegalDialect<MhloDialect>(is_legal);
  target.addDynamicallyLegalDialect<chlo::ChloDialect>(is_legal);

  LogicalResult result =
      applyPartialConversion(op, target, std::move(patterns));
  if (failed(result)) {
    signalPassFailure();
  }
}

}  // end namespace

std::unique_ptr<OperationPass<func::FuncOp>> createConvertMHLOQuantToIntPass() {
  return std::make_unique<ConvertMHLOQuantToInt>();
}

}  // end namespace mhlo
}  // end namespace mlir
