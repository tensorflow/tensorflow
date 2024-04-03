/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>

#include "absl/algorithm/container.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project  // NOLINT: Required to register quantization dialect.
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/common/uniform_quantized_types.h"

#define DEBUG_TYPE "stablehlo-compose-uniform-quantized-type"

namespace mlir {
namespace odml {
namespace {

using ::mlir::quant::CreateI8F32UniformQuantizedPerAxisType;
using ::mlir::quant::CreateI8F32UniformQuantizedType;
using ::mlir::quant::TryCast;
using ::mlir::quant::UniformQuantizedPerAxisType;
using ::mlir::quant::UniformQuantizedType;

#define GEN_PASS_DEF_COMPOSEUNIFORMQUANTIZEDTYPEPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h.inc"

// These strings are used to identify the uniform_quantize / uniform_dequantize
// functions.
constexpr StringRef kUniformQuantizeFunctionNameSubstring = "uniform_quantize";
constexpr StringRef kUniformDequantizeFunctionNameSubstring =
    "uniform_dequantize";

class ComposeUniformQuantizedTypePass
    : public impl::ComposeUniformQuantizedTypePassBase<
          ComposeUniformQuantizedTypePass> {
 private:
  void runOnOperation() override;
};

// Tests whether a `stablehlo::ConvertOp` is a i8 -> f32 cast.
bool IsI8ToF32Cast(stablehlo::ConvertOp convert_op) {
  const bool is_i8_operand =
      convert_op.getOperand().getType().getElementType().isInteger(/*width=*/8);
  const bool is_f32_result =
      convert_op.getResult().getType().getElementType().isa<Float32Type>();
  return is_i8_operand && is_f32_result;
}

// Tests whether a `stablehlo::ConvertOp` is a i8 -> i32 cast.
bool IsI8ToI32Cast(stablehlo::ConvertOp convert_op) {
  const bool is_i8_operand =
      convert_op.getOperand().getType().getElementType().isInteger(/*width=*/8);
  const bool is_i32_result =
      convert_op.getResult().getType().getElementType().isInteger(/*width=*/32);
  return is_i8_operand && is_i32_result;
}

// Tests whether a `stablehlo::ConvertOp` is a i32 -> f32 cast.
bool IsI32ToF32Cast(stablehlo::ConvertOp convert_op) {
  const bool is_i32_operand =
      convert_op.getOperand().getType().getElementType().isInteger(
          /*width=*/32);
  const bool is_f32_result =
      convert_op.getResult().getType().getElementType().isa<Float32Type>();
  return is_i32_operand && is_f32_result;
}

// Matches the zero points operand for the uniform_quantize and
// uniform_dequantize functions. Returns `failure()` if it doesn't match.
LogicalResult MatchZeroPointsOperand(Value zero_points) {
  if (!zero_points) {
    LLVM_DEBUG(llvm::dbgs() << "Zero point value is empty.\n");
    return failure();
  }

  auto zero_points_type = zero_points.getType().dyn_cast_or_null<TensorType>();
  if (!zero_points_type) {
    LLVM_DEBUG(llvm::dbgs() << "Zero point value should be a tensor type. Got: "
                            << zero_points_type << ".\n");
    return failure();
  }

  if (Type zero_points_element_type = zero_points_type.getElementType();
      !zero_points_element_type.isa<IntegerType>()) {
    LLVM_DEBUG(llvm::dbgs() << "Zero point should be an integer type. Got: "
                            << zero_points_element_type << ".\n");
    return failure();
  }

  if (zero_points_type.getNumElements() != 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "Zero point should contain a single element. Has: "
               << zero_points_type.getNumElements() << ".\n");
    return failure();
  }

  auto zero_point_constant_op =
      dyn_cast_or_null<stablehlo::ConstantOp>(zero_points.getDefiningOp());
  if (!zero_point_constant_op) {
    LLVM_DEBUG(llvm::dbgs()
               << "Zero point should be a constant. Instead it is defined by: "
               << zero_point_constant_op << ".\n");
    return failure();
  }

  return success();
}

// Matches the inverse scales operand for the uniform_quantize and
// uniform_dequantize functions. Returns `failure()` if it doesn't match.
LogicalResult MatchInverseScalesOperand(Value inverse_scales) {
  if (!inverse_scales) {
    LLVM_DEBUG(llvm::dbgs() << "Inverse scales value is empty.\n");
    return failure();
  }

  auto inverse_scales_type =
      inverse_scales.getType().dyn_cast_or_null<TensorType>();
  if (!inverse_scales_type) {
    LLVM_DEBUG(llvm::dbgs() << "Inverse scales should be a tensor type. Got: "
                            << inverse_scales_type << ".\n");
    return failure();
  }

  if (Type inverse_scales_element_type = inverse_scales_type.getElementType();
      !inverse_scales_element_type.isa<FloatType>()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Inverse scales element should be a float type. Got: "
               << inverse_scales_element_type << ".\n");
    return failure();
  }

  if (inverse_scales_type.getNumElements() != 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "Inverse scales should contain only one element. Has: "
               << inverse_scales_type.getNumElements() << ".\n");
    return failure();
  }

  auto inverse_scale_constant_op =
      dyn_cast_or_null<stablehlo::ConstantOp>(inverse_scales.getDefiningOp());
  if (!inverse_scale_constant_op) {
    llvm::dbgs()
        << "Inverse scales should be a constant. Instead, it was defined by: "
        << inverse_scale_constant_op << ".\n";
    return failure();
  }

  return success();
}

// Matches the following pattern that represents uniform quantization:
//
// `call @uniform_quantize(%input, %inverse_scale, %zero_point)`
//
// Provides helper functions to access the operands and the callee.
class UniformQuantizeFunctionCallPattern {
 public:
  // Returns Failure if it doesn't match. Returns the "wrapper" for the uniform
  // dequantization function call pattern when matched.
  static FailureOr<UniformQuantizeFunctionCallPattern> Match(
      func::CallOp call_op) {
    if (!call_op.getCallee().contains(kUniformQuantizeFunctionNameSubstring)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match @uniform_quantize function call pattern. "
                    "The name doesn't contain uniform_quantize.\n");
      return failure();
    }

    Value input_value = call_op.getOperand(0);
    if (!input_value) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match @uniform_quantize function call pattern. "
                    "Input value is empty.\n");
      return failure();
    }

    auto input_value_type =
        input_value.getType().dyn_cast_or_null<TensorType>();
    if (!input_value_type) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match @uniform_quantize function call pattern. "
                    "Input value's type must be a TensorType.\n");
      return failure();
    }

    if (Type input_element_type = input_value_type.getElementType();
        !input_element_type.isa<FloatType>()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match @uniform_quantize function call pattern. "
                    "Input value's element type must be a float. Got: "
                 << input_element_type << ".\n");
      return failure();
    }

    if (failed(MatchInverseScalesOperand(call_op.getOperand(1)))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match the inverse scales operand of "
                    "the @uniform_quantize call pattern.\n");
      return failure();
    }

    if (failed(MatchZeroPointsOperand(call_op.getOperand(2)))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match the zero point operand of "
                                 "the @uniform_quantize call pattern.\n");
      return failure();
    }

    return UniformQuantizeFunctionCallPattern(call_op);
  }

  Value GetInputValue() { return call_op_.getOperand(0); }

  Value GetInverseScalesValue() { return call_op_.getOperand(1); }

  Value GetZeroPointsValue() { return call_op_.getOperand(2); }

  stablehlo::ConstantOp GetZeroPointsConstantOp() {
    return cast<stablehlo::ConstantOp>(GetZeroPointsValue().getDefiningOp());
  }

  stablehlo::ConstantOp GetInverseScalesConstantOp() {
    return cast<stablehlo::ConstantOp>(GetInverseScalesValue().getDefiningOp());
  }

  ElementsAttr GetZeroPointsValueAttr() {
    return GetZeroPointsConstantOp().getValue();
  }

  ElementsAttr GetInverseScalesValueAttr() {
    return GetInverseScalesConstantOp().getValue();
  }

  func::CallOp GetCallOp() { return call_op_; }

  FlatSymbolRefAttr GetFunction() { return call_op_.getCalleeAttr(); }

 private:
  explicit UniformQuantizeFunctionCallPattern(func::CallOp call_op)
      : call_op_(call_op) {}

  func::CallOp call_op_;
};

// Matches the following pattern that represents uniform dequantization.
//
// `call @uniform_dequantize(%input, %zero_point, %inverse_scale)`
//
// Provides helper functions to access the operands and the callee.
class UniformDequantizeFunctionCallPattern {
 public:
  // Returns Failure if it doesn't match. Returns the "wrapper" for the uniform
  // dequantization function call pattern when matched.
  static FailureOr<UniformDequantizeFunctionCallPattern> Match(
      func::CallOp call_op) {
    if (!call_op.getCallee().contains(
            kUniformDequantizeFunctionNameSubstring)) {
      llvm::dbgs() << "Failed to match uniformDequantizeCallOp - name doesn't "
                      "contain uniform_quantize.\n";
      return failure();
    }

    Value input_value = call_op.getOperand(0);
    if (!input_value) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match @uniform_dequantize call "
                                 "pattern. Input value is empty.\n");
      return failure();
    }

    auto input_value_type =
        input_value.getType().dyn_cast_or_null<TensorType>();
    if (!input_value_type) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match @uniform_dequantize call pattern. Input "
                    "value's type must be a TensorType. Got:"
                 << input_value_type << ".\n");
      return failure();
    }

    if (Type input_element_type = input_value_type.getElementType();
        !input_element_type.isa<IntegerType>()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match @uniform_dequantize call pattern. Input "
                    "value's element type must be integer. Got: "
                 << input_element_type << ".\n");
      return failure();
    }

    if (failed(MatchInverseScalesOperand(call_op.getOperand(1)))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match the inverse scales operand of "
                    "the @uniform_dequantize call pattern.\n");
      return failure();
    }

    if (failed(MatchZeroPointsOperand(call_op.getOperand(2)))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match the zero point operand of "
                                 "the @uniform_dequantize call pattern.\n");
      return failure();
    }

    return UniformDequantizeFunctionCallPattern(call_op);
  }

  Value GetInputValue() { return call_op_.getOperand(0); }

  Value GetInverseScalesValue() { return call_op_.getOperand(1); }

  Value GetZeroPointsValue() { return call_op_.getOperand(2); }

  stablehlo::ConstantOp GetZeroPointsConstantOp() {
    return cast<stablehlo::ConstantOp>(GetZeroPointsValue().getDefiningOp());
  }

  stablehlo::ConstantOp GetInverseScalesConstantOp() {
    return cast<stablehlo::ConstantOp>(GetInverseScalesValue().getDefiningOp());
  }

  ElementsAttr GetZeroPointsValueAttr() {
    return GetZeroPointsConstantOp().getValue();
  }

  ElementsAttr GetInverseScalesValueAttr() {
    return GetInverseScalesConstantOp().getValue();
  }

  func::CallOp GetCallOp() { return call_op_; }

  FlatSymbolRefAttr GetFunction() { return call_op_.getCalleeAttr(); }

 private:
  explicit UniformDequantizeFunctionCallPattern(func::CallOp call_op)
      : call_op_(call_op) {}

  func::CallOp call_op_;
};

// Matches the pattern for quantized convolution op and rewrites it to use
// uniform quantized types.
//
// Currently assumes asymmetric per-tensor quantization for activations and
// symmetric per-channel quantization for filters.
//
// This pattern represents the following derived equation, where:
// * rn = real (expressed) value for tensor n
// * qn = quantized value for tensor n
// * sn = scale for tensor n
// * zn = zero point for tensor n
//
// r3 = r1 * r2
//    = s1 (q1 - z1) * s2 (q2 - z2)
//    = s1 s2 (q1 q2 - q1 z2 - q2 z1 + z1 z2)
//
// * z2 is zero, because it assumes symmetric quantization for the filter:
//
//    = s1 s2 (q1 q2 - q2 z1)
//
// In StableHLO text representation, the pattern is as the following
// (simplified):
//
// ```
// %0 = // Input tensor r1.
// %1 = stablehlo.constant  // Input inverse scale 1 / s1.
// %2 = stablehlo.constant  // Input zero point z1.
// %3 = call @uniform_quantize(%0, %1, %2)  // Quantize input (q1).
// %4 = stablehlo.convert %3  // i8 -> f32 cast trick for input.
// %5 = stablehlo.constant  // Quantized filter tensor q2.
// %6 = stablehlo.convert %5  // Optional: i8 -> f32 cast trick for filter.
// %7 = stablehlo.convolution(%4, %6)  // q1 * q2 (disguised in f32).
// %8 = stablehlo.reshape %2  // z1
// %9 = stablehlo.broadcast_in_dim %8
// %10 = stablehlo.convert %9  // i8 -> f32 cast trick for z1.
// %11 = stablehlo.convert %5  // i8 -> f32 cast trick for filter.
// %12 = stablehlo.convolution(%10, %11)  // q2 * z1
// %13 = stablehlo.subtract %7, %12  // q1 * q2 - q2 * z1
// %14 = stablehlo.constant  // Merged scale s1 * s2, precalculated.
// %15 = stablehlo.broadcast_in_dim %14
// %16 = stablehlo.multiply %13 %15  // r3 = s1 s2 (q1 q2 - q2 z1)
//
// The following quant -> dequant pattern is a no-op, but is required to
// retrieve the quantization parameters for the output tensor.
//
// %17 = stablehlo.constant  // Output inverse scale 1 / s3.
// %18 = stablehlo.constant  // Output zero point z3.
// %19 = call @uniform_quantize_0(%16, %17, %18)  // r3 -> q3
// %20 = call @uniform_dequantize(%19, %17, %18)  // q3 -> r3
// ```
//
// The rewritten pattern looks like:
//
// ```
// %1 = stablehlo.uniform_quantize %0  // Input f32 -> uniform quantized type.
// %2 = stablehlo.constant  // Filter uniform quantized type.
// %3 = stablehlo.convolution(%1, %2)   // In uniform quantized type.
// %4 = stablehlo.uniform_dequantize %3  // Dequantize the output.
// ```
class ComposeUniformQuantizedConvolutionOp
    : public OpRewritePattern<stablehlo::ConvolutionOp> {
 public:
  using OpRewritePattern<stablehlo::ConvolutionOp>::OpRewritePattern;

  LogicalResult match(stablehlo::ConvolutionOp op) const final {
    // Verify operands' types.
    for (Type operand_type : op.getOperandTypes()) {
      if (Type element_type = operand_type.cast<TensorType>().getElementType();
          !element_type.isa<Float32Type>()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Failed to match. The operand type must be a float. Got: "
                   << element_type << ".\n");
        return failure();
      }
    }

    // Match the subgraph for the input operand.
    auto input_i8_to_f32_convert_op = dyn_cast_or_null<stablehlo::ConvertOp>(
        op.getOperand(0).getDefiningOp());
    if (!input_i8_to_f32_convert_op) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match. Input is not defined by a "
                                 "stablehlo::ConvertOp.\n");
      return failure();
    }

    if (!IsI8ToF32Cast(input_i8_to_f32_convert_op)) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Failed to match. The ConvertOp is not an i8->f32 type cast.\n");
      return failure();
    }

    auto uniform_quantize_call_op = dyn_cast_or_null<func::CallOp>(
        input_i8_to_f32_convert_op.getOperand().getDefiningOp());
    if (!uniform_quantize_call_op) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match: Input is not quantized by a "
                                 "uniform_quantize function.\n");
      return failure();
    }

    auto uniform_quantize_call_pattern_for_input =
        UniformQuantizeFunctionCallPattern::Match(uniform_quantize_call_op);
    if (failed(uniform_quantize_call_pattern_for_input)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match uniform_quantized call op for input.\n");
      return failure();
    }

    // Match the subgraph that receives the convolution output.
    Value conv_output_value = op.getResult();
    if (auto output_element_type =
            conv_output_value.getType().cast<TensorType>().getElementType();
        !output_element_type.isa<FloatType>()) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Failed to match. Output type is expected to be a float. Got: "
          << output_element_type << ".\n");
      return failure();
    }

    if (!op->hasOneUse()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match op - doesn't have a single use.\n");
      return failure();
    }

    auto subtract_op = dyn_cast_or_null<stablehlo::SubtractOp>(
        *conv_output_value.user_begin());
    if (!subtract_op) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match subtract_op.\n");
      return failure();
    }
    if (!subtract_op->hasOneUse()) {
      llvm::dbgs() << "Failed to match op - doesn't have a single use.\n";
      return failure();
    }

    // This convolution represents the "q2 * z1" term, which is subtraced from
    // the first convolution result.
    auto other_conv_op = dyn_cast_or_null<stablehlo::ConvolutionOp>(
        subtract_op.getOperand(1).getDefiningOp());
    if (!other_conv_op) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match other_conv_op.\n");
      return failure();
    }

    // The filter of the convolution may have a `ConvertOp` after the constant
    // op.
    if (!isa<stablehlo::ConstantOp>(
            other_conv_op.getOperand(1).getDefiningOp()) &&
        !isa<stablehlo::ConvertOp>(
            other_conv_op.getOperand(1).getDefiningOp())) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match filter of other_conv_op.\n");
      return failure();
    }

    auto other_zp_i8_to_f32_convert_op = dyn_cast_or_null<stablehlo::ConvertOp>(
        other_conv_op.getOperand(0).getDefiningOp());
    if (!other_zp_i8_to_f32_convert_op) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match other_zp_i8_to_f32_convert_op.\n");
      return failure();
    }

    if (!(other_zp_i8_to_f32_convert_op.getResult()
              .getType()
              .getElementType()
              .isa<Float32Type>() &&
          other_zp_i8_to_f32_convert_op.getOperand()
              .getType()
              .getElementType()
              .isa<IntegerType>())) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Failed to match. The ConvertOp is not an i8->f32 type cast.\n");
      return failure();
    }

    auto other_zp_broadcast_in_dim_op =
        dyn_cast_or_null<stablehlo::BroadcastInDimOp>(
            other_zp_i8_to_f32_convert_op.getOperand().getDefiningOp());
    if (!other_zp_broadcast_in_dim_op) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match other_zp_broadcast_in_dim_op.\n");
      return failure();
    }

    auto other_zp_reshape_op = dyn_cast_or_null<stablehlo::ReshapeOp>(
        other_zp_broadcast_in_dim_op.getOperand().getDefiningOp());
    if (!other_zp_reshape_op) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match other_zp_reshape_op.\n");
      return failure();
    }

    auto other_input_zero_points_constant_op =
        dyn_cast_or_null<stablehlo::ConstantOp>(
            other_zp_reshape_op.getOperand().getDefiningOp());
    if (!other_input_zero_points_constant_op) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match other_input_zero_points_constant_op.\n");
      return failure();
    }

    auto combined_scale_multiply_op = dyn_cast_or_null<stablehlo::MulOp>(
        *subtract_op.getResult().user_begin());
    if (!combined_scale_multiply_op) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match combined_scale_multiply_op.\n");
      return failure();
    }
    if (!combined_scale_multiply_op->hasOneUse()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match op - doesn't have a single use.\n");
      return failure();
    }

    auto scale_combined_broadcast_in_dim_op =
        dyn_cast_or_null<stablehlo::BroadcastInDimOp>(
            combined_scale_multiply_op.getOperand(1).getDefiningOp());
    if (!scale_combined_broadcast_in_dim_op) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match scale_combined_broadcast_in_dim_op.\n");
      return failure();
    }

    // s1 * s2
    auto combined_scale_constant_op = dyn_cast_or_null<stablehlo::ConstantOp>(
        scale_combined_broadcast_in_dim_op.getOperand().getDefiningOp());
    if (!combined_scale_constant_op) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match combined_scale_constant_op.\n");
      return failure();
    }

    // Quantize -> Dequantize following r3.
    auto output_uniform_quantize_call_op = dyn_cast_or_null<func::CallOp>(
        *combined_scale_multiply_op.getResult().user_begin());
    if (!output_uniform_quantize_call_op->hasOneUse()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match op - doesn't have a single use.\n");
      return failure();
    }

    if (failed(UniformQuantizeFunctionCallPattern::Match(
            output_uniform_quantize_call_op))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match op output_uniform_quantize_call_op\n");
      return failure();
    }

    auto output_uniform_dequantize_call_op = dyn_cast_or_null<func::CallOp>(
        *output_uniform_quantize_call_op.getResult(0).user_begin());
    if (!output_uniform_dequantize_call_op) {
      return failure();
    }
    if (failed(UniformDequantizeFunctionCallPattern::Match(
            output_uniform_dequantize_call_op))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match output_uniform_dequantize_call_op.\n");
      return failure();
    }

    // The filter of the convolution may have a `ConvertOp` after the constant
    // op.
    Operation* filter_op = op.getOperand(1).getDefiningOp();
    if (!isa<stablehlo::ConstantOp>(filter_op) &&
        !(isa<stablehlo::ConvertOp>(filter_op) &&
          isa<stablehlo::ConstantOp>(
              filter_op->getOperand(0).getDefiningOp()))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match filter_constant_op.\n");
      return failure();
    }

    return success();
  }

  void rewrite(stablehlo::ConvolutionOp op,
               PatternRewriter& rewriter) const final {
    // Rewrite `call @uniform_quantize` -> `stablehlo.uniform_quantize`.
    auto input_i8_to_f32_convert_op =
        cast<stablehlo::ConvertOp>(op.getOperand(0).getDefiningOp());
    auto uniform_quantize_call_op = cast<func::CallOp>(
        input_i8_to_f32_convert_op.getOperand().getDefiningOp());

    auto uniform_quantize_call_pattern_for_input =
        *UniformQuantizeFunctionCallPattern::Match(uniform_quantize_call_op);
    const double input_inverse_scales_value =
        uniform_quantize_call_pattern_for_input.GetInverseScalesValueAttr()
            .getSplatValue<APFloat>()
            .convertToFloat();
    const double input_scale_value = 1.0 / input_inverse_scales_value;
    const int64_t input_zero_point_value =
        uniform_quantize_call_pattern_for_input.GetZeroPointsValueAttr()
            .getSplatValue<APInt>()
            .getSExtValue();

    Value input_value = uniform_quantize_call_pattern_for_input.GetInputValue();
    UniformQuantizedType input_quantized_element_type =
        CreateI8F32UniformQuantizedType(
            uniform_quantize_call_op.getLoc(), *rewriter.getContext(),
            input_scale_value, input_zero_point_value);
    auto input_uniform_quantize_op =
        rewriter.create<stablehlo::UniformQuantizeOp>(
            uniform_quantize_call_op.getLoc(),
            /*result=*/
            input_value.getType().cast<TensorType>().clone(
                input_quantized_element_type),
            /*operand=*/input_value);

    rewriter.replaceAllUsesWith(input_i8_to_f32_convert_op.getResult(),
                                input_uniform_quantize_op.getResult());

    // Rewrite filter constant.
    Operation* filter_op = op.getOperand(1).getDefiningOp();

    // Retrieve the i8 filter values.
    DenseElementsAttr filter_i8_value_attr = nullptr;
    if (auto filter_constant_op =
            dyn_cast_or_null<stablehlo::ConstantOp>(filter_op);
        filter_constant_op) {
      // This is i8 values disguised as f32 (due to the upcast trick). Simply
      // cast them to i8.
      ElementsAttr filter_value = filter_constant_op.getValue();
      filter_i8_value_attr = filter_value.cast<DenseFPElementsAttr>().mapValues(
          rewriter.getI8Type(), [](const APFloat& val) -> APInt {
            APSInt convertedInt(/*BitWidth=*/8, /*isUnsigned=*/false);
            bool ignored;
            val.convertToInteger(convertedInt, APFloat::rmTowardZero, &ignored);
            return convertedInt;
          });
    } else if (isa<stablehlo::ConvertOp>(filter_op) &&
               isa<stablehlo::ConstantOp>(
                   filter_op->getOperand(0).getDefiningOp())) {
      filter_i8_value_attr =
          cast<stablehlo::ConstantOp>(filter_op->getOperand(0).getDefiningOp())
              .getValue()
              .cast<DenseIntElementsAttr>();
    }

    // Create Uniform Quantized constant for the filter.
    auto subtract_op =
        cast<stablehlo::SubtractOp>(*op.getResult().user_begin());
    auto other_conv_op = cast<stablehlo::ConvolutionOp>(
        subtract_op.getOperand(1).getDefiningOp());
    auto combined_scale_multiply_op =
        cast<stablehlo::MulOp>(*subtract_op.getResult().user_begin());

    auto scale_combined_broadcast_in_dim_op = cast<stablehlo::BroadcastInDimOp>(
        combined_scale_multiply_op.getOperand(1).getDefiningOp());
    auto combined_scale_constant_op = cast<stablehlo::ConstantOp>(
        scale_combined_broadcast_in_dim_op.getOperand().getDefiningOp());

    SmallVector<double> filter_scale_values;
    for (const auto combined_scale_value : combined_scale_constant_op.getValue()
                                               .cast<DenseFPElementsAttr>()
                                               .getValues<float>()) {
      // UniformQuantizedPerAxisType requires scales to have double dtype.
      const double filter_scale_value = static_cast<double>(
          combined_scale_value * input_inverse_scales_value);
      filter_scale_values.emplace_back(filter_scale_value);
    }

    // Assumes it is symmetric.
    SmallVector<int64_t> filter_zero_point_values(
        /*Size=*/filter_scale_values.size(), /*Value=*/0);

    // Use quantization dimension = 3 that corresponds to the output channel
    // dimension, assuming the filter format is `[0, 1, i, o]`.
    // TODO: b/291029962 - Lift the assumption above and retrieve the
    // quantization dimension from the `dimension_numbers` attribute.
    UniformQuantizedPerAxisType filter_quantized_element_type =
        CreateI8F32UniformQuantizedPerAxisType(
            filter_op->getLoc(), *rewriter.getContext(), filter_scale_values,
            filter_zero_point_values,
            /*quantization_dimension=*/3);

    // Create a new constant op for the filter in i8.
    auto quantized_filter_constant_op = rewriter.create<stablehlo::ConstantOp>(
        filter_op->getLoc(),
        /*output=*/
        filter_i8_value_attr.getType().clone(filter_quantized_element_type),
        /*value=*/filter_i8_value_attr);

    // Replace filter uses with uniform quantized filter.
    rewriter.replaceAllUsesWith(filter_op->getResult(0),
                                quantized_filter_constant_op.getResult());

    // Replace conv op with a new convolution op that has quantized output type.
    // Quantize -> Dequantize following r3.
    auto output_uniform_quantize_call_op = cast<func::CallOp>(
        *combined_scale_multiply_op.getResult().user_begin());

    auto output_uniform_quantize_call_pattern =
        *UniformQuantizeFunctionCallPattern::Match(
            output_uniform_quantize_call_op);

    const int output_zero_point_value =
        output_uniform_quantize_call_pattern.GetZeroPointsValueAttr()
            .getSplatValue<APInt>()
            .getSExtValue();
    const double output_inverse_scale_value =
        output_uniform_quantize_call_pattern.GetInverseScalesValueAttr()
            .getSplatValue<APFloat>()
            .convertToFloat();

    UniformQuantizedType output_uniform_quantized_type =
        CreateI8F32UniformQuantizedType(
            output_uniform_quantize_call_op.getLoc(), *rewriter.getContext(),
            /*scale=*/1.0 / output_inverse_scale_value,
            output_zero_point_value);

    Value conv_output_value = op.getResult();
    auto output_uniform_quantized_tensor_type = RankedTensorType::getChecked(
        rewriter.getUnknownLoc(),
        /*shape=*/conv_output_value.getType().cast<TensorType>().getShape(),
        output_uniform_quantized_type);

    SmallVector<Type> new_conv_output_types = {
        output_uniform_quantized_tensor_type};
    auto new_conv_op_with_output_type =
        rewriter.create<stablehlo::ConvolutionOp>(
            op.getLoc(), new_conv_output_types, op.getOperands(),
            op->getAttrs());

    rewriter.replaceAllUsesWith(op.getResult(),
                                new_conv_op_with_output_type.getResult());

    auto new_output_dequant_op =
        rewriter.create<stablehlo::UniformDequantizeOp>(
            rewriter.getUnknownLoc(),
            /*operand=*/new_conv_op_with_output_type);

    auto output_uniform_dequantize_call_op = cast<func::CallOp>(
        *output_uniform_quantize_call_op.getResult(0).user_begin());

    rewriter.replaceAllUsesWith(output_uniform_dequantize_call_op.getResult(0),
                                new_output_dequant_op.getResult());

    // Erase unused ops in the reverse order.
    rewriter.eraseOp(output_uniform_dequantize_call_op);
    rewriter.eraseOp(output_uniform_quantize_call_op);
    rewriter.eraseOp(combined_scale_multiply_op);
    rewriter.eraseOp(subtract_op);
    rewriter.eraseOp(other_conv_op);
    rewriter.eraseOp(op);
    rewriter.eraseOp(input_i8_to_f32_convert_op);
    rewriter.eraseOp(uniform_quantize_call_op);
  }
};

// Matches the pattern for quantized dot_general op and rewrites it to use
// uniform quantized types.
//
// Currently assumes asymmetric per-tensor quantization for activations and
// symmetric per-channel quantization for filters.
//
// This pattern represents the following derived equation, where:
// * rn = real (expressed) value for tensor n
// * qn = quantized value for tensor n
// * sn = scale for tensor n
// * zn = zero point for tensor n
//
// r3 = r1 * r2
//    = s1 (q1 - z1) * s2 (q2 - z2)
//    = s1 s2 (q1 q2 - q1 z2 - q2 z1 + z1 z2)
//
// * z2 is zero, because it assumes symmetric quantization for the filter:
//
//    = s1 s2 (q1 q2 - q2 z1)
//
// In StableHLO text representation, the pattern is as the following
// (simplified):
//
// ```
// %0 = // Input tensor r1.
// %1 = stablehlo.constant  // Input inverse scale 1 / s1.
// %2 = stablehlo.constant  // Input zero point z1.
// %3 = stablehlo.constant  // Quantized filter tensor q2.
// %4 = stablehlo.constant  // Precalculated q2 * z1
// %5 = stablehlo.constant  // Merged scale s1 * s2, precalculated.
// %6 = call @uniform_quantize(%0, %1, %2)  // Quantize input (q1).
// %7 = stablehlo.convert %6  // i8 -> f32 cast trick for input.
// %8 = stablehlo.convert %3  // i8 -> f32 cast trick for filter, optional.
// %9 = stablehlo.dot_general(%7, %8)  // q1 * q2 (disguised in f32).
// %10 = stablehlo.convert %4  // i32 -> f32 cast for q2 * z1.
// %11 = stablehlo.broadcast_in_dim %10  // Optional.
// %12 = stablehlo.subtract %9, %11  // q1 * q2 - q2 * z1
// %13 = stablehlo.broadcast_in_dim %5  // Optional.
// %14 = stablehlo.multiply %12 %13  // r3 = s1 s2 (q1 q2 - q2 z1)
//
// The following quant -> dequant pattern is a no-op, but is required to
// retrieve the quantization parameters for the output tensor.
//
// %15 = stablehlo.constant  // Output inverse scale 1 / s3.
// %16 = stablehlo.constant  // Output zero point z3.
// %17 = call @uniform_quantize_0(%15, %15, %16)  // r3 -> q3
// %18 = call @uniform_dequantize(%17, %15, %16)  // q3 -> r3
// ```
//
// The rewritten pattern looks like:
//
// ```
// %1 = stablehlo.uniform_quantize %0  // Input f32 -> uniform quantized type.
// %2 = stablehlo.constant  // Filter uniform quantized type.
// %3 = stablehlo.dot_general(%1, %2)   // In uniform quantized type.
// %4 = stablehlo.uniform_dequantize %3  // Dequantize the output.
// ```
//
// Note that the i8->f32 cast trick for the filter (%8) is optional. When the
// cast isn't present, the filter constant (%3) should be i8 quantized values
// disguised in f32.
class ComposeUniformQuantizedDotGeneralOp
    : public OpRewritePattern<stablehlo::DotGeneralOp> {
 public:
  using OpRewritePattern<stablehlo::DotGeneralOp>::OpRewritePattern;
  LogicalResult match(stablehlo::DotGeneralOp op) const final {
    auto input_i8_to_f32_convert_op =
        TryCast<stablehlo::ConvertOp>(op.getOperand(0).getDefiningOp(),
                                      /*name=*/"input_i8_to_f32_convert_op");
    if (failed(input_i8_to_f32_convert_op)) return failure();

    if (!IsI8ToF32Cast(*input_i8_to_f32_convert_op)) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match input_i8_to_f32_convert_op. "
                                 "It should be a i8->f32 cast.\n");
      return failure();
    }

    if (failed(MatchFilter(op.getOperand(1)))) return failure();

    auto input_quantize_call_op = TryCast<func::CallOp>(
        input_i8_to_f32_convert_op->getOperand().getDefiningOp(),
        /*name=*/"input_quantize_call_op");
    if (failed(input_quantize_call_op)) return failure();

    // Uniform quantization pattern found for the input DotGeneralOp input.
    auto input_uniform_quantize_call_pattern =
        UniformQuantizeFunctionCallPattern::Match(*input_quantize_call_op);
    if (failed(input_uniform_quantize_call_pattern)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match input uniform quantize call pattern.\n");
      return failure();
    }

    auto subtract_op = TryCast<stablehlo::SubtractOp>(
        *op.getResult().user_begin(), /*name=*/"subtract_op");
    if (failed(subtract_op)) return failure();

    Value subtract_op_second_operand = subtract_op->getOperand(1);
    if (auto broadcast_in_dim_op = TryCast<stablehlo::BroadcastInDimOp>(
            subtract_op_second_operand.getDefiningOp(),
            /*name=*/"broadcast_in_dim_for_subtract_op_operand");
        succeeded(broadcast_in_dim_op)) {
      subtract_op_second_operand = broadcast_in_dim_op->getOperand();
    }

    auto input_zp_filter_prod_convert_op = TryCast<stablehlo::ConvertOp>(
        subtract_op_second_operand.getDefiningOp(),
        /*name=*/"input_zp_filter_prod_convert_op");
    if (failed(input_zp_filter_prod_convert_op)) return failure();

    if (!IsI32ToF32Cast(*input_zp_filter_prod_convert_op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match input_zp_filter_prod_convert_op. "
                    "It should be a i32->f32 cast.\n");
      return failure();
    }

    // z1 * q2
    auto input_zp_filter_prod_constant_op = TryCast<stablehlo::ConstantOp>(
        input_zp_filter_prod_convert_op->getOperand().getDefiningOp(),
        /*name=*/"input_zp_filter_prod_constant_op");
    if (failed(input_zp_filter_prod_constant_op)) return failure();

    auto multiply_op = TryCast<stablehlo::MulOp>(
        *subtract_op->getResult().user_begin(), /*name=*/"multiply_op");
    if (failed(multiply_op)) return failure();

    Value multiply_op_second_operand = multiply_op->getOperand(1);
    if (auto broadcast_in_dim_op = TryCast<stablehlo::BroadcastInDimOp>(
            multiply_op_second_operand.getDefiningOp(),
            /*name=*/"broadcast_in_dim_for_multiply_op_operand");
        succeeded(broadcast_in_dim_op)) {
      multiply_op_second_operand = broadcast_in_dim_op->getOperand();
    }

    auto merged_scale_constant_op = TryCast<stablehlo::ConstantOp>(
        multiply_op_second_operand.getDefiningOp(),
        /*name=*/"merged_scale_constant_op");
    if (failed(merged_scale_constant_op)) return failure();

    auto output_uniform_quantize_call_op =
        TryCast<func::CallOp>(*multiply_op->getResult().user_begin(),
                              /*name=*/"output_quantize_call_op");
    if (failed(output_uniform_quantize_call_op)) return failure();

    auto output_uniform_quantize_call_pattern =
        UniformQuantizeFunctionCallPattern::Match(
            *output_uniform_quantize_call_op);
    if (failed(output_uniform_quantize_call_pattern)) {
      llvm::dbgs() << "Failed match uniform quantize call pattern.\n";
      return failure();
    }

    auto output_uniform_dequantize_call_op = TryCast<func::CallOp>(
        *output_uniform_quantize_call_op->getResult(0).user_begin(),
        /*name=*/"output_uniform_dequantize_call_op");
    if (failed(output_uniform_dequantize_call_op)) return failure();

    auto output_uniform_dequantize_call_pattern =
        UniformDequantizeFunctionCallPattern::Match(
            *output_uniform_dequantize_call_op);
    if (failed(output_uniform_dequantize_call_pattern)) {
      llvm::dbgs() << "Failed to match output uniform quantize call pattern.\n";
      return failure();
    }

    return success();
  }

  void rewrite(stablehlo::DotGeneralOp op,
               PatternRewriter& rewriter) const final {
    // Build uniform quantized type for input.
    auto input_i8_to_f32_convert_op =
        cast<stablehlo::ConvertOp>(op.getOperand(0).getDefiningOp());
    auto input_uniform_quantize_call_op = cast<func::CallOp>(
        input_i8_to_f32_convert_op.getOperand().getDefiningOp());
    auto input_uniform_quantize_call_pattern =
        UniformQuantizeFunctionCallPattern::Match(
            input_uniform_quantize_call_op);

    const float input_inverse_scale_value =
        input_uniform_quantize_call_pattern->GetInverseScalesValueAttr()
            .getSplatValue<APFloat>()
            .convertToFloat();
    const float input_scale_value = 1.0 / input_inverse_scale_value;

    const int8_t input_zero_point_value =
        input_uniform_quantize_call_pattern->GetZeroPointsValueAttr()
            .getSplatValue<APInt>()
            .getSExtValue();

    const UniformQuantizedType input_uniform_quantized_type =
        CreateI8F32UniformQuantizedType(
            input_uniform_quantize_call_op.getLoc(), *rewriter.getContext(),
            input_scale_value, input_zero_point_value);

    Value input_value = input_uniform_quantize_call_pattern->GetInputValue();
    auto input_uniform_quantize_op =
        rewriter.create<stablehlo::UniformQuantizeOp>(
            input_i8_to_f32_convert_op.getLoc(),
            /*result=*/
            input_value.getType().cast<TensorType>().clone(
                input_uniform_quantized_type),
            /*operand=*/input_value);

    rewriter.replaceAllUsesWith(input_i8_to_f32_convert_op.getResult(),
                                input_uniform_quantize_op.getResult());

    // Build uniform quantized type for filter.
    Value filter_value = op.getOperand(1);
    stablehlo::ConstantOp filter_constant_op =
        GetFilterConstantOp(filter_value);
    auto filter_value_attr =
        filter_constant_op.getValue().cast<DenseElementsAttr>();
    if (filter_value_attr.getElementType().isF32()) {
      // This is i8 values disguised as f32 (due to the upcast trick). Simply
      // cast them to i8.
      filter_value_attr =
          filter_value_attr.cast<DenseFPElementsAttr>().mapValues(
              rewriter.getI8Type(), [](const APFloat& val) -> APInt {
                APSInt converted_int(/*BitWidth=*/8, /*isUnsigned=*/false);
                bool ignored;
                val.convertToInteger(converted_int, APFloat::rmTowardZero,
                                     &ignored);
                return converted_int;
              });
    }

    auto subtract_op =
        cast<stablehlo::SubtractOp>(*op.getResult().user_begin());

    Value subtract_op_second_operand = subtract_op.getOperand(1);
    if (auto broadcast_in_dim_op =
            dyn_cast_or_null<stablehlo::BroadcastInDimOp>(
                subtract_op_second_operand.getDefiningOp());
        broadcast_in_dim_op) {
      // Ignore BroadcastInDimOp - it is optional.
      subtract_op_second_operand = broadcast_in_dim_op.getOperand();
    }

    auto multiply_op =
        cast<stablehlo::MulOp>(*subtract_op.getResult().user_begin());

    Value multiply_op_second_operand = multiply_op.getOperand(1);
    if (auto broadcast_in_dim_op =
            dyn_cast_or_null<stablehlo::BroadcastInDimOp>(
                multiply_op_second_operand.getDefiningOp());
        broadcast_in_dim_op) {
      // Ignore BroadcastInDimOp - it is optional.
      multiply_op_second_operand = broadcast_in_dim_op.getOperand();
    }

    // s1 * s2
    auto merged_scale_constant_op =
        cast<stablehlo::ConstantOp>(multiply_op_second_operand.getDefiningOp());
    SmallVector<double> filter_scale_values;
    for (const auto merged_scale : merged_scale_constant_op.getValue()
                                       .cast<DenseFPElementsAttr>()
                                       .getValues<float>()) {
      // (s1 * s2) * (1 / s1) = s2
      // UniformQuantizedPerAxisType requires scales to have double dtype.
      filter_scale_values.push_back(
          static_cast<double>(merged_scale * input_inverse_scale_value));
    }

    SmallVector<int64_t> filter_zero_point_values(
        /*Size=*/filter_scale_values.size(), /*Value=*/0);

    const int quantization_dimension = GetFilterQuantizationDimension(
        op.getDotDimensionNumbers(),
        filter_value_attr.getType().cast<TensorType>().getRank());
    const UniformQuantizedPerAxisType filter_uniform_quantized_type =
        CreateI8F32UniformQuantizedPerAxisType(
            filter_constant_op.getLoc(), *rewriter.getContext(),
            filter_scale_values, filter_zero_point_values,
            quantization_dimension);

    // Create a new constant op for the filter in i8.
    auto quantized_filter_constant_op = rewriter.create<stablehlo::ConstantOp>(
        filter_constant_op.getLoc(),
        /*output=*/
        filter_constant_op.getResult().getType().cast<TensorType>().clone(
            filter_uniform_quantized_type),
        /*value=*/filter_value_attr);

    rewriter.replaceAllUsesWith(filter_value,
                                quantized_filter_constant_op.getResult());

    // Recreate stablehlo::DotGeneralOp with a uniform quantized output type.
    auto output_uniform_quantize_call_op =
        cast<func::CallOp>(*multiply_op.getResult().user_begin());

    auto output_uniform_quantize_call_pattern =
        UniformQuantizeFunctionCallPattern::Match(
            output_uniform_quantize_call_op);

    auto output_uniform_dequantize_call_op = cast<func::CallOp>(
        *output_uniform_quantize_call_op.getResult(0).user_begin());

    auto output_uniform_dequantize_call_pattern =
        UniformDequantizeFunctionCallPattern::Match(
            output_uniform_dequantize_call_op);

    const auto inverse_output_scale_value =
        output_uniform_quantize_call_pattern->GetInverseScalesValueAttr()
            .getSplatValue<APFloat>()
            .convertToFloat();
    const float output_scale_value = 1.0 / inverse_output_scale_value;

    const int64_t output_zero_point_value =
        output_uniform_quantize_call_pattern->GetZeroPointsValueAttr()
            .getSplatValue<APInt>()
            .getSExtValue();

    const UniformQuantizedType output_uniform_quantized_type =
        CreateI8F32UniformQuantizedType(
            output_uniform_quantize_call_op.getLoc(), *rewriter.getContext(),
            output_scale_value, output_zero_point_value);

    auto new_dot_general_op = rewriter.create<stablehlo::DotGeneralOp>(
        op.getLoc(), /*resultType0=*/
        op.getResult().getType().cast<TensorType>().clone(
            output_uniform_quantized_type),
        /*lhs=*/op.getLhs(), /*rhs=*/op.getRhs(),
        /*dot_dimension_numbers=*/op.getDotDimensionNumbers(),
        /*precision_config=*/op.getPrecisionConfigAttr());

    rewriter.replaceAllUsesWith(op.getResult(), new_dot_general_op.getResult());

    auto new_output_dequant_op =
        rewriter.create<stablehlo::UniformDequantizeOp>(
            output_uniform_dequantize_call_op.getLoc(),
            /*operand=*/new_dot_general_op);

    rewriter.replaceAllUsesWith(output_uniform_dequantize_call_op.getResult(0),
                                new_output_dequant_op.getResult());

    // Erase unused ops after the transformation.
    rewriter.eraseOp(output_uniform_dequantize_call_pattern->GetCallOp());
    rewriter.eraseOp(output_uniform_quantize_call_pattern->GetCallOp());
    rewriter.eraseOp(multiply_op);
    rewriter.eraseOp(subtract_op);
    rewriter.eraseOp(input_i8_to_f32_convert_op);
    rewriter.eraseOp(input_uniform_quantize_call_pattern->GetCallOp());
  }

  // Quantization dimension corresponds to the output feature dimension, i.e.
  // not contracted nor batched.
  int64_t GetFilterQuantizationDimension(
      const stablehlo::DotDimensionNumbersAttr& dot_dimension_numbers,
      const int64_t filter_rank) const {
    // Register all dimensions as candidates.
    auto seq_range = llvm::seq(int64_t{0}, filter_rank);
    SmallVector<int64_t> quantization_dimension_candidates(seq_range.begin(),
                                                           seq_range.end());

    // Erase all contracting dimensions from the candidates.
    for (const int64_t contracting_dim :
         dot_dimension_numbers.getRhsContractingDimensions()) {
      auto contracting_dim_itr =
          absl::c_find(quantization_dimension_candidates, contracting_dim);
      quantization_dimension_candidates.erase(contracting_dim_itr);
    }

    // Erase all batching dimensions from the candidates.
    for (const int64_t batching_dim :
         dot_dimension_numbers.getRhsBatchingDimensions()) {
      auto batching_dim_itr =
          absl::c_find(quantization_dimension_candidates, batching_dim);
      quantization_dimension_candidates.erase(batching_dim_itr);
    }

    return quantization_dimension_candidates[0];
  }

 private:
  // Returns the filter constant op. The resulting constant's element type is
  // either i8 (when i8->f32 cast is present) or f32.
  stablehlo::ConstantOp GetFilterConstantOp(Value filter_value) const {
    Operation* filter_op = filter_value.getDefiningOp();

    auto f32_filter_constant_op = dyn_cast<stablehlo::ConstantOp>(filter_op);
    if (f32_filter_constant_op) {
      return f32_filter_constant_op;
    } else {
      // Build uniform quantized type for filter.
      auto filter_i8_to_f32_convert_op = cast<stablehlo::ConvertOp>(filter_op);

      return cast<stablehlo::ConstantOp>(
          filter_i8_to_f32_convert_op.getOperand().getDefiningOp());
    }
  }

  LogicalResult MatchFilter(Value filter_value) const {
    auto filter_constant_op = TryCast<stablehlo::ConstantOp>(
        filter_value.getDefiningOp(), /*name=*/"float_filter_constant_op");
    if (succeeded(filter_constant_op) &&
        filter_constant_op->getResult().getType().getElementType().isF32()) {
      return success();
    }

    auto filter_i8_to_f32_convert_op =
        TryCast<stablehlo::ConvertOp>(filter_value.getDefiningOp(),
                                      /*name=*/"filter_i8_to_f32_convert_op");
    if (failed(filter_i8_to_f32_convert_op)) return failure();

    if (!IsI8ToF32Cast(*filter_i8_to_f32_convert_op)) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match filter_i8_to_f32_convert_op. "
                                 "It should be a i8->f32 cast.\n");
      return failure();
    }

    return TryCast<stablehlo::ConstantOp>(
        filter_i8_to_f32_convert_op->getOperand().getDefiningOp(),
        /*name=*/"filter_constant_op");
  }
};

// Matches the pattern for quantized dot_general op and rewrites it to use
// uniform quantized types when both operands are activations.
//
// Currently assumes asymmetric per-tensor quantization for both activations.
//
// This pattern represents the following derived equation, where:
// * rn = real (expressed) value for tensor n
// * qn = quantized value for tensor n
// * sn = scale for tensor n
// * zn = zero point for tensor n
//
// r3 = r1 * r2
//    = s1 (q1 - z1) * s2 (q2 - z2)
//    = s1 s2 (q1 - z1) * (q2 - z2)
//
// Unlike `ComposeUniformQuantizedDotGeneralOp`, the pattern assumes that the
// term "(q1 - z1) * (q2 - z2)" is not expanded. This is done to reduce
// unnecessary op count. Also, the subtractions "(q - z)" are performed in i32
// to avoid underflows.
//
// In StableHLO text representation, the pattern is as the following
// (simplified):
//
// ```
// %0 = // Input tensor r1.
// %1 = // Input tensor r2.
// %2 = stablehlo.constant  // Input 1 inverse scale 1 / s1.
// %3 = stablehlo.constant  // Input 1 zero point z1 (i8).
// %4 = stablehlo.constant  // Input 1 zero point z1 (i32).
// %5 = stablehlo.constant  // Input 2 inverse scale 1 / s2.
// %6 = stablehlo.constant  // Input 2 zero point z2 (i8).
// %7 = stablehlo.constant  // Input 2 zero point z2 (i32).
// %8 = stablehlo.constant  // Input 3 inverse scale 1 / s3.
// %9 = stablehlo.constant  // Input 3 zero point z3.
// %10 = stablehlo.constant  // s1 * s2.
// %11 = call @uniform_quantize(%0, %2, %3)  // Quantize input (q1).
// %12 = call @uniform_quantize_0(%1, %5, %6)  // Quantize input (q2).
// %13 = stablehlo.convert %11  // i8->i32 cast for q1.
// %14 = stablehlo.convert %3  // [Optional] i8->i32 cast for z1.
// %15 = stablehlo.broadcast_in_dim %14  // Operand = %4 if no `convert` above.
// %16 = stablehlo.subtract %13, %15  // q1 - z1
// %17 = stablehlo.convert %12  // i8->i32 cast for q2.
// %18 = stablehlo.convert %6  // [Optional] i8->i32 cast for z2.
// %19 = stablehlo.broadcast_in_dim %18  // Operand = %7 if no `convert` above.
// %20 = stablehlo.subtract %17, %19  // q2 - z2
// %21 = stablehlo.dot_general(%16, %20)  // (q1 - z1) * (q2 - z2).
// %22 = stablehlo.convert %21  // i32 -> f32 cast.
// %23 = stablehlo.broadcast_in_dim %10
// %24 = stablehlo.multiply %22 %23  // * s1 s2
//
// The following quant -> dequant pattern is a no-op, but is required to
// retrieve the quantization parameters for the output tensor.
//
// %25 = call @uniform_quantize_1(%24, %8, %9)  // r3 -> q3
// %26 = call @uniform_dequantize(%25, %8, %9)  // q3 -> r3
// ```
//
// The rewritten pattern looks like:
//
// ```
// %2 = stablehlo.uniform_quantize %0  // Input 1 f32->uniform quantized type.
// %3 = stablehlo.uniform_quantize %1  // Input 2 f32->uniform quantized type.
// %4 = stablehlo.dot_general(%2, %3)   // In uniform quantized type.
// %5 = stablehlo.uniform_dequantize %4  // Dequantize the output.
// ```
class ComposeUniformQuantizedDotGeneralOpWithTwoQuantizedActivations
    : public OpRewritePattern<stablehlo::DotGeneralOp> {
 public:
  using OpRewritePattern<stablehlo::DotGeneralOp>::OpRewritePattern;

  LogicalResult match(stablehlo::DotGeneralOp op) const final {
    // q1 - z1
    if (failed(MatchQuantizedOperand(op.getOperand(0)))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match quantized operand pattern for LHS.\n");
      return failure();
    }

    // q2 - z2
    if (failed(MatchQuantizedOperand(op.getOperand(1)))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match quantized operand pattern for RHS.\n");
      return failure();
    }

    // Go downstream from `op`.
    // * s1 s2
    auto output_i32_to_f32_convert_op = TryCast<stablehlo::ConvertOp>(
        *op.getResult().user_begin(), /*name=*/"output_i32_to_f32_convert_op");
    if (failed(output_i32_to_f32_convert_op)) return failure();

    auto combined_scale_multiply_op = TryCast<stablehlo::MulOp>(
        *output_i32_to_f32_convert_op->getResult().user_begin(),
        /*name=*/"combined_scale_multiply_op");
    if (failed(combined_scale_multiply_op)) return failure();

    // call @uniform_quantize()
    auto output_uniform_quantize_call_op = TryCast<func::CallOp>(
        *combined_scale_multiply_op->getResult().user_begin(),
        /*name=*/"output_quantize_call_op");
    if (failed(output_uniform_quantize_call_op)) return failure();

    auto output_uniform_quantize_call_pattern =
        UniformQuantizeFunctionCallPattern::Match(
            *output_uniform_quantize_call_op);
    if (failed(output_uniform_quantize_call_pattern)) {
      llvm::dbgs() << "Failed match uniform quantize call pattern.\n";
      return failure();
    }

    // call @uniform_dequantize()
    auto output_uniform_dequantize_call_op = TryCast<func::CallOp>(
        *output_uniform_quantize_call_op->getResult(0).user_begin(),
        /*name=*/"output_uniform_dequantize_call_op");
    if (failed(output_uniform_dequantize_call_op)) return failure();

    auto output_uniform_dequantize_call_pattern =
        UniformDequantizeFunctionCallPattern::Match(
            *output_uniform_dequantize_call_op);
    if (failed(output_uniform_dequantize_call_pattern)) {
      llvm::dbgs() << "Failed to match output uniform quantize call pattern.\n";
      return failure();
    }

    return success();
  }

  void rewrite(stablehlo::DotGeneralOp op,
               PatternRewriter& rewriter) const final {
    // Build uniform quantized type for input 1 (lhs).
    auto input1_zero_point_subtract_op =
        cast<stablehlo::SubtractOp>(op.getOperand(0).getDefiningOp());
    auto input1_i8_to_i32_convert_op = cast<stablehlo::ConvertOp>(
        input1_zero_point_subtract_op.getOperand(0).getDefiningOp());
    auto input1_uniform_quantize_call_op = cast<func::CallOp>(
        input1_i8_to_i32_convert_op.getOperand().getDefiningOp());
    auto input1_uniform_quantize_call_pattern =
        UniformQuantizeFunctionCallPattern::Match(
            input1_uniform_quantize_call_op);

    const float input1_inverse_scale_value =
        input1_uniform_quantize_call_pattern->GetInverseScalesValueAttr()
            .getSplatValue<APFloat>()
            .convertToFloat();
    const float input1_scale_value = 1.0 / input1_inverse_scale_value;

    const int8_t input1_zero_point_value =
        input1_uniform_quantize_call_pattern->GetZeroPointsValueAttr()
            .getSplatValue<APInt>()
            .getSExtValue();

    const UniformQuantizedType input1_uniform_quantized_type =
        CreateI8F32UniformQuantizedType(
            input1_uniform_quantize_call_op.getLoc(), *rewriter.getContext(),
            input1_scale_value, input1_zero_point_value);

    Value input1_value = input1_uniform_quantize_call_pattern->GetInputValue();
    auto input1_uniform_quantize_op =
        rewriter.create<stablehlo::UniformQuantizeOp>(
            input1_uniform_quantize_call_op.getLoc(),
            /*result=*/
            input1_value.getType().cast<TensorType>().clone(
                input1_uniform_quantized_type),
            /*operand=*/input1_value);

    rewriter.replaceAllUsesWith(input1_zero_point_subtract_op.getResult(),
                                input1_uniform_quantize_op.getResult());

    // Build uniform quantized type for input 2 (rhs).
    auto input2_zero_point_subtract_op =
        cast<stablehlo::SubtractOp>(op.getOperand(1).getDefiningOp());
    auto input2_i8_to_i32_convert_op = cast<stablehlo::ConvertOp>(
        input2_zero_point_subtract_op.getOperand(0).getDefiningOp());
    auto input2_uniform_quantize_call_op = cast<func::CallOp>(
        input2_i8_to_i32_convert_op.getOperand().getDefiningOp());
    auto input2_uniform_quantize_call_pattern =
        UniformQuantizeFunctionCallPattern::Match(
            input2_uniform_quantize_call_op);

    const float input2_inverse_scale_value =
        input2_uniform_quantize_call_pattern->GetInverseScalesValueAttr()
            .getSplatValue<APFloat>()
            .convertToFloat();
    const float input2_scale_value = 1.0 / input2_inverse_scale_value;

    const int8_t input2_zero_point_value =
        input2_uniform_quantize_call_pattern->GetZeroPointsValueAttr()
            .getSplatValue<APInt>()
            .getSExtValue();

    const UniformQuantizedType input2_uniform_quantized_type =
        CreateI8F32UniformQuantizedType(
            input2_uniform_quantize_call_op.getLoc(), *rewriter.getContext(),
            input2_scale_value, input2_zero_point_value);

    Value input2_value = input2_uniform_quantize_call_pattern->GetInputValue();
    auto input2_uniform_quantize_op =
        rewriter.create<stablehlo::UniformQuantizeOp>(
            input2_uniform_quantize_call_op.getLoc(),
            /*result=*/
            input2_value.getType().cast<TensorType>().clone(
                input2_uniform_quantized_type),
            /*operand=*/input2_value);

    rewriter.replaceAllUsesWith(input2_zero_point_subtract_op.getResult(),
                                input2_uniform_quantize_op.getResult());

    // Recreate stablehlo::DotGeneralOp with a uniform quantized output type.
    // * s1 s2
    auto output_i32_to_f32_convert_op =
        cast<stablehlo::ConvertOp>(*op.getResult().user_begin());
    auto combined_scale_multiply_op = cast<stablehlo::MulOp>(
        *output_i32_to_f32_convert_op.getResult().user_begin());

    // call @uniform_quantize()
    auto output_uniform_quantize_call_op = cast<func::CallOp>(
        *combined_scale_multiply_op.getResult().user_begin());

    auto output_uniform_quantize_call_pattern =
        UniformQuantizeFunctionCallPattern::Match(
            output_uniform_quantize_call_op);

    // call @uniform_dequantize()
    auto output_uniform_dequantize_call_op = cast<func::CallOp>(
        *output_uniform_quantize_call_op.getResult(0).user_begin());

    auto output_uniform_dequantize_call_pattern =
        UniformDequantizeFunctionCallPattern::Match(
            output_uniform_dequantize_call_op);

    const auto inverse_output_scale_value =
        output_uniform_quantize_call_pattern->GetInverseScalesValueAttr()
            .getSplatValue<APFloat>()
            .convertToFloat();
    const float output_scale_value = 1.0 / inverse_output_scale_value;

    const int64_t output_zero_point_value =
        output_uniform_quantize_call_pattern->GetZeroPointsValueAttr()
            .getSplatValue<APInt>()
            .getSExtValue();

    const UniformQuantizedType output_uniform_quantized_type =
        CreateI8F32UniformQuantizedType(
            output_uniform_quantize_call_op.getLoc(), *rewriter.getContext(),
            output_scale_value, output_zero_point_value);

    auto new_dot_general_op = rewriter.create<stablehlo::DotGeneralOp>(
        op.getLoc(), /*resultType0=*/
        op.getResult().getType().cast<TensorType>().clone(
            output_uniform_quantized_type),
        /*lhs=*/op.getLhs(), /*rhs=*/op.getRhs(),
        /*dot_dimension_numbers=*/op.getDotDimensionNumbers(),
        /*precision_config=*/op.getPrecisionConfigAttr());

    rewriter.replaceAllUsesWith(op.getResult(), new_dot_general_op.getResult());

    auto new_output_dequant_op =
        rewriter.create<stablehlo::UniformDequantizeOp>(
            output_uniform_dequantize_call_op.getLoc(),
            /*operand=*/new_dot_general_op);

    rewriter.replaceAllUsesWith(output_uniform_dequantize_call_op.getResult(0),
                                new_output_dequant_op.getResult());

    // Erase unused ops after the transformation.
    rewriter.eraseOp(output_uniform_dequantize_call_pattern->GetCallOp());
    rewriter.eraseOp(output_uniform_quantize_call_pattern->GetCallOp());
    rewriter.eraseOp(combined_scale_multiply_op);
    rewriter.eraseOp(output_i32_to_f32_convert_op);

    rewriter.eraseOp(input1_zero_point_subtract_op);
    rewriter.eraseOp(input1_i8_to_i32_convert_op);
    rewriter.eraseOp(input1_uniform_quantize_call_pattern->GetCallOp());

    rewriter.eraseOp(input2_zero_point_subtract_op);
    rewriter.eraseOp(input2_i8_to_i32_convert_op);
    rewriter.eraseOp(input2_uniform_quantize_call_pattern->GetCallOp());
  }

 private:
  // Determines whether the `operand` is a result of quantized operand pattern,
  // represented as `(q - z)` where `q` is the quantized value and `z` is the
  // zero point. Returns `success()` iff `operand` matches the pattern,
  // `failure()` otherwise.
  LogicalResult MatchQuantizedOperand(Value operand) const {
    // q - z
    auto input_zero_point_subtract_op =
        TryCast<stablehlo::SubtractOp>(operand.getDefiningOp(),
                                       /*name=*/"input_zero_point_subtract_op");
    if (failed(input_zero_point_subtract_op)) return failure();

    // z
    auto input_zero_point_broadcast_in_dim_op =
        TryCast<stablehlo::BroadcastInDimOp>(
            input_zero_point_subtract_op->getOperand(1).getDefiningOp(),
            /*name=*/"input_zero_point_broadcast_in_dim_op");
    if (failed(input_zero_point_broadcast_in_dim_op)) return failure();

    Value input_zero_point_constant_value =
        input_zero_point_broadcast_in_dim_op->getOperand();
    // Match optional i8->i32 conversion for z.
    if (auto input_zero_point_i8_to_i32_convert_op =
            TryCast<stablehlo::ConvertOp>(
                input_zero_point_constant_value.getDefiningOp(),
                /*name=*/"input_zero_point_i8_to_i32_convert_op");
        succeeded(input_zero_point_i8_to_i32_convert_op)) {
      if (!IsI8ToI32Cast(*input_zero_point_i8_to_i32_convert_op)) {
        return failure();
      }
      input_zero_point_constant_value =
          input_zero_point_i8_to_i32_convert_op->getOperand();
    }

    auto input_zero_point_constant_op = TryCast<stablehlo::ConstantOp>(
        input_zero_point_constant_value.getDefiningOp(),
        /*name=*/"input_zero_point_constant_op");
    if (failed(input_zero_point_constant_op)) return failure();

    // q
    auto input_i8_to_i32_convert_op = TryCast<stablehlo::ConvertOp>(
        input_zero_point_subtract_op->getOperand(0).getDefiningOp(),
        /*name=*/"input_i8_to_i32_convert_op");
    if (failed(input_i8_to_i32_convert_op)) return failure();

    if (!IsI8ToI32Cast(*input_i8_to_i32_convert_op)) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Failed to match. The ConvertOp is not an i8->i32 type cast.\n");
      return failure();
    }

    auto input_uniform_quantize_call_op = TryCast<func::CallOp>(
        input_i8_to_i32_convert_op->getOperand().getDefiningOp(),
        /*name=*/"input_uniform_quantize_call_op");
    if (failed(input_uniform_quantize_call_op)) return failure();

    auto input_uniform_quantize_call_pattern =
        UniformQuantizeFunctionCallPattern::Match(
            *input_uniform_quantize_call_op);
    if (failed(input_uniform_quantize_call_pattern)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match input uniform quantize call pattern.\n");
      return failure();
    }

    return success();
  }
};

void ComposeUniformQuantizedTypePass::runOnOperation() {
  ModuleOp module_op = getOperation();
  MLIRContext& ctx = getContext();

  RewritePatternSet patterns(&ctx);
  patterns.add<ComposeUniformQuantizedConvolutionOp,
               ComposeUniformQuantizedDotGeneralOp,
               ComposeUniformQuantizedDotGeneralOpWithTwoQuantizedActivations>(
      &ctx);

  if (failed(applyPatternsAndFoldGreedily(module_op, std::move(patterns)))) {
    module_op.emitError()
        << "Failed to compose stablehlo uniform quantized types.";
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateComposeUniformQuantizedTypePass() {
  return std::make_unique<ComposeUniformQuantizedTypePass>();
}

static PassRegistration<ComposeUniformQuantizedTypePass> pass;

}  // namespace odml
}  // namespace mlir
