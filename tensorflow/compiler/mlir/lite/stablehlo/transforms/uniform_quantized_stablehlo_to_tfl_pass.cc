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
#include <iterator>
#include <limits>
#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project  // NOLINT: Required to register quantization dialect.
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

#define DEBUG_TYPE "uniform-quantized-stablehlo-to-tfl"

namespace mlir {
namespace odml {
namespace {

using ::mlir::quant::QuantizedType;
using ::mlir::quant::UniformQuantizedPerAxisType;
using ::mlir::quant::UniformQuantizedType;

#define GEN_PASS_DEF_UNIFORMQUANTIZEDSTABLEHLOTOTFLPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h.inc"

class UniformQuantizedStablehloToTflPass
    : public impl::UniformQuantizedStablehloToTflPassBase<
          UniformQuantizedStablehloToTflPass> {
 private:
  void runOnOperation() override;
};

// Determines whether the storage type of a quantized type is supported by
// `tfl.quantize` or `tfl.dequantize` ops. ui8, i8 and i16 are supported.
bool IsSupportedByTfliteQuantizeOrDequantizeOps(IntegerType storage_type) {
  if ((storage_type.isSigned() &&
       !(storage_type.getWidth() == 8 || storage_type.getWidth() == 16)) ||
      (!storage_type.isSigned() && storage_type.getWidth() != 8)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Uniform quantize / dequantize op only supports ui8, i8 or "
                  "i16 for the storage type of uniform quantized type. Got: "
               << storage_type << ".\n");
    return false;
  }
  return true;
}

// Returns true iff the storage type of `quantized_type` is 8-bit integer.
bool IsStorageTypeI8(QuantizedType quantized_type) {
  const Type storage_type = quantized_type.getStorageType();
  return storage_type.isInteger(/*width=*/8);
}

// Returns true iff the expressed type of `quantized_type` is f32.
bool IsExpressedTypeF32(QuantizedType quantized_type) {
  const Type expressed_type = quantized_type.getExpressedType();
  return expressed_type.isa<Float32Type>();
}

// Returns true iff `type` is a uniform quantized type whose storage type is
// 8-bit integer and expressed type is f32.
bool IsI8F32UniformQuantizedType(const Type type) {
  auto quantized_type = type.dyn_cast_or_null<UniformQuantizedType>();
  if (!quantized_type) {
    LLVM_DEBUG(llvm::dbgs()
               << "Expected a uniform quantized type. Got: " << type << ".\n");
    return false;
  }

  if (!IsStorageTypeI8(quantized_type)) {
    LLVM_DEBUG(llvm::dbgs() << "Expected an i8 storage type. Got: "
                            << quantized_type << ".\n");
    return false;
  }

  if (!IsExpressedTypeF32(quantized_type)) {
    LLVM_DEBUG(llvm::dbgs() << "Expected an f32 expressed type. Got: "
                            << quantized_type << ".\n");
    return false;
  }

  return true;
}

// Returns true iff `type` is a uniform quantized per-axis (per-channel) type
// whose storage type is 8-bit integer and expressed type is f32.
bool IsI8F32UniformQuantizedPerAxisType(const Type type) {
  auto quantized_per_axis_type =
      type.dyn_cast_or_null<UniformQuantizedPerAxisType>();
  if (!quantized_per_axis_type) {
    LLVM_DEBUG(llvm::dbgs()
               << "Expected a uniform quantized type. Got: " << type << ".\n");
    return false;
  }

  if (!IsStorageTypeI8(quantized_per_axis_type)) {
    LLVM_DEBUG(llvm::dbgs() << "Expected an i8 storage type. Got: "
                            << quantized_per_axis_type << ".\n");
    return false;
  }

  if (!IsExpressedTypeF32(quantized_per_axis_type)) {
    LLVM_DEBUG(llvm::dbgs() << "Expected an f32 expressed type. Got: "
                            << quantized_per_axis_type << ".\n");
    return false;
  }

  return true;
}

// Bias scales for matmul-like ops should be input scale * filter scale. Here it
// is assumed that the input is per-tensor quantized and filter is per-channel
// quantized.
SmallVector<double> GetBiasScales(const double input_scale,
                                  const ArrayRef<double> filter_scales) {
  SmallVector<double> bias_scales;
  absl::c_transform(filter_scales, std::back_inserter(bias_scales),
                    [input_scale](const double filter_scale) -> double {
                      return filter_scale * input_scale;
                    });
  return bias_scales;
}

// stablehlo.uniform_quantize -> tfl.quantize
class RewriteUniformQuantizeOp
    : public OpRewritePattern<stablehlo::UniformQuantizeOp> {
  using OpRewritePattern<stablehlo::UniformQuantizeOp>::OpRewritePattern;

  // Determines whether the input and output types are compatible with
  // `tfl.quantize`. See the definition for the `QUANTIZE` kernel for the
  // detailed limitations
  // (https://github.com/tensorflow/tensorflow/blob/8f145d579aa0ee7f4187af32dbbf4e12fdabbffe/tensorflow/lite/kernels/quantize.cc#L105).
  LogicalResult match(stablehlo::UniformQuantizeOp op) const override {
    const Type input_element_type =
        op.getOperand().getType().cast<TensorType>().getElementType();
    if (!input_element_type.isa<FloatType>()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Uniform quantize op's input should be a float type. Got: "
                 << input_element_type << ".\n");
      return failure();
    }

    // Output type of `UniformQuantizeOp` is guaranteed to be a quantized
    // tensor with integer storage type.
    const auto output_storage_type = op.getResult()
                                         .getType()
                                         .cast<TensorType>()
                                         .getElementType()
                                         .cast<QuantizedType>()
                                         .getStorageType()
                                         .cast<IntegerType>();
    if (!IsSupportedByTfliteQuantizeOrDequantizeOps(output_storage_type)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match storage type of output quantized type.\n");
      return failure();
    }

    return success();
  }

  void rewrite(stablehlo::UniformQuantizeOp op,
               PatternRewriter& rewriter) const override {
    Type output_type = *op->getResultTypes().begin();
    rewriter.replaceOpWithNewOp<TFL::QuantizeOp>(
        op, output_type, /*input=*/op.getOperand(),
        /*qtype=*/TypeAttr::get(output_type));
  }
};

// stablehlo.uniform_dequantize -> tfl.dequantize
class RewriteUniformDequantizeOp
    : public OpRewritePattern<stablehlo::UniformDequantizeOp> {
  using OpRewritePattern<stablehlo::UniformDequantizeOp>::OpRewritePattern;

  // Determines whether the input and output types are compatible with
  // `tfl.dequantize`. See the definition for the `DEQUANTIZE` kernel for the
  // detailed limitations
  // (https://github.com/tensorflow/tensorflow/blob/8f145d579aa0ee7f4187af32dbbf4e12fdabbffe/tensorflow/lite/kernels/dequantize.cc#L52).
  LogicalResult match(stablehlo::UniformDequantizeOp op) const override {
    const auto input_storage_type = op.getOperand()
                                        .getType()
                                        .cast<TensorType>()
                                        .getElementType()
                                        .cast<QuantizedType>()
                                        .getStorageType()
                                        .cast<IntegerType>();
    if (!IsSupportedByTfliteQuantizeOrDequantizeOps(input_storage_type)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match storage type of input quantized type.\n");
      return failure();
    }

    // Output type is guaranteed to be a float tensor for a valid StableHLO.
    const auto output_element_type = op.getResult()
                                         .getType()
                                         .cast<TensorType>()
                                         .getElementType()
                                         .cast<FloatType>();
    if (!output_element_type.isa<Float32Type>()) {
      LLVM_DEBUG(llvm::dbgs() << "Uniform dequantize op's output element type "
                                 "should be f32. Got: "
                              << output_element_type << ".\n");
      return failure();
    }

    return success();
  }

  void rewrite(stablehlo::UniformDequantizeOp op,
               PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<TFL::DequantizeOp>(
        op, /*resultTypes=*/op->getResultTypes(), /*input=*/op.getOperand());
  }
};

// Rewrites `stablehlo.convolution` -> `tfl.conv_2d` when it accepts uniform
// quantized tensors.
//
// Conditions for the conversion:
//   * Input and output tensors are per-tensor uniform quantized (i8->f32)
//     tensors.
//   * The filter tensor is constant a per-channel uniform quantized (i8->f32)
//     tensor.
//   * Convolution is a 2D convolution op and both the input's and filter's
//     shape is 4 dimensional.
//   * The filter tensor's format is `[0, 1, i, o]`.
//   * Not a depthwise convolution.
//   * Does not consider bias add fusion.
// TODO: b/294771704 - Support bias quantization.
class RewriteQuantizedConvolutionOp
    : public OpRewritePattern<stablehlo::ConvolutionOp> {
 public:
  using OpRewritePattern<stablehlo::ConvolutionOp>::OpRewritePattern;

  static LogicalResult MatchInput(Value input) {
    auto input_type = input.getType().cast<TensorType>();
    if (input_type.getRank() != 4) {
      LLVM_DEBUG(llvm::dbgs() << "Only 2D convolution op is supported. "
                                 "Expected input rank of 4. Got: "
                              << input_type.getRank() << ".\n");
      return failure();
    }

    if (const auto input_element_type = input_type.getElementType();
        !IsI8F32UniformQuantizedType(input_element_type)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Expected an i8->f32 uniform quantized type. Got: "
                 << input_element_type << ".\n");
      return failure();
    }

    return success();
  }

  static LogicalResult MatchFilter(Value filter) {
    auto filter_type = filter.getType().cast<TensorType>();
    if (filter_type.getRank() != 4) {
      LLVM_DEBUG(llvm::dbgs() << "Only 2D convolution op is supported. "
                                 "Expected filter rank of 4. Got: "
                              << filter_type.getRank() << ".\n");
      return failure();
    }

    const Type filter_element_type = filter_type.getElementType();
    if (!IsI8F32UniformQuantizedPerAxisType(filter_type.getElementType())) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Expected a per-channel uniform quantized (i8->f32) type. Got: "
          << filter_element_type << "\n");
      return failure();
    }

    if (filter_element_type.cast<UniformQuantizedPerAxisType>()
            .getQuantizedDimension() != 3) {
      LLVM_DEBUG(llvm::dbgs() << "Quantized dimension should be 3. Got: "
                              << filter_element_type << "\n");
      return failure();
    }

    if (Operation* filter_op = filter.getDefiningOp();
        filter_op == nullptr || !isa<stablehlo::ConstantOp>(filter_op)) {
      LLVM_DEBUG(llvm::dbgs() << "Filter should be a constant.\n");
      return failure();
    }

    return success();
  }

  static LogicalResult MatchOutput(Value output) {
    const Type output_element_type =
        output.getType().cast<TensorType>().getElementType();
    if (!IsI8F32UniformQuantizedType(output_element_type)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Expected a uniform quantized (i8->f32) type. Got: "
                 << output_element_type << ".\n");
      return failure();
    }

    return success();
  }

  LogicalResult match(stablehlo::ConvolutionOp op) const override {
    stablehlo::ConvDimensionNumbersAttr dimension_numbers =
        op.getDimensionNumbers();

    const int64_t output_dimension =
        dimension_numbers.getKernelOutputFeatureDimension();
    if (output_dimension != 3) {
      LLVM_DEBUG(llvm::dbgs() << "Expected kernel output feature == 3. Got: "
                              << output_dimension << ".\n");
      return failure();
    }

    const int64_t input_dimension =
        dimension_numbers.getKernelInputFeatureDimension();
    if (input_dimension != 2) {
      LLVM_DEBUG(llvm::dbgs() << "Expected kernel input feature == 2. Got: "
                              << input_dimension << ".\n");
      return failure();
    }

    if (failed(MatchInput(op.getOperand(0)))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match input for quantized convolution_op.\n");
      return failure();
    }

    if (failed(MatchFilter(op.getOperand(1)))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match filter for quantized convolution_op.\n");
      return failure();
    }

    if (failed(MatchOutput(op.getResult()))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match output for quantized convolution_op.\n");
      return failure();
    }

    return success();
  }

  void rewrite(stablehlo::ConvolutionOp op,
               PatternRewriter& rewriter) const override {
    Value filter_value = op.getOperand(1);
    Operation* filter_op = filter_value.getDefiningOp();

    auto filter_uniform_quantized_type =
        filter_value.getType()
            .cast<TensorType>()
            .getElementType()
            .cast<UniformQuantizedPerAxisType>();

    // Create a new quantized tensor type for the filter. This is required
    // because the quantized dimension is changed from 3 -> 0. `TFL::Conv2DOp`
    // requires the quantized dimension to be 0 because it accepts a filter
    // tensor of format OHWI
    // (https://github.com/tensorflow/tensorflow/blob/5430e5e238f868ce977df96ba89c9c1d31fbe8fa/tensorflow/compiler/mlir/lite/ir/tfl_ops.td#L933).
    // The quantized dimension should correspond to the output feature
    // dimension.
    auto new_filter_quantized_type = UniformQuantizedPerAxisType::getChecked(
        filter_op->getLoc(), /*flags=*/true,
        /*storageType=*/filter_uniform_quantized_type.getStorageType(),
        filter_uniform_quantized_type.getExpressedType(),
        filter_uniform_quantized_type.getScales(),
        filter_uniform_quantized_type.getZeroPoints(),
        /*quantizedDimension=*/0,
        filter_uniform_quantized_type.getStorageTypeMin(),
        filter_uniform_quantized_type.getStorageTypeMax());

    auto filter_constant_value_attr = cast<DenseIntElementsAttr>(
        cast<stablehlo::ConstantOp>(filter_value.getDefiningOp()).getValue());

    // Using TransposeOp doesn't work because the quantized dimension
    // changes which violates the constraint for the TransposeOp that the
    // input's and output's element type should be the same.
    const DenseIntElementsAttr new_filter_value_attr = TransposeFilterValue(
        filter_op->getLoc(), rewriter, filter_constant_value_attr);

    auto new_filter_result_type = RankedTensorType::getChecked(
        filter_op->getLoc(),
        /*shape=*/new_filter_value_attr.getShapedType().getShape(),
        /*type=*/new_filter_quantized_type);

    auto new_filter_constant_op = rewriter.create<TFL::QConstOp>(
        filter_op->getLoc(), /*output=*/TypeAttr::get(new_filter_result_type),
        new_filter_value_attr);

    SmallVector<double> bias_scales =
        GetBiasScales(/*input_scale=*/op.getOperand(0)
                          .getType()
                          .cast<TensorType>()
                          .getElementType()
                          .cast<UniformQuantizedType>()
                          .getScale(),
                      /*filter_scales=*/new_filter_quantized_type.getScales());

    // Create a bias filled with zeros. Mimics the behavior of no bias add.
    const int64_t num_output_features = new_filter_result_type.getShape()[0];
    const SmallVector<int64_t, 1> bias_shape = {num_output_features};
    auto bias_quantized_type = UniformQuantizedPerAxisType::getChecked(
        op.getLoc(), /*flags=*/true,
        /*storageType=*/rewriter.getI32Type(),  // i32 for bias
        /*expressedType=*/rewriter.getF32Type(),
        /*scales=*/std::move(bias_scales),
        /*zeroPoints=*/new_filter_quantized_type.getZeroPoints(),  // Zeros.
        /*quantizedDimension=*/0,
        /*storageTypeMin=*/std::numeric_limits<int32_t>::min(),
        /*storageTypeMax=*/std::numeric_limits<int32_t>::max());
    auto bias_type = RankedTensorType::getChecked(op.getLoc(), bias_shape,
                                                  bias_quantized_type);

    // Create a bias constant. It should have values of 0.
    auto bias_value_type = RankedTensorType::getChecked(op.getLoc(), bias_shape,
                                                        rewriter.getI32Type());
    auto bias_value = DenseIntElementsAttr::get(
        bias_value_type, APInt(/*numBits=*/32, /*value=*/0, /*isSigned=*/true));
    auto bias = rewriter.create<TFL::QConstOp>(
        op.getLoc(), /*output=*/TypeAttr::get(bias_type),
        /*value=*/bias_value);

    // Determine the attributes for the TFL::Conv2DOp.
    // TODO: b/294808863 - Use `padding = "SAME"` if the padding attribute
    // matches the semantics.
    Value input_value = op.getOperand(0);
    if (const DenseIntElementsAttr padding_attr = op.getPaddingAttr();
        !IsPaddingValid(padding_attr)) {
      // Add an extra tfl.pad_op if there are explicit padding values. This
      // extra pad op will allow us to always set the `padding` attribute of the
      // newly created tfl.conv_2d op as "VALID".
      TFL::PadOp pad_op =
          CreateTflPadOp(op.getLoc(), padding_attr, input_value, rewriter);
      input_value = pad_op.getResult();
    }

    const auto [stride_h, stride_w] = GetStrides(op);
    const auto [dilation_h_factor, dilation_w_factor] = GetDilationFactors(op);

    auto tfl_conv2d_op = rewriter.create<TFL::Conv2DOp>(
        op.getLoc(), /*output=*/op.getResult().getType(), /*input=*/input_value,
        /*filter=*/new_filter_constant_op, /*bias=*/bias.getResult(),
        /*dilation_h_factor=*/rewriter.getI32IntegerAttr(dilation_h_factor),
        /*dilation_w_factor=*/rewriter.getI32IntegerAttr(dilation_w_factor),
        /*fused_activation_function=*/rewriter.getStringAttr("NONE"),
        /*padding=*/rewriter.getStringAttr("VALID"),
        /*stride_h=*/rewriter.getI32IntegerAttr(stride_h),
        /*stride_w=*/rewriter.getI32IntegerAttr(stride_w));

    rewriter.replaceAllUsesWith(op.getResult(), tfl_conv2d_op.getResult());
    rewriter.eraseOp(op);
  }

 private:
  // Create a `tfl.pad` op to apply explicit padding to the input tensor that
  // correspond to the `padding` attribute from the `stablehlo.convolution` op.
  TFL::PadOp CreateTflPadOp(Location loc,
                            const DenseIntElementsAttr& padding_attr,
                            Value input_value,
                            PatternRewriter& rewriter) const {
    auto padding_values = padding_attr.getValues<int64_t>();
    // [[h_l, h_r], [w_l, w_r]].
    DCHECK_EQ(padding_attr.size(), 4);

    // In StableHLO the padding attribute doesn't include the padding values for
    // input and output feature dimensions (because they are 0 anyways). In
    // TFLite, padding values for input and output feature dimensions should be
    // explicitly set to 0s. Note that TFLite's input tensor is formatted as
    // OHWI. The resulting pad values becomes: [[0, 0], [h_l, h_r], [w_l, w_r],
    // [0, 0]]
    SmallVector<int32_t, 8> tfl_pad_values = {0, 0};  // For output feature dim.
    for (const int64_t padding_value : padding_values) {
      tfl_pad_values.push_back(static_cast<int32_t>(padding_value));
    }
    // For input feature dim.
    tfl_pad_values.push_back(0);
    tfl_pad_values.push_back(0);

    const auto input_tensor_type =
        input_value.getType().cast<RankedTensorType>();
    const int64_t rank = input_tensor_type.getRank();

    SmallVector<int64_t> padded_output_tensor_shape =
        InferPaddedTensorShape(input_tensor_type.getShape(), tfl_pad_values);

    auto padded_output_tensor_type = RankedTensorType::get(
        padded_output_tensor_shape, input_tensor_type.getElementType());

    // The pad values is provided as a const op.
    auto pad_value_const_op = rewriter.create<TFL::ConstOp>(
        loc, /*value=*/DenseIntElementsAttr::get(
            RankedTensorType::get({rank, 2}, rewriter.getIntegerType(32)),
            tfl_pad_values));

    return rewriter.create<TFL::PadOp>(
        loc, /*output=*/padded_output_tensor_type, input_value,
        /*padding=*/pad_value_const_op.getResult());
  }

  // Infers the output tensor's shape after padding `tfl_pad_values` to the
  // `tensor_shape`. `tfl_pad_values` should be formatted as `[[l_0, r_0], [l_1,
  // r_1], ..., [l_n, r_n]]`, where `l_x` and `r_x` are the left and paddings
  // for the x-th dimension, respectively.
  SmallVector<int64_t> InferPaddedTensorShape(
      const ArrayRef<int64_t> tensor_shape,
      const ArrayRef<int32_t> tfl_pad_values) const {
    SmallVector<int64_t> padded_shape(tensor_shape.begin(), tensor_shape.end());
    for (int i = 0; i < padded_shape.size(); ++i) {
      // Left padding + right padding.
      const int32_t padded = tfl_pad_values[i * 2] + tfl_pad_values[i * 2 + 1];
      padded_shape[i] += padded;
    }

    return padded_shape;
  }

  // Transposes the filter tensor to match the filter tensor format for
  // `tfl.conv_2d`. This function performs the following index permutation
  // only: (3, 0, 1, 2). The filter value is assumed to be of `[0, 1, i, o]`
  // format. The `tfl.conv_2d` accepts the filter of `[o, 0, 1, i]`.
  // TODO: b/291598373 - Lift the assumption about the filter tensor's format
  // and generalize the transpose.
  DenseIntElementsAttr TransposeFilterValue(
      Location loc, PatternRewriter& rewriter,
      const DenseIntElementsAttr& filter_value_attr) const {
    ArrayRef<int64_t> filter_shape =
        filter_value_attr.getShapedType().getShape();
    SmallVector<int8_t> filter_constant_values;
    for (const auto filter_val : filter_value_attr.getValues<int8_t>()) {
      filter_constant_values.push_back(filter_val);
    }

    SmallVector<int8_t> new_filter_constant_values(
        filter_constant_values.size(), 0);

    SmallVector<int64_t> new_filter_shape;
    SmallVector<int64_t, 4> transpose_dims = {3, 0, 1, 2};
    for (int i = 0; i < filter_shape.size(); ++i) {
      new_filter_shape.push_back(filter_shape[transpose_dims[i]]);
    }

    auto get_array_idx = [](ArrayRef<int64_t> shape, const int i, const int j,
                            const int k, const int l) -> int64_t {
      return (i * shape[1] * shape[2] * shape[3]) + (j * shape[2] * shape[3]) +
             (k * shape[3]) + l;
    };

    // Transpose the filter value.
    for (int i = 0; i < filter_shape[0]; ++i) {
      for (int j = 0; j < filter_shape[1]; ++j) {
        for (int k = 0; k < filter_shape[2]; ++k) {
          for (int l = 0; l < filter_shape[3]; ++l) {
            // [i][j][k][l] -> [l][i][j][k]
            const int old_idx = get_array_idx(filter_shape, i, j, k, l);
            const int new_idx = get_array_idx(new_filter_shape, l, i, j, k);

            new_filter_constant_values[new_idx] =
                filter_constant_values[old_idx];
          }
        }
      }
    }

    // Create the new filter constant.
    auto new_filter_value_attr_type =
        RankedTensorType::getChecked(loc, new_filter_shape,
                                     /*elementType=*/rewriter.getI8Type());
    auto new_filter_constant_value_attr = DenseIntElementsAttr::get(
        new_filter_value_attr_type, new_filter_constant_values);

    return new_filter_constant_value_attr;
  }

  // Determines if the padding attribute corresponds to "VALID"
  // (https://www.tensorflow.org/api_docs/python/tf/nn).
  bool IsPaddingValid(const DenseIntElementsAttr& padding_attr) const {
    // If padding_attr is empty, it defaults to splat 0s.
    return !padding_attr || (padding_attr.isSplat() &&
                             padding_attr.getSplatValue<int64_t>() == 0);
  }

  // Returns the stride amount for the height and width, respectively.
  std::pair<int64_t, int64_t> GetStrides(stablehlo::ConvolutionOp op) const {
    const DenseIntElementsAttr window_strides_attr = op.getWindowStridesAttr();
    if (!window_strides_attr) {
      return {1, 1};  // Default values.
    }

    const auto window_strides_attr_value =
        window_strides_attr.getValues<int64_t>();
    // It is guaranteed from the spec that it has two values:
    // https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convolution.
    return {window_strides_attr_value[0], window_strides_attr_value[1]};
  }

  // Returns the dilation amount for the height and width, respectively.
  std::pair<int64_t, int64_t> GetDilationFactors(
      stablehlo::ConvolutionOp op) const {
    const DenseIntElementsAttr lhs_dilation_attr = op.getLhsDilationAttr();
    if (!lhs_dilation_attr) {
      return {1, 1};  // Default values.
    }

    const auto lhs_dilation_attr_value = lhs_dilation_attr.getValues<int64_t>();
    // It is guaranteed from the spec that it has two values:
    // https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convolution.
    return {lhs_dilation_attr_value[0], lhs_dilation_attr_value[1]};
  }
};

// Rewrites full-integer quantized `stablehlo.dot_general` ->`tfl.batch_matmul`
// when it accepts uniform quantized tensors.
//
// Since transpose and reshape of quantized tensors are not natively supported
// at the moment, the conversion condition is relatively strict, following
// (https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/batch-mat-mul-v3)
//
// Conditions for the conversion :
//   * size(batching_dimensions) <= 3 (TFLite support restriction)
//   * size(contracting_dimensions) = 1
//   * Input (lhs) and output tensors are per-tensor uniform quantized (i8->f32)
//     tensors (full integer) with shape [..., r_x, c_x] or [..., c_x, r_x].
//   * The rhs tensor is a per-tensor uniform quantized (i8->f32) tensor
//     (constant or activation) with shape [..., r_y, c_y] or [..., c_y, r_y].
//
// TODO: b/293650675 - Relax the conversion condition to support dot_general in
// general.
class RewriteFullIntegerQuantizedDotGeneralOp
    : public OpRewritePattern<stablehlo::DotGeneralOp> {
 public:
  using OpRewritePattern<stablehlo::DotGeneralOp>::OpRewritePattern;

  static LogicalResult MatchLhs(
      Value lhs, stablehlo::DotDimensionNumbersAttr dimension_numbers) {
    auto lhs_type = lhs.getType().cast<TensorType>();
    if (!(IsI8F32UniformQuantizedType(lhs_type.getElementType()))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Expected a per-tensor uniform "
                    "quantized (i8->f32) input for dot_general. Got: "
                 << lhs_type << "\n");
      return failure();
    }
    if (!lhs_type.hasRank()) {
      LLVM_DEBUG(llvm::dbgs() << "Expected lhs of dot_general has rank. Got: "
                              << lhs_type << "\n");
      return failure();
    }
    const int lhs_rank = lhs_type.getRank();
    auto lhs_contracting_dim =
        dimension_numbers.getLhsContractingDimensions()[0];
    if ((lhs_contracting_dim != lhs_rank - 1) &&
        (lhs_contracting_dim != lhs_rank - 2)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Not supported lhs contracting dim for dot_general.\n");
      return failure();
    }
    return success();
  }

  static LogicalResult MatchRhs(
      Value rhs, stablehlo::DotDimensionNumbersAttr dimension_numbers) {
    if (!rhs.getType().cast<TensorType>().hasRank()) {
      LLVM_DEBUG(llvm::dbgs() << "Expected rhs of dot_general has rank. Got: "
                              << rhs.getType() << "\n");
      return failure();
    }
    const int rhs_rank = rhs.getType().cast<TensorType>().getRank();
    auto rhs_contracting_dim =
        dimension_numbers.getRhsContractingDimensions()[0];
    if ((rhs_contracting_dim != rhs_rank - 1) &&
        (rhs_contracting_dim != rhs_rank - 2)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Not supported rhs contracting dim for dot_general.\n");
      return failure();
    }

    auto rhs_type = rhs.getType().cast<TensorType>();
    if (!(IsI8F32UniformQuantizedType(rhs_type.getElementType()))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Expected a per-tensor uniform "
                    "quantized (i8->f32) weight for dot_general. Got: "
                 << rhs_type << "\n");
      return failure();
    }
    return success();
  }

  LogicalResult match(stablehlo::DotGeneralOp op) const override {
    stablehlo::DotDimensionNumbersAttr dimension_numbers =
        op.getDotDimensionNumbers();

    // Check one side is enough since
    // (C1) size(lhs_batching_dimensions) = size(rhs_batching_dimensions).
    if (dimension_numbers.getLhsBatchingDimensions().size() > 3) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Failed to match batch dimention for quantized dot_general.\n");
      return failure();
    }
    // Check one side is enough since
    // (C2) size(lhs_contracting_dimensions) = size(rhs_contracting_dimensions).
    if (dimension_numbers.getLhsContractingDimensions().size() != 1) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Failed to match contract dimention for quantized dot_general.\n");
      return failure();
    }

    if (failed(MatchLhs(op.getLhs(), dimension_numbers))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match input for quantized dot_general.\n");
      return failure();
    }
    if (failed(MatchRhs(op.getRhs(), dimension_numbers))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match weight for quantized dot_general.\n");
      return failure();
    }

    return success();
  }

  void rewrite(stablehlo::DotGeneralOp op,
               PatternRewriter& rewriter) const override {
    Value rhs_value = op.getRhs();
    Operation* rhs_op = rhs_value.getDefiningOp();

    stablehlo::DotDimensionNumbersAttr dimension_numbers =
        op.getDotDimensionNumbers();
    Value input_value = op.getLhs();
    const int lhs_rank = input_value.getType().cast<TensorType>().getRank();
    auto lhs_contracting_dim =
        dimension_numbers.getLhsContractingDimensions()[0];
    BoolAttr adj_x =
        (lhs_contracting_dim == lhs_rank - 2 ? rewriter.getBoolAttr(true)
                                             : rewriter.getBoolAttr(false));
    auto rhs_contracting_dim =
        dimension_numbers.getRhsContractingDimensions()[0];
    const int rhs_rank = rhs_value.getType().cast<TensorType>().getRank();
    BoolAttr adj_y =
        (rhs_contracting_dim == rhs_rank - 1 ? rewriter.getBoolAttr(true)
                                             : rewriter.getBoolAttr(false));

    // Set to `nullptr` because this attribute only matters when the input is
    // dynamic-range quantized.
    BoolAttr asymmetric_quantize_inputs = nullptr;

    // Create BMM assuming rhs is activation.
    auto tfl_batchmatmul_op = rewriter.create<TFL::BatchMatMulOp>(
        op.getLoc(), /*output=*/op.getResult().getType(), /*input=*/input_value,
        /*filter=*/rhs_value, adj_x, adj_y, asymmetric_quantize_inputs);

    // Update BMM if rhs is a constant.
    auto const_rhs = dyn_cast_or_null<stablehlo::ConstantOp>(rhs_op);
    if (const_rhs) {
      auto rhs_uniform_quantized_type = rhs_value.getType().cast<ShapedType>();
      auto rhs_constant_value_attr =
          cast<DenseIntElementsAttr>(const_rhs.getValue());
      auto rhs_constant_op = rewriter.create<TFL::QConstOp>(
          rhs_op->getLoc(),
          /*output=*/TypeAttr::get(rhs_uniform_quantized_type),
          rhs_constant_value_attr);
      tfl_batchmatmul_op = rewriter.create<TFL::BatchMatMulOp>(
          op.getLoc(), /*output=*/op.getResult().getType(),
          /*input=*/input_value, /*filter=*/rhs_constant_op.getResult(), adj_x,
          adj_y, asymmetric_quantize_inputs);
    }

    rewriter.replaceAllUsesWith(op.getResult(), tfl_batchmatmul_op.getResult());
  }
};

// Rewrites `stablehlo.dot_general` -> `tfl.fully_connected` when it accepts
// uniform quantized tensors with per-axis quantized filter tensor (rhs).
//
// Conditions for the conversion:
//   * Input and output tensors are per-tensor uniform quantized (i8->f32)
//     tensors.
//   * The filter tensor is constant a per-channel uniform quantized (i8->f32)
//     tensor. The quantization dimension should be 1 (the non-contracting
//     dimension).
//   * The input tensor's rank is either 2 or 3. The last dimension of the input
//     tensor should be the contracting dimension, i.e. [..., c_x, r_x].
//   * The filter tensor's rank is 2. The contracting dimension should be the
//     first dimension (dim 0), i.e. [c_y, r_y] where c_y == r_x.
//   * Does not consider activation fusion.
//   * Does not consider bias add fusion.
//
// TODO: b/294983811 - Merge this pattern into
// `RewriteFullIntegerQuantizedDotGeneralOp`.
// TODO: b/295264927 - `stablehlo.dot_general` with per-axis quantized operands
// is not specified in the StableHLO dialect. Update the spec to allow this.
class RewriteQuantizedDotGeneralOpToTflFullyConnectedOp
    : public OpRewritePattern<stablehlo::DotGeneralOp> {
  using OpRewritePattern<stablehlo::DotGeneralOp>::OpRewritePattern;

 public:
  LogicalResult match(stablehlo::DotGeneralOp op) const override {
    const stablehlo::DotDimensionNumbersAttr dot_dimension_nums =
        op.getDotDimensionNumbers();
    if (const int num_rhs_contracting_dims =
            dot_dimension_nums.getRhsContractingDimensions().size();
        num_rhs_contracting_dims != 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Expected number of contracting dimensions to be 1. Got: "
                 << num_rhs_contracting_dims << ".\n");
      return failure();
    }

    if (failed(MatchInput(op.getOperand(0)))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match input for quantized dot_general op.\n");
      return failure();
    }

    if (failed(MatchFilter(op.getOperand(1)))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match filter for quantized dot_general op.\n");
      return failure();
    }

    if (failed(MatchOutput(op.getResult()))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match output for quantized dot_general op.\n");
      return failure();
    }

    return success();
  }

  void rewrite(stablehlo::DotGeneralOp op,
               PatternRewriter& rewriter) const override {
    // Create the new filter constant - transpose filter value
    // from [i, o] -> [o, i]. This is because we assume `[i, o]` format for
    // `stablehlo.dot_general` (i.e. contracting dimension == 1) whereas
    // `tfl.fully_connected` accepts an OI format.
    auto filter_constant_op =
        cast<stablehlo::ConstantOp>(op.getOperand(1).getDefiningOp());

    TFL::QConstOp new_filter_constant_op =
        CreateTflConstOpForFilter(filter_constant_op, rewriter);
    const Value input_value = op.getOperand(0);
    const double input_scale = input_value.getType()
                                   .cast<TensorType>()
                                   .getElementType()
                                   .cast<UniformQuantizedType>()
                                   .getScale();
    TFL::QConstOp bias_constant_op = CreateTflConstOpForBias(
        op.getLoc(), input_scale, new_filter_constant_op, rewriter);

    const Value result_value = op.getResult();
    // Set to `nullptr` because this attribute only matters when the input is
    // dynamic-range quantized.
    const BoolAttr asymmetric_quantize_inputs = nullptr;
    auto tfl_fully_connected_op = rewriter.create<TFL::FullyConnectedOp>(
        op.getLoc(), /*output=*/result_value.getType(),
        /*input=*/input_value, /*filter=*/new_filter_constant_op.getResult(),
        /*bias=*/bias_constant_op.getResult(),
        /*fused_activation_function=*/rewriter.getStringAttr("NONE"),
        /*weights_format=*/rewriter.getStringAttr("DEFAULT"),
        /*keep_num_dims=*/rewriter.getBoolAttr(false),
        asymmetric_quantize_inputs);

    rewriter.replaceAllUsesWith(result_value,
                                tfl_fully_connected_op.getResult(0));
    rewriter.eraseOp(op);
  }

 private:
  static LogicalResult MatchInput(Value input) {
    auto input_type = input.getType().cast<TensorType>();
    if (!input_type.hasRank() ||
        !(input_type.getRank() == 2 || input_type.getRank() == 3)) {
      LLVM_DEBUG(llvm::dbgs() << "Input expected to have rank of 2 or 3. Got: "
                              << input_type << ".\n");
      return failure();
    }

    if (const auto input_element_type = input_type.getElementType();
        !IsI8F32UniformQuantizedType(input_element_type)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Expected an i8->f32 uniform quantized type. Got: "
                 << input_element_type << ".\n");
      return failure();
    }

    return success();
  }

  static LogicalResult MatchFilter(Value filter) {
    auto filter_type = filter.getType().cast<TensorType>();
    if (!filter_type.hasRank() || filter_type.getRank() != 2) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Filter tensor expected to have a tensor rank of 2. Got: "
                 << filter_type << ".\n");
      return failure();
    }

    const Type filter_element_type = filter_type.getElementType();
    if (!IsI8F32UniformQuantizedPerAxisType(filter_type.getElementType())) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Expected a per-channel uniform quantized (i8->f32) type. Got: "
          << filter_element_type << "\n");
      return failure();
    }

    if (filter_element_type.cast<UniformQuantizedPerAxisType>()
            .getQuantizedDimension() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "Quantized dimension should be 1. Got: "
                              << filter_element_type << "\n");
      return failure();
    }

    if (Operation* filter_op = filter.getDefiningOp();
        filter_op == nullptr || !isa<stablehlo::ConstantOp>(filter_op)) {
      LLVM_DEBUG(llvm::dbgs() << "Filter should be a constant.\n");
      return failure();
    }

    return success();
  }

  static LogicalResult MatchOutput(Value output) {
    const Type output_element_type =
        output.getType().cast<TensorType>().getElementType();
    if (!IsI8F32UniformQuantizedType(output_element_type)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Expected a uniform quantized (i8->f32) type. Got: "
                 << output_element_type << ".\n");
      return failure();
    }

    return success();
  }

  // Creates a new `tfl.qconst` op for the quantized filter. Transposes the
  // filter value from [i, o] -> [o, i]. This is because we assume `[i, o]`
  // format for `stablehlo.dot_general` (i.e. contracting dimension == 1)
  // whereas `tfl.fully_connected` accepts an OI format.
  TFL::QConstOp CreateTflConstOpForFilter(
      stablehlo::ConstantOp filter_constant_op,
      PatternRewriter& rewriter) const {
    const auto filter_values = filter_constant_op.getValue()
                                   .cast<DenseIntElementsAttr>()
                                   .getValues<int8_t>();

    ArrayRef<int64_t> filter_shape =
        filter_constant_op.getType().cast<TensorType>().getShape();

    // Reverse the shapes. This makes sense because it assumes that the filter
    // tensor has rank of 2 (no batch dimension).
    SmallVector<int64_t, 2> new_filter_shape(filter_shape.rbegin(),
                                             filter_shape.rend());

    // Construct the value array of transposed filter. Assumes 2D matrix.
    SmallVector<int8_t> new_filter_values(filter_values.size(), /*Value=*/0);
    for (int i = 0; i < filter_shape[0]; ++i) {
      for (int j = 0; j < filter_shape[1]; ++j) {
        const int old_idx = i * filter_shape[1] + j;
        const int new_idx = j * filter_shape[0] + i;
        new_filter_values[new_idx] = filter_values[old_idx];
      }
    }

    auto new_filter_value_attr_type = RankedTensorType::getChecked(
        filter_constant_op.getLoc(), new_filter_shape,
        /*elementType=*/rewriter.getI8Type());

    auto filter_quantized_type = filter_constant_op.getResult()
                                     .getType()
                                     .cast<TensorType>()
                                     .getElementType()
                                     .cast<UniformQuantizedPerAxisType>();

    auto new_filter_quantized_type = UniformQuantizedPerAxisType::getChecked(
        filter_constant_op.getLoc(), /*flags=*/true,
        /*storageType=*/filter_quantized_type.getStorageType(),
        /*expressedType=*/filter_quantized_type.getExpressedType(),
        /*scales=*/filter_quantized_type.getScales(),
        /*zeroPoints=*/filter_quantized_type.getZeroPoints(),
        /*quantizedDimension=*/0, /*storageTypeMin=*/llvm::minIntN(8),
        /*storageTypeMax=*/llvm::maxIntN(8));

    // Required because the quantized dimension is changed from 3 -> 0.
    auto new_filter_result_type = RankedTensorType::getChecked(
        filter_constant_op.getLoc(), /*shape=*/new_filter_shape,
        /*type=*/new_filter_quantized_type);

    auto new_filter_constant_value_attr = DenseIntElementsAttr::get(
        new_filter_value_attr_type, new_filter_values);
    return rewriter.create<TFL::QConstOp>(
        filter_constant_op.getLoc(),
        /*output=*/TypeAttr::get(new_filter_result_type),
        /*value=*/new_filter_constant_value_attr);
  }

  // Creates a new `tfl.qconst` op for the bias. The bias values are 0s, because
  // this bias a dummy bias (note that bias fusion is not considered for this
  // transformation). The quantization scale for the bias is input scale *
  // filter scale. `filter_const_op` is used to retrieve the filter scales and
  // the size of the bias constant.
  TFL::QConstOp CreateTflConstOpForBias(const Location loc,
                                        const double input_scale,
                                        TFL::QConstOp filter_const_op,
                                        PatternRewriter& rewriter) const {
    const ArrayRef<int64_t> filter_shape =
        filter_const_op.getResult().getType().getShape();
    const auto filter_quantized_element_type =
        filter_const_op.getResult()
            .getType()
            .getElementType()
            .cast<UniformQuantizedPerAxisType>();

    // The storage type is i32 for bias, which is the precision used for
    // accumulation.
    auto bias_quantized_type = UniformQuantizedPerAxisType::getChecked(
        loc, /*flags=*/true, /*storageType=*/rewriter.getI32Type(),
        /*expressedType=*/rewriter.getF32Type(), /*scales=*/
        GetBiasScales(input_scale, filter_quantized_element_type.getScales()),
        /*zeroPoints=*/filter_quantized_element_type.getZeroPoints(),
        /*quantizedDimension=*/0, /*storageTypeMin=*/llvm::minIntN(8),
        /*storageTypeMax=*/llvm::maxIntN(8));

    SmallVector<int64_t, 1> bias_shape = {filter_shape[0]};
    auto bias_type =
        RankedTensorType::getChecked(loc, bias_shape, bias_quantized_type);

    auto bias_value_type = RankedTensorType::getChecked(
        loc, std::move(bias_shape), rewriter.getI32Type());
    auto bias_value = DenseIntElementsAttr::get(
        bias_value_type, APInt(/*numBits=*/32, /*value=*/0, /*isSigned=*/true));

    return rewriter.create<TFL::QConstOp>(
        loc, /*output=*/TypeAttr::get(bias_type), /*value=*/bias_value);
  }
};

void UniformQuantizedStablehloToTflPass::runOnOperation() {
  func::FuncOp func_op = getOperation();
  MLIRContext& ctx = getContext();

  RewritePatternSet patterns(&ctx);
  patterns.add<RewriteUniformQuantizeOp, RewriteUniformDequantizeOp,
               RewriteQuantizedConvolutionOp,
               RewriteFullIntegerQuantizedDotGeneralOp,
               RewriteQuantizedDotGeneralOpToTflFullyConnectedOp>(&ctx);

  if (failed(applyPatternsAndFoldGreedily(func_op, std::move(patterns)))) {
    func_op.emitError() << "Failed to convert stablehlo ops with uniform "
                           "quantized types to tflite ops.";
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateUniformQuantizedStablehloToTflPass() {
  return std::make_unique<UniformQuantizedStablehloToTflPass>();
}

static PassRegistration<UniformQuantizedStablehloToTflPass> pass;

}  // namespace odml
}  // namespace mlir
