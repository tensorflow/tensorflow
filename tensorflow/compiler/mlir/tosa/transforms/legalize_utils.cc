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
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/kernels/internal/common.h"
#include "tensorflow/compiler/mlir/lite/kernels/internal/quantization_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_common.h"
#include "xla/tsl/framework/fixedpoint/FixedPoint.h"

// Implements legalization and post-legalization optimization helper functions

namespace mlir {
namespace tosa {

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
  auto e_ty = mlir::cast<ShapedType>(input_value.getType()).getElementType();
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

  DenseI64ArrayAttr shape_attr = rewriter.getDenseI64ArrayAttr(static_dims);
  auto output_ty = tensorflow::GetTypeFromTFTensorShape(static_dims, e_ty);
  return rewriter
      .create<tosa::ReshapeOp>(op->getLoc(), output_ty, input_value, shape_attr)
      .getResult();
}

// Create a TOSA rescale op from TFLite scaling multiplier, scaling shift, zero
// points and rounding mode
Value buildRescale(PatternRewriter& rewriter, Operation* op,
                   ShapedType output_type, Value input_val,
                   int32_t scale_multiplier, int32_t scale_shit,
                   int64_t input_zp, int64_t output_zp, bool double_round,
                   bool scale32) {
  auto rescale_op = CreateOpAndInfer<tosa::RescaleOp>(
      rewriter, op->getLoc(), output_type, input_val,
      rewriter.getI32IntegerAttr(static_cast<int32_t>(input_zp)),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(output_zp)),
      rewriter.getDenseI32ArrayAttr({scale_multiplier}),
      rewriter.getDenseI8ArrayAttr({static_cast<int8_t>(scale_shit)}),
      rewriter.getBoolAttr(scale32), rewriter.getBoolAttr(double_round),
      rewriter.getBoolAttr(false));

  return rescale_op.getResult();
}

// Create a TOSA rescale op from TFLite scaling, zero points and rounding mode
Value buildRescale(PatternRewriter& rewriter, Operation* op,
                   ShapedType output_type, Value input_val, double scale,
                   int64_t input_zp, int64_t output_zp, bool double_round,
                   bool scale32) {
  int32_t multiplier;
  int32_t shift;

  int32_t scale_width = scale32 ? 32 : 16;

  computeMultiplierAndShift(scale, multiplier, shift, scale_width);

  return buildRescale(rewriter, op, output_type, input_val, multiplier, shift,
                      input_zp, output_zp, double_round, scale32);
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

  return buildRescale(rewriter, op, output_type, input_val,
                      input_scale_multiplier, input_scale_shift, input_zp,
                      /*input_zp=*/0, IsTFLDoubleRoundingMode(),
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

  // Potentially check input_shape == output_shape here
  return buildRescale(rewriter, op, output_type, input_val, output_scale,
                      /*input_zp=*/0, output_zp, IsTFLDoubleRoundingMode(),
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
  bool double_round = scale32;

  if (auto weight_per_tensor_qtype =
          dyn_cast<mlir::quant::UniformQuantizedType>(
              weight_type.getElementType())) {
    // Per-tensor quantization
    double weight_scale = weight_per_tensor_qtype.getScale();

    int32_t multiplier;
    int32_t shift;

    double op_tensor_scale = (input_scale * weight_scale) / output_scale;

    computeMultiplierAndShift(op_tensor_scale, multiplier, shift, scale_width);

    auto rescale_op = CreateOpAndInfer<tosa::RescaleOp>(
        rewriter, op->getLoc(), output_type, conv_val,
        rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(output_zp),
        rewriter.getDenseI32ArrayAttr({multiplier}),
        rewriter.getDenseI8ArrayAttr({static_cast<int8_t>(shift)}),
        rewriter.getBoolAttr(scale32), rewriter.getBoolAttr(double_round),
        rewriter.getBoolAttr(false));

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

      computeMultiplierAndShift(op_channel_scale, multiplier, shift,
                                scale_width);

      multiplier_arr.push_back(multiplier);
      shift_arr.push_back(static_cast<int8_t>(shift));
    }

    auto rescale_op = CreateOpAndInfer<tosa::RescaleOp>(
        rewriter, op->getLoc(), output_type, conv_val,
        rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(output_zp),
        rewriter.getDenseI32ArrayAttr(multiplier_arr),
        rewriter.getDenseI8ArrayAttr(shift_arr), rewriter.getBoolAttr(scale32),
        rewriter.getBoolAttr(double_round), rewriter.getBoolAttr(true));

    return rescale_op.getResult();

  } else {
    op->emitOpError("buildConvRescaleOp: unknown weight quantized type");
    return nullptr;
  }
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
      // Assume that any value close to 0 (or negtive values) represents the max
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
// Follow PopulateLookupTable() tensorflow/lite/kernels/activations.cc
Value getTosaConst8bitTable(PatternRewriter& rewriter, Operation* op,
                            double input_scale, int32_t input_zp,
                            double output_scale, int32_t output_zp,
                            std::function<double(double)> func) {
  SmallVector<int8_t, 256> table;

  for (int32_t i = -128; i < 128; i++) {
    double dequantized = input_scale * (i - input_zp);
    double transformed = func(dequantized);

    double max = (output_scale > 1.0) ? DBL_MAX : (DBL_MAX * output_scale);
    if (transformed >= max) {
      table.push_back(INT8_MAX);
      continue;
    }

    int32_t rescaled = std::llround(transformed / output_scale);
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

// Create a 16-bit TOSA TABLE constant tensor with int16[513] array.
// Output is restricted to [-1.0, 1.0].
// Follow gen_lut() tensorflow/lite/kernels/internal/common.h
Value getTosaConst16bitTable(PatternRewriter& rewriter, Operation* op,
                             std::function<double(double)> func, double min,
                             double max) {
  SmallVector<int16_t, 513> table;

  double step = (max - min) / 512.0f;
  double half_step = step / 2.0f;
  for (int32_t i = 0; i < 512; i++) {
    int32_t sample_val = std::llround(func(min + (i * step)) * 32768.0);
    double midpoint_interp_val =
        std::round(((func(min + (i + 1) * step) * 32768.0) +
                    std::round(func(min + (i * step)) * 32768.0)) /
                   2.0);
    double midpoint_val =
        std::round(func(min + (i * step) + half_step) * 32768.0);
    double midpoint_err = midpoint_interp_val - midpoint_val;
    int32_t bias = std::llround(midpoint_err / 2.0);

    table.push_back(static_cast<int16_t>(
        std::min(std::max(sample_val - bias, -32768), 32767)));
  }

  int32_t max_val = std::llround(func(max) * 32768.0);
  table.push_back(
      static_cast<int16_t>(std::min(std::max(max_val, -32768), 32767)));

  auto const_type =
      tensorflow::GetTypeFromTFTensorShape({513}, rewriter.getIntegerType(16));
  auto const_attr = DenseElementsAttr::get(const_type, llvm::ArrayRef(table));

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

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
                                  float val) {
  auto const_type =
      tensorflow::GetTypeFromTFTensorShape({}, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, val);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Create a 32-bit integer constant operator from an int
Value getTosaConstTensorSingleI32(PatternRewriter& rewriter, Operation* op,
                                  int32_t val) {
  auto const_type =
      tensorflow::GetTypeFromTFTensorShape({}, rewriter.getIntegerType(32));
  auto const_attr = DenseElementsAttr::get(const_type, val);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Create an expected bitwidth integer constant operator based on the type
// parameter.
Value getTosaConstTensorScalarInt(ImplicitLocOpBuilder& builder, Type type,
                                  int64_t val) {
  auto bit_width = type.getIntOrFloatBitWidth();
  auto const_type = tensorflow::GetTypeFromTFTensorShape(
      {}, builder.getIntegerType(bit_width));
  auto const_attr =
      SplatElementsAttr::get(const_type, builder.getIntegerAttr(type, val));

  auto const_op =
      builder.create<tosa::ConstOp>(builder.getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Create a vector from a 32-bit value tensor.  Returns the size of
// the new vector or -1 on error.
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
    tensorflow::Status status = tensorflow::GetWindowedOutputSizeVerbose(
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

    int total_padding = ((ifm_size - 1) * dim_stride + filter_size - ofm_size);
    total_padding = total_padding > 0 ? total_padding : 0;

    pad_before = total_padding / 2;
    pad_after = total_padding - pad_before;

    computed_paddings.push_back(pad_before);
    computed_paddings.push_back(pad_after);
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
  if (failed(applyPatternsAndFoldGreedily(func, patterns, config))) {
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
    auto ety = op.getValue().getShapedType().getElementType();
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

}  // namespace tosa
}  // namespace mlir
