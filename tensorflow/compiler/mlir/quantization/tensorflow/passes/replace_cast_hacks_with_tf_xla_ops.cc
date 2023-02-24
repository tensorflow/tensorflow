/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/tf_to_xla_attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace mlir::quant {
namespace {

// Replaces mixed-type Conv and Matmul cast hacks with TF XLA ops.
// TODO(b/228403741): Support conversion for dynamic-shaped TF ops.
class ReplaceCastHacksWithTFXLAOpsPass
    : public PassWrapper<ReplaceCastHacksWithTFXLAOpsPass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceCastHacksWithTFXLAOpsPass)

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-replace-cast-hacks-with-tf-xla-ops";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Replace mixed-type Conv and Matmul cast hacks with TF XLA ops.";
  }

  void runOnOperation() override;
};

// Generates params for the XLA Convolution op.
void PrepareXlaConvParams(OpBuilder &builder, Location loc, ArrayAttr strides,
                          ArrayAttr dilations, int feature_group_cnt,
                          Value &window_strides, Value &lhs_dilation,
                          Value &rhs_dilation, Value &feature_group_count,
                          int num_dims) {
  SmallVector<int32_t> lhs_dilation_values(num_dims - 2, 1);
  SmallVector<int32_t> stride_values, rhs_dilation_values;
  for (int64_t i : llvm::seq<int64_t>(1, num_dims - 1)) {
    stride_values.push_back(strides[i].cast<IntegerAttr>().getInt());
    rhs_dilation_values.push_back(dilations[i].cast<IntegerAttr>().getInt());
  }
  window_strides = Create1DConstValue<int32_t>(builder, loc, stride_values);
  lhs_dilation = Create1DConstValue<int32_t>(builder, loc, lhs_dilation_values);
  rhs_dilation = Create1DConstValue<int32_t>(builder, loc, rhs_dilation_values);

  feature_group_count =
      CreateScalarConstValue<int32_t>(builder, loc, feature_group_cnt);
}

// Calculates other_tensor_zp * tensor for zero point offset calculation.
Value CreateZeroPointPartialOffset(OpBuilder &builder, Location loc,
                                   Value tensor, int8_t other_tensor_zp,
                                   const ArrayRef<int64_t> output_dims) {
  if (other_tensor_zp == 0) {
    return CreateScalarConstValue<int32_t>(builder, loc, 0);
  }

  auto shape = tensor.getType().template cast<ShapedType>();
  SmallVector<int64_t> non_output_indices;
  for (int64_t i : llvm::seq<int64_t>(0, shape.getRank())) {
    if (absl::c_count(output_dims, i) == 0) {
      non_output_indices.push_back(i);
    }
  }

  auto reduction_indices_value =
      Create1DConstValue<int64_t>(builder, loc, non_output_indices);
  auto zp = CreateScalarConstValue<int32_t>(builder, loc, other_tensor_zp);

  TensorType tensor_type = tensor.getType().dyn_cast<TensorType>();
  Value tensor_i32 = builder.create<TF::CastOp>(
      loc, tensor_type.clone(builder.getIntegerType(32)), tensor);
  auto reduced =
      builder.create<TF::SumOp>(loc, tensor_i32, reduction_indices_value,
                                /*keep_dims=*/builder.getBoolAttr(true));
  auto mul_op = builder.create<TF::MulOp>(loc, zp, reduced);

  llvm::SmallVector<Value> folded_results = ConstantFoldOpIfPossible(mul_op);
  return folded_results.front();
}

// Calculates zero-point offset by reducing the weight and multiply it with zp.
// Originally, we have:
//   output = (int8_input - input_zp) * (int8_weight - weight_zp)
// So, offset = input_zp * int8_weight + weight_zp * int8_input
// - input_zp * weight_zp.
// This function calculates the `offset` value mentioned above. Note that the
// `output_dims` is the weight dimensions that are not contracted, so they
// appear in the output shape.
Value CalculateZeroPointOffset(OpBuilder &builder, Location loc, Value input,
                               Value weight, int8_t input_zp, int8_t weight_zp,
                               const ArrayRef<int64_t> input_output_dims,
                               const ArrayRef<int64_t> weight_output_dims) {
  Value zp_input_contribution = CreateZeroPointPartialOffset(
      builder, loc, input, weight_zp, input_output_dims);
  Value zp_weight_contribution = CreateZeroPointPartialOffset(
      builder, loc, weight, input_zp, weight_output_dims);

  auto weight_shape = weight.getType().template cast<ShapedType>();
  SmallVector<int64_t> weight_non_output_indices;
  for (auto i : llvm::seq<int64_t>(0, weight_shape.getRank())) {
    if (absl::c_count(weight_output_dims, i) == 0) {
      weight_non_output_indices.push_back(i);
    }
  }

  if (input_zp != 0 && weight_zp != 0) {
    // Add two contributions, and a zeropoint modification term
    // Consider two quantized matrices P, Q with zero points z, w. Let's say the
    // dimensions are l X n, n X m.
    // What we want to calculate is: R = matmul(P-z, Q-w).
    // Then r_ij = sigma(k) (p_ik - z) * (q_kj - w)
    //           = sigma(k)(p_ik * q_kj) - w * sigma(k)p_ik - z * sigma(k)q_kj
    //             + sigma(k)z*w.
    // zp_input_contribution = z * sigma(k)q_kj
    // zp_weight_contribution = w * sigma(k)p_ik
    // In case z != 0 and w != 0, we need to additionally calculate sigma(k)z*w,
    // which is: # of reduced dim(n in this case) * input_zp * weight_zp
    int32_t static_dim_total = 1;
    Value accum_dynamic_dim = nullptr;
    llvm::SmallVector<int64_t> weight_non_output_dynamic_indices;
    for (const int64_t weight_idx : weight_non_output_indices) {
      if (weight_shape.isDynamicDim(weight_idx)) {
        weight_non_output_dynamic_indices.push_back(weight_idx);
      } else {
        static_dim_total *= weight_shape.getDimSize(weight_idx);
      }
    }

    if (!weight_non_output_dynamic_indices.empty()) {
      // Has dynamic shapes.
      auto weight_shape_op = builder.create<TF::ShapeOp>(
          loc, weight, /*use32Bit=*/builder.getBoolAttr(false));

      auto slice_output_type = RankedTensorType::get({1}, builder.getI64Type());
      auto slice_stride = CreateConstValue<int64_t>(builder, loc, {1}, {1});
      for (int64_t weight_idx : weight_non_output_dynamic_indices) {
        auto start = CreateConstValue<int64_t>(builder, loc, {1}, {weight_idx});
        auto end =
            CreateConstValue<int64_t>(builder, loc, {1}, {weight_idx + 1});
        auto sliced_shape_op = builder.create<TF::StridedSliceOp>(
            loc, slice_output_type, weight_shape_op, start, end, slice_stride);
        if (accum_dynamic_dim == nullptr) {
          accum_dynamic_dim = sliced_shape_op->getResults().front();
        } else {
          accum_dynamic_dim =
              builder
                  .create<TF::MulOp>(loc, accum_dynamic_dim, sliced_shape_op)
                  ->getResults()
                  .front();
        }
      }
    }

    const int32_t zp_constant_offset = static_cast<int32_t>(input_zp) *
                                       static_cast<int32_t>(weight_zp) *
                                       static_dim_total;
    auto zp_offset_value =
        CreateScalarConstValue<int32_t>(builder, loc, zp_constant_offset);
    if (accum_dynamic_dim != nullptr) {
      accum_dynamic_dim =
          builder
              .create<mlir::TF::CastOp>(
                  loc, mlir::RankedTensorType::get({1}, builder.getI32Type()),
                  accum_dynamic_dim)
              ->getResults()
              .front();
      auto mul_op =
          builder.create<TF::MulOp>(loc, accum_dynamic_dim, zp_offset_value);
      zp_offset_value = mul_op->getResults().front();
    }

    auto offset_sum = builder.create<TF::AddOp>(loc, zp_input_contribution,
                                                zp_weight_contribution);
    auto offset_op =
        builder.create<TF::SubOp>(loc, offset_sum, zp_offset_value);

    llvm::SmallVector<Value> folded_results =
        ConstantFoldOpIfPossible(offset_op);
    return folded_results.front();
  }

  if (input_zp != 0) return zp_weight_contribution;
  return zp_input_contribution;
}

// Helper function to create a XlaConvV2Op for Conv2DOp, DepthwiseConv2DOp and
// Conv3DOp.
Value CreateXlaConvOp(OpBuilder &builder, Location loc, Value input,
                      Value filter, Value input_zp, Value conv_output,
                      ArrayAttr strides, ArrayAttr dilations,
                      StringAttr conv_padding, ArrayAttr explicit_paddings,
                      int feature_group_cnt, bool four_bit = false,
                      int num_dims = 4) {
  int32_t input_zp_value;
  if (!GetSplatValue(input_zp, input_zp_value)) {
    emitError(loc,
              "zero point is expected to be a constant with a single value");
    return {};
  }
  if (strides.size() != num_dims || dilations.size() != num_dims) {
    emitError(loc,
              absl::StrFormat(
                  "strides and dilations are expected to be %d-element arrays",
                  num_dims));
    return {};
  }

  xla::ConvolutionDimensionNumbers dnums;
  // Input: [N, H, W, C] for Conv2D or [N, D, H, W, C] for Conv3D.
  dnums.set_input_batch_dimension(0);
  dnums.set_input_feature_dimension(num_dims - 1);
  // Kernel: [K, K, I, O] for Conv2D or [K, K, K, I, O] for Conv3D.
  dnums.set_kernel_input_feature_dimension(num_dims - 2);
  dnums.set_kernel_output_feature_dimension(num_dims - 1);
  // Output: [N, H, W, C] for Conv2D or [N, D, H, W, C] for Conv3D.
  dnums.set_output_batch_dimension(0);
  dnums.set_output_feature_dimension(num_dims - 1);

  for (int64_t i : llvm::seq<int64_t>(1, num_dims - 1)) {
    dnums.add_input_spatial_dimensions(i);
    dnums.add_kernel_spatial_dimensions(i - 1);
    dnums.add_output_spatial_dimensions(i);
  }

  Value padding, window_strides, lhs_dilation, rhs_dilation,
      feature_group_count;
  PrepareXlaConvParams(builder, loc, strides, dilations, feature_group_cnt,
                       /*window_strides=*/window_strides,
                       /*lhs_dilation=*/lhs_dilation,
                       /*rhs_dilation=*/rhs_dilation,
                       /*feature_group_count=*/feature_group_count,
                       /*num_dims=*/num_dims);

  input = CalculatePaddingAndPadIfNeeded(
      builder, loc, input, filter, input_zp_value, strides, dilations,
      conv_padding, explicit_paddings, padding, num_dims);

  std::string precision_config_str;
  if (four_bit) {
    input = PackOperand(builder, loc, input, /*pack_dim=*/num_dims - 1);
    filter = PackOperand(builder, loc, filter, /*pack_dim=*/num_dims - 2);
    xla::PrecisionConfig precision_config;
    precision_config.add_operand_precision(xla::PrecisionConfig::PACKED_NIBBLE);
    precision_config.add_operand_precision(xla::PrecisionConfig::PACKED_NIBBLE);
    precision_config_str = precision_config.SerializeAsString();
  }
  Value xla_conv_output =
      builder
          .create<TF::XlaConvV2Op>(
              loc, /*output_type=*/conv_output.getType(),
              /*lhs=*/input,
              /*rhs=*/filter, window_strides, padding, lhs_dilation,
              rhs_dilation, feature_group_count,
              builder.getStringAttr(dnums.SerializeAsString()),
              /*precision_config=*/builder.getStringAttr(precision_config_str))
          .getOutput();

  // Dynamic-range quantization wil always fall into this case.
  if (input_zp_value == 0) return xla_conv_output;

  Value zp_offset = CalculateZeroPointOffset(
      builder, loc, input, filter, input_zp_value,
      /*weight_zp=*/0,
      /*input_output_dims=*/ArrayRef<int64_t>({0}),
      /*weight_output_dims=*/ArrayRef<int64_t>({num_dims - 1}));
  return builder.create<TF::SubOp>(loc, xla_conv_output, zp_offset).getZ();
}

// Creates a XlaConvV2Op from TF Conv2DOp and returns its output. The returned
// value will be used as an input of the next op.
Value CreateXlaConvOpFromTfConv2dOp(OpBuilder &builder, Location loc,
                                    Value input, Value filter, Value input_zp,
                                    Value conv_output, ArrayAttr strides,
                                    ArrayAttr dilations,
                                    StringAttr conv_padding,
                                    ArrayAttr explicit_paddings) {
  auto input_shape = input.getType().template cast<ShapedType>();
  auto filter_shape = filter.getType().template cast<ShapedType>();
  if (!input_shape.hasRank() || input_shape.getRank() != 4 ||
      !filter_shape.hasRank() || filter_shape.getRank() != 4) {
    emitError(loc, "input and filter are expected to be 4D tensors");
    return {};
  }

  const int feature_group_cnt =
      input_shape.getDimSize(3) / filter_shape.getDimSize(2);
  return CreateXlaConvOp(builder, loc, input, filter, input_zp, conv_output,
                         strides, dilations, conv_padding, explicit_paddings,
                         feature_group_cnt);
}

// Creates a XlaConvV2Op from TF DepthwiseConv2DOp and returns its output.
Value CreateXlaConvOpFromTfDepthwiseConv2dOp(
    OpBuilder &builder, Location loc, Value input, Value filter, Value input_zp,
    Value conv_output, ArrayAttr strides, ArrayAttr dilations,
    StringAttr conv_padding, ArrayAttr explicit_paddings) {
  auto input_shape = input.getType().template cast<ShapedType>();
  auto filter_shape = filter.getType().template cast<ShapedType>();
  if (!input_shape.hasRank() || input_shape.getRank() != 4 ||
      !filter_shape.hasRank() || filter_shape.getRank() != 4) {
    emitError(loc, "input and filter are expected to be 4D tensors");
    return {};
  }
  const int feature_group_cnt = input_shape.getDimSize(3);

  // Reshape the filter to [K, K, 1, I * O].
  llvm::SmallVector<int64_t> new_filter_shape{
      filter_shape.getDimSize(0), filter_shape.getDimSize(1), 1,
      filter_shape.getDimSize(2) * filter_shape.getDimSize(3)};
  Value new_filter = builder.create<TF::ReshapeOp>(
      loc,
      RankedTensorType::get(new_filter_shape, filter_shape.getElementType()),
      filter, Create1DConstValue(builder, loc, new_filter_shape));
  return CreateXlaConvOp(builder, loc, input, new_filter, input_zp, conv_output,
                         strides, dilations, conv_padding, explicit_paddings,
                         feature_group_cnt);
}

// Creates a XlaConvV2Op from TF Conv3DOp and returns its output.
Value CreateXlaConvOpFromTfConv3dOp(OpBuilder &builder, Location loc,
                                    Value input, Value filter, Value input_zp,
                                    Value conv_output, ArrayAttr strides,
                                    ArrayAttr dilations,
                                    StringAttr conv_padding) {
  auto input_shape = input.getType().template cast<ShapedType>();
  auto filter_shape = filter.getType().template cast<ShapedType>();
  if (!input_shape.hasRank() || input_shape.getRank() != 5 ||
      !filter_shape.hasRank() || filter_shape.getRank() != 5) {
    emitError(loc, "input and filter are expected to be 5D tensors");
    return {};
  }
  const int feature_group_cnt =
      input_shape.getDimSize(4) / filter_shape.getDimSize(3);

  return CreateXlaConvOp(builder, loc, input, filter, input_zp, conv_output,
                         strides, dilations, conv_padding,
                         /*explicit_paddings=*/nullptr, feature_group_cnt,
                         /*four_bit=*/false, /*num_dims=*/5);
}

// Helper function to create an XlaDotV2Op.
Value CreateXlaDotV2Op(OpBuilder &builder, Location loc, Value input,
                       Value weight, Value input_zp, Value weight_zp,
                       Value output, const xla::DotDimensionNumbers &dnums,
                       bool four_bit = false) {
  int32_t input_zp_value = 0;
  int32_t weight_zp_value = 0;
  if (input_zp != nullptr && !GetSplatValue(input_zp, input_zp_value)) {
    emitError(loc,
              "zero point is expected to be a constant with a single value");
    return {};
  }

  if (weight_zp != nullptr && !GetSplatValue(weight_zp, weight_zp_value)) {
    emitError(loc,
              "zero point is expected to be a constant with a single value");
    return {};
  }

  std::string precision_config_str;
  if (four_bit) {
    input = PackOperand(builder, loc, input, /*pack_dim=*/1);
    weight = PackOperand(builder, loc, weight, /*pack_dim=*/0);
    xla::PrecisionConfig precision_config;
    precision_config.add_operand_precision(xla::PrecisionConfig::PACKED_NIBBLE);
    precision_config.add_operand_precision(xla::PrecisionConfig::PACKED_NIBBLE);
    precision_config_str = precision_config.SerializeAsString();
  }

  Value dot_result =
      builder
          .create<TF::XlaDotV2Op>(
              loc, /*output=*/output.getType(),
              /*lhs=*/input,
              /*rhs=*/weight,
              /*dimension_numbers=*/
              builder.getStringAttr(dnums.SerializeAsString()),
              /*precision_config=*/builder.getStringAttr(precision_config_str))
          .getResult();

  if (input_zp_value == 0) return dot_result;

  auto input_shape = input.getType().template cast<ShapedType>();
  auto weight_shape = weight.getType().template cast<ShapedType>();
  SmallVector<int64_t> input_output_dims(input_shape.getRank() - 2);
  SmallVector<int64_t> weight_output_dims(weight_shape.getRank() - 2);
  absl::c_iota(input_output_dims, 0);
  absl::c_iota(weight_output_dims, 0);
  input_output_dims.push_back(weight_shape.getRank() - 2);
  weight_output_dims.push_back(weight_shape.getRank() - 1);

  Value zp_offset = CalculateZeroPointOffset(
      builder, loc, input, weight, input_zp_value, weight_zp_value,
      ArrayRef<int64_t>(input_output_dims),
      ArrayRef<int64_t>(weight_output_dims));
  return builder.create<TF::SubOp>(loc, dot_result, zp_offset);
}

Value CreateXlaDotV2OpFromTfMatMulOp(OpBuilder &builder, Location loc,
                                     Value input, Value weight, Value input_zp,
                                     Value weight_zp, Value output,
                                     BoolAttr transpose_a,
                                     BoolAttr transpose_b) {
  // Transpose and constant-fold the weight if needed.
  if (transpose_b.getValue()) {
    Value perm = Create1DConstValue<int32_t>(builder, loc, {1, 0});
    auto transpose_op = builder.create<TF::TransposeOp>(loc, weight, perm);
    weight = ConstantFoldOpIfPossible(transpose_op).front();
  }

  xla::DotDimensionNumbers dnums;
  dnums.add_rhs_contracting_dimensions(0);
  if (transpose_a.getValue()) {
    dnums.add_lhs_contracting_dimensions(0);
  } else {
    dnums.add_lhs_contracting_dimensions(1);
  }

  return CreateXlaDotV2Op(builder, loc, input, weight, input_zp, weight_zp,
                          output, dnums);
}

// Gets the broadcasted shapes of the input and weight of the BatchMatMul op
// from their types. If there are dynamic dimesions, these shapes couldn't be
// used as the arguments for the BroadcastTo ops.
llvm::Optional<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
GetBroadcastShapesForBatchMatmul(ShapedType input_type,
                                 ShapedType weight_type) {
  ArrayRef<int64_t> input_shape = input_type.getShape();
  ArrayRef<int64_t> weight_shape = weight_type.getShape();

  const int64_t num_matmul_dim = 2;
  const int64_t num_input_batch_dim = input_type.getRank() - num_matmul_dim;
  const int64_t num_weight_batch_dim = weight_type.getRank() - num_matmul_dim;

  ArrayRef<int64_t> input_batch_dims =
      input_shape.slice(0, num_input_batch_dim);
  ArrayRef<int64_t> weight_batch_dims =
      weight_shape.slice(0, num_weight_batch_dim);
  ArrayRef<int64_t> input_matmul_dims =
      input_shape.slice(num_input_batch_dim, num_matmul_dim);
  ArrayRef<int64_t> weight_matmul_dims =
      weight_shape.slice(num_weight_batch_dim, num_matmul_dim);

  SmallVector<int64_t> broadcasted_batch_dims;
  if (!OpTrait::util::getBroadcastedShape(input_batch_dims, weight_batch_dims,
                                          broadcasted_batch_dims)) {
    return std::nullopt;
  }
  SmallVector<int64_t> broadcasted_input_shape(broadcasted_batch_dims);
  broadcasted_input_shape.append(input_matmul_dims.begin(),
                                 input_matmul_dims.end());
  SmallVector<int64_t> broadcasted_weight_shape(broadcasted_batch_dims);
  broadcasted_weight_shape.append(weight_matmul_dims.begin(),
                                  weight_matmul_dims.end());

  return std::make_pair(std::move(broadcasted_input_shape),
                        std::move(broadcasted_weight_shape));
}

// Broadcasts batch dimensions of the input and weight of the BatchMatMul
// op. In XLA, shapes are all constants, so all operations created in this
// function, except BroadcastTo, are expected to be folded.
void BroadcastBatchDimensionsForBatchMatMul(OpBuilder &builder, Location loc,
                                            Value &input, Value &weight) {
  ShapedType input_type = input.getType().template cast<ShapedType>();
  ShapedType weight_type = weight.getType().template cast<ShapedType>();
  const int32_t input_rank = input_type.getRank();
  const int32_t weight_rank = weight_type.getRank();
  const int32_t broadcasted_rank = std::max(input_rank, weight_rank);

  const int32_t num_matmul_dim = 2;
  const int32_t num_input_batch_dim = input_rank - num_matmul_dim;
  const int32_t num_weight_batch_dim = weight_rank - num_matmul_dim;
  if (num_input_batch_dim == 0 && num_weight_batch_dim == 0) return;

  // If the broadcasted shapes can be calculated statically, only add two
  // BroadcastTo ops for input and weight.
  auto broadcasted_shapes_or =
      GetBroadcastShapesForBatchMatmul(input_type, weight_type);
  if (!broadcasted_shapes_or.has_value()) return;
  const auto broadcasted_input_type = RankedTensorType::get(
      broadcasted_shapes_or->first, input_type.getElementType());
  const auto broadcasted_weight_type = RankedTensorType::get(
      broadcasted_shapes_or->second, weight_type.getElementType());

  if (broadcasted_input_type.hasStaticShape() &&
      broadcasted_weight_type.hasStaticShape()) {
    input = builder.create<TF::BroadcastToOp>(
        loc, broadcasted_input_type, input,
        Create1DConstValue(builder, loc, broadcasted_shapes_or->first));
    weight = builder.create<TF::BroadcastToOp>(
        loc, broadcasted_weight_type, weight,
        Create1DConstValue(builder, loc, broadcasted_shapes_or->second));
    return;
  }

  const Value zero = Create1DConstValue<int32_t>(builder, loc, {0});
  const Value num_matmul_dim_value =
      Create1DConstValue<int32_t>(builder, loc, {num_matmul_dim});
  const Value num_input_batch_dim_value =
      Create1DConstValue<int32_t>(builder, loc, {num_input_batch_dim});
  const Value num_weight_batch_dim_value =
      Create1DConstValue<int32_t>(builder, loc, {num_weight_batch_dim});

  // Decompose the input and weight shape into batch and matmul dimensions.
  Value input_shape = builder.create<TF::ShapeOp>(
      loc, input, /*use32Bit=*/builder.getBoolAttr(false));
  Value input_batch_dims = builder.create<TF::SliceOp>(
      loc, RankedTensorType::get({num_input_batch_dim}, builder.getI64Type()),
      input_shape, zero, num_input_batch_dim_value);
  Value input_matmul_dims = builder.create<TF::SliceOp>(
      loc, RankedTensorType::get({num_matmul_dim}, builder.getI64Type()),
      input_shape, num_input_batch_dim_value, num_matmul_dim_value);

  Value weight_shape = builder.create<TF::ShapeOp>(
      loc, weight, /*use32Bit=*/builder.getBoolAttr(false));
  Value weight_batch_dims = builder.create<TF::SliceOp>(
      loc, RankedTensorType::get({num_weight_batch_dim}, builder.getI64Type()),
      weight_shape, zero, num_weight_batch_dim_value);
  Value weight_matmul_dims = builder.create<TF::SliceOp>(
      loc, RankedTensorType::get({num_matmul_dim}, builder.getI64Type()),
      weight_shape, num_weight_batch_dim_value, num_matmul_dim_value);

  // Calculate the broadcasted shapes.
  Value broadcasted_batch_dims = builder.create<TF::BroadcastArgsOp>(
      loc,
      RankedTensorType::get({broadcasted_rank - num_matmul_dim},
                            builder.getI64Type()),
      input_batch_dims, weight_batch_dims);
  Type broadcasted_shape_type =
      RankedTensorType::get({broadcasted_rank}, builder.getI64Type());

  const Value zero_scalar = CreateScalarConstValue<int32_t>(builder, loc, 0);
  Value broacasted_input_shape = builder.create<TF::ConcatOp>(
      loc, broadcasted_shape_type, /*concat_dim=*/zero_scalar,
      ValueRange{broadcasted_batch_dims, input_matmul_dims});
  Value broacasted_weight_shape = builder.create<TF::ConcatOp>(
      loc, broadcasted_shape_type, /*concat_dim=*/zero_scalar,
      ValueRange{broadcasted_batch_dims, weight_matmul_dims});

  // Broadcast input and weight with the calculated shapes.
  input = builder.create<TF::BroadcastToOp>(loc, broadcasted_input_type, input,
                                            broacasted_input_shape);
  weight = builder.create<TF::BroadcastToOp>(loc, broadcasted_weight_type,
                                             weight, broacasted_weight_shape);
}

Value CreateXlaDotV2OpFromTfBatchMatMulOp(OpBuilder &builder, Location loc,
                                          Value input, Value weight,
                                          Value input_zp, Value weight_zp,
                                          Value output, BoolAttr adj_x,
                                          BoolAttr adj_y) {
  // TensorFlow BatchMatMulOp allows the batch dimensions to be broadcastable
  // while the XlaDotV2Op doesn't. So we have to broadcast them beforehand.
  BroadcastBatchDimensionsForBatchMatMul(builder, loc, input, weight);

  // Both input and weight have the same rank after broadcasting.
  ShapedType weight_shape = weight.getType().template cast<ShapedType>();
  int num_batch_dim = weight_shape.getRank() - 2;

  // Transpose and constant-fold the weight if needed.
  if (adj_y.getValue()) {
    SmallVector<int32_t> perm_values(num_batch_dim);
    absl::c_iota(perm_values, 0);
    perm_values.push_back(num_batch_dim + 1);
    perm_values.push_back(num_batch_dim);
    Value perm = Create1DConstValue<int32_t>(builder, loc, perm_values);
    auto transpose_op = builder.create<TF::TransposeOp>(loc, weight, perm);
    weight = ConstantFoldOpIfPossible(transpose_op).front();
  }

  xla::DotDimensionNumbers dnums;
  for (int i : llvm::seq<int32_t>(0, num_batch_dim)) {
    dnums.add_lhs_batch_dimensions(i);
    dnums.add_rhs_batch_dimensions(i);
  }
  dnums.add_rhs_contracting_dimensions(num_batch_dim);
  if (adj_x.getValue()) {
    dnums.add_lhs_contracting_dimensions(num_batch_dim);
  } else {
    dnums.add_lhs_contracting_dimensions(num_batch_dim + 1);
  }

  return CreateXlaDotV2Op(builder, loc, input, weight, input_zp, weight_zp,
                          output, dnums);
}

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/replace_cast_hacks_with_tf_xla_ops.inc"

void ReplaceCastHacksWithTFXLAOpsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  populateWithGenerated(patterns);
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    func.emitError() << "quant-replace-cast-hacks-with-tf-xla-ops failed.";
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateReplaceCastHacksWithTFXLAOpsPass() {
  return std::make_unique<ReplaceCastHacksWithTFXLAOpsPass>();
}

static PassRegistration<ReplaceCastHacksWithTFXLAOpsPass> pass;

}  // namespace mlir::quant
