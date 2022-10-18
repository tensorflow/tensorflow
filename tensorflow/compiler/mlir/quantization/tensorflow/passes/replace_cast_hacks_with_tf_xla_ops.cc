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

#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>

#include "absl/strings/str_format.h"
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

// Calculates zero-point offset by reducing weights and multiply it with zp.
Value CalculateZeroPointOffset(OpBuilder &builder, Location loc, Value filter,
                               int8_t input_zp, int output_dim) {
  auto weight_shape = filter.getType().template cast<ShapedType>();
  SmallVector<int64_t> weight_non_output_indices;
  for (int64_t i : llvm::seq<int64_t>(0, weight_shape.getRank())) {
    if (i != output_dim) weight_non_output_indices.push_back(i);
  }

  Value reduction_indices_value =
      Create1DConstValue<int64_t>(builder, loc, weight_non_output_indices);
  Value zp = CreateScalarConstValue<int32_t>(builder, loc, input_zp);

  TensorType filter_type = filter.getType().dyn_cast<TensorType>();
  Value filter_i32 = builder.create<TF::CastOp>(
      loc, filter_type.clone(builder.getIntegerType(32)), filter);
  auto zp_mul_output_type = RankedTensorType::get(
      {weight_shape.getDimSize(output_dim)}, builder.getIntegerType(32));
  auto reduced = builder.create<TF::SumOp>(
      loc, zp_mul_output_type, filter_i32, reduction_indices_value,
      /*keep_dims=*/builder.getBoolAttr(false));
  TF::MulOp mul_op = builder.create<TF::MulOp>(loc, zp, reduced);
  llvm::SmallVector<Value> folded_results = ConstantFoldOpIfPossible(mul_op);
  return folded_results.front();
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
          .output();
  if (input_zp_value == 0) return xla_conv_output;

  Value zp_offset = CalculateZeroPointOffset(builder, loc, /*filter=*/filter,
                                             /*input_zp=*/input_zp_value,
                                             /*output_dim=*/num_dims - 1);
  return builder.create<TF::SubOp>(loc, xla_conv_output, zp_offset).z();
}

// Creates a XlaConvV2Op from TF Conv2DOp and returns its output.
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
                       Value weight, Value input_zp, Value output,
                       const xla::DotDimensionNumbers &dnums,
                       bool four_bit = false) {
  int32_t input_zp_value;
  if (!GetSplatValue(input_zp, input_zp_value)) {
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

  Value zp_offset =
      CalculateZeroPointOffset(builder, loc, weight, input_zp_value,
                               /*output_dim=*/1);
  return builder.create<TF::SubOp>(loc, dot_result, zp_offset);
}

Value CreateXlaDotV2OpFromTfMatMulOp(OpBuilder &builder, Location loc,
                                     Value input, Value weight, Value input_zp,
                                     Value output, BoolAttr transpose_a,
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

  return CreateXlaDotV2Op(builder, loc, input, weight, input_zp, output, dnums);
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
