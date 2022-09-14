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
#include <string>
#include <utility>

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
                          Value &rhs_dilation, Value &feature_group_count) {
  const int stride_h = strides[1].cast<IntegerAttr>().getInt();
  const int stride_w = strides[2].cast<IntegerAttr>().getInt();
  window_strides =
      Create1DConstValue<int32_t>(builder, loc, {stride_h, stride_w});

  const int dilation_h = dilations[1].cast<IntegerAttr>().getInt();
  const int dilation_w = dilations[2].cast<IntegerAttr>().getInt();
  lhs_dilation = Create1DConstValue<int32_t>(builder, loc, {1, 1});
  rhs_dilation =
      Create1DConstValue<int32_t>(builder, loc, {dilation_h, dilation_w});

  feature_group_count =
      CreateScalarConstValue<int32_t>(builder, loc, feature_group_cnt);
}

// Calculates zero-point offset by reducing weights and multiply it with zp.
Value CalculateZeroPointOffset(
    OpBuilder &builder, Location loc, Value filter, int8_t input_zp,
    int output_dim, const SmallVector<int64_t> &weight_non_output_indices) {
  Value reduction_indices_value =
      Create1DConstValue<int64_t>(builder, loc, weight_non_output_indices);
  Value zp = CreateScalarConstValue<int32_t>(builder, loc, input_zp);

  TensorType filter_type = filter.getType().dyn_cast<TensorType>();
  Value filter_i32 = builder.create<TF::CastOp>(
      loc, filter_type.clone(builder.getIntegerType(32)), filter);
  auto zp_mul_output_type =
      RankedTensorType::get({output_dim}, builder.getIntegerType(32));
  auto reduced = builder.create<TF::SumOp>(
      loc, zp_mul_output_type, filter_i32, reduction_indices_value,
      /*keep_dims=*/builder.getBoolAttr(false));
  TF::MulOp mul_op = builder.create<TF::MulOp>(loc, zp, reduced);
  llvm::SmallVector<Value> folded_results = ConstantFoldOpIfPossible(mul_op);
  return folded_results.front();
}

// Helper function to create a XlaConvV2Op for Conv2DOp and DepthwiseConv2DOp.
Value CreateXLAConvOp(OpBuilder &builder, Location loc, Value input,
                      Value filter, Value input_zp, Value conv_output,
                      ArrayAttr strides, ArrayAttr dilations,
                      StringAttr conv_padding, ArrayAttr explicit_paddings,
                      int feature_group_cnt) {
  int32_t input_zp_value;
  if (!GetSplatValue(input_zp, input_zp_value)) {
    emitError(loc,
              "zero point is expected to be a constant with a single value");
    return {};
  }
  if (strides.size() != 4 || dilations.size() != 4) {
    emitError(loc, "strides and dilations are expected to be 4-element arrays");
    return {};
  }
  ShapedType filter_shape = filter.getType().template cast<ShapedType>();
  SmallVector<int64_t> filter_non_output_indices = {0, 1, 2};
  xla::ConvolutionDimensionNumbers dnums;
  // Input: [N, H, W, C].
  dnums.set_input_batch_dimension(0);
  dnums.set_input_feature_dimension(3);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(2);
  // Kernel: [K, K, I, O].
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  // Output: [N, H, W, C].
  dnums.set_output_batch_dimension(0);
  dnums.set_output_feature_dimension(3);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(2);

  Value padding, window_strides, lhs_dilation, rhs_dilation,
      feature_group_count;
  PrepareXlaConvParams(builder, loc, strides, dilations, feature_group_cnt,
                       /*window_strides=*/window_strides,
                       /*lhs_dilation=*/lhs_dilation,
                       /*rhs_dilation=*/rhs_dilation,
                       /*feature_group_count=*/feature_group_count);

  input = CalculatePaddingAndPadIfNeeded(
      builder, loc, input, filter, input_zp_value, strides, dilations,
      conv_padding, explicit_paddings, padding);
  Value xla_conv_output =
      builder
          .create<TF::XlaConvV2Op>(
              loc, /*output_type=*/conv_output.getType(),
              /*lhs=*/input,
              /*rhs=*/filter, window_strides, padding, lhs_dilation,
              rhs_dilation, feature_group_count,
              builder.getStringAttr(dnums.SerializeAsString()),
              /*precision_config=*/builder.getStringAttr(""))
          .output();
  if (input_zp_value == 0) return xla_conv_output;

  Value zp_offset = CalculateZeroPointOffset(
      builder, loc, /*filter=*/filter, /*input_zp=*/input_zp_value,
      /*output_dim=*/filter_shape.getDimSize(3),
      /*weight_non_output_indices=*/filter_non_output_indices);
  return builder.create<TF::SubOp>(loc, xla_conv_output, zp_offset);
}

// Creates a XlaConvV2Op from TF Conv2DOp and returns its output.
Value CreateXLAConvOpFromTFConv2DOp(OpBuilder &builder, Location loc,
                                    Value input, Value filter, Value input_zp,
                                    Value conv_output, ArrayAttr strides,
                                    ArrayAttr dilations,
                                    StringAttr conv_padding,
                                    ArrayAttr explicit_paddings) {
  ShapedType input_shape = input.getType().template cast<ShapedType>();
  ShapedType filter_shape = filter.getType().template cast<ShapedType>();
  if (!input_shape.hasRank() || input_shape.getRank() != 4 ||
      !filter_shape.hasRank() || filter_shape.getRank() != 4) {
    emitError(loc, "input and filter are expected to be 4D tensors");
    return {};
  }

  const int feature_group_cnt =
      input_shape.getDimSize(3) / filter_shape.getDimSize(2);
  return CreateXLAConvOp(builder, loc, input, filter, input_zp, conv_output,
                         strides, dilations, conv_padding, explicit_paddings,
                         feature_group_cnt);
}

// Creates a XlaConvV2Op from TF DepthConv2DOp and returns its output.
Value CreateXLAConvOpFromTFDepthwiseConv2DOp(
    OpBuilder &builder, Location loc, Value input, Value filter, Value input_zp,
    Value conv_output, ArrayAttr strides, ArrayAttr dilations,
    StringAttr conv_padding, ArrayAttr explicit_paddings) {
  ShapedType input_shape = input.getType().template cast<ShapedType>();
  ShapedType filter_shape = filter.getType().template cast<ShapedType>();
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
  return CreateXLAConvOp(builder, loc, input, new_filter, input_zp, conv_output,
                         strides, dilations, conv_padding, explicit_paddings,
                         feature_group_cnt);
}

// Helper function to create cast-hack Matmul.
// TODO(b/242661546): Replace cast-hack Matmul with XlaDotV2Op.
Value CreateCastHackMatmul(OpBuilder &builder, Location loc, Value input,
                           Value weight, Value input_zp, Value output,
                           BoolAttr transpose_a) {
  int32_t input_zp_value;
  if (!GetSplatValue(input_zp, input_zp_value)) {
    emitError(loc,
              "zero point is expected to be a constant with a single value");
    return {};
  }

  TensorType input_type = input.getType().dyn_cast<TensorType>();
  Value input_i32 = builder.create<TF::CastOp>(
      loc, input_type.clone(builder.getIntegerType(32)), input);

  TensorType weight_type = weight.getType().dyn_cast<TensorType>();
  Value weight_identity =
      builder.create<TF::IdentityOp>(loc, weight.getType(), weight);
  Value weight_i32 = builder.create<TF::CastOp>(
      loc, weight_type.clone(builder.getIntegerType(32)), weight_identity);

  Value dot_result =
      builder
          .create<TF::MatMulOp>(loc, /*output=*/output.getType(),
                                /*lhs=*/input_i32,
                                /*rhs=*/weight_i32,
                                /*transpose_a=*/transpose_a,
                                /*transpose_b=*/builder.getBoolAttr(false))
          .getResult();

  ShapedType weight_shape = weight.getType().template cast<ShapedType>();
  SmallVector<int64_t> filter_non_output_indices = {0};
  Value zp_offset = CalculateZeroPointOffset(
      builder, loc, /*filter=*/weight, /*input_zp=*/input_zp_value,
      /*output_dim=*/weight_shape.getDimSize(1),
      /*weight_non_output_indices=*/filter_non_output_indices);
  return builder.create<TF::SubOp>(loc, dot_result, zp_offset);
}

Value AddCastHackToTFMatMulOp(OpBuilder &builder, Location loc, Value input,
                              Value weight, Value input_zp, Value output,
                              BoolAttr transpose_a, BoolAttr transpose_b) {
  // Transpose and constantf-fold the weight if needed.
  if (transpose_b.getValue()) {
    Value perm = Create1DConstValue<int32_t>(builder, loc, {1, 0});
    auto transpose_op = builder.create<TF::TransposeOp>(loc, weight, perm);
    weight = ConstantFoldOpIfPossible(transpose_op).front();
  }

  return CreateCastHackMatmul(builder, loc, input, weight, input_zp, output,
                              transpose_a);
}

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/replace_cast_hacks_with_tf_xla_ops.inc"

void ReplaceCastHacksWithTFXLAOpsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateReplaceCastHacksWithTFXLAOpsPass() {
  return std::make_unique<ReplaceCastHacksWithTFXLAOpsPass>();
}

static PassRegistration<ReplaceCastHacksWithTFXLAOpsPass> pass;

}  // namespace mlir::quant
