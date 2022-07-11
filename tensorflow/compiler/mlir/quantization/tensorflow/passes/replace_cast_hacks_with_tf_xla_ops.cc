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
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/lite/kernels/padding.h"

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
  const int stride_h = strides[1].cast<mlir::IntegerAttr>().getInt();
  const int stride_w = strides[2].cast<mlir::IntegerAttr>().getInt();
  window_strides =
      Create1DConstValue<int32_t>(builder, loc, {stride_h, stride_w});

  const int dilation_h = dilations[1].cast<mlir::IntegerAttr>().getInt();
  const int dilation_w = dilations[2].cast<mlir::IntegerAttr>().getInt();
  lhs_dilation = Create1DConstValue<int32_t>(builder, loc, {1, 1});
  rhs_dilation =
      Create1DConstValue<int32_t>(builder, loc, {dilation_h, dilation_w});

  feature_group_count =
      CreateScalarConstValue<int32_t>(builder, loc, feature_group_cnt);
}

// For non-zero padding (input_zp != 0), adds Pad op before convolution.
Value CalculatePaddingAndPadIfNeeded(
    OpBuilder &builder, Location loc, Value input, Value filter,
    int8_t input_zp_value, ArrayAttr strides, ArrayAttr dilations,
    StringAttr conv_padding, ArrayAttr explicit_paddings, Value &padding) {
  ShapedType input_shape = input.getType().template cast<ShapedType>();
  ShapedType filter_shape = filter.getType().template cast<ShapedType>();

  int padding_h_before, padding_h_after, padding_w_before, padding_w_after;
  if (conv_padding.strref().equals("EXPLICIT")) {
    if (explicit_paddings.size() != 8) {
      mlir::emitError(loc,
                      "explicit_paddings are expected to be 8-element arrays");
      return {};
    }
    padding_h_before = explicit_paddings[2].cast<mlir::IntegerAttr>().getInt();
    padding_h_after = explicit_paddings[3].cast<mlir::IntegerAttr>().getInt();
    padding_w_before = explicit_paddings[4].cast<mlir::IntegerAttr>().getInt();
    padding_w_after = explicit_paddings[5].cast<mlir::IntegerAttr>().getInt();
  } else {
    TfLitePadding tflite_padding = conv_padding.strref().equals("VALID")
                                       ? kTfLitePaddingValid
                                       : kTfLitePaddingSame;
    int output_height, output_width;
    const int stride_h = strides[1].cast<mlir::IntegerAttr>().getInt();
    const int stride_w = strides[2].cast<mlir::IntegerAttr>().getInt();
    const int dilation_h = dilations[1].cast<mlir::IntegerAttr>().getInt();
    const int dilation_w = dilations[2].cast<mlir::IntegerAttr>().getInt();
    TfLitePaddingValues padding_values = tflite::ComputePaddingHeightWidth(
        stride_h, stride_w, dilation_h, dilation_w,
        /*in_height=*/input_shape.getDimSize(1),
        /*in_width=*/input_shape.getDimSize(2),
        /*filter_height=*/filter_shape.getDimSize(0),
        /*filter_width=*/filter_shape.getDimSize(1), tflite_padding,
        &output_height, &output_width);
    padding_h_before = padding_values.height;
    padding_h_after = padding_values.height + padding_values.height_offset;
    padding_w_before = padding_values.width;
    padding_w_after = padding_values.width + padding_values.width_offset;
  }

  if (conv_padding.strref().equals("VALID") || input_zp_value == 0 ||
      (padding_h_before == 0 && padding_h_after == 0 && padding_w_before == 0 &&
       padding_w_after == 0)) {
    padding = CreateConstValue<int32_t>(
        builder, loc, {2, 2},
        {padding_h_before, padding_h_after, padding_w_before, padding_w_after});
    return input;
  }
  padding = CreateConstValue<int32_t>(builder, loc, {2, 2}, {0, 0, 0, 0});

  Value temp_padding =
      CreateConstValue<int32_t>(builder, loc, {4, 2},
                                {0, 0, padding_h_before, padding_h_after,
                                 padding_w_before, padding_w_after, 0, 0});
  SmallVector<int64_t> output_shape(input_shape.getShape().begin(),
                                    input_shape.getShape().end());
  output_shape[1] += padding_h_before + padding_h_after;
  output_shape[2] += padding_w_before + padding_w_after;
  return builder.create<TF::PadV2Op>(
      loc, RankedTensorType::get(output_shape, builder.getI8Type()), input,
      temp_padding,
      CreateScalarConstValue<int8_t>(builder, loc, input_zp_value));
}

// Calculates zero-point offset by reducing weights and multiply it with zp.
Value CalculateZeroPointOffset(
    OpBuilder &builder, Location loc, Value filter, int8_t input_zp,
    int output_dim, const SmallVector<int64_t> &weight_non_output_indices) {
  Value reduction_indices_value =
      Create1DConstValue<int64_t>(builder, loc, weight_non_output_indices);
  Value zp = CreateScalarConstValue<int32_t>(builder, loc, input_zp);

  auto zp_mul_output_type =
      RankedTensorType::get({output_dim}, builder.getIntegerType(32));
  auto reduced = builder.create<TF::SumOp>(
      loc, zp_mul_output_type, filter, reduction_indices_value,
      /*keep_dims=*/builder.getBoolAttr(false));
  return builder.create<TF::MulOp>(loc, zp, reduced).getResult();
}

// Helper function to create a XlaConvV2Op for Conv2DOp and DepthwiseConv2DOp.
Value CreateXLAConvOp(
    OpBuilder &builder, Location loc, Value input, Value filter, Value input_zp,
    Value conv_output, ArrayAttr strides, ArrayAttr dilations,
    StringAttr conv_padding, ArrayAttr explicit_paddings, int feature_group_cnt,
    const SmallVector<int64_t> &filter_non_output_indices,
    const xla::ConvolutionDimensionNumbers &dimension_numbers) {
  int32_t input_zp_value;
  if (!GetSplatValue(input_zp, input_zp_value)) {
    mlir::emitError(
        loc, "zero point is expected to be a constant with a single value");
    return {};
  }
  if (strides.size() != 4 || dilations.size() != 4) {
    mlir::emitError(
        loc, "strides and dilations are expected to be 4-element arrays");
    return {};
  }
  ShapedType input_shape = input.getType().template cast<ShapedType>();
  ShapedType filter_shape = filter.getType().template cast<ShapedType>();
  if (!input_shape.hasRank() || input_shape.getRank() != 4 ||
      !filter_shape.hasRank() || filter_shape.getRank() != 4) {
    mlir::emitError(loc, "input and filter are expected to be 4D tensors");
    return {};
  }

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
  auto filter_type = filter.getType().dyn_cast<TensorType>();
  Value filter_i8 = builder.create<TF::CastOp>(
      loc, filter_type.clone(builder.getIntegerType(8)), filter);
  Value xla_conv_output =
      builder
          .create<TF::XlaConvV2Op>(
              loc, /*output=*/conv_output.getType(),
              /*lhs=*/input,
              /*rhs=*/filter_i8, window_strides, padding, lhs_dilation,
              rhs_dilation, feature_group_count,
              builder.getStringAttr(dimension_numbers.SerializeAsString()),
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
  const int feature_group_cnt = 1;
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
  return CreateXLAConvOp(builder, loc, input, filter, input_zp, conv_output,
                         strides, dilations, conv_padding, explicit_paddings,
                         feature_group_cnt, filter_non_output_indices, dnums);
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
