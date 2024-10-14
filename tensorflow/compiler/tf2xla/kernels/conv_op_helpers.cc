/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// XLA-specific Ops for 2D convolution.

#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal_util.h"
#include "xla/util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_grad_shape_utils.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tsl/platform/tensor_float_32_utils.h"

namespace tensorflow {
namespace {

xla::PrecisionConfig GetPrecisionConfig() {
  xla::PrecisionConfig::Precision precision =
      tsl::tensor_float_32_execution_enabled() ? xla::PrecisionConfig::DEFAULT
                                               : xla::PrecisionConfig::HIGHEST;
  xla::PrecisionConfig config;
  const int num_inputs = 2;
  config.mutable_operand_precision()->Reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    config.add_operand_precision(precision);
  }
  return config;
}

// Returns the expanded size of a filter used for depthwise convolution.
// If `shape` is [H, W, ..., M, N] returns [H, W, ..., 1, M*N].
xla::Shape GroupedFilterShapeForDepthwiseConvolution(
    const xla::Shape& filter_shape) {
  int64_t input_feature_dim = filter_shape.dimensions_size() - 2;
  int64_t output_feature_dim = filter_shape.dimensions_size() - 1;
  int64_t depthwise_multiplier = filter_shape.dimensions(output_feature_dim);
  int64_t input_feature = filter_shape.dimensions(input_feature_dim);

  // Create a [H, W, ..., 1, M*N] reshape of the filter.
  xla::Shape grouped_filter_shape = filter_shape;
  grouped_filter_shape.set_dimensions(input_feature_dim, 1);
  grouped_filter_shape.set_dimensions(output_feature_dim,
                                      depthwise_multiplier * input_feature);
  return grouped_filter_shape;
}

// Returns the transposed filter for use in BackpropInput of group convolution.
xla::XlaOp TransposeFilterForGroupConvolutionBackpropInput(
    xla::XlaOp filter, const xla::Shape& filter_shape, int64_t num_groups,
    int num_spatial_dims) {
  // 1. Reshape from [H, W, ..., filter_in_depth, out_depth] to [H, W, ...,
  // filter_in_depth, G, out_depth / G]
  int num_dims = filter_shape.dimensions_size();
  CHECK_GE(num_dims, 2);  // Crash OK
  xla::Shape new_shape = filter_shape;
  new_shape.set_dimensions(num_dims - 1, num_groups);
  new_shape.add_dimensions(filter_shape.dimensions(num_dims - 1) / num_groups);
  xla::XlaOp result = xla::Reshape(filter, new_shape.dimensions());

  // 2. Transpose to [H, W, ..., G, filter_in_depth, out_depth / G]
  std::vector<int64_t> transpose_dims(num_dims + 1);
  std::iota(transpose_dims.begin(), transpose_dims.end(), 0);
  std::swap(transpose_dims[num_spatial_dims],
            transpose_dims[num_spatial_dims + 1]);
  result = xla::Transpose(result, transpose_dims);

  // 3. Reshape to [H, W, ..., in_depth, out_depth / G]
  result = xla::Collapse(result, {num_spatial_dims, num_spatial_dims + 1});
  return result;
}

// Reshapes a filter of shape [H, W, ..., M, N] to [H, W, ..., 1, M*N]. Used to
// build a depthwise convolution.
xla::XlaOp ReshapeFilterForDepthwiseConvolution(const xla::Shape& filter_shape,
                                                xla::XlaOp filter) {
  return xla::Reshape(
      filter,
      GroupedFilterShapeForDepthwiseConvolution(filter_shape).dimensions());
}

// Performs some basic checks on ConvOpAttrs that are true for all kinds of XLA
// convolutions (as currently implemented).
absl::Status CheckConvAttrs(const ConvOpAttrs& attrs) {
  const int num_dims = attrs.num_spatial_dims + 2;
  const int attrs_strides_size = attrs.strides.size();
  if (attrs_strides_size != num_dims) {
    return errors::InvalidArgument("Sliding window strides field must specify ",
                                   num_dims, " dimensions");
  }
  int batch_dim = GetTensorBatchDimIndex(num_dims, attrs.data_format);
  int feature_dim = GetTensorFeatureDimIndex(num_dims, attrs.data_format);
  if (attrs.strides[batch_dim] != 1 || attrs.strides[feature_dim] != 1) {
    return errors::Unimplemented(
        "Current implementation does not yet support strides in the batch and "
        "depth dimensions.");
  }
  const int attrs_dilations_size = attrs.dilations.size();
  if (attrs_dilations_size != num_dims) {
    return errors::InvalidArgument("Dilations field must specify ", num_dims,
                                   " dimensions");
  }
  if (attrs.dilations[batch_dim] != 1 || attrs.dilations[feature_dim] != 1) {
    return errors::Unimplemented(
        "Current implementation does not support dilations in the batch and "
        "depth dimensions.");
  }
  for (int i = 0; i < attrs.num_spatial_dims; ++i) {
    int input_dim = GetTensorSpatialDimIndex(num_dims, attrs.data_format, i);
    if (attrs.dilations[input_dim] < 1) {
      return errors::Unimplemented("Dilation values must be positive; ", i,
                                   "th spatial dimension had dilation ",
                                   attrs.dilations[input_dim]);
    }
  }
  return absl::OkStatus();
}

// Wrapper around ConvBackpropComputeDimensions that converts from XLA shapes
// to TensorShapes.
absl::Status ConvBackpropComputeDimensionsV2XlaShapes(
    StringPiece label, int num_spatial_dims, const xla::Shape& input_shape,
    const xla::Shape& filter_shape, const xla::Shape& out_backprop_shape,
    absl::Span<const int32> dilations, const std::vector<int32>& strides,
    Padding padding, TensorFormat data_format, ConvBackpropDimensions* dims,
    absl::Span<const int64_t> explicit_paddings) {
  TensorShape input_tensor_shape, filter_tensor_shape,
      out_backprop_tensor_shape;
  TF_RETURN_IF_ERROR(XLAShapeToTensorShape(input_shape, &input_tensor_shape));
  TF_RETURN_IF_ERROR(XLAShapeToTensorShape(filter_shape, &filter_tensor_shape));
  TF_RETURN_IF_ERROR(
      XLAShapeToTensorShape(out_backprop_shape, &out_backprop_tensor_shape));
  return ConvBackpropComputeDimensionsV2(
      label, num_spatial_dims, input_tensor_shape, filter_tensor_shape,
      out_backprop_tensor_shape, dilations, strides, padding, explicit_paddings,
      data_format, dims);
}

}  // anonymous namespace

std::vector<DataType> GetXlaConvTypesForNonGpu() {
  return {DT_FLOAT, DT_BFLOAT16, DT_HALF, DT_DOUBLE, DT_INT32};
}

std::vector<DataType> GetXlaConvTypesForGpu() {
  return {DT_FLOAT, DT_BFLOAT16, DT_HALF, DT_DOUBLE};
}

absl::StatusOr<ConvOpAttrs> ConvOpAttrs::Create(int num_spatial_dims,
                                                bool depthwise,
                                                OpKernelConstruction* ctx) {
  ConvOpAttrs attrs;
  attrs.num_spatial_dims = num_spatial_dims;
  attrs.depthwise = depthwise;
  TF_RETURN_IF_ERROR(ctx->GetAttr("dilations", &attrs.dilations));
  TF_RETURN_IF_ERROR(ctx->GetAttr("strides", &attrs.strides));
  TF_RETURN_IF_ERROR(ctx->GetAttr("padding", &attrs.padding));
  if (attrs.padding == EXPLICIT) {
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("explicit_paddings", &attrs.explicit_paddings));
  }

  string data_format;
  TF_RETURN_IF_ERROR(ctx->GetAttr("data_format", &data_format));
  if (!FormatFromString(data_format, &attrs.data_format)) {
    return errors::InvalidArgument("Invalid data format: ", data_format);
  }

  TF_RETURN_IF_ERROR(CheckValidPadding(attrs.padding, attrs.explicit_paddings,
                                       /*num_dims=*/num_spatial_dims + 2,
                                       attrs.data_format));

  return attrs;
}

absl::StatusOr<ConvNDOpAttrs> ConvNDOpAttrs::Create(OpKernelConstruction* ctx) {
  ConvNDOpAttrs attrs;
  TF_RETURN_IF_ERROR(ctx->GetAttr("groups", &attrs.groups));
  TF_RETURN_IF_ERROR(ctx->GetAttr("batch_dims", &attrs.batch_dims));
  if (attrs.batch_dims < 1) {
    return absl::InvalidArgumentError("batch_dims must be non-negative.");
  }
  TF_RETURN_IF_ERROR(ctx->GetAttr("dilations", &attrs.dilations));
  TF_RETURN_IF_ERROR(ctx->GetAttr("strides", &attrs.strides));
  TF_RETURN_IF_ERROR(ctx->GetAttr("padding", &attrs.padding));
  if (attrs.padding == EXPLICIT) {
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("explicit_paddings", &attrs.explicit_paddings));
  }

  string data_format_str;
  TF_RETURN_IF_ERROR(ctx->GetAttr("data_format", &data_format_str));
  if (!(data_format_str == "CHANNELS_LAST" ||
        data_format_str == "CHANNELS_FIRST")) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unknown data format: ", data_format_str));
  }
  attrs.data_format =
      data_format_str == "CHANNELS_LAST" ? FORMAT_NHWC : FORMAT_NCHW;

  return attrs;
}

absl::StatusOr<xla::XlaOp> MakeXlaForwardConvOp(StringPiece /*type_string*/,
                                                xla::XlaOp conv_input,
                                                xla::XlaOp filter,
                                                const ConvOpAttrs& attrs) {
  TF_RETURN_IF_ERROR(CheckConvAttrs(attrs));

  auto* builder = conv_input.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape input_shape, builder->GetShape(conv_input));
  // Filter has the form [filter_rows, filter_cols, ..., in_depth, out_depth]
  TF_ASSIGN_OR_RETURN(xla::Shape filter_shape, builder->GetShape(filter));

  // For 2D convolution, there should be 4 dimensions.
  int num_dims = attrs.num_spatial_dims + 2;
  if (input_shape.dimensions_size() != num_dims) {
    return errors::InvalidArgument("input must be ", num_dims, "-dimensional",
                                   input_shape.DebugString());
  }
  if (filter_shape.dimensions_size() != num_dims) {
    return errors::InvalidArgument(
        "filter must be ", num_dims,
        "-dimensional: ", filter_shape.DebugString());
  }

  // The last two dimensions of the filter are the input and output shapes.
  int batch_dim = GetTensorBatchDimIndex(num_dims, attrs.data_format);
  int feature_dim = GetTensorFeatureDimIndex(num_dims, attrs.data_format);

  int64_t filter_in_depth = filter_shape.dimensions(attrs.num_spatial_dims),
          out_depth = filter_shape.dimensions(attrs.num_spatial_dims + 1),
          in_depth = input_shape.dimensions(feature_dim);
  // The 'C' dimension for input is in_depth.
  // It must be a multiple of the filter's in_depth.
  if (in_depth % filter_in_depth != 0) {
    return errors::InvalidArgument(
        "Depth of input must be a multiple of depth of filter: ", in_depth,
        " vs ", filter_in_depth);
  }
  int64_t feature_group_count = in_depth / filter_in_depth;
  if (out_depth % feature_group_count != 0) {
    return errors::InvalidArgument(
        "Depth of output must be a multiple of the number of groups: ",
        out_depth, " vs ", feature_group_count);
  }

  if (attrs.depthwise) {
    filter = ReshapeFilterForDepthwiseConvolution(filter_shape, filter);
  }

  xla::ConvolutionDimensionNumbers dims;
  std::vector<int64_t> window_strides(attrs.num_spatial_dims);
  std::vector<int64_t> lhs_dilation(attrs.num_spatial_dims, 1);
  std::vector<int64_t> rhs_dilation(attrs.num_spatial_dims);
  std::vector<std::pair<int64_t, int64_t>> padding(attrs.num_spatial_dims);

  dims.set_input_batch_dimension(batch_dim);
  dims.set_output_batch_dimension(batch_dim);
  dims.set_input_feature_dimension(feature_dim);
  dims.set_output_feature_dimension(feature_dim);
  dims.set_kernel_input_feature_dimension(attrs.num_spatial_dims);
  dims.set_kernel_output_feature_dimension(attrs.num_spatial_dims + 1);
  xla::PaddingType padding_type = xla::PaddingType::PADDING_INVALID;
  for (int i = 0; i < attrs.num_spatial_dims; ++i) {
    const int64_t dim =
        GetTensorSpatialDimIndex(num_dims, attrs.data_format, i);
    if (input_shape.is_dynamic_dimension(dim)) {
      TF_RET_CHECK(attrs.padding == VALID || attrs.padding == SAME)
          << "Dynamic convolution only supports valid and same padding";
      if (attrs.padding == VALID) {
        padding_type = xla::PaddingType::PADDING_VALID;
      }
      if (attrs.padding == SAME) {
        padding_type = xla::PaddingType::PADDING_SAME;
      }
    }
    dims.add_input_spatial_dimensions(dim);
    dims.add_kernel_spatial_dimensions(i);
    dims.add_output_spatial_dimensions(dim);
    window_strides[i] = attrs.strides.at(dim);
    rhs_dilation[i] = attrs.dilations.at(dim);

    if (attrs.padding == EXPLICIT) {
      padding[i] = {attrs.explicit_paddings.at(dim * 2),
                    attrs.explicit_paddings.at(dim * 2 + 1)};
    }

    int64_t unused_output_size;
    TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
        input_shape.dimensions(dim), filter_shape.dimensions(i),
        rhs_dilation[i], window_strides[i], attrs.padding, &unused_output_size,
        &padding[i].first, &padding[i].second));
  }
  xla::PrecisionConfig precision_config = GetPrecisionConfig();

  if (padding_type != xla::PaddingType::PADDING_INVALID) {
    return xla::DynamicConvForward(
        conv_input, filter, window_strides, padding, lhs_dilation, rhs_dilation,
        dims,
        /*feature_group_count=*/attrs.depthwise ? in_depth
                                                : feature_group_count,
        /*batch_group_count=*/1, &precision_config, padding_type);
  }

  return xla::ConvGeneralDilated(
      conv_input, filter, window_strides, padding, lhs_dilation, rhs_dilation,
      dims,
      /*feature_group_count=*/attrs.depthwise ? in_depth : feature_group_count,
      /*batch_group_count=*/1, &precision_config);
}

absl::StatusOr<xla::XlaOp> MakeXlaBackpropInputConvOp(
    StringPiece type_string, const xla::Shape& input_shape, xla::XlaOp filter,
    xla::XlaOp out_backprop, const ConvOpAttrs& attrs,
    xla::XlaOp* input_sizes) {
  TF_RETURN_IF_ERROR(CheckConvAttrs(attrs));

  int num_dims = attrs.num_spatial_dims + 2;
  int batch_dim = GetTensorBatchDimIndex(num_dims, attrs.data_format);
  int feature_dim = GetTensorFeatureDimIndex(num_dims, attrs.data_format);

  auto* builder = filter.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape filter_shape, builder->GetShape(filter));
  TF_ASSIGN_OR_RETURN(xla::Shape out_backprop_shape,
                      builder->GetShape(out_backprop));

  int64_t in_depth = input_shape.dimensions(feature_dim),
          filter_in_depth = filter_shape.dimensions(attrs.num_spatial_dims),
          feature_group_count =
              attrs.depthwise ? filter_in_depth : in_depth / filter_in_depth;

  xla::Shape grouped_filter_shape =
      attrs.depthwise ? GroupedFilterShapeForDepthwiseConvolution(filter_shape)
                      : filter_shape;
  // Reuse dimension computation logic from conv_grad_shape_utils.cc.
  ConvBackpropDimensions dims;
  TF_RETURN_IF_ERROR(ConvBackpropComputeDimensionsV2XlaShapes(
      type_string, attrs.num_spatial_dims, input_shape, grouped_filter_shape,
      out_backprop_shape, attrs.dilations, attrs.strides, attrs.padding,
      attrs.data_format, &dims, attrs.explicit_paddings));

  // The input gradients are computed by a convolution of the output
  // gradients and the filter, with some appropriate padding. See the
  // comment at the top of conv_grad_shape_utils.h for details.

  xla::ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(batch_dim);
  dnums.set_output_batch_dimension(batch_dim);
  dnums.set_input_feature_dimension(feature_dim);
  dnums.set_output_feature_dimension(feature_dim);

  // TF filter shape is [ H, W, ..., inC, outC ]
  // Transpose the input and output features for computing the gradient.
  dnums.set_kernel_input_feature_dimension(attrs.num_spatial_dims + 1);
  dnums.set_kernel_output_feature_dimension(attrs.num_spatial_dims);

  std::vector<int64_t> kernel_spatial_dims(attrs.num_spatial_dims);
  std::vector<std::pair<int64_t, int64_t>> padding(attrs.num_spatial_dims);
  std::vector<int64_t> lhs_dilation(attrs.num_spatial_dims);
  std::vector<int64_t> rhs_dilation(attrs.num_spatial_dims);
  std::vector<int64_t> ones(attrs.num_spatial_dims, 1);
  xla::PaddingType padding_type = xla::PaddingType::PADDING_INVALID;
  for (int i = 0; i < attrs.num_spatial_dims; ++i) {
    int64_t dim = GetTensorSpatialDimIndex(num_dims, attrs.data_format, i);
    if (out_backprop_shape.is_dynamic_dimension(dim)) {
      TF_RET_CHECK(attrs.padding == VALID || attrs.padding == SAME)
          << "Dynamic convolution only supports valid and same padding";
      if (attrs.padding == VALID) {
        padding_type = xla::PaddingType::PADDING_VALID;
      }
      if (attrs.padding == SAME) {
        padding_type = xla::PaddingType::PADDING_SAME;
      }
    }
    dnums.add_input_spatial_dimensions(dim);
    dnums.add_kernel_spatial_dimensions(i);
    dnums.add_output_spatial_dimensions(dim);

    kernel_spatial_dims[i] = i;
    padding[i] = {dims.spatial_dims[i].pad_before,
                  dims.spatial_dims[i].pad_after};
    lhs_dilation[i] = dims.spatial_dims[i].stride;
    rhs_dilation[i] = attrs.dilations[dim];
  }
  xla::PrecisionConfig precision_config = GetPrecisionConfig();

  if (feature_group_count != 1 && !attrs.depthwise) {
    filter = TransposeFilterForGroupConvolutionBackpropInput(
        filter, filter_shape, feature_group_count, attrs.num_spatial_dims);
  }
  // Mirror the filter in the spatial dimensions.
  filter = xla::Rev(filter, kernel_spatial_dims);
  if (padding_type != xla::PaddingType::PADDING_INVALID) {
    TF_RET_CHECK(input_sizes != nullptr);
    return xla::DynamicConvInputGrad(
        *input_sizes, out_backprop, filter, /*window_strides=*/ones, padding,
        lhs_dilation, rhs_dilation, dnums,
        /*feature_group_count=*/
        feature_group_count,
        /*batch_group_count=*/1, &precision_config, padding_type);
  }
  // activation gradients
  //   = gradients (with padding and dilation) <conv> mirrored_weights
  return xla::ConvGeneralDilated(out_backprop, filter, /*window_strides=*/ones,
                                 padding, lhs_dilation, rhs_dilation, dnums,
                                 /*feature_group_count=*/
                                 feature_group_count,
                                 /*batch_group_count=*/1, &precision_config);
}

absl::StatusOr<xla::XlaOp> MakeXlaBackpropFilterConvOp(
    StringPiece type_string, xla::XlaOp activations,
    const xla::Shape& filter_shape, xla::XlaOp gradients,
    const ConvOpAttrs& attrs) {
  TF_RETURN_IF_ERROR(CheckConvAttrs(attrs));

  auto* builder = activations.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape activations_shape,
                      builder->GetShape(activations));
  TF_ASSIGN_OR_RETURN(xla::Shape out_backprop_shape,
                      builder->GetShape(gradients));
  xla::XlaOp filter_backprop;

  xla::Shape input_shape = activations_shape;
  xla::Shape output_shape = out_backprop_shape;

  TensorShape input_tensor_shape, filter_tensor_shape, output_tensor_shape;
  TF_RETURN_IF_ERROR(XLAShapeToTensorShape(filter_shape, &filter_tensor_shape));
  TF_RETURN_IF_ERROR(XLAShapeToTensorShape(input_shape, &input_tensor_shape));
  TF_RETURN_IF_ERROR(XLAShapeToTensorShape(output_shape, &output_tensor_shape));

  const xla::Shape grouped_filter_shape =
      attrs.depthwise ? GroupedFilterShapeForDepthwiseConvolution(filter_shape)
                      : filter_shape;
  // Reuse dimension computation logic from conv_grad_shape_utils.cc.
  ConvBackpropDimensions dims;
  // The filter gradients are computed by a convolution of the input
  // activations and the output gradients, with some appropriate padding.
  // See the comment at the top of conv_grad_shape_utils.h for details.
  xla::ConvolutionDimensionNumbers dnums;

  TF_RETURN_IF_ERROR(ConvBackpropComputeDimensionsV2XlaShapes(
      type_string, attrs.num_spatial_dims, activations_shape,
      grouped_filter_shape, out_backprop_shape, attrs.dilations, attrs.strides,
      attrs.padding, attrs.data_format, &dims, attrs.explicit_paddings));

  // Obtain some useful dimensions:
  // The last two dimensions of the filter are the input and output shapes.
  int num_dims = attrs.num_spatial_dims + 2;
  int n_dim = GetTensorBatchDimIndex(num_dims, attrs.data_format);
  int c_dim = GetTensorFeatureDimIndex(num_dims, attrs.data_format);
  int64_t in_depth = input_shape.dimensions(c_dim),
          filter_in_depth = filter_shape.dimensions(attrs.num_spatial_dims),
          batch_group_count =
              attrs.depthwise ? filter_in_depth : in_depth / filter_in_depth;

  std::vector<std::pair<int64_t, int64_t>> padding(attrs.num_spatial_dims);
  std::vector<int64_t> rhs_dilation(attrs.num_spatial_dims);
  std::vector<int64_t> window_strides(attrs.num_spatial_dims);
  std::vector<int64_t> ones(attrs.num_spatial_dims, 1);

  // Swap n_dim and c_dim in the activations.
  dnums.set_input_batch_dimension(c_dim);
  dnums.set_input_feature_dimension(n_dim);

  // The gradients become the RHS of the convolution.
  // The gradients have shape [batch, out_rows, out_cols, ..., out_depth]
  // where the batch becomes the input feature for the convolution.
  dnums.set_kernel_input_feature_dimension(n_dim);
  dnums.set_kernel_output_feature_dimension(c_dim);

  dnums.set_output_batch_dimension(attrs.num_spatial_dims);
  dnums.set_output_feature_dimension(attrs.num_spatial_dims + 1);

  // Tensorflow filter shape is [ H, W, ..., inC, outC ].
  for (int i = 0; i < attrs.num_spatial_dims; ++i) {
    dnums.add_output_spatial_dimensions(i);
  }
  xla::PaddingType padding_type = xla::PaddingType::PADDING_INVALID;
  for (int64_t i = 0; i < attrs.num_spatial_dims; ++i) {
    int64_t dim = GetTensorSpatialDimIndex(num_dims, attrs.data_format, i);
    if (activations_shape.is_dynamic_dimension(dim)) {
      TF_RET_CHECK(attrs.padding == VALID || attrs.padding == SAME)
          << "Dynamic convolution only supports valid and same padding";
      if (attrs.padding == VALID) {
        padding_type = xla::PaddingType::PADDING_VALID;
      }
      if (attrs.padding == SAME) {
        padding_type = xla::PaddingType::PADDING_SAME;
      }
    }
    dnums.add_input_spatial_dimensions(dim);
    dnums.add_kernel_spatial_dimensions(dim);
    rhs_dilation[i] = dims.spatial_dims[i].stride;
    window_strides[i] = attrs.dilations[dim];

    // We will also need to pad the input with zeros such that after the
    // convolution, we get the right size for the filter.
    // The padded_in_rows should be such that when we convolve this with the
    // expanded_out_rows as a filter, we should get filter_rows back.

    const int64_t padded_in_size =
        dims.spatial_dims[i].expanded_output_size +
        (dims.spatial_dims[i].filter_size - 1) * attrs.dilations[dim];

    // However it can be smaller than input_rows: in this
    // case it means some of the inputs are not used.
    //
    // An example is to have input_cols = 3, filter_cols = 2 and stride = 2:
    //
    // INPUT =  [ A  B  C ]
    //
    // FILTER = [ x y ]
    //
    // and the output will only have one column: a = A * x + B * y
    //
    // and input "C" is not used at all.
    //
    // We apply negative padding in this case.
    const int64_t pad_total = padded_in_size - dims.spatial_dims[i].input_size;

    // + For the EXPLICIT padding, we pad the top/left side with the explicit
    //   padding and pad the bottom/right side with the remaining space.
    // + For the VALID padding, we don't pad anything on the top/left side
    //   and pad the bottom/right side with the remaining space.
    // + For the SAME padding, we pad top/left side the same as bottom/right
    //   side.
    //
    // In addition, if the padded input size is smaller than the input size,
    // we need to ignore some training elements of the input. We do this by
    // applying negative padding on the right/bottom.
    const int64_t pad_before =
        attrs.padding == Padding::EXPLICIT ? attrs.explicit_paddings[2 * dim]
        : attrs.padding == Padding::SAME   ? std::max<int64_t>(pad_total / 2, 0)
                                           : 0;
    padding[i] = {pad_before, pad_total - pad_before};
  }
  xla::PrecisionConfig precision_config = GetPrecisionConfig();

  // Besides padding the input, we will also expand output_rows to
  //    expanded_out_rows = (output_rows - 1) * stride + 1
  // with zeros in between:
  //
  //      a . . . b . . . c . . . d . . . e
  //
  // This is done by specifying the window dilation factors in the
  // convolution HLO below.
  if (padding_type != xla::PaddingType::PADDING_INVALID) {
    filter_backprop = xla::DynamicConvKernelGrad(
        activations, gradients, window_strides, padding, /*lhs_dilation=*/ones,
        rhs_dilation, dnums,
        /*feature_group_count=*/1,
        /*batch_group_count=*/batch_group_count, &precision_config,
        padding_type);
  } else {
    filter_backprop = xla::ConvGeneralDilated(
        activations, gradients, window_strides, padding, /*lhs_dilation=*/ones,
        rhs_dilation, dnums,
        /*feature_group_count=*/1,
        /*batch_group_count=*/batch_group_count, &precision_config);
  }

  if (attrs.depthwise) {
    filter_backprop = xla::Reshape(filter_backprop, filter_shape.dimensions());
  }

  return filter_backprop;
}

}  // namespace tensorflow
