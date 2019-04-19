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

#include "tensorflow/compiler/xla/client/lib/conv_op_helpers.h"

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"

namespace xla {
namespace {

// Returns the expanded size of a filter used for depthwise convolution.
// If `shape` is [H, W, ..., M, N] returns [H, W, ..., M, M*N].
Shape ExpandedFilterShapeForDepthwiseConvolution(const Shape& shape) {
  int num_dims = shape.dimensions_size();
  CHECK_GE(num_dims, 2);  // Crash OK
  Shape expanded_shape = shape;
  expanded_shape.set_dimensions(
      num_dims - 1,
      shape.dimensions(num_dims - 2) * shape.dimensions(num_dims - 1));
  return expanded_shape;
}

// Returns the transposed filter for use in BackpropInput of group convolution.
XlaOp TransposeFilterForGroupConvolutionBackpropInput(const XlaOp& filter,
                                                      const Shape& filter_shape,
                                                      int64 num_groups,
                                                      int num_spatial_dims) {
  // 1. Reshape from [H, W, ..., filter_in_depth, out_depth] to [H, W, ...,
  // filter_in_depth, G, out_depth / G]
  int num_dims = filter_shape.dimensions_size();
  CHECK_GE(num_dims, 2);  // Crash OK
  Shape new_shape = filter_shape;
  new_shape.set_dimensions(num_dims - 1, num_groups);
  new_shape.add_dimensions(filter_shape.dimensions(num_dims - 1) / num_groups);
  XlaOp result = Reshape(filter, new_shape.dimensions());

  // 2. Transpose to [H, W, ..., G, filter_in_depth, out_depth / G]
  std::vector<int64> transpose_dims(num_dims + 1);
  std::iota(transpose_dims.begin(), transpose_dims.end(), 0);
  std::swap(transpose_dims[num_spatial_dims],
            transpose_dims[num_spatial_dims + 1]);
  result = Transpose(result, transpose_dims);

  // 3. Reshape to [H, W, ..., in_depth, out_depth / G]
  result = Collapse(result, {num_spatial_dims, num_spatial_dims + 1});
  return result;
}

// Returns the transposed input for use in BackpropFilter of group convolution.
XlaOp TransposeInputForGroupConvolutionBackpropFilter(const XlaOp& input,
                                                      const Shape& input_shape,
                                                      int64 num_groups,
                                                      int batch_dim,
                                                      int depth_dim) {
  // 1. Reshape the depth_dim C into [G, C/G]
  int num_dims = input_shape.dimensions_size();
  std::vector<int64> reshape_dims = input_shape.dimensions();
  reshape_dims[depth_dim] = reshape_dims[depth_dim] / num_groups;
  reshape_dims.insert(reshape_dims.begin() + depth_dim, num_groups);
  XlaOp result = Reshape(input, reshape_dims);

  // 2. Transpose G to the axis before N, e.g.: [G, N, H, W, C/G]
  std::vector<int64> transpose_dims(num_dims + 1);
  std::iota(transpose_dims.begin(), transpose_dims.end(),
            0);  // e.g.: [0, 1, 2, 3, 4] -> [N, H, W, G, C/G]
  transpose_dims.erase(transpose_dims.begin() + depth_dim);
  transpose_dims.insert(
      transpose_dims.begin() + batch_dim,
      depth_dim);  // e.g.: [3, 0, 1, 2, 4] -> [G, N, H, W, C/G]
  result = Transpose(result, transpose_dims);

  // 3. Merge [G, N] to [G*N]
  result = Collapse(result, {batch_dim, batch_dim + 1});
  return result;
}

// Create a mask for depthwise convolution that will make a normal convolution
// produce the same results as a depthwise convolution. For a [2, 2, 3, 2]
// depthwise filter this returns a [2, 2, 3, 6] tensor
//   1 1 0 0 0 0   1 1 0 0 0 0
//   0 0 1 1 0 0   0 0 1 1 0 0
//   0 0 0 0 1 1   0 0 0 0 1 1
//
//   1 1 0 0 0 0   1 1 0 0 0 0
//   0 0 1 1 0 0   0 0 1 1 0 0
//   0 0 0 0 1 1   0 0 0 0 1 1
//
// The first step is to create a iota A with iota_dimension = 2
//   0 0 0 0 0 0   0 0 0 0 0 0
//   1 1 1 1 1 1   1 1 1 1 1 1
//   2 2 2 2 2 2   2 2 2 2 2 2
//
//   0 0 0 0 0 0   0 0 0 0 0 0
//   1 1 1 1 1 1   1 1 1 1 1 1
//   2 2 2 2 2 2   2 2 2 2 2 2
//
// and another iota B with iota_dimension = 3
//   0 1 2 3 4 5  0 1 2 3 4 5
//   0 1 2 3 4 5  0 1 2 3 4 5
//   0 1 2 3 4 5  0 1 2 3 4 5
//
//   0 1 2 3 4 5  0 1 2 3 4 5
//   0 1 2 3 4 5  0 1 2 3 4 5
//   0 1 2 3 4 5  0 1 2 3 4 5
//
// and divide B by 2 to get
//   0 0 1 1 2 2  0 0 1 1 2 2
//   0 0 1 1 2 2  0 0 1 1 2 2
//   0 0 1 1 2 2  0 0 1 1 2 2
//
//   0 0 1 1 2 2  0 0 1 1 2 2
//   0 0 1 1 2 2  0 0 1 1 2 2
//   0 0 1 1 2 2  0 0 1 1 2 2
//
// Finally compare A and B and return the result at the beginning of the
// comment.
XlaOp CreateExpandedFilterMask(const Shape& filter_shape, XlaBuilder* builder) {
  Shape expanded_filter_shape =
      ExpandedFilterShapeForDepthwiseConvolution(filter_shape);
  int64 depthwise_multiplier =
      filter_shape.dimensions(filter_shape.dimensions_size() - 1);

  // Create two iotas with the shape of the expanded filter, one of them with
  // the iota dimension chosen as the feature dimension, and the other a iota
  // with the iota dimension chosen as the expanded output feature dimension.
  std::vector<int64> iota_dimensions(expanded_filter_shape.dimensions().begin(),
                                     expanded_filter_shape.dimensions().end());
  Shape iota_shape = ShapeUtil::MakeShape(S32, iota_dimensions);
  XlaOp input_feature_iota =
      Iota(builder, iota_shape, /*iota_dimension=*/iota_dimensions.size() - 2);
  XlaOp expanded_feature_iota =
      Iota(builder, iota_shape, /*iota_dimension=*/iota_dimensions.size() - 1);

  // Divide 'expanded_feature_iota' by the depthwise_multiplier to create
  // [0 0 1 1 2 2] ... in the example in the function comment.
  expanded_feature_iota = Div(
      expanded_feature_iota,
      ConstantR0WithType(builder, PrimitiveType::S32, depthwise_multiplier));

  // Compare 'input_feature_iota' with 'expanded_feature_iota' to create a
  // diagonal predicate.
  return Eq(expanded_feature_iota, input_feature_iota);
}

// Reshapes a filter of shape [H, W, ..., M, N] to [H, W, ..., 1, M*N]. Used to
// build a depthwise convolution.
XlaOp ReshapeFilterForDepthwiseConvolution(const Shape& filter_shape,
                                           const XlaOp& filter) {
  int64 input_feature_dim = filter_shape.dimensions_size() - 2;
  int64 output_feature_dim = filter_shape.dimensions_size() - 1;
  int64 depthwise_multiplier = filter_shape.dimensions(output_feature_dim);
  int64 input_feature = filter_shape.dimensions(input_feature_dim);

  // Create a [H, W, ..., 1, N*M] reshape of the filter.
  Shape implicit_broadcast_filter_shape = filter_shape;
  implicit_broadcast_filter_shape.set_dimensions(input_feature_dim, 1);
  implicit_broadcast_filter_shape.set_dimensions(
      output_feature_dim, depthwise_multiplier * input_feature);
  return Reshape(filter,
                 AsInt64Slice(implicit_broadcast_filter_shape.dimensions()));
}

// Reduces the results of the convolution with an expanded filter to the
// non-expanded filter.
XlaOp ContractFilterForDepthwiseBackprop(const Shape& filter_shape,
                                         const XlaOp& filter_backprop,
                                         XlaBuilder* builder) {
  auto masked_expanded_filter =
      Select(CreateExpandedFilterMask(filter_shape, builder), filter_backprop,
             ZerosLike(filter_backprop));

  auto elem_type = filter_shape.element_type();
  return Reshape(
      // This reduce does not need inputs to be converted with
      // XlaHelpers::SumAccumulationType() since the select above guarantees
      // that only one element is non zero, so there cannot be accumulated
      // precision error.
      Reduce(masked_expanded_filter, Zero(builder, elem_type),
             CreateScalarAddComputation(elem_type, builder),
             {filter_shape.dimensions_size() - 2}),
      AsInt64Slice(filter_shape.dimensions()));
}

// Performs some basic checks on ConvOpAttrs that are true for all kinds of XLA
// convolutions (as currently implemented).
Status CheckConvAttrs(const ConvOpAttrs& attrs) {
  const int num_dims = attrs.num_spatial_dims + 2;
  if (attrs.strides.size() != num_dims) {
    return InvalidArgument(
        "Sliding window strides field must specify %d dimensions", num_dims);
  }
  int batch_dim = attrs.data_format.input_batch_dimension();
  int feature_dim = attrs.data_format.input_feature_dimension();
  if (attrs.strides[batch_dim] != 1 || attrs.strides[feature_dim] != 1) {
    return Unimplemented(
        "Current implementation does not yet support strides in the batch and "
        "depth dimensions.");
  }
  if (attrs.dilations.size() != num_dims) {
    return InvalidArgument("Dilations field must specify %d dimensions",
                           num_dims);
  }
  if (attrs.dilations[batch_dim] != 1 || attrs.dilations[feature_dim] != 1) {
    return Unimplemented(
        "Current implementation does not support dilations in the batch and "
        "depth dimensions.");
  }
  for (int i = 0; i < attrs.num_spatial_dims; ++i) {
    int input_dim = attrs.data_format.input_spatial_dimensions(i);
    if (attrs.dilations[input_dim] < 1) {
      return Unimplemented(
          "Dilation values must be positive; %dth spatial dimension had "
          "dilation %d",
          i, attrs.dilations[input_dim]);
    }
  }
  return Status::OK();
}

// Information about a single spatial dimension for a convolution
// backpropagation.
struct ConvBackpropSpatialDimension {
  int64 input_size;
  int64 filter_size;
  int64 output_size;
  int64 stride;
  int64 dilation;
  // Output size after scaling by the stride.
  int64 expanded_output_size;
  // Number of padding elements to be added before/after this dimension of
  // the input when computing Conv?DBackpropInput.
  int64 pad_before, pad_after;
};

// Computed dimensions for a backwards convolution.
struct ConvBackpropDimensions {
  // Information about each spatial dimension.
  std::vector<ConvBackpropSpatialDimension> spatial_dims;
  // Batch size.
  int64 batch_size;
  // Input and output feature depth.
  int64 in_depth, out_depth;
};

Status ConvBackpropExtractAndVerifyDimension(
    absl::Span<const int64> input_shape, absl::Span<const int64> filter_shape,
    absl::Span<const int64> output_shape, absl::Span<const int32> dilations,
    const std::vector<int32>& strides, int64 padding_before,
    int64 padding_after, int spatial_dim, int filter_spatial_dim,
    ConvBackpropSpatialDimension* dim) {
  dim->input_size = input_shape.at(spatial_dim);
  dim->filter_size = filter_shape.at(filter_spatial_dim);
  dim->output_size = output_shape.at(spatial_dim);
  dim->stride = strides[spatial_dim];
  dim->dilation = dilations[spatial_dim];
  int64 effective_filter_size = (dim->filter_size - 1) * dim->dilation + 1;
  int64 out_size = (dim->input_size + padding_before + padding_after -
                    effective_filter_size + dim->stride) /
                   dim->stride;
  if (dim->output_size != out_size) {
    return InvalidArgument(
        "ConvBackpropExtractAndVerifyDimension: Size of out_backprop doesn't "
        "match computed: actual = %ld, "
        "computed = %ld, spatial_dim: %d, input: %ld, filter: %ld, output: "
        "%ld, stride: %ld, dilation: %ld",
        dim->output_size, out_size, spatial_dim, dim->input_size,
        dim->filter_size, dim->output_size, dim->stride, dim->dilation);
  }

  dim->expanded_output_size = (dim->output_size - 1) * dim->stride + 1;
  const auto padded_out_size = dim->input_size + effective_filter_size - 1;
  dim->pad_before = effective_filter_size - 1 - padding_before;
  dim->pad_after =
      padded_out_size - dim->expanded_output_size - dim->pad_before;
  VLOG(2) << "ConvBackpropExtractAndVerifyDimension: expanded_out = "
          << dim->expanded_output_size
          << ", effective_filter_size = " << effective_filter_size
          << ", padded_out = " << padded_out_size
          << ", pad_before = " << dim->pad_before
          << ", pad_after = " << dim->pad_after
          << ", dilation = " << dim->dilation << ", strides = " << dim->stride;
  return Status::OK();
}

// Verifies that the dimensions all match, and computes sizes/padding for the
// spatial dimensions.
Status ConvBackpropComputeDimensions(
    absl::string_view label, int num_spatial_dims,
    absl::Span<const int64> input_shape, absl::Span<const int64> filter_shape,
    absl::Span<const int64> out_backprop_shape,
    absl::Span<const int32> dilations, const std::vector<int32>& strides,
    absl::Span<const int64> explicit_paddings,
    const ConvolutionDimensionNumbers& data_format,
    ConvBackpropDimensions* dims) {
  // The + 2 in the following line is for the batch and feature dimensions.
  const int num_dims = num_spatial_dims + 2;
  if (input_shape.size() != num_dims) {
    return InvalidArgument("%s: input must be %d-dimensional", label, num_dims);
  }
  if (filter_shape.size() != num_dims) {
    return InvalidArgument("%s: filter must be %d-dimensional", label,
                           num_dims);
  }
  if (out_backprop_shape.size() != num_dims) {
    return InvalidArgument("%s: out_backprop must be %d-dimensional", label,
                           num_dims);
  }
  int batch_dim = data_format.input_batch_dimension();
  dims->batch_size = input_shape.at(batch_dim);
  if (dims->batch_size != out_backprop_shape.at(batch_dim)) {
    return InvalidArgument(
        "%s: input and out_backprop must have the same batch size, input "
        "batch: %ld outbackprop batch: %ld batch_dim: %d",
        label, dims->batch_size, out_backprop_shape.at(batch_dim), batch_dim);
  }

  int feature_dim = data_format.input_feature_dimension();
  dims->in_depth = input_shape.at(feature_dim);
  // The input and output feature dimensions are the second last and last
  // dimensions of the filter Tensor.
  VLOG(2) << "input vs filter_in depth " << dims->in_depth << " "
          << filter_shape.at(num_dims - 2);
  if (dims->in_depth % filter_shape.at(num_dims - 2)) {
    return InvalidArgument(
        "%s: input depth must be evenly divisible by filter depth", label);
  }
  dims->out_depth = filter_shape.at(num_dims - 1);
  if (dims->out_depth != out_backprop_shape.at(feature_dim)) {
    return InvalidArgument(
        "%s: filter and out_backprop must have the same out_depth", label);
  }
  dims->spatial_dims.resize(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    int image_dim = data_format.input_spatial_dimensions(i);
    int64 padding_before = -1, padding_after = -1;
    padding_before = explicit_paddings[2 * image_dim];
    padding_after = explicit_paddings[2 * image_dim + 1];
    TF_RETURN_IF_ERROR(ConvBackpropExtractAndVerifyDimension(
        input_shape, filter_shape, out_backprop_shape, dilations, strides,
        padding_before, padding_after, image_dim, i, &dims->spatial_dims[i]));
  }
  return Status::OK();
}

}  // anonymous namespace

StatusOr<XlaOp> MakeXlaForwardConvOp(absl::string_view /*type_string*/,
                                     XlaOp conv_input, XlaOp filter,
                                     const ConvOpAttrs& attrs,
                                     const PrecisionConfig* precision_config) {
  TF_RETURN_IF_ERROR(CheckConvAttrs(attrs));

  auto* builder = conv_input.builder();
  TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(conv_input));
  // Filter has the form [filter_rows, filter_cols, ..., in_depth, out_depth]
  TF_ASSIGN_OR_RETURN(Shape filter_shape, builder->GetShape(filter));

  // For 2D convolution, there should be 4 dimensions.
  int num_dims = attrs.num_spatial_dims + 2;
  if (input_shape.dimensions_size() != num_dims) {
    return InvalidArgument("input must be %d-dimensional: %s", num_dims,
                           input_shape.DebugString());
  }
  if (filter_shape.dimensions_size() != num_dims) {
    return InvalidArgument("filter must be %d-dimensional: %s", num_dims,
                           filter_shape.DebugString());
  }

  // The last two dimensions of the filter are the input and output shapes.
  int batch_dim = attrs.data_format.input_batch_dimension();
  int feature_dim = attrs.data_format.input_feature_dimension();

  int64 filter_in_depth = filter_shape.dimensions(attrs.num_spatial_dims),
        out_depth = filter_shape.dimensions(attrs.num_spatial_dims + 1),
        in_depth = input_shape.dimensions(feature_dim);
  // The 'C' dimension for input is in_depth.
  // It must be a multiple of the filter's in_depth.
  if (in_depth % filter_in_depth != 0) {
    return InvalidArgument(
        "Depth of input must be a multiple of depth of filter: %d vs %d",
        in_depth, filter_in_depth);
  }
  int64 feature_group_count = in_depth / filter_in_depth;
  if (out_depth % feature_group_count != 0) {
    return InvalidArgument(
        "Depth of output must be a multiple of the number of groups: %d vs %d",
        out_depth, feature_group_count);
  }

  if (attrs.depthwise) {
    filter = ReshapeFilterForDepthwiseConvolution(filter_shape, filter);
  }

  ConvolutionDimensionNumbers dims;
  std::vector<int64> window_strides(attrs.num_spatial_dims);
  std::vector<int64> lhs_dilation(attrs.num_spatial_dims, 1);
  std::vector<int64> rhs_dilation(attrs.num_spatial_dims);
  std::vector<std::pair<int64, int64>> padding(attrs.num_spatial_dims);

  dims.set_input_batch_dimension(batch_dim);
  dims.set_output_batch_dimension(batch_dim);
  dims.set_input_feature_dimension(feature_dim);
  dims.set_output_feature_dimension(feature_dim);
  dims.set_kernel_input_feature_dimension(attrs.num_spatial_dims);
  dims.set_kernel_output_feature_dimension(attrs.num_spatial_dims + 1);

  for (int i = 0; i < attrs.num_spatial_dims; ++i) {
    const int64 dim = attrs.data_format.input_spatial_dimensions(i);
    dims.add_input_spatial_dimensions(dim);
    dims.add_kernel_spatial_dimensions(i);
    dims.add_output_spatial_dimensions(dim);
    window_strides[i] = attrs.strides.at(dim);
    rhs_dilation[i] = attrs.dilations.at(dim);
    padding[i] = {attrs.explicit_paddings.at(dim * 2),
                  attrs.explicit_paddings.at(dim * 2 + 1)};
  }

  return ConvGeneralDilated(
      conv_input, filter, window_strides, padding, lhs_dilation, rhs_dilation,
      dims,
      /*feature_group_count=*/attrs.depthwise ? in_depth : feature_group_count,
      /*batch_group_count=*/1, precision_config);
}

StatusOr<XlaOp> MakeXlaBackpropInputConvOp(
    absl::string_view type_string, const Shape& input_shape, XlaOp filter,
    XlaOp out_backprop, const ConvOpAttrs& attrs,
    const PrecisionConfig* precision_config) {
  TF_RETURN_IF_ERROR(CheckConvAttrs(attrs));

  int batch_dim = attrs.data_format.input_batch_dimension();
  int feature_dim = attrs.data_format.input_feature_dimension();

  auto* builder = filter.builder();
  TF_ASSIGN_OR_RETURN(Shape filter_shape, builder->GetShape(filter));
  TF_ASSIGN_OR_RETURN(Shape out_backprop_shape,
                      builder->GetShape(out_backprop));

  int64 in_depth = input_shape.dimensions(feature_dim),
        filter_in_depth = filter_shape.dimensions(attrs.num_spatial_dims),
        feature_group_count = in_depth / filter_in_depth;

  Shape expanded_filter_shape =
      attrs.depthwise ? ExpandedFilterShapeForDepthwiseConvolution(filter_shape)
                      : filter_shape;
  // Reuse dimension computation logic from conv_grad_ops.cc.
  ConvBackpropDimensions dims;
  TF_RETURN_IF_ERROR(ConvBackpropComputeDimensions(
      type_string, attrs.num_spatial_dims, input_shape.dimensions(),
      expanded_filter_shape.dimensions(), out_backprop_shape.dimensions(),
      attrs.dilations, attrs.strides, attrs.explicit_paddings,
      attrs.data_format, &dims));

  // The input gradients are computed by a convolution of the output
  // gradients and the filter, with some appropriate padding. See the
  // comment at the top of conv_grad_ops.h for details.

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(batch_dim);
  dnums.set_output_batch_dimension(batch_dim);
  dnums.set_input_feature_dimension(feature_dim);
  dnums.set_output_feature_dimension(feature_dim);

  // TF filter shape is [ H, W, ..., inC, outC ]
  // Transpose the input and output features for computing the gradient.
  dnums.set_kernel_input_feature_dimension(attrs.num_spatial_dims + 1);
  dnums.set_kernel_output_feature_dimension(attrs.num_spatial_dims);

  std::vector<int64> kernel_spatial_dims(attrs.num_spatial_dims);
  std::vector<std::pair<int64, int64>> padding(attrs.num_spatial_dims);
  std::vector<int64> lhs_dilation(attrs.num_spatial_dims);
  std::vector<int64> rhs_dilation(attrs.num_spatial_dims);
  std::vector<int64> ones(attrs.num_spatial_dims, 1);
  for (int i = 0; i < attrs.num_spatial_dims; ++i) {
    int64 dim = attrs.data_format.input_spatial_dimensions(i);
    dnums.add_input_spatial_dimensions(dim);
    dnums.add_kernel_spatial_dimensions(i);
    dnums.add_output_spatial_dimensions(dim);

    kernel_spatial_dims[i] = i;
    padding[i] = {dims.spatial_dims[i].pad_before,
                  dims.spatial_dims[i].pad_after};
    lhs_dilation[i] = dims.spatial_dims[i].stride;
    rhs_dilation[i] = attrs.dilations[dim];
  }

  if (feature_group_count != 1 && !attrs.depthwise) {
    filter = TransposeFilterForGroupConvolutionBackpropInput(
        filter, filter_shape, feature_group_count, attrs.num_spatial_dims);
  }
  // Mirror the filter in the spatial dimensions.
  filter = Rev(filter, kernel_spatial_dims);

  // activation gradients
  //   = gradients (with padding and dilation) <conv> mirrored_weights
  return ConvGeneralDilated(
      out_backprop, filter, /*window_strides=*/ones, padding, lhs_dilation,
      rhs_dilation, dnums,
      /*feature_group_count=*/
      attrs.depthwise ? out_backprop_shape.dimensions(feature_dim) /
                            filter_shape.dimensions(attrs.num_spatial_dims + 1)
                      : feature_group_count,
      /*batch_group_count=*/1, precision_config);
}

StatusOr<XlaOp> MakeXlaBackpropFilterConvOp(
    absl::string_view type_string, XlaOp activations, const Shape& filter_shape,
    XlaOp out_backprop, const ConvOpAttrs& attrs,
    const PrecisionConfig* precision_config) {
  TF_RETURN_IF_ERROR(CheckConvAttrs(attrs));

  auto* builder = activations.builder();
  TF_ASSIGN_OR_RETURN(Shape activations_shape, builder->GetShape(activations));
  TF_ASSIGN_OR_RETURN(Shape out_backprop_shape,
                      builder->GetShape(out_backprop));
  XlaOp filter_backprop;

  Shape input_shape = activations_shape;
  Shape output_shape = out_backprop_shape;

  const Shape expanded_filter_shape =
      attrs.depthwise ? ExpandedFilterShapeForDepthwiseConvolution(filter_shape)
                      : filter_shape;
  // Reuse dimension computation logic from conv_grad_ops.cc.
  ConvBackpropDimensions dims;
  // The filter gradients are computed by a convolution of the input
  // activations and the output gradients, with some appropriate padding.
  // See the comment at the top of conv_grad_ops.h for details.
  ConvolutionDimensionNumbers dnums;

  TF_RETURN_IF_ERROR(ConvBackpropComputeDimensions(
      type_string, attrs.num_spatial_dims, activations_shape.dimensions(),
      expanded_filter_shape.dimensions(), out_backprop_shape.dimensions(),
      attrs.dilations, attrs.strides, attrs.explicit_paddings,
      attrs.data_format, &dims));

  // Obtain some useful dimensions:
  // The last two dimensions of the filter are the input and output shapes.
  int num_dims = attrs.num_spatial_dims + 2;
  int n_dim = attrs.data_format.input_batch_dimension();
  int c_dim = attrs.data_format.input_feature_dimension();
  int64 in_depth = input_shape.dimensions(c_dim),
        filter_in_depth = filter_shape.dimensions(attrs.num_spatial_dims),
        feature_group_count = in_depth / filter_in_depth;

  // The activations (inputs) form the LHS of the convolution.
  // Activations have shape: [batch, in_rows, in_cols, ..., in_depth]
  // For the gradient computation, we need to:
  // 1. In the case of group convolution, move the num_groups dimension before
  // the batch dimension
  // 2. Swap the roles of the batch and feature dimensions.
  if (feature_group_count != 1 && !attrs.depthwise) {
    activations = TransposeInputForGroupConvolutionBackpropFilter(
        activations, input_shape, feature_group_count, n_dim, c_dim);
  }

  // In the case of depthwise convolution with no multiplier,
  // the computation can be done by the batch_group_count parameter.
  bool use_batch_group_count =
      filter_shape.dimensions(num_dims - 1) == 1 && attrs.depthwise;

  std::vector<std::pair<int64, int64>> padding(attrs.num_spatial_dims);
  std::vector<int64> rhs_dilation(attrs.num_spatial_dims);
  std::vector<int64> window_strides(attrs.num_spatial_dims);
  std::vector<int64> ones(attrs.num_spatial_dims, 1);

  // Swap n_dim and c_dim in the activations.
  dnums.set_input_batch_dimension(c_dim);
  dnums.set_input_feature_dimension(n_dim);

  // The gradients become the RHS of the convolution.
  // The gradients have shape [batch, out_rows, out_cols, ..., out_depth]
  // where the batch becomes the input feature for the convolution.
  dnums.set_kernel_input_feature_dimension(n_dim);
  dnums.set_kernel_output_feature_dimension(c_dim);

  // The dimension swap below is needed because filter shape is KH,KW,F,DM.
  if (use_batch_group_count) {
    dnums.set_output_batch_dimension(attrs.num_spatial_dims + 1);
    dnums.set_output_feature_dimension(attrs.num_spatial_dims);
  } else {
    dnums.set_output_batch_dimension(attrs.num_spatial_dims);
    dnums.set_output_feature_dimension(attrs.num_spatial_dims + 1);
  }

  // Tensorflow filter shape is [ H, W, ..., inC, outC ].
  for (int i = 0; i < attrs.num_spatial_dims; ++i) {
    dnums.add_output_spatial_dimensions(i);
  }

  for (int64 i = 0; i < attrs.num_spatial_dims; ++i) {
    int64 dim = attrs.data_format.input_spatial_dimensions(i);
    dnums.add_input_spatial_dimensions(dim);
    dnums.add_kernel_spatial_dimensions(dim);
    rhs_dilation[i] = dims.spatial_dims[i].stride;
    window_strides[i] = attrs.dilations[dim];

    // We will also need to pad the input with zeros such that after the
    // convolution, we get the right size for the filter.
    // The padded_in_rows should be such that when we convolve this with the
    // expanded_out_rows as a filter, we should get filter_rows back.

    const int64 padded_in_size =
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
    const int64 pad_total = padded_in_size - dims.spatial_dims[i].input_size;

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
    const int64 pad_before = attrs.explicit_paddings[2 * dim];
    padding[i] = {pad_before, pad_total - pad_before};
  }

  // Besides padding the input, we will also expand output_rows to
  //    expanded_out_rows = (output_rows - 1) * stride + 1
  // with zeros in between:
  //
  //      a . . . b . . . c . . . d . . . e
  //
  // This is done by specifying the window dilation factors in the
  // convolution HLO below.

  filter_backprop = ConvGeneralDilated(
      activations, out_backprop, window_strides, padding, /*lhs_dilation=*/ones,
      rhs_dilation, dnums,
      /*feature_group_count=*/feature_group_count,
      /*batch_group_count=*/use_batch_group_count ? dims.in_depth : 1,
      precision_config);

  if (!use_batch_group_count && attrs.depthwise) {
    filter_backprop = ContractFilterForDepthwiseBackprop(
        filter_shape, filter_backprop, activations.builder());
  }

  return filter_backprop;
}

}  // namespace xla
