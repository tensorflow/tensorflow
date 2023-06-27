/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/nn_ops.cc.

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/conv_grad_shape_utils.h"

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

// Compute padding for the given spatial dimension.
int ConvBackpropDimensions::SpatialPadding(const Padding& padding,
                                           int dim) const {
  return (padding == VALID)
             ? 0
             : std::max<int>(
                   0, static_cast<int>((output_size(dim) - 1) * stride(dim) +
                                       (filter_size(dim) - 1) * dilation(dim) +
                                       1 - input_size(dim)));
}

namespace {

Status ConvBackpropExtractAndVerifyDimension(
    StringPiece label, const TensorShape& input_shape,
    const TensorShape& filter_shape, const TensorShape& output_shape,
    const gtl::ArraySlice<int32> dilations, const std::vector<int32>& strides,
    Padding padding, int64_t padding_before, int64_t padding_after,
    int spatial_dim, int filter_spatial_dim,
    ConvBackpropSpatialDimension* dim) {
  dim->input_size = input_shape.dim_size(spatial_dim);
  dim->filter_size = filter_shape.dim_size(filter_spatial_dim);
  dim->output_size = output_shape.dim_size(spatial_dim);
  dim->stride = strides[spatial_dim];
  dim->dilation = dilations[spatial_dim];
  int64_t out_size = 0;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerboseV2(
      dim->input_size, dim->filter_size, dim->dilation, dim->stride, padding,
      &out_size, &padding_before, &padding_after));
  if (dim->output_size != out_size) {
    return errors::InvalidArgument(
        label, ": Size of out_backprop doesn't match computed: ", "actual = ",
        dim->output_size, ", computed = ", out_size,
        " spatial_dim: ", spatial_dim, " input: ", dim->input_size,
        " filter: ", dim->filter_size, " output: ", dim->output_size,
        " stride: ", dim->stride, " dilation: ", dim->dilation);
  }

  int64_t effective_filter_size = (dim->filter_size - 1) * dim->dilation + 1;
  dim->expanded_output_size = (dim->output_size - 1) * dim->stride + 1;
  const auto padded_out_size = dim->input_size + effective_filter_size - 1;
  dim->pad_before = effective_filter_size - 1 - padding_before;
  dim->pad_after =
      padded_out_size - dim->expanded_output_size - dim->pad_before;
  VLOG(2) << label << ": expanded_out = " << dim->expanded_output_size
          << ", effective_filter_size = " << effective_filter_size
          << ", padded_out = " << padded_out_size
          << ", pad_before = " << dim->pad_before
          << ", pad_after = " << dim->pad_after
          << ", dilation = " << dim->dilation << ", strides = " << dim->stride;
  return OkStatus();
}

}  // namespace

Status ConvBackpropComputeDimensionsV2(
    StringPiece label, int num_spatial_dims, const TensorShape& input_shape,
    const TensorShape& filter_shape, const TensorShape& out_backprop_shape,
    const gtl::ArraySlice<int32>& dilations, const std::vector<int32>& strides,
    Padding padding, absl::Span<const int64_t> explicit_paddings,
    TensorFormat data_format, ConvBackpropDimensions* dims) {
  // The + 2 in the following line is for the batch and feature dimensions.
  const int num_dims = num_spatial_dims + 2;
  if (input_shape.dims() != num_dims) {
    return errors::InvalidArgument(label, ": input must be ", num_dims,
                                   "-dimensional");
  }
  if (filter_shape.dims() != num_dims) {
    return errors::InvalidArgument(label, ": filter must be ", num_dims,
                                   "-dimensional");
  }
  if (out_backprop_shape.dims() != num_dims) {
    return errors::InvalidArgument(label, ": out_backprop must be ", num_dims,
                                   "-dimensional");
  }
  int batch_dim = GetTensorBatchDimIndex(num_dims, data_format);
  dims->batch_size = input_shape.dim_size(batch_dim);
  if (dims->batch_size != out_backprop_shape.dim_size(batch_dim)) {
    return errors::InvalidArgument(
        label, ": input and out_backprop must have the same batch size.",
        " Input batch: ", dims->batch_size,
        ", outbackprop batch: ", out_backprop_shape.dim_size(batch_dim),
        ", batch_dim: ", batch_dim);
  }

  int feature_dim = GetTensorFeatureDimIndex(num_dims, data_format);
  dims->in_depth = input_shape.dim_size(feature_dim);
  // The input and output feature dimensions are the second last and last
  // dimensions of the filter Tensor.
  VLOG(2) << "input vs filter_in depth " << dims->in_depth << " "
          << filter_shape.dim_size(num_dims - 2);
  if (filter_shape.dim_size(num_dims - 2) <= 0) {
    return errors ::InvalidArgument(
        label, ": filter depth must be strictly greated than zero");
  }
  if (dims->in_depth % filter_shape.dim_size(num_dims - 2)) {
    return errors::InvalidArgument(
        label, ": input depth must be evenly divisible by filter depth");
  }
  dims->out_depth = filter_shape.dim_size(num_dims - 1);
  if (dims->out_depth != out_backprop_shape.dim_size(feature_dim)) {
    return errors::InvalidArgument(
        label, ": filter and out_backprop must have the same out_depth");
  }
  dims->spatial_dims.resize(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    int image_dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
    int64_t padding_before = -1, padding_after = -1;
    if (padding == EXPLICIT) {
      padding_before = explicit_paddings[2 * image_dim];
      padding_after = explicit_paddings[2 * image_dim + 1];
    }
    TF_RETURN_IF_ERROR(ConvBackpropExtractAndVerifyDimension(
        label, input_shape, filter_shape, out_backprop_shape, dilations,
        strides, padding, padding_before, padding_after, image_dim, i,
        &dims->spatial_dims[i]));
  }
  return OkStatus();
}

Status ConvBackpropComputeDimensions(StringPiece label, int num_spatial_dims,
                                     const TensorShape& input_shape,
                                     const TensorShape& filter_shape,
                                     const TensorShape& out_backprop_shape,
                                     const std::vector<int32>& strides,
                                     Padding padding, TensorFormat data_format,
                                     ConvBackpropDimensions* dims) {
  static constexpr std::array<int32, 5> one_dilations = {{1, 1, 1, 1, 1}};
  return ConvBackpropComputeDimensionsV2(
      label, num_spatial_dims, input_shape, filter_shape, out_backprop_shape,
      one_dilations, strides, padding, /*explicit_paddings=*/{}, data_format,
      dims);
}

Status Conv2DBackpropComputeInputShape(const Tensor& input_sizes,
                                       const TensorShape& filter_shape,
                                       const TensorShape& out_backprop_shape,
                                       const TensorFormat& data_format,
                                       TensorShape* input_shape) {
  if (!TensorShapeUtils::IsVector(input_sizes.shape())) {
    return errors::InvalidArgument(
        "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
        input_sizes.dims());
  }

  if (input_sizes.dim_size(0) == 4) {
    return TensorShapeUtils::MakeShape(input_sizes.vec<int32>(), input_shape);
  }

  if (input_sizes.dim_size(0) == 2) {
    const int batch_size = GetTensorDim(out_backprop_shape, data_format, 'N');
    const int output_height = input_sizes.vec<int32>()(0);
    const int output_width = input_sizes.vec<int32>()(1);
    const int output_depth = filter_shape.dim_size(2);
    if (output_height < 0 || output_width < 0) {
      return errors::InvalidArgument(
          "Conv2DBackpropInput: elements of input_sizes must be >= 0, not ",
          output_height, "x", output_width);
    }
    return ShapeFromFormatWithStatus(data_format, batch_size, output_height,
                                     output_width, output_depth, input_shape);
  }

  return errors::InvalidArgument(
      "Conv2DBackpropInput requires input_sizes to "
      "contain 4 values or 2 values, but got: ",
      input_sizes.dim_size(0));
}

}  // namespace tensorflow
