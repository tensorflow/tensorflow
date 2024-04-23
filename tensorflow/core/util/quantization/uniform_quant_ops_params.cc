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
#include "tensorflow/core/util/quantization/uniform_quant_ops_params.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"

namespace tensorflow {
namespace {

using tensorflow::errors::InvalidArgument;

Status ValidDim(int64_t dims, int64_t dim) {
  if (dim < 0 || dim >= dims) {
    return InvalidArgument(
        "Each dimension number must be in region [0, rank). Given rank ", dims,
        " and dimension number value ", dim);
  }
  return absl::OkStatus();
}

Status ValidSpatialDimensions(
    int64_t dims, const protobuf::RepeatedField<int64_t>& spatial_dimensions) {
  if (spatial_dimensions.size() != dims - 2) {
    return InvalidArgument(
        "Spatial dimensions size must be rank - 2. Given rank ", dims,
        " and spatial dimensions size ", spatial_dimensions.size());
  }
  for (int i = 0; i < spatial_dimensions.size(); ++i) {
    TF_RETURN_IF_ERROR(ValidDim(dims, spatial_dimensions.Get(i)));
  }
  return absl::OkStatus();
}

}  // namespace

Status UniformQuantizedConvolutionParams::LoadFromAttrs(
    const OpKernelConstruction& context) {
  return LoadFromAttrsInternal(context);
}

Status UniformQuantizedConvolutionParams::LoadFromAttrs(
    const shape_inference::InferenceContext& context) {
  return LoadFromAttrsInternal(context);
}

Status UniformQuantizedConvolutionParams::ValidateOrFillParamsAndValidateShape(
    const TensorShape& lhs_shape, const TensorShape& rhs_shape) {
  if (lhs_shape.dims() != rhs_shape.dims()) {
    return InvalidArgument(
        "lhs and rhs must have same dims. Given lhs and rhs of shapes: ",
        lhs_shape.DebugString(), rhs_shape.DebugString());
  }
  const int64_t dims = lhs_shape.dims();
  if (dims <= 2) {
    return InvalidArgument("lhs and rhs shape dims must be at least 3. Given: ",
                           dims);
  }

  const int64_t num_spatial_dims = dims - 2;

  if (window_strides_.empty()) {
    window_strides_.resize(num_spatial_dims, 1);
  } else if (window_strides_.size() != num_spatial_dims) {
    return InvalidArgument("Size of window_strides Attr must be dims - 2.");
  } else if (!absl::c_all_of(window_strides_,
                             [](int stride) { return stride >= 1; })) {
    return InvalidArgument(
        "All elements of window_strides must be >= 1. Given ",
        absl::StrJoin(window_strides_, ", "));
  }

  if (lhs_dilation_.empty()) {
    lhs_dilation_.resize(num_spatial_dims, 1);
  } else if (lhs_dilation_.size() != num_spatial_dims) {
    return InvalidArgument("Size of lhs_dilation Attr must be dims - 2.");
  } else if (!absl::c_all_of(lhs_dilation_, [](const int dilation) {
               return dilation >= 1;
             })) {
    return InvalidArgument("All elements of lhs_dilation must be >= 1. Given ",
                           absl::StrJoin(lhs_dilation_, ", "));
  }

  if (rhs_dilation_.empty()) {
    rhs_dilation_.resize(num_spatial_dims, 1);
  } else if (rhs_dilation_.size() != num_spatial_dims) {
    return InvalidArgument("Size of rhs_dilation Attr must be dims - 2.");
  } else if (!absl::c_all_of(rhs_dilation_, [](const int dilation) {
               return dilation >= 1;
             })) {
    return InvalidArgument("All elements of rhs_dilation must be >= 1. Given ",
                           absl::StrJoin(rhs_dilation_, ", "));
  }

  if (dimension_numbers_.input_spatial_dimensions_size() == 0) {
    // dimension_numbers Attr string was empty.
    dimension_numbers_.set_input_batch_dimension(0);
    dimension_numbers_.set_input_feature_dimension(1);
    for (int64_t i = 0; i < num_spatial_dims; ++i) {
      dimension_numbers_.add_input_spatial_dimensions(2 + i);
    }

    dimension_numbers_.set_kernel_output_feature_dimension(0);
    dimension_numbers_.set_kernel_input_feature_dimension(1);
    for (int64_t i = 0; i < num_spatial_dims; ++i) {
      dimension_numbers_.add_kernel_spatial_dimensions(2 + i);
    }

    dimension_numbers_.set_output_batch_dimension(0);
    dimension_numbers_.set_output_feature_dimension(1);
    for (int64_t i = 0; i < num_spatial_dims; ++i) {
      dimension_numbers_.add_output_spatial_dimensions(2 + i);
    }
  } else {
    TF_RETURN_IF_ERROR(
        ValidDim(dims, dimension_numbers_.input_batch_dimension()));
    TF_RETURN_IF_ERROR(
        ValidDim(dims, dimension_numbers_.input_feature_dimension()));
    TF_RETURN_IF_ERROR(ValidSpatialDimensions(
        dims, dimension_numbers_.input_spatial_dimensions()));

    TF_RETURN_IF_ERROR(
        ValidDim(dims, dimension_numbers_.kernel_input_feature_dimension()));
    TF_RETURN_IF_ERROR(
        ValidDim(dims, dimension_numbers_.kernel_output_feature_dimension()));
    TF_RETURN_IF_ERROR(ValidSpatialDimensions(
        dims, dimension_numbers_.kernel_spatial_dimensions()));

    TF_RETURN_IF_ERROR(
        ValidDim(dims, dimension_numbers_.output_batch_dimension()));
    TF_RETURN_IF_ERROR(
        ValidDim(dims, dimension_numbers_.output_batch_dimension()));
    TF_RETURN_IF_ERROR(ValidSpatialDimensions(
        dims, dimension_numbers_.output_spatial_dimensions()));
  }

  // Validate lhs_shape, rhs_shape, feature_group_count, and batch_group_count.
  if (feature_group_count_ <= 0) {
    return InvalidArgument(
        "feature_group_count must be a positive integer, given: ",
        feature_group_count_);
  }
  const int64_t lhs_feature_count =
      lhs_shape.dim_size(dimension_numbers_.input_feature_dimension());
  if (lhs_feature_count % feature_group_count_) {
    return InvalidArgument(
        "feature_group_count must divide lhs feature dimension size, but ",
        feature_group_count_, " does not divide ", lhs_feature_count);
  }
  const int64_t rhs_input_feature_count =
      rhs_shape.dim_size(dimension_numbers_.kernel_input_feature_dimension());
  if (lhs_feature_count % rhs_input_feature_count) {
    return InvalidArgument(
        "rhs input feature dimension must divide lhs feature dimension "
        "size, but ",
        rhs_input_feature_count, " does not divide ", lhs_feature_count);
  }
  if (lhs_feature_count / feature_group_count_ != rhs_input_feature_count) {
    return InvalidArgument(
        "lhs feature dimension size divided by feature_group_count must equal "
        "the rhs input feature dimension size, but ",
        lhs_feature_count, " / ", feature_group_count_,
        " != ", rhs_input_feature_count);
  }
  const int64_t rhs_output_feature_count =
      rhs_shape.dim_size(dimension_numbers_.kernel_output_feature_dimension());
  if (rhs_output_feature_count % feature_group_count_) {
    return InvalidArgument(
        "rhs output dimension size must be a multiple of feature_group_count, "
        "but ",
        rhs_output_feature_count, " is not a multiple of ",
        feature_group_count_);
  }

  if (batch_group_count_ <= 0) {
    return InvalidArgument(
        "batch_group_count Attr must be a positive integer. Given: ",
        batch_group_count_);
  }
  const int64_t lhs_batch_count =
      lhs_shape.dim_size(dimension_numbers_.input_batch_dimension());
  if (lhs_batch_count % batch_group_count_) {
    return InvalidArgument(
        "batch_group_count must divide lhs batch dimension size, but ",
        batch_group_count_, " does not divide ", lhs_batch_count);
  }
  if (rhs_output_feature_count % batch_group_count_) {
    return InvalidArgument(
        "rhs output dimension size must be a multiple of batch_group_count, "
        "but ",
        rhs_output_feature_count, " is not a multiple of ", batch_group_count_);
  }

  return ValidateOrFillPaddingList(lhs_shape, rhs_shape);
}

absl::StatusOr<TensorShape>
UniformQuantizedConvolutionParams::CalculateOutputShape(
    const TensorShape& lhs_shape, const TensorShape& rhs_shape) const {
  // Given that lhs_shape, rhs_shape and Op Attrs (feature_group_count,
  // batch_group_count) are valid, calculate output shape.
  std::vector<int64_t> output_shape_buf(lhs_shape.dims());

  output_shape_buf[dimension_numbers_.output_batch_dimension()] =
      lhs_shape.dim_size(dimension_numbers_.input_batch_dimension()) /
      batch_group_count_;
  output_shape_buf[dimension_numbers_.output_feature_dimension()] =
      rhs_shape.dim_size(dimension_numbers_.kernel_output_feature_dimension());

  for (int i = 0; i < dimension_numbers_.input_spatial_dimensions_size(); ++i) {
    const int64_t lhs_size_dilated = DilatedSize(
        lhs_shape.dim_size(dimension_numbers_.input_spatial_dimensions(i)),
        lhs_dilation_[i]);
    const int64_t rhs_size_dilated = DilatedSize(
        rhs_shape.dim_size(dimension_numbers_.kernel_spatial_dimensions(i)),
        rhs_dilation_[i]);

    const int64_t output_size_numerator =
        lhs_size_dilated + padding_list_[2 * i] + padding_list_[2 * i + 1] -
        rhs_size_dilated + 1;
    const int64_t output_size_denominator = window_strides_[i];
    // output_size = ceil(output_size_numerator / output_size_denominator).
    output_shape_buf[dimension_numbers_.output_spatial_dimensions(i)] =
        (output_size_numerator + output_size_denominator - 1) /
        output_size_denominator;
  }

  TensorShape output_shape;
  TF_RETURN_IF_ERROR(
      TensorShape::BuildTensorShape(output_shape_buf, &output_shape));
  return output_shape;
}

template <typename ContextT>
Status UniformQuantizedConvolutionParams::LoadFromAttrsInternal(
    const ContextT& context) {
  TF_RETURN_IF_ERROR(context.GetAttr("window_strides", &window_strides_));
  TF_RETURN_IF_ERROR(context.GetAttr("lhs_dilation", &lhs_dilation_));
  TF_RETURN_IF_ERROR(context.GetAttr("rhs_dilation", &rhs_dilation_));
  TF_RETURN_IF_ERROR(context.GetAttr("batch_group_count", &batch_group_count_));
  TF_RETURN_IF_ERROR(
      context.GetAttr("feature_group_count", &feature_group_count_));

  TF_RETURN_IF_ERROR(context.GetAttr("padding", &padding_));
  TF_RETURN_IF_ERROR(context.GetAttr("explicit_padding", &padding_list_));
  if (padding_ != "EXPLICIT" && padding_ != "SAME" && padding_ != "VALID") {
    return InvalidArgument(
        "padding Attr must be one of [EXPLICIT | SAME | VALID], but given: ",
        padding_);
  } else if (padding_ != "EXPLICIT" && !padding_list_.empty()) {
    return InvalidArgument(
        "If padding Attr is not 'EXPLICIT', explicit_padding Attr must be "
        "empty. Given padding ",
        padding_, " and explicit_padding of size ", padding_list_.size());
  }

  std::string dimension_numbers_str;
  TF_RETURN_IF_ERROR(
      context.GetAttr("dimension_numbers", &dimension_numbers_str));
  if (dimension_numbers_str.empty()) {
    dimension_numbers_.Clear();
  } else if (!dimension_numbers_.ParseFromString(dimension_numbers_str)) {
    return InvalidArgument("Error parsing convolution dimension numbers.");
  }
  return absl::OkStatus();
}

Status UniformQuantizedConvolutionParams::ValidateOrFillPaddingList(
    const TensorShape& lhs_shape, const TensorShape& rhs_shape) {
  const int64_t dims = lhs_shape.dims();
  const int64_t padding_list_size = 2 * (dims - 2);

  if (padding_ == "EXPLICIT") {
    if (padding_list_.size() != padding_list_size) {
      return InvalidArgument(
          "Size of explicit_padding Attr must be 2 * (rank - 2). Given rank ",
          dims, " and explicit_padding of size ", padding_list_.size());
    } else if (!absl::c_all_of(padding_list_,
                               [](int elem) { return elem >= 0; })) {
      return InvalidArgument("All explicit_padding elems must be >= 0, Given ",
                             absl::StrJoin(padding_list_, ", "));
    }
  } else if (padding_ == "VALID") {
    padding_list_.resize(padding_list_size, 0);
  } else {
    padding_list_.resize(padding_list_size);
    for (int i = 0; i < dimension_numbers_.input_spatial_dimensions_size();
         ++i) {
      const int64_t stride = window_strides_[i];
      const int64_t lhs_size_dilated = DilatedSize(
          lhs_shape.dim_size(dimension_numbers_.input_spatial_dimensions(i)),
          lhs_dilation_[i]);
      const int64_t rhs_size_dilated = DilatedSize(
          rhs_shape.dim_size(dimension_numbers_.kernel_spatial_dimensions(i)),
          rhs_dilation_[i]);

      const int64_t output_size = (lhs_size_dilated + stride - 1) / stride;

      const int64_t total_padding = std::max(
          (output_size - 1) * stride + rhs_size_dilated - lhs_size_dilated,
          static_cast<int64_t>(0));
      const int64_t padding_begin = total_padding / 2;
      const int64_t padding_end = total_padding - padding_begin;
      padding_list_[2 * i] = padding_begin;
      padding_list_[2 * i + 1] = padding_end;
    }
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
