/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/kernels/conv_op_attrs.h"

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {
namespace {

// Converts the tensor data format to the one required by the XLA convolution
// library.
xla::ConvolutionDimensionNumbers MakeConvolutionDimensionNumbers(
    TensorFormat data_format, int num_spatial_dims) {
  int num_dims = num_spatial_dims + 2;
  int batch_dimension = GetTensorBatchDimIndex(num_dims, data_format);
  int feature_dimension = GetTensorFeatureDimIndex(num_dims, data_format);
  xla::ConvolutionDimensionNumbers conv_dim_numbers;
  for (int spatial_dim = 0; spatial_dim < num_spatial_dims; ++spatial_dim) {
    conv_dim_numbers.add_input_spatial_dimensions(
        GetTensorSpatialDimIndex(num_dims, data_format, spatial_dim));
  }
  conv_dim_numbers.set_input_batch_dimension(batch_dimension);
  conv_dim_numbers.set_input_feature_dimension(feature_dimension);
  return conv_dim_numbers;
}

}  // namespace

xla::StatusOr<ConvOpAttrs> ConvOpAttrs::Create(
    int num_spatial_dims, bool depthwise,
    tensorflow::OpKernelConstruction* ctx) {
  ConvOpAttrs attrs;
  attrs.num_spatial_dims = num_spatial_dims;
  attrs.depthwise = depthwise;
  TF_RETURN_IF_ERROR(ctx->GetAttr("dilations", &attrs.dilations));
  TF_RETURN_IF_ERROR(ctx->GetAttr("strides", &attrs.strides));
  TF_RETURN_IF_ERROR(ctx->GetAttr("padding", &attrs.padding));
  if (attrs.padding == tensorflow::EXPLICIT) {
    TF_RETURN_IF_ERROR(
        ctx->GetAttr("explicit_paddings", &attrs.explicit_paddings));
  }

  string data_format;
  TF_RETURN_IF_ERROR(ctx->GetAttr("data_format", &data_format));
  if (!FormatFromString(data_format, &attrs.data_format)) {
    return errors::InvalidArgument("Invalid data format: ", data_format);
  }

  return attrs;
}

xla::StatusOr<xla::ConvOpAttrs> ConvOpAttrs::ToXla(
    const TensorShape& input_shape, const TensorShape& filter_shape) const {
  xla::ConvOpAttrs xla_attrs;
  xla_attrs.depthwise = depthwise;
  xla_attrs.num_spatial_dims = num_spatial_dims;
  xla_attrs.dilations = dilations;
  xla_attrs.strides = strides;
  xla_attrs.data_format =
      MakeConvolutionDimensionNumbers(data_format, num_spatial_dims);
  if (padding == Padding::EXPLICIT) {
    xla_attrs.explicit_paddings = explicit_paddings;
    return xla_attrs;
  }
  int num_dims = num_spatial_dims + 2;
  xla_attrs.explicit_paddings.resize(2 * num_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    const int64 dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
    int64 unused_output_size;
    TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerboseV2(
        input_shape.dim_size(dim), filter_shape.dim_size(i), dilations.at(dim),
        strides.at(dim), padding, &unused_output_size,
        &xla_attrs.explicit_paddings[dim * 2],
        &xla_attrs.explicit_paddings[dim * 2 + 1]));
  }
  return xla_attrs;
}

}  // namespace tensorflow
