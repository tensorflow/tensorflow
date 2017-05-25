/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

Status GetWindowedOutputSizeVerbose(int64 input_size, int64 filter_size,
                                    int64 stride, Padding padding_type,
                                    int64* output_size, int64* padding_before,
                                    int64* padding_after) {
  if (stride <= 0) {
    return errors::InvalidArgument("Stride must be > 0, but got ", stride);
  }

  // See also the parallel implementation in GetWindowedOutputSizeFromDims.
  switch (padding_type) {
    case Padding::VALID:
      *output_size = (input_size - filter_size + stride) / stride;
      *padding_before = *padding_after = 0;
      break;
    case Padding::SAME:
      *output_size = (input_size + stride - 1) / stride;
      const int64 padding_needed =
          std::max(0LL, (*output_size - 1) * stride + filter_size - input_size);
      // For odd values of total padding, add more padding at the 'right'
      // side of the given dimension.
      *padding_before = padding_needed / 2;
      *padding_after = padding_needed - *padding_before;
      break;
  }
  if (*output_size < 0) {
    return errors::InvalidArgument("computed output size would be negative");
  }
  return Status::OK();
}

Status GetWindowedOutputSize(int64 input_size, int64 filter_size, int64 stride,
                             Padding padding_type, int64* output_size,
                             int64* padding) {
  int64 padding_after_unused;
  return GetWindowedOutputSizeVerbose(input_size, filter_size, stride,
                                      padding_type, output_size, padding,
                                      &padding_after_unused);
}

Status Get3dOutputSize(const std::array<int64, 3>& input,
                       const std::array<int64, 3>& window,
                       const std::array<int64, 3>& strides,
                       Padding padding_type, std::array<int64, 3>* output_ptr,
                       std::array<int64, 3>* padding_ptr) {
  for (size_t i = 0; i < input.size(); ++i) {
    TF_RETURN_IF_ERROR(GetWindowedOutputSize(input[i], window[i], strides[i],
                                             padding_type, &(*output_ptr)[i],
                                             &(*padding_ptr)[i]));
  }
  return Status::OK();
}

namespace shape_inference {

Status GetWindowedOutputSizeFromDims(
    shape_inference::InferenceContext* c,
    shape_inference::DimensionHandle input_size,
    shape_inference::DimensionOrConstant filter_size, int64 stride,
    Padding padding_type, shape_inference::DimensionHandle* output_size) {
  if (stride <= 0) {
    return errors::InvalidArgument("Stride must be > 0, but got ", stride);
  }

  // See also the parallel implementation in GetWindowedOutputSizeVerbose.
  switch (padding_type) {
    case Padding::VALID:
      TF_RETURN_IF_ERROR(c->Subtract(input_size, filter_size, output_size));
      TF_RETURN_IF_ERROR(c->Add(*output_size, stride, output_size));
      TF_RETURN_IF_ERROR(c->Divide(*output_size, stride,
                                   false /* evenly_divisible */, output_size));
      break;
    case Padding::SAME:
      TF_RETURN_IF_ERROR(c->Add(input_size, stride - 1, output_size));
      TF_RETURN_IF_ERROR(c->Divide(*output_size, stride,
                                   false /* evenly_divisible */, output_size));
      break;
  }
  return Status::OK();
}

Status UnchangedShape(shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(0));
  return Status::OK();
}

Status MatMulShape(shape_inference::InferenceContext* c) {
  ShapeHandle a;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));

  ShapeHandle b;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b));

  bool transpose_a, transpose_b;
  TF_RETURN_IF_ERROR(c->GetAttr("transpose_a", &transpose_a));
  TF_RETURN_IF_ERROR(c->GetAttr("transpose_b", &transpose_b));
  DimensionHandle output_rows = transpose_a ? c->Dim(a, 1) : c->Dim(a, 0);
  DimensionHandle output_cols = transpose_b ? c->Dim(b, 0) : c->Dim(b, 1);

  // Validate that the inner shapes are compatible.
  DimensionHandle inner_a = transpose_a ? c->Dim(a, 0) : c->Dim(a, 1);
  DimensionHandle inner_b = transpose_b ? c->Dim(b, 1) : c->Dim(b, 0);
  DimensionHandle merged;
  TF_RETURN_IF_ERROR(c->Merge(inner_a, inner_b, &merged));

  c->set_output(0, c->Matrix(output_rows, output_cols));
  return Status::OK();
}

Status BiasAddShape(shape_inference::InferenceContext* c) {
  ShapeHandle input_shape;

  // Fetch the data_format attribute, which may not exist.
  string data_format;
  Status s = c->GetAttr("data_format", &data_format);

  if (s.ok() && data_format == "NCHW") {
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 3, &input_shape));
  } else {
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input_shape));
  }

  ShapeHandle bias_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &bias_shape));
  DimensionHandle bias_dim = c->Dim(bias_shape, 0);

  // If rank unknown, return unknown shape.
  if (!c->RankKnown(input_shape)) {
    c->set_output(0, c->UnknownShape());
    return Status::OK();
  }

  // Output has the same shape as the input, and matches the length of
  // the bias in its bias dimension.
  ShapeHandle output_shape;
  if (s.ok() && data_format == "NCHW") {
    // Merge the length of bias_shape into the third to last dimension
    ShapeHandle first;
    TF_RETURN_IF_ERROR(c->Subshape(input_shape, 0, -3, &first));

    ShapeHandle last;
    TF_RETURN_IF_ERROR(c->Subshape(input_shape, -2, &last));

    DimensionHandle input_bias_dim = c->Dim(input_shape, -3);
    DimensionHandle merged_bias_dim;
    TF_RETURN_IF_ERROR(c->Merge(input_bias_dim, bias_dim, &merged_bias_dim));
    ShapeHandle merged_bias = c->Vector(merged_bias_dim);

    ShapeHandle temp;
    TF_RETURN_IF_ERROR(c->Concatenate(first, merged_bias, &temp));
    TF_RETURN_IF_ERROR(c->Concatenate(temp, last, &output_shape));
  } else {
    ShapeHandle all_but_bias;
    TF_RETURN_IF_ERROR(c->Subshape(input_shape, 0, -1, &all_but_bias));

    DimensionHandle input_bias_dim = c->Dim(input_shape, -1);
    DimensionHandle merged_bias_dim;
    TF_RETURN_IF_ERROR(c->Merge(input_bias_dim, bias_dim, &merged_bias_dim));

    ShapeHandle merged_bias = c->Vector(merged_bias_dim);
    TF_RETURN_IF_ERROR(
        c->Concatenate(all_but_bias, merged_bias, &output_shape));
  }

  c->set_output(0, output_shape);
  return Status::OK();
}

Status BiasAddGradShape(shape_inference::InferenceContext* c) {
  ShapeHandle input_shape;
  // Fetch the data_format attribute, which may not exist.
  string data_format;
  Status s = c->GetAttr("data_format", &data_format);

  if (s.ok() && data_format == "NCHW") {
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 3, &input_shape));
    c->set_output(0, c->Vector(c->Dim(input_shape, -3)));
  } else {
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input_shape));
    c->set_output(0, c->Vector(c->Dim(input_shape, -1)));
  }

  return Status::OK();
}

Status Conv2DShape(shape_inference::InferenceContext* c) {
  ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
  ShapeHandle filter_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &filter_shape));

  string data_format;
  Status s = c->GetAttr("data_format", &data_format);

  std::vector<int32> strides;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));

  if (strides.size() != 4) {
    return errors::InvalidArgument(
        "Conv2D requires the stride attribute to contain 4 values, but got: ",
        strides.size());
  }

  int32 stride_rows, stride_cols;

  if (s.ok() && data_format == "NCHW") {
    // Convert input shape to default NHWC for inference
    auto dim = [&](char dimension) {
      return c->Dim(input_shape, GetTensorDimIndex<2>(FORMAT_NCHW, dimension));
    };
    input_shape = c->MakeShape({{dim('N'), dim('0'), dim('1'), dim('C')}});
    stride_rows = strides[2];
    stride_cols = strides[3];
  } else {
    stride_rows = strides[1];
    stride_cols = strides[2];
  }

  DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
  DimensionHandle in_rows_dim = c->Dim(input_shape, 1);
  DimensionHandle in_cols_dim = c->Dim(input_shape, 2);
  DimensionHandle filter_rows_dim = c->Dim(filter_shape, 0);
  DimensionHandle filter_cols_dim = c->Dim(filter_shape, 1);
  DimensionHandle output_depth_dim = c->Dim(filter_shape, 3);

  DimensionHandle unused;
  TF_RETURN_IF_ERROR(
      c->Merge(c->Dim(input_shape, 3), c->Dim(filter_shape, 2), &unused));

  Padding padding;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

  DimensionHandle output_rows, output_cols;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
      c, in_rows_dim, filter_rows_dim, stride_rows, padding, &output_rows));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
      c, in_cols_dim, filter_cols_dim, stride_cols, padding, &output_cols));

  ShapeHandle output_shape;
  if (data_format == "NCHW") {
    output_shape = c->MakeShape(
        {batch_size_dim, output_depth_dim, output_rows, output_cols});
  } else {
    output_shape = c->MakeShape(
        {batch_size_dim, output_rows, output_cols, output_depth_dim});
  }

  c->set_output(0, output_shape);
  return Status::OK();
}

// TODO(mjanusz): Unify all conv/pooling shape functions.
Status Conv3DShape(shape_inference::InferenceContext* c) {
  ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &input_shape));
  ShapeHandle filter_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 5, &filter_shape));

  string data_format;
  Status s = c->GetAttr("data_format", &data_format);

  std::vector<int32> strides;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
  if (strides.size() != 5) {
    return errors::InvalidArgument(
        "Conv3D requires the stride attribute to contain 5 values, but got: ",
        strides.size());
  }

  int32 stride_planes, stride_rows, stride_cols;
  if (s.ok() && data_format == "NCDHW") {
    // Convert input_shape to NDHWC.
    auto dim = [&](char dimension) {
      return c->Dim(input_shape, GetTensorDimIndex<3>(FORMAT_NCHW, dimension));
    };
    input_shape =
        c->MakeShape({{dim('N'), dim('0'), dim('1'), dim('2'), dim('C')}});
    stride_planes = strides[2];
    stride_cols = strides[3];
    stride_rows = strides[4];
  } else {
    stride_planes = strides[1];
    stride_rows = strides[2];
    stride_cols = strides[3];
  }

  DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
  DimensionHandle in_planes_dim = c->Dim(input_shape, 1);
  DimensionHandle in_rows_dim = c->Dim(input_shape, 2);
  DimensionHandle in_cols_dim = c->Dim(input_shape, 3);

  DimensionHandle filter_planes_dim = c->Dim(filter_shape, 0);
  DimensionHandle filter_rows_dim = c->Dim(filter_shape, 1);
  DimensionHandle filter_cols_dim = c->Dim(filter_shape, 2);
  DimensionHandle output_depth_dim = c->Dim(filter_shape, 4);

  DimensionHandle unused;
  TF_RETURN_IF_ERROR(
      c->Merge(c->Dim(input_shape, 4), c->Dim(filter_shape, 3), &unused));

  Padding padding;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));
  DimensionHandle output_planes, output_rows, output_cols;

  TF_RETURN_IF_ERROR(
      GetWindowedOutputSizeFromDims(c, in_planes_dim, filter_planes_dim,
                                    stride_planes, padding, &output_planes));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
      c, in_rows_dim, filter_rows_dim, stride_rows, padding, &output_rows));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
      c, in_cols_dim, filter_cols_dim, stride_cols, padding, &output_cols));

  ShapeHandle output_shape;
  if (data_format == "NCDHW") {
    output_shape = c->MakeShape({batch_size_dim, output_depth_dim,
                                 output_planes, output_rows, output_cols});
  } else {
    output_shape = c->MakeShape({batch_size_dim, output_planes, output_rows,
                                 output_cols, output_depth_dim});
  }
  c->set_output(0, output_shape);
  return Status::OK();
}

Status DepthwiseConv2DNativeShape(shape_inference::InferenceContext* c) {
  ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
  ShapeHandle filter_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &filter_shape));

  std::vector<int32> strides;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));

  if (strides.size() != 4) {
    return errors::InvalidArgument(
        "DepthwiseConv2D requires the stride attribute to contain 4 values, "
        "but got: ",
        strides.size());
  }

  string data_format;
  Status s = c->GetAttr("data_format", &data_format);
  int32 stride_rows;
  int32 stride_cols;
  if (s.ok() && data_format == "NCHW") {
    // Convert input shape to default NHWC for inference
    input_shape =
        c->MakeShape({{c->Dim(input_shape, 0), c->Dim(input_shape, 2),
                       c->Dim(input_shape, 3), c->Dim(input_shape, 1)}});
    stride_rows = strides[2];
    stride_cols = strides[3];
  } else {
    stride_rows = strides[1];
    stride_cols = strides[2];
  }

  DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
  DimensionHandle in_rows_dim = c->Dim(input_shape, 1);
  DimensionHandle in_cols_dim = c->Dim(input_shape, 2);

  DimensionHandle filter_rows_dim = c->Dim(filter_shape, 0);
  DimensionHandle filter_cols_dim = c->Dim(filter_shape, 1);
  DimensionHandle input_depth = c->Dim(filter_shape, 2);
  DimensionHandle depth_multiplier = c->Dim(filter_shape, 3);

  // Check that the input depths are compatible.
  TF_RETURN_IF_ERROR(
      c->Merge(c->Dim(input_shape, 3), input_depth, &input_depth));

  DimensionHandle output_depth;
  TF_RETURN_IF_ERROR(c->Multiply(input_depth, depth_multiplier, &output_depth));

  Padding padding;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

  // TODO(mrry,shlens): Raise an error if the stride would cause
  // information in the input to be ignored. This will require a change
  // in the kernel implementation.
  DimensionHandle output_rows, output_cols;

  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
      c, in_rows_dim, filter_rows_dim, stride_rows, padding, &output_rows));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
      c, in_cols_dim, filter_cols_dim, stride_cols, padding, &output_cols));

  ShapeHandle output_shape;
  if (data_format == "NCHW") {
    output_shape =
        c->MakeShape({batch_size_dim, output_depth, output_rows, output_cols});
  } else {
    output_shape =
        c->MakeShape({batch_size_dim, output_rows, output_cols, output_depth});
  }
  c->set_output(0, output_shape);
  return Status::OK();
}

Status AvgPoolShape(shape_inference::InferenceContext* c) {
  ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

  string data_format;
  Status s = c->GetAttr("data_format", &data_format);

  std::vector<int32> strides;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
  if (strides.size() != 4) {
    return errors::InvalidArgument(
        "AvgPool requires the stride attribute to contain 4 values, but "
        "got: ",
        strides.size());
  }

  std::vector<int32> kernel_sizes;
  TF_RETURN_IF_ERROR(c->GetAttr("ksize", &kernel_sizes));
  if (kernel_sizes.size() != 4) {
    return errors::InvalidArgument(
        "AvgPool requires the ksize attribute to contain 4 values, but got: ",
        kernel_sizes.size());
  }

  int32 stride_rows, stride_cols;
  int32 kernel_rows, kernel_cols;

  if (s.ok() && data_format == "NCHW") {
    // Convert input shape to default NHWC for inference.
    auto dim = [&](char dimension) {
      return c->Dim(input_shape, GetTensorDimIndex<2>(FORMAT_NCHW, dimension));
    };
    input_shape = c->MakeShape({{dim('N'), dim('0'), dim('1'), dim('C')}});
    stride_rows = strides[2];
    stride_cols = strides[3];
    kernel_rows = kernel_sizes[2];
    kernel_cols = kernel_sizes[3];
  } else {
    stride_rows = strides[1];
    stride_cols = strides[2];
    kernel_rows = kernel_sizes[1];
    kernel_cols = kernel_sizes[2];
  }

  DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
  DimensionHandle in_rows_dim = c->Dim(input_shape, 1);
  DimensionHandle in_cols_dim = c->Dim(input_shape, 2);
  DimensionHandle output_depth_dim = c->Dim(input_shape, 3);

  Padding padding;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

  // TODO(mrry,shlens): Raise an error if the stride would cause
  // information in the input to be ignored. This will require a change
  // in the kernel implementation.

  DimensionHandle output_rows, output_cols;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
      c, in_rows_dim, kernel_rows, stride_rows, padding, &output_rows));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
      c, in_cols_dim, kernel_cols, stride_cols, padding, &output_cols));

  ShapeHandle output_shape;
  if (data_format == "NCHW") {
    output_shape = c->MakeShape(
        {batch_size_dim, output_depth_dim, output_rows, output_cols});
  } else {
    output_shape = c->MakeShape(
        {batch_size_dim, output_rows, output_cols, output_depth_dim});
  }

  c->set_output(0, output_shape);
  return Status::OK();
}

Status MaxPoolShape(shape_inference::InferenceContext* c) {
  ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

  string data_format;
  Status s = c->GetAttr("data_format", &data_format);

  std::vector<int32> strides;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
  if (strides.size() != 4) {
    return errors::InvalidArgument(
        "MaxPool requires the stride attribute to contain 4 values, but "
        "got: ",
        strides.size());
  }

  std::vector<int32> kernel_sizes;
  TF_RETURN_IF_ERROR(c->GetAttr("ksize", &kernel_sizes));
  if (kernel_sizes.size() != 4) {
    return errors::InvalidArgument(
        "MaxPool requires the ksize attribute to contain 4 values, but got: ",
        kernel_sizes.size());
  }

  int32 stride_rows, stride_cols, stride_depth;
  int32 kernel_rows, kernel_cols, kernel_depth;

  if (s.ok() && data_format == "NCHW") {
    // Convert input shape to default NHWC for inference.
    auto dim = [&](char dimension) {
      return c->Dim(input_shape, GetTensorDimIndex<2>(FORMAT_NCHW, dimension));
    };
    input_shape = c->MakeShape({{dim('N'), dim('0'), dim('1'), dim('C')}});
    stride_depth = strides[1];
    stride_rows = strides[2];
    stride_cols = strides[3];
    kernel_depth = kernel_sizes[1];
    kernel_rows = kernel_sizes[2];
    kernel_cols = kernel_sizes[3];
  } else {
    stride_rows = strides[1];
    stride_cols = strides[2];
    stride_depth = strides[3];
    kernel_rows = kernel_sizes[1];
    kernel_cols = kernel_sizes[2];
    kernel_depth = kernel_sizes[3];
  }

  DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
  DimensionHandle in_rows_dim = c->Dim(input_shape, 1);
  DimensionHandle in_cols_dim = c->Dim(input_shape, 2);
  DimensionHandle in_depth_dim = c->Dim(input_shape, 3);

  Padding padding;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

  ShapeHandle output_shape;
  DimensionHandle output_rows, output_cols, output_depth;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
      c, in_rows_dim, kernel_rows, stride_rows, padding, &output_rows));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
      c, in_cols_dim, kernel_cols, stride_cols, padding, &output_cols));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
      c, in_depth_dim, kernel_depth, stride_depth, padding, &output_depth));

  output_shape =
      c->MakeShape({batch_size_dim, output_rows, output_cols, output_depth});
  if (data_format == "NCHW") {
    // Convert output shape back to expected NCHW data format.
    auto dim = [&](char dimension) {
      return c->Dim(output_shape, GetTensorDimIndex<2>(FORMAT_NHWC, dimension));
    };
    output_shape = c->MakeShape({{dim('N'), dim('C'), dim('0'), dim('1')}});
  }

  c->set_output(0, output_shape);
  return Status::OK();
}

Status Pool3DShape(shape_inference::InferenceContext* c) {
  ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &input_shape));

  string data_format;
  Status s = c->GetAttr("data_format", &data_format);

  std::vector<int32> strides;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
  if (strides.size() != 5) {
    return errors::InvalidArgument(
        "Pool3D ops require the stride attribute to contain 5 values, but "
        "got: ",
        strides.size());
  }

  std::vector<int32> kernel_sizes;
  TF_RETURN_IF_ERROR(c->GetAttr("ksize", &kernel_sizes));
  if (kernel_sizes.size() != 5) {
    return errors::InvalidArgument(
        "Pool3D requires the ksize attribute to contain 5 values, but got: ",
        kernel_sizes.size());
  }

  int32 stride_planes, stride_rows, stride_cols;
  int32 kernel_planes, kernel_rows, kernel_cols;

  if (s.ok() && data_format == "NCDHW") {
    // Convert input_shape to NDHWC.
    auto dim = [&](char dimension) {
      return c->Dim(input_shape, GetTensorDimIndex<3>(FORMAT_NCHW, dimension));
    };
    input_shape =
        c->MakeShape({{dim('N'), dim('0'), dim('1'), dim('2'), dim('C')}});
    stride_planes = strides[2];
    stride_rows = strides[3];
    stride_cols = strides[4];
    kernel_planes = kernel_sizes[2];
    kernel_rows = kernel_sizes[3];
    kernel_cols = kernel_sizes[4];
  } else {
    stride_planes = strides[1];
    stride_rows = strides[2];
    stride_cols = strides[3];
    kernel_planes = kernel_sizes[1];
    kernel_rows = kernel_sizes[2];
    kernel_cols = kernel_sizes[3];
  }

  DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
  DimensionHandle in_planes_dim = c->Dim(input_shape, 1);
  DimensionHandle in_rows_dim = c->Dim(input_shape, 2);
  DimensionHandle in_cols_dim = c->Dim(input_shape, 3);
  DimensionHandle output_depth_dim = c->Dim(input_shape, 4);

  Padding padding;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

  // TODO(mrry,shlens): Raise an error if the stride would cause
  // information in the input to be ignored. This will require a change
  // in the kernel implementation.
  DimensionHandle output_planes, output_rows, output_cols;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
      c, in_planes_dim, kernel_planes, stride_planes, padding, &output_planes));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
      c, in_rows_dim, kernel_rows, stride_rows, padding, &output_rows));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
      c, in_cols_dim, kernel_cols, stride_cols, padding, &output_cols));

  ShapeHandle output_shape;
  if (data_format == "NCDHW") {
    output_shape = c->MakeShape({batch_size_dim, output_depth_dim,
                                 output_planes, output_rows, output_cols});
  } else {
    output_shape = c->MakeShape({batch_size_dim, output_planes, output_rows,
                                 output_cols, output_depth_dim});
  }

  c->set_output(0, output_shape);
  return Status::OK();
}

Status UnknownShape(shape_inference::InferenceContext* c) {
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->UnknownShape());
  }
  return Status::OK();
}

Status ReductionShape(InferenceContext* c) {
  ShapeHandle input = c->input(0);

  ShapeHandle indices;
  // Older versions of TensorFlow accidentally allowed higher rank tensors like
  // [[1,2]] or [[1],[2]] to represent axis=[1,2].
  if (c->graph_def_version() < 21) {
    indices = c->input(1);
  } else {
    TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 1, &indices));
  }

  bool keep_dims;
  TF_RETURN_IF_ERROR(c->GetAttr("keep_dims", &keep_dims));

  const Tensor* reduction_indices_t = c->input_tensor(1);
  if (reduction_indices_t == nullptr || !c->RankKnown(input)) {
    // If we do not have the reduction values at runtime, or the
    // rank of the input, we don't know the output shape.

    if (keep_dims && c->RankKnown(input)) {
      // output rank matches input input if <keep_dims>.
      c->set_output(0, c->UnknownShapeOfRank(c->Rank(input)));
      return Status::OK();
    } else {
      return shape_inference::UnknownShape(c);
    }
  }

  const int32 input_rank = c->Rank(input);
  std::set<int32> true_indices;
  auto reduction_indices = reduction_indices_t->flat<int32>();
  for (int i = 0; i < reduction_indices_t->NumElements(); ++i) {
    int32 reduction_index = reduction_indices(i);
    if (reduction_index < -input_rank || reduction_index >= input_rank) {
      return errors::InvalidArgument("Invalid reduction dimension ",
                                     reduction_index, " for input with ",
                                     input_rank, " dimensions.");
    }

    int32 wrapped_index = reduction_index;
    if (wrapped_index < 0) {
      wrapped_index += input_rank;
    }

    true_indices.insert(wrapped_index);
  }

  std::vector<DimensionHandle> dims;
  for (int i = 0; i < input_rank; ++i) {
    if (true_indices.count(i) > 0) {
      if (keep_dims) {
        dims.emplace_back(c->MakeDim(1));
      }
    } else {
      dims.emplace_back(c->Dim(input, i));
    }
  }

  c->set_output(0, c->MakeShape(dims));
  return Status::OK();
}

Status ConcatShapeHelper(InferenceContext* c, int start_value_index,
                         int end_value_index, int dim_index) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(dim_index), 0, &unused));
  const Tensor* concat_dim_t = c->input_tensor(dim_index);
  if (concat_dim_t == nullptr) {
    // Return an unknown shape with same rank as inputs, or an unknown rank
    // if no input's rank is known.

    // Find rank.
    int32 rank = InferenceContext::kUnknownRank;
    for (int i = start_value_index; i < end_value_index; ++i) {
      if (rank == InferenceContext::kUnknownRank) rank = c->Rank(c->input(i));
      if (rank != InferenceContext::kUnknownRank) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), rank, &unused));
      }
    }
    if (rank == InferenceContext::kUnknownRank) {
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    }
    if (rank == 0) {
      return errors::InvalidArgument(
          "Can't concatenate scalars (use tf.stack instead)");
    }
    // Build result of <rank> different unknown dims.
    std::vector<DimensionHandle> dims;
    for (int i = 0; i < rank; ++i) dims.push_back(c->UnknownDim());
    c->set_output(0, c->MakeShape(dims));
    return Status::OK();
  }

  // Merge all the non-concat dims, and sum the concat dim to make an output
  // shape.
  const int32 concat_dim = concat_dim_t->scalar<int32>()();

  // Minimum required number of dimensions.
  const int min_rank = concat_dim < 0 ? -concat_dim : concat_dim + 1;

  ShapeHandle output_before;
  ShapeHandle output_after;

  ShapeHandle input = c->input(end_value_index - 1);
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(input, min_rank, &input));
  TF_RETURN_IF_ERROR(c->Subshape(input, 0, concat_dim, &output_before));
  DimensionHandle output_middle = c->Dim(input, concat_dim);
  if (concat_dim == -1) {
    output_after = c->Scalar();  // no dimensions.
  } else {
    TF_RETURN_IF_ERROR(c->Subshape(input, concat_dim + 1, &output_after));
  }

  for (int i = end_value_index - 2; i >= start_value_index; --i) {
    ShapeHandle before;
    ShapeHandle after;
    input = c->input(i);
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(input, min_rank, &input));
    TF_RETURN_IF_ERROR(c->Subshape(input, 0, concat_dim, &before));
    DimensionHandle middle = c->Dim(input, concat_dim);
    if (concat_dim == -1) {
      after = c->Scalar();
    } else {
      TF_RETURN_IF_ERROR(c->Subshape(input, concat_dim + 1, &after));
    }

    TF_RETURN_IF_ERROR(c->Merge(before, output_before, &output_before));
    TF_RETURN_IF_ERROR(c->Add(output_middle, middle, &output_middle));
    TF_RETURN_IF_ERROR(c->Merge(after, output_after, &output_after));
  }

  ShapeHandle s;
  TF_RETURN_IF_ERROR(
      c->Concatenate(output_before, c->Vector(output_middle), &s));
  TF_RETURN_IF_ERROR(c->Concatenate(s, output_after, &s));
  c->set_output(0, s);
  return Status::OK();
}

Status ConcatShape(InferenceContext* c, int num_inputs_to_concat) {
  return ConcatShapeHelper(c, 1 /* start_value_index */,
                           1 + num_inputs_to_concat /* end_value_index */,
                           0 /* dim_index */);
}

Status ConcatV2Shape(InferenceContext* c) {
  return ConcatShapeHelper(c, 0 /* start_value_index */,
                           c->num_inputs() - 1 /* end_value_index */,
                           c->num_inputs() - 1 /* dim_index */);
}

Status BroadcastBinaryOpShapeFn(InferenceContext* c) {
  ShapeHandle shape_x = c->input(0);
  ShapeHandle shape_y = c->input(1);
  if (!c->RankKnown(shape_x) || !c->RankKnown(shape_y)) {
    c->set_output(0, c->UnknownShape());
    return Status::OK();
  }
  const int32 rank_x = c->Rank(shape_x);
  const int32 rank_y = c->Rank(shape_y);
  const int32 rank_out = std::max(rank_x, rank_y);

  // To compute the broadcast dimensions, we zip together shape_x and shape_y
  // and
  // pad with 1 to make them the same length.
  std::vector<DimensionHandle> dims;
  DimensionHandle dim_one;
  if (rank_x != rank_y) dim_one = c->MakeDim(1);
  for (int i = 0; i < rank_out; ++i) {
    const auto dim_x = i < (rank_out - rank_x)
                           ? dim_one
                           : c->Dim(shape_x, i - (rank_out - rank_x));
    const bool dim_y_is_one = (i < (rank_out - rank_y));
    const auto dim_y =
        dim_y_is_one ? dim_one : c->Dim(shape_y, i - (rank_out - rank_y));
    if (!c->ValueKnown(dim_x) || !c->ValueKnown(dim_y)) {
      // One or both dimensions is unknown.
      //
      // - If either dimension is greater than 1, we assume that the program is
      // correct, and the other dimension will be broadcast to match it.
      // TODO(cwhipkey): For shape inference, if we eliminate the shape checks
      // in C++ op code, we must still assert that the unknown dim is either 1
      // or the same as the known dim.
      // - If either dimension is 1, the other dimension is the output.
      if (c->Value(dim_x) > 1) {
        dims.push_back(dim_x);
      } else if (c->Value(dim_y) > 1) {
        dims.push_back(dim_y);
      } else if (c->Value(dim_x) == 1) {
        dims.push_back(dim_y);
      } else if (c->Value(dim_y) == 1) {
        dims.push_back(dim_x);
      } else {
        dims.push_back(c->UnknownDim());
      }
    } else if (c->Value(dim_x) == 1 || c->Value(dim_y) == 1) {
      if (c->Value(dim_x) == 1 && !dim_y_is_one) {
        // We will broadcast dim_x to dim_y.
        dims.push_back(dim_y);
      } else {
        DCHECK_EQ(c->Value(dim_y), 1);
        // We will broadcast dim_y to dim_x.
        dims.push_back(dim_x);
      }
    } else {
      DimensionHandle dim;
      TF_RETURN_IF_ERROR(c->Merge(dim_x, dim_y, &dim));
      dims.push_back(dim);
    }
  }

  c->set_output(0, c->MakeShape(dims));
  return Status::OK();
}

Status RandomShape(shape_inference::InferenceContext* c) {
  shape_inference::ShapeHandle out;
  TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
  c->set_output(0, out);
  return Status::OK();
}

Status ValidateSparseTensor(InferenceContext* c, ShapeHandle indices_shape,
                            ShapeHandle values_shape, ShapeHandle shape_shape) {
  // Validate ranks.
  ShapeHandle unused_shape;
  TF_RETURN_IF_ERROR(c->WithRank(indices_shape, 2, &unused_shape));
  TF_RETURN_IF_ERROR(c->WithRank(values_shape, 1, &unused_shape));
  TF_RETURN_IF_ERROR(c->WithRank(shape_shape, 1, &unused_shape));

  // Number of elements in indices and values must match.
  DimensionHandle num_index_elements_dim = c->Dim(indices_shape, 0);
  if (c->ValueKnown(num_index_elements_dim)) {
    DimensionHandle num_values_elements_dim = c->Dim(values_shape, 0);
    if (c->ValueKnown(num_values_elements_dim)) {
      int64 num_index_elements = c->Value(num_index_elements_dim);
      int64 num_values_elements = c->Value(num_values_elements_dim);
      if (num_index_elements != num_values_elements) {
        return errors::InvalidArgument("Number of elements in index (",
                                       num_index_elements, ") and values (",
                                       num_values_elements, ") do not match.");
      }
    }
  }

  // Rank embedded in indices must match shape.
  DimensionHandle index_rank_dim = c->Dim(indices_shape, 1);
  if (c->ValueKnown(index_rank_dim)) {
    DimensionHandle shape_rank_dim = c->Dim(shape_shape, 0);
    if (c->ValueKnown(shape_rank_dim)) {
      int64 index_rank = c->Value(index_rank_dim);
      int32 shape_rank = c->Value(shape_rank_dim);
      if (index_rank != shape_rank) {
        return errors::InvalidArgument("Index rank (", index_rank,
                                       ") and shape rank (", shape_rank,
                                       ") do not match.");
      }
    }
  }

  return Status::OK();
}

}  // namespace shape_inference
}  // namespace tensorflow
