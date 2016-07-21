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

namespace tensorflow {

Status GetWindowedOutputSizeVerbose(int64 input_size, int64 filter_size,
                                    int64 stride, Padding padding_type,
                                    int64* output_size, int64* padding_before,
                                    int64* padding_after) {
  if (stride <= 0) {
    return errors::InvalidArgument("Stride must be > 0, but got ", stride);
  }

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

Status UnchangedShape(shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(0));
  return Status::OK();
}

Status MatMulShape(shape_inference::InferenceContext* c) {
  const Shape* a;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));

  const Shape* b;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b));

  bool transpose_a, transpose_b;
  TF_RETURN_IF_ERROR(c->GetAttr("transpose_a", &transpose_a));
  TF_RETURN_IF_ERROR(c->GetAttr("transpose_b", &transpose_b));
  const Dimension* output_rows = transpose_a ? c->Dim(a, 1) : c->Dim(a, 0);
  const Dimension* output_cols = transpose_b ? c->Dim(b, 0) : c->Dim(b, 1);

  // Validate that the inner shapes are compatible.
  const Dimension* inner_a = transpose_a ? c->Dim(a, 0) : c->Dim(a, 1);
  const Dimension* inner_b = transpose_b ? c->Dim(b, 1) : c->Dim(b, 0);
  const Dimension* merged;
  TF_RETURN_IF_ERROR(c->Merge(inner_a, inner_b, &merged));

  c->set_output(0, c->Matrix(output_rows, output_cols));
  return Status::OK();
}

Status BiasAddShape(shape_inference::InferenceContext* c) {
  const Shape* input_shape;

  // Fetch the data_format attribute, which may not exist.
  string data_format;
  Status s = c->GetAttr("data_format", &data_format);

  if (s.ok() && data_format == "NCHW") {
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 4, &input_shape));
  } else {
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input_shape));
  }

  const Shape* bias_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &bias_shape));
  const Dimension* bias_dim = c->Dim(bias_shape, 0);

  // If rank unknown, return unknown shape.
  if (!c->RankKnown(input_shape)) {
    c->set_output(0, c->UnknownShape());
    return Status::OK();
  }

  // Output has the same shape as the input, and matches the length of
  // the bias in its bias dimension.
  const Shape* output_shape;
  if (s.ok() && data_format == "NCHW") {
    // Merge the length of bias_shape into the third to last dimension
    const Shape* first;
    TF_RETURN_IF_ERROR(c->Subshape(input_shape, 0, -3, &first));

    const Shape* last;
    TF_RETURN_IF_ERROR(c->Subshape(input_shape, -2, &last));

    const Dimension* input_bias_dim = c->Dim(input_shape, -3);
    const Dimension* merged_bias_dim;
    TF_RETURN_IF_ERROR(c->Merge(input_bias_dim, bias_dim, &merged_bias_dim));
    const Shape* merged_bias = c->Vector(merged_bias_dim);

    const Shape* temp;
    TF_RETURN_IF_ERROR(c->Concatenate(first, merged_bias, &temp));
    TF_RETURN_IF_ERROR(c->Concatenate(temp, last, &output_shape));
  } else {
    const Shape* all_but_bias;
    TF_RETURN_IF_ERROR(c->Subshape(input_shape, 0, -1, &all_but_bias));

    const Dimension* input_bias_dim = c->Dim(input_shape, -1);
    const Dimension* merged_bias_dim;
    TF_RETURN_IF_ERROR(c->Merge(input_bias_dim, bias_dim, &merged_bias_dim));

    const Shape* merged_bias = c->Vector(merged_bias_dim);
    TF_RETURN_IF_ERROR(
        c->Concatenate(all_but_bias, merged_bias, &output_shape));
  }

  c->set_output(0, output_shape);
  return Status::OK();
}

Status BiasAddGradShape(shape_inference::InferenceContext* c) {
  const Shape* input_shape;
  // Fetch the data_format attribute, which may not exist.
  string data_format;
  Status s = c->GetAttr("data_format", &data_format);

  if (s.ok() && data_format == "NCHW") {
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 4, &input_shape));
    c->set_output(0, c->Vector(c->Dim(input_shape, -3)));
  } else {
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input_shape));
    c->set_output(0, c->Vector(c->Dim(input_shape, -1)));
  }

  return Status::OK();
}

Status Conv2DShape(shape_inference::InferenceContext* c) {
  const Shape* input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
  const Shape* filter_shape;
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
    input_shape =
        c->MakeShape({{c->Dim(input_shape, 0), c->Dim(input_shape, 2),
                       c->Dim(input_shape, 3), c->Dim(input_shape, 1)}});
    stride_rows = strides[2];
    stride_cols = strides[3];
  } else {
    stride_rows = strides[1];
    stride_cols = strides[2];
  }

  const Dimension* batch_size_dim = c->Dim(input_shape, 0);
  const Dimension* in_rows_dim = c->Dim(input_shape, 1);
  const Dimension* in_cols_dim = c->Dim(input_shape, 2);
  const Dimension* filter_rows_dim = c->Dim(filter_shape, 0);
  const Dimension* filter_cols_dim = c->Dim(filter_shape, 1);
  const Dimension* output_depth_dim = c->Dim(filter_shape, 3);

  // At the moment we need to know the values of several fields.
  auto CheckKnownDim = [&c](const Dimension* dim, const char* name) {
    if (!c->ValueKnown(dim)) {
      return errors::InvalidArgument("Cannot infer shape because dimension ",
                                     name, " is not known.");
    }
    return Status::OK();
  };
  TF_RETURN_IF_ERROR(CheckKnownDim(in_rows_dim, "in_rows"));
  TF_RETURN_IF_ERROR(CheckKnownDim(in_cols_dim, "in_cols"));
  TF_RETURN_IF_ERROR(CheckKnownDim(filter_rows_dim, "filter_rows"));
  TF_RETURN_IF_ERROR(CheckKnownDim(filter_cols_dim, "filter_cols"));

  auto in_rows = c->Value(in_rows_dim);
  auto in_cols = c->Value(in_cols_dim);
  auto filter_rows = c->Value(filter_rows_dim);
  auto filter_cols = c->Value(filter_cols_dim);

  const Dimension* unused;
  TF_RETURN_IF_ERROR(
      c->Merge(c->Dim(input_shape, 3), c->Dim(filter_shape, 2), &unused));

  Padding padding;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

  int64 output_rows, output_cols;
  int64 padding_before, padding_after;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
      in_rows, filter_rows, stride_rows, padding, &output_rows, &padding_before,
      &padding_after));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
      in_cols, filter_cols, stride_cols, padding, &output_cols, &padding_before,
      &padding_after));

  const Shape* output_shape;
  if (data_format == "NCHW") {
    output_shape =
        c->MakeShape({{batch_size_dim, output_depth_dim,
                       c->MakeDim(output_rows), c->MakeDim(output_cols)}});
  } else {
    output_shape = c->MakeShape({{batch_size_dim, c->MakeDim(output_rows),
                                  c->MakeDim(output_cols), output_depth_dim}});
  }

  c->set_output(0, output_shape);
  return Status::OK();
}

}  // namespace shape_inference
}  // namespace tensorflow
