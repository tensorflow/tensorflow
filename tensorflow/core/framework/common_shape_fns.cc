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

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/einsum_op_util.h"

namespace tensorflow {

namespace shape_inference {

// The V2 version computes windowed output size with arbitrary dilation_rate,
// while the original version only handles the cases where dilation_rates equal
// to 1.
Status GetWindowedOutputSizeFromDimsV2(
    shape_inference::InferenceContext* c,
    shape_inference::DimensionHandle input_size,
    shape_inference::DimensionOrConstant filter_size, int64 dilation_rate,
    int64 stride, Padding padding_type, int64 padding_before,
    int64 padding_after, shape_inference::DimensionHandle* output_size) {
  if (stride <= 0) {
    return errors::InvalidArgument("Stride must be > 0, but got ", stride);
  }

  if (dilation_rate < 1) {
    return errors::InvalidArgument("Dilation rate must be >= 1, but got ",
                                   dilation_rate);
  }

  // See also the parallel implementation in GetWindowedOutputSizeVerbose.
  switch (padding_type) {
    case Padding::VALID:
      padding_before = padding_after = 0;
      TF_FALLTHROUGH_INTENDED;
    case Padding::EXPLICIT:
      TF_RETURN_IF_ERROR(
          c->Add(input_size, padding_before + padding_after, &input_size));
      if (dilation_rate > 1) {
        DimensionHandle window_size;
        TF_RETURN_IF_ERROR(
            c->Subtract(c->MakeDim(filter_size), 1, &window_size));
        TF_RETURN_IF_ERROR(
            c->Multiply(window_size, dilation_rate, &window_size));
        TF_RETURN_IF_ERROR(c->Add(window_size, 1, &window_size));
        TF_RETURN_IF_ERROR(c->Subtract(input_size, window_size, output_size));
      } else {
        TF_RETURN_IF_ERROR(c->Subtract(input_size, filter_size, output_size));
      }
      TF_RETURN_IF_ERROR(c->Add(*output_size, stride, output_size));
      TF_RETURN_IF_ERROR(c->Divide(*output_size, stride,
                                   /*evenly_divisible=*/false, output_size));
      break;
    case Padding::SAME:
      TF_RETURN_IF_ERROR(c->Add(input_size, stride - 1, output_size));
      TF_RETURN_IF_ERROR(c->Divide(*output_size, stride,
                                   /*evenly_divisible=*/false, output_size));
      break;
  }
  return Status::OK();
}

Status GetWindowedOutputSizeFromDims(
    shape_inference::InferenceContext* c,
    shape_inference::DimensionHandle input_size,
    shape_inference::DimensionOrConstant filter_size, int64 stride,
    Padding padding_type, shape_inference::DimensionHandle* output_size) {
  if (padding_type == Padding::EXPLICIT) {
    return errors::Internal(
        "GetWindowedOutputSizeFromDims does not handle EXPLICIT padding; call "
        "GetWindowedOutputSizeFromDimsV2 instead");
  }
  return GetWindowedOutputSizeFromDimsV2(c, input_size, filter_size,
                                         /*dilation_rate=*/1, stride,
                                         padding_type,
                                         // Give dummy values of -1 to
                                         // padding_before and padding_after,
                                         // since explicit padding is not used.
                                         -1, -1, output_size);
}

Status UnchangedShape(shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(0));
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data != nullptr) {
    c->set_output_handle_shapes_and_types(0, *handle_data);
  }
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

namespace {

// Validate that an Einsum subscript contains exactly one or zero ellipsis; and
// that periods (.) occur only within an ellipses (...).
Status ValidateEinsumEllipsis(absl::string_view subscript,
                              bool* found_ellipsis) {
  const int num_periods = absl::c_count(subscript, '.');
  if (num_periods != 0 && num_periods != 3) {
    return errors::InvalidArgument(
        "Expected at most one ellipsis (...), but found ", num_periods,
        " periods (.) in the input subscript: ", subscript);
  }
  if (num_periods == 3 && !absl::StrContains(subscript, "...")) {
    return errors::InvalidArgument(
        "Periods found outside of ellipsis in subscript: ", subscript);
  }
  *found_ellipsis = num_periods > 0;
  return Status::OK();
}

}  // namespace

Status EinsumShape(shape_inference::InferenceContext* c) {
  // We assume that the equation has a valid format. Either (x),(y)->(z)
  // or (x)->(z), where each of (x), (y) and (z) are concatenation of zero or
  // more latin alphabets and contains at most one ellipsis ('...').
  string equation;
  TF_RETURN_IF_ERROR(c->GetAttr("equation", &equation));
  gtl::InlinedVector<string, 2> input_labels;
  string output_labels;
  TF_RETURN_IF_ERROR(
      ParseEinsumEquation(equation, &input_labels, &output_labels));

  if (c->num_inputs() == 0 || c->num_inputs() > 2) {
    return errors::InvalidArgument("Expected either 1 or 2 inputs but got: ",
                                   c->num_inputs());
  }
  if (c->num_inputs() != input_labels.size()) {
    return errors::InvalidArgument("Expected ", input_labels.size(),
                                   " inputs for equation ", equation,
                                   " but got: ", c->num_inputs());
  }

  // Validate input subscripts, build the label to dimension mapping and obtain
  // the broadcast shapes that map to ellipsis.
  absl::flat_hash_map<char, DimensionHandle> label_to_dimension;
  gtl::InlinedVector<ShapeHandle, 2> input_bcast_shapes(c->num_inputs());
  for (int i = 0; i < c->num_inputs(); ++i) {
    bool has_ellipsis = false;
    TF_RETURN_IF_ERROR(ValidateEinsumEllipsis(input_labels[i], &has_ellipsis));
    ShapeHandle input_shape = c->input(i);
    // Validate that the input rank is sufficient for the given number of named
    // labels.
    if (c->RankKnown(input_shape)) {
      if (has_ellipsis) {
        const int num_named_labels =
            static_cast<int>(input_labels[i].size()) - 3;
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            c->WithRankAtLeast(input_shape, num_named_labels, &input_shape),
            " for ", i, "th input and equation: ", equation);
      } else {
        const int num_named_labels = static_cast<int>(input_labels[i].size());
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            c->WithRank(input_shape, num_named_labels, &input_shape), " for ",
            i, "th input and equation: ", equation);
      }
    }

    bool seen_ellipsis = false;
    input_bcast_shapes[i] = c->Scalar();
    // Run through the input labels; populate label_to_dimension mapping and
    // compute the broadcast shapes corresponding to the ellipsis (if present).
    for (int label_idx = 0; label_idx < input_labels[i].size(); ++label_idx) {
      const char label = input_labels[i][label_idx];
      // Calculate the input axis that the current label is referring to. After
      // the ellipsis, the axis may be found by using negative indices; i.e the
      // (rank - k)th dimension corresponds to the (num_labels - k)th label.
      const int64 axis_before_ellipsis = label_idx;
      const int64 axis_after_ellipsis =
          c->RankKnown(input_shape)
              ? label_idx + c->Rank(input_shape) - input_labels[i].size()
              : -1;

      // Populate the input broadcast shape when we encounter an ellipsis (...).
      if (label == '.') {
        if (!c->RankKnown(input_shape)) {
          input_bcast_shapes[i] = c->UnknownShape();
        } else {
          // The broadcast shape runs till the named label right after the
          // ellipsis, the label with index (label_idx + 3).
          TF_RETURN_IF_ERROR(c->Subshape(input_shape, axis_before_ellipsis,
                                         axis_after_ellipsis + 3,
                                         &input_bcast_shapes[i]));
        }
        label_idx += 2;  // Skip the rest of the ellipsis.
        seen_ellipsis = true;
        continue;
      }
      // Obtain the dimension that the current label corresponds to.
      int64 axis = seen_ellipsis ? axis_after_ellipsis : axis_before_ellipsis;
      DimensionHandle new_dim = c->RankKnown(input_shape)
                                    ? c->Dim(input_shape, axis)
                                    : c->UnknownDim();
      // If we've seen this label before, make sure previous and current
      // dimensions are compatible.
      if (label_to_dimension.contains(label)) {
        DimensionHandle merged;
        TF_RETURN_IF_ERROR(
            c->Merge(label_to_dimension[label], new_dim, &merged));
        label_to_dimension[label] = merged;
      } else {
        label_to_dimension[label] = new_dim;
      }
    }
  }

  // For two inputs, broadcast the two input broadcast shapes to create the
  // output broadcast shape. For one input, just copy the single broadcast
  // shape.
  ShapeHandle output_bcast_shape;
  if (input_bcast_shapes.size() == 1) {
    output_bcast_shape = input_bcast_shapes[0];
  } else if (input_bcast_shapes.size() == 2) {
    TF_RETURN_IF_ERROR(BroadcastBinaryOpOutputShapeFnHelper(
        c, input_bcast_shapes[0], input_bcast_shapes[1], true,
        &output_bcast_shape));
  }

  bool output_has_ellipsis = false;
  TF_RETURN_IF_ERROR(
      ValidateEinsumEllipsis(output_labels, &output_has_ellipsis));
  if (output_has_ellipsis) {
    // If the output subscript has ellipsis and the output broadcast rank is
    // unknown, then the output shape should have unknown rank.
    if (!c->RankKnown(output_bcast_shape)) {
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    }
  } else {
    // If the output subscripts don't have ellipsis then make sure the output
    // broadcasting shape is empty.
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        c->WithRankAtMost(output_bcast_shape, 0, &output_bcast_shape),
        " for einsum equation '", equation,
        "' without ellipsis (...) in the output subscripts where input(s) have "
        "non-empty broadcasting shape");
    output_bcast_shape = c->Scalar();
  }

  // Create the output shape from output labels and label_to_dimension mapping.
  std::vector<DimensionHandle> output_dims;
  for (int label_idx = 0; label_idx < output_labels.size(); ++label_idx) {
    const char label = output_labels[label_idx];
    // Append the output_bcast_shape when the ellipsis is encountered.
    if (label == '.') {
      for (int k = 0; k < c->Rank(output_bcast_shape); ++k) {
        output_dims.push_back(c->Dim(output_bcast_shape, k));
      }
      label_idx += 2;  // Skip the rest of the ellipsis.
      continue;
    }
    auto dimension_it = label_to_dimension.find(label);
    if (dimension_it == label_to_dimension.end()) {
      return errors::InvalidArgument(
          "Einsum output subscripts for equation '", equation, "' has label '",
          label, "' which is not present in the input subscripts");
    }
    output_dims.push_back(dimension_it->second);
  }
  c->set_output(0, c->MakeShape(output_dims));
  return Status::OK();
}

Status BatchMatMulV2Shape(shape_inference::InferenceContext* c) {
  ShapeHandle a_shape;
  ShapeHandle b_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &a_shape));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &b_shape));

  // Determine output rows and columns.
  bool adj_x;
  bool adj_y;
  TF_RETURN_IF_ERROR(c->GetAttr("adj_x", &adj_x));
  TF_RETURN_IF_ERROR(c->GetAttr("adj_y", &adj_y));
  DimensionHandle output_rows = c->Dim(a_shape, adj_x ? -1 : -2);
  DimensionHandle output_cols = c->Dim(b_shape, adj_y ? -2 : -1);

  // Inner dimensions should be compatible.
  DimensionHandle inner_merged;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(a_shape, adj_x ? -2 : -1),
                              c->Dim(b_shape, adj_y ? -1 : -2), &inner_merged));

  // Batch dimensions should broadcast with each other.
  ShapeHandle a_batch_shape;
  ShapeHandle b_batch_shape;
  ShapeHandle output_batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(a_shape, 0, -2, &a_batch_shape));
  TF_RETURN_IF_ERROR(c->Subshape(b_shape, 0, -2, &b_batch_shape));

  TF_RETURN_IF_ERROR(BroadcastBinaryOpOutputShapeFnHelper(
      c, a_batch_shape, b_batch_shape, true, &output_batch_shape));

  ShapeHandle output_shape;
  TF_RETURN_IF_ERROR(c->Concatenate(
      output_batch_shape, c->Matrix(output_rows, output_cols), &output_shape));

  c->set_output(0, output_shape);
  return Status::OK();
}

Status BatchMatMulShape(shape_inference::InferenceContext* c) {
  ShapeHandle a_shape;
  ShapeHandle b_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &a_shape));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &b_shape));

  // Determine output rows and cols.
  bool adj_x;
  bool adj_y;
  TF_RETURN_IF_ERROR(c->GetAttr("adj_x", &adj_x));
  TF_RETURN_IF_ERROR(c->GetAttr("adj_y", &adj_y));
  DimensionHandle output_rows = c->Dim(a_shape, adj_x ? -1 : -2);
  DimensionHandle output_cols = c->Dim(b_shape, adj_y ? -2 : -1);

  // Batch dims match between inputs.
  ShapeHandle a_batch_dims;
  ShapeHandle b_batch_dims;
  ShapeHandle batch_dims;
  TF_RETURN_IF_ERROR(c->Subshape(a_shape, 0, -2, &a_batch_dims));
  TF_RETURN_IF_ERROR(c->Subshape(b_shape, 0, -2, &b_batch_dims));
  TF_RETURN_IF_ERROR(c->Merge(a_batch_dims, b_batch_dims, &batch_dims));

  // Assert inner dims match.
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(a_shape, adj_x ? -2 : -1),
                              c->Dim(b_shape, adj_y ? -1 : -2), &unused));

  ShapeHandle out;
  TF_RETURN_IF_ERROR(
      c->Concatenate(batch_dims, c->Matrix(output_rows, output_cols), &out));
  c->set_output(0, out);
  return Status::OK();
}

// --------------------------------------------------------------------------

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
    TF_RETURN_IF_ERROR(c->Subshape(input_shape, 0, 1, &first));

    ShapeHandle last;
    TF_RETURN_IF_ERROR(c->Subshape(input_shape, 2, &last));

    DimensionHandle input_bias_dim = c->Dim(input_shape, 1);
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
    c->set_output(0, c->Vector(c->Dim(input_shape, 1)));
  } else {
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input_shape));
    c->set_output(0, c->Vector(c->Dim(input_shape, -1)));
  }

  return Status::OK();
}

Status CheckFormatConstraintsOnShape(const TensorFormat tensor_format,
                                     const ShapeHandle shape_handle,
                                     const string& tensor_name,
                                     shape_inference::InferenceContext* c) {
  if (tensor_format == FORMAT_NCHW_VECT_C) {
    // Check that the vect dim has size 4.
    const int num_dims = c->Rank(shape_handle);
    DimensionHandle vect_dim = c->Dim(
        shape_handle, GetTensorInnerFeatureDimIndex(num_dims, tensor_format));
    DimensionHandle unused_vect_dim;
    TF_RETURN_IF_ERROR(c->WithValue(vect_dim, 4, &unused_vect_dim));
  }

  return Status::OK();
}

Status MakeShapeFromFormat(TensorFormat format, DimensionOrConstant N,
                           const std::vector<DimensionOrConstant>& spatial,
                           DimensionOrConstant C, ShapeHandle* out,
                           shape_inference::InferenceContext* context) {
  const int num_dims = GetTensorDimsFromSpatialDims(spatial.size(), format);
  std::vector<DimensionHandle> dims_actual(num_dims);
  dims_actual[GetTensorBatchDimIndex(num_dims, format)] = context->MakeDim(N);
  int outer_c_index = GetTensorFeatureDimIndex(num_dims, format);
  dims_actual[outer_c_index] = context->MakeDim(C);
  if (format == FORMAT_NCHW_VECT_C) {
    dims_actual[GetTensorInnerFeatureDimIndex(num_dims, format)] =
        context->MakeDim(4);
  } else if (format == FORMAT_NHWC_VECT_W) {
    dims_actual[GetTensorInnerWidthDimIndex(num_dims, format)] =
        context->MakeDim(4);
  }
  for (int spatial_dim = 0; spatial_dim < spatial.size(); spatial_dim++) {
    dims_actual[GetTensorSpatialDimIndex(num_dims, format, spatial_dim)] =
        context->MakeDim(spatial[spatial_dim]);
  }
  *out = context->MakeShape(dims_actual);
  return Status::OK();
}

Status DimensionsFromShape(ShapeHandle shape, TensorFormat format,
                           DimensionHandle* batch_dim,
                           gtl::MutableArraySlice<DimensionHandle> spatial_dims,
                           DimensionHandle* filter_dim,
                           InferenceContext* context) {
  const int32 rank = GetTensorDimsFromSpatialDims(spatial_dims.size(), format);
  // Batch.
  *batch_dim = context->Dim(shape, GetTensorBatchDimIndex(rank, format));
  // Spatial.
  for (int spatial_dim_index = 0; spatial_dim_index < spatial_dims.size();
       ++spatial_dim_index) {
    spatial_dims[spatial_dim_index] = context->Dim(
        shape, GetTensorSpatialDimIndex(rank, format, spatial_dim_index));
  }
  // Channel.
  *filter_dim = context->Dim(shape, GetTensorFeatureDimIndex(rank, format));
  if (format == FORMAT_NCHW_VECT_C) {
    TF_RETURN_IF_ERROR(context->Multiply(
        *filter_dim,
        context->Dim(shape, GetTensorInnerFeatureDimIndex(rank, format)),
        filter_dim));
  }
  return Status::OK();
}

Status ShapeFromDimensions(DimensionHandle batch_dim,
                           gtl::ArraySlice<DimensionHandle> spatial_dims,
                           DimensionHandle filter_dim, TensorFormat format,
                           InferenceContext* context, ShapeHandle* shape) {
  const int32 rank = GetTensorDimsFromSpatialDims(spatial_dims.size(), format);
  std::vector<DimensionHandle> out_dims(rank);

  // Batch.
  out_dims[tensorflow::GetTensorBatchDimIndex(rank, format)] = batch_dim;
  // Spatial.
  for (int spatial_dim_index = 0; spatial_dim_index < spatial_dims.size();
       ++spatial_dim_index) {
    out_dims[tensorflow::GetTensorSpatialDimIndex(
        rank, format, spatial_dim_index)] = spatial_dims[spatial_dim_index];
  }
  // Channel.
  if (format == tensorflow::FORMAT_NCHW_VECT_C) {
    // When format is NCHW_VECT_C, factor the feature map count
    // into the outer feature count and the inner feature count (=4).
    TF_RETURN_IF_ERROR(context->Divide(
        filter_dim, 4, /*evenly_divisible=*/true,
        &out_dims[tensorflow::GetTensorFeatureDimIndex(rank, format)]));
    out_dims[GetTensorInnerFeatureDimIndex(rank, format)] = context->MakeDim(4);
  } else {
    out_dims[tensorflow::GetTensorFeatureDimIndex(rank, format)] = filter_dim;
  }

  *shape = context->MakeShape(out_dims);
  return tensorflow::Status::OK();
}

namespace {

Status Conv2DShapeImpl(shape_inference::InferenceContext* c,
                       bool supports_explicit_padding) {
  string data_format_str, filter_format_str;
  if (!c->GetAttr("data_format", &data_format_str).ok()) {
    data_format_str = "NHWC";
  }
  if (!c->GetAttr("filter_format", &filter_format_str).ok()) {
    filter_format_str = "HWIO";
  }

  TensorFormat data_format;
  if (!FormatFromString(data_format_str, &data_format)) {
    return errors::InvalidArgument("Invalid data format string: ",
                                   data_format_str);
  }
  FilterTensorFormat filter_format;
  if (!FilterFormatFromString(filter_format_str, &filter_format)) {
    return errors::InvalidArgument("Invalid filter format string: ",
                                   filter_format_str);
  }

  constexpr int num_spatial_dims = 2;
  const int rank = GetTensorDimsFromSpatialDims(num_spatial_dims, data_format);
  ShapeHandle conv_input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), rank, &conv_input_shape));
  TF_RETURN_IF_ERROR(CheckFormatConstraintsOnShape(
      data_format, conv_input_shape, "conv_input", c));

  // The filter rank should match the input (4 for NCHW, 5 for NCHW_VECT_C).
  ShapeHandle filter_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), rank, &filter_shape));
  TF_RETURN_IF_ERROR(
      CheckFormatConstraintsOnShape(data_format, filter_shape, "filter", c));

  std::vector<int32> dilations;
  TF_RETURN_IF_ERROR(c->GetAttr("dilations", &dilations));

  if (dilations.size() != 4) {
    return errors::InvalidArgument(
        "Conv2D requires the dilation attribute to contain 4 values, but got: ",
        dilations.size());
  }

  std::vector<int32> strides;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));

  // strides.size() should be 4 (NCHW) even if the input is 5 (NCHW_VECT_C).
  if (strides.size() != 4) {
    return errors::InvalidArgument("Conv2D on data format ", data_format_str,
                                   " requires the stride attribute to contain"
                                   " 4 values, but got: ",
                                   strides.size());
  }

  const int32 stride_rows = GetTensorDim(strides, data_format, 'H');
  const int32 stride_cols = GetTensorDim(strides, data_format, 'W');
  const int32 dilation_rows = GetTensorDim(dilations, data_format, 'H');
  const int32 dilation_cols = GetTensorDim(dilations, data_format, 'W');

  DimensionHandle batch_size_dim;
  DimensionHandle input_depth_dim;
  gtl::InlinedVector<DimensionHandle, 2> input_spatial_dims(2);
  TF_RETURN_IF_ERROR(DimensionsFromShape(
      conv_input_shape, data_format, &batch_size_dim,
      absl::MakeSpan(input_spatial_dims), &input_depth_dim, c));

  DimensionHandle output_depth_dim = c->Dim(
      filter_shape, GetFilterDimIndex<num_spatial_dims>(filter_format, 'O'));
  DimensionHandle filter_rows_dim = c->Dim(
      filter_shape, GetFilterDimIndex<num_spatial_dims>(filter_format, 'H'));
  DimensionHandle filter_cols_dim = c->Dim(
      filter_shape, GetFilterDimIndex<num_spatial_dims>(filter_format, 'W'));
  DimensionHandle filter_input_depth_dim;
  if (filter_format == FORMAT_OIHW_VECT_I) {
    TF_RETURN_IF_ERROR(c->Multiply(
        c->Dim(filter_shape,
               GetFilterDimIndex<num_spatial_dims>(filter_format, 'I')),
        c->Dim(filter_shape,
               GetFilterTensorInnerInputChannelsDimIndex(rank, filter_format)),
        &filter_input_depth_dim));
  } else {
    filter_input_depth_dim = c->Dim(
        filter_shape, GetFilterDimIndex<num_spatial_dims>(filter_format, 'I'));
  }

  // Check that the input tensor and the filter tensor agree on the channel
  // count.
  if (c->ValueKnown(input_depth_dim) && c->ValueKnown(filter_input_depth_dim)) {
    int64 input_depth_value = c->Value(input_depth_dim),
          filter_input_depth_value = c->Value(filter_input_depth_dim);
    if (input_depth_value % filter_input_depth_value != 0)
      return errors::InvalidArgument(
          "Depth of input (", input_depth_value,
          ") is not a multiple of input depth of filter (",
          filter_input_depth_value, ")");
    if (input_depth_value != filter_input_depth_value) {
      int64 num_groups = input_depth_value / filter_input_depth_value;
      if (c->ValueKnown(output_depth_dim)) {
        int64 output_depth_value = c->Value(output_depth_dim);
        if (output_depth_value % num_groups != 0)
          return errors::InvalidArgument(
              "Depth of output (", output_depth_value,
              ") is not a multiple of the number of groups (", num_groups, ")");
      }
    }
  }

  Padding padding;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

  std::vector<int64> explicit_paddings;
  if (supports_explicit_padding) {
    Status s = c->GetAttr("explicit_paddings", &explicit_paddings);
    // Use the default value, which is an empty list, if the attribute is not
    // found. Otherwise return the error to the caller.
    if (!s.ok() && !errors::IsNotFound(s)) {
      return s;
    }
    TF_RETURN_IF_ERROR(CheckValidPadding(padding, explicit_paddings,
                                         /*num_dims=*/4, data_format));
  } else {
    CHECK(padding != Padding::EXPLICIT);  // Crash ok.
  }

  DimensionHandle output_rows, output_cols;
  int64 pad_rows_before = -1, pad_rows_after = -1;
  int64 pad_cols_before = -1, pad_cols_after = -1;
  if (padding == Padding::EXPLICIT) {
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'H',
                             &pad_rows_before, &pad_rows_after);
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'W',
                             &pad_cols_before, &pad_cols_after);
  }
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDimsV2(
      c, input_spatial_dims[0], filter_rows_dim, dilation_rows, stride_rows,
      padding, pad_rows_before, pad_rows_after, &output_rows));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDimsV2(
      c, input_spatial_dims[1], filter_cols_dim, dilation_cols, stride_cols,
      padding, pad_cols_before, pad_cols_after, &output_cols));

  ShapeHandle output_shape;
  TF_RETURN_IF_ERROR(
      ShapeFromDimensions(batch_size_dim, {output_rows, output_cols},
                          output_depth_dim, data_format, c, &output_shape));
  c->set_output(0, output_shape);
  return Status::OK();
}

}  // namespace

// Shape function for Conv2D-like operations that support explicit padding.
Status Conv2DShapeWithExplicitPadding(shape_inference::InferenceContext* c) {
  return Conv2DShapeImpl(c, true);
}

// Shape function for Conv2D-like operations that do not support explicit
// padding.
Status Conv2DShape(shape_inference::InferenceContext* c) {
  return Conv2DShapeImpl(c, false);
}

// TODO(mjanusz): Unify all conv/pooling shape functions.
Status Conv3DShape(shape_inference::InferenceContext* c) {
  ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &input_shape));
  ShapeHandle filter_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 5, &filter_shape));

  string data_format;
  Status s = c->GetAttr("data_format", &data_format);

  std::vector<int32> dilations;
  TF_RETURN_IF_ERROR(c->GetAttr("dilations", &dilations));

  if (dilations.size() != 5) {
    return errors::InvalidArgument(
        "Conv3D requires the dilation attribute to contain 5 values, but got: ",
        dilations.size());
  }

  std::vector<int32> strides;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
  if (strides.size() != 5) {
    return errors::InvalidArgument(
        "Conv3D requires the stride attribute to contain 5 values, but got: ",
        strides.size());
  }

  int32 stride_planes, stride_rows, stride_cols;
  int32 dilation_planes, dilation_rows, dilation_cols;
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
    dilation_planes = dilations[2];
    dilation_cols = dilations[3];
    dilation_rows = dilations[4];
  } else {
    stride_planes = strides[1];
    stride_rows = strides[2];
    stride_cols = strides[3];
    dilation_planes = dilations[1];
    dilation_cols = dilations[2];
    dilation_rows = dilations[3];
  }

  DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
  DimensionHandle in_planes_dim = c->Dim(input_shape, 1);
  DimensionHandle in_rows_dim = c->Dim(input_shape, 2);
  DimensionHandle in_cols_dim = c->Dim(input_shape, 3);
  DimensionHandle input_depth_dim = c->Dim(input_shape, 4);

  DimensionHandle filter_planes_dim = c->Dim(filter_shape, 0);
  DimensionHandle filter_rows_dim = c->Dim(filter_shape, 1);
  DimensionHandle filter_cols_dim = c->Dim(filter_shape, 2);
  DimensionHandle filter_input_depth_dim = c->Dim(filter_shape, 3);
  DimensionHandle output_depth_dim = c->Dim(filter_shape, 4);

  // Check that the input tensor and the filter tensor agree on the channel
  // count.
  if (c->ValueKnown(input_depth_dim) && c->ValueKnown(filter_input_depth_dim)) {
    int64 input_depth_value = c->Value(input_depth_dim),
          filter_input_depth_value = c->Value(filter_input_depth_dim);
    if (input_depth_value % filter_input_depth_value != 0)
      return errors::InvalidArgument(
          "Depth of input (", input_depth_value,
          ") is not a multiple of input depth of filter (",
          filter_input_depth_value, ")");
    if (input_depth_value != filter_input_depth_value) {
      int64 num_groups = input_depth_value / filter_input_depth_value;
      if (c->ValueKnown(output_depth_dim)) {
        int64 output_depth_value = c->Value(output_depth_dim);
        if (output_depth_value % num_groups != 0)
          return errors::InvalidArgument(
              "Depth of output (", output_depth_value,
              ") is not a multiple of the number of groups (", num_groups, ")");
      }
    }
  }

  Padding padding;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));
  DimensionHandle output_planes, output_rows, output_cols;

  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDimsV2(
      c, in_planes_dim, filter_planes_dim, dilation_planes, stride_planes,
      padding, -1, -1, &output_planes));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDimsV2(
      c, in_rows_dim, filter_rows_dim, dilation_rows, stride_rows, padding, -1,
      -1, &output_rows));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDimsV2(
      c, in_cols_dim, filter_cols_dim, dilation_cols, stride_cols, padding, -1,
      -1, &output_cols));

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

  std::vector<int32> dilations;
  if (!c->GetAttr("dilations", &dilations).ok()) {
    dilations.resize(4, 1);
  }

  if (dilations.size() != 4) {
    return errors::InvalidArgument(
        "DepthwiseConv2D requires the dilations attribute to contain 4 values, "
        "but got: ",
        dilations.size());
  }

  string data_format;
  Status s = c->GetAttr("data_format", &data_format);
  int32 stride_rows;
  int32 stride_cols;
  int32 dilation_rows;
  int32 dilation_cols;
  if (s.ok() && data_format == "NCHW") {
    // Canonicalize input shape to NHWC so the shape inference code below can
    // process it.
    input_shape =
        c->MakeShape({{c->Dim(input_shape, 0), c->Dim(input_shape, 2),
                       c->Dim(input_shape, 3), c->Dim(input_shape, 1)}});
    stride_rows = strides[2];
    stride_cols = strides[3];
    dilation_rows = dilations[2];
    dilation_cols = dilations[3];
  } else {
    stride_rows = strides[1];
    stride_cols = strides[2];
    dilation_rows = dilations[1];
    dilation_cols = dilations[2];
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

  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDimsV2(
      c, in_rows_dim, filter_rows_dim, dilation_rows, stride_rows, padding, -1,
      -1, &output_rows));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDimsV2(
      c, in_cols_dim, filter_cols_dim, dilation_cols, stride_cols, padding, -1,
      -1, &output_cols));

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
  string data_format_str;
  TensorFormat data_format;
  Status s = c->GetAttr("data_format", &data_format_str);
  if (s.ok()) {
    FormatFromString(data_format_str, &data_format);
  } else {
    data_format = FORMAT_NHWC;
  }

  const int rank = (data_format == FORMAT_NCHW_VECT_C) ? 5 : 4;
  ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), rank, &input_shape));

  TF_RETURN_IF_ERROR(
      CheckFormatConstraintsOnShape(data_format, input_shape, "input", c));

  std::vector<int32> strides;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
  if (strides.size() != 4) {
    return errors::InvalidArgument(
        "AvgPool requires the stride attribute to contain 4 values, but got: ",
        strides.size());
  }

  std::vector<int32> kernel_sizes;
  TF_RETURN_IF_ERROR(c->GetAttr("ksize", &kernel_sizes));
  if (kernel_sizes.size() != 4) {
    return errors::InvalidArgument(
        "AvgPool requires the ksize attribute to contain 4 values, but got: ",
        kernel_sizes.size());
  }

  int32 stride_rows = GetTensorDim(strides, data_format, 'H');
  int32 stride_cols = GetTensorDim(strides, data_format, 'W');
  int32 kernel_rows = GetTensorDim(kernel_sizes, data_format, 'H');
  int32 kernel_cols = GetTensorDim(kernel_sizes, data_format, 'W');

  constexpr int num_spatial_dims = 2;
  DimensionHandle batch_size_dim = c->Dim(
      input_shape, GetTensorDimIndex<num_spatial_dims>(data_format, 'N'));
  DimensionHandle in_rows_dim = c->Dim(
      input_shape, GetTensorDimIndex<num_spatial_dims>(data_format, 'H'));
  DimensionHandle in_cols_dim = c->Dim(
      input_shape, GetTensorDimIndex<num_spatial_dims>(data_format, 'W'));
  DimensionHandle depth_dim = c->Dim(
      input_shape, GetTensorDimIndex<num_spatial_dims>(data_format, 'C'));

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
  TF_RETURN_IF_ERROR(MakeShapeFromFormat(data_format, batch_size_dim,
                                         {output_rows, output_cols}, depth_dim,
                                         &output_shape, c));
  c->set_output(0, output_shape);
  return Status::OK();
}

Status FusedBatchNormShape(shape_inference::InferenceContext* c) {
  ShapeHandle x;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &x));

  bool is_training;
  TF_RETURN_IF_ERROR(c->GetAttr("is_training", &is_training));
  float exponential_avg_factor;
  if (!c->GetAttr("exponential_avg_factor", &exponential_avg_factor).ok()) {
    exponential_avg_factor = 1.0f;  // default value
  }
  int number_inputs = (is_training && exponential_avg_factor == 1.0f) ? 3 : 5;
  string data_format_str;
  TF_RETURN_IF_ERROR(c->GetAttr("data_format", &data_format_str));
  TensorFormat data_format;
  if (!FormatFromString(data_format_str, &data_format)) {
    return errors::InvalidArgument("Invalid data format string: ",
                                   data_format_str);
  }
  int channel_dim_index = GetTensorFeatureDimIndex(4, data_format);
  DimensionHandle channel_dim = c->Dim(x, channel_dim_index);

  // covers scale, offset, and if is_training is false, mean, variance
  for (int i = 1; i < number_inputs; ++i) {
    ShapeHandle vec;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &vec));
    TF_RETURN_IF_ERROR(c->Merge(channel_dim, c->Dim(vec, 0), &channel_dim));
  }

  ShapeHandle y;
  TF_RETURN_IF_ERROR(c->ReplaceDim(x, channel_dim_index, channel_dim, &y));
  c->set_output(0, y);
  ShapeHandle vector_shape = c->Vector(channel_dim);
  c->set_output(1, vector_shape);
  c->set_output(2, vector_shape);
  c->set_output(3, vector_shape);
  c->set_output(4, vector_shape);
  return Status::OK();
}

Status FusedBatchNormV3Shape(shape_inference::InferenceContext* c) {
  TF_RETURN_IF_ERROR(FusedBatchNormShape(c));
  c->set_output(5, c->UnknownShape());
  return Status::OK();
}

Status FusedBatchNormExShape(shape_inference::InferenceContext* c) {
  TF_RETURN_IF_ERROR(FusedBatchNormV3Shape(c));

  string data_format_str;
  TF_RETURN_IF_ERROR(c->GetAttr("data_format", &data_format_str));
  TensorFormat data_format;
  if (!FormatFromString(data_format_str, &data_format)) {
    return errors::InvalidArgument("Invalid data format string: ",
                                   data_format_str);
  }
  ShapeHandle x;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &x));

  int channel_dim_index = GetTensorFeatureDimIndex(4, data_format);
  DimensionHandle channel_dim = c->Dim(x, channel_dim_index);

  // This is a cuDNN implementation constraint.
  if (c->ValueKnown(channel_dim) && c->Value(channel_dim) % 4 != 0) {
    return errors::InvalidArgument(
        "_FusedBatchNormEx channel dimension must be divisible by 4.");
  }

  return Status::OK();
}

Status FusedBatchNormGradShape(shape_inference::InferenceContext* c) {
  ShapeHandle y_backprop;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &y_backprop));
  ShapeHandle x;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &x));

  bool is_training;
  TF_RETURN_IF_ERROR(c->GetAttr("is_training", &is_training));
  string data_format_str;
  TF_RETURN_IF_ERROR(c->GetAttr("data_format", &data_format_str));
  TensorFormat data_format;
  if (!FormatFromString(data_format_str, &data_format)) {
    return errors::InvalidArgument("Invalid data format string: ",
                                   data_format_str);
  }
  int channel_dim_index = GetTensorFeatureDimIndex(4, data_format);
  DimensionHandle channel_dim = c->Dim(y_backprop, channel_dim_index);
  TF_RETURN_IF_ERROR(
      c->Merge(channel_dim, c->Dim(x, channel_dim_index), &channel_dim));

  // covers scale, mean (reserve_space_1), variance (reserve_space_2)
  for (int i = 2; i < 5; ++i) {
    ShapeHandle vec;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &vec));
    TF_RETURN_IF_ERROR(c->Merge(channel_dim, c->Dim(vec, 0), &channel_dim));
  }

  ShapeHandle x_backprop;
  TF_RETURN_IF_ERROR(
      c->ReplaceDim(y_backprop, channel_dim_index, channel_dim, &x_backprop));
  c->set_output(0, x_backprop);
  c->set_output(1, c->Vector(channel_dim));
  c->set_output(2, c->Vector(channel_dim));
  c->set_output(3, c->Vector(0));
  c->set_output(4, c->Vector(0));
  return Status::OK();
}

Status ReadDiagIndex(InferenceContext* c, const Tensor* diag_index_tensor,
                     int32* lower_diag_index, int32* upper_diag_index) {
  // This function assumes that the shape of diag_index_tensor is fully defined.
  if (diag_index_tensor->dims() == 0) {
    *lower_diag_index = diag_index_tensor->scalar<int32>()();
    *upper_diag_index = *lower_diag_index;
  } else {
    int32 num_elements = diag_index_tensor->dim_size(0);
    if (num_elements == 1) {
      *lower_diag_index = diag_index_tensor->vec<int32>()(0);
      *upper_diag_index = *lower_diag_index;
    } else if (num_elements == 2) {
      *lower_diag_index = diag_index_tensor->vec<int32>()(0);
      *upper_diag_index = diag_index_tensor->vec<int32>()(1);
    } else {
      return errors::InvalidArgument(
          "diag_index must be a vector with one or two elements. It has ",
          num_elements, " elements.");
    }
  }
  return Status::OK();
}

Status MatrixDiagPartV2Shape(shape_inference::InferenceContext* c) {
  ShapeHandle input_shape, diag_index_shape, unused_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input_shape));
  TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 1, &diag_index_shape));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused_shape));

  const Tensor* diag_index_tensor = c->input_tensor(1);
  if (!c->RankKnown(input_shape) || !c->FullyDefined(diag_index_shape) ||
      diag_index_tensor == nullptr) {
    c->set_output(0, c->UnknownShape());
    return Status::OK();
  }
  int32 lower_diag_index = 0;
  int32 upper_diag_index = 0;
  TF_RETURN_IF_ERROR(ReadDiagIndex(c, diag_index_tensor, &lower_diag_index,
                                   &upper_diag_index));
  if (lower_diag_index > upper_diag_index) {
    return errors::InvalidArgument(
        "lower_diag_index is greater than upper_diag_index");
  }

  // Validates lower_diag_index and upper_diag_index.
  const int32 input_rank = c->Rank(input_shape);
  const int32 num_rows = c->Value(c->Dim(input_shape, input_rank - 2));
  const int32 num_cols = c->Value(c->Dim(input_shape, input_rank - 1));
  int32 max_diag_len = InferenceContext::kUnknownDim;
  if (num_rows != InferenceContext::kUnknownDim &&
      num_cols != InferenceContext::kUnknownDim) {
    if (lower_diag_index != 0 &&  // For when num_rows or num_cols == 0.
        (-num_rows >= lower_diag_index || lower_diag_index >= num_cols)) {
      return errors::InvalidArgument("lower_diag_index is out of bound.");
    }
    if (upper_diag_index != 0 &&  // For when num_rows or num_cols == 0.
        (-num_rows >= upper_diag_index || upper_diag_index >= num_cols)) {
      return errors::InvalidArgument("upper_diag_index is out of bound.");
    }
    max_diag_len = std::min(num_rows + std::min(upper_diag_index, 0),
                            num_cols - std::max(lower_diag_index, 0));
  }

  std::vector<DimensionHandle> dims;
  dims.reserve(input_rank - 2);
  for (int i = 0; i < input_rank - 2; ++i) {
    dims.push_back(c->Dim(input_shape, i));
  }
  if (lower_diag_index < upper_diag_index) {
    dims.push_back(c->MakeDim(upper_diag_index - lower_diag_index + 1));
  }
  dims.push_back(c->MakeDim(max_diag_len));
  c->set_output(0, c->MakeShape(dims));
  return Status::OK();
}

Status MatrixDiagV2Shape(shape_inference::InferenceContext* c) {
  // Checks input ranks.
  ShapeHandle input_shape, diag_index_shape, unused_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &input_shape));
  TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 1, &diag_index_shape));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused_shape));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused_shape));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused_shape));

  // Reads the diagonal indices.
  const Tensor* diag_index_tensor = c->input_tensor(1);
  if (!c->RankKnown(input_shape) || !c->FullyDefined(diag_index_shape) ||
      diag_index_tensor == nullptr) {
    c->set_output(0, c->UnknownShape());
    return Status::OK();
  }
  int32 lower_diag_index = 0;
  int32 upper_diag_index = 0;
  TF_RETURN_IF_ERROR(ReadDiagIndex(c, diag_index_tensor, &lower_diag_index,
                                   &upper_diag_index));
  if (lower_diag_index > upper_diag_index) {
    return errors::InvalidArgument(
        "lower_diag_index is greater than upper_diag_index");
  }

  // Checks if the number of diagonals provided matches what we imply from
  // lower_diag_index and upper_diag_index.
  const int32 input_rank = c->Rank(input_shape);
  if (lower_diag_index < upper_diag_index) {
    const int32 num_diags = c->Value(c->Dim(input_shape, input_rank - 2));
    const int32 other_dim = c->Value(c->Dim(input_shape, input_rank - 1));

    if (num_diags != (upper_diag_index - lower_diag_index + 1)) {
      return errors::InvalidArgument(
          "The number of rows of `diagonal` doesn't match the number of "
          "diagonals implied from `d_lower` and `d_upper`.\n",
          "num_diags = ", num_diags, ", d_lower = ", lower_diag_index,
          ", d_upper = ", upper_diag_index, " ", input_rank, " ", other_dim);
    }
  }

  // Reads num_rows and num_cols.
  const Tensor* num_rows_tensor = c->input_tensor(2);
  const Tensor* num_cols_tensor = c->input_tensor(3);
  int64 num_rows = -1;
  int64 num_cols = -1;
  if (num_rows_tensor != nullptr) {
    TF_RETURN_IF_ERROR(c->GetScalarFromTensor(num_rows_tensor, &num_rows));
  }
  if (num_cols_tensor != nullptr) {
    TF_RETURN_IF_ERROR(c->GetScalarFromTensor(num_cols_tensor, &num_cols));
  }

  // Infers the missing num_rows or num_cols: If both are missing, assume
  // output is square. Otherwise, use the smallest possible value. Also
  // validates the provided values.
  const int32 max_diag_len = c->Value(c->Dim(input_shape, input_rank - 1));
  const int32 min_num_rows = max_diag_len - std::min(upper_diag_index, 0);
  const int32 min_num_cols = max_diag_len + std::max(lower_diag_index, 0);
  if (num_rows == -1 && num_cols == -1) {  // Special case.
    num_rows = std::max(min_num_rows, min_num_cols);
    num_cols = num_rows;
  }
  if (num_rows == -1) {
    num_rows = min_num_rows;
  } else if (num_rows < min_num_rows) {
    return errors::InvalidArgument("num_rows is too small");
  }
  if (num_cols == -1) {
    num_cols = min_num_cols;
  } else if (num_cols < min_num_cols) {
    return errors::InvalidArgument("num_cols is too small.");
  }
  // At least one of them must match the minimum length.
  if (num_rows != min_num_rows && num_cols != min_num_cols) {
    return errors::InvalidArgument(
        "num_rows and num_cols are not consistent with lower_diag_index, "
        "upper_diag_index, and the length of the given diagonals.\n",
        "num_rows = ", num_rows, " != min_num_rows = ", min_num_rows,
        ", num_cols = ", num_cols, " != min_num_cols = ", min_num_cols);
  }

  // Sets output shape.
  ShapeHandle output_shape;
  const DimensionHandle output_row_dim = c->MakeDim(num_rows);
  const DimensionHandle output_col_dim = c->MakeDim(num_cols);
  if (lower_diag_index == upper_diag_index) {
    TF_RETURN_IF_ERROR(c->ReplaceDim(input_shape, input_rank - 1,
                                     output_row_dim, &output_shape));
    TF_RETURN_IF_ERROR(
        c->Concatenate(output_shape, c->Vector(output_col_dim), &output_shape));
  } else {
    TF_RETURN_IF_ERROR(c->ReplaceDim(input_shape, input_rank - 2,
                                     output_row_dim, &output_shape));
    TF_RETURN_IF_ERROR(c->ReplaceDim(output_shape, input_rank - 1,
                                     output_col_dim, &output_shape));
  }
  c->set_output(0, output_shape);
  return Status::OK();
}

Status MatrixSetDiagV2Shape(shape_inference::InferenceContext* c) {
  ShapeHandle input_shape, diag_shape, diag_index_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input_shape));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &diag_shape));
  TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(2), 1, &diag_index_shape));

  int32 lower_diag_index = 0;
  int32 upper_diag_index = 0;
  bool diag_index_known = false;
  const Tensor* diag_index_tensor = c->input_tensor(2);
  if (diag_index_tensor != nullptr && c->FullyDefined(diag_index_shape)) {
    diag_index_known = true;
    TF_RETURN_IF_ERROR(ReadDiagIndex(c, diag_index_tensor, &lower_diag_index,
                                     &upper_diag_index));
    if (lower_diag_index > upper_diag_index) {
      return errors::InvalidArgument(
          "lower_diag_index is greater than upper_diag_index");
    }
  }

  // Do more checks when input rank is known.
  if (c->RankKnown(input_shape)) {
    int32 input_rank = c->Rank(input_shape);

    // If diag_index is set, we know the exact rank of diagonal.
    if (diag_index_known) {
      TF_RETURN_IF_ERROR(c->WithRank(
          c->input(1),
          (lower_diag_index == upper_diag_index) ? input_rank - 1 : input_rank,
          &diag_shape));
    } else {
      TF_RETURN_IF_ERROR(
          c->WithRankAtLeast(c->input(1), input_rank - 1, &diag_shape));
      TF_RETURN_IF_ERROR(
          c->WithRankAtMost(c->input(1), input_rank, &diag_shape));
    }

    // Validates lower_diag_index and upper_diag_index.
    const int32 num_rows = c->Value(c->Dim(input_shape, input_rank - 2));
    const int32 num_cols = c->Value(c->Dim(input_shape, input_rank - 1));
    if (num_rows != InferenceContext::kUnknownDim &&
        num_cols != InferenceContext::kUnknownDim) {
      if (lower_diag_index != 0 &&  // For when num_rows or num_cols == 0.
          (-num_rows >= lower_diag_index || lower_diag_index >= num_cols)) {
        return errors::InvalidArgument("lower_diag_index is out of bound.");
      }
      if (upper_diag_index != 0 &&  // For when num_rows or num_cols == 0.
          (-num_rows >= upper_diag_index || upper_diag_index >= num_cols)) {
        return errors::InvalidArgument("upper_diag_index is out of bound.");
      }
    }
  }

  ShapeHandle output_shape = input_shape;
  if (c->RankKnown(diag_shape) && !c->FullyDefined(input_shape)) {
    // Try to infer parts of shape from diag.
    ShapeHandle diag_prefix;
    TF_RETURN_IF_ERROR(c->Subshape(
        diag_shape, 0, (lower_diag_index == upper_diag_index) ? -1 : -2,
        &diag_prefix));

    // The inner matrices can be rectangular, so we can't pinpoint their
    // exact height and width by just lower_diag_index, upper_diag_index,
    // and the longest length of given diagonals.
    TF_RETURN_IF_ERROR(
        c->Concatenate(diag_prefix, c->UnknownShapeOfRank(2), &diag_shape));
    TF_RETURN_IF_ERROR(c->Merge(input_shape, diag_shape, &output_shape));
  }
  c->set_output(0, output_shape);
  return Status::OK();
}

Status MaxPoolShape(shape_inference::InferenceContext* c) {
  string data_format_str;
  TensorFormat data_format;
  Status s = c->GetAttr("data_format", &data_format_str);
  if (s.ok()) {
    FormatFromString(data_format_str, &data_format);
  } else {
    data_format = FORMAT_NHWC;
  }

  const int rank = (data_format == FORMAT_NCHW_VECT_C) ? 5 : 4;
  ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), rank, &input_shape));

  TF_RETURN_IF_ERROR(
      CheckFormatConstraintsOnShape(data_format, input_shape, "input", c));

  std::vector<int32> strides;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
  if (strides.size() != 4) {
    return errors::InvalidArgument(
        "MaxPool requires the stride attribute to contain 4 values, but got: ",
        strides.size());
  }

  std::vector<int32> kernel_sizes;
  TF_RETURN_IF_ERROR(c->GetAttr("ksize", &kernel_sizes));
  if (kernel_sizes.size() != 4) {
    return errors::InvalidArgument(
        "MaxPool requires the ksize attribute to contain 4 values, but got: ",
        kernel_sizes.size());
  }

  int32 stride_depth = GetTensorDim(strides, data_format, 'C');
  int32 stride_rows = GetTensorDim(strides, data_format, 'H');
  int32 stride_cols = GetTensorDim(strides, data_format, 'W');
  int32 kernel_depth = GetTensorDim(kernel_sizes, data_format, 'C');
  int32 kernel_rows = GetTensorDim(kernel_sizes, data_format, 'H');
  int32 kernel_cols = GetTensorDim(kernel_sizes, data_format, 'W');

  constexpr int num_spatial_dims = 2;
  DimensionHandle batch_size_dim = c->Dim(
      input_shape, GetTensorDimIndex<num_spatial_dims>(data_format, 'N'));
  DimensionHandle in_rows_dim = c->Dim(
      input_shape, GetTensorDimIndex<num_spatial_dims>(data_format, 'H'));
  DimensionHandle in_cols_dim = c->Dim(
      input_shape, GetTensorDimIndex<num_spatial_dims>(data_format, 'W'));
  DimensionHandle in_depth_dim = c->Dim(
      input_shape, GetTensorDimIndex<num_spatial_dims>(data_format, 'C'));

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

  TF_RETURN_IF_ERROR(MakeShapeFromFormat(data_format, batch_size_dim,
                                         {output_rows, output_cols},
                                         output_depth, &output_shape, c));

  c->set_output(0, output_shape);
  return Status::OK();
}

Status MaxPoolV2Shape(shape_inference::InferenceContext* c, int num_inputs) {
  string data_format_str;
  TensorFormat data_format;
  Status s = c->GetAttr("data_format", &data_format_str);
  if (s.ok()) {
    FormatFromString(data_format_str, &data_format);
  } else {
    data_format = FORMAT_NHWC;
  }

  const int rank = (data_format == FORMAT_NCHW_VECT_C) ? 5 : 4;
  ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), rank, &input_shape));

  TF_RETURN_IF_ERROR(
      CheckFormatConstraintsOnShape(data_format, input_shape, "input", c));

  std::vector<int32> kernel_sizes;
  std::vector<int32> strides;

  if (c->num_inputs() + 2 == num_inputs) {
    TF_RETURN_IF_ERROR(c->GetAttr("ksize", &kernel_sizes));

    TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
  } else {
    // Verify shape of ksize and strides input.
    ShapeHandle size;
    DimensionHandle unused;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs() - 2), 1, &size));
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(size, 0), 4, &unused));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(c->num_inputs() - 1), 1, &size));
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(size, 0), 4, &unused));

    const Tensor* kernel_sizes_tensor = c->input_tensor(c->num_inputs() - 2);
    if (kernel_sizes_tensor == nullptr) {
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    }
    kernel_sizes.resize(kernel_sizes_tensor->shape().num_elements());
    auto kernel_sizes_vec = kernel_sizes_tensor->flat<int32>();
    std::copy_n(&kernel_sizes_vec(0), kernel_sizes.size(),
                kernel_sizes.begin());

    const Tensor* strides_tensor = c->input_tensor(c->num_inputs() - 1);
    if (strides_tensor == nullptr) {
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    }
    strides.resize(strides_tensor->shape().num_elements());
    auto strides_vec = strides_tensor->flat<int32>();
    std::copy_n(&strides_vec(0), strides.size(), strides.begin());
  }

  if (strides.size() != 4) {
    return errors::InvalidArgument(
        "MaxPool requires the stride attribute to contain 4 values, but "
        "got: ",
        strides.size());
  }
  if (kernel_sizes.size() != 4) {
    return errors::InvalidArgument(
        "MaxPool requires the ksize attribute to contain 4 values, but got: ",
        kernel_sizes.size());
  }

  int32 stride_depth = GetTensorDim(strides, data_format, 'C');
  int32 stride_rows = GetTensorDim(strides, data_format, 'H');
  int32 stride_cols = GetTensorDim(strides, data_format, 'W');
  int32 kernel_depth = GetTensorDim(kernel_sizes, data_format, 'C');
  int32 kernel_rows = GetTensorDim(kernel_sizes, data_format, 'H');
  int32 kernel_cols = GetTensorDim(kernel_sizes, data_format, 'W');

  constexpr int num_spatial_dims = 2;
  DimensionHandle batch_size_dim = c->Dim(
      input_shape, GetTensorDimIndex<num_spatial_dims>(data_format, 'N'));
  DimensionHandle in_rows_dim = c->Dim(
      input_shape, GetTensorDimIndex<num_spatial_dims>(data_format, 'H'));
  DimensionHandle in_cols_dim = c->Dim(
      input_shape, GetTensorDimIndex<num_spatial_dims>(data_format, 'W'));
  DimensionHandle in_depth_dim = c->Dim(
      input_shape, GetTensorDimIndex<num_spatial_dims>(data_format, 'C'));

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

  TF_RETURN_IF_ERROR(MakeShapeFromFormat(data_format, batch_size_dim,
                                         {output_rows, output_cols},
                                         output_depth, &output_shape, c));

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

template <typename T>
Status ReductionShapeHelper(const Tensor* reduction_indices_t,
                            const int32 input_rank,
                            std::set<int64>* true_indices) {
  auto reduction_indices = reduction_indices_t->flat<T>();
  for (int i = 0; i < reduction_indices_t->NumElements(); ++i) {
    const T reduction_index = reduction_indices(i);
    if (reduction_index < -input_rank || reduction_index >= input_rank) {
      return errors::InvalidArgument("Invalid reduction dimension ",
                                     reduction_index, " for input with ",
                                     input_rank, " dimensions.");
    }

    auto wrapped_index = reduction_index;
    if (wrapped_index < 0) {
      wrapped_index += input_rank;
    }

    true_indices->insert(wrapped_index);
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
  std::set<int64> true_indices;
  if (reduction_indices_t->dtype() == DataType::DT_INT32) {
    TF_RETURN_IF_ERROR(ReductionShapeHelper<int32>(reduction_indices_t,
                                                   input_rank, &true_indices));
  } else if (reduction_indices_t->dtype() == DataType::DT_INT64) {
    TF_RETURN_IF_ERROR(ReductionShapeHelper<int64>(reduction_indices_t,
                                                   input_rank, &true_indices));
  } else {
    return errors::InvalidArgument(
        "reduction_indices can only be int32 or int64");
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
        break;
      }
    }
    if (rank == InferenceContext::kUnknownRank) {
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    } else if (rank == 0) {
      return errors::InvalidArgument(
          "Can't concatenate scalars (use tf.stack instead)");
    } else {
      for (int i = start_value_index; i < end_value_index; ++i) {
        // Check that all the inputs are of the correct rank.
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), rank, &unused));
      }
    }
    // Build result of <rank> different unknown dims.
    std::vector<DimensionHandle> dims;
    dims.reserve(rank);
    for (int i = 0; i < rank; ++i) dims.push_back(c->UnknownDim());
    c->set_output(0, c->MakeShape(dims));
    return Status::OK();
  }

  // Merge all the non-concat dims, and sum the concat dim to make an output
  // shape.
  int64 concat_dim;
  if (concat_dim_t->dtype() == DT_INT32) {
    concat_dim = static_cast<int64>(concat_dim_t->flat<int32>()(0));
  } else {
    concat_dim = concat_dim_t->flat<int64>()(0);
  }

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

Status QuantizedConcatV2Shape(InferenceContext* c, int num_inputs_to_concat) {
  return ConcatShapeHelper(c, 0 /* start_value_index */,
                           num_inputs_to_concat /* end_value_index */,
                           num_inputs_to_concat /* dim_index */);
}

Status BroadcastBinaryOpOutputShapeFnHelper(InferenceContext* c,
                                            ShapeHandle shape_x,
                                            ShapeHandle shape_y,
                                            bool incompatible_shape_error,
                                            ShapeHandle* out) {
  CHECK_NOTNULL(out);
  if (!c->RankKnown(shape_x) || !c->RankKnown(shape_y)) {
    *out = c->UnknownShape();
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
        if (!incompatible_shape_error) {
          *out = c->UnknownShape();
          return Status::OK();
        }
        dims.push_back(dim_x);
      } else if (c->Value(dim_y) > 1) {
        if (!incompatible_shape_error) {
          *out = c->UnknownShape();
          return Status::OK();
        }
        dims.push_back(dim_y);
      } else if (c->Value(dim_x) == 1) {
        dims.push_back(dim_y);
      } else if (c->Value(dim_y) == 1) {
        dims.push_back(dim_x);
      } else if (dim_y.SameHandle(dim_x)) {
        dims.push_back(dim_x);
      } else {
        if (!incompatible_shape_error) {
          *out = c->UnknownShape();
          return Status::OK();
        }
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
      Status s = c->Merge(dim_x, dim_y, &dim);
      if (!s.ok()) {
        if (!incompatible_shape_error) {
          *out = c->MakeShape({});
          return Status::OK();
        }
        return s;
      }
      dims.push_back(dim);
    }
  }

  *out = c->MakeShape(dims);
  return Status::OK();
}

Status RandomShape(shape_inference::InferenceContext* c) {
  shape_inference::ShapeHandle out;
  TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
  c->set_output(0, out);
  return Status::OK();
}

Status UnsortedSegmentReductionShapeFn(InferenceContext* c) {
  ShapeHandle s_data = c->input(0);
  ShapeHandle s_segment_ids = c->input(1);
  ShapeHandle s_num_segments = c->input(2);
  TF_RETURN_IF_ERROR(c->WithRank(s_num_segments, 0, &s_num_segments));

  ShapeHandle out;

  // Leading dimensions of data must be compatible with dimensions of
  // <s_segment_ids>.
  if (c->RankKnown(s_segment_ids)) {
    TF_RETURN_IF_ERROR(
        c->MergePrefix(s_data, s_segment_ids, &s_data, &s_segment_ids));

    // Get the value of the num_segments input tensor.
    DimensionHandle num_segments_dim;
    TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(2, &num_segments_dim));

    // Output is {segment_id_rank} + s_data[segment_id_rank:].
    ShapeHandle s_data_suffix;
    TF_RETURN_IF_ERROR(
        c->Subshape(s_data, c->Rank(s_segment_ids), &s_data_suffix));
    TF_RETURN_IF_ERROR(
        c->Concatenate(c->Vector(num_segments_dim), s_data_suffix, &out));
  } else {
    out = c->UnknownShape();
  }
  c->set_output(0, out);
  return Status::OK();
}

namespace {

// This SliceHelper processes the output shape of the `slice`
// when the tensor of `sizes` is available.
template <typename T>
Status SliceHelper(InferenceContext* c, ShapeHandle begin_value,
                   const Tensor* sizes_value,
                   std::vector<DimensionHandle>* dims) {
  auto sizes_vec = sizes_value->vec<T>();
  for (int i = 0; i < sizes_value->NumElements(); ++i) {
    DimensionHandle dim = c->Dim(c->input(0), i);
    if (sizes_vec(i) != -1) {
      auto dim_val = c->Value(dim);
      if (sizes_vec(i) < 0) {
        return errors::InvalidArgument(
            "Out of bounds slicing on dimension ", i, " of length ", dim_val,
            ": sizes vector cannot be < -1, but was ", sizes_vec(i));
      }

      dims->emplace_back(c->MakeDim(sizes_vec(i)));
    } else {
      DimensionHandle result;
      TF_RETURN_IF_ERROR(c->Subtract(dim, c->Dim(begin_value, i), &result));
      dims->emplace_back(result);
    }
  }

  return Status::OK();
}
}  // namespace

Status SliceShape(InferenceContext* c) {
  ShapeHandle input = c->input(0);
  ShapeHandle begin_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &begin_shape));
  ShapeHandle sizes_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &sizes_shape));

  // Merge to check compatibility of begin and sizes tensors.
  TF_RETURN_IF_ERROR(c->Merge(begin_shape, sizes_shape, &begin_shape));

  DimensionHandle ndims = c->Dim(begin_shape, 0);
  if (c->ValueKnown(ndims)) {
    TF_RETURN_IF_ERROR(c->WithRank(input, c->Value(ndims), &input));
  }

  // NOTE(mrry): Use MakeShapeFromShapeTensor to handle partially-known
  // values, even though the `begin` value does not represent a shape.
  ShapeHandle begin_value;
  TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &begin_value));

  // We check the tensor value here and will only use
  // `MakeShapeFromShapeTensor` when `sizes_value` is null.
  // The reason is that `sizes` might contain -1, which can't
  // be represented (-1 in the ShapeHandle would mean "unknown").
  const Tensor* sizes_value = c->input_tensor(2);

  if (sizes_value != nullptr) {
    TF_RETURN_IF_ERROR(
        c->WithRank(begin_value, sizes_value->NumElements(), &begin_value));
    std::vector<DimensionHandle> dims;
    // If the begin and sizes tensors are available, then
    // we can be precise about the shape of the output.
    if (sizes_value->dtype() == DT_INT64) {
      TF_RETURN_IF_ERROR(
          SliceHelper<int64>(c, begin_value, sizes_value, &dims));
    } else {
      TF_RETURN_IF_ERROR(
          SliceHelper<int32>(c, begin_value, sizes_value, &dims));
    }
    c->set_output(0, c->MakeShape(dims));
    return Status::OK();
  } else {
    // In case `sizes` is not available (`sizes_value` is null),
    // we could try to use `MakeShapeFromShapeTensor` here.
    // If sizes contain -1, we will simply consider it as `Unknown`.
    // This is less than ideal but still an improvement of shape inference.
    // The following is an example that returns [None, 1, None] with this
    // code path:
    //   z = tf.zeros((1, 2, 3))
    //   m = tf.slice(z, [0, 0, 0], [tf.constant(1) + 0, 1, -1])
    //   m.get_shape().as_list()
    ShapeHandle sizes_value;
    TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &sizes_value));
    if (c->RankKnown(sizes_value)) {
      TF_RETURN_IF_ERROR(
          c->WithRank(begin_value, c->Rank(sizes_value), &begin_value));
      std::vector<DimensionHandle> dims;
      dims.reserve(c->Rank(sizes_value));
      for (int i = 0; i < c->Rank(sizes_value); ++i) {
        dims.emplace_back(c->Dim(sizes_value, i));
      }
      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    }
    // We might know the rank of the input.
    if (c->RankKnown(input)) {
      c->set_output(0, c->UnknownShapeOfRank(c->Rank(input)));
      return Status::OK();
    } else {
      return shape_inference::UnknownShape(c);
    }
  }

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

Status ValidateVariableResourceHandle(
    InferenceContext* c, std::vector<ShapeAndType>* shape_and_type) {
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data == nullptr || handle_data->empty()) {
    shape_and_type->emplace_back(c->UnknownShape(), DT_INVALID);
  } else {
    *shape_and_type = *handle_data;
    DataType value_dtype;
    TF_RETURN_IF_ERROR(c->GetAttr("dtype", &value_dtype));
    if (shape_and_type->at(0).dtype != value_dtype) {
      return errors::InvalidArgument(
          "Trying to read variable with wrong dtype. "
          "Expected ",
          DataTypeString(shape_and_type->at(0).dtype), " got ",
          DataTypeString(value_dtype));
    }
  }
  return Status::OK();
}

Status GatherNdShape(InferenceContext* c) {
  ShapeHandle params;
  std::vector<ShapeAndType> handle_shape_and_type;
  if (c->input_handle_shapes_and_types(0) != nullptr) {
    TF_RETURN_IF_ERROR(
        ValidateVariableResourceHandle(c, &handle_shape_and_type));
    params = handle_shape_and_type[0].shape;
  } else {
    params = c->input(0);
  }
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &indices));
  DimensionHandle r_dim = c->Dim(indices, -1);

  if (!c->RankKnown(params) || !c->ValueKnown(r_dim)) {
    c->set_output(0, c->UnknownShape());
    return Status::OK();
  }

  if (c->Value(r_dim) > c->Rank(params)) {
    return errors::InvalidArgument(
        "indices.shape[-1] must be <= params.rank, but saw indices shape: ",
        c->DebugString(indices), " and params shape: ", c->DebugString(params));
  }

  // Remove r_dim from indices to get output.
  ShapeHandle indices_slice;
  ShapeHandle params_slice;
  TF_RETURN_IF_ERROR(c->Subshape(indices, 0, -1, &indices_slice));
  TF_RETURN_IF_ERROR(c->Subshape(params, c->Value(r_dim), &params_slice));
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->Concatenate(indices_slice, params_slice, &out));
  c->set_output(0, out);
  return Status::OK();
}

Status ScatterNdUpdateShape(InferenceContext* c) {
  ShapeHandle input_shape = c->input(0);
  if (c->input_handle_shapes_and_types(0) != nullptr) {
    // This is called for tf.scatter_nd_update; input is a Variable handle.
    const auto& shape_and_type = *(c->input_handle_shapes_and_types(0));
    if (shape_and_type.size() == 1) {
      input_shape = shape_and_type[0].shape;
    }
  }
  ShapeHandle indices_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &indices_shape));
  ShapeHandle updates_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 1, &updates_shape));

  if (c->Value(c->NumElements(input_shape)) == 0 &&
      (c->Value(c->NumElements(indices_shape)) > 0 ||
       c->Value(c->NumElements(updates_shape)) > 0)) {
    return errors::InvalidArgument(
        "Indices and updates specified for empty output shape");
  }

  if (c->RankKnown(indices_shape) && c->RankKnown(updates_shape)) {
    const int64 num_outer_dims = c->Rank(indices_shape) - 1;
    const DimensionHandle index_size = c->Dim(indices_shape, -1);

    // We can only do more validation if the last dimension of indices
    // is a known value.
    if (c->ValueKnown(index_size)) {
      const int64 ix = c->Value(index_size);
      ShapeHandle unused;
      ShapeHandle prefix_indices;
      TF_RETURN_IF_ERROR(
          c->Subshape(indices_shape, 0, num_outer_dims, &prefix_indices));
      ShapeHandle prefix_updates;
      TF_RETURN_IF_ERROR(
          c->Subshape(updates_shape, 0, num_outer_dims, &prefix_updates));

      Status s = c->Merge(prefix_indices, prefix_updates, &unused);
      if (!s.ok()) {
        return errors::InvalidArgument(
            "The outer ", num_outer_dims,
            " dimensions of indices.shape=", c->DebugString(indices_shape),
            " must match the outer ", num_outer_dims,
            " dimensions of updates.shape=", c->DebugString(updates_shape),
            ": ", s.error_message());
      }

      ShapeHandle input_suffix;
      TF_RETURN_IF_ERROR(c->Subshape(input_shape, ix, &input_suffix));
      ShapeHandle suffix_updates;
      TF_RETURN_IF_ERROR(
          c->Subshape(updates_shape, num_outer_dims, &suffix_updates));
      s = c->Merge(input_suffix, suffix_updates, &unused);
      if (!s.ok()) {
        return errors::InvalidArgument(
            "The inner ", c->Rank(input_shape) - ix,
            " dimensions of input.shape=", c->DebugString(input_shape),
            " must match the inner ", c->Rank(updates_shape) - num_outer_dims,
            " dimensions of updates.shape=", c->DebugString(updates_shape),
            ": ", s.error_message());
      }
    }
  }

  if (c->input_handle_shapes_and_types(0) == nullptr && c->num_outputs() > 0) {
    // This is called for tf.scatter_nd; output is a tensor with this shape.
    c->set_output(0, input_shape);
  }
  return Status::OK();
}

Status ExplicitShape(InferenceContext* c) {
  PartialTensorShape shape;
  TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape));
  ShapeHandle output_shape;
  TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shape, &output_shape));
  c->set_output(0, output_shape);
  return Status::OK();
}

Status ExplicitShapes(InferenceContext* c) {
  std::vector<PartialTensorShape> shapes;
  TF_RETURN_IF_ERROR(c->GetAttr("shapes", &shapes));
  if (shapes.empty()) {
    return errors::Internal("shapes attribute is empty");
  }
  for (int i = 0; i < shapes.size(); ++i) {
    ShapeHandle output_shape;
    TF_RETURN_IF_ERROR(
        c->MakeShapeFromPartialTensorShape(shapes[i], &output_shape));
    c->set_output(i, output_shape);
  }
  return Status::OK();
}

Status SparseReduceShapeFn(InferenceContext* c) {
  // Input 0: input_indices
  // Input 1: input_values
  // Input 2: input_shape
  // Input 3: reduction_axes
  // Attr: keep_dims
  bool keep_dims = false;
  TF_RETURN_IF_ERROR(c->GetAttr("keep_dims", &keep_dims));

  const Tensor* shape_tensor = c->input_tensor(2);
  const Tensor* axes_tensor = c->input_tensor(3);
  if (shape_tensor != nullptr && axes_tensor != nullptr) {
    auto shape_vec = shape_tensor->flat<int64>();
    auto axes_vec = axes_tensor->flat<int32>();

    int64 ndims = shape_vec.size();
    absl::flat_hash_set<int64> axes;
    for (int i = 0; i < axes_vec.size(); i++) {
      axes.insert((axes_vec(i) + ndims) % ndims);
    }

    std::vector<DimensionHandle> dims;
    if (keep_dims) {
      dims.reserve(ndims);
      for (int d = 0; d < ndims; ++d) {
        if (axes.find(d) == axes.end()) {
          dims.push_back(c->MakeDim(shape_vec(d)));
        } else {
          dims.push_back(c->MakeDim(1));
        }
      }
    } else {
      for (int d = 0; d < ndims; ++d) {
        if (axes.find(d) == axes.end()) {
          dims.push_back(c->MakeDim(shape_vec(d)));
        }
      }
    }

    c->set_output(0, c->MakeShape(dims));
    return Status::OK();
  }
  return UnknownShape(c);
}

}  // namespace shape_inference

}  // namespace tensorflow
