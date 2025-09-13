/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/shlo/ops/convolution.h"

#include <algorithm>
#include <cstddef>
#include <set>
#include <string>
#include <type_traits>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/convolution_helper_functions.h"
#include "tensorflow/lite/experimental/shlo/ops/dot_general.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

template <DataType storage_type>
absl::Status PrepareImpl(ConvolutionOp& op, const Tensor& lhs,
                         const Tensor& rhs, Tensor& output) {
  using StorageT = StorageType<storage_type>;
  const int64_t* padding_buffer =
      op.attributes.padding.GetDataAs<DataType::kSI64>();

  // Constraints Check
  if (op.attributes.precision_configs.size() != 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: Size of precision_config "
        "must be two.");
  }
  Axis rank = lhs.Rank();
  if (lhs.Rank() != rhs.Rank()) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: rank(lhs) == rank(rhs)");
  }
  if (output.Rank() != lhs.Rank()) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: rank(output) == "
        "lhs.Rank()");
  }
  if (!lhs.IsQuantized()) {
    SHLO_REF_RETURN_ON_ERROR(
        CheckSameBaselineType(CheckCtx("Convolution"), lhs, rhs));
    SHLO_REF_RETURN_ON_ERROR(
        CheckSameBaselineType(CheckCtx("Convolution"), lhs, output));
  }
  if (op.attributes.window_strides.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: size(window_stride) = "
        "rank - 2");
  }
  if (!IsGreaterThanZero(op.attributes.window_strides)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: 0 < window_stride");
  }
  if (op.attributes.padding.shape().Dim(0) != rank - 2 ||
      op.attributes.padding.shape().Dim(1) != 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: shape(padding) = [rank - "
        "2, 2]");
  }
  if (op.attributes.lhs_dilation.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: shape(lhs_dilation) == "
        "rank - 2");
  }
  if (!IsGreaterThanZero(op.attributes.lhs_dilation)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: 0 < lhs_dilation");
  }
  if (op.attributes.rhs_dilation.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: shape(rhs_dilation) == "
        "rank - 2");
  }
  if (!IsGreaterThanZero(op.attributes.rhs_dilation)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: 0 < rhs_dilation");
  }
  if (lhs.shape().Dim(static_cast<Axis>(op.attributes.input_batch_dimension)) %
          op.attributes.batch_group_count !=
      0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: "
        "Dim(lhs,input_batch_dimension) % batch_group_count = 0");
  }
  if (lhs.shape().Dim(
          static_cast<Axis>(op.attributes.input_feature_dimension)) %
          op.attributes.feature_group_count !=
      0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: "
        "Dim(lhs,input_feature_dimension) % (feature_group_count) = 0");
  }
  if (op.attributes.input_spatial_dimensions.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: "
        "size(input_spatial_dimensions) = rank - 2");
  }
  if (!IsUnique(op.attributes.input_batch_dimension,
                op.attributes.input_feature_dimension,
                op.attributes.input_spatial_dimensions)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: "
        "isUnique(input_dimensions)");
  }
  if (!IsInRange(op.attributes.input_batch_dimension,
                 op.attributes.input_feature_dimension,
                 op.attributes.input_spatial_dimensions, rank)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: 0 <= input_dimensions < "
        "rank");
  }
  if (rhs.shape().Dim(
          static_cast<Axis>(op.attributes.kernel_input_feature_dimension)) !=
      lhs.shape().Dim(
          static_cast<Axis>(op.attributes.input_feature_dimension)) /
          op.attributes.feature_group_count) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: "
        "Dim(rhs,kernel_input_feature_dimension) = "
        "Dim(lhs,input_feature_dimension) / feature_group_count");
  }
  if (rhs.shape().Dim(
          static_cast<Axis>(op.attributes.kernel_output_feature_dimension)) %
          op.attributes.batch_group_count !=
      0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: "
        "Dim(rhs,kernel_output_feature_dimension) % batch_group_count = 0");
  }
  if (rhs.shape().Dim(
          static_cast<Axis>(op.attributes.kernel_output_feature_dimension)) %
          op.attributes.feature_group_count !=
      0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: "
        "Dim(rhs,kernel_output_feature_dimension) % (feature_group_count) = 0");
  }
  if (op.attributes.kernel_spatial_dimensions.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: "
        "size(kernel_spatial_dimensions) = rank - 2");
  }
  if (!IsUnique(op.attributes.kernel_output_feature_dimension,
                op.attributes.kernel_input_feature_dimension,
                op.attributes.kernel_spatial_dimensions)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: "
        "isUnique(kernel_dimensions)");
  }
  if (!IsInRange(op.attributes.kernel_output_feature_dimension,
                 op.attributes.kernel_input_feature_dimension,
                 op.attributes.kernel_spatial_dimensions, rank)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: 0<= kernel_dimensions < "
        "rank");
  }
  if (op.attributes.output_spatial_dimensions.size() != rank - 2) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: "
        "size(output_spatial_dimensions) = rank - 2");
  }
  if (!IsUnique(op.attributes.output_batch_dimension,
                op.attributes.output_feature_dimension,
                op.attributes.output_spatial_dimensions)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: "
        "isUnique(output_dimensions)");
  }
  if (!IsInRange(op.attributes.output_batch_dimension,
                 op.attributes.output_feature_dimension,
                 op.attributes.output_spatial_dimensions, rank)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: 0 <= output_dimensions < "
        "rank");
  }
  if (op.attributes.feature_group_count <= 0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: 0 < feature_group_count");
  }
  if (op.attributes.batch_group_count <= 0) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: 0 < batch_group_count");
  }
  if (op.attributes.batch_group_count != 1 &&
      op.attributes.feature_group_count != 1) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: batch_group_count == 1 or "
        "feature_group_count "
        "== 1");
  }
  if (output.shape().Dim(
          static_cast<Axis>(op.attributes.output_batch_dimension)) !=
      lhs.shape().Dim(static_cast<Axis>(op.attributes.input_batch_dimension)) /
          op.attributes.batch_group_count) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: "
        "output.shape().Dim(output_batch_dimension) == "
        "lhs.shape().Dim(input_batch_dimension) / batch_group_count");
  }
  if (output.shape().Dim(
          static_cast<Axis>(op.attributes.output_feature_dimension)) !=
      rhs.shape().Dim(
          static_cast<Axis>(op.attributes.kernel_output_feature_dimension))) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: "
        "output.shape().Dim(output_feature_dimension) == "
        "rhs.shape().Dim(kernel_output_feature_dimension)");
  }
  if (!CheckOutputSpatial(op, lhs, rhs, output)) {
    return absl::FailedPreconditionError(
        "stablehlo.convolution constraint violation: "
        "output.shape().Dim(spatial_dim) is not "
        "properly set");
  }
  if (lhs.IsQuantized() || rhs.IsQuantized() || output.IsQuantized()) {
    if (!(lhs.IsQuantized() && rhs.IsQuantized() && output.IsQuantized())) {
      return absl::FailedPreconditionError(
          "stablehlo.convolution constraint violation: lhs.IsQuantized() && "
          "rhs.IsQuantized() && "
          "output.IsQuantized()");
    }
    if (rhs.IsPerTensorQuantized()) {
      if (!(output.IsPerTensorQuantized())) {
        return absl::FailedPreconditionError(
            "stablehlo.convolution constraint violation: If "
            "is_per_tensor_quantized(rhs), then "
            "is_per_tensor_quantized(output)");
      }
    }
    if (rhs.IsPerAxisQuantized()) {
      if (rhs.quantized_per_axis_element_type().QuantizedDimension() !=
          op.attributes.kernel_output_feature_dimension) {
        return absl::FailedPreconditionError(
            "stablehlo.convolution constraint violation:  If "
            "is_per_axis_quantized(rhs), then "
            "quantization_dimension(rhs) = "
            "op.attributes.kernel_output_feature_dimension");
      }
    }
    if (output.IsPerAxisQuantized()) {
      if (output.quantized_per_axis_element_type().QuantizedDimension() !=
          op.attributes.output_feature_dimension) {
        return absl::FailedPreconditionError(
            "stablehlo.convolution constraint violation:  If "
            "is_per_axis_quantized(output), then "
            "quantization_dimension(output) = "
            "op.attributes.output_feature_dimension");
      }
    }
  }

  // Transpose prepare
  absl::InlinedVector<Axis, kMaxNumDimensions> lhs_permutation_values(
      lhs.Rank(), 0);
  lhs_permutation_values[0] = op.attributes.input_batch_dimension;
  lhs_permutation_values[1] = op.attributes.input_feature_dimension;
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> lhs_shape_dims(
      lhs.Rank(), 0);
  lhs_shape_dims[0] =
      lhs.shape().Dim(static_cast<Axis>(op.attributes.input_batch_dimension));
  lhs_shape_dims[1] =
      lhs.shape().Dim(static_cast<Axis>(op.attributes.input_feature_dimension));
  for (size_t i = 0; i < lhs.Rank() - 2; ++i) {
    lhs_shape_dims[i + 2] = lhs.shape().Dim(
        static_cast<Axis>(op.attributes.input_spatial_dimensions[i]));
    lhs_permutation_values[i + 2] = op.attributes.input_spatial_dimensions[i];
  }

  op.lhs_transposed_data =
      std::vector<std::byte>(lhs.NumElements() * sizeof(StorageT));
  const Shape lhs_transposed_shape(lhs_shape_dims);
  Tensor lhs_transposed{.type = TensorType{.shape = lhs_transposed_shape,
                                           .element_type = storage_type},
                        .data = op.lhs_transposed_data.data()};

  absl::InlinedVector<Axis, kMaxNumDimensions> rhs_permutation_values(
      rhs.Rank(), 0);
  rhs_permutation_values[0] = op.attributes.kernel_output_feature_dimension;
  rhs_permutation_values[1] = op.attributes.kernel_input_feature_dimension;
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> rhs_shape_dims(
      rhs.Rank(), 0);
  rhs_shape_dims[0] = rhs.shape().Dim(
      static_cast<Axis>(op.attributes.kernel_output_feature_dimension));
  rhs_shape_dims[1] = rhs.shape().Dim(
      static_cast<Axis>(op.attributes.kernel_input_feature_dimension));
  for (size_t i = 0; i < rhs.Rank() - 2; ++i) {
    rhs_shape_dims[i + 2] = rhs.shape().Dim(
        static_cast<Axis>(op.attributes.kernel_spatial_dimensions[i]));
    rhs_permutation_values[i + 2] = op.attributes.kernel_spatial_dimensions[i];
  }

  op.rhs_transposed_data =
      std::vector<std::byte>(rhs.NumElements() * sizeof(StorageT));
  const Shape rhs_transposed_shape(rhs_shape_dims);
  Tensor rhs_transposed{.type = TensorType{.shape = rhs_transposed_shape,
                                           .element_type = storage_type},
                        .data = op.rhs_transposed_data.data()};

  absl::InlinedVector<Axis, kMaxNumDimensions> output_permutation_values(
      output.Rank(), 0);
  output_permutation_values[op.attributes.output_batch_dimension] = 0;
  output_permutation_values[op.attributes.output_feature_dimension] = 1;
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> output_shape_dims(
      output.Rank(), 0);
  output_shape_dims[0] = output.shape().Dim(
      static_cast<Axis>(op.attributes.output_batch_dimension));
  output_shape_dims[1] = output.shape().Dim(
      static_cast<Axis>(op.attributes.output_feature_dimension));
  for (size_t i = 0; i < output.Rank() - 2; ++i) {
    output_shape_dims[i + 2] = output.shape().Dim(
        static_cast<Axis>(op.attributes.output_spatial_dimensions[i]));
    output_permutation_values[op.attributes.output_spatial_dimensions[i]] =
        i + 2;
  }

  op.output_transposed_data =
      std::vector<std::byte>(output.NumElements() * sizeof(StorageT));
  const Shape output_transposed_shape(output_shape_dims);
  Tensor output_transposed{.type = TensorType{.shape = output_transposed_shape,
                                              .element_type = storage_type},
                           .data = op.output_transposed_data.data()};
  // transpose prepare end

  // DotGeneral prepare
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> dims(
      rhs_transposed.Rank(), 0);
  size_t rhs_transposed_tensor_size = 1;
  dims[0] = 1;
  for (size_t i = 1; i < rhs_transposed.Rank(); ++i) {
    dims[i] = rhs_transposed.shape().Dim(i);
    rhs_transposed_tensor_size *= rhs_transposed.shape().Dim(i);
  }
  const Shape rhs_dot_general_shape(dims);
  op.rhs_dot_general_data =
      std::vector<std::byte>(rhs_transposed_tensor_size * sizeof(StorageT));
  Tensor rhs_dot_general{.type = TensorType{.shape = rhs_dot_general_shape,
                                            .element_type = storage_type},
                         .data = op.rhs_dot_general_data.data()};

  op.lhs_dot_general_data =
      std::vector<std::byte>(rhs_transposed_tensor_size * sizeof(StorageT));
  Tensor lhs_dot_general{.type = TensorType{.shape = rhs_dot_general_shape,
                                            .element_type = storage_type},
                         .data = op.lhs_dot_general_data.data()};

  absl::InlinedVector<Axis, kMaxNumDimensions>
      lhs_contracting_dimensions_values(lhs_transposed.Rank() - 1);
  for (size_t i = 0; i < lhs_transposed.Rank() - 1; ++i) {
    lhs_contracting_dimensions_values[i] = i + 1;
  }
  absl::Span<Axis> lhs_contracting_dimensions(
      lhs_contracting_dimensions_values);

  absl::InlinedVector<Axis, kMaxNumDimensions>
      rhs_contracting_dimensions_values(rhs_transposed.Rank() - 1);
  for (size_t i = 0; i < rhs_transposed.Rank() - 1; ++i) {
    rhs_contracting_dimensions_values[i] = i + 1;
  }
  absl::Span<Axis> rhs_contracting_dimensions(
      rhs_contracting_dimensions_values);

  std::vector<StorageT> dot_general_output_values(1);
  dot_general_output_values[0] = 0;
  op.output_dot_general_data = std::vector<std::byte>(
      reinterpret_cast<std::byte*>(dot_general_output_values.data()),
      reinterpret_cast<std::byte*>(dot_general_output_values.data() +
                                   dot_general_output_values.size()));
  const Shape dot_general_output_shape{{1}};
  Tensor output_dot_general{
      .type = TensorType{.shape = dot_general_output_shape,
                         .element_type = storage_type},
      .data = op.output_dot_general_data.data()};

  absl::Span<Axis> lhs_batching_dimensions;
  absl::Span<Axis> rhs_batching_dimensions;

  const size_t lhs_rank = lhs.Rank();
  const size_t rhs_rank = rhs.Rank();

  op.lhs_result_dims = CalculateResultDimensions(
      lhs_rank, lhs_batching_dimensions, lhs_contracting_dimensions);
  op.rhs_result_dims = CalculateResultDimensions(
      rhs_rank, rhs_batching_dimensions, rhs_contracting_dimensions);
  // Dot general prepare end

  // padding prepare
  op.pad_input_offset = 0;
  op.pad_output_offset = 0;
  int64_t lhs_padded_spatials[lhs_transposed.Rank() - 2];
  int64_t lhs_padded_tensor_size = 1;
  for (size_t i = lhs_transposed.Rank() - 1; i > 1; --i) {
    lhs_padded_spatials[i - 2] = lhs_transposed.shape().Dim(i) +
                                 (op.attributes.lhs_dilation[i - 2] - 1) *
                                     (lhs_transposed.shape().Dim(i) - 1) +
                                 padding_buffer[2 * (i - 2)] +
                                 padding_buffer[(2 * (i - 2)) + 1];
    lhs_padded_tensor_size *= lhs_padded_spatials[i - 2];
  }

  lhs_padded_tensor_size *=
      lhs_transposed.shape().Dim(0) * lhs_transposed.shape().Dim(1);
  op.lhs_padded_data =
      std::vector<std::byte>(lhs_padded_tensor_size * sizeof(StorageT));
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> lhs_padding_shape_dims(
      lhs_transposed.Rank(), 0);
  lhs_padding_shape_dims[0] = lhs_transposed.shape().Dim(0);
  lhs_padding_shape_dims[1] = lhs_transposed.shape().Dim(1);
  for (size_t i = 0; i < lhs_transposed.Rank() - 2; ++i) {
    lhs_padding_shape_dims[i + 2] =
        static_cast<int64_t>(lhs_padded_spatials[i]);
  }
  const Shape lhs_padding_shape(lhs_padding_shape_dims);
  Tensor lhs_padded{.type = TensorType{.shape = lhs_padding_shape,
                                       .element_type = storage_type},
                    .data = op.lhs_padded_data.data()};
  int64_t pad_output_shape[kMaxNumDimensions];
  std::copy(lhs_padding_shape_dims.data(),
            lhs_padding_shape_dims.data() + lhs_transposed.Rank(),
            pad_output_shape);
  int64_t edge_pad_high[kMaxNumDimensions];
  int64_t edge_pad_low[kMaxNumDimensions];
  int64_t interior_pad[kMaxNumDimensions];
  edge_pad_high[0] = edge_pad_low[0] = interior_pad[0] = 0;
  edge_pad_high[1] = edge_pad_low[1] = interior_pad[1] = 0;
  for (int64_t i = 2; i < lhs_transposed.Rank(); ++i) {
    edge_pad_low[i] = padding_buffer[2 * (i - 2)];
    edge_pad_high[i] = padding_buffer[2 * (i - 2) + 1];
    interior_pad[i] = op.attributes.lhs_dilation[i - 2] - 1;
  }
  int64_t pad_rank = lhs_transposed.Rank();
  int64_t pad_output_dimension_sizes[kMaxNumDimensions];
  pad_output_dimension_sizes[pad_rank - 1] = 1;
  op.pad_output_strides.resize(pad_rank, 0);
  op.pad_output_strides[pad_rank - 1] = interior_pad[pad_rank - 1] + 1;
  for (int64_t i = pad_rank - 2; i >= 0; --i) {
    pad_output_dimension_sizes[i] =
        pad_output_shape[i + 1] * pad_output_dimension_sizes[i + 1];
    op.pad_output_strides[i] =
        pad_output_dimension_sizes[i] * (interior_pad[i] + 1);
  }
  for (int64_t i = 0; i < pad_rank; ++i) {
    op.pad_output_offset +=
        std::max<int64_t>(edge_pad_low[i], 0) * pad_output_dimension_sizes[i];
  }
  op.pad_input_strides.resize(pad_rank, 0);
  op.pad_input_strides[pad_rank - 1] = 1;
  for (int64_t i = pad_rank - 1; i >= 1; --i) {
    op.pad_input_strides[i - 1] =
        lhs_transposed.shape().Dim(i) * op.pad_input_strides[i];
  }
  auto DivNegRoundAwayOrZero = [](int64_t num, int64_t denum) -> int64_t {
    return num < 0 ? (num - denum + 1) / denum : 0;
  };
  for (int64_t i = 0; i < pad_rank; ++i) {
    op.pad_input_shape.push_back(
        lhs_transposed.shape().Dim(i) +
        DivNegRoundAwayOrZero(edge_pad_low[i], interior_pad[i] + 1) +
        DivNegRoundAwayOrZero(edge_pad_high[i], interior_pad[i] + 1));
  }

  for (int64_t i = 0; i < pad_rank; ++i) {
    op.pad_input_offset -=
        DivNegRoundAwayOrZero(edge_pad_low[i], interior_pad[i] + 1) *
        op.pad_input_strides[i];
    if (edge_pad_low[i] < 0) {
      int64_t tmp_offset =
          ((interior_pad[i] + 1 + edge_pad_low[i]) % (interior_pad[i] + 1));
      if (tmp_offset < 0) {
        tmp_offset += interior_pad[i] + 1;
      }
      op.pad_output_offset += tmp_offset * pad_output_dimension_sizes[i];
    }
  }
  // padding prepare end

  // Split prepare
  int64_t num_splits =
      op.attributes.batch_group_count * op.attributes.feature_group_count;
  int64_t split_dimension = 0;
  for (int64_t i = 0; i < num_splits; ++i) {
    absl::InlinedVector<DimensionSize, kMaxNumDimensions> rhs_split_dims(
        rhs_transposed.Rank(), 0);
    for (size_t i = 0; i < rhs_transposed.Rank(); ++i) {
      if (i == split_dimension) {
        rhs_split_dims[i] = (rhs_transposed.shape().Dim(i) / num_splits);
      } else {
        rhs_split_dims[i] = rhs_transposed.shape().Dim(i);
      }
    }
    const Shape rhs_split_shape(rhs_split_dims);
    op.rhs_splits_data.push_back(std::vector<std::byte>(
        (rhs_transposed.NumElements() / num_splits) * sizeof(StorageT)));
    Tensor rhs_split{.type = TensorType{.shape = rhs_split_shape,
                                        .element_type = storage_type},
                     .data = op.rhs_splits_data.back().data()};
    op.rhs_splits.push_back(rhs_split);
  }

  if (op.attributes.feature_group_count > 1) {
    split_dimension = 1;
  }

  for (int64_t i = 0; i < num_splits; ++i) {
    absl::InlinedVector<DimensionSize, kMaxNumDimensions> lhs_split_dims(
        lhs_padded.Rank(), 0);
    for (size_t i = 0; i < lhs_padded.Rank(); ++i) {
      if (i == split_dimension) {
        lhs_split_dims[i] = (lhs_padded.shape().Dim(i) / num_splits);
      } else {
        lhs_split_dims[i] = lhs_padded.shape().Dim(i);
      }
    }
    const Shape lhs_split_shape(lhs_split_dims);
    op.lhs_splits_data.push_back(std::vector<std::byte>(
        (lhs_padded.NumElements() / num_splits) * sizeof(StorageT)));
    Tensor lhs_split{.type = TensorType{.shape = lhs_split_shape,
                                        .element_type = storage_type},
                     .data = op.lhs_splits_data.back().data()};
    op.lhs_splits.push_back(lhs_split);
  }
  // split prepare end

  op.lhs_permutations = std::move(lhs_permutation_values);
  op.lhs_transposed = std::move(lhs_transposed);
  op.rhs_permutations = std::move(rhs_permutation_values);
  op.rhs_transposed = std::move(rhs_transposed);
  op.output_permutations = std::move(output_permutation_values);
  op.output_transposed = std::move(output_transposed);
  op.lhs_dot_general = std::move(lhs_dot_general);
  op.rhs_dot_general = std::move(rhs_dot_general);
  op.output_dot_general = std::move(output_dot_general);
  op.lhs_padded = std::move(lhs_padded);
  op.lhs_contracting_dimensions = std::move(lhs_contracting_dimensions_values);
  op.rhs_contracting_dimensions = std::move(rhs_contracting_dimensions_values);

  return absl::OkStatus();
}

// Convolution
template <DataType storage_type>
absl::Status ConvolutionImpl(ConvolutionOp& op, size_t& output_channel,
                             const Tensor& lhs, const Tensor& rhs,
                             Tensor& output) {
  using StorageT = StorageType<storage_type>;
  using int64_t = StorageType<DataType::kSI64>;
  const StorageT* lhs_buffer = lhs.GetDataAs<storage_type>();
  const StorageT* rhs_buffer = rhs.GetDataAs<storage_type>();
  StorageT* output_buffer = output.GetDataAs<storage_type>();

  size_t rhs_tensor_size = 1;
  size_t rhs_spacial_size = 1;
  size_t output_spacial_size = 1;
  for (size_t i = 1; i < rhs.Rank(); ++i) {
    rhs_tensor_size *= rhs.shape().Dim(i);
    if (i > 1) {
      output_spacial_size *= output.shape().Dim(i);
      rhs_spacial_size *= rhs.shape().Dim(i);
    }
  }

  Tensor lhs_slice = op.lhs_dot_general;
  Tensor rhs_slice = op.rhs_dot_general;
  Tensor dot_general_output = op.output_dot_general;

  StorageT* lhs_slice_pointer = lhs_slice.GetDataAs<storage_type>();
  StorageT* rhs_slice_pointer = rhs_slice.GetDataAs<storage_type>();

  for (size_t i = 0; i < lhs.shape().Dim(0); ++i) {
    for (size_t j = 0; j < output_spacial_size; ++j) {
      int64_t output_dims[output.Rank()];
      size_t output_depth = 1;
      for (size_t m = output.Rank() - 1; m > 1; --m) {
        output_dims[m] = (j / output_depth) % output.shape().Dim(m);
        output_depth *= output.shape().Dim(m);
      }
      for (size_t k = 0; k < lhs.shape().Dim(1); ++k) {
        for (size_t l = 0; l < rhs_spacial_size; ++l) {
          int64_t filter_spacials[rhs.Rank() - 2];
          size_t depth = 1;
          for (size_t m = rhs.Rank() - 1; m > 1; --m) {
            filter_spacials[m - 2] = (l / depth) % rhs.shape().Dim(m);
            depth *= rhs.shape().Dim(m);
          }

          int64_t lhs_dims[lhs.Rank()];
          lhs_dims[0] = i;
          lhs_dims[1] = k;
          depth = 1;
          size_t lhs_index = 0;
          for (int64_t m = lhs.Rank() - 1; m >= 0; --m) {
            if (m > 1)
              lhs_dims[m] =
                  output_dims[m] * op.attributes.window_strides[m - 2] +
                  filter_spacials[m - 2] * op.attributes.rhs_dilation[m - 2];
            lhs_index += lhs_dims[m] * depth;
            depth *= lhs.shape().Dim(m);
          }

          l += k * rhs_spacial_size;
          lhs_slice_pointer[l] = lhs_buffer[lhs_index];
          l -= k * rhs_spacial_size;
        }
      }
      for (size_t k = 0; k < rhs.shape().Dim(0); ++k) {
        size_t batch_skip = k * rhs_tensor_size;
        std::copy(rhs_buffer + batch_skip,
                  rhs_buffer + batch_skip + rhs_tensor_size, rhs_slice_pointer);

        absl::Span<Axis> lhs_batching_span;
        absl::Span<Axis> rhs_batching_span;
        absl::Span<Axis> lhs_contracting_span(op.lhs_contracting_dimensions);
        absl::Span<Axis> rhs_contracting_span(op.rhs_contracting_dimensions);
        absl::Status state = DotGeneralImpl<storage_type>(
            op, lhs_slice, rhs_slice, lhs_batching_span, rhs_batching_span,
            lhs_contracting_span, rhs_contracting_span, dot_general_output);

        StorageT* dor_general_output_buffer =
            dot_general_output.GetDataAs<storage_type>();

        output_dims[0] = i;
        output_dims[1] = output_channel + k;
        output_depth = 1;
        size_t output_index = 0;
        for (int64_t m = output.Rank() - 1; m >= 0; --m) {
          output_index += output_dims[m] * output_depth;
          output_depth *= output.shape().Dim(m);
        }
        output_buffer[output_index] = dor_general_output_buffer[0];
      }
    }
  }
  output_channel += rhs.shape().Dim(0);

  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status EvaluateImpl(ConvolutionOp& op, const Tensor& lhs,
                          const Tensor& rhs, Tensor& output) {
  absl::Status transpose_status = TransposeEvaluateImpl<storage_type>(
      lhs, op.lhs_permutations, op.lhs_transposed);

  transpose_status = TransposeEvaluateImpl<storage_type>(
      rhs, op.rhs_permutations, op.rhs_transposed);

  PaddingOp<storage_type>(op, op.lhs_transposed);

  // spliting the lhs and rhs
  size_t output_channel = 0;

  if (op.attributes.feature_group_count > 1) {
    Split<storage_type>(op.lhs_padded, op.attributes.feature_group_count, 1,
                        op.lhs_splits);
    Split<storage_type>(op.rhs_transposed, op.attributes.feature_group_count, 0,
                        op.rhs_splits);

    for (int64_t i = 0; i < op.attributes.feature_group_count; ++i) {
      absl::Status status =
          ConvolutionImpl<storage_type>(op, output_channel, op.lhs_splits[i],
                                        op.rhs_splits[i], op.output_transposed);
    }
    transpose_status = TransposeEvaluateImpl<storage_type>(
        op.output_transposed, op.output_permutations, output);
    return absl::OkStatus();
  } else if (op.attributes.batch_group_count > 1) {
    Split<storage_type>(op.lhs_padded, op.attributes.batch_group_count, 0,
                        op.lhs_splits);
    Split<storage_type>(op.rhs_transposed, op.attributes.batch_group_count, 0,
                        op.rhs_splits);

    for (int64_t i = 0; i < op.attributes.batch_group_count; ++i) {
      absl::Status status =
          ConvolutionImpl<storage_type>(op, output_channel, op.lhs_splits[i],
                                        op.rhs_splits[i], op.output_transposed);
    }
    transpose_status = TransposeEvaluateImpl<storage_type>(
        op.output_transposed, op.output_permutations, output);
    return absl::OkStatus();
  }

  absl::Status status =
      ConvolutionImpl<storage_type>(op, output_channel, op.lhs_padded,
                                    op.rhs_transposed, op.output_transposed);
  transpose_status = TransposeEvaluateImpl<storage_type>(
      op.output_transposed, op.output_permutations, output);

  return absl::OkStatus();
}

ConvolutionOp Create(const ConvolutionOp::Attributes& attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  DISPATCH_INT_FLOAT(PrepareImpl, lhs.StorageType(), op, lhs, rhs, output);
}

absl::Status Evaluate(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
  DISPATCH_INT_FLOAT(EvaluateImpl, output.tensor_element_type(), op, lhs, rhs,
                     output);
}
}  // namespace shlo_ref