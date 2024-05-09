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

  // DotGeneral prepare
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> dims(rhs.Rank(), 0);
  size_t rhs_tensor_size = 1;
  dims[0] = 1;
  for (size_t i = 1; i < rhs.Rank(); ++i) {
    dims[i] = rhs.shape().Dim(i);
    rhs_tensor_size *= rhs.shape().Dim(i);
  }
  const Shape rhs_dot_general_shape(dims);
  op.rhs_dot_general_data =
      std::vector<std::byte>(rhs_tensor_size * sizeof(StorageT));
  Tensor rhs_dot_general{.type = TensorType{.shape = rhs_dot_general_shape,
                                            .element_type = storage_type},
                         .data = op.rhs_dot_general_data.data()};

  op.lhs_dot_general_data =
      std::vector<std::byte>(rhs_tensor_size * sizeof(StorageT));
  Tensor lhs_dot_general{.type = TensorType{.shape = rhs_dot_general_shape,
                                            .element_type = storage_type},
                         .data = op.lhs_dot_general_data.data()};

  absl::InlinedVector<Axis, kMaxNumDimensions>
      lhs_contracting_dimensions_values(lhs.Rank() - 1);
  for (size_t i = 0; i < lhs.Rank() - 1; ++i) {
    lhs_contracting_dimensions_values[i] = i + 1;
  }
  absl::Span<Axis> lhs_contracting_dimensions(
      lhs_contracting_dimensions_values);

  absl::InlinedVector<Axis, kMaxNumDimensions>
      rhs_contracting_dimensions_values(rhs.Rank() - 1);
  for (size_t i = 0; i < rhs.Rank() - 1; ++i) {
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

  op.lhs_dot_general = std::move(lhs_dot_general);
  op.rhs_dot_general = std::move(rhs_dot_general);
  op.output_dot_general = std::move(output_dot_general);
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
  size_t output_channel = 0;
  absl::Status status =
      ConvolutionImpl<storage_type>(op, output_channel, lhs, rhs, output);
  return absl::OkStatus();
}

ConvolutionOp Create(const ConvolutionOp::Attributes& attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  DISPATCH_INT_FLOAT(PrepareImpl, lhs.StorageType(), op, lhs, rhs, output);
  return absl::OkStatus();
}

absl::Status Evaluate(ConvolutionOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
  DISPATCH_INT_FLOAT(EvaluateImpl, output.tensor_element_type(), op, lhs, rhs,
                     output);
}
}  // namespace shlo_ref