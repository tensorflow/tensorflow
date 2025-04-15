/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/convolution_lib.h"

#include <cstddef>
#include <cstdint>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/runtime/buffer_use.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

absl::InlinedVector<BufferUse, 4> ConvolutionBufferUses(
    const ConvolutionSlices& slices) {
  return {BufferUse::Read(slices.input_buffer),
          BufferUse::Read(slices.kernel_buffer),
          BufferUse::Write(slices.output_buffer)};
}

ConvolutionCanonicalDims::Dims::Dims(absl::Span<const int64_t> dims)
    : rank(dims.size()), x(dims[0]), y(dims[1]), z(rank == 3 ? dims[2] : 0) {}

static size_t GetConvolutionRank(const Shape& input_shape) {
  // Convolution rank is the number of spatial dimensions. Besides spatial
  // dimensions, input shape contains two other dimensions (batch size and the
  // number of channels).
  return input_shape.dimensions().size() - 2;
}

static absl::Status ValidateConvolutionShapes(
    const Shape& input_shape, const Shape& kernel_shape,
    const Shape& output_shape, const ConvolutionDimensionNumbers& dnums) {
  // Convolution rank.
  int64_t convolution_rank = GetConvolutionRank(input_shape);
  if (convolution_rank > 3 || convolution_rank < 1) {
    return InvalidArgument("ConvolutionThunk: Incorrect convolution rank (%d)",
                           convolution_rank);
  }

  // Rank of input, kernel and output buffers.
  if (input_shape.dimensions().size() != kernel_shape.dimensions().size() ||
      input_shape.dimensions().size() != output_shape.dimensions().size()) {
    return InvalidArgument(
        "ConvolutionThunk: Buffer ranks mismatch. Input rank (%d) vs kernel "
        "rank (%d) vs output rank (%d)",
        input_shape.dimensions().size(), kernel_shape.dimensions().size(),
        output_shape.dimensions().size());
  }

  // Batch size.
  auto input_batch = input_shape.dimensions(dnums.input_batch_dimension());
  auto output_batch = output_shape.dimensions(dnums.output_batch_dimension());
  if (input_batch != output_batch) {
    return InvalidArgument(
        "ConvolutionThunk: Batch sizes mismatch. Input batch (%d) vs output "
        "batch (%d)",
        input_batch, output_batch);
  }

  // Output channels / kernel filters.
  auto kernel_filters =
      kernel_shape.dimensions(dnums.kernel_output_feature_dimension());
  auto output_channels =
      output_shape.dimensions(dnums.output_feature_dimension());
  if (kernel_filters != output_channels) {
    return InvalidArgument(
        "ConvolutionThunk: Output channels mismatch. Kernel filters count (%d) "
        "should be the same as output channels count (%d)",
        kernel_filters, output_channels);
  }

  return absl::OkStatus();
}

bool IsSupportedType(PrimitiveType primitive_type) {
  return primitive_type == PrimitiveType::F16 ||
         primitive_type == PrimitiveType::F32;
}

absl::StatusOr<ConvolutionCanonicalDims> GetConvolutionCanonicalDims(
    const ConvolutionSlices& slices, const ConvolutionDimensionNumbers& dnums,
    const Window& window, int64_t feature_group_count) {
  TF_RETURN_IF_ERROR(ValidateConvolutionShapes(
      slices.input_shape, slices.kernel_shape, slices.output_shape, dnums));

  auto primitive_type = slices.input_shape.element_type();
  if (!IsSupportedType(primitive_type)) {
    return InvalidArgument("ConvolutionThunk: Unsupported element type (%s)",
                           PrimitiveType_Name(primitive_type));
  }

  absl::InlinedVector<int64_t, 2> input_dims;
  absl::InlinedVector<int64_t, 2> kernel_dims;
  absl::InlinedVector<int64_t, 2> output_dims;

  // We lower 1D convolutions into calls to the same Eigen function as 2D
  // convolutions, except that we pretend that the 1D convolution is really
  // a 2D convolution with the missing dimension set to 1.  We also adjust
  // the padding, dilation parameters as needed.

  int64_t convolution_rank = GetConvolutionRank(slices.input_shape);
  if (convolution_rank == 1) {
    input_dims.push_back(1);
    kernel_dims.push_back(1);
    output_dims.push_back(1);
  }

  // Input tensor.
  int64_t input_batch =
      slices.input_shape.dimensions(dnums.input_batch_dimension());
  for (int d : dnums.input_spatial_dimensions()) {
    input_dims.push_back(slices.input_shape.dimensions(d));
  }
  int64_t input_channels =
      slices.input_shape.dimensions(dnums.input_feature_dimension());

  // Kernel tensor.
  for (int d : dnums.kernel_spatial_dimensions()) {
    kernel_dims.push_back(slices.kernel_shape.dimensions(d));
  }
  int64_t kernel_channels =
      slices.kernel_shape.dimensions(dnums.kernel_input_feature_dimension());
  int64_t kernel_filters =
      slices.kernel_shape.dimensions(dnums.kernel_output_feature_dimension());

  // Output tensor.
  for (int d : dnums.output_spatial_dimensions()) {
    output_dims.push_back(slices.output_shape.dimensions(d));
  }

  // Extract the window stride for the convolution.
  absl::InlinedVector<int64_t, 2> strides;
  absl::InlinedVector<int64_t, 2> padding_before;
  absl::InlinedVector<int64_t, 2> padding_after;
  absl::InlinedVector<int64_t, 2> base_dilation;
  absl::InlinedVector<int64_t, 2> window_dilation;

  if (convolution_rank == 1) {
    strides.push_back(1);
    padding_before.push_back(0);
    padding_after.push_back(0);
    base_dilation.push_back(1);
    window_dilation.push_back(1);
  }

  for (const auto& d : window.dimensions()) {
    strides.push_back(d.stride());
    padding_before.push_back(d.padding_low());
    padding_after.push_back(d.padding_high());
    base_dilation.push_back(d.base_dilation());
    window_dilation.push_back(d.window_dilation());
  }

  auto valid_num_dims = [](absl::Span<const int64_t> xs) {
    return xs.size() >= 2 && xs.size() <= 3;
  };

  TF_RET_CHECK(valid_num_dims(input_dims));
  TF_RET_CHECK(valid_num_dims(kernel_dims));
  TF_RET_CHECK(valid_num_dims(output_dims));
  TF_RET_CHECK(valid_num_dims(strides));
  TF_RET_CHECK(valid_num_dims(padding_before));
  TF_RET_CHECK(valid_num_dims(padding_after));
  TF_RET_CHECK(valid_num_dims(base_dilation));
  TF_RET_CHECK(valid_num_dims(window_dilation));

  using Dims = ConvolutionCanonicalDims::Dims;
  return ConvolutionCanonicalDims{
      input_batch,         Dims(input_dims),    input_channels,
      Dims(kernel_dims),   kernel_channels,     kernel_filters,
      Dims(output_dims),   Dims(strides),       Dims(padding_before),
      Dims(padding_after), Dims(base_dilation), Dims(window_dilation),
      feature_group_count};
}

}  // namespace xla::cpu
