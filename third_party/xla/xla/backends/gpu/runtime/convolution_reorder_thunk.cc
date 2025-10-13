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

#include "xla/backends/gpu/runtime/convolution_reorder_thunk.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/convolution_filter_thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

static se::dnn::FilterDescriptor CreateFilterDescriptor(
    const ConvolutionFilterDimensions& filter_dimensions) {
  se::dnn::FilterDescriptor filter_desc(/*ndims=*/2);
  filter_desc.set_layout(se::dnn::FilterLayout::kOutputInputYX32);
  filter_desc.set_output_feature_map_count(
      filter_dimensions.output_feature_map_count());
  filter_desc.set_input_feature_map_count(
      filter_dimensions.input_feature_map_count());
  filter_desc.set_input_filter_height(filter_dimensions.input_filter_height());
  filter_desc.set_input_filter_width(filter_dimensions.input_filter_width());
  return filter_desc;
}

ConvolutionReorderThunk::ConvolutionReorderThunk(
    ThunkInfo thunk_info, ConvolutionFilterDimensions filter_dimensions,
    absl::InlinedVector<BufferAllocation::Slice, 2> operand_slices,
    absl::InlinedVector<BufferAllocation::Slice, 2> result_slices)
    : Thunk(Kind::kConvolutionReorder, thunk_info),
      filter_descriptor_(CreateFilterDescriptor(filter_dimensions)),
      operand_buffers_(operand_slices),
      result_buffers_(result_slices) {}

absl::Status ConvolutionReorderThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  bool has_bias = operand_buffers_.size() > 1;
  CHECK_EQ(operand_buffers_.size(), result_buffers_.size());

  const auto& buffer_allocations = *params.buffer_allocations;

  auto filter_input = se::DeviceMemory<int8_t>(
      buffer_allocations.GetDeviceAddress(operand_buffers_[0]));
  auto filter_output = se::DeviceMemory<int8_t>(
      buffer_allocations.GetDeviceAddress(result_buffers_[0]));
  auto bias_input =
      has_bias ? std::make_optional(se::DeviceMemory<float>(
                     buffer_allocations.GetDeviceAddress(operand_buffers_[1])))
               : std::nullopt;
  auto bias_output =
      has_bias ? std::make_optional(se::DeviceMemory<float>(
                     buffer_allocations.GetDeviceAddress(result_buffers_[1])))
               : std::nullopt;

  auto dnn = params.stream->parent()->AsDnn();
  if (dnn == nullptr) {
    return absl::InternalError("No DNN for stream.");
  }
  return dnn->CudnnReorderConvolutionFilterAndBias(
      params.stream, filter_descriptor_, filter_input, &filter_output,
      std::move(bias_input), std::move(bias_output));
}

}  // namespace gpu
}  // namespace xla
