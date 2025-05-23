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

#include "xla/backends/cpu/runtime/xnnpack/xnn_convolution_thunk.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xnnpack.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/convolution_lib.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_fusion_thunk.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_interop.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

absl::StatusOr<xnn_subgraph_t> XnnConvolutionThunk::BuildConvolutionSubgraph(
    absl::Span<const Argument> arguments, absl::Span<const Result> results,
    absl::Span<const se::DeviceMemoryBase> arguments_buffers) {
  xnn_subgraph_t subgraph = nullptr;
  XNN_RETURN_IF_ERROR(xnn_create_subgraph(/*external_value_ids=*/3,
                                          /*flags=*/0, &subgraph));

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  uint32_t out_id = XNN_INVALID_VALUE_ID;

  auto dims = [](absl::Span<const int64_t> dims) -> std::vector<size_t> {
    return {dims.begin(), dims.end()};
  };

  VLOG(3) << absl::StreamFormat(
      "Create XNNPACK convolution: input_shape=%s kernel_shape=%s out_shape=%s",
      convolution_slices_.input_shape.ToString(true),
      convolution_slices_.kernel_shape.ToString(true),
      convolution_slices_.output_shape.ToString(true));

  std::vector<size_t> input_dims =
      dims(convolution_slices_.input_shape.dimensions());
  std::vector<size_t> kernel_dims =
      dims(convolution_slices_.kernel_shape.dimensions());
  std::vector<size_t> out_dims =
      dims(convolution_slices_.output_shape.dimensions());

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(),
      nullptr,
      /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, kernel_dims.size(), kernel_dims.data(),
      /*data=*/arguments_buffers[1].opaque(),
      /*external_id=*/1, /*flags=*/0, &kernel_id));

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, out_dims.size(), out_dims.data(), nullptr,
      /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id));

  auto& ds = convolution_canonical_dims_;
  XNN_RETURN_IF_ERROR(xnn_define_convolution_2d(
      subgraph,  //
      /*input_padding_top=*/ds.padding_before.x,
      /*input_padding_right=*/ds.padding_before.y,
      /*input_padding_bottom=*/ds.padding_after.x,
      /*input_padding_left=*/ds.padding_after.y,
      /*kernel_height=*/ds.kernel_dims.x,
      /*kernel_width=*/ds.kernel_dims.y,
      /*subsampling_height=*/ds.strides.x,
      /*subsampling_width=*/ds.strides.y,
      /*dilation_height=*/ds.base_dilation.x,
      /*dilation_width=*/ds.base_dilation.y,
      /*groups=*/ds.feature_group_count,
      /*group_input_channels=*/ds.input_channels,
      /*group_output_channels=*/ds.kernel_filters,
      /*output_min=*/std::numeric_limits<float>::lowest(),
      /*output_max=*/std::numeric_limits<float>::max(), input_id, kernel_id,
      /*bias_id=*/XNN_INVALID_VALUE_ID, out_id,
      /*flags=*/XNN_FLAG_TENSORFLOW_SAME_PADDING));

  return subgraph;
}

absl::StatusOr<std::unique_ptr<XnnConvolutionThunk>>
XnnConvolutionThunk::Create(
    Options options, Info info, BufferAllocation::Slice input_buffer,
    const Shape& input_shape, BufferAllocation::Slice kernel_buffer,
    const Shape& kernel_shape, BufferAllocation::Slice output_buffer,
    const Shape& output_shape, const ConvolutionDimensionNumbers& dnums,
    const Window& window, int64_t feature_group_count) {
  TF_RETURN_IF_ERROR(InitializeXnnPack());

  if (dnums.kernel_input_feature_dimension() != 3 ||
      dnums.kernel_output_feature_dimension() != 0) {
    return InvalidArgument(
        "XNNPACK convolution expects kernel (filter) in OHWI format");
  }

  ConvolutionSlices slices = {input_buffer, input_shape,   kernel_buffer,
                              kernel_shape, output_buffer, output_shape};

  TF_ASSIGN_OR_RETURN(
      ConvolutionCanonicalDims canonical_dims,
      GetConvolutionCanonicalDims(slices, dnums, window, feature_group_count));

  return absl::WrapUnique(new XnnConvolutionThunk(
      std::move(options), std::move(info), std::move(slices),
      std::move(canonical_dims), dnums, window));
}

static std::vector<XnnFusionThunk::Argument> ConvolutionArguments(
    const ConvolutionSlices& slices) {
  return {XnnFusionThunk::Argument{slices.input_buffer, slices.input_shape},
          XnnFusionThunk::Argument{slices.kernel_buffer, slices.kernel_shape}};
}

static std::vector<XnnFusionThunk::Result> ConvolutionResults(
    const ConvolutionSlices& slices) {
  return {XnnFusionThunk::Result{slices.output_buffer, slices.output_shape}};
}

XnnConvolutionThunk::XnnConvolutionThunk(
    Options options, Info info, ConvolutionSlices convolution_slices,
    ConvolutionCanonicalDims convolution_canonical_dims,
    ConvolutionDimensionNumbers dnums, Window window)
    : XnnFusionThunk(XnnFusionKind::kConvolution, std::move(options),
                     std::move(info), ConvolutionArguments(convolution_slices),
                     ConvolutionResults(convolution_slices),
                     CapturingBuilder(std::bind(
                         &XnnConvolutionThunk::BuildConvolutionSubgraph, this,
                         std::placeholders::_1, std::placeholders::_2,
                         std::placeholders::_3)),
                     /*captured_arguments=*/{1}),
      convolution_slices_(std::move(convolution_slices)),
      convolution_canonical_dims_(std::move(convolution_canonical_dims)),
      dnums_(std::move(dnums)),
      window_(std::move(window)) {}

std::string XnnConvolutionThunk::fusion_kind() const { return "convolution"; }

std::string XnnConvolutionThunk::fusion_description() const {
  return absl::StrFormat("convolution_rank=%d",
                         convolution_canonical_dims_.convolution_rank());
}

std::vector<std::string> XnnConvolutionThunk::fusion_details() const {
  return {absl::StrCat(convolution_canonical_dims_)};
}

std::string XnnConvolutionThunk::argument_name(size_t index) const {
  return index == 0 ? "input" : "kernel";
}

std::string XnnConvolutionThunk::result_name(size_t index) const {
  return "out";
}

}  // namespace xla::cpu
