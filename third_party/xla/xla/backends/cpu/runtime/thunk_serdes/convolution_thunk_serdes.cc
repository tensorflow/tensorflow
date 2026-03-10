/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/thunk_serdes/convolution_thunk_serdes.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/convolution_dims.h"
#include "xla/backends/cpu/runtime/convolution_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes_utils.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

namespace xla::cpu {
namespace {

absl::Status ConvolutionThunkToProto(const Thunk& thunk, ThunkProto& proto) {
  const auto& convolution_thunk =
      tsl::down_cast<const ConvolutionThunk&>(thunk);

  ConvolutionThunkProto* convolution_thunk_proto =
      proto.mutable_convolution_thunk();

  const std::string dnums_as_str =
      convolution_thunk.dnums().SerializeAsString();
  convolution_thunk_proto->mutable_dimension_numbers()->ParseFromString(
      dnums_as_str);

  const std::string window_as_str =
      convolution_thunk.window().SerializeAsString();
  convolution_thunk_proto->mutable_window()->ParseFromString(window_as_str);

  convolution_thunk_proto->set_feature_group_count(
      convolution_thunk.feature_group_count());

  const ConvolutionSlices& convolution_slices =
      convolution_thunk.convolution_slices();

  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      convolution_slices.input_buffer, convolution_slices.input_shape,
      convolution_thunk_proto->mutable_input_buffer_shape()));

  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      convolution_slices.output_buffer, convolution_slices.output_shape,
      convolution_thunk_proto->mutable_output_buffer_shape()));

  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      convolution_slices.kernel_buffer, convolution_slices.kernel_shape,
      convolution_thunk_proto->mutable_kernel_buffer_shape()));

  convolution_thunk_proto->mutable_options()->set_multi_threaded(
      convolution_thunk.options().multi_threaded);

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Thunk>> ConvolutionThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  // Parse options.
  ConvolutionThunk::Options options;
  options.multi_threaded = proto.convolution_thunk().options().multi_threaded();

  // Dimension numbers.
  ConvolutionDimensionNumbers dnums =
      proto.convolution_thunk().dimension_numbers();

  // Window.
  Window window = proto.convolution_thunk().window();

  // Feature group count.
  int64_t feature_group_count = proto.convolution_thunk().feature_group_count();

  TF_ASSIGN_OR_RETURN(
      auto input_slice_shape,
      DeserializeSliceShapeFromProto(
          proto.convolution_thunk().input_buffer_shape(), buffer_allocations));
  TF_ASSIGN_OR_RETURN(
      auto kernel_slice_shape,
      DeserializeSliceShapeFromProto(
          proto.convolution_thunk().kernel_buffer_shape(), buffer_allocations));
  TF_ASSIGN_OR_RETURN(
      auto output_slice_shape,
      DeserializeSliceShapeFromProto(
          proto.convolution_thunk().output_buffer_shape(), buffer_allocations));

  const auto& [input_buffer, input_shape] = input_slice_shape;
  const auto& [kernel_buffer, kernel_shape] = kernel_slice_shape;
  const auto& [output_buffer, output_shape] = output_slice_shape;

  return ConvolutionThunk::Create(
      std::move(info), std::move(options), std::move(input_buffer), input_shape,
      std::move(kernel_buffer), kernel_shape, std::move(output_buffer),
      output_shape, dnums, window, feature_group_count);
}

}  // namespace

void RegisterConvolutionThunkSerDes() {
  CHECK_OK(ThunkSerDesRegistry::Get().Register(Thunk::Kind::kConvolution,
                                               ConvolutionThunkToProto,
                                               ConvolutionThunkFromProto));
}

// Static initialization to register the serdes.
static bool convolution_thunk_serdes_registered = []() {
  RegisterConvolutionThunkSerDes();
  return true;
}();

}  // namespace xla::cpu
