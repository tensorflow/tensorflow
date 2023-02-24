/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/runtime/conv_reorder.h"

#include <optional>
#include <utility>

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::xla::runtime::CustomCall;
using ::xla::runtime::FlatMemrefView;
using ::xla::runtime::StridedMemrefView;

se::dnn::FilterDescriptor GetFilterDescriptor(
    absl::Span<const int64_t> filter_dims) {
  se::dnn::FilterDescriptor filter_desc(2);
  filter_desc.set_layout(se::dnn::FilterLayout::kOutputInputYX32);
  filter_desc.set_output_feature_map_count(filter_dims[0]);
  filter_desc.set_input_feature_map_count(filter_dims[1]);
  filter_desc.set_input_filter_height(filter_dims[2]);
  filter_desc.set_input_filter_width(filter_dims[3]);
  return filter_desc;
}

absl::Status ConvReorderFilterImpl(
    const ServiceExecutableRunOptions* run_options,
    StridedMemrefView input_view, StridedMemrefView output_view,
    absl::Span<const int64_t> filter_dims) {
  auto input = se::DeviceMemory<int8_t>(GetDeviceAddress(input_view));
  auto output = se::DeviceMemory<int8_t>(GetDeviceAddress(output_view));

  auto executed = run_options->stream()->CudnnReorderConvolutionFilterAndBias(
      GetFilterDescriptor(filter_dims), input, &output, std::nullopt,
      std::nullopt);
  if (!executed.ok()) return ToAbslStatus(executed);

  return absl::OkStatus();
}

absl::Status ConvReorderFilterAndBiasImpl(
    const ServiceExecutableRunOptions* run_options,
    StridedMemrefView filter_input_view, FlatMemrefView bias_input_view,
    StridedMemrefView filter_output_view, FlatMemrefView bias_output_view,
    absl::Span<const int64_t> filter_dims) {
  auto filter_input =
      se::DeviceMemory<int8_t>(GetDeviceAddress(filter_input_view));
  auto filter_output =
      se::DeviceMemory<int8_t>(GetDeviceAddress(filter_output_view));
  auto bias_input = se::DeviceMemory<float>(GetDeviceAddress(bias_input_view));
  auto bias_output =
      se::DeviceMemory<float>(GetDeviceAddress(bias_output_view));

  auto executed = run_options->stream()->CudnnReorderConvolutionFilterAndBias(
      GetFilterDescriptor(filter_dims), filter_input, &filter_output,
      std::make_optional(bias_input), std::make_optional(bias_output));
  if (!executed.ok()) return ToAbslStatus(executed);

  return absl::OkStatus();
}

}  // namespace

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    ConvReorderFilter, FunctionWrapper<ConvReorderFilterImpl>(), checks,
    CustomCall::Bind("xla.gpu.conv.reorder.filter")
        .UserData<const ServiceExecutableRunOptions*>()
        .Arg<StridedMemrefView>()  // filter_input
        .Arg<StridedMemrefView>()  // filter_output
        .Attr<absl::Span<const int64_t>>("filter_dims"));

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    ConvReorderFilterAndBias, FunctionWrapper<ConvReorderFilterAndBiasImpl>(),
    checks,
    CustomCall::Bind("xla.gpu.conv.reorder.filter_and_bias")
        .UserData<const ServiceExecutableRunOptions*>()
        .Arg<StridedMemrefView>()  // filter_input
        .Arg<FlatMemrefView>()     // bias_input
        .Arg<StridedMemrefView>()  // filter_output
        .Arg<FlatMemrefView>()     // bias_output
        .Attr<absl::Span<const int64_t>>("filter_dims"));

void RegisterConvReorderCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.conv.reorder.filter", ConvReorderFilter);
  registry.Register("xla.gpu.conv.reorder.filter_and_bias",
                    ConvReorderFilterAndBias);
}

}  // namespace gpu
}  // namespace xla
