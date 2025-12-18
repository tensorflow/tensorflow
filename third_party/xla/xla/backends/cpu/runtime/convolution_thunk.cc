/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/convolution_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/cpu/runtime/convolution_dims.h"
#include "xla/backends/cpu/runtime/convolution_lib.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/executable_run_options.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

#define EIGEN_USE_THREADS
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

auto MakeRunOptions(const Eigen::ThreadPoolDevice* threadpool) {
  ExecutableRunOptions run_options;
  run_options.set_intra_op_thread_pool(threadpool);
  return run_options;
}

absl::StatusOr<std::unique_ptr<ConvolutionThunk>> ConvolutionThunk::Create(
    Info info, Options options, BufferAllocation::Slice input_buffer,
    const Shape& input_shape, BufferAllocation::Slice kernel_buffer,
    const Shape& kernel_shape, BufferAllocation::Slice output_buffer,
    const Shape& output_shape, const ConvolutionDimensionNumbers& dnums,
    const Window& window, int64_t feature_group_count) {
  ConvolutionSlices slices = {input_buffer, input_shape,   kernel_buffer,
                              kernel_shape, output_buffer, output_shape};

  TF_ASSIGN_OR_RETURN(
      ConvolutionCanonicalDims canonical_dims,
      GetConvolutionCanonicalDims(slices, dnums, window, feature_group_count));

  return absl::WrapUnique(new ConvolutionThunk(std::move(info), options, slices,
                                               canonical_dims, dnums, window));
}

ConvolutionThunk::ConvolutionThunk(
    Info info, Options options, ConvolutionSlices convolution_slices,
    ConvolutionCanonicalDims convolution_canonical_dims,
    ConvolutionDimensionNumbers dnums, Window window)
    : Thunk(Kind::kConvolution, std::move(info)),
      options_(options),
      convolution_slices_(convolution_slices),
      convolution_canonical_dims_(convolution_canonical_dims),
      dnums_(std::move(dnums)),
      window_(std::move(window)) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> ConvolutionThunk::Execute(
    const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(se::DeviceAddressBase input_data,
                      params.buffer_allocations->GetDeviceAddress(
                          convolution_slices_.input_buffer));
  TF_ASSIGN_OR_RETURN(se::DeviceAddressBase kernel_data,
                      params.buffer_allocations->GetDeviceAddress(
                          convolution_slices_.kernel_buffer));
  TF_ASSIGN_OR_RETURN(se::DeviceAddressBase output_data,
                      params.buffer_allocations->GetDeviceAddress(
                          convolution_slices_.output_buffer));

  VLOG(3) << absl::StreamFormat("Convolution: %v", convolution_canonical_dims_);

  VLOG(3) << absl::StreamFormat("  input: %s in slice %s (%p)",
                                convolution_slices_.input_shape.ToString(true),
                                convolution_slices_.input_buffer.ToString(),
                                input_data.opaque());
  VLOG(3) << absl::StreamFormat("  kernel: %s in slice %s (%p)",
                                convolution_slices_.kernel_shape.ToString(true),
                                convolution_slices_.kernel_buffer.ToString(),
                                kernel_data.opaque());
  VLOG(3) << absl::StreamFormat("  output: %s in slice %s (%p)",
                                convolution_slices_.output_shape.ToString(true),
                                convolution_slices_.output_buffer.ToString(),
                                output_data.opaque());

  // Eigen convolution
  if (convolution_canonical_dims_.convolution_rank() == 2) {
    return HandleEigen2DConvolution(params, input_data, kernel_data,
                                    output_data);
  } else {
    return HandleEigen3DConvolution(params, input_data, kernel_data,
                                    output_data);
  }
}

tsl::AsyncValueRef<Thunk::ExecuteEvent>
ConvolutionThunk::HandleEigen2DConvolution(const ExecuteParams& params,
                                           se::DeviceAddressBase input,
                                           se::DeviceAddressBase kernel,
                                           se::DeviceAddressBase output) {
  auto dispatch = [&](auto type_tag) {
    auto execute_event = tsl::MakeConstructedAsyncValueRef<ExecuteEvent>();

    using ScalarType = decltype(type_tag);
    internal::EigenConv2D(
        params.intra_op_threadpool, static_cast<ScalarType*>(output.opaque()),
        static_cast<const ScalarType*>(input.opaque()),
        static_cast<const ScalarType*>(kernel.opaque()),
        convolution_canonical_dims_.input_batch,
        convolution_canonical_dims_.input_dims.x,
        convolution_canonical_dims_.input_dims.y,
        convolution_canonical_dims_.input_channels,
        convolution_canonical_dims_.kernel_dims.x,
        convolution_canonical_dims_.kernel_dims.y,
        convolution_canonical_dims_.kernel_channels,
        convolution_canonical_dims_.kernel_filters,
        convolution_canonical_dims_.output_dims.x,
        convolution_canonical_dims_.output_dims.y,
        convolution_canonical_dims_.strides.x,
        convolution_canonical_dims_.strides.y,
        convolution_canonical_dims_.padding_before.x,
        convolution_canonical_dims_.padding_after.x,
        convolution_canonical_dims_.padding_before.y,
        convolution_canonical_dims_.padding_after.y,
        convolution_canonical_dims_.base_dilation.x,
        convolution_canonical_dims_.base_dilation.y,
        convolution_canonical_dims_.window_dilation.x,
        convolution_canonical_dims_.window_dilation.y,
        convolution_canonical_dims_.feature_group_count,
        [execute_event] { execute_event.SetStateConcrete(); });

    return execute_event;
  };

  PrimitiveType input_type = convolution_slices_.input_shape.element_type();

  switch (input_type) {
    case PrimitiveType::F16:
      return dispatch(Eigen::half{});
    case PrimitiveType::F32:
      return dispatch(float{});
    default:
      return Internal("Unsupported Conv2D input type: %s",
                      primitive_util::LowercasePrimitiveTypeName(input_type));
  }
}

tsl::AsyncValueRef<Thunk::ExecuteEvent>
ConvolutionThunk::HandleEigen3DConvolution(const ExecuteParams& params,
                                           se::DeviceAddressBase input,
                                           se::DeviceAddressBase kernel,
                                           se::DeviceAddressBase output) {
  auto dispatch = [&](auto type_tag) {
    auto execute_event = tsl::MakeConstructedAsyncValueRef<ExecuteEvent>();

    using ScalarType = decltype(type_tag);
    internal::EigenConv3D(
        params.intra_op_threadpool, static_cast<ScalarType*>(output.opaque()),
        static_cast<const ScalarType*>(input.opaque()),
        static_cast<const ScalarType*>(kernel.opaque()),
        convolution_canonical_dims_.input_batch,
        convolution_canonical_dims_.input_dims.x,
        convolution_canonical_dims_.input_dims.y,
        convolution_canonical_dims_.input_dims.z,
        convolution_canonical_dims_.input_channels,
        convolution_canonical_dims_.kernel_dims.x,
        convolution_canonical_dims_.kernel_dims.y,
        convolution_canonical_dims_.kernel_dims.z,
        convolution_canonical_dims_.kernel_channels,
        convolution_canonical_dims_.kernel_filters,
        convolution_canonical_dims_.output_dims.x,
        convolution_canonical_dims_.output_dims.y,
        convolution_canonical_dims_.output_dims.z,
        convolution_canonical_dims_.strides.x,
        convolution_canonical_dims_.strides.y,
        convolution_canonical_dims_.strides.z,
        convolution_canonical_dims_.padding_before.x,
        convolution_canonical_dims_.padding_after.x,
        convolution_canonical_dims_.padding_before.y,
        convolution_canonical_dims_.padding_after.y,
        convolution_canonical_dims_.padding_before.z,
        convolution_canonical_dims_.padding_after.z,
        convolution_canonical_dims_.base_dilation.x,
        convolution_canonical_dims_.base_dilation.y,
        convolution_canonical_dims_.base_dilation.z,
        convolution_canonical_dims_.window_dilation.x,
        convolution_canonical_dims_.window_dilation.y,
        convolution_canonical_dims_.window_dilation.z,
        convolution_canonical_dims_.feature_group_count,
        [execute_event] { execute_event.SetStateConcrete(); });

    return execute_event;
  };

  PrimitiveType input_type = convolution_slices_.input_shape.element_type();

  switch (input_type) {
    case PrimitiveType::F16:
      return dispatch(Eigen::half{});
    case PrimitiveType::F32:
      return dispatch(float{});
    default:
      return Internal("Unsupported Conv3D input type: %s",
                      primitive_util::LowercasePrimitiveTypeName(input_type));
  }
}

}  // namespace xla::cpu
