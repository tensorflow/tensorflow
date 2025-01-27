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

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/convolution_lib.h"
#include "xla/backends/cpu/runtime/convolution_thunk_internal.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/executable_run_options.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime_conv2d_acl.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/traceme.h"

#define EIGEN_USE_THREADS
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

auto GetConvolutionRank(const Shape& input_shape) {
  // Convolution rank is the number of spatial dimensions. Besides spatial
  // dimensions, input shape contains two other dimensions (batch size and the
  // number of channels).
  return input_shape.dimensions_size() - 2;
}

absl::Status ValidateShapes(const Shape& input_shape, const Shape& kernel_shape,
                            const Shape& output_shape,
                            const ConvolutionDimensionNumbers& dnums) {
  // Convolution rank.
  int64_t convolution_rank = GetConvolutionRank(input_shape);
  if (convolution_rank > 3 || convolution_rank < 1) {
    return InvalidArgument("ConvolutionThunk: Incorrect convolution rank (%d)",
                           convolution_rank);
  }

  // Rank of input, kernel and output buffers.
  if (input_shape.dimensions_size() != kernel_shape.dimensions_size() ||
      input_shape.dimensions_size() != output_shape.dimensions_size()) {
    return InvalidArgument(
        "ConvolutionThunk: Buffer ranks mismatch. Input rank (%d) vs kernel "
        "rank (%d) vs output rank (%d)",
        input_shape.dimensions_size(), kernel_shape.dimensions_size(),
        output_shape.dimensions_size());
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

bool CanUseACL(const ConvolutionThunk::Options& options,
               PrimitiveType primitive_type, int64_t convolution_rank) {
  return options.use_acl && primitive_type == PrimitiveType::F32 &&
         convolution_rank == 2;
}

auto MakeRunOptions(const Eigen::ThreadPoolDevice* threadpool) {
  ExecutableRunOptions run_options;
  run_options.set_intra_op_thread_pool(threadpool);
  return run_options;
}

}  // namespace

absl::StatusOr<std::unique_ptr<ConvolutionThunk>> ConvolutionThunk::Create(
    Info info, Options options, BufferAllocation::Slice input_buffer,
    const Shape& input_shape, BufferAllocation::Slice kernel_buffer,
    const Shape& kernel_shape, BufferAllocation::Slice output_buffer,
    const Shape& output_shape, const ConvolutionDimensionNumbers& dnums,
    const Window& window, int64_t feature_group_count) {
  TF_RETURN_IF_ERROR(
      ValidateShapes(input_shape, kernel_shape, output_shape, dnums));
  auto primitive_type = input_shape.element_type();
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

  int64_t convolution_rank = GetConvolutionRank(input_shape);
  if (convolution_rank == 1) {
    input_dims.push_back(1);
    kernel_dims.push_back(1);
    output_dims.push_back(1);
  }

  // Turn off ACL if not supported for given primitive type and convolution
  // rank.
  options.use_acl = CanUseACL(options, primitive_type, convolution_rank);

  // Input tensor.
  int64_t input_batch = input_shape.dimensions(dnums.input_batch_dimension());
  for (int d : dnums.input_spatial_dimensions()) {
    input_dims.push_back(input_shape.dimensions(d));
  }
  int64_t input_channels =
      input_shape.dimensions(dnums.input_feature_dimension());

  // Kernel tensor.
  for (int d : dnums.kernel_spatial_dimensions()) {
    kernel_dims.push_back(kernel_shape.dimensions(d));
  }
  int64_t kernel_channels =
      kernel_shape.dimensions(dnums.kernel_input_feature_dimension());
  int64_t kernel_filters =
      kernel_shape.dimensions(dnums.kernel_output_feature_dimension());

  // Output tensor.
  for (int d : dnums.output_spatial_dimensions()) {
    output_dims.push_back(output_shape.dimensions(d));
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
  TF_RET_CHECK(valid_num_dims(input_dims)) << input_dims.size();
  TF_RET_CHECK(valid_num_dims(kernel_dims));
  TF_RET_CHECK(valid_num_dims(output_dims));
  TF_RET_CHECK(valid_num_dims(strides));
  TF_RET_CHECK(valid_num_dims(padding_before));
  TF_RET_CHECK(valid_num_dims(padding_after));
  TF_RET_CHECK(valid_num_dims(base_dilation));
  TF_RET_CHECK(valid_num_dims(window_dilation));

  ConvolutionSlices convolution_slices = {input_buffer,  input_shape,
                                          kernel_buffer, kernel_shape,
                                          output_buffer, output_shape};

  using Dims = ConvolutionCanonicalDims::Dims;
  ConvolutionCanonicalDims convolution_canonical_dims = {
      input_batch,         Dims(input_dims),    input_channels,
      Dims(kernel_dims),   kernel_channels,     kernel_filters,
      Dims(output_dims),   Dims(strides),       Dims(padding_before),
      Dims(padding_after), Dims(base_dilation), Dims(window_dilation),
      feature_group_count};

  return absl::WrapUnique(
      new ConvolutionThunk(std::move(info), options, convolution_slices,
                           convolution_canonical_dims, dnums, window));
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
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase input_data,
                      params.buffer_allocations->GetDeviceAddress(
                          convolution_slices_.input_buffer));
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase kernel_data,
                      params.buffer_allocations->GetDeviceAddress(
                          convolution_slices_.kernel_buffer));
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase output_data,
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

  if (options_.multi_threaded && params.intra_op_threadpool == nullptr) {
    return Internal(
        "Intra-op threadpool must be provided for ConvolutionThunk in "
        "multi-threaded mode.");
  }

  if (options_.use_acl) {
    HandleACLConvolution(params, input_data, kernel_data, output_data);
    return OkExecuteEvent();
  }

  // Eigen convolution
  if (convolution_canonical_dims_.convolution_rank() == 2) {
    return HandleEigen2DConvolution(params, input_data, kernel_data,
                                    output_data);
  } else {
    return HandleEigen3DConvolution(params, input_data, kernel_data,
                                    output_data);
  }
}

void ConvolutionThunk::HandleACLConvolution(const ExecuteParams& params,
                                            se::DeviceMemoryBase input,
                                            se::DeviceMemoryBase kernel,
                                            se::DeviceMemoryBase output) {
  // NOTE: This is the basic support for ACL. Performance was not
  // benchmarked and is likely not good, the design could be improved
  // (e.g. creating run_options is a hack).
  auto run_options = MakeRunOptions(params.intra_op_threadpool);
  __xla_cpu_runtime_ACLConv2DF32(
      &run_options, static_cast<float*>(output.opaque()),
      static_cast<float*>(input.opaque()), static_cast<float*>(kernel.opaque()),
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
      convolution_canonical_dims_.feature_group_count);
}

tsl::AsyncValueRef<Thunk::ExecuteEvent>
ConvolutionThunk::HandleEigen2DConvolution(const ExecuteParams& params,
                                           se::DeviceMemoryBase input,
                                           se::DeviceMemoryBase kernel,
                                           se::DeviceMemoryBase output) {
  auto dispatch = [&](auto type_tag, const auto& eigen_device,
                      tsl::CountDownAsyncValueRef<ExecuteEvent> count_down) {
    using scalar_type = decltype(type_tag);
    internal::EigenConv2D(
        eigen_device, static_cast<scalar_type*>(output.opaque()),
        static_cast<scalar_type*>(input.opaque()),
        static_cast<scalar_type*>(kernel.opaque()),
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
        convolution_canonical_dims_.feature_group_count, std::move(count_down),
        /*use_thunk_runtime=*/true);
  };

  PrimitiveType input_type = convolution_slices_.input_shape.element_type();

  // Execute convolution in the intra-op threadpool.
  if (options_.multi_threaded) {
    tsl::CountDownAsyncValueRef<ExecuteEvent> count_down(
        convolution_canonical_dims_.feature_group_count);
    auto execute_event = count_down.AsRef();

    if (input_type == PrimitiveType::F16) {
      dispatch(Eigen::half{}, *params.intra_op_threadpool,
               std::move(count_down));
    } else {
      dispatch(float{}, *params.intra_op_threadpool, std::move(count_down));
    }
    return execute_event;
  }

  // Execute convolution in the caller thread.
  if (input_type == PrimitiveType::F16) {
    dispatch(Eigen::half{}, Eigen::DefaultDevice(), /*count_down=*/{});
  } else {
    dispatch(float{}, Eigen::DefaultDevice(), /*count_down=*/{});
  }

  return OkExecuteEvent();
}

tsl::AsyncValueRef<Thunk::ExecuteEvent>
ConvolutionThunk::HandleEigen3DConvolution(const ExecuteParams& params,
                                           se::DeviceMemoryBase input,
                                           se::DeviceMemoryBase kernel,
                                           se::DeviceMemoryBase output) {
  auto dispatch = [&](auto type_tag, const auto& eigen_device,
                      tsl::CountDownAsyncValueRef<ExecuteEvent> count_down) {
    using scalar_type = decltype(type_tag);
    internal::EigenConv3D(
        eigen_device, static_cast<scalar_type*>(output.opaque()),
        static_cast<scalar_type*>(input.opaque()),
        static_cast<scalar_type*>(kernel.opaque()),
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
        convolution_canonical_dims_.feature_group_count, std::move(count_down));
  };

  PrimitiveType input_type = convolution_slices_.input_shape.element_type();

  // Execute convolution in the intra-op threadpool.
  if (options_.multi_threaded) {
    tsl::CountDownAsyncValueRef<ExecuteEvent> count_down(
        convolution_canonical_dims_.feature_group_count);
    auto execute_event = count_down.AsRef();

    if (input_type == PrimitiveType::F16) {
      dispatch(Eigen::half{}, *params.intra_op_threadpool,
               std::move(count_down));
    } else {
      dispatch(float{}, *params.intra_op_threadpool, std::move(count_down));
    }
    return execute_event;
  }

  // Execute convolution in the caller thread.
  if (input_type == PrimitiveType::F16) {
    dispatch(Eigen::half{}, Eigen::DefaultDevice(), /*count_down=*/{});
  } else {
    dispatch(float{}, Eigen::DefaultDevice(), /*count_down=*/{});
  }
  return OkExecuteEvent();
}

}  // namespace xla::cpu
