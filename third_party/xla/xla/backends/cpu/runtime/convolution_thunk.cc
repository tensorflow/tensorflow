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

#define EIGEN_USE_THREADS

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/backends/cpu/runtime/convolution_thunk_internal.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/executable_run_options.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime_conv2d_acl.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

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

  return absl::WrapUnique(new ConvolutionThunk(
      std::move(info), std::move(input_buffer), input_shape,
      std::move(kernel_buffer), kernel_shape, std::move(output_buffer),
      output_shape, input_batch, input_dims, input_channels, kernel_dims,
      kernel_channels, kernel_filters, output_dims, strides, padding_before,
      padding_after, base_dilation, window_dilation, feature_group_count,
      options));
}

ConvolutionThunk::ConvolutionThunk(
    Info info, BufferAllocation::Slice input_buffer, const Shape& input_shape,
    BufferAllocation::Slice kernel_buffer, const Shape& kernel_shape,
    BufferAllocation::Slice output_buffer, const Shape& output_shape,
    int64_t input_batch, const absl::InlinedVector<int64_t, 2>& input_dims,
    int64_t input_channels, const absl::InlinedVector<int64_t, 2>& kernel_dims,
    int64_t kernel_channels, int64_t kernel_filters,
    const absl::InlinedVector<int64_t, 2>& output_dims,
    const absl::InlinedVector<int64_t, 2>& strides,
    const absl::InlinedVector<int64_t, 2>& padding_before,
    const absl::InlinedVector<int64_t, 2>& padding_after,
    const absl::InlinedVector<int64_t, 2>& base_dilation,
    const absl::InlinedVector<int64_t, 2>& window_dilation,
    int64_t feature_group_count, Options options)
    : Thunk(Kind::kConvolution, std::move(info)),
      input_buffer_(input_buffer),
      input_shape_(input_shape),
      kernel_buffer_(kernel_buffer),
      kernel_shape_(kernel_shape),
      output_buffer_(output_buffer),
      output_shape_(output_shape),
      input_batch_(input_batch),
      input_dims_(input_dims),
      input_channels_(input_channels),
      kernel_dims_(kernel_dims),
      kernel_channels_(kernel_channels),
      kernel_filters_(kernel_filters),
      output_dims_(output_dims),
      strides_(strides),
      padding_before_(padding_before),
      padding_after_(padding_after),
      base_dilation_(base_dilation),
      window_dilation_(window_dilation),
      feature_group_count_(feature_group_count),
      convolution_rank_(input_dims.size()),
      options_(options) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> ConvolutionThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase input_data,
      params.buffer_allocations->GetDeviceAddress(input_buffer_));
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase kernel_data,
      params.buffer_allocations->GetDeviceAddress(kernel_buffer_));
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase output_data,
      params.buffer_allocations->GetDeviceAddress(output_buffer_));

  VLOG(3) << absl::StreamFormat(
      "Convolution: input_batch=%d input_dims=%s input_channels=%d "
      "kernel_dims=%s kernel_channels=%d kernel_filters=%d output_dims=%s "
      "strides=%s padding_before=%s padding_after=%s base_dilation=%s "
      "window_dilation=%s feature_group_count=%d",
      input_batch_, ToString(input_dims_), input_channels_,
      ToString(kernel_dims_), kernel_channels_, kernel_filters_,
      ToString(output_dims_), ToString(strides_), ToString(padding_before_),
      ToString(padding_after_), ToString(base_dilation_),
      ToString(window_dilation_), feature_group_count_);

  VLOG(3) << absl::StreamFormat("  input: %s in slice %s (%p)",
                                input_shape_.ToString(true),
                                input_buffer_.ToString(), input_data.opaque());
  VLOG(3) << absl::StreamFormat(
      "  kernel: %s in slice %s (%p)", kernel_shape_.ToString(true),
      kernel_buffer_.ToString(), kernel_data.opaque());
  VLOG(3) << absl::StreamFormat(
      "  output: %s in slice %s (%p)", output_shape_.ToString(true),
      output_buffer_.ToString(), output_data.opaque());

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
  if (convolution_rank_ == 2) {
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
      input_batch_, input_dims_.x, input_dims_.y, input_channels_,
      kernel_dims_.x, kernel_dims_.y, kernel_channels_, kernel_filters_,
      output_dims_.x, output_dims_.y, strides_.x, strides_.y, padding_before_.x,
      padding_after_.x, padding_before_.y, padding_after_.y, base_dilation_.x,
      base_dilation_.y, window_dilation_.x, window_dilation_.y,
      feature_group_count_);
}

tsl::AsyncValueRef<Thunk::ExecuteEvent>
ConvolutionThunk::HandleEigen2DConvolution(const ExecuteParams& params,
                                           se::DeviceMemoryBase input,
                                           se::DeviceMemoryBase kernel,
                                           se::DeviceMemoryBase output) {
  auto dispatch = [&](auto type_tag, const auto& eigen_device,
                      std::function<void()> done_callback = nullptr) {
    using scalar_type = decltype(type_tag);
    internal::EigenConv2D(
        eigen_device, static_cast<scalar_type*>(output.opaque()),
        static_cast<scalar_type*>(input.opaque()),
        static_cast<scalar_type*>(kernel.opaque()), input_batch_, input_dims_.x,
        input_dims_.y, input_channels_, kernel_dims_.x, kernel_dims_.y,
        kernel_channels_, kernel_filters_, output_dims_.x, output_dims_.y,
        strides_.x, strides_.y, padding_before_.x, padding_after_.x,
        padding_before_.y, padding_after_.y, base_dilation_.x, base_dilation_.y,
        window_dilation_.x, window_dilation_.y, feature_group_count_,
        std::move(done_callback), /*use_thunk_runtime=*/true);
  };

  if (options_.multi_threaded) {
    tsl::CountDownAsyncValueRef<ExecuteEvent> state(feature_group_count_);
    auto done_callback = [state]() mutable { state.CountDown(); };
    if (input_shape_.element_type() == PrimitiveType::F16) {
      dispatch(Eigen::half{}, *params.intra_op_threadpool, done_callback);
    } else {
      dispatch(float(), *params.intra_op_threadpool, done_callback);
    }
    return state.AsRef();
  } else {
    if (input_shape_.element_type() == PrimitiveType::F16) {
      dispatch(Eigen::half{}, Eigen::DefaultDevice());
    } else {
      dispatch(float{}, Eigen::DefaultDevice());
    }
    return OkExecuteEvent();
  }
}

tsl::AsyncValueRef<Thunk::ExecuteEvent>
ConvolutionThunk::HandleEigen3DConvolution(const ExecuteParams& params,
                                           se::DeviceMemoryBase input,
                                           se::DeviceMemoryBase kernel,
                                           se::DeviceMemoryBase output) {
  auto dispatch = [&](auto type_tag, const auto& eigen_device,
                      std::function<void()> done_callback = nullptr) {
    using scalar_type = decltype(type_tag);
    internal::EigenConv3D(
        eigen_device, static_cast<scalar_type*>(output.opaque()),
        static_cast<scalar_type*>(input.opaque()),
        static_cast<scalar_type*>(kernel.opaque()), input_batch_, input_dims_.x,
        input_dims_.y, input_dims_.z, input_channels_, kernel_dims_.x,
        kernel_dims_.y, kernel_dims_.z, kernel_channels_, kernel_filters_,
        output_dims_.x, output_dims_.y, output_dims_.z, strides_.x, strides_.y,
        strides_.z, padding_before_.x, padding_after_.x, padding_before_.y,
        padding_after_.y, padding_before_.z, padding_after_.z, base_dilation_.x,
        base_dilation_.y, base_dilation_.z, window_dilation_.x,
        window_dilation_.y, window_dilation_.z, feature_group_count_,
        std::move(done_callback));
  };

  if (options_.multi_threaded) {
    tsl::CountDownAsyncValueRef<ExecuteEvent> state(feature_group_count_);
    auto done_callback = [state]() mutable { state.CountDown(); };
    if (input_shape_.element_type() == PrimitiveType::F16) {
      dispatch(Eigen::half{}, *params.intra_op_threadpool, done_callback);
    } else {
      dispatch(float{}, *params.intra_op_threadpool, done_callback);
    }
    return state.AsRef();
  } else {
    if (input_shape_.element_type() == PrimitiveType::F16) {
      dispatch(Eigen::half{}, Eigen::DefaultDevice());
    } else {
      dispatch(float{}, Eigen::DefaultDevice());
    }
    return OkExecuteEvent();
  }
}

ConvolutionThunk::Dims::Dims(const absl::InlinedVector<int64_t, 2>& dims)
    : x(dims[0]), y(dims[1]), z(dims.size() == 3 ? dims[2] : 0) {}

}  // namespace xla::cpu
