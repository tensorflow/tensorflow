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

#include "xla/backends/gpu/runtime/buffers_nan_count_thunk.h"

#include <cstdint>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/buffer_debug_log.h"
#include "xla/stream_executor/gpu/buffer_debug_nan_count_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"

namespace xla::gpu {

namespace se = stream_executor;

absl::Status BuffersDebugNanCountThunk::Initialize(
    const InitializeParams& params) {
  if (params.executor->GetPlatform()->id() != se::cuda::kCudaPlatformId) {
    VLOG(1)
        << "Buffer nan-counting not supported on non-CUDA platforms, skipping";
    return absl::OkStatus();
  }
  if (!params.executor->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastPascal()) {
    VLOG(1)
        << "Buffer nan-counting not supported on CUDA architectures older than "
           "Pascal due to missing atomic fetch_add with system scope, skipping";
    return absl::OkStatus();
  }

  se::gpu::GpuKernelRegistry registry =
      se::gpu::GpuKernelRegistry::GetGlobalRegistry();
  TF_ASSIGN_OR_RETURN(
      kernel_f32_, registry.LoadKernel<se::gpu::BufferDebugNanCountF32Kernel>(
                       params.executor));
  TF_ASSIGN_OR_RETURN(
      kernel_bf16_, registry.LoadKernel<se::gpu::BufferDebugNanCountBf16Kernel>(
                        params.executor));

  VLOG(1) << "NanCount kernel loaded";
  return absl::OkStatus();
}

absl::Status BuffersDebugNanCountThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::StreamExecutor* executor = params.stream->parent();
  if (!kernel_f32_.has_value()) {
    // Initialize didn't load the kernel. This can happen when we're running on
    // an unsupported platform.
    VLOG(1) << "NanCount kernel not loaded, skipping";
    return absl::OkStatus();
  }

  VLOG(1) << "BuffersDebugNanCountThunk::ExecuteOnStream";

  const se::ThreadDim thread_dim(
      executor->GetDeviceDescription().threads_per_block_limit(), 1, 1);

  se::DeviceMemory<uint8_t> log_ptr(
      params.buffer_allocations->GetDeviceAddress(log_slice_));
  se::gpu::BufferDebugLog buffer_debug_log =
      se::gpu::BufferDebugLog::FromDeviceMemoryUnchecked(log_ptr);

  for (const auto& [entry_id, buffer] : buffers_) {
    PrimitiveType buffer_type = buffer.element_type();
    se::DeviceMemoryBase device_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer);
    if (buffer_type == PrimitiveType::F32) {
      VLOG(1) << "F32 buffer detected with id: " << entry_id
              << " and size: " << device_buffer.size();
      se::DeviceMemory<float> f32_buffer(device_buffer);
      TF_RETURN_IF_ERROR(kernel_f32_->Launch(
          thread_dim, se::BlockDim(1, 1, 1), params.stream, entry_id,
          f32_buffer, f32_buffer.size(), buffer_debug_log.GetDeviceHeader(),
          buffer_debug_log.GetDeviceEntries()));
    } else if (buffer_type == PrimitiveType::BF16) {
      VLOG(1) << "BF16 buffer detected with id: " << entry_id
              << " and size: " << device_buffer.size();
      se::DeviceMemory<Eigen::bfloat16> bf16_buffer(device_buffer);
      TF_RETURN_IF_ERROR(kernel_bf16_->Launch(
          thread_dim, se::BlockDim(1, 1, 1), params.stream, entry_id,
          bf16_buffer, bf16_buffer.size(), buffer_debug_log.GetDeviceHeader(),
          buffer_debug_log.GetDeviceEntries()));
    } else {
      VLOG(1) << "Unsupported primitive type for NaN counting: "
              << PrimitiveType_Name(buffer_type);
    }
  }

  return absl::OkStatus();
}

std::string BuffersDebugNanCountThunk::ToString(int indent) const {
  std::string result;
  absl::StrAppend(&result, ", buffers = ", buffers_.size());
  for (const auto& [buffer_id, buffer] : buffers_) {
    absl::StrAppend(&result, "\n", std::string(indent + 2, ' '),
                    "buffer_id: ", buffer_id, ", buffer: ", buffer.ToString());
  }
  return result;
}

}  // namespace xla::gpu
