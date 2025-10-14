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

#include "xla/backends/gpu/runtime/sdc_thunk.h"

#include <cstdint>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/cuda/sdc_log.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/sdc_xor_checksum_kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

namespace se = stream_executor;

absl::Status SdcThunk::Initialize(const InitializeParams& params) {
  if (params.executor->GetPlatform()->id() != se::cuda::kCudaPlatformId) {
    VLOG(1) << "[SDC LOG] Not supported on non-CUDA platforms, skipping";
    return absl::OkStatus();
  }
  if (!params.executor->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastPascal()) {
    VLOG(1) << "[SDC LOG] Not supported on CUDA architectures older than "
               "Pascal due to missing atomic fetch_add with system scope, "
               "skipping";
    return absl::OkStatus();
  }

  se::gpu::GpuKernelRegistry registry =
      se::gpu::GpuKernelRegistry::GetGlobalRegistry();
  TF_ASSIGN_OR_RETURN(
      kernel_,
      registry.LoadKernel<se::gpu::SdcXorChecksumKernel>(params.executor));

  VLOG(1) << "[SDC LOG] SDC kernel loaded";
  return absl::OkStatus();
}

absl::Status SdcThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::StreamExecutor* executor = params.stream->parent();
  if (!kernel_.has_value()) {
    // Initialize didn't load the kernel. This can happen when we're running on
    // an unsupported platform.
    VLOG(1) << "[SDC LOG] SDC kernel not loaded, skipping";
    return absl::OkStatus();
  }

  VLOG(1) << "[SDC LOG] SdcThunk::ExecuteOnStream";

  const se::ThreadDim thread_dim(
      executor->GetDeviceDescription().threads_per_block_limit(), 1, 1);

  se::DeviceMemory<uint8_t> log_ptr(
      params.buffer_allocations->GetDeviceAddress(log_slice_));
  se::cuda::SdcLog sdc_log =
      se::cuda::SdcLog::FromDeviceMemoryUnchecked(log_ptr);

  for (const auto& [entry_id, buffer] : buffers_) {
    se::DeviceMemory<uint8_t> device_buffer(
        params.buffer_allocations->GetDeviceAddress(buffer));

    TF_RETURN_IF_ERROR(
        kernel_->Launch(thread_dim, se::BlockDim(1, 1, 1), params.stream,
                        entry_id, device_buffer, device_buffer.size(),
                        sdc_log.GetDeviceHeader(), sdc_log.GetDeviceEntries()));
  }

  return absl::OkStatus();
}

std::string SdcThunk::ToString(int indent) const {
  std::string result;
  absl::StrAppend(&result, ", buffers = ", buffers_.size());
  for (const auto& [buffer_id, buffer] : buffers_) {
    absl::StrAppend(&result, "\n", std::string(indent + 2, ' '),
                    "buffer_id: ", buffer_id, ", buffer: ", buffer.ToString());
  }
  return result;
}

}  // namespace xla::gpu
