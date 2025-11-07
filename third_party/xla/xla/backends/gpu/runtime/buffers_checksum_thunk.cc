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

#include "xla/backends/gpu/runtime/buffers_checksum_thunk.h"

#include <cstdint>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_entry_metadata_store.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/buffer_debug_log.h"
#include "xla/stream_executor/gpu/buffer_debug_xor_checksum_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

namespace se = stream_executor;

absl::Status BuffersDebugChecksumThunk::Initialize(
    const InitializeParams& params) {
  if (params.executor->GetPlatform()->id() != se::cuda::kCudaPlatformId) {
    VLOG(1)
        << "Buffer checksumming not supported on non-CUDA platforms, skipping";
    return absl::OkStatus();
  }
  if (!params.executor->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastPascal()) {
    VLOG(1)
        << "Buffer checksumming not supported on CUDA architectures older than "
           "Pascal due to missing atomic fetch_add with system scope, skipping";
    return absl::OkStatus();
  }

  se::gpu::GpuKernelRegistry registry =
      se::gpu::GpuKernelRegistry::GetGlobalRegistry();
  TF_ASSIGN_OR_RETURN(
      kernel_, registry.LoadKernel<se::gpu::BufferDebugXorChecksumKernel>(
                   params.executor));

  VLOG(1) << "Checksum kernel loaded";
  return absl::OkStatus();
}

absl::Status BuffersDebugChecksumThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::StreamExecutor* executor = params.stream->parent();
  if (!kernel_.has_value()) {
    // Initialize didn't load the kernel. This can happen when we're running on
    // an unsupported platform.
    VLOG(1) << "Checksum kernel not loaded, skipping";
    return absl::OkStatus();
  }

  VLOG(1) << "BuffersDebugChecksumThunk::ExecuteOnStream";
  const uint32_t execution_id = execution_count_.fetch_add(1);

  const se::ThreadDim thread_dim(
      executor->GetDeviceDescription().threads_per_block_limit(), 1, 1);

  se::DeviceMemory<uint8_t> log_ptr(
      params.buffer_allocations->GetDeviceAddress(log_slice_));
  se::gpu::BufferDebugLog buffer_debug_log =
      se::gpu::BufferDebugLog::FromDeviceMemoryUnchecked(log_ptr);

  for (const auto& [buffer_idx, buffer] : checked_thunk_buffers_) {
    BufferDebugLogEntryMetadataStore::Metadata metadata{
        /*thunk_id*/ checked_thunk_id_,
        /*buffer_idx*/ buffer_idx,
        /*execution_id*/ execution_id,
        /*is_input*/ runs_before_checked_thunk_,
        /*check_type*/ BufferDebugLogEntryProto::CHECK_TYPE_CHECKSUM,
    };
    const BufferDebugLogEntryId log_entry_id =
        metadata_store_->AssignId(metadata);

    se::DeviceMemory<uint8_t> device_buffer(
        params.buffer_allocations->GetDeviceAddress(buffer));

    TF_RETURN_IF_ERROR(kernel_->Launch(
        thread_dim, se::BlockDim(1, 1, 1), params.stream, log_entry_id,
        device_buffer, device_buffer.size(), buffer_debug_log.GetDeviceHeader(),
        buffer_debug_log.GetDeviceEntries<BufferDebugLogEntry>()));
  }

  return absl::OkStatus();
}

std::string BuffersDebugChecksumThunk::ToString(int indent) const {
  std::string result;
  absl::StrAppend(&result, ", buffers = ", checked_thunk_buffers_.size());
  for (const auto& [buffer_idx, buffer] : checked_thunk_buffers_) {
    absl::StrAppend(&result, "\n", std::string(indent + 2, ' '),
                    "buffer_idx: ", buffer_idx,
                    ", buffer: ", buffer.ToString());
  }
  return result;
}

}  // namespace xla::gpu
