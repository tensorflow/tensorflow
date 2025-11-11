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

#include "xla/backends/gpu/runtime/buffers_float_check_thunk.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_entry_metadata_store.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/buffer_debug_float_check_kernel.h"
#include "xla/stream_executor/gpu/buffer_debug_log.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"

namespace xla::gpu {

namespace se = stream_executor;

absl::Status BuffersDebugFloatCheckThunk::Initialize(
    const InitializeParams& params) {
  if (params.executor->GetPlatform()->id() != se::cuda::kCudaPlatformId) {
    VLOG(1) << "Buffer float checking not supported on non-CUDA platforms, "
               "skipping";
    return absl::OkStatus();
  }
  if (!params.executor->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastPascal()) {
    VLOG(1)
        << "Buffer float checking not supported on CUDA architectures older "
           "than "
           "Pascal due to missing atomic fetch_add with system scope, skipping";
    return absl::OkStatus();
  }

  {
    absl::MutexLock lock(kernels_mutex_);
    if (!kernels_.contains(params.executor)) {
      se::gpu::GpuKernelRegistry registry =
          se::gpu::GpuKernelRegistry::GetGlobalRegistry();
      TF_ASSIGN_OR_RETURN(
          auto kernel_f32,
          registry.LoadKernel<se::gpu::BufferDebugFloatCheckF32Kernel>(
              params.executor));
      TF_ASSIGN_OR_RETURN(
          auto kernel_bf16,
          registry.LoadKernel<se::gpu::BufferDebugFloatCheckBf16Kernel>(
              params.executor));
      kernels_[params.executor] = std::make_unique<Kernels>(
          Kernels{std::move(kernel_f32), std::move(kernel_bf16)});
    }
  }

  VLOG(1) << "FloatCheck kernel loaded";
  return absl::OkStatus();
}

absl::Status BuffersDebugFloatCheckThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::StreamExecutor* executor = params.stream->parent();

  Kernels* kernels = nullptr;
  {
    absl::MutexLock lock(kernels_mutex_);
    auto kernel_it = kernels_.find(executor);
    if (kernel_it == kernels_.end()) {
      // Initialize didn't load the kernel. This can happen when we're running
      // on an unsupported platform.
      VLOG(1) << "FloatCheck kernels not loaded on device "
              << executor->device_ordinal() << ", skipping";
      return absl::OkStatus();
    }
    kernels = kernel_it->second.get();
  }

  VLOG(1) << "BuffersDebugFloatCheckThunk::ExecuteOnStream";

  const se::ThreadDim thread_dim(
      executor->GetDeviceDescription().threads_per_block_limit(), 1, 1);

  se::DeviceMemory<uint8_t> log_ptr(
      params.buffer_allocations->GetDeviceAddress(log_slice_));
  se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry> buffer_debug_log =
      se::gpu::BufferDebugLog<
          BufferDebugFloatCheckEntry>::FromDeviceMemoryUnchecked(log_ptr);
  const uint32_t execution_id = execution_count_.fetch_add(1);

  for (const auto& [buffer_idx, buffer] : checked_thunk_buffers_) {
    BufferDebugLogEntryMetadataStore::Metadata metadata{
        checked_thunk_id_,
        buffer_idx,
        execution_id,
        /*is_input=*/runs_before_checked_thunk_,
        BufferDebugLogEntryProto::CHECK_TYPE_FLOAT_CHECKS,
    };
    const BufferDebugLogEntryId entry_id = metadata_store_->AssignId(metadata);

    PrimitiveType buffer_type = buffer.element_type();
    se::DeviceMemoryBase device_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer);
    if (buffer_type == PrimitiveType::F32) {
      VLOG(1) << "F32 buffer detected with id: " << entry_id
              << " and size: " << device_buffer.size();
      se::DeviceMemory<float> f32_buffer(device_buffer);
      TF_RETURN_IF_ERROR(kernels->f32.Launch(
          thread_dim, se::BlockDim(1, 1, 1), params.stream, entry_id,
          f32_buffer, f32_buffer.size(), buffer_debug_log.GetDeviceHeader(),
          buffer_debug_log.GetDeviceEntries()));
    } else if (buffer_type == PrimitiveType::BF16) {
      VLOG(1) << "BF16 buffer detected with id: " << entry_id
              << " and size: " << device_buffer.size();
      se::DeviceMemory<Eigen::bfloat16> bf16_buffer(device_buffer);
      TF_RETURN_IF_ERROR(kernels->bf16.Launch(
          thread_dim, se::BlockDim(1, 1, 1), params.stream, entry_id,
          bf16_buffer, bf16_buffer.size(), buffer_debug_log.GetDeviceHeader(),
          buffer_debug_log.GetDeviceEntries()));
    } else {
      VLOG(1) << "Unsupported primitive type for float checking: "
              << PrimitiveType_Name(buffer_type);
    }
  }

  return absl::OkStatus();
}

std::string BuffersDebugFloatCheckThunk::ToString(int indent) const {
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
