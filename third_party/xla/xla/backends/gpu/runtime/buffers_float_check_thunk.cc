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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_entry_metadata_store.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/buffer_debug_float_check_kernel.h"
#include "xla/stream_executor/gpu/buffer_debug_log.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/util.h"

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
           "than Pascal due to missing atomic fetch_add with system scope, "
           "skipping";
    return absl::OkStatus();
  }

  {
    absl::MutexLock lock(kernels_mutex_);
    if (!kernels_.contains(params.executor)) {
      se::gpu::GpuKernelRegistry registry =
          se::gpu::GpuKernelRegistry::GetGlobalRegistry();
      TF_ASSIGN_OR_RETURN(
          auto kernel_append,
          registry
              .LoadKernel<se::gpu::BufferDebugAppendFloatCheckResultsKernel>(
                  params.executor));
      kernels_[params.executor] =
          std::make_unique<Kernels>(Kernels{std::move(kernel_append)});
      VLOG(1) << "NanCount kernels loaded";
    }
  }

  return absl::OkStatus();
}

template <typename T>
se::BlockDim GetBlockDimForBuffer(se::Stream* stream,
                                  se::DeviceMemory<T> buffer,
                                  int64_t max_blocks) {
  const int64_t num_elements = buffer.size() / sizeof(T);
  const se::DeviceDescription& desc = stream->parent()->GetDeviceDescription();
  const int64_t num_blocks =
      std::min(xla::CeilOfRatio(num_elements, desc.threads_per_block_limit()),
               max_blocks);
  return se::BlockDim(num_blocks);
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

  se::DeviceAddress<xla::gpu::FloatCheckResult> results_ptr(
      params.buffer_allocations->GetDeviceAddress(results_slice_));
  const size_t results_size_elements =
      results_slice_.size() / sizeof(xla::gpu::FloatCheckResult);
  CHECK_GE(results_size_elements, checked_thunk_buffers_.size())
      << "results_slice_ is too small to hold results for all buffers, this "
         "should have been caught during initialization";

  se::DeviceAddress<xla::gpu::BufferDebugLogEntryId> ids_ptr(
      params.buffer_allocations->GetDeviceAddress(ids_slice_));
  const size_t ids_size_elements =
      ids_slice_.size() / sizeof(xla::gpu::BufferDebugLogEntryId);
  CHECK_GE(ids_size_elements, checked_thunk_buffers_.size())
      << "ids_slice_ is too small to hold ids for all buffers, this should "
         "have been caught during initialization";

  se::DeviceAddress<uint8_t> log_ptr(
      params.buffer_allocations->GetDeviceAddress(log_slice_));
  se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry> buffer_debug_log =
      se::gpu::BufferDebugLog<
          BufferDebugFloatCheckEntry>::FromDeviceAddressUnchecked(log_ptr);
  const uint32_t execution_id = execution_count_.fetch_add(1);

  std::vector<BufferDebugLogEntryId> entry_ids;

  for (const auto& [buffer_idx, buffer] : checked_thunk_buffers_) {
    BufferDebugLogEntryMetadataStore::Metadata metadata{
        checked_thunk_info_.thunk_id,
        buffer_idx,
        execution_id,
        /*is_input=*/false,
        BufferDebugLogEntryProto::CHECK_TYPE_FLOAT_CHECKS,
        checked_thunk_info_.profile_annotation,
    };
    se::DeviceAddress<FloatCheckResult> result_ptr =
        results_ptr.GetSlice(entry_ids.size(), 1);
    const BufferDebugLogEntryId entry_id = metadata_store_->AssignId(metadata);
    entry_ids.push_back(entry_id);

    PrimitiveType buffer_type = buffer.element_type();
    se::DeviceAddressBase device_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer);
    if (buffer_type == PrimitiveType::F32) {
      VLOG(1) << "F32 buffer detected with id: " << entry_id
              << " and size: " << device_buffer.size();
      se::DeviceAddress<float> f32_buffer(device_buffer);
      TF_RETURN_IF_ERROR(
          se::gpu::CheckFloats<float>(f32_buffer, result_ptr, params.stream));
    } else if (buffer_type == PrimitiveType::BF16) {
      VLOG(1) << "BF16 buffer detected with id: " << entry_id
              << " and size: " << device_buffer.size();
      se::DeviceAddress<Eigen::bfloat16> bf16_buffer(device_buffer);
      TF_RETURN_IF_ERROR(se::gpu::CheckFloats<Eigen::bfloat16>(
          bf16_buffer, result_ptr, params.stream));
    } else {
      VLOG(1) << "Unsupported primitive type for float checking: "
              << PrimitiveType_Name(buffer_type);
      continue;
    }
  }

  TF_RETURN_IF_ERROR(
      params.stream->Memcpy(&ids_ptr, entry_ids.data(),
                            entry_ids.size() * sizeof(BufferDebugLogEntryId)));
  // Operations on the same stream perform in sequence, so at this point the
  // results of the previous FloatCheck operations are available.
  TF_RETURN_IF_ERROR(kernels->append.Launch(
      se::ThreadDim(1, 1, 1), se::BlockDim(1, 1, 1), params.stream, results_ptr,
      ids_ptr, entry_ids.size(), buffer_debug_log.GetDeviceHeader(),
      buffer_debug_log.GetDeviceEntries()));

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
