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

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_entry_metadata_store.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
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

BuffersDebugFloatCheckThunk::BuffersDebugFloatCheckThunk(
    ThunkInfo info, const ThunkInfo& checked_thunk_info,
    BufferAllocation::Slice log_slice, BufferAllocation::Slice tmp_slice,
    absl::flat_hash_map<size_t, BufferAllocation::Slice> checked_thunk_buffers,
    std::shared_ptr<BufferDebugLogEntryMetadataStore> metadata_store)
    : Thunk(Thunk::Kind::kBuffersDebugFloatCheck, std::move(info)),
      log_slice_(log_slice),
      tmp_slice_(tmp_slice),
      checked_thunk_info_(checked_thunk_info),
      checked_thunk_buffers_(std::move(checked_thunk_buffers)),
      metadata_store_(std::move(metadata_store)) {
  absl::erase_if(
      checked_thunk_buffers_,
      [this](const std::pair<size_t, BufferAllocation::Slice>& pair) {
        if (pair.second.size() == 0) {
          VLOG(1) << "Buffer " << pair.first << " in thunk "
                  << checked_thunk_info_.thunk_id
                  << " has zero size, skipping float check";
          return true;
        }
        return false;
      });
}
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
          auto kernel_f32,
          registry.LoadKernel<se::gpu::BufferDebugFloatCheckF32Kernel>(
              params.executor));
      TF_ASSIGN_OR_RETURN(
          auto kernel_bf16,
          registry.LoadKernel<se::gpu::BufferDebugFloatCheckBf16Kernel>(
              params.executor));
      TF_ASSIGN_OR_RETURN(
          auto kernel_f64,
          registry.LoadKernel<se::gpu::BufferDebugFloatCheckF64Kernel>(
              params.executor));
      TF_ASSIGN_OR_RETURN(
          auto kernel_reduce,
          registry.LoadKernel<
              se::gpu::BufferDebugAppendReducedFloatCheckResultsKernel>(
              params.executor));
      kernels_[params.executor] = std::make_unique<Kernels>(
          Kernels{std::move(kernel_f32), std::move(kernel_bf16),
                  std::move(kernel_f64), std::move(kernel_reduce)});
      VLOG(1) << "NanCount kernels loaded";
    }
  }

  return absl::OkStatus();
}

template <typename T>
se::BlockDim GetBlockDimForBuffer(se::Stream* stream,
                                  se::DeviceAddress<T> buffer,
                                  int64_t max_blocks) {
  const int64_t num_elements = buffer.size() / sizeof(T);
  const se::DeviceDescription& desc = stream->parent()->GetDeviceDescription();
  const int64_t num_blocks =
      std::min(xla::CeilOfRatio(num_elements, desc.threads_per_block_limit()),
               max_blocks);
  return se::BlockDim(num_blocks);
}

template <typename T>
constexpr const char* FloatTypeString() {
  if constexpr (std::is_same_v<T, float>) {
    return "F32";
  } else if constexpr (std::is_same_v<T, Eigen::bfloat16>) {
    return "BF16";
  } else if constexpr (std::is_same_v<T, double>) {
    return "F64";
  } else {
    static_assert(false, "Unsupported type");
  }
}

template <typename T, typename MapKernel, typename ReduceAppendKernel>
absl::Status CheckFloatsAndLog(
    se::Stream* stream, const BufferDebugLogEntryId entry_id,
    se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry>& buffer_debug_log,
    se::DeviceAddressBase device_buffer,
    se::DeviceAddress<xla::gpu::FloatCheckResult> tmp_ptr,
    MapKernel& map_kernel, ReduceAppendKernel& reduce_append_kernel) {
  VLOG(1) << FloatTypeString<T>() << " buffer detected with id: " << entry_id
          << " and size: " << device_buffer.size();

  se::DeviceAddress<T> buffer(device_buffer);
  // The kernels assume 1024 threads per block.
  const se::ThreadDim thread_dim(1024);
  const se::BlockDim block_dim =
      GetBlockDimForBuffer<T>(stream, buffer, tmp_ptr.ElementCount());
  const size_t num_blocks = block_dim.x * block_dim.y * block_dim.z;

  TF_RETURN_IF_ERROR(map_kernel.Launch(thread_dim, block_dim, stream, buffer,
                                       buffer.ElementCount(), tmp_ptr,
                                       tmp_ptr.ElementCount()));
  // Operations on the same stream perform in sequence, so at this point the
  // results of the previous FloatCheck operation are available.
  TF_RETURN_IF_ERROR(reduce_append_kernel.Launch(
      thread_dim, se::BlockDim(1, 1, 1), stream, tmp_ptr,
      std::min(tmp_ptr.ElementCount(), num_blocks), entry_id,
      buffer_debug_log.GetDeviceHeader(), buffer_debug_log.GetDeviceEntries()));

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

  se::DeviceAddress<xla::gpu::FloatCheckResult> tmp_ptr(
      params.buffer_allocations->GetDeviceAddress(tmp_slice_));
  const size_t tmp_size_elements =
      tmp_slice_.size() / sizeof(xla::gpu::FloatCheckResult);
  CHECK_GT(tmp_size_elements, 0)
      << "tmp_slice_ is too small to hold any results, this should have been "
         "caught during initialization";

  se::DeviceAddress<uint8_t> log_ptr(
      params.buffer_allocations->GetDeviceAddress(log_slice_));
  se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry> buffer_debug_log =
      se::gpu::BufferDebugLog<
          BufferDebugFloatCheckEntry>::FromDeviceAddressUnchecked(log_ptr);
  const uint32_t execution_id = execution_count_.fetch_add(1);

  for (const auto& [buffer_idx, buffer] : checked_thunk_buffers_) {
    BufferDebugLogEntryMetadataStore::Metadata metadata{
        checked_thunk_info_.thunk_id,
        buffer_idx,
        execution_id,
        /*is_input=*/false,
        BufferDebugLogEntryProto::CHECK_TYPE_FLOAT_CHECKS,
        checked_thunk_info_.profile_annotation,
    };
    const BufferDebugLogEntryId entry_id = metadata_store_->AssignId(metadata);

    PrimitiveType buffer_type = buffer.element_type();
    se::DeviceAddressBase device_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer);

    if (buffer_type == PrimitiveType::F32) {
      TF_RETURN_IF_ERROR(CheckFloatsAndLog<float>(
          params.stream, entry_id, buffer_debug_log, device_buffer, tmp_ptr,
          kernels->f32, kernels->reduce));
    } else if (buffer_type == PrimitiveType::BF16) {
      TF_RETURN_IF_ERROR(CheckFloatsAndLog<Eigen::bfloat16>(
          params.stream, entry_id, buffer_debug_log, device_buffer, tmp_ptr,
          kernels->bf16, kernels->reduce));
    } else if (buffer_type == PrimitiveType::F64) {
      TF_RETURN_IF_ERROR(CheckFloatsAndLog<double>(
          params.stream, entry_id, buffer_debug_log, device_buffer, tmp_ptr,
          kernels->f64, kernels->reduce));
    } else {
      VLOG(1) << "Unsupported primitive type for float checking: "
              << PrimitiveType_Name(buffer_type);
      continue;
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
