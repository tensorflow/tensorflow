/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/stream_executor/tpu/tpu_executor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/allocator_stats.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/tpu/tpu_event.h"
#include "xla/stream_executor/tpu/tpu_stream.h"
#include "xla/tpu/c_api_conversions.h"
#include "xla/tpu/c_api_decl.h"
#include "xla/tpu/status_helper.h"
#include "xla/tpu/tpu_executor_api.h"
#include "xla/tpu/tpu_topology.h"
#include "xla/tsl/c/tsl_status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace stream_executor {
namespace tpu {

TpuExecutor::~TpuExecutor() { ExecutorApiFn()->TpuExecutor_FreeFn(executor_); }

absl::Status TpuExecutor::Init() {
  StatusHelper status;
  ExecutorApiFn()->TpuExecutor_InitFn(executor_, status.c_status);
  return status.status();
}

bool TpuExecutor::SynchronizeAllActivity() {
  return ExecutorApiFn()->TpuExecutor_SynchronizeAllActivityFn(executor_);
}

tensorflow::tpu::TpuCoreLocationExternal TpuExecutor::GetCoreLocationExternal()
    const {
  return tensorflow::tpu::TpuCoreLocationExternal(
      ExecutorApiFn()->TpuExecutor_GetCoreLocationFn(executor_));
}

void TpuExecutor::DeallocateStream(Stream* stream) {
  ExecutorApiFn()->TpuExecutor_DeallocateStreamFn(executor_,
                                                  get_stream(stream));
  absl::MutexLock lock(&tpu_platform().mutex());
  stream_map().erase(stream);
}

absl::StatusOr<std::unique_ptr<Stream>> TpuExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  SE_Stream* tpu_stream = ExecutorApiFn()->TpuStream_NewFn(executor_);
  if (tpu_stream == nullptr) {
    return absl::InternalError("Failed to create TPU stream");
  }
  auto stream = std::make_unique<tensorflow::tpu::TpuStream>(
      tpu_stream, this, executor_, &tpu_platform());
  {
    absl::MutexLock lock(&tpu_platform().mutex());
    stream_map()[stream.get()] = tpu_stream;
  }
  return std::move(stream);
}

absl::StatusOr<std::unique_ptr<Event>> TpuExecutor::CreateEvent() {
  SE_Event* se_event = ExecutorApiFn()->TpuEvent_NewFn(executor_);
  auto tpu_event = std::make_unique<TpuEvent>(se_event, platform_);
  tpu_platform().InsertEvent(tpu_event.get(), se_event);

  StatusHelper status;
  ExecutorApiFn()->TpuExecutor_AllocateEventFn(executor_, se_event,
                                               status.c_status);
  RETURN_IF_ERROR(status.status());

  return std::move(tpu_event);
}

DeviceAddressBase TpuExecutor::Allocate(uint64_t size, int64_t memory_space) {
  SE_DeviceAddressBase se_base =
      ExecutorApiFn()->TpuExecutor_AllocateFn(executor_, size, memory_space);
  return ApiConverter::FromC(se_base);
}

void TpuExecutor::Deallocate(const DeviceAddressBase& memory) {
  SE_DeviceAddressBase se_base = ApiConverter::ToC(memory);
  ExecutorApiFn()->TpuExecutor_DeallocateFn(executor_, &se_base);
}

void TpuExecutor::Deallocate(DeviceAddressBase* memory) {
  SE_DeviceAddressBase se_base = ApiConverter::ToC(*memory);
  ExecutorApiFn()->TpuExecutor_DeallocateFn(executor_, &se_base);
}

bool TpuExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  int64_t _free;
  int64_t _total;
  if (ExecutorApiFn()->TpuExecutor_DeviceMemoryUsageFn(executor_, &_free,
                                                       &_total)) {
    *free = _free;
    *total = _total;
    return true;
  }
  return false;
}

std::optional<stream_executor::AllocatorStats>
TpuExecutor::GetAllocatorStats() {
  SE_AllocatorStats c_stats;
  if (ExecutorApiFn()->TpuExecutor_GetAllocatorStatsFn(executor_, &c_stats)) {
    AllocatorStats stats;
    stats.num_allocs = c_stats.num_allocs;
    stats.bytes_in_use = c_stats.bytes_in_use;
    stats.peak_bytes_in_use = c_stats.peak_bytes_in_use;
    stats.largest_alloc_size = c_stats.largest_alloc_size;
    if (c_stats.has_bytes_limit) {
      stats.bytes_limit = c_stats.bytes_limit;
    }
    stats.bytes_reserved = c_stats.bytes_reserved;
    stats.peak_bytes_reserved = c_stats.peak_bytes_reserved;
    if (c_stats.has_bytes_reservable_limit) {
      stats.bytes_reservable_limit = c_stats.bytes_reservable_limit;
    }
    stats.largest_free_block_bytes = c_stats.largest_free_block_bytes;
    return stats;
  }
  return {};
}

void TpuExecutor::DequeueOutfeed(int32_t outfeed_queue_index,
                                 absl::Span<uint8_t> bytes,
                                 StatusCallback done) {
  if (ExecutorApiFn()->TpuExecutor_DequeueOutfeedV2Fn != nullptr) {
    StatusCallback* done_cb = new StatusCallback(std::move(done));
    SE_DequeueOutfeedCallback c_callback = [](TF_Status* c_status, void* ctx) {
      StatusCallback* cb = static_cast<StatusCallback*>(ctx);
      absl::Status status = StatusHelper::FromC(c_status);
      // c_callback takes ownership of c_status allocated by
      // TpuExecutor_DequeueOutfeedV2.
      ExecutorApiFn()->TpuStatus_FreeFn(c_status);
      (*cb)(status);
      delete cb;
    };
    ExecutorApiFn()->TpuExecutor_DequeueOutfeedV2Fn(
        executor_, outfeed_queue_index, bytes.data(), bytes.size(), c_callback,
        done_cb);
  } else {
    // Legacy fallback: TpuExecutor_DequeueOutfeedFn relies on a stack-allocated
    // StatusHelper. Note: This exhibits potential asynchronous lifetime
    // limitations on older runtimes if the outfeed dequeue completes
    // asynchronously after DequeueOutfeed returns. DequeueOutfeedV2 safely
    // resolves this by heap-allocating the status callback context.
    StatusHelper status;
    ExecutorApiFn()->TpuExecutor_DequeueOutfeedFn(
        executor_, outfeed_queue_index, bytes.data(), bytes.size(),
        status.c_status);
    done(status.status());
  }
}

absl::Status TpuExecutor::EnqueueInfeed(int32_t infeed_queue_index,
                                        absl::Span<const uint8_t> bytes) {
  StatusHelper status;
  ExecutorApiFn()->TpuExecutor_EnqueueInfeedFn(executor_, infeed_queue_index,
                                               bytes.data(), bytes.size(),
                                               status.c_status);
  return status.status();
}

absl::Status TpuExecutor::SynchronousMemcpy(DeviceAddressBase* device_dst,
                                            const void* host_src,
                                            uint64_t size) {
  StatusHelper status;
  SE_DeviceAddressBase se_base = ApiConverter::ToC(*device_dst);
  ExecutorApiFn()->TpuExecutor_SynchronousMemcpyFromHostFn(
      executor_, &se_base, host_src, size, status.c_status);
  return status.status();
}

absl::Status TpuExecutor::SynchronousMemcpy(void* host_dst,
                                            const DeviceAddressBase& device_src,
                                            uint64_t size) {
  StatusHelper status;
  SE_DeviceAddressBase se_base = ApiConverter::ToC(device_src);
  ExecutorApiFn()->TpuExecutor_SynchronousMemcpyToHostFn(
      executor_, host_dst, &se_base, size, status.c_status);
  return status.status();
}

absl::Status TpuExecutor::UnloadAllPrograms() {
  StatusHelper status;
  ExecutorApiFn()->TpuExecutor_UnloadAllProgramsFn(executor_, status.c_status);
  return status.status();
}

absl::Status TpuExecutor::EnqueueCompactionOnStreamForHbm(
    Stream* compaction_stream) {
  StatusHelper status;
  ExecutorApiFn()->TpuExecutor_EnqueueCompactionOnStreamForHbmFn(
      executor_, get_stream(compaction_stream), status.c_status);
  return status.status();
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
TpuExecutor::CreateDeviceDescription() const {
  StatusHelper status;
  SE_DeviceDescription* description =
      ExecutorApiFn()->TpuDeviceDescription_NewFn();
  absl::Cleanup cleanup = [description]() {
    ExecutorApiFn()->TpuDeviceDescription_FreeFn(description);
  };
  ExecutorApiFn()->TpuExecutor_CreateDeviceDescriptionFn(executor_, description,
                                                         status.c_status);
  if (status.status().ok()) {
    stream_executor::DeviceDescription desc;
    CHECK_NE(description->device_vendor, nullptr);
    desc.set_device_vendor(description->device_vendor);
    desc.set_name(description->name);
    desc.set_clock_rate_ghz(description->clock_rate_ghz);
    desc.set_core_count(description->core_count);
    desc.set_ecc_enabled(description->ecc_enabled);
    desc.set_device_memory_size(description->device_memory_size);
    desc.set_platform_version(description->platform_version);
    if (description->driver_version != nullptr) {
      absl::StatusOr<SemanticVersion> version_or =
          SemanticVersion::ParseFromString(description->driver_version);
      if (version_or.ok()) {
        desc.set_driver_version(version_or.value());
      }
    }
    if (description->runtime_version != nullptr) {
      absl::StatusOr<SemanticVersion> version_or =
          SemanticVersion::ParseFromString(description->runtime_version);
      if (version_or.ok()) {
        desc.set_runtime_version(version_or.value());
      }
    }
    if (description->pci_bus_id != nullptr) {
      desc.set_pci_bus_id(description->pci_bus_id);
    }
    return std::make_unique<DeviceDescription>(std::move(desc));
  }
  return status.status();
}

}  // namespace tpu
}  // namespace stream_executor
