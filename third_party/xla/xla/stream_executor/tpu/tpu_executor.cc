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

#include "absl/cleanup/cleanup.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/stream_executor/allocator_stats.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/tpu/c_api_conversions.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_event.h"
#include "xla/stream_executor/tpu/tpu_executor_api.h"
#include "xla/stream_executor/tpu/tpu_stream.h"
#include "xla/stream_executor/tpu/tpu_topology.h"
#include "xla/tsl/c/tsl_status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/status.h"

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

absl::Status TpuExecutor::BlockHostUntilDone(Stream* stream) {
  StatusHelper status;
  ExecutorApiFn()->TpuExecutor_BlockHostUntilDoneFn(
      executor_, get_stream(stream), status.c_status);
  return status.status();
}

tensorflow::tpu::TpuCoreLocationExternal TpuExecutor::GetCoreLocationExternal()
    const {
  return tensorflow::tpu::TpuCoreLocationExternal(
      ExecutorApiFn()->TpuExecutor_GetCoreLocationFn(executor_));
}

void TpuExecutor::DeallocateStream(Stream* stream) {
  ExecutorApiFn()->TpuExecutor_DeallocateStreamFn(executor_,
                                                  get_stream(stream));
  tpu_platform().mutex().Lock();
  stream_map().erase(stream);
  tpu_platform().mutex().Unlock();
}

absl::StatusOr<std::unique_ptr<Stream>> TpuExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  SE_Stream* tpu_stream = ExecutorApiFn()->TpuStream_NewFn(executor_);
  auto stream = std::make_unique<tensorflow::tpu::TpuStream>(
      tpu_stream, this, executor_, &tpu_platform());
  tpu_platform().mutex().Lock();
  stream_map()[stream.get()] = tpu_stream;
  tpu_platform().mutex().Unlock();
  return std::move(stream);
}

absl::StatusOr<std::unique_ptr<Event>> TpuExecutor::CreateEvent() {
  SE_Event* se_event = ExecutorApiFn()->TpuEvent_NewFn(executor_);
  auto tpu_event = std::make_unique<TpuEvent>(se_event, platform_);
  tpu_platform().InsertEvent(tpu_event.get(), se_event);

  StatusHelper status;
  ExecutorApiFn()->TpuExecutor_AllocateEventFn(executor_, se_event,
                                               status.c_status);
  TF_RETURN_IF_ERROR(status.status());

  return std::move(tpu_event);
}

DeviceMemoryBase TpuExecutor::Allocate(uint64_t size, int64_t memory_space) {
  SE_DeviceMemoryBase se_base =
      ExecutorApiFn()->TpuExecutor_AllocateFn(executor_, size, memory_space);
  return ApiConverter::FromC(se_base);
}

void TpuExecutor::Deallocate(const DeviceMemoryBase& memory) {
  SE_DeviceMemoryBase se_base = ApiConverter::ToC(memory);
  ExecutorApiFn()->TpuExecutor_DeallocateFn(executor_, &se_base);
}

void TpuExecutor::Deallocate(DeviceMemoryBase* memory) {
  SE_DeviceMemoryBase se_base = ApiConverter::ToC(*memory);
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
    ::stream_executor::AllocatorStats stats;
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
  StatusHelper status;
  ExecutorApiFn()->TpuExecutor_DequeueOutfeedFn(executor_, outfeed_queue_index,
                                                bytes.data(), bytes.size(),
                                                status.c_status);
  done(status.status());
}

absl::Status TpuExecutor::EnqueueInfeed(int32_t infeed_queue_index,
                                        absl::Span<const uint8_t> bytes) {
  StatusHelper status;
  ExecutorApiFn()->TpuExecutor_EnqueueInfeedFn(executor_, infeed_queue_index,
                                               bytes.data(), bytes.size(),
                                               status.c_status);
  return status.status();
}

absl::Status TpuExecutor::Memcpy(
    Stream* stream, void* host_dst,
    const ::stream_executor::DeviceMemoryBase& device_src, uint64_t size) {
  StatusHelper status;
  SE_DeviceMemoryBase se_base = ApiConverter::ToC(device_src);
  ExecutorApiFn()->TpuExecutor_MemcpyToHostFn(
      executor_, get_stream(stream), host_dst, &se_base, size, status.c_status);
  return status.status();
}

absl::Status TpuExecutor::Memcpy(
    Stream* stream, ::stream_executor::DeviceMemoryBase* device_dst,
    const void* host_src, uint64_t size) {
  StatusHelper status;
  SE_DeviceMemoryBase se_base = ApiConverter::ToC(*device_dst);
  ExecutorApiFn()->TpuExecutor_MemcpyFromHostFn(
      executor_, get_stream(stream), &se_base, host_src, size, status.c_status);
  return status.status();
}

absl::Status TpuExecutor::SynchronousMemcpy(
    ::stream_executor::DeviceMemoryBase* device_dst, const void* host_src,
    uint64_t size) {
  StatusHelper status;
  SE_DeviceMemoryBase se_base = ApiConverter::ToC(*device_dst);
  ExecutorApiFn()->TpuExecutor_SynchronousMemcpyFromHostFn(
      executor_, &se_base, host_src, size, status.c_status);
  return status.status();
}

absl::Status TpuExecutor::SynchronousMemcpy(
    void* host_dst, const ::stream_executor::DeviceMemoryBase& device_src,
    uint64_t size) {
  StatusHelper status;
  SE_DeviceMemoryBase se_base = ApiConverter::ToC(device_src);
  ExecutorApiFn()->TpuExecutor_SynchronousMemcpyToHostFn(
      executor_, host_dst, &se_base, size, status.c_status);
  return status.status();
}

bool TpuExecutor::MemcpyDeviceToDevice(
    Stream* stream, ::stream_executor::DeviceMemoryBase* gpu_dst,
    const ::stream_executor::DeviceMemoryBase& host_src, uint64_t size) {
  LOG(FATAL) << __func__ << " not supported on TpuExecutor";
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

struct HostCallbackContext {
  absl::AnyInvocable<absl::Status() &&> callback;
};

TSL_Status* HostCallbackTrampoline(void* ctx) {
  HostCallbackContext* host_ctx = reinterpret_cast<HostCallbackContext*>(ctx);
  absl::Status status = std::move(host_ctx->callback)();
  TSL_Status* c_status = ExecutorApiFn()->TpuStatus_CreateFn(
      status.raw_code(), absl::StatusMessageAsCStr(status));
  delete host_ctx;
  return c_status;
}

bool TpuExecutor::HostCallback(Stream* stream,
                               absl::AnyInvocable<absl::Status() &&> callback) {
  HostCallbackContext* ctx = new HostCallbackContext{std::move(callback)};
  return ExecutorApiFn()->TpuExecutor_HostCallbackFn(
      executor_, get_stream(stream), &HostCallbackTrampoline, ctx);
}

absl::StatusOr<std::unique_ptr<::stream_executor::DeviceDescription>>
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
    stream_executor::internal::DeviceDescriptionBuilder builder;
    CHECK_NE(description->device_vendor, nullptr);
    builder.set_device_vendor(description->device_vendor);
    builder.set_name(description->name);
    builder.set_clock_rate_ghz(description->clock_rate_ghz);
    builder.set_core_count(description->core_count);
    builder.set_ecc_enabled(description->ecc_enabled);
    builder.set_device_memory_size(description->device_memory_size);
    builder.set_platform_version(description->platform_version);
    return builder.Build();
  }
  return status.status();
}

}  // namespace tpu
}  // namespace stream_executor
