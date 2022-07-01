/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/tpu/tpu_executor.h"

#include "absl/cleanup/cleanup.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_event.h"
#include "tensorflow/stream_executor/tpu/tpu_stream.h"
#include "tensorflow/stream_executor/tpu/tpu_timer.h"

using stream_executor::DeviceMemoryBase;

namespace tensorflow {
namespace tpu {

namespace {
using ::stream_executor::port::Status;
}  // namespace

TpuExecutor::~TpuExecutor() {
  tpu::ExecutorApiFn()->TpuExecutor_FreeFn(executor_);
}

Status TpuExecutor::Init(int device_ordinal,
                         ::stream_executor::DeviceOptions device_options) {
  StatusHelper status;
  SE_DeviceOptions* options =
      tpu::ExecutorApiFn()->TpuExecutor_NewDeviceOptionsFn(
          device_options.flags());
  tpu::ExecutorApiFn()->TpuExecutor_InitFn(executor_, device_ordinal, options,
                                           status.c_status);
  tpu::ExecutorApiFn()->TpuExecutor_FreeDeviceOptionsFn(options);
  return status.status();
}

int TpuExecutor::PlatformDeviceCount() {
  return tpu::ExecutorApiFn()->TpuExecutor_PlatformDeviceCountFn(executor_);
}

void TpuExecutor::SyncAndForgetFailedStreams() {
  tpu::ExecutorApiFn()->TpuExecutor_SyncAndForgetFailedStreamsFn(executor_);
}

bool TpuExecutor::SynchronizeAllActivity() {
  return tpu::ExecutorApiFn()->TpuExecutor_SynchronizeAllActivityFn(executor_);
}

Status TpuExecutor::BlockHostUntilDone(Stream* stream) {
  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_BlockHostUntilDoneFn(
      executor_, get_stream(stream->implementation()), status.c_status);
  return status.status();
}

Status TpuExecutor::BlockUntilDoneOrFailed() {
  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_BlockUntilDoneOrFailedFn(executor_,
                                                             status.c_status);
  return status.status();
}

Status TpuExecutor::GetStatus(Stream* stream) {
  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_GetStatusFn(
      executor_, get_stream(stream->implementation()), status.c_status);
  return status.status();
}

tpu::TpuCoreLocationExternal TpuExecutor::GetCoreLocationExternal() const {
  return tpu::TpuCoreLocationExternal(
      tpu::ExecutorApiFn()->TpuExecutor_GetCoreLocationFn(executor_));
}

bool TpuExecutor::AllocateStream(Stream* stream) {
  return tpu::ExecutorApiFn()->TpuExecutor_AllocateStreamFn(
      executor_, get_stream(stream->implementation()));
}

void TpuExecutor::DeallocateStream(Stream* stream) {
  tpu::ExecutorApiFn()->TpuExecutor_DeallocateStreamFn(
      executor_, get_stream(stream->implementation()));
  tpu_platform().mutex().lock();
  stream_map().erase(stream->implementation());
  tpu_platform().mutex().unlock();
}

bool TpuExecutor::CreateStreamDependency(Stream* dependent, Stream* other) {
  return tpu::ExecutorApiFn()->TpuExecutor_CreateStreamDependencyFn(
      executor_, get_stream(dependent->implementation()),
      get_stream(other->implementation()));
}

Status TpuExecutor::AllocateEvent(Event* event) { return OkStatus(); }

Status TpuExecutor::DeallocateEvent(Event* event) {
  tpu_platform().EraseEvent(event->implementation());
  return OkStatus();
}

// AllocateTimer/DeallocateTimer have no specialization.
bool TpuExecutor::AllocateTimer(Timer* timer) { return true; }

void TpuExecutor::DeallocateTimer(Timer* timer) {}

bool TpuExecutor::StartTimer(Stream* stream, ::stream_executor::Timer* timer) {
  return tpu::ExecutorApiFn()->TpuExecutor_StartTimerFn(
      executor_, get_stream(stream->implementation()),
      timer_map_.at(timer->implementation()));
}

bool TpuExecutor::StopTimer(Stream* stream, ::stream_executor::Timer* timer) {
  return tpu::ExecutorApiFn()->TpuExecutor_StopTimerFn(
      executor_, get_stream(stream->implementation()),
      timer_map_.at(timer->implementation()));
}

stream_executor::Event::Status TpuExecutor::PollForEventStatus(
    stream_executor::Event* event) {
  auto se_event = tpu_platform().LookupEvent(event->implementation());
  return stream_executor::Event::Status(
      tpu::ExecutorApiFn()->TpuExecutor_PollForEventStatusFn(executor_,
                                                             se_event));
}

Status TpuExecutor::RecordEvent(Stream* stream,
                                ::stream_executor::Event* event) {
  StatusHelper status;
  auto se_event = tpu_platform().LookupEvent(event->implementation());
  tpu::ExecutorApiFn()->TpuExecutor_RecordEventFn(
      executor_, get_stream(stream->implementation()), se_event,
      status.c_status);
  return status.status();
}

Status TpuExecutor::WaitForEvent(Stream* stream,
                                 ::stream_executor::Event* event) {
  StatusHelper status;
  auto se_event = tpu_platform().LookupEvent(event->implementation());
  tpu::ExecutorApiFn()->TpuExecutor_WaitForEventFn(
      executor_, get_stream(stream->implementation()), se_event,
      status.c_status);
  return status.status();
}

// Implementations for Timer, Stream, Event
// We need to map these implementations to internal equivalents -- thus we
// allocate the internal Timer, Stream and Event operations here, and map
// the implementations to the internal values. The "wrapper" interfaces are
// responsible for deallocating the internal value when they are destroyed.

// Called by Timer::Timer
std::unique_ptr<::stream_executor::internal::TimerInterface>
TpuExecutor::GetTimerImplementation() {
  SE_Timer* tpu_timer = tpu::ExecutorApiFn()->TpuTimer_NewFn(executor_);
  auto ptr = absl::make_unique<TpuTimer>(tpu_timer);
  timer_map_[ptr.get()] = tpu_timer;
  return ptr;
}

// Called by Stream::Stream
std::unique_ptr<::stream_executor::internal::StreamInterface>
TpuExecutor::GetStreamImplementation() {
  SE_Stream* tpu_stream = tpu::ExecutorApiFn()->TpuStream_NewFn(executor_);
  auto ptr = absl::make_unique<tpu::TpuStream>(tpu_stream);
  tpu_platform().mutex().lock();
  stream_map()[ptr.get()] = tpu_stream;
  tpu_platform().mutex().unlock();
  return ptr;
}

// Called by Event::Event
std::unique_ptr<::stream_executor::internal::EventInterface>
TpuExecutor::CreateEventImplementation() {
  SE_Event* tpu_event = tpu::ExecutorApiFn()->TpuEvent_NewFn(executor_);
  auto ptr = absl::make_unique<TpuEvent>(tpu_event);
  tpu_platform().InsertEvent(ptr.get(), tpu_event);
  return ptr;
}

DeviceMemoryBase TpuExecutor::Allocate(uint64_t size, int64_t memory_space) {
  SE_DeviceMemoryBase se_base = tpu::ExecutorApiFn()->TpuExecutor_AllocateFn(
      executor_, size, memory_space);
  return ApiConverter::FromC(se_base);
}

void TpuExecutor::Deallocate(const DeviceMemoryBase& memory) {
  SE_DeviceMemoryBase se_base = ApiConverter::ToC(memory);
  tpu::ExecutorApiFn()->TpuExecutor_DeallocateFn(executor_, &se_base);
}

void TpuExecutor::Deallocate(DeviceMemoryBase* memory) {
  SE_DeviceMemoryBase se_base = ApiConverter::ToC(*memory);
  tpu::ExecutorApiFn()->TpuExecutor_DeallocateFn(executor_, &se_base);
}

bool TpuExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  int64_t _free;
  int64_t _total;
  if (tpu::ExecutorApiFn()->TpuExecutor_DeviceMemoryUsageFn(executor_, &_free,
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
  if (tpu::ExecutorApiFn()->TpuExecutor_GetAllocatorStatsFn(executor_,
                                                            &c_stats)) {
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

Status TpuExecutor::WaitForInfeedReady(int32_t infeed_queue_index) {
  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_WaitForInfeedReadyFn(
      executor_, infeed_queue_index, status.c_status);
  return status.status();
}

Status TpuExecutor::WaitForOutfeedReady(int32_t outfeed_queue_index) {
  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_WaitForOutfeedReadyFn(
      executor_, outfeed_queue_index, status.c_status);
  return status.status();
}

void TpuExecutor::DequeueOutfeed(int32_t outfeed_queue_index,
                                 absl::Span<uint8> bytes, StatusCallback done) {
  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_DequeueOutfeedFn(
      executor_, outfeed_queue_index, bytes.data(), bytes.size(),
      status.c_status);
  done(status.status());
}

Status TpuExecutor::EnqueueInfeed(int32_t infeed_queue_index,
                                  absl::Span<const uint8> bytes) {
  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_EnqueueInfeedFn(
      executor_, infeed_queue_index, bytes.data(), bytes.size(),
      status.c_status);
  return status.status();
}

bool TpuExecutor::Memcpy(Stream* stream, void* host_dst,
                         const ::stream_executor::DeviceMemoryBase& device_src,
                         uint64_t size) {
  SE_DeviceMemoryBase se_base = ApiConverter::ToC(device_src);
  return tpu::ExecutorApiFn()->TpuExecutor_MemcpyToHostFn(
      executor_, get_stream(stream->implementation()), host_dst, &se_base,
      size);
}

bool TpuExecutor::Memcpy(Stream* stream,
                         ::stream_executor::DeviceMemoryBase* device_dst,
                         const void* host_src, uint64_t size) {
  SE_DeviceMemoryBase se_base = ApiConverter::ToC(*device_dst);
  return tpu::ExecutorApiFn()->TpuExecutor_MemcpyFromHostFn(
      executor_, get_stream(stream->implementation()), &se_base, host_src,
      size);
}

Status TpuExecutor::SynchronousMemcpy(
    ::stream_executor::DeviceMemoryBase* device_dst, const void* host_src,
    uint64_t size) {
  StatusHelper status;
  SE_DeviceMemoryBase se_base = ApiConverter::ToC(*device_dst);
  tpu::ExecutorApiFn()->TpuExecutor_SynchronousMemcpyFromHostFn(
      executor_, &se_base, host_src, size, status.c_status);
  return status.status();
}

Status TpuExecutor::SynchronousMemcpy(
    void* host_dst, const ::stream_executor::DeviceMemoryBase& device_src,
    uint64_t size) {
  StatusHelper status;
  SE_DeviceMemoryBase se_base = ApiConverter::ToC(device_src);
  tpu::ExecutorApiFn()->TpuExecutor_SynchronousMemcpyToHostFn(
      executor_, host_dst, &se_base, size, status.c_status);
  return status.status();
}

Status TpuExecutor::SynchronousMemcpyDeviceToDevice(
    ::stream_executor::DeviceMemoryBase* device_dst,
    const ::stream_executor::DeviceMemoryBase& device_src, uint64_t size) {
  return ::stream_executor::port::UnimplementedError(
      "This operation not supported on TPU");
}

bool TpuExecutor::MemcpyDeviceToDevice(
    Stream* stream, ::stream_executor::DeviceMemoryBase* gpu_dst,
    const ::stream_executor::DeviceMemoryBase& host_src, uint64_t size) {
  LOG(FATAL) << __func__ << " not supported on TpuExecutor";
}

Status TpuExecutor::UnloadAllPrograms() {
  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_UnloadAllProgramsFn(executor_,
                                                        status.c_status);
  return status.status();
}

Status TpuExecutor::EnqueueCompactionOnStreamForHbm(Stream* compaction_stream) {
  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_EnqueueCompactionOnStreamForHbmFn(
      executor_, get_stream(compaction_stream->implementation()),
      status.c_status);
  return status.status();
}

struct HostCallbackContext {
  std::function<Status()> callback;
};

TF_Status* HostCallbackTrampoline(void* ctx) {
  HostCallbackContext* host_ctx = reinterpret_cast<HostCallbackContext*>(ctx);
  Status status = host_ctx->callback();
  TF_Status* c_status = tpu::ExecutorApiFn()->TpuStatus_CreateFn(
      status.code(), status.error_message().c_str());
  delete host_ctx;
  return c_status;
}

bool TpuExecutor::HostCallback(Stream* stream,
                               std::function<Status()> callback) {
  HostCallbackContext* ctx = new HostCallbackContext{callback};
  return tpu::ExecutorApiFn()->TpuExecutor_HostCallbackFn(
      executor_, get_stream(stream->implementation()), &HostCallbackTrampoline,
      ctx);
}

TpuExecutor::StatusOr<std::unique_ptr<::stream_executor::DeviceDescription>>
TpuExecutor::CreateDeviceDescription() const {
  StatusHelper status;
  SE_DeviceDescription* description =
      tpu::ExecutorApiFn()->TpuDeviceDescription_NewFn();
  absl::Cleanup cleanup = [description]() {
    tpu::ExecutorApiFn()->TpuDeviceDescription_FreeFn(description);
  };
  tpu::ExecutorApiFn()->TpuExecutor_CreateDeviceDescriptionFn(
      executor_, description, status.c_status);
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
}  // namespace tensorflow
