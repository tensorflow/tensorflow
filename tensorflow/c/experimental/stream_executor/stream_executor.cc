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
// This file extends/implements core stream executor base classes in terms of
// the C API defined in stream_executor.h. A class "CSomething" represents a
// "Something" that can be manipulated via calls in the C interface and a C
// struct called "SP_Something".
//
// This file also contains stream_executor::Platform registration for pluggable
// device.
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"

#include <string>

#include "tensorflow/c/experimental/stream_executor/stream_executor_internal.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/stream_executor/timer.h"

using tensorflow::StatusFromTF_Status;

namespace stream_executor {
using tensorflow::StringPiece;
using OwnedTFStatus = std::unique_ptr<TF_Status, TFStatusDeleter>;

namespace {

#define VALIDATE_STRUCT_SIZE(STRUCT_NAME, STRUCT_OBJ, SIZE_VALUE_NAME) \
  do {                                                                 \
    if (STRUCT_OBJ.struct_size == 0) {                                 \
      return port::FailedPreconditionError(                            \
          "struct_size field in " #STRUCT_NAME                         \
          " must be set to " #SIZE_VALUE_NAME ".");                    \
    }                                                                  \
  } while (0)

#define VALIDATE_MEMBER(STRUCT_NAME, STRUCT_OBJ, NAME)           \
  do {                                                           \
    if (STRUCT_OBJ.NAME == 0) {                                  \
      return port::FailedPreconditionError(                      \
          "'" #NAME "' field in " #STRUCT_NAME " must be set."); \
    }                                                            \
  } while (0)

port::Status ValidateDeviceType(StringPiece type) {
  // Validate device type. Device type must start with a capital letter and
  // consist of capital letters and underscores. Reasoning behind this decision:
  // * At the minimum we want to disallow '/' and ':' since
  //   these characters are used in device spec, for e.g.
  //   /job:foo/replica:12/device:GPU:1.
  // * Underscores seem useful, for e.g. XLA_GPU uses underscores.
  // * Allowing lowercase might get confusing. For example, say someone
  //   registers a new type called "Gpu". It might be confusing for users that
  //   "Gpu" is not the same device type as "GPU".
  //   Note that lowercase "cpu" and "gpu" are currently supported only for
  //   legacy reasons:
  //   https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/python/framework/device_spec.py;l=46;drc=d3a378f9665d8eee827c74cb9ecbee81e4c288dd
  static const LazyRE2 kTfDeviceTypeRegEx = {"[A-Z][A-Z_]*"};
  bool matches = RE2::FullMatch(type, *kTfDeviceTypeRegEx);
  if (!matches) {
    return port::FailedPreconditionError(
        tensorflow::strings::StrCat("Device name/type '", type, "' must match ",
                                    kTfDeviceTypeRegEx->pattern(), "."));
  }
  return port::Status::OK();
}

port::Status ValidateSPPlatform(const SP_Platform& platform) {
  VALIDATE_STRUCT_SIZE(SP_Platform, platform, SP_PLATFORM_STRUCT_SIZE);
  VALIDATE_MEMBER(SP_Platform, platform, name);
  VALIDATE_MEMBER(SP_Platform, platform, type);
  TF_RETURN_IF_ERROR(ValidateDeviceType(platform.name));
  TF_RETURN_IF_ERROR(ValidateDeviceType(platform.type));
  // `visible_device_count` could be 0 at initialization time.
  return port::Status::OK();
}

port::Status ValidateSPPlatformFns(const SP_PlatformFns& platform_fns) {
  VALIDATE_STRUCT_SIZE(SP_PlatformFns, platform_fns,
                       SP_PLATFORM_FNS_STRUCT_SIZE);
  VALIDATE_MEMBER(SP_PlatformFns, platform_fns, create_device);
  VALIDATE_MEMBER(SP_PlatformFns, platform_fns, destroy_device);
  VALIDATE_MEMBER(SP_PlatformFns, platform_fns, create_stream_executor);
  VALIDATE_MEMBER(SP_PlatformFns, platform_fns, destroy_stream_executor);
  VALIDATE_MEMBER(SP_PlatformFns, platform_fns, create_timer_fns);
  VALIDATE_MEMBER(SP_PlatformFns, platform_fns, destroy_timer_fns);
  VALIDATE_MEMBER(SP_PlatformFns, platform_fns, create_device_fns);
  VALIDATE_MEMBER(SP_PlatformFns, platform_fns, destroy_device_fns);
  return port::Status::OK();
}

port::Status ValidateSPTimerFns(const SP_TimerFns& timer_fns) {
  VALIDATE_STRUCT_SIZE(SP_TimerFns, timer_fns, SP_TIMER_FNS_STRUCT_SIZE);
  VALIDATE_MEMBER(SP_TimerFns, timer_fns, nanoseconds);
  return port::Status::OK();
}

port::Status ValidateSPAllocatorStats(const SP_AllocatorStats& stats) {
  VALIDATE_STRUCT_SIZE(SP_AllocatorStats, stats, SP_ALLOCATORSTATS_STRUCT_SIZE);
  // All other fields could theoretically be zero/null.
  return port::Status::OK();
}

port::Status ValidateSPDeviceMemoryBase(const SP_DeviceMemoryBase& mem) {
  VALIDATE_STRUCT_SIZE(SP_DeviceMemoryBase, mem,
                       SP_DEVICE_MEMORY_BASE_STRUCT_SIZE);
  // All other fields could theoretically be zero/null.
  return port::Status::OK();
}

port::Status ValidateSPDevice(const SP_Device& device) {
  VALIDATE_STRUCT_SIZE(SP_Device, device, SP_DEVICE_STRUCT_SIZE);
  // All other fields could theoretically be zero/null.
  return port::Status::OK();
}

port::Status ValidateSPDeviceFns(const SP_DeviceFns& device_fns) {
  VALIDATE_STRUCT_SIZE(SP_DeviceFns, device_fns, SP_DEVICE_FNS_STRUCT_SIZE);
  // All other fields could theoretically be zero/null.
  return port::Status::OK();
}

port::Status ValidateSPStreamExecutor(const SP_StreamExecutor& se,
                                      const SP_Platform& platform) {
  VALIDATE_STRUCT_SIZE(SP_StreamExecutor, se, SP_STREAM_EXECUTOR_STRUCT_SIZE);
  VALIDATE_MEMBER(SP_StreamExecutor, se, allocate);
  VALIDATE_MEMBER(SP_StreamExecutor, se, deallocate);
  VALIDATE_MEMBER(SP_StreamExecutor, se, get_allocator_stats);
  VALIDATE_MEMBER(SP_StreamExecutor, se, host_memory_allocate);
  VALIDATE_MEMBER(SP_StreamExecutor, se, host_memory_deallocate);
  if (platform.supports_unified_memory) {
    VALIDATE_MEMBER(SP_StreamExecutor, se, unified_memory_allocate);
    VALIDATE_MEMBER(SP_StreamExecutor, se, unified_memory_deallocate);
  }
  VALIDATE_MEMBER(SP_StreamExecutor, se, device_memory_usage);
  VALIDATE_MEMBER(SP_StreamExecutor, se, create_stream);
  VALIDATE_MEMBER(SP_StreamExecutor, se, destroy_stream);
  VALIDATE_MEMBER(SP_StreamExecutor, se, create_stream_dependency);
  VALIDATE_MEMBER(SP_StreamExecutor, se, get_stream_status);
  VALIDATE_MEMBER(SP_StreamExecutor, se, create_event);
  VALIDATE_MEMBER(SP_StreamExecutor, se, destroy_event);
  VALIDATE_MEMBER(SP_StreamExecutor, se, get_event_status);
  VALIDATE_MEMBER(SP_StreamExecutor, se, record_event);
  VALIDATE_MEMBER(SP_StreamExecutor, se, wait_for_event);
  VALIDATE_MEMBER(SP_StreamExecutor, se, create_timer);
  VALIDATE_MEMBER(SP_StreamExecutor, se, destroy_timer);
  VALIDATE_MEMBER(SP_StreamExecutor, se, start_timer);
  VALIDATE_MEMBER(SP_StreamExecutor, se, stop_timer);
  VALIDATE_MEMBER(SP_StreamExecutor, se, memcpy_dtoh);
  VALIDATE_MEMBER(SP_StreamExecutor, se, memcpy_htod);
  VALIDATE_MEMBER(SP_StreamExecutor, se, sync_memcpy_dtoh);
  VALIDATE_MEMBER(SP_StreamExecutor, se, sync_memcpy_htod);
  VALIDATE_MEMBER(SP_StreamExecutor, se, block_host_for_event);
  VALIDATE_MEMBER(SP_StreamExecutor, se, synchronize_all_activity);
  VALIDATE_MEMBER(SP_StreamExecutor, se, host_callback);
  return port::Status::OK();
}

port::Status ValidateSEPlatformRegistrationParams(
    const SE_PlatformRegistrationParams& params) {
  VALIDATE_STRUCT_SIZE(SE_PlatformRegistrationParams, params,
                       SE_PLATFORM_REGISTRATION_PARAMS_STRUCT_SIZE);
  VALIDATE_MEMBER(SE_PlatformRegistrationParams, params, destroy_platform);
  VALIDATE_MEMBER(SE_PlatformRegistrationParams, params, destroy_platform_fns);
  return port::Status::OK();
}
#undef VALIDATE_MEMBER

// Converts SE_EventStatus to Event::Status.
Event::Status SEEventStatusToEventStatus(SE_EventStatus s) {
  switch (s) {
    case SE_EVENT_ERROR:
      return Event::Status::kError;
    case SE_EVENT_PENDING:
      return Event::Status::kPending;
    case SE_EVENT_COMPLETE:
      return Event::Status::kComplete;
    default:
      return Event::Status::kUnknown;
  }
}

// Converts DeviceMemoryBase to a C struct.
SP_DeviceMemoryBase DeviceMemoryBaseToC(const DeviceMemoryBase* mem) {
  SP_DeviceMemoryBase device_memory_base{SP_DEVICE_MEMORY_BASE_STRUCT_SIZE};
  // `opaque` field inside SP_DeviceMemoryBase is not const.
  // Therefore, we need to cast away the constness before setting it.
  device_memory_base.opaque = const_cast<void*>(mem->opaque());
  device_memory_base.size = mem->size();
  device_memory_base.payload = mem->payload();
  return device_memory_base;
}

DeviceMemoryBase DeviceMemoryBaseFromC(const SP_DeviceMemoryBase& mem) {
  DeviceMemoryBase base(mem.opaque, mem.size);
  base.SetPayload(mem.payload);
  return base;
}

// Wrapper that allows passing std::function across C API.
struct HostCallbackContext {
  std::function<port::Status()> callback;
};

// This wrapper allows calling `HostCallbackContext::callback` across C API.
// This function matches `SE_StatusCallbackFn` signature and will be passed as
// `callback_fn` to `host_callback` in `SP_StreamExecutor`.
void HostCallbackTrampoline(void* ctx, TF_Status* status) {
  HostCallbackContext* host_ctx = static_cast<HostCallbackContext*>(ctx);
  port::Status s = host_ctx->callback();
  Set_TF_Status_from_Status(status, s);
  delete host_ctx;
}

class CStreamExecutor : public internal::StreamExecutorInterface {
 public:
  explicit CStreamExecutor(SP_Device device, SP_DeviceFns* device_fns,
                           SP_StreamExecutor* stream_executor,
                           SP_Platform* platform, SP_PlatformFns* platform_fns,
                           SP_TimerFns* timer_fns, const std::string& name,
                           int visible_device_count)
      : device_(std::move(device)),
        device_fns_(device_fns),
        stream_executor_(stream_executor),
        platform_(platform),
        platform_fns_(platform_fns),
        timer_fns_(timer_fns),
        platform_name_(name),
        visible_device_count_(visible_device_count) {}

  ~CStreamExecutor() override {
    platform_fns_->destroy_device(platform_, &device_);
  }

  port::Status Init(int device_ordinal, DeviceOptions device_options) override {
    return port::Status::OK();
  }

  DeviceMemoryBase Allocate(uint64 size, int64 memory_space) override {
    SP_DeviceMemoryBase mem = {SP_DEVICE_MEMORY_BASE_STRUCT_SIZE};
    stream_executor_->allocate(&device_, size, memory_space, &mem);
    port::Status status = ValidateSPDeviceMemoryBase(mem);
    if (!status.ok()) {
      LOG(ERROR) << status.error_message();
    }
    return DeviceMemoryBaseFromC(mem);
  }
  DeviceMemoryBase Allocate(uint64 size) {
    return Allocate(size, /*memory_space=*/0);
  }
  void* GetSubBuffer(DeviceMemoryBase* parent, uint64 offset,
                     uint64 size) override {
    LOG(FATAL) << "GetSubBuffer is not supported by pluggable device.";
  }

  void Deallocate(DeviceMemoryBase* mem) override {
    SP_DeviceMemoryBase device_memory_base = DeviceMemoryBaseToC(mem);
    stream_executor_->deallocate(&device_, &device_memory_base);
  }

  void* HostMemoryAllocate(uint64 size) override {
    return stream_executor_->host_memory_allocate(&device_, size);
  }

  void HostMemoryDeallocate(void* mem) override {
    stream_executor_->host_memory_deallocate(&device_, mem);
  }

  bool HostMemoryRegister(void* mem, uint64 size) override { return false; }
  bool HostMemoryUnregister(void* mem) override { return false; }

  void* UnifiedMemoryAllocate(uint64 size) override {
    CHECK(stream_executor_->unified_memory_allocate);
    return stream_executor_->unified_memory_allocate(&device_, size);
  }

  void UnifiedMemoryDeallocate(void* mem) override {
    CHECK(stream_executor_->unified_memory_deallocate);
    stream_executor_->unified_memory_deallocate(&device_, mem);
  }

  absl::optional<AllocatorStats> GetAllocatorStats() override {
    SP_AllocatorStats c_stats{SP_ALLOCATORSTATS_STRUCT_SIZE};
    TF_Bool has_stats =
        stream_executor_->get_allocator_stats(&device_, &c_stats);
    if (!has_stats) {
      return absl::nullopt;
    }
    port::Status status = ValidateSPAllocatorStats(c_stats);
    if (!status.ok()) {
      LOG(ERROR) << status.error_message();
      return absl::nullopt;
    }
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
  bool SynchronizeAllActivity() override {
    OwnedTFStatus c_status(TF_NewStatus());
    stream_executor_->synchronize_all_activity(&device_, c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
      return false;
    }
    return true;
  }
  port::Status SynchronousMemZero(DeviceMemoryBase* location,
                                  uint64 size) override {
    // TODO(annarev): figure out if we should support memzero/memset
    // functionality by allocating on host and then copying to device.
    return port::UnimplementedError(
        "SynchronousMemZero is not supported by pluggable device.");
  }
  port::Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                                 uint64 size) override {
    return port::UnimplementedError(
        "SynchronousMemSet is not supported by pluggable device.");
  }
  port::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                 const void* host_src, uint64 size) override {
    OwnedTFStatus c_status(TF_NewStatus());
    SP_DeviceMemoryBase device_memory_base = DeviceMemoryBaseToC(gpu_dst);
    stream_executor_->sync_memcpy_htod(&device_, &device_memory_base, host_src,
                                       size, c_status.get());
    return StatusFromTF_Status(c_status.get());
  }
  port::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceMemoryBase& gpu_src,
                                 uint64 size) override {
    OwnedTFStatus c_status(TF_NewStatus());
    SP_DeviceMemoryBase device_memory_base = DeviceMemoryBaseToC(&gpu_src);
    stream_executor_->sync_memcpy_dtoh(&device_, host_dst, &device_memory_base,
                                       size, c_status.get());
    return StatusFromTF_Status(c_status.get());
  }
  port::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase* gpu_dst,
                                               const DeviceMemoryBase& gpu_src,
                                               uint64 size) override {
    OwnedTFStatus c_status(TF_NewStatus());
    SP_DeviceMemoryBase device_mem_dst = DeviceMemoryBaseToC(gpu_dst);
    SP_DeviceMemoryBase device_mem_src = DeviceMemoryBaseToC(&gpu_src);
    stream_executor_->sync_memcpy_dtod(&device_, &device_mem_dst,
                                       &device_mem_src, size, c_status.get());
    return StatusFromTF_Status(c_status.get());
  }
  port::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                       uint64 size) override {
    return port::UnimplementedError(
        "MemZero is not supported by pluggable device.");
  }
  port::Status Memset(Stream* stream, DeviceMemoryBase* location, uint8 pattern,
                      uint64 size) override {
    return port::UnimplementedError(
        "Memset is not supported by pluggable device.");
  }
  port::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                        uint32 pattern, uint64 size) override {
    return port::UnimplementedError(
        "Memset32 is not supported by pluggable device.");
  }
  bool Memcpy(Stream* stream, void* host_dst, const DeviceMemoryBase& gpu_src,
              uint64 size) override {
    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    SP_DeviceMemoryBase device_mem_src = DeviceMemoryBaseToC(&gpu_src);
    stream_executor_->memcpy_dtoh(&device_, stream_handle, host_dst,
                                  &device_mem_src, size, c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
      return false;
    }
    return true;
  }
  bool Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst, const void* host_src,
              uint64 size) override {
    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    SP_DeviceMemoryBase device_mem_dst = DeviceMemoryBaseToC(gpu_dst);
    stream_executor_->memcpy_htod(&device_, stream_handle, &device_mem_dst,
                                  host_src, size, c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
      return false;
    }
    return true;
  }
  bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                            const DeviceMemoryBase& gpu_src,
                            uint64 size) override {
    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    SP_DeviceMemoryBase device_mem_dst = DeviceMemoryBaseToC(gpu_dst);
    SP_DeviceMemoryBase device_mem_src = DeviceMemoryBaseToC(&gpu_src);
    stream_executor_->memcpy_dtod(&device_, stream_handle, &device_mem_dst,
                                  &device_mem_src, size, c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
      return false;
    }
    return true;
  }
  bool HostCallback(Stream* stream,
                    std::function<port::Status()> callback) override {
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    HostCallbackContext* ctx = new HostCallbackContext{callback};
    return stream_executor_->host_callback(&device_, stream_handle,
                                           &HostCallbackTrampoline, ctx);
  }
  port::Status AllocateEvent(Event* event) override {
    DCHECK(event != nullptr);
    return static_cast<CEvent*>(event->implementation())->Create();
  }
  port::Status DeallocateEvent(Event* event) override {
    static_cast<CEvent*>(event->implementation())->Destroy();
    return port::Status::OK();
  }
  port::Status RecordEvent(Stream* stream, Event* event) override {
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    return static_cast<CEvent*>(event->implementation())->Record(stream_handle);
  }
  port::Status WaitForEvent(Stream* stream, Event* event) override {
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    SP_Event event_handle =
        static_cast<CEvent*>(event->implementation())->Handle();
    OwnedTFStatus c_status(TF_NewStatus());
    stream_executor_->wait_for_event(&device_, stream_handle, event_handle,
                                     c_status.get());
    port::Status s = StatusFromTF_Status(c_status.get());
    return s;
  }
  Event::Status PollForEventStatus(Event* event) override {
    SP_Event event_handle =
        static_cast<CEvent*>(event->implementation())->Handle();
    SE_EventStatus event_status =
        stream_executor_->get_event_status(&device_, event_handle);
    return SEEventStatusToEventStatus(event_status);
  }
  bool AllocateStream(Stream* stream) override {
    DCHECK(stream != nullptr);
    port::Status status =
        static_cast<CStream*>(stream->implementation())->Create();
    // TODO(annarev): update AllocateStream to return status instead
    // (similar to AllocateEvent).
    return status.ok();
  }
  void DeallocateStream(Stream* stream) override {
    static_cast<CStream*>(stream->implementation())->Destroy();
  }
  bool CreateStreamDependency(Stream* dependent, Stream* other) override {
    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream dependent_handle =
        static_cast<CStream*>(dependent->implementation())->Handle();
    SP_Stream other_handle =
        static_cast<CStream*>(other->implementation())->Handle();
    stream_executor_->create_stream_dependency(&device_, dependent_handle,
                                               other_handle, c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
      return false;
    }
    return true;
  }
  bool AllocateTimer(Timer* timer) override {
    port::Status status =
        static_cast<CTimer*>(timer->implementation())->Create();
    // TODO(annarev): change return value of AllocateTimer
    // to status (similar to AllocateEvent).
    return status.ok();
  }
  void DeallocateTimer(Timer* timer) override {
    static_cast<CTimer*>(timer->implementation())->Destroy();
  }
  bool StartTimer(Stream* stream, Timer* timer) override {
    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    SP_Timer timer_handle =
        static_cast<CTimer*>(timer->implementation())->Handle();
    stream_executor_->start_timer(&device_, stream_handle, timer_handle,
                                  c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
      return false;
    }
    return true;
  }
  bool StopTimer(Stream* stream, Timer* timer) override {
    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    SP_Timer timer_handle =
        static_cast<CTimer*>(timer->implementation())->Handle();
    stream_executor_->stop_timer(&device_, stream_handle, timer_handle,
                                 c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
      return false;
    }
    return true;
  }
  port::Status BlockHostForEvent(Stream* stream, Event* event) {
    OwnedTFStatus c_status(TF_NewStatus());
    SP_Event event_handle =
        static_cast<CEvent*>(event->implementation())->Handle();
    stream_executor_->block_host_for_event(&device_, event_handle,
                                           c_status.get());
    return StatusFromTF_Status(c_status.get());
  }

  port::Status BlockHostUntilDone(Stream* stream) override {
    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();

    // If `block_host_until_done` is set, use it.
    if (stream_executor_->block_host_until_done != nullptr) {
      stream_executor_->block_host_until_done(&device_, stream_handle,
                                              c_status.get());
      return StatusFromTF_Status(c_status.get());
    }
    // Create and record an event and then wait for it.
    SP_Event event_handle;
    stream_executor_->create_event(&device_, &event_handle, c_status.get());
    TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status.get()));
    stream_executor_->record_event(&device_, stream_handle, event_handle,
                                   c_status.get());
    port::Status s = StatusFromTF_Status(c_status.get());
    if (!s.ok()) {
      stream_executor_->destroy_event(&device_, event_handle);
      return s;
    }
    stream_executor_->block_host_for_event(&device_, event_handle,
                                           c_status.get());
    stream_executor_->destroy_event(&device_, event_handle);
    return StatusFromTF_Status(c_status.get());
  }

  port::Status GetStatus(Stream* stream) override {
    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    stream_executor_->get_stream_status(&device_, stream_handle,
                                        c_status.get());
    return StatusFromTF_Status(c_status.get());
  }
  int PlatformDeviceCount() override { return visible_device_count_; }
  port::Status EnablePeerAccessTo(StreamExecutorInterface* other) override {
    return port::UnimplementedError(
        "EnablePeerAccessTo is not supported by pluggable device.");
  }
  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override {
    return false;
  }

  bool DeviceMemoryUsage(int64* free, int64* total) const override {
    static_assert(sizeof(int64_t) == sizeof(tensorflow::int64),
                  "64-bit int types should match in size");
    return stream_executor_->device_memory_usage(
        &device_, reinterpret_cast<int64_t*>(free),
        reinterpret_cast<int64_t*>(total));
  }

  // Creates a new DeviceDescription object.
  // Ownership is transferred to the caller.
  port::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    OwnedTFStatus c_status(TF_NewStatus());

    internal::DeviceDescriptionBuilder builder;
    if (device_.hardware_name != nullptr) {
      builder.set_name(device_.hardware_name);
    }
    if (device_.device_vendor != nullptr) {
      builder.set_device_vendor(device_.device_vendor);
    }
    if (device_.pci_bus_id != nullptr) {
      builder.set_pci_bus_id(device_.pci_bus_id);
    }

    if (device_fns_->get_numa_node != nullptr) {
      int32_t numa_node = device_fns_->get_numa_node(&device_);
      if (numa_node >= 0) {
        builder.set_numa_node(numa_node);
      }
    }

    if (device_fns_->get_memory_bandwidth != nullptr) {
      int64_t memory_bandwidth = device_fns_->get_memory_bandwidth(&device_);
      if (memory_bandwidth >= 0) {
        builder.set_memory_bandwidth(memory_bandwidth);
      }
    }
    // TODO(annarev): Add gflops field in DeviceDescription and set it here.
    // TODO(annarev): Perhaps add `supports_unified_memory` in
    // DeviceDescription.
    return builder.Build();
  }

  // Each call creates a new instance of the platform-specific implementation of
  // the corresponding interface type.
  std::unique_ptr<internal::EventInterface> CreateEventImplementation()
      override {
    return std::unique_ptr<internal::EventInterface>(
        new CEvent(&device_, stream_executor_));
  }
  std::unique_ptr<internal::KernelInterface> CreateKernelImplementation()
      override {
    LOG(FATAL)
        << "CreateKernelImplementation is not supported by pluggable device.";
  }
  std::unique_ptr<internal::StreamInterface> GetStreamImplementation()
      override {
    return std::unique_ptr<internal::StreamInterface>(
        new CStream(&device_, stream_executor_));
  }
  std::unique_ptr<internal::TimerInterface> GetTimerImplementation() override {
    return std::unique_ptr<internal::TimerInterface>(
        new CTimer(&device_, stream_executor_, timer_fns_));
  }

 private:
  SP_Device device_;
  SP_DeviceFns* device_fns_;
  SP_StreamExecutor* stream_executor_;
  SP_Platform* platform_;
  SP_PlatformFns* platform_fns_;
  SP_TimerFns* timer_fns_;
  std::string platform_name_;
  int visible_device_count_;
};
}  // namespace

CPlatform::CPlatform(SP_Platform platform,
                     void (*destroy_platform)(SP_Platform*),
                     SP_PlatformFns platform_fns,
                     void (*destroy_platform_fns)(SP_PlatformFns*),
                     SP_DeviceFns device_fns, SP_StreamExecutor stream_executor,
                     SP_TimerFns timer_fns)
    : platform_(std::move(platform)),
      destroy_platform_(destroy_platform),
      platform_fns_(std::move(platform_fns)),
      destroy_platform_fns_(destroy_platform_fns),
      device_fns_(std::move(device_fns)),
      stream_executor_(std::move(stream_executor)),
      timer_fns_(std::move(timer_fns)),
      name_(platform.name) {}

CPlatform::~CPlatform() {
  executor_cache_.DestroyAllExecutors();
  platform_fns_.destroy_device_fns(&platform_, &device_fns_);
  platform_fns_.destroy_stream_executor(&platform_, &stream_executor_);
  platform_fns_.destroy_timer_fns(&platform_, &timer_fns_);
  destroy_platform_(&platform_);
  destroy_platform_fns_(&platform_fns_);
}

port::StatusOr<std::unique_ptr<DeviceDescription>>
CPlatform::DescriptionForDevice(int ordinal) const {
  // TODO(annarev): see if we can get StreamExecutor instance
  // and call GetDeviceDescription. executor_cache_.Get would need
  // to be made const for it to work.
  internal::DeviceDescriptionBuilder builder;
  builder.set_name(name_);
  return builder.Build();
}
port::StatusOr<StreamExecutor*> CPlatform::ExecutorForDevice(int ordinal) {
  stream_executor::StreamExecutorConfig config;
  config.ordinal = ordinal;
  return GetExecutor(config);
}
port::StatusOr<StreamExecutor*> CPlatform::ExecutorForDeviceWithPluginConfig(
    int ordinal, const PluginConfig& plugin_config) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = plugin_config;
  return GetExecutor(config);
}
port::StatusOr<StreamExecutor*> CPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}
port::StatusOr<std::unique_ptr<StreamExecutor>> CPlatform::GetUncachedExecutor(
    const StreamExecutorConfig& config) {
  // Fill device creation params
  SE_CreateDeviceParams device_params{SE_CREATE_DEVICE_PARAMS_STRUCT_SIZE};
  SP_Device device{SP_DEVICE_STRUCT_SIZE};
  device_params.device = &device;
  device_params.ext = nullptr;
  device_params.ordinal = config.ordinal;
  OwnedTFStatus c_status(TF_NewStatus());

  // Create Device
  platform_fns_.create_device(&platform_, &device_params, c_status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status.get()));
  TF_RETURN_IF_ERROR(ValidateSPDevice(device));

  // Get Device Count
  int visible_device_count = 0;
  platform_fns_.get_device_count(&platform_, &visible_device_count,
                                 c_status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status.get()));

  auto executor = absl::make_unique<CStreamExecutor>(
      std::move(device), &device_fns_, &stream_executor_, &platform_,
      &platform_fns_, &timer_fns_, name_, visible_device_count);
  auto result = absl::make_unique<StreamExecutor>(this, std::move(executor),
                                                  config.ordinal);
  return result;
}

port::Status InitStreamExecutorPlugin(void* dso_handle) {
  tensorflow::Env* env = tensorflow::Env::Default();

  // Step 1: Load symbol for `TF_InitPlugin`
  void* dso_symbol;
  TF_RETURN_IF_ERROR(
      env->GetSymbolFromLibrary(dso_handle, "SE_InitPlugin", &dso_symbol));

  // Step 2: Call `TF_InitPlugin`
  auto init_fn = reinterpret_cast<SEInitPluginFn>(dso_symbol);
  return InitStreamExecutorPlugin(init_fn);
}

port::Status InitStreamExecutorPlugin(SEInitPluginFn init_fn) {
  SE_PlatformRegistrationParams params{
      SE_PLATFORM_REGISTRATION_PARAMS_STRUCT_SIZE};
  SP_Platform platform{SP_PLATFORM_STRUCT_SIZE};
  SP_PlatformFns platform_fns{SP_PLATFORM_FNS_STRUCT_SIZE};
  params.major_version = SE_MAJOR;
  params.minor_version = SE_MINOR;
  params.patch_version = SE_PATCH;
  params.platform = &platform;
  params.platform_fns = &platform_fns;

  OwnedTFStatus c_status(TF_NewStatus());
  init_fn(&params, c_status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(c_status.get()));
  TF_RETURN_IF_ERROR(ValidateSEPlatformRegistrationParams(params));
  TF_RETURN_IF_ERROR(ValidateSPPlatform(platform));
  TF_RETURN_IF_ERROR(ValidateSPPlatformFns(platform_fns));

  // Fill SP_DeviceFns creation params
  SE_CreateDeviceFnsParams device_fns_params{
      SE_CREATE_DEVICE_FNS_PARAMS_STRUCT_SIZE};
  SP_DeviceFns device_fns{SP_DEVICE_FNS_STRUCT_SIZE};
  device_fns_params.device_fns = &device_fns;

  // Create StreamExecutor
  platform_fns.create_device_fns(&platform, &device_fns_params, c_status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(c_status.get()));
  TF_RETURN_IF_ERROR(ValidateSPDeviceFns(device_fns));

  // Fill stream executor creation params
  SE_CreateStreamExecutorParams se_params{
      SE_CREATE_STREAM_EXECUTOR_PARAMS_STRUCT_SIZE};
  SP_StreamExecutor se{SP_STREAMEXECUTOR_STRUCT_SIZE};
  se_params.stream_executor = &se;

  // Create StreamExecutor
  platform_fns.create_stream_executor(&platform, &se_params, c_status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(c_status.get()));
  TF_RETURN_IF_ERROR(ValidateSPStreamExecutor(se, platform));

  SP_TimerFns timer_fns{SP_TIMER_FNS_STRUCT_SIZE};
  platform_fns.create_timer_fns(&platform, &timer_fns, c_status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(c_status.get()));
  TF_RETURN_IF_ERROR(ValidateSPTimerFns(timer_fns));

  // Register new platform
  std::string platform_name = std::string(platform.name);
  std::unique_ptr<stream_executor::CPlatform> cplatform(
      new stream_executor::CPlatform(
          std::move(platform), params.destroy_platform, std::move(platform_fns),
          params.destroy_platform_fns, std::move(device_fns), std::move(se),
          std::move(timer_fns)));
  SE_CHECK_OK(stream_executor::MultiPlatformManager::RegisterPlatform(
      std::move(cplatform)));

  // TODO(annarev): Add pluggable device registration here.
  // TODO(annarev): Return `use_bfc_allocator` value in some way so that it is
  // available in `PluggableDeviceProcessState` once the latter is checked in.
  return port::Status::OK();
}
}  // namespace stream_executor
