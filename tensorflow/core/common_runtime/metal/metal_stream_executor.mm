/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/metal/metal_stream_executor.h"

#import <Metal/Metal.h>

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/common_runtime/metal/metal_buffer_registry.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

id<MTLDevice> MTLDeviceOf(const SP_Device* device) {
  return static_cast<id<MTLDevice>>(device->device_handle);
}

void Ok(TF_Status* status) { TF_SetStatus(status, TF_OK, ""); }

void Fail(TF_Status* status, TF_Code code, const std::string& message) {
  TF_SetStatus(status, code, message.c_str());
}

// Resolves a device address to the buffer that backs it, failing `status` with
// a diagnosable message rather than crashing if the address is not one we
// handed out.
bool ResolveOrFail(const void* address, const char* what, id<MTLBuffer>* buffer,
                   size_t* offset, TF_Status* status) {
  if (MetalBufferRegistry::Global().Lookup(address, buffer, offset)) return true;
  Fail(status, TF_INVALID_ARGUMENT,
       std::string("Metal: ") + what +
           " does not belong to any live Metal allocation.");
  return false;
}

// Reports a stream that a previous command buffer left in a failed state. Once
// a stream has failed, every later operation on it is unreliable, so callbacks
// surface the original error instead of appearing to succeed.
bool StreamAlreadyFailed(SP_Stream stream, TF_Status* status) {
  absl::MutexLock lock(&stream->mu);
  if (!stream->failed) return false;
  TF_SetStatus(status, TF_INTERNAL,
               ("Metal: stream previously failed: " + stream->failure_message)
                   .c_str());
  return true;
}

/*** ALLOCATION ***/

void Allocate(const SP_Device* device, uint64_t size, int64_t memory_space,
              SP_DeviceMemoryBase* mem) {
  mem->struct_size = SP_DEVICE_MEMORY_BASE_STRUCT_SIZE;
  mem->ext = nullptr;
  mem->opaque = nullptr;
  mem->size = 0;
  mem->payload = 0;

  // Metal returns nil for a zero-length buffer, but core does request
  // zero-byte allocations. Rounding up to one byte keeps a successful
  // zero-byte allocation distinguishable from an out-of-memory failure, which
  // core detects by a null opaque.
  const uint64_t bytes = std::max<uint64_t>(size, 1);

  // Shared storage, never private: the whole backend depends on device
  // addresses being host-addressable. See MetalBufferRegistry.
  id<MTLBuffer> buffer =
      [MTLDeviceOf(device) newBufferWithLength:bytes
                                       options:MTLResourceStorageModeShared];
  if (buffer == nil) return;

  void* address = MetalBufferRegistry::Global().Register(buffer);
  // The registry took its own reference; drop the one newBuffer gave us.
  [buffer release];
  if (address == nullptr) return;

  mem->opaque = address;
  mem->size = size;
}

void Deallocate(const SP_Device* device, SP_DeviceMemoryBase* mem) {
  if (mem == nullptr || mem->opaque == nullptr) return;
  MetalBufferRegistry::Global().Unregister(mem->opaque);
  mem->opaque = nullptr;
  mem->size = 0;
}

void* HostMemoryAllocate(const SP_Device* device, uint64_t size) {
  if (size == 0) return nullptr;
  // Page aligned so the result can be wrapped with newBufferWithBytesNoCopy:
  // without an intermediate staging copy, should a future kernel need it.
  const size_t alignment = static_cast<size_t>(getpagesize());
  const size_t rounded = ((size + alignment - 1) / alignment) * alignment;
  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment, rounded) != 0) return nullptr;
  return ptr;
}

void HostMemoryDeallocate(const SP_Device* device, void* mem) { free(mem); }

void* UnifiedMemoryAllocate(const SP_Device* device, uint64_t bytes) {
  SP_DeviceMemoryBase mem{SP_DEVICE_MEMORY_BASE_STRUCT_SIZE};
  Allocate(device, bytes, /*memory_space=*/0, &mem);
  return mem.opaque;
}

void UnifiedMemoryDeallocate(const SP_Device* device, void* location) {
  MetalBufferRegistry::Global().Unregister(location);
}

TF_Bool GetAllocatorStats(const SP_Device* device, SP_AllocatorStats* stats) {
  const MetalBufferRegistry::Stats snapshot =
      MetalBufferRegistry::Global().GetStats();
  stats->struct_size = SP_ALLOCATORSTATS_STRUCT_SIZE;
  stats->num_allocs = snapshot.num_allocs;
  stats->bytes_in_use = snapshot.bytes_in_use;
  stats->peak_bytes_in_use = snapshot.peak_bytes_in_use;
  stats->largest_alloc_size = snapshot.largest_alloc_size;
  // Metal imposes no fixed per-process budget, so there is no limit to report.
  stats->has_bytes_limit = 0;
  stats->bytes_limit = 0;
  stats->bytes_reserved = 0;
  stats->peak_bytes_reserved = 0;
  stats->has_bytes_reservable_limit = 0;
  stats->bytes_reservable_limit = 0;
  stats->largest_free_block_bytes = 0;
  return true;
}

TF_Bool DeviceMemoryUsage(const SP_Device* device, int64_t* free,
                          int64_t* total) {
  id<MTLDevice> mtl = MTLDeviceOf(device);
  // recommendedMaxWorkingSetSize is the budget Metal will let this process use
  // before it starts paging, which is the closest analogue to "total device
  // memory" on a unified memory system, where physical RAM is shared with the
  // rest of the machine.
  const int64_t budget = static_cast<int64_t>([mtl recommendedMaxWorkingSetSize]);
  const int64_t used = static_cast<int64_t>([mtl currentAllocatedSize]);
  *total = budget;
  *free = std::max<int64_t>(budget - used, 0);
  return true;
}

/*** STREAMS ***/

void CreateStream(const SP_Device* device, SP_Stream* stream,
                  TF_Status* status) {
  id<MTLDevice> mtl = MTLDeviceOf(device);
  id<MTLCommandQueue> queue = [mtl newCommandQueue];
  if (queue == nil) {
    Fail(status, TF_RESOURCE_EXHAUSTED,
         "Metal: could not create an MTLCommandQueue.");
    return;
  }
  id<MTLSharedEvent> event = [mtl newSharedEvent];
  if (event == nil) {
    [queue release];
    Fail(status, TF_RESOURCE_EXHAUSTED,
         "Metal: could not create an MTLSharedEvent for stream ordering.");
    return;
  }

  *stream = new SP_Stream_st(queue, event);
  // SP_Stream_st retained both; drop the references the new* calls gave us.
  [queue release];
  [event release];

  StateOf(device)->AddStream(*stream);
  Ok(status);
}

void DestroyStream(const SP_Device* device, SP_Stream stream) {
  if (stream == nullptr) return;
  // Outstanding command buffers hold references to the queue and the event,
  // but their completion handlers touch the stream itself, so it must outlive
  // them. Drain before freeing.
  uint64_t target = 0;
  {
    absl::MutexLock lock(&stream->mu);
    target = stream->last_enqueued;
  }
  if (target > 0) {
    [stream->order_event waitUntilSignaledValue:target timeoutMS:UINT64_MAX];
  }
  StateOf(device)->RemoveStream(stream);
  delete stream;
}

void CreateStreamDependency(const SP_Device* device, SP_Stream dependent,
                            SP_Stream other, TF_Status* status) {
  ScopedAutoreleasePool pool;
  if (StreamAlreadyFailed(other, status)) return;

  uint64_t other_value = 0;
  {
    absl::MutexLock lock(&other->mu);
    other_value = other->last_enqueued;
  }
  // Nothing has ever been enqueued on `other`, so there is nothing to wait for.
  if (other_value == 0) {
    Ok(status);
    return;
  }

  OrderedCommandBuffer buffer(dependent);
  if (!buffer.ok()) {
    Fail(status, TF_RESOURCE_EXHAUSTED,
         "Metal: could not create a command buffer for the stream dependency.");
    return;
  }
  [buffer.get() encodeWaitForEvent:other->order_event value:other_value];
  buffer.Commit();
  Ok(status);
}

void GetStreamStatus(const SP_Device* device, SP_Stream stream,
                     TF_Status* status) {
  if (StreamAlreadyFailed(stream, status)) return;
  Ok(status);
}

// SP_StreamOptions and the callback that takes it were added to the
// StreamExecutor C API after the last TensorFlow release. An in-tree build
// always has them. A build against an installed TensorFlow may not, which is
// what TF_METAL_NO_STREAM_OPTIONS says; the plugin then simply offers one
// fewer callback, and core falls back to create_stream.
#if !defined(TF_METAL_NO_STREAM_OPTIONS)
void CreateStreamWithOptions(const SP_Device* device,
                             const SP_StreamOptions* options, SP_Stream* stream,
                             TF_Status* status) {
  // Metal command queues have no priority control, so the priority hint in
  // `options` has nothing to map onto and is deliberately ignored.
  CreateStream(device, stream, status);
}
#endif  // TF_METAL_NO_STREAM_OPTIONS

/*** EVENTS ***/

void CreateEvent(const SP_Device* device, SP_Event* event, TF_Status* status) {
  id<MTLSharedEvent> shared_event = [MTLDeviceOf(device) newSharedEvent];
  if (shared_event == nil) {
    Fail(status, TF_RESOURCE_EXHAUSTED,
         "Metal: could not create an MTLSharedEvent.");
    return;
  }
  *event = new SP_Event_st(shared_event);
  [shared_event release];
  Ok(status);
}

void DestroyEvent(const SP_Device* device, SP_Event event) { delete event; }

SE_EventStatus GetEventStatus(const SP_Device* device, SP_Event event) {
  uint64_t target = 0;
  {
    absl::MutexLock lock(&event->mu);
    target = event->target;
  }
  // Never recorded: core treats an unrecorded event as already complete.
  if (target == 0) return SE_EVENT_COMPLETE;
  return event->event.signaledValue >= target ? SE_EVENT_COMPLETE
                                              : SE_EVENT_PENDING;
}

void RecordEvent(const SP_Device* device, SP_Stream stream, SP_Event event,
                 TF_Status* status) {
  ScopedAutoreleasePool pool;
  if (StreamAlreadyFailed(stream, status)) return;

  uint64_t value = 0;
  {
    absl::MutexLock lock(&event->mu);
    value = ++event->target;
  }

  OrderedCommandBuffer buffer(stream);
  if (!buffer.ok()) {
    Fail(status, TF_RESOURCE_EXHAUSTED,
         "Metal: could not create a command buffer to record the event.");
    return;
  }
  [buffer.get() encodeSignalEvent:event->event value:value];
  buffer.Commit();
  Ok(status);
}

void WaitForEvent(const SP_Device* const device, SP_Stream stream,
                  SP_Event event, TF_Status* const status) {
  ScopedAutoreleasePool pool;
  if (StreamAlreadyFailed(stream, status)) return;

  uint64_t target = 0;
  {
    absl::MutexLock lock(&event->mu);
    target = event->target;
  }
  if (target == 0) {
    Ok(status);
    return;
  }

  OrderedCommandBuffer buffer(stream);
  if (!buffer.ok()) {
    Fail(status, TF_RESOURCE_EXHAUSTED,
         "Metal: could not create a command buffer to wait on the event.");
    return;
  }
  [buffer.get() encodeWaitForEvent:event->event value:target];
  buffer.Commit();
  Ok(status);
}

void BlockHostForEvent(const SP_Device* device, SP_Event event,
                       TF_Status* status) {
  uint64_t target = 0;
  {
    absl::MutexLock lock(&event->mu);
    target = event->target;
  }
  if (target == 0) {
    Ok(status);
    return;
  }
  if (![event->event waitUntilSignaledValue:target timeoutMS:UINT64_MAX]) {
    Fail(status, TF_INTERNAL, "Metal: timed out waiting for an event.");
    return;
  }
  Ok(status);
}

/*** TIMERS ***/

void CreateTimer(const SP_Device* device, SP_Timer* timer, TF_Status* status) {
  *timer = new SP_Timer_st();
  Ok(status);
}

void DestroyTimer(const SP_Device* device, SP_Timer timer) { delete timer; }

void StartTimer(const SP_Device* device, SP_Stream stream, SP_Timer timer,
                TF_Status* status) {
  ScopedAutoreleasePool pool;
  OrderedCommandBuffer buffer(stream);
  if (!buffer.ok()) {
    Fail(status, TF_RESOURCE_EXHAUSTED,
         "Metal: could not create a command buffer to start the timer.");
    return;
  }
  // Metal reports timings on the command buffer, so an otherwise empty buffer
  // is what marks the point in the stream. GPUEndTime of this empty buffer is
  // the moment the GPU reached this point.
  [buffer.get() addCompletedHandler:^(id<MTLCommandBuffer> completed) {
    absl::MutexLock lock(&timer->mu);
    timer->start_seconds = completed.GPUEndTime;
    timer->started = true;
  }];
  buffer.Commit();
  Ok(status);
}

void StopTimer(const SP_Device* device, SP_Stream stream, SP_Timer timer,
               TF_Status* status) {
  ScopedAutoreleasePool pool;
  OrderedCommandBuffer buffer(stream);
  if (!buffer.ok()) {
    Fail(status, TF_RESOURCE_EXHAUSTED,
         "Metal: could not create a command buffer to stop the timer.");
    return;
  }
  [buffer.get() addCompletedHandler:^(id<MTLCommandBuffer> completed) {
    absl::MutexLock lock(&timer->mu);
    timer->end_seconds = completed.GPUEndTime;
    timer->stopped = true;
  }];
  // Waited on, so that `nanoseconds` has both timestamps by the time core
  // reads them. The start handler is ordered before this one and has therefore
  // also run.
  buffer.CommitAndWait();
  Ok(status);
}

uint64_t TimerNanoseconds(SP_Timer timer) {
  absl::MutexLock lock(&timer->mu);
  if (!timer->started || !timer->stopped) return 0;
  const double elapsed = timer->end_seconds - timer->start_seconds;
  if (elapsed <= 0.0) return 0;
  return static_cast<uint64_t>(elapsed * 1e9);
}

/*** TRANSFERS ***/

// Every device address is host-addressable (see MetalBufferRegistry), so a
// transfer is a memcpy. What it still needs is stream ordering: the copy must
// not overtake GPU work already enqueued, and later work must not overtake the
// copy. CommitWithHostCompletion provides exactly that, without the staging
// buffer and second copy a blit-based implementation would require.
void EnqueueHostCopy(SP_Stream stream, void* dst, const void* src,
                     uint64_t size, const char* what, TF_Status* status) {
  ScopedAutoreleasePool pool;
  if (size == 0) {
    Ok(status);
    return;
  }
  if (dst == nullptr || src == nullptr) {
    Fail(status, TF_INVALID_ARGUMENT,
         std::string("Metal: null address passed to ") + what + ".");
    return;
  }
  if (StreamAlreadyFailed(stream, status)) return;

  OrderedCommandBuffer buffer(stream);
  if (!buffer.ok()) {
    Fail(status, TF_RESOURCE_EXHAUSTED,
         std::string("Metal: could not create a command buffer for ") + what +
             ".");
    return;
  }
  buffer.CommitWithHostCompletion(^{
    std::memcpy(dst, src, size);
  });
  Ok(status);
}

void MemcpyDToH(const SP_Device* device, SP_Stream stream, void* host_dst,
                const SP_DeviceMemoryBase* device_src, uint64_t size,
                TF_Status* status) {
  EnqueueHostCopy(stream, host_dst, device_src->opaque, size,
                  "a device-to-host copy", status);
}

void MemcpyHToD(const SP_Device* device, SP_Stream stream,
                SP_DeviceMemoryBase* device_dst, const void* host_src,
                uint64_t size, TF_Status* status) {
  EnqueueHostCopy(stream, device_dst->opaque, host_src, size,
                  "a host-to-device copy", status);
}

void MemcpyDToD(const SP_Device* device, SP_Stream stream,
                SP_DeviceMemoryBase* device_dst,
                const SP_DeviceMemoryBase* device_src, uint64_t size,
                TF_Status* status) {
  EnqueueHostCopy(stream, device_dst->opaque, device_src->opaque, size,
                  "a device-to-device copy", status);
}

// The synchronous variants take no stream, so they drain the whole device
// before copying. Draining first is what makes the plain memcpy safe.
void SyncCopy(const SP_Device* device, void* dst, const void* src,
              uint64_t size, TF_Status* status) {
  if (size == 0) {
    Ok(status);
    return;
  }
  if (dst == nullptr || src == nullptr) {
    Fail(status, TF_INVALID_ARGUMENT,
         "Metal: null address passed to a synchronous copy.");
    return;
  }
  StateOf(device)->BlockUntilIdle();
  std::memcpy(dst, src, size);
  Ok(status);
}

void SyncMemcpyDToH(const SP_Device* device, void* host_dst,
                    const SP_DeviceMemoryBase* device_src, uint64_t size,
                    TF_Status* status) {
  SyncCopy(device, host_dst, device_src->opaque, size, status);
}

void SyncMemcpyHToD(const SP_Device* device, SP_DeviceMemoryBase* device_dst,
                    const void* host_src, uint64_t size, TF_Status* status) {
  SyncCopy(device, device_dst->opaque, host_src, size, status);
}

void SyncMemcpyDToD(const SP_Device* device, SP_DeviceMemoryBase* device_dst,
                    const SP_DeviceMemoryBase* device_src, uint64_t size,
                    TF_Status* status) {
  SyncCopy(device, device_dst->opaque, device_src->opaque, size, status);
}

/*** SYNCHRONISATION ***/

void BlockHostUntilDone(const SP_Device* device, SP_Stream stream,
                        TF_Status* status) {
  uint64_t target = 0;
  {
    absl::MutexLock lock(&stream->mu);
    target = stream->last_enqueued;
  }
  if (target > 0 && ![stream->order_event waitUntilSignaledValue:target
                                                       timeoutMS:UINT64_MAX]) {
    Fail(status, TF_INTERNAL, "Metal: timed out draining a stream.");
    return;
  }
  if (StreamAlreadyFailed(stream, status)) return;
  Ok(status);
}

void SynchronizeAllActivity(const SP_Device* device, TF_Status* status) {
  StateOf(device)->BlockUntilIdle();
  Ok(status);
}

/*** FILLS ***/

// Byte fills go through the blit encoder: it is the one operation Metal does
// natively over a whole buffer range, and it keeps large fills off the CPU.
void EncodeFill(SP_Stream stream, SP_DeviceMemoryBase* location, uint8_t value,
                uint64_t size, const char* what, TF_Status* status) {
  ScopedAutoreleasePool pool;
  if (size == 0) {
    Ok(status);
    return;
  }
  if (StreamAlreadyFailed(stream, status)) return;

  id<MTLBuffer> buffer = nil;
  size_t offset = 0;
  if (!ResolveOrFail(location->opaque, what, &buffer, &offset, status)) return;

  // Metal traps on a range that runs past the end of the buffer, so reject it
  // here with a message that names the sizes involved.
  if (offset + size > [buffer length]) {
    Fail(status, TF_INVALID_ARGUMENT,
         std::string("Metal: ") + what + " of " + std::to_string(size) +
             " bytes at offset " + std::to_string(offset) +
             " runs past the end of a " + std::to_string([buffer length]) +
             " byte allocation.");
    return;
  }

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    Fail(status, TF_RESOURCE_EXHAUSTED,
         std::string("Metal: could not create a command buffer for ") + what +
             ".");
    return;
  }
  id<MTLBlitCommandEncoder> encoder = [command_buffer.get() blitCommandEncoder];
  [encoder fillBuffer:buffer range:NSMakeRange(offset, size) value:value];
  [encoder endEncoding];
  command_buffer.Commit();
  Ok(status);
}

void MemZero(const SP_Device* device, SP_Stream stream,
             SP_DeviceMemoryBase* location, uint64_t size, TF_Status* status) {
  EncodeFill(stream, location, 0, size, "a device memory zero fill", status);
}

void Memset(const SP_Device* device, SP_Stream stream,
            SP_DeviceMemoryBase* location, uint8_t pattern, uint64_t size,
            TF_Status* status) {
  EncodeFill(stream, location, pattern, size, "a device memory fill", status);
}

void Memset32(const SP_Device* device, SP_Stream stream,
              SP_DeviceMemoryBase* location, uint32_t pattern, uint64_t size,
              TF_Status* status) {
  ScopedAutoreleasePool pool;
  if (size == 0) {
    Ok(status);
    return;
  }
  if (size % sizeof(uint32_t) != 0) {
    Fail(status, TF_INVALID_ARGUMENT,
         "Metal: memset32 size must be a multiple of 4 bytes.");
    return;
  }
  // A blit fill writes one byte value, so a repeating 32-bit pattern that is
  // not four equal bytes has to be written word by word. Doing it on the host
  // is correct on a unified memory system; a compute shader would be faster
  // for large buffers and is left for the kernel work.
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&pattern);
  if (bytes[0] == bytes[1] && bytes[1] == bytes[2] && bytes[2] == bytes[3]) {
    EncodeFill(stream, location, bytes[0], size, "a device memory 32-bit fill",
               status);
    return;
  }

  if (StreamAlreadyFailed(stream, status)) return;
  OrderedCommandBuffer buffer(stream);
  if (!buffer.ok()) {
    Fail(status, TF_RESOURCE_EXHAUSTED,
         "Metal: could not create a command buffer for a 32-bit fill.");
    return;
  }
  void* destination = location->opaque;
  const uint64_t words = size / sizeof(uint32_t);
  buffer.CommitWithHostCompletion(^{
    uint32_t* out = static_cast<uint32_t*>(destination);
    for (uint64_t i = 0; i < words; ++i) out[i] = pattern;
  });
  Ok(status);
}

/*** HOST CALLBACKS ***/

TF_Bool HostCallback(const SP_Device* device, SP_Stream stream,
                     SE_StatusCallbackFn callback_fn, void* callback_arg) {
  ScopedAutoreleasePool pool;
  OrderedCommandBuffer buffer(stream);
  if (!buffer.ok()) return false;
  buffer.CommitWithHostCompletion(^{
    TF_Status* status = TF_NewStatus();
    callback_fn(callback_arg, status);
    TF_DeleteStatus(status);
  });
  return true;
}

}  // namespace

void PopulateStreamExecutor(SP_StreamExecutor* se) {
  se->struct_size = SP_STREAMEXECUTOR_STRUCT_SIZE;
  se->ext = nullptr;

  se->allocate = Allocate;
  se->deallocate = Deallocate;
  se->host_memory_allocate = HostMemoryAllocate;
  se->host_memory_deallocate = HostMemoryDeallocate;
  se->unified_memory_allocate = UnifiedMemoryAllocate;
  se->unified_memory_deallocate = UnifiedMemoryDeallocate;
  se->get_allocator_stats = GetAllocatorStats;
  se->device_memory_usage = DeviceMemoryUsage;

  se->create_stream = CreateStream;
  se->destroy_stream = DestroyStream;
  se->create_stream_dependency = CreateStreamDependency;
  se->get_stream_status = GetStreamStatus;
#if !defined(TF_METAL_NO_STREAM_OPTIONS)
  se->create_stream_with_options = CreateStreamWithOptions;
#endif  // TF_METAL_NO_STREAM_OPTIONS

  se->create_event = CreateEvent;
  se->destroy_event = DestroyEvent;
  se->get_event_status = GetEventStatus;
  se->record_event = RecordEvent;
  se->wait_for_event = WaitForEvent;

  se->create_timer = CreateTimer;
  se->destroy_timer = DestroyTimer;
  se->start_timer = StartTimer;
  se->stop_timer = StopTimer;

  se->memcpy_dtoh = MemcpyDToH;
  se->memcpy_htod = MemcpyHToD;
  se->memcpy_dtod = MemcpyDToD;
  se->sync_memcpy_dtoh = SyncMemcpyDToH;
  se->sync_memcpy_htod = SyncMemcpyHToD;
  se->sync_memcpy_dtod = SyncMemcpyDToD;

  se->block_host_for_event = BlockHostForEvent;
  se->block_host_until_done = BlockHostUntilDone;
  se->synchronize_all_activity = SynchronizeAllActivity;

  se->mem_zero = MemZero;
  se->memset = Memset;
  se->memset32 = Memset32;

  se->host_callback = HostCallback;
}

void PopulateTimerFns(SP_TimerFns* timer_fns) {
  timer_fns->struct_size = SP_TIMER_FNS_STRUCT_SIZE;
  timer_fns->ext = nullptr;
  timer_fns->nanoseconds = TimerNanoseconds;
}

}  // namespace metal
}  // namespace tensorflow
