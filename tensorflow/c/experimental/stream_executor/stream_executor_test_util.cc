/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/stream_executor/stream_executor_test_util.h"

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"

namespace stream_executor {
namespace test_util {

/*** Functions for creating SP_StreamExecutor ***/
void Allocate(const SP_Device* const device, uint64_t size,
              int64_t memory_space, SP_DeviceMemoryBase* const mem) {}
void Deallocate(const SP_Device* const device, SP_DeviceMemoryBase* const mem) {
}
void* HostMemoryAllocate(const SP_Device* const device, uint64_t size) {
  return nullptr;
}
void HostMemoryDeallocate(const SP_Device* const device, void* mem) {}
TF_Bool GetAllocatorStats(const SP_Device* const device,
                          SP_AllocatorStats* const stats) {
  return true;
}
TF_Bool DeviceMemoryUsage(const SP_Device* const device, int64_t* const free,
                          int64_t* const total) {
  return true;
}
void CreateStream(const SP_Device* const device, SP_Stream* stream,
                  TF_Status* const status) {
  *stream = nullptr;
}
void DestroyStream(const SP_Device* const device, SP_Stream stream) {}
void CreateStreamDependency(const SP_Device* const device, SP_Stream dependent,
                            SP_Stream other, TF_Status* const status) {}
void GetStreamStatus(const SP_Device* const device, SP_Stream stream,
                     TF_Status* const status) {}
void CreateEvent(const SP_Device* const device, SP_Event* event,
                 TF_Status* const status) {
  *event = nullptr;
}
void DestroyEvent(const SP_Device* const device, SP_Event event) {}
SE_EventStatus GetEventStatus(const SP_Device* const device, SP_Event event) {
  return SE_EVENT_UNKNOWN;
}
void RecordEvent(const SP_Device* const device, SP_Stream stream,
                 SP_Event event, TF_Status* const status) {}
void WaitForEvent(const SP_Device* const device, SP_Stream stream,
                  SP_Event event, TF_Status* const status) {}
void CreateTimer(const SP_Device* const device, SP_Timer* timer,
                 TF_Status* const status) {}
void DestroyTimer(const SP_Device* const device, SP_Timer timer) {}
void StartTimer(const SP_Device* const device, SP_Stream stream, SP_Timer timer,
                TF_Status* const status) {}
void StopTimer(const SP_Device* const device, SP_Stream stream, SP_Timer timer,
               TF_Status* const status) {}
void MemcpyDToH(const SP_Device* const device, SP_Stream stream, void* host_dst,
                const SP_DeviceMemoryBase* const device_src, uint64_t size,
                TF_Status* const status) {}
void MemcpyHToD(const SP_Device* const device, SP_Stream stream,
                SP_DeviceMemoryBase* const device_dst, const void* host_src,
                uint64_t size, TF_Status* const status) {}
void SyncMemcpyDToH(const SP_Device* const device, void* host_dst,
                    const SP_DeviceMemoryBase* const device_src, uint64_t size,
                    TF_Status* const status) {}
void SyncMemcpyHToD(const SP_Device* const device,
                    SP_DeviceMemoryBase* const device_dst, const void* host_src,
                    uint64_t size, TF_Status* const status) {}
void BlockHostForEvent(const SP_Device* const device, SP_Event event,
                       TF_Status* const status) {}
void SynchronizeAllActivity(const SP_Device* const device,
                            TF_Status* const status) {}
TF_Bool HostCallback(const SP_Device* const device, SP_Stream stream,
                     SE_StatusCallbackFn const callback_fn,
                     void* const callback_arg) {
  return true;
}

void MemZero(const SP_Device* device, SP_Stream stream,
             SP_DeviceMemoryBase* location, uint64_t size, TF_Status* status) {}

void Memset(const SP_Device* device, SP_Stream stream,
            SP_DeviceMemoryBase* location, uint8_t pattern, uint64_t size,
            TF_Status* status) {}

void Memset32(const SP_Device* device, SP_Stream stream,
              SP_DeviceMemoryBase* location, uint32_t pattern, uint64_t size,
              TF_Status* status) {}

void PopulateDefaultStreamExecutor(SP_StreamExecutor* se) {
  *se = {SP_STREAMEXECUTOR_STRUCT_SIZE};
  se->allocate = Allocate;
  se->deallocate = Deallocate;
  se->host_memory_allocate = HostMemoryAllocate;
  se->host_memory_deallocate = HostMemoryDeallocate;
  se->get_allocator_stats = GetAllocatorStats;
  se->device_memory_usage = DeviceMemoryUsage;
  se->create_stream = CreateStream;
  se->destroy_stream = DestroyStream;
  se->create_stream_dependency = CreateStreamDependency;
  se->get_stream_status = GetStreamStatus;
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
  se->sync_memcpy_dtoh = SyncMemcpyDToH;
  se->sync_memcpy_htod = SyncMemcpyHToD;
  se->block_host_for_event = BlockHostForEvent;
  se->synchronize_all_activity = SynchronizeAllActivity;
  se->host_callback = HostCallback;
  se->mem_zero = MemZero;
  se->memset = Memset;
  se->memset32 = Memset32;
}

void PopulateDefaultDeviceFns(SP_DeviceFns* device_fns) {
  *device_fns = {SP_DEVICE_FNS_STRUCT_SIZE};
}

/*** Functions for creating SP_TimerFns ***/
uint64_t Nanoseconds(SP_Timer timer) { return timer->timer_id; }

void PopulateDefaultTimerFns(SP_TimerFns* timer_fns) {
  timer_fns->nanoseconds = Nanoseconds;
}

/*** Functions for creating SP_Platform ***/
void CreateTimerFns(const SP_Platform* platform, SP_TimerFns* timer_fns,
                    TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  PopulateDefaultTimerFns(timer_fns);
}
void DestroyTimerFns(const SP_Platform* platform, SP_TimerFns* timer_fns) {}

void CreateStreamExecutor(const SP_Platform* platform,
                          SE_CreateStreamExecutorParams* params,
                          TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  PopulateDefaultStreamExecutor(params->stream_executor);
}
void DestroyStreamExecutor(const SP_Platform* platform, SP_StreamExecutor* se) {
}
void GetDeviceCount(const SP_Platform* platform, int* device_count,
                    TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  *device_count = kDeviceCount;
}
void CreateDevice(const SP_Platform* platform, SE_CreateDeviceParams* params,
                  TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  params->device->struct_size = {SP_DEVICE_STRUCT_SIZE};
}
void DestroyDevice(const SP_Platform* platform, SP_Device* device) {}

void CreateDeviceFns(const SP_Platform* platform,
                     SE_CreateDeviceFnsParams* params, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  params->device_fns->struct_size = {SP_DEVICE_FNS_STRUCT_SIZE};
}
void DestroyDeviceFns(const SP_Platform* platform, SP_DeviceFns* device_fns) {}

void PopulateDefaultPlatform(SP_Platform* platform,
                             SP_PlatformFns* platform_fns) {
  *platform = {SP_PLATFORM_STRUCT_SIZE};
  platform->name = kDeviceName;
  platform->type = kDeviceType;
  platform_fns->get_device_count = GetDeviceCount;
  platform_fns->create_device = CreateDevice;
  platform_fns->destroy_device = DestroyDevice;
  platform_fns->create_device_fns = CreateDeviceFns;
  platform_fns->destroy_device_fns = DestroyDeviceFns;
  platform_fns->create_stream_executor = CreateStreamExecutor;
  platform_fns->destroy_stream_executor = DestroyStreamExecutor;
  platform_fns->create_timer_fns = CreateTimerFns;
  platform_fns->destroy_timer_fns = DestroyTimerFns;
}

/*** Functions for creating SE_PlatformRegistrationParams ***/
void DestroyPlatform(SP_Platform* platform) {}
void DestroyPlatformFns(SP_PlatformFns* platform_fns) {}

void PopulateDefaultPlatformRegistrationParams(
    SE_PlatformRegistrationParams* const params) {
  PopulateDefaultPlatform(params->platform, params->platform_fns);
  params->destroy_platform = DestroyPlatform;
  params->destroy_platform_fns = DestroyPlatformFns;
}

}  // namespace test_util
}  // namespace stream_executor
