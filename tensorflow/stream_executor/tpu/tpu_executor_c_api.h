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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_C_API_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_C_API_H_

#include <stddef.h>
#include <stdint.h>

#include "tensorflow/c/tf_attrtype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/tpu/kernels/tpu_util_c_api.h"
#include "tensorflow/core/tpu/libtftpu.h"

typedef struct SE_Platform SE_Platform;
typedef struct SE_StreamExecutor SE_StreamExecutor;
typedef struct SE_Stream SE_Stream;
typedef struct SE_Event SE_Event;
typedef struct SE_Timer SE_Timer;

typedef struct SE_PlatformId {
  void* id;  // aka stream_executor::Platform::Id
} SE_PlatformId;
typedef struct SE_StreamExecutorConfig SE_StreamExecutorConfig;
typedef struct SE_DeviceOptions SE_DeviceOptions;
typedef SE_Status* (*SE_StatusCallbackFn)(void*);

typedef struct SE_DeviceMemoryBase {
  void* opaque;
  uint64_t size;
  uint64_t payload;
} SE_DeviceMemoryBase;

typedef struct SE_AllocatorStats {
  int64_t num_allocs;
  int64_t bytes_in_use;
  int64_t peak_bytes_in_use;
  int64_t largest_alloc_size;

  bool has_bytes_limit;
  int64_t bytes_limit;

  int64_t bytes_reserved;
  int64_t peak_bytes_reserved;

  bool has_bytes_reservable_limit;
  int64_t bytes_reservable_limit;

  int64_t largest_free_block_bytes;
} SE_AllocatorStats;

typedef struct SE_DeviceDescription {
  char* device_vendor;
  char* platform_version;
  char* driver_version;
  char* runtime_version;
  char* pci_bus_id;
  char* name;

  int64_t thread_dim_limit_x;
  int64_t thread_dim_limit_y;
  int64_t thread_dim_limit_z;
  int64_t block_dim_limit_x;
  int64_t block_dim_limit_y;
  int64_t block_dim_limit_z;

  int64_t threads_per_core_limit;
  int64_t threads_per_block_limit;
  int64_t threads_per_warp;

  int64_t registers_per_core_limit;
  int64_t registers_per_block_limit;

  int64_t device_address_bits;
  int64_t device_memory_size;
  int64_t memory_bandwidth;

  int64_t shared_memory_per_core;
  int64_t shared_memory_per_block;

  float clock_rate_ghz;

  int cuda_compute_capability_major;
  int cuda_compute_capability_minor;

  int rocm_amdgpu_isa_version;

  int numa_node;
  int core_count;
  bool ecc_enabled;
} SE_DeviceDescription;

typedef struct XLA_TransferManager XLA_TransferManager;

typedef struct XLA_ComputationPlacer XLA_ComputationPlacer;

// Represents an XLA shape tree.
// Shapes are flattened in default traversal order.
typedef struct XLA_Shape {
  char* bytes;
  size_t size;
} XLA_Shape;

// Represents a leaf node for a XLA shaped buffer.
typedef struct XLA_ShapedBuffer {
  XLA_Shape on_host_shape;
  XLA_Shape on_device_shape;
  int device_ordinal;

  SE_DeviceMemoryBase* bases;
  size_t count;
} XLA_ShapedBuffer;

// Represents a leaf XLA literal.
typedef struct XLA_Literal {
  char** buffers;
  size_t* sizes;
  size_t count;
  XLA_Shape shape;
} XLA_Literal;

typedef void (*XLA_CallbackFn)(void*);
typedef void (*XLA_StatusCallbackFn)(void*, SE_Status*);

extern "C" {

SE_Platform* TpuPlatform_New();
void TpuPlatform_Free(SE_Platform* platform);
void TpuPlatform_Initialize(SE_Platform* platform, size_t options_size,
                            const char** options_key,
                            const char** options_value, SE_Status* status);
bool TpuPlatform_Initialized(SE_Platform* platform);
SE_StreamExecutor* TpuPlatform_GetExecutor(SE_Platform* platform,
                                           SE_StreamExecutorConfig* config,
                                           SE_Status* status);
SE_PlatformId TpuPlatform_Id(SE_Platform* platform);
int64_t TpuPlatform_VisibleDeviceCount(SE_Platform* platform);
int64_t TpuPlatform_TpuMemoryLimit(SE_Platform* platform);
bool TpuPlatform_ShouldRegisterTpuDeviceToDeviceCopy(SE_Platform* platform);

void TpuExecutor_Init(SE_StreamExecutor* executor, int device_ordinal,
                      SE_DeviceOptions* device_options, SE_Status* status);
void TpuExecutor_Free(SE_StreamExecutor* executor);

int TpuExecutor_PlatformDeviceCount(SE_StreamExecutor* executor);

SE_DeviceMemoryBase TpuExecutor_Allocate(SE_StreamExecutor* executor,
                                         uint64_t size, int64_t memory_space);
void TpuExecutor_Deallocate(SE_StreamExecutor* executor,
                            SE_DeviceMemoryBase* memory);
bool TpuExecutor_GetAllocatorStats(SE_StreamExecutor* executor,
                                   SE_AllocatorStats* stats);
bool TpuExecutor_DeviceMemoryUsage(SE_StreamExecutor* executor, int64_t* free,
                                   int64_t* total);

bool TpuExecutor_AllocateStream(SE_StreamExecutor* executor, SE_Stream* stream);
void TpuExecutor_DeallocateStream(SE_StreamExecutor* executor,
                                  SE_Stream* stream);
bool TpuExecutor_CreateStreamDependency(SE_StreamExecutor* executor,
                                        SE_Stream* dependent, SE_Stream* other);
void TpuExecutor_GetStatus(SE_StreamExecutor* executor, SE_Stream* stream,
                           SE_Status* status);

void TpuExecutor_AllocateEvent(SE_StreamExecutor* executor, SE_Event* event,
                               SE_Status* status);
void TpuExecutor_DeallocateEvent(SE_StreamExecutor* executor, SE_Event* event,
                                 SE_Status* status);
int TpuExecutor_PollForEventStatus(SE_StreamExecutor* executor,
                                   SE_Event* event);
void TpuExecutor_RecordEvent(SE_StreamExecutor* executor, SE_Stream* stream,
                             SE_Event* event, SE_Status* status);
void TpuExecutor_WaitForEvent(SE_StreamExecutor* executor, SE_Stream* stream,
                              SE_Event* event, SE_Status* status);

bool TpuExecutor_AllocateTimer(SE_StreamExecutor* executor, SE_Timer* timer);
void TpuExecutor_DeallocateTimer(SE_StreamExecutor* executor, SE_Timer* timer);
bool TpuExecutor_StartTimer(SE_StreamExecutor* executor, SE_Stream* stream,
                            SE_Timer* timer);
bool TpuExecutor_StopTimer(SE_StreamExecutor* executor, SE_Stream* stream,
                           SE_Timer* timer);

void TpuExecutor_SynchronousMemcpyToHost(SE_StreamExecutor* executor,
                                         void* host_dst,
                                         const SE_DeviceMemoryBase* device_src,
                                         uint64_t size, SE_Status* status);
void TpuExecutor_SynchronousMemcpyFromHost(SE_StreamExecutor* executor,
                                           SE_DeviceMemoryBase* device_dst,
                                           const void* host_src, uint64_t size,
                                           SE_Status* status);
bool TpuExecutor_MemcpyToHost(SE_StreamExecutor* executor, SE_Stream* stream,
                              void* host_dst,
                              const SE_DeviceMemoryBase* device_src,
                              uint64_t size);

bool TpuExecutor_MemcpyFromHost(SE_StreamExecutor* executor, SE_Stream* stream,
                                SE_DeviceMemoryBase* device_dst,
                                const void* host_src, uint64_t size);

void TpuExecutor_EnqueueInfeed(SE_StreamExecutor* executor,
                               int32_t infeed_queue_index, const uint8_t* data,
                               int64_t size, SE_Status* status);
void TpuExecutor_DequeueOutfeed(SE_StreamExecutor* executor,
                                int32_t outfeed_queue_index, uint8_t* data,
                                int64_t size, SE_Status* status);
void TpuExecutor_WaitForInfeedReady(SE_StreamExecutor* executor,
                                    int32_t infeed_queue_index,
                                    SE_Status* status);
void TpuExecutor_WaitForOutfeedReady(SE_StreamExecutor* executor,
                                     int32_t outfeed_queue_index,
                                     SE_Status* status);

void TpuExecutor_BlockHostUntilDone(SE_StreamExecutor* executor,
                                    SE_Stream* stream, SE_Status* status);
void TpuExecutor_BlockUntilDoneOrFailed(SE_StreamExecutor* executor,
                                        SE_Status* status);
void TpuExecutor_SyncAndForgetFailedStreams(SE_StreamExecutor* executor);
bool TpuExecutor_SynchronizeAllActivity(SE_StreamExecutor* executor);

SE_Stream* TpuStream_New(SE_StreamExecutor* parent);
void TpuStream_Free(SE_Stream*);
void* TpuStream_Stream(SE_Stream*);
bool TpuStream_Status(SE_Stream*);
bool TpuStream_IsSameSharedMemoryLocation(SE_Stream*, SE_Stream*);
void TpuStream_TpuEnqueueOnDeviceSendRecvLocal(SE_Stream* stream,
                                               SE_DeviceMemoryBase send_buffer,
                                               SE_DeviceMemoryBase recv_buffer,
                                               SE_Status* status);

SE_Event* TpuEvent_New(SE_StreamExecutor* parent);
void TpuEvent_Free(SE_Event*);

SE_Timer* TpuTimer_New(SE_StreamExecutor* parent);
void TpuTimer_Free(SE_Timer*);
int64_t TpuTimer_Nanoseconds(SE_Timer*);
int64_t TpuTimer_Microseconds(SE_Timer*);

SE_Status* TpuStatus_New();
SE_Status* TpuStatus_Create(int32_t code, const char* msg);
void TpuStatus_Free(SE_Status* status);
const char* TpuStatus_Message(SE_Status* status);
int TpuStatus_Code(SE_Status* status);
bool TpuStatus_Ok(SE_Status* status);

SE_StreamExecutorConfig* TpuStreamExecutorConfig_Default();
void TpuStreamExecutorConfig_SetOrdinal(SE_StreamExecutorConfig*, int ordinal);
void TpuStreamExecutorConfig_Free(SE_StreamExecutorConfig*);

SE_DeviceDescription* TpuDeviceDescription_New();
void TpuDeviceDescription_Free(SE_DeviceDescription* description);
void TpuExecutor_CreateDeviceDescription(SE_StreamExecutor* executor,
                                         SE_DeviceDescription* description,
                                         SE_Status* status);

SE_DeviceOptions* TpuExecutor_NewDeviceOptions(unsigned flags);
void TpuExecutor_FreeDeviceOptions(SE_DeviceOptions* options);

bool TpuExecutor_HostCallback(SE_StreamExecutor* executor, SE_Stream* stream,
                              SE_StatusCallbackFn callback_fn, void* ctx);

XLA_TransferManager* TpuTransferManager_New();
void TpuTransferManager_Free(XLA_TransferManager* manager);
SE_PlatformId TpuTransferManager_PlatformId(XLA_TransferManager* manager);
void TpuTransferManager_HostShapeToDeviceShape(XLA_TransferManager* manager,
                                               XLA_Shape* host_shape,
                                               XLA_Shape* device_shape);
void TpuTransferManager_TransferLiteralToDeviceAsync(
    XLA_TransferManager* manager, SE_Stream* stream, XLA_Literal* literal,
    XLA_ShapedBuffer* device_buffer, SE_Status* status);
void TpuTransferManager_TransferLiteralFromDevice(
    XLA_TransferManager* manager, SE_Stream* stream,
    XLA_ShapedBuffer* device_buffer, XLA_Literal* literal,
    XLA_StatusCallbackFn callback, void* ctx);

int64_t TpuTransferManager_GetByteSizeRequirement(XLA_TransferManager* manager,
                                                  XLA_Shape* shape);
void TpuTransferManager_WriteSingleTupleIndexTable(
    XLA_TransferManager* manager, SE_Stream* stream,
    SE_DeviceMemoryBase* elements, size_t elements_len, XLA_Shape* shape,
    SE_DeviceMemoryBase* region, SE_Status* status);

XLA_ComputationPlacer* TpuComputationPlacer_New();
void TpuComputationPlacer_Free(XLA_ComputationPlacer* placer);

struct TfTpu_ExecutorApiFn {
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_Initialize);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_Initialized);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_GetExecutor);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_Id);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_VisibleDeviceCount);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_TpuMemoryLimit);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_ShouldRegisterTpuDeviceToDeviceCopy);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_Init);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_PlatformDeviceCount);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_Allocate);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_Deallocate);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_GetAllocatorStats);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_DeviceMemoryUsage);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_AllocateStream);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_DeallocateStream);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_CreateStreamDependency);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_GetStatus);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_AllocateEvent);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_DeallocateEvent);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_PollForEventStatus);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_RecordEvent);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_WaitForEvent);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_AllocateTimer);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_DeallocateTimer);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_StartTimer);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_StopTimer);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_SynchronousMemcpyToHost);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_SynchronousMemcpyFromHost);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_MemcpyToHost);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_MemcpyFromHost);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_EnqueueInfeed);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_DequeueOutfeed);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_WaitForInfeedReady);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_WaitForOutfeedReady);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_BlockHostUntilDone);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_BlockUntilDoneOrFailed);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_SyncAndForgetFailedStreams);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_SynchronizeAllActivity);

  TFTPU_ADD_FN_IN_STRUCT(TpuStream_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuStream_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuStream_Stream);
  TFTPU_ADD_FN_IN_STRUCT(TpuStream_Status);
  TFTPU_ADD_FN_IN_STRUCT(TpuStream_IsSameSharedMemoryLocation);
  TFTPU_ADD_FN_IN_STRUCT(TpuStream_TpuEnqueueOnDeviceSendRecvLocal);

  TFTPU_ADD_FN_IN_STRUCT(TpuEvent_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuEvent_Free);

  TFTPU_ADD_FN_IN_STRUCT(TpuTimer_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuTimer_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuTimer_Nanoseconds);
  TFTPU_ADD_FN_IN_STRUCT(TpuTimer_Microseconds);

  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Create);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Message);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Code);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Ok);

  TFTPU_ADD_FN_IN_STRUCT(TpuStreamExecutorConfig_Default);
  TFTPU_ADD_FN_IN_STRUCT(TpuStreamExecutorConfig_SetOrdinal);
  TFTPU_ADD_FN_IN_STRUCT(TpuStreamExecutorConfig_Free);

  TFTPU_ADD_FN_IN_STRUCT(TpuDeviceDescription_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuDeviceDescription_Free);

  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_CreateDeviceDescription);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_NewDeviceOptions);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_FreeDeviceOptions);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_HostCallback);

  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_PlatformId);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_HostShapeToDeviceShape);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_TransferLiteralToDeviceAsync);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_TransferLiteralFromDevice);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_GetByteSizeRequirement);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_WriteSingleTupleIndexTable);

  TFTPU_ADD_FN_IN_STRUCT(TpuComputationPlacer_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuComputationPlacer_Free);
};
}

// extern "C"

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_C_API_H_
