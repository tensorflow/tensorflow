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

#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_C_API_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_C_API_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/libtftpu.h"

extern "C" {

SE_Platform* TpuPlatform_New();
void TpuPlatform_Free(SE_Platform* platform);
void TpuPlatform_Initialize(SE_Platform* platform, TF_Status* status);
bool TpuPlatform_Initialized(SE_Platform* platform);
SE_StreamExecutor* TpuPlatform_GetExecutor(SE_Platform* platform, int ordinal,
                                           TF_Status* status);
SE_PlatformId TpuPlatform_Id(SE_Platform* platform);
int64_t TpuPlatform_VisibleDeviceCount(SE_Platform* platform);
bool TpuPlatform_ShouldRegisterTpuDeviceToDeviceCopy(SE_Platform* platform);
const SE_TpuTopology* TpuPlatform_GetTopologyPtr(SE_Platform* platform);
SE_TpuTopology_Host* TpuPlatform_GetHostLocation(SE_Platform* platform);
TpuRuntimeVersion TpuPlatform_GetRuntimeVersion(SE_Platform* platform);

void TpuExecutor_Init(SE_StreamExecutor* executor, TF_Status* status);
void TpuExecutor_Free(SE_StreamExecutor* executor);

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
                           TF_Status* status);

SE_TpuTopology_Core* TpuExecutor_GetCoreLocation(SE_StreamExecutor* executor);

void TpuExecutor_AllocateEvent(SE_StreamExecutor* executor, SE_Event* event,
                               TF_Status* status);
void TpuExecutor_RecordEvent(SE_StreamExecutor* executor, SE_Stream* stream,
                             SE_Event* event, TF_Status* status);
void TpuExecutor_WaitForEvent(SE_StreamExecutor* executor, SE_Stream* stream,
                              SE_Event* event, TF_Status* status);

void TpuExecutor_SynchronousMemcpyToHost(SE_StreamExecutor* executor,
                                         void* host_dst,
                                         const SE_DeviceMemoryBase* device_src,
                                         uint64_t size, TF_Status* status);
void TpuExecutor_SynchronousMemcpyFromHost(SE_StreamExecutor* executor,
                                           SE_DeviceMemoryBase* device_dst,
                                           const void* host_src, uint64_t size,
                                           TF_Status* status);
void TpuExecutor_MemcpyToHost(SE_StreamExecutor* executor, SE_Stream* stream,
                              void* host_dst,
                              const SE_DeviceMemoryBase* device_src,
                              uint64_t size, TF_Status* status);

void TpuExecutor_MemcpyFromHost(SE_StreamExecutor* executor, SE_Stream* stream,
                                SE_DeviceMemoryBase* device_dst,
                                const void* host_src, uint64_t size,
                                TF_Status* status);

void TpuExecutor_EnqueueInfeed(SE_StreamExecutor* executor,
                               int32_t infeed_queue_index, const uint8_t* data,
                               int64_t size, TF_Status* status);
void TpuExecutor_DequeueOutfeed(SE_StreamExecutor* executor,
                                int32_t outfeed_queue_index, uint8_t* data,
                                int64_t size, TF_Status* status);

void TpuExecutor_BlockHostUntilDone(SE_StreamExecutor* executor,
                                    SE_Stream* stream, TF_Status* status);
bool TpuExecutor_SynchronizeAllActivity(SE_StreamExecutor* executor);

void TpuExecutor_UnloadAllPrograms(SE_StreamExecutor* executor,
                                   TF_Status* status);
void TpuExecutor_EnqueueCompactionOnStreamForHbm(SE_StreamExecutor* executor,
                                                 SE_Stream* compaction_stream,
                                                 TF_Status* status);

SE_Stream* TpuStream_New(SE_StreamExecutor* parent);
void TpuStream_Free(SE_Stream*);
void* TpuStream_Stream(SE_Stream*);
bool TpuStream_Status(SE_Stream*);
bool TpuStream_IsSameSharedMemoryLocation(SE_Stream*, SE_Stream*);
void TpuStream_EnqueueTransferHostToDevice(SE_Stream* stream,
                                           SE_DeviceMemoryBase device_dst,
                                           void* host_src, uint64_t size,
                                           TF_Status* status);
void TpuStream_EnqueueTransferDeviceToHost(SE_Stream* stream,
                                           SE_DeviceMemoryBase device_src,
                                           void* host_dst, uint64_t size,
                                           TF_Status* status);
void TpuStream_TpuEnqueueOnDeviceSendRecvLocal(SE_Stream* stream,
                                               SE_DeviceMemoryBase send_buffer,
                                               SE_DeviceMemoryBase recv_buffer,
                                               TF_Status* status);

SE_Event* TpuEvent_New(SE_StreamExecutor* parent);
void TpuEvent_Free(SE_Event*);

TF_Status* TpuStatus_New();
TF_Status* TpuStatus_Create(int32_t code, const char* msg);
void TpuStatus_Set(TF_Status* status, int32_t code, const char* msg,
                   int32_t len);
void TpuStatus_Free(TF_Status* status);
const char* TpuStatus_Message(TF_Status* status);
int TpuStatus_Code(TF_Status* status);
bool TpuStatus_Ok(TF_Status* status);

SE_DeviceDescription* TpuDeviceDescription_New();
void TpuDeviceDescription_Free(SE_DeviceDescription* description);
void TpuExecutor_CreateDeviceDescription(SE_StreamExecutor* executor,
                                         SE_DeviceDescription* description,
                                         TF_Status* status);

bool TpuExecutor_HostCallback(SE_StreamExecutor* executor, SE_Stream* stream,
                              SE_StatusCallback callback_fn, void* ctx);

XLA_TransferManager* TpuTransferManager_New();
void TpuTransferManager_Free(XLA_TransferManager* manager);
SE_PlatformId TpuTransferManager_PlatformId(XLA_TransferManager* manager);
void TpuTransferManager_HostShapeToDeviceShape(XLA_TransferManager* manager,
                                               XLA_Shape* host_shape,
                                               XLA_Shape* device_shape);
void TpuTransferManager_TransferLiteralToDeviceAsync(
    XLA_TransferManager* manager, SE_Stream* stream, XLA_Literal* literal,
    XLA_ShapedBuffer* device_buffer, TF_Status* status);
void TpuTransferManager_TransferLiteralFromDevice(
    XLA_TransferManager* manager, SE_Stream* stream,
    XLA_ShapedBuffer* device_buffer, XLA_Literal* literal,
    XLA_StatusCallbackFn callback, void* ctx);
int64_t TpuTransferManager_GetByteSizeRequirement(XLA_TransferManager* manager,
                                                  XLA_Shape* shape);
void TpuTransferManager_ChooseCompactLayoutForShape(
    XLA_TransferManager* manager, XLA_Shape* host_shape, XLA_Shape* output,
    TF_Status* status);
bool TpuTransferManager_CanShapedBufferBeAccessedNow(
    XLA_TransferManager* manager, SE_StreamExecutor* executor,
    XLA_ShapedBuffer* device_buffer);
bool TpuTransferManager_CanBufferBeAccessedNow(
    XLA_TransferManager* manager, SE_StreamExecutor* executor,
    SE_DeviceMemoryBase* device_buffer);
void TpuTransferManager_WriteSingleTupleIndexTable(
    XLA_TransferManager* manager, SE_Stream* stream,
    SE_DeviceMemoryBase* elements, size_t elements_len, XLA_Shape* shape,
    SE_DeviceMemoryBase* region, TF_Status* status);
void TpuTransferManager_GetInfeedLayout(XLA_Shape* shape,
                                        XLA_Shape* infeed_shape);
void TpuTransferManager_LinearizeToBuffers(
    XLA_TransferManager* manager, XLA_Literal* c_literal,
    XLA_Shape* c_device_shape, char*** buffers_array, int64_t** buffers_size,
    int64_t* buffers_array_size, TF_Status* status);
void TpuTransferManager_FreeBuffers(char** buffers_array, int64_t* buffers_size,
                                    int64_t buffers_array_size);
void TpuTransferManager_TransferLiteralToInfeed(XLA_TransferManager* manager,
                                                SE_StreamExecutor* executor,
                                                XLA_Literal* c_literal,
                                                TF_Status* status);
void TpuTransferManager_TransferBuffersToInfeed(XLA_TransferManager* manager,
                                                SE_StreamExecutor* executor,
                                                uint32_t** buffers_array,
                                                int64_t* buffers_size_in_uint32,
                                                int64_t buffers_array_size,
                                                TF_Status* status);
void TpuTransferManager_TransferLiteralFromOutfeed(
    XLA_TransferManager* manager, SE_StreamExecutor* executor,
    XLA_Shape* shape /*deprecated*/, XLA_Literal* c_literal, TF_Status* status);
void TpuTransferManager_ResetDevices(XLA_TransferManager* manager,
                                     SE_StreamExecutor** executors,
                                     int64_t num_executors, TF_Status* status);
void TpuTransferManager_ReadDynamicShapes(SE_Stream* stream,
                                          XLA_ShapedBuffer* buffer,
                                          const XLA_Shape& original_shape,
                                          XLA_Shape* updated_shape,
                                          TF_Status* status);

XLA_ComputationPlacer* TpuComputationPlacer_New();
void TpuComputationPlacer_Free(XLA_ComputationPlacer* placer);
// `assignment` should be a preallocated array of size `replicate_count` *
// `computation_count`. The assignment will be constructed as a 2D array where
// assignment[replica][computation] = device_id.
void TpuComputationPlacer_AssignDevices(XLA_ComputationPlacer* placer,
                                        int replica_count,
                                        int computation_count, int* assignment,
                                        TF_Status* status);
void TpuComputationPlacer_AssignLocalDevices(SE_TpuTopology_Host* host,
                                             int replica_count,
                                             int computation_count,
                                             int* assignment,
                                             TF_Status* status);

int TpuTopology_LogicalDevicesPerHost(const SE_TpuTopology* tpu_topology,
                                      TpuCoreTypeEnum tpu_core_type);
int TpuTopology_LogicalDevicesPerChip(const SE_TpuTopology* tpu_topology,
                                      TpuCoreTypeEnum tpu_core_type);
int TpuTopology_HostCount(const SE_TpuTopology* tpu_topology);
int TpuTopology_ChipsPerHost(const SE_TpuTopology* tpu_topology);

int TpuTopology_ChipBounds_X(const SE_TpuTopology* tpu_topology);
int TpuTopology_ChipBounds_Y(const SE_TpuTopology* tpu_topology);
int TpuTopology_ChipBounds_Z(const SE_TpuTopology* tpu_topology);
bool TpuTopology_HasChip(const SE_TpuTopology* tpu_topology, int x, int y,
                         int z);
SE_TpuTopology_Core* TpuTopology_CoreForId(const SE_TpuTopology* tpu_topology,
                                           TpuCoreTypeEnum tpu_core_type,
                                           int id);
SE_TpuTopology_Core* TpuTopology_Core(const SE_TpuTopology* tpu_topology,
                                      TpuCoreTypeEnum tpu_core_type, int x,
                                      int y, int z, int index);
int TpuTopology_NumCores(const SE_TpuTopology* tpu_topology,
                         TpuCoreTypeEnum tpu_core_type);
// 'cores' should be a preallocated array of size TpuTopology_NumCores.
void TpuTopology_Cores(const SE_TpuTopology* tpu_topology,
                       TpuCoreTypeEnum tpu_core_type,
                       SE_TpuTopology_Core** cores);
int TpuTopology_IdForHost(const SE_TpuTopology* tpu_topology, int x, int y,
                          int z);
TpuVersionEnum TpuTopology_Version(const SE_TpuTopology* tpu_topology);
void TpuCoreLocation_ChipCoordinates(SE_TpuTopology_Core* tpu_core_location,
                                     int* x, int* y, int* z);
void TpuCoreLocation_HostCoordinates(SE_TpuTopology_Core* tpu_core_location,
                                     int* x, int* y, int* z);
int TpuCoreLocation_Index(SE_TpuTopology_Core* tpu_core_location);
int TpuCoreLocation_Id(SE_TpuTopology_Core* tpu_core_location);

int TpuHostLocation_Id(SE_TpuTopology_Host* tpu_host_location);
int TpuHostLocation_NumCores(SE_TpuTopology_Host* tpu_host_location,
                             TpuCoreTypeEnum tpu_core_type);
// 'cores' should be a preallocated array of size TpuHostLocation_NumCores.
void TpuHostLocation_Cores(SE_TpuTopology_Host* tpu_host_location,
                           TpuCoreTypeEnum tpu_core_type,
                           SE_TpuTopology_Core** cores);

// Async collective offloading.
// Safe to call multiple times.
void TpuAsyncCollectiveOffloadHelper_Init();
// Must be called after TpuAsyncCollectiveOffloadHelper_Init.
void TpuAsyncCollectiveOffloadHelper_Shutdown();

// C API for XLA::Compiler interface

TFTPU_CAPI_EXPORT Tpu_Compiler* TpuCompiler_New();
TFTPU_CAPI_EXPORT void TpuCompiler_Free(Tpu_Compiler* compiler);

TFTPU_CAPI_EXPORT void TpuCompiler_RunHloPasses(
    Tpu_Compiler* compiler, XLA_HloModule* se_hlo_module,
    SE_StreamExecutor* stream_executor, SE_DeviceMemoryAllocator* allocator,
    XLA_HloModule* result, TF_Status* status);

TFTPU_CAPI_EXPORT void TpuCompiler_RunBackend(
    Tpu_Compiler* compiler, XLA_HloModule* se_hlo_module,
    SE_StreamExecutor* stream_executor, SE_DeviceMemoryAllocator* allocator,
    SE_Executable** result, TF_Status* status);

TFTPU_CAPI_EXPORT void TpuCompiler_Compile(
    Tpu_Compiler* compiler, XLA_HloModuleGroup* se_hlo_module_group,
    SE_StreamExecutorList* stream_exec_lists, int num_lists,
    SE_DeviceMemoryAllocator* allocator, SE_Executable** executables,
    TF_Status* status);

TFTPU_CAPI_EXPORT int64_t TpuCompiler_ShapeSize(Tpu_Compiler* compiler,
                                                XLA_Shape* c_shape);

TFTPU_CAPI_EXPORT void TpuCompiler_DefaultDeviceShapeRepresentation(
    Tpu_Compiler* compiler, XLA_Shape* host_shape, XLA_Shape* device_shape);

TFTPU_CAPI_EXPORT void TpuExecutable_ExecuteAsyncOnStream(
    SE_Executable* executable, SE_ExecutableRunOptions* se_options,
    SE_ExecutionInput** se_arguments, int se_arguments_size,
    SE_ExecutionOutput* se_output, TF_Status* status);

// This frees the XLA_ShapeIndex* array allocated when se_output is returned by
// TpuExecutable_ExecuteAsyncOnStream.
TFTPU_CAPI_EXPORT void TpuExecutable_FreeXlaShapeIndexArray(
    XLA_ShapeIndex* array);

// This frees the SE_MaybeOwningDeviceMemory* array allocated when se_output is
// returned by TpuExecutable_ExecuteAsyncOnStream.
// Note that this only frees the heap-allocated array itself, and does not
// free any of the underlying device memory.
TFTPU_CAPI_EXPORT void TpuExecutable_FreeMaybeOwningDeviceMemoryArray(
    SE_MaybeOwningDeviceMemory* array);

TFTPU_CAPI_EXPORT void TpuExecutable_Fingerprint(SE_Executable* executable,
                                                 const char** fingerprint,
                                                 size_t* size);

// The serialization format is not guaranteed to be stable over time and has no
// compatibility guarantees (i.e. this is not a suitable long-term storage
// format). TpuExecutableSerialize_FreeHandle should be called after 'handle' is
// no longer needed. 'handle' is set to nullptr on error.
TFTPU_CAPI_EXPORT void TpuExecutable_Serialize(
    SE_Executable* executable, SE_ExecutableSerializationHandle** handle,
    TF_Status* status);

// Returns the size of the serialized executable in bytes, i.e. the size of the
// array that should be passed to TpuExecutableSerialize_WriteToArray. `handle`
// must be non-null.
TFTPU_CAPI_EXPORT size_t
TpuExecutableSerialize_GetByteSize(SE_ExecutableSerializationHandle* handle);

// Writes the serialized executable to `serialized`, which must be of size
// `serialized_size`. `serialized_size` should must be at least
// `TpuExecutableSerialize_GetByteSize(handle)`. `handle` must be non-null.
TFTPU_CAPI_EXPORT void TpuExecutableSerialize_WriteToArray(
    SE_ExecutableSerializationHandle* handle, int serialized_size,
    uint8_t* serialized, TF_Status* status);

// Safe to call if 'handle' is null.
TFTPU_CAPI_EXPORT void TpuExecutableSerialize_FreeHandle(
    SE_ExecutableSerializationHandle* handle);

TFTPU_CAPI_EXPORT void TpuExecutable_Deserialize(int serialized_size,
                                                 const uint8_t* serialized,
                                                 SE_Executable** executable,
                                                 TF_Status* status);

// Caller is responsible for freeing the returned module's proto and its
// config's proto.
TFTPU_CAPI_EXPORT XLA_HloModule
TpuExecutable_HloModule(SE_Executable* executable);

TFTPU_CAPI_EXPORT void TpuExecutable_Free(SE_Executable*);

// Converts an XLA `Shape` into its equivalent TPU `Shape` representation.
TFTPU_CAPI_EXPORT void XlaShapeToTpuShapeRepresentation(
    XLA_Shape* serialized_xla_shape, int data_type, bool use_fast_memory,
    XLA_Shape* serialized_tpu_shape, TF_Status* status);

TFTPU_CAPI_EXPORT void XlaShapeToTpuPaddedShape(XLA_Shape* serialized_xla_shape,
                                                XLA_Shape* padded_shape,
                                                TF_Status* status);

struct TfTpu_ExecutorApiFn {
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_Initialize);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_Initialized);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_GetExecutor);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_Id);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_VisibleDeviceCount);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_ShouldRegisterTpuDeviceToDeviceCopy);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_GetTopologyPtr);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_GetHostLocation);
  TFTPU_ADD_FN_IN_STRUCT(TpuPlatform_GetRuntimeVersion);

  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_Init);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_Allocate);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_Deallocate);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_GetAllocatorStats);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_DeviceMemoryUsage);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_AllocateStream);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_DeallocateStream);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_CreateStreamDependency);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_GetStatus);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_GetCoreLocation);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_AllocateEvent);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_RecordEvent);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_WaitForEvent);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_SynchronousMemcpyToHost);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_SynchronousMemcpyFromHost);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_MemcpyToHost);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_MemcpyFromHost);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_EnqueueInfeed);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_DequeueOutfeed);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_BlockHostUntilDone);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_SynchronizeAllActivity);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_UnloadAllPrograms);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_EnqueueCompactionOnStreamForHbm);

  TFTPU_ADD_FN_IN_STRUCT(TpuStream_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuStream_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuStream_Stream);
  TFTPU_ADD_FN_IN_STRUCT(TpuStream_Status);
  TFTPU_ADD_FN_IN_STRUCT(TpuStream_IsSameSharedMemoryLocation);
  TFTPU_ADD_FN_IN_STRUCT(TpuStream_EnqueueTransferHostToDevice);
  TFTPU_ADD_FN_IN_STRUCT(TpuStream_EnqueueTransferDeviceToHost);
  TFTPU_ADD_FN_IN_STRUCT(TpuStream_TpuEnqueueOnDeviceSendRecvLocal);

  TFTPU_ADD_FN_IN_STRUCT(TpuEvent_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuEvent_Free);

  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Create);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Set);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Message);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Code);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Ok);

  TFTPU_ADD_FN_IN_STRUCT(TpuDeviceDescription_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuDeviceDescription_Free);

  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_CreateDeviceDescription);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutor_HostCallback);

  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_PlatformId);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_HostShapeToDeviceShape);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_TransferLiteralToDeviceAsync);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_TransferLiteralFromDevice);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_GetByteSizeRequirement);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_ChooseCompactLayoutForShape);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_CanShapedBufferBeAccessedNow);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_CanBufferBeAccessedNow);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_WriteSingleTupleIndexTable);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_GetInfeedLayout);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_LinearizeToBuffers);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_FreeBuffers);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_TransferLiteralToInfeed);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_TransferBuffersToInfeed);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_TransferLiteralFromOutfeed);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_ResetDevices);
  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_ReadDynamicShapes);

  TFTPU_ADD_FN_IN_STRUCT(TpuComputationPlacer_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuComputationPlacer_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuComputationPlacer_AssignDevices);
  TFTPU_ADD_FN_IN_STRUCT(TpuComputationPlacer_AssignLocalDevices);

  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_LogicalDevicesPerHost);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_LogicalDevicesPerChip);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_HostCount);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_ChipsPerHost);

  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_ChipBounds_X);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_ChipBounds_Y);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_ChipBounds_Z);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_HasChip);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_CoreForId);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_Core);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_NumCores);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_Cores);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_IdForHost);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_Version);

  TFTPU_ADD_FN_IN_STRUCT(TpuCoreLocation_ChipCoordinates);
  TFTPU_ADD_FN_IN_STRUCT(TpuCoreLocation_HostCoordinates);
  TFTPU_ADD_FN_IN_STRUCT(TpuCoreLocation_Index);
  TFTPU_ADD_FN_IN_STRUCT(TpuCoreLocation_Id);

  TFTPU_ADD_FN_IN_STRUCT(TpuHostLocation_Id);
  TFTPU_ADD_FN_IN_STRUCT(TpuHostLocation_NumCores);
  TFTPU_ADD_FN_IN_STRUCT(TpuHostLocation_Cores);

  TFTPU_ADD_FN_IN_STRUCT(TpuCompiler_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompiler_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompiler_RunHloPasses);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompiler_RunBackend);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompiler_Compile);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompiler_ShapeSize);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompiler_DefaultDeviceShapeRepresentation);

  TFTPU_ADD_FN_IN_STRUCT(TpuExecutable_ExecuteAsyncOnStream);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutable_FreeXlaShapeIndexArray);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutable_FreeMaybeOwningDeviceMemoryArray);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutable_Fingerprint);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutable_Serialize);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutableSerialize_GetByteSize);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutableSerialize_WriteToArray);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutableSerialize_FreeHandle);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutable_Deserialize);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutable_HloModule);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutable_Free);

  TFTPU_ADD_FN_IN_STRUCT(XlaShapeToTpuShapeRepresentation);
  TFTPU_ADD_FN_IN_STRUCT(XlaShapeToTpuPaddedShape);

  TFTPU_ADD_FN_IN_STRUCT(TpuAsyncCollectiveOffloadHelper_Init);
  TFTPU_ADD_FN_IN_STRUCT(TpuAsyncCollectiveOffloadHelper_Shutdown);
};
}

// extern "C"

#endif  // XLA_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_C_API_H_
