/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_CLIENT_LIBTPU_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_CLIENT_LIBTPU_H_

#include <stdbool.h>
#include <stdint.h>

#define TPUDRIVER_CAPI_EXPORT __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

// ------------------- TPU Driver Support -----------------------

struct TpuDriverFn;

typedef struct TpuDriver TpuDriver;

typedef struct TpuEvent TpuEvent;

typedef struct TpuBufferHandleInternal TpuBufferHandleInternal;

typedef struct TpuCompiledProgramHandleInternal
    TpuCompiledProgramHandleInternal;

typedef struct TpuLoadedProgramHandleInternal TpuLoadedProgramHandleInternal;

typedef struct TpuBufferHandle {
  TpuBufferHandleInternal* internal_handle;
  TpuEvent* event;
  int64_t size_in_bytes;
} TpuBufferHandle;

typedef struct TpuCompiledProgramHandle {
  TpuCompiledProgramHandleInternal* internal_handle;
  TpuEvent* event;
} TpuCompiledProgramHandle;

typedef struct TpuLoadedProgramHandle {
  TpuLoadedProgramHandleInternal* internal_handle;
  TpuEvent* event;
} TpuLoadedProgramHandle;

// HloProto is a serialized xla::HloProto buffer.
typedef struct HloProto {
  void* buffer;
  int32_t size;
} HloProto;

// DeviceAssignment is a serialized xla::DeviceAssignmentProto buffer.
typedef struct DeviceAssignment {
  void* bytes;
  int32_t size;
} DeviceAssignment;

typedef struct TpuStatus {
  int32_t code;
  char* msg;
} TpuStatus;

typedef struct CompiledProgramShape {
  struct TpuStatus* status;
  void* bytes;
  int32_t size;
} CompiledProgramShape;

typedef struct TpuAllocationShape {
  void* bytes;
  int32_t size;
} TpuAllocationShape;

typedef struct TpuSystemInfo {
  void* bytes;
  int32_t size;
} TpuSystemInfo;

typedef void(PrototypeTpuDriver_Initialize)(struct TpuDriverFn* driver_fn,
                                            bool initialize);
typedef struct TpuDriver*(PrototypeTpuDriver_Open)(const char* worker);
typedef void(PrototypeTpuDriver_Close)(struct TpuDriver* driver);
typedef struct TpuStatus*(PrototypeTpuDriver_Reset)(struct TpuDriver* driver);

typedef struct TpuSystemInfo*(PrototypeTpuDriver_QuerySystemInfo)(
    struct TpuDriver* driver);

typedef void(PrototypeTpuDriver_FreeSystemInfo)(struct TpuSystemInfo* info);

// TODO(frankchn): Make this not a hard-coded constant.
const int32_t MemoryRegion_HBM = 1;

typedef int64_t(PrototypeTpuDriver_ComputeLinearizedBytesFromShape)(
    struct TpuDriver* driver, const struct TpuAllocationShape shape);

typedef struct TpuStatus*(PrototypeTpuDriver_LinearizeShape)(
    struct TpuDriver* driver, void* dst, const void* src,
    const struct TpuAllocationShape shape);

typedef struct TpuStatus*(PrototypeTpuDriver_DelinearizeShape)(
    struct TpuDriver* driver, void* dst, const void* src,
    const struct TpuAllocationShape shape);

typedef struct TpuCompiledProgramHandle*(PrototypeTpuDriver_CompileProgram)(
    struct TpuDriver* driver, const struct HloProto hlo_proto,
    int32_t num_replicas, int32_t eventc, struct TpuEvent** eventv);

typedef struct TpuCompiledProgramHandle*(
    PrototypeTpuDriver_CompileProgramFromText)(struct TpuDriver* driver,
                                               const char* hlo_text,
                                               int32_t num_replicas,
                                               int32_t eventc,
                                               struct TpuEvent** eventv);

/* Note: We are not responsible for freeing the event within the
 * TpuCompiledProgramHandle. You have to call FreeEvent separately to ensure
 * that memory does not leak.
 */
typedef void(PrototypeTpuDriver_FreeCompiledProgramHandle)(
    struct TpuCompiledProgramHandle* handle);

typedef struct TpuLoadedProgramHandle*(PrototypeTpuDriver_LoadProgram)(
    struct TpuDriver* driver, int32_t core_id,
    const struct TpuCompiledProgramHandle* compiled_program_handle,
    int32_t eventc, struct TpuEvent** eventv);

/* Note: We are not responsible for freeing the event within the
 * TpuLoadedProgramHandle. You have to call FreeEvent separately to ensure that
 * memory does not leak.
 */
typedef struct TpuEvent*(PrototypeTpuDriver_UnloadProgram)(
    struct TpuDriver* driver,
    struct TpuLoadedProgramHandle* loaded_program_handle, int32_t eventc,
    struct TpuEvent** eventv);

typedef struct TpuEvent*(PrototypeTpuDriver_ExecuteProgram)(
    struct TpuDriver* driver, struct TpuLoadedProgramHandle* handle,
    int32_t inputc, struct TpuBufferHandle** input_buffer_handle,
    int32_t outputc, struct TpuBufferHandle** output_buffer_handle,
    struct DeviceAssignment device_assignment, int32_t eventc,
    struct TpuEvent** eventv);

typedef struct TpuBufferHandle*(PrototypeTpuDriver_AllocateTuple)(
    struct TpuDriver* driver, int32_t core_id, int32_t memory_region,
    int32_t bufferc, struct TpuBufferHandle** buffer_handle, int32_t eventc,
    struct TpuEvent** eventv);

typedef struct TpuBufferHandle*(PrototypeTpuDriver_Allocate)(
    struct TpuDriver* driver, int32_t core_id, int32_t memory_region,
    int64_t num_bytes, int32_t eventc, struct TpuEvent** eventv);

typedef struct TpuBufferHandle*(PrototypeTpuDriver_AllocateShape)(
    struct TpuDriver* driver, int32_t core_id, int32_t memory_region,
    const struct TpuAllocationShape shape, int32_t eventc,
    struct TpuEvent** eventv);

/* Note: We are not responsible for freeing the event within the
 * TpuBufferHandle. You have to call FreeEvent separately to ensure that memory
 * does not leak.
 */
typedef struct TpuEvent*(PrototypeTpuDriver_Deallocate)(
    struct TpuDriver* driver, struct TpuBufferHandle* buffer_handle,
    int32_t eventc, struct TpuEvent** eventv);

typedef struct TpuEvent*(PrototypeTpuDriver_TransferToDevice)(
    struct TpuDriver* driver, const void* src, struct TpuBufferHandle* dst,
    int32_t eventc, struct TpuEvent** eventv);

typedef struct TpuEvent*(PrototypeTpuDriver_TransferFromDevice)(
    struct TpuDriver* driver, struct TpuBufferHandle* src, void* dst,
    int32_t eventc, struct TpuEvent** eventv);

typedef struct TpuEvent*(PrototypeTpuDriver_TransferFromDeviceToDevice)(
    struct TpuDriver* driver, struct TpuBufferHandle* src,
    struct TpuBufferHandle* dst, int32_t eventc, struct TpuEvent** eventv);

typedef struct CompiledProgramShape*(
    PrototypeTpuDriver_GetCompiledProgramShape)(
    struct TpuCompiledProgramHandle* handle);

typedef void(PrototypeTpuDriver_FreeCompiledProgramShape)(
    struct CompiledProgramShape* shape);

typedef void(PrototypeTpuDriver_EventAddCallback)(
    struct TpuEvent* event,
    void (*callback_fn)(struct TpuStatus*, void* additional_info),
    void* additional_info);

typedef struct TpuStatus*(PrototypeTpuDriver_EventAwait)(struct TpuEvent* event,
                                                         int64_t timeout_in_us);

typedef void(PrototypeTpuDriver_FreeEvent)(struct TpuEvent* event);

typedef void(PrototypeTpuDriver_FreeStatus)(struct TpuStatus* status);

typedef const char*(PrototypeTpuDriver_Version)();

TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_Initialize TpuDriver_Initialize;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_Open TpuDriver_Open;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_Close TpuDriver_Close;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_Reset TpuDriver_Reset;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_QuerySystemInfo
    TpuDriver_QuerySystemInfo;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_FreeSystemInfo
    TpuDriver_FreeSystemInfo;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_ComputeLinearizedBytesFromShape
    TpuDriver_ComputeLinearizedBytesFromShape;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_LinearizeShape
    TpuDriver_LinearizeShape;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_DelinearizeShape
    TpuDriver_DelinearizeShape;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_CompileProgram
    TpuDriver_CompileProgram;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_CompileProgramFromText
    TpuDriver_CompileProgramFromText;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_FreeCompiledProgramHandle
    TpuDriver_FreeCompiledProgramHandle;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_LoadProgram
    TpuDriver_LoadProgram;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_UnloadProgram
    TpuDriver_UnloadProgram;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_ExecuteProgram
    TpuDriver_ExecuteProgram;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_AllocateTuple
    TpuDriver_AllocateTuple;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_Allocate TpuDriver_Allocate;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_AllocateShape
    TpuDriver_AllocateShape;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_Deallocate TpuDriver_Deallocate;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_TransferToDevice
    TpuDriver_TransferToDevice;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_TransferFromDevice
    TpuDriver_TransferFromDevice;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_TransferFromDeviceToDevice
    TpuDriver_TransferFromDeviceToDevice;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_GetCompiledProgramShape
    TpuDriver_GetCompiledProgramShape;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_FreeCompiledProgramShape
    TpuDriver_FreeCompiledProgramShape;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_EventAddCallback
    TpuDriver_EventAddCallback;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_EventAwait TpuDriver_EventAwait;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_FreeEvent TpuDriver_FreeEvent;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_FreeStatus TpuDriver_FreeStatus;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_Version TpuDriver_Version;

#ifdef __cplusplus
}
#endif

struct TpuDriverFn {
  PrototypeTpuDriver_Open* TpuDriver_Open;                          // NOLINT
  PrototypeTpuDriver_Close* TpuDriver_Close;                        // NOLINT
  PrototypeTpuDriver_Reset* TpuDriver_Reset;                        // NOLINT
  PrototypeTpuDriver_ComputeLinearizedBytesFromShape*
      TpuDriver_ComputeLinearizedBytesFromShape;                    // NOLINT
  PrototypeTpuDriver_QuerySystemInfo* TpuDriver_QuerySystemInfo;    // NOLINT
  PrototypeTpuDriver_FreeSystemInfo* TpuDriver_FreeSystemInfo;      // NOLINT
  PrototypeTpuDriver_LinearizeShape* TpuDriver_LinearizeShape;      // NOLINT
  PrototypeTpuDriver_DelinearizeShape* TpuDriver_DelinearizeShape;  // NOLINT
  PrototypeTpuDriver_CompileProgram* TpuDriver_CompileProgram;      // NOLINT
  PrototypeTpuDriver_CompileProgramFromText*
      TpuDriver_CompileProgramFromText;                             // NOLINT
  PrototypeTpuDriver_FreeCompiledProgramHandle*
      TpuDriver_FreeCompiledProgramHandle;                          // NOLINT
  PrototypeTpuDriver_LoadProgram* TpuDriver_LoadProgram;            // NOLINT
  PrototypeTpuDriver_UnloadProgram* TpuDriver_UnloadProgram;        // NOLINT
  PrototypeTpuDriver_ExecuteProgram* TpuDriver_ExecuteProgram;      // NOLINT
  PrototypeTpuDriver_AllocateTuple* TpuDriver_AllocateTuple;        // NOLINT
  PrototypeTpuDriver_Allocate* TpuDriver_Allocate;                  // NOLINT
  PrototypeTpuDriver_AllocateShape* TpuDriver_AllocateShape;        // NOLINT
  PrototypeTpuDriver_Deallocate* TpuDriver_Deallocate;              // NOLINT
  PrototypeTpuDriver_TransferToDevice* TpuDriver_TransferToDevice;  // NOLINT
  PrototypeTpuDriver_TransferFromDevice*
      TpuDriver_TransferFromDevice;  // NOLINT
  PrototypeTpuDriver_TransferFromDeviceToDevice*
      TpuDriver_TransferFromDeviceToDevice;                         // NOLINT
  PrototypeTpuDriver_GetCompiledProgramShape*
      TpuDriver_GetCompiledProgramShape;  // NOLINT
  PrototypeTpuDriver_FreeCompiledProgramShape*
      TpuDriver_FreeCompiledProgramShape;                           // NOLINT
  PrototypeTpuDriver_EventAddCallback* TpuDriver_EventAddCallback;  // NOLINT
  PrototypeTpuDriver_EventAwait* TpuDriver_EventAwait;              // NOLINT
  PrototypeTpuDriver_FreeEvent* TpuDriver_FreeEvent;                // NOLINT
  PrototypeTpuDriver_FreeStatus* TpuDriver_FreeStatus;              // NOLINT

  PrototypeTpuDriver_Version* TpuDriver_Version;                    // NOLINT
};

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_CLIENT_LIBTPU_H_
