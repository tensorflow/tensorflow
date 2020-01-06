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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_CLIENT_C_API_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_CLIENT_C_API_H_

#include <stdint.h>

#define TPUDRIVER_CAPI_EXPORT __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

struct TpuDriverFn;

typedef struct TpuDriver TpuDriver;

typedef struct TpuEvent TpuEvent;

typedef struct TpuBufferHandleInternal TpuBufferHandleInternal;

typedef struct TpuCompiledProgramHandleInternal
    TpuCompiledProgramHandleInternal;

typedef struct TpuLoadedProgramHandleInternal TpuLoadedProgramHandleInternal;
typedef struct HloProtoInternal HloProtoInternal;

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

typedef struct HloProto {
  HloProtoInternal* internal_hlo_proto;
} HloProto;

typedef struct DeviceAssignment {
  int replica_count;
  int computation_count;
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

typedef void(PrototypeTpuDriver_Initialize)(struct TpuDriverFn* driver_fn);
typedef struct TpuDriver*(PrototypeTpuDriver_Open)(const char* worker);
typedef void(PrototypeTpuDriver_Close)(struct TpuDriver* driver);

// TODO(frankchn): Make this not a hard-coded constant.
const int32_t MemoryRegion_HBM = 1;

typedef struct TpuCompiledProgramHandle*(PrototypeTpuDriver_CompileProgram)(
    struct TpuDriver* driver, const struct HloProto hlo_proto,
    int32_t num_replicas, int32_t eventc, struct TpuEvent** eventv);

typedef struct TpuCompiledProgramHandle*(
    PrototypeTpuDriver_CompileProgramFromText)(struct TpuDriver* driver,
                                               const char* hlo_text,
                                               int32_t num_replicas,
                                               int32_t eventc,
                                               struct TpuEvent** eventv);

typedef struct TpuLoadedProgramHandle*(PrototypeTpuDriver_LoadProgram)(
    struct TpuDriver* driver, int32_t core_id,
    const struct TpuCompiledProgramHandle* compiled_program_handle,
    int32_t eventc, struct TpuEvent** eventv);

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

typedef void(PrototypeTpuDriver_CreateDeviceAssignment)(int replica_count,
                                                        int computation_count);

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
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_CompileProgram
    TpuDriver_CompileProgram;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_CompileProgramFromText
    TpuDriver_CompileProgramFromText;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_LoadProgram
    TpuDriver_LoadProgram;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_UnloadProgram
    TpuDriver_UnloadProgram;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_ExecuteProgram
    TpuDriver_ExecuteProgram;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_AllocateTuple
    TpuDriver_AllocateTuple;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_Allocate TpuDriver_Allocate;
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
  PrototypeTpuDriver_CompileProgram* TpuDriver_CompileProgram;      // NOLINT
  PrototypeTpuDriver_CompileProgramFromText*
      TpuDriver_CompileProgramFromText;                             // NOLINT
  PrototypeTpuDriver_LoadProgram* TpuDriver_LoadProgram;            // NOLINT
  PrototypeTpuDriver_UnloadProgram* TpuDriver_UnloadProgram;        // NOLINT
  PrototypeTpuDriver_ExecuteProgram* TpuDriver_ExecuteProgram;      // NOLINT
  PrototypeTpuDriver_AllocateTuple* TpuDriver_AllocateTuple;        // NOLINT
  PrototypeTpuDriver_Allocate* TpuDriver_Allocate;                  // NOLINT
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

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_CLIENT_C_API_H_
