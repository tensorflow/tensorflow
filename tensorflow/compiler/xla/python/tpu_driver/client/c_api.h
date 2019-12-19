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

typedef struct TpuBufferHandle {
  TpuBufferHandleInternal* internal_handle;
  TpuEvent* event;
} TpuBufferHandle;

typedef void(PrototypeTpuDriver_Initialize)(struct TpuDriverFn* driver_fn);
typedef struct TpuDriver*(PrototypeTpuDriver_Open)(const char* worker);
typedef void(PrototypeTpuDriver_Close)(struct TpuDriver* driver);

const int32_t MemoryRegion_HBM = 1;

typedef struct TpuBufferHandle*(PrototypeTpuDriver_Allocate)(
    struct TpuDriver* driver, int32_t core_id, int32_t memory_region,
    int64_t num_bytes, int32_t eventc, struct TpuEvent** eventv);

typedef struct TpuEvent*(PrototypeTpuDriver_Deallocate)(
    struct TpuDriver* driver, struct TpuBufferHandle* buffer_handle,
    int32_t eventc, struct TpuEvent** eventv);

typedef void(PrototypeTpuDriver_FreeEvent)(struct TpuEvent* event);

typedef const char*(PrototypeTpuDriver_Version)();

TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_Initialize TpuDriver_Initialize;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_Open TpuDriver_Open;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_Close TpuDriver_Close;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_Allocate TpuDriver_Allocate;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_Deallocate TpuDriver_Deallocate;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_FreeEvent TpuDriver_FreeEvent;
TPUDRIVER_CAPI_EXPORT extern PrototypeTpuDriver_Version TpuDriver_Version;

#ifdef __cplusplus
}
#endif

struct TpuDriverFn {
  PrototypeTpuDriver_Open* TpuDriver_Open;              // NOLINT
  PrototypeTpuDriver_Close* TpuDriver_Close;            // NOLINT
  PrototypeTpuDriver_Allocate* TpuDriver_Allocate;      // NOLINT
  PrototypeTpuDriver_Deallocate* TpuDriver_Deallocate;  // NOLINT
  PrototypeTpuDriver_FreeEvent* TpuDriver_FreeEvent;    // NOLINT
  PrototypeTpuDriver_Version* TpuDriver_Version;        // NOLINT
};

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_CLIENT_C_API_H_
