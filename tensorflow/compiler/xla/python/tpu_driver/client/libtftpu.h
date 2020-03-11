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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_CLIENT_LIBTFTPU_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_CLIENT_LIBTFTPU_H_

#include <stdbool.h>
#include <stdint.h>

#define TPUDRIVER_CAPI_EXPORT __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TpuStatus {
  int32_t code;
  char* msg;
} TpuStatus;

typedef struct TpuEagerRequest {
  int64_t size;
  void* data;
} TpuEagerRequest;

typedef struct TpuEagerResponse {
  struct TpuStatus* status;
  int64_t size;
  char* data;
} TpuEagerResponse;

typedef struct TpuEagerService TpuEagerService;

typedef struct TpuEagerService*(PrototypeTfEager_CreateEagerService)();

typedef void(PrototypeTfEager_FreeEagerService)(
    struct TpuEagerService* service);

typedef struct TpuEagerResponse*(PrototypeTfEager_CreateContext)(
    struct TpuEagerService* service, struct TpuEagerRequest request);

typedef struct TpuEagerResponse*(PrototypeTfEager_UpdateContext)(
    struct TpuEagerService* service, struct TpuEagerRequest request);

typedef struct TpuEagerResponse*(PrototypeTfEager_Enqueue)(
    struct TpuEagerService* service, struct TpuEagerRequest request);

typedef struct TpuEagerResponse*(PrototypeTfEager_WaitQueueDone)(
    struct TpuEagerService* service, struct TpuEagerRequest request);

typedef struct TpuEagerResponse*(PrototypeTfEager_KeepAlive)(
    struct TpuEagerService* service, struct TpuEagerRequest request);

typedef struct TpuEagerResponse*(PrototypeTfEager_CloseContext)(
    struct TpuEagerService* service, struct TpuEagerRequest request);

typedef void(PrototypeTfEager_FreeTpuEagerResponse)(
    struct TpuEagerResponse* response);

TPUDRIVER_CAPI_EXPORT extern PrototypeTfEager_CreateEagerService
    TfEager_CreateEagerService;

TPUDRIVER_CAPI_EXPORT extern PrototypeTfEager_FreeEagerService
    TfEager_FreeEagerService;

TPUDRIVER_CAPI_EXPORT extern PrototypeTfEager_CreateContext
    TfEager_CreateContext;

TPUDRIVER_CAPI_EXPORT extern PrototypeTfEager_UpdateContext
    TfEager_UpdateContext;

TPUDRIVER_CAPI_EXPORT extern PrototypeTfEager_Enqueue TfEager_Enqueue;

TPUDRIVER_CAPI_EXPORT extern PrototypeTfEager_WaitQueueDone
    TfEager_WaitQueueDone;

TPUDRIVER_CAPI_EXPORT extern PrototypeTfEager_KeepAlive TfEager_KeepAlive;

TPUDRIVER_CAPI_EXPORT extern PrototypeTfEager_CloseContext TfEager_CloseContext;

TPUDRIVER_CAPI_EXPORT extern PrototypeTfEager_FreeTpuEagerResponse
    TfEager_FreeTpuEagerResponse;

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_CLIENT_LIBTFTPU_H_
