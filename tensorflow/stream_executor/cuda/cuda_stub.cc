/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"

// Implements the CUDA driver API by forwarding to CUDA loaded from the DSO.

namespace {
// Returns DSO handle or null if loading the DSO fails.
void* GetDsoHandle() {
#ifdef PLATFORM_GOOGLE
  return nullptr;
#else
  static auto handle = []() -> void* {
    auto handle_or =
        stream_executor::internal::DsoLoader::GetCudaDriverDsoHandle();
    if (!handle_or.ok()) return nullptr;
    return handle_or.ValueOrDie();
  }();
  return handle;
#endif
}

template <typename T>
T LoadSymbol(const char* symbol_name) {
  void* symbol = nullptr;
  if (auto handle = GetDsoHandle()) {
    stream_executor::port::Env::Default()
        ->GetSymbolFromLibrary(handle, symbol_name, &symbol)
        .IgnoreError();
  }
  return reinterpret_cast<T>(symbol);
}

CUresult GetSymbolNotFoundError() {
  return CUDA_ERROR_SHARED_OBJECT_INIT_FAILED;
}
}  // namespace

#if CUDA_VERSION < 8000
#error CUDA version earlier than 8 is not supported.
#endif

// Forward-declare types introduced in CUDA 9.0.
typedef struct CUDA_LAUNCH_PARAMS_st CUDA_LAUNCH_PARAMS;

#ifndef __CUDA_DEPRECATED
#define __CUDA_DEPRECATED
#endif

#if CUDA_VERSION < 10000
// Define fake enums introduced in CUDA 10.0.
typedef enum CUgraphNodeType_enum {} CUgraphNodeType;
typedef enum CUstreamCaptureStatus_enum {} CUstreamCaptureStatus;
typedef enum CUexternalMemoryHandleType_enum {} CUexternalMemoryHandleType;
typedef enum CUexternalSemaphoreHandleType_enum {
} CUexternalSemaphoreHandleType;
#endif

// Forward-declare types introduced in CUDA 10.0.
typedef struct CUextMemory_st* CUexternalMemory;
typedef struct CUextSemaphore_st* CUexternalSemaphore;
typedef struct CUgraph_st* CUgraph;
typedef struct CUgraphNode_st* CUgraphNode;
typedef struct CUgraphExec_st* CUgraphExec;
typedef struct CUDA_KERNEL_NODE_PARAMS_st CUDA_KERNEL_NODE_PARAMS;
typedef struct CUDA_MEMSET_NODE_PARAMS_st CUDA_MEMSET_NODE_PARAMS;
typedef struct CUDA_HOST_NODE_PARAMS_st CUDA_HOST_NODE_PARAMS;
typedef struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC;
typedef struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC;
typedef struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC;
typedef struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC;
typedef struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS;
typedef struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS;
typedef void(CUDA_CB* CUhostFn)(void* userData);

// For now only one stub implementation is needed. If a function that is not
// available in the given CUDA release, the corresponding wrapper returns
// CUDA_ERROR_SHARED_OBJECT_INIT_FAILED.
#include "tensorflow/stream_executor/cuda/cuda_10_0.inc"
