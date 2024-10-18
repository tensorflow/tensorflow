// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_TENSOR_BUFFER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_TENSOR_BUFFER_H_

#include <memory.h>
#include <stddef.h>

#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_event.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"

#if LRT_HAS_AHWB_SUPPORT
#include <android/hardware_buffer.h>
#else
// Define a place holder AHardwareBuffer struct just to enable compilation.
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
typedef struct AHardwareBuffer AHardwareBuffer;
#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // LRT_HAS_AHWB_SUPPORT

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITE_RT_DEFINE_HANDLE(LrtTensorBuffer);

#define LRT_HOST_MEMORY_BUFFER_ALIGNMENT 64

typedef enum {
  kLrtTensorBufferTypeUnknown = 0,
  kLrtTensorBufferTypeHostMemory = 1,
  kLrtTensorBufferTypeAhwb = 2,
  kLrtTensorBufferTypeIon = 3,
  kLrtTensorBufferTypeDmaBuf = 4,
  kLrtTensorBufferTypeFastRpc = 5,
} LrtTensorBufferType;

typedef void (*LrtHostMemoryDeallocator)(void* addr);
typedef void (*LrtAhwbDeallocator)(AHardwareBuffer* ahwb);
typedef void (*LrtIonDeallocator)(void* ion_buffer_addr);
typedef void (*LrtDmaBufDeallocator)(void* dmabuf_buffer_addr);
typedef void (*LrtFastRpcDeallocator)(void* fastrpc_buffer_addr);

// /////////////////////////////////////////////////////////////////////////////
// TensorBuffers.
// /////////////////////////////////////////////////////////////////////////////

// Create a tensor buffer from an existing host memory buffer of a given size,
// with optional host memory buffer deallocator (it can be NULL). Return an
// error if the passed host memory buffer doesn't satisfy
// LRT_HOST_MEMORY_BUFFER_ALIGNMENT alignment.
LrtStatus LrtCreateTensorBufferFromHostMemory(
    const LrtRankedTensorType* tensor_type, void* host_buffer_addr,
    size_t host_buffer_size, LrtHostMemoryDeallocator deallocator,
    LrtTensorBuffer* buffer);

// Return an error if the backing buffer is not allocated on the host memory.
LrtStatus LrtGetTensorBufferHostMemory(LrtTensorBuffer tensor_buffer,
                                       void** host_memory_addr);

#if LRT_HAS_AHWB_SUPPORT
// Create a tensor buffer from an existing AHardwareBuffer, with optional
// AHardwareBuffer deallocator (it can be NULL). An non-zero `buffer_offset` can
// be used to specify multiple tensor buffers sharing the same underlying AHWB,
// in which case the provided AHWB must be sufficiently large to accomodate for
// the allocation needed for all tensor buffers sharing it.
LrtStatus LrtCreateTensorBufferFromAhwb(const LrtRankedTensorType* tensor_type,
                                        AHardwareBuffer* ahwb,
                                        size_t ahwb_offset,
                                        LrtAhwbDeallocator deallocator,
                                        LrtTensorBuffer* buffer);

// Return an error if the backing buffer is not an AhardwareBuffer.
LrtStatus LrtGetTensorBufferAhwb(LrtTensorBuffer tensor_buffer,
                                 AHardwareBuffer** ahwb);
#endif  // LRT_HAS_AHWB_SUPPORT

#if LRT_HAS_ION_SUPPORT
// Create a tensor buffer from an existing ION buffer of a given size, with
// optional ION buffer deallocator (it can be NULL). An non-zero
// `ion_buffer_offset` can be used to specify multiple tensor buffers sharing
// the same underlying ION buffer, in which case parameter `ion_buffer_size`
// must be the entire size of the underlying ION memory buffer, including the
// allocation needed for all tensor buffers sharing it.
LrtStatus LrtCreateTensorBufferFromIonBuffer(
    const LrtRankedTensorType* tensor_type, void* ion_buffer_addr,
    int ion_buffer_fd, size_t ion_buffer_size, size_t ion_buffer_offset,
    LrtIonDeallocator deallocator, LrtTensorBuffer* buffer);

// Return an error if the backing buffer is not an ION buffer.
LrtStatus LrtGetTensorBufferIonBuffer(LrtTensorBuffer buffer,
                                      void** ion_buffer_addr,
                                      int* ion_buffer_fd);
#endif  // LRT_HAS_ION_SUPPORT

#if LRT_HAS_DMABUF_SUPPORT
// Create a tensor buffer from an existing DMA-BUF buffer of a given size, with
// optional DMA-BUF buffer deallocator (it can be NULL). An non-zero
// `dmabuf_buffer_offset` can be used to specify multiple tensor buffers sharing
// the same underlying ION buffer, in which case parameter `ion_buffer_size`
// must be the entire size of the underlying ION memory buffer, including the
// allocation needed for all tensor buffers sharing it.
LrtStatus LrtCreateTensorBufferFromDmaBufBuffer(
    const LrtRankedTensorType* tensor_type, void* dmabuf_buffer_addr,
    int dmabuf_buffer_fd, size_t dmabuf_buffer_size,
    size_t dmabuf_buffer_offset, LrtDmaBufDeallocator deallocator,
    LrtTensorBuffer* buffer);

// Return an error if the backing buffer is not an DMA-BUF buffer.
LrtStatus LrtGetTensorBufferDmaBufBuffer(LrtTensorBuffer tensor_buffer,
                                         void** dmabuf_buffer_addr,
                                         int* dmabuf_buffer_fd);
#endif  // LRT_HAS_DMABUF_SUPPORT

#if LRT_HAS_FASTRPC_SUPPORT
// Create a tensor buffer from an existing FastRPC memory buffer of a given
// size, with optional FastRPC memory buffer deallocator (it can be NULL). An
// non-zero `fastrpc_buffer_offset` can be used to specify multiple tensor
// buffers sharing the same underlying FastRPC memory buffer, in which case
// parameter `fastrpc_buffer_size` must be the entire size of the underlying
// FastRPC memory buffer, including the allocation needed for all tensor buffers
// sharing it.
LrtStatus LrtCreateTensorBufferFromFastRpcBuffer(
    const LrtRankedTensorType* tensor_type, void* fastrpc_buffer_addr,
    int fastrpc_fd, size_t fastrpc_buffer_size, size_t fastrpc_buffer_offset,
    LrtFastRpcDeallocator deallocator, LrtTensorBuffer* buffer);

// Return an error if the backing buffer is not a FastRPC memory buffer.
LrtStatus LrtGetTensorBufferFastRpcBuffer(LrtTensorBuffer tensor_buffer,
                                          void** fastrpc_buffer_addr,
                                          int* fastrpc_buffer_fd);
#endif  // LRT_HAS_FASTRPC_SUPPORT

// Create a buffer backed by managed memory for a given size.
LrtStatus LrtCreateManagedTensorBuffer(LrtTensorBufferType buffer_type,
                                       const LrtRankedTensorType* tensor_type,
                                       size_t buffer_size,
                                       LrtTensorBuffer* buffer);

LrtStatus LrtGetTensorBufferType(LrtTensorBuffer tensor_buffer,
                                 LrtTensorBufferType* buffer_type);

LrtStatus LrtGetTensorBufferTensorType(LrtTensorBuffer tensor_buffer,
                                       LrtRankedTensorType* tensor_type);

LrtStatus LrtGetTensorBufferSize(LrtTensorBuffer tensor_buffer, size_t* size);

LrtStatus LrtGetTensorBufferOffset(LrtTensorBuffer tensor_buffer,
                                   size_t* offset);

// Lock a tensor buffer and map it to host memory, optionally syncronizing on a
// given input event (parameter `event` can be NULL).
LrtStatus LrtLockTensorBuffer(LrtTensorBuffer tensor_buffer,
                              void** host_mem_addr, LrtEvent event);

// Unlock a tensor buffer and (potentially) unmap it from host memory.
LrtStatus LrtUnlockTensorBuffer(LrtTensorBuffer buffer);

void LrtDestroyTensorBuffer(LrtTensorBuffer buffer);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_TENSOR_BUFFER_H_
