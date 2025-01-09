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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_TENSOR_BUFFER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_TENSOR_BUFFER_H_

#include <memory.h>
#include <stddef.h>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_event.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"

#if LITERT_HAS_AHWB_SUPPORT
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
#endif  // LITERT_HAS_AHWB_SUPPORT

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LiteRtTensorBuffer);

#define LITERT_HOST_MEMORY_BUFFER_ALIGNMENT 64

typedef enum {
  kLiteRtTensorBufferTypeUnknown = 0,
  kLiteRtTensorBufferTypeHostMemory = 1,
  kLiteRtTensorBufferTypeAhwb = 2,
  kLiteRtTensorBufferTypeIon = 3,
  kLiteRtTensorBufferTypeDmaBuf = 4,
  kLiteRtTensorBufferTypeFastRpc = 5,
} LiteRtTensorBufferType;

typedef void (*LiteRtHostMemoryDeallocator)(void* addr);
typedef void (*LiteRtAhwbDeallocator)(AHardwareBuffer* ahwb);
typedef void (*LiteRtIonDeallocator)(void* ion_buffer_addr);
typedef void (*LiteRtDmaBufDeallocator)(void* dmabuf_buffer_addr);
typedef void (*LiteRtFastRpcDeallocator)(void* fastrpc_buffer_addr);

// /////////////////////////////////////////////////////////////////////////////
// TensorBuffers.
// /////////////////////////////////////////////////////////////////////////////

// Create a tensor buffer from an existing host memory buffer of a given size,
// with optional host memory buffer deallocator (it can be NULL). Return an
// error if the passed host memory buffer doesn't satisfy
// LITERT_HOST_MEMORY_BUFFER_ALIGNMENT alignment.
LiteRtStatus LiteRtCreateTensorBufferFromHostMemory(
    const LiteRtRankedTensorType* tensor_type, void* host_buffer_addr,
    size_t host_buffer_size, LiteRtHostMemoryDeallocator deallocator,
    LiteRtTensorBuffer* buffer);

// Return an error if the backing buffer is not allocated on the host memory.
LiteRtStatus LiteRtGetTensorBufferHostMemory(LiteRtTensorBuffer tensor_buffer,
                                             void** host_memory_addr);

#if LITERT_HAS_AHWB_SUPPORT
// Create a tensor buffer from an existing AHardwareBuffer, with optional
// AHardwareBuffer deallocator (it can be NULL). An non-zero `buffer_offset` can
// be used to specify multiple tensor buffers sharing the same underlying AHWB,
// in which case the provided AHWB must be sufficiently large to accomodate for
// the allocation needed for all tensor buffers sharing it.
LiteRtStatus LiteRtCreateTensorBufferFromAhwb(
    const LiteRtRankedTensorType* tensor_type, AHardwareBuffer* ahwb,
    size_t ahwb_offset, LiteRtAhwbDeallocator deallocator,
    LiteRtTensorBuffer* buffer);

// Return an error if the backing buffer is not an AhardwareBuffer.
LiteRtStatus LiteRtGetTensorBufferAhwb(LiteRtTensorBuffer tensor_buffer,
                                       AHardwareBuffer** ahwb);
#endif  // LITERT_HAS_AHWB_SUPPORT

#if LITERT_HAS_ION_SUPPORT
// Create a tensor buffer from an existing ION buffer of a given size, with
// optional ION buffer deallocator (it can be NULL). An non-zero
// `ion_buffer_offset` can be used to specify multiple tensor buffers sharing
// the same underlying ION buffer, in which case parameter `ion_buffer_size`
// must be the entire size of the underlying ION memory buffer, including the
// allocation needed for all tensor buffers sharing it.
LiteRtStatus LiteRtCreateTensorBufferFromIonBuffer(
    const LiteRtRankedTensorType* tensor_type, void* ion_buffer_addr,
    int ion_buffer_fd, size_t ion_buffer_size, size_t ion_buffer_offset,
    LiteRtIonDeallocator deallocator, LiteRtTensorBuffer* buffer);

// Return an error if the backing buffer is not an ION buffer.
LiteRtStatus LiteRtGetTensorBufferIonBuffer(LiteRtTensorBuffer buffer,
                                            void** ion_buffer_addr,
                                            int* ion_buffer_fd);
#endif  // LITERT_HAS_ION_SUPPORT

#if LITERT_HAS_DMABUF_SUPPORT
// Create a tensor buffer from an existing DMA-BUF buffer of a given size, with
// optional DMA-BUF buffer deallocator (it can be NULL). An non-zero
// `dmabuf_buffer_offset` can be used to specify multiple tensor buffers sharing
// the same underlying ION buffer, in which case parameter `ion_buffer_size`
// must be the entire size of the underlying ION memory buffer, including the
// allocation needed for all tensor buffers sharing it.
LiteRtStatus LiteRtCreateTensorBufferFromDmaBufBuffer(
    const LiteRtRankedTensorType* tensor_type, void* dmabuf_buffer_addr,
    int dmabuf_buffer_fd, size_t dmabuf_buffer_size,
    size_t dmabuf_buffer_offset, LiteRtDmaBufDeallocator deallocator,
    LiteRtTensorBuffer* buffer);

// Return an error if the backing buffer is not an DMA-BUF buffer.
LiteRtStatus LiteRtGetTensorBufferDmaBufBuffer(LiteRtTensorBuffer tensor_buffer,
                                               void** dmabuf_buffer_addr,
                                               int* dmabuf_buffer_fd);
#endif  // LITERT_HAS_DMABUF_SUPPORT

#if LITERT_HAS_FASTRPC_SUPPORT
// Create a tensor buffer from an existing FastRPC memory buffer of a given
// size, with optional FastRPC memory buffer deallocator (it can be NULL). An
// non-zero `fastrpc_buffer_offset` can be used to specify multiple tensor
// buffers sharing the same underlying FastRPC memory buffer, in which case
// parameter `fastrpc_buffer_size` must be the entire size of the underlying
// FastRPC memory buffer, including the allocation needed for all tensor buffers
// sharing it.
LiteRtStatus LiteRtCreateTensorBufferFromFastRpcBuffer(
    const LiteRtRankedTensorType* tensor_type, void* fastrpc_buffer_addr,
    int fastrpc_fd, size_t fastrpc_buffer_size, size_t fastrpc_buffer_offset,
    LiteRtFastRpcDeallocator deallocator, LiteRtTensorBuffer* buffer);

// Return an error if the backing buffer is not a FastRPC memory buffer.
LiteRtStatus LiteRtGetTensorBufferFastRpcBuffer(
    LiteRtTensorBuffer tensor_buffer, void** fastrpc_buffer_addr,
    int* fastrpc_buffer_fd);
#endif  // LITERT_HAS_FASTRPC_SUPPORT

// Create a buffer backed by managed memory for a given size.
LiteRtStatus LiteRtCreateManagedTensorBuffer(
    LiteRtTensorBufferType buffer_type,
    const LiteRtRankedTensorType* tensor_type, size_t buffer_size,
    LiteRtTensorBuffer* buffer);

// Create a duplicate of the current tensor buffer. It will increase the
// reference count of a managed tensor buffer. And the number decreases when
// LiteRtDestroyTensorBuffer() is called.
LiteRtStatus LiteRtDuplicateTensorBuffer(LiteRtTensorBuffer tensor_buffer);

LiteRtStatus LiteRtGetTensorBufferType(LiteRtTensorBuffer tensor_buffer,
                                       LiteRtTensorBufferType* buffer_type);

LiteRtStatus LiteRtGetTensorBufferTensorType(
    LiteRtTensorBuffer tensor_buffer, LiteRtRankedTensorType* tensor_type);

LiteRtStatus LiteRtGetTensorBufferSize(LiteRtTensorBuffer tensor_buffer,
                                       size_t* size);

LiteRtStatus LiteRtGetTensorBufferOffset(LiteRtTensorBuffer tensor_buffer,
                                         size_t* offset);

LiteRtStatus LiteRtHasTensorBufferEvent(LiteRtTensorBuffer tensor_buffer,
                                        bool* has_event);

LiteRtStatus LiteRtGetTensorBufferEvent(LiteRtTensorBuffer tensor_buffer,
                                        LiteRtEvent* event);

LiteRtStatus LiteRtSetTensorBufferEvent(LiteRtTensorBuffer tensor_buffer,
                                        LiteRtEvent event);

LiteRtStatus LiteRtClearTensorBufferEvent(LiteRtTensorBuffer tensor_buffer);

// Lock a tensor buffer and map it to host memory, optionally synchronizing on a
// given input event (parameter `event` can be NULL).
LiteRtStatus LiteRtLockTensorBuffer(LiteRtTensorBuffer tensor_buffer,
                                    void** host_mem_addr, LiteRtEvent event);

// Unlock a tensor buffer and (potentially) unmap it from host memory.
LiteRtStatus LiteRtUnlockTensorBuffer(LiteRtTensorBuffer buffer);

// Destroy a tensor buffer. If the tensor buffer is managed, the number of
// references to it is decreased and released the underlying TensorBufferT when
// the last reference is removed.
void LiteRtDestroyTensorBuffer(LiteRtTensorBuffer buffer);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_TENSOR_BUFFER_H_
