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

#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_event.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/core/logging.h"
#include "tensorflow/lite/experimental/lrt/core/tensor_buffer.h"

LrtStatus LrtCreateTensorBufferFromHostMemory(
    const LrtRankedTensorType* tensor_type, void* host_buffer_addr, size_t size,
    LrtHostMemoryDeallocator deallocator, LrtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !host_buffer_addr || !tensor_buffer) {
    return kLrtStatusErrorInvalidArgument;
  }
  auto created_tensor_buffer = LrtTensorBufferT::CreateFromHostMemory(
      *tensor_type,
      absl::MakeSpan(static_cast<uint8_t*>(host_buffer_addr), size),
      deallocator);
  if (!created_tensor_buffer.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s", created_tensor_buffer.status().message());
    return kLrtStatusErrorRuntimeFailure;
  }
  *tensor_buffer = created_tensor_buffer->release();
  return kLrtStatusOk;
}

#if LRT_HAS_AHWB_SUPPORT
LrtStatus LrtCreateTensorBufferFromAhwb(const LrtRankedTensorType* tensor_type,
                                        AHardwareBuffer* ahwb,
                                        size_t ahwb_offset,
                                        LrtAhwbDeallocator deallocator,
                                        LrtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !ahwb || !tensor_buffer) {
    return kLrtStatusErrorInvalidArgument;
  }
  auto created_tensor_buffer = LrtTensorBufferT::CreateFromAhwb(
      *tensor_type, ahwb, ahwb_offset, deallocator);
  if (!created_tensor_buffer.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s", created_tensor_buffer.status().message());
    return kLrtStatusErrorRuntimeFailure;
  }
  *tensor_buffer = created_tensor_buffer->release();
  return kLrtStatusOk;
}

LrtStatus LrtGetTensorBufferAhwb(LrtTensorBuffer tensor_buffer,
                                 AHardwareBuffer** ahwb) {
  if (!tensor_buffer || !ahwb) {
    return kLrtStatusErrorInvalidArgument;
  }

  auto ahwb_buffer = tensor_buffer->GetAhwbBuffer();
  if (!ahwb_buffer.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s", ahwb_buffer.status().message());
    return kLrtStatusErrorRuntimeFailure;
  }

  *ahwb = *ahwb_buffer;
  return kLrtStatusOk;
}
#endif  // LRT_HAS_AHWB_SUPPORT

#if LRT_HAS_ION_SUPPORT
LrtStatus LrtCreateTensorBufferFromIonBuffer(
    const LrtRankedTensorType* tensor_type, void* ion_buffer_addr,
    int ion_buffer_fd, size_t ion_buffer_size, size_t ion_buffer_offset,
    LrtIonDeallocator deallocator, LrtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !tensor_buffer) {
    return kLrtStatusErrorInvalidArgument;
  }
  auto created_tensor_buffer = LrtTensorBufferT::CreateFromIonBuffer(
      *tensor_type, ion_buffer_addr, ion_buffer_fd, ion_buffer_size,
      ion_buffer_offset, deallocator);
  if (!created_tensor_buffer.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s", created_tensor_buffer.status().message());
    return kLrtStatusErrorRuntimeFailure;
  }
  *tensor_buffer = created_tensor_buffer->release();
  return kLrtStatusOk;
}

LrtStatus LrtGetTensorBufferIonBuffer(LrtTensorBuffer tensor_buffer,
                                      void** ion_buffer_addr,
                                      int* ion_buffer_fd) {
  if (!tensor_buffer || !ion_buffer_addr || !ion_buffer_fd) {
    return kLrtStatusErrorInvalidArgument;
  }

  auto ion_buffer = tensor_buffer->GetIonBuffer();
  if (!ion_buffer.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s", ion_buffer.status().message());
    return kLrtStatusErrorRuntimeFailure;
  }

  *ion_buffer_addr = ion_buffer->first;
  *ion_buffer_fd = ion_buffer->second;
  return kLrtStatusOk;
}
#endif  // LRT_HAS_ION_SUPPORT

#if LRT_HAS_DMABUF_SUPPORT
LrtStatus LrtCreateTensorBufferFromDmaBufBuffer(
    const LrtRankedTensorType* tensor_type, void* dmabuf_buffer_addr,
    int dmabuf_buffer_fd, size_t dmabuf_buffer_size,
    size_t dmabuf_buffer_offset, LrtDmaBufDeallocator deallocator,
    LrtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !tensor_buffer) {
    return kLrtStatusErrorInvalidArgument;
  }
  auto created_tensor_buffer = LrtTensorBufferT::CreateFromDmaBufBuffer(
      *tensor_type, dmabuf_buffer_addr, dmabuf_buffer_fd, dmabuf_buffer_size,
      dmabuf_buffer_offset, deallocator);
  if (!created_tensor_buffer.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s", created_tensor_buffer.status().message());
    return kLrtStatusErrorRuntimeFailure;
  }
  *tensor_buffer = created_tensor_buffer->release();
  return kLrtStatusOk;
}

LrtStatus LrtGetTensorBufferDmaBufBuffer(LrtTensorBuffer tensor_buffer,
                                         void** dmabuf_buffer_addr,
                                         int* dmabuf_buffer_fd) {
  if (!tensor_buffer || !dmabuf_buffer_addr || !dmabuf_buffer_fd) {
    return kLrtStatusErrorInvalidArgument;
  }

  auto dmabuf_buffer = tensor_buffer->GetDmaBufBuffer();
  if (!dmabuf_buffer.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s", dmabuf_buffer.status().message());
    return kLrtStatusErrorRuntimeFailure;
  }

  *dmabuf_buffer_addr = dmabuf_buffer->first;
  *dmabuf_buffer_fd = dmabuf_buffer->second;
  return kLrtStatusOk;
}
#endif  // LRT_HAS_DMABUF_SUPPORT

#if LRT_HAS_FASTRPC_SUPPORT
LrtStatus LrtCreateTensorBufferFromFastRpcBuffer(
    const LrtRankedTensorType* tensor_type, void* fastrpc_buffer_addr,
    int fastrpc_buffer_fd, size_t fastrpc_buffer_size,
    size_t fastrpc_buffer_offset, LrtFastRpcDeallocator deallocator,
    LrtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !tensor_buffer) {
    return kLrtStatusErrorInvalidArgument;
  }
  auto created_tensor_buffer = LrtTensorBufferT::CreateFromFastRpcBuffer(
      *tensor_type, fastrpc_buffer_addr, fastrpc_buffer_fd, fastrpc_buffer_size,
      fastrpc_buffer_offset, deallocator);
  if (!created_tensor_buffer.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s", created_tensor_buffer.status().message());
    return kLrtStatusErrorRuntimeFailure;
  }
  *tensor_buffer = created_tensor_buffer->release();
  return kLrtStatusOk;
}

LrtStatus LrtGetTensorBufferFastRpcBuffer(LrtTensorBuffer tensor_buffer,
                                          void** fastrpc_buffer_addr,
                                          int* fastrpc_buffer_fd) {
  if (!tensor_buffer || !fastrpc_buffer_addr || !fastrpc_buffer_fd) {
    return kLrtStatusErrorInvalidArgument;
  }

  auto fastrpc_buffer = tensor_buffer->GetFastRpcBuffer();
  if (!fastrpc_buffer.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s", fastrpc_buffer.status().message());
    return kLrtStatusErrorRuntimeFailure;
  }

  *fastrpc_buffer_addr = fastrpc_buffer->first;
  *fastrpc_buffer_fd = fastrpc_buffer->second;
  return kLrtStatusOk;
}
#endif  // LRT_HAS_FASTRPC_SUPPORT

LrtStatus LrtCreateManagedTensorBuffer(LrtTensorBufferType buffer_type,
                                       const LrtRankedTensorType* tensor_type,
                                       size_t buffer_size,
                                       LrtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !tensor_buffer) {
    return kLrtStatusErrorInvalidArgument;
  }
  auto created_tensor_buffer =
      LrtTensorBufferT::CreateManaged(buffer_type, *tensor_type, buffer_size);
  if (!created_tensor_buffer.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s", created_tensor_buffer.status().message());
    return kLrtStatusErrorRuntimeFailure;
  }
  *tensor_buffer = created_tensor_buffer->release();
  return kLrtStatusOk;
}

LrtStatus LrtGetTensorBufferType(LrtTensorBuffer tensor_buffer,
                                 LrtTensorBufferType* buffer_type) {
  if (!tensor_buffer || !buffer_type) {
    return kLrtStatusErrorInvalidArgument;
  }
  *buffer_type = tensor_buffer->buffer_type();
  return kLrtStatusOk;
}

LrtStatus LrtGetTensorBufferTensorType(LrtTensorBuffer tensor_buffer,
                                       LrtRankedTensorType* tensor_type) {
  if (!tensor_buffer || !tensor_type) {
    return kLrtStatusErrorInvalidArgument;
  }
  *tensor_type = tensor_buffer->tensor_type();
  return kLrtStatusOk;
}

LrtStatus LrtGetTensorBufferSize(LrtTensorBuffer tensor_buffer,
                                 size_t* buffer_size) {
  if (!tensor_buffer || !buffer_size) {
    return kLrtStatusErrorInvalidArgument;
  }
  *buffer_size = tensor_buffer->buffer_size();
  return kLrtStatusOk;
}

LrtStatus LrtGetTensorBufferOffset(LrtTensorBuffer tensor_buffer,
                                   size_t* buffer_offset) {
  if (!tensor_buffer || !buffer_offset) {
    return kLrtStatusErrorInvalidArgument;
  }
  *buffer_offset = tensor_buffer->buffer_offset();
  return kLrtStatusOk;
}

LrtStatus LrtGetTensorBufferHostMemory(LrtTensorBuffer tensor_buffer,
                                       void** host_memory_addr) {
  if (!tensor_buffer || !host_memory_addr) {
    return kLrtStatusErrorInvalidArgument;
  }

  auto host_buffer = tensor_buffer->GetHostBuffer();
  if (!host_buffer.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s", host_buffer.status().message());
    return kLrtStatusErrorRuntimeFailure;
  }

  *host_memory_addr = *host_buffer;
  return kLrtStatusOk;
}

LrtStatus LrtLockTensorBuffer(LrtTensorBuffer tensor_buffer,
                              void** host_mem_addr, LrtEvent event) {
  if (!tensor_buffer || !host_mem_addr) {
    return kLrtStatusErrorInvalidArgument;
  }

  auto mapped_addr = tensor_buffer->Lock(event);
  if (!mapped_addr.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s", mapped_addr.status().message());
    return kLrtStatusErrorRuntimeFailure;
  }

  *host_mem_addr = *mapped_addr;
  return kLrtStatusOk;
}

LrtStatus LrtUnlockTensorBuffer(LrtTensorBuffer tensor_buffer) {
  if (!tensor_buffer) {
    return kLrtStatusErrorInvalidArgument;
  }

  if (auto status = tensor_buffer->Unlock(); !status.ok()) {
    LITE_RT_LOG(LRT_ERROR, "%s", status.message());
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;
}

void LrtDestroyTensorBuffer(LrtTensorBuffer tensor_buffer) {
  delete tensor_buffer;
}
