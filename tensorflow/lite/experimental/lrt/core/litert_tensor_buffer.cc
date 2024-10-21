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

#include "tensorflow/lite/experimental/lrt/c/litert_tensor_buffer.h"

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/lrt/c/litert_common.h"
#include "tensorflow/lite/experimental/lrt/c/litert_event.h"
#include "tensorflow/lite/experimental/lrt/c/litert_logging.h"
#include "tensorflow/lite/experimental/lrt/c/litert_model.h"
#include "tensorflow/lite/experimental/lrt/core/tensor_buffer.h"

LiteRtStatus LiteRtCreateTensorBufferFromHostMemory(
    const LiteRtRankedTensorType* tensor_type, void* host_buffer_addr,
    size_t size, LiteRtHostMemoryDeallocator deallocator,
    LiteRtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !host_buffer_addr || !tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto created_tensor_buffer = LiteRtTensorBufferT::CreateFromHostMemory(
      *tensor_type,
      absl::MakeSpan(static_cast<uint8_t*>(host_buffer_addr), size),
      deallocator);
  if (!created_tensor_buffer.ok()) {
    LITERT_LOG(LITERT_ERROR, "%s", created_tensor_buffer.status().message());
    return kLiteRtStatusErrorRuntimeFailure;
  }
  *tensor_buffer = created_tensor_buffer->release();
  return kLiteRtStatusOk;
}

#if LITERT_HAS_AHWB_SUPPORT
LiteRtStatus LiteRtCreateTensorBufferFromAhwb(
    const LiteRtRankedTensorType* tensor_type, AHardwareBuffer* ahwb,
    size_t ahwb_offset, LiteRtAhwbDeallocator deallocator,
    LiteRtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !ahwb || !tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto created_tensor_buffer = LiteRtTensorBufferT::CreateFromAhwb(
      *tensor_type, ahwb, ahwb_offset, deallocator);
  if (!created_tensor_buffer.ok()) {
    LITERT_LOG(LITERT_ERROR, "%s", created_tensor_buffer.status().message());
    return kLiteRtStatusErrorRuntimeFailure;
  }
  *tensor_buffer = created_tensor_buffer->release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferAhwb(LiteRtTensorBuffer tensor_buffer,
                                       AHardwareBuffer** ahwb) {
  if (!tensor_buffer || !ahwb) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto ahwb_buffer = tensor_buffer->GetAhwbBuffer();
  if (!ahwb_buffer.ok()) {
    LITERT_LOG(LITERT_ERROR, "%s", ahwb_buffer.status().message());
    return kLiteRtStatusErrorRuntimeFailure;
  }

  *ahwb = *ahwb_buffer;
  return kLiteRtStatusOk;
}
#endif  // LITERT_HAS_AHWB_SUPPORT

#if LITERT_HAS_ION_SUPPORT
LiteRtStatus LiteRtCreateTensorBufferFromIonBuffer(
    const LiteRtRankedTensorType* tensor_type, void* ion_buffer_addr,
    int ion_buffer_fd, size_t ion_buffer_size, size_t ion_buffer_offset,
    LiteRtIonDeallocator deallocator, LiteRtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto created_tensor_buffer = LiteRtTensorBufferT::CreateFromIonBuffer(
      *tensor_type, ion_buffer_addr, ion_buffer_fd, ion_buffer_size,
      ion_buffer_offset, deallocator);
  if (!created_tensor_buffer.ok()) {
    LITERT_LOG(LITERT_ERROR, "%s", created_tensor_buffer.status().message());
    return kLiteRtStatusErrorRuntimeFailure;
  }
  *tensor_buffer = created_tensor_buffer->release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferIonBuffer(LiteRtTensorBuffer tensor_buffer,
                                            void** ion_buffer_addr,
                                            int* ion_buffer_fd) {
  if (!tensor_buffer || !ion_buffer_addr || !ion_buffer_fd) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto ion_buffer = tensor_buffer->GetIonBuffer();
  if (!ion_buffer.ok()) {
    LITERT_LOG(LITERT_ERROR, "%s", ion_buffer.status().message());
    return kLiteRtStatusErrorRuntimeFailure;
  }

  *ion_buffer_addr = ion_buffer->first;
  *ion_buffer_fd = ion_buffer->second;
  return kLiteRtStatusOk;
}
#endif  // LITERT_HAS_ION_SUPPORT

#if LITERT_HAS_DMABUF_SUPPORT
LiteRtStatus LiteRtCreateTensorBufferFromDmaBufBuffer(
    const LiteRtRankedTensorType* tensor_type, void* dmabuf_buffer_addr,
    int dmabuf_buffer_fd, size_t dmabuf_buffer_size,
    size_t dmabuf_buffer_offset, LiteRtDmaBufDeallocator deallocator,
    LiteRtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto created_tensor_buffer = LiteRtTensorBufferT::CreateFromDmaBufBuffer(
      *tensor_type, dmabuf_buffer_addr, dmabuf_buffer_fd, dmabuf_buffer_size,
      dmabuf_buffer_offset, deallocator);
  if (!created_tensor_buffer.ok()) {
    LITERT_LOG(LITERT_ERROR, "%s", created_tensor_buffer.status().message());
    return kLiteRtStatusErrorRuntimeFailure;
  }
  *tensor_buffer = created_tensor_buffer->release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferDmaBufBuffer(LiteRtTensorBuffer tensor_buffer,
                                               void** dmabuf_buffer_addr,
                                               int* dmabuf_buffer_fd) {
  if (!tensor_buffer || !dmabuf_buffer_addr || !dmabuf_buffer_fd) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto dmabuf_buffer = tensor_buffer->GetDmaBufBuffer();
  if (!dmabuf_buffer.ok()) {
    LITERT_LOG(LITERT_ERROR, "%s", dmabuf_buffer.status().message());
    return kLiteRtStatusErrorRuntimeFailure;
  }

  *dmabuf_buffer_addr = dmabuf_buffer->first;
  *dmabuf_buffer_fd = dmabuf_buffer->second;
  return kLiteRtStatusOk;
}
#endif  // LITERT_HAS_DMABUF_SUPPORT

#if LITERT_HAS_FASTRPC_SUPPORT
LiteRtStatus LiteRtCreateTensorBufferFromFastRpcBuffer(
    const LiteRtRankedTensorType* tensor_type, void* fastrpc_buffer_addr,
    int fastrpc_buffer_fd, size_t fastrpc_buffer_size,
    size_t fastrpc_buffer_offset, LiteRtFastRpcDeallocator deallocator,
    LiteRtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto created_tensor_buffer = LiteRtTensorBufferT::CreateFromFastRpcBuffer(
      *tensor_type, fastrpc_buffer_addr, fastrpc_buffer_fd, fastrpc_buffer_size,
      fastrpc_buffer_offset, deallocator);
  if (!created_tensor_buffer.ok()) {
    LITERT_LOG(LITERT_ERROR, "%s", created_tensor_buffer.status().message());
    return kLiteRtStatusErrorRuntimeFailure;
  }
  *tensor_buffer = created_tensor_buffer->release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferFastRpcBuffer(
    LiteRtTensorBuffer tensor_buffer, void** fastrpc_buffer_addr,
    int* fastrpc_buffer_fd) {
  if (!tensor_buffer || !fastrpc_buffer_addr || !fastrpc_buffer_fd) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto fastrpc_buffer = tensor_buffer->GetFastRpcBuffer();
  if (!fastrpc_buffer.ok()) {
    LITERT_LOG(LITERT_ERROR, "%s", fastrpc_buffer.status().message());
    return kLiteRtStatusErrorRuntimeFailure;
  }

  *fastrpc_buffer_addr = fastrpc_buffer->first;
  *fastrpc_buffer_fd = fastrpc_buffer->second;
  return kLiteRtStatusOk;
}
#endif  // LITERT_HAS_FASTRPC_SUPPORT

LiteRtStatus LiteRtCreateManagedTensorBuffer(
    LiteRtTensorBufferType buffer_type,
    const LiteRtRankedTensorType* tensor_type, size_t buffer_size,
    LiteRtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto created_tensor_buffer = LiteRtTensorBufferT::CreateManaged(
      buffer_type, *tensor_type, buffer_size);
  if (!created_tensor_buffer.ok()) {
    LITERT_LOG(LITERT_ERROR, "%s", created_tensor_buffer.status().message());
    return kLiteRtStatusErrorRuntimeFailure;
  }
  *tensor_buffer = created_tensor_buffer->release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferType(LiteRtTensorBuffer tensor_buffer,
                                       LiteRtTensorBufferType* buffer_type) {
  if (!tensor_buffer || !buffer_type) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *buffer_type = tensor_buffer->buffer_type();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferTensorType(
    LiteRtTensorBuffer tensor_buffer, LiteRtRankedTensorType* tensor_type) {
  if (!tensor_buffer || !tensor_type) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *tensor_type = tensor_buffer->tensor_type();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferSize(LiteRtTensorBuffer tensor_buffer,
                                       size_t* buffer_size) {
  if (!tensor_buffer || !buffer_size) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *buffer_size = tensor_buffer->buffer_size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferOffset(LiteRtTensorBuffer tensor_buffer,
                                         size_t* buffer_offset) {
  if (!tensor_buffer || !buffer_offset) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *buffer_offset = tensor_buffer->buffer_offset();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferHostMemory(LiteRtTensorBuffer tensor_buffer,
                                             void** host_memory_addr) {
  if (!tensor_buffer || !host_memory_addr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto host_buffer = tensor_buffer->GetHostBuffer();
  if (!host_buffer.ok()) {
    LITERT_LOG(LITERT_ERROR, "%s", host_buffer.status().message());
    return kLiteRtStatusErrorRuntimeFailure;
  }

  *host_memory_addr = *host_buffer;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtLockTensorBuffer(LiteRtTensorBuffer tensor_buffer,
                                    void** host_mem_addr, LiteRtEvent event) {
  if (!tensor_buffer || !host_mem_addr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto mapped_addr = tensor_buffer->Lock(event);
  if (!mapped_addr.ok()) {
    LITERT_LOG(LITERT_ERROR, "%s", mapped_addr.status().message());
    return kLiteRtStatusErrorRuntimeFailure;
  }

  *host_mem_addr = *mapped_addr;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtUnlockTensorBuffer(LiteRtTensorBuffer tensor_buffer) {
  if (!tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  if (auto status = tensor_buffer->Unlock(); !status.ok()) {
    LITERT_LOG(LITERT_ERROR, "%s", status.message());
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

void LiteRtDestroyTensorBuffer(LiteRtTensorBuffer tensor_buffer) {
  delete tensor_buffer;
}
