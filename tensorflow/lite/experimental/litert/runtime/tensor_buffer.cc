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

#include "tensorflow/lite/experimental/litert/runtime/tensor_buffer.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_event.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/core/util/tensor_type_util.h"
#include "tensorflow/lite/experimental/litert/runtime/ahwb_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/dmabuf_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/event.h"
#include "tensorflow/lite/experimental/litert/runtime/fastrpc_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/ion_buffer.h"

namespace {

template <typename T>
void Copy(size_t array_size, const T*& array, std::vector<T>& vec) {
  vec.clear();
  vec.reserve(array_size);
  std::copy(array, array + array_size, std::back_inserter(vec));
  array = vec.data();
}

}  // namespace

LiteRtTensorBufferT::LiteRtTensorBufferT(
    const LiteRtRankedTensorType& tensor_type,
    LiteRtTensorBufferType buffer_type, size_t buffer_size,
    size_t buffer_offset)
    : tensor_type_(tensor_type),
      buffer_type_(buffer_type),
      buffer_size_(buffer_size),
      buffer_offset_(buffer_offset),
      ref_(1) {
  // Copy local memory passed by the caller.
  Copy(tensor_type_.layout.rank, tensor_type_.layout.dimensions, dimensions_);
  if (tensor_type_.layout.strides) {
    Copy(tensor_type_.layout.rank, tensor_type_.layout.strides, strides_);
  }
}

LiteRtTensorBufferT::~LiteRtTensorBufferT() {
  switch (buffer_type()) {
    case kLiteRtTensorBufferTypeUnknown:
      // Nothing to do.
      break;
    case kLiteRtTensorBufferTypeHostMemory:
      if (auto& buffer = std::get<HostBuffer>(buffer_); buffer.deallocator) {
        buffer.deallocator(buffer.addr);
      }
      break;
    case kLiteRtTensorBufferTypeAhwb:
      if (auto& buffer = std::get<AhwbBuffer>(buffer_); buffer.deallocator) {
        buffer.deallocator(buffer.ahwb);
      }
      break;
    case kLiteRtTensorBufferTypeIon:
      if (auto& buffer = std::get<IonBuffer>(buffer_); buffer.deallocator) {
        buffer.deallocator(buffer.addr);
      }
      break;
    case kLiteRtTensorBufferTypeDmaBuf:
      if (auto& buffer = std::get<DmaBufBuffer>(buffer_); buffer.deallocator) {
        buffer.deallocator(buffer.addr);
      }
      break;
    case kLiteRtTensorBufferTypeFastRpc:
      if (auto& buffer = std::get<FastRpcBuffer>(buffer_); buffer.deallocator) {
        buffer.deallocator(buffer.addr);
      }
      break;
  }
}

absl::StatusOr<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateFromHostMemory(
    const LiteRtRankedTensorType& tensor_type, absl::Span<uint8_t> host_memory,
    LiteRtHostMemoryDeallocator deallocator) {
  Ptr tensor_buffer(new LiteRtTensorBufferT(
      tensor_type, kLiteRtTensorBufferTypeHostMemory, host_memory.size()));
  tensor_buffer->buffer_ = HostBuffer{
      .addr = host_memory.data(),
      .deallocator = deallocator,
  };

  if (auto status = tensor_buffer->IsValid(); !status.ok()) {
    return status;
  }

  return tensor_buffer;
}

absl::StatusOr<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedOnHostMemory(
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  void* host_memory_ptr;
  if (auto rc = ::posix_memalign(
          &host_memory_ptr, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT, buffer_size);
      rc) {
    return absl::InternalError("Failed to allocate aligned memory");
  }

  LiteRtHostMemoryDeallocator deallocator = ::free;
  auto tensor_buffer = CreateFromHostMemory(
      tensor_type,
      absl::MakeSpan(static_cast<uint8_t*>(host_memory_ptr), buffer_size),
      deallocator);
  if (!tensor_buffer.ok()) {
    free(host_memory_ptr);
    return tensor_buffer.status();
  }

  return std::move(*tensor_buffer);
}

absl::StatusOr<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromAhwb(
    const LiteRtRankedTensorType& tensor_type, AHardwareBuffer* ahwb,
    size_t ahwb_offset, LiteRtAhwbDeallocator deallocator) {
  auto buffer_size = litert::internal::AhwbBuffer::GetSize(ahwb);
  if (!buffer_size.ok()) {
    return buffer_size.status();
  }

  Ptr tensor_buffer(new LiteRtTensorBufferT(
      tensor_type, kLiteRtTensorBufferTypeAhwb, *buffer_size, ahwb_offset));
  tensor_buffer->buffer_ = AhwbBuffer{
      .ahwb = ahwb,
      .deallocator = deallocator,
  };

  if (auto status = tensor_buffer->IsValid(); !status.ok()) {
    return status;
  }

  return tensor_buffer;
}

absl::StatusOr<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedAhwbBuffer(
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  auto buffer = litert::internal::AhwbBuffer::Alloc(buffer_size);
  if (!buffer.ok()) {
    return buffer.status();
  }
  return CreateFromAhwb(tensor_type, buffer->ahwb, /*ahwb_offset=*/0,
                        /*deallocator=*/litert::internal::AhwbBuffer::Free);
}

absl::StatusOr<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateFromIonBuffer(
    const LiteRtRankedTensorType& tensor_type, void* ion_buffer_addr,
    int ion_buffer_fd, size_t ion_buffer_size, size_t ion_buffer_offset,
    LiteRtIonDeallocator deallocator) {
  if (!ion_buffer_addr) {
    return absl::InvalidArgumentError("Invalid ION buffer address");
  }
  if (ion_buffer_fd < 0) {
    return absl::InvalidArgumentError("Invalid ION buffer fd");
  }

  Ptr tensor_buffer(
      new LiteRtTensorBufferT(tensor_type, kLiteRtTensorBufferTypeIon,
                              ion_buffer_size, ion_buffer_offset));
  tensor_buffer->buffer_ = IonBuffer{
      .addr = ion_buffer_addr,
      .fd = ion_buffer_fd,
      .deallocator = deallocator,
  };

  if (auto status = tensor_buffer->IsValid(); !status.ok()) {
    return status;
  }

  return tensor_buffer;
}

absl::StatusOr<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedIonBuffer(
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  auto buffer = litert::internal::IonBuffer::Alloc(
      buffer_size, /*alignment=*/LITERT_HOST_MEMORY_BUFFER_ALIGNMENT);
  if (!buffer.ok()) {
    return buffer.status();
  }
  return CreateFromIonBuffer(tensor_type, buffer->addr, buffer->fd, buffer_size,
                             /*ion_buffer_offset=*/0,
                             litert::internal::IonBuffer::Free);
}

absl::StatusOr<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateFromDmaBufBuffer(
    const LiteRtRankedTensorType& tensor_type, void* dmabuf_buffer_addr,
    int dmabuf_buffer_fd, size_t dmabuf_buffer_size,
    size_t dmabuf_buffer_offset, LiteRtDmaBufDeallocator deallocator) {
  if (!dmabuf_buffer_addr) {
    return absl::InvalidArgumentError("Invalid DMA-BUF buffer address");
  }
  if (dmabuf_buffer_fd < 0) {
    return absl::InvalidArgumentError("Invalid DMA-BUF buffer fd");
  }

  Ptr tensor_buffer(
      new LiteRtTensorBufferT(tensor_type, kLiteRtTensorBufferTypeDmaBuf,
                              dmabuf_buffer_size, dmabuf_buffer_offset));
  tensor_buffer->buffer_ = DmaBufBuffer{
      .addr = dmabuf_buffer_addr,
      .fd = dmabuf_buffer_fd,
      .deallocator = deallocator,
  };

  if (auto status = tensor_buffer->IsValid(); !status.ok()) {
    return status;
  }

  return tensor_buffer;
}

absl::StatusOr<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedDmaBufBuffer(
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  auto buffer = litert::internal::DmaBufBuffer::Alloc(buffer_size);
  if (!buffer.ok()) {
    return buffer.status();
  }
  return CreateFromDmaBufBuffer(tensor_type, buffer->addr, buffer->fd,
                                buffer_size, /*dmabuf_buffer_offset=*/0,
                                litert::internal::DmaBufBuffer::Free);
}

absl::StatusOr<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateFromFastRpcBuffer(
    const LiteRtRankedTensorType& tensor_type, void* fastrpc_buffer_addr,
    int fastrpc_buffer_fd, size_t fastrpc_buffer_size,
    size_t fastrpc_buffer_offset, LiteRtFastRpcDeallocator deallocator) {
  if (!fastrpc_buffer_addr) {
    return absl::InvalidArgumentError("Invalid FastRPC buffer address");
  }
  if (fastrpc_buffer_fd < 0) {
    return absl::InvalidArgumentError("Invalid FastRPC buffer fd");
  }

  Ptr tensor_buffer(
      new LiteRtTensorBufferT(tensor_type, kLiteRtTensorBufferTypeFastRpc,
                              fastrpc_buffer_size, fastrpc_buffer_offset));
  tensor_buffer->buffer_ = FastRpcBuffer{
      .addr = fastrpc_buffer_addr,
      .fd = fastrpc_buffer_fd,
      .deallocator = deallocator,
  };

  if (auto status = tensor_buffer->IsValid(); !status.ok()) {
    return status;
  }

  return tensor_buffer;
}

absl::StatusOr<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedFastRpcBuffer(
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  auto buffer = litert::internal::FastRpcBuffer::Alloc(buffer_size);
  if (!buffer.ok()) {
    return buffer.status();
  }
  return CreateFromFastRpcBuffer(tensor_type, buffer->addr, buffer->fd,
                                 buffer_size, /*fastrpc_buffer_offset=*/0,
                                 litert::internal::FastRpcBuffer::Free);
}

absl::StatusOr<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateManaged(
    LiteRtTensorBufferType buffer_type,
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  switch (buffer_type) {
    case kLiteRtTensorBufferTypeHostMemory:
      return CreateManagedOnHostMemory(tensor_type, buffer_size);
    case kLiteRtTensorBufferTypeAhwb:
      return CreateManagedAhwbBuffer(tensor_type, buffer_size);
    case kLiteRtTensorBufferTypeIon:
      return CreateManagedIonBuffer(tensor_type, buffer_size);
    case kLiteRtTensorBufferTypeDmaBuf:
      return CreateManagedDmaBufBuffer(tensor_type, buffer_size);
    case kLiteRtTensorBufferTypeFastRpc:
      return CreateManagedFastRpcBuffer(tensor_type, buffer_size);
    default:
      return absl::InvalidArgumentError("Unexpected tensor type");
  }
}

absl::Status LiteRtTensorBufferT::IsValid() const {
  // Check for static dimensions.
  for (auto i = 0; i < tensor_type_.layout.rank; ++i) {
    if (tensor_type_.layout.dimensions[i] <= 0) {
      return absl::InternalError(
          "TensorBuffer must have all static dimensions");
    }
  }

  // Check for valid offset.
  if (buffer_offset() >= buffer_size()) {
    return absl::InternalError("Invalid buffer offset");
  }

  // Check for sufficient size.
  if (auto num_bytes = litert::internal::GetNumPackedBytes(tensor_type_);
      !num_bytes.ok()) {
    return num_bytes.status();
  } else if (*num_bytes > buffer_size() - buffer_offset()) {
    return absl::InternalError("Insufficient buffer size");
  }

  // Check for proper alignment.
  if (buffer_type() == kLiteRtTensorBufferTypeHostMemory) {
    auto host_buffer = GetHostBuffer();
    if (!host_buffer.ok()) {
      return host_buffer.status();
    }
    if (reinterpret_cast<uintptr_t>(*host_buffer) %
        LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
      return absl::InternalError("Unaligned host memory pointer");
    }
  }

  return {};
}

absl::StatusOr<void*> LiteRtTensorBufferT::GetHostBuffer() const {
  if (buffer_type_ != kLiteRtTensorBufferTypeHostMemory) {
    return absl::InternalError("Unexpected tensor buffer type");
  }
  return std::get<HostBuffer>(buffer_).addr;
}

absl::StatusOr<AHardwareBuffer*> LiteRtTensorBufferT::GetAhwbBuffer() const {
  if (buffer_type_ != kLiteRtTensorBufferTypeAhwb) {
    return absl::InternalError("Unexpected tensor buffer type");
  }
  return std::get<AhwbBuffer>(buffer_).ahwb;
}

absl::StatusOr<std::pair<void*, int>> LiteRtTensorBufferT::GetIonBuffer()
    const {
  if (buffer_type_ != kLiteRtTensorBufferTypeIon) {
    return absl::InternalError("Unexpected tensor buffer type");
  }
  auto buffer = std::get<IonBuffer>(buffer_);
  return std::make_pair(buffer.addr, buffer.fd);
}

absl::StatusOr<std::pair<void*, int>> LiteRtTensorBufferT::GetDmaBufBuffer()
    const {
  if (buffer_type_ != kLiteRtTensorBufferTypeDmaBuf) {
    return absl::InternalError("Unexpected tensor buffer type");
  }
  auto buffer = std::get<DmaBufBuffer>(buffer_);
  return std::make_pair(buffer.addr, buffer.fd);
}

absl::StatusOr<std::pair<void*, int>> LiteRtTensorBufferT::GetFastRpcBuffer()
    const {
  if (buffer_type_ != kLiteRtTensorBufferTypeFastRpc) {
    return absl::InternalError("Unexpected tensor buffer type");
  }
  auto buffer = std::get<FastRpcBuffer>(buffer_);
  return std::make_pair(buffer.addr, buffer.fd);
}

absl::StatusOr<void*> LiteRtTensorBufferT::Lock(LiteRtEvent event) {
  if (event) {
    // Only AHWB supports waiting on an input sync fence when locking the
    // buffer. For all other buffer types we wait here.
    if (buffer_type() != kLiteRtTensorBufferTypeAhwb) {
      if (auto status = event->Wait(/*timeout_in_ms*/ -1);
          status != kLiteRtStatusOk) {
        return absl::InternalError("Failed to wait on input event");
      }
    }
  }

  switch (buffer_type()) {
    case kLiteRtTensorBufferTypeHostMemory:
      return *GetHostBuffer();
    case kLiteRtTensorBufferTypeAhwb:
      return litert::internal::AhwbBuffer::Lock(*GetAhwbBuffer(), event);
    case kLiteRtTensorBufferTypeIon:
      return GetIonBuffer()->first;
    case kLiteRtTensorBufferTypeDmaBuf:
      return GetDmaBufBuffer()->first;
    case kLiteRtTensorBufferTypeFastRpc:
      return GetFastRpcBuffer()->first;
    default:
      return absl::InternalError("Unexpected tensor buffer type");
  }
}

absl::Status LiteRtTensorBufferT::Unlock() {
  if (buffer_type() == kLiteRtTensorBufferTypeAhwb) {
    auto ahwb = std::get<AhwbBuffer>(buffer_).ahwb;
    return litert::internal::AhwbBuffer::Unlock(ahwb);
  }

  return {};
}
