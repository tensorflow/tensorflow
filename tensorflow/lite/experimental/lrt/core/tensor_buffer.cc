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

#include "tensorflow/lite/experimental/lrt/core/tensor_buffer.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <utility>
#include <vector>

#if LRT_HAS_AHWB_SUPPORT
#include <android/hardware_buffer.h>
#endif  // LRT_HAS_AHWB_SUPPORT

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_event.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/core/dmabuf_buffer.h"
#include "tensorflow/lite/experimental/lrt/core/fastrpc_buffer.h"
#include "tensorflow/lite/experimental/lrt/core/ion_buffer.h"
#include "tensorflow/lite/experimental/lrt/core/utils.h"

namespace {

template <typename T>
void Copy(size_t array_size, const T*& array, std::vector<T>& vec) {
  vec.clear();
  vec.reserve(array_size);
  std::copy(array, array + array_size, std::back_inserter(vec));
  array = vec.data();
}

}  // namespace

LrtTensorBufferT::LrtTensorBufferT(const LrtRankedTensorType& tensor_type,
                                   LrtTensorBufferType buffer_type,
                                   size_t buffer_size, size_t buffer_offset)
    : tensor_type_(tensor_type),
      buffer_type_(buffer_type),
      buffer_size_(buffer_size),
      buffer_offset_(buffer_offset) {
  // Copy local memory passed by the caller.
  Copy(tensor_type_.layout.rank, tensor_type_.layout.dimensions, dimensions_);
  if (tensor_type_.layout.strides) {
    Copy(tensor_type_.layout.rank, tensor_type_.layout.strides, strides_);
  }
}

LrtTensorBufferT::~LrtTensorBufferT() {
  switch (buffer_type()) {
    case kLrtTensorBufferTypeUnknown:
      // Nothing to do.
      break;
    case kLrtTensorBufferTypeHostMemory:
      if (auto& buffer = std::get<HostBuffer>(buffer_); buffer.deallocator) {
        buffer.deallocator(buffer.addr);
      }
      break;
    case kLrtTensorBufferTypeAhwb:
      if (auto& buffer = std::get<AhwbBuffer>(buffer_); buffer.deallocator) {
        buffer.deallocator(buffer.ahwb);
      }
      break;
    case kLrtTensorBufferTypeIon:
      if (auto& buffer = std::get<IonBuffer>(buffer_); buffer.deallocator) {
        buffer.deallocator(buffer.addr);
      }
      break;
    case kLrtTensorBufferTypeDmaBuf:
      if (auto& buffer = std::get<DmaBufBuffer>(buffer_); buffer.deallocator) {
        buffer.deallocator(buffer.addr);
      }
      break;
    case kLrtTensorBufferTypeFastRpc:
      if (auto& buffer = std::get<FastRpcBuffer>(buffer_); buffer.deallocator) {
        buffer.deallocator(buffer.addr);
      }
      break;
  }
}

absl::StatusOr<LrtTensorBufferT::Ptr> LrtTensorBufferT::CreateFromHostMemory(
    const LrtRankedTensorType& tensor_type, absl::Span<uint8_t> host_memory,
    LrtHostMemoryDeallocator deallocator) {
  Ptr tensor_buffer(new LrtTensorBufferT(
      tensor_type, kLrtTensorBufferTypeHostMemory, host_memory.size()));
  tensor_buffer->buffer_ = HostBuffer{
      .addr = host_memory.data(),
      .deallocator = deallocator,
  };

  if (auto status = tensor_buffer->IsValid(); !status.ok()) {
    return status;
  }

  return tensor_buffer;
}

absl::StatusOr<LrtTensorBufferT::Ptr>
LrtTensorBufferT::CreateManagedOnHostMemory(
    const LrtRankedTensorType& tensor_type, size_t buffer_size) {
  void* host_memory_ptr;
  if (auto rc = ::posix_memalign(&host_memory_ptr,
                                 LRT_HOST_MEMORY_BUFFER_ALIGNMENT, buffer_size);
      rc) {
    return absl::InternalError("Failed to allocate aligned memory");
  }

  LrtHostMemoryDeallocator deallocator = ::free;
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

absl::StatusOr<LrtTensorBufferT::Ptr> LrtTensorBufferT::CreateFromAhwb(
    const LrtRankedTensorType& tensor_type, AHardwareBuffer* ahwb,
    size_t ahwb_offset, LrtAhwbDeallocator deallocator) {
#if LRT_HAS_AHWB_SUPPORT
  AHardwareBuffer_Desc ahwb_desc;
  AHardwareBuffer_describe(ahwb, &ahwb_desc);
  size_t buffer_size = static_cast<size_t>(ahwb_desc.width) * ahwb_desc.height *
                       ahwb_desc.layers;

  Ptr tensor_buffer(new LrtTensorBufferT(tensor_type, kLrtTensorBufferTypeAhwb,
                                         buffer_size, ahwb_offset));
  tensor_buffer->buffer_ = AhwbBuffer{
      .ahwb = ahwb,
      .deallocator = deallocator,
  };

  if (auto status = tensor_buffer->IsValid(); !status.ok()) {
    return status;
  }

  return tensor_buffer;
#else
  return absl::InternalError(
      "AHardwareBuffers are not supported on this platform");
#endif  // LRT_HAS_AHWB_SUPPORT
}

absl::StatusOr<LrtTensorBufferT::Ptr> LrtTensorBufferT::CreateManagedAhwbBuffer(
    const LrtRankedTensorType& tensor_type, size_t buffer_size) {
#if LRT_HAS_AHWB_SUPPORT
  AHardwareBuffer* ahwb;
  AHardwareBuffer_Desc ahwb_desc = {
      .width = static_cast<uint32_t>(buffer_size),
      .height = 1,
      .layers = 1,
      .format = AHARDWAREBUFFER_FORMAT_BLOB,
      .usage = AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY |
               AHARDWAREBUFFER_USAGE_CPU_READ_RARELY |
               AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER};
  if (AHardwareBuffer_allocate(&ahwb_desc, &ahwb) != 0) {
    return absl::InternalError("Failed to allocate AHWB");
  }
  return CreateFromAhwb(tensor_type, ahwb, /*ahwb_offset=*/0,
                        /*deallocator=*/AHardwareBuffer_release);
#else
  return absl::InternalError(
      "AHardwareBuffers are not supported on this platform");
#endif  // LRT_HAS_AHWB_SUPPORT
}

absl::StatusOr<LrtTensorBufferT::Ptr> LrtTensorBufferT::CreateFromIonBuffer(
    const LrtRankedTensorType& tensor_type, void* ion_buffer_addr,
    int ion_buffer_fd, size_t ion_buffer_size, size_t ion_buffer_offset,
    LrtFastRpcDeallocator deallocator) {
  Ptr tensor_buffer(new LrtTensorBufferT(tensor_type, kLrtTensorBufferTypeIon,
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

absl::StatusOr<LrtTensorBufferT::Ptr> LrtTensorBufferT::CreateManagedIonBuffer(
    const LrtRankedTensorType& tensor_type, size_t buffer_size) {
  auto buffer = lrt::internal::IonBuffer::Alloc(
      buffer_size, /*alignment=*/LRT_HOST_MEMORY_BUFFER_ALIGNMENT);
  if (!buffer.ok()) {
    return buffer.status();
  }
  return CreateFromIonBuffer(tensor_type, buffer->addr, buffer->fd, buffer_size,
                             /*ion_buffer_offset=*/0,
                             lrt::internal::IonBuffer::Free);
}

absl::StatusOr<LrtTensorBufferT::Ptr> LrtTensorBufferT::CreateFromDmaBufBuffer(
    const LrtRankedTensorType& tensor_type, void* dmabuf_buffer_addr,
    int dmabuf_buffer_fd, size_t dmabuf_buffer_size,
    size_t dmabuf_buffer_offset, LrtFastRpcDeallocator deallocator) {
  Ptr tensor_buffer(
      new LrtTensorBufferT(tensor_type, kLrtTensorBufferTypeDmaBuf,
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

absl::StatusOr<LrtTensorBufferT::Ptr>
LrtTensorBufferT::CreateManagedDmaBufBuffer(
    const LrtRankedTensorType& tensor_type, size_t buffer_size) {
  auto buffer = lrt::internal::DmaBufBuffer::Alloc(buffer_size);
  if (!buffer.ok()) {
    return buffer.status();
  }
  return CreateFromDmaBufBuffer(tensor_type, buffer->addr, buffer->fd,
                                buffer_size, /*dmabuf_buffer_offset=*/0,
                                lrt::internal::DmaBufBuffer::Free);
}

absl::StatusOr<LrtTensorBufferT::Ptr> LrtTensorBufferT::CreateFromFastRpcBuffer(
    const LrtRankedTensorType& tensor_type, void* fastrpc_buffer_addr,
    int fastrpc_buffer_fd, size_t fastrpc_buffer_size,
    size_t fastrpc_buffer_offset, LrtFastRpcDeallocator deallocator) {
  Ptr tensor_buffer(
      new LrtTensorBufferT(tensor_type, kLrtTensorBufferTypeFastRpc,
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

absl::StatusOr<LrtTensorBufferT::Ptr>
LrtTensorBufferT::CreateManagedFastRpcBuffer(
    const LrtRankedTensorType& tensor_type, size_t buffer_size) {
  auto buffer = lrt::internal::FastRpcBuffer::Alloc(buffer_size);
  if (!buffer.ok()) {
    return buffer.status();
  }
  return CreateFromFastRpcBuffer(tensor_type, buffer->addr, buffer->fd,
                                 buffer_size, /*fastrpc_buffer_offset=*/0,
                                 lrt::internal::FastRpcBuffer::Free);
}

absl::StatusOr<LrtTensorBufferT::Ptr> LrtTensorBufferT::CreateManaged(
    LrtTensorBufferType buffer_type, const LrtRankedTensorType& tensor_type,
    size_t buffer_size) {
  switch (buffer_type) {
    case kLrtTensorBufferTypeHostMemory:
      return CreateManagedOnHostMemory(tensor_type, buffer_size);
    case kLrtTensorBufferTypeAhwb:
      return CreateManagedAhwbBuffer(tensor_type, buffer_size);
    case kLrtTensorBufferTypeIon:
      return CreateManagedIonBuffer(tensor_type, buffer_size);
    case kLrtTensorBufferTypeDmaBuf:
      return CreateManagedDmaBufBuffer(tensor_type, buffer_size);
    case kLrtTensorBufferTypeFastRpc:
      return CreateManagedFastRpcBuffer(tensor_type, buffer_size);
    default:
      return absl::InvalidArgumentError("Unexpected tensor type");
  }
}

absl::Status LrtTensorBufferT::IsValid() const {
  // Check for static dimensions.
  for (auto i = 0; i < tensor_type_.layout.rank; ++i) {
    if (tensor_type_.layout.dimensions[i] <= 0) {
      return absl::InternalError(
          "TensorBuffer must have all static dimensions");
    }
  }

  // Check for sufficient size.
  if (auto num_bytes = lrt::internal::GetNumPackedBytes(tensor_type_);
      !num_bytes.ok()) {
    return num_bytes.status();
  } else if (*num_bytes > buffer_size() - buffer_offset()) {
    return absl::InternalError("Insufficient buffer size");
  }

  // Check for alignment.
  if (buffer_type() == kLrtTensorBufferTypeHostMemory) {
    auto host_buffer = GetHostBuffer();
    if (!host_buffer.ok()) {
      return host_buffer.status();
    }
    if (reinterpret_cast<uintptr_t>(*host_buffer) %
        LRT_HOST_MEMORY_BUFFER_ALIGNMENT) {
      return absl::InternalError("Unaligned host memory pointer");
    }
  }

  return {};
}

absl::StatusOr<void*> LrtTensorBufferT::GetHostBuffer() const {
  if (buffer_type_ != kLrtTensorBufferTypeHostMemory) {
    return absl::InternalError("Unexpected tensor buffer type");
  }
  return std::get<HostBuffer>(buffer_).addr;
}

absl::StatusOr<AHardwareBuffer*> LrtTensorBufferT::GetAhwbBuffer() const {
  if (buffer_type_ != kLrtTensorBufferTypeAhwb) {
    return absl::InternalError("Unexpected tensor buffer type");
  }
  return std::get<AhwbBuffer>(buffer_).ahwb;
}

absl::StatusOr<std::pair<void*, int>> LrtTensorBufferT::GetIonBuffer() const {
  if (buffer_type_ != kLrtTensorBufferTypeIon) {
    return absl::InternalError("Unexpected tensor buffer type");
  }
  auto buffer = std::get<IonBuffer>(buffer_);
  return std::make_pair(buffer.addr, buffer.fd);
}

absl::StatusOr<std::pair<void*, int>> LrtTensorBufferT::GetDmaBufBuffer()
    const {
  if (buffer_type_ != kLrtTensorBufferTypeDmaBuf) {
    return absl::InternalError("Unexpected tensor buffer type");
  }
  auto buffer = std::get<DmaBufBuffer>(buffer_);
  return std::make_pair(buffer.addr, buffer.fd);
}

absl::StatusOr<std::pair<void*, int>> LrtTensorBufferT::GetFastRpcBuffer()
    const {
  if (buffer_type_ != kLrtTensorBufferTypeFastRpc) {
    return absl::InternalError("Unexpected tensor buffer type");
  }
  auto buffer = std::get<FastRpcBuffer>(buffer_);
  return std::make_pair(buffer.addr, buffer.fd);
}

absl::StatusOr<void*> LrtTensorBufferT::Lock(LrtEvent event) {
  if (event) {
    // Only AHWB supports waiting on an input sync fence when locking the
    // buffer. For all other buffer types we wait here.
    if (buffer_type() != kLrtTensorBufferTypeAhwb) {
      if (auto status = LrtEventWait(event, /*timeout_in_ms*/ -1);
          status != kLrtStatusOk) {
        return absl::InternalError("Failed to wait on input event");
      }
    }
  }

  switch (buffer_type()) {
    case kLrtTensorBufferTypeHostMemory:
      return *GetHostBuffer();
    case kLrtTensorBufferTypeAhwb: {
#if LRT_HAS_AHWB_SUPPORT
      auto ahwb = *GetAhwbBuffer();
      int fence = -1;
      if (event) {
        if (auto status = LrtEventGetSyncFenceFd(event, &fence);
            status != kLrtStatusOk) {
          return absl::InternalError("Failed to get sync fence fd from event");
        }
      }
      void* host_addr;
      if (AHardwareBuffer_lock(ahwb,
                               AHARDWAREBUFFER_USAGE_CPU_READ_RARELY |
                                   AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY,
                               fence, /*rect=*/nullptr, &host_addr) != 0) {
        return absl::InternalError("Failed to lock AHWB");
      }
      return host_addr;
#else
      return absl::InternalError("AHWB is not supported on this platform");
#endif  // LRT_HAS_AHWB_SUPPORT
    }
    case kLrtTensorBufferTypeIon:
      return GetIonBuffer()->first;
    case kLrtTensorBufferTypeDmaBuf:
      return GetDmaBufBuffer()->first;
    case kLrtTensorBufferTypeFastRpc:
      return GetFastRpcBuffer()->first;
    default:
      return absl::InternalError("Unexpected tensor buffer type");
  }
}

absl::Status LrtTensorBufferT::Unlock() {
  if (buffer_type() == kLrtTensorBufferTypeAhwb) {
#if LRT_HAS_AHWB_SUPPORT
    auto ahwb = std::get<AhwbBuffer>(buffer_).ahwb;
    if (AHardwareBuffer_unlock(ahwb, /*fence=*/nullptr) != 0) {
      return absl::InternalError("Failed to unlock AHWB");
    }
#else
    return absl::InternalError("AHWB is not supported on this platform");
#endif  // LRT_HAS_AHWB_SUPPORT
  }

  return {};
}
