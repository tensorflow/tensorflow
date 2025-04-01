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

#include <stdlib.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_event.h"
#include "tensorflow/lite/experimental/litert/c/litert_gl_types.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_types.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_utils.h"
#include "tensorflow/lite/experimental/litert/core/util/tensor_type_util.h"
#include "tensorflow/lite/experimental/litert/runtime/ahwb_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/dmabuf_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/event.h"
#include "tensorflow/lite/experimental/litert/runtime/fastrpc_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/gl_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/gl_texture.h"
#include "tensorflow/lite/experimental/litert/runtime/ion_buffer.h"

#if LITERT_HAS_OPENCL_SUPPORT
#include <CL/cl.h>
#include "tensorflow/lite/experimental/litert/runtime/open_cl_buffer.h"
#endif  // LITERT_HAS_OPENCL_SUPPORT

using litert::Expected;
using litert::Unexpected;

namespace {

template <typename T>
void Copy(size_t array_size, const T* array, std::vector<T>& vec) {
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
    case kLiteRtTensorBufferTypeOpenCl:
      // internal opencl buffer is auto-disposed by the
      // litert::internal::OpenClBuffer destructor.
      break;
    case kLiteRtTensorBufferTypeGlBuffer:
      // internal gl buffer is auto-disposed by the
      // litert::internal::GlBuffer destructor.
    case kLiteRtTensorBufferTypeGlTexture:
      // internal gl texture is auto-disposed by the
      // litert::internal::GlTexture destructor.
      break;
  }
}

Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromHostMemory(
    const LiteRtRankedTensorType& tensor_type, absl::Span<uint8_t> host_memory,
    LiteRtHostMemoryDeallocator deallocator) {
  Ptr tensor_buffer(new LiteRtTensorBufferT(
      tensor_type, kLiteRtTensorBufferTypeHostMemory, host_memory.size()));
  tensor_buffer->buffer_ = HostBuffer{
      .addr = host_memory.data(),
      .deallocator = deallocator,
  };

  if (auto status = tensor_buffer->IsValid(); !status) {
    return Unexpected(status.Error());
  }

  return tensor_buffer;
}

Expected<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedOnHostMemory(
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  void* host_memory_ptr;
  if (auto rc = posix_memalign(
          &host_memory_ptr, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT, buffer_size);
      rc) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to allocate aligned memory");
  }

  LiteRtHostMemoryDeallocator deallocator = ::free;
  LITERT_ASSIGN_OR_RETURN(
      LiteRtTensorBufferT::Ptr tensor_buffer,
      CreateFromHostMemory(
          tensor_type,
          absl::MakeSpan(static_cast<uint8_t*>(host_memory_ptr), buffer_size),
          deallocator));

  return std::move(tensor_buffer);
}

Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromAhwb(
    const LiteRtRankedTensorType& tensor_type, AHardwareBuffer* ahwb,
    size_t ahwb_offset, LiteRtAhwbDeallocator deallocator) {
  LITERT_ASSIGN_OR_RETURN(size_t buffer_size,
                          litert::internal::AhwbBuffer::GetSize(ahwb));

  Ptr tensor_buffer(new LiteRtTensorBufferT(
      tensor_type, kLiteRtTensorBufferTypeAhwb, buffer_size, ahwb_offset));
  tensor_buffer->buffer_ = AhwbBuffer{
      .ahwb = ahwb,
      .deallocator = deallocator,
  };

  if (auto status = tensor_buffer->IsValid(); !status) {
    return Unexpected(status.Error());
  }

  return tensor_buffer;
}

Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateManagedAhwbBuffer(
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  LITERT_ASSIGN_OR_RETURN(litert::internal::AhwbBuffer buffer,
                          litert::internal::AhwbBuffer::Alloc(buffer_size));
  return CreateFromAhwb(tensor_type, buffer.ahwb, /*ahwb_offset=*/0,
                        /*deallocator=*/litert::internal::AhwbBuffer::Free);
}

Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromIonBuffer(
    const LiteRtRankedTensorType& tensor_type, void* ion_buffer_addr,
    int ion_buffer_fd, size_t ion_buffer_size, size_t ion_buffer_offset,
    LiteRtIonDeallocator deallocator) {
  if (!ion_buffer_addr) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid ION buffer address");
  }
  if (ion_buffer_fd < 0) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid ION buffer fd");
  }

  Ptr tensor_buffer(
      new LiteRtTensorBufferT(tensor_type, kLiteRtTensorBufferTypeIon,
                              ion_buffer_size, ion_buffer_offset));
  tensor_buffer->buffer_ = IonBuffer{
      .addr = ion_buffer_addr,
      .fd = ion_buffer_fd,
      .deallocator = deallocator,
  };

  if (auto status = tensor_buffer->IsValid(); !status) {
    return Unexpected(status.Error());
  }

  return tensor_buffer;
}

Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateManagedIonBuffer(
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  auto buffer = litert::internal::IonBuffer::Alloc(
      buffer_size, /*alignment=*/LITERT_HOST_MEMORY_BUFFER_ALIGNMENT);
  if (!buffer) {
    return Unexpected(buffer.Error());
  }
  return CreateFromIonBuffer(tensor_type, buffer->addr, buffer->fd, buffer_size,
                             /*ion_buffer_offset=*/0,
                             litert::internal::IonBuffer::Free);
}

Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromDmaBufBuffer(
    const LiteRtRankedTensorType& tensor_type, void* dmabuf_buffer_addr,
    int dmabuf_buffer_fd, size_t dmabuf_buffer_size,
    size_t dmabuf_buffer_offset, LiteRtDmaBufDeallocator deallocator) {
  if (!dmabuf_buffer_addr) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid DMA-BUF buffer address");
  }
  if (dmabuf_buffer_fd < 0) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid DMA-BUF buffer fd");
  }

  Ptr tensor_buffer(
      new LiteRtTensorBufferT(tensor_type, kLiteRtTensorBufferTypeDmaBuf,
                              dmabuf_buffer_size, dmabuf_buffer_offset));
  tensor_buffer->buffer_ = DmaBufBuffer{
      .addr = dmabuf_buffer_addr,
      .fd = dmabuf_buffer_fd,
      .deallocator = deallocator,
  };

  if (auto status = tensor_buffer->IsValid(); !status) {
    return Unexpected(status.Error());
  }

  return tensor_buffer;
}

Expected<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedDmaBufBuffer(
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  auto buffer = litert::internal::DmaBufBuffer::Alloc(buffer_size);
  if (!buffer) {
    return Unexpected(buffer.Error());
  }
  return CreateFromDmaBufBuffer(tensor_type, buffer->addr, buffer->fd,
                                buffer_size, /*dmabuf_buffer_offset=*/0,
                                litert::internal::DmaBufBuffer::Free);
}

Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromFastRpcBuffer(
    const LiteRtRankedTensorType& tensor_type, void* fastrpc_buffer_addr,
    int fastrpc_buffer_fd, size_t fastrpc_buffer_size,
    size_t fastrpc_buffer_offset, LiteRtFastRpcDeallocator deallocator) {
  if (!fastrpc_buffer_addr) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid FastRPC buffer address");
  }
  if (fastrpc_buffer_fd < 0) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid FastRPC buffer fd");
  }

  Ptr tensor_buffer(
      new LiteRtTensorBufferT(tensor_type, kLiteRtTensorBufferTypeFastRpc,
                              fastrpc_buffer_size, fastrpc_buffer_offset));
  tensor_buffer->buffer_ = FastRpcBuffer{
      .addr = fastrpc_buffer_addr,
      .fd = fastrpc_buffer_fd,
      .deallocator = deallocator,
  };

  if (auto status = tensor_buffer->IsValid(); !status) {
    return Unexpected(status.Error());
  }

  return tensor_buffer;
}

Expected<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedFastRpcBuffer(
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  auto buffer = litert::internal::FastRpcBuffer::Alloc(buffer_size);
  if (!buffer) {
    return Unexpected(buffer.Error());
  }
  return CreateFromFastRpcBuffer(tensor_type, buffer->addr, buffer->fd,
                                 buffer_size, /*fastrpc_buffer_offset=*/0,
                                 litert::internal::FastRpcBuffer::Free);
}

#if LITERT_HAS_OPENCL_SUPPORT
Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromOpenClBuffer(
    const LiteRtRankedTensorType& tensor_type, cl_mem buffer,
    size_t buffer_size, LiteRtOpenClDeallocator deallocator) {
  Ptr tensor_buffer(new LiteRtTensorBufferT(
      tensor_type, kLiteRtTensorBufferTypeOpenCl, buffer_size));
  tensor_buffer->buffer_.emplace<litert::internal::OpenClBuffer>(
      buffer, buffer_size, deallocator);
  return tensor_buffer;
}

Expected<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedOpenClBuffer(
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  auto buffer = litert::internal::OpenClBuffer::Alloc(buffer_size);
  if (!buffer) {
    return Unexpected(buffer.Error());
  }
  Ptr tensor_buffer(new LiteRtTensorBufferT(
      tensor_type, kLiteRtTensorBufferTypeOpenCl, buffer_size));
  tensor_buffer->buffer_.emplace<litert::internal::OpenClBuffer>(
      std::move(*buffer));
  return tensor_buffer;
}
#endif  // LITERT_HAS_OPENCL_SUPPORT

Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromGlBuffer(
    const LiteRtRankedTensorType& tensor_type, LiteRtGLenum target,
    LiteRtGLuint id, size_t size_bytes, size_t offset,
    LiteRtGlBufferDeallocator deallocator) {
  Ptr tensor_buffer(new LiteRtTensorBufferT(
      tensor_type, kLiteRtTensorBufferTypeGlBuffer, size_bytes));
  tensor_buffer->buffer_.emplace<litert::internal::GlBuffer>(
      target, id, size_bytes, offset, deallocator);
  return tensor_buffer;
}

Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateManagedGlBuffer(
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  auto buffer = litert::internal::GlBuffer::Alloc(buffer_size);
  if (!buffer) {
    return Unexpected(buffer.Error());
  }
  Ptr tensor_buffer(new LiteRtTensorBufferT(
      tensor_type, kLiteRtTensorBufferTypeGlBuffer, buffer_size));
  tensor_buffer->buffer_.emplace<litert::internal::GlBuffer>(
      std::move(*buffer));
  return tensor_buffer;
}

Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromGlTexture(
    const LiteRtRankedTensorType& tensor_type, LiteRtGLenum target,
    LiteRtGLuint id, LiteRtGLenum format, size_t size_bytes, LiteRtGLint layer,
    LiteRtGlTextureDeallocator deallocator) {
  Ptr tensor_buffer(new LiteRtTensorBufferT(
      tensor_type, kLiteRtTensorBufferTypeGlTexture, size_bytes));
  tensor_buffer->buffer_.emplace<litert::internal::GlTexture>(
      litert::internal::GlTexture(target, id, format, size_bytes, layer,
                                  deallocator));
  return tensor_buffer;
}

Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateManaged(
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
    case kLiteRtTensorBufferTypeOpenCl: {
#if LITERT_HAS_OPENCL_SUPPORT
      return CreateManagedOpenClBuffer(tensor_type, buffer_size);
#else
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "OpenCL buffers are not supported.");
#endif  // LITERT_HAS_OPENCL_SUPPORT
    }
    case kLiteRtTensorBufferTypeGlBuffer: {
      return CreateManagedGlBuffer(tensor_type, buffer_size);
    }
    case kLiteRtTensorBufferTypeGlTexture: {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "LiteRT does not support managed GL textures.");
    }
    default:
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Unexpected tensor type");
  }
}

Expected<void> LiteRtTensorBufferT::IsValid() {
  // Check for static dimensions.
  for (auto i = 0; i < tensor_type_.layout.rank; ++i) {
    if (tensor_type_.layout.dimensions[i] <= 0) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "TensorBuffer must have all static dimensions");
    }
  }

  // Check for valid offset.
  if (buffer_offset() >= buffer_size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid buffer offset");
  }

  // Check for sufficient size.
  if (auto num_bytes = litert::internal::GetNumPackedBytes(tensor_type_);
      !num_bytes) {
    return Unexpected(num_bytes.Error());
  } else if (*num_bytes > buffer_size() - buffer_offset()) {
    const std::string error_message = absl::StrFormat(
        "Insufficient buffer size: Required %d bytes, actual size %d bytes",
        *num_bytes, buffer_size() - buffer_offset());
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, error_message);
  }

  // Check for proper alignment.
  if (buffer_type() == kLiteRtTensorBufferTypeHostMemory) {
    auto host_buffer = GetHostBuffer();
    if (!host_buffer) {
      return Unexpected(host_buffer.Error());
    }
    if (reinterpret_cast<uintptr_t>(*host_buffer) %
        LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Unaligned host memory pointer");
    }
  }

  return {};
}

Expected<void*> LiteRtTensorBufferT::GetHostBuffer() {
  if (buffer_type_ == kLiteRtTensorBufferTypeHostMemory) {
    return std::get<HostBuffer>(buffer_).addr;
  }
  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      absl::StrFormat("Cannot get %s buffer from %s tensor buffer",
                      BufferTypeToString(kLiteRtTensorBufferTypeHostMemory),
                      BufferTypeToString(buffer_type_)));
}

Expected<AHardwareBuffer*> LiteRtTensorBufferT::GetAhwbBuffer() {
  if (buffer_type_ == kLiteRtTensorBufferTypeAhwb) {
    return std::get<AhwbBuffer>(buffer_).ahwb;
  }
  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      absl::StrFormat("Cannot get %s buffer from %s tensor buffer",
                      BufferTypeToString(kLiteRtTensorBufferTypeAhwb),
                      BufferTypeToString(buffer_type_)));
}

Expected<std::pair<void*, int>> LiteRtTensorBufferT::GetIonBuffer() {
  if (buffer_type_ == kLiteRtTensorBufferTypeIon) {
    auto buffer = std::get<IonBuffer>(buffer_);
    return std::make_pair(buffer.addr, buffer.fd);
  }
  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      absl::StrFormat("Cannot get %s buffer from %s tensor buffer",
                      BufferTypeToString(kLiteRtTensorBufferTypeIon),
                      BufferTypeToString(buffer_type_)));
}

Expected<std::pair<void*, int>> LiteRtTensorBufferT::GetDmaBufBuffer() {
  if (buffer_type_ == kLiteRtTensorBufferTypeDmaBuf) {
    auto buffer = std::get<DmaBufBuffer>(buffer_);
    return std::make_pair(buffer.addr, buffer.fd);
  }
  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      absl::StrFormat("Cannot get %s buffer from %s tensor buffer",
                      BufferTypeToString(kLiteRtTensorBufferTypeDmaBuf),
                      BufferTypeToString(buffer_type_)));
}

Expected<std::pair<void*, int>> LiteRtTensorBufferT::GetFastRpcBuffer() {
  if (buffer_type_ == kLiteRtTensorBufferTypeFastRpc) {
    auto buffer = std::get<FastRpcBuffer>(buffer_);
    return std::make_pair(buffer.addr, buffer.fd);
  }

  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      absl::StrFormat("Cannot get %s buffer from %s tensor buffer",
                      BufferTypeToString(kLiteRtTensorBufferTypeFastRpc),
                      BufferTypeToString(buffer_type_)));
}

#if LITERT_HAS_OPENCL_SUPPORT
Expected<litert::internal::OpenClBuffer*>
LiteRtTensorBufferT::GetOpenClBuffer() {
  if (buffer_type_ == kLiteRtTensorBufferTypeOpenCl) {
    return &std::get<litert::internal::OpenClBuffer>(buffer_);
  }
  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      absl::StrFormat("Cannot get %s buffer from %s tensor buffer",
                      BufferTypeToString(kLiteRtTensorBufferTypeOpenCl),
                      BufferTypeToString(buffer_type_)));
}
#endif  // LITERT_HAS_OPENCL_SUPPORT

Expected<litert::internal::GlTexture*> LiteRtTensorBufferT::GetGlTexture() {
  if (buffer_type_ != kLiteRtTensorBufferTypeGlTexture) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Unexpected tensor buffer type");
  }
  return &std::get<litert::internal::GlTexture>(buffer_);
}

Expected<litert::internal::GlBuffer*> LiteRtTensorBufferT::GetGlBuffer() {
  if (buffer_type_ == kLiteRtTensorBufferTypeGlBuffer) {
    return &std::get<litert::internal::GlBuffer>(buffer_);
  }
#if LITERT_HAS_AHWB_SUPPORT
  if (buffer_type_ == kLiteRtTensorBufferTypeAhwb) {
    if (auto it = memory_backed_buffers_.find(kLiteRtTensorBufferTypeGlBuffer);
        it != memory_backed_buffers_.end()) {
      BufferVariant& memory_backed_buffer = it->second;
      return &std::get<litert::internal::GlBuffer>(memory_backed_buffer);
    }
    // Create a new GL buffer from the AHWB buffer if not found.
    litert::internal::AhwbBuffer ahwb_buffer = {
        .ahwb = std::get<AhwbBuffer>(buffer_).ahwb};

    LITERT_ASSIGN_OR_RETURN(
        litert::internal::GlBuffer gl_buffer_from_ahwb,
        litert::internal::GlBuffer::AllocFromAhwbBuffer(ahwb_buffer));

    auto [it, inserted] = memory_backed_buffers_.insert(
        {kLiteRtTensorBufferTypeGlBuffer, std::move(gl_buffer_from_ahwb)});
    LITERT_RETURN_IF_ERROR(
        inserted == true,
        Unexpected(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to insert GL buffer into memory backed buffers"));
    return &std::get<litert::internal::GlBuffer>(it->second);
  }
#endif  // LITERT_HAS_AHWB_SUPPORT

  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      absl::StrFormat("Cannot get %s buffer from %s tensor buffer",
                      BufferTypeToString(kLiteRtTensorBufferTypeGlBuffer),
                      BufferTypeToString(buffer_type_)));
}

Expected<void*> LiteRtTensorBufferT::Lock() {
  if (event_ != nullptr) {
    // Only AHWB supports waiting on an input sync fence when locking the
    // buffer. For all other buffer types we wait here.
    if (buffer_type() != kLiteRtTensorBufferTypeAhwb) {
      LITERT_RETURN_IF_ERROR(event_->Wait(/*timeout_in_ms=*/-1));
    }
  }

  switch (buffer_type()) {
    case kLiteRtTensorBufferTypeHostMemory:
      return *GetHostBuffer();
    case kLiteRtTensorBufferTypeAhwb:
      return litert::internal::AhwbBuffer::Lock(
          *GetAhwbBuffer(), event_ != nullptr ? event_.get() : nullptr);
    case kLiteRtTensorBufferTypeIon:
      return GetIonBuffer()->first;
    case kLiteRtTensorBufferTypeDmaBuf:
      return GetDmaBufBuffer()->first;
    case kLiteRtTensorBufferTypeFastRpc:
      return GetFastRpcBuffer()->first;
    case kLiteRtTensorBufferTypeOpenCl: {
#if LITERT_HAS_OPENCL_SUPPORT
      auto opencl_buffer = *GetOpenClBuffer();
      auto host_memory_ptr = opencl_buffer->Lock<float>();
      if (host_memory_ptr.HasValue()) {
        return Expected<void*>(host_memory_ptr.Value());
      } else {
        return Unexpected(host_memory_ptr.Error());
      }
#else
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "OpenCL buffers are not supported");
#endif  // LITERT_HAS_OPENCL_SUPPORT
    }
    case kLiteRtTensorBufferTypeGlBuffer: {
#if LITERT_HAS_OPENGL_SUPPORT
      auto gl_buffer = *GetGlBuffer();
      auto host_memory_ptr = gl_buffer->Lock<float>();
      if (host_memory_ptr.HasValue()) {
        return Expected<void*>(host_memory_ptr.Value());
      } else {
        return Unexpected(host_memory_ptr.Error());
      }
#else
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "OpenGL buffers are not supported");
#endif  // LITERT_HAS_OPENGL_SUPPORT
    }
    default:
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Unexpected tensor buffer type");
  }
}

Expected<void> LiteRtTensorBufferT::Unlock() {
  switch (buffer_type()) {
    case kLiteRtTensorBufferTypeAhwb: {
      auto ahwb = std::get<AhwbBuffer>(buffer_).ahwb;
      return litert::internal::AhwbBuffer::Unlock(ahwb);
    }
    case kLiteRtTensorBufferTypeOpenCl: {
#if LITERT_HAS_OPENCL_SUPPORT
      auto opencl_buffer = *GetOpenClBuffer();
      return opencl_buffer->Unlock<float>();
#else
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "OpenCL buffers are not supported");
#endif  // LITERT_HAS_OPENCL_SUPPORT
    }
    case kLiteRtTensorBufferTypeGlBuffer: {
#if LITERT_HAS_OPENGL_SUPPORT
      auto gl_buffer = *GetGlBuffer();
      return gl_buffer->Unlock<float>();
#else
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "OpenGL buffers are not supported");
#endif  // LITERT_HAS_OPENGL_SUPPORT
    }
    default:
      return {};
  }
}
