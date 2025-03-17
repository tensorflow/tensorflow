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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_TENSOR_BUFFER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_TENSOR_BUFFER_H_

#include <cstddef>
#include <cstring>
#include <utility>

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_event.h"
#include "tensorflow/lite/experimental/litert/c/litert_gl_types.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_types.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/cc/litert_event.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_handle.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"

#if LITERT_HAS_OPENCL_SUPPORT
#include <CL/cl.h>
#endif

namespace litert {

// Tensor and associated backing buffer. C++ equivalent of LiteRtTensorBuffer.
class TensorBuffer
    : public internal::Handle<LiteRtTensorBuffer, LiteRtDestroyTensorBuffer> {
 public:
  TensorBuffer() = default;

  // Parameter `owned` indicates if the created TensorBuffer object should take
  // ownership of the provided `tensor_buffer` handle.
  explicit TensorBuffer(LiteRtTensorBuffer tensor_buffer, bool owned = true)
      : internal::Handle<LiteRtTensorBuffer, LiteRtDestroyTensorBuffer>(
            tensor_buffer, owned) {}

  // Creates a duplicate of the current TensorBuffer object. The returned
  // object is reference counted so the underlying LiteRtTensorBuffer handle is
  // not released with the destructor until the last reference is removed.
  Expected<TensorBuffer> Duplicate() const {
    if (!IsOwned()) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Cannot duplicate a non-owned tensor buffer");
    }
    LITERT_RETURN_IF_ERROR(LiteRtDuplicateTensorBuffer(Get()));
    return TensorBuffer(Get());
  }

  static Expected<TensorBuffer> CreateManaged(
      LiteRtTensorBufferType buffer_type, const RankedTensorType& tensor_type,
      size_t buffer_size) {
    LiteRtTensorBuffer tensor_buffer;
    auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);
    LITERT_RETURN_IF_ERROR(LiteRtCreateManagedTensorBuffer(
        buffer_type, &litert_tensor_type, buffer_size, &tensor_buffer));
    return TensorBuffer(tensor_buffer);
  }

  // Creates a TensorBuffer object that wraps the provided host memory.
  // The provided host memory is not owned by the TensorBuffer object and must
  // outlive the TensorBuffer object.
  static Expected<TensorBuffer> CreateFromHostMemory(
      const RankedTensorType& tensor_type, void* host_mem_addr,
      size_t buffer_size) {
    LiteRtTensorBuffer tensor_buffer;
    auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);

    LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferFromHostMemory(
        &litert_tensor_type, host_mem_addr, buffer_size,
        /*deallocator=*/nullptr, &tensor_buffer));
    return TensorBuffer(tensor_buffer);
  }

  // Creates a TensorBuffer object that wraps an Android Hardware Buffer. Note
  // that the provided AHardwareBuffer is not owned by the TensorBuffer object
  // and must outlive the TensorBuffer object. The `ahwb_offset` parameter
  // specifies the offset in bytes from the start of the AHardwareBuffer where
  // the tensor data starts.
  static Expected<TensorBuffer> CreateFromAhwb(
      const RankedTensorType& tensor_type, AHardwareBuffer* ahwb,
      size_t ahwb_offset) {
#if LITERT_HAS_AHWB_SUPPORT
    LiteRtTensorBuffer tensor_buffer;
    auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);

    LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferFromAhwb(
        &litert_tensor_type, ahwb, ahwb_offset,
        /*deallocator=*/nullptr, &tensor_buffer));
    return TensorBuffer(tensor_buffer);
#else
    return litert::Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        "AHardwareBuffer is not supported on this platform");
#endif
  }

  litert::Expected<AHardwareBuffer*> GetAhwb() const {
#if LITERT_HAS_AHWB_SUPPORT
    AHardwareBuffer* ahwb;
    LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferAhwb(Get(), &ahwb));
    return ahwb;
#else
    return litert::Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        "AHardwareBuffer is not supported on this platform");
#endif
  }

  struct DmaBuf {
    void* addr;
    int fd;
  };

  litert::Expected<DmaBuf> GetDmaBuf() const {
#if LITERT_HAS_DMABUF_SUPPORT
    DmaBuf dma_buf;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetTensorBufferDmaBufBuffer(Get(), &dma_buf.addr, &dma_buf.fd));
    return dma_buf;
#else
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "DMA-BUF is not supported on this platform");
#endif
  }

#if LITERT_HAS_OPENCL_SUPPORT
  Expected<cl_mem> GetOpenClBuffer() const {
    cl_mem cl_mem;
    LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferOpenClBuffer(Get(), &cl_mem));
    return cl_mem;
  }
#endif

  struct GlBuffer {
    LiteRtGLenum target;
    LiteRtGLuint id;
    size_t size_bytes;
    size_t offset;
  };

  static Expected<TensorBuffer> CreateFromGlBuffer(
      const RankedTensorType& tensor_type, LiteRtGLenum target, LiteRtGLuint id,
      size_t size_bytes, size_t offset) {
    LiteRtTensorBuffer tensor_buffer;
    auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);
    LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferFromGlBuffer(
        &litert_tensor_type, target, id, size_bytes, offset,
        /*deallocator=*/nullptr, &tensor_buffer));
    return TensorBuffer(tensor_buffer);
  }

  Expected<GlBuffer> GetGlBuffer() const {
    GlBuffer gl_buffer;
    LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferGlBuffer(
        Get(), &gl_buffer.target, &gl_buffer.id, &gl_buffer.size_bytes,
        &gl_buffer.offset));
    return gl_buffer;
  }
  struct GlTexture {
    LiteRtGLenum target;
    LiteRtGLuint id;
    LiteRtGLenum format;
    size_t size_bytes;
    LiteRtGLint layer;
  };
  static Expected<TensorBuffer> CreateFromGlTexture(
      const RankedTensorType& tensor_type, LiteRtGLenum target, LiteRtGLuint id,
      LiteRtGLenum format, size_t size_bytes, LiteRtGLint layer) {
    LiteRtTensorBuffer tensor_buffer;
    auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);
    LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferFromGlTexture(
        &litert_tensor_type, target, id, format, size_bytes, layer,
        /*deallocator=*/nullptr, &tensor_buffer));
    return TensorBuffer(tensor_buffer);
  }

  Expected<GlTexture> GetGlTexture() const {
    GlTexture gl_texture;
    LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferGlTexture(
        Get(), &gl_texture.target, &gl_texture.id, &gl_texture.format,
        &gl_texture.size_bytes, &gl_texture.layer));
    return gl_texture;
  }

  Expected<LiteRtTensorBufferType> BufferType() const {
    LiteRtTensorBufferType tensor_buffer_type;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetTensorBufferType(Get(), &tensor_buffer_type));
    return tensor_buffer_type;
  }

  Expected<RankedTensorType> TensorType() const {
    LiteRtRankedTensorType tensor_type;
    if (auto status = LiteRtGetTensorBufferTensorType(Get(), &tensor_type);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to get tensor type");
    }
    return RankedTensorType(tensor_type);
  }

  Expected<size_t> Size() const {
    size_t size;
    LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferSize(Get(), &size));
    return size;
  }

  Expected<size_t> Offset() const {
    size_t offset;
    LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferOffset(Get(), &offset));
    return offset;
  }

  bool HasEvent() const {
    bool has_event;
    internal::AssertOk(LiteRtHasTensorBufferEvent, Get(), &has_event);
    return has_event;
  }

  Expected<Event> GetEvent() const {
    LiteRtEvent event;
    LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferEvent(Get(), &event));
    return Event(event, /*owned=*/false);
  }

  // The function takes ownership of the passed event e.
  Expected<void> SetEvent(Event&& e) {
    if (!e.IsOwned()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Expected an owned event");
    }
    LITERT_RETURN_IF_ERROR(LiteRtSetTensorBufferEvent(Get(), e.Release()));
    return {};
  }

  Expected<void> ClearEvent() {
    LITERT_RETURN_IF_ERROR(LiteRtClearTensorBufferEvent(Get()));
    return {};
  }

  Expected<void*> Lock() {
    void* host_mem_addr;
    LITERT_RETURN_IF_ERROR(LiteRtLockTensorBuffer(Get(), &host_mem_addr));
    return host_mem_addr;
  }

  Expected<void> Unlock() {
    LITERT_RETURN_IF_ERROR(LiteRtUnlockTensorBuffer(Get()));
    return {};
  }

  // Writes data from the user provided Span<const T> to the tensor buffer.
  // It returns an error if the provided buffer is bigger than the size of the
  // tensor buffer.
  template <typename T>
  Expected<void> Write(absl::Span<const T> data) {
    LITERT_ASSIGN_OR_RETURN(void* host_mem_addr, Lock());
    LITERT_ASSIGN_OR_RETURN(size_t size, Size());
    if (size < data.size() * sizeof(T)) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          "TensorBuffer size is smaller than the given data size");
    }
    std::memcpy(host_mem_addr, data.data(), data.size() * sizeof(T));
    Unlock();
    return {};
  }

  // Reads data into the user provided Span<T> from the tensor buffer.
  // If the provided buffer is smaller than the size of the tensor buffer, the
  // data will be read up to the size of the provided buffer.
  // It returns an error if the provided buffer is bigger than the size of the
  // tensor buffer.
  template <typename T>
  Expected<void> Read(absl::Span<T> data) {
    LITERT_ASSIGN_OR_RETURN(void* host_mem_addr, Lock());
    LITERT_ASSIGN_OR_RETURN(size_t size, Size());
    size_t total_read_size = data.size() * sizeof(T);
    if (size < total_read_size) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          "TensorBuffer size is smaller than the given data size");
    }
    std::memcpy(data.data(), host_mem_addr, total_read_size);
    Unlock();
    return {};
  }
};

class TensorBufferScopedLock {
 public:
  TensorBufferScopedLock(const TensorBufferScopedLock& arg) = delete;
  TensorBufferScopedLock(TensorBufferScopedLock&& arg) = default;
  ~TensorBufferScopedLock() { (void)LiteRtUnlockTensorBuffer(tensor_buffer_); }

  template <typename T = void>
  static Expected<std::pair<TensorBufferScopedLock, T*>> Create(
      TensorBuffer& tensor_buffer) {
    return Create<T>(tensor_buffer.Get());
  }

  template <typename T = void>
  static Expected<std::pair<TensorBufferScopedLock, T*>> Create(
      LiteRtTensorBuffer tensor_buffer) {
    void* host_mem_addr;
    LITERT_RETURN_IF_ERROR(
        LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr));
    return std::make_pair(TensorBufferScopedLock(tensor_buffer),
                          static_cast<T*>(host_mem_addr));
  }

 private:
  explicit TensorBufferScopedLock(LiteRtTensorBuffer& tensor_buffer)
      : tensor_buffer_(tensor_buffer) {}

  LiteRtTensorBuffer tensor_buffer_;
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_TENSOR_BUFFER_H_
