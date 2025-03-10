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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_TENSOR_BUFFER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_TENSOR_BUFFER_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#if LITERT_HAS_OPENGL_SUPPORT
#include <GLES3/gl31.h>
#include <GLES3/gl32.h>
#endif  // LITERT_HAS_OPENGL_SUPPORT
#include "absl/types/span.h"
#include <CL/cl.h>
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_event.h"
#include "tensorflow/lite/experimental/litert/c/litert_layout.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_event.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#if LITERT_HAS_OPENGL_SUPPORT
#include "tensorflow/lite/experimental/litert/runtime/gl_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/gl_texture.h"
#endif  // LITERT_HAS_OPENGL_SUPPORT
#include "tensorflow/lite/experimental/litert/runtime/open_cl_buffer.h"

class LiteRtTensorBufferT {
 public:
  using Ptr = std::unique_ptr<LiteRtTensorBufferT>;

  ~LiteRtTensorBufferT();

  // Make this class non-copiable because it includes raw pointers and resource
  // handles.
  LiteRtTensorBufferT(const LiteRtTensorBufferT&) = delete;
  LiteRtTensorBufferT(LiteRtTensorBufferT&&) = delete;
  LiteRtTensorBufferT& operator=(const LiteRtTensorBufferT&) = delete;
  LiteRtTensorBufferT& operator=(LiteRtTensorBufferT&&) = delete;

  static litert::Expected<Ptr> CreateFromHostMemory(
      const LiteRtRankedTensorType& tensor_type,
      absl::Span<uint8_t> host_memory,
      LiteRtHostMemoryDeallocator deallocator = nullptr);

  static litert::Expected<Ptr> CreateFromAhwb(
      const LiteRtRankedTensorType& tensor_type, AHardwareBuffer* ahwb,
      size_t ahwb_offset, LiteRtAhwbDeallocator deallocator = nullptr);

  static litert::Expected<Ptr> CreateFromIonBuffer(
      const LiteRtRankedTensorType& tensor_type, void* ion_buffer_addr,
      int ion_buffer_fd, size_t ion_buffer_size, size_t ion_buffer_offset,
      LiteRtIonDeallocator deallocator = nullptr);

  static litert::Expected<Ptr> CreateFromDmaBufBuffer(
      const LiteRtRankedTensorType& tensor_type, void* dmabuf_buffer_addr,
      int dmabuf_buffer_fd, size_t dmabuf_buffer_size,
      size_t dmabuf_buffer_offset,
      LiteRtDmaBufDeallocator deallocator = nullptr);

  static litert::Expected<Ptr> CreateFromFastRpcBuffer(
      const LiteRtRankedTensorType& tensor_type, void* fastrpc_buffer_addr,
      int fastrpc_buffer_fd, size_t fastrpc_buffer_size,
      size_t fastrpc_buffer_offset,
      LiteRtFastRpcDeallocator deallocator = nullptr);

  static litert::Expected<Ptr> CreateFromOpenClBuffer(
      const LiteRtRankedTensorType& tensor_type, cl_mem buffer,
      size_t opencl_buffer_size, LiteRtOpenClDeallocator deallocator = nullptr);

#if LITERT_HAS_OPENGL_SUPPORT
  static litert::Expected<Ptr> CreateFromGlBuffer(
      const LiteRtRankedTensorType& tensor_type, GLenum target, GLuint id,
      size_t bytes_size, size_t offset,
      LiteRtGlBufferDeallocator deallocator = nullptr);

  static litert::Expected<Ptr> CreateFromGlTexture(
      const LiteRtRankedTensorType& tensor_type, GLenum target, GLuint id,
      GLenum format, size_t size_bytes, GLint layer,
      LiteRtGlTextureDeallocator deallocator = nullptr);
#endif  // LITERT_HAS_OPENGL_SUPPORT

  static litert::Expected<Ptr> CreateManaged(
      LiteRtTensorBufferType buffer_type,
      const LiteRtRankedTensorType& tensor_type, size_t buffer_size);

  LiteRtRankedTensorType tensor_type() const { return tensor_type_; }
  LiteRtTensorBufferType buffer_type() const { return buffer_type_; }
  size_t buffer_size() const { return buffer_size_; }
  size_t buffer_offset() const { return buffer_offset_; }

  bool HasEvent() const { return event_.has_value(); }

  litert::Expected<LiteRtEvent> GetEvent() const {
    if (!HasEvent()) {
      return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                           "TensorBuffer has no event");
    }
    return event_->Get();
  }

  void SetEvent(LiteRtEvent e) { event_ = litert::Event(e, true); }
  void ClearEvent() { event_ = std::nullopt; }

  litert::Expected<void*> GetHostBuffer();
  litert::Expected<AHardwareBuffer*> GetAhwbBuffer();
  litert::Expected<std::pair<void*, int>> GetIonBuffer();
  litert::Expected<std::pair<void*, int>> GetDmaBufBuffer();
  litert::Expected<std::pair<void*, int>> GetFastRpcBuffer();
  litert::Expected<litert::internal::OpenClBuffer*> GetOpenClBuffer();
#if LITERT_HAS_OPENGL_SUPPORT
  litert::Expected<litert::internal::GlTexture*> GetGlTexture();
  litert::Expected<litert::internal::GlBuffer*> GetGlBuffer();
#endif  // LITERT_HAS_OPENGL_SUPPORT

  litert::Expected<void*> Lock();
  litert::Expected<void> Unlock();

  // Used to duplicate the current tensor buffer. Internally it increases
  // reference count to the underlying buffer.
  void Duplicate() const { Ref(); }

  // Increments reference count by one.
  void Ref() const { ref_.fetch_add(1, std::memory_order_relaxed); }

  // Decrements reference count by one.  If the count remains
  // positive, returns false.  When the count reaches zero, returns
  // true.
  bool Unref() const {
    if (ref_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      return true;
    }
    return false;
  }

  // Gets the current reference count.
  int RefCount() const { return ref_.load(std::memory_order_relaxed); }

 private:
  struct HostBuffer {
    void* addr;
    LiteRtHostMemoryDeallocator deallocator;
  };

  struct AhwbBuffer {
    AHardwareBuffer* ahwb;
    LiteRtAhwbDeallocator deallocator;
  };

  struct IonBuffer {
    void* addr;
    int fd;
    LiteRtIonDeallocator deallocator;
  };

  struct DmaBufBuffer {
    void* addr;
    int fd;
    LiteRtDmaBufDeallocator deallocator;
  };

  struct FastRpcBuffer {
    void* addr;
    int fd;
    LiteRtFastRpcDeallocator deallocator;
  };

  LiteRtTensorBufferT(const LiteRtRankedTensorType& tensor_type,
                      LiteRtTensorBufferType buffer_type, size_t buffer_size,
                      size_t buffer_offset = 0);

  static litert::Expected<Ptr> CreateManagedOnHostMemory(
      const LiteRtRankedTensorType& tensor_type, size_t buffer_size);

  static litert::Expected<Ptr> CreateManagedAhwbBuffer(
      const LiteRtRankedTensorType& tensor_type, size_t buffer_size);

  static litert::Expected<Ptr> CreateManagedIonBuffer(
      const LiteRtRankedTensorType& tensor_type, size_t buffer_size);

  static litert::Expected<Ptr> CreateManagedDmaBufBuffer(
      const LiteRtRankedTensorType& tensor_type, size_t buffer_size);

  static litert::Expected<Ptr> CreateManagedFastRpcBuffer(
      const LiteRtRankedTensorType& tensor_type, size_t buffer_size);

  static litert::Expected<Ptr> CreateManagedOpenClBuffer(
      const LiteRtRankedTensorType& tensor_type, size_t buffer_size);

#if LITERT_HAS_OPENGL_SUPPORT
  static litert::Expected<Ptr> CreateManagedGlBuffer(
      const LiteRtRankedTensorType& tensor_type, size_t buffer_size);
#endif  // LITERT_HAS_OPENGL_SUPPORT

  litert::Expected<void> IsValid();

  LiteRtRankedTensorType tensor_type_;
  std::vector<std::decay_t<decltype(LiteRtLayout::dimensions[0])>> dimensions_;
  std::vector<std::decay_t<decltype(LiteRtLayout::strides[0])>> strides_;
  LiteRtTensorBufferType buffer_type_;
  size_t buffer_size_;
  size_t buffer_offset_;
  std::variant<HostBuffer, AhwbBuffer, IonBuffer, DmaBufBuffer, FastRpcBuffer,
               litert::internal::OpenClBuffer
#if LITERT_HAS_OPENGL_SUPPORT
               ,
               litert::internal::GlBuffer, litert::internal::GlTexture
#endif  // LITERT_HAS_OPENGL_SUPPORT
               >
      buffer_;
  std::optional<litert::Event> event_;
  mutable std::atomic_int_fast32_t ref_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_TENSOR_BUFFER_H_
