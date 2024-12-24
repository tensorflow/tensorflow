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
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/cc/litert_event.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_handle.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"

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
    if (auto status = LiteRtDuplicateTensorBuffer(Get());
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to duplicate managed tensor buffer");
    }
    return TensorBuffer(Get());
  }

  static Expected<TensorBuffer> CreateManaged(
      LiteRtTensorBufferType buffer_type, const RankedTensorType& tensor_type,
      size_t buffer_size) {
    LiteRtTensorBuffer tensor_buffer;
    auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);
    if (auto status = LiteRtCreateManagedTensorBuffer(
            buffer_type, &litert_tensor_type, buffer_size, &tensor_buffer);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to create managed tensor buffer");
    }
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

    if (auto status = LiteRtCreateTensorBufferFromHostMemory(
            &litert_tensor_type, host_mem_addr, buffer_size,
            /*deallocator=*/nullptr, &tensor_buffer);
        status != kLiteRtStatusOk) {
      return Unexpected(status,
                        "Failed to create tensor buffer from host memory");
    }
    return TensorBuffer(tensor_buffer);
  }

  litert::Expected<AHardwareBuffer*> GetAhwb() const {
#if LITERT_HAS_AHWB_SUPPORT
    AHardwareBuffer* ahwb;
    if (LiteRtGetTensorBufferAhwb(Get(), &ahwb) == kLiteRtStatusOk) {
      return ahwb;
    } else {
      return litert::Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          "Failed to get AHardwareBuffer from tensor buffer");
    }
#else
    return litert::Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        "AHardwareBuffer is not supported on this platform");
#endif
  }

  Expected<LiteRtTensorBufferType> BufferType() const {
    LiteRtTensorBufferType tensor_buffer_type;
    if (auto status = LiteRtGetTensorBufferType(Get(), &tensor_buffer_type);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to get tensor buffer type");
    }
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
    if (auto status = LiteRtGetTensorBufferSize(Get(), &size);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to get tensor size");
    }
    return size;
  }

  Expected<size_t> Offset() const {
    size_t offset;
    if (auto status = LiteRtGetTensorBufferOffset(Get(), &offset);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to get tensor offset");
    }
    return offset;
  }

  bool HasEvent() const {
    bool has_event;
    internal::AssertOk(LiteRtHasTensorBufferEvent, Get(), &has_event);
    return has_event;
  }

  Expected<Event> GetEvent() const {
    LiteRtEvent event;
    if (auto status = LiteRtGetTensorBufferEvent(Get(), &event);
        status != kLiteRtStatusOk) {
      return Error(status, "Failed to get tensor buffer event");
    }
    return Event(event, /*owned=*/false);
  }

  Expected<void> SetEvent(Event e) {
    if (auto status = LiteRtSetTensorBufferEvent(Get(), e.Get());
        status != kLiteRtStatusOk) {
      return Error(status, "Failed to set tensor buffer event");
    }
    return {};
  }

  Expected<void> ClearEvent() {
    if (auto status = LiteRtClearTensorBufferEvent(Get());
        status != kLiteRtStatusOk) {
      return Error(status, "Failed to clear tensor buffer event");
    }
    return {};
  }

  Expected<void*> Lock(LiteRtEvent event = nullptr) {
    void* host_mem_addr;
    if (auto status = LiteRtLockTensorBuffer(Get(), &host_mem_addr, event);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to lock the tensor buffer");
    }
    return host_mem_addr;
  }

  Expected<void> Unlock() {
    if (auto status = LiteRtUnlockTensorBuffer(Get());
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to unlock the tensor buffer");
    }
    return {};
  }

  // Writes data from the user provided Span<const T> to the tensor buffer.
  // It returns an error if the provided buffer is bigger than the size of the
  // tensor buffer.
  template <typename T>
  Expected<void> Write(absl::Span<const T> data) {
    auto host_mem_addr = Lock();
    if (!host_mem_addr) {
      return host_mem_addr.Error();
    }
    auto size = Size();
    if (!size) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to get TensorBuffer size");
    }
    if (*size < data.size() * sizeof(T)) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          "TensorBuffer size is smaller than the given data size");
    }
    std::memcpy(*host_mem_addr, data.data(), data.size() * sizeof(T));
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
    auto host_mem_addr = Lock();
    if (!host_mem_addr) {
      return host_mem_addr.Error();
    }
    auto size = Size();
    if (!size) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to get TensorBuffer size");
    }
    size_t total_read_size = data.size() * sizeof(T);
    if (*size < total_read_size) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          "TensorBuffer size is smaller than the given data size");
    }
    std::memcpy(data.data(), *host_mem_addr, total_read_size);
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
      TensorBuffer& tensor_buffer, LiteRtEvent event = nullptr) {
    return Create<T>(tensor_buffer.Get(), event);
  }

  template <typename T = void>
  static Expected<std::pair<TensorBufferScopedLock, T*>> Create(
      LiteRtTensorBuffer tensor_buffer, LiteRtEvent event = nullptr) {
    void* host_mem_addr;
    if (auto status =
            LiteRtLockTensorBuffer(tensor_buffer, &host_mem_addr, event);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to lock the tensor buffer");
    }
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
