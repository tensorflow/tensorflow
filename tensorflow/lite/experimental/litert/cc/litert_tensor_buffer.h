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
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_event.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
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
  absl::StatusOr<TensorBuffer> Duplicate() const {
    if (!IsOwned()) {
      return absl::FailedPreconditionError(
          "Cannot duplicate a non-owned tensor buffer");
    }
    if (auto status = LiteRtDuplicateTensorBuffer(Get());
        status != kLiteRtStatusOk) {
      return absl::InternalError("Failed to duplicate managed tensor buffer");
    }
    return TensorBuffer(Get());
  }

  static absl::StatusOr<TensorBuffer> CreateManaged(
      LiteRtTensorBufferType buffer_type, const RankedTensorType& tensor_type,
      size_t buffer_size) {
    LiteRtTensorBuffer tensor_buffer;
    auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);
    if (auto status = LiteRtCreateManagedTensorBuffer(
            buffer_type, &litert_tensor_type, buffer_size, &tensor_buffer);
        status != kLiteRtStatusOk) {
      return absl::InternalError("Failed to create managed tensor buffer");
    }
    return TensorBuffer(tensor_buffer);
  }

  absl::StatusOr<LiteRtTensorBufferType> BufferType() const {
    LiteRtTensorBufferType tensor_buffer_type;
    if (auto status = LiteRtGetTensorBufferType(Get(), &tensor_buffer_type);
        status != kLiteRtStatusOk) {
      return absl::InternalError("Failed to get tensor buffer type");
    }
    return tensor_buffer_type;
  }

  absl::StatusOr<RankedTensorType> TensorType() const {
    LiteRtRankedTensorType tensor_type;
    if (auto status = LiteRtGetTensorBufferTensorType(Get(), &tensor_type);
        status != kLiteRtStatusOk) {
      return absl::InternalError("Failed to get tensor type");
    }
    return RankedTensorType(tensor_type);
  }

  absl::StatusOr<size_t> Size() const {
    size_t size;
    if (auto status = LiteRtGetTensorBufferSize(Get(), &size);
        status != kLiteRtStatusOk) {
      return absl::InternalError("Failed to get tensor size");
    }
    return size;
  }

  absl::StatusOr<size_t> Offset() const {
    size_t offset;
    if (auto status = LiteRtGetTensorBufferOffset(Get(), &offset);
        status != kLiteRtStatusOk) {
      return absl::InternalError("Failed to get tensor offset");
    }
    return offset;
  }

  absl::StatusOr<void*> Lock(LiteRtEvent event = nullptr) {
    void* host_mem_addr;
    if (auto status = LiteRtLockTensorBuffer(Get(), &host_mem_addr, event);
        status != kLiteRtStatusOk) {
      return absl::InternalError("Failed to lock the tensor buffer");
    }
    return host_mem_addr;
  }

  absl::Status Unlock() {
    if (auto status = LiteRtUnlockTensorBuffer(Get());
        status != kLiteRtStatusOk) {
      return absl::InternalError("Failed to unlock the tensor buffer");
    }
    return {};
  }
};

class TensorBufferScopedLock {
 public:
  ~TensorBufferScopedLock() { (void)tensor_buffer_.Unlock(); }

  static absl::StatusOr<std::pair<TensorBufferScopedLock, void*>> Create(
      TensorBuffer& tensor_buffer, LiteRtEvent event = nullptr) {
    auto addr = tensor_buffer.Lock(event);
    if (!addr.ok()) {
      return addr.status();
    }
    return std::make_pair(TensorBufferScopedLock(tensor_buffer), *addr);
  }

 private:
  explicit TensorBufferScopedLock(TensorBuffer& tensor_buffer)
      : tensor_buffer_(tensor_buffer) {}
  TensorBuffer& tensor_buffer_;
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_TENSOR_BUFFER_H_
