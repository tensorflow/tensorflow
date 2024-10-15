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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITE_RT_TENSOR_BUFFER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITE_RT_TENSOR_BUFFER_H_

#include <cstddef>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_event.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_handle.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_model.h"

namespace lrt {

// Tensor and associated backing buffer. C++ equivalent of LrtTensorBuffer.
class TensorBuffer {
 public:
  TensorBuffer() = default;

  // Parameter `owned` indicates if the created TensorBuffer object should take
  // ownership of the provided `tensor_buffer` handle.
  explicit TensorBuffer(LrtTensorBuffer tensor_buffer, bool owned = true)
      : handle_(tensor_buffer, owned ? LrtDestroyTensorBuffer : nullptr) {}

  static absl::StatusOr<TensorBuffer> CreateManaged(
      LrtTensorBufferType buffer_type, const RankedTensorType& tensor_type,
      size_t buffer_size) {
    LrtTensorBuffer tensor_buffer;
    auto& lrt_tensor_type =
        static_cast<const LrtRankedTensorType&>(tensor_type);
    if (auto status = LrtCreateManagedTensorBuffer(
            buffer_type, &lrt_tensor_type, buffer_size, &tensor_buffer);
        status != kLrtStatusOk) {
      return absl::InternalError("Failed to create managed tensor buffer");
    }
    return TensorBuffer(tensor_buffer);
  }

  explicit operator LrtTensorBuffer() { return handle_.Get(); }

  absl::StatusOr<LrtTensorBufferType> BufferType() const {
    LrtTensorBufferType tensor_buffer_type;
    if (auto status =
            LrtGetTensorBufferType(handle_.Get(), &tensor_buffer_type);
        status != kLrtStatusOk) {
      return absl::InternalError("Failed to get tensor buffer type");
    }
    return tensor_buffer_type;
  }

  absl::StatusOr<RankedTensorType> TensorType() const {
    LrtRankedTensorType tensor_type;
    if (auto status = LrtGetTensorBufferTensorType(handle_.Get(), &tensor_type);
        status != kLrtStatusOk) {
      return absl::InternalError("Failed to get tensor type");
    }
    return RankedTensorType(tensor_type);
  }

  absl::StatusOr<size_t> Size() const {
    size_t size;
    if (auto status = LrtGetTensorBufferSize(handle_.Get(), &size);
        status != kLrtStatusOk) {
      return absl::InternalError("Failed to get tensor size");
    }
    return size;
  }

  absl::StatusOr<size_t> Offset() const {
    size_t offset;
    if (auto status = LrtGetTensorBufferOffset(handle_.Get(), &offset);
        status != kLrtStatusOk) {
      return absl::InternalError("Failed to get tensor offset");
    }
    return offset;
  }

  absl::StatusOr<void*> Lock(LrtEvent event = nullptr) {
    void* host_mem_addr;
    if (auto status = LrtLockTensorBuffer(handle_.Get(), &host_mem_addr, event);
        status != kLrtStatusOk) {
      return absl::InternalError("Failed to lock the tensor buffer");
    }
    return host_mem_addr;
  }

  absl::Status Unlock() {
    if (auto status = LrtUnlockTensorBuffer(handle_.Get());
        status != kLrtStatusOk) {
      return absl::InternalError("Failed to unlock the tensor buffer");
    }
    return {};
  }

 private:
  internal::Handle<LrtTensorBuffer> handle_;
};

class TensorBufferScopedLock {
 public:
  ~TensorBufferScopedLock() { (void)tensor_buffer_.Unlock(); }

  static absl::StatusOr<std::pair<TensorBufferScopedLock, void*>> Create(
      TensorBuffer& tensor_buffer, LrtEvent event = nullptr) {
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

}  // namespace lrt

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITE_RT_TENSOR_BUFFER_H_
