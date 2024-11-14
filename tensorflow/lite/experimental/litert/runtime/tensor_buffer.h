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
#include <memory>
#include <type_traits>
#include <variant>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"

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

  static absl::StatusOr<Ptr> CreateFromHostMemory(
      const LiteRtRankedTensorType& tensor_type,
      absl::Span<uint8_t> host_memory,
      LiteRtHostMemoryDeallocator deallocator = nullptr);

  static absl::StatusOr<Ptr> CreateFromAhwb(
      const LiteRtRankedTensorType& tensor_type, AHardwareBuffer* ahwb,
      size_t ahwb_offset, LiteRtAhwbDeallocator deallocator = nullptr);

  static absl::StatusOr<Ptr> CreateFromIonBuffer(
      const LiteRtRankedTensorType& tensor_type, void* ion_buffer_addr,
      int ion_buffer_fd, size_t ion_buffer_size, size_t ion_buffer_offset,
      LiteRtIonDeallocator deallocator = nullptr);

  static absl::StatusOr<Ptr> CreateFromDmaBufBuffer(
      const LiteRtRankedTensorType& tensor_type, void* dmabuf_buffer_addr,
      int dmabuf_buffer_fd, size_t dmabuf_buffer_size,
      size_t dmabuf_buffer_offset,
      LiteRtDmaBufDeallocator deallocator = nullptr);

  static absl::StatusOr<Ptr> CreateFromFastRpcBuffer(
      const LiteRtRankedTensorType& tensor_type, void* fastrpc_buffer_addr,
      int fastrpc_buffer_fd, size_t fastrpc_buffer_size,
      size_t fastrpc_buffer_offset,
      LiteRtFastRpcDeallocator deallocator = nullptr);

  static absl::StatusOr<Ptr> CreateManaged(
      LiteRtTensorBufferType buffer_type,
      const LiteRtRankedTensorType& tensor_type, size_t buffer_size);

  LiteRtRankedTensorType tensor_type() const { return tensor_type_; }
  LiteRtTensorBufferType buffer_type() const { return buffer_type_; }
  size_t buffer_size() const { return buffer_size_; }
  size_t buffer_offset() const { return buffer_offset_; }

  absl::StatusOr<void*> GetHostBuffer() const;
  absl::StatusOr<AHardwareBuffer*> GetAhwbBuffer() const;
  absl::StatusOr<std::pair<void*, int>> GetIonBuffer() const;
  absl::StatusOr<std::pair<void*, int>> GetDmaBufBuffer() const;
  absl::StatusOr<std::pair<void*, int>> GetFastRpcBuffer() const;

  absl::StatusOr<void*> Lock(LiteRtEvent event = nullptr);
  absl::Status Unlock();

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

  static absl::StatusOr<Ptr> CreateManagedOnHostMemory(
      const LiteRtRankedTensorType& tensor_type, size_t buffer_size);

  static absl::StatusOr<Ptr> CreateManagedAhwbBuffer(
      const LiteRtRankedTensorType& tensor_type, size_t buffer_size);

  static absl::StatusOr<Ptr> CreateManagedIonBuffer(
      const LiteRtRankedTensorType& tensor_type, size_t buffer_size);

  static absl::StatusOr<Ptr> CreateManagedDmaBufBuffer(
      const LiteRtRankedTensorType& tensor_type, size_t buffer_size);

  static absl::StatusOr<Ptr> CreateManagedFastRpcBuffer(
      const LiteRtRankedTensorType& tensor_type, size_t buffer_size);

  absl::Status IsValid() const;

  LiteRtRankedTensorType tensor_type_;
  std::vector<std::decay_t<decltype(LiteRtLayout::dimensions[0])>> dimensions_;
  std::vector<std::decay_t<decltype(LiteRtLayout::strides[0])>> strides_;
  LiteRtTensorBufferType buffer_type_;
  size_t buffer_size_;
  size_t buffer_offset_;
  std::variant<HostBuffer, AhwbBuffer, IonBuffer, DmaBufBuffer, FastRpcBuffer>
      buffer_;
  mutable std::atomic_int_fast32_t ref_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_TENSOR_BUFFER_H_
