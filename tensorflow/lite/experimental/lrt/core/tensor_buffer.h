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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_TENSOR_BUFFER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_TENSOR_BUFFER_H_

#include <memory>
#include <type_traits>
#include <variant>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"

class LrtTensorBufferT {
 public:
  using Ptr = std::unique_ptr<LrtTensorBufferT>;

  ~LrtTensorBufferT();

  // Make this class non-copiable because it includes raw pointers and resource
  // handles.
  LrtTensorBufferT(const LrtTensorBufferT&) = delete;
  LrtTensorBufferT(LrtTensorBufferT&&) = delete;
  LrtTensorBufferT& operator=(const LrtTensorBufferT&) = delete;
  LrtTensorBufferT& operator=(LrtTensorBufferT&&) = delete;

  static absl::StatusOr<Ptr> CreateFromHostMemory(
      const LrtRankedTensorType& tensor_type, absl::Span<uint8_t> host_memory,
      LrtHostMemoryDeallocator deallocator = nullptr);

  static absl::StatusOr<Ptr> CreateFromAhwb(
      const LrtRankedTensorType& tensor_type, AHardwareBuffer* ahwb,
      size_t ahwb_offset, LrtAhwbDeallocator deallocator = nullptr);

  static absl::StatusOr<Ptr> CreateFromIonBuffer(
      const LrtRankedTensorType& tensor_type, void* ion_buffer_addr,
      int ion_buffer_fd, size_t ion_buffer_size, size_t ion_buffer_offset,
      LrtFastRpcDeallocator deallocator = nullptr);

  static absl::StatusOr<Ptr> CreateFromDmaBufBuffer(
      const LrtRankedTensorType& tensor_type, void* dmabuf_buffer_addr,
      int dmabuf_buffer_fd, size_t dmabuf_buffer_size,
      size_t dmabuf_buffer_offset, LrtFastRpcDeallocator deallocator = nullptr);

  static absl::StatusOr<Ptr> CreateFromFastRpcBuffer(
      const LrtRankedTensorType& tensor_type, void* fastrpc_buffer_addr,
      int fastrpc_buffer_fd, size_t fastrpc_buffer_size,
      size_t fastrpc_buffer_offset,
      LrtFastRpcDeallocator deallocator = nullptr);

  static absl::StatusOr<Ptr> CreateManaged(
      LrtTensorBufferType buffer_type, const LrtRankedTensorType& tensor_type,
      size_t buffer_size);

  LrtRankedTensorType tensor_type() const { return tensor_type_; }
  LrtTensorBufferType buffer_type() const { return buffer_type_; }
  size_t buffer_size() const { return buffer_size_; }
  size_t buffer_offset() const { return buffer_offset_; }

  absl::StatusOr<void*> GetHostBuffer() const;
  absl::StatusOr<AHardwareBuffer*> GetAhwbBuffer() const;
  absl::StatusOr<std::pair<void*, int>> GetIonBuffer() const;
  absl::StatusOr<std::pair<void*, int>> GetDmaBufBuffer() const;
  absl::StatusOr<std::pair<void*, int>> GetFastRpcBuffer() const;

  absl::StatusOr<void*> Lock(LrtEvent event = nullptr);
  absl::Status Unlock();

 private:
  struct HostBuffer {
    void* addr;
    LrtHostMemoryDeallocator deallocator;
  };

  struct AhwbBuffer {
    AHardwareBuffer* ahwb;
    LrtAhwbDeallocator deallocator;
  };

  struct IonBuffer {
    void* addr;
    int fd;
    LrtIonDeallocator deallocator;
  };

  struct DmaBufBuffer {
    void* addr;
    int fd;
    LrtDmaBufDeallocator deallocator;
  };

  struct FastRpcBuffer {
    void* addr;
    int fd;
    LrtFastRpcDeallocator deallocator;
  };

  LrtTensorBufferT(const LrtRankedTensorType& tensor_type,
                   LrtTensorBufferType buffer_type, size_t buffer_size,
                   size_t buffer_offset = 0);

  static absl::StatusOr<Ptr> CreateManagedOnHostMemory(
      const LrtRankedTensorType& tensor_type, size_t buffer_size);

  static absl::StatusOr<Ptr> CreateManagedAhwbBuffer(
      const LrtRankedTensorType& tensor_type, size_t buffer_size);

  static absl::StatusOr<Ptr> CreateManagedIonBuffer(
      const LrtRankedTensorType& tensor_type, size_t buffer_size);

  static absl::StatusOr<Ptr> CreateManagedDmaBufBuffer(
      const LrtRankedTensorType& tensor_type, size_t buffer_size);

  static absl::StatusOr<Ptr> CreateManagedFastRpcBuffer(
      const LrtRankedTensorType& tensor_type, size_t buffer_size);

  absl::Status IsValid() const;

  LrtRankedTensorType tensor_type_;
  std::vector<std::decay_t<decltype(LrtLayout::dimensions[0])>> dimensions_;
  std::vector<std::decay_t<decltype(LrtLayout::strides[0])>> strides_;
  LrtTensorBufferType buffer_type_;
  size_t buffer_size_;
  size_t buffer_offset_;
  std::variant<HostBuffer, AhwbBuffer, IonBuffer, DmaBufBuffer, FastRpcBuffer>
      buffer_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_TENSOR_BUFFER_H_
