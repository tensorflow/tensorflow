/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_BUFFER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_BUFFER_H_

#include <string>
#include <vector>

#import <Metal/Metal.h>

#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/metal/gpu_object.h"

namespace tflite {
namespace gpu {
namespace metal {

class Buffer : public GPUObject {
 public:
  Buffer() = default;
  Buffer(id<MTLBuffer> buffer, size_t size_in_bytes);
  explicit Buffer(id<MTLBuffer> buffer);

  // Move only
  Buffer(Buffer&& buffer);
  Buffer& operator=(Buffer&& buffer);
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  ~Buffer();

  // for profiling and memory statistics
  uint64_t GetMemorySizeInBytes() const { return size_; }

  id<MTLBuffer> GetMemoryPtr() const { return buffer_; }

  // Writes data to a buffer. Data should point to a region that
  // has exact size in bytes as size_in_bytes(constructor parameter).
  template <typename T>
  absl::Status WriteData(absl::Span<T> data);
  template <typename T>
  absl::Status WriteData(id<MTLCommandQueue> command_queue, absl::Span<T> data,
                         bool wait_for_completion);

  // Reads data from Buffer into CPU memory.
  template <typename T>
  absl::Status ReadData(std::vector<T>* result) const;

  absl::Status GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                               GPUResourcesWithValue* resources) const override;

  absl::Status CreateFromBufferDescriptor(const BufferDescriptor& desc, id<MTLDevice> device);

 private:
  void Release();

  id<MTLBuffer> buffer_ = nullptr;
  size_t size_;
  bool owner_ = true;
};

Buffer CreateBufferShared(id<MTLBuffer> buffer);

absl::Status CreateBuffer(size_t size_in_bytes, const void* data, id<MTLDevice> device,
                          Buffer* result);

template <typename T>
absl::Status Buffer::WriteData(const absl::Span<T> data) {
  if (sizeof(T) * data.size() > size_) {
    return absl::InvalidArgumentError(
        "absl::Span<T> data size is greater from buffer allocated size.");
  }
  std::memcpy([buffer_ contents], data.data(), size_);
  return absl::OkStatus();
}

template <typename T>
absl::Status Buffer::WriteData(id<MTLCommandQueue> command_queue, absl::Span<T> data,
                               bool wait_for_completion) {
  if (sizeof(T) * data.size() > size_) {
    return absl::InvalidArgumentError(
        "absl::Span<T> data size is greater from buffer allocated size.");
  }
  WriteDataToBuffer(buffer_, /*buffer_offset=*/0, command_queue, data.data(), size_,
                    wait_for_completion);
  return absl::OkStatus();
}

template <typename T>
absl::Status Buffer::ReadData(std::vector<T>* result) const {
  if (size_ % sizeof(T) != 0) {
    return absl::UnknownError("Wrong element size(typename T is not correct?");
  }

  const int elements_count = size_ / sizeof(T);
  result->resize(elements_count);
  std::memcpy(result->data(), [buffer_ contents], size_);

  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_BUFFER_H_
