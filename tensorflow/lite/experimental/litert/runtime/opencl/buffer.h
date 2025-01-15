// Copyright 2024 The TensorFlow Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file is a copy of third_party/ml_drift/cl/buffer.h.
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_BUFFER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include <CL/cl.h>
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_command_queue.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_context.h"

namespace litert {
namespace cl {

// Buffer represent linear GPU data storage with arbitrary data format.
// Buffer is moveable but not copyable.
class Buffer {
 public:
  Buffer() = default;  // just for using Buffer as a class members
  Buffer(cl_mem buffer, size_t size_in_bytes, bool is_sub_buffer = false);
  explicit Buffer(cl_mem buffer);

  // Move only
  Buffer(Buffer&& buffer);
  Buffer& operator=(Buffer&& buffer);
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  ~Buffer() { Release(); }

  // for profiling and memory statistics
  uint64_t GetMemorySizeInBytes() const { return size_; }

  cl_mem GetMemoryPtr() const { return buffer_; }

  bool IsSubBuffer() const { return is_sub_buffer_; }

  // Writes data to a buffer. Data should point to a region that
  // has exact size in bytes as size_in_bytes(constructor parameter).
  template <typename T>
  absl::Status WriteData(ClCommandQueue* queue, absl::Span<T> data);

  // Reads data from Buffer into CPU memory.
  template <typename T>
  absl::Status ReadData(ClCommandQueue* queue, std::vector<T>* result) const;

 private:
  void Release();

  cl_mem buffer_ = nullptr;
  size_t size_ = 0;
  bool is_sub_buffer_ = false;
  bool owner_ = true;
};

Buffer CreateBufferShared(cl_mem buffer);

absl::Status CreateClBuffer(cl_context context, int size_in_bytes,
                            bool read_only, void* data, cl_mem* result);

absl::Status CreateBuffer(size_t size_in_bytes, bool gpu_read_only,
                          const void* data, ClContext* context, Buffer* result);

absl::Status CreateReadOnlyBuffer(size_t size_in_bytes, ClContext* context,
                                  Buffer* result);

absl::Status CreateReadOnlyBuffer(size_t size_in_bytes, const void* data,
                                  ClContext* context, Buffer* result);

absl::Status CreateReadWriteBuffer(size_t size_in_bytes, ClContext* context,
                                   Buffer* result);

absl::Status CreateReadWriteSubBuffer(const Buffer& parent,
                                      size_t origin_in_bytes,
                                      size_t size_in_bytes, ClContext* context,
                                      Buffer* result);

template <typename T>
absl::Status Buffer::WriteData(ClCommandQueue* queue,
                               const absl::Span<T> data) {
  if (sizeof(T) * data.size() > size_) {
    return absl::InvalidArgumentError(
        "absl::Span<T> data size is greater from buffer allocated size.");
  }
  auto status = queue->EnqueueWriteBuffer(buffer_, size_, data.data());
  if (!status.ok()) {
    return status;
  }
  return absl::OkStatus();
}

template <typename T>
absl::Status Buffer::ReadData(ClCommandQueue* queue,
                              std::vector<T>* result) const {
  if (size_ % sizeof(T) != 0) {
    return absl::UnknownError("Wrong element size(typename T is not correct?");
  }

  const int elements_count = size_ / sizeof(T);
  result->resize(elements_count);

  return queue->EnqueueReadBuffer(buffer_, size_, result->data());
}

}  // namespace cl
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_BUFFER_H_
