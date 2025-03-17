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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPEN_CL_BUFFER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPEN_CL_BUFFER_H_

#include <cstddef>
#include <cstdlib>
#include <utility>

#include "absl/synchronization/mutex.h"
#include <CL/cl.h>
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/opencl_wrapper.h"

namespace litert::internal {

/**
 * The OpenCL buffer class that provides GPU memory allocation and two-way sync
 * between the CPU memory and the GPU OpenCL buffer.
 */
class OpenClBuffer {
 public:
  OpenClBuffer(OpenClBuffer&& other) {
    data_ = other.data_;
    buffer_ = std::move(other.buffer_);
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }

  OpenClBuffer(litert::cl::Buffer buffer, size_t size)
      : buffer_(std::move(buffer)), size_(size) {}

  OpenClBuffer(cl_mem buffer, size_t size, LiteRtOpenClDeallocator deallocator)
      : deallocator_(deallocator), size_(size) {
    if (deallocator_ != nullptr) {
      buffer_ = litert::cl::CreateBufferShared(buffer);
    } else {  // The buffer will be deallocated automatically.
      buffer_ = litert::cl::Buffer(buffer, size);
    }
  }

  ~OpenClBuffer() {
    if (deallocator_ != nullptr) {
      deallocator_(buffer_.GetMemoryPtr());
    }
    if (data_ != nullptr) {
      free(data_);
    };
  }

  cl_mem GetMemoryPtr() { return buffer_.GetMemoryPtr(); }
  // Allocates a CPU memory and conducts a copy from the OpenCL buffer to the
  // CPU memory.
  template <typename T>
  Expected<T*> Lock();

  // Writes the data from the CPU memory to the OpenCL buffer.
  template <typename T>
  Expected<void> Unlock();

  static bool IsSupported();
  static Expected<OpenClBuffer> Alloc(size_t bytes_size);
  size_t size_bytes() const { return size_; }

 private:
  absl::Mutex mutex_;
  // The cpu memory buffer pointer.
  void* data_ = nullptr;
  litert::cl::Buffer buffer_;
  LiteRtOpenClDeallocator deallocator_ = nullptr;
  // The size of the buffer in bytes.
  size_t size_ = 0;
};

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPEN_CL_BUFFER_H_
