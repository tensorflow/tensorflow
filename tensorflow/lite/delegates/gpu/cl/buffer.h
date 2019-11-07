/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_BUFFER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_BUFFER_H_

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

// Buffer represent linear GPU data storage with arbitrary data format.
// Buffer is moveable but not copyable.
class Buffer {
 public:
  Buffer() {}  // just for using Buffer as a class members
  Buffer(cl_mem buffer, size_t size_in_bytes);

  // Move only
  Buffer(Buffer&& buffer);
  Buffer& operator=(Buffer&& buffer);
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  ~Buffer();

  // for profiling and memory statistics
  uint64_t GetMemorySizeInBytes() const { return size_; }

  cl_mem GetMemoryPtr() const { return buffer_; }

  // Writes data to a buffer. Data should point to a region that
  // has exact size in bytes as size_in_bytes(constructor parameter).
  template <typename T>
  Status WriteData(CLCommandQueue* queue, const absl::Span<T> data);

  // Reads data from Buffer into CPU memory.
  template <typename T>
  Status ReadData(CLCommandQueue* queue, std::vector<T>* result) const;

 private:
  void Release();

  cl_mem buffer_ = nullptr;
  size_t size_;
};

Status CreateReadOnlyBuffer(size_t size_in_bytes, CLContext* context,
                            Buffer* result);

Status CreateReadOnlyBuffer(size_t size_in_bytes, const void* data,
                            CLContext* context, Buffer* result);

Status CreateReadWriteBuffer(size_t size_in_bytes, CLContext* context,
                             Buffer* result);

template <typename T>
Status Buffer::WriteData(CLCommandQueue* queue, const absl::Span<T> data) {
  if (size_ != sizeof(T) * data.size()) {
    return InvalidArgumentError(
        "absl::Span<T> data size is different from buffer allocated size.");
  }
  RETURN_IF_ERROR(queue->EnqueueWriteBuffer(buffer_, size_, data.data()));
  return OkStatus();
}

template <typename T>
Status Buffer::ReadData(CLCommandQueue* queue, std::vector<T>* result) const {
  if (size_ % sizeof(T) != 0) {
    return UnknownError("Wrong element size(typename T is not correct?");
  }

  const int elements_count = size_ / sizeof(T);
  result->resize(elements_count);

  return queue->EnqueueReadBuffer(buffer_, size_, result->data());
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_BUFFER_H_
