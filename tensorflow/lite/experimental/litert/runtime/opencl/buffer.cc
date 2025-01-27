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

// This file is a copy of third_party/ml_drift/cl/buffer.cc.
#include "tensorflow/lite/experimental/litert/runtime/opencl/buffer.h"

#include <cstddef>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_context.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/opencl_wrapper.h"

namespace litert {
namespace cl {
absl::Status CreateClBuffer(cl_context context, int size_in_bytes,
                            bool read_only, void* data, cl_mem* result) {
  cl_mem_flags flags = read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
  if (data) {
    flags |= CL_MEM_COPY_HOST_PTR;
  }
  cl_int error_code;
  *result = clCreateBuffer(context, flags, size_in_bytes, data, &error_code);
  if (!*result) {
    return absl::UnknownError(
        absl::StrCat("Failed to allocate device memory (clCreateBuffer): ",
                     std::to_string(error_code)));
  }
  return absl::OkStatus();
}
absl::Status CreateBuffer(size_t size_in_bytes, bool gpu_read_only,
                          const void* data, ClContext* context,
                          Buffer* result) {
  cl_mem buffer;
  auto status = CreateClBuffer(context->context(), size_in_bytes, gpu_read_only,
                               const_cast<void*>(data), &buffer);
  if (!status.ok()) {
    return status;
  }
  *result = Buffer(buffer, size_in_bytes);

  return absl::OkStatus();
}

Buffer::Buffer(cl_mem buffer, size_t size_in_bytes, bool is_sub_buffer)
    : buffer_(buffer), size_(size_in_bytes), is_sub_buffer_(is_sub_buffer) {}

Buffer::Buffer(cl_mem buffer)
    : buffer_(buffer), size_(0), is_sub_buffer_(false), owner_(false) {}

Buffer::Buffer(Buffer&& buffer)
    : buffer_(buffer.buffer_),
      size_(buffer.size_),
      is_sub_buffer_(buffer.is_sub_buffer_),
      owner_(buffer.owner_) {
  buffer.buffer_ = nullptr;
  buffer.size_ = 0;
  buffer.is_sub_buffer_ = false;
}

Buffer& Buffer::operator=(Buffer&& buffer) {
  if (this != &buffer) {
    Release();
    std::swap(size_, buffer.size_);
    std::swap(buffer_, buffer.buffer_);
    std::swap(is_sub_buffer_, buffer.is_sub_buffer_);
    std::swap(owner_, buffer.owner_);
  }
  return *this;
}

void Buffer::Release() {
  if (owner_ && buffer_) {
    clReleaseMemObject(buffer_);
    buffer_ = nullptr;
    size_ = 0;
    is_sub_buffer_ = false;
  }
}

Buffer CreateBufferShared(cl_mem buffer) { return Buffer(buffer); }

absl::Status CreateReadOnlyBuffer(size_t size_in_bytes, ClContext* context,
                                  Buffer* result) {
  return CreateBuffer(size_in_bytes, true, nullptr, context, result);
}

absl::Status CreateReadOnlyBuffer(size_t size_in_bytes, const void* data,
                                  ClContext* context, Buffer* result) {
  return CreateBuffer(size_in_bytes, true, data, context, result);
}

absl::Status CreateReadWriteBuffer(size_t size_in_bytes, ClContext* context,
                                   Buffer* result) {
  return CreateBuffer(size_in_bytes, false, nullptr, context, result);
}

}  // namespace cl
}  // namespace litert
