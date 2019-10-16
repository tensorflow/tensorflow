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

#include "tensorflow/lite/delegates/gpu/cl/buffer.h"

#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
Status CreateBuffer(size_t size_in_bytes, bool gpu_read_only, const void* data,
                    CLContext* context, Buffer* result) {
  cl_mem_flags flags = gpu_read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
  if (data != nullptr) {
    flags |= CL_MEM_COPY_HOST_PTR;
  }
  cl_int error_code;
  cl_mem buffer = clCreateBuffer(context->context(), flags, size_in_bytes,
                                 const_cast<void*>(data), &error_code);
  if (!buffer) {
    return UnknownError(
        absl::StrCat("Failed to allocate device memory with clCreateBuffer",
                     CLErrorCodeToString(error_code)));
  }

  *result = Buffer(buffer, size_in_bytes);

  return OkStatus();
}
}  // namespace

Buffer::Buffer(cl_mem buffer, size_t size_in_bytes)
    : buffer_(buffer), size_(size_in_bytes) {}

Buffer::Buffer(Buffer&& buffer) : buffer_(buffer.buffer_), size_(buffer.size_) {
  buffer.buffer_ = nullptr;
  buffer.size_ = 0;
}

Buffer& Buffer::operator=(Buffer&& buffer) {
  if (this != &buffer) {
    Release();
    std::swap(size_, buffer.size_);
    std::swap(buffer_, buffer.buffer_);
  }
  return *this;
}

Buffer::~Buffer() { Release(); }

void Buffer::Release() {
  if (buffer_) {
    clReleaseMemObject(buffer_);
    buffer_ = nullptr;
    size_ = 0;
  }
}

Status CreateReadOnlyBuffer(size_t size_in_bytes, CLContext* context,
                            Buffer* result) {
  return CreateBuffer(size_in_bytes, true, nullptr, context, result);
}

Status CreateReadOnlyBuffer(size_t size_in_bytes, const void* data,
                            CLContext* context, Buffer* result) {
  return CreateBuffer(size_in_bytes, true, data, context, result);
}

Status CreateReadWriteBuffer(size_t size_in_bytes, CLContext* context,
                             Buffer* result) {
  return CreateBuffer(size_in_bytes, false, nullptr, context, result);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
