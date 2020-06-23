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

absl::Status CreateBuffer(size_t size_in_bytes, bool gpu_read_only,
                          const void* data, CLContext* context,
                          Buffer* result) {
  cl_mem_flags flags = gpu_read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
  if (data != nullptr) {
    flags |= CL_MEM_COPY_HOST_PTR;
  }
  cl_int error_code;
  cl_mem buffer = clCreateBuffer(context->context(), flags, size_in_bytes,
                                 const_cast<void*>(data), &error_code);
  if (!buffer) {
    return absl::UnknownError(
        absl::StrCat("Failed to allocate device memory with clCreateBuffer",
                     CLErrorCodeToString(error_code)));
  }

  *result = Buffer(buffer, size_in_bytes);

  return absl::OkStatus();
}
}  // namespace

GPUResources BufferDescriptor::GetGPUResources(AccessType access_type) const {
  GPUResources resources;
  GPUBufferDescriptor desc;
  desc.data_type = element_type;
  desc.access_type = access_type;
  desc.element_size = element_size;
  resources.buffers.push_back({"buffer", desc});
  return resources;
}

absl::Status BufferDescriptor::PerformSelector(
    const std::string& selector, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (selector == "Read") {
    return PerformReadSelector(args, result);
  } else {
    return absl::NotFoundError(absl::StrCat(
        "BufferDescriptor don't have selector with name - ", selector));
  }
}

absl::Status BufferDescriptor::PerformReadSelector(
    const std::vector<std::string>& args, std::string* result) const {
  if (args.size() != 1) {
    return absl::NotFoundError(
        absl::StrCat("BufferDescriptor Read require one argument, but ",
                     args.size(), " was passed"));
  }
  *result = absl::StrCat("buffer[", args[0], "]");
  return absl::OkStatus();
}

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

GPUResourcesWithValue Buffer::GetGPUResources(AccessType access_type) const {
  GPUResourcesWithValue resources;
  resources.buffers.push_back({"buffer", buffer_});
  return resources;
}

absl::Status CreateReadOnlyBuffer(size_t size_in_bytes, CLContext* context,
                                  Buffer* result) {
  return CreateBuffer(size_in_bytes, true, nullptr, context, result);
}

absl::Status CreateReadOnlyBuffer(size_t size_in_bytes, const void* data,
                                  CLContext* context, Buffer* result) {
  return CreateBuffer(size_in_bytes, true, data, context, result);
}

absl::Status CreateReadWriteBuffer(size_t size_in_bytes, CLContext* context,
                                   Buffer* result) {
  return CreateBuffer(size_in_bytes, false, nullptr, context, result);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
