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

#include <string>

#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

absl::Status CreateBuffer(size_t size_in_bytes, bool gpu_read_only,
                          const void* data, CLContext* context,
                          Buffer* result) {
  cl_mem buffer;
  RETURN_IF_ERROR(CreateCLBuffer(context->context(), size_in_bytes,
                                 gpu_read_only, const_cast<void*>(data),
                                 &buffer));
  *result = Buffer(buffer, size_in_bytes);

  return absl::OkStatus();
}

absl::Status CreateSubBuffer(const Buffer& parent, size_t origin_in_bytes,
                             size_t size_in_bytes, bool gpu_read_only,
                             CLContext* context, Buffer* result) {
  cl_mem buffer;
  if (parent.IsSubBuffer()) {
    return absl::InvalidArgumentError(
        "Cannot create a sub-buffer from a sub-buffer!");
  }
  RETURN_IF_ERROR(CreateCLSubBuffer(context->context(), parent.GetMemoryPtr(),
                                    origin_in_bytes, size_in_bytes,
                                    gpu_read_only, &buffer));
  *result = Buffer(buffer, size_in_bytes, /*is_sub_buffer=*/true);

  return absl::OkStatus();
}
}  // namespace

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

absl::Status Buffer::GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                                     GPUResourcesWithValue* resources) const {
  const auto* buffer_desc = dynamic_cast<const BufferDescriptor*>(obj_ptr);
  if (!buffer_desc) {
    return absl::InvalidArgumentError("Expected BufferDescriptor on input.");
  }

  resources->buffers.push_back({"buffer", buffer_});
  return absl::OkStatus();
}

absl::Status Buffer::CreateFromBufferDescriptor(const BufferDescriptor& desc,
                                                CLContext* context) {
  bool read_only = desc.memory_type == MemoryType::CONSTANT;
  uint8_t* data_ptr = desc.data.empty()
                          ? nullptr
                          : const_cast<unsigned char*>(desc.data.data());
  size_ = desc.size;
  return CreateCLBuffer(context->context(), desc.size, read_only, data_ptr,
                        &buffer_);
}

Buffer CreateBufferShared(cl_mem buffer) { return Buffer(buffer); }

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

absl::Status CreateReadWriteSubBuffer(const Buffer& parent,
                                      size_t origin_in_bytes,
                                      size_t size_in_bytes, CLContext* context,
                                      Buffer* result) {
  return CreateSubBuffer(parent, origin_in_bytes, size_in_bytes,
                         /*gpu_read_only=*/false, context, result);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
