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

#include "tensorflow/lite/delegates/gpu/metal/buffer.h"

#include <utility>

namespace tflite {
namespace gpu {
namespace metal {

Buffer::Buffer(id<MTLBuffer> buffer, size_t size_in_bytes)
    : buffer_(buffer), size_(size_in_bytes) {}

Buffer::Buffer(id<MTLBuffer> buffer)
    : buffer_(buffer), size_(0), owner_(false) {}

Buffer::Buffer(Buffer&& buffer)
    : buffer_(buffer.buffer_), size_(buffer.size_), owner_(buffer.owner_) {
  buffer.buffer_ = nullptr;
  buffer.size_ = 0;
}

Buffer& Buffer::operator=(Buffer&& buffer) {
  if (this != &buffer) {
    Release();
    std::swap(size_, buffer.size_);
    std::swap(buffer_, buffer.buffer_);
    std::swap(owner_, buffer.owner_);
  }
  return *this;
}

Buffer::~Buffer() { Release(); }

void Buffer::Release() {
  if (owner_ && buffer_) {
    buffer_ = nullptr;
    size_ = 0;
    owner_ = false;
  }
}

absl::Status Buffer::GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                                     GPUResourcesWithValue* resources) const {
  const auto* buffer_desc = dynamic_cast<const BufferDescriptor*>(obj_ptr);
  if (!buffer_desc) {
    return absl::InvalidArgumentError("Expected BufferDescriptor on input.");
  }

  resources->buffers.push_back({"buffer", {buffer_, 0}});
  return absl::OkStatus();
}

absl::Status Buffer::CreateFromBufferDescriptor(const BufferDescriptor& desc,
                                                id<MTLDevice> device) {
  size_ = desc.size;
  if (desc.data.empty()) {
    buffer_ =
        [device newBufferWithLength:size_ options:MTLResourceStorageModeShared];
  } else {
    buffer_ = [device newBufferWithBytes:desc.data.data()
                                  length:size_
                                 options:MTLResourceStorageModeShared];
  }
  return absl::OkStatus();
}

Buffer CreateBufferShared(id<MTLBuffer> buffer) { return Buffer(buffer); }

absl::Status CreateBuffer(size_t size_in_bytes, const void* data,
                          id<MTLDevice> device, Buffer* result) {
  id<MTLBuffer> buffer;
  if (data) {
    buffer = [device newBufferWithBytes:data
                                 length:size_in_bytes
                                options:MTLResourceStorageModeShared];
  } else {
    buffer = [device newBufferWithLength:size_in_bytes
                                 options:MTLResourceStorageModeShared];
  }

  *result = Buffer(buffer, size_in_bytes);

  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
