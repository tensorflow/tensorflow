/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/metal/linear_storage.h"

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/metal/common.h"

namespace tflite {
namespace gpu {
namespace metal {

void LinearStorage::Release() {
  if (buffer_) {
    buffer_ = nullptr;
  }
  if (texture_) {
    texture_ = nullptr;
  }
}

LinearStorage::LinearStorage(LinearStorage&& storage)
    : GPUObject(std::move(storage)),
      buffer_(storage.buffer_),
      texture_(storage.texture_),
      depth_(storage.depth_),
      storage_type_(storage.storage_type_) {
  storage.buffer_ = nullptr;
  storage.texture_ = nullptr;
}

LinearStorage& LinearStorage::operator=(LinearStorage&& storage) {
  if (this != &storage) {
    Release();
    std::swap(buffer_, storage.buffer_);
    std::swap(texture_, storage.texture_);
    std::swap(depth_, storage.depth_);
    std::swap(storage_type_, storage.storage_type_);
    GPUObject::operator=(std::move(storage));
  }
  return *this;
}

absl::Status LinearStorage::GetGPUResources(
    const GPUObjectDescriptor* obj_ptr,
    GPUResourcesWithValue* resources) const {
  const auto* linear_desc =
      dynamic_cast<const TensorLinearDescriptor*>(obj_ptr);
  if (!linear_desc) {
    return absl::InvalidArgumentError(
        "Expected TensorLinearDescriptor on input.");
  }

  resources->ints.push_back({"length", depth_});

  if (storage_type_ == LinearStorageType::BUFFER) {
    resources->buffers.push_back({"buffer", buffer_});
  } else {
    resources->images2d.push_back({"tex2d", texture_});
  }

  return absl::OkStatus();
}

absl::Status LinearStorage::CreateFromTensorLinearDescriptor(
    const TensorLinearDescriptor& desc, id<MTLDevice> device) {
  storage_type_ = desc.storage_type;
  depth_ = desc.size;
  uint8_t* data_ptr = desc.data.empty()
                          ? nullptr
                          : const_cast<unsigned char*>(desc.data.data());
  const int float4_size = desc.element_type == DataType::FLOAT32
                              ? sizeof(float) * 4
                              : sizeof(half) * 4;
  if (storage_type_ == LinearStorageType::BUFFER) {
    bool read_only = desc.memory_type == MemoryType::CONSTANT;
    uint8_t* data_ptr = desc.data.empty()
                            ? nullptr
                            : const_cast<unsigned char*>(desc.data.data());
    buffer_ = [device newBufferWithBytes:data_ptr
                                  length:depth_ * float4_size
                                 options:MTLResourceStorageModeShared];
    if (!buffer_) {
      return absl::UnknownError("Failed to allocate id<MTLBuffer>");
    }

    return absl::OkStatus();
  } else {
    MTLPixelFormat pixel_format = desc.element_type == DataType::FLOAT32
                                      ? MTLPixelFormatRGBA32Float
                                      : MTLPixelFormatRGBA16Float;
    MTLTextureDescriptor* texture_desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format
                                                           width:depth_
                                                          height:1
                                                       mipmapped:NO];
    texture_desc.textureType = MTLTextureType2D;
    texture_desc.usage = MTLTextureUsageShaderRead;
    texture_desc.storageMode = MTLStorageModePrivate;

    texture_ = [device newTextureWithDescriptor:texture_desc];
    if (!texture_) {
      return absl::UnknownError("Failed to allocate id<MTLTexture>");
    }

    WriteDataToTexture2D(texture_, device, data_ptr);

    return absl::OkStatus();
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
