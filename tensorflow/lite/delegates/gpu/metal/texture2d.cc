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

#include "tensorflow/lite/delegates/gpu/metal/texture2d.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

// Creates new 4-channel 2D texture with cl_channel_type elements
absl::Status CreateTexture2D(int width, int height, DataType type, void* data,
                             id<MTLDevice> device, Texture2D* result) {
  MTLPixelFormat pixel_format = DataTypeToRGBAPixelFormat(type);

  MTLTextureDescriptor* texture_desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format
                                                         width:width
                                                        height:height
                                                     mipmapped:NO];
  texture_desc.textureType = MTLTextureType2D;
  texture_desc.usage = MTLTextureUsageShaderRead;
  texture_desc.storageMode = MTLStorageModePrivate;

  id<MTLTexture> texture = [device newTextureWithDescriptor:texture_desc];
  if (!texture) {
    return absl::UnknownError("Failed to allocate id<MTLTexture>");
  }

  if (data) {
    WriteDataToTexture2D(texture, device, data);
  }

  *result = Texture2D(texture, width, height, pixel_format);

  return absl::OkStatus();
}
}  // namespace

Texture2D::Texture2D(id<MTLTexture> texture, int width, int height,
                     MTLPixelFormat pixel_format)
    : texture_(texture),
      width_(width),
      height_(height),
      pixel_format_(pixel_format) {}

Texture2D::Texture2D(Texture2D&& texture)
    : texture_(texture.texture_),
      width_(texture.width_),
      height_(texture.height_),
      pixel_format_(texture.pixel_format_) {
  texture.texture_ = nullptr;
  texture.width_ = 0;
  texture.height_ = 0;
}

Texture2D& Texture2D::operator=(Texture2D&& texture) {
  if (this != &texture) {
    Release();
    std::swap(pixel_format_, texture.pixel_format_);
    std::swap(width_, texture.width_);
    std::swap(height_, texture.height_);
    std::swap(texture_, texture.texture_);
  }
  return *this;
}

void Texture2D::Release() {
  if (texture_) {
    texture_ = nullptr;
    width_ = 0;
    height_ = 0;
  }
}

absl::Status Texture2D::GetGPUResources(
    const GPUObjectDescriptor* obj_ptr,
    GPUResourcesWithValue* resources) const {
  const auto* texture_desc = dynamic_cast<const Texture2DDescriptor*>(obj_ptr);
  if (!texture_desc) {
    return absl::InvalidArgumentError("Expected Texture2DDescriptor on input.");
  }

  resources->images2d.push_back({"tex2d", texture_});
  return absl::OkStatus();
}

absl::Status Texture2D::CreateFromTexture2DDescriptor(
    const Texture2DDescriptor& desc, id<MTLDevice> device) {
  width_ = desc.size.x;
  height_ = desc.size.y;
  pixel_format_ = DataTypeToRGBAPixelFormat(desc.element_type, desc.normalized);
  uint8_t* data_ptr = desc.data.empty()
                          ? nullptr
                          : const_cast<unsigned char*>(desc.data.data());

  MTLTextureDescriptor* texture_desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format_
                                                         width:width_
                                                        height:height_
                                                     mipmapped:NO];
  texture_desc.textureType = MTLTextureType2D;
  texture_desc.usage = MTLTextureUsageShaderRead;
  texture_desc.storageMode = MTLStorageModePrivate;

  texture_ = [device newTextureWithDescriptor:texture_desc];
  if (!texture_) {
    return absl::UnknownError("Failed to allocate id<MTLTexture>");
  }

  if (data_ptr) {
    WriteDataToTexture2D(texture_, device, data_ptr);
  }

  return absl::OkStatus();
}

// Creates new 4-channel 2D texture with f32 elements
absl::Status CreateTexture2DRGBA32F(int width, int height, id<MTLDevice> device,
                                    Texture2D* result) {
  return CreateTexture2D(width, height, DataType::FLOAT32, nullptr, device,
                         result);
}

// Creates new 4-channel 2D texture with f16 elements
absl::Status CreateTexture2DRGBA16F(int width, int height, id<MTLDevice> device,
                                    Texture2D* result) {
  return CreateTexture2D(width, height, DataType::FLOAT16, nullptr, device,
                         result);
}

absl::Status CreateTexture2DRGBA(DataType type, int width, int height,
                                 id<MTLDevice> device, Texture2D* result) {
  return CreateTexture2D(width, height, type, nullptr, device, result);
}

absl::Status CreateTexture2DRGBA(DataType type, int width, int height,
                                 void* data, id<MTLDevice> device,
                                 Texture2D* result) {
  return CreateTexture2D(width, height, type, data, device, result);
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
