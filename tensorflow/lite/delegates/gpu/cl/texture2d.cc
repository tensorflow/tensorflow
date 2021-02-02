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

#include "tensorflow/lite/delegates/gpu/cl/texture2d.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

// Creates new 4-channel 2D texture with cl_channel_type elements
absl::Status CreateTexture2D(int width, int height, DataType type, void* data,
                             CLContext* context, Texture2D* result) {
  cl_mem texture;
  cl_channel_type channel_type = DataTypeToChannelType(type);
  RETURN_IF_ERROR(CreateRGBAImage2D(context->context(), width, height,
                                    channel_type, data, &texture));
  *result = Texture2D(texture, width, height, channel_type);

  return absl::OkStatus();
}
}  // namespace

Texture2D::Texture2D(cl_mem texture, int width, int height,
                     cl_channel_type type)
    : texture_(texture), width_(width), height_(height), channel_type_(type) {}

Texture2D::Texture2D(Texture2D&& texture)
    : texture_(texture.texture_),
      width_(texture.width_),
      height_(texture.height_),
      channel_type_(texture.channel_type_) {
  texture.texture_ = nullptr;
  texture.width_ = 0;
  texture.height_ = 0;
}

Texture2D& Texture2D::operator=(Texture2D&& texture) {
  if (this != &texture) {
    Release();
    std::swap(channel_type_, texture.channel_type_);
    std::swap(width_, texture.width_);
    std::swap(height_, texture.height_);
    std::swap(texture_, texture.texture_);
  }
  return *this;
}

void Texture2D::Release() {
  if (texture_) {
    clReleaseMemObject(texture_);
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
    const Texture2DDescriptor& desc, CLContext* context) {
  width_ = desc.size.x;
  height_ = desc.size.y;
  channel_type_ = DataTypeToChannelType(desc.element_type, desc.normalized);
  uint8_t* data_ptr = desc.data.empty()
                          ? nullptr
                          : const_cast<unsigned char*>(desc.data.data());
  return CreateRGBAImage2D(context->context(), desc.size.x, desc.size.y,
                           channel_type_, data_ptr, &texture_);
}

// Creates new 4-channel 2D texture with f32 elements
absl::Status CreateTexture2DRGBA32F(int width, int height, CLContext* context,
                                    Texture2D* result) {
  return CreateTexture2D(width, height, DataType::FLOAT32, nullptr, context,
                         result);
}

// Creates new 4-channel 2D texture with f16 elements
absl::Status CreateTexture2DRGBA16F(int width, int height, CLContext* context,
                                    Texture2D* result) {
  return CreateTexture2D(width, height, DataType::FLOAT16, nullptr, context,
                         result);
}

absl::Status CreateTexture2DRGBA(DataType type, int width, int height,
                                 CLContext* context, Texture2D* result) {
  return CreateTexture2D(width, height, type, nullptr, context, result);
}

absl::Status CreateTexture2DRGBA(DataType type, int width, int height,
                                 void* data, CLContext* context,
                                 Texture2D* result) {
  return CreateTexture2D(width, height, type, data, context, result);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
