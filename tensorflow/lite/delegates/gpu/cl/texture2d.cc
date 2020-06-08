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
absl::Status CreateTexture2D(int width, int height, cl_channel_type type,
                             void* data, CLContext* context,
                             Texture2D* result) {
  cl_image_desc desc;
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = width;
  desc.image_height = height;
  desc.image_depth = 0;
  desc.image_row_pitch = 0;
  desc.image_slice_pitch = 0;
  desc.num_mip_levels = 0;
  desc.num_samples = 0;
  desc.buffer = nullptr;

  cl_image_format format;
  format.image_channel_order = CL_RGBA;
  format.image_channel_data_type = type;

  cl_mem_flags flags = CL_MEM_READ_WRITE;
  if (data != nullptr) {
    flags |= CL_MEM_COPY_HOST_PTR;
  }

  cl_int error_code;
  cl_mem texture = CreateImage2DLegacy(context->context(), flags, &format,
                                       &desc, data, &error_code);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to create Texture2D (clCreateImage)",
                     CLErrorCodeToString(error_code)));
  }

  *result = Texture2D(texture, width, height, type);

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

Texture2D::~Texture2D() { Release(); }

void Texture2D::Release() {
  if (texture_) {
    clReleaseMemObject(texture_);
    texture_ = nullptr;
    width_ = 0;
    height_ = 0;
  }
}

// Creates new 4-channel 2D texture with f32 elements
absl::Status CreateTexture2DRGBA32F(int width, int height, CLContext* context,
                                    Texture2D* result) {
  return CreateTexture2D(width, height, CL_FLOAT, nullptr, context, result);
}

// Creates new 4-channel 2D texture with f16 elements
absl::Status CreateTexture2DRGBA16F(int width, int height, CLContext* context,
                                    Texture2D* result) {
  return CreateTexture2D(width, height, CL_HALF_FLOAT, nullptr, context,
                         result);
}

absl::Status CreateTexture2DRGBA(DataType type, int width, int height,
                                 CLContext* context, Texture2D* result) {
  if (type == DataType::FLOAT32) {
    return CreateTexture2D(width, height, CL_FLOAT, nullptr, context, result);
  } else {
    return CreateTexture2D(width, height, CL_HALF_FLOAT, nullptr, context,
                           result);
  }
}

absl::Status CreateTexture2DRGBA(DataType type, int width, int height,
                                 void* data, CLContext* context,
                                 Texture2D* result) {
  if (type == DataType::FLOAT32) {
    return CreateTexture2D(width, height, CL_FLOAT, data, context, result);
  } else {
    return CreateTexture2D(width, height, CL_HALF_FLOAT, data, context, result);
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
