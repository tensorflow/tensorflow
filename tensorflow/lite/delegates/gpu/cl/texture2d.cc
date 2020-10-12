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

Texture2DDescriptor::Texture2DDescriptor(Texture2DDescriptor&& desc)
    : GPUObjectDescriptor(std::move(desc)),
      element_type(desc.element_type),
      normalized(desc.normalized),
      normalized_type(desc.normalized_type),
      size(desc.size),
      data(std::move(desc.data)) {}

Texture2DDescriptor& Texture2DDescriptor::operator=(
    Texture2DDescriptor&& desc) {
  if (this != &desc) {
    std::swap(element_type, desc.element_type);
    std::swap(normalized, desc.normalized);
    std::swap(normalized_type, desc.normalized_type);
    std::swap(size, desc.size);
    data = std::move(desc.data);
    GPUObjectDescriptor::operator=(std::move(desc));
  }
  return *this;
}

void Texture2DDescriptor::Release() { data.clear(); }

GPUResources Texture2DDescriptor::GetGPUResources() const {
  GPUResources resources;
  GPUImage2DDescriptor desc;
  desc.data_type = element_type;
  desc.access_type = access_type_;
  resources.images2d.push_back({"tex2d", desc});
  return resources;
}

absl::Status Texture2DDescriptor::PerformSelector(
    const std::string& selector, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (selector == "Read") {
    return PerformReadSelector(args, result);
  } else {
    return absl::NotFoundError(absl::StrCat(
        "Texture2DDescriptor don't have selector with name - ", selector));
  }
}

absl::Status Texture2DDescriptor::PerformReadSelector(
    const std::vector<std::string>& args, std::string* result) const {
  if (args.size() != 2) {
    return absl::NotFoundError(
        absl::StrCat("Texture2DDescriptor Read require two arguments, but ",
                     args.size(), " was passed"));
  }
  std::string read;
  switch (element_type) {
    case DataType::FLOAT32:
      read = "read_imagef";
      break;
    case DataType::FLOAT16:
      read = "read_imageh";
      break;
    case DataType::INT8:
    case DataType::INT16:
    case DataType::INT32:
      if (normalized) {
        read = normalized_type == DataType::FLOAT16 ? "read_imageh"
                                                    : "read_imagef";
      } else {
        read = "read_imagei";
      }
      break;
    case DataType::UINT8:
    case DataType::UINT16:
    case DataType::UINT32:
      if (normalized) {
        read = normalized_type == DataType::FLOAT16 ? "read_imageh"
                                                    : "read_imagef";
      } else {
        read = "read_imageui";
      }
      break;
    default:
      read = "unknown_type";
      break;
  }
  *result = absl::StrCat(read, "(tex2d, smp_none, (int2)(", args[0],
                         ", " + args[1] + "))");
  return absl::OkStatus();
}

absl::Status Texture2DDescriptor::CreateGPUObject(CLContext* context,
                                                  GPUObjectPtr* result) const {
  Texture2D gpu_texture;
  RETURN_IF_ERROR(gpu_texture.CreateFromTexture2DDescriptor(*this, context));
  *result = absl::make_unique<Texture2D>(std::move(gpu_texture));
  return absl::OkStatus();
}

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
