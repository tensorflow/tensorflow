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

#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"

namespace tflite {
namespace gpu {
namespace cl {

LinearStorage::LinearStorage(int depth, LinearStorageType storage_type,
                             DataType data_type)
    : depth_(depth), storage_type_(storage_type), data_type_(data_type) {}

LinearStorage::LinearStorage(LinearStorage&& storage)
    : texture_storage_(std::move(storage.texture_storage_)),
      buffer_storage_(std::move(storage.buffer_storage_)),
      memory_(storage.memory_),
      depth_(storage.depth_),
      name_(std::move(storage.name_)),
      storage_type_(storage.storage_type_),
      data_type_(storage.data_type_) {
  storage.memory_ = nullptr;
}

LinearStorage& LinearStorage::operator=(LinearStorage&& storage) {
  if (this != &storage) {
    texture_storage_ = std::move(storage.texture_storage_);
    buffer_storage_ = std::move(storage.buffer_storage_);
    std::swap(memory_, storage.memory_);
    std::swap(depth_, storage.depth_);
    name_ = std::move(storage.name_);
    std::swap(storage_type_, storage.storage_type_);
    std::swap(data_type_, storage.data_type_);
  }
  return *this;
}

std::string LinearStorage::ReadLinearFLT4(const std::string& z_coord) const {
  if (storage_type_ == LinearStorageType::BUFFER) {
    return absl::StrCat(name_, "[", z_coord, "]");
  } else {
    return absl::StrCat("READ_IMAGE(", name_, ", smp_none, (int2)(", z_coord,
                        ", 0))");
  }
}

std::string LinearStorage::GetDeclaration() const {
  if (storage_type_ == LinearStorageType::BUFFER) {
    return absl::StrCat("__global FLT4* ", name_);
  } else {
    return absl::StrCat("__read_only image2d_t ", name_);
  }
}

LinearStorageType DeduceLinearStorageType(
    TensorStorageType tensor_storage_type) {
  if (tensor_storage_type == TensorStorageType::BUFFER) {
    return LinearStorageType::BUFFER;
  } else {
    return LinearStorageType::TEXTURE_2D;
  }
}

Status CreateBufferLinearStorage(int size, DataType data_type, void* data,
                                 CLContext* context, LinearStorage* result) {
  const int float4_size =
      data_type == DataType::FLOAT32 ? sizeof(float4) : sizeof(half4);
  *result = LinearStorage(size, LinearStorageType::BUFFER, data_type);
  RETURN_IF_ERROR(CreateReadOnlyBuffer(float4_size * size, data, context,
                                       &result->buffer_storage_));
  result->memory_ = result->buffer_storage_.GetMemoryPtr();
  return OkStatus();
}

Status CreateTextureLinearStorage(int size, DataType data_type, void* data,
                                  CLContext* context, LinearStorage* result) {
  *result = LinearStorage(size, LinearStorageType::TEXTURE_2D, data_type);
  RETURN_IF_ERROR(CreateTexture2DRGBA(data_type, size, 1, data, context,
                                      &result->texture_storage_));
  result->memory_ = result->texture_storage_.GetMemoryPtr();
  return OkStatus();
}

Status CreateLinearStorage(const LinearStorageCreateInfo& creation_info,
                           int size, void* data, CLContext* context,
                           LinearStorage* result) {
  if (creation_info.storage_type == LinearStorageType::BUFFER) {
    return CreateBufferLinearStorage(size, creation_info.data_type, data,
                                     context, result);
  } else {
    return CreateTextureLinearStorage(size, creation_info.data_type, data,
                                      context, result);
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
