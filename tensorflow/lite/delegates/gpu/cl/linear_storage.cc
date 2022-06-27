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

#include <utility>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_image_format.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

void LinearStorage::Release() {
  if (memory_) {
    clReleaseMemObject(memory_);
    memory_ = nullptr;
  }
}

LinearStorage::LinearStorage(LinearStorage&& storage)
    : GPUObject(std::move(storage)),
      memory_(storage.memory_),
      depth_(storage.depth_),
      storage_type_(storage.storage_type_) {
  storage.memory_ = nullptr;
}

LinearStorage& LinearStorage::operator=(LinearStorage&& storage) {
  if (this != &storage) {
    Release();
    std::swap(memory_, storage.memory_);
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
    resources->buffers.push_back({"buffer", memory_});
  } else {
    resources->images2d.push_back({"tex2d", memory_});
  }

  return absl::OkStatus();
}

absl::Status LinearStorage::CreateFromTensorLinearDescriptor(
    const TensorLinearDescriptor& desc, CLContext* context) {
  storage_type_ = desc.storage_type;
  depth_ = desc.size;
  uint8_t* data_ptr = desc.data.empty()
                          ? nullptr
                          : const_cast<unsigned char*>(desc.data.data());
  if (storage_type_ == LinearStorageType::BUFFER) {
    bool read_only = desc.memory_type == MemoryType::CONSTANT;
    uint8_t* data_ptr = desc.data.empty()
                            ? nullptr
                            : const_cast<unsigned char*>(desc.data.data());
    const int float4_size = desc.element_type == DataType::FLOAT32
                                ? sizeof(float) * 4
                                : sizeof(half) * 4;
    return CreateCLBuffer(context->context(), depth_ * float4_size, read_only,
                          data_ptr, &memory_);
  } else {
    return CreateRGBAImage2D(context->context(), depth_, 1,
                             DataTypeToChannelType(desc.element_type), data_ptr,
                             &memory_);
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
