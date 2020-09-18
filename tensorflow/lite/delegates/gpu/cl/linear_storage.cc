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

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

TensorLinearDescriptor::TensorLinearDescriptor(TensorLinearDescriptor&& desc)
    : GPUObjectDescriptor(std::move(desc)),
      storage_type(desc.storage_type),
      element_type(desc.element_type),
      memory_type(desc.memory_type),
      size(desc.size),
      data(std::move(desc.data)) {}

TensorLinearDescriptor& TensorLinearDescriptor::operator=(
    TensorLinearDescriptor&& desc) {
  if (this != &desc) {
    std::swap(storage_type, desc.storage_type);
    std::swap(element_type, desc.element_type);
    std::swap(memory_type, desc.memory_type);
    std::swap(size, desc.size);
    data = std::move(desc.data);
    GPUObjectDescriptor::operator=(std::move(desc));
  }
  return *this;
}

void TensorLinearDescriptor::Release() { data.clear(); }

GPUResources TensorLinearDescriptor::GetGPUResources() const {
  GPUResources resources;
  resources.ints.push_back("length");
  if (storage_type == LinearStorageType::BUFFER) {
    GPUBufferDescriptor desc;
    desc.data_type = element_type;
    desc.access_type = access_type_;
    desc.element_size = 4;
    desc.memory_type = memory_type;
    resources.buffers.push_back({"buffer", desc});
  } else {
    GPUImage2DDescriptor desc;
    desc.data_type = element_type;
    desc.access_type = access_type_;
    resources.images2d.push_back({"tex2d", desc});
  }
  return resources;
}

absl::Status TensorLinearDescriptor::PerformSelector(
    const std::string& selector, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (selector == "Length") {
    *result = "length";
    return absl::OkStatus();
  } else if (selector == "Read") {
    return PerformReadSelector(args, result);
  } else if (selector == "GetPtr") {
    if (storage_type != LinearStorageType::BUFFER) {
      return absl::InvalidArgumentError(
          "GetPtr selector supported for LinearStorageType::BUFFER only.");
    }
    *result = "buffer";
    return absl::OkStatus();
  } else {
    return absl::NotFoundError(absl::StrCat(
        "TensorLinearDescriptor don't have selector with name - ", selector));
  }
}

absl::Status TensorLinearDescriptor::PerformReadSelector(
    const std::vector<std::string>& args, std::string* result) const {
  if (args.size() != 1) {
    return absl::NotFoundError(
        absl::StrCat("TensorLinearDescriptor Read require one argument, but ",
                     args.size(), " was passed"));
  }
  if (storage_type == LinearStorageType::BUFFER) {
    *result = absl::StrCat("buffer[", args[0], "]");
    return absl::OkStatus();
  } else {
    const std::string read =
        element_type == DataType::FLOAT16 ? "read_imageh" : "read_imagef";
    *result = absl::StrCat(read, "(tex2d, smp_none, (int2)(", args[0], ", 0))");
    return absl::OkStatus();
  }
}

absl::Status TensorLinearDescriptor::CreateGPUObject(
    CLContext* context, GPUObjectPtr* result) const {
  LinearStorage gpu_storage;
  RETURN_IF_ERROR(gpu_storage.CreateFromTensorLinearDescriptor(*this, context));
  *result = absl::make_unique<LinearStorage>(std::move(gpu_storage));
  return absl::OkStatus();
}

void TensorLinearDescriptor::UploadLinearData(
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src,
    int aligned_size) {
  size = aligned_size == 0 ? DivideRoundUp(src.shape.v, 4) : aligned_size;
  if (element_type == DataType::FLOAT32) {
    data.resize(size * sizeof(float) * 4);
    float* gpu_data = reinterpret_cast<float*>(data.data());
    for (int i = 0; i < size * 4; ++i) {
      if (i < src.shape.v) {
        gpu_data[i] = src.data[i];
      } else {
        gpu_data[i] = 0.0f;
      }
    }
  } else {
    data.resize(size * sizeof(half) * 4);
    half* gpu_data = reinterpret_cast<half*>(data.data());
    for (int i = 0; i < size * 4; ++i) {
      if (i < src.shape.v) {
        gpu_data[i] = src.data[i];
      } else {
        gpu_data[i] = 0.0f;
      }
    }
  }
}

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
    return CreateFloatRGBAImage2D(context->context(), depth_, 1,
                                  desc.element_type, data_ptr, &memory_);
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

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
