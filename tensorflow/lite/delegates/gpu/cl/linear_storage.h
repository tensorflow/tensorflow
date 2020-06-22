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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_LINEAR_STORAGE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_LINEAR_STORAGE_H_

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/gpu_object.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/cl/texture2d.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

enum class LinearStorageType { BUFFER, TEXTURE_2D };

struct TensorLinearDescriptor : public GPUObjectDescriptor {
  LinearStorageType storage_type;
  DataType element_type;  // FLOAT32 or FLOAT16

  absl::Status PerformSelector(const std::string& selector,
                               const std::vector<std::string>& args,
                               const std::vector<std::string>& template_args,
                               std::string* result) const override;

  GPUResources GetGPUResources(AccessType access_type) const override;
  absl::Status PerformReadSelector(const std::vector<std::string>& args,
                                   std::string* result) const;
};

struct LinearStorageCreateInfo {
  LinearStorageType storage_type;
  DataType data_type;
  std::string name;      // optional
  int aligned_size = 0;  // optional, to pad with zeroes
};

LinearStorageType DeduceLinearStorageType(
    TensorStorageType tensor_storage_type);

// Represent GPU 1D-array of FLT4(float4/half4) values
// Can use inside texture2d or buffer
class LinearStorage : public GPUObject {
 public:
  LinearStorage() {}

  // Move only
  LinearStorage(LinearStorage&& storage);
  LinearStorage& operator=(LinearStorage&& storage);
  LinearStorage(const LinearStorage&) = delete;
  LinearStorage& operator=(const LinearStorage&) = delete;

  void SetName(const std::string& name) { name_ = name; }
  cl_mem GetMemoryPtr() const { return memory_; }
  std::string ReadLinearFLT4(const std::string& z_coord) const;
  std::string GetDeclaration() const;

  GPUResourcesWithValue GetGPUResources(AccessType access_type) const override;

 private:
  friend absl::Status CreateTextureLinearStorage(int size, DataType data_type,
                                                 void* data, CLContext* context,
                                                 LinearStorage* result);
  friend absl::Status CreateBufferLinearStorage(int size, DataType data_type,
                                                void* data, CLContext* context,
                                                LinearStorage* result);

  LinearStorage(int depth, LinearStorageType storage_type, DataType data_type);

  Texture2D texture_storage_;
  Buffer buffer_storage_;
  cl_mem memory_ = nullptr;  // Just a reference to texture_storage_ or
                             // buffer_storage_ memory, not an owner
  int depth_;
  std::string name_;
  LinearStorageType storage_type_;
  DataType data_type_;
};

absl::Status CreateBufferLinearStorage(int size, DataType data_type, void* data,
                                       CLContext* context,
                                       LinearStorage* result);

absl::Status CreateTextureLinearStorage(int size, DataType data_type,
                                        void* data, CLContext* context,
                                        LinearStorage* result);

absl::Status CreateLinearStorage(const LinearStorageCreateInfo& creation_info,
                                 int size, void* data, CLContext* context,
                                 LinearStorage* result);

template <DataType T>
absl::Status CreateLinearStorage(const LinearStorageCreateInfo& creation_info,
                                 const tflite::gpu::Tensor<Linear, T>& tensor,
                                 CLContext* context, LinearStorage* result) {
  int size = creation_info.aligned_size != 0 ? creation_info.aligned_size
                                             : tensor.shape.v;
  const int depth = DivideRoundUp(size, 4);
  if (creation_info.data_type == DataType::FLOAT32) {
    std::vector<float4> gpu_data(depth);
    CopyLinearFLT4(tensor, absl::MakeSpan(gpu_data));
    RETURN_IF_ERROR(CreateLinearStorage(creation_info, depth, gpu_data.data(),
                                        context, result));
  } else {
    std::vector<half4> gpu_data(depth);
    CopyLinearFLT4(tensor, absl::MakeSpan(gpu_data));
    RETURN_IF_ERROR(CreateLinearStorage(creation_info, depth, gpu_data.data(),
                                        context, result));
  }
  result->SetName(creation_info.name);
  return absl::OkStatus();
}

template <DataType T>
absl::Status CreateLinearStorage(const TensorLinearDescriptor& descriptor,
                                 const tflite::gpu::Tensor<Linear, T>& tensor,
                                 CLContext* context, LinearStorage* result) {
  LinearStorageCreateInfo creation_info;
  creation_info.storage_type = descriptor.storage_type;
  creation_info.data_type = descriptor.element_type;
  int size = creation_info.aligned_size != 0 ? creation_info.aligned_size
                                             : tensor.shape.v;
  const int depth = DivideRoundUp(size, 4);
  if (creation_info.data_type == DataType::FLOAT32) {
    std::vector<float4> gpu_data(depth);
    CopyLinearFLT4(tensor, absl::MakeSpan(gpu_data));
    RETURN_IF_ERROR(CreateLinearStorage(creation_info, depth, gpu_data.data(),
                                        context, result));
  } else {
    std::vector<half4> gpu_data(depth);
    CopyLinearFLT4(tensor, absl::MakeSpan(gpu_data));
    RETURN_IF_ERROR(CreateLinearStorage(creation_info, depth, gpu_data.data(),
                                        context, result));
  }
  result->SetName(creation_info.name);
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_LINEAR_STORAGE_H_
