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
  MemoryType memory_type = MemoryType::GLOBAL;  // applicable for BUFFER

  absl::Status PerformSelector(const std::string& selector,
                               const std::vector<std::string>& args,
                               const std::vector<std::string>& template_args,
                               std::string* result) const override;

  GPUResources GetGPUResources() const override;
  absl::Status PerformReadSelector(const std::vector<std::string>& args,
                                   std::string* result) const;
};

LinearStorageType DeduceLinearStorageType(
    TensorStorageType tensor_storage_type);

// Represent GPU 1D-array of FLT4(float4/half4) values
// Can use inside texture2d or buffer
class LinearStorage : public GPUObject {
 public:
  LinearStorage() {}

  virtual ~LinearStorage() {}

  // Move only
  LinearStorage(LinearStorage&& storage);
  LinearStorage& operator=(LinearStorage&& storage);
  LinearStorage(const LinearStorage&) = delete;
  LinearStorage& operator=(const LinearStorage&) = delete;

  absl::Status GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                               GPUResourcesWithValue* resources) const override;

 private:
  friend absl::Status CreateLinearStorage(LinearStorageType storage_type,
                                          DataType data_type, int size,
                                          void* data, CLContext* context,
                                          LinearStorage* result);

  LinearStorage(int depth, LinearStorageType storage_type);

  Texture2D texture_storage_;
  Buffer buffer_storage_;

  int depth_;
  LinearStorageType storage_type_;
};

absl::Status CreateLinearStorage(LinearStorageType storage_type,
                                 DataType data_type, int size, void* data,
                                 CLContext* context, LinearStorage* result);

template <DataType T>
absl::Status CreateLinearStorage(const TensorLinearDescriptor& descriptor,
                                 const tflite::gpu::Tensor<Linear, T>& tensor,
                                 CLContext* context, LinearStorage* result) {
  const int depth = DivideRoundUp(tensor.shape.v, 4);
  if (descriptor.element_type == DataType::FLOAT32) {
    std::vector<float4> gpu_data(depth);
    CopyLinearFLT4(tensor, absl::MakeSpan(gpu_data));
    RETURN_IF_ERROR(CreateLinearStorage(descriptor.storage_type,
                                        descriptor.element_type, depth,
                                        gpu_data.data(), context, result));
  } else {
    std::vector<half4> gpu_data(depth);
    CopyLinearFLT4(tensor, absl::MakeSpan(gpu_data));
    RETURN_IF_ERROR(CreateLinearStorage(descriptor.storage_type,
                                        descriptor.element_type, depth,
                                        gpu_data.data(), context, result));
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_LINEAR_STORAGE_H_
