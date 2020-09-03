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
#include "tensorflow/lite/delegates/gpu/cl/gpu_object.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
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

  // optional
  int size = 0;
  std::vector<uint8_t> data;

  TensorLinearDescriptor() = default;
  TensorLinearDescriptor(const TensorLinearDescriptor&) = default;
  TensorLinearDescriptor& operator=(const TensorLinearDescriptor&) = default;
  TensorLinearDescriptor(TensorLinearDescriptor&& desc);
  TensorLinearDescriptor& operator=(TensorLinearDescriptor&& desc);

  void UploadLinearData(
      const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src,
      int aligned_size = 0);

  absl::Status PerformSelector(const std::string& selector,
                               const std::vector<std::string>& args,
                               const std::vector<std::string>& template_args,
                               std::string* result) const override;

  GPUResources GetGPUResources() const override;
  absl::Status PerformReadSelector(const std::vector<std::string>& args,
                                   std::string* result) const;

  absl::Status CreateGPUObject(CLContext* context,
                               GPUObjectPtr* result) const override;
  void Release() override;
};

LinearStorageType DeduceLinearStorageType(
    TensorStorageType tensor_storage_type);

// Represent GPU 1D-array of FLT4(float4/half4) values
// Can use inside texture2d or buffer
class LinearStorage : public GPUObject {
 public:
  LinearStorage() {}
  ~LinearStorage() override { Release(); }

  // Move only
  LinearStorage(LinearStorage&& storage);
  LinearStorage& operator=(LinearStorage&& storage);
  LinearStorage(const LinearStorage&) = delete;
  LinearStorage& operator=(const LinearStorage&) = delete;

  absl::Status GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                               GPUResourcesWithValue* resources) const override;

  absl::Status CreateFromTensorLinearDescriptor(
      const TensorLinearDescriptor& desc, CLContext* context);

 private:
  void Release();

  cl_mem memory_ = nullptr;
  int depth_;
  LinearStorageType storage_type_;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_LINEAR_STORAGE_H_
