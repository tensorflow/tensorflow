/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_TENSOR_LINEAR_DESC_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_TENSOR_LINEAR_DESC_H_

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {

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
  TensorLinearDescriptor(TensorLinearDescriptor&& desc) = default;
  TensorLinearDescriptor& operator=(TensorLinearDescriptor&& desc) = default;

  void UploadLinearData(
      const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src,
      int aligned_size = 0);

  absl::Status PerformSelector(const GpuInfo& gpu_info,
                               const std::string& selector,
                               const std::vector<std::string>& args,
                               const std::vector<std::string>& template_args,
                               std::string* result) const override;

  GPUResources GetGPUResources(const GpuInfo& gpu_info) const override;
  absl::Status PerformReadSelector(const GpuInfo& gpu_info,
                                   const std::vector<std::string>& args,
                                   std::string* result) const;

  void Release() override;
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_TENSOR_LINEAR_DESC_H_
