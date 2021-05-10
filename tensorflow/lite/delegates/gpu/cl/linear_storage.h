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
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/gpu_object.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_linear_desc.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

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
