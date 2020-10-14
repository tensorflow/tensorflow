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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_MEAN_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_MEAN_H_

#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

class Mean : public GPUOperation {
 public:
  Mean() = default;
  Mean(const OperationDef& definition, const DeviceInfo& device_info);

  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const DeviceInfo& device_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override {
    work_groups->push_back(work_group_size_);
  }
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  Mean(Mean&& operation);
  Mean& operator=(Mean&& operation);
  Mean(const Mean&) = delete;
  Mean& operator=(const Mean&) = delete;

 private:
  std::string GetMeanKernelCode(const OperationDef& op_def,
                                const int3& work_group_size);
};

Mean CreateMean(const OperationDef& definition, const DeviceInfo& device_info);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_MEAN_H_
