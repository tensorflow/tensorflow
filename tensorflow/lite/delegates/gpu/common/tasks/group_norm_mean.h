/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_GROUP_NORM_MEAN_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_GROUP_NORM_MEAN_H_

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

class GroupNormMean : public GPUOperation {
 public:
  GroupNormMean() = default;
  GroupNormMean(const BHWC& shape, 
                const OperationDef& definition,
                const GpuInfo& gpu_info);

  void GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info,
    std::vector<int3>* work_groups) const override;

  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  GroupNormMean(GroupNormMean&& operation);
  GroupNormMean& operator=(GroupNormMean&& operation);
  GroupNormMean(const GroupNormMean&) = delete;
  GroupNormMean& operator=(const GroupNormMean&) = delete;

 private:

  std::string GetGroupNormMeanCode(const OperationDef& op_def, 
                                   const GpuInfo& gpu_info,
                                   const int3& work_group_size,
                                   std::vector<int>& axis_to_reduce);

  bool use_wg_reduction_;
};


GroupNormMean CreateGroupNormMean(const BHWC& shape,
                                 const GpuInfo& gpu_info,
                                 const OperationDef& op_def);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_GROUP_NORM_MEAN_H_