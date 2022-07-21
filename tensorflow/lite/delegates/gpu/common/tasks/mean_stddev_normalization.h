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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_LSTM_NORMALIZATION_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_LSTM_NORMALIZATION_H_

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

// Implements tensor_utils::MeanStddevNormalization
class MeanStdDevNormalization : public GPUOperation {
 public:
  explicit MeanStdDevNormalization(const OperationDef& definition,
                                   const GpuInfo& gpu_info, const BHWC& shape,
                                   float variance_bias, bool two_step);

  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override {
    work_groups->push_back(work_group_size_);
  }
  int3 GetGridSize() const override;

  // Move only
  MeanStdDevNormalization(MeanStdDevNormalization&& kernel) = default;
  MeanStdDevNormalization& operator=(MeanStdDevNormalization&& kernel) =
      default;
  MeanStdDevNormalization(const MeanStdDevNormalization&) = delete;
  MeanStdDevNormalization& operator=(const MeanStdDevNormalization&) = delete;

 private:
  std::string GetNormalizationCode(const GpuInfo& gpu_info, bool channels_x4,
                                   bool two_step);
};

// std dev can be calculated in single step, but two step algorithm can
// provide more stable and robust results
MeanStdDevNormalization CreateMeanStdDevNormalization(
    const OperationDef& definition, const GpuInfo& gpu_info, const BHWC& shape,
    float variance_bias = 1.0e-8f, bool two_step = true);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_LSTM_NORMALIZATION_H_
