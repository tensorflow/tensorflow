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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_REDUCE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_REDUCE_H_

#include <map>
#include <set>

#include "tensorflow/lite/delegates/gpu/common/kernel_info.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

class Reduce : public GPUOperation {
 public:
  Reduce() = default;
  Reduce(const std::map<Axis, int>& axis_to_reduce, OperationType op_type,
         const OperationDef& definition, const GpuInfo& gpu_info);

  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override;
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  Reduce(Reduce&& operation);
  Reduce& operator=(Reduce&& operation);
  Reduce(const Reduce&) = delete;
  Reduce& operator=(const Reduce&) = delete;

 private:
  std::string GetReduceKernelCode(const OperationDef& op_def,
                                  const int3& work_group_size,
                                  const std::vector<Axis>& axis_to_reduce,
                                  OperationType op_type);

  bool use_wg_reduction_;
};

Reduce CreateReduce(const std::set<Axis>& axis_to_reduce, const BHWC& src_shape,
                    OperationType op_type, const OperationDef& definition,
                    const GpuInfo& gpu_info);

Reduce CreateReduce(const std::set<Axis>& axis_to_reduce,
                    const BHWDC& src_shape, OperationType op_type,
                    const OperationDef& definition, const GpuInfo& gpu_info);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_REDUCE_H_
