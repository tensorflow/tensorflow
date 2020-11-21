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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_WORK_GROUP_PICKING_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_WORK_GROUP_PICKING_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/kernel_info.h"
#include "tensorflow/lite/delegates/gpu/common/task/tuning_type.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/workgroup_selection.h"

namespace tflite {
namespace gpu {
namespace cl {

// multiplier can be power of two only
void GetPossibleWorkGroupsXYMultipleOf(int multiplier, const GpuInfo& gpu_info,
                                       const KernelInfo& kernel_info,
                                       const int3& grid,
                                       WorkGroupSizeAlignment z_alignment,
                                       std::vector<int3>* work_groups);

void GetPossibleWorkGroupsXMultipleOf(int multiplier, const GpuInfo& gpu_info,
                                      const KernelInfo& kernel_info,
                                      const int3& grid,
                                      WorkGroupSizeAlignment z_alignment,
                                      std::vector<int3>* work_groups);

int3 GetWorkGroupXY128ConvLinear(const int3& grid);

int3 GetWorkGroupXY128Simple(const int3& grid);
int3 GetWorkGroupXY128Conv(const int3& grid);

bool XY128RequiresMoreWorkGroupsThenXY128Linear(int width, int height);

void GetPossibleWorkGroups(TuningType tuning_type, const GpuInfo& gpu_info,
                           const KernelInfo& kernel_info, const int3& grid,
                           std::vector<int3>* work_groups);

void GetPossibleWorkGroupsConv(TuningType tuning_type, const GpuInfo& gpu_info,
                               const KernelInfo& kernel_info, const int3& grid,
                               std::vector<int3>* work_groups);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_WORK_GROUP_PICKING_H_
