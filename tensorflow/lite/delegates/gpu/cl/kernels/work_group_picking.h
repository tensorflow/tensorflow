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

#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/tuning_parameters.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/workgroup_selection.h"

namespace tflite {
namespace gpu {
namespace cl {

// writes best_work_group if successful
// Here and later you can find XY128, this is because 128 is SIMD width of A6xx
// And XY128 means that work_group_size.x * work_group_size.y % 128 = 0
// We need it to correctly work with constants uploading on A6xx
Status GetBestWorkGroupXY128(const TuningParameters& params,
                             const CLKernel& kernel, const int3& grid,
                             WorkGroupSizeAlignment z_alignment,
                             int3* best_work_group);

Status GetBestWorkGroupXY128Linear(const TuningParameters& params,
                                   const CLKernel& kernel, const int3& grid,
                                   WorkGroupSizeAlignment z_alignment,
                                   int3* best_work_group);

int3 GetWorkGroupXY128ConvLinear(const int3& grid);

int3 GetWorkGroupXY128Simple(const int3& grid);
int3 GetWorkGroupXY128Conv(const int3& grid);

bool XY128RequiresMoreWorkGroupsThenXY128Linear(int width, int height);

Status GetBestWorkGroup(const TuningParameters& params, const CLKernel& kernel,
                        const int3& grid, int3* best_work_group);

Status GetBestWorkGroupConv(const TuningParameters& params,
                            const CLKernel& kernel, const int3& grid,
                            int3* best_work_group);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_WORK_GROUP_PICKING_H_
