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

#include "tensorflow/lite/delegates/gpu/gl/workgroups/calculator.h"

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/shader_code.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

uint64_t CalculateProduct(const uint3& value) {
  return static_cast<uint64_t>(value.x) * value.y * value.z;
}

void MaybeShrinkWorkgroup(const GpuInfo& gpu_info, uint3* wg) {
  while (wg->x > gpu_info.GetMaxWorkGroupSizeForX()) {
    wg->x /= 2;
  }

  while (wg->y > gpu_info.GetMaxWorkGroupSizeForY()) {
    wg->y /= 2;
  }

  while (wg->z > gpu_info.GetMaxWorkGroupSizeForZ()) {
    wg->z /= 2;
  }

  // Code below decreases amount of invocations per workgroup in a balanced way.
  // As example, workgroup size is x=16, y=8, z=8 (16x8x8 = 1024), but
  // max_work_group_total_size = 512. We need to fit this limit and we can
  // reduce workgroup size in different ways, but we want to use the most
  // balanced way. So code below will find the maximal of three dimensions and
  // reduce it, so the whole workgroup is kept balanced by all dimensions. And
  // the final reduced workgroup will be x=8, y=8, z=8 for the given example.
  while (CalculateProduct(*wg) > gpu_info.GetMaxWorkGroupTotalSize()) {
    unsigned int* max = &wg->x;
    if (wg->y > *max) max = &wg->y;
    if (wg->z > *max) max = &wg->z;
    *max = *max /= 2;
  }
}

}  // namespace

WorkgroupsCalculator::WorkgroupsCalculator(const GpuInfo& gpu_info)
    : gpu_info_{gpu_info} {}

uint3 WorkgroupsCalculator::Calculate(const ShaderCode& shader_code) const {
  uint3 workgroup_size = shader_code.recommended_workgroup;
  if (workgroup_size == kEmptyWorkgroupSize) {
    workgroup_size = CalculateInternal(shader_code);
  }
  MaybeShrinkWorkgroup(gpu_info_, &workgroup_size);
  return workgroup_size;
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
