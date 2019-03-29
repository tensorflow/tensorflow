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

#include "tensorflow/lite/delegates/gpu/gl/workgroups/default_calculator.h"

#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/workgroups/calculator.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class DefaultWorkgroupsCalculator : public WorkgroupsCalculator {
 public:
  explicit DefaultWorkgroupsCalculator(const GpuInfo& gpu_info)
      : WorkgroupsCalculator(gpu_info) {}
  uint3 CalculateInternal(const ShaderCode& shader_code) const final {
    const auto& workload = shader_code.workload;
    if (workload.z >= 64) {
      return uint3(4, 4, 64);
    }
    if (workload.z >= 32) {
      return uint3(8, 4, 32);
    }
    if (workload.z >= 16) {
      return uint3(8, 8, 16);
    }
    if (workload.z >= 8) {
      return uint3(16, 8, 8);
    }
    if (workload.z >= 4) {
      return uint3(16, 16, 4);
    }
    if (workload.z >= 2) {
      return uint3(32, 16, 2);
    }
    return uint3(32, 32, 1);
  }
};

class WorkgroupsCalculatorForMali : public WorkgroupsCalculator {
 public:
  explicit WorkgroupsCalculatorForMali(const GpuInfo& gpu_info)
      : WorkgroupsCalculator(gpu_info) {}
  uint3 CalculateInternal(const ShaderCode& shader_code) const final {
    const auto& workload = shader_code.workload;
    if (workload.z >= 32) {
      return uint3(2, 2, 32);
    }
    if (workload.z >= 16) {
      return uint3(4, 2, 16);
    }
    if (workload.z >= 8) {
      return uint3(4, 4, 8);
    }
    if (workload.z >= 4) {
      return uint3(8, 4, 4);
    }
    if (workload.z >= 2) {
      return uint3(8, 8, 2);
    }
    return uint3(16, 8, 1);
  }
};

}  // namespace

std::unique_ptr<WorkgroupsCalculator> NewDefaultWorkgroupsCalculator(
    const GpuInfo& gpu_info) {
  if (gpu_info.type == GpuType::MALI) {
    return absl::make_unique<WorkgroupsCalculatorForMali>(gpu_info);
  } else {
    return absl::make_unique<DefaultWorkgroupsCalculator>(gpu_info);
  }
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
