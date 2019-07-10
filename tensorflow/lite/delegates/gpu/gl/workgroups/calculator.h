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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_WORKGROUPS_CALCULATOR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_WORKGROUPS_CALCULATOR_H_

#include <memory>

#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/shader_code.h"
#include "tensorflow/lite/delegates/gpu/gl/gpu_info.h"

namespace tflite {
namespace gpu {
namespace gl {

constexpr uint3 kEmptyWorkgroupSize(0, 0, 0);

// Calculates workgroup size for the given shader code in a model graph.
//
// Potentially there are multiple implementations possible:
//   - per-operation type hard-coded constants
//   - statistic-based calculator that uses aggregated stats for all operations
class WorkgroupsCalculator {
 public:
  explicit WorkgroupsCalculator(const GpuInfo& gpu_info);

  virtual ~WorkgroupsCalculator() = default;

  // Uses shader code recommended work group size if available and doesn't
  // exceed max work group invocations num, otherwise work group size from
  // passed calculator.
  uint3 Calculate(const ShaderCode& shader_code) const;

 protected:
  virtual uint3 CalculateInternal(const ShaderCode& shader_code) const = 0;

 private:
  GpuInfo gpu_info_;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_WORKGROUPS_CALCULATOR_H_
