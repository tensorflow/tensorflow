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

#include "tensorflow/lite/delegates/gpu/gl/workgroups/best_effort_calculator.h"

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/gl/workgroups/calculator.h"
#include "tensorflow/lite/delegates/gpu/gl/workgroups/default_calculator.h"

#ifndef TFLITE_GPU_BINARY_RELEASE
#include "tensorflow/lite/delegates/gpu/gl/workgroups/calculator_from_metadata.h"
#endif

namespace tflite {
namespace gpu {
namespace gl {

std::unique_ptr<WorkgroupsCalculator> BestEffortWorkgroupsCalculator(
    const uint8_t* metadata, const GpuInfo& gpu_info) {
#ifndef TFLITE_GPU_BINARY_RELEASE
  std::unique_ptr<WorkgroupsCalculator> calculator_from_metadata =
      NewWorkgroupsCalculatorFromMetadata(metadata, gpu_info);
  if (calculator_from_metadata) {
    return calculator_from_metadata;
  }
#endif
  return NewDefaultWorkgroupsCalculator(gpu_info);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
