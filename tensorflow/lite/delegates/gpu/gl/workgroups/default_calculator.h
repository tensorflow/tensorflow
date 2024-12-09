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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_WORKGROUPS_DEFAULT_CALCULATOR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_WORKGROUPS_DEFAULT_CALCULATOR_H_

#include <memory>

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/gl/workgroups/calculator.h"

namespace tflite {
namespace gpu {
namespace gl {

// Creates new workgroups calculator for the general case or specificly for Mali
std::unique_ptr<WorkgroupsCalculator> NewDefaultWorkgroupsCalculator(
    const GpuInfo& gpu_info);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_WORKGROUPS_DEFAULT_CALCULATOR_H_
