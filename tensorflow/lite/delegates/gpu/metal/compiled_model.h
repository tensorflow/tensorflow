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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMPILED_MODEL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMPILED_MODEL_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {

using CompiledModel = std::vector<ComputeTaskDescriptorPtr>;

// Receives input CompiledModel, validates, optimizes it and returns output
// CompiledModel. No shader compilation or memory allocation happen here, this
// function just does high-level operations fusion.
Status ValidateOptimizeModel(const std::vector<ValueId>& input_buffers,
                             const std::vector<ValueId>& output_buffers,
                             const CompiledModel& input, CompiledModel* output);

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMPILED_MODEL_H_
