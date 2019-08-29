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

#include "tensorflow/lite/delegates/gpu/metal/kernels/hard_swish.h"

#include <memory>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {

std::vector<ComputeTaskDescriptorPtr> HardSwish(int id, ValueId input_id,
                                                ValueId output_id,
                                                const RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = true;
  desc->shader_source = R"(
      FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid) {
        return value * clamp(value / 6.0f + FLT4(0.5f), FLT4(0.0f), FLT4(1.0f));
      }
  )";
  desc->input_buffers = {{input_id}};
  desc->output_buffer = {output_id};
  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
