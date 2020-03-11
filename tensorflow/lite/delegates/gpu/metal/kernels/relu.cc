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

#include "tensorflow/lite/delegates/gpu/metal/kernels/relu.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {

std::vector<ComputeTaskDescriptorPtr> ReLU(int id, ValueId input_id,
                                           ValueId output_id,
                                           const ReLUAttributes& attr) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = true;
  const std::string min_func =
      attr.alpha == 0 ? "FLT4(0.0f)" : "min(value * params.x, 0.0f)";
  const std::string parameters =
      "FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid, float2 params) "
      "{\n";
  if (attr.clip != 0.0) {
    desc->shader_source = parameters + "  return FLT4(clamp(value, " +
                          min_func + ", FLT4(params.y)));\n}";
  } else {
    desc->shader_source =
        parameters + "  return FLT4(max(value, " + min_func + "));\n}";
  }
  desc->input_buffers = {{input_id}};
  desc->output_buffer = {output_id};
  desc->uniform_buffers = {
      {"constant float2&",
       [attr](const std::map<ValueId, BHWC>& buffers) {
         return GetByteBuffer(std::vector<float>{attr.alpha, attr.clip});
       }},
  };
  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
