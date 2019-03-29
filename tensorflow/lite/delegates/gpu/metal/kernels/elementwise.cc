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

#include "tensorflow/lite/delegates/gpu/metal/kernels/elementwise.h"

#include <unordered_map>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {

std::vector<ComputeTaskDescriptorPtr> Elementwise(int id, ValueId input_id,
                                                  ValueId output_id,
                                                  OperationType op_type) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = true;

  const std::unordered_map<OperationType, std::string> functors{
      {OperationType::ABS, "abs(value)"},
      {OperationType::SIN, "sin(value)"},
      {OperationType::COS, "cos(value)"},
      {OperationType::LOG, "log(value)"},
      {OperationType::SQRT, "sqrt(value)"},
      {OperationType::RSQRT, "1.0 / sqrt(value)"},
      {OperationType::SQUARE, "value * value"},
      {OperationType::SIGMOID, "1.0 / (1.0 + exp(-1.0 * value))"},
      {OperationType::TANH, "tanh(value)"},
  };

  if (functors.count(op_type) == 0) {
    return {};
  }

  desc->shader_source =
      "FLT4 linkable$0(FLT4 value, int linear_index, uint3 gid) {\n";
  desc->shader_source += "    return " + functors.at(op_type) + ";\n";
  desc->shader_source += "  }";

  desc->input_buffers = {{input_id}};
  desc->output_buffer = {output_id};
  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
