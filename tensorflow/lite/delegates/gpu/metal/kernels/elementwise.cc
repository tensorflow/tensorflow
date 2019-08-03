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

#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace metal {

namespace {

std::string GetElementwiseWithTwoInputsCode(int src_count,
                                            OperationType op_type) {
  std::string code = R"(
    #include <metal_stdlib>
    using namespace metal;

    struct uniforms {
      int4 src_size;
    };

    $0
    kernel void ComputeFunction(
                                $1
                                uint3 gid[[thread_position_in_grid]]) {
      if (static_cast<int>(gid.x) >= params.src_size.x ||
          static_cast<int>(gid.y) >= params.src_size.y) {
        return;
      }

      int linear_index = (int(gid.z) * params.src_size.y + int(gid.y)) *
        params.src_size.x + int(gid.x);
        )";

  switch (op_type) {
    case OperationType::SUB: {
      code +=
          " FLT4 value = src_buffer0[linear_index] - "
          "src_buffer1[linear_index];";
      break;
    }
    case OperationType::DIV: {
      code +=
          " FLT4 value = src_buffer0[linear_index] / "
          "src_buffer1[linear_index];";
      break;
    }
    case OperationType::POW: {
      code +=
          " FLT4 value = pow(src_buffer0[linear_index], "
          "src_buffer1[linear_index]);";
      break;
    }
    case OperationType::SQUARED_DIFF: {
      code += R"(
     FLT4 src_0 = src_buffer0[linear_index];
     FLT4 src_1 = src_buffer1[linear_index];
     FLT4 value = (src_0 - src_1) * (src_0 - src_1);
   )";
      break;
    }
    default: {
      return "";
    }
  }
  code += R"(
      $2
      dst_buffer[linear_index] = value;
    })";
  return code;
}
}  // namespace

std::vector<ComputeTaskDescriptorPtr> ElementwiseWithTwoInputs(
    int id, std::vector<ValueId> input_ids, ValueId output_id,
    OperationType op_type) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source =
      GetElementwiseWithTwoInputsCode(input_ids.size(), op_type);

  for (int i = 0; i < input_ids.size(); ++i) {
    const std::string buffer_name =
        "device FLT4* const src_buffer" + std::to_string(i);
    desc->input_buffers.push_back({input_ids[i], buffer_name});
  }

  desc->output_buffer = {output_id, "device FLT4* dst_buffer",
                         [input_ids](const std::map<ValueId, BHWC>& buffers) {
                           return buffers.find(input_ids[0])->second;
                         }};

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_ids](const std::map<ValueId, BHWC>& buffers) {
         const auto& dimension = buffers.find(input_ids[0])->second;
         std::vector<int> uniform_params = {dimension.w, dimension.h, 0, 0};
         return VectorToUint8Vector(uniform_params);
       }},
  };

  desc->resize_function = [input_ids](const std::map<ValueId, BHWC>& buffers) {
    const auto& src_dim = buffers.find(input_ids[0])->second;
    const uint3 groups_size{16, 16, 1};
    int groups_x = IntegralDivideRoundUp(src_dim.w, groups_size.x);
    int groups_y = IntegralDivideRoundUp(src_dim.h, groups_size.y);
    const int dst_layers = IntegralDivideRoundUp(src_dim.c, 4);
    int groups_z = IntegralDivideRoundUp(dst_layers, groups_size.z);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };
  return {desc};
}

std::vector<ComputeTaskDescriptorPtr> ElementwiseWithOneInput(
    int id, ValueId input_id, ValueId output_id, OperationType op_type) {
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
