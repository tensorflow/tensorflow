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

#include "tensorflow/lite/delegates/gpu/metal/kernels/mul.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/substitute.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {

std::vector<ComputeTaskDescriptorPtr> Multiply(
    int id, ValueId input_id, ValueId output_id,
    const MultiplyScalarAttributes& attr, const RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  auto multiplier = absl::get_if<float>(&attr.param);
  auto mul_buffer =
      absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr.param);
  const bool scalar = multiplier != nullptr;
  std::string code = R"(
    #include <metal_stdlib>
    using namespace metal;
    $0
    kernel void ComputeFunction(
                                $1
                                uint3 gid[[thread_position_in_grid]]
                                ) {
        if (int(gid.x) >= size.x || int(gid.y) >= size.y) {
            return;
        }
        const int linear_index = (gid.z * size.y + gid.y) * size.x + gid.x;
        FLT4 value = FLT4(0.0f);
    )";
  if (scalar) {
    code += "value = input_buffer[linear_index] * multiplier;\n";
  } else {
    code += "value = input_buffer[linear_index] * mul_buf[gid.z];\n";
  }
  code += "$2\n";
  code += "output_buffer[linear_index] = value;\n";
  code += "}\n";
  desc->shader_source = code;
  desc->input_buffers = {{input_id, "device FLT4* const input_buffer"}};
  desc->output_buffer = {output_id, "device FLT4* output_buffer", [input_id](const std::map<ValueId, BHWC>& buffers) {
    return buffers.find(input_id)->second;
  }};
  if (scalar) {
    std::vector<uint8_t> multiplier_bits =
        GetByteBuffer(std::vector<float>{*multiplier});
    desc->uniform_buffers = {
        {"constant float& multiplier",
         [multiplier_bits](const std::map<ValueId, BHWC>& buffers) {
           return multiplier_bits;
         }},
        {"constant int2& size", [input_id](const std::map<ValueId, BHWC>& buffers) {
            const auto& dimension = buffers.find(input_id)->second;
            std::vector<int> uniform_params = {dimension.w, dimension.h};
            return VectorToUint8Vector(uniform_params);
        }}
    };
  } else {
    auto coeffs = GetByteBufferConverted(mul_buffer->data, options.storage_precision);
    desc->uniform_buffers = {
      {"constant int2& size", [input_id](const std::map<ValueId, BHWC>& buffers) {
          const auto& dimension = buffers.find(input_id)->second;
          std::vector<int> uniform_params = {dimension.w, dimension.h};
          return VectorToUint8Vector(uniform_params);
      }}
    };
    desc->immutable_buffers = {
        {"device FLT4* const mul_buf", coeffs},
    };

    desc->resize_function = [input_id](const std::map<ValueId, BHWC>& buffers) {
      const auto& src_dim = buffers.find(input_id)->second;
      const uint3 groups_size{16, 16, 1};
      int groups_x = IntegralDivideRoundUp(src_dim.w, groups_size.x);
      int groups_y = IntegralDivideRoundUp(src_dim.h, groups_size.y);
      const int dst_layers = IntegralDivideRoundUp(src_dim.c, 4);
      int groups_z = IntegralDivideRoundUp(dst_layers, groups_size.z);
      return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
    };
  }
  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
