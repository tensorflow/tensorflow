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

#include "tensorflow/lite/delegates/gpu/metal/kernels/softmax.h"

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {

std::vector<ComputeTaskDescriptorPtr> Softmax(int id, ValueId input_id,
                                              ValueId output_id,
                                              int channels_count,
                                              const RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source = R"(
    #include <metal_stdlib>
    using namespace metal;
    constant int src_channels = )";
  desc->shader_source += std::to_string(channels_count);
  desc->shader_source += R"(;
    $0
    kernel void ComputeFunction(
                                $1
                                uint3 gid[[thread_position_in_grid]]) {
      if (int(gid.x) >= size.x || int(gid.y) >= size.y) {
        return;
      }
      float shift = 0.0f;
      int remaining_channels = src_channels % 4;

      float sum = 0.0f;
      for (int d = 0; d < src_channels / 4; ++d) {
        int buffer_index = (d * size.y + gid.y) * size.x + gid.x;
        sum += dot(float4(1.0f), exp(float4(input_buffer[buffer_index]) - shift));
      }
      if (remaining_channels > 0) {
        int buffer_index = ((src_channels / 4) * size.y + gid.y) * size.x + gid.x;
        float4 last_element = float4(input_buffer[buffer_index]);
        sum += exp(last_element.x - shift);
        if (remaining_channels > 1) sum += exp(last_element.y - shift);
        if (remaining_channels == 3) sum += exp(last_element.z - shift);
      }

      for (int d = 0; d < (src_channels + 3) / 4; ++d) {
        const int linear_index = (d * size.y + gid.y) * size.x + gid.x;
        FLT4 value = FLT4(exp(float4(input_buffer[linear_index]) - shift) / sum);
        $2
        output_buffer[linear_index] = value;
      }
    }
  )";

  desc->input_buffers = {
      {input_id, "device FLT4* const input_buffer"},
  };

  desc->output_buffer = {output_id, "device FLT4* output_buffer",
                         [input_id](const std::map<ValueId, BHWC>& buffers) {
                           return buffers.find(input_id)->second;
                         }};

  desc->uniform_buffers = {
      {"constant int2& size",
       [output_id](const std::map<ValueId, BHWC>& buffers) {
         const auto& dimension = buffers.find(output_id)->second;
         std::vector<int> sizes{dimension.w, dimension.h};
         return VectorToUint8Vector(sizes);
       }},
  };

  desc->resize_function = [output_id](const std::map<ValueId, BHWC>& buffers) {
    uint3 groups_size{16, 16, 1};
    const auto& dimension = buffers.find(output_id)->second;
    uint3 groups_count{IntegralDivideRoundUp(dimension.w, groups_size.x),
                       IntegralDivideRoundUp(dimension.h, groups_size.y), 1};
    return std::make_pair(groups_size, groups_count);
  };

  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
