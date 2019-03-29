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

#include "tensorflow/lite/delegates/gpu/metal/kernels/sub.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

std::string GetSubCode(int src_count) {
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
      FLT4 value = src_buffer0[linear_index] - src_buffer1[linear_index];

      $2 
      dst_buffer[linear_index] = value;
    })";
  return code;
}
}  // namespace

std::vector<ComputeTaskDescriptorPtr> Sub(int id,
                                          std::vector<ValueId> input_ids,
                                          ValueId output_id) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source = GetSubCode(input_ids.size());

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

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
