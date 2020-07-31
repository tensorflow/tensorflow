/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/metal/kernels/mean.h"

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
#include "tensorflow/lite/delegates/gpu/metal/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {

std::string GetMeanCode(const int3& work_group_size) {
  const std::string wg_x = std::to_string(work_group_size.x);
  const std::string wg_y = std::to_string(work_group_size.y);
  std::string c = R"(
    #include <metal_stdlib>
    using namespace metal;
    struct uniforms {
      int4 src_size;
      float4 inv_multipliers;
    };

    $0
    kernel void ComputeFunction(
                                $1
                                uint tid[[thread_index_in_threadgroup]],
                                uint3 tid3d[[thread_position_in_threadgroup]],
                                uint3 gid[[thread_position_in_grid]]) {
  int local_x = static_cast<int>(tid3d.x);
  int local_y = static_cast<int>(tid3d.y);
  int local_id = static_cast<int>(tid);
  int S = static_cast<int>(gid.z);
  if (S >= params.src_size.z) return;
)";
  c += "  threadgroup float4 accum[" +
       std::to_string(work_group_size.x * work_group_size.y) + "];\n";
  c += "  accum[local_id] = float4(0.0f);\n";
  c += "  int src_offset = S * params.src_size.x * params.src_size.y;\n";
  c += "  for (int s_y = local_y; s_y < params.src_size.y; s_y += " + wg_y +
       ") {\n";
  c += "    for (int s_x = local_x; s_x < params.src_size.x; s_x += " + wg_x +
       ") {\n";
  c += "      int src_index = src_offset + s_y * params.src_size.x + s_x;\n";
  c += "      accum[local_id] += float4(src_buffer[src_index]);\n";
  c += "    }\n";
  c += "  }\n";
  c += "  accum[local_id] *= params.inv_multipliers.x;\n";
  c += "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  const int total_size = work_group_size.x * work_group_size.y;
  int offset = 1;
  int reminder = total_size / 4;
  for (; reminder >= 8; reminder /= 4, offset *= 4) {
    c += "  if (local_id < " + std::to_string(reminder) + ") {\n";
    c += "    int t = local_id * " + std::to_string(offset * 4) + ";\n";
    c += "    float4 sum = accum[t + " + std::to_string(offset) + "];\n";
    c += "    sum += accum[t + " + std::to_string(offset * 2) + "];\n";
    c += "    sum += accum[t + " + std::to_string(offset * 3) + "];\n";
    c += "    accum[t] += sum;\n";
    c += "  }\n";
    c += "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  }
  c += "  float4 sum = accum[0];\n";
  reminder *= 4;
  for (int i = 1; i < reminder; ++i) {
    c += "  sum += accum[" + std::to_string(offset * i) + "];\n";
  }
  c += "  FLT4 value = FLT4(sum * params.inv_multipliers.y);\n";
  c += R"(
  const int linear_index = static_cast<int>(gid.z);
  $2
  dst_buffer[linear_index] = value;
}
)";
  return c;
}

std::vector<ComputeTaskDescriptorPtr> Mean(int id, ValueId input_id,
                                           ValueId output_id,
                                           const MeanAttributes& attr) {
  if (attr.dims != std::set<Axis>({Axis::HEIGHT, Axis::WIDTH})) {
    // Mean calculation is supported only for height and width
    return {};
  }

  const int3 work_group_size = int3(16, 16, 1);

  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  std::string code = GetMeanCode(work_group_size);
  desc->shader_source = code;

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {output_id, "device FLT4* dst_buffer",
                         [input_id](const std::map<ValueId, BHWC>& buffers) {
                           const auto& input_dimension =
                               buffers.find(input_id)->second;
                           return BHWC(1, 1, 1, input_dimension.c);
                         }};
  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id, output_id,
        work_group_size](const std::map<ValueId, BHWC>& buffers) {
         const auto& src_shape = buffers.find(input_id)->second;
         const int src_slices = DivideRoundUp(src_shape.c, 4);
         struct uniforms {
           int4 src_size;
           float4 inv_multipliers;
         };
         uniforms params;
         params.src_size = {src_shape.w, src_shape.h, src_slices, 0};
         const double total_size = src_shape.w * src_shape.h;
         const double size_0 = work_group_size.x * work_group_size.y;
         const double size_1 = total_size / size_0;
         params.inv_multipliers.x = 1.0 / size_1;
         params.inv_multipliers.y = 1.0 / size_0;
         const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&params);
         return std::vector<uint8_t>(ptr, ptr + sizeof(uniforms));
       }},
  };

  desc->resize_function = [output_id, work_group_size](
                              const std::map<ValueId, BHWC>& buffers) {
    BHWC dst_shape = buffers.find(output_id)->second;
    const int dst_slices = DivideRoundUp(dst_shape.c, 4);
    const int groups_z = DivideRoundUp(dst_slices, work_group_size.z);
    return std::make_pair(work_group_size, uint3{1, 1, groups_z});
  };
  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
