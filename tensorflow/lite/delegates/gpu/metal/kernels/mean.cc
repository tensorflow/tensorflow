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

std::string GetMeanCode() {
  std::string shader_source = R"(
    #include <metal_stdlib>
    using namespace metal;
    struct uniforms {
      int4 src_size;
      int4 dst_size;
    };

    $0
    kernel void ComputeFunction(
                                $1
                                uint3 gid[[thread_position_in_grid]]) {
      if (static_cast<int>(gid.x) >= params.dst_size.x ||
          static_cast<int>(gid.y) >= params.dst_size.y ||
          static_cast<int>(gid.z) >= params.dst_size.z) {
        return;
      }

      float4 sum = float4(0.0);
      float size = float( params.src_size.x * params.src_size.y);
      for (int w = 0; w < params.src_size.x; w++) {
        for (int h = 0; h < params.src_size.y; h++) {
          const int buffer_index =
            (gid.z * params.src_size.y + h) * params.src_size.x + w;
          sum += src_buffer[buffer_index];
        }
      }
      sum /= size;
      const int linear_index =
      (gid.z * params.dst_size.y + int(gid.y)) * params.dst_size.x + int(gid.x);

      FLT4 value = FLT4(sum);
      $2
      output_buffer[linear_index] = value;
    }
  )";
  return shader_source;
}

std::vector<ComputeTaskDescriptorPtr> Mean(int id, ValueId input_id,
                                           ValueId output_id,
                                           const MeanAttributes& attr) {
  if (attr.dims != std::set<Axis>({Axis::HEIGHT, Axis::WIDTH})) {
    // Mean calculation is supported only for height and width
    return {};
  }

  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  std::string code = GetMeanCode();
  desc->shader_source = code;

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {output_id, "device FLT4* output_buffer",
                         [input_id](const std::map<ValueId, BHWC>& buffers) {
                           const auto& input_dimension =
                               buffers.find(input_id)->second;
                           return BHWC(1, 1, 1, input_dimension.c);
                         }};
  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id, output_id](const std::map<ValueId, BHWC>& buffers) {
         const auto& dimension = buffers.find(input_id)->second;
         const auto& output_dimension = buffers.find(output_id)->second;
         std::vector<int> uniform_params = {
             dimension.w,
             dimension.h,
             IntegralDivideRoundUp(dimension.c, 4),
             0,
             output_dimension.w,
             output_dimension.h,
             IntegralDivideRoundUp(dimension.c, 4),
             0};
         return GetByteBuffer(uniform_params);
       }},
  };

  desc->resize_function = [output_id](const std::map<ValueId, BHWC>& buffers) {
    BHWC dst_shape = buffers.find(output_id)->second;
    const uint3 grid =
        uint3(dst_shape.w, dst_shape.h, IntegralDivideRoundUp(dst_shape.c, 4));
    const uint3 groups_size = GetWorkGroupSizeForGrid(grid);
    int groups_x = IntegralDivideRoundUp(grid.x, groups_size.x);
    int groups_y = IntegralDivideRoundUp(grid.y, groups_size.y);
    int groups_z = IntegralDivideRoundUp(grid.z, groups_size.z);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };
  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
