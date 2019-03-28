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

#include "tensorflow/lite/delegates/gpu/metal/kernels/upsample.h"

#include <map>
#include <memory>
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

std::vector<ComputeTaskDescriptorPtr> Upsample(
    int id, ValueId input_id, ValueId output_id,
    const Upsample2DAttributes& attr) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;

  desc->shader_source = R"(
    #include <metal_stdlib>
    using namespace metal;
    $0
    kernel void ComputeFunction(
                                $1
                                uint3 gid[[thread_position_in_grid]]) {
      if (int(gid.x) >= size.z || int(gid.y) >= size.w) {
        return;
      }
      const float2 tex_coord = float2(gid.xy) * scale;
      int4 st;
      const int2 borders = size.xy - int2(1, 1);
      st.xy = clamp(int2(tex_coord), int2(0, 0), borders);
      st.zw = min(st.xy + int2(1, 1), borders);
      const float2 t = tex_coord - float2(st.xy); //interpolating factors
      const int src_index0 = (gid.z * size.y + st.y) * size.x + st.x;
      const int src_index1 = (gid.z * size.y + st.y) * size.x + st.z;
      const int src_index2 = (gid.z * size.y + st.w) * size.x + st.x;
      const int src_index3 = (gid.z * size.y + st.w) * size.x + st.z;
      FLT4 tex11 = src_buffer[src_index0];
      FLT4 tex21 = src_buffer[src_index1];
      FLT4 tex12 = src_buffer[src_index2];
      FLT4 tex22 = src_buffer[src_index3];
      // bilinear interpolation
      FLT4 value = mix(mix(tex11, tex21, static_cast<FLT>(t.x)),
                       mix(tex12, tex22, static_cast<FLT>(t.x)), static_cast<FLT>(t.y));
      const int linear_index = (gid.z * size.w + gid.y) * size.z + gid.x;
      $2
      output_buffer[linear_index] = value;
    }
  )";

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {
      output_id, "device FLT4* output_buffer",
      [input_id, attr](const std::map<ValueId, BHWC>& buffers) {
        return CalculateOutputShape(buffers.find(input_id)->second, attr);
      }};

  desc->uniform_buffers = {
      {"constant int4& size",
       [input_id, output_id](const std::map<ValueId, BHWC>& buffers) {
         const auto& dimension = buffers.find(input_id)->second;
         const auto& output_dimension = buffers.find(output_id)->second;
         std::vector<int> sizes = {
             dimension.w,
             dimension.h,
             output_dimension.w,
             output_dimension.h,
         };
         return VectorToUint8Vector(sizes);
       }},
      {"constant float2& scale",
       [input_id, output_id, attr](const std::map<ValueId, BHWC>& buffers) {
         const auto& input_dimensions = buffers.find(input_id)->second;
         const auto& output_dimensions = buffers.find(output_id)->second;
         std::vector<float> sizes = {
             CalculateResizeScale(input_dimensions.w, output_dimensions.w,
                                  attr),
             CalculateResizeScale(input_dimensions.h, output_dimensions.h,
                                  attr),
         };
         return VectorToUint8Vector(sizes);
       }},
  };

  desc->resize_function = [output_id](const std::map<ValueId, BHWC>& buffers) {
    const uint3 groups_size{16, 16, 1};
    const auto& dst_dim = buffers.find(output_id)->second;
    int groups_x = IntegralDivideRoundUp(dst_dim.w, groups_size.x);
    int groups_y = IntegralDivideRoundUp(dst_dim.h, groups_size.y);
    const int dst_layers = IntegralDivideRoundUp(dst_dim.c, 4);
    int groups_z = IntegralDivideRoundUp(dst_layers, groups_size.z);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };
  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
