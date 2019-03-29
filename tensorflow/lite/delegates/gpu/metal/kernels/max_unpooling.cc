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

#include "tensorflow/lite/delegates/gpu/metal/kernels/max_unpooling.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

std::string GetMaxUnpoolingCode(const HW& kernel_size) {
  std::string shader_source = R"(
    #include <metal_stdlib>
    using namespace metal;
    constant int window_w = $0;
    struct uniforms {
      int2 src_size;
      int2 dst_size;
      int2 stride;
      int2 offset;
    };

    $$0
    kernel void ComputeFunction(
                                $$1
                                uint3 gid[[thread_position_in_grid]]) {
      int X = static_cast<int>(gid.x);
      int Y = static_cast<int>(gid.y);
      if (X >= params.dst_size.x || Y >= params.dst_size.y) {
        return;
      }

      int src_x = (X + params.offset.x) / params.stride.x;
      int src_y = (Y + params.offset.y) / params.stride.y;

      bool outside = src_x < 0 || src_y < 0 ||
        src_x >= params.src_size.x || src_y >= params.src_size.y;

      int src_index = (gid.z * params.src_size.y + src_y) * params.src_size.x + src_x;
      int linear_index = (gid.z * params.dst_size.y + Y) * params.dst_size.x + X;

      int4 indexes = outside ? int4(0) : int4(src_indices_buffer[src_index]);
      FLT4 src_color = outside ? FLT4(0.0f) : src_buffer[src_index];

      int t_x = X - (src_x * params.stride.x - params.offset.x);
      int t_y = Y - (src_y * params.stride.y - params.offset.y);
      int t_index = t_y * window_w + t_x;

      FLT4 value;
      value.x = t_index == indexes.x ? src_color.x : 0.0;
      value.y = t_index == indexes.y ? src_color.y : 0.0;
      value.z = t_index == indexes.z ? src_color.z : 0.0;
      value.w = t_index == indexes.w ? src_color.w : 0.0;

      $$2
      output_buffer[linear_index] = value;
    }
  )";
  return absl::Substitute(shader_source, kernel_size.w);
}
}  // namespace

std::vector<ComputeTaskDescriptorPtr> MaxUnpooling(
    int id, ValueId input_id, ValueId input_indices_id, ValueId output_id,
    const MaxUnpooling2DAttributes& params) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source = GetMaxUnpoolingCode(params.kernel);

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
      {input_indices_id, "device FLT4* const src_indices_buffer"},
  };

  desc->output_buffer = {output_id, "device FLT4* output_buffer",
                         [input_id, input_indices_id,
                          params](const std::map<ValueId, BHWC>& buffers) {
                           return CalculateOutputShape(
                               buffers.find(input_id)->second, params);
                         }};

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id, input_indices_id, output_id,
        params](const std::map<ValueId, BHWC>& buffers) {
         const auto& dimension = buffers.find(input_id)->second;
         const auto& output_dimension = buffers.find(output_id)->second;
         std::vector<int> uniform_params{
             dimension.w,
             dimension.h,
             output_dimension.w,
             output_dimension.h,
             params.strides.w,
             params.strides.h,
             params.padding.prepended.w,
             params.padding.prepended.h,
         };
         return VectorToUint8Vector(uniform_params);
       }},
  };

  desc->resize_function = [input_id, input_indices_id,
                           params](const std::map<ValueId, BHWC>& buffers) {
    const auto& src_shape = buffers.find(input_id)->second;
    BHWC dst_shape = CalculateOutputShape(src_shape, params);
    const uint3 groups_size{16, 16, 1};
    int groups_x = IntegralDivideRoundUp(dst_shape.w, groups_size.x);
    int groups_y = IntegralDivideRoundUp(dst_shape.h, groups_size.y);
    int groups_z = IntegralDivideRoundUp(dst_shape.c, 4);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };

  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
