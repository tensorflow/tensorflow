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

#include "tensorflow/lite/delegates/gpu/metal/kernels/reshape.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/util.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {
std::string GetReshapeCode() {
  std::string code = R"(
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
  const int3 igid = int3(gid);

  if (igid.x >= params.dst_size.x || igid.y >= params.dst_size.y ||
      igid.z * 4 >= params.dst_size.z) return;

  FLT4 value;

  for (int i = 0; i < 4; ++i) {
    const int dst_channel = igid.z * 4 + i;
    if (dst_channel < params.dst_size.z) {
      int p = dst_channel + params.dst_size.z * igid.x + params.dst_size.w * igid.y;
      int src_y = p / params.src_size.w;
      int t0 = p - src_y * params.src_size.w;  // p % params.src_size.w;
      int src_x = t0 / params.src_size.z;
      int src_z = t0 - src_x * params.src_size.z;  // t0 % params.src_size.z;
      int src_layer = src_z >> 2;
      int src_channel = src_z & 3;
      int src_linear_id = (src_layer * params.src_size.y + src_y) * params.src_size.x + src_x;
      value[i] = src_buffer[src_linear_id][src_channel];
    }
  }

  int linear_index = (igid.z * params.dst_size.y + igid.y) * params.dst_size.x + igid.x;
  $2
  dst_buffer[linear_index] = value;
})";
  return code;
}

std::string GetReshapex4Code() {
  std::string code = R"(
#include <metal_stdlib>
using namespace metal;

struct uniforms {
  int4 src_size;
  int4 dst_size;
  int2 plane_xz;
  int2 dummy0;  // dummy, for alignment
  int4 dummy1;  // dummy, for alignment
};

$0
kernel void ComputeFunction(
                            $1
                            uint3 gid[[thread_position_in_grid]]) {
  int X = gid.x;
  int Y = gid.y;
  int Z = gid.z;

  if (X >= params.dst_size.x || Y >= params.dst_size.y || Z >= params.dst_size.z) return;

  int p = Z + params.dst_size.z * X + params.plane_xz.y * Y;
  int src_y = p / params.plane_xz.x;
  int t0 = p - src_y * params.plane_xz.x;  // p % params.plane_xz.x;
  int src_x = t0 / params.src_size.z;
  int src_z = t0 - src_x * params.src_size.z;  // t0 % params.src_size.z;

  int src_index = src_z * params.src_size.w + src_y * params.src_size.x + src_x;
  int linear_index = Z * params.dst_size.w + Y * params.dst_size.x + X;
  FLT4 value = src_buffer[src_index];
  $2
  dst_buffer[linear_index] = value;
})";
  return code;
}

}  // namespace

std::vector<ComputeTaskDescriptorPtr> Reshape(int id, ValueId input_id,
                                              ValueId output_id,
                                              const ReshapeAttributes& attr) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source = GetReshapeCode();

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {
      output_id, "device FLT4* dst_buffer",
      [input_id, attr](const std::map<ValueId, BHWC>& buffers) {
        int batch = buffers.find(input_id)->second.b;
        return BHWC{batch, attr.new_shape.h, attr.new_shape.w,
                    attr.new_shape.c};
      }};

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id, output_id](const std::map<ValueId, BHWC>& buffers) {
         const auto& src_dim = buffers.find(input_id)->second;
         const auto& dst_dim = buffers.find(output_id)->second;
         std::vector<int> uniform_params{
             // int4 src_size
             src_dim.w,
             src_dim.h,
             src_dim.c,
             src_dim.c * src_dim.w,
             // int4 dst_size
             dst_dim.w,
             dst_dim.h,
             dst_dim.c,
             dst_dim.c * dst_dim.w,
         };
         return GetByteBuffer(uniform_params);
       }},
  };

  desc->resize_function = [attr](const std::map<ValueId, BHWC>& buffers) {
    const uint3 grid = uint3(attr.new_shape.w, attr.new_shape.h,
                             IntegralDivideRoundUp(attr.new_shape.c, 4));
    const uint3 groups_size = GetWorkGroupSizeForGrid(grid);
    int groups_x = IntegralDivideRoundUp(grid.x, groups_size.x);
    int groups_y = IntegralDivideRoundUp(grid.y, groups_size.y);
    int groups_z = IntegralDivideRoundUp(grid.z, groups_size.z);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };

  return {desc};
}

std::vector<ComputeTaskDescriptorPtr> Reshapex4(int id, ValueId input_id,
                                                ValueId output_id,
                                                const ReshapeAttributes& attr) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source = GetReshapex4Code();

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {
      output_id, "device FLT4* dst_buffer",
      [input_id, attr](const std::map<ValueId, BHWC>& buffers) {
        int batch = buffers.find(input_id)->second.b;
        return BHWC{batch, attr.new_shape.h, attr.new_shape.w,
                    attr.new_shape.c};
      }};

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id, output_id](const std::map<ValueId, BHWC>& buffers) {
         const auto& src_dim = buffers.find(input_id)->second;
         const auto& dst_dim = buffers.find(output_id)->second;
         std::vector<int32_t> uniform_params{
             // int4 src_size
             src_dim.w, src_dim.h, IntegralDivideRoundUp(src_dim.c, 4),
             src_dim.w * src_dim.h,
             // int4 dst_size
             dst_dim.w, dst_dim.h, IntegralDivideRoundUp(dst_dim.c, 4),
             dst_dim.w * dst_dim.h,
             // int2 plane_xz
             src_dim.w * IntegralDivideRoundUp(src_dim.c, 4),
             dst_dim.w * IntegralDivideRoundUp(dst_dim.c, 4),
             0,  // dummy, for alignment
             0,  // dummy, for alignment
             0,  // dummy, for alignment
             0,  // dummy, for alignment
             0,  // dummy, for alignment
             0   // dummy, for alignment
         };
         return GetByteBuffer(uniform_params);
       }},
  };

  desc->resize_function = [attr](const std::map<ValueId, BHWC>& buffers) {
    const uint3 grid = uint3(attr.new_shape.w, attr.new_shape.h,
                             IntegralDivideRoundUp(attr.new_shape.c, 4));
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
