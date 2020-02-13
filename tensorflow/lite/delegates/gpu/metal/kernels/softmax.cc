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
namespace {
std::string GetSoftmax1x1Code() {
  std::string code = R"(
#include <metal_stdlib>
using namespace metal;

struct uniforms {
  int4 size;
  float4 mask;
};

$0

kernel void ComputeFunction($1
                            uint tid[[thread_index_in_threadgroup]],
                            uint3 ugid[[thread_position_in_grid]])
{
  int offset = 0;
  float sum = 0.0f;
  int s = 0;
  do {
    if (offset + tid < params.size.x) {
      float4 mask_temp = offset + tid == params.size.x - 1 ? params.mask : float4(1.0h);
      float4 src = float4(src_buffer[offset + tid]);
      sum += dot(mask_temp, exp(src));
      offset += 32;
    }
    s++;
  } while (s < params.size.y);

  threadgroup float4 tmp[8];
  threadgroup float* tmpx1 = (threadgroup float*)tmp;
  tmpx1[tid] = sum;
  BARRIER(mem_flags::mem_threadgroup);
  if (tid == 0) {
    sum = dot(float4(1.0f), tmp[0]);
    sum += dot(float4(1.0f), tmp[1]);
    sum += dot(float4(1.0f), tmp[2]);
    sum += dot(float4(1.0f), tmp[3]);
    sum += dot(float4(1.0f), tmp[4]);
    sum += dot(float4(1.0f), tmp[5]);
    sum += dot(float4(1.0f), tmp[6]);
    sum += dot(float4(1.0f), tmp[7]);
    tmpx1[0] = 1.0 / sum;
  }
  BARRIER(mem_flags::mem_threadgroup);
  sum = tmpx1[0];

  offset = 0;
  s = 0;
  do {
    if (offset + tid < params.size.x) {
      int linear_index = offset + tid;
      FLT4 value = FLT4(exp(float4(src_buffer[linear_index])) * sum);
      uint3 gid = uint3(0, 0, linear_index);
      $2
      dst_buffer[linear_index] = value;
      offset += 32;
    }
    s++;
  } while (s < params.size.y);
})";
  return code;
}
}  // namespace

std::vector<ComputeTaskDescriptorPtr> Softmax(int id, ValueId input_id,
                                              ValueId output_id,
                                              int channels_count) {
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
         return GetByteBuffer(sizes);
       }},
  };

  desc->resize_function = [output_id](const std::map<ValueId, BHWC>& buffers) {
    uint3 groups_size{8, 4, 1};
    const auto& dimension = buffers.find(output_id)->second;
    uint3 groups_count{IntegralDivideRoundUp(dimension.w, groups_size.x),
                       IntegralDivideRoundUp(dimension.h, groups_size.y), 1};
    return std::make_pair(groups_size, groups_count);
  };

  return {desc};
}

std::vector<ComputeTaskDescriptorPtr> Softmax1x1(int id, ValueId input_id,
                                                 ValueId output_id,
                                                 int channels_count) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source = GetSoftmax1x1Code();

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {output_id, "device FLT4* dst_buffer",
                         [input_id](const std::map<ValueId, BHWC>& buffers) {
                           return buffers.find(input_id)->second;
                         }};

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [channels_count](const std::map<ValueId, BHWC>& buffers) {
         const int src_depth = IntegralDivideRoundUp(channels_count, 4);
         struct uniforms {
           int4 size;
           float4 mask;
         };
         uniforms params;
         params.size = {src_depth, IntegralDivideRoundUp(src_depth, 32), 1, 1};
         params.mask = {0.0f, 0.0f, 0.0f, 0.0f};
         const int reminder = channels_count % 4 == 0 ? 4 : channels_count % 4;
         for (int i = 0; i < reminder; ++i) {
           params.mask[i] = 1.0f;
         }
         const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&params);
         return std::vector<uint8_t>(ptr, ptr + sizeof(uniforms));
       }},
  };

  desc->resize_function = [](const std::map<ValueId, BHWC>& buffers) {
    return std::make_pair(uint3{32u, 1u, 1u}, uint3{1u, 1u, 1u});
  };

  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
