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

#include "tensorflow/lite/delegates/gpu/metal/kernels/fully_connected.h"

#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/environment.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

std::string GetFullyConnectedCode(bool shared_memory, int src_channels,
                                  int dst_channels) {
  const int src_depth = IntegralDivideRoundUp(src_channels, 4);
  std::stringstream code;
  code << R"(
    #include <metal_stdlib>
    using namespace metal;

    struct uniforms {
      uint src_depth;
      uint dst_channels;
      uint out_channels;
      uint dummy;
    };

    $$0
    kernel void ComputeFunction(
                                $$1
                                uint3 tid[[thread_position_in_threadgroup]],
                                uint tid_index[[thread_index_in_threadgroup]],
                                uint3 ugid[[thread_position_in_grid]]) {

)";
  if (shared_memory) {
    code << R"(
  float summa = 0.0f;
  threadgroup FLT4 local_vector[32];
  for (int j = 0; j < $0; ++j) {
    local_vector[tid_index] = j * 32 + tid_index >= params.src_depth ?
      FLT4(0.0f) : vector[j * 32 + tid_index];
    BARRIER(mem_flags::mem_threadgroup);
    for (uint i = 0, counter = j * 32 + tid.y * 8; i < 8; ++i, ++counter) {
      summa += dot(local_vector[tid.y * 8 + i], matrix[counter * params.dst_channels + ugid.x]);
    }
    BARRIER(mem_flags::mem_none);
  }
  )";
  } else {
    code << R"(
  float summa = 0.0f;
  uint counter = ugid.y * $0;
  for (uint i = 0; i < $0; ++i, ++counter) {
    )";
    if (src_depth % 4 != 0) {
      code << "    if (counter >= params.src_depth) continue;" << std::endl;
    }
    code << "    summa += dot(vector[counter], matrix[counter * "
            "params.dst_channels + ugid.x]);"
         << std::endl;
    code << "  }" << std::endl;
  }
  code << R"(

  threadgroup float temp[8][4];
  temp[tid.x][tid.y] = summa;
  BARRIER(mem_flags::mem_threadgroup);
  if (tid.y == 0) {
    summa += temp[tid.x][1];
    summa += temp[tid.x][2];
    summa += temp[tid.x][3];
    temp[tid.x][0] = summa;
  }
  BARRIER(mem_flags::mem_threadgroup);
  if (tid.y == 0 && tid.x % 4 == 0 && ugid.x < params.out_channels) {
    const int linear_index = ugid.x / 4;
    FLT4 value = FLT4(temp[tid.x][0], temp[tid.x + 1][0], temp[tid.x + 2][0], temp[tid.x + 3][0]) +
      biases[linear_index];
    uint3 gid = uint3(1u, 1u, uint(linear_index));
    $$2
    result[linear_index] = value;
  }
}
  )";
  const int src_depth_sub_groups = shared_memory
                                       ? IntegralDivideRoundUp(src_depth, 32)
                                       : IntegralDivideRoundUp(src_depth, 4);
  return absl::Substitute(code.str(), src_depth_sub_groups);
}
}  // namespace

std::vector<ComputeTaskDescriptorPtr> FullyConnected(
    int id, ValueId input_id, ValueId output_id,
    const FullyConnectedAttributes& attr, const RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  auto gpu_type = GetGpuType();
  bool shared = gpu_type == GpuType::kA7 || gpu_type == GpuType::kA8;
  desc->shader_source =
      GetFullyConnectedCode(shared, attr.weights.shape.i, attr.weights.shape.o);

  desc->input_buffers = {
      {input_id, "device FLT4* const vector"},
  };

  desc->output_buffer = {
      output_id, "device FLT4* result",
      [input_id, attr](const std::map<ValueId, BHWC>& buffers) {
        return CalculateOutputShape(buffers.find(input_id)->second, attr);
      }};

  const int src_depth = IntegralDivideRoundUp(attr.weights.shape.i, 4);
  const int src_depth_aligned = AlignByN(src_depth, shared ? 32 : 4);
  const int dst_channels_aligned = AlignByN(attr.weights.shape.o, 8);

  int counter = 0;
  std::vector<float> filters_reordered(dst_channels_aligned *
                                       src_depth_aligned * 4);
  for (int j = 0; j < src_depth_aligned; ++j) {
    for (int i = 0; i < dst_channels_aligned; ++i) {
      for (int k = 0; k < 4; ++k) {
        if (j * 4 + k >= attr.weights.shape.i || i >= attr.weights.shape.o) {
          filters_reordered[counter++] = 0.0f;
        } else {
          const int f_index =
              attr.weights.shape.LinearIndex({i, 0, 0, j * 4 + k});
          filters_reordered[counter++] = attr.weights.data[f_index];
        }
      }
    }
  }

  desc->immutable_buffers = {
      {"device FLT4* const matrix",
       GetByteBufferConverted(filters_reordered, options.storage_precision)},
      {"device FLT4* const biases",
       GetByteBufferConvertedResized(attr.bias.data, options.storage_precision,
                                     attr.weights.shape.o)},
  };

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [attr](const std::map<ValueId, BHWC>& buffers) {
         std::vector<uint32_t> uniform_params{
             static_cast<uint32_t>(
                 IntegralDivideRoundUp(attr.weights.shape.i, 4)),
             static_cast<uint32_t>(AlignByN(attr.weights.shape.o, 8)),
             static_cast<uint32_t>(attr.weights.shape.o),
             static_cast<uint32_t>(0),
         };
         return GetByteBuffer(uniform_params);
       }},
  };

  desc->resize_function = [attr](const std::map<ValueId, BHWC>& buffers) {
    const uint3 groups_size{8, 4, 1};
    const int dst_channels_aligned = AlignByN(attr.weights.shape.o, 8);
    int groups_x = IntegralDivideRoundUp(dst_channels_aligned, groups_size.x);
    return std::make_pair(groups_size, uint3{groups_x, 1, 1});
  };

  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
