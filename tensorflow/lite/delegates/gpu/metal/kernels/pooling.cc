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

#include "tensorflow/lite/delegates/gpu/metal/kernels/pooling.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/util.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

std::string GetMaxPoolingCode(const HW& kernel_size) {
  std::string shader_source = R"(
    #include <metal_stdlib>
    using namespace metal;
    constant int window_w = $0;
    constant int window_h = $1;
    struct uniforms {
      int4 src_size;
      int4 dst_size;
      int2 stride;
      int2 offset;
    };

    $$0
    kernel void ComputeFunction(
                                $$1
                                uint3 gid[[thread_position_in_grid]]) {
      if (static_cast<int>(gid.x) >= params.dst_size.x ||
          static_cast<int>(gid.y) >= params.dst_size.y ||
          static_cast<int>(gid.z) >= params.dst_size.z) {
        return;
      }

      FLT4 maximum = FLT4(-10000.0);
      for (int a = 0; a < window_h; ++a) {
        for (int b = 0; b < window_w; ++b) {
          const int2 coords = int2(gid.xy) * params.stride - params.offset + int2(b, a);
          bool outside = coords.x < 0 || coords.y < 0 || coords.x >= params.src_size.x ||
            coords.y >= params.src_size.y;
          const int buffer_index = (gid.z * params.src_size.y + coords.y) *
            params.src_size.x + coords.x;
          FLT4 src_color = outside ? FLT4(-10000.0) : src_buffer[buffer_index];
          maximum = max(maximum, src_color);
        }
      }
      const int linear_index = (gid.z * params.dst_size.y + int(gid.y)) * params.dst_size.x +
        int(gid.x);
      FLT4 value = maximum;
      $$2
      output_buffer[linear_index] = value;
    }
  )";
  return absl::Substitute(shader_source, kernel_size.w, kernel_size.h);
}

std::string GetMaxPoolingIndicesCode(const HW& kernel_size) {
  std::string shader_source = R"(
    #include <metal_stdlib>
    using namespace metal;
    constant int window_w = $0;
    constant int window_h = $1;
    struct uniforms {
      int4 src_size;
      int4 dst_size;
      int2 stride;
      int2 offset;
    };

    $$0
    kernel void ComputeFunction(
                                $$1
                                uint3 gid[[thread_position_in_grid]]) {
      if (static_cast<int>(gid.x) >= params.dst_size.x ||
          static_cast<int>(gid.y) >= params.dst_size.y ||
          static_cast<int>(gid.z) >= params.dst_size.z) {
        return;
      }

      FLT4 maximum = FLT4(-10000.0);
      ushort4 indexes = ushort4(0);
      ushort index_counter = 0;
      for (int a = 0; a < window_h; ++a) {
        for (int b = 0; b < window_w; ++b) {
          const int2 coords = int2(gid.xy) * params.stride - params.offset + int2(b, a);
          bool outside = coords.x < 0 || coords.y < 0 || coords.x >= params.src_size.x ||
            coords.y >= params.src_size.y;
          const int buffer_index = (gid.z * params.src_size.y + coords.y) *
            params.src_size.x + coords.x;
          FLT4 src_color = outside ? FLT4(-10000.0) : src_buffer[buffer_index];
          if (src_color.x > maximum.x) {
            indexes.x = index_counter;
            maximum.x = src_color.x;
          }
          if (src_color.y > maximum.y) {
            indexes.y = index_counter;
            maximum.y = src_color.y;
          }
          if (src_color.z > maximum.z) {
            indexes.z = index_counter;
            maximum.z = src_color.z;
          }
          if (src_color.w > maximum.w) {
            indexes.w = index_counter;
            maximum.w = src_color.w;
          }
          index_counter++;
        }
      }
      const int linear_index = (gid.z * params.dst_size.y + int(gid.y)) * params.dst_size.x +
        int(gid.x);
      FLT4 value = static_cast<FLT4>(indexes) + FLT4(0.1);
      $$2
      output_buffer[linear_index] = value;
    }
  )";
  return absl::Substitute(shader_source, kernel_size.w, kernel_size.h);
}

std::string GetAveragePoolingCode(const HW& kernel_size) {
  std::string shader_source = R"(
  #include <metal_stdlib>
  using namespace metal;
  constant int window_w = $0;
  constant int window_h = $1;
  constant float multiplier = $2;
  struct uniforms {
    int4 src_size;
    int4 dst_size;
    int2 stride;
    int2 offset;
  };
  $$0
  kernel void ComputeFunction(
                              $$1
                              uint tid[[thread_index_in_threadgroup]],
                              uint3 gid[[thread_position_in_grid]]) {
    if (static_cast<int>(gid.x) >= params.dst_size.x ||
        static_cast<int>(gid.y) >= params.dst_size.y ||
        static_cast<int>(gid.z) >= params.dst_size.z) {
      return;
    }

    float4 sum = float4(0.0f);
    for (int a = 0; a < window_h; ++a) {
      for (int b = 0; b < window_w; ++b) {
        const int2 coords = int2(gid.xy) * params.stride - params.offset + int2(b, a);
        bool outside = coords.x < 0 || coords.y < 0 || coords.x >= params.src_size.x ||
          coords.y >= params.src_size.y;
        const int buffer_index = (gid.z * params.src_size.y + coords.y) *
          params.src_size.x + coords.x;
        const float4 src_color = outside ? float4(0.0f) : float4(src_buffer[buffer_index]);
        sum += src_color;
      }
    }
    const int linear_index = (gid.z * params.dst_size.y + int(gid.y)) * params.dst_size.x +
      int(gid.x);
    FLT4 value = FLT4(sum * multiplier);
    $$2
    output_buffer[linear_index] = value;
  }
)";
  float multiplier = 1.0f / static_cast<float>(kernel_size.w * kernel_size.h);
  return absl::Substitute(shader_source, kernel_size.w, kernel_size.h,
                          multiplier);
}

ComputeTaskDescriptorPtr PoolingInternal(int id, ValueId input_id,
                                         ValueId output_id,
                                         const Pooling2DAttributes& params,
                                         bool generate_indices) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  if (params.type == PoolingType::MAX) {
    desc->shader_source = generate_indices
                              ? GetMaxPoolingIndicesCode(params.kernel)
                              : GetMaxPoolingCode(params.kernel);
  } else if (params.type == PoolingType::AVERAGE) {
    desc->shader_source = GetAveragePoolingCode(params.kernel);
  }

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {
      output_id, "device FLT4* output_buffer",
      [input_id, params](const std::map<ValueId, BHWC>& buffers) {
        return CalculateOutputShape(buffers.find(input_id)->second, params);
      }};

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id, output_id, params](const std::map<ValueId, BHWC>& buffers) {
         const auto& dimension = buffers.find(input_id)->second;
         const auto& output_dimension = buffers.find(output_id)->second;
         std::vector<int> uniform_params = {
             dimension.w,
             dimension.h,
             IntegralDivideRoundUp(dimension.c, 4),
             dimension.w * dimension.h,
             output_dimension.w,
             output_dimension.h,
             IntegralDivideRoundUp(dimension.c, 4),
             output_dimension.w * output_dimension.h,
             params.strides.w,
             params.strides.h,
             params.padding.prepended.w,
             params.padding.prepended.h,
         };
         return VectorToUint8Vector(uniform_params);
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

  return desc;
}

}  // namespace

std::vector<ComputeTaskDescriptorPtr> Pooling(
    int id, ValueId input_id, const std::vector<ValueId>& output_ids,
    const Pooling2DAttributes& params) {
  std::vector<ComputeTaskDescriptorPtr> descriptors;
  descriptors.push_back(
      PoolingInternal(id, input_id, output_ids[0], params, false));
  if (params.type == PoolingType::MAX && params.output_indices) {
    descriptors.push_back(
        PoolingInternal(id, input_id, output_ids[1], params, true));
  }
  return descriptors;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
