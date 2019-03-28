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

#include "tensorflow/lite/delegates/gpu/metal/kernels/depth_wise_convolution.h"

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {

std::vector<ComputeTaskDescriptorPtr> DepthWiseConvolution(
    int id, ValueId input_id, ValueId output_id,
    const DepthwiseConvolution2DAttributes& attr,
    const RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  std::string shader_source = R"(
    #include <metal_stdlib>
    using namespace metal;
    constant int kernel_x = $0;
    constant int kernel_y = $1;
    struct uniforms {
      int4 stride;
      int4 padding;
      int4 dillation;
      int4 size;
      int4 channel_multiplier;
    };
    $$0
    kernel void ComputeFunction(
                                $$1
                                uint tid[[thread_index_in_threadgroup]],
                                uint3 gid[[thread_position_in_grid]]) {
      const bool outside = static_cast<int>(gid.x) >= params.size.z ||
        static_cast<int>(gid.y) >= params.size.w;
      if (outside) {
        return;
      }
      device FLT4* temp = filters + gid.z * kernel_y * kernel_x;
      float4 sum0 = float4(0.0f, 0.0f, 0.0f, 0.0f);

      for(int ky = 0; ky < kernel_y; ++ky) {
        for(int kx = 0; kx < kernel_x; ++kx) {
          int2 coords  = int2(gid.xy) * params.stride.xy + int2(kx, ky) * params.dillation.xy -
            params.padding.xy;
          const bool outside = coords.x < 0 || coords.y < 0 ||
            coords.x >= params.size.x || coords.y >= params.size.y;
          if (outside) continue;

          const int src_layer = gid.z;
          const int src_index = (src_layer * params.size.y + coords.y) * params.size.x + coords.x;
          sum0 += float4(src_buffer[src_index]) * float4(temp[ky * kernel_x + kx]);
        }
      }
      FLT4 res = FLT4(sum0 + float4(biases[gid.z]));
      const int linear_index = (gid.z * params.size.w + int(gid.y)) * params.size.z + int(gid.x);
      FLT4 value = res;
      $$2
      output_buffer[linear_index] = value;
    }
  )";
  desc->shader_source = absl::Substitute(shader_source, attr.weights.shape.w,
                                         attr.weights.shape.h);

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {
      output_id, "device FLT4* output_buffer",
      [input_id, attr](const std::map<ValueId, BHWC>& buffers) {
        auto out_shape =
            CalculateOutputShape(buffers.find(input_id)->second, attr);
        return out_shape;
      }};

  const int num_output_channels = attr.weights.shape.i * attr.weights.shape.o;
  BHWC reordered_dims{1, attr.weights.shape.h, attr.weights.shape.w,
                      num_output_channels};
  std::vector<float> filters_reordered(GetElementsSizeForPHWC4(reordered_dims),
                                       0.0f);
  if (!ConvertToPHWC4(
           absl::MakeConstSpan(attr.weights.data.data(),
                               attr.weights.data.size()),
           reordered_dims,
           absl::MakeSpan(filters_reordered.data(), filters_reordered.size()))
           .ok()) {
    return {};
  }
  auto filters = options.storage_precision == RuntimeOptions::Precision::FP32
                     ? VectorToUint8Vector(filters_reordered)
                     : VectorFloatToHalf(filters_reordered);
  auto biases = options.storage_precision == RuntimeOptions::Precision::FP32
                    ? VectorToUint8Vector(attr.bias.data)
                    : VectorFloatToHalf(attr.bias.data);
  desc->immutable_buffers = {
      {"device FLT4* const filters", filters},
      {"device FLT4* const biases", biases},
  };

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id, output_id, attr](const std::map<ValueId, BHWC>& buffers) {
         const auto& dimension = buffers.find(input_id)->second;
         const auto& output_dimension = buffers.find(output_id)->second;
         std::vector<int> uniform_params{
             attr.strides.w,
             attr.strides.h,
             1,
             1,
             attr.padding.prepended.w,
             attr.padding.prepended.h,
             1,
             1,
             attr.dilations.w,
             attr.dilations.h,
             1,
             1,
             dimension.w,
             dimension.h,
             output_dimension.w,
             output_dimension.h,
             attr.weights.shape.o,
             0,
             0,
             0,
         };
         return VectorToUint8Vector(uniform_params);
       }},
  };

  desc->resize_function = [output_id](const std::map<ValueId, BHWC>& buffers) {
    const auto& dimension = buffers.find(output_id)->second;
    uint3 groups_size{8, 4, 1};
    uint3 groups_count{IntegralDivideRoundUp(dimension.w, groups_size.x),
                       IntegralDivideRoundUp(dimension.h, groups_size.y),
                       IntegralDivideRoundUp(dimension.c, 4)};
    return std::make_pair(groups_size, groups_count);
  };

  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
