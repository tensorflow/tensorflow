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

#include "tensorflow/lite/delegates/gpu/metal/kernels/convolution_generic.h"

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

int GetNumOutputSlices(int dst_channels) {
  const int dst_depth = IntegralDivideRoundUp(dst_channels, 4);
  if (dst_depth % 4 == 0) {
    return 4;
  } else if (dst_depth % 3 == 0) {
    return 3;
  } else if (dst_depth % 2 == 0) {
    return 2;
  } else {
    return 1;
  }
}

std::string GetKernelCode(const Convolution2DAttributes& params, int z_out) {
  std::string code;
  code.reserve(16 * 1024);  // Reserve large enough buffer.
  std::string channels[4] = {"x", "y", "z", "w"};
  code += R"(
#include <metal_stdlib>
using namespace metal;

struct uniforms {
  int4 src_size;
  int4 dst_size;
  int4 stride_padding;
  int4 kernel_dilation;
  uint4 work_group_size;
};
$0

kernel void ComputeFunction(
                            $1
                            uint3 group_id[[threadgroup_position_in_grid]],
                            uint3 tid3d[[thread_position_in_threadgroup]])
{
  int gid_x = group_id.y * params.work_group_size.x + tid3d.x;
  int gid_y = (group_id.z * params.work_group_size.y + tid3d.y) * 2;
  )";
  code += "  int gid_z = (group_id.x * params.work_group_size.z + tid3d.z) * " +
          std::to_string(z_out) + "u;\n";
  for (int i = 0; i < z_out; ++i) {
    const std::string s_i = std::to_string(i);
    code += "  float4 r" + s_i + " = float4(0.0f, 0.0f, 0.0f, 0.0f);\n";
    code += "  float4 l" + s_i + " = float4(0.0f, 0.0f, 0.0f, 0.0f);\n";
  }
  code += R"(
  device FLT4* tmp = filters + gid_z * 4 * params.src_size.w * params.kernel_dilation.x * params.kernel_dilation.y;

  int y0 = gid_y * params.stride_padding.y + params.stride_padding.w;
  int y1 = (gid_y + 1) * params.stride_padding.y + params.stride_padding.w;
  int x0 = gid_x * params.stride_padding.x + params.stride_padding.z;
  
  int y = 0;
  do {
    int coord_y0 = y * params.kernel_dilation.w + y0;
    int coord_y1 = y * params.kernel_dilation.w + y1;
    bool y0_out = coord_y0 < 0 || coord_y0 >= params.src_size.y;
    bool y1_out = coord_y1 < 0 || coord_y1 >= params.src_size.y;
    coord_y0 = clamp(coord_y0, 0, params.src_size.y - 1);
    coord_y1 = clamp(coord_y1, 0, params.src_size.y - 1);
    int x = 0;
    do {
      int coord_x0 = x * params.kernel_dilation.z + x0;
      bool x0_out = coord_x0 < 0 || coord_x0 >= params.src_size.x;
      coord_x0 = clamp(coord_x0, 0, params.src_size.x - 1);
      FLT m0 = !(y0_out || x0_out);
      FLT m1 = !(y1_out || x0_out);
      int s = 0;
      device FLT4* src_loc_0 = src_buffer + coord_y0 * params.src_size.x + coord_x0;
      device FLT4* src_loc_1 = src_buffer + coord_y1 * params.src_size.x + coord_x0;
      do {
        FLT4 src_0 = *src_loc_0 * m0;
        FLT4 src_1 = *src_loc_1 * m1;
        src_loc_0 += params.src_size.z;
        src_loc_1 += params.src_size.z;
    )";
  for (int i = 0; i < z_out * 4; ++i) {
    const std::string s_i = std::to_string(i);
    code += "        r" + std::to_string(i / 4) + "." + channels[i % 4] +
            " += dot(tmp[" + s_i + "], src_0);\n";
    code += "        l" + std::to_string(i / 4) + "." + channels[i % 4] +
            " += dot(tmp[" + s_i + "], src_1);\n";
  }

  code += "        tmp += " + std::to_string(z_out * 4) + ";\n";
  code += R"(
        s += 1;
      } while (s < params.src_size.w);
      x++;
    } while (x < params.kernel_dilation.x);
    y++;
  } while (y < params.kernel_dilation.y);
  const int offset_0 = gid_z * params.dst_size.z + gid_y * params.dst_size.x + gid_x;
  const int offset_1 = offset_0 + params.dst_size.x;
  bool p0_in = gid_x < params.dst_size.x && gid_y < params.dst_size.y;
  bool p1_in = gid_x < params.dst_size.x && gid_y + 1 < params.dst_size.y;

  device FLT4* bias_loc = biases + gid_z;
  )";
  for (int i = 0; i < z_out; ++i) {
    const std::string s_i = std::to_string(i);
    code += "  r" + s_i + " += float4(bias_loc[" + s_i + "]);\n";
    code += "  l" + s_i + " += float4(bias_loc[" + s_i + "]);\n";
  }
  code += R"(
  if (gid_x >= params.dst_size.x || gid_y >= params.dst_size.y) {
      return;
  }
  )";
  for (int i = 0; i < z_out; ++i) {
    const std::string s_i = std::to_string(i);
    code += "  if (gid_z + " + s_i + "< params.dst_size.w) {\n";
    code += "    if (p0_in) {\n";
    code += "      FLT4 value = FLT4(r" + s_i + ");\n";
    code += "      int linear_index = offset_0 + params.dst_size.z * " + s_i +
            ";\n";
    code += "      uint3 gid = uint3(gid_x, gid_y, gid_z + " + s_i + ");\n";
    code += "      $2\n";
    code += "      dst_buffer[linear_index] = value;\n";
    code += "    }\n";
    code += "    if (p1_in) {\n";
    code += "      FLT4 value = FLT4(l" + s_i + ");\n";
    code += "      int linear_index = offset_1 + params.dst_size.z * " + s_i +
            ";\n";
    code += "      uint3 gid = uint3(gid_x, gid_y + 1, gid_z + " + s_i + ");\n";
    code += "      $2\n";
    code += "      dst_buffer[linear_index] = value;\n";
    code += "    }\n";
    code += "  }\n";
  }
  code += "  }\n";
  return code;
}

// Reorder weights to make the weights memory access pattern cache friendly for
// ConvolutionGeneric
std::vector<float> ReorderWeights(const Convolution2DAttributes& params,
                                  int z_out) {
  const int dst_depth = IntegralDivideRoundUp(params.weights.shape.o, 4);
  const int src_depth = IntegralDivideRoundUp(params.weights.shape.i, 4);
  std::vector<float> weights_reordered(params.weights.shape.w *
                                       params.weights.shape.h * dst_depth * 4 *
                                       src_depth * 4);
  int counter = 0;
  for (int d = 0; d < IntegralDivideRoundUp(dst_depth, z_out); ++d) {
    for (int y = 0; y < params.weights.shape.h; ++y) {
      for (int x = 0; x < params.weights.shape.w; ++x) {
        for (int s = 0; s < src_depth; ++s) {
          for (int k = 0; k < z_out; ++k) {
            for (int j = 0; j < 4; ++j) {
              for (int i = 0; i < 4; ++i) {
                int src_ch = s * 4 + i;
                int dst_ch = (d * z_out + k) * 4 + j;
                if (src_ch >= params.weights.shape.i ||
                    dst_ch >= params.weights.shape.o) {
                  weights_reordered[counter++] = 0.0f;
                } else {
                  const int f_index =
                      params.weights.shape.LinearIndex({dst_ch, y, x, src_ch});
                  weights_reordered[counter++] = params.weights.data[f_index];
                }
              }
            }
          }
        }
      }
    }
  }
  return weights_reordered;
}
}  // namespace

static std::vector<uint8_t> GetUniformBuffer(
    const BHWC& src_size, const BHWC& dst_size,
    const Convolution2DAttributes& params) {
  std::vector<int> uniform_params = {
      src_size.w,
      src_size.h,
      src_size.w * src_size.h,
      IntegralDivideRoundUp(src_size.c, 4),
      dst_size.w,
      dst_size.h,
      dst_size.w * dst_size.h,
      IntegralDivideRoundUp(dst_size.c, 4),
      params.strides.w,
      params.strides.h,
      -params.padding.prepended.w,
      -params.padding.prepended.h,
      params.weights.shape.w,
      params.weights.shape.h,
      params.dilations.w,
      params.dilations.h,
      8u,
      4u,
      1u,
      1u,  // dummy, for alignment
  };
  return VectorToUint8Vector(uniform_params);
}

std::vector<ComputeTaskDescriptorPtr> ConvolutionGeneric(
    int id, ValueId input_id, ValueId output_id,
    const Convolution2DAttributes& params,
    const metal::RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  const int z_out = GetNumOutputSlices(params.weights.shape.o);
  desc->shader_source = GetKernelCode(params, z_out);

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {
      output_id, "device FLT4* dst_buffer",
      [input_id, params](const std::map<ValueId, BHWC>& buffers) {
        auto out_shape =
            CalculateOutputShape(buffers.find(input_id)->second, params);
        return out_shape;
      }};

  auto weights_reordered = ReorderWeights(params, z_out);
  auto weights =
      options.storage_precision == metal::RuntimeOptions::Precision::FP32
          ? VectorToUint8Vector(weights_reordered)
          : VectorFloatToHalf(weights_reordered);
  auto biases =
      options.storage_precision == metal::RuntimeOptions::Precision::FP32
          ? VectorToUint8Vector(params.bias.data)
          : VectorFloatToHalf(params.bias.data);
  desc->immutable_buffers = {
      {"device FLT4* const filters", weights},
      {"device FLT4* const biases", biases},
  };

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id, output_id, params](const std::map<ValueId, BHWC>& buffers) {
         const auto& input_dimensions = buffers.find(input_id)->second;
         const auto& output_dimensions = buffers.find(output_id)->second;
         return GetUniformBuffer(input_dimensions, output_dimensions, params);
       }},
  };

  desc->resize_function = [output_id,
                           params](const std::map<ValueId, BHWC>& buffers) {
    const auto& output_dims = buffers.find(output_id)->second;
    const int dst_depth = IntegralDivideRoundUp(params.weights.shape.o, 4);
    const uint3 group_size{8, 4, 1};
    int groups_x = IntegralDivideRoundUp(output_dims.w, group_size.x);
    int groups_y = IntegralDivideRoundUp(
        IntegralDivideRoundUp(output_dims.h, 2), group_size.y);
    const int z_out = GetNumOutputSlices(params.weights.shape.o);
    int groups_z = IntegralDivideRoundUp(
        IntegralDivideRoundUp(dst_depth, z_out), group_size.z);
    return std::make_pair(group_size, uint3{groups_z, groups_x, groups_y});
  };

  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
