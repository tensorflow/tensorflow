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

#include "tensorflow/lite/delegates/gpu/metal/kernels/conv.h"

#include <cmath>
#include <cstdint>
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
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

int GetNumOutputSlices(int dst_channels) {
  const int dst_depth = IntegralDivideRoundUp(dst_channels, 4);
  if (dst_depth % 4 == 0) {
    return 4;
  } else if (dst_depth % 2 == 0) {
    return 2;
  } else {
    return 1;
  }
}

int GetSrcBatchSize(int dst_channels) {
  const int dst_depth = IntegralDivideRoundUp(dst_channels, 4);
  if (dst_depth % 4 == 0) {
    return 2;
  } else if (dst_depth % 2 == 0) {
    return 4;
  } else {
    return 8;
  }
}

std::string GetValuesDeclarationPart(int num_output_slices, bool is_1x1) {
  std::string code;
  for (int d = 0; d < num_output_slices; ++d) {
    code += absl::Substitute(R"(
    float4 sum$0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    )",
                             d);
  }
  if (is_1x1) {
    code += absl::Substitute(R"(
    threadgroup FLT4 temp[32];
    device FLT4* f_offseted = weights + (gid.z + params.z_offset.x) * $0 * src_offset;
    )",
                             num_output_slices * 4);
  } else {
    code += absl::Substitute(R"(
    threadgroup FLT4 temp[32];
    device FLT4* f_offseted = weights + (gid.z + params.z_offset.x) * $0 * src_offset *
    kernel_y * kernel_x;
         )",
                             num_output_slices * 4);
  }
  return code;
}

std::string GetLocalMemoryUploadPart() {
  std::string code = R"(
    BARRIER(mem_flags::mem_none);
    temp[tid] = f_offseted[tid];
    f_offseted += 32;
    BARRIER(mem_flags::mem_threadgroup);
    )";
  return code;
}

std::string GetSummationPart(int num_output_slices, int index) {
  std::string code = R"(
      {
        const FLT4 src = src_buffer[src_adress];
        src_adress += params.dillation_layer_offsets.z;
    )";
  for (int d = 0; d < num_output_slices; ++d) {
    code += absl::Substitute(R"(
        sum$6.x += dot(temp[$0 * $1 + $2], src) * multiplier;
        sum$6.y += dot(temp[$0 * $1 + $3], src) * multiplier;
        sum$6.z += dot(temp[$0 * $1 + $4], src) * multiplier;
        sum$6.w += dot(temp[$0 * $1 + $5], src) * multiplier;
      )",
                             index, num_output_slices * 4, d * 4 + 0, d * 4 + 1,
                             d * 4 + 2, d * 4 + 3, d);
  }
  code += "}";
  return code;
}

std::string GetBiasReadingPart(int num_output_slices) {
  std::string code = absl::Substitute(R"(
     {
         gid.z = (gid.z + params.z_offset.x) * $0;
         BARRIER(mem_flags::mem_none);
         if (tid < $0) {
             temp[tid] = biases[gid.z + tid];
         }
         BARRIER(mem_flags::mem_threadgroup);
         if (outside) {
             return;
         }
     })",
                                      num_output_slices);
  return code;
}

std::string GetWritingPart(int num_output_slices) {
  std::string code;
  for (int d = 0; d < num_output_slices; ++d) {
    code += absl::Substitute(R"(
     {
         int dst_adress = int(gid.y) * params.size.z + int(gid.x);
         FLT4 value = FLT4(sum$0) + temp[$0];
         const int linear_index = gid.z * params.dillation_layer_offsets.w + dst_adress;
         $$2
         dst_buffer[linear_index + params.z_offset.y] = value;
         gid.z += 1;
     })",
                             d);
  }
  return code;
}

std::string GetKernelForConv(const Convolution2DAttributes& params) {
  const int num_output_slices = GetNumOutputSlices(params.weights.shape.o);
  std::string code;
  code.reserve(16 * 1024);  // Reserve large enough buffer.
  const bool is_1x1 =
      params.weights.shape.w == 1 && params.weights.shape.h == 1;
  const bool is_strided = params.strides.w > 1 || params.strides.h > 1;
  const int src_group_size = GetSrcBatchSize(params.weights.shape.o);

  const int src_depth = IntegralDivideRoundUp(params.weights.shape.i, 4);
  const int src_groups = src_depth / src_group_size;
  const int src_depth_aligned = AlignByN(src_depth, src_group_size);
  const int reminder_src_depth = src_depth - src_groups * src_group_size;

  code = absl::Substitute(R"(
    #include <metal_stdlib>
    using namespace metal;
    constant int src_depth_groups = $0;
    constant int src_offset = $1;
    constant int kernel_x = $2;
    constant int kernel_y = $3;
    struct uniforms {
      int4 stride_padding;
      int4 dillation_layer_offsets;
      int4 size;
      int4 z_offset;
    };
    $$0
    kernel void ComputeFunction(
                                $$1
                                uint tid[[thread_index_in_threadgroup]],
                                uint3 gid[[thread_position_in_grid]])
      {
        const bool outside = static_cast<int>(gid.x) >= params.size.z ||
          static_cast<int>(gid.y) >= params.size.w;
  )",
                          src_groups, src_depth_aligned, params.weights.shape.w,
                          params.weights.shape.h);
  code += GetValuesDeclarationPart(num_output_slices, is_1x1);

  if (!is_1x1) {
    code += R"(
      for(int ky = 0; ky < kernel_y; ++ky) {
        for(int kx = 0; kx < kernel_x; ++kx) {
          int2 coords = int2(gid.xy) * params.stride_padding.xy + int2(kx, ky) *
            params.dillation_layer_offsets.xy - params.stride_padding.zw;
          const bool el_outside = coords.x < 0 || coords.y < 0 || coords.x >= params.size.x ||
            coords.y >= params.size.y;
          const FLT multiplier = el_outside ? 0.0f : 1.0f;
    )";
  } else {
    code += "const FLT multiplier = 1.0f;\n";
    code += "int2 coords = int2(gid.xy)";
    if (is_strided) {
      code += " * params.stride_padding.xy";
    }
    code += ";\n";
  }
  code += R"(
    coords = clamp(coords, int2(0, 0), int2(params.size.x - 1, params.size.y - 1));
    int src_adress = coords.y * params.size.x + coords.x;
    for(int s = 0; s < src_depth_groups; ++s) {
  )";
  code += GetLocalMemoryUploadPart();
  for (int sub_s = 0; sub_s < src_group_size; ++sub_s) {
    code += GetSummationPart(num_output_slices, sub_s);
  }
  code += R"(
    }
  )";
  if (reminder_src_depth != 0) {
    code += GetLocalMemoryUploadPart();
    for (int sub_s = 0; sub_s < reminder_src_depth; ++sub_s) {
      code += GetSummationPart(num_output_slices, sub_s);
    }
  }
  if (!is_1x1) {
    code += R"(
        }
      }
    )";
  }
  code += GetBiasReadingPart(num_output_slices);
  code += GetWritingPart(num_output_slices);
  code += "  }";
  return code;
}

// Reorder weights to make the weights memory access pattern cache friendly for
// GPU
std::vector<float> ReorderWeightsForConvShared(
    const Convolution2DAttributes& params) {
  const int dst_batch_size = GetNumOutputSlices(params.weights.shape.o) * 4;
  const int src_batch_size = GetSrcBatchSize(params.weights.shape.o);
  BHWC input_dimensions{params.weights.shape.o, params.weights.shape.h,
                        params.weights.shape.w, params.weights.shape.i};
  const int gpu_simd_size = dst_batch_size * src_batch_size;
  const int weights_width = AlignByN(input_dimensions.c, gpu_simd_size);
  const int weights_height = AlignByN(input_dimensions.b, dst_batch_size);
  const int weights_channels = params.weights.shape.w * params.weights.shape.h;
  const int weights_aligned_size =
      weights_width * weights_height * weights_channels;
  std::vector<float> weights_reordered(weights_aligned_size);
  float* destination = weights_reordered.data();
  const int dst_groups =
      IntegralDivideRoundUp(input_dimensions.b, dst_batch_size);
  const int src_sub_groups =
      IntegralDivideRoundUp(input_dimensions.c, 4 * src_batch_size);
  for (int group = 0; group < dst_groups; ++group) {
    for (int y = 0; y < params.weights.shape.h; ++y) {
      for (int x = 0; x < params.weights.shape.w; ++x) {
        for (int sub_group = 0; sub_group < src_sub_groups; ++sub_group) {
          for (int s = 0; s < src_batch_size; ++s) {
            for (int d = 0; d < dst_batch_size; ++d) {
              int output_index = group * dst_batch_size + d;
              for (int i = 0; i < 4; ++i) {
                int input_index = (sub_group * src_batch_size + s) * 4 + i;
                if (input_index >= input_dimensions.c ||
                    output_index >= input_dimensions.b) {
                  // Padding with zero
                  *destination++ = 0.0f;
                } else {
                  int linear_index =
                      input_index +
                      input_dimensions.c *
                          (x + input_dimensions.w *
                                   (y + input_dimensions.h * output_index));
                  *destination++ = params.weights.data[linear_index];
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

std::vector<uint8_t> GetUniformBufferForConvShared(
    const BHWC& input_dimensions, const BHWC& output_dimensions,
    const Convolution2DAttributes& params) {
  std::vector<int> uniform_params = {
      params.strides.w,
      params.strides.h,
      params.padding.prepended.w,
      params.padding.prepended.h,
      params.dilations.w,
      params.dilations.h,
      input_dimensions.w * input_dimensions.h,
      output_dimensions.w * output_dimensions.h,
      input_dimensions.w,
      input_dimensions.h,
      output_dimensions.w,
      output_dimensions.h,
      // TODO(chirkov): use z_offset for concat table optimization
      /*z_offset.x=*/0,
      /*z_offset.y=*/0,
      /*z_offset.z=*/0,
      /*z_offset.w=*/0,
  };
  return VectorToUint8Vector(uniform_params);
}

std::string GetKernelForConv1x1(const Convolution2DAttributes& params,
                                int z_out) {
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
  int gid_y = (group_id.z * params.work_group_size.y + tid3d.y) << 1u;
  )";
  code += "  int gid_z = (group_id.x * params.work_group_size.z + tid3d.z) * " +
          std::to_string(z_out) + "u;\n";
  for (int i = 0; i < z_out; ++i) {
    const std::string s_i = std::to_string(i);
    code += "  ACCUM_FLT4 r" + s_i + " = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);\n";
    code += "  ACCUM_FLT4 l" + s_i + " = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);\n";
  }
  code += R"(
  device FLT4* tmp = filters + gid_z * 4 * params.src_size.w;

  int y0 = clamp(gid_y, 0, params.src_size.y - 1);
  int y1 = clamp(gid_y + 1, 0, params.src_size.y - 1);
  int x0 = clamp(gid_x, 0, params.src_size.x - 1);

  int s = 0;

  device FLT4* src_loc_0 = src_buffer + y0 * params.src_size.x + x0;
  device FLT4* src_loc_1 = src_buffer + y1 * params.src_size.x + x0;
  do {
    FLT4 src_0 = *src_loc_0;
    FLT4 src_1 = *src_loc_1;
    src_loc_0 += params.src_size.z;
    src_loc_1 += params.src_size.z;
    )";
  for (int i = 0; i < z_out * 4; ++i) {
    const std::string s_i = std::to_string(i);
    code += "    r" + std::to_string(i / 4) + "." + channels[i % 4] +
            " += dot(tmp[" + s_i + "], src_0);\n";
    code += "    l" + std::to_string(i / 4) + "." + channels[i % 4] +
            " += dot(tmp[" + s_i + "], src_1);\n";
  }

  code += "    tmp += " + std::to_string(z_out * 4) + ";\n";
  code += R"(
    s += 1;
  } while (s < params.src_size.w);
  const int offset_0 = gid_z * params.dst_size.z + gid_y * params.dst_size.x + gid_x;
  const int offset_1 = offset_0 + params.dst_size.x;
  bool y0_in = gid_y < params.dst_size.y;
  bool y1_in = gid_y + 1 < params.dst_size.y;

  device FLT4* bias_loc = biases + gid_z;
  )";
  for (int i = 0; i < z_out; ++i) {
    const std::string s_i = std::to_string(i);
    code += "  r" + s_i + " += TO_ACCUM4_TYPE(bias_loc[" + s_i + "]);\n";
    code += "  l" + s_i + " += TO_ACCUM4_TYPE(bias_loc[" + s_i + "]);\n";
  }
  code += R"(
  if (gid_x >= params.dst_size.x || gid_y >= params.dst_size.y) {
      return;
  }
  )";
  for (int i = 0; i < z_out; ++i) {
    const std::string s_i = std::to_string(i);
    code += "  if (gid_z + " + s_i + "< params.dst_size.w) {\n";
    code += "    if (y0_in) {\n";
    code += "      FLT4 value = FLT4(r" + s_i + ");\n";
    code += "      int linear_index = offset_0 + params.dst_size.z * " + s_i +
            ";\n";
    code += "      uint3 gid = uint3(gid_x, gid_y, gid_z + " + s_i + ");\n";
    code += "      $2\n";
    code += "      dst_buffer[linear_index] = value;\n";
    code += "    }\n";
    code += "    if (y1_in) {\n";
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

std::string GetKernelForConvGeneric(const Convolution2DAttributes& params,
                                    int z_out) {
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
    code += "  ACCUM_FLT4 r" + s_i + " = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);\n";
    code += "  ACCUM_FLT4 l" + s_i + " = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);\n";
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
    code += "  r" + s_i + " += TO_ACCUM4_TYPE(bias_loc[" + s_i + "]);\n";
    code += "  l" + s_i + " += TO_ACCUM4_TYPE(bias_loc[" + s_i + "]);\n";
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

std::string GetKernelForConvPrecise(int z_out) {
  std::string channels[4] = {"x", "y", "z", "w"};
  std::string code;
  code.reserve(16 * 1024);  // Reserve large enough buffer.
  code += R"(
#include <metal_stdlib>
using namespace metal;

struct uniforms {
    int4 src_size;
    int4 dst_size;
    int4 stride_padding;
    int4 kernel_dilation;
    int4 slices;
};
$0

kernel void ComputeFunction(
                            $1
                            uint3 ugid[[thread_position_in_grid]])
{
    int linear_id = ugid.x;
    int gid_z = linear_id / params.slices.y;
    int linear_xy = (linear_id - gid_z * params.slices.y) << 1;
    )";
  code += "    gid_z *= " + std::to_string(z_out) + ";\n";
  code += R"(
    int gid_y0 = linear_xy / params.slices.x;
    int gid_x0 = linear_xy - gid_y0 * params.slices.x;
    linear_xy += 1;
    int gid_y1 = linear_xy / params.slices.x;
    int gid_x1 = linear_xy - gid_y1 * params.slices.x;

    if (gid_z >= params.dst_size.w) return;
    )";
  for (int i = 0; i < z_out; ++i) {
    const std::string s_i = std::to_string(i);
    code += "  ACCUM_FLT4 r" + s_i + " = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);\n";
    code += "  ACCUM_FLT4 l" + s_i + " = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);\n";
  }
  code += R"(
    device FLT4* tmp = filters + gid_z * 4 * params.src_size.w *
                       params.kernel_dilation.x * params.kernel_dilation.y;

    int y0 = gid_y0 * params.stride_padding.y + params.stride_padding.w;
    int y1 = gid_y1 * params.stride_padding.y + params.stride_padding.w;
    int x0 = gid_x0 * params.stride_padding.x + params.stride_padding.z;
    int x1 = gid_x1 * params.stride_padding.x + params.stride_padding.z;
)";
  code += R"(
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
    int coord_x1 = x * params.kernel_dilation.z + x1;
    bool x0_out = coord_x0 < 0 || coord_x0 >= params.src_size.x;
    bool x1_out = coord_x1 < 0 || coord_x1 >= params.src_size.x;
    coord_x0 = clamp(coord_x0, 0, params.src_size.x - 1);
    coord_x1 = clamp(coord_x1, 0, params.src_size.x - 1);
    FLT m0 = !(y0_out || x0_out);
    FLT m1 = !(y1_out || x1_out);
    device FLT4* src_loc_0 = src_buffer + coord_y0 * params.src_size.x + coord_x0;
    device FLT4* src_loc_1 = src_buffer + coord_y1 * params.src_size.x + coord_x1;
    int s = 0;
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
  const int offset_0 = gid_z * params.dst_size.z + gid_y0 * params.dst_size.x + gid_x0;
  const int offset_1 = gid_z * params.dst_size.z + gid_y1 * params.dst_size.x + gid_x1;
  bool p0_in = gid_x0 < params.dst_size.x && gid_y0 < params.dst_size.y;
  bool p1_in = gid_x1 < params.dst_size.x && gid_y1 < params.dst_size.y;

  device FLT4* bias_loc = biases + gid_z;
  )";
  for (int i = 0; i < z_out; ++i) {
    const std::string s_i = std::to_string(i);
    code += "  r" + s_i + " += TO_ACCUM4_TYPE(bias_loc[" + s_i + "]);\n";
    code += "  l" + s_i + " += TO_ACCUM4_TYPE(bias_loc[" + s_i + "]);\n";
  }
  for (int i = 0; i < z_out; ++i) {
    const std::string s_i = std::to_string(i);
    code += "  if (gid_z + " + s_i + "< params.dst_size.w) {\n";
    code += "    if (p0_in) {\n";
    code += "      FLT4 value = FLT4(r" + s_i + ");\n";
    code += "      int linear_index = offset_0 + params.dst_size.z * " + s_i +
            ";\n";
    code += "      uint3 gid = uint3(gid_x0, gid_y0, gid_z + " + s_i + ");\n";
    code += "      $2\n";
    code += "      dst_buffer[linear_index] = value;\n";
    code += "    }\n";
    code += "    if (p1_in) {\n";
    code += "      FLT4 value = FLT4(l" + s_i + ");\n";
    code += "      int linear_index = offset_1 + params.dst_size.z * " + s_i +
            ";\n";
    code += "      uint3 gid = uint3(gid_x1, gid_y1, gid_z + " + s_i + ");\n";
    code += "      $2\n";
    code += "      dst_buffer[linear_index] = value;\n";
    code += "    }\n";
    code += "  }\n";
  }
  code += "  }\n";
  return code;
}

// Reorder weights to make the weights memory access pattern cache friendly for
// Convolution1x1/ConvolutionGeneric
std::vector<float> ReorderWeightsForConv(const Convolution2DAttributes& params,
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

uint3 GetWorkGroupForConv() { return {8, 4, 1}; }
uint3 GetWorkGroupForConvPrecise() { return {32, 1, 1}; }

std::vector<uint8_t> GetUniformBufferForConv(
    const BHWC& src_size, const BHWC& dst_size,
    const Convolution2DAttributes& params) {
  const int3 group_size = GetWorkGroupForConv();
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
      group_size.x,
      group_size.y,
      group_size.z,
      1u,  // dummy, for alignment
  };
  return VectorToUint8Vector(uniform_params);
}

std::vector<uint8_t> GetUniformBufferForConvPrecise(
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
      dst_size.w,
      IntegralDivideRoundUp(dst_size.w * dst_size.h, 2),
      0u,  // dummy, for alignment
      0u,  // dummy, for alignment
  };
  return VectorToUint8Vector(uniform_params);
}

uint3 GetGroupsCountForConv(const uint3& group_size, const BHWC& dst_shape) {
  const int dst_depth = IntegralDivideRoundUp(dst_shape.c, 4);
  int groups_x = IntegralDivideRoundUp(dst_shape.w, group_size.x);
  int groups_y = IntegralDivideRoundUp(IntegralDivideRoundUp(dst_shape.h, 2),
                                       group_size.y);
  const int z_out = GetNumOutputSlices(dst_shape.c);
  int groups_z = IntegralDivideRoundUp(IntegralDivideRoundUp(dst_depth, z_out),
                                       group_size.z);
  return {groups_x, groups_y, groups_z};
}

uint3 GetGroupsCountForConvPrecise(const uint3& group_size,
                                   const BHWC& dst_shape) {
  const int z_out = GetNumOutputSlices(dst_shape.c);
  const int dst_depth = IntegralDivideRoundUp(dst_shape.c, 4);
  int xy_size = IntegralDivideRoundUp(dst_shape.w * dst_shape.h, 2);
  int z_size = IntegralDivideRoundUp(dst_depth, z_out);
  int task_size = xy_size * z_size;
  return {IntegralDivideRoundUp(task_size, group_size.x), 1, 1};
}

int GetConvolutionThreadsCount(const BHWC& dst_shape) {
  const uint3 group_size = GetWorkGroupForConv();
  const uint3 groups_count = GetGroupsCountForConv(group_size, dst_shape);
  return groups_count.x * groups_count.y * groups_count.z * group_size.x *
         group_size.y * group_size.z;
}

int GetConvolutionPreciseThreadsCount(const BHWC& dst_shape) {
  const uint3 group_size = GetWorkGroupForConvPrecise();
  const uint3 groups_count =
      GetGroupsCountForConvPrecise(group_size, dst_shape);
  return groups_count.x * groups_count.y * groups_count.z * group_size.x *
         group_size.y * group_size.z;
}

}  // namespace

std::vector<ComputeTaskDescriptorPtr> Convolution(
    int id, ValueId input_id, ValueId output_id,
    const Convolution2DAttributes& params, const RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source = GetKernelForConv(params);

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {
      output_id, "device FLT4* dst_buffer",
      [input_id, params](const std::map<ValueId, BHWC>& buffers) {
        return CalculateOutputShape(buffers.find(input_id)->second, params);
      }};

  auto weights_reordered = ReorderWeightsForConvShared(params);
  auto weights = options.storage_precision == RuntimeOptions::Precision::FP32
                     ? VectorToUint8Vector(weights_reordered)
                     : VectorFloatToHalf(weights_reordered);
  auto biases = options.storage_precision == RuntimeOptions::Precision::FP32
                    ? VectorToUint8Vector(params.bias.data)
                    : VectorFloatToHalf(params.bias.data);
  desc->immutable_buffers = {
      {"device FLT4* const weights", weights},
      {"device FLT4* const biases", biases},
  };

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id, output_id, params](const std::map<ValueId, BHWC>& buffers) {
         const auto& input_dimensions = buffers.find(input_id)->second;
         const auto& output_dimensions = buffers.find(output_id)->second;
         return GetUniformBufferForConvShared(input_dimensions,
                                              output_dimensions, params);
       }},
  };

  desc->resize_function = [output_id,
                           params](const std::map<ValueId, BHWC>& buffers) {
    const auto& output_dims = buffers.find(output_id)->second;
    const int num_output_slices = GetNumOutputSlices(params.weights.shape.o);
    const uint3 group_size{8, 4, 1};
    int groups_x = IntegralDivideRoundUp(output_dims.w, group_size.x);
    int groups_y = IntegralDivideRoundUp(output_dims.h, group_size.y);
    const int dst_depth = IntegralDivideRoundUp(params.weights.shape.o, 4);
    int groups_z = IntegralDivideRoundUp(dst_depth, num_output_slices);
    return std::make_pair(group_size, uint3{groups_x, groups_y, groups_z});
  };

  return {desc};
}

std::vector<ComputeTaskDescriptorPtr> Convolution1x1(
    int id, ValueId input_id, ValueId output_id,
    const Convolution2DAttributes& params,
    const metal::RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  const int z_out = GetNumOutputSlices(params.weights.shape.o);
  desc->shader_source = GetKernelForConv1x1(params, z_out);

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

  auto weights_reordered = ReorderWeightsForConv(params, z_out);
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
         return GetUniformBufferForConv(input_dimensions, output_dimensions,
                                        params);
       }},
  };

  desc->resize_function = [output_id,
                           params](const std::map<ValueId, BHWC>& buffers) {
    const auto& output_dims = buffers.find(output_id)->second;
    const uint3 group_size = GetWorkGroupForConv();
    const uint3 groups_count = GetGroupsCountForConv(group_size, output_dims);
    return std::make_pair(
        group_size, uint3{groups_count.z, groups_count.x, groups_count.y});
  };

  return {desc};
}

bool CheckConvolution1x1Support(const Convolution2DAttributes& attr) {
  return attr.weights.shape.h == 1 && attr.weights.shape.w == 1 &&
         attr.strides.h == 1 && attr.strides.w == 1 && attr.dilations.h == 1 &&
         attr.dilations.w == 1 && attr.padding.prepended.h == 0 &&
         attr.padding.prepended.w == 0 && attr.padding.appended.h == 0 &&
         attr.padding.appended.w == 0;
}

std::vector<ComputeTaskDescriptorPtr> ConvolutionGeneric(
    int id, ValueId input_id, ValueId output_id,
    const Convolution2DAttributes& params,
    const metal::RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  const int z_out = GetNumOutputSlices(params.weights.shape.o);
  desc->shader_source = GetKernelForConvGeneric(params, z_out);

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

  auto weights_reordered = ReorderWeightsForConv(params, z_out);
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
         return GetUniformBufferForConv(input_dimensions, output_dimensions,
                                        params);
       }},
  };

  desc->resize_function = [output_id,
                           params](const std::map<ValueId, BHWC>& buffers) {
    const auto& output_dims = buffers.find(output_id)->second;
    const uint3 group_size = GetWorkGroupForConv();
    const uint3 groups_count = GetGroupsCountForConv(group_size, output_dims);
    return std::make_pair(
        group_size, uint3{groups_count.z, groups_count.x, groups_count.y});
  };

  return {desc};
}

std::vector<ComputeTaskDescriptorPtr> ConvolutionPrecise(
    int id, ValueId input_id, ValueId output_id,
    const Convolution2DAttributes& params,
    const metal::RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  const int z_out = GetNumOutputSlices(params.weights.shape.o);
  desc->shader_source = GetKernelForConvPrecise(z_out);

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

  auto weights_reordered = ReorderWeightsForConv(params, z_out);
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
         return GetUniformBufferForConvPrecise(input_dimensions,
                                               output_dimensions, params);
       }},
  };

  desc->resize_function = [output_id,
                           params](const std::map<ValueId, BHWC>& buffers) {
    const auto& output_dims = buffers.find(output_id)->second;
    const uint3 group_size = GetWorkGroupForConvPrecise();
    const uint3 groups_count =
        GetGroupsCountForConvPrecise(group_size, output_dims);
    return std::make_pair(group_size, groups_count);
  };

  return {desc};
}

float GetThreadsRatioUsualToPreciseConvolution(const BHWC& dst_shape) {
  return static_cast<float>(GetConvolutionThreadsCount(dst_shape)) /
         static_cast<float>(GetConvolutionPreciseThreadsCount(dst_shape));
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
