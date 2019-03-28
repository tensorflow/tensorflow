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

#include "tensorflow/lite/delegates/gpu/metal/kernels/convolution.h"

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

std::string GetKernel(const Convolution2DAttributes& params) {
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
std::vector<float> ReorderWeights(const Convolution2DAttributes& params) {
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

std::vector<uint8_t> GetUniformBuffer(const BHWC& input_dimensions,
                                      const BHWC& output_dimensions,
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

}  // namespace

std::vector<ComputeTaskDescriptorPtr> Convolution(
    int id, ValueId input_id, ValueId output_id,
    const Convolution2DAttributes& params, const RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source = GetKernel(params);

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {
      output_id, "device FLT4* dst_buffer",
      [input_id, params](const std::map<ValueId, BHWC>& buffers) {
        return CalculateOutputShape(buffers.find(input_id)->second, params);
      }};

  auto weights_reordered = ReorderWeights(params);
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
         return GetUniformBuffer(input_dimensions, output_dimensions, params);
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

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
