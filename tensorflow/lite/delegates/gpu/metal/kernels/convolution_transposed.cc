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

#include "tensorflow/lite/delegates/gpu/metal/kernels/convolution_transposed.h"

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
#include "tensorflow/lite/delegates/gpu/metal/environment.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

const int kThreadGroupWidth = 16;
const int kThreadGroupHeight = 4;

std::string GetDeconvolution(const ConvolutionTransposedAttributes& attr) {
  std::string constant_args = R"(
    constant short2 padding = {$0, $1};
    constant short2 stride = {$2, $3};
    constant short2 kernel_size = {$4, $5};
    constant short2 inner_size = {$6, $7};
    constant short2 kernel_offset = {$8, $9};
  )";
  std::string shader_source = R"(
    #include <metal_stdlib>
    using namespace metal;

    struct FilterStripe {
      FLT4 vals[$0];
    };

    constant int src_depth = $1;
    constant int dst_depth = $2;
    constant int dst_channels = $3;
    constant int dst_channels_aligned = $4;

    $5

    struct uniforms {
      int2 src_size;
      int2 dst_size;
    };

    $$0
    kernel void ComputeFunction(
                                $$1
                                uint2 ugid[[thread_position_in_grid]]) {
      if (static_cast<int>(ugid.x) >= params.dst_size.x ||
          static_cast<int>(ugid.y) >= params.dst_size.y) {
        return;
      }

      float out[$4];
      for (short l = 0; l < dst_depth * 4; ++l) {
        out[l] = float(0.0f);
      }

      short2 offset = (short2(ugid) + padding - kernel_offset);
      offset.x = offset.x % stride.x;
      offset.y = offset.y % stride.y;
      offset += stride;
      offset.x = offset.x % stride.x;
      offset.y = offset.y % stride.y;
      short2 f_offset;
      f_offset.x = offset.x == 0 ? 0 : (stride.x - offset.x);
      f_offset.y = offset.y == 0 ? 0 : (stride.y - offset.y);
      for (int ky = 0; ky < inner_size.y; ++ky) {
        for (int kx = 0; kx < inner_size.x; ++kx) {
          short2 index = short2(kx, ky) * stride + f_offset;
          bool inside_kernel = index.x < kernel_size.x && index.y < kernel_size.y;
          const short2 src_coord = (short2(ugid) + index + padding - kernel_offset) / stride;
          index = kernel_size - short2(1, 1) - index;
          bool outside = src_coord.x < 0 || src_coord.y < 0 ||
            src_coord.x >= params.src_size.x || src_coord.y >= params.src_size.y;
          const int kernel_index = index.y * kernel_size.x + index.x;
          bool belong = inside_kernel && !outside;
          if (belong) {
            for (int l = 0; l < src_depth; ++l) {
              const int src_index = (l * params.src_size.y + src_coord.y)
                * params.src_size.x + src_coord.x;
              FLT4 srcColor = src_buffer[src_index];
              for (int k = 0; k < dst_channels; ++k) {
                out[k] += dot(srcColor, filters[kernel_index].vals[l * dst_channels_aligned + k]);
              }
            }
          }
        }
      }

      for (short l = 0; l < dst_depth; ++l) {
        FLT4 value = FLT4(out[l * 4], out[l * 4 + 1], out[l * 4 + 2], out[l * 4 + 3]) + biases[l];
        const int linear_index = (l * params.dst_size.y + int(ugid.y))
          * params.dst_size.x + int(ugid.x);
        uint3 gid = uint3(ugid.x, ugid.y, uint(l));
        $$2
        dst_buffer[linear_index] = value;
      }
    }
  )";
  const int kernel_x = attr.weights.shape.w;
  const int kernel_y = attr.weights.shape.h;
  const int inner_size_x = (kernel_x - 1) / attr.stride.w + 1;
  const int inner_size_y = (kernel_y - 1) / attr.stride.h + 1;
  std::string constant_args_inplaced = absl::Substitute(
      constant_args, attr.padding.prepended.w, attr.padding.prepended.h,
      attr.stride.w, attr.stride.h, kernel_x, kernel_y, inner_size_x,
      inner_size_y, kernel_x - 1, kernel_y - 1);
  const int src_depth = IntegralDivideRoundUp(attr.weights.shape.i, 4);
  const int dst_depth = IntegralDivideRoundUp(attr.weights.shape.o, 4);
  const int dst_channels_aligned = AlignByN(attr.weights.shape.o, 4);
  return absl::Substitute(shader_source, src_depth * dst_channels_aligned,
                          src_depth, dst_depth, attr.weights.shape.o,
                          dst_channels_aligned, constant_args_inplaced);
}

std::string GetDeconvolutionShared(const ConvolutionTransposedAttributes& attr,
                                   int workgroup_x, int workgroup_y) {
  std::string constant_args = R"(
    constant short2 padding = {$0, $1};
    constant short2 stride = {$2, $3};
    constant short2 kernel_size = {$4, $5};
    constant short2 inner_size = {$6, $7};
    constant short2 kernel_offset = {$8, $9};
  )";
  std::string shader_source = R"(
    #include <metal_stdlib>
    using namespace metal;

    struct FilterStripe {
      FLT4 vals[$0];
    };

    constant int src_depth = $1;
    constant int dst_depth = $2;
    constant int dst_channels = $3;
    constant int dst_channels_aligned = $4;

    $5

    constant short2 src_local_size = {$6, $7};

    struct uniforms {
      int2 src_size;
      int2 dst_size;
    };

    $$0
    kernel void ComputeFunction(
                                $$1
                                uint2 tid[[thread_position_in_threadgroup]],
                                uint2 ugid[[thread_position_in_grid]]) {
      float out[$4];
      for (short l = 0; l < dst_depth * 4; ++l) {
        out[l] = float(0.0f);
      }

      short2 offset = (short2(ugid) + padding - kernel_offset);
      offset.x = offset.x % stride.x;
      offset.y = offset.y % stride.y;
      offset += stride;
      offset.x = offset.x % stride.x;
      offset.y = offset.y % stride.y;
      short2 f_offset;
      f_offset.x = offset.x == 0 ? 0 : stride.x - offset.x;
      f_offset.y = offset.y == 0 ? 0 : stride.y - offset.y;

      short2 first_gid = short2((ugid.x / $8) * $8, (ugid.y / $9) * $9);

      short2 shared_offset = (first_gid + padding - kernel_offset);
      shared_offset.x = shared_offset.x % stride.x;
      shared_offset.y = shared_offset.y % stride.y;
      shared_offset += stride;
      shared_offset.x = shared_offset.x % stride.x;
      shared_offset.y = shared_offset.y % stride.y;
      short2 shared_f_offset;
      shared_f_offset.x = shared_offset.x == 0 ? 0 : (stride.x - shared_offset.x);
      shared_f_offset.y = shared_offset.y == 0 ? 0 : (stride.y - shared_offset.y);

      short2 first_index = short2(0, 0) * stride + shared_f_offset;
      const short2 first_src_coord = (first_gid + first_index + padding - kernel_offset) / stride;
      threadgroup FLT4 src_shared[$6][$7][$1];
      if (static_cast<int>(tid.x) < src_local_size.x &&
          static_cast<int>(tid.y) < src_local_size.y) {
        for (int z = 0; z < src_depth; ++z) {
          const short2 src_coord = first_src_coord + short2(tid);
          bool outside = src_coord.x < 0 || src_coord.y < 0 ||
            src_coord.x >= params.src_size.x || src_coord.y >= params.src_size.y;
          const int src_index = (z * params.src_size.y + src_coord.y)
            * params.src_size.x + src_coord.x;
          FLT4 src = !outside ? src_buffer[src_index] : FLT4(0.0f);
          src_shared[tid.x][tid.y][z] = src;
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (static_cast<int>(ugid.x) >= params.dst_size.x ||
          static_cast<int>(ugid.y) >= params.dst_size.y) {
        return;
      }

      for (int ky = 0; ky < inner_size.y; ++ky) {
        for (int kx = 0; kx < inner_size.x; ++kx) {
          short2 index = short2(kx, ky) * stride + f_offset;
          bool inside_kernel = index.x < kernel_size.x && index.y < kernel_size.y;
          const short2 src_coord = (short2(ugid) + index + padding - kernel_offset) / stride;
          index = kernel_size - short2(1, 1) - index;
          bool outside = src_coord.x < 0 || src_coord.y < 0 ||
            src_coord.x >= params.src_size.x || src_coord.y >= params.src_size.y;
          const int kernel_index = index.y * kernel_size.x + index.x;
          bool belong = inside_kernel && !outside;
          if (belong) {
            for (int k = 0; k < dst_channels; ++k) {
              for (int l = 0; l < src_depth; ++l) {
                short2 src_index = src_coord - first_src_coord;
                out[k] += dot(src_shared[src_index.x][src_index.y][l],
                              filters[kernel_index].vals[l * dst_channels_aligned + k]);
              }
            }
          }
        }
      }

      for (short l = 0; l < dst_depth; ++l) {
        FLT4 value = FLT4(out[l * 4], out[l * 4 + 1], out[l * 4 + 2], out[l * 4 + 3]) + biases[l];
        const int linear_index = (l * params.dst_size.y + int(ugid.y))
          * params.dst_size.x + int(ugid.x);
        uint3 gid = uint3(ugid.x, ugid.y, uint(l));
        $$2
        dst_buffer[linear_index] = value;
      }
    }
  )";
  const int kernel_x = attr.weights.shape.w;
  const int kernel_y = attr.weights.shape.h;
  const int inner_size_x = (kernel_x - 1) / attr.stride.w + 1;
  const int inner_size_y = (kernel_y - 1) / attr.stride.h + 1;
  std::string constant_args_inplaced = absl::Substitute(
      constant_args, attr.padding.prepended.w, attr.padding.prepended.h,
      attr.stride.w, attr.stride.h, kernel_x, kernel_y, inner_size_x,
      inner_size_y, kernel_x - 1, kernel_y - 1);
  const int src_depth = IntegralDivideRoundUp(attr.weights.shape.i, 4);
  const int dst_depth = IntegralDivideRoundUp(attr.weights.shape.o, 4);
  const int dst_channels_aligned = AlignByN(attr.weights.shape.o, 4);
  const int src_local_size_x = (workgroup_x + kernel_x) / attr.stride.w;
  const int src_local_size_y = (workgroup_y + kernel_y) / attr.stride.h;
  return absl::Substitute(
      shader_source, src_depth * dst_channels_aligned, src_depth, dst_depth,
      attr.weights.shape.o, dst_channels_aligned, constant_args_inplaced,
      src_local_size_x, src_local_size_y, workgroup_x, workgroup_y);
}

}  // namespace

std::vector<ComputeTaskDescriptorPtr> ConvolutionTransposed(
    int id, ValueId input_id, ValueId output_id,
    const ConvolutionTransposedAttributes& params,
    const RuntimeOptions& options) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;

  const int src_local_size_x =
      (kThreadGroupWidth + params.weights.shape.w) / params.stride.w;
  const int src_local_size_y =
      (kThreadGroupHeight + params.weights.shape.h) / params.stride.h;
  const int src_depth = IntegralDivideRoundUp(params.weights.shape.i, 4);
  const int shared_size =
      sizeof(float) * 4 * src_depth * src_local_size_x * src_local_size_y;
  int gpu_type = GetAppleSocVersion();
  if (shared_size < 1000 * 16 && (gpu_type == 7 || gpu_type == 8)) {
    desc->shader_source =
        GetDeconvolutionShared(params, kThreadGroupWidth, kThreadGroupHeight);
  } else {
    desc->shader_source = GetDeconvolution(params);
  }

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {
      output_id, "device FLT4* dst_buffer",
      [input_id, params](const std::map<ValueId, BHWC>& buffers) {
        return CalculateOutputShape(buffers.find(input_id)->second, params);
      }};

  const int src_ch_aligned = AlignByN(params.weights.shape.i, 4);
  const int dst_ch_aligned = AlignByN(params.weights.shape.o, 4);
  const int kernel_x = params.weights.shape.w;
  const int kernel_y = params.weights.shape.h;
  const int filters_aligned_size =
      src_ch_aligned * dst_ch_aligned * kernel_x * kernel_y;
  std::vector<float> filters_reordered(filters_aligned_size);

  int counter = 0;
  for (int y = 0; y < kernel_y; ++y) {
    for (int x = 0; x < kernel_x; ++x) {
      for (int ch = 0; ch < src_depth; ++ch) {
        for (int f = 0; f < dst_ch_aligned; ++f) {
          for (int i = 0; i < 4; ++i) {
            if (ch * 4 + i >= params.weights.shape.i ||
                f >= params.weights.shape.o) {
              filters_reordered[counter++] = 0.0f;
            } else {
              const int f_index =
                  params.weights.shape.LinearIndex({f, y, x, ch * 4 + i});
              filters_reordered[counter++] = params.weights.data[f_index];
            }
          }
        }
      }
    }
  }

  auto filters = options.storage_precision == RuntimeOptions::Precision::FP32
                     ? VectorToUint8Vector(filters_reordered)
                     : VectorFloatToHalf(filters_reordered);
  auto biases = options.storage_precision == RuntimeOptions::Precision::FP32
                    ? VectorToUint8Vector(params.bias.data)
                    : VectorFloatToHalf(params.bias.data);
  desc->immutable_buffers = {
      {"device FilterStripe* const filters", filters},
      {"constant FLT4* const biases", biases},
  };

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id, output_id](const std::map<ValueId, BHWC>& buffers) {
         const auto& dimension = buffers.find(input_id)->second;
         const auto& output_dimension = buffers.find(output_id)->second;
         std::vector<int> uniform_params{
             dimension.w,
             dimension.h,
             output_dimension.w,
             output_dimension.h,
         };
         return VectorToUint8Vector(uniform_params);
       }},
  };

  desc->resize_function = [input_id,
                           params](const std::map<ValueId, BHWC>& buffers) {
    const uint3 groups_size{kThreadGroupWidth, kThreadGroupHeight, 1};
    BHWC dst_shape =
        CalculateOutputShape(buffers.find(input_id)->second, params);
    int groups_x = IntegralDivideRoundUp(dst_shape.w, groups_size.x);
    int groups_y = IntegralDivideRoundUp(dst_shape.h, groups_size.y);
    int groups_z = 1;
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };

  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
