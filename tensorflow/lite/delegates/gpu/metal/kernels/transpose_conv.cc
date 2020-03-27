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

#include "tensorflow/lite/delegates/gpu/metal/kernels/transpose_conv.h"

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

struct GridParams {
  uint rect_offsets[4];
  uint widths[4];
  short2 origins[4];
  uint elements_count;
};

struct Params3x3 {
  short2 inner_size;
  short2 src_offset;
  short2 dst_offset;
};

void Init3x3(const ConvolutionTransposedAttributes& attr, const int2& src_size,
             const int2& dst_size, GridParams* grid_params,
             Params3x3* params3x3) {
  short2 src_size_scaled;
  src_size_scaled.x = (src_size.x - 1) * 2;
  src_size_scaled.y = (src_size.y - 1) * 2;
  short2 top_left_src, bottom_right_src;
  top_left_src.x = 1 - attr.padding.prepended.w;
  top_left_src.y = 1 - attr.padding.prepended.h;
  bottom_right_src.x = top_left_src.x + src_size_scaled.x;
  bottom_right_src.y = top_left_src.y + src_size_scaled.y;
  short2 top_left_inner, bottom_right_inner;
  if (top_left_src.x >= 0) {
    top_left_inner.x = top_left_src.x;
  } else {
    top_left_inner.x = std::abs(top_left_src.x % 2);
  }
  if (top_left_src.y >= 0) {
    top_left_inner.y = top_left_src.y;
  } else {
    top_left_inner.y = std::abs(top_left_src.y % 2);
  }

  if (bottom_right_src.x <= dst_size.x) {
    bottom_right_inner.x = bottom_right_src.x;
  } else {
    bottom_right_inner.x = dst_size.x;
  }
  if (top_left_src.x % 2 == 0) {
    bottom_right_inner.x -= bottom_right_inner.x % 2;
  } else {
    if (bottom_right_inner.x % 2 == 0) {
      bottom_right_inner.x -= 1;
    }
  }
  bottom_right_inner.x -= 1;

  if (bottom_right_src.y <= dst_size.y) {
    bottom_right_inner.y = bottom_right_src.y;
  } else {
    bottom_right_inner.y = dst_size.y;
  }
  if (top_left_src.y % 2 == 0) {
    bottom_right_inner.y -= bottom_right_inner.y % 2;
  } else {
    if (bottom_right_inner.y % 2 == 0) {
      bottom_right_inner.y -= 1;
    }
  }
  bottom_right_inner.y -= 1;

  params3x3->dst_offset = top_left_inner;
  params3x3->src_offset.x = (top_left_inner.x - top_left_src.x) / 2;
  params3x3->src_offset.y = (top_left_inner.y - top_left_src.y) / 2;
  params3x3->inner_size.x =
      std::max(0, bottom_right_inner.x - top_left_inner.x + 1) / 2;
  params3x3->inner_size.y =
      std::max(0, bottom_right_inner.y - top_left_inner.y + 1) / 2;

  short2 top_rect, bottom_rect, left_rect, right_rect;

  top_rect.x = dst_size.x;
  top_rect.y = top_left_inner.y;

  bottom_rect.x = dst_size.x;
  bottom_rect.y = dst_size.y - bottom_right_inner.y - 1;

  left_rect.x = top_left_inner.x;
  left_rect.y = dst_size.y - top_rect.y - bottom_rect.y;

  right_rect.x = dst_size.x - bottom_right_inner.x - 1;
  right_rect.y = left_rect.y;

  grid_params->widths[0] = top_rect.x;
  grid_params->widths[1] = left_rect.x;
  grid_params->widths[2] = right_rect.x;
  grid_params->widths[3] = bottom_rect.x;

  grid_params->rect_offsets[0] = 0;
  grid_params->rect_offsets[1] =
      grid_params->rect_offsets[0] + top_rect.x * top_rect.y;
  grid_params->rect_offsets[2] =
      grid_params->rect_offsets[1] + left_rect.x * left_rect.y;
  grid_params->rect_offsets[3] =
      grid_params->rect_offsets[2] + right_rect.x * right_rect.y;
  grid_params->elements_count =
      grid_params->rect_offsets[3] + bottom_rect.x * bottom_rect.y;

  grid_params->origins[0] = short2(0, 0);
  grid_params->origins[1] = short2(int16_t(0), int16_t(top_rect.y));
  grid_params->origins[2] =
      short2(int16_t(dst_size.x - right_rect.x), int16_t(top_rect.y));
  grid_params->origins[3] = short2(0, dst_size.y - bottom_rect.y);
}

std::string GetDeconvolutionBorder(
    const ConvolutionTransposedAttributes& attr) {
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
      uint rect_offsets[4];
      uint widths[4];
      short2 origins[4];
      uint elements_count;
    };

    short2 GetGridIdByLinearId(uint linear_id, constant uniforms& params);

    short2 GetGridIdByLinearId(uint linear_id, constant uniforms& params) {
      int index = 0;
      index = linear_id >= params.rect_offsets[0] ? 0 : index;
      index = linear_id >= params.rect_offsets[1] ? 1 : index;
      index = linear_id >= params.rect_offsets[2] ? 2 : index;
      index = linear_id >= params.rect_offsets[3] ? 3 : index;

      const uint rect_index = linear_id - params.rect_offsets[index];

      const uint rect_width = params.widths[index];
      const short2 offset = short2(rect_index % rect_width, rect_index / rect_width);
      return params.origins[index] + offset;
    }

    $$0
    kernel void ComputeFunction(
                                $$1
                                uint linear_id[[thread_position_in_grid]]) {
      if (linear_id >= params.elements_count) {
        return;
      }
      short2 gid_sh = GetGridIdByLinearId(linear_id, params);

      float out[$4];
      for (short l = 0; l < dst_depth * 4; ++l) {
        out[l] = float(0.0f);
      }

      short2 offset = gid_sh + padding - kernel_offset;
      offset.x = offset.x % stride.x;
      offset.y = offset.y % stride.y;
      offset += stride;
      offset.x = offset.x % stride.x;
      offset.y = offset.y % stride.y;
      short2 f_offset;
      f_offset.x = offset.x == 0 ? 0 : stride.x - offset.x;
      f_offset.y = offset.y == 0 ? 0 : stride.y - offset.y;
      for (int ky = 0; ky < inner_size.y; ++ky) {
        for (int kx = 0; kx < inner_size.x; ++kx) {
          short2 index = short2(kx, ky) * stride + f_offset;
          bool inside_kernel = index.x < kernel_size.x && index.y < kernel_size.y;
          const short2 src_coord = (gid_sh + index + padding - kernel_offset) / stride;
          index = kernel_size - short2(1, 1) - index;
          bool outside = src_coord.x < 0 || src_coord.y < 0 ||
            src_coord.x >= params.src_size.x || src_coord.y >= params.src_size.y;
          const int kernel_index = index.y * kernel_size.x + index.x;
          bool belong = inside_kernel && !outside;
          if (belong) {
            for (int l = 0; l < src_depth; ++l) {
              const int src_index = (l * params.src_size.y + src_coord.y) *
                params.src_size.x + src_coord.x;
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
        const int linear_index = (l * params.dst_size.y + int(gid_sh.y)) *
          params.dst_size.x + int(gid_sh.x);
        uint3 gid = uint3(uint(gid_sh.x), uint(gid_sh.y), uint(l));
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

std::string GetDeconvolution3x3(const ConvolutionTransposedAttributes& attr) {
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

      struct uniforms {
      int2 src_size;
      int2 dst_size;
      short2 inner_size;
      short2 src_offset;
      short2 dst_offset;
    };

    $$0
    kernel void ComputeFunction(
                                $$1
                                uint tid[[thread_index_in_threadgroup]],
                                uint2 ugid[[thread_position_in_grid]]) {
      if (static_cast<int>(ugid.x) >= params.inner_size.x ||
          static_cast<int>(ugid.y) >= params.inner_size.y) {
        return;
      }

      float out[$4];
      short2 src_coord_0 = short2(ugid) + params.src_offset;
      short2 dst_coord = short2(ugid) * 2 + params.dst_offset;

      for (short l = 0; l < dst_depth * 4; ++l) {
        out[l] = float(0.0f);
      }

      for (int l = 0; l < src_depth; ++l) {
        const int src_index_0 = (l * params.src_size.y + src_coord_0.y) *
          params.src_size.x + src_coord_0.x;
        FLT4 srcColor_0 = src_buffer[src_index_0];
        for (int k = 0; k < dst_channels; ++k) {
          out[k] += dot(srcColor_0, filters[4].vals[l * dst_channels_aligned + k]);
        }
      }

      for (short l = 0; l < dst_depth; ++l) {
        FLT4 value = FLT4(out[l * 4], out[l * 4 + 1], out[l * 4 + 2], out[l * 4 + 3]) + biases[l];
        const int linear_index = (l * params.dst_size.y + int(dst_coord.y)) *
          params.dst_size.x + int(dst_coord.x);
        uint3 gid = uint3(uint(dst_coord.x), uint(dst_coord.y), uint(l));
        $$2
        dst_buffer[linear_index] = value;
      }

      short2 src_coord_1 = src_coord_0 + short2(1, 0);
      dst_coord += short2(1, 0);

      for (short l = 0; l < dst_depth * 4; ++l) {
        out[l] = float(0.0f);
      }

      for (int l = 0; l < src_depth; ++l) {
        const int src_index_0 = (l * params.src_size.y + src_coord_0.y) *
          params.src_size.x + src_coord_0.x;
        const int src_index_1 = (l * params.src_size.y + src_coord_1.y) *
          params.src_size.x + src_coord_1.x;
        FLT4 srcColor_0 = src_buffer[src_index_0];
        FLT4 srcColor_1 = src_buffer[src_index_1];
        for (int k = 0; k < dst_channels; ++k) {
          out[k] += dot(srcColor_0, filters[5].vals[l * dst_channels_aligned + k]);
          out[k] += dot(srcColor_1, filters[3].vals[l * dst_channels_aligned + k]);
        }
      }

      for (short l = 0; l < dst_depth; ++l) {
        FLT4 value = FLT4(out[l * 4], out[l * 4 + 1], out[l * 4 + 2], out[l * 4 + 3]) + biases[l];
        const int linear_index = (l * params.dst_size.y + int(dst_coord.y)) *
          params.dst_size.x + int(dst_coord.x);
        uint3 gid = uint3(uint(dst_coord.x), uint(dst_coord.y), uint(l));
        $$2
        dst_buffer[linear_index] = value;
      }

      short2 src_coord_2 = src_coord_0 + short2(0, 1);
      dst_coord += short2(-1, 1);

      for (short l = 0; l < dst_depth * 4; ++l) {
        out[l] = float(0.0f);
      }

      for (int l = 0; l < src_depth; ++l) {
        const int src_index_0 = (l * params.src_size.y + src_coord_0.y) *
          params.src_size.x + src_coord_0.x;
        const int src_index_2 = (l * params.src_size.y + src_coord_2.y) *
          params.src_size.x + src_coord_2.x;
        FLT4 srcColor_0 = src_buffer[src_index_0];
        FLT4 srcColor_2 = src_buffer[src_index_2];
        for (int k = 0; k < dst_channels; ++k) {
          out[k] += dot(srcColor_0, filters[7].vals[l * dst_channels_aligned + k]);
          out[k] += dot(srcColor_2, filters[1].vals[l * dst_channels_aligned + k]);
        }
      }

      for (short l = 0; l < dst_depth; ++l) {
        FLT4 value = FLT4(out[l * 4], out[l * 4 + 1], out[l * 4 + 2], out[l * 4 + 3]) + biases[l];
        const int linear_index = (l * params.dst_size.y + int(dst_coord.y)) *
          params.dst_size.x + int(dst_coord.x);
        uint3 gid = uint3(uint(dst_coord.x), uint(dst_coord.y), uint(l));
        $$2
        dst_buffer[linear_index] = value;
      }

      short2 src_coord_3 = src_coord_0 + short2(1, 1);
      dst_coord += short2(1, 0);

      for (short l = 0; l < dst_depth * 4; ++l) {
        out[l] = float(0.0f);
      }

      for (int l = 0; l < src_depth; ++l) {
        const int src_index_0 = (l * params.src_size.y + src_coord_0.y) *
          params.src_size.x + src_coord_0.x;
        const int src_index_1 = (l * params.src_size.y + src_coord_1.y) *
          params.src_size.x + src_coord_1.x;
        const int src_index_2 = (l * params.src_size.y + src_coord_2.y) *
          params.src_size.x + src_coord_2.x;
        const int src_index_3 = (l * params.src_size.y + src_coord_3.y) *
          params.src_size.x + src_coord_3.x;
        FLT4 srcColor_0 = src_buffer[src_index_0];
        FLT4 srcColor_1 = src_buffer[src_index_1];
        FLT4 srcColor_2 = src_buffer[src_index_2];
        FLT4 srcColor_3 = src_buffer[src_index_3];
        for (int k = 0; k < dst_channels; ++k) {
          out[k] += dot(srcColor_0, filters[8].vals[l * dst_channels_aligned + k]);
          out[k] += dot(srcColor_1, filters[6].vals[l * dst_channels_aligned + k]);
          out[k] += dot(srcColor_2, filters[2].vals[l * dst_channels_aligned + k]);
          out[k] += dot(srcColor_3, filters[0].vals[l * dst_channels_aligned + k]);
        }
      }

      for (short l = 0; l < dst_depth; ++l) {
        FLT4 value = FLT4(out[l * 4], out[l * 4 + 1], out[l * 4 + 2], out[l * 4 + 3]) + biases[l];
        const int linear_index = (l * params.dst_size.y + int(dst_coord.y)) *
          params.dst_size.x + int(dst_coord.x);
        uint3 gid = uint3(uint(dst_coord.x), uint(dst_coord.y), uint(l));
        $$2
        dst_buffer[linear_index] = value;
      }
    }
  )";

  const int src_depth = IntegralDivideRoundUp(attr.weights.shape.i, 4);
  const int dst_depth = IntegralDivideRoundUp(attr.weights.shape.o, 4);
  const int dst_channels_aligned = AlignByN(attr.weights.shape.o, 4);
  return absl::Substitute(shader_source, src_depth * dst_channels_aligned,
                          src_depth, dst_depth, attr.weights.shape.o,
                          dst_channels_aligned);
}

std::string GetDeconvolutionShared3x3(
    const ConvolutionTransposedAttributes& attr) {
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

      struct uniforms {
      int2 src_size;
      int2 dst_size;
      short2 inner_size;
      short2 src_offset;
      short2 dst_offset;
    };

    $$0
    kernel void ComputeFunction(
                                $$1
                                uint tid[[thread_index_in_threadgroup]],
                                uint2 ugid[[thread_position_in_grid]]) {

      float out[$4];
      for (short l = 0; l < dst_depth * 4; ++l) {
        out[l] = float(0.0f);
      }

      threadgroup FilterStripe stripes[4];
      threadgroup_barrier(mem_flags::mem_none);
      if (tid < dst_channels) {
        for (int l = 0; l < src_depth; ++l) {
          stripes[0].vals[l * dst_channels_aligned + tid]
            = filters[4].vals[l * dst_channels_aligned + tid];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      bool inside_grid = (static_cast<int>(ugid.x) < params.inner_size.x)
        && (static_cast<int>(ugid.y) < params.inner_size.y);

      short2 src_coord_0 = short2(ugid) + params.src_offset;
      short2 dst_coord = short2(ugid) * 2 + params.dst_offset;

      if (inside_grid) {
        for (short l = 0; l < dst_depth * 4; ++l) {
          out[l] = float(0.0f);
        }

        for (int l = 0; l < src_depth; ++l) {
          const int src_index_0 = (l * params.src_size.y + src_coord_0.y) *
            params.src_size.x + src_coord_0.x;
          FLT4 srcColor_0 = src_buffer[src_index_0];
          for (int k = 0; k < dst_channels; ++k) {
            out[k] += dot(srcColor_0, stripes[0].vals[l * dst_channels_aligned + k]);
          }
        }

        for (short l = 0; l < dst_depth; ++l) {
          FLT4 value = FLT4(out[l * 4], out[l * 4 + 1], out[l * 4 + 2], out[l * 4 + 3]) + biases[l];
          const int linear_index = (l * params.dst_size.y + int(dst_coord.y)) *
            params.dst_size.x + int(dst_coord.x);
          uint3 gid = uint3(ugid.x, ugid.y, uint(l));
          $$2
          dst_buffer[linear_index] = value;
        }
      }

      short2 src_coord_1 = src_coord_0 + short2(1, 0);
      dst_coord += short2(1, 0);

      threadgroup_barrier(mem_flags::mem_none);
      if (tid < dst_channels) {
        for (int l = 0; l < src_depth; ++l) {
          stripes[0].vals[l * dst_channels_aligned + tid]
            = filters[5].vals[l * dst_channels_aligned + tid];
          stripes[1].vals[l * dst_channels_aligned + tid]
            = filters[3].vals[l * dst_channels_aligned + tid];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (inside_grid) {
        for (short l = 0; l < dst_depth * 4; ++l) {
          out[l] = float(0.0f);
        }

        for (int l = 0; l < src_depth; ++l) {
          const int src_index_0 = (l * params.src_size.y + src_coord_0.y) *
            params.src_size.x + src_coord_0.x;
          const int src_index_1 = (l * params.src_size.y + src_coord_1.y) *
            params.src_size.x + src_coord_1.x;
          FLT4 srcColor_0 = src_buffer[src_index_0];
          FLT4 srcColor_1 = src_buffer[src_index_1];
          for (int k = 0; k < dst_channels; ++k) {
            out[k] += dot(srcColor_0, stripes[0].vals[l * dst_channels_aligned + k]);
            out[k] += dot(srcColor_1, stripes[1].vals[l * dst_channels_aligned + k]);
          }
        }

        for (short l = 0; l < dst_depth; ++l) {
          FLT4 value = FLT4(out[l * 4], out[l * 4 + 1], out[l * 4 + 2], out[l * 4 + 3]) + biases[l];
          const int linear_index = (l * params.dst_size.y + int(dst_coord.y)) *
            params.dst_size.x + int(dst_coord.x);
          uint3 gid = uint3(ugid.x, ugid.y, uint(l));
          $$2
          dst_buffer[linear_index] = value;
        }
      }

      short2 src_coord_2 = src_coord_0 + short2(0, 1);
      dst_coord += short2(-1, 1);

      threadgroup_barrier(mem_flags::mem_none);
      if (tid < dst_channels) {
        for (int l = 0; l < src_depth; ++l) {
          stripes[0].vals[l * dst_channels_aligned + tid]
            = filters[7].vals[l * dst_channels_aligned + tid];
          stripes[1].vals[l * dst_channels_aligned + tid]
            = filters[1].vals[l * dst_channels_aligned + tid];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (inside_grid) {
        for (short l = 0; l < dst_depth * 4; ++l) {
          out[l] = float(0.0f);
        }

        for (int l = 0; l < src_depth; ++l) {
          const int src_index_0 = (l * params.src_size.y + src_coord_0.y) *
            params.src_size.x + src_coord_0.x;
          const int src_index_2 = (l * params.src_size.y + src_coord_2.y) *
            params.src_size.x + src_coord_2.x;
          FLT4 srcColor_0 = src_buffer[src_index_0];
          FLT4 srcColor_2 = src_buffer[src_index_2];
          for (int k = 0; k < dst_channels; ++k) {
            out[k] += dot(srcColor_0, stripes[0].vals[l * dst_channels_aligned + k]);
            out[k] += dot(srcColor_2, stripes[1].vals[l * dst_channels_aligned + k]);
          }
        }

        for (short l = 0; l < dst_depth; ++l) {
          FLT4 value = FLT4(out[l * 4], out[l * 4 + 1], out[l * 4 + 2], out[l * 4 + 3]) + biases[l];
          const int linear_index = (l * params.dst_size.y + int(dst_coord.y)) *
            params.dst_size.x + int(dst_coord.x);
          uint3 gid = uint3(ugid.x, ugid.y, uint(l));
          $$2
          dst_buffer[linear_index] = value;
        }
      }

            short2 src_coord_3 = src_coord_0 + short2(1, 1);
      dst_coord += short2(1, 0);

      threadgroup_barrier(mem_flags::mem_none);
      if (tid < dst_channels) {
        for (int l = 0; l < src_depth; ++l) {
          stripes[0].vals[l * dst_channels_aligned + tid]
            = filters[8].vals[l * dst_channels_aligned + tid];
          stripes[1].vals[l * dst_channels_aligned + tid]
            = filters[6].vals[l * dst_channels_aligned + tid];
          stripes[2].vals[l * dst_channels_aligned + tid]
            = filters[2].vals[l * dst_channels_aligned + tid];
          stripes[3].vals[l * dst_channels_aligned + tid]
            = filters[0].vals[l * dst_channels_aligned + tid];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (inside_grid) {
        for (short l = 0; l < dst_depth * 4; ++l) {
          out[l] = float(0.0f);
        }

        for (int l = 0; l < src_depth; ++l) {
          const int src_index_0 = (l * params.src_size.y + src_coord_0.y) *
            params.src_size.x + src_coord_0.x;
          const int src_index_1 = (l * params.src_size.y + src_coord_1.y) *
            params.src_size.x + src_coord_1.x;
          const int src_index_2 = (l * params.src_size.y + src_coord_2.y) *
            params.src_size.x + src_coord_2.x;
          const int src_index_3 = (l * params.src_size.y + src_coord_3.y) *
            params.src_size.x + src_coord_3.x;
          FLT4 srcColor_0 = src_buffer[src_index_0];
          FLT4 srcColor_1 = src_buffer[src_index_1];
          FLT4 srcColor_2 = src_buffer[src_index_2];
          FLT4 srcColor_3 = src_buffer[src_index_3];
          for (int k = 0; k < dst_channels; ++k) {
            out[k] += dot(srcColor_0, stripes[0].vals[l * dst_channels_aligned + k]);
            out[k] += dot(srcColor_1, stripes[1].vals[l * dst_channels_aligned + k]);
            out[k] += dot(srcColor_2, stripes[2].vals[l * dst_channels_aligned + k]);
            out[k] += dot(srcColor_3, stripes[3].vals[l * dst_channels_aligned + k]);
          }
        }

        for (short l = 0; l < dst_depth; ++l) {
          FLT4 value = FLT4(out[l * 4], out[l * 4 + 1], out[l * 4 + 2], out[l * 4 + 3]) + biases[l];
          const int linear_index = (l * params.dst_size.y + int(dst_coord.y)) *
            params.dst_size.x + int(dst_coord.x);
          uint3 gid = uint3(ugid.x, ugid.y, uint(l));
          $$2
          dst_buffer[linear_index] = value;
        }
      }
    }
  )";
  const int src_depth = IntegralDivideRoundUp(attr.weights.shape.i, 4);
  const int dst_depth = IntegralDivideRoundUp(attr.weights.shape.o, 4);
  const int dst_channels_aligned = AlignByN(attr.weights.shape.o, 4);
  return absl::Substitute(shader_source, src_depth * dst_channels_aligned,
                          src_depth, dst_depth, attr.weights.shape.o,
                          dst_channels_aligned);
}

std::string GetDeconvolution4x4(const int2& block_size, bool use_local_mem) {
  std::string c = R"(
    #include <metal_stdlib>
    using namespace metal;

    struct uniforms {
      int4 src_size;
      int4 dst_size;
      int filter_offset;
      int3 dummy_0;
    };
)";
  c += R"(
    $0
    kernel void ComputeFunction(
                                $1
                                uint3 group_id[[threadgroup_position_in_grid]],
                                uint3 tid3d[[thread_position_in_threadgroup]],
                                uint3 ugid[[thread_position_in_grid]]) {
)";
  c += "  int X = static_cast<int>(group_id.y * 8u + tid3d.x);\n";
  c += "  int Y = static_cast<int>(group_id.z * 4u + tid3d.y);\n";
  c += "  int Z = static_cast<int>(group_id.x * 1u + tid3d.z);\n";
  c += "  X *= " + std::to_string(block_size.x) + ";\n";
  c += "  Y *= " + std::to_string(block_size.y) + ";\n";
  if (!use_local_mem) {
    c += "  if (X * 2 > params.dst_size.x || Y * 2 > params.dst_size.y || Z >= "
         "params.dst_size.z) return;\n";
  }
  for (int y = 0; y < block_size.y; ++y) {
    for (int x = 0; x < block_size.x; ++x) {
      const std::string block = std::to_string(x) + std::to_string(y);
      c += "  ACCUM_FLT4 r_" + block +
           "_00 = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);\n";
      c += "  ACCUM_FLT4 r_" + block +
           "_10 = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);\n";
      c += "  ACCUM_FLT4 r_" + block +
           "_01 = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);\n";
      c += "  ACCUM_FLT4 r_" + block +
           "_11 = ACCUM_FLT4(0.0f, 0.0f, 0.0f, 0.0f);\n";
    }
  }
  c += "  int f_offset = Z * params.filter_offset;\n";
  if (use_local_mem) {
    c += "  threadgroup FLT4 weights_cache[64];\n";
    c += "  int local_id = static_cast<int>(tid3d.y * 8u + tid3d.x);\n";
  }
  for (int x = 0; x < block_size.x + 1; ++x) {
    const std::string sx = std::to_string(x);
    const std::string xc =
        x == 0 ? std::string("X - 1") : "X + " + std::to_string(x - 1);
    c += "  bool in_x" + sx + " = " + xc + " >= 0 && " + xc +
         " < params.src_size.x;\n";
    c += "  int xc" + sx + " = clamp(" + xc + ", 0, params.src_size.x - 1);\n";
  }
  for (int y = 0; y < block_size.y + 1; ++y) {
    const std::string sy = std::to_string(y);
    const std::string yc =
        y == 0 ? std::string("Y - 1") : "Y + " + std::to_string(y - 1);
    c += "  bool in_y" + std::to_string(y) + " = " + yc + " >= 0 && " + yc +
         " < params.src_size.y;\n";
    c += "  int yc" + sy + " = clamp(" + yc + ", 0, params.src_size.y - 1);\n";
  }
  for (int y = 0; y < block_size.y + 1; ++y) {
    for (int x = 0; x < block_size.x + 1; ++x) {
      const std::string sx = std::to_string(x);
      const std::string sy = std::to_string(y);
      c += "  FLT m_" + sx + sy + " = in_x" + sx + " && in_y" + sy + ";\n";
      c += "  device FLT4* src_ptr_" + sx + sy + " = src_buffer + yc" + sy +
           " * params.src_size.x + xc" + sx + ";\n";
    }
  }
  c += "  for (int s = 0; s < params.src_size.z; ++s) {\n";
  if (use_local_mem) {
    c += "    BARRIER(mem_flags::mem_none);\n";
    c += "    weights_cache[local_id] = filters[f_offset + local_id];\n";
    c += "    weights_cache[local_id + 32] = filters[f_offset + local_id + "
         "32];\n";
  } else {
    c += "    device FLT4* weights_cache = filters + f_offset;\n";
  }
  for (int y = 0; y < block_size.y + 1; ++y) {
    for (int x = 0; x < block_size.x + 1; ++x) {
      const std::string id = std::to_string(x) + std::to_string(y);
      c += "    FLT4 src_" + id + " = *src_ptr_" + id + " * m_" + id +
           "; src_ptr_" + id + " += params.src_size.w;\n";
    }
  }
  c += "    f_offset += 64;\n";
  if (use_local_mem) {
    c += "    BARRIER(mem_flags::mem_threadgroup);\n";
  }
  for (int i = 0; i < 16; ++i) {
    const int result_sub_pixel_id = i % 4;
    const int src_pixel_id = i / 4;
    const int weights_offset = i * 4;
    for (int y = 0; y < block_size.y; ++y) {
      for (int x = 0; x < block_size.x; ++x) {
        const std::string block = std::to_string(x) + std::to_string(y);
        const std::string R = "r_" + block + "_" +
                              std::to_string(result_sub_pixel_id % 2) +
                              std::to_string(result_sub_pixel_id / 2);
        const std::string S = "src_" + std::to_string(src_pixel_id % 2 + x) +
                              std::to_string(src_pixel_id / 2 + y);
        c += "    " + R + ".x += dot(" + S + ", weights_cache[" +
             std::to_string(weights_offset + 0) + "]);\n";
        c += "    " + R + ".y += dot(" + S + ", weights_cache[" +
             std::to_string(weights_offset + 1) + "]);\n";
        c += "    " + R + ".z += dot(" + S + ", weights_cache[" +
             std::to_string(weights_offset + 2) + "]);\n";
        c += "    " + R + ".w += dot(" + S + ", weights_cache[" +
             std::to_string(weights_offset + 3) + "]);\n";
      }
    }
  }
  c += "  }\n";
  c += "\n";
  if (use_local_mem) {
    c += "  if (X * 2 > params.dst_size.x || Y * 2 > params.dst_size.y || Z >= "
         "params.dst_size.z) return;\n";
  }
  c += "  X = X * 2 - 1;\n";
  c += "  Y = Y * 2 - 1;\n";
  c += "\n";
  c += "  const int dst_offset = (Z * params.dst_size.y + Y) * "
       "params.dst_size.x "
       "+ X;\n";
  c += "  FLT4 bias_val = biases[Z];\n";
  for (int y = 0; y < block_size.y; ++y) {
    for (int x = 0; x < block_size.x; ++x) {
      for (int sub_y = 0; sub_y < 2; ++sub_y) {
        for (int sub_x = 0; sub_x < 2; ++sub_x) {
          const int x_offset = x * 2 + sub_x;
          const int y_offset = y * 2 + sub_y;
          const std::string block = std::to_string(x) + std::to_string(y);
          const std::string R = "r_" + block + "_" + std::to_string(sub_x) +
                                std::to_string(sub_y);
          const std::string dst_x = "X + " + std::to_string(x_offset);
          const std::string dst_y = "Y + " + std::to_string(y_offset);
          const std::string x_check = x_offset == 0
                                          ? std::string("X >= 0")
                                          : dst_x + " < params.dst_size.x";
          const std::string y_check = y_offset == 0
                                          ? std::string("Y >= 0")
                                          : dst_y + " < params.dst_size.y";
          c += "  if (" + x_check + " && " + y_check + ") {\n";
          c += "    FLT4 value = FLT4(" + R + ") + bias_val;\n";
          c += "    int linear_index = dst_offset + params.dst_size.x * " +
               std::to_string(y_offset) + " + " + std::to_string(x_offset) +
               ";\n";
          c += "    uint3 gid = uint3(" + dst_x + ", " + dst_y + ", Z);\n";
          c += "    $2\n";
          c += "    dst_buffer[linear_index] = value;\n";
          c += "  }\n";
        }
      }
    }
  }
  c += "}\n";
  return c;
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
  auto gpu_type = GetGpuType();
  if (shared_size < 1000 * 16 &&
      (gpu_type == GpuType::kA7 || gpu_type == GpuType::kA8)) {
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

  auto filters =
      GetByteBufferConverted(filters_reordered, options.storage_precision);
  desc->immutable_buffers = {
      {"device FilterStripe* const filters", filters},
      {"device FLT4* const biases",
       GetByteBufferConvertedResized(params.bias.data,
                                     options.storage_precision,
                                     params.weights.shape.o)},
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
         return GetByteBuffer(uniform_params);
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

std::vector<ComputeTaskDescriptorPtr> ConvolutionTransposed3x3(
    int id, ValueId input_id, ValueId output_id,
    const ConvolutionTransposedAttributes& params,
    const RuntimeOptions& options) {
  const int kThreadGroupWidth = 16;
  const int kThreadGroupHeight = 4;

  auto border_desc = std::make_shared<ComputeTaskDescriptor>();
  border_desc->id = id;
  border_desc->is_linkable = false;

  border_desc->shader_source = GetDeconvolutionBorder(params);

  border_desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  border_desc->output_buffer = {
      output_id, "device FLT4* dst_buffer",
      [input_id, params](const std::map<ValueId, BHWC>& buffers) {
        const auto& src_shape = buffers.find(input_id)->second;
        BHWC dst_shape = CalculateOutputShape(src_shape, params);
        return BHWC{src_shape.b, dst_shape.h, dst_shape.w, dst_shape.c};
      }};

  const int src_depth = IntegralDivideRoundUp(params.weights.shape.i, 4);
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

  auto filters =
      GetByteBufferConverted(filters_reordered, options.storage_precision);
  auto biases = GetByteBufferConvertedResized(
      params.bias.data, options.storage_precision, params.weights.shape.o);
  border_desc->immutable_buffers = {
      {"device FilterStripe* const filters", filters},
      {"constant FLT4* const biases", biases},
  };

  border_desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id, output_id, params](const std::map<ValueId, BHWC>& buffers) {
         const auto& src_dim = buffers.find(input_id)->second;
         const auto& dst_dim = buffers.find(output_id)->second;
         GridParams grid_params;
         Params3x3 params3x3;
         Init3x3(params, int2(src_dim.w, src_dim.h), int2(dst_dim.w, dst_dim.h),
                 &grid_params, &params3x3);
         int* ptr = reinterpret_cast<int*>(&grid_params);
         std::vector<int> uniform_params{
             src_dim.w,
             src_dim.h,
             dst_dim.w,
             dst_dim.h,
             /*uint GridParams.rect_offsets[4]*/
             ptr[0],
             ptr[1],
             ptr[2],
             ptr[3],
             /*uint GridParams.widths[4]*/
             ptr[4],
             ptr[5],
             ptr[6],
             ptr[7],
             /*short2 GridParams.origins[4]*/
             ptr[8],
             ptr[9],
             ptr[10],
             ptr[11],
             /*uint GridParams.elements_count*/
             ptr[12],
         };
         return GetByteBuffer(uniform_params);
       }},
  };

  border_desc->resize_function =
      [input_id, params](const std::map<ValueId, BHWC>& buffers) {
        const uint3 groups_size{kThreadGroupWidth * kThreadGroupHeight, 1, 1};
        const auto& src_shape = buffers.find(input_id)->second;
        BHWC dst_shape = CalculateOutputShape(src_shape, params);
        GridParams grid_params;
        Params3x3 params3x3;
        Init3x3(params, int2(src_shape.w, src_shape.h),
                int2(dst_shape.w, dst_shape.h), &grid_params, &params3x3);
        if (grid_params.elements_count == 0) {
          return std::make_pair(groups_size, uint3{0, 0, 0});
        }
        int groups_x =
            IntegralDivideRoundUp(grid_params.elements_count, groups_size.x);
        int groups_y = 1;
        int groups_z = 1;
        return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
      };

  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;

  const int shared_size = sizeof(float) * 4 * src_depth * dst_ch_aligned * 4;
  auto gpu_type = GetGpuType();
  if (shared_size < (1024 * 16 - 32) &&
      (gpu_type == GpuType::kA7 || gpu_type == GpuType::kA8) &&
      dst_ch_aligned <= kThreadGroupWidth * kThreadGroupHeight) {
    desc->shader_source = GetDeconvolutionShared3x3(params);
  } else {
    desc->shader_source = GetDeconvolution3x3(params);
  }

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {
      output_id, "device FLT4* dst_buffer",
      [input_id, params](const std::map<ValueId, BHWC>& buffers) {
        const auto& src_shape = buffers.find(input_id)->second;
        BHWC dst_shape = CalculateOutputShape(src_shape, params);
        return BHWC{src_shape.b, dst_shape.h, dst_shape.w, dst_shape.c};
      }};

  desc->immutable_buffers = {
      {"device FilterStripe* const filters", filters},
      {"constant FLT4* const biases", biases},
  };

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id, output_id, params](const std::map<ValueId, BHWC>& buffers) {
         const auto& src_shape = buffers.find(input_id)->second;
         const auto& dst_shape = buffers.find(output_id)->second;
         GridParams grid_params;
         Params3x3 params3x3;
         Init3x3(params, int2(src_shape.w, src_shape.h),
                 int2(dst_shape.w, dst_shape.h), &grid_params, &params3x3);
         int* ptr = reinterpret_cast<int*>(&params3x3);
         std::vector<int> uniform_params{
             src_shape.w,
             src_shape.h,
             dst_shape.w,
             dst_shape.h,
             /*short2 Params3x3.inner_size*/ ptr[0],
             /*short2 Params3x3.src_offset*/ ptr[1],
             /*short2 Params3x3.dst_offset*/ ptr[2],
         };
         return GetByteBuffer(uniform_params);
       }},
  };

  desc->resize_function = [input_id,
                           params](const std::map<ValueId, BHWC>& buffers) {
    const uint3 groups_size{kThreadGroupWidth, kThreadGroupHeight, 1};
    const auto& src_shape = buffers.find(input_id)->second;
    BHWC dst_shape = CalculateOutputShape(src_shape, params);
    GridParams grid_params;
    Params3x3 params3x3;
    Init3x3(params, int2(src_shape.w, src_shape.h),
            int2(dst_shape.w, dst_shape.h), &grid_params, &params3x3);
    if (params3x3.inner_size.x * params3x3.inner_size.y == 0) {
      return std::make_pair(groups_size, uint3{0, 0, 0});
    }
    int groups_x = IntegralDivideRoundUp(params3x3.inner_size.x, groups_size.x);
    int groups_y = IntegralDivideRoundUp(params3x3.inner_size.y, groups_size.y);
    int groups_z = 1;
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };

  return {border_desc, desc};
}

std::vector<ComputeTaskDescriptorPtr> ConvolutionTransposed4x4(
    int id, ValueId input_id, ValueId output_id,
    const ConvolutionTransposedAttributes& params,
    const RuntimeOptions& options) {
  const int src_depth = IntegralDivideRoundUp(params.weights.shape.i, 4);
  const int dst_depth = IntegralDivideRoundUp(params.weights.shape.o, 4);
  const int kernel_x = 4;
  const int kernel_y = 4;

  const int flt_count = kernel_x * kernel_y * src_depth * dst_depth * 4 * 4;
  std::vector<float> gpu_data(flt_count);

  const int remap[16] = {10, 11, 14, 15, 8, 9, 12, 13, 2, 3, 6, 7, 0, 1, 4, 5};

  int counter = 0;
  for (int d = 0; d < dst_depth; ++d) {
    for (int s = 0; s < src_depth; ++s) {
      for (int y = 0; y < kernel_y; ++y) {
        for (int x = 0; x < kernel_x; ++x) {
          const int kernel_index = remap[y * kernel_x + x];
          const int kernel_index_x = kernel_index % kernel_x;
          const int kernel_index_y = kernel_index / kernel_x;
          float4 filters[4];
          for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
              const int s_ch = s * 4 + i;
              const int d_ch = d * 4 + j;
              if (s_ch < params.weights.shape.i &&
                  d_ch < params.weights.shape.o) {
                const int f_index = params.weights.shape.LinearIndex(
                    {d_ch, kernel_index_y, kernel_index_x, s_ch});
                filters[j][i] = params.weights.data[f_index];
              } else {
                filters[j][i] = 0.0f;
              }
            }
          }
          for (int i = 0; i < 4; ++i) {
            gpu_data[counter++] = filters[i].x;
            gpu_data[counter++] = filters[i].y;
            gpu_data[counter++] = filters[i].z;
            gpu_data[counter++] = filters[i].w;
          }
        }
      }
    }
  }

  auto filters = GetByteBufferConverted(gpu_data, options.storage_precision);
  auto biases = GetByteBufferConvertedResized(
      params.bias.data, options.storage_precision, params.weights.shape.o);

  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;

  const auto gpu_type = GetGpuType();
  const bool powervr = gpu_type == GpuType::kA7 || gpu_type == GpuType::kA8 ||
                       gpu_type == GpuType::kA9 || gpu_type == GpuType::kA10;
  const bool recommended_2x =
      !powervr && options.storage_precision == RuntimeOptions::Precision::FP16;
  const bool use_local_mem = powervr;
  const int2 block_size(recommended_2x ? 2 : 1, 1);
  desc->shader_source = GetDeconvolution4x4(block_size, use_local_mem);

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {
      output_id, "device FLT4* dst_buffer",
      [input_id, params](const std::map<ValueId, BHWC>& buffers) {
        const auto& src_shape = buffers.find(input_id)->second;
        BHWC dst_shape = CalculateOutputShape(src_shape, params);
        return BHWC{src_shape.b, dst_shape.h, dst_shape.w, dst_shape.c};
      }};

  desc->immutable_buffers = {
      {"device FLT4* const filters", filters},
      {"device FLT4* const biases", biases},
  };

  desc->uniform_buffers = {
      {"constant uniforms& params",
       [input_id, output_id, params](const std::map<ValueId, BHWC>& buffers) {
         const auto& src_shape = buffers.find(input_id)->second;
         const auto& dst_shape = buffers.find(output_id)->second;
         const int src_depth = IntegralDivideRoundUp(src_shape.c, 4);
         std::vector<int> uniform_params{
             src_shape.w,
             src_shape.h,
             src_depth,
             src_shape.w * src_shape.h,
             dst_shape.w,
             dst_shape.h,
             IntegralDivideRoundUp(dst_shape.c, 4),
             0,
             4 * 16 * src_depth,
             0,
             0,
             0,
         };
         return GetByteBuffer(uniform_params);
       }},
  };

  desc->resize_function = [output_id, block_size,
                           params](const std::map<ValueId, BHWC>& buffers) {
    const auto& dst_shape = buffers.find(output_id)->second;
    const int grid_x = IntegralDivideRoundUp(dst_shape.w + 2, 2 * block_size.x);
    const int grid_y = IntegralDivideRoundUp(dst_shape.h + 2, 2 * block_size.y);
    const int grid_z = IntegralDivideRoundUp(dst_shape.c, 4);
    const uint3 group_size{8, 4, 1};
    int groups_x = IntegralDivideRoundUp(grid_x, group_size.x);
    int groups_y = IntegralDivideRoundUp(grid_y, group_size.y);
    int groups_z = IntegralDivideRoundUp(grid_z, group_size.z);
    return std::make_pair(group_size, uint3{groups_z, groups_x, groups_y});
  };

  return {desc};
}

bool CheckConvolutionTransposed4x4Support(
    const ConvolutionTransposedAttributes& attr) {
  return attr.weights.shape.w == 4 && attr.weights.shape.h == 4 &&
         attr.stride.w == 2 && attr.stride.h == 2 &&
         attr.padding.prepended.w == 1 && attr.padding.prepended.h == 1;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
