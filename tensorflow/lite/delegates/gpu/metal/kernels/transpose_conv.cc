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
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

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

    constant int src_depth = $1;
    constant int dst_depth = $2;
    constant int dst_channels = $3;
    constant int dst_channels_aligned = $4;

    $5

    $$0
    kernel void ComputeFunction(
                                $$1
                                uint2 ugid[[thread_position_in_grid]]) {
      if (static_cast<int>(ugid.x) >= args.dst_tensor.Width() ||
          static_cast<int>(ugid.y) >= args.dst_tensor.Height()) {
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
            src_coord.x >= args.src_tensor.Width() || src_coord.y >= args.src_tensor.Height();
          const int kernel_index = index.y * kernel_size.x + index.x;
          device FLT4* weights_cache = args.weights.GetPtr() + kernel_index * $0;
          bool belong = inside_kernel && !outside;
          if (belong) {
            for (int l = 0; l < src_depth; ++l) {
              FLT4 srcColor = args.src_tensor.Read(src_coord.x, src_coord.y, l);
              for (int k = 0; k < dst_channels; ++k) {
                out[k] += dot(srcColor, weights_cache[l * dst_channels_aligned + k]);
              }
            }
          }
        }
      }

      for (short l = 0; l < dst_depth; ++l) {
        FLT4 value = FLT4(out[l * 4], out[l * 4 + 1], out[l * 4 + 2], out[l * 4 + 3]) + args.biases.Read(l);
        args.dst_tensor.GetAddress(linear_index, ugid.x, ugid.y, l);
        $$2
        args.dst_tensor.Write(value, ugid.x, ugid.y, l);
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
  const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
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

    constant int src_depth = $1;
    constant int dst_depth = $2;
    constant int dst_channels = $3;
    constant int dst_channels_aligned = $4;

    $5

    constant short2 src_local_size = {$6, $7};

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
            src_coord.x >= args.src_tensor.Width() || src_coord.y >= args.src_tensor.Height();
          FLT4 src = !outside ? args.src_tensor.Read(src_coord.x, src_coord.y, z) : FLT4(0.0f);
          src_shared[tid.x][tid.y][z] = src;
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (static_cast<int>(ugid.x) >= args.dst_tensor.Width() ||
          static_cast<int>(ugid.y) >= args.dst_tensor.Height()) {
        return;
      }

      for (int ky = 0; ky < inner_size.y; ++ky) {
        for (int kx = 0; kx < inner_size.x; ++kx) {
          short2 index = short2(kx, ky) * stride + f_offset;
          bool inside_kernel = index.x < kernel_size.x && index.y < kernel_size.y;
          const short2 src_coord = (short2(ugid) + index + padding - kernel_offset) / stride;
          index = kernel_size - short2(1, 1) - index;
          bool outside = src_coord.x < 0 || src_coord.y < 0 ||
            src_coord.x >= args.src_tensor.Width() || src_coord.y >= args.src_tensor.Height();
          const int kernel_index = index.y * kernel_size.x + index.x;
          device FLT4* weights_cache = args.weights.GetPtr() + kernel_index * $0;
          bool belong = inside_kernel && !outside;
          if (belong) {
            for (int k = 0; k < dst_channels; ++k) {
              for (int l = 0; l < src_depth; ++l) {
                short2 src_index = src_coord - first_src_coord;
                out[k] += dot(src_shared[src_index.x][src_index.y][l],
                              weights_cache[l * dst_channels_aligned + k]);
              }
            }
          }
        }
      }

      for (short l = 0; l < dst_depth; ++l) {
        FLT4 value = FLT4(out[l * 4], out[l * 4 + 1], out[l * 4 + 2], out[l * 4 + 3]) + args.biases.Read(l);
        args.dst_tensor.GetAddress(linear_index, ugid.x, ugid.y, l);
        $$2
        args.dst_tensor.Write(value, ugid.x, ugid.y, l);
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
  const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
  const int dst_channels_aligned = AlignByN(attr.weights.shape.o, 4);
  const int src_local_size_x = (workgroup_x + kernel_x) / attr.stride.w;
  const int src_local_size_y = (workgroup_y + kernel_y) / attr.stride.h;
  return absl::Substitute(
      shader_source, src_depth * dst_channels_aligned, src_depth, dst_depth,
      attr.weights.shape.o, dst_channels_aligned, constant_args_inplaced,
      src_local_size_x, src_local_size_y, workgroup_x, workgroup_y);
}

std::string GetDeconvolution4x4(const int2& block_size,
                                const GpuInfo& gpu_info) {
  bool use_local_mem = false;
  if (gpu_info.IsApple() && gpu_info.apple_info.IsBionic()) {
    use_local_mem = true;
  }
  if (gpu_info.IsIntel()) {
    use_local_mem = true;
  }
  const std::string barrier = gpu_info.IsWaveSizeEqualTo32()
                                  ? "SIMDGROUP_BARRIER"
                                  : "threadgroup_barrier";
  std::string c = R"(
#include <metal_stdlib>
using namespace metal;
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
    c += "  if (X * 2 > args.dst_tensor.Width() || Y * 2 > "
         "args.dst_tensor.Height() || Z >= args.dst_tensor.Slices()) return;\n";
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
  c += "  int f_offset = Z * 64 * args.src_tensor.Slices();\n";
  if (use_local_mem) {
    c += "  threadgroup FLT4 weights_cache[64];\n";
    c += "  int local_id = static_cast<int>(tid3d.y * 8u + tid3d.x);\n";
  }
  for (int x = 0; x < block_size.x + 1; ++x) {
    const std::string sx = std::to_string(x);
    const std::string xc =
        x == 0 ? std::string("X - 1") : "X + " + std::to_string(x - 1);
    c += "  bool in_x" + sx + " = " + xc + " >= 0 && " + xc +
         " < args.src_tensor.Width();\n";
    c += "  int xc" + sx + " = clamp(" + xc +
         ", 0, args.src_tensor.Width() - 1);\n";
  }
  for (int y = 0; y < block_size.y + 1; ++y) {
    const std::string sy = std::to_string(y);
    const std::string yc =
        y == 0 ? std::string("Y - 1") : "Y + " + std::to_string(y - 1);
    c += "  bool in_y" + std::to_string(y) + " = " + yc + " >= 0 && " + yc +
         " < args.src_tensor.Height();\n";
    c += "  int yc" + sy + " = clamp(" + yc +
         ", 0, args.src_tensor.Height() - 1);\n";
  }
  for (int y = 0; y < block_size.y + 1; ++y) {
    for (int x = 0; x < block_size.x + 1; ++x) {
      const std::string sx = std::to_string(x);
      const std::string sy = std::to_string(y);
      c += "  FLT m_" + sx + sy + " = in_x" + sx + " && in_y" + sy + ";\n";
      c += "  device FLT4* src_ptr_" + sx + sy +
           " = args.src_tensor.GetHandle() + args.src_tensor.GetWHOffset(xc" +
           sx + ", yc" + sy + ");\n";
    }
  }
  c += "  for (int s = 0; s < args.src_tensor.Slices(); ++s) {\n";
  if (use_local_mem) {
    c += "    " + barrier + "(mem_flags::mem_none);\n";
    c += "    weights_cache[local_id] = args.weights.Read(f_offset + "
         "local_id);\n";
    c += "    weights_cache[local_id + 32] = args.weights.Read(f_offset + "
         "local_id + 32);\n";
  } else {
    c += "    device FLT4* weights_cache = args.weights.GetPtr() + f_offset;\n";
  }
  for (int y = 0; y < block_size.y + 1; ++y) {
    for (int x = 0; x < block_size.x + 1; ++x) {
      const std::string id = std::to_string(x) + std::to_string(y);
      c += "    FLT4 src_" + id + " = *src_ptr_" + id + " * m_" + id +
           "; src_ptr_" + id + " += args.src_tensor.SliceStride();\n";
    }
  }
  c += "    f_offset += 64;\n";
  if (use_local_mem) {
    c += "    " + barrier + "(mem_flags::mem_threadgroup);\n";
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
    c += "  if (X * 2 > args.dst_tensor.Width() || Y * 2 > "
         "args.dst_tensor.Height() || Z >= args.dst_tensor.Slices()) return;\n";
  }
  c += "  X = X * 2 - 1;\n";
  c += "  Y = Y * 2 - 1;\n";
  c += "  FLT4 bias_val = args.biases.Read(Z);\n";
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
          const std::string x_check =
              x_offset == 0 ? std::string("X >= 0")
                            : dst_x + " < args.dst_tensor.Width()";
          const std::string y_check =
              y_offset == 0 ? std::string("Y >= 0")
                            : dst_y + " < args.dst_tensor.Height()";
          c += "  if (" + x_check + " && " + y_check + ") {\n";
          c += "    FLT4 value = FLT4(" + R + ") + bias_val;\n";
          std::string dst_coords = dst_x + ", " + dst_y + ", Z";
          c += "    args.dst_tensor.GetAddress(linear_index, " + dst_coords +
               ");\n";
          c += "    uint3 gid = uint3(" + dst_coords + ");\n";
          c += "    $2\n";
          c += "    args.dst_tensor.Write(value, " + dst_coords + ");\n";
          c += "  }\n";
        }
      }
    }
  }
  c += "}\n";
  return c;
}

}  // namespace

ComputeTaskDescriptor ConvolutionTransposed(
    const OperationDef& definition,
    const ConvolutionTransposedAttributes& params, const GpuInfo& gpu_info) {
  ComputeTaskDescriptor desc(definition);
  desc.tensors_as_args = true;

  const int src_local_size_x =
      (kThreadGroupWidth + params.weights.shape.w) / params.stride.w;
  const int src_local_size_y =
      (kThreadGroupHeight + params.weights.shape.h) / params.stride.h;
  const int src_depth = DivideRoundUp(params.weights.shape.i, 4);
  const int shared_size =
      sizeof(float) * 4 * src_depth * src_local_size_x * src_local_size_y;
  if (shared_size < 1000 * 16 &&
      gpu_info.apple_info.IsLocalMemoryPreferredOverGlobal()) {
    desc.shader_source =
        GetDeconvolutionShared(params, kThreadGroupWidth, kThreadGroupHeight);
  } else {
    desc.shader_source = GetDeconvolution(params);
  }

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

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

  auto data_type = DeduceDataTypeFromPrecision(definition.precision);
  BufferDescriptor weights_desc;
  weights_desc.element_type = data_type;
  weights_desc.element_size = 4;
  weights_desc.data = GetByteBufferConverted(filters_reordered, data_type);
  weights_desc.size = weights_desc.data.size();
  desc.args.AddObject(
      "weights", absl::make_unique<BufferDescriptor>(std::move(weights_desc)));

  BufferDescriptor bias_desc;
  bias_desc.element_type = data_type;
  bias_desc.element_size = 4;
  bias_desc.data = GetByteBufferConvertedResized(params.bias.data, data_type,
                                                 params.weights.shape.o);
  bias_desc.size = bias_desc.data.size();
  desc.args.AddObject(
      "biases", absl::make_unique<BufferDescriptor>(std::move(bias_desc)));

  desc.resize_function = [](const std::vector<BHWC>& src_shapes,
                            const std::vector<BHWC>& dst_shapes) {
    const uint3 groups_size{kThreadGroupWidth, kThreadGroupHeight, 1};
    int groups_x = DivideRoundUp(dst_shapes[0].w, groups_size.x);
    int groups_y = DivideRoundUp(dst_shapes[0].h, groups_size.y);
    int groups_z = 1;
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };

  return desc;
}

ComputeTaskDescriptor ConvolutionTransposed4x4(
    const OperationDef& definition,
    const ConvolutionTransposedAttributes& params, const GpuInfo& gpu_info) {
  const int src_depth = DivideRoundUp(params.weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(params.weights.shape.o, 4);
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

  auto data_type = DeduceDataTypeFromPrecision(definition.precision);
  auto filters = GetByteBufferConverted(gpu_data, data_type);
  auto biases = GetByteBufferConvertedResized(params.bias.data, data_type,
                                              params.weights.shape.o);

  ComputeTaskDescriptor desc(definition);
  desc.tensors_as_args = true;

  bool recommended_2x = false;
  if (gpu_info.IsApple()) {
    if (gpu_info.apple_info.IsBionic() &&
        definition.precision == CalculationsPrecision::F16) {
      recommended_2x = true;
    }
  } else {
    if (definition.precision == CalculationsPrecision::F16) {
      recommended_2x = true;
    }
  }

  const int2 block_size(recommended_2x ? 2 : 1, 1);
  desc.shader_source = GetDeconvolution4x4(block_size, gpu_info);

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  BufferDescriptor weights_desc;
  weights_desc.element_type = data_type;
  weights_desc.element_size = 4;
  weights_desc.data = filters;
  weights_desc.size = weights_desc.data.size();
  desc.args.AddObject(
      "weights", absl::make_unique<BufferDescriptor>(std::move(weights_desc)));

  BufferDescriptor bias_desc;
  bias_desc.element_type = data_type;
  bias_desc.element_size = 4;
  bias_desc.data = biases;
  bias_desc.size = bias_desc.data.size();
  desc.args.AddObject(
      "biases", absl::make_unique<BufferDescriptor>(std::move(bias_desc)));

  desc.resize_function = [block_size](const std::vector<BHWC>& src_shapes,
                                      const std::vector<BHWC>& dst_shapes) {
    const int grid_x = DivideRoundUp(dst_shapes[0].w + 2, 2 * block_size.x);
    const int grid_y = DivideRoundUp(dst_shapes[0].h + 2, 2 * block_size.y);
    const int grid_z = DivideRoundUp(dst_shapes[0].c, 4);
    const uint3 group_size{8, 4, 1};
    int groups_x = DivideRoundUp(grid_x, group_size.x);
    int groups_y = DivideRoundUp(grid_y, group_size.y);
    int groups_z = DivideRoundUp(grid_z, group_size.z);
    return std::make_pair(group_size, uint3{groups_z, groups_x, groups_y});
  };

  return desc;
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
