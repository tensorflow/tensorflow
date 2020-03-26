/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/metal/kernels/winograd.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/common/winograd_util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {
std::string GetKernelWinograd4x4To36() {
  std::string c;
  c += R"(
#include <metal_stdlib>
using namespace metal;

struct uniforms {
    int4 src_size;
    int4 dst_size;
    int2 padding;
    int2 dummy0;
};
)";
  auto bt_mat = BtMatrixForWinograd4x4To6x6();
  c += "constant FLT Bt[36] = {\n";
  for (int y = 0; y < 6; ++y) {
    c += "\t";
    for (int x = 0; x < 6; ++x) {
      c += absl::StrFormat("%.10f", bt_mat[y * 6 + x]) + "f, ";
    }
    c += "\n";
  }
  c += "};\n";
  c += R"(

$0

kernel void ComputeFunction($1
                            uint3 ugid[[thread_position_in_grid]])
{
  int3 gid = int3(ugid.x * 4, ugid.y * 4, ugid.z);

  if (gid.x >= U.src_size.x || gid.y >= U.src_size.y) return;

  FLT4 I[6][6];
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x) {
      I[y][x] = FLT4(0.0f);
    }
  }
  const int src_base = gid.z * U.src_size.y * U.src_size.x;
)";
  for (int y = 0; y < 6; ++y) {
    const std::string s_y = std::to_string(y);
    c += "  {\n";
    c += "    int coord_y = gid.y + " + s_y + " + U.padding.y;\n";
    c += "    bool in_y = FLT(coord_y >= 0 && coord_y < U.src_size.y);\n";
    c += "    coord_y = clamp(coord_y, 0, U.src_size.y - 1);\n";
    c += "    const int src_adress_y = src_base + coord_y * U.src_size.x;\n";
    for (int x = 0; x < 6; ++x) {
      const std::string s_x = std::to_string(x);
      c += "    {\n";
      c += "      int coord_x = gid.x + " + s_x + " + U.padding.x;\n";
      c += "      bool in_x = FLT(coord_x >= 0 && coord_x < U.src_size.x);\n";
      c += "      FLT mult = FLT(in_y && in_x);\n";
      c += "      coord_x = clamp(coord_x, 0, U.src_size.x - 1);\n";
      c += "      FLT4 src = src_buffer[src_adress_y + coord_x] * mult;\n";
      c += "      I[0][" + s_x + "] += Bt[" + std::to_string(y) + "] * src;\n";
      c += "      I[1][" + s_x + "] += Bt[" + std::to_string(y + 6) +
           "] * src;\n";
      c += "      I[2][" + s_x + "] += Bt[" + std::to_string(y + 12) +
           "] * src;\n";
      c += "      I[3][" + s_x + "] += Bt[" + std::to_string(y + 18) +
           "] * src;\n";
      c += "      I[4][" + s_x + "] += Bt[" + std::to_string(y + 24) +
           "] * src;\n";
      c += "      I[5][" + s_x + "] += Bt[" + std::to_string(y + 30) +
           "] * src;\n";
      c += "    }\n";
    }
    c += "  }\n";
  }
  c += R"(

  int dst_x = ugid.y * (U.src_size.x + 3) / 4 + ugid.x;
  int dst_adress = gid.z * U.dst_size.y * U.dst_size.x + dst_x;
  for (int y = 0; y < 6; ++y) {
    dst_buffer[dst_adress] = I[y][0] + Bt[2] * I[y][2] + Bt[4] * I[y][4];
    dst_adress += U.dst_size.x;
    dst_buffer[dst_adress] = Bt[7] * I[y][1] + Bt[8] * I[y][2] + Bt[9] * I[y][3] + Bt[10] * I[y][4];
    dst_adress += U.dst_size.x;
    dst_buffer[dst_adress] = Bt[13] * I[y][1] + Bt[14] * I[y][2] + Bt[15] * I[y][3] + Bt[16] * I[y][4];
    dst_adress += U.dst_size.x;
    dst_buffer[dst_adress] = Bt[19] * I[y][1] + Bt[20] * I[y][2] + Bt[21] * I[y][3] + Bt[22] * I[y][4];
    dst_adress += U.dst_size.x;
    dst_buffer[dst_adress] = Bt[25] * I[y][1] + Bt[26] * I[y][2] + Bt[27] * I[y][3] + Bt[28] * I[y][4];
    dst_adress += U.dst_size.x;
    dst_buffer[dst_adress] = Bt[31] * I[y][1] + Bt[33] * I[y][3] + I[y][5];
    dst_adress += U.dst_size.x;
  }
}
)";
  return c;
}

std::string GetKernelWinograd36To4x4() {
  std::string c;
  c += R"(
#include <metal_stdlib>
using namespace metal;

struct uniforms {
    int4 src_size;
    int4 dst_size;
};
)";
  auto at_mat = AtMatrixForWinograd4x4To6x6();
  c += "constant FLT At[24] = {\n";
  for (int y = 0; y < 4; ++y) {
    c += "\t";
    for (int x = 0; x < 6; ++x) {
      c += absl::StrFormat("%.10f", at_mat[y * 6 + x]) + "f, ";
    }
    c += "\n";
  }
  c += "};\n";
  c += R"(

$0

kernel void ComputeFunction($1
                            uint3 global_ids[[thread_position_in_grid]])
{
  int3 gid = int3(global_ids.x, global_ids.y, global_ids.z);

  int tile_id = global_ids.x;
  int tiles_count_x = (U.dst_size.x + 3) / 4;
  int tile_x = (tile_id % tiles_count_x) * 4;
  int tile_y = (tile_id / tiles_count_x) * 4;
  if (tile_x >= U.dst_size.x || tile_y >= U.dst_size.y) return;

  int src_adress = gid.z * U.src_size.y * U.src_size.x + gid.x;
  FLT4 I[4][6];
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 6; ++x) {
      I[y][x] = 0.0f;
    }
  }
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x, src_adress += U.src_size.x) {
      FLT4 src = src_buffer[src_adress];
      I[0][x] += src * At[y];
      I[1][x] += src * At[y + 6];
      I[2][x] += src * At[y + 12];
      I[3][x] += src * At[y + 18];
    }
  }

  FLT4 bias_val = biases[gid.z];
  int dst_adress = (gid.z * U.dst_size.y + tile_y) * U.dst_size.x + tile_x;
  for (int y = 0; y < 4 && tile_y + y < U.dst_size.y; ++y) {
    FLT4 t0 = I[y][1] + I[y][2];
    FLT4 t1 = I[y][3] + I[y][4];
    if (tile_x < U.dst_size.x) {
      FLT4 value = I[y][0] + t0 + t1 + bias_val;
      int linear_index = dst_adress;
      uint3 ugid = uint3(tile_x, tile_y + y, global_ids.z);
      $2
      dst_buffer[linear_index] = value;
    }
    FLT4 t2 = I[y][1] - I[y][2];
    FLT4 t3 = I[y][3] - I[y][4];
    if (tile_x + 1 < U.dst_size.x) {
      FLT4 value = t2 * At[7] + t3 * At[9] + bias_val;
      int linear_index = dst_adress + 1;
      uint3 ugid = uint3(tile_x + 1, tile_y + y, global_ids.z);
      $2
      dst_buffer[linear_index] = value;
    }
    if (tile_x + 2 < U.dst_size.x) {
      FLT4 value = t0 * At[13] + t1 * At[15] + bias_val;
      int linear_index = dst_adress + 2;
      uint3 ugid = uint3(tile_x + 2, tile_y + y, global_ids.z);
      $2
      dst_buffer[linear_index] = value;
    }
    if (tile_x + 3 < U.dst_size.x) {
      FLT4 value = t2 * At[19] + t3 * At[21] + I[y][5] + bias_val;
      uint3 ugid = uint3(tile_x + 3, tile_y + y, global_ids.z);
      int linear_index = dst_adress + 3;
      $2
      dst_buffer[linear_index] = value;
    }
    dst_adress += U.dst_size.x;
  }
}
)";
  return c;
}
}  // namespace

std::vector<ComputeTaskDescriptorPtr> Winograd4x4To36(
    int id, ValueId input_id, ValueId output_id,
    const Winograd4x4To36Attributes& attr) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source = GetKernelWinograd4x4To36();

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {output_id, "device FLT4* dst_buffer",
                         [input_id](const std::map<ValueId, BHWC>& buffers) {
                           const auto src_shape =
                               buffers.find(input_id)->second;
                           BHWC dst_shape;
                           dst_shape.b = src_shape.b;
                           dst_shape.h = 36;
                           dst_shape.w = IntegralDivideRoundUp(src_shape.w, 4) *
                                         IntegralDivideRoundUp(src_shape.h, 4);
                           dst_shape.c = src_shape.c;
                           return dst_shape;
                         }};

  desc->uniform_buffers = {
      {"constant uniforms& U",
       [input_id, output_id, attr](const std::map<ValueId, BHWC>& buffers) {
         const auto& src_shape = buffers.find(input_id)->second;
         const auto& dst_shape = buffers.find(output_id)->second;
         std::vector<int> sizes = {
             src_shape.w,
             src_shape.h,
             IntegralDivideRoundUp(src_shape.c, 4),
             0,
             dst_shape.w,
             dst_shape.h,
             IntegralDivideRoundUp(dst_shape.c, 4),
             0,
             -attr.padding.prepended.w,
             -attr.padding.prepended.h,
             0,
             0,
         };
         return GetByteBuffer(sizes);
       }},
  };

  desc->resize_function = [input_id](const std::map<ValueId, BHWC>& buffers) {
    const uint3 groups_size{8, 4, 1};
    const auto& src_shape = buffers.find(input_id)->second;
    int grid_x = IntegralDivideRoundUp(src_shape.w, 4);
    int grid_y = IntegralDivideRoundUp(src_shape.h, 4);
    int grid_z = IntegralDivideRoundUp(src_shape.c, 4);
    int groups_x = IntegralDivideRoundUp(grid_x, groups_size.x);
    int groups_y = IntegralDivideRoundUp(grid_y, groups_size.y);
    int groups_z = IntegralDivideRoundUp(grid_z, groups_size.z);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };
  return {desc};
}

std::vector<ComputeTaskDescriptorPtr> Winograd36To4x4(
    int id, ValueId input_id, ValueId output_id, const RuntimeOptions& options,
    const Winograd36To4x4Attributes& attr) {
  auto desc = std::make_shared<ComputeTaskDescriptor>();
  desc->id = id;
  desc->is_linkable = false;
  desc->shader_source = GetKernelWinograd36To4x4();

  desc->input_buffers = {
      {input_id, "device FLT4* const src_buffer"},
  };

  desc->output_buffer = {
      output_id, "device FLT4* dst_buffer",
      [input_id, attr](const std::map<ValueId, BHWC>& buffers) {
        const auto src_shape = buffers.find(input_id)->second;
        BHWC dst_shape;
        dst_shape.b = src_shape.b;
        dst_shape.h = attr.output_shape.h;
        dst_shape.w = attr.output_shape.w;
        dst_shape.c = src_shape.c;
        return dst_shape;
      }};

  desc->immutable_buffers = {
      {"device FLT4* const biases",
       GetByteBufferConvertedResized(attr.biases.data,
                                     options.storage_precision,
                                     AlignByN(attr.output_shape.c, 4))},
  };

  desc->uniform_buffers = {
      {"constant uniforms& U",
       [input_id, output_id](const std::map<ValueId, BHWC>& buffers) {
         const auto& src_shape = buffers.find(input_id)->second;
         const auto& dst_shape = buffers.find(output_id)->second;
         std::vector<int> sizes = {
             src_shape.w, src_shape.h, IntegralDivideRoundUp(src_shape.c, 4), 0,
             dst_shape.w, dst_shape.h, IntegralDivideRoundUp(dst_shape.c, 4), 0,
         };
         return GetByteBuffer(sizes);
       }},
  };

  desc->resize_function = [input_id](const std::map<ValueId, BHWC>& buffers) {
    const uint3 groups_size{32, 1, 1};
    const auto& src_shape = buffers.find(input_id)->second;
    int grid_x = src_shape.w;
    int grid_y = 1;
    int grid_z = IntegralDivideRoundUp(src_shape.c, 4);
    int groups_x = IntegralDivideRoundUp(grid_x, groups_size.x);
    int groups_y = IntegralDivideRoundUp(grid_y, groups_size.y);
    int groups_z = IntegralDivideRoundUp(grid_z, groups_size.z);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };
  return {desc};
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
