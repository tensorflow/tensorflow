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
kernel void ComputeFunction($0
                            uint3 ugid[[thread_position_in_grid]])
{
  int3 gid = int3(ugid.x * 4, ugid.y * 4, ugid.z);

  if (ugid.x >= args.tiles_x || ugid.y >= args.tiles_y) return;

  FLT4 I[6][6];
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x) {
      I[y][x] = FLT4(0.0f);
    }
  }
  const int src_base = gid.z * args.src_tensor.Height() * args.src_tensor.Width();
)";
  for (int y = 0; y < 6; ++y) {
    const std::string s_y = std::to_string(y);
    c += "  {\n";
    c += "    int coord_y = gid.y + " + s_y + " + args.padding_y;\n";
    c += "    bool in_y = FLT(coord_y >= 0 && coord_y < "
         "args.src_tensor.Height());\n";
    c += "    coord_y = clamp(coord_y, 0, args.src_tensor.Height() - 1);\n";
    c += "    const int src_adress_y = src_base + coord_y * "
         "args.src_tensor.Width();\n";
    for (int x = 0; x < 6; ++x) {
      const std::string s_x = std::to_string(x);
      c += "    {\n";
      c += "      int coord_x = gid.x + " + s_x + " + args.padding_x;\n";
      c += "      bool in_x = FLT(coord_x >= 0 && coord_x < "
           "args.src_tensor.Width());\n";
      c += "      FLT mult = FLT(in_y && in_x);\n";
      c += "      coord_x = clamp(coord_x, 0, args.src_tensor.Width() - 1);\n";
      c += "      FLT4 src = args.src_tensor.Read(src_adress_y + coord_x) * "
           "mult;\n";
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

  int dst_x = ugid.y * args.tiles_x + ugid.x;
  args.dst_tensor.GetAddress(dst_adress, dst_x, 0, gid.z);
  for (int y = 0; y < 6; ++y) {
    FLT4 value = I[y][0] + Bt[2] * I[y][2] + Bt[4] * I[y][4];
    args.dst_tensor.WriteLinear(value, dst_adress);
    dst_adress += args.dst_tensor.Width();
    value = Bt[7] * I[y][1] + Bt[8] * I[y][2] + Bt[9] * I[y][3] + Bt[10] * I[y][4];
    args.dst_tensor.WriteLinear(value, dst_adress);
    dst_adress += args.dst_tensor.Width();
    value = Bt[13] * I[y][1] + Bt[14] * I[y][2] + Bt[15] * I[y][3] + Bt[16] * I[y][4];
    args.dst_tensor.WriteLinear(value, dst_adress);
    dst_adress += args.dst_tensor.Width();
    value = Bt[19] * I[y][1] + Bt[20] * I[y][2] + Bt[21] * I[y][3] + Bt[22] * I[y][4];
    args.dst_tensor.WriteLinear(value, dst_adress);
    dst_adress += args.dst_tensor.Width();
    value = Bt[25] * I[y][1] + Bt[26] * I[y][2] + Bt[27] * I[y][3] + Bt[28] * I[y][4];
    args.dst_tensor.WriteLinear(value, dst_adress);
    dst_adress += args.dst_tensor.Width();
    value = Bt[31] * I[y][1] + Bt[33] * I[y][3] + I[y][5];
    args.dst_tensor.WriteLinear(value, dst_adress);
    dst_adress += args.dst_tensor.Width();
  }
}
)";
  return c;
}

std::string GetKernelWinograd4x4To36TileX6() {
  std::string c;
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

kernel void ComputeFunction($0
                            uint3 global_ids[[thread_position_in_grid]])
{
  int DST_X = global_ids.x;
  int DST_Y = global_ids.y;
  int DST_Z = global_ids.z;
  if (DST_X >= args.tiles_y || DST_Y >= 6 || DST_Z >= args.dst_tensor.Slices()) {
    return;
  }
  int tile_x = (DST_X % args.tiles_x) * 4;
  int tile_y = (DST_X / args.tiles_x) * 4;
  FLT4 I0, I1, I2, I3, I4, I5;
  FLT bt_ar[6];
  FLT4 t0 = args.bt_arr.Read(DST_Y * 2 + 0);
  FLT4 t1 = args.bt_arr.Read(DST_Y * 2 + 1);
  DST_Y *= 6;
  bt_ar[0] = t0.x;
  bt_ar[1] = t0.y;
  bt_ar[2] = t0.z;
  bt_ar[3] = t0.w;
  bt_ar[4] = t1.x;
  bt_ar[5] = t1.y;
)";
  auto read_src = [&](const std::string& src, const std::string& xs) {
    c += "    FLT4 " + src + " = args.src_tensor.Read(src_a_" + xs +
         " + offset) * m" + xs + "_x;\n";
  };
  for (int x = 0; x < 6; ++x) {
    const std::string xs = std::to_string(x);
    c += "  int xc" + xs + " = tile_x + args.padding_x + " + xs + ";\n";
    c += "  FLT m" + xs + "_x = xc" + xs + " >= 0 && xc" + xs +
         " < args.src_tensor.Width();\n";
    c += "  bool inx" + xs + " = (xc" + xs + " >= 0 && xc" + xs +
         " < args.src_tensor.Width());\n";
    c += "  xc" + xs + " = clamp(xc" + xs +
         ", 0, args.src_tensor.Width() - 1);\n";
    c += "  int src_a_" + xs +
         " = DST_Z * args.src_tensor.Width() * args.src_tensor.Height() + xc" +
         xs + ";\n";
  }
  c += "  {\n";
  c += "    int yc = tile_y + args.padding_y;\n";
  c += "    bool iny = (yc >= 0 && yc < args.src_tensor.Height());\n";
  c += "    yc = clamp(yc, 0, args.src_tensor.Height() - 1);\n";
  c += "    int offset = yc * args.src_tensor.Width();\n";
  c += "    FLT bt = bt_ar[0] * FLT(iny);\n";
  for (int x = 0; x < 6; ++x) {
    const std::string xs = std::to_string(x);
    const std::string src = "src" + xs;
    read_src(src, xs);
    c += "    I" + xs + " = bt * " + src + ";\n";
  }
  c += "  }\n";
  for (int y = 1; y < 6; ++y) {
    const std::string ys = std::to_string(y);
    c += "  {\n";
    c += "    int yc = tile_y + args.padding_y + (" + ys + ");\n";
    c += "    bool iny = (yc >= 0 && yc < args.src_tensor.Height());\n";
    c += "    yc = clamp(yc, 0, args.src_tensor.Height() - 1);\n";
    c += "    int offset = yc * args.src_tensor.Width();\n";
    c += "    FLT bt = bt_ar[" + ys + "] * FLT(iny);\n";
    for (int x = 0; x < 6; ++x) {
      const std::string xs = std::to_string(x);
      const std::string src = "src" + xs;
      read_src(src, xs);
      c += "    I" + xs + " += bt * " + src + ";\n";
    }
    c += "  }\n";
  }
  c += R"(
  {
    FLT4 r0 = I0 + Bt[2] * I2 + Bt[4] * I4;
    args.dst_tensor.Write(r0, DST_X, DST_Y, DST_Z);
    DST_Y++;
  }
  {
    FLT4 r0 = Bt[7] * I1 + Bt[8] * I2 + Bt[9] * I3 + Bt[10] * I4;
    args.dst_tensor.Write(r0, DST_X, DST_Y, DST_Z);
    DST_Y++;
  }
  {
    FLT4 r0 = Bt[13] * I1 + Bt[14] * I2 + Bt[15] * I3 + Bt[16] * I4;
    args.dst_tensor.Write(r0, DST_X, DST_Y, DST_Z);
    DST_Y++;
  }
  {
    FLT4 r0 = Bt[19] * I1 + Bt[20] * I2 + Bt[21] * I3 + Bt[22] * I4;
    args.dst_tensor.Write(r0, DST_X, DST_Y, DST_Z);
    DST_Y++;
  }
  {
    FLT4 r0 = Bt[25] * I1 + Bt[26] * I2 + Bt[27] * I3 + Bt[28] * I4;
    args.dst_tensor.Write(r0, DST_X, DST_Y, DST_Z);
    DST_Y++;
  }
  {
    FLT4 r0 = Bt[31] * I1 + Bt[33] * I3 + I5;
    args.dst_tensor.Write(r0, DST_X, DST_Y, DST_Z);
  }
}
)";
  return c;
}

std::string GetKernelWinograd36To4x4() {
  std::string c;
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

kernel void ComputeFunction($0
                            uint3 global_ids[[thread_position_in_grid]])
{
  int tile_id = global_ids.x;
  int Z = static_cast<int>(global_ids.z);
  int tiles_count_x = (args.dst_tensor.Width() + 3) / 4;
  int tile_x = (tile_id % tiles_count_x) * 4;
  int tile_y = (tile_id / tiles_count_x) * 4;
  if (tile_x >= args.dst_tensor.Width() || tile_y >= args.dst_tensor.Height()) return;

  int src_adress = Z * args.src_tensor.Height() * args.src_tensor.Width() + tile_id;
  FLT4 I[4][6];
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 6; ++x) {
      I[y][x] = 0.0f;
    }
  }
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x, src_adress += args.src_tensor.Width()) {
      FLT4 src = args.src_tensor.Read(src_adress);
      I[0][x] += src * At[y];
      I[1][x] += src * At[y + 6];
      I[2][x] += src * At[y + 12];
      I[3][x] += src * At[y + 18];
    }
  }

  FLT4 bias_val = args.biases.Read(Z);
  for (int y = 0; y < 4 && tile_y + y < args.dst_tensor.Height(); ++y) {
    FLT4 t0 = I[y][1] + I[y][2];
    FLT4 t1 = I[y][3] + I[y][4];
    if (tile_x < args.dst_tensor.Width()) {
      FLT4 value = I[y][0] + t0 + t1 + bias_val;
      args.dst_tensor.Write(value, tile_x, tile_y + y, global_ids.z);
    }
    FLT4 t2 = I[y][1] - I[y][2];
    FLT4 t3 = I[y][3] - I[y][4];
    if (tile_x + 1 < args.dst_tensor.Width()) {
      FLT4 value = t2 * At[7] + t3 * At[9] + bias_val;
      args.dst_tensor.Write(value, tile_x + 1, tile_y + y, global_ids.z);
    }
    if (tile_x + 2 < args.dst_tensor.Width()) {
      FLT4 value = t0 * At[13] + t1 * At[15] + bias_val;
      args.dst_tensor.Write(value, tile_x + 2, tile_y + y, global_ids.z);
    }
    if (tile_x + 3 < args.dst_tensor.Width()) {
      FLT4 value = t2 * At[19] + t3 * At[21] + I[y][5] + bias_val;
      args.dst_tensor.Write(value, tile_x + 3, tile_y + y, global_ids.z);
    }
  }
}
)";
  return c;
}

std::string GetKernelWinograd36To4x4Tile4x1() {
  std::string c;
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

kernel void ComputeFunction($0
                            uint3 global_ids[[thread_position_in_grid]])
{
  int tile_id = global_ids.x;
  int DST_Y = global_ids.y;
  int DST_Z = global_ids.z;
  int tile_x = (tile_id % args.tiles_x) * 4;
  int tile_y = (tile_id / args.tiles_x) * 4 + DST_Y;
  if (tile_x >= args.dst_tensor.Width() || tile_y >= args.dst_tensor.Height() || DST_Z >= args.dst_tensor.Slices()) {
    return;
  }
  FLT4 I0, I1, I2, I3, I4, I5;
  FLT at_ar[6];
  FLT4 t00 = args.at_arr.Read(DST_Y * 2 + 0);
  FLT4 t01 = args.at_arr.Read(DST_Y * 2 + 1);
  at_ar[0] = t00.x;
  at_ar[1] = t00.y;
  at_ar[2] = t00.z;
  at_ar[3] = t00.w;
  at_ar[4] = t01.x;
  at_ar[5] = t01.y;
  int src_adress = DST_Z * args.src_tensor.Height() * args.src_tensor.Width() + tile_id;
  int src_adress_final;
  {
    FLT at = at_ar[0];
)";
  for (int x = 0; x < 6; ++x) {
    const std::string yc = std::to_string(x);
    const std::string src = "src" + std::to_string(x);
    c += "    src_adress_final = src_adress + args.src_tensor.Width() * " + yc +
         ";\n";
    c += "    FLT4 " + src + " = args.src_tensor.Read(src_adress_final);\n";
    c += "    I" + std::to_string(x) + " = at * " + src + ";\n";
  }
  c += "  }\n";
  for (int y = 1; y < 6; ++y) {
    c += "  {\n";
    c += "    FLT at = at_ar[" + std::to_string(y) + "];\n";
    for (int x = 0; x < 6; ++x) {
      const std::string yc = std::to_string(y * 6 + x);
      const std::string src = "src" + std::to_string(x);
      c += "    src_adress_final = src_adress + args.src_tensor.Width() * " +
           yc + ";\n";
      c += "    FLT4 " + src + " = args.src_tensor.Read(src_adress_final);\n";
      c += "    I" + std::to_string(x) + " += at * " + src + ";\n";
    }
    c += "  }\n";
  }
  c += R"(
  FLT4 t0 = I1 + I2;
  FLT4 t1 = I3 + I4;
  FLT4 bias_val = args.biases.Read(DST_Z);
  if (tile_x < args.dst_tensor.Width()) {
    FLT4 value = I0 + t0 + t1 + bias_val;
    args.dst_tensor.Write(value, tile_x, tile_y, global_ids.z);
  }
  FLT4 t2 = I1 - I2;
  FLT4 t3 = I3 - I4;
  if (tile_x + 1 < args.dst_tensor.Width()) {
    FLT4 value = t2 * At[7] + t3 * At[9] + bias_val;
    args.dst_tensor.Write(value, tile_x + 1, tile_y, global_ids.z);
  }
  if (tile_x + 2 < args.dst_tensor.Width()) {
    FLT4 value = t0 * At[13] + t1 * At[15] + bias_val;
    args.dst_tensor.Write(value, tile_x + 2, tile_y, global_ids.z);
  }
  if (tile_x + 3 < args.dst_tensor.Width()) {
    FLT4 value = t2 * At[19] + t3 * At[21] + I5 + bias_val;
    args.dst_tensor.Write(value, tile_x + 3, tile_y, global_ids.z);
  }
}
)";
  return c;
}
}  // namespace

ComputeTaskDescriptor Winograd4x4To36(const OperationDef& definition,
                                      const Winograd4x4To36Attributes& attr) {
  ComputeTaskDescriptor desc(definition);
  desc.shader_source = GetKernelWinograd4x4To36();

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.args.AddInt("padding_x", -attr.padding.prepended.w);
  desc.args.AddInt("padding_y", -attr.padding.prepended.h);
  desc.args.AddInt("tiles_x");
  desc.args.AddInt("tiles_y");

  desc.update_function = {[attr](const std::vector<BHWC>& src_shapes,
                                 const std::vector<BHWC>& dst_shapes,
                                 ArgumentsBinder* args) -> absl::Status {
    int new_width = src_shapes[0].w + attr.padding.prepended.w +
                    attr.padding.appended.w - 2;
    int new_height = src_shapes[0].h + attr.padding.prepended.h +
                     attr.padding.appended.h - 2;
    int tiles_x = DivideRoundUp(new_width, 4);
    int tiles_y = DivideRoundUp(new_height, 4);
    RETURN_IF_ERROR(args->SetInt("tiles_x", tiles_x));
    RETURN_IF_ERROR(args->SetInt("tiles_y", tiles_y));
    return absl::OkStatus();
  }};

  desc.resize_function = [attr](const std::vector<BHWC>& src_shapes,
                                const std::vector<BHWC>& dst_shapes) {
    const uint3 groups_size{8, 4, 1};
    int new_width = src_shapes[0].w + attr.padding.prepended.w +
                    attr.padding.appended.w - 2;
    int new_height = src_shapes[0].h + attr.padding.prepended.h +
                     attr.padding.appended.h - 2;
    int grid_x = DivideRoundUp(new_width, 4);
    int grid_y = DivideRoundUp(new_height, 4);
    int grid_z = DivideRoundUp(src_shapes[0].c, 4);
    int groups_x = DivideRoundUp(grid_x, groups_size.x);
    int groups_y = DivideRoundUp(grid_y, groups_size.y);
    int groups_z = DivideRoundUp(grid_z, groups_size.z);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };
  return desc;
}

ComputeTaskDescriptor Winograd4x4To36TileX6(
    const OperationDef& definition, const Winograd4x4To36Attributes& attr) {
  ComputeTaskDescriptor desc(definition);
  desc.shader_source = GetKernelWinograd4x4To36TileX6();

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  std::vector<float> bt_aligned(6 * 8);
  auto bt_mat = BtMatrixForWinograd4x4To6x6();
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x) {
      bt_aligned[y * 8 + x] = bt_mat[y * 6 + x];
    }
    bt_aligned[y * 8 + 6] = 0.0f;
    bt_aligned[y * 8 + 7] = 0.0f;
  }

  auto data_type = DeduceDataTypeFromPrecision(definition.precision);
  BufferDescriptor buf_desc;
  buf_desc.element_type = data_type;
  buf_desc.element_size = 4;
  buf_desc.data = GetByteBufferConverted(bt_aligned, data_type);
  buf_desc.size = buf_desc.data.size();

  desc.args.AddObject("bt_arr",
                      absl::make_unique<BufferDescriptor>(std::move(buf_desc)));

  desc.args.AddInt("padding_x", -attr.padding.prepended.w);
  desc.args.AddInt("padding_y", -attr.padding.prepended.h);
  desc.args.AddInt("tiles_x");
  desc.args.AddInt("tiles_y");

  desc.update_function = {[attr](const std::vector<BHWC>& src_shapes,
                                 const std::vector<BHWC>& dst_shapes,
                                 ArgumentsBinder* args) -> absl::Status {
    int new_width = src_shapes[0].w + attr.padding.prepended.w +
                    attr.padding.appended.w - 2;
    int new_height = src_shapes[0].h + attr.padding.prepended.h +
                     attr.padding.appended.h - 2;
    int tiles_x = DivideRoundUp(new_width, 4);
    int tiles_y = DivideRoundUp(new_height, 4);
    RETURN_IF_ERROR(args->SetInt("tiles_x", tiles_x));
    RETURN_IF_ERROR(args->SetInt("tiles_y", tiles_x * tiles_y));
    return absl::OkStatus();
  }};

  desc.resize_function = [](const std::vector<BHWC>& src_shapes,
                            const std::vector<BHWC>& dst_shapes) {
    const uint3 groups_size{4, 6, 1};
    int grid_x = dst_shapes[0].w;
    int grid_y = 6;
    int grid_z = DivideRoundUp(dst_shapes[0].c, 4);
    int groups_x = DivideRoundUp(grid_x, groups_size.x);
    int groups_y = DivideRoundUp(grid_y, groups_size.y);
    int groups_z = DivideRoundUp(grid_z, groups_size.z);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };
  return desc;
}

ComputeTaskDescriptor Winograd36To4x4(const OperationDef& definition,
                                      const Winograd36To4x4Attributes& attr) {
  ComputeTaskDescriptor desc(definition);
  desc.shader_source = GetKernelWinograd36To4x4();

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  auto data_type = DeduceDataTypeFromPrecision(definition.precision);
  BufferDescriptor bias_desc;
  bias_desc.element_type = data_type;
  bias_desc.element_size = 4;
  bias_desc.data = GetByteBufferConvertedResized(
      attr.biases.data, data_type, AlignByN(attr.output_shape.c, 4));
  bias_desc.size = bias_desc.data.size();

  desc.args.AddObject(
      "biases", absl::make_unique<BufferDescriptor>(std::move(bias_desc)));

  desc.resize_function = [](const std::vector<BHWC>& src_shapes,
                            const std::vector<BHWC>& dst_shapes) {
    const uint3 groups_size{32, 1, 1};
    int grid_x = src_shapes[0].w;
    int grid_y = 1;
    int grid_z = DivideRoundUp(src_shapes[0].c, 4);
    int groups_x = DivideRoundUp(grid_x, groups_size.x);
    int groups_y = DivideRoundUp(grid_y, groups_size.y);
    int groups_z = DivideRoundUp(grid_z, groups_size.z);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };
  return desc;
}

ComputeTaskDescriptor Winograd36To4x4Tile4x1(
    const OperationDef& definition, const Winograd36To4x4Attributes& attr) {
  ComputeTaskDescriptor desc(definition);
  desc.shader_source = GetKernelWinograd36To4x4Tile4x1();

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  std::vector<float> at_aligned(4 * 8);
  auto at_mat = AtMatrixForWinograd4x4To6x6();
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 6; ++x) {
      at_aligned[y * 8 + x] = at_mat[y * 6 + x];
    }
    at_aligned[y * 8 + 6] = 0.0f;
    at_aligned[y * 8 + 7] = 0.0f;
  }

  auto data_type = DeduceDataTypeFromPrecision(definition.precision);
  BufferDescriptor bias_desc;
  bias_desc.element_type = data_type;
  bias_desc.element_size = 4;
  bias_desc.data = GetByteBufferConvertedResized(
      attr.biases.data, data_type, AlignByN(attr.output_shape.c, 4));
  bias_desc.size = bias_desc.data.size();

  desc.args.AddObject(
      "biases", absl::make_unique<BufferDescriptor>(std::move(bias_desc)));

  BufferDescriptor buf_desc;
  buf_desc.element_type = data_type;
  buf_desc.element_size = 4;
  buf_desc.data = GetByteBufferConverted(at_aligned, data_type);
  buf_desc.size = buf_desc.data.size();

  desc.args.AddObject("at_arr",
                      absl::make_unique<BufferDescriptor>(std::move(buf_desc)));

  desc.args.AddInt("tiles_x");
  desc.args.AddInt("tiles_y");

  desc.update_function = {[attr](const std::vector<BHWC>& src_shapes,
                                 const std::vector<BHWC>& dst_shapes,
                                 ArgumentsBinder* args) -> absl::Status {
    const int tiles_x = DivideRoundUp(dst_shapes[0].w, 4);
    const int tiles_y = DivideRoundUp(dst_shapes[0].h, 4);
    RETURN_IF_ERROR(args->SetInt("tiles_x", tiles_x));
    RETURN_IF_ERROR(args->SetInt("tiles_y", tiles_y));
    return absl::OkStatus();
  }};

  desc.resize_function = [](const std::vector<BHWC>& src_shapes,
                            const std::vector<BHWC>& dst_shapes) {
    const uint3 groups_size{8, 4, 1};
    const int tiles_x = DivideRoundUp(dst_shapes[0].w, 4);
    const int tiles_y = DivideRoundUp(dst_shapes[0].h, 4);
    int grid_x = tiles_x * tiles_y;
    int grid_y = 4;
    int grid_z = DivideRoundUp(dst_shapes[0].c, 4);
    int groups_x = DivideRoundUp(grid_x, groups_size.x);
    int groups_y = DivideRoundUp(grid_y, groups_size.y);
    int groups_z = DivideRoundUp(grid_z, groups_size.z);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };
  return desc;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
