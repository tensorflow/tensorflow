/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/tasks/tile.h"

#include <string>
#include <utility>
#include <vector>

namespace tflite {
namespace gpu {

namespace {
std::string GetTileCode(const OperationDef& op_def, bool src_channels_x4) {
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id = GLOBAL_ID_1;\n";
    c += "  int Y = linear_id / args.dst_tensor.Depth();\n";
    c += "  int Z = linear_id % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = GLOBAL_ID_1;\n";
  }
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  std::string dst_coords = "X, Y";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    dst_coords += ", Z";
  }
  dst_coords += ", S";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    dst_coords += ", B";
  }
  std::string src_coords = "src_x, src_y";
  if (op_def.src_tensors[0].HasAxis(Axis::DEPTH)) {
    src_coords += ", src_z";
  }
  src_coords += ", src_s";
  if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
    src_coords += ", src_b";
  }
  c += "  int src_x = X % args.src_tensor.Width();\n";
  c += "  int src_y = Y % args.src_tensor.Height();\n";
  if (op_def.src_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int src_z = Z % args.src_tensor.Depth();\n";
  }
  if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int src_b = B % args.src_tensor.Batch();\n";
  }
  if (src_channels_x4) {
    c += "  int src_s = S % args.src_tensor.Slices();\n";
    c += "  args.src_tensor::type result = args.src_tensor.Read(" + src_coords +
         ");\n";
  } else {
    c += "  args.src_tensor::scalar_type tmp[4];\n";
    c += "  tmp[0] = args.src_tensor::scalar_zero_value;\n";
    c += "  tmp[1] = args.src_tensor::scalar_zero_value;\n";
    c += "  tmp[2] = args.src_tensor::scalar_zero_value;\n";
    c += "  tmp[3] = args.src_tensor::scalar_zero_value;\n";
    c += "  for (int i = 0; i < 4; ++i) {\n";
    c += "    int dst_c = 4 * S + i;\n";
    c += "    int src_s = dst_c % args.src_tensor.Channels();\n";
    c += "    args.src_tensor.ReadPerChannel(tmp[i], " + src_coords + ");\n";
    c += "  }\n";
    c += "  args.src_tensor::type result;\n";
    c += "  result.x = tmp[0];\n";
    c += "  result.y = tmp[1];\n";
    c += "  result.z = tmp[2];\n";
    c += "  result.w = tmp[3];\n";
  }
  c += "  args.dst_tensor.Write(result, " + dst_coords + ");\n";
  c += "}\n";
  return c;
}
}  // namespace

GPUOperation CreateTile(const OperationDef& op_def, int src_channels) {
  GPUOperation op(op_def);
  op.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  op.code_ = GetTileCode(op_def, src_channels % 4 == 0);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
