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

#include "tensorflow/lite/delegates/gpu/common/tasks/reshape.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {
std::string GetReshapeCode(const OperationDef& op_def) {
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int Z = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  FLT temps[4];\n";
  c += "  temps[0] = INIT_FLT(0.0f);\n";
  c += "  temps[1] = INIT_FLT(0.0f);\n";
  c += "  temps[2] = INIT_FLT(0.0f);\n";
  c += "  temps[3] = INIT_FLT(0.0f);\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int base = B;\n";
  } else {
    c += "  int base = 0;\n";
  }
  c += "  base = ((base * args.dst_tensor.Height() + Y) * "
       "args.dst_tensor.Width() + X) * args.dst_tensor.Channels() + Z * 4;\n";
  c += "  for (int i = 0; i < 4; ++i) {\n";
  c += "    int dst_channel = Z * 4 + i;\n";
  c += "    if (dst_channel < args.dst_tensor.Channels()) {;\n";
  c += "      int p = base + i;\n";
  c += "      int src_c = p % args.src_tensor.Channels();\n";
  c += "      p = p / args.src_tensor.Channels();\n";
  c += "      int src_x = p % args.src_tensor.Width();\n";
  c += "      p = p / args.src_tensor.Width();\n";
  c += "      int src_y = p % args.src_tensor.Height();\n";
  if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int src_b = p / args.src_tensor.Height();\n";
    c += "  args.src_tensor.SetBatchRef(src_b);\n";
  }
  c += "      int src_z = src_c / 4;\n";
  c += "      int src_sub_ch = src_c % 4;\n";
  c += "      FLT4 t = args.src_tensor.Read(src_x, src_y, src_z);\n";
  c += "      FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
  c += "      temps[i] = t_ar[src_sub_ch];\n";
  c += "    }\n";
  c += "  }\n";
  c += "  FLT4 result;\n";
  c += "  result.x = temps[0];\n";
  c += "  result.y = temps[1];\n";
  c += "  result.z = temps[2];\n";
  c += "  result.w = temps[3];\n";
  c += "  args.dst_tensor.Write(result, X, Y, Z);\n";
  c += "}\n";
  return c;
}

}  // namespace

GPUOperation CreateReshape(const OperationDef& definition) {
  GPUOperation op(definition);
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.code_ = GetReshapeCode(definition);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
