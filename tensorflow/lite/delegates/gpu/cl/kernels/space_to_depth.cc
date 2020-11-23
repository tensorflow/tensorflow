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

#include "tensorflow/lite/delegates/gpu/cl/kernels/space_to_depth.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
std::string GetSpaceToDepthCode(const OperationDef& op_def) {
  std::string c;
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = get_global_id(0);\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = get_global_id(0);\n";
  }
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  FLT tmp[4];\n";
  c += "  tmp[0] = (FLT)(0.0f);\n";
  c += "  tmp[1] = (FLT)(0.0f);\n";
  c += "  tmp[2] = (FLT)(0.0f);\n";
  c += "  tmp[3] = (FLT)(0.0f);\n";
  c += "  for (int i = 0; i < 4; ++i) {\n";
  c += "    int dst_c = 4 * Z + i;\n";
  c += "    int block_id = dst_c / args.src_tensor.Channels();\n";
  c += "    int src_x = X * args.block_size + block_id % args.block_size;\n";
  c += "    int src_y = Y * args.block_size + block_id / args.block_size;\n";
  c += "    int src_c = dst_c % args.src_tensor.Channels();\n";
  c += "    int src_z = src_c / 4;\n";
  c += "    FLT4 t =  args.src_tensor.Read(src_x, src_y, src_z);\n";
  c += "    FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
  c += "    tmp[i] = t_ar[src_c % 4];\n";
  c += "  }\n";
  c += "  FLT4 result = (FLT4)(tmp[0], tmp[1], tmp[2], tmp[3]);\n";
  c += "  args.dst_tensor.Write(result, X, Y, Z);\n";
  c += "}\n";
  return c;
}
}  // namespace

GPUOperation CreateSpaceToDepth(const OperationDef& op_def,
                                const SpaceToDepthAttributes& attr) {
  GPUOperation op(op_def);
  op.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  op.args_.AddInt("block_size", attr.block_size);
  op.code_ = GetSpaceToDepthCode(op_def);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
