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

#include "tensorflow/lite/delegates/gpu/common/tasks/transpose.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {
std::string GetTransposeCode(const OperationDef& op_def,
                             const TransposeAttributes& attr) {
  const std::string batch_id =
      op_def.dst_tensors[0].HasAxis(Axis::BATCH) ? "B" : "0";
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
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  args.src_tensor::scalar_type temps[4];\n";
  c += "  temps[0] = args.src_tensor::scalar_zero_value;\n";
  c += "  temps[1] = args.src_tensor::scalar_zero_value;\n";
  c += "  temps[2] = args.src_tensor::scalar_zero_value;\n";
  c += "  temps[3] = args.src_tensor::scalar_zero_value;\n";
  int remap[4];
  remap[attr.perm.b] = 0;
  remap[attr.perm.h] = 1;
  remap[attr.perm.w] = 2;
  remap[attr.perm.c] = 3;
  if (attr.perm.c == 3) {  // optimized reading when no channels permutation
    const std::string bhw[] = {batch_id, "Y", "X"};
    if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
      c += "  args.src_tensor.SetBatchRef(" + bhw[remap[0]] + ");\n";
    }
    c += "  int s_y = " + bhw[remap[1]] + ";\n";
    c += "  int s_x = " + bhw[remap[2]] + ";\n";
    c += "  args.src_tensor::type t = args.src_tensor.Read(s_x, s_y, S);\n";
    c += "  temps[0] = t.x;\n";
    c += "  temps[1] = t.y;\n";
    c += "  temps[2] = t.z;\n";
    c += "  temps[3] = t.w;\n";
  } else {
    c += "  for (int i = 0; i < 4; ++i) {\n";
    c += "    int dst_channel = S * 4 + i;\n";
    c += "    if (dst_channel < args.dst_tensor.Channels()) {\n";
    const std::string bhwc[] = {batch_id, "Y", "X", "dst_channel"};
    if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
      c += "      args.src_tensor.SetBatchRef(" + bhwc[remap[0]] + ");\n";
    }
    c += "      int s_y = " + bhwc[remap[1]] + ";\n";
    c += "      int s_x = " + bhwc[remap[2]] + ";\n";
    c += "      int s_c = " + bhwc[remap[3]] + ";\n";
    c += "      args.src_tensor.ReadPerChannel(temps[i], s_x, s_y, s_c);\n";
    c += "    }\n";
    c += "  }\n";
  }
  c += "  args.src_tensor::type result;\n";
  c += "  result.x = temps[0];\n";
  c += "  result.y = temps[1];\n";
  c += "  result.z = temps[2];\n";
  c += "  result.w = temps[3];\n";
  c += "  args.dst_tensor.Write(result, X, Y, S);\n";
  c += "}\n";
  return c;
}
}  // namespace

GPUOperation CreateTranspose(const OperationDef& definition,
                             const TransposeAttributes& attr) {
  GPUOperation op(definition);
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.code_ = GetTransposeCode(definition, attr);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
