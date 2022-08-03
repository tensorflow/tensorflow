/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/gpu/common/tasks/select_v2.h"

#include <string>
#include <utility>

namespace tflite {
namespace gpu {

std::string GetSelectV2Code(const OperationDef& op_def,
                            const SelectV2Attributes& attr, GPUOperation* op) {
  op->AddSrcTensor("cond_tensor", op_def.src_tensors[0]);
  op->AddSrcTensor("true_tensor", op_def.src_tensors[1]);
  op->AddSrcTensor("else_tensor", op_def.src_tensors[2]);
  op->AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.cond_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += attr.broadcast_true ? "" : "  args.true_tensor.SetBatchRef(B);\n";
    c += attr.broadcast_false ? "" : "  args.else_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int Z = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  FLT4 true_val, else_val;\n";
  if (!attr.broadcast_true) {
    c += "  true_val = args.true_tensor.Read(X, Y, Z);\n";
  } else {
    c += "  true_val = INIT_FLT4(args.true_tensor.Read(0, 0, 0, 0).x);\n";
  }
  if (!attr.broadcast_false) {
    c += "  else_val = args.else_tensor.Read(X, Y, Z);\n";
  } else {
    c += "  else_val = INIT_FLT4(args.else_tensor.Read(0, 0, 0, 0).x);\n";
  }
  c += "  bool should_gather_rows = \n";
  if (attr.broadcast_true && attr.broadcast_false) {
    c += "      true;\n";
  } else {
    c += "      args.dst_tensor.Slices() != args.cond_tensor.Slices();\n";
  }
  c += "  FLT4 res;\n";
  c += "  if (should_gather_rows) {\n";
  c += "    bool cond = args.cond_tensor.Read<bool>(X, 0, 0).x;\n";
  c += "    res = cond ? true_val : else_val;\n";
  c += "  } else {\n";
  c += "    bool4 cond = args.cond_tensor.Read<bool>(0, Y, Z);\n";
  c += "    res = true_val;\n";
  c += "    res.x = cond.x ? true_val.x : else_val.x;\n";
  c += "    res.y = cond.y ? true_val.y : else_val.y;\n";
  c += "    res.z = cond.z ? true_val.z : else_val.z;\n";
  c += "    res.w = cond.w ? true_val.w : else_val.w;\n";
  c += "  }\n;";
  c += "  args.dst_tensor.Write(res, X, Y, Z);\n";
  c += "}\n";
  return c;
}

GPUOperation CreateSelectV2(const OperationDef& definition,
                            const SelectV2Attributes& attr) {
  GPUOperation op(definition);
  op.code_ = GetSelectV2Code(definition, attr, &op);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  op.args_.AddInt("broadcast_true", attr.broadcast_true);
  op.args_.AddInt("broadcast_else", attr.broadcast_false);
  return op;
}

}  // namespace gpu
}  // namespace tflite
