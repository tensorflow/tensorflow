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
#include "tensorflow/lite/delegates/gpu/common/tasks/one_hot.h"

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/operations.h"

namespace tflite {
namespace gpu {

std::string GetOneHotCode(const OperationDef& op_def,
                          const OneHotAttributes& attr, GPUOperation* op) {
  op->AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op->AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int Z = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  int idx = Z * 4;\n";
  c += "  int hot_idx = args.src_tensor.Read(0, 0, 0).x;\n";
  c += "  FLT4 res = INIT_FLT4(args.off_value);\n";
  c += "  if ((hot_idx >= idx) && (hot_idx < (idx + 4))) {\n";
  c += "    res.x = (idx + 0) == hot_idx ? args.on_value : args.off_value;\n";
  c += "    res.y = (idx + 1) == hot_idx ? args.on_value : args.off_value;\n";
  c += "    res.z = (idx + 2) == hot_idx ? args.on_value : args.off_value;\n";
  c += "    res.w = (idx + 3) == hot_idx ? args.on_value : args.off_value;\n";
  c += "  }\n";
  c += "  args.dst_tensor.Write(res, X, Y, Z);\n";
  c += "}\n";
  return c;
}

GPUOperation CreateOneHot(const OperationDef& definition,
                          const OneHotAttributes& attr) {
  GPUOperation op(definition);
  op.code_ = GetOneHotCode(definition, attr, &op);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  if (definition.precision == CalculationsPrecision::F32) {
    op.args_.AddFloat("on_value", attr.on_value);
    op.args_.AddFloat("off_value", attr.off_value);
  } else {
    op.args_.AddHalf("on_value", half(attr.on_value));
    op.args_.AddHalf("off_value", half(attr.off_value));
  }
  return op;
}

}  // namespace gpu
}  // namespace tflite
