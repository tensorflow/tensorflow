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

#include "tensorflow/lite/delegates/gpu/common/tasks/cast.h"

#include <map>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/task/util.h"

namespace tflite {
namespace gpu {
namespace {
std::string GetCastKernelCode(const OperationDef& op_def,
                              const GpuInfo& gpu_info) {
  std::string c;
  std::string coords = "X, Y";
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = GLOBAL_ID_1;\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int Z = linear_id_1 % args.dst_tensor.Depth();\n";
    coords += ", Z";
  } else {
    c += "  int Y = GLOBAL_ID_1;\n";
  }
  coords += ", S";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id_0 = GLOBAL_ID_0;\n";
    c += "  int X = linear_id_0 / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id_0 % args.dst_tensor.Batch();\n";
    coords += ", B";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  args.src_tensor::type src_value = args.src_tensor.Read(" + coords +
       ");\n";
  const std::string conversion =
      GetTypeConversion(gpu_info, op_def.src_tensors[0].data_type,
                        op_def.dst_tensors[0].data_type, 4);
  if (conversion.empty()) {
    c += "  args.dst_tensor::type result = src_value;\n";
  } else {
    c += "  args.dst_tensor::type result = " + conversion + "(src_value);\n";
  }
  c += "  args.dst_tensor.Write(result, " + coords + ");\n";
  c += "}\n";
  return c;
}
}  // namespace

GPUOperation CreateCast(const OperationDef& definition,
                        const GpuInfo& gpu_info) {
  GPUOperation op(definition);
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.code_ = GetCastKernelCode(definition, gpu_info);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace gpu
}  // namespace tflite
