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

#include "tensorflow/lite/delegates/gpu/common/tasks/softmax.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {
std::string GetSoftmaxKernelCode(const OperationDef& op_def) {
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int X = GLOBAL_ID_0;\n";
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) "
       "return; \n";
  c += "  float sum = 0.0f;\n";
  c += "  float maximum = args.src_tensor.Read<float>(X, Y, 0).x;\n";
  c += "  for (int d = 0; d < args.dst_tensor.Slices(); ++d) {\n";
  c += "    float4 t = args.src_tensor.Read<float>(X, Y, d);\n";
  c += "    maximum = max(maximum, t.x);\n";
  c += "    if (d * 4 + 1 < args.dst_tensor.Channels()) maximum = max(maximum, "
       "t.y);\n";
  c += "    if (d * 4 + 2 < args.dst_tensor.Channels()) maximum = max(maximum, "
       "t.z);\n";
  c += "    if (d * 4 + 3 < args.dst_tensor.Channels()) maximum = max(maximum, "
       "t.w);\n";
  c += "  }\n";
  c += "  for (int d = 0; d < args.dst_tensor.Slices(); ++d) {\n";
  c += "    float4 t = args.src_tensor.Read<float>(X, Y, d) - "
       "INIT_FLOAT4(maximum);\n";
  c += "    sum += exp(t.x);\n";
  c += "    if (d * 4 + 1 < args.dst_tensor.Channels()) sum += exp(t.y);\n";
  c += "    if (d * 4 + 2 < args.dst_tensor.Channels()) sum += exp(t.z);\n";
  c += "    if (d * 4 + 3 < args.dst_tensor.Channels()) sum += exp(t.w);\n";
  c += "  }\n";
  c += "  for (int d = 0; d < args.dst_tensor.Slices(); ++d) {\n";
  c += "    float4 t = args.src_tensor.Read<float>(X, Y, d) - "
       "INIT_FLOAT4(maximum);\n";
  c += "    t = exp(t) / sum;\n";
  c += "    FLT4 result = TO_FLT4(t);\n";
  c += "    args.dst_tensor.Write(result, X, Y, d);\n";
  c += "  }\n";
  c += "}\n";
  return c;
}
}  // namespace

GPUOperation CreateSoftmax(const OperationDef& definition) {
  GPUOperation op(definition);
  auto src_desc = definition.src_tensors[0];
  if (definition.IsBatchSupported()) {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  op.AddSrcTensor("src_tensor", src_desc);
  auto dst_desc = definition.dst_tensors[0];
  if (definition.IsBatchSupported()) {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  op.AddDstTensor("dst_tensor", dst_desc);
  op.code_ = GetSoftmaxKernelCode(definition);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_ZIs1;
  return op;
}

}  // namespace gpu
}  // namespace tflite
