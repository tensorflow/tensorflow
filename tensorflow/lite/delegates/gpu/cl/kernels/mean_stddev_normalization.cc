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

#include "tensorflow/lite/delegates/gpu/cl/kernels/mean_stddev_normalization.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"

namespace tflite {
namespace gpu {
namespace cl {

MeanStdDevNormalization::MeanStdDevNormalization(const OperationDef& definition)
    : GPUOperation(definition) {
  code_ = GetNormalizationCode(definition_);
}

std::string MeanStdDevNormalization::GetNormalizationCode(
    const OperationDef& op_def) {
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  c += "  size_t B = get_global_id(0);\n";
  c += "  if (B >= args.src_tensor.Batch()) { return; }\n";
  c += "  if (get_global_id(1) > 0) { return; }\n";  // ?!?
  c += "  float sum = 0.0f;\n";
  c += "  for (int S = 0; S < args.src_tensor.Slices(); ++S) {\n";
  c += "    const float4 t = args.src_tensor.Read<float>(0, 0, S, B);\n";
  c += "    sum += t.x;\n";
  c += "    if (S * 4 + 1 < args.src_tensor.Channels()) sum += t.y;\n";
  c += "    if (S * 4 + 2 < args.src_tensor.Channels()) sum += t.z;\n";
  c += "    if (S * 4 + 3 < args.src_tensor.Channels()) sum += t.w;\n";
  c += "  }\n";
  c += "  float mean = sum / args.src_tensor.Channels();\n";
  c += "  float sum_diff_sq = 0.0f;\n";
  c += "  for (int S = 0; S < args.src_tensor.Slices(); ++S) {\n";
  c += "    const float4 t = args.src_tensor.Read<float>(0, 0, S, B);\n";
  c += "    float4 diff = t - (float4)(mean, mean, mean, mean);";
  c += "    if (S * 4 + 1 >= args.src_tensor.Channels()) diff.y = 0.0f;\n";
  c += "    if (S * 4 + 2 >= args.src_tensor.Channels()) diff.z = 0.0f;\n";
  c += "    if (S * 4 + 3 >= args.src_tensor.Channels()) diff.w = 0.0f;\n";
  c += "    float dotprod = dot(diff, diff);\n";
  c += "    sum_diff_sq += dotprod;\n";
  c += "  }\n";
  c += "  const float variance = sum_diff_sq / args.src_tensor.Channels();\n";
  c += "  const float stddev_inv =  rsqrt(variance + 1.0e-8f);\n";
  c += "  for (int S = 0; S < args.src_tensor.Slices(); ++S) {\n";
  c += "    float4 t = args.src_tensor.Read<float>(0, 0, S, B);\n";
  c += "    t = (t - mean) * stddev_inv;\n";
  c += "    FLT4 result = TO_FLT4(t);\n";
  c += "    args.dst_tensor.Write(result, 0, 0, S, B);\n";
  c += "  }\n";
  c += "}\n";
  return c;
}

int3 MeanStdDevNormalization::GetGridSize() const {
  const int grid_x = dst_[0]->Batch();
  const int grid_y = 1;
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

MeanStdDevNormalization CreateMeanStdDevNormalization(
    const OperationDef& definition) {
  return MeanStdDevNormalization(definition);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
