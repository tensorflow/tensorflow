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
std::string GetExpCalculation(const std::string& in_val,
                              const std::string& m_val,
                              const std::string& n_val,
                              const std::string& exp_func) {
  std::string c;
  c += "  " + n_val + " = floor(" + in_val + " * 1.44269504089f);\n";
  c += "  " + m_val + " = " + exp_func + "(" + in_val + " - " + n_val +
       " * 0.69314718056f);\n";
  return c;
}

std::string AddValue(const std::string& m_i, const std::string& n_i,
                     const std::string& m_sum, const std::string& n_sum,
                     const std::string& pow_func) {
  std::string c;
  c += "  n_max = max(" + n_i + ", " + n_sum + ");\n";
  c += "  " + m_sum + " = " + m_i + " * " + pow_func + "(2.0f, " + n_i +
       " - n_max) + " + m_sum + " * " + pow_func + "(2.0f, " + n_sum +
       " - n_max);\n";
  c += "  " + n_sum + " = n_max;\n";
  return c;
}

std::string GetSoftmaxTwoPassKernelCode(const OperationDef& op_def,
                                        const GpuInfo& gpu_info) {
  std::string c;
  std::string exp_func = "exp";
  std::string pow_func = "pow";
  if (gpu_info.IsApiOpenCl()) {
    exp_func = "native_exp";
    pow_func = "native_powr";
  } else if (gpu_info.IsApiMetal()) {
    pow_func = "powr";
  }
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) "
       "return; \n";
  c += "  float m_sum, n_sum;\n";
  c += "  float src_val = args.src_tensor.Read<float>(X, Y, 0).x;\n";
  c += GetExpCalculation("src_val", "m_sum", "n_sum", exp_func);
  c += "  m_sum = 0.0f;\n";
  c += "  for (int d = 0; d < args.dst_tensor.Slices(); ++d) {\n";
  c += "    float4 t = args.src_tensor.Read<float>(X, Y, d);\n";
  c += "    float4 m_i, n_i;\n";
  c += "    float n_max;\n";
  c += GetExpCalculation("t", "m_i", "n_i", exp_func);
  c += AddValue("m_i.x", "n_i.x", "m_sum", "n_sum", pow_func);
  c += "    if (d * 4 + 1 < args.dst_tensor.Channels()) {\n";
  c += AddValue("m_i.y", "n_i.y", "m_sum", "n_sum", pow_func);
  c += "    }\n";
  c += "    if (d * 4 + 2 < args.dst_tensor.Channels()) {\n";
  c += AddValue("m_i.z", "n_i.z", "m_sum", "n_sum", pow_func);
  c += "    }\n";
  c += "    if (d * 4 + 3 < args.dst_tensor.Channels()) {\n";
  c += AddValue("m_i.w", "n_i.w", "m_sum", "n_sum", pow_func);
  c += "    }\n";
  c += "  }\n";
  c += "  for (int d = 0; d < args.dst_tensor.Slices(); ++d) {\n";
  c += "    float4 t = args.src_tensor.Read<float>(X, Y, d);\n";
  c += "    float4 m_i, n_i;\n";
  c += "    FLT4 result;\n";
  c += GetExpCalculation("t", "m_i", "n_i", exp_func);
  c += "    result = TO_FLT4(m_i * " + pow_func +
       "(2.0f, n_i - n_sum) / m_sum);\n";
  c += "    args.dst_tensor.Write(result, X, Y, d);\n";
  c += "  }\n";
  c += "}\n";
  return c;
}

std::string GetSoftmaxThreePassKernelCode(const OperationDef& op_def) {
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
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

GPUOperation CreateSoftmax(const OperationDef& definition,
                           const GpuInfo& gpu_info, const BHWC& shape) {
  GPUOperation op(definition);
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  if ((gpu_info.IsAdreno() && gpu_info.adreno_info.IsAdreno6xxOrHigher() &&
       gpu_info.IsApiOpenCl()) ||
      (gpu_info.IsApple() && gpu_info.IsApiMetal())) {
    op.code_ = GetSoftmaxTwoPassKernelCode(definition, gpu_info);
  } else {
    op.code_ = GetSoftmaxThreePassKernelCode(definition);
  }
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_ZIs1;
  return op;
}

}  // namespace gpu
}  // namespace tflite
