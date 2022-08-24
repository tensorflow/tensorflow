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

#include "tensorflow/lite/delegates/gpu/common/tasks/softmax1x1.h"

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"

namespace tflite {
namespace gpu {

Softmax1x1::Softmax1x1(const OperationDef& definition)
    : GPUOperation(definition) {
  work_group_size_ = int3(32, 1, 1);
  code_ = GetSoftmaxKernelCode(definition_);
}

Softmax1x1::Softmax1x1(Softmax1x1&& kernel) : GPUOperation(std::move(kernel)) {}

Softmax1x1& Softmax1x1::operator=(Softmax1x1&& kernel) {
  if (this != &kernel) {
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

std::string Softmax1x1::GetSoftmaxKernelCode(const OperationDef& op_def) {
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddFloat("mask_x");
  args_.AddFloat("mask_y");
  args_.AddFloat("mask_z");
  args_.AddFloat("mask_w");

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = GROUP_ID_1;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  if (B >= args.dst_tensor.Batch()) return;\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GROUP_ID_1;\n";
  }
  c += "  int Y = GROUP_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width()) return;\n";
  c += "  if (Y >= args.dst_tensor.Height()) return;\n";
  c += "  float4 mask = INIT_FLOAT4v4(args.mask_x, args.mask_y, args.mask_z, "
       "args.mask_w);\n";
  c +=
      "  float4 maxx4 = INIT_FLOAT4(args.src_tensor.Read<float>(X, Y, 0).x);\n";
  c += "  int tid = LOCAL_ID_0;\n";
  c += "  for (int s = tid; s < args.src_tensor.Slices(); s += 32) {\n";
  c += "    float4 mask_a = s == args.src_tensor.Slices() - 1 ? mask : "
       "INIT_FLOAT4(1.0f);\n";
  c += "    float4 mask_b = INIT_FLOAT4(1.0f) - mask_a;\n";
  c += "    float4 src = args.src_tensor.Read<float>(X, Y, s);\n";
  c += "    src = src * mask_a + mask_b * src.x;\n";
  c += "    maxx4 = max(maxx4, src);\n";
  c += "  }\n";
  c += "  float maximum = max(maxx4.x, maxx4.y);\n";
  c += "  maximum = max(maximum, maxx4.z);\n";
  c += "  maximum = max(maximum, maxx4.w);\n";
  c += "  __local float loc_mem[32];\n";
  c += "  loc_mem[tid] = maximum;\n";
  c += "  LOCAL_MEM_BARRIER;\n";
  c += "  if (tid % 8 == 0) {\n";
  c += "    maximum = max(loc_mem[tid], loc_mem[tid + 1]);\n";
  c += "    maximum = max(maximum, loc_mem[tid + 2]);\n";
  c += "    maximum = max(maximum, loc_mem[tid + 3]);\n";
  c += "    maximum = max(maximum, loc_mem[tid + 4]);\n";
  c += "    maximum = max(maximum, loc_mem[tid + 5]);\n";
  c += "    maximum = max(maximum, loc_mem[tid + 6]);\n";
  c += "    maximum = max(maximum, loc_mem[tid + 7]);\n";
  c += "    loc_mem[tid] = maximum;\n";
  c += "  }\n";
  c += "  LOCAL_MEM_BARRIER;\n";
  c += "  if (tid == 0) {\n";
  c += "    maximum = max(loc_mem[0], loc_mem[8]);\n";
  c += "    maximum = max(maximum, loc_mem[16]);\n";
  c += "    maximum = max(maximum, loc_mem[24]);\n";
  c += "    loc_mem[0] = maximum;\n";
  c += "  }\n";
  c += "  LOCAL_MEM_BARRIER;\n";
  c += "  maximum = loc_mem[0];\n";
  c += "  float sum = 0.0f;\n";
  c += "  for (int s = tid; s < args.src_tensor.Slices(); s += 32) {\n";
  c += "    float4 mask_temp = s == args.src_tensor.Slices() - 1 ? mask : "
       "INIT_FLOAT4(1.0f);\n";
  c += "    float4 src = args.src_tensor.Read<float>(X, Y, s) - "
       "INIT_FLOAT4(maximum);\n";
  c += "    sum += dot(mask_temp, exp(src));\n";
  c += "  }\n";
  c += "  LOCAL_MEM_BARRIER;\n";
  c += "  loc_mem[tid] = sum;\n";
  c += "  LOCAL_MEM_BARRIER;\n";
  c += "  if (tid % 8 == 0) {\n";
  c += "    sum = loc_mem[tid] + loc_mem[tid + 1];\n";
  c += "    sum += loc_mem[tid + 2];\n";
  c += "    sum += loc_mem[tid + 3];\n";
  c += "    sum += loc_mem[tid + 4];\n";
  c += "    sum += loc_mem[tid + 5];\n";
  c += "    sum += loc_mem[tid + 6];\n";
  c += "    sum += loc_mem[tid + 7];\n";
  c += "    loc_mem[tid] = sum;\n";
  c += "  }\n";
  c += "  LOCAL_MEM_BARRIER;\n";
  c += "  if (tid == 0) {\n";
  c += "    sum = loc_mem[0] + loc_mem[8] + loc_mem[16] + loc_mem[24];\n";
  c += "    loc_mem[0] = 1.0f / sum;\n";
  c += "  }\n";
  c += "  LOCAL_MEM_BARRIER;\n";
  c += "  sum = loc_mem[0];\n";
  c += "\n";
  c += "  int dst_s = GLOBAL_ID_0;\n";
  c += "  if (dst_s < args.dst_tensor.Slices()) {\n";
  c += "    float4 src = args.src_tensor.Read<float>(X, Y, dst_s) - "
       "INIT_FLOAT4(maximum);\n";
  c += "    FLT4 res = TO_FLT4(exp(src) * sum);\n";
  c += "    args.dst_tensor.Write(res, X, Y, dst_s);\n";
  c += "  }\n";
  c += "}\n";
  return c;
}

absl::Status Softmax1x1::BindArguments(ArgumentsBinder* args) {
  float4 mask = GetMaskForLastPlane(src_[0]->Channels());
  RETURN_IF_ERROR(args->SetFloat("mask_x", mask.x));
  RETURN_IF_ERROR(args->SetFloat("mask_y", mask.y));
  RETURN_IF_ERROR(args->SetFloat("mask_z", mask.z));
  RETURN_IF_ERROR(args->SetFloat("mask_w", mask.w));
  return absl::OkStatus();
}

int3 Softmax1x1::GetGridSize() const {
  return int3(dst_[0]->Slices(), dst_[0]->Width() * dst_[0]->Batch(),
              dst_[0]->Height());
}

Softmax1x1 CreateSoftmax1x1(const OperationDef& definition) {
  return Softmax1x1(definition);
}

}  // namespace gpu
}  // namespace tflite
