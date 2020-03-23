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

#include "tensorflow/lite/delegates/gpu/cl/kernels/mean.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetMeanKernelCode(
    const OperationDef& op_def,
    const std::vector<ElementwiseOperation*>& linked_operations,
    const int3& work_group_size) {
  TensorCodeGenerator src_tensor(
      "src_data", WHSPoint{"src_size.x", "src_size.y", "src_size.z"},
      op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor("dst_data", WHSPoint{"1", "1", "src_size.z"},
                                 op_def.dst_tensors[0]);

  std::string c = GetCommonDefines(op_def.precision);
  const std::string wg_x = std::to_string(work_group_size.x);
  const std::string wg_y = std::to_string(work_group_size.y);
  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ);
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,           \n";
  c += "    float2 inv_multipliers   \n";
  c += ") {\n";
  c += "  __local float4 accum[" +
       std::to_string(work_group_size.x * work_group_size.y) + "];\n";
  c += "  int local_x = get_local_id(0);\n";
  c += "  int local_y = get_local_id(1);\n";
  c += "  int local_id = local_y * " + wg_x + " + local_x;\n";
  c += "  int S = get_global_id(2);\n";
  c += "  if (S >= src_size.z) return;\n";
  c += "  accum[local_id] = (float4)(0.0f);\n";
  c += "  for (int s_y = local_y; s_y < src_size.y; s_y += " + wg_y + ") {\n";
  c += "    for (int s_x = local_x; s_x < src_size.x; s_x += " + wg_x + ") {\n";
  c += "        accum[local_id] += " +
       src_tensor.ReadAsFloatWHS("s_x", "s_y", "S") + ";\n";
  c += "    }\n";
  c += "  }\n";
  c += "  accum[local_id] *= inv_multipliers.x;\n";
  c += "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  const int total_size = work_group_size.x * work_group_size.y;
  int offset = 1;
  int reminder = total_size / 4;
  for (; reminder >= 8; reminder /= 4, offset *= 4) {
    c += "  if (local_id < " + std::to_string(reminder) + ") {\n";
    c += "    int t = local_id * " + std::to_string(offset * 4) + ";\n";
    c += "    float4 sum = accum[t + " + std::to_string(offset) + "];\n";
    c += "    sum += accum[t + " + std::to_string(offset * 2) + "];\n";
    c += "    sum += accum[t + " + std::to_string(offset * 3) + "];\n";
    c += "    accum[t] += sum;\n";
    c += "  }\n";
    c += "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  }
  c += "  float4 sum = accum[0];\n";
  reminder *= 4;
  for (int i = 1; i < reminder; ++i) {
    c += "  sum += accum[" + std::to_string(offset * i) + "];\n";
  }
  c += "  FLT4 result = TO_FLT4(sum * inv_multipliers.y);\n";
  c += PostProcess(linked_operations, {"result", "0", "0", "S"});
  c += "  " + dst_tensor.WriteWHS("result", "0", "0", "S");
  c += "}\n";
  return c;
}
}  // namespace

Mean::Mean(Mean&& operation)
    : GPUOperation(std::move(operation)),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

Mean& Mean::operator=(Mean&& operation) {
  if (this != &operation) {
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

absl::Status Mean::Compile(const CreationContext& creation_context) {
  if (creation_context.device->IsAdreno3xx()) {
    work_group_size_ = int3(16, 8, 1);
  }
  const auto code =
      GetMeanKernelCode(definition_, linked_operations_, work_group_size_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status Mean::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWBatchedHSB()));
  const double total_size = src_[0]->Width() * src_[0]->Height();
  const double size_0 = work_group_size_.x * work_group_size_.y;
  const double size_1 = total_size / size_0;
  RETURN_IF_ERROR(kernel_.SetBytesAuto(float2(1.0 / size_1, 1.0 / size_0)));
  return absl::OkStatus();
}

int3 Mean::GetGridSize() const {
  const int grid_x = work_group_size_.x * dst_[0]->Batch();
  const int grid_y = work_group_size_.y;
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

absl::Status Mean::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Mean CreateMean(const OperationDef& definition) { return Mean(definition); }

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
