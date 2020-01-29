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

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetMeanKernelCode(
    const OperationDef& op_def,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor(
      "src_data", WHSPoint{"src_size.x", "src_size.y", "src_size.z"},
      op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor("dst_data", WHSPoint{"1", "1", "src_size.z"},
                                 op_def.dst_tensors[0]);

  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ);
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size \n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int S = get_global_id(2);\n";
  c += "  if (X >= 1 || Y >= 1 || S >= src_size.z) return;\n";
  c += "  float4 sum = (float4)(0.0f);\n";
  c += "  for (int y = 0; y < src_size.y; ++y) {\n";
  c += "    for (int x = 0; x < src_size.x; ++x) {\n";
  c += "      sum += " + src_tensor.ReadAsFloatWHS("x", "y", "S") + ";\n";
  c += "    }\n";
  c += "  }\n";
  c += "  sum /= (float)(src_size.x * src_size.y);\n";
  c += "  FLT4 result = TO_FLT4(sum);\n";
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

Status Mean::Compile(const CreationContext& creation_context) {
  const auto code = GetMeanKernelCode(definition_, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status Mean::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWBatchedHSB()));
  return OkStatus();
}

int3 Mean::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

Status Mean::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status Mean::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Mean CreateMean(const OperationDef& definition) { return Mean(definition); }

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
