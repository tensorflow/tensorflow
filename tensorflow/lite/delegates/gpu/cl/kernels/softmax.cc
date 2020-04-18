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

#include "tensorflow/lite/delegates/gpu/cl/kernels/softmax.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetSoftmaxKernelCode(
    const OperationDef& op_def,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor("src_data",
                                 WHSPoint{"size.x", "size.y", "size.z"},
                                 op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor("dst_data",
                                 WHSPoint{"size.x", "size.y", "size.z"},
                                 op_def.dst_tensors[0]);

  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ);
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 size,\n";
  c += "    float4 mask\n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  if (X >= size.x || Y >= size.y) return; \n";
  c += "  float sum = 0.0f;\n";
  c += "  for (int d = 0; d < size.z; ++d) {\n";
  c += "    float4 mask_temp = d == size.z - 1 ? mask : (float4)(1.0f);\n";
  c += "    float4 t = " + src_tensor.ReadAsFloatWHS("X", "Y", "d") + ";\n";
  c += "    sum += dot(mask_temp, exp(t));\n";
  c += "  }\n";
  c += "  for (int d = 0; d < size.z; ++d) {\n";
  c += "    float4 t = " + src_tensor.ReadAsFloatWHS("X", "Y", "d") + ";\n";
  c += "    t = exp(t) / sum;\n";
  c += "    FLT4 result = TO_FLT4(t);\n";
  c += PostProcess(linked_operations, {"result", "X", "Y", "d"});
  c += "    " + dst_tensor.WriteWHS("result", "X", "Y", "d");
  c += "  }\n";
  c += "}\n";
  return c;
}
}  // namespace

Softmax::Softmax(Softmax&& kernel)
    : GPUOperation(std::move(kernel)),
      kernel_(std::move(kernel.kernel_)),
      work_group_size_(kernel.work_group_size_) {}

Softmax& Softmax::operator=(Softmax&& kernel) {
  if (this != &kernel) {
    kernel_ = std::move(kernel.kernel_);
    std::swap(work_group_size_, kernel.work_group_size_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

absl::Status Softmax::Compile(const CreationContext& creation_context) {
  const auto code = GetSoftmaxKernelCode(definition_, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status Softmax::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWBatchedHSB()));
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(GetMaskForLastPlane(src_[0]->Channels())));
  return absl::OkStatus();
}

int3 Softmax::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

absl::Status Softmax::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

absl::Status Softmax::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Softmax CreateSoftmax(const OperationDef& definition) {
  return Softmax(definition);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
