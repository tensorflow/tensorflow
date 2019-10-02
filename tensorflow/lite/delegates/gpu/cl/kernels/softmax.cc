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
  TensorCodeGenerator src_tensor("src_data", "size", op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor("dst_data", "size", op_def.dst_tensors[0]);

  const std::string batch_id = op_def.batch_support ? "batch_id" : "";
  std::string code = GetCommonDefines(op_def.precision);
  code += "__kernel void main_function(\n";
  code += src_tensor.GetDeclaration(AccessType::READ);
  code += GetArgsDeclaration(linked_operations);
  code += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  code += "    int4 size,       \n";
  if (op_def.batch_support) {
    code += "    int BATCH_SIZE,  \n";
  }
  code += "    float4 mask      \n";
  code += ") {\n";
  code += "  int X = get_global_id(0);\n";
  code += "  int Y = get_global_id(1);\n";
  code += "  if (X >= size.x || Y >= size.y) return; \n";
  if (op_def.batch_support) {
    code += "  int batch_id = get_global_id(2);\n";
    code += "  if (batch_id >= BATCH_SIZE) return;\n";
  }
  code += "  float sum = 0.0f;\n";
  code += "  for (int d = 0; d < size.w - 1; ++d) {\n";
  code += "    float4 t = " +
          src_tensor.ReadAsFloat4D("X", "Y", "d", batch_id,
                                   TextureAddressMode::DONT_CARE) +
          ";\n";
  code += "    sum += dot((float4)(1.0f), exp(t));\n";
  code += "  }\n";
  code += "  {\n";
  code += "    float4 t = " +
          src_tensor.ReadAsFloat4D("X", "Y", "size.w - 1", batch_id,
                                   TextureAddressMode::DONT_CARE) +
          ";\n";
  code += "    sum += dot(mask, exp(t));\n";
  code += "  }\n";
  code += "  for (int d = 0; d < size.w; ++d) {\n";
  code += "    float4 t = " +
          src_tensor.ReadAsFloat4D("X", "Y", "d", batch_id,
                                   TextureAddressMode::DONT_CARE) +
          ";\n";
  code += "    t = exp(t) / sum;\n";
  code += "    FLT4 result = TO_FLT4(t);\n";
  const LinkingContext context{"result", "X", "Y", "d"};
  code += PostProcess(linked_operations, context);
  code += "    " + dst_tensor.Write4D("result", "X", "Y", "d", batch_id);
  code += "  }\n";
  code += "}\n";
  return code;
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

Status Softmax::Compile(const CreationContext& creation_context) {
  const auto code = GetSoftmaxKernelCode(definition_, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status Softmax::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetSizeWithDepth()));
  if (definition_.batch_support) {
    RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->Batch()));
  }
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(GetMaskForLastPlane(src_[0]->Channels())));
  return OkStatus();
}

int3 Softmax::GetGridSize() const {
  const int grid_x = dst_[0]->Width();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Batch();
  return int3(grid_x, grid_y, grid_z);
}

Status Softmax::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status Softmax::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Softmax CreateSoftmax(const OperationDef& definition) {
  return Softmax(definition);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
