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
    const TensorDescriptor& src_descriptor,
    const TensorDescriptor& dst_descriptor, CalculationsPrecision precision,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor("src_data", "size", src_descriptor);
  TensorCodeGenerator dst_tensor("dst_data", "size", dst_descriptor);

  std::string code = GetCommonDefines(precision);
  code += "__kernel void main_function(\n";
  code += src_tensor.GetDeclaration(AccessType::READ);
  code += GetArgsDeclaration(linked_operations);
  code += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  code += "    int4 size,\n";
  code += "    float4 mask\n";
  code += ") {\n";
  code += "  int X = get_global_id(0);\n";
  code += "  int Y = get_global_id(1);\n";
  code += "  if (X >= size.x || Y >= size.y) { \n";
  code += "    return; \n";
  code += "  } \n";
  code += "  float sum = 0.0f;\n";
  code += "  for (int d = 0; d < size.w - 1; ++d) {\n";
  code +=
      "    float4 t = " +
      src_tensor.ReadAsFloat3D("X", "Y", "d", TextureAddressMode::DONT_CARE) +
      ";\n";
  code += "    sum += dot((float4)(1.0f), exp(t));\n";
  code += "  }\n";
  code += "  {\n";
  code += "    float4 t = " +
          src_tensor.ReadAsFloat3D("X", "Y", "size.w - 1",
                                   TextureAddressMode::DONT_CARE) +
          ";\n";
  code += "    sum += dot(mask, exp(t));\n";
  code += "  }\n";
  code += "  for (int d = 0; d < size.w; ++d) {\n";
  code += "    " + src_tensor.GetAddress("address", "X", "Y", "d") + "\n";
  code += "    float4 t = " +
          src_tensor.ReadAsFloat3D("address", TextureAddressMode::DONT_CARE) +
          ";\n";
  code += "    t = exp(t) / sum;\n";
  code += "    FLT4 result = TO_FLT4(t);\n";
  code += PostProcess(linked_operations, "result", "d", "address");
  code += "    " + dst_tensor.Write3D("result", "address");
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
  const auto code = GetSoftmaxKernelCode(
      definition_.src_tensors[0], definition_.dst_tensors[0],
      definition_.precision, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status Softmax::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(GetMaskForLastPlane(src_[0]->Channels())));
  return OkStatus();
}

int3 Softmax::GetGridSize() const {
  const int grid_x = dst_[0]->Width();
  const int grid_y = dst_[0]->Height();
  const int grid_z = 1;
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
