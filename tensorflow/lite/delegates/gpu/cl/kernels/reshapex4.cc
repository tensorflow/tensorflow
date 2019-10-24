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

#include "tensorflow/lite/delegates/gpu/cl/kernels/reshapex4.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetReshapeBatchedCode(
    const OperationDef& op_def,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor(
      "src_data", {"src_size.x", "src_size.y", "src_size.z", "src_size.w"},
      op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor(
      "dst_data", {"dst_size.x", "dst_size.y", "dst_size.z", "dst_size.w"},
      op_def.dst_tensors[0]);

  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ);
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,             \n";
  c += "    int4 dst_size              \n";
  c += ") {\n";
  c += "  int linear_id = get_global_id(0);\n";
  c += "  int X = linear_id / dst_size.w;\n";
  c += "  int B = linear_id % dst_size.w;\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z || B >= "
       "dst_size.w) return;\n";
  c += "  int dst_bhwc4 = ((B * dst_size.y + Y) * dst_size.x + X) * dst_size.z "
       "+ Z;\n";
  c += "  int src_z = dst_bhwc4 % src_size.z;\n";
  c += "  dst_bhwc4 = dst_bhwc4 / src_size.z;\n";
  c += "  int src_x = dst_bhwc4 % src_size.x;\n";
  c += "  dst_bhwc4 = dst_bhwc4 / src_size.x;\n";
  c += "  int src_y = dst_bhwc4 % src_size.y;\n";
  c += "  int src_b = dst_bhwc4 / src_size.y;\n";
  c += "  FLT4 result =" +
       src_tensor.Read4D("src_x", "src_y", "src_z", "src_b") + ";\n";
  const LinkingContext context{"result", "X * dst_size.w + B", "Y", "Z"};
  c += PostProcess(linked_operations, context);
  c += "  " + dst_tensor.Write4D("result", "X", "Y", "Z", "B");
  c += "}\n";
  return c;
}

std::string GetReshapeCode(
    const OperationDef& op_def,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor("src_data",
                                 {"src_size.x", "src_size.y", "src_size.z"},
                                 op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor("dst_data",
                                 {"dst_size.x", "dst_size.y", "dst_size.z"},
                                 op_def.dst_tensors[0]);

  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ);
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,             \n";
  c += "    int4 dst_size              \n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;\n";
  c += "  int dst_hwc4 = (Y * dst_size.x + X) * dst_size.z + Z;\n";
  c += "  int src_z = dst_hwc4 % src_size.z;\n";
  c += "  dst_hwc4 = dst_hwc4 / src_size.z;\n";
  c += "  int src_x = dst_hwc4 % src_size.x;\n";
  c += "  int src_y = dst_hwc4 / src_size.x;\n";
  c += "  FLT4 result =" + src_tensor.Read3D("src_x", "src_y", "src_z") + ";\n";
  const LinkingContext context{"result", "X", "Y", "Z"};
  c += PostProcess(linked_operations, context);
  c += "  " + dst_tensor.Write3D("result", "X", "Y", "Z");
  c += "}\n";
  return c;
}
}  // namespace

Reshapex4::Reshapex4(Reshapex4&& operation)
    : GPUOperation(std::move(operation)),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

Reshapex4& Reshapex4::operator=(Reshapex4&& operation) {
  if (this != &operation) {
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status Reshapex4::Compile(const CreationContext& creation_context) {
  const auto code = definition_.batch_support
                        ? GetReshapeBatchedCode(definition_, linked_operations_)
                        : GetReshapeCode(definition_, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status Reshapex4::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWHDB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWHDB()));

  return OkStatus();
}

int3 Reshapex4::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Depth();
  return int3(grid_x, grid_y, grid_z);
}

Status Reshapex4::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status Reshapex4::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Reshapex4 CreateReshapex4(const OperationDef& definition) {
  return Reshapex4(definition);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
