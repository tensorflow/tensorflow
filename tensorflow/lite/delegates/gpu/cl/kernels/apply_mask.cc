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

#include "tensorflow/lite/delegates/gpu/cl/kernels/apply_mask.h"

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetApplyMaskKernelCode(
    const OperationDef& definition,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src("src_data", "src_size", definition.src_tensors[0]);
  TensorCodeGenerator mask("src_mask", "src_size_1", definition.src_tensors[1]);
  TensorCodeGenerator dst("dst_data", "dst_size", definition.dst_tensors[0]);

  std::string c = GetCommonDefines(definition.precision);

  c += "__kernel void main_function(\n";
  c += src.GetDeclaration(AccessType::READ) + ",\n";
  c += mask.GetDeclaration(AccessType::READ) + ",\n";
  c += dst.GetDeclaration(AccessType::WRITE);
  c += GetArgsDeclaration(linked_operations);
  c += "    int apply_mask_type,\n";
  c += "    int4 src_size,\n";
  c += "    int4 src_size_1,\n";
  c += "    int4 dst_size  \n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y) return;\n";
  c += "  FLT4 result = " + src.Read3D("X", "Y", "Z") + ";\n";
  c += "  if (apply_mask_type == 1) {\n";
  c += "    result *= " + mask.Read3D("X", "Y", "Z") + ";\n";
  c += "  } else if (apply_mask_type == 2) {\n";
  c += "    result *= " + mask.Read3D("0", "0", "Z") + ";\n";
  c += "  } else {\n";
  c += "    result *= " + mask.Read3D("X", "Y", "0") + ".x;\n";
  c += "  }\n";
  c += "  " + dst.GetAddress("dst_adr", "X", "Y", "Z");
  c += PostProcess(linked_operations, "result", "Z", "dst_adr");
  c += "  " + dst.Write3D("result", "dst_adr");
  c += "}\n";
  return c;
}

int GetMaskType(int4 src_size, int4 mask_size) {
  if (mask_size.z == 1) {
    return 0;
  } else if (src_size.x == mask_size.x && src_size.y == mask_size.y) {
    return 1;
  } else {
    return 2;
  }
}

}  // namespace

ApplyMask::ApplyMask(ApplyMask&& operation)
    : GPUOperation(std::move(operation)),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

ApplyMask& ApplyMask::operator=(ApplyMask&& operation) {
  if (this != &operation) {
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status ApplyMask::Compile(const CreationContext& creation_context) {
  const auto code = GetApplyMaskKernelCode(definition_, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status ApplyMask::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[1]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(int32_t(
      GetMaskType(src_[0]->GetSizeWithDepth(), src_[1]->GetSizeWithDepth()))));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[1]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetSizeWithDepth()));
  return OkStatus();
}

int3 ApplyMask::GetGridSize() const {
  return int3(dst_[0]->Width(), dst_[0]->Height(), dst_[0]->Depth());
}

Status ApplyMask::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status ApplyMask::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

ApplyMask CreateApplyMask(const OperationDef& definition) {
  return ApplyMask(definition);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
