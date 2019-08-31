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

#include "tensorflow/lite/delegates/gpu/cl/kernels/padding.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetPaddingCode(
    const TensorDescriptor& src_descriptor,
    const TensorDescriptor& dst_descriptor, CalculationsPrecision precision,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor("src_data", "src_size", src_descriptor);
  TensorCodeGenerator dst_tensor("dst_data", "dst_size", dst_descriptor);

  std::string code = GetCommonDefines(precision);
  const std::string channels[] = {".x", ".y", ".z", ".w"};

  code += "__kernel void main_function(\n";
  code += src_tensor.GetDeclaration(AccessType::READ);
  code += GetArgsDeclaration(linked_operations);
  code += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  code += "    int4 src_size,      \n";
  code += "    int4 dst_size,      \n";
  code += "    int4 prepended      \n";
  code += ") {\n";
  code += "  int X = get_global_id(0);\n";
  code += "  int Y = get_global_id(1);\n";
  code += "  int Z = get_global_id(2);\n";
  code += "  if (X >= dst_size.x || Y >= dst_size.y) return; \n";
  code += "  FLT4 result = (FLT4)(0.0);\n";
  code += "  int s_x = X - prepended.x;\n";
  code += "  int s_y = Y - prepended.y;\n";
  code += "  bool inside_x = s_x >= 0 && s_x < src_size.x;\n";
  code += "  bool inside_y = s_y >= 0 && s_y < src_size.y;\n";
  code += "  if (inside_x && inside_y) {\n";
  code += "    int start_channel = Z * 4;\n";
  for (int i = 0; i < 4; ++i) {
    const auto& s = channels[i];
    code += "    {\n";
    code += "    int channel = start_channel + " + std::to_string(i) + ";\n";
    code += "    int s_z = channel - prepended.z;\n";
    code += "    if (s_z >= 0 && s_z < src_size.z) {\n";
    code += "      FLT4 t = " +
            src_tensor.Read3D("s_x", "s_y", "s_z / 4",
                              TextureAddressMode::DONT_CARE) +
            ";\n";
    code += "      FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
    code += "      result" + s + " = t_ar[s_z % 4];\n";
    code += "    }\n";
    code += "    }\n";
  }
  code += "  }\n";
  code += "  " + dst_tensor.GetAddress("address", "X", "Y", "Z") + "\n";
  code += PostProcess(linked_operations, "result", "Z", "address");
  code += "  " + dst_tensor.Write3D("result", "address");
  code += "}\n";

  return code;
}
}  // namespace

Padding::Padding(const OperationDef& definition, const PadAttributes& attr)
    : GPUOperation(definition) {
  SetPrepended(int3(attr.prepended.w, attr.prepended.h, attr.prepended.c));
}

Padding::Padding(Padding&& kernel)
    : GPUOperation(std::move(kernel)),
      prepended_(kernel.prepended_),
      kernel_(std::move(kernel.kernel_)),
      work_group_size_(kernel.work_group_size_) {}

Padding& Padding::operator=(Padding&& kernel) {
  if (this != &kernel) {
    std::swap(prepended_, kernel.prepended_);
    kernel_ = std::move(kernel.kernel_);
    std::swap(work_group_size_, kernel.work_group_size_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

void Padding::SetPrepended(const int3& prepended) {
  prepended_.x = prepended.x;
  prepended_.y = prepended.y;
  prepended_.z = prepended.z;
  prepended_.w = 0;
}

Status Padding::Compile(const CreationContext& creation_context) {
  const auto code =
      GetPaddingCode(definition_.src_tensors[0], definition_.dst_tensors[0],
                     definition_.precision, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status Padding::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(prepended_));
  return OkStatus();
}

int3 Padding::GetGridSize() const {
  const int grid_x = dst_[0]->Width();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Depth();
  return int3(grid_x, grid_y, grid_z);
}

Status Padding::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status Padding::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Padding CreatePadding(const OperationDef& definition,
                      const PadAttributes& attr) {
  return Padding(definition, attr);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
