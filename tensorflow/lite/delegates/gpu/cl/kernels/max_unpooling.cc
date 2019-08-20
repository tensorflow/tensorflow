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

#include "tensorflow/lite/delegates/gpu/cl/kernels/max_unpooling.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetMaxUnoolingKernelCode(
    const TensorDescriptor& src_descriptor,
    const TensorDescriptor& src_ind_descriptor,
    const TensorDescriptor& dst_descriptor, CalculationsPrecision precision,
    const std::vector<ElementwiseOperation*>& linked_operations,
    bool manual_boundary_check) {
  TensorCodeGenerator src("src_data", "src_size", src_descriptor);
  TensorCodeGenerator src_ind("src_data_indices", "src_size",
                              src_ind_descriptor);
  TensorCodeGenerator dst("dst_data", "dst_size", dst_descriptor);

  std::string code = GetCommonDefines(precision);

  code += "__kernel void main_function(\n";
  code += src.GetDeclaration(AccessType::READ) + ",\n";
  code += src_ind.GetDeclaration(AccessType::READ);
  code += GetArgsDeclaration(linked_operations);
  code += dst.GetDeclaration(AccessType::WRITE) + ",\n";
  code += "    int4 src_size,      \n";
  code += "    int4 dst_size,      \n";
  code += "    int2 kernel_size,   \n";
  code += "    int2 padding,       \n";
  code += "    int2 stride         \n";
  code += ") {\n";
  code += "  int X = get_global_id(0);\n";
  code += "  int Y = get_global_id(1);\n";
  code += "  int Z = get_global_id(2);\n";
  code += "  if (X >= dst_size.x || Y >= dst_size.y) return; \n";
  code += "  int src_x = (X + padding.x) / stride.x;\n";
  code += "  int src_y = (Y + padding.y) / stride.y;\n";
  code += "  " + src.GetAddress("src_adr", "src_x", "src_y", "Z") + "\n";
  if (manual_boundary_check) {
    code += "  bool outside = src_x < 0 || src_y < 0 ||";
    code += "  src_x >= src_size.x || src_y >= src_size.y;\n";
    code += "  FLT4 src = (FLT4)(0.0f);\n";
    code += "  int4 ind = (int4)(0);\n";
    code += "  if (!outside) {\n";
    code +=
        "    src = " + src.Read3D("src_adr", TextureAddressMode::DONT_CARE) +
        ";\n";
    code += "    ind = convert_int4(" +
            src_ind.Read3D("src_adr", TextureAddressMode::DONT_CARE) + ");\n";
    code += "  }\n";
  } else {
    code += "  FLT4 src = " + src.Read3D("src_adr") + ";\n";
    code += "  int4 ind = convert_int4(" + src_ind.Read3D("src_adr") + ");\n";
  }
  code += "  int t_x = X - (src_x * stride.x - padding.x);\n";
  code += "  int t_y = Y - (src_y * stride.y - padding.y);\n";
  code += "  int t_index = t_y * kernel_size.x + t_x;\n";
  code += "  FLT4 result;\n";
  const std::string channels[] = {".x", ".y", ".z", ".w"};
  for (int i = 0; i < 4; ++i) {
    const auto& s = channels[i];
    code += "  result" + s + "= t_index == ind" + s + "? src" + s + ": 0.0f;\n";
  }
  code += "  " + dst.GetAddress("address", "X", "Y", "Z") + "\n";
  code += PostProcess(linked_operations, "result", "Z", "address");
  code += "  " + dst.Write3D("result", "address");
  code += "}\n";

  return code;
}
}  // namespace

MaxUnpooling::MaxUnpooling(const OperationDef& definition,
                           const MaxUnpooling2DAttributes& attr)
    : GPUOperation(definition),
      stride_(attr.strides.w, attr.strides.h),
      padding_(attr.padding.appended.w, attr.padding.appended.h),
      kernel_size_(attr.kernel.w, attr.kernel.h) {}

MaxUnpooling::MaxUnpooling(MaxUnpooling&& kernel)
    : GPUOperation(std::move(kernel)),
      stride_(kernel.stride_),
      padding_(kernel.padding_),
      kernel_size_(kernel.kernel_size_),
      kernel_(std::move(kernel.kernel_)),
      work_group_size_(kernel.work_group_size_) {}

MaxUnpooling& MaxUnpooling::operator=(MaxUnpooling&& kernel) {
  if (this != &kernel) {
    std::swap(stride_, kernel.stride_);
    std::swap(padding_, kernel.padding_);
    std::swap(kernel_size_, kernel.kernel_size_);
    kernel_ = std::move(kernel.kernel_);
    std::swap(work_group_size_, kernel.work_group_size_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

Status MaxUnpooling::Compile(const CreationContext& creation_context) {
  const bool manual_boundary_check =
      definition_.src_tensors[0].storage_type == TensorStorageType::BUFFER ||
      creation_context.device->IsAdreno3xx();
  const auto code = GetMaxUnoolingKernelCode(
      definition_.src_tensors[0], definition_.src_tensors[1],
      definition_.dst_tensors[0], definition_.precision, linked_operations_,
      manual_boundary_check);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status MaxUnpooling::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[1]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(kernel_size_));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(padding_));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(stride_));

  return OkStatus();
}

int3 MaxUnpooling::GetGridSize() const {
  const int grid_x = dst_[0]->Width();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Depth();
  return int3(grid_x, grid_y, grid_z);
}

Status MaxUnpooling::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status MaxUnpooling::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

MaxUnpooling CreateMaxUnpooling(const OperationDef& definition,
                                const MaxUnpooling2DAttributes& attr) {
  return MaxUnpooling(definition, attr);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
