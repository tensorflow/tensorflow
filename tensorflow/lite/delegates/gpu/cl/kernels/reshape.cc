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

#include "tensorflow/lite/delegates/gpu/cl/kernels/reshape.h"

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
  c += "    int4 dst_size,             \n";
  c += "    int src_channels,          \n";
  c += "    int dst_channels           \n";
  c += ") {\n";
  c += "  int linear_id = get_global_id(0);\n";
  c += "  int X = linear_id / dst_size.w;\n";
  c += "  int B = linear_id % dst_size.w;\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z || B >= "
       "dst_size.w) return;\n";
  c += "  FLT temps[4];\n";
  c += "  temps[0] = (FLT)(0.0f);\n";
  c += "  temps[1] = (FLT)(0.0f);\n";
  c += "  temps[2] = (FLT)(0.0f);\n";
  c += "  temps[3] = (FLT)(0.0f);\n";
  c += "  int base = ((B * dst_size.y + Y)* dst_size.x + X)* dst_channels + Z "
       "* 4;\n";
  c += "  for (int i = 0; i < 4; ++i) {\n";
  c += "    int dst_channel = Z * 4 + i;\n";
  c += "    if (dst_channel < dst_channels) {;\n";
  c += "      int p = base + i;\n";
  c += "      int src_c = p % src_channels;\n";
  c += "      p = p / src_channels;\n";
  c += "      int src_x = p % src_size.x;\n";
  c += "      p = p / src_size.x;\n";
  c += "      int src_y = p % src_size.y;\n";
  c += "      int src_b = p / src_size.y;\n";
  c += "      int src_z = src_c / 4;\n";
  c += "      int src_sub_ch = src_c % 4;\n";
  c +=
      "      FLT4 t =" + src_tensor.Read4D("src_x", "src_y", "src_z", "src_b") +
      ";\n";
  c += "      FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
  c += "      temps[i] = t_ar[src_sub_ch];\n";
  c += "    }\n";
  c += "  }\n";
  c += "  FLT4 result = (FLT4)(temps[0], temps[1], temps[2], temps[3]);\n";
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
  c += "    int4 dst_size,             \n";
  c += "    int src_channels,          \n";
  c += "    int dst_channels           \n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  FLT temps[4];\n";
  c += "  temps[0] = (FLT)(0.0f);\n";
  c += "  temps[1] = (FLT)(0.0f);\n";
  c += "  temps[2] = (FLT)(0.0f);\n";
  c += "  temps[3] = (FLT)(0.0f);\n";
  c += "  for (int i = 0; i < 4; ++i) {\n";
  c += "    int dst_channel = Z * 4 + i;\n";
  c += "    if (dst_channel < dst_channels) {;\n";
  c += "      int p = dst_channel + dst_channels * (X + dst_size.x * Y);\n";
  c += "      int src_c = p % src_channels;\n";
  c += "      p = p / src_channels;\n";
  c += "      int src_x = p % src_size.x;\n";
  c += "      int src_y = p / src_size.x;\n";
  c += "      int src_z = src_c / 4;\n";
  c += "      int src_sub_ch = src_c % 4;\n";
  c += "      FLT4 t =" + src_tensor.Read3D("src_x", "src_y", "src_z") + ";\n";
  c += "      FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
  c += "      temps[i] = t_ar[src_sub_ch];\n";
  c += "    }\n";
  c += "  }\n";
  c += "  FLT4 result = (FLT4)(temps[0], temps[1], temps[2], temps[3]);\n";
  const LinkingContext context{"result", "X", "Y", "Z"};
  c += PostProcess(linked_operations, context);
  c += "  " + dst_tensor.Write3D("result", "X", "Y", "Z");
  c += "}\n";
  return c;
}
}  // namespace

Reshape::Reshape(Reshape&& operation)
    : GPUOperation(std::move(operation)),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

Reshape& Reshape::operator=(Reshape&& operation) {
  if (this != &operation) {
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status Reshape::Compile(const CreationContext& creation_context) {
  const auto code = definition_.batch_support
                        ? GetReshapeBatchedCode(definition_, linked_operations_)
                        : GetReshapeCode(definition_, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status Reshape::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWHSB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWHSB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->Channels()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->Channels()));

  return OkStatus();
}

int3 Reshape::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

Status Reshape::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status Reshape::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Reshape CreateReshape(const OperationDef& definition) {
  return Reshape(definition);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
