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

#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_weights_converter.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetConverterToConvWeightsCode(const OperationDef& op_def) {
  TensorCodeGenerator src_tensor(
      "src_data",
      WHSBPoint{"src_size.x", "src_size.y", "src_size.z", "src_size.w"},
      op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor(
      "dst_data",
      WHSBPoint{"dst_size.x", "dst_size.y", "dst_size.z", "dst_size.w"},
      op_def.dst_tensors[0]);

  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ) + ",\n";
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size              \n";
  c += ") {\n";
  c += "  int O = get_global_id(0) * 4;\n";
  c += "  int I = get_global_id(1);\n";
  c += "  if (O >= src_size.w || I >= src_size.z) return;\n";
  c += "  FLT4 v0 =" + src_tensor.ReadWHSB("0", "0", "I", "O + 0") + ";\n";
  c += "  FLT4 v1 =" + src_tensor.ReadWHSB("0", "0", "I", "O + 1") + ";\n";
  c += "  FLT4 v2 =" + src_tensor.ReadWHSB("0", "0", "I", "O + 2") + ";\n";
  c += "  FLT4 v3 =" + src_tensor.ReadWHSB("0", "0", "I", "O + 3") + ";\n";
  c += "  FLT4 r0 = (FLT4)(v0.x, v1.x, v2.x, v3.x);\n";
  c += "  FLT4 r1 = (FLT4)(v0.y, v1.y, v2.y, v3.y);\n";
  c += "  FLT4 r2 = (FLT4)(v0.z, v1.z, v2.z, v3.z);\n";
  c += "  FLT4 r3 = (FLT4)(v0.w, v1.w, v2.w, v3.w);\n";
  c += "  int d_index = O / 16;\n";
  c += "  int s_index = I;\n";
  c += "  int k_index = (O % 16) / 4;\n";
  c += "  int dst_offset = (d_index * src_size.z + s_index) * 4 + k_index;\n";
  c += "  int address0 = dst_offset * 4 + 0;\n";
  c += "  int address1 = dst_offset * 4 + 1;\n";
  c += "  int address2 = dst_offset * 4 + 2;\n";
  c += "  int address3 = dst_offset * 4 + 3;\n";
  c += "  " + dst_tensor.Write("r0", "address0");
  c += "  " + dst_tensor.Write("r1", "address1");
  c += "  " + dst_tensor.Write("r2", "address2");
  c += "  " + dst_tensor.Write("r3", "address3");
  c += "}\n";
  return c;
}
}  // namespace

ConverterToConvWeights::ConverterToConvWeights(
    ConverterToConvWeights&& operation)
    : GPUOperation(std::move(operation)),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

ConverterToConvWeights& ConverterToConvWeights::operator=(
    ConverterToConvWeights&& operation) {
  if (this != &operation) {
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

absl::Status ConverterToConvWeights::Compile(
    const CreationContext& creation_context) {
  std::string code = GetConverterToConvWeightsCode(definition_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status ConverterToConvWeights::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWHSB()));
  return absl::OkStatus();
}

int3 ConverterToConvWeights::GetGridSize() const {
  const int grid_x = DivideRoundUp(src_[0]->Batch(), 4);
  const int grid_y = src_[0]->Slices();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

absl::Status ConverterToConvWeights::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

absl::Status ConverterToConvWeights::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

ConverterToConvWeights CreateConverterToConvWeights(
    const OperationDef& definition) {
  return ConverterToConvWeights(definition);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
