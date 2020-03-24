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

#include "tensorflow/lite/delegates/gpu/cl/kernels/concat_xy.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetConcatKernelCode(
    const OperationDef& op_def, int tensors_count,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  std::vector<TensorCodeGenerator> srcs(tensors_count);
  for (int i = 0; i < tensors_count; ++i) {
    const std::string tensor_name = "src_data_" + std::to_string(i);
    const std::string width = "src_size_" + std::to_string(i) + ".x";
    const std::string height = "src_size_" + std::to_string(i) + ".y";
    srcs[i] =
        TensorCodeGenerator(tensor_name, WHSPoint{width, height, "dst_size.z"},
                            op_def.src_tensors[i]);
  }
  TensorCodeGenerator dst("dst_data",
                          WHSPoint{"dst_size.x", "dst_size.y", "dst_size.z"},
                          op_def.dst_tensors[0]);

  std::string c = GetCommonDefines(op_def.precision);

  c += "__kernel void main_function(\n";
  for (const auto& src : srcs) {
    c += src.GetDeclaration(AccessType::READ) + ",\n";
  }
  c += dst.GetDeclaration(AccessType::WRITE);
  c += GetArgsDeclaration(linked_operations);
  for (int i = 0; i < tensors_count; ++i) {
    const std::string uniform_name = "src_size_" + std::to_string(i);
    c += "    int4 " + uniform_name + ",\n";
  }
  c += "    int4 dst_size  \n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (Z >= dst_size.z) return;\n";
  for (int i = 0; i < tensors_count; ++i) {
    const std::string size_name = "src_size_" + std::to_string(i);
    c += "  if (X < " + size_name + ".x && Y < " + size_name + ".y) { \n";
    c += "    FLT4 result = " + srcs[i].ReadWHS("X", "Y", "Z") + ";\n";
    c += "    int dst_x = X + " + size_name + ".z;\n";
    c += "    int dst_y = Y + " + size_name + ".w;\n";
    const LinkingContext context{"result", "dst_x", "dst_y", "Z"};
    c += PostProcess(linked_operations, context);
    c += "    " + dst.WriteWHS("result", "dst_x", "dst_y", "Z");
    c += "  } \n";
  }
  c += "}\n";
  return c;
}

}  // namespace

ConcatXY::ConcatXY(ConcatXY&& operation)
    : GPUOperation(std::move(operation)),
      attr_(operation.attr_),
      tensors_count_(operation.tensors_count_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

ConcatXY& ConcatXY::operator=(ConcatXY&& operation) {
  if (this != &operation) {
    attr_ = operation.attr_;
    tensors_count_ = operation.tensors_count_;
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status ConcatXY::Compile(const CreationContext& creation_context) {
  const auto code =
      GetConcatKernelCode(definition_, tensors_count_, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status ConcatXY::BindArguments() {
  kernel_.ResetBindingCounter();
  for (int i = 0; i < tensors_count_; ++i) {
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[i]->GetMemoryPtr()));
  }
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  int x_offset = 0;
  int y_offset = 0;
  for (int i = 0; i < tensors_count_; ++i) {
    const int width = src_[i]->Width() * src_[i]->Batch();
    const int height = src_[i]->Height();
    RETURN_IF_ERROR(
        kernel_.SetBytesAuto(int4(width, height, x_offset, y_offset)));
    x_offset += attr_.axis == Axis::WIDTH ? width : 0;
    y_offset += attr_.axis == Axis::HEIGHT ? height : 0;
  }
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWBatchedHSB()));
  return OkStatus();
}

int3 ConcatXY::GetGridSize() const {
  int max_src_width = 0;
  int max_src_height = 0;
  for (int i = 0; i < tensors_count_; ++i) {
    max_src_width = std::max(max_src_width, src_[i]->Width());
    max_src_height = std::max(max_src_height, src_[i]->Height());
  }

  const int grid_x = max_src_width * dst_[0]->Batch();
  const int grid_y = max_src_height;
  const int grid_z = dst_[0]->Slices();

  return int3(grid_x, grid_y, grid_z);
}

Status ConcatXY::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status ConcatXY::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

ConcatXY CreateConcatXY(const OperationDef& definition,
                        const ConcatAttributes& attr, int tensors_count) {
  return ConcatXY(definition, attr, tensors_count);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
