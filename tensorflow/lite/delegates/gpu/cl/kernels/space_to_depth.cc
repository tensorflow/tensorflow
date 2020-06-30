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

#include "tensorflow/lite/delegates/gpu/cl/kernels/space_to_depth.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetSpaceToDepthCode(const OperationDef& op_def, Arguments* args) {
  args->AddObjectRef(
      "src_tensor", AccessType::READ,
      absl::make_unique<TensorDescriptor>(op_def.src_tensors[0]));
  args->AddObjectRef(
      "dst_tensor", AccessType::WRITE,
      absl::make_unique<TensorDescriptor>(op_def.dst_tensors[0]));
  args->AddInt("block_size");

  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = get_global_id(0);\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = get_global_id(0);\n";
  }
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  FLT tmp[4];\n";
  c += "  tmp[0] = (FLT)(0.0f);\n";
  c += "  tmp[1] = (FLT)(0.0f);\n";
  c += "  tmp[2] = (FLT)(0.0f);\n";
  c += "  tmp[3] = (FLT)(0.0f);\n";
  c += "  for (int i = 0; i < 4; ++i) {\n";
  c += "    int dst_c = 4 * Z + i;\n";
  c += "    int block_id = dst_c / args.src_tensor.Channels();\n";
  c += "    int src_x = X * args.block_size + block_id % args.block_size;\n";
  c += "    int src_y = Y * args.block_size + block_id / args.block_size;\n";
  c += "    int src_c = dst_c % args.src_tensor.Channels();\n";
  c += "    int src_z = src_c / 4;\n";
  c += "    FLT4 t =  args.src_tensor.Read(src_x, src_y, src_z);\n";
  c += "    FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
  c += "    tmp[i] = t_ar[src_c % 4];\n";
  c += "  }\n";
  c += "  FLT4 result = (FLT4)(tmp[0], tmp[1], tmp[2], tmp[3]);\n";
  c += "  args.dst_tensor.Write(result, X, Y, Z);\n";
  c += "}\n";
  return c;
}

}  // namespace

SpaceToDepth::SpaceToDepth(SpaceToDepth&& operation)
    : GPUOperation(std::move(operation)),
      attr_(operation.attr_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

SpaceToDepth& SpaceToDepth::operator=(SpaceToDepth&& operation) {
  if (this != &operation) {
    attr_ = operation.attr_;
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

absl::Status SpaceToDepth::Compile(const CreationContext& creation_context) {
  std::string code = GetSpaceToDepthCode(definition_, &args_);
  std::string element_wise_code;
  RETURN_IF_ERROR(
      MergeOperations(linked_operations_, &args_, &element_wise_code));
  RETURN_IF_ERROR(args_.TransformToCLCode(creation_context.device->GetInfo(),
                                          {{"dst_tensor", element_wise_code}},
                                          &code));
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status SpaceToDepth::BindArguments() {
  RETURN_IF_ERROR(args_.SetObjectRef("src_tensor", src_[0]));
  RETURN_IF_ERROR(args_.SetObjectRef("dst_tensor", dst_[0]));
  RETURN_IF_ERROR(args_.SetInt("block_size", attr_.block_size));
  RETURN_IF_ERROR(SetArguments(linked_operations_, &args_));
  return args_.Bind(kernel_.kernel());
}

int3 SpaceToDepth::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

absl::Status SpaceToDepth::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

absl::Status SpaceToDepth::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

SpaceToDepth CreateSpaceToDepth(const OperationDef& op_def,
                                const SpaceToDepthAttributes& attr) {
  return SpaceToDepth(op_def, attr);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
