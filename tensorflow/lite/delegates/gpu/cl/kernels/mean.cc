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

#include "tensorflow/lite/delegates/gpu/cl/kernels/mean.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetMeanKernelCode(const OperationDef& op_def,
                              const int3& work_group_size, Arguments* args) {
  args->AddObjectRef(
      "src_tensor", AccessType::READ,
      absl::make_unique<TensorDescriptor>(op_def.src_tensors[0]));
  args->AddObjectRef(
      "dst_tensor", AccessType::WRITE,
      absl::make_unique<TensorDescriptor>(op_def.dst_tensors[0]));
  args->AddFloat("inv_multiplier_1");
  args->AddFloat("inv_multiplier_2");

  std::string c = GetCommonDefines(op_def.precision);
  const std::string wg_x = std::to_string(work_group_size.x);
  const std::string wg_y = std::to_string(work_group_size.y);
  c += "__kernel void main_function(\n";
  c += "$0) {\n";
  c += "  __local float4 accum[" +
       std::to_string(work_group_size.x * work_group_size.y) + "];\n";
  c += "  int local_x = get_local_id(0);\n";
  c += "  int local_y = get_local_id(1);\n";
  c += "  int local_id = local_y * " + wg_x + " + local_x;\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id_2 = get_global_id(2);\n";
    c += "  int S = linear_id_2 / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id_2 % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int S = get_global_id(2);\n";
  }
  c += "  if (S >= args.dst_tensor.Slices()) return;\n";
  c += "  accum[local_id] = (float4)(0.0f);\n";
  c += "  for (int s_y = local_y; s_y < args.src_tensor.Height(); s_y += " +
       wg_y + ") {\n";
  c += "    for (int s_x = local_x; s_x < args.src_tensor.Width(); s_x += " +
       wg_x + ") {\n";
  c += "      accum[local_id] += args.src_tensor.Read<float>(s_x, s_y, S);\n";
  c += "    }\n";
  c += "  }\n";
  c += "  accum[local_id] *= args.inv_multiplier_1;\n";
  c += "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  const int total_size = work_group_size.x * work_group_size.y;
  int offset = 1;
  int reminder = total_size / 4;
  for (; reminder >= 8; reminder /= 4, offset *= 4) {
    c += "  if (local_id < " + std::to_string(reminder) + ") {\n";
    c += "    int t = local_id * " + std::to_string(offset * 4) + ";\n";
    c += "    float4 sum = accum[t + " + std::to_string(offset) + "];\n";
    c += "    sum += accum[t + " + std::to_string(offset * 2) + "];\n";
    c += "    sum += accum[t + " + std::to_string(offset * 3) + "];\n";
    c += "    accum[t] += sum;\n";
    c += "  }\n";
    c += "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  }
  c += "  float4 sum = accum[0];\n";
  reminder *= 4;
  for (int i = 1; i < reminder; ++i) {
    c += "  sum += accum[" + std::to_string(offset * i) + "];\n";
  }
  c += "  FLT4 result = TO_FLT4(sum * args.inv_multiplier_2);\n";
  c += "  args.dst_tensor.Write(result, 0, 0, S);\n";
  c += "}\n";
  return c;
}
}  // namespace

Mean::Mean(Mean&& operation)
    : GPUOperation(std::move(operation)),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

Mean& Mean::operator=(Mean&& operation) {
  if (this != &operation) {
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

absl::Status Mean::Compile(const CreationContext& creation_context) {
  if (creation_context.device->IsAdreno3xx()) {
    work_group_size_ = int3(16, 8, 1);
  }
  std::string code = GetMeanKernelCode(definition_, work_group_size_, &args_);
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

absl::Status Mean::BindArguments() {
  RETURN_IF_ERROR(args_.SetObjectRef("src_tensor", src_[0]));
  RETURN_IF_ERROR(args_.SetObjectRef("dst_tensor", dst_[0]));
  const double total_size = src_[0]->Width() * src_[0]->Height();
  const double size_0 = work_group_size_.x * work_group_size_.y;
  const double size_1 = total_size / size_0;
  RETURN_IF_ERROR(args_.SetFloat("inv_multiplier_1", 1.0 / size_1));
  RETURN_IF_ERROR(args_.SetFloat("inv_multiplier_2", 1.0 / size_0));
  RETURN_IF_ERROR(SetArguments(linked_operations_, &args_));
  return args_.Bind(kernel_.kernel());
}

int3 Mean::GetGridSize() const {
  const int grid_x = work_group_size_.x;
  const int grid_y = work_group_size_.y;
  const int grid_z = dst_[0]->Slices() * dst_[0]->Batch();
  return int3(grid_x, grid_y, grid_z);
}

absl::Status Mean::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Mean CreateMean(const OperationDef& definition) { return Mean(definition); }

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
