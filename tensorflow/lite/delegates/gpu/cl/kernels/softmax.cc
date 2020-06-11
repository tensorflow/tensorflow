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

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetSoftmaxKernelCode(
    const OperationDef& op_def,
    const std::vector<ElementwiseOperation*>& linked_operations,
    Arguments* args) {
  auto src_desc = absl::make_unique<TensorDescriptor>(op_def.src_tensors[0]);
  if (op_def.IsBatchSupported()) {
    src_desc->SetStateVar("BatchedWidth", "true");
  }
  args->AddObjectRef("src_tensor", AccessType::READ, std::move(src_desc));
  auto dst_desc = absl::make_unique<TensorDescriptor>(op_def.dst_tensors[0]);
  if (op_def.IsBatchSupported()) {
    dst_desc->SetStateVar("BatchedWidth", "true");
  }
  args->AddObjectRef("dst_tensor", AccessType::WRITE, std::move(dst_desc));

  std::string c = GetCommonDefines(op_def.precision);
  std::string linked_args = GetArgsDeclaration(linked_operations);
  if (linked_args[0] == ',') {
    linked_args[0] = ' ';
  }
  c += "__kernel void main_function(\n";
  c += linked_args;
  c += "$0) {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) "
       "return; \n";
  c += "  float sum = 0.0f;\n";
  c += "  for (int d = 0; d < args.dst_tensor.Slices(); ++d) {\n";
  c += "    float4 t = args.src_tensor.Read<float>(X, Y, d);\n";
  c += "    sum += exp(t.x);\n";
  c += "    if (d * 4 + 1 < args.dst_tensor.Channels()) sum += exp(t.y);\n";
  c += "    if (d * 4 + 2 < args.dst_tensor.Channels()) sum += exp(t.z);\n";
  c += "    if (d * 4 + 3 < args.dst_tensor.Channels()) sum += exp(t.w);\n";
  c += "  }\n";
  c += "  for (int d = 0; d < args.dst_tensor.Slices(); ++d) {\n";
  c += "    float4 t = args.src_tensor.Read<float>(X, Y, d);\n";
  c += "    t = exp(t) / sum;\n";
  c += "    FLT4 result = TO_FLT4(t);\n";
  c += PostProcess(linked_operations, {"result", "X", "Y", "d"});
  c += "    args.dst_tensor.Write(result, X, Y, d);\n";
  c += "  }\n";
  c += "}\n";
  return c;
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

absl::Status Softmax::Compile(const CreationContext& creation_context) {
  std::string code =
      GetSoftmaxKernelCode(definition_, linked_operations_, &args_);
  RETURN_IF_ERROR(args_.TransformToCLCode({}, &code));
  code = absl::Substitute(code, args_.GetListOfArgs());
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

absl::Status Softmax::BindArguments() {
  RETURN_IF_ERROR(args_.SetObjectRef("src_tensor", src_[0]));
  RETURN_IF_ERROR(args_.SetObjectRef("dst_tensor", dst_[0]));
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(args_.Bind(kernel_.kernel(), kernel_.GetBindingCounter()));
  return absl::OkStatus();
}

int3 Softmax::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

absl::Status Softmax::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

absl::Status Softmax::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Softmax CreateSoftmax(const OperationDef& definition) {
  return Softmax(definition);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
