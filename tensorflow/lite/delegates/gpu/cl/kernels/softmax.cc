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

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

Softmax::Softmax(Softmax&& kernel) : GPUOperation(std::move(kernel)) {}

Softmax& Softmax::operator=(Softmax&& kernel) {
  if (this != &kernel) {
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

std::string Softmax::GetSoftmaxKernelCode(const OperationDef& op_def) {
  auto src_desc = op_def.src_tensors[0];
  if (op_def.IsBatchSupported()) {
    src_desc.SetStateVar("BatchedWidth", "true");
  }
  AddSrcTensor("src_tensor", src_desc);
  auto dst_desc = op_def.dst_tensors[0];
  if (op_def.IsBatchSupported()) {
    dst_desc.SetStateVar("BatchedWidth", "true");
  }
  AddDstTensor("dst_tensor", dst_desc);

  std::string c = GetCommonDefines(op_def.precision);
  c += "__kernel void main_function(\n";
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
  c += "    args.dst_tensor.Write(result, X, Y, d);\n";
  c += "  }\n";
  c += "}\n";
  return c;
}

absl::Status Softmax::Compile(const CreationContext& creation_context) {
  std::string code = GetSoftmaxKernelCode(definition_);
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

int3 Softmax::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

Softmax CreateSoftmax(const OperationDef& definition) {
  return Softmax(definition);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
