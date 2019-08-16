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

#include "tensorflow/lite/delegates/gpu/cl/kernels/softmax1x1.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetSoftmaxKernelCode(
    const TensorDescriptor& src_descriptor,
    const TensorDescriptor& dst_descriptor, CalculationsPrecision precision,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor("src_data", "tensor_size", src_descriptor);
  TensorCodeGenerator dst_tensor("dst_data", "tensor_size", dst_descriptor);

  std::string code = GetCommonDefines(precision);
  code += "__kernel void main_function(\n";
  code += src_tensor.GetDeclaration(AccessType::READ);
  code += GetArgsDeclaration(linked_operations);
  code += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  code += "    int4 tensor_size,\n";
  code += "    int2 size,\n";
  code += "    float4 mask\n";
  code += ") {\n";
  code += "  int offset = 0;\n";
  code += "  float sum = 0.0f;\n";
  code += "  int s = 0;\n";
  code += "  int tid = get_local_id(0);\n";
  code += "  do {\n";
  code += "    int z = offset + tid;\n";
  code += "    if (z < size.x) {\n";
  code += "      float4 mask_temp = z == size.x - 1 ? mask : (float4)(1.0f);\n";
  code +=
      "      float4 src = " + src_tensor.ReadAsFloat3D("0", "0", "z") + ";\n";
  code += "      sum += dot(mask_temp, exp(src));\n";
  code += "      offset += 32;\n";
  code += "    }\n";
  code += "    s++;\n";
  code += "  } while (s < size.y);\n";
  code += "\n";
  code += "  __local float4 tmp[8];\n";
  code += "  __local float* tmpx1 = (__local float*)tmp;\n";
  code += "  tmpx1[tid] = sum;\n";
  code += "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  code += "  if (tid == 0) {\n";
  code += "    sum = dot((float4)(1.0f), tmp[0]);\n";
  code += "    sum += dot((float4)(1.0f), tmp[1]);\n";
  code += "    sum += dot((float4)(1.0f), tmp[2]);\n";
  code += "    sum += dot((float4)(1.0f), tmp[3]);\n";
  code += "    sum += dot((float4)(1.0f), tmp[4]);\n";
  code += "    sum += dot((float4)(1.0f), tmp[5]);\n";
  code += "    sum += dot((float4)(1.0f), tmp[6]);\n";
  code += "    sum += dot((float4)(1.0f), tmp[7]);\n";
  code += "    tmpx1[0] = 1.0f / sum;\n";
  code += "  }\n";
  code += "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  code += "  sum = tmpx1[0];\n";
  code += "\n";
  code += "  offset = 0;\n";
  code += "  s = 0;\n";
  code += "  do {\n";
  code += "    int z = offset + tid;\n";
  code += "    if (z < size.x) {\n";
  code += "    " + dst_tensor.GetAddress("address", "0", "0", "z") + "\n";
  code += "      FLT4 value = TO_FLT4(exp(" +
          src_tensor.ReadAsFloat3D("address") + ") * sum);\n";
  code += PostProcess(linked_operations, "value", "z", "address");
  code += "    " + dst_tensor.Write3D("value", "address");
  code += "      offset += 32;\n";
  code += "    }\n";
  code += "    s++;\n";
  code += "  } while (s < size.y);\n";
  code += "}\n";
  return code;
}
}  // namespace

Softmax1x1::Softmax1x1(Softmax1x1&& kernel)
    : GPUOperation(std::move(kernel)), kernel_(std::move(kernel.kernel_)) {}

Softmax1x1& Softmax1x1::operator=(Softmax1x1&& kernel) {
  if (this != &kernel) {
    kernel_ = std::move(kernel.kernel_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

Status Softmax1x1::Compile(const CreationContext& creation_context) {
  const auto code = GetSoftmaxKernelCode(
      definition_.src_tensors[0], definition_.dst_tensors[0],
      definition_.precision, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status Softmax1x1::AddToQueue(CLCommandQueue* queue) {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetSizeWithDepth()));
  const int depth = src_[0]->Depth();
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(int2(depth, IntegralDivideRoundUp(depth, 32))));
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(GetMaskForLastPlane(src_[0]->Channels())));

  return queue->DispatchImplicit(kernel_, {32u, 1u, 1u}, {32u, 1u, 1u});
}

Softmax1x1 CreateSoftmax1x1(const OperationDef& definition) {
  return Softmax1x1(definition);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
