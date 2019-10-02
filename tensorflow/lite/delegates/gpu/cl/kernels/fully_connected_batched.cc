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

#include "tensorflow/lite/delegates/gpu/cl/kernels/fully_connected_batched.h"

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
std::string GetFullyConnectedBatchedKernelCode(
    const OperationDef& op_def,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor("src_data", "src_size", op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor("dst_data", "dst_size", op_def.dst_tensors[0]);

  std::string c = GetCommonDefines(op_def.precision);

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ) + ",\n";
  c += "    __read_only image2d_t filters,\n";
  c += "    __read_only image2d_t biases";
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,             \n";
  c += "    int4 dst_size,             \n";
  c += "    int BATCH_SIZE             \n";
  c += ") {\n";
  c += "  int Z = get_global_id(0);\n";
  c += "  int B = get_global_id(1);\n";
  c += "  if (Z >= dst_size.w || B >= BATCH_SIZE) return;\n";
  c += "  ACCUM_FLT4 s = (ACCUM_FLT4)(0.0f);\n";
  c += "  for (int i = 0; i < src_size.w; ++i) {\n";
  c += "    FLT4 v = " +
       src_tensor.Read4D("0", "0", "i", "B", TextureAddressMode::DONT_CARE) +
       ";\n";
  c += "    FLT4 m0 = READ_IMAGE(filters, smp_none, (int2)(Z * 4 + 0, i));\n";
  c += "    FLT4 m1 = READ_IMAGE(filters, smp_none, (int2)(Z * 4 + 1, i));\n";
  c += "    FLT4 m2 = READ_IMAGE(filters, smp_none, (int2)(Z * 4 + 2, i));\n";
  c += "    FLT4 m3 = READ_IMAGE(filters, smp_none, (int2)(Z * 4 + 3, i));\n";
  c += "    s.x += (v.x * m0.s0 + v.y * m0.s1 + v.z * m0.s2 + v.w * m0.s3);\n";
  c += "    s.y += (v.x * m1.s0 + v.y * m1.s1 + v.z * m1.s2 + v.w * m1.s3);\n";
  c += "    s.z += (v.x * m2.s0 + v.y * m2.s1 + v.z * m2.s2 + v.w * m2.s3);\n";
  c += "    s.w += (v.x * m3.s0 + v.y * m3.s1 + v.z * m3.s2 + v.w * m3.s3);\n";
  c += "  }\n";
  c += "  FLT4 r0 = TO_FLT4(s) + READ_IMAGE(biases, smp_none, (int2)(Z, "
       "0));\n";
  const LinkingContext context{"r0", "0", "0", "Z"};
  c += PostProcess(linked_operations, context);
  c += "  " + dst_tensor.Write4D("r0", "0", "0", "Z", "B") + "\n";
  c += "}\n";
  return c;
}
}  // namespace

FullyConnectedBatched::FullyConnectedBatched(const OperationDef& definition)
    : GPUOperation(definition) {}

FullyConnectedBatched::FullyConnectedBatched(FullyConnectedBatched&& kernel)
    : GPUOperation(std::move(kernel)),
      weights_(std::move(kernel.weights_)),
      biases_(std::move(kernel.biases_)),
      kernel_(std::move(kernel.kernel_)),
      work_group_size_(kernel.work_group_size_) {}

FullyConnectedBatched& FullyConnectedBatched::operator=(
    FullyConnectedBatched&& kernel) {
  if (this != &kernel) {
    weights_ = std::move(kernel.weights_);
    biases_ = std::move(kernel.biases_);
    kernel_ = std::move(kernel.kernel_);
    std::swap(work_group_size_, kernel.work_group_size_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

Status FullyConnectedBatched::Compile(const CreationContext& creation_context) {
  const auto code =
      GetFullyConnectedBatchedKernelCode(definition_, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status FullyConnectedBatched::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_.GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(biases_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->Batch()));
  return OkStatus();
}

int3 FullyConnectedBatched::GetGridSize() const {
  const int grid_x = dst_[0]->Depth();
  const int grid_y = dst_[0]->Batch();
  return int3(grid_x, grid_y, 1);
}

Status FullyConnectedBatched::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status FullyConnectedBatched::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Status CreateFullyConnectedBatched(const CreationContext& creation_context,
                                   const OperationDef& definition,
                                   const FullyConnectedAttributes& attr,
                                   FullyConnectedBatched* result) {
  *result = FullyConnectedBatched(definition);
  RETURN_IF_ERROR(
      result->UploadWeights(attr.weights, creation_context.context));
  LinearStorageCreateInfo create_info;
  create_info.storage_type = LinearStorageType::TEXTURE_2D;
  create_info.data_type = definition.GetDataType();
  create_info.aligned_size = attr.weights.shape.o;
  RETURN_IF_ERROR(CreateLinearStorage(
      create_info, attr.bias, creation_context.context, &result->biases_));
  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
