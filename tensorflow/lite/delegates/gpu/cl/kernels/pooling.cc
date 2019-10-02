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

#include "tensorflow/lite/delegates/gpu/cl/kernels/pooling.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetAveragePoolingKernelCode(
    const TensorDescriptor& src_descriptor,
    const TensorDescriptor& dst_descriptor, CalculationsPrecision precision,
    const CLDevice& device,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor("src_data", "src_size", src_descriptor);
  TensorCodeGenerator dst_tensor("dst_data", "dst_size", dst_descriptor);

  const auto address_mode = GetFastestZeroMode(device);

  std::string code = GetCommonDefines(precision);

  const bool manual_clamp =
      src_descriptor.storage_type == TensorStorageType::BUFFER ||
      src_descriptor.storage_type == TensorStorageType::IMAGE_BUFFER;

  code += "__kernel void main_function(\n";
  code += src_tensor.GetDeclaration(AccessType::READ);
  code += GetArgsDeclaration(linked_operations);
  code += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  code += "    int4 src_size,             \n";
  code += "    int4 dst_size,             \n";
  code += "    int2 kernel_size,          \n";
  code += "    int2 padding,              \n";
  code += "    int2 stride                \n";
  code += ") {\n";
  code += "  int X = get_global_id(0);\n";
  code += "  int Y = get_global_id(1);\n";
  code += "  int Z = get_global_id(2);\n";
  code +=
      "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.w) return; \n";
  code += "  float4 r = (float4)(0.0f);\n";
  code += "  float window_size = 0.0;\n";
  code += "  for (int ky = 0; ky < kernel_size.y; ++ky) {\n";
  code += "    int y_c = Y * stride.y - padding.y + ky;\n";
  code += "    bool outside_y = y_c < 0 || y_c >= src_size.y;\n";
  code += "    for (int kx = 0; kx < kernel_size.x; ++kx) {\n";
  code += "      int x_c = X * stride.x - padding.x + kx;\n";
  code += "      bool outside = outside_y || x_c < 0 || x_c >= src_size.x;\n";
  if (manual_clamp) {
    code += "     r += !outside ? " +
            src_tensor.ReadAsFloat3D("x_c", "y_c", "Z",
                                     TextureAddressMode::DONT_CARE) +
            " : (float4)(0.0f);\n";
  } else {
    code += "      r += " +
            src_tensor.ReadAsFloat3D("x_c", "y_c", "Z", address_mode) + ";\n";
  }
  code += "        window_size += !outside ? 1.0 : 0.0;\n";
  code += "    }\n";
  code += "  }\n";
  // If window_size==0, window covered nothing. This situation is a sign of
  // incorrectly constructed operation. NaNs are expected as output.
  code += "  FLT4 result = TO_FLT4(r / window_size);\n";
  const LinkingContext context{"result", "X", "Y", "Z"};
  code += PostProcess(linked_operations, context);
  code += "  " + dst_tensor.Write3D("result", "X", "Y", "Z");
  code += "}\n";

  return code;
}

std::string GetMaxPoolingKernelCode(
    const TensorDescriptor& src_descriptor,
    const TensorDescriptor& dst_descriptor, CalculationsPrecision precision,
    const std::vector<ElementwiseOperation*>& linked_operations,
    bool output_indices) {
  TensorCodeGenerator src_tensor("src_data", "src_size", src_descriptor);
  TensorCodeGenerator dst_tensor("dst_data", "dst_size", dst_descriptor);
  TensorCodeGenerator indices_tensor("dst_indices", "dst_size", dst_descriptor);

  std::string code = GetCommonDefines(precision);

  code += "__kernel void main_function(\n";
  code += src_tensor.GetDeclaration(AccessType::READ);
  code += GetArgsDeclaration(linked_operations);
  code += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  if (output_indices) {
    code += indices_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  }
  code += "    int4 src_size,             \n";
  code += "    int4 dst_size,             \n";
  code += "    int2 kernel_size,          \n";
  code += "    int2 padding,              \n";
  code += "    int2 stride                \n";
  code += ") {\n";
  code += "  int X = get_global_id(0);\n";
  code += "  int Y = get_global_id(1);\n";
  code += "  int Z = get_global_id(2);\n";
  code +=
      "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.w) return; \n";
  code += "  FLT4 maximum = (FLT4)(-10000.0f);\n";
  if (output_indices) {
    code += "  FLT4 indexes = (FLT4)(0.0f);\n";
    code += "  FLT index_counter = (FLT)(0.1f);\n";
  }
  code += "  for (int ky = 0; ky < kernel_size.y; ++ky) {\n";
  code += "    int y_c = Y * stride.y - padding.y + ky;\n";
  code += "    bool outside_y = y_c < 0 || y_c >= src_size.y;\n";
  code += "    for (int kx = 0; kx < kernel_size.x; ++kx) {\n";
  code += "      int x_c = X * stride.x - padding.x + kx;\n";
  code += "      bool outside_x = x_c < 0 || x_c >= src_size.x;\n";
  code += "      if (!outside_x && !outside_y) {\n";
  code += "        FLT4 src = " +
          src_tensor.Read3D("x_c", "y_c", "Z", TextureAddressMode::DONT_CARE) +
          ";\n";
  if (output_indices) {
    code += "        if (src.x > maximum.x) {\n";
    code += "          indexes.x = index_counter;\n";
    code += "          maximum.x = src.x;\n";
    code += "        }\n";
    code += "        if (src.y > maximum.y) {\n";
    code += "          indexes.y = index_counter;\n";
    code += "          maximum.y = src.y;\n";
    code += "        }\n";
    code += "        if (src.z > maximum.z) {\n";
    code += "          indexes.z = index_counter;\n";
    code += "          maximum.z = src.z;\n";
    code += "        }\n";
    code += "        if (src.w > maximum.w) {\n";
    code += "          indexes.w = index_counter;\n";
    code += "          maximum.w = src.w;\n";
    code += "        }\n";
    code += "        index_counter += (FLT)(1.0f);\n";
  }
  code += "        maximum = max(src, maximum);\n";
  code += "      };\n";
  code += "    }\n";
  code += "  }\n";
  const LinkingContext context{"maximum", "X", "Y", "Z"};
  code += PostProcess(linked_operations, context);
  code += "  " + dst_tensor.Write3D("maximum", "X", "Y", "Z");
  if (output_indices) {
    code += "  " + indices_tensor.Write3D("indexes", "X", "Y", "Z");
  }
  code += "}\n";

  return code;
}

}  // namespace

Pooling::Pooling(const OperationDef& definition,
                 const Pooling2DAttributes& attr)
    : GPUOperation(definition),
      stride_(attr.strides.w, attr.strides.h),
      padding_(attr.padding.prepended.w, attr.padding.prepended.h),
      kernel_size_(attr.kernel.w, attr.kernel.h),
      type_(attr.type),
      output_indices_(attr.output_indices) {}

Pooling::Pooling(Pooling&& kernel)
    : GPUOperation(std::move(kernel)),
      stride_(kernel.stride_),
      padding_(kernel.padding_),
      kernel_size_(kernel.kernel_size_),
      type_(kernel.type_),
      output_indices_(kernel.output_indices_),
      kernel_(std::move(kernel.kernel_)),
      work_group_size_(kernel.work_group_size_) {}

Pooling& Pooling::operator=(Pooling&& kernel) {
  if (this != &kernel) {
    std::swap(stride_, kernel.stride_);
    std::swap(padding_, kernel.padding_);
    std::swap(kernel_size_, kernel.kernel_size_);
    std::swap(type_, kernel.type_);
    std::swap(output_indices_, kernel.output_indices_);
    kernel_ = std::move(kernel.kernel_);
    std::swap(work_group_size_, kernel.work_group_size_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

Status Pooling::Compile(const CreationContext& creation_context) {
  std::string code;
  switch (type_) {
    case PoolingType::AVERAGE:
      code = GetAveragePoolingKernelCode(
          definition_.src_tensors[0], definition_.dst_tensors[0],
          definition_.precision, *creation_context.device, linked_operations_);
      break;
    case PoolingType::MAX:
      code = GetMaxPoolingKernelCode(
          definition_.src_tensors[0], definition_.dst_tensors[0],
          definition_.precision, linked_operations_, output_indices_);
      break;
    default:
      return InvalidArgumentError(
          "You should create another kernel with this params");
      break;
  }
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status Pooling::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  if (output_indices_) {
    RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[1]->GetMemoryPtrForWriting()));
  }
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(kernel_size_));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(padding_));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(stride_));

  return OkStatus();
}

int3 Pooling::GetGridSize() const {
  const int grid_x = dst_[0]->Width();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Depth();
  return int3(grid_x, grid_y, grid_z);
}

Status Pooling::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status Pooling::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Pooling CreatePooling(const OperationDef& definition,
                      const Pooling2DAttributes& attr) {
  return Pooling(definition, attr);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
