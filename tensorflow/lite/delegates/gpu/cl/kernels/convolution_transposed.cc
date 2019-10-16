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

#include "tensorflow/lite/delegates/gpu/cl/kernels/convolution_transposed.h"

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GenerateConvolutionTransposedCode(
    const OperationDef& op_def, const LinearStorage& biases,
    const CLDevice& device,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  const TensorCodeGenerator::SizeVariablesNames src_size(
      "src_size.x", "src_size.y", "src_size.z", "src_size.w");
  const TensorCodeGenerator::SizeVariablesNames dst_size(
      "dst_size.x", "dst_size.y", "dst_size.z", "dst_size.w");
  TensorCodeGenerator src_tensor("src_data", src_size, op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor("dst_data", dst_size, op_def.dst_tensors[0]);
  const auto src_tensor_type = op_def.src_tensors[0].storage_type;

  const std::string batch_id = op_def.batch_support ? "B" : "";
  std::string c = GetCommonDefines(op_def.precision);

  switch (op_def.precision) {
    case CalculationsPrecision::F32:
    case CalculationsPrecision::F16:
      if (src_tensor_type == TensorStorageType::BUFFER) {
        c += "#define CONV(R, S)   \\\n";
        c += "R += S.x * f0.s0123; \\\n";
        c += "R += S.y * f0.s4567; \\\n";
        c += "R += S.z * f0.s89ab; \\\n";
        c += "R += S.w * f0.scdef;   \n";
      } else {
        c += "#define CONV(R, S)  \\\n";
        c += "R += S.x * f[0];    \\\n";
        c += "R += S.y * f[1];    \\\n";
        c += "R += S.z * f[2];    \\\n";
        c += "R += S.w * f[3];      \n";
      }
      break;
    case CalculationsPrecision::F32_F16:
      if (src_tensor_type == TensorStorageType::BUFFER) {
        c += "#define CONV(R, S) \\\n";
        c += "R += convert_float4(S.x * f0.s0123 + S.y * f0.s4567 + S.z * "
             "f0.s89ab + S.w * f0.scdef);\n";
      } else {
        c += "#define CONV(R, S) \\\n";
        c += "R += convert_float4(S.x * f[0] + S.y * f[1]";
        c += "+ S.z * f[2] + S.w * f[3]);\n";
      }
      break;
  }

  switch (op_def.precision) {
    case CalculationsPrecision::F32:
      c += "#define FLT16 float16\n";
      break;
    case CalculationsPrecision::F32_F16:
    case CalculationsPrecision::F16:
      c += "#define FLT16 half16\n";
      break;
  }

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ) + ",\n";
  if (src_tensor_type == TensorStorageType::BUFFER) {
    c += "    __global FLT16* filters,  \n";
    c += "    __global FLT4* biases";
  } else {
    c += "    __read_only image2d_t filters,  \n";
    c += "    __read_only image2d_t biases";
  }
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int2 kernel_size,          \n";
  c += "    int2 stride,               \n";
  c += "    int2 padding,              \n";
  c += "    int2 k_offset,        \n";
  c += "    int2 inner_size,           \n";
  c += "    int4 src_size,             \n";
  c += "    int4 dst_size              \n";
  c += ") {\n";
  if (op_def.batch_support) {
    c += "  int linear_id = get_global_id(0);\n";
    c += "  int X = linear_id / dst_size.w;\n";
    c += "  int B = linear_id % dst_size.w;\n";
  } else {
    c += "  int X = get_global_id(0);\n";
  }
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y || Z >= dst_size.z) return;\n";
  if (src_tensor_type == TensorStorageType::BUFFER) {
    c += "  int f_base = Z * src_size.z * kernel_size.x * kernel_size.y;\n";
  }
  c += "  int2 offset = (int2)(X, Y) + padding - k_offset;\n";
  c += "  offset.x = offset.x % stride.x;\n";
  c += "  offset.y = offset.y % stride.y;\n";
  c += "  offset += stride;\n";
  c += "  offset.x = offset.x % stride.x;\n";
  c += "  offset.y = offset.y % stride.y;\n";
  c += "  int2 f_offset;\n";
  c += "  f_offset.x = offset.x == 0 ? 0 : stride.x - offset.x;\n";
  c += "  f_offset.y = offset.y == 0 ? 0 : stride.y - offset.y;\n";
  c += "  ACCUM_FLT4 r0 = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n";
  c += "  for (int ky = 0; ky < inner_size.y; ++ky) {\n";
  c += "    int index_y = ky * stride.y + f_offset.y;\n";
  c += "    bool inside_y = index_y < kernel_size.y;\n";
  c += "    int s_y = (Y + index_y + padding.y - k_offset.y) / stride.y;\n";
  c += "    index_y = kernel_size.y - 1 - index_y;\n";
  c += "    bool out_y = s_y < 0 || s_y >= src_size.y;\n";
  c += "    for (int kx = 0; kx < inner_size.x; ++kx) {\n";
  c += "      int index_x = kx * stride.x + f_offset.x;\n";
  c += "      bool inside_kernel = index_x < kernel_size.x && inside_y;\n";
  c += "      int s_x = (X + index_x + padding.x - k_offset.x) / stride.x;\n";
  c += "      index_x = kernel_size.x - 1 - index_x;\n";
  c += "      bool out_x = s_x < 0 || s_x >= src_size.x;\n";
  c += "      int kernel_index = index_y * kernel_size.x + index_x;\n";
  c += "      if (inside_kernel && !(out_x || out_y)) {\n";
  if (src_tensor_type == TensorStorageType::BUFFER) {
    c += "        int f_offset = f_base + kernel_index * src_size.z;\n";
  } else {
    c += "        int x_c = kernel_index * src_size.z * 4;\n";
  }
  c += "        for (int l = 0; l < src_size.z; ++l) {\n";
  c += "          FLT4 src =" + src_tensor.Read4D("s_x", "s_y", "l", batch_id) +
       ";\n";
  if (src_tensor_type == TensorStorageType::BUFFER) {
    c += "          FLT16 f0 = filters[f_offset]; f_offset++;\n";
  } else {
    c += "          FLT4 f[4];\n";
    c += "          f[0] = READ_IMAGE(filters, smp_none, (int2)(x_c, Z)); "
         "x_c++;\n";
    c += "          f[1] = READ_IMAGE(filters, smp_none, (int2)(x_c, Z)); "
         "x_c++;\n";
    c += "          f[2] = READ_IMAGE(filters, smp_none, (int2)(x_c, Z)); "
         "x_c++;\n";
    c += "          f[3] = READ_IMAGE(filters, smp_none, (int2)(x_c, Z)); "
         "x_c++;\n";
  }
  c += "          CONV(r0, src);\n";
  c += "        }\n";
  c += "      }\n";
  c += "    }\n";
  c += "  }\n";
  c += "  FLT4 bias_val = " + biases.ReadLinearFLT4("Z") + ";\n";
  c += "  FLT4 res0 = TO_FLT4(r0) + bias_val;\n";
  std::string x_3dcoord = op_def.batch_support ? "X * dst_size.w + B" : "X";
  const LinkingContext context{"res0", x_3dcoord, "Y", "Z"};
  c += PostProcess(linked_operations, context);
  c += "  " + dst_tensor.Write4D("res0", "X", "Y", "Z", batch_id) + "\n";
  c += "}\n";

  return c;
}
}  // namespace

ConvolutionTransposed::ConvolutionTransposed(
    const OperationDef& definition, const ConvolutionTransposedAttributes& attr)
    : GPUOperation(definition),
      kernel_size_(attr.weights.shape.w, attr.weights.shape.h),
      stride_(attr.stride.w, attr.stride.h),
      padding_(attr.padding.prepended.w, attr.padding.prepended.h),
      src_channels_(attr.weights.shape.i),
      dst_channels_(attr.weights.shape.o) {
  const int inner_size_x = (kernel_size_.x - 1) / stride_.x + 1;
  const int inner_size_y = (kernel_size_.y - 1) / stride_.y + 1;
  inner_size_ = int2(inner_size_x, inner_size_y);
  kernel_offset_ = int2(kernel_size_.x - 1, kernel_size_.y - 1);
}

ConvolutionTransposed::ConvolutionTransposed(ConvolutionTransposed&& kernel)
    : GPUOperation(std::move(kernel)),
      biases_(std::move(kernel.biases_)),
      weights_tex2d_(std::move(kernel.weights_tex2d_)),
      weights_buf_(std::move(kernel.weights_buf_)),
      weights_(kernel.weights_),
      kernel_size_(kernel.kernel_size_),
      stride_(kernel.stride_),
      padding_(kernel.padding_),
      kernel_offset_(kernel.kernel_offset_),
      inner_size_(kernel.inner_size_),
      src_channels_(kernel.src_channels_),
      dst_channels_(kernel.dst_channels_),
      kernel_(std::move(kernel.kernel_)),
      work_group_size_(kernel.work_group_size_) {}

ConvolutionTransposed& ConvolutionTransposed::operator=(
    ConvolutionTransposed&& kernel) {
  if (this != &kernel) {
    biases_ = std::move(kernel.biases_);
    weights_tex2d_ = std::move(kernel.weights_tex2d_);
    weights_buf_ = std::move(kernel.weights_buf_);
    std::swap(weights_, kernel.weights_);
    std::swap(kernel_size_, kernel.kernel_size_);
    std::swap(stride_, kernel.stride_);
    std::swap(padding_, kernel.padding_);
    std::swap(kernel_offset_, kernel.kernel_offset_);
    std::swap(inner_size_, kernel.inner_size_);
    std::swap(src_channels_, kernel.src_channels_);
    std::swap(dst_channels_, kernel.dst_channels_);
    kernel_ = std::move(kernel.kernel_);
    std::swap(work_group_size_, kernel.work_group_size_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

Status ConvolutionTransposed::Compile(const CreationContext& creation_context) {
  const auto code = GenerateConvolutionTransposedCode(
      definition_, biases_, *creation_context.device, linked_operations_);

  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status ConvolutionTransposed::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(biases_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(kernel_size_));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(stride_));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(padding_));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(kernel_offset_));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(inner_size_));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWHDB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWHDB()));
  return OkStatus();
}

int3 ConvolutionTransposed::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Depth();
  return int3(grid_x, grid_y, grid_z);
}

Status ConvolutionTransposed::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroupConv(params, kernel_, GetGridSize(),
                              &work_group_size_);
}

Status ConvolutionTransposed::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Status CreateConvolutionTransposed(const CreationContext& creation_context,
                                   const OperationDef& definition,
                                   const ConvolutionTransposedAttributes& attr,
                                   ConvolutionTransposed* result) {
  *result = ConvolutionTransposed(definition, attr);
  RETURN_IF_ERROR(
      result->UploadWeights(attr.weights, creation_context.context));
  LinearStorageCreateInfo create_info;
  create_info.storage_type =
      DeduceLinearStorageType(definition.GetPrimaryStorageType());
  create_info.data_type = definition.GetDataType();
  create_info.name = "biases";
  create_info.aligned_size = attr.weights.shape.o;
  RETURN_IF_ERROR(CreateLinearStorage(
      create_info, attr.bias, creation_context.context, &result->biases_));

  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
